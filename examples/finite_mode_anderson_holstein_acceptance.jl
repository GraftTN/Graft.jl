using Graft
using Graft.TestUtils
using LinearAlgebra
using Printf

function chain_topology(sites)
    return TreeTopology(
        last(sites),
        [sites[index + 1] => sites[index]
         for index in (length(sites) - 1):-1:1])
end

function finite_mode_model(action::FiniteModeAction, nmax::Int)
    spin = spin_ops()
    boson_libraries = [boson_ops(nmax)
                       for _ in action.boson_frequencies]
    bath_count = length(action.bath_energies)
    spinful = action.orbital_convention === :spinful
    impurity_sites = spinful ? [:d_up, :d_dn] : [:d]
    bath_sites = Dict(
        impurity => [Symbol(:bath_, impurity, :_, index)
                     for index in 1:bath_count]
        for impurity in impurity_sites)
    boson_sites = [Symbol(:ph_, index)
                   for index in eachindex(action.boson_frequencies)]
    sites = Symbol[]
    for impurity in impurity_sites
        push!(sites, impurity)
        append!(sites, bath_sites[impurity])
    end
    append!(sites, boson_sites)
    topology = chain_topology(sites)
    physical_spaces = Dict(site => spin.P for site in sites
                           if site ∉ boson_sites)
    for (site, library) in zip(boson_sites, boson_libraries)
        physical_spaces[site] = library.P
    end

    generator = OpSum()
    density = OpSum()
    for impurity in impurity_sites
        generator += Term(
            -action.mu, SiteOp(impurity, :N, spin.N))
        density += Term(1.0, SiteOp(impurity, :N, spin.N))
        for index in 1:bath_count
            bath = bath_sites[impurity][index]
            epsilon = action.bath_energies[index]
            coupling = action.bath_couplings[index]
            generator += Term(
                epsilon, SiteOp(bath, :N, spin.N))
            generator += Term(
                coupling,
                SiteOp(impurity, :Cd, spin.Sm),
                SiteOp(bath, :C, spin.Sp))
            generator += Term(
                conj(coupling),
                SiteOp(impurity, :C, spin.Sp),
                SiteOp(bath, :Cd, spin.Sm))
        end
    end
    double_occupancy = OpSum()
    if spinful
        generator += Term(
            action.static_interaction,
            SiteOp(:d_up, :N, spin.N),
            SiteOp(:d_dn, :N, spin.N))
        double_occupancy += Term(
            1.0,
            SiteOp(:d_up, :N, spin.N),
            SiteOp(:d_dn, :N, spin.N))
    end

    boson_number = OpSum()
    anchor = first(impurity_sites)
    for (site, library, omega, coupling) in zip(
            boson_sites, boson_libraries,
            action.boson_frequencies, action.boson_couplings)
        generator += Term(omega, SiteOp(site, :N, library.N))
        boson_number += Term(1.0, SiteOp(site, :N, library.N))
        for impurity in impurity_sites
            generator += Term(
                coupling,
                SiteOp(impurity, :N, spin.N),
                SiteOp(site, :X, library.X))
        end
        generator += Term(
            -coupling * action.n0,
            SiteOp(anchor, :I, spin.I),
            SiteOp(site, :X, library.X))
    end
    annihilator = OpSum() +
        Term(1.0, SiteOp(first(impurity_sites), :C, spin.Sp))
    creator = OpSum() +
        Term(1.0, SiteOp(first(impurity_sites), :Cd, spin.Sm))
    return (;
        topology, physical_spaces, generator, density, double_occupancy,
        boson_number, annihilator, creator)
end

struct DenseThermalCache
    energies::Vector{Float64}
    vectors::Matrix{ComplexF64}
    shifted_weights::Vector{Float64}
    partition::Float64
    logZ::Float64
    beta::Float64
end

function DenseThermalCache(matrix, beta)
    decomposition = eigen(Hermitian(Matrix{ComplexF64}(matrix)))
    energies = real.(decomposition.values)
    minimum_energy = minimum(energies)
    weights = exp.(-beta .* (energies .- minimum_energy))
    partition = sum(weights)
    return DenseThermalCache(
        energies, decomposition.vectors, weights, partition,
        -beta * minimum_energy + log(partition), beta)
end

eigenbasis(cache, operator) =
    cache.vectors' * Matrix{ComplexF64}(operator) * cache.vectors

function thermal_expectation(cache, operator)
    transformed = eigenbasis(cache, operator)
    return sum(diag(transformed) .* cache.shifted_weights) / cache.partition
end

function thermal_correlation(cache, operator_a, operator_b, times)
    A = eigenbasis(cache, operator_a)
    B = eigenbasis(cache, operator_b)
    minimum_energy = minimum(cache.energies)
    shifted = cache.energies .- minimum_energy
    return ComplexF64[
        sum(
            exp(-(cache.beta - tau) * shifted[m] - tau * shifted[n]) *
            A[m, n] * B[n, m]
            for m in eachindex(shifted), n in eachindex(shifted)) /
        cache.partition
        for tau in times
    ]
end

function benchmark_cell(label, action, nmax, time_count)
    started = time()
    model = finite_mode_model(action, nmax)
    H = dense_hamiltonian(
        model.generator, model.topology, model.physical_spaces)
    density_operator = dense_hamiltonian(
        model.density, model.topology, model.physical_spaces)
    boson_operator = dense_hamiltonian(
        model.boson_number, model.topology, model.physical_spaces)
    annihilator = dense_hamiltonian(
        model.annihilator, model.topology, model.physical_spaces)
    creator = dense_hamiltonian(
        model.creator, model.topology, model.physical_spaces)
    cache = DenseThermalCache(H, action.beta)
    times = collect(range(0.0, action.beta; length=time_count))
    density = thermal_expectation(cache, density_operator)
    green = CorrelatorSeries(
        times,
        -thermal_correlation(
            cache, annihilator, creator, times),
        (; beta=action.beta))
    chi = CorrelatorSeries(
        times,
        thermal_correlation(
            cache, density_operator, density_operator, times) .-
            density^2,
        (; beta=action.beta))
    giw = matsubara_transform(
        green; statistics=:fermionic, indices=0:7)
    chiiv = matsubara_transform(
        chi; statistics=:bosonic, indices=0:7)

    data = ThermalBenchmarkDatum[
        ThermalBenchmarkDatum(:logZ, :scalar, 0.0, cache.logZ),
        ThermalBenchmarkDatum(:density, :scalar, 0.0, density),
        ThermalBenchmarkDatum(
            :boson_occupation, :scalar, 0.0,
            thermal_expectation(cache, boson_operator)),
    ]
    if action.orbital_convention === :spinful
        double_operator = dense_hamiltonian(
            model.double_occupancy,
            model.topology, model.physical_spaces)
        push!(data, ThermalBenchmarkDatum(
            :double_occupancy, :scalar, 0.0,
            thermal_expectation(cache, double_operator)))
    end
    append!(data, [
        ThermalBenchmarkDatum(:Gtau, :tau, tau, value)
        for (tau, value) in green
    ])
    append!(data, [
        ThermalBenchmarkDatum(:chi_nn, :tau, tau, value)
        for (tau, value) in chi
    ])
    append!(data, [
        ThermalBenchmarkDatum(:Giw, :fermionic_iw, frequency, value)
        for (frequency, value) in giw
    ])
    append!(data, [
        ThermalBenchmarkDatum(:chi_nn_iv, :bosonic_iv, frequency, value)
        for (frequency, value) in chiiv
    ])
    return FiniteModeBenchmarkCell(
        label, action, :ed, :plain, nmax, data;
        truncation=(; scheme=:dense_exact),
        propagation_grid=times,
        wall_time=time() - started)
end

function actions(full)
    result = Pair{Symbol,FiniteModeAction}[
        :spinless_one_bath_one_boson => FiniteModeAction(
            beta=2.0, mu=0.1, n0=0.5,
            bath_energies=[-0.35], bath_couplings=[0.25],
            boson_frequencies=[1.0], boson_couplings=[0.16]),
        :spinful_one_bath_one_boson => FiniteModeAction(
            beta=2.0, mu=1.0, static_interaction=2.0, n0=1.0,
            bath_energies=[-0.35], bath_couplings=[0.25],
            boson_frequencies=[1.0], boson_couplings=[0.16],
            orbital_convention=:spinful),
    ]
    full && push!(result,
        :spinful_two_bath_two_boson => FiniteModeAction(
            beta=2.0, mu=1.0, static_interaction=2.0, n0=1.0,
            bath_energies=[-0.45, 0.4],
            bath_couplings=[0.22, 0.17],
            boson_frequencies=[1.0, 1.3],
            boson_couplings=[0.12, 0.08],
            orbital_convention=:spinful))
    return result
end

function with_cutoff_budget(high, low)
    data = ThermalBenchmarkDatum[]
    for datum in high.data
        reference = only(filter(low.data) do candidate
            candidate.observable === datum.observable &&
                candidate.axis === datum.axis &&
                candidate.coordinate == datum.coordinate
        end)
        push!(data, ThermalBenchmarkDatum(
            datum.observable, datum.axis, datum.coordinate, datum.value;
            stderr=datum.stderr,
            deterministic_error=datum.deterministic_error,
            cutoff_error=abs(datum.value - reference.value),
            lbo_error=datum.lbo_error))
    end
    return data
end

function main()
    full = lowercase(get(ENV, "GRAFT_P4_FULL", "false")) in
        ("1", "true", "yes", "on")
    cutoffs = full ? [3, 4, 5] : [1, 2]
    time_count = full ? 65 : 17
    tolerance = parse(
        Float64, get(ENV, "GRAFT_P4_TARGET_UNCERTAINTY",
                     full ? "1e-3" : "5e-2"))
    output = get(
        ENV, "GRAFT_P4_OUTPUT",
        "finite_mode_anderson_holstein_benchmark.csv")
    ctseg_directory = get(ENV, "GRAFT_CTSEG_RESULTS_DIR", "")

    cells = FiniteModeBenchmarkCell[]
    accepted = true
    for (label, action) in actions(full)
        local_cells = FiniteModeBenchmarkCell[]
        for nmax in cutoffs
            cell = benchmark_cell(label, action, nmax, time_count)
            push!(local_cells, cell)
            push!(cells, cell)
            @printf(
                "%s nmax=%d logZ=%.10f density=%.10f wall=%.3fs\n",
                label, nmax,
                real(only(d.value for d in cell.data
                          if d.observable === :logZ)),
                real(only(d.value for d in cell.data
                          if d.observable === :density)),
                cell.wall_time)
        end
        report = assess_boson_cutoff(
            local_cells; target_uncertainty=tolerance)
        accepted &= report.converged
        final_rows = filter(
            row -> (row.low_nmax, row.high_nmax) == report.final_pair,
            report.rows)
        worst = final_rows[argmax(getfield.(final_rows, :absolute_error))]
        println(label, " cutoff ", report.final_pair,
                " converged=", report.converged,
                " worst=", worst.observable, "/", worst.axis,
                " error=", worst.absolute_error)

        input_path = string(output, ".", label, ".ctseg_input.csv")
        write_ctseg_input_csv(input_path, action)
        if !isempty(ctseg_directory)
            artifact_path = joinpath(
                ctseg_directory, string(label, ".ctseg_results.csv"))
            artifact = read_ctseg_results_csv(artifact_path)
            reference = with_cutoff_budget(
                local_cells[end], local_cells[end - 1])
            comparison = compare_ctseg(reference, artifact; action)
            accepted &= comparison.passed
            println(label, " CTSEG accepted=", comparison.passed)
        end
    end
    write_finite_mode_benchmark_csv(output, cells)
    println("mode=", full ? "full" : "smoke",
            " accepted=", accepted, " records=", output)
    full && !accepted && error("finite-mode P4/P5 acceptance gate failed")
end

main()
