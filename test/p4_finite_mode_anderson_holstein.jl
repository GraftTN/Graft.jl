function _p4_spinless_model(action::FiniteModeAction, nmax)
    length(action.bath_energies) == 1 ||
        throw(ArgumentError("fixture expects one fermionic bath mode"))
    length(action.boson_frequencies) == 1 ||
        throw(ArgumentError("fixture expects one boson mode"))
    S = spin_ops()
    B = boson_ops(nmax)
    topo = TreeTopology(:ph, [:ph => :bath, :bath => :d])
    phys = Dict(:d => S.P, :bath => S.P, :ph => B.P)
    epsilon = only(action.bath_energies)
    V = only(action.bath_couplings)
    omega = only(action.boson_frequencies)
    g = only(action.boson_couplings)
    H = OpSum() +
        Term(-action.mu, SiteOp(:d, :N, S.N)) +
        Term(epsilon, SiteOp(:bath, :N, S.N)) +
        Term(V, SiteOp(:d, :Cd, S.Sm), SiteOp(:bath, :C, S.Sp)) +
        Term(conj(V), SiteOp(:d, :C, S.Sp), SiteOp(:bath, :Cd, S.Sm)) +
        Term(omega, SiteOp(:ph, :N, B.N)) +
        Term(g, SiteOp(:d, :N, S.N), SiteOp(:ph, :X, B.X)) +
        Term(-g * action.n0, SiteOp(:d, :I, S.I), SiteOp(:ph, :X, B.X))
    density = OpSum() + Term(1.0, SiteOp(:d, :N, S.N))
    boson_number = OpSum() + Term(1.0, SiteOp(:ph, :N, B.N))
    return (; S, B, topo, phys, H, density, boson_number)
end

function _p4_benchmark_data(logz, density, boson, green, chi)
    giw = matsubara_transform(green; statistics=:fermionic, indices=0:1)
    chiiv = matsubara_transform(chi; statistics=:bosonic, indices=0:1)
    data = ThermalBenchmarkDatum[
        ThermalBenchmarkDatum(:logZ, :scalar, 0.0, logz),
        ThermalBenchmarkDatum(:density, :scalar, 0.0, density),
        ThermalBenchmarkDatum(:boson_occupation, :scalar, 0.0, boson),
    ]
    append!(data, [
        ThermalBenchmarkDatum(:Gtau, :tau, tau, value)
        for (tau, value) in zip(green.times, green.values)
    ])
    append!(data, [
        ThermalBenchmarkDatum(:chi_nn, :tau, tau, value)
        for (tau, value) in zip(chi.times, chi.values)
    ])
    append!(data, [
        ThermalBenchmarkDatum(:Giw, :fermionic_iw, frequency, value)
        for (frequency, value) in giw
    ])
    append!(data, [
        ThermalBenchmarkDatum(:chi_nn_iv, :bosonic_iv, frequency, value)
        for (frequency, value) in chiiv
    ])
    return data
end

function _p4_max_bond(psi)
    topo = topology(psi)
    return maximum(
        (dim(domain(psi.tensors[node])[1])
         for node in 1:nnodes(topo) if topo.parent[node] != 0);
        init=1)
end

function _p4_ppdress_observable(observable, model, nmax)
    dressed, _, _ = ppdress(
        observable, model.topo, model.phys;
        nmax, boson_sites=[:ph])
    return dressed
end

function _p4_ppdress_local(site, name, operator, model, nmax)
    observable = OpSum() + Term(1.0, SiteOp(site, name, operator))
    dressed = _p4_ppdress_observable(observable, model, nmax)
    return only(only(collect(dressed)).ops).op
end

@graft_testset "P4 finite-mode Anderson-Holstein ED/plain/PP" begin
    action = FiniteModeAction(
        beta=0.2,
        mu=0.2,
        n0=0.5,
        bath_energies=[-0.1],
        bath_couplings=[0.18],
        boson_frequencies=[0.7],
        boson_couplings=[0.12])
    nmax = 1
    model = _p4_spinless_model(action, nmax)
    Hd = dense_hamiltonian(model.H, model.topo, model.phys)
    Nd = dense_hamiltonian(model.density, model.topo, model.phys)
    Bd = dense_hamiltonian(model.boson_number, model.topo, model.phys)
    Cd = dense_hamiltonian(
        OpSum() + Term(1.0, SiteOp(:d, :C, model.S.Sp)),
        model.topo, model.phys)
    Cdd = dense_hamiltonian(
        OpSum() + Term(1.0, SiteOp(:d, :Cd, model.S.Sm)),
        model.topo, model.phys)
    times = collect(range(0.0, action.beta; length=3))

    density_ed = exact_thermal_expect(Hd, Nd, action.beta)
    green_ed = CorrelatorSeries(
        times,
        -exact_thermal_correlator(
            Hd, Cd, Cdd, action.beta, times),
        (; beta=action.beta))
    chi_ed = CorrelatorSeries(
        times,
        exact_thermal_correlator(
            Hd, Nd, Nd, action.beta, times) .- density_ed^2,
        (; beta=action.beta))
    ed_data = _p4_benchmark_data(
        exact_thermal_logZ(Hd, action.beta),
        density_ed,
        exact_thermal_expect(Hd, Bd, action.beta),
        green_ed,
        chi_ed)
    ed_cell = FiniteModeBenchmarkCell(
        :spinless_one_bath_one_boson,
        action, :ed, :plain, nmax, ed_data;
        truncation=(; scheme=:none),
        propagation_grid=times)

    evolver = TDVP2(
        trunc=TruncationScheme(maxdim=4, atol=1e-10),
        krylovdim=12, tol=1e-10, verbose=false)
    problem = purification_problem(
        model.H, model.topo, model.phys; hermitian=true)
    trajectory = thermalize(
        Purified(), problem, action.beta;
        evolver, nsteps=4,
        save_betas=sort(unique(action.beta .- times)))
    density = thermal_expect(
        trajectory, physical_ttno(problem, model.density))
    boson = thermal_expect(
        trajectory, physical_ttno(problem, model.boson_number))
    green = thermal_correlator(
        Purified(), problem,
        :d => model.S.Sp, :d => model.S.Sm,
        action.beta, times;
        evolver, trajectory, prop_nsteps=2)
    green = CorrelatorSeries(green.times, -green.values, green.metadata)
    chi = thermal_correlator(
        Purified(), problem,
        :d => model.S.N, :d => model.S.N,
        action.beta, times;
        evolver, trajectory, prop_nsteps=2, connected=true)
    plain_data = _p4_benchmark_data(
        trajectory.final.logZ, density, boson, green, chi)
    plain_cell = FiniteModeBenchmarkCell(
        :spinless_one_bath_one_boson,
        action, :graft, :plain, nmax, plain_data;
        max_bond_dimension=_p4_max_bond(trajectory.final.psi),
        truncation=(; maxdim=4, atol=1e-10, nsteps=4),
        propagation_grid=times)

    @test maximum(abs.(
        getfield.(plain_data, :value) .- getfield.(ed_data, :value))) < 1e-3
    @test plain_cell.max_bond_dimension > 1

    if GRAFT_EXTENDED_TESTS
        Hpp, topopp, physpp = ppdress(
            model.H, model.topo, model.phys;
            nmax, boson_sites=[:ph])
        problempp = purification_problem(
            Hpp, topopp, physpp;
            hermitian=true, pp_pairs=Dict(:ph => :ph_B1))
        densitypp_op = _p4_ppdress_observable(
            model.density, model, nmax)
        bosonpp_op = _p4_ppdress_observable(
            model.boson_number, model, nmax)
        Cpp = _p4_ppdress_local(
            :d, :C, model.S.Sp, model, nmax)
        Cdpp = _p4_ppdress_local(
            :d, :Cd, model.S.Sm, model, nmax)
        Npp = _p4_ppdress_local(
            :d, :N, model.S.N, model, nmax)
        densitypp_ttno = physical_ttno(problempp, densitypp_op)
        bosonpp_ttno = physical_ttno(problempp, bosonpp_op)
        trajectorypp = thermalize(
            Purified(), problempp, action.beta;
            evolver, nsteps=4,
            save_betas=sort(unique(action.beta .- times)))
        densitypp = thermal_expect(
            trajectorypp, densitypp_ttno)
        bosonpp = thermal_expect(
            trajectorypp, bosonpp_ttno)
        greenpp = thermal_correlator(
            Purified(), problempp,
            :d => Cpp, :d => Cdpp,
            action.beta, times;
            evolver, trajectory=trajectorypp, prop_nsteps=2)
        greenpp = CorrelatorSeries(
            greenpp.times, -greenpp.values, greenpp.metadata)
        chipp = thermal_correlator(
            Purified(), problempp,
            :d => Npp, :d => Npp,
            action.beta, times;
            evolver, trajectory=trajectorypp, prop_nsteps=2, connected=true)
        pp_data = _p4_benchmark_data(
            trajectorypp.final.logZ, densitypp, bosonpp, greenpp, chipp)
        pp_cell = FiniteModeBenchmarkCell(
            :spinless_one_bath_one_boson,
            action, :graft, :pp_untruncated, nmax, pp_data;
            max_bond_dimension=_p4_max_bond(trajectorypp.final.psi),
            truncation=(; pp_bond=:untruncated, maxdim=4, nsteps=4),
            propagation_grid=times)

        pp_report = compare_representations(
            plain_cell, pp_cell; tolerance=1e-3)
        @test pp_report.passed
        @test maximum(row.absolute_error for row in pp_report.rows) < 1e-3
    end
    @test ed_cell.action === action
end
