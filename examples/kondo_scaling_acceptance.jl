using Graft
using Graft.TestUtils
using Graft.Backend: Vect, FermionParity, dim, domain
using LinearAlgebra: norm
using Printf
using Random

function semicircular_bath(nbath::Int)
    return gauss_semicircular_bath(nbath)
end

function anderson_model(U::Real, energies, couplings)
    fermion = fermion_ops_z2()
    length(energies) == length(couplings) ||
        throw(ArgumentError("bath energies and couplings must have equal length"))
    nbath = length(energies)
    bath_up = [Symbol(:bath_up_, j) for j in 1:nbath]
    bath_dn = [Symbol(:bath_dn_, j) for j in 1:nbath]
    sites = [:d_up; bath_up; :d_dn; bath_dn]
    topology = TreeTopology(
        first(sites), [sites[j] => sites[j + 1] for j in 1:(length(sites) - 1)])
    physical_spaces = Dict(site => fermion.P for site in sites)
    hamiltonian = OpSum() +
        Term(U, SiteOp(:d_up, :N, fermion.N),
                SiteOp(:d_dn, :N, fermion.N)) +
        Term(-U / 2, SiteOp(:d_up, :N, fermion.N)) +
        Term(-U / 2, SiteOp(:d_dn, :N, fermion.N))
    for (spin, bath) in ((:d_up, bath_up), (:d_dn, bath_dn))
        for j in eachindex(bath)
            hamiltonian += Term(
                energies[j], SiteOp(bath[j], :N, fermion.N))
            hamiltonian += Term(
                couplings[j],
                SiteOp(spin, :Cd, fermion.Cd),
                SiteOp(bath[j], :C, fermion.C))
            hamiltonian += Term(
                couplings[j],
                SiteOp(spin, :C, fermion.C),
                SiteOp(bath[j], :Cd, fermion.Cd))
        end
    end
    double_occupancy = OpSum() +
        Term(1.0, SiteOp(:d_up, :N, fermion.N),
                  SiteOp(:d_dn, :N, fermion.N))
    return (; topology, physical_spaces, hamiltonian, double_occupancy)
end

function parity_bond(maxdim::Int)
    maxdim >= 2 || throw(ArgumentError("maxdim must be at least two"))
    even = FermionParity(0)
    odd = FermionParity(1)
    even_dim = cld(maxdim, 2)
    odd_dim = fld(maxdim, 2)
    return Vect[FermionParity](even => even_dim, odd => odd_dim)
end

max_bond_dimension(psi) = maximum(
    dim(domain(psi.tensors[node])[1])
    for node in 1:nnodes(topology(psi))
    if topology(psi).parent[node] != 0
)

function ground_double_occupancy(model, rng, bond, nsweeps)
    operator = ttno_from_opsum(
        model.hamiltonian, model.topology, model.physical_spaces;
        hermitian=true)
    observable = ttno_from_opsum(
        model.double_occupancy, model.topology, model.physical_spaces;
        hermitian=true)
    state = random_ttns(
        rng, ComplexF64, model.topology, model.physical_spaces, bond)
    dmrg2!(
        state, operator;
        trunc=TruncationScheme(maxdim=dim(bond), atol=1e-10),
        nsweeps, krylovdim=32, verbose=false)
    return state, real(expect(state, observable))
end

function fixed_manifold_transform(template)
    return state -> begin
        target = copy(template)
        fit!(
            target, (state,);
            nsweeps=2, tol=1e-10, normalize=false, verbose=false)
        state_norm = norm(target)
        isfinite(state_norm) && state_norm > 0 ||
            error("product state could not be embedded in the fixed manifold")
        normalize!(target)
        return target
    end
end

function run_curve(U, betas, energies, couplings, sample_counts,
                   burnin, maxdim, ground_sweeps, panel_steps, rng)
    model = anderson_model(U, energies, couplings)
    problem = purification_problem(
        model.hamiltonian, model.topology, model.physical_spaces;
        hermitian=true)
    observable = physical_ttno(
        problem, model.double_occupancy; hermitian=true, doubled=false)
    bond = parity_bond(maxdim)
    ground_state, D0 = ground_double_occupancy(
        model, rng, bond, ground_sweeps)
    template = random_ttns(
        rng, ComplexF64, model.topology, model.physical_spaces, bond)
    transform = fixed_manifold_transform(template)
    values = Float64[]
    errors = Float64[]
    walltimes = Float64[]
    maxbonds = Int[]
    for (beta, nsamples) in zip(betas, sample_counts)
        grid = logarithmic_time_grid(
            0.01, beta / 2; nsteps_per_panel=panel_steps)
        representation = METTS(
            ; rng, collapse_basis=:computational, burnin,
            nsamples, thin=1)
        evolver = ImplicitLogTime(
            ; scheme=LogTrapezoid(), krylovdim=10, maxiter=1,
            tol=1e-8, fit_nsweeps=1, fit_tol=1e-8,
            energy_shift=true)
        started = time()
        trajectory = thermalize(
            representation, problem, beta;
            evolver, tau_grid=grid,
            initial_state=ground_state, collapse_initial=true,
            state_transform=transform)
        statistics = metts_statistics(trajectory, observable)
        push!(values, real(statistics.mean))
        push!(errors, statistics.stderr)
        push!(walltimes, time() - started)
        push!(maxbonds, maximum(
            max_bond_dimension(sample.state) for sample in trajectory.samples))
        @printf(
            "U=%.4g beta=%.4g D=%.10g stderr=%.3g wall=%.3fs maxbond=%d\n",
            U, beta, real(statistics.mean), statistics.stderr,
            walltimes[end], maxbonds[end])
    end
    return (; D0, values, errors, walltimes, maxbonds)
end

function paper_sample_counts(betas)
    first_count = 50_000.0
    last_count = 15_000.0
    return [
        round(Int, first_count +
            (last_count - first_count) *
            (log2(beta) - 1) / 9)
        for beta in betas
    ]
end

function acceptance_bath(full)
    if !full
        energies, couplings = semicircular_bath(1)
        return energies, ComplexF64.(couplings), nothing
    end
    default_path = joinpath(
        @__DIR__, "data", "kondo_semicircular_bath_29.csv")
    path = get(ENV, "GRAFT_KONDO_BATH_CSV", default_path)
    energies, couplings = read_bath_csv(path)
    length(energies) == 29 ||
        error("paper Kondo bath must contain exactly 29 sites")
    report = validate_semicircular_bath(
        energies, couplings;
        omega_min=pi / 1024, omega_max=100,
        npoints=4096, tolerance=1e-6)
    report.accepted || error(
        "paper Kondo bath failed pointwise hybridization gate: " *
        "max_error=$(report.max_error) at omega=$(report.worst_frequency)")
    return energies, couplings, report
end

function main()
    full = lowercase(get(ENV, "GRAFT_KONDO_FULL", "false")) in
        ("1", "true", "yes", "on")
    interactions = full ? [8.0, 10.0, 12.0, 15.0] : [2.0]
    betas = full ? Float64[2^k for k in 1:10] : [0.04]
    energies, couplings, bath_report = acceptance_bath(full)
    sample_override = get(ENV, "GRAFT_KONDO_NSAMPLES", "")
    sample_counts = isempty(sample_override) ?
        (full ? paper_sample_counts(betas) : fill(1, length(betas))) :
        fill(parse(Int, sample_override), length(betas))
    burnin = parse(Int, get(
        ENV, "GRAFT_KONDO_BURNIN", full ? "1000" : "0"))
    maxdim = parse(Int, get(
        ENV, "GRAFT_KONDO_MAXDIM", full ? "200" : "4"))
    ground_sweeps = full ? 4 : 1
    panel_steps = full ? 4 : 1
    output_path = get(
        ENV, "GRAFT_KONDO_OUTPUT", "kondo_scaling_acceptance.csv")
    dry_run = lowercase(get(ENV, "GRAFT_KONDO_DRY_RUN", "false")) in
        ("1", "true", "yes", "on")
    if dry_run
        println("mode=", full ? "paper" : "smoke",
                " nbath=", length(energies),
                " interactions=", join(interactions, ';'),
                " betas=", join(betas, ';'),
                " nsamples=", join(sample_counts, ';'),
                " burnin=", burnin,
                " maxdim=", maxdim,
                " dry_run=true")
        bath_report === nothing || @printf(
            "bath max error: %.6e at omega %.6e\n",
            bath_report.max_error, bath_report.worst_frequency)
        return nothing
    end
    rng = Xoshiro(260602930)

    curves = Vector{Vector{Float64}}()
    ground = Float64[]
    records = NamedTuple[]
    for U in interactions
        result = run_curve(
            U, betas, energies, couplings, sample_counts,
            burnin, maxdim, ground_sweeps, panel_steps, rng)
        push!(curves, result.values)
        push!(ground, result.D0)
        for i in eachindex(betas)
            push!(records, (;
                U, beta=betas[i], temperature=inv(betas[i]),
                double_occupancy=result.values[i],
                stderr=result.errors[i], D0=result.D0,
                nsamples=sample_counts[i],
                walltime=result.walltimes[i],
                maxbond=result.maxbonds[i]))
        end
    end

    open(output_path, "w") do io
        println(io,
            "U,beta,temperature,double_occupancy,stderr,D0,nsamples,walltime,maxbond")
        for row in records
            @printf(io, "%.8g,%.8g,%.12g,%.12g,%.12g,%.12g,%d,%.6f,%d\n",
                    row.U, row.beta, row.temperature,
                    row.double_occupancy, row.stderr, row.D0,
                    row.nsamples, row.walltime, row.maxbond)
        end
    end

    accepted = false
    try
        scaling = fit_kondo_scaling(
            interactions, inv.(betas), curves, ground;
            low_points=min(3, length(betas)))
        accepted = full && abs(scaling.exponent - 0.21) <= 0.05 &&
                   scaling.r2 >= 0.95
        @printf("Kondo exponent: %.6f (paper target 0.21), R2=%.6f\n",
                scaling.exponent, scaling.r2)
    catch err
        println("Scaling fit unavailable for this run: ", sprint(showerror, err))
    end
    println("mode=", full ? "paper" : "smoke",
            " nbath=", length(energies),
            " nsamples=", join(sample_counts, ';'),
            " maxdim=", maxdim, " accepted=", accepted)
    bath_report === nothing || @printf(
        "bath max error: %.6e at omega %.6e\n",
        bath_report.max_error, bath_report.worst_frequency)
    println("records=", output_path)
    full && !accepted &&
        error("paper-scale Kondo acceptance gate failed")
    return nothing
end

main()
