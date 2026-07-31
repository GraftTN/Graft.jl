# Minimally entangled typical thermal states.

"""
    METTSSample

One normalized METTS after imaginary-time preparation. `outcomes` records the
local collapse outcome that seeded this sample, in physical-site order.
"""
struct METTSSample{S<:ElementarySpace,T<:Number}
    state::TTNS{S,T}
    beta::Float64
    log_weight::Float64
    outcomes::Dict{Symbol,Int}
    chain_step::Int
    basis::Symbol
end

"""
    METTSTrajectory

Completed serial Markov chain. The copied RNG state and final product state are
sufficient to resume the chain without invalidating completed samples.
"""
struct METTSTrajectory{S<:ElementarySpace,T<:Number,R<:AbstractRNG}
    samples::Vector{METTSSample{S,T}}
    final_product::TTNS{S,T}
    final_outcomes::Dict{Symbol,Int}
    rng::R
    beta::Float64
    burnin::Int
    thin::Int
    total_steps::Int
    metadata::NamedTuple
end

"""
    DistributedMETTSTrajectory

One independently seeded METTS chain per rank. `local_chain` contains only this
rank's samples; observables and statistics perform explicit collectives
through `context`.
"""
struct DistributedMETTSTrajectory{L<:METTSTrajectory,
                                  C<:AbstractDistributedContext}
    local_chain::L
    context::C
    global_nsamples::Int
    samples_per_rank::Vector{Int}
    metadata::NamedTuple
end

"""Autocorrelation-aware scalar statistics for a METTS observable."""
struct METTSStatistics{T<:Number}
    mean::T
    variance::Float64
    stderr::Float64
    tau_int::Float64
    effective_samples::Float64
    nsamples::Int
end

"""
    metts_statistics(values; maxlag=nothing)

Estimate mean, variance, integrated autocorrelation time, effective sample
count, and standard error. The positive-sequence estimator stops at the first
nonpositive normalized autocorrelation.
"""
function metts_statistics(values::AbstractVector{<:Number};
                          maxlag::Union{Nothing,Integer}=nothing)
    n = length(values)
    n >= 2 || throw(ArgumentError("METTS statistics require at least two samples"))
    limit = maxlag === nothing ? min(n - 1, max(1, floor(Int, sqrt(n)))) :
            Int(maxlag)
    0 <= limit < n || throw(ArgumentError("maxlag must lie in 0:$(n - 1)"))
    mu = sum(values) / n
    centered = values .- mu
    variance = Float64(sum(abs2, centered) / (n - 1))
    if iszero(variance)
        return METTSStatistics(mu, 0.0, 0.0, 0.5, Float64(n), n)
    end
    covariance0 = sum(abs2, centered) / n
    tau = 0.5
    for lag in 1:limit
        covariance = real(sum(conj(centered[i]) * centered[i + lag]
                              for i in 1:(n - lag))) / (n - lag)
        rho = covariance / covariance0
        rho > 0 || break
        tau += rho
    end
    neff = clamp(n / (2tau), 1.0, Float64(n))
    return METTSStatistics(mu, variance, sqrt(variance / neff), tau, neff, n)
end

"""
    thermalize(rep::METTS, problem, beta; evolver, tau_grid=:uniform,
               nsteps=nothing, initial_state=nothing,
               collapse_initial=false)

Run a serial METTS Markov chain on the original physical topology. Every chain
step prepares `exp(-beta*K/2)|i>`, normalizes it, and performs a complete local
Born collapse to seed the next step. Burn-in and thinning are applied before
states enter the returned trajectory. With `collapse_initial=true`, the
supplied `initial_state` is first Born-sampled into a product state; this
supports ground-state-seeded chains.
"""
function thermalize(rep::METTS, problem::PurificationProblem, beta::Real;
                    evolver::Evolver,
                    tau_grid=:uniform,
                    nsteps=nothing,
                    initial_state=nothing,
                    collapse_initial::Bool=false,
                    resume_from=nothing,
                    state_transform=identity,
                    distributed::Union{Nothing,AbstractDistributedContext}=nothing)
    distributed === nothing || return _distributed_metts(
        rep, problem, beta, distributed;
        evolver, tau_grid, nsteps, initial_state, collapse_initial,
        resume_from, state_transform)
    beta >= 0 || throw(ArgumentError("beta must be nonnegative"))
    _check_evolver_no_normalize(evolver)
    isempty(problem.pp_ancilla_of) || throw(ArgumentError(
        "METTS for constrained PP pairs requires a logical-pair collapse basis"))
    basis_kind = _check_collapse_basis(rep.collapse_basis)
    initial_state === nothing || resume_from === nothing ||
        throw(ArgumentError("initial_state and resume_from are mutually exclusive"))
    collapse_initial && initial_state === nothing &&
        throw(ArgumentError("collapse_initial requires initial_state"))
    resumed = resume_from !== nothing
    if resumed
        _validate_metts_resume(resume_from, problem, beta, basis_kind, :metts)
    end
    rng = resumed ? copy(resume_from.rng) : rep.rng
    current, current_outcomes = if resumed
        copy(resume_from.final_product), copy(resume_from.final_outcomes)
    elseif initial_state === nothing
        _random_product_state(rng, problem, basis_kind, 1)
    else
        topology(initial_state) == problem.topo_orig ||
            throw(ArgumentError("initial METTS state has the wrong topology"))
        collapse_initial ?
            _collapse_to_product(
                rng, initial_state, problem.phys_orig, basis_kind, 1) :
            (copy(initial_state), Dict{Symbol,Int}())
    end

    start_step = resumed ? resume_from.total_steps : 0
    run_burnin = resumed ? 0 : rep.burnin
    run_steps = run_burnin + rep.nsamples * rep.thin
    total_steps = start_step + run_steps
    saved = resumed ? copy(resume_from.samples) : METTSSample[]
    previous_count = length(saved)
    for local_step in 1:run_steps
        chain_step = start_step + local_step
        seed = state_transform(copy(current))
        seed isa TTNS && topology(seed) == problem.topo_orig ||
            throw(ArgumentError("METTS state_transform must return a TTNS on the original topology"))
        seed_norm = norm(seed)
        isfinite(seed_norm) && seed_norm > 0 ||
            throw(ArgumentError(
                "METTS state_transform produced a zero or nonfinite state"))
        normalize!(seed)
        sample_state, log_weight = _prepare_metts(
            seed, problem.K_orig, Float64(beta), evolver, tau_grid, nsteps)
        if local_step > run_burnin &&
           (local_step - run_burnin - 1) % rep.thin == 0
            phase = _collapse_phase(basis_kind, chain_step)
            push!(saved, METTSSample(
                copy(sample_state), Float64(beta), 2log_weight,
                copy(current_outcomes), chain_step, phase))
        end
        current, current_outcomes = _collapse_to_product(
            rng, sample_state, problem.phys_orig, basis_kind, chain_step + 1)
    end
    length(saved) == previous_count + rep.nsamples ||
        throw(AssertionError("METTS sampling schedule produced the wrong sample count"))
    typed_samples = convert(Vector{typeof(first(saved))}, saved)
    metadata = (;
        representation=:metts,
        problem_hash=hash(problem.topo_orig),
        collapse_basis=basis_kind,
        symmetry_ergodicity=_symmetry_ergodicity(problem.phys_orig, basis_kind),
    )
    return METTSTrajectory(
        typed_samples, current, current_outcomes, copy(rng), Float64(beta),
        resumed ? resume_from.burnin : rep.burnin,
        rep.thin, total_steps, metadata)
end

"""
    thermalize(rep::HybridMETTS, problem, beta; evolver,
               tau_grid=:uniform, nsteps=nothing)

Run partial-purification METTS. `sampled_sites` are collapsed; every other
logical group is reset to its canonical infinite-temperature coevaluation
before each imaginary-time preparation. This reset prevents repeated thermal
evolution of the uncollapsed subsystem.
"""
function thermalize(rep::HybridMETTS, problem::PurificationProblem, beta::Real;
                    evolver::Evolver,
                    tau_grid=:uniform,
                    nsteps=nothing,
                    resume_from=nothing,
                    state_transform=identity,
                    distributed::Union{Nothing,AbstractDistributedContext}=nothing)
    distributed === nothing || return _distributed_metts(
        rep, problem, beta, distributed;
        evolver, tau_grid, nsteps, resume_from, state_transform)
    beta >= 0 || throw(ArgumentError("beta must be nonnegative"))
    _check_evolver_no_normalize(evolver)
    basis_kind = _check_collapse_basis(rep.collapse_basis)
    sampled_sites = _validate_hybrid_sites(problem, rep.sampled_sites)
    resumed = resume_from !== nothing
    if resumed
        _validate_metts_resume(
            resume_from, problem, beta, basis_kind, :hybrid_metts;
            sampled_sites)
    end
    rng = resumed ? copy(resume_from.rng) : rep.rng
    current_outcomes = resumed ? copy(resume_from.final_outcomes) :
        _random_outcomes(rng, problem.phys_doubled, sampled_sites, basis_kind, 1)
    current = resumed ? copy(resume_from.final_product) :
        _hybrid_seed(problem, sampled_sites, current_outcomes, basis_kind, 1)

    start_step = resumed ? resume_from.total_steps : 0
    run_burnin = resumed ? 0 : rep.burnin
    run_steps = run_burnin + rep.nsamples * rep.thin
    total_steps = start_step + run_steps
    saved = resumed ? copy(resume_from.samples) : METTSSample[]
    previous_count = length(saved)
    for local_step in 1:run_steps
        chain_step = start_step + local_step
        seed = state_transform(copy(current))
        seed isa TTNS && topology(seed) == problem.topo_doubled ||
            throw(ArgumentError("HybridMETTS state_transform must return a TTNS on the doubled topology"))
        seed_norm = norm(seed)
        isfinite(seed_norm) && seed_norm > 0 ||
            throw(ArgumentError(
                "HybridMETTS state_transform produced a zero or nonfinite state"))
        normalize!(seed)
        sample_state, log_weight = _prepare_metts(
            seed, problem.K, Float64(beta), evolver, tau_grid, nsteps)
        if local_step > run_burnin &&
           (local_step - run_burnin - 1) % rep.thin == 0
            push!(saved, METTSSample(
                copy(sample_state), Float64(beta), 2log_weight,
                copy(current_outcomes), chain_step,
                _collapse_phase(basis_kind, chain_step)))
        end
        next_step = chain_step + 1
        _, current_outcomes = _collapse_descriptors(
            rng, sample_state, problem.phys_doubled, sampled_sites,
            basis_kind, next_step)
        current = _hybrid_seed(
            problem, sampled_sites, current_outcomes, basis_kind, next_step)
    end
    length(saved) == previous_count + rep.nsamples ||
        throw(AssertionError("HybridMETTS sampling schedule produced the wrong sample count"))
    typed_samples = convert(Vector{typeof(first(saved))}, saved)
    purified_sites = sort!(
        setdiff(collect(keys(problem.phys_orig)), sampled_sites); by=string)
    metadata = (;
        representation=:hybrid_metts,
        problem_hash=hash(problem.topo_orig),
        collapse_basis=basis_kind,
        sampled_sites=copy(sampled_sites),
        purified_sites,
        resets_purified_groups=true,
        symmetry_ergodicity=_symmetry_ergodicity(
            Dict(site => problem.phys_orig[site] for site in sampled_sites),
            basis_kind),
    )
    return METTSTrajectory(
        typed_samples, current, current_outcomes, copy(rng), Float64(beta),
        resumed ? resume_from.burnin : rep.burnin,
        rep.thin, total_steps, metadata)
end

function _validate_metts_resume(trajectory::METTSTrajectory,
                                problem::PurificationProblem,
                                beta, basis, representation;
                                sampled_sites=nothing)
    trajectory.metadata.representation === representation ||
        throw(ArgumentError("cannot resume $representation from $(trajectory.metadata.representation)"))
    trajectory.metadata.problem_hash == hash(problem.topo_orig) ||
        throw(ArgumentError("METTS checkpoint belongs to a different problem"))
    trajectory.metadata.collapse_basis === basis ||
        throw(ArgumentError("collapse basis differs from the checkpoint"))
    isapprox(trajectory.beta, Float64(beta); atol=0, rtol=0) ||
        throw(ArgumentError("beta differs from the METTS checkpoint"))
    if sampled_sites !== nothing
        trajectory.metadata.sampled_sites == sampled_sites ||
            throw(ArgumentError("HybridMETTS sampled_sites differ from the checkpoint"))
    end
    return trajectory
end

function thermal_expect(traj::METTSTrajectory, O::TTNO)
    isempty(traj.samples) && throw(ArgumentError("METTS trajectory has no samples"))
    topology(O) == topology(first(traj.samples).state) ||
        throw(ArgumentError("METTS observable has the wrong topology; use physical_ttno(...; doubled=false)"))
    return sum(expect(sample.state, O) for sample in traj.samples) /
           length(traj.samples)
end

function metts_statistics(traj::METTSTrajectory, O::TTNO; kwargs...)
    topology(O) == topology(first(traj.samples).state) ||
        throw(ArgumentError("METTS observable has the wrong topology"))
    return metts_statistics([expect(sample.state, O) for sample in traj.samples];
                            kwargs...)
end

function _distributed_sample_counts(nsamples::Int, size::Int)
    nsamples >= size || throw(ArgumentError(
        "distributed METTS needs at least one sample per rank"))
    quotient, remainder = divrem(nsamples, size)
    return [quotient + (rank <= remainder ? 1 : 0) for rank in 1:size]
end

function _rank_rng(rng::AbstractRNG, rank::Int)
    base = rand(copy(rng), UInt64)
    stream = base ⊻ (UInt64(rank + 1) * 0x9e3779b97f4a7c15)
    return Xoshiro(stream)
end

function _distributed_metts(
        rep::METTS, problem, beta, context;
        evolver, tau_grid, nsteps, initial_state, collapse_initial,
        resume_from, state_transform)
    rank = distributed_rank(context)
    size = distributed_size(context)
    counts = _distributed_sample_counts(rep.nsamples, size)
    local_resume = resume_from === nothing ? nothing : begin
        resume_from isa DistributedMETTSTrajectory || throw(ArgumentError(
            "distributed METTS can only resume a distributed trajectory"))
        distributed_size(resume_from.context) == size || throw(ArgumentError(
            "distributed METTS checkpoint has a different rank count"))
        resume_from.local_chain
    end
    local_rep = METTS(
        _rank_rng(rep.rng, rank), rep.collapse_basis, rep.burnin,
        counts[rank + 1], rep.thin)
    chain = thermalize(
        local_rep, problem, beta;
        evolver, tau_grid, nsteps, initial_state, collapse_initial,
        resume_from=local_resume, state_transform)
    return distributed_trajectory(chain, context)
end

function _distributed_metts(
        rep::HybridMETTS, problem, beta, context;
        evolver, tau_grid, nsteps, resume_from, state_transform)
    rank = distributed_rank(context)
    size = distributed_size(context)
    counts = _distributed_sample_counts(rep.nsamples, size)
    local_resume = resume_from === nothing ? nothing : begin
        resume_from isa DistributedMETTSTrajectory || throw(ArgumentError(
            "distributed HybridMETTS can only resume a distributed trajectory"))
        distributed_size(resume_from.context) == size || throw(ArgumentError(
            "distributed HybridMETTS checkpoint has a different rank count"))
        resume_from.local_chain
    end
    local_rep = HybridMETTS(
        _rank_rng(rep.rng, rank), rep.sampled_sites, rep.collapse_basis,
        rep.burnin, counts[rank + 1], rep.thin)
    chain = thermalize(
        local_rep, problem, beta;
        evolver, tau_grid, nsteps, resume_from=local_resume, state_transform)
    return distributed_trajectory(chain, context)
end

function distributed_trajectory(chain::METTSTrajectory, context)
    rank = distributed_rank(context)
    size = distributed_size(context)
    counts = zeros(Int, size)
    counts[rank + 1] = length(chain.samples)
    distributed_allreduce_sum!(context, counts)
    metadata = merge(chain.metadata, (;
        distributed=true,
        nranks=size,
        samples_per_rank=copy(counts),
    ))
    return DistributedMETTSTrajectory(
        chain, context, sum(counts), counts, metadata)
end

function thermal_expect(traj::DistributedMETTSTrajectory, O::TTNO)
    local_values = [
        expect(sample.state, O) for sample in traj.local_chain.samples]
    buffer = ComplexF64[sum(local_values), length(local_values)]
    distributed_allreduce_sum!(traj.context, buffer)
    real(buffer[2]) > 0 ||
        throw(ArgumentError("distributed METTS trajectory has no samples"))
    return buffer[1] / real(buffer[2])
end

function metts_statistics(
        traj::DistributedMETTSTrajectory, O::TTNO; maxlag=nothing)
    local_values = [
        expect(sample.state, O) for sample in traj.local_chain.samples]
    chains = distributed_allgather(traj.context, local_values)
    return _metts_statistics_chains(chains; maxlag)
end

function _metts_statistics_chains(chains; maxlag=nothing)
    n = sum(length, chains)
    n >= 2 || throw(ArgumentError(
        "distributed METTS statistics require at least two samples"))
    max_chain = maximum(length, chains)
    limit = maxlag === nothing ?
        min(max_chain - 1, max(1, floor(Int, sqrt(n)))) : Int(maxlag)
    0 <= limit < max_chain || throw(ArgumentError(
        "maxlag must lie in 0:$(max_chain - 1) for independent chains"))
    mu = sum(sum, chains) / n
    variance_sum = sum(
        sum(abs2(value - mu) for value in chain) for chain in chains)
    variance = Float64(variance_sum / (n - 1))
    iszero(variance) &&
        return METTSStatistics(mu, 0.0, 0.0, 0.5, Float64(n), n)
    covariance0 = variance_sum / n
    tau = 0.5
    for lag in 1:limit
        covariance_sum = zero(real(mu))
        pairs = 0
        for chain in chains
            for i in 1:(length(chain) - lag)
                covariance_sum += real(
                    conj(chain[i] - mu) * (chain[i + lag] - mu))
                pairs += 1
            end
        end
        pairs == 0 && break
        rho = covariance_sum / pairs / covariance0
        rho > 0 || break
        tau += rho
    end
    neff = clamp(n / (2tau), 1.0, Float64(n))
    return METTSStatistics(
        mu, variance, sqrt(variance / neff), tau, neff, n)
end

function _prepare_metts(product::TTNS, K::TTNO, beta::Float64,
                        evolver::Evolver, tau_grid, nsteps)
    psi = copy(product)
    iszero(beta) && return psi, 0.0
    grid = _build_grid(beta, tau_grid, nsteps)
    ev = _fresh_evolver_thermal(evolver)
    log_amplitude = 0.0
    for i in 1:(length(grid) - 1)
        dtau = grid[i + 1] - grid[i]
        step!(ev, psi, K, -dtau)
        nrm = norm(psi)
        isfinite(nrm) && nrm > 0 ||
            throw(ArgumentError("METTS imaginary-time step produced zero or nonfinite norm"))
        normalize!(psi)
        log_amplitude += log(nrm)
    end
    return psi, log_amplitude
end

function _check_collapse_basis(basis)
    basis in (:computational, :fourier, :alternating) ||
        throw(ArgumentError("collapse_basis must be :computational, :fourier, or :alternating"))
    return basis
end

_collapse_phase(kind::Symbol, step::Int) =
    kind === :alternating ? (isodd(step) ? :computational : :fourier) : kind

function _collapse_to_product(rng::AbstractRNG, psi::TTNS,
                              phys::Dict{Symbol,S}, basis_kind::Symbol,
                              step::Int) where {S<:ElementarySpace}
    sites = [nodeid(topology(psi), n) for n in postorder(topology(psi))
             if haskey(phys, nodeid(topology(psi), n))]
    descriptors, outcomes = _collapse_descriptors(
        rng, psi, phys, sites, basis_kind, step)
    return _metts_product_state(eltype(psi), topology(psi), phys, descriptors),
           outcomes
end

function _collapse_descriptors(rng::AbstractRNG, psi::TTNS,
                               phys::Dict{Symbol,S}, sites,
                               basis_kind::Symbol,
                               step::Int) where {S<:ElementarySpace}
    phase = _collapse_phase(basis_kind, step)
    projected = copy(psi)
    descriptors = Dict{Symbol,Any}()
    outcomes = Dict{Symbol,Int}()
    for site in sites
        choices = _local_collapse_choices(eltype(psi), phys[site], phase)
        candidates = [
            apply_local(projected, projector, site)
            for (projector, _) in choices
        ]
        probabilities = Float64[max(0.0, norm(candidate)^2)
                                for candidate in candidates]
        total = sum(probabilities)
        isfinite(total) && total > 100eps(Float64) ||
            throw(ArgumentError("collapse probabilities vanished at $site"))
        probabilities ./= total
        chosen = _sample_categorical(rng, probabilities)
        _, descriptor = choices[chosen]
        projected = candidates[chosen]
        normalize!(projected)
        descriptors[site] = descriptor
        outcomes[site] = chosen
    end
    return descriptors, outcomes
end

function _random_product_state(rng::AbstractRNG, problem::PurificationProblem,
                               basis_kind::Symbol, step::Int)
    sites = sort!(collect(keys(problem.phys_orig)); by=string)
    outcomes = _random_outcomes(
        rng, problem.phys_orig, sites, basis_kind, step)
    descriptors = _descriptors_from_outcomes(
        problem.elt, problem.phys_orig, sites, outcomes, basis_kind, step)
    return _metts_product_state(problem.elt, problem.topo_orig,
                                problem.phys_orig, descriptors), outcomes
end

function _random_outcomes(rng::AbstractRNG, phys, sites,
                          basis_kind::Symbol, step::Int)
    phase = _collapse_phase(basis_kind, step)
    outcomes = Dict{Symbol,Int}()
    for site in sites
        choices = _local_collapse_choices(ComplexF64, phys[site], phase)
        outcomes[site] = rand(rng, eachindex(choices))
    end
    return outcomes
end

function _descriptors_from_outcomes(::Type{T}, phys, sites, outcomes,
                                    basis_kind::Symbol, step::Int) where {T<:Number}
    phase = _collapse_phase(basis_kind, step)
    descriptors = Dict{Symbol,Any}()
    for site in sites
        choices = _local_collapse_choices(T, phys[site], phase)
        chosen = outcomes[site]
        chosen in eachindex(choices) ||
            throw(ArgumentError("collapse outcome $chosen is invalid at $site"))
        descriptors[site] = choices[chosen][2]
    end
    return descriptors
end

function _hybrid_seed(problem::PurificationProblem, sampled_sites,
                      outcomes, basis_kind::Symbol, step::Int)
    phase = _collapse_phase(basis_kind, step)
    seed = infinite_temperature_state(problem).psi
    for site in sampled_sites
        choices = _local_collapse_choices(
            problem.elt, problem.phys_doubled[site], phase)
        chosen = outcomes[site]
        chosen in eachindex(choices) ||
            throw(ArgumentError("collapse outcome $chosen is invalid at $site"))
        seed = apply_local(seed, choices[chosen][1], site)
        normalize!(seed)
    end
    return seed
end

function _validate_hybrid_sites(problem::PurificationProblem, requested)
    physical = Set(keys(problem.phys_orig))
    all(site -> site in physical, requested) ||
        throw(ArgumentError("HybridMETTS sampled_sites must be original physical sites"))
    sampled = Set(requested)
    for group in problem.logical_groups
        overlap = count(site -> site in sampled, group)
        overlap in (0, length(group)) || throw(ArgumentError(
            "HybridMETTS must sample an entire logical group $(group)"))
    end
    length(sampled) < length(physical) || throw(ArgumentError(
        "HybridMETTS needs at least one purified physical site; use METTS when all sites are sampled"))
    return sort!(collect(sampled); by=string)
end

function _sample_categorical(rng::AbstractRNG, probabilities)
    target = rand(rng)
    cumulative = 0.0
    for i in eachindex(probabilities)
        cumulative += probabilities[i]
        target <= cumulative && return i
    end
    return lastindex(probabilities)
end

function _local_collapse_choices(::Type{T}, P::ComplexSpace,
                                 basis::Symbol) where {T<:Number}
    d = dim(P)
    vectors = _basis_vectors(T, d, basis)
    return [(TensorMap(v * v', P ← P), v) for v in vectors]
end

function _local_collapse_choices(::Type{T}, P::ElementarySpace,
                                 basis::Symbol) where {T<:Number}
    choices = Tuple{AbstractTensorMap,Any}[]
    for q in sectors(P)
        d = dim(P, q)
        for v in _basis_vectors(T, d, basis)
            projector = zeros(T, P ← P)
            for (sector, block_) in blocks(projector)
                sector == q && (block_ .= v * v')
            end
            push!(choices, (projector, (q, v)))
        end
    end
    return choices
end

function _basis_vectors(::Type{T}, d::Int, basis::Symbol) where {T<:Number}
    if basis === :computational || d == 1
        return [T[i == j ? 1 : 0 for i in 1:d] for j in 1:d]
    end
    CT = promote_type(T, ComplexF64)
    return [CT[exp(2pi * im * (row - 1) * (column - 1) / d) / sqrt(d)
               for row in 1:d] for column in 1:d]
end

function _metts_product_state(::Type{T}, topo::TreeTopology,
                              phys::Dict{Symbol,ComplexSpace},
                              descriptors) where {T<:Number}
    unit = oneunit(ComplexSpace)
    tensors = map(1:nnodes(topo)) do n
        K = nchildren(topo, n)
        vector = get(descriptors, nodeid(topo, n), nothing)
        cod = K == 0 ? one(unit) : reduce(⊗, ntuple(_ -> unit, K))
        if vector === nothing
            TensorMap(ones(T, ntuple(_ -> 1, K + 1)), cod ← unit)
        else
            array = reshape(T.(vector), ntuple(_ -> 1, K)..., length(vector), 1)
            TensorMap(array, cod ⊗ phys[nodeid(topo, n)] ← unit)
        end
    end
    return TTNS(topo, tensors, topo.root)
end

function _metts_product_state(::Type{T}, topo::TreeTopology,
                              phys::Dict{Symbol,S},
                              descriptors) where {T<:Number,S<:ElementarySpace}
    Q = sectortype(first(values(phys)))
    unitq = one(Q)
    qlocal = fill(unitq, nnodes(topo))
    for (site, descriptor) in descriptors
        qlocal[nodeindex(topo, site)] = descriptor[1]
    end
    qsub = fill(unitq, nnodes(topo))
    for n in postorder(topo)
        q = qlocal[n]
        for child in topo.children[n]
            q = only(q ⊗ qsub[child])
        end
        qsub[n] = q
    end
    edgespace(n) = Vect[Q](qsub[n] => 1)
    rootspace = Vect[Q](qsub[topo.root] => 1)
    tensors = map(1:nnodes(topo)) do n
        cods = S[edgespace(child) for child in topo.children[n]]
        site = nodeid(topo, n)
        haskey(phys, site) && push!(cods, phys[site])
        cod = isempty(cods) ? oneunit(S) : reduce(⊗, cods)
        dom = topo.parent[n] == 0 ? rootspace : edgespace(n)
        tensor = zeros(T, cod ← dom)
        codcoord = _metts_basis_coord(cods, unitq)
        domcoord = _metts_basis_coord(S[dom], unitq)
        child_indices = ntuple(_ -> 1, nchildren(topo, n))
        if haskey(phys, site)
            q, vector = descriptors[site]
            offset = sum(
                (dim(phys[site], sector) for sector in sectors(phys[site])
                 if _sector_before(phys[site], sector, q));
                init=0)
            for local_index in eachindex(vector)
                codindex = (child_indices..., offset + local_index)
                cq, row = codcoord[codindex]
                dq, column = domcoord[(1,)]
                cq == dq && _set_block_entry!(tensor, cq, row, column,
                                              T(vector[local_index]))
            end
        else
            cq, row = codcoord[child_indices]
            dq, column = domcoord[(1,)]
            cq == dq && _set_block_entry!(tensor, cq, row, column, one(T))
        end
        tensor
    end
    return TTNS(topo, tensors, topo.root)
end

function _sector_before(P, candidate, target)
    for q in sectors(P)
        q == target && return false
        q == candidate && return true
    end
    return false
end

function _metts_basis_coord(legs::Vector{S}, unitq) where {S<:ElementarySpace}
    coord = Dict{Tuple,Tuple{typeof(unitq),Int}}()
    isempty(legs) && (coord[()] = (unitq, 1); return coord)
    legsectors = [collect(sectors(V)) for V in legs]
    offsets = map(legs) do V
        result = Dict{typeof(unitq),Int}()
        offset = 0
        for q in sectors(V)
            result[q] = offset
            offset += dim(V, q)
        end
        result
    end
    rows = Dict{typeof(unitq),Int}()
    for sector_indices in CartesianIndices(Tuple(length.(legsectors)))
        selected = ntuple(j -> legsectors[j][sector_indices[j]], length(legs))
        fused = unitq
        for q in selected
            fused = only(fused ⊗ q)
        end
        for degeneracies in CartesianIndices(
                ntuple(j -> dim(legs[j], selected[j]), length(legs)))
            index = ntuple(j -> offsets[j][selected[j]] + degeneracies[j],
                           length(legs))
            row = get(rows, fused, 0) + 1
            rows[fused] = row
            coord[index] = (fused, row)
        end
    end
    return coord
end

function _set_block_entry!(tensor, q, row, column, value)
    for (sector, block_) in blocks(tensor)
        sector == q && (block_[row, column] = value)
    end
    return tensor
end

function _symmetry_ergodicity(phys, basis)
    first_space = first(values(phys))
    spacetype(first_space) === ComplexSpace && return :alternating_local_bases
    basis === :computational && return :fixed_global_sector
    any(dim(P, q) > 1 for P in values(phys) for q in sectors(P)) &&
        return :within_sector_rotations
    return :fixed_global_sector
end
