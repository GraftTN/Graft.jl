# Thermal driver: thermalize, thermal_expect, thermal_correlator.
# Implements §05 plan §2.4–§2.5.

using ..Evolution: CorrelatorSeries

"""
    thermalize(rep::Purified, problem::PurificationProblem, beta;
               evolver, tau_grid=:uniform, nsteps=nothing,
               save_betas=Float64[]) -> PurificationTrajectory

Prepare the thermal state `|Ψ_β⟩ = (e^{-βK/2} ⊗ I_a)|I⟩` by delegating all
propagation to the supplied `Evolver` through imaginary-time steps
`step!(ev, ψ, K, -δτ)`.

The driver owns **all** normalization: after each step the TTNS is renormalized
and the removed norm is accumulated in `log_amplitude`. At completion,
`logZ = log_hilbert_dim + 2 * log_amplitude`.

Grid contract (propagation time = β/2):
- `beta ≥ 0`; `beta == 0` returns the canonical `|I⟩`
- `tau_grid=:uniform` with positive `nsteps` uses `range(0, beta/2; length=nsteps+1)`
- an explicit real vector must start at `0`, end at `beta/2`, be strictly increasing
- `save_betas` requests snapshots at physical inverse temperatures `b` (mapped
  to propagation time `b/2`); off-grid requests use one fractional step from the
  nearest preceding grid point

Norm-bookkeeping: an evolver that rescales inside `step!` corrupts
`log_amplitude`. Detectable cases (`hasproperty(ev, :normalize) && ev.normalize`)
are rejected with `ArgumentError`.
"""
function thermalize(rep::Purified, problem::PurificationProblem, beta::Real;
                    evolver::Evolver, tau_grid=:uniform, nsteps=nothing,
                    save_betas=Float64[])
    _validate_aux_evolution(rep.aux_evolution)
    _check_evolver_no_normalize(evolver)
    beta >= 0 || throw(ArgumentError("beta must be nonnegative"))

    state0 = infinite_temperature_state(problem)

    if beta == 0
        return PurificationTrajectory(
            state0, Dict{Float64,PurifiedState}(0.0 => state0), [0.0],
            (; evolver_type=typeof(evolver), problem_hash=hash(problem.topo_orig),))
    end

    grid = _build_grid(Float64(beta), tau_grid, nsteps)
    psi = copy(state0.psi)
    log_amp = 0.0
    K = problem.K
    ev = _fresh_evolver_thermal(evolver)

    grid_states = Dict{Float64,Tuple{TTNS,Float64}}()
    grid_states[0.0] = (copy(psi), 0.0)

    for i in 1:(length(grid) - 1)
        dtau = grid[i + 1] - grid[i]
        iszero(dtau) && continue
        step!(ev, psi, K, -dtau)
        nrm = norm(psi)
        normalize!(psi)
        log_amp += log(nrm)
        grid_states[grid[i + 1]] = (copy(psi), log_amp)
    end

    logZ_final = problem.log_hilbert_dim + 2 * log_amp
    final_state = PurifiedState(
        psi, Float64(beta), log_amp, logZ_final,
        (; problem_hash=hash(problem.topo_orig),))

    checkpoints = Dict{Float64,PurifiedState}()
    checkpoints[0.0] = state0
    checkpoints[Float64(beta)] = final_state

    all_saves = sort(unique(Float64.(save_betas)))
    for b in all_saves
        b == 0.0 && continue
        b == Float64(beta) && continue
        0 <= b <= beta || throw(ArgumentError("save_beta $b outside [0, $beta]"))

        t_target = b / 2
        t_prev = maximum(t for t in grid if t <= t_target + 1e-14)

        if abs(t_prev - t_target) < 1e-14
            psi_save, la_save = grid_states[t_prev]
        else
            psi_prev, la_prev = grid_states[t_prev]
            psi_save = copy(psi_prev)
            la_save = la_prev
            dtau_off = t_target - t_prev
            ev_off = _fresh_evolver_thermal(evolver)
            step!(ev_off, psi_save, K, -dtau_off)
            n_off = norm(psi_save)
            normalize!(psi_save)
            la_save += log(n_off)
        end
        logZ_save = problem.log_hilbert_dim + 2 * la_save
        checkpoints[b] = PurifiedState(
            psi_save, b, la_save, logZ_save,
            (; problem_hash=hash(problem.topo_orig),))
    end

    return PurificationTrajectory(
        final_state, checkpoints, grid,
        (; evolver_type=typeof(evolver), problem_hash=hash(problem.topo_orig),))
end

"""
    thermal_realtime_correlator(rep, problem, A, B, times;
        evolver, trajectory, aux_hamiltonian=nothing,
        aux_evolver=evolver, metadata=(;)) -> CorrelatorSeries

Compute `tr(rho * A(t) * B) / Z` by evolving both the thermal reference and
`B|Psi_beta>` with the physical Hamiltonian. For
`Purified(aux_evolution=:backward)`, both states additionally evolve with
`+im * H_aux * dt` on ancillas. This common ancilla unitary leaves the
correlator invariant and implements the Karrasch-Barthel gauge evolution.

Neutral generators get `H_aux = transpose(H)` automatically. Generators with
charged local factors require an explicit `aux_hamiltonian`, because a
factorwise transpose is not sufficient to certify fermionic braiding signs.
Passing a TTNO directly as `rep.aux_evolution` is the custom-Hamiltonian form.
"""
function thermal_realtime_correlator(
        rep::Purified, problem::PurificationProblem, A, B, times;
        evolver::Evolver,
        trajectory::PurificationTrajectory,
        aux_hamiltonian=nothing,
        aux_evolver::Evolver=evolver,
        metadata::NamedTuple=(;))
    _validate_trajectory(trajectory, problem, trajectory.final.beta, evolver)
    eltype(trajectory.final.psi) <: Complex ||
        throw(ArgumentError("finite-temperature real-time evolution requires a complex-eltype state"))
    Asite, Aop = _local_insertion_thermal(A)
    Bsite, Bop = _local_insertion_thermal(B)
    values_t = collect(times)
    all(t -> isreal(t) && real(t) >= 0, values_t) ||
        throw(ArgumentError("real-time points must be real and nonnegative"))
    all(diff(real.(values_t)) .>= 0) ||
        throw(ArgumentError("real-time points must be nondecreasing"))

    reference = copy(trajectory.final.psi)
    inserted = apply_local(trajectory.final.psi, Bop, Bsite)
    physical_ev_ref = _fresh_evolver_thermal(evolver)
    physical_ev_inserted = _fresh_evolver_thermal(evolver)
    Haux = _resolve_aux_hamiltonian(rep, problem, aux_hamiltonian)
    aux_ev_ref = Haux === nothing ? nothing : _fresh_evolver_thermal(aux_evolver)
    aux_ev_inserted = Haux === nothing ? nothing : _fresh_evolver_thermal(aux_evolver)

    vals = Vector{ComplexF64}(undef, length(values_t))
    previous = 0.0
    for i in eachindex(values_t)
        current = Float64(real(values_t[i]))
        dt = current - previous
        if !iszero(dt)
            step!(physical_ev_ref, reference, problem.K, -im * dt)
            step!(physical_ev_inserted, inserted, problem.K, -im * dt)
            if Haux !== nothing
                step!(aux_ev_ref, reference, Haux, +im * dt)
                step!(aux_ev_inserted, inserted, Haux, +im * dt)
            end
        end
        acted = apply_local(inserted, Aop, Asite)
        vals[i] = inner(reference, acted)
        previous = current
    end
    meta = merge(metadata, (;
        beta=trajectory.final.beta,
        Asite,
        Bsite,
        aux_evolution=Haux === nothing ? :none : :backward,
        evolver_type=typeof(evolver),
        aux_evolver_type=Haux === nothing ? Nothing : typeof(aux_evolver),
    ))
    return CorrelatorSeries(Float64.(real.(values_t)), vals, meta)
end

function _validate_aux_evolution(mode)
    mode === :none && return :none
    mode === :backward && return :backward
    mode isa TTNO && return :custom
    throw(ArgumentError(
        "aux_evolution must be :none, :backward, or a custom auxiliary TTNO"))
end

function _resolve_aux_hamiltonian(rep::Purified,
                                  problem::PurificationProblem,
                                  supplied)
    mode = _validate_aux_evolution(rep.aux_evolution)
    if mode === :none
        supplied === nothing || throw(ArgumentError(
            "aux_hamiltonian was supplied but aux_evolution=:none"))
        return nothing
    elseif mode === :custom
        supplied === nothing || throw(ArgumentError(
            "custom auxiliary Hamiltonian was supplied twice"))
        Haux = rep.aux_evolution
    elseif supplied !== nothing
        Haux = supplied
    else
        Haux = _automatic_aux_hamiltonian(problem)
    end
    Haux isa TTNO ||
        throw(ArgumentError("auxiliary Hamiltonian must be a TTNO"))
    topology(Haux) == problem.topo_doubled ||
        throw(ArgumentError("auxiliary Hamiltonian has the wrong topology"))
    return Haux
end

function _automatic_aux_hamiltonian(problem::PurificationProblem)
    auxiliary = OpSum()
    for term in problem.generator
        mapped = SiteOp[]
        for factor in term.ops
            (numin(factor.op) == 1 && numout(factor.op) == 1) || throw(ArgumentError(
                "automatic aux_evolution=:backward supports only neutral local factors; " *
                "pass aux_hamiltonian explicitly for charged/fermionic generators"))
            ancilla = get(problem.ancilla_of, factor.site, nothing)
            ancilla === nothing && throw(ArgumentError(
                "no thermal ancilla is associated with $(factor.site)"))
            Paux = problem.phys_doubled[ancilla]
            matrix = collect(transpose(convert(Array, factor.op)))
            auxop = try
                TensorMap(matrix, Paux ← Paux)
            catch err
                throw(ArgumentError(
                    "cannot transpose local factor $(factor.name) onto ancilla $ancilla; " *
                    "pass aux_hamiltonian explicitly (cause: $(sprint(showerror, err)))"))
            end
            push!(mapped, SiteOp(ancilla, Symbol(factor.name, :_auxT), auxop))
        end
        auxiliary += Term(term.coeff, mapped)
    end
    return ttno_from_opsum(
        auxiliary, problem.topo_doubled, problem.phys_doubled;
        hermitian=problem.hermitian, elt=problem.elt)
end

"""
    thermal_expect(state::PurifiedState, O::TTNO) -> Number

Thermal expectation `⟨O⟩_β = ⟨Ψ_β|O|Ψ_β⟩ / ⟨Ψ_β|Ψ_β⟩`. Since `state.psi` is
normalized and carries the full thermal state (ancillas carry identity for
physical operators), this is just `expect(state.psi, O)`.
"""
function thermal_expect(state::PurifiedState, O::TTNO)
    return expect(state.psi, O)
end

"""Convenience: evaluate on the final state of a trajectory."""
thermal_expect(traj::PurificationTrajectory, O::TTNO) = thermal_expect(traj.final, O)

"""
    thermal_correlator(rep::Purified, problem::PurificationProblem,
                       A, B, beta, taus;
                       evolver, prep_grid=:uniform, prep_nsteps=nothing,
                       prop_grid=:uniform, prop_nsteps=nothing,
                       prop_max_step=nothing,
                       trajectory=nothing, connected=false,
                       metadata=(;), threaded=false,
                       minbatch=2,
                       task_memory_cap_bytes=nothing,
                       task_workspace_memory_bytes=nothing) -> CorrelatorSeries

Thermal correlator `C_AB(τ) = tr(e^{-(β-τ)K} A e^{-τK} B) / Z` using the stable
β-τ preparation formula (§05 plan §1.3, §2.5).

For each `τ`:
1. obtain the normalized saved state `|ψ_b⟩` at `b=β-τ`
2. when `connected=true`, replace the insertions by
   `δA = A - ⟨A⟩_β I` and `δB = B - ⟨B⟩_β I`
3. build `bra = δA†|ψ_b⟩` and `ket = δB|ψ_b⟩` (or use `A`, `B` for the
   unconnected correlator)
4. normalize `ket` into `ScaledTTNS`, propagate by `e^{-τK}`
5. evaluate `C = e^{2ℓ_b+ℓ_k-2ℓ_β} ⟨δA†ψ_b|k_τ⟩`

Thus a connected correlator is measured directly in the fluctuation sector;
the implementation never forms `C_AB(τ) - ⟨A⟩⟨B⟩` from two extensive or
nearly equal final results.  The centers are thermal means of the supplied
operators, not model-specific constants such as `N - 1`.

`A` and `B` are `site => op` local insertions. The returned series does NOT
include a fermionic minus sign; the caller constructs `G(τ) = -C_{d,d†}(τ)`
explicitly.

Threaded execution is fail-closed. `task_memory_cap_bytes` bounds retained
checkpoints/results plus every concurrently admitted item.
`task_workspace_memory_bytes` is the measured conservative scratch allowance
for one evolver/solver item; because an arbitrary `Evolver` does not expose a
generic workspace-size oracle, omitting it forces an observable
`:missing_task_workspace_memory` serial fallback. Returned metadata contains
the complete fan-out diagnostics.

For a uniform propagation grid, `prop_max_step` selects the smallest number
of steps for each individual `τ` such that `Δτ ≤ prop_max_step`. This avoids
giving short-`τ` items the same step count as `τ=β` while preserving one
caller-selected maximum step size. It is mutually exclusive with
`prop_nsteps` and with an explicit `prop_grid`.
"""
function thermal_correlator(rep::Purified, problem::PurificationProblem,
                           A, B, beta::Real, taus;
                           evolver::Evolver,
                           prep_grid=:uniform, prep_nsteps=nothing,
                           prop_grid=:uniform, prop_nsteps=nothing,
                           prop_max_step::Union{Nothing,Real}=nothing,
                           trajectory=nothing,
                           connected::Bool=false,
                           metadata::NamedTuple=(;),
                           threaded::Bool=false,
                           minbatch::Integer=2,
                           task_memory_cap_bytes::Union{Nothing,Integer}=nothing,
                           task_workspace_memory_bytes::Union{Nothing,Integer}=nothing)
    minbatch >= 1 || throw(ArgumentError("minbatch must be positive"))
    task_memory_cap_bytes === nothing || task_memory_cap_bytes >= 0 ||
        throw(ArgumentError("task_memory_cap_bytes must be nonnegative"))
    task_workspace_memory_bytes === nothing ||
        task_workspace_memory_bytes >= 0 ||
        throw(ArgumentError(
            "task_workspace_memory_bytes must be nonnegative"))
    if prop_max_step !== nothing
        isfinite(prop_max_step) && prop_max_step > 0 ||
            throw(ArgumentError("prop_max_step must be finite and positive"))
        prop_nsteps === nothing ||
            throw(ArgumentError(
                "prop_max_step and prop_nsteps are mutually exclusive"))
        prop_grid === :uniform ||
            throw(ArgumentError(
                "prop_max_step requires prop_grid=:uniform"))
    end
    beta_value = Float64(beta)
    beta_value >= 0 || throw(ArgumentError("beta must be nonnegative"))
    tau_values = Float64.(collect(taus))
    all(tau -> 0 <= tau <= beta_value, tau_values) ||
        throw(ArgumentError("all taus must lie in [0, beta]"))

    Asite, Aop = _local_insertion_thermal(A)
    Bsite, Bop = _local_insertion_thermal(B)
    K = problem.K

    if trajectory !== nothing
        traj = trajectory
        _validate_trajectory(traj, problem, beta, evolver)
    else
        save_betas = sort(unique(vcat(beta_value .- tau_values, [beta_value])))
        traj = thermalize(rep, problem, beta;
                         evolver=evolver, tau_grid=prep_grid, nsteps=prep_nsteps,
                         save_betas=save_betas)
    end

    l_beta = traj.final.log_amplitude
    p_nsteps = prop_nsteps === nothing ?
        max(length(traj.tau_grid) - 1, 1) : prop_nsteps
    propagation_step_counts = [
        iszero(tau) ? 0 :
            (prop_max_step === nothing ?
                p_nsteps : max(ceil(Int, tau / prop_max_step), 1))
        for tau in tau_values
    ]

    Abar = zero(ComplexF64)
    Bbar = zero(ComplexF64)
    if connected
        Abar = thermal_expect(
            traj.final, physical_ttno(problem, _opsum_from_local(Asite, Aop)))
        Bbar = thermal_expect(
            traj.final, physical_ttno(problem, _opsum_from_local(Bsite, Bop)))
        Aop = Aop - Abar * id(problem.phys_doubled[Asite])
        Bop = Bop - Bbar * id(problem.phys_doubled[Bsite])
    end

    # Resolve Dict-backed checkpoints and strip any warm cache from the
    # evolver before entering the threaded region. Each task clones this clean
    # template only when propagation is needed, so mutable state is task-local
    # without retaining one evolver per time point.
    evolver_template = _fresh_evolver_thermal(evolver)
    tau_items = [(i, tau, state_at(traj, beta_value - tau; atol=1e-10))
                 for (i, tau) in enumerate(tau_values)]
    vals = Vector{ComplexF64}(undef, length(tau_values))
    state_payloads = [_thermal_ttns_payload_bytes(item[3].psi)
                      for item in tau_items]
    retained_memory_bytes = sum(state_payloads; init=0) +
        Base.summarysize(evolver_template) +
        sizeof(eltype(vals)) * length(vals)
    item_memory_bytes = [
        4 * payload + Base.summarysize(evolver_template) +
            something(task_workspace_memory_bytes, 0)
        for payload in state_payloads
    ]
    missing_workspace_model = threaded &&
        task_workspace_memory_bytes === nothing
    fanout = bounded_threaded_foreach(
        tau_items;
        threaded=threaded && !missing_workspace_model,
        minbatch,
        retained_memory_bytes,
        memory_cap_bytes=task_memory_cap_bytes,
        item_memory_bytes=(index, _) -> item_memory_bytes[index],
        item_id=(index, item) -> (; tau_index=index, tau=item[2]),
    ) do item
        i, tau, state_b = item
        l_b = state_b.log_amplitude

        # TTNS `copy` is intentionally structural and may retain TensorMap
        # block storage. Give bra and ket separate deep clones: in addition to
        # isolating different τ tasks, this prevents normalization/evolution
        # of ket from mutating backing arrays still observed by bra.
        bra = apply_local(deepcopy(state_b.psi), adjoint(Aop), Asite)
        ket = apply_local(deepcopy(state_b.psi), Bop, Bsite)
        n_ket = norm(ket)
        if iszero(n_ket)
            vals[i] = 0
        else
            normalize!(ket)
            l_k = log(n_ket)

            if tau > 0
                ev = _fresh_evolver_thermal(evolver_template)
                item_nsteps = propagation_step_counts[i]
                pgrid = _build_prop_grid(tau, prop_grid, item_nsteps)
                for j in 1:(length(pgrid) - 1)
                    dtau = pgrid[j + 1] - pgrid[j]
                    iszero(dtau) && continue
                    step!(ev, ket, K, -dtau)
                    nrm = norm(ket)
                    normalize!(ket)
                    l_k += log(nrm)
                end
            end

            overlap = inner(bra, ket)
            vals[i] = exp(2 * l_b + l_k - 2 * l_beta) * overlap
        end
    end
    missing_workspace_model &&
        (fanout = _thermal_fanout_fallback(
            fanout, :missing_task_workspace_memory))

    meta = merge(metadata, (; beta=Float64(beta), Asite, Bsite,
                             connected,
                             centering=connected ? :thermal_mean_insertion : :none,
                             Abar, Bbar, evolver_type=typeof(evolver),
                             propagation_step_counts,
                             propagation_total_steps=sum(
                                 propagation_step_counts; init=0),
                             propagation_max_step=prop_max_step,
                             fanout,))
    return CorrelatorSeries(tau_values, vals, meta)
end

function _thermal_fanout_fallback(
    diagnostics::BoundedFanoutDiagnostics,
    fallback::Symbol,
)
    return BoundedFanoutDiagnostics(
        diagnostics.mode,
        fallback,
        diagnostics.item_count,
        diagnostics.worker_limit,
        diagnostics.batch_count,
        diagnostics.max_batch_items,
        diagnostics.retained_memory_bytes,
        diagnostics.peak_admitted_bytes,
        diagnostics.memory_cap_bytes,
        diagnostics.completed_items,
        diagnostics.cancelled_items,
    )
end

function _thermal_tensor_payload_bytes(tensor)
    bytes = 0
    for (_, block_) in blocks(tensor)
        T = eltype(block_)
        bytes += isbitstype(T) ? sizeof(T) * length(block_) :
            Base.summarysize(block_)
    end
    return bytes
end

function _thermal_ttns_payload_bytes(psi::TTNS)
    return sum(_thermal_tensor_payload_bytes, psi.tensors; init=0)
end

function _check_evolver_no_normalize(evolver)
    if hasproperty(evolver, :normalize) && getproperty(evolver, :normalize)
        throw(ArgumentError(
            "evolver with normalize=true is incompatible with purification; " *
            "the driver owns all normalization (§05 plan §2.4)"))
    end
end

function _fresh_evolver_thermal(evolver::Evolver)
    evrun = deepcopy(evolver)
    if hasproperty(evrun, :cache)
        setproperty!(evrun, :cache, nothing)
    end
    return evrun
end

function _build_grid(beta::Float64, tau_grid, nsteps)
    if tau_grid == :uniform
        nsteps === nothing && throw(ArgumentError(":uniform grid requires nsteps"))
        nsteps > 0 || throw(ArgumentError("nsteps must be positive"))
        return collect(range(0.0, beta / 2; length=nsteps + 1))
    elseif tau_grid isa AbstractVector
        grid = Float64.(collect(tau_grid))
        length(grid) >= 2 || throw(ArgumentError("grid must have at least 2 points"))
        isapprox(grid[1], 0.0; atol=1e-14) ||
            throw(ArgumentError("grid must start at 0; got $(grid[1])"))
        isapprox(grid[end], beta / 2; atol=1e-14) ||
            throw(ArgumentError("grid must end at beta/2=$(beta/2); got $(grid[end])"))
        all(diff(grid) .> 0) ||
            throw(ArgumentError("grid must be strictly increasing"))
        return grid
    else
        throw(ArgumentError("unknown tau_grid: $tau_grid (expected :uniform or a Vector)"))
    end
end

function _build_prop_grid(tau::Float64, prop_grid, nsteps)
    if prop_grid == :uniform
        nsteps === nothing && throw(ArgumentError(":uniform prop_grid requires nsteps"))
        nsteps > 0 || throw(ArgumentError("nsteps must be positive"))
        return collect(range(0.0, tau; length=nsteps + 1))
    elseif prop_grid isa AbstractVector
        grid = Float64.(collect(prop_grid))
        length(grid) >= 2 || throw(ArgumentError("prop_grid must have at least 2 points"))
        isapprox(grid[1], 0.0; atol=1e-14) ||
            throw(ArgumentError("prop_grid must start at 0; got $(grid[1])"))
        isapprox(grid[end], tau; atol=1e-14) ||
            throw(ArgumentError("prop_grid must end at tau=$tau; got $(grid[end])"))
        all(diff(grid) .> 0) ||
            throw(ArgumentError("prop_grid must be strictly increasing"))
        return grid
    else
        throw(ArgumentError("unknown prop_grid: $prop_grid (expected :uniform or a Vector)"))
    end
end

function _local_insertion_thermal(x)
    x isa Pair || throw(ArgumentError("local insertion must be `site => op`"))
    x.first isa Symbol || throw(ArgumentError("local insertion site must be a Symbol"))
    x.second isa AbstractTensorMap ||
        throw(ArgumentError("local insertion operator must be an AbstractTensorMap"))
    return x.first, x.second
end

function _validate_trajectory(traj, problem, beta, evolver)
    traj.metadata.problem_hash == hash(problem.topo_orig) ||
        throw(ArgumentError("trajectory topology does not match problem"))
    isapprox(traj.final.beta, Float64(beta); atol=1e-14) ||
        throw(ArgumentError("trajectory beta $(traj.final.beta) does not match request $beta"))
end

function _opsum_from_local(site::Symbol, op::AbstractTensorMap)
    H = OpSum()
    H += Term(1.0, SiteOp(site, :O, op))
    return H
end
