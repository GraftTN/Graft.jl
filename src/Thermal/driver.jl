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
    rep.aux_evolution === :none ||
        throw(ArgumentError("only aux_evolution=:none is supported in v1; :backward and :custom belong to the future finite-T real-time driver"))
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
                       trajectory=nothing, connected=false,
                       metadata=(;)) -> CorrelatorSeries

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
"""
function thermal_correlator(rep::Purified, problem::PurificationProblem,
                           A, B, beta::Real, taus;
                           evolver::Evolver,
                           prep_grid=:uniform, prep_nsteps=nothing,
                           prop_grid=:uniform, prop_nsteps=nothing,
                           trajectory=nothing,
                           connected::Bool=false,
                           metadata::NamedTuple=(;))
    Asite, Aop = _local_insertion_thermal(A)
    Bsite, Bop = _local_insertion_thermal(B)
    K = problem.K

    if trajectory !== nothing
        traj = trajectory
        _validate_trajectory(traj, problem, beta, evolver)
    else
        save_betas = sort(unique(vcat([beta - Float64(tau) for tau in taus], [Float64(beta)])))
        traj = thermalize(rep, problem, beta;
                         evolver=evolver, tau_grid=prep_grid, nsteps=prep_nsteps,
                         save_betas=save_betas)
    end

    l_beta = traj.final.log_amplitude
    p_nsteps = prop_nsteps === nothing ? max(length(traj.tau_grid) - 1, 1) : prop_nsteps

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

    vals = Vector{ComplexF64}(undef, length(taus))
    for (i, tau) in enumerate(taus)
        tau = Float64(tau)
        b = Float64(beta) - tau
        state_b = state_at(traj, b; atol=1e-10)
        l_b = state_b.log_amplitude

        bra = apply_local(state_b.psi, adjoint(Aop), Asite)
        ket = apply_local(state_b.psi, Bop, Bsite)
        n_ket = norm(ket)
        if iszero(n_ket)
            vals[i] = 0
            continue
        end
        normalize!(ket)
        l_k = log(n_ket)

        if tau > 0
            ev = _fresh_evolver_thermal(evolver)
            pgrid = _build_prop_grid(tau, prop_grid, p_nsteps)
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

    meta = merge(metadata, (; beta=Float64(beta), Asite, Bsite,
                             connected,
                             centering=connected ? :thermal_mean_insertion : :none,
                             Abar, Bbar, evolver_type=typeof(evolver),))
    return CorrelatorSeries(collect(Float64.(taus)), vals, meta)
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
