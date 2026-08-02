# TDVP-only controlled bond expansion strategies.
#
# PredictorCBE follows the projected two-site energy-variation construction in
# pyTTN (Apache-2.0, arXiv:2503.15460). PredictorLegacyCBE preserves Graft's
# former finite-step predictor. NaiveCBE uses the direct double complement and
# an explicit-RNG randomized SVD, following the algorithmic lineage of
# FiniteMPS (MIT). LGVDCBE provides the strict chain-only update ordering and
# alternating integrator from Li--Gleis--von Delft (arXiv:2208.10972,
# arXiv:2207.14712). The factorized C2/C3 regrouping was independently
# implemented from the paper and cross-checked against Shimpei Goto's MIT-
# licensed CBEAlgorithms/ShrewdSelection.h (2024); no source was copied. The
# full-C1 oracle below remains test/reference machinery only.

using Random: Xoshiro

"""Abstract, TDVP-only controlled-bond-expansion strategy."""
abstract type AbstractCBE end

"""Typed diagnostic emitted by a CBE selector or splice."""
abstract type AbstractCBEInfo end

@enum LGVDSweepDirection::UInt8 begin
    LGVDLeftToRight
    LGVDRightToLeft
end

"""A factorized shrewd-selection result in the next-site edge frame."""
struct LGVDSelection{M}
    complement::M
    preselected_rank::Int
    selected_rank::Int
    c2_discarded_norm::Float64
    c3_discarded_norm::Float64
    selector::Symbol
end

"""Per-edge diagnostics for one strict LGVD first-order sweep."""
struct LGVDEdgeInfo
    current::Int
    next::Int
    direction::LGVDSweepDirection
    rank_before::Int
    preselected_rank::Int
    selected_rank::Int
    expanded_rank::Int
    final_rank::Int
    c2_discarded_norm::Float64
    c3_discarded_norm::Float64
    trim_discarded_norm::Float64
    discarded_weight::Float64
    embedding_error::Float64
    state_preserving::Bool
    saturated_rotation_enabled::Bool
end

"""Defensive error for internal paths that bypass the strict LGVD schedule."""
struct LGVDShrewdSelectionUnavailable <: Exception
    reason::Symbol
end


function Base.showerror(io::IO, err::LGVDShrewdSelectionUnavailable)
    print(io,
          "strict LGVDCBE rejected an unsupported internal path (", err.reason,
          "); the full-C1 reference is intentionally not a production fallback")
end

"""
    CBESelectionInfo

Diagnostics for Predictor, legacy, and naive selectors. Predictor reports
`solver=:krylov_svd` only for a converged dense TensorMap solve. Nontrivial
symmetry sectors use the honest `:exact_svd_sector_fallback`; no cross-sector
ordering of independently seeded Ritz problems is guessed.
"""
struct CBESelectionInfo <: AbstractCBEInfo
    strategy::Symbol
    solver::Symbol
    available_rank::Int
    selected_rank::Int
    added_rank::Int
    projection_norm::Float64
    core_norm::Float64
    singular_threshold::Float64
    score_max::Float64
    state_preserving::Bool
end

"""
    LGVDCBEInfo

Strict Li--Gleis--von Delft step diagnostics. Completed production calls carry
the full edge-report vector and alternating composition phase; injected test
selectors may use `completed=false` plus an explicit `blocking_reason`.
"""
struct LGVDCBEInfo <: AbstractCBEInfo
    selector::Symbol
    c1_rank::Int
    preselected_rank::Int
    selected_rank::Int
    added_rank::Int
    projection_error_estimate::Float64
    preselection_threshold::Float64
    final_selection_threshold::Float64
    trim_threshold::Float64
    discarded_weight::Float64
    state_preserving::Bool
    shrewd_applied::Bool
    strict_schedule_applied::Bool
    phase_before::LGVDSweepDirection
    phase_after::LGVDSweepDirection
    edge_reports::Vector{LGVDEdgeInfo}
    completed::Bool
    blocking_reason::Union{Nothing,Symbol}
end

"""
    PredictorCBE(; max_add=32, spawn_threshold=1e-10, neigs=2, krylovdim=4)

Default pyTTN-style projected two-site energy-variation selector. Directions
are the leading left singular vectors of the double-complement core `M`,
equivalently the leading eigenvectors of `M*M'`. A direction is spawned when
`abs(dz) * sigma / norm(theta)` exceeds `spawn_threshold`.

Dense maps use KrylovKit's partial Golub--Kahan solve of the corresponding
normal-operator singular problem. Nontrivial sector maps retain an exact
TensorKit SVD fallback because a single TensorMap Krylov start does not safely
rank Ritz values across charge sectors.
"""
struct PredictorCBE <: AbstractCBE
    max_add::Int
    spawn_threshold::Float64
    neigs::Int
    krylovdim::Int
end
function PredictorCBE(; max_add::Integer=32, spawn_threshold::Real=1e-10,
                      neigs::Integer=2, krylovdim::Integer=4)
    max_add >= 0 || throw(ArgumentError("PredictorCBE max_add must be nonnegative"))
    spawn_threshold >= 0 || throw(ArgumentError(
        "PredictorCBE spawn_threshold must be nonnegative"))
    neigs >= 1 || throw(ArgumentError("PredictorCBE neigs must be positive"))
    krylovdim >= neigs || throw(ArgumentError(
        "PredictorCBE krylovdim must be at least neigs"))
    return PredictorCBE(Int(max_add), Float64(spawn_threshold),
                        Int(neigs), Int(krylovdim))
end

"""
    PredictorLegacyCBE(; max_add=32, enr_rtol=1e-10, enr_atol=1e-12)

Regression-only finite-step Graft predictor. It evolves a throwaway two-site
tensor with `exp(dz*h2)`, projects the resulting one-sided basis, and retains
the historical final truncating SVD. It is not the projected-variance method.
"""
struct PredictorLegacyCBE <: AbstractCBE
    max_add::Int
    enr_rtol::Float64
    enr_atol::Float64
end
function PredictorLegacyCBE(; max_add::Integer=32, enr_rtol::Real=1e-10,
                            enr_atol::Real=1e-12)
    max_add >= 0 || throw(ArgumentError(
        "PredictorLegacyCBE max_add must be nonnegative"))
    enr_rtol >= 0 || throw(ArgumentError(
        "PredictorLegacyCBE enr_rtol must be nonnegative"))
    enr_atol >= 0 || throw(ArgumentError(
        "PredictorLegacyCBE enr_atol must be nonnegative"))
    return PredictorLegacyCBE(Int(max_add), Float64(enr_rtol), Float64(enr_atol))
end

"""
    NaiveCBE(; rng, max_add=32, rsvd=true, ...)

FiniteMPS-style direct double-complement selection. Randomized SVD is Graft's
default policy and requires an explicit caller-owned RNG; `rsvd=false` selects
the deterministic full-SVD reference. No execution path touches global RNG.
"""
struct NaiveCBE{R<:AbstractRNG} <: AbstractCBE
    rng::R
    max_add::Int
    rsvd::Bool
    oversample::Int
    poweriter::Int
    enr_rtol::Float64
    enr_atol::Float64
end
function NaiveCBE(; rng::R, max_add::Integer=32, rsvd::Bool=true,
                  oversample::Integer=8, poweriter::Integer=0,
                  enr_rtol::Real=1e-10, enr_atol::Real=1e-12) where {R<:AbstractRNG}
    max_add >= 0 || throw(ArgumentError("NaiveCBE max_add must be nonnegative"))
    oversample >= 0 || throw(ArgumentError("NaiveCBE oversample must be nonnegative"))
    poweriter >= 0 || throw(ArgumentError("NaiveCBE poweriter must be nonnegative"))
    enr_rtol >= 0 || throw(ArgumentError("NaiveCBE enr_rtol must be nonnegative"))
    enr_atol >= 0 || throw(ArgumentError("NaiveCBE enr_atol must be nonnegative"))
    return NaiveCBE{R}(rng, Int(max_add), rsvd, Int(oversample),
                       Int(poweriter), Float64(enr_rtol), Float64(enr_atol))
end

"""
    LGVDCBE(; max_add=32, preselection_threshold=1e-4,
            final_selection_threshold=1e-6, trim_threshold=1e-12)

Configuration for strict chain CBE-TDVP. The defaults are the empirical paper
thresholds epsilon-prime, epsilon-tilde, and epsilon-final. Strict execution
uses a dedicated expansion-before-evolution sweep and the alternating
`LRL`/`RLR` composition. Production selection uses the paper's factorized
C2/C3 contractions; the full-C1 oracle is never used as a fallback.
"""
struct LGVDCBE <: AbstractCBE
    max_add::Int
    preselection_threshold::Float64
    final_selection_threshold::Float64
    trim_threshold::Float64
end
function LGVDCBE(; max_add::Integer=32,
                 preselection_threshold::Real=1e-4,
                 final_selection_threshold::Real=1e-6,
                 trim_threshold::Real=1e-12)
    max_add >= 0 || throw(ArgumentError("LGVDCBE max_add must be nonnegative"))
    preselection_threshold >= 0 || throw(ArgumentError(
        "LGVDCBE preselection_threshold must be nonnegative"))
    final_selection_threshold >= 0 || throw(ArgumentError(
        "LGVDCBE final_selection_threshold must be nonnegative"))
    trim_threshold >= 0 || throw(ArgumentError(
        "LGVDCBE trim_threshold must be nonnegative"))
    return LGVDCBE(Int(max_add), Float64(preselection_threshold),
                   Float64(final_selection_threshold), Float64(trim_threshold))
end

"""
    TDVP1_CBE(; cbe=PredictorCBE(), order=2, trunc=TruncationScheme(maxdim=100), ...)

One-site projector-splitting TDVP whose bond-expansion mathematics is selected
by the concrete `cbe::AbstractCBE` strategy. CBE strategies are consumed only
through this evolver.
"""
Base.@kwdef mutable struct TDVP1_CBE{C<:AbstractCBE} <: Evolver
    cbe::C = PredictorCBE()
    order::Int = 2
    trunc::TruncationScheme = TruncationScheme(; maxdim=100)
    enabled::Bool = true
    krylovdim::Int = 30
    tol::Float64 = 1e-12
    threaded_channels::Bool = false
    channel_slices::Int = 2
    channel_minbatch::Int = 2
    channel_min_flops::Real = 1_000_000
    channel_memory_cap_bytes::Union{Nothing,Real} = nothing
    distributed::Union{Nothing,AbstractDistributedContext} = nothing
    verbose::Bool = true
    cache::Union{Nothing,EnvCache} = nothing
    last_cbe_info::Union{Nothing,AbstractCBEInfo} = nothing
    lgvd_phase::LGVDSweepDirection = LGVDLeftToRight
end

_cbe_max_add(cbe::AbstractCBE) = cbe.max_add

"""Return `(child, parent)` for the active edge, independent of sweep direction."""
function _cbe_orient_edge(t::TreeTopology, u::Int, v::Int)
    t.parent[u] == v && return u, v
    t.parent[v] == u && return v, u
    throw(ArgumentError("CBE endpoints must be adjacent"))
end

"""
Put the active edge leg in the domain and return the explicit two-site leg
groups. The groups preserve the non-reversing order of `two_site_tensor`.
"""
function _cbe_edge_frames(ψ::TTNS, u::Int, v::Int)
    t = topology(ψ)
    n, m = _cbe_orient_edge(t, u, v)
    An, Am = ψ.tensors[n], ψ.tensors[m]
    k = childslot(t, m, n)
    pn = numout(An)
    total = pn + numind(Am) - 1
    if u == n
        active = An
        partner = permute(Am, (Backend._others(numind(Am), k), (k,)))
        active_legs = ntuple(identity, pn)
        partner_legs = ntuple(j -> pn + j, total - pn)
    else
        active = permute(Am, (Backend._others(numind(Am), k), (k,)))
        partner = An
        active_legs = ntuple(j -> pn + j, total - pn)
        partner_legs = ntuple(identity, pn)
    end
    return (; n, m, active, partner, active_legs, partner_legs)
end

"""
Right injection for the partner complement. `transpose(adjoint(Np))`
reverses its codomain factors; reverse them once more to match the explicit,
non-reversing domain group used for `Zs`.
"""
function _cbe_right_injection(Np::AbstractTensorMap)
    injection = transpose(adjoint(Np))
    no = numout(injection)
    ni = numin(injection)
    return permute(injection,
                   (reverse(ntuple(identity, no)),
                    ntuple(i -> no + i, ni)))
end

"""
Construct the factors of the canonical double-complement map without yet
materializing `M`. `Z = h2(two_site_tensor(...))`; no finite-step evolution is
performed. Naive RSVD sketches these factors directly.
"""
function _cbe_projected_factors(ev::TDVP1_CBE, ψ::TTNS, H::TTNO,
                                u::Int, v::Int)
    frames = _cbe_edge_frames(ψ, u, v)
    Q, _ = left_orth(frames.active)
    Na = left_null(Q)
    Np = left_null(frames.partner)
    Θ = two_site_tensor(ψ, frames.n, frames.m)
    h2 = eff_h2(ev.cache::EnvCache, ψ, H, frames.n, frames.m;
                threaded_channels=ev.threaded_channels,
                channel_slices=ev.channel_slices,
                channel_minbatch=ev.channel_minbatch,
                channel_min_flops=ev.channel_min_flops,
                channel_memory_cap_bytes=ev.channel_memory_cap_bytes,
                distributed=ev.distributed)
    Z = h2(Θ)
    Zs = permute(Z, (frames.active_legs, frames.partner_legs))
    injection = _cbe_right_injection(Np)
    return (; frames, Q, Na, Np, Θ, Zs, injection)
end

"""Materialize `M = Na' * Zs * injection` for exact/reference selectors."""
function _cbe_projected_core(ev::TDVP1_CBE, ψ::TTNS, H::TTNO,
                             u::Int, v::Int)
    factors = _cbe_projected_factors(ev, ψ, H, u, v)
    (; Na, Zs, injection) = factors
    M = Na' * Zs * injection
    return merge(factors, (; M))
end

_cbe_rank(A::AbstractTensorMap) = dim(domain(A))
_cbe_singular_values(A::AbstractTensorMap) = sort!(Float64.(collect(svd_vals(A))); rev=true)

function _cbe_exact_directions(M::AbstractTensorMap, maxrank::Int;
                               atol::Real=0, rtol::Real=0)
    maxrank <= 0 && return nothing
    U, _, _ = split_svd(M, TruncationScheme(; maxdim=maxrank,
                                             atol=Float64(atol),
                                             rtol=Float64(rtol)))
    return _cbe_rank(U) == 0 ? nothing : U
end

_cbe_projected_apply(factors, x::AbstractTensorMap) =
    factors.Na' * (factors.Zs * (factors.injection * x))
_cbe_projected_adjoint_apply(factors, x::AbstractTensorMap) =
    factors.injection' * (factors.Zs' * (factors.Na * x))

function _cbe_dense_krylov_directions(M::AbstractTensorMap, maxrank::Int,
                                      krylovdim::Int, cutoff::Real)
    maxrank <= 0 && return nothing, Float64[], :krylov_svd
    Vcod = fuse(codomain(M))
    Vdom = fuse(domain(M))
    # A single TensorMap start represents one conserved sector. It is a safe
    # global start only for the trivial dense sector; graded maps fall back to
    # TensorKit's exact sector-aware SVD in the caller.
    sectortype(Vcod) === Trivial && sectortype(Vdom) === Trivial ||
        return nothing, Float64[], :exact_svd_sector_fallback

    K = Contractions._rsvd_probe_space(Vdom, 1)
    Ω = Contractions._rsvd_random_probe(
        Xoshiro(0xcbe2503154600001), scalartype(M), domain(M) ← K;
        threaded=false)
    x0 = M * Ω
    norm(x0) > eps(Float64) * max(norm(M), 1) ||
        return nothing, Float64[], :exact_svd_start_fallback

    problem_dim = min(dim(Vcod), dim(Vdom))
    requested = min(maxrank, problem_dim)
    problem_dim > requested ||
        return nothing, Float64[], :exact_svd_small_fallback
    kd = max(requested + 1, min(krylovdim, problem_dim))
    vals, lefts, _, info = KrylovKit.svdsolve(
        (x -> M * x, x -> M' * x), x0, requested, :LR;
        krylovdim=kd, tol=max(eps(Float64), min(Float64(cutoff), 1e-10)),
        maxiter=20)
    info.converged >= requested ||
        return nothing, Float64.(vals), :exact_svd_convergence_fallback
    keep = findall(σ -> σ >= cutoff, vals)
    resize!(keep, min(length(keep), requested))
    isempty(keep) && return nothing, Float64.(vals), :krylov_svd
    directions = reduce(catdomain, lefts[keep])
    return directions, Float64.(vals), :krylov_svd
end

"""
Exact state-preserving zero-weight splice. No SVD is permitted here: old
content is never truncated or rotated during expansion.
"""
function _cbe_state_preserving_splice(tA::AbstractTensorMap,
                                      Q::AbstractTensorMap,
                                      E::AbstractTensorMap)
    if isdual(domain(Q)[1]) != isdual(domain(E)[1])
        E = flip(E, numind(E))
    end
    U = catdomain(Q, E)
    R = U' * tA
    return U, R
end

function _cbe_no_growth_split(tA::AbstractTensorMap, Q::AbstractTensorMap)
    return Q, Q' * tA
end

function _cbe_predictor_directions(strategy::PredictorCBE,
                                   core, dz::Number, room::Int)
    core_norm = Float64(norm(core.Θ))
    projection_norm = Float64(norm(core.M))
    denom = max(core_norm, eps(Float64))
    cutoff = iszero(dz) ? Inf : strategy.spawn_threshold * denom / Float64(abs(dz))
    budget = min(room, strategy.max_add, strategy.neigs)
    directions, sigmas, solver = if isfinite(cutoff) && budget > 0
        try
            _cbe_dense_krylov_directions(
                core.M, budget, strategy.krylovdim, cutoff)
        catch
            (nothing, Float64[], :exact_svd_error_fallback)
        end
    else
        (nothing, Float64[], :krylov_svd)
    end
    if startswith(String(solver), "exact_svd") && isfinite(cutoff) && budget > 0
        sigmas = _cbe_singular_values(core.M)
        directions = _cbe_exact_directions(core.M, budget; atol=cutoff)
    end
    score_max = isempty(sigmas) ? 0.0 : Float64(abs(dz)) * first(sigmas) / denom
    selected = directions === nothing ? 0 : _cbe_rank(directions)
    available = min(dim(codomain(core.M)), dim(domain(core.M)))
    info = CBESelectionInfo(:predictor, solver, available, selected,
                            selected, projection_norm, core_norm,
                            Float64(cutoff), score_max, true)
    return directions, info
end

function _cbe_implicit_rsvd_directions(strategy::NaiveCBE, factors, budget::Int)
    budget <= 0 && return nothing, nothing
    Vrest = fuse(domain(factors.injection))
    sketchdim = min(dim(Vrest), budget + strategy.oversample)
    K = Contractions._rsvd_probe_space(Vrest, sketchdim)
    Ω = Contractions._rsvd_random_probe(
        strategy.rng, scalartype(factors.Zs), domain(factors.injection) ← K;
        threaded=false)
    Y = _cbe_projected_apply(factors, Ω)
    for _ in 1:strategy.poweriter
        Z, _ = left_orth(_cbe_projected_adjoint_apply(factors, Y); alg=:qr)
        Y = _cbe_projected_apply(factors, Z)
    end
    sketch, _ = left_orth(Y; alg=:qr)
    # Only this rank-sketch-by-partner-complement core is materialized. The
    # full active-complement-by-partner-complement map is never formed.
    small = (sketch' * factors.Na') * factors.Zs * factors.injection
    localdirs = _cbe_exact_directions(
        small, budget; atol=strategy.enr_atol, rtol=strategy.enr_rtol)
    return localdirs === nothing ? nothing : sketch * localdirs, small
end

function _cbe_naive_directions(strategy::NaiveCBE, factors, room::Int)
    budget = min(room, strategy.max_add)
    directions, small, solver = if budget <= 0
        (nothing, nothing, strategy.rsvd ? :implicit_rsvd : :exact_svd)
    elseif strategy.rsvd
        dirs, sketched = _cbe_implicit_rsvd_directions(strategy, factors, budget)
        (dirs, sketched, :implicit_rsvd)
    else
        M = factors.Na' * factors.Zs * factors.injection
        (_cbe_exact_directions(M, budget;
                               atol=strategy.enr_atol,
                               rtol=strategy.enr_rtol), M, :exact_svd)
    end
    sigmas = small === nothing ? Float64[] : _cbe_singular_values(small)
    selected = directions === nothing ? 0 : _cbe_rank(directions)
    available = min(dim(domain(factors.Na)), dim(domain(factors.Np)))
    info = CBESelectionInfo(:naive, solver, available, selected, selected,
                            small === nothing ? 0.0 : Float64(norm(small)),
                            Float64(norm(factors.Θ)), strategy.enr_atol,
                            isempty(sigmas) ? 0.0 : first(sigmas), true)
    return directions, info
end

"""Private full-SVD oracle for validating the implicit Naive range finder."""
function _cbe_naive_exact_oracle(strategy::NaiveCBE, core, room::Int)
    budget = min(room, strategy.max_add)
    sigmas = _cbe_singular_values(core.M)
    directions = budget <= 0 ? nothing :
        _cbe_exact_directions(core.M, budget;
                              atol=strategy.enr_atol, rtol=strategy.enr_rtol)
    selected = directions === nothing ? 0 : _cbe_rank(directions)
    info = CBESelectionInfo(:naive, :exact_svd_oracle,
                            length(sigmas), selected, selected,
                            Float64(norm(core.M)), Float64(norm(core.Θ)),
                            strategy.enr_atol,
                            isempty(sigmas) ? 0.0 : first(sigmas), true)
    return directions, info
end

"""Exact full-C1 dense reference selector for strict LGVD integration tests."""
function _lgvd_c1_directions(strategy::LGVDCBE, core, room::Int)
    sigmas = _cbe_singular_values(core.M)
    scale = isempty(sigmas) ? 0.0 : first(sigmas)
    pre_cut = strategy.preselection_threshold * scale
    final_cut = strategy.final_selection_threshold * scale
    preselected = count(>=(pre_cut), sigmas)
    budget = min(room, strategy.max_add)
    directions = _cbe_exact_directions(core.M, budget; atol=final_cut)
    selected = directions === nothing ? 0 : _cbe_rank(directions)
    info = LGVDCBEInfo(:full_c1, length(sigmas), preselected, selected,
                       selected, Float64(norm(core.M)),
                       strategy.preselection_threshold,
                       strategy.final_selection_threshold,
                       strategy.trim_threshold, 0.0, true, false, false,
                       LGVDLeftToRight, LGVDLeftToRight, LGVDEdgeInfo[],
                       false, :reference_only)
    return directions, info
end

"""Return a deterministic endpoint path or reject before strict LGVD mutation."""
function _lgvd_chain_path(t::TreeTopology)
    nnodes(t) >= 2 || throw(ArgumentError(
        "strict LGVDCBE requires a chain with at least two physical sites"))
    all(length(neighbors(t, n)) <= 2 for n in 1:nnodes(t)) ||
        throw(ArgumentError(
            "strict LGVDCBE supports MPS-chain topology only; branching-tree CBE is not implemented"))
    endpoints = [n for n in 1:nnodes(t) if length(neighbors(t, n)) == 1]
    length(endpoints) == 2 || throw(ArgumentError(
        "strict LGVDCBE requires exactly two chain endpoints"))
    return path_between(t, minimum(endpoints), maximum(endpoints))
end

function _lgvd_require_chain(t::TreeTopology)
    _lgvd_chain_path(t)
    return nothing
end

function _lgvd_preflight(ev::TDVP1_CBE{LGVDCBE}, ψ::TTNS, H::TTNO)
    topology(H) == topology(ψ) || throw(ArgumentError(
        "strict LGVDCBE requires matching TTNS and TTNO topologies"))
    ev.order == 2 || throw(ArgumentError(
        "strict LGVDCBE implements only the alternating symmetric order-2 composition"))
    all(hasphys(ψ, n) && hasphys(H, n) for n in 1:nnodes(topology(ψ))) ||
        throw(ArgumentError(
            "strict LGVDCBE requires one physical leg on every chain node"))
    ev.distributed === nothing || throw(ArgumentError(
        "strict LGVDCBE distributed selection has not been validated"))
    return _lgvd_chain_path(topology(ψ))
end

function _cbe_projected_split(strategy::PredictorCBE, ev::TDVP1_CBE,
                              ψ::TTNS, H::TTNO, u::Int, v::Int, dz::Number)
    factors = _cbe_projected_core(ev, ψ, H, u, v)
    oldrank = _cbe_rank(factors.Q)
    room = max(0, ev.trunc.maxdim - oldrank)
    directions, info = _cbe_predictor_directions(strategy, factors, dz, room)
    if directions === nothing
        U, R = _cbe_no_growth_split(factors.frames.active, factors.Q)
    else
        E = factors.Na * directions
        U, R = _cbe_state_preserving_splice(
            factors.frames.active, factors.Q, E)
    end
    ev.last_cbe_info = info
    return U, R
end


function _cbe_projected_split(strategy::NaiveCBE, ev::TDVP1_CBE,
                              ψ::TTNS, H::TTNO, u::Int, v::Int, ::Number)
    factors = _cbe_projected_factors(ev, ψ, H, u, v)
    oldrank = _cbe_rank(factors.Q)
    room = max(0, ev.trunc.maxdim - oldrank)
    directions, info = _cbe_naive_directions(strategy, factors, room)
    if directions === nothing
        U, R = _cbe_no_growth_split(factors.frames.active, factors.Q)
    else
        E = factors.Na * directions
        U, R = _cbe_state_preserving_splice(
            factors.frames.active, factors.Q, E)
    end
    ev.last_cbe_info = info
    return U, R
end

function _split_link_up(ev::TDVP1_CBE, ψ::TTNS, H::TTNO,
                        u::Int, v::Int, dz::Number)
    ev.enabled || return _split_link_up(TDVP1(), ψ, H, u, v, dz)
    return _split_link_up(ev.cbe, ev, ψ, H, u, v, dz)
end
function _split_link_down(ev::TDVP1_CBE, ψ::TTNS, H::TTNO,
                          u::Int, v::Int, dz::Number)
    ev.enabled || return _split_link_down(TDVP1(), ψ, H, u, v, dz)
    return _split_link_down(ev.cbe, ev, ψ, H, u, v, dz)
end

function _split_link_up(strategy::Union{PredictorCBE,NaiveCBE},
                        ev::TDVP1_CBE, ψ::TTNS, H::TTNO,
                        u::Int, v::Int, dz::Number)
    U, R = _cbe_projected_split(strategy, ev, ψ, H, u, v, dz)
    ψ.tensors[u] = U
    return R
end
function _split_link_down(strategy::Union{PredictorCBE,NaiveCBE},
                          ev::TDVP1_CBE, ψ::TTNS, H::TTNO,
                          u::Int, v::Int, dz::Number)
    A = ψ.tensors[u]
    N, No = numind(A), numout(A)
    U, R = _cbe_projected_split(strategy, ev, ψ, H, u, v, dz)
    ψ.tensors[u] = permute(U, Backend._restore_perm(N, No, childslot(ψ.topo, u, v)))
    return transpose(R)
end

# Historical finite-step predictor and truncating splice, retained byte-for-
# byte in numerical order except that policy now lives in PredictorLegacyCBE.
function _cbe_legacy_predictor(strategy::PredictorLegacyCBE,
                               ev::TDVP1_CBE, ψ::TTNS, H::TTNO,
                               u::Int, v::Int, dz::Number)
    t = ψ.topo
    n, m = _cbe_orient_edge(t, u, v)
    Θ = two_site_tensor(ψ, n, m)
    h2 = eff_h2(ev.cache::EnvCache, ψ, H, n, m;
                threaded_channels=ev.threaded_channels,
                channel_slices=ev.channel_slices,
                channel_minbatch=ev.channel_minbatch,
                channel_min_flops=ev.channel_min_flops,
                channel_memory_cap_bytes=ev.channel_memory_cap_bytes,
                distributed=ev.distributed)
    Θ, _ = _effective_exponentiate(
        ev.distributed, h2, dz, Θ; ishermitian=ishermitian(H),
        krylovdim=ev.krylovdim, tol=ev.tol)
    pn = numout(ψ.tensors[n])
    NΘ = numind(Θ)
    Θs = permute(Θ, (ntuple(identity, pn),
                     ntuple(j -> pn + j, NΘ - pn)))
    U, _, Vh = split_svd(Θs, ev.trunc)
    if u == n
        return U
    end
    k = childslot(t, m, n)
    Km = numind(ψ.tensors[m]) - 1
    p1 = ntuple(j -> j == k ? 1 : 1 + (j < k ? j : j - 1), Km)
    return permute(Vh, (p1, (numind(Vh),)))
end

function _cbe_legacy_enrich_split(strategy::PredictorLegacyCBE,
                                  ev::TDVP1_CBE,
                                  tA::AbstractTensorMap,
                                  tP::AbstractTensorMap)
    expanded = tA
    room = min(strategy.max_add, ev.trunc.maxdim - dim(domain(tA)))
    if room > 0 && dim(codomain(tA)) > dim(domain(tA))
        N = left_null(tA)
        if dim(domain(N)) > 0
            M = N' * tP
            Um, _, _ = svd_trunc(M; trunc=truncrank(room) &
                trunctol(; atol=strategy.enr_atol, rtol=strategy.enr_rtol))
            if dim(domain(Um)) > 0
                E = N * Um
                if isdual(domain(tA)[1]) != isdual(domain(E)[1])
                    E = flip(E, numind(E))
                end
                expanded = catdomain(tA, E)
            end
        end
    end
    U, _, _ = split_svd(expanded, ev.trunc)
    R = U' * tA
    ev.last_cbe_info = CBESelectionInfo(
        :predictor_legacy, :finite_step_svd, dim(codomain(expanded)),
        max(0, dim(domain(U)) - dim(domain(tA))),
        max(0, dim(domain(U)) - dim(domain(tA))), 0.0, Float64(norm(tA)),
        strategy.enr_atol, 0.0, false)
    return U, R
end

function _split_link_up(strategy::PredictorLegacyCBE, ev::TDVP1_CBE,
                        ψ::TTNS, H::TTNO, u::Int, v::Int, dz::Number)
    A = ψ.tensors[u]
    P = _cbe_legacy_predictor(strategy, ev, ψ, H, u, v, dz)
    U, R = _cbe_legacy_enrich_split(strategy, ev, A, P)
    ψ.tensors[u] = U
    return R
end
function _split_link_down(strategy::PredictorLegacyCBE, ev::TDVP1_CBE,
                          ψ::TTNS, H::TTNO, u::Int, v::Int, dz::Number)
    k = childslot(ψ.topo, u, v)
    A = ψ.tensors[u]
    P = _cbe_legacy_predictor(strategy, ev, ψ, H, u, v, dz)
    N, No = numind(A), numout(A)
    frame = (Backend._others(N, k), (k,))
    U, R = _cbe_legacy_enrich_split(
        strategy, ev, permute(A, frame), permute(P, frame))
    ψ.tensors[u] = permute(U, Backend._restore_perm(N, No, k))
    return transpose(R)
end

function _lgvd_next_frame(ψ::TTNS, current::Int, next::Int)
    t = topology(ψ)
    if t.parent[next] == current
        return (; tensor=ψ.tensors[next], kind=:child, slot=0,
                original_numind=numind(ψ.tensors[next]),
                original_numout=numout(ψ.tensors[next]))
    elseif t.parent[current] == next
        slot = childslot(t, next, current)
        A = ψ.tensors[next]
        return (; tensor=permute(A, (Backend._others(numind(A), slot), (slot,))),
                kind=:parent, slot, original_numind=numind(A),
                original_numout=numout(A))
    end
    throw(ArgumentError("strict LGVDCBE update endpoints must be adjacent"))
end

function _lgvd_restore_next_frame(frame, tensor::AbstractTensorMap)
    frame.kind === :child && return tensor
    return permute(tensor, Backend._restore_perm(
        frame.original_numind, frame.original_numout, frame.slot))
end

function _lgvd_zero_pad!(ev::TDVP1_CBE{LGVDCBE}, ψ::TTNS,
                         current::Int, next::Int,
                         selection::LGVDSelection)
    cache = ev.cache::EnvCache
    t = topology(ψ)
    n, m = _cbe_orient_edge(t, current, next)
    before = two_site_tensor(ψ, n, m)
    frame = _lgvd_next_frame(ψ, current, next)
    rank_before = _cbe_rank(frame.tensor)
    selection.selected_rank >= 0 || throw(ArgumentError(
        "LGVD selection rank must be nonnegative"))

    E = selection.complement
    if E === nothing || selection.selected_rank == 0
        E === nothing || _cbe_rank(E) == 0 || throw(ArgumentError(
            "LGVD zero-rank selection carried nonempty directions"))
        return (; rank_before, expanded_rank=rank_before,
                embedding_error=0.0, state_preserving=true)
    end
    E isa AbstractTensorMap || throw(ArgumentError(
        "LGVD selection complement must be a TensorMap"))
    _cbe_rank(E) == selection.selected_rank || throw(ArgumentError(
        "LGVD selection rank does not match its complement"))
    codomain(E) == codomain(frame.tensor) || throw(ArgumentError(
        "LGVD complement is not in the next-site edge frame"))
    if isdual(domain(frame.tensor)[1]) != isdual(domain(E)[1])
        E = flip(E, numind(E))
    end
    orthogonality = Float64(norm(frame.tensor' * E))
    isometry_defect = Float64(norm(E' * E - id(domain(E))))
    scale = max(Float64(norm(frame.tensor)), Float64(norm(E)), 1.0)
    tolerance = 1_000 * eps(Float64) * scale
    orthogonality <= tolerance || throw(ArgumentError(
        "LGVD selected directions are not orthogonal to the kept frame"))
    isometry_defect <= tolerance || throw(ArgumentError(
        "LGVD selected directions are not isometric"))

    expanded = catdomain(frame.tensor, E)
    injection = expanded' * frame.tensor
    next_tensor = _lgvd_restore_next_frame(frame, expanded)
    current_tensor = if t.parent[next] == current
        absorb_on_leg(ψ.tensors[current], injection,
                      childslot(t, current, next))
    else
        transform_leg(ψ.tensors[current], injection, parentleg(ψ, current))
    end

    # Both tensors are built and space-checked before either state write.
    update_tensor!(ψ, next, next_tensor; caches=(cache,), gauge=false)
    update_tensor!(ψ, current, current_tensor; caches=(cache,))
    after = two_site_tensor(ψ, n, m)
    embedding_error = Float64(norm(after - before))
    state_preserving = embedding_error <=
        1_000 * eps(Float64) * max(Float64(norm(before)), 1.0)
    return (; rank_before, expanded_rank=_cbe_rank(expanded),
            embedding_error, state_preserving)
end

function _lgvd_site_forward!(ev::TDVP1_CBE{LGVDCBE}, ψ::TTNS, H::TTNO,
                             site::Int, dz::Number; hermitian::Bool)
    cache = ev.cache::EnvCache
    h1 = eff_h1(cache, ψ, H, site;
                threaded_channels=ev.threaded_channels,
                channel_slices=ev.channel_slices,
                channel_minbatch=ev.channel_minbatch,
                channel_min_flops=ev.channel_min_flops,
                channel_memory_cap_bytes=ev.channel_memory_cap_bytes,
                distributed=ev.distributed)
    A, _ = _effective_exponentiate(
        ev.distributed, h1, dz, ψ.tensors[site];
        ishermitian=hermitian, krylovdim=ev.krylovdim, tol=ev.tol)
    update_tensor!(ψ, site, A; caches=(cache,))
    return ψ
end

function _lgvd_trim_scheme(ev::TDVP1_CBE{LGVDCBE})
    return TruncationScheme(
        maxdim=ev.trunc.maxdim,
        atol=ev.trunc.atol,
        rtol=max(ev.trunc.rtol, ev.cbe.trim_threshold),
        discarded_weight=ev.trunc.discarded_weight)
end

function _lgvd_trim_toward!(ev::TDVP1_CBE{LGVDCBE}, ψ::TTNS,
                            current::Int, next::Int)
    cache = ev.cache::EnvCache
    t = topology(ψ)
    A = ψ.tensors[current]
    input_norm = Float64(norm(A))
    if t.parent[current] == next
        Q, S, Vh, discarded = split_svd_with_error(
            A, _lgvd_trim_scheme(ev))
        link = S * Vh
    elseif t.parent[next] == current
        Q, C, discarded = svd_factor_leg_with_error(
            A, childslot(t, current, next), _lgvd_trim_scheme(ev))
        link = transpose(C)
    else
        throw(ArgumentError("strict LGVDCBE trim endpoints must be adjacent"))
    end
    update_tensor!(ψ, current, Q; caches=(cache,))
    discarded_norm = Float64(discarded)
    discarded_weight = iszero(input_norm) ?
        (iszero(discarded_norm) ? 0.0 : Inf) :
        abs2(discarded_norm / input_norm)
    final_rank = t.parent[current] == next ?
        dim(codomain(link)) : dim(domain(link))
    return (; link, discarded_norm, discarded_weight, final_rank)
end

function _lgvd_backward_link_move!(ev::TDVP1_CBE{LGVDCBE}, ψ::TTNS,
                                   H::TTNO, current::Int, next::Int,
                                   dz::Number, link::AbstractTensorMap;
                                   hermitian::Bool)
    cache = ev.cache::EnvCache
    t = topology(ψ)
    n, m = _cbe_orient_edge(t, current, next)
    k0 = eff_h0(cache, ψ, H, n, m)
    evolved, _ = _effective_exponentiate(
        ev.distributed, k0, -dz, link;
        ishermitian=hermitian, krylovdim=ev.krylovdim, tol=ev.tol)
    evolved = Networks.pivotal_link(evolved)
    next_tensor = if current == n
        absorb_on_leg(ψ.tensors[next], evolved, childslot(t, next, current))
    else
        ψ.tensors[next] * evolved
    end
    update_tensor!(ψ, next, next_tensor; caches=(cache,), gauge=false)
    ψ.center = next
    return ψ
end

function _lgvd_update_edge!(ev::TDVP1_CBE{LGVDCBE}, ψ::TTNS, H::TTNO,
                            current::Int, next::Int, dz::Number,
                            direction::LGVDSweepDirection, selector;
                            hermitian::Bool)
    @assert center(ψ) == current
    selection = selector(ev.cbe, ev, ψ, H, current, next)
    selection isa LGVDSelection || throw(ArgumentError(
        "LGVD selector must return LGVDSelection"))
    padding = _lgvd_zero_pad!(ev, ψ, current, next, selection)
    _lgvd_site_forward!(ev, ψ, H, current, dz; hermitian)
    trim = _lgvd_trim_toward!(ev, ψ, current, next)
    _lgvd_backward_link_move!(
        ev, ψ, H, current, next, dz, trim.link; hermitian)
    return LGVDEdgeInfo(
        current, next, direction, padding.rank_before,
        selection.preselected_rank, selection.selected_rank,
        padding.expanded_rank, trim.final_rank,
        selection.c2_discarded_norm, selection.c3_discarded_norm,
        trim.discarded_norm, trim.discarded_weight,
        padding.embedding_error, padding.state_preserving,
        padding.rank_before >= ev.trunc.maxdim && selection.selected_rank > 0)
end

function _lgvd_sweep!(ev::TDVP1_CBE{LGVDCBE}, ψ::TTNS, H::TTNO,
                      dz::Number, path::Vector{Int},
                      direction::LGVDSweepDirection, selector,
                      reports::Vector{LGVDEdgeInfo})
    sweep_path = direction === LGVDLeftToRight ? path : reverse(path)
    cache = ev.cache::EnvCache
    move_center!(ψ, first(sweep_path); cache)
    hermitian = ishermitian(H)
    for i in 1:(length(sweep_path) - 1)
        push!(reports, _lgvd_update_edge!(
            ev, ψ, H, sweep_path[i], sweep_path[i + 1], dz,
            direction, selector; hermitian))
    end
    _lgvd_site_forward!(ev, ψ, H, last(sweep_path), dz; hermitian)
    return ψ
end

_lgvd_reverse(direction::LGVDSweepDirection) =
    direction === LGVDLeftToRight ? LGVDRightToLeft : LGVDLeftToRight

function _lgvd_step_with_selector!(ev::TDVP1_CBE{LGVDCBE}, ψ::TTNS,
                                   H::TTNO, dz::Number, selector;
                                   selector_name::Symbol=:injected_selector,
                                   shrewd_applied::Bool=false)
    path = _lgvd_preflight(ev, ψ, H)
    cache_reused = ev.cache !== nothing && ev.cache.topo == topology(ψ)
    cache_reused || (ev.cache = EnvCache(topology(ψ)))
    phase_before = ev.lgvd_phase
    reverse_direction = _lgvd_reverse(phase_before)
    reports = LGVDEdgeInfo[]
    _lgvd_sweep!(ev, ψ, H, dz / 4, path, phase_before, selector, reports)
    _lgvd_sweep!(ev, ψ, H, dz / 2, path, reverse_direction, selector, reports)
    _lgvd_sweep!(ev, ψ, H, dz / 4, path, phase_before, selector, reports)
    ev.lgvd_phase = reverse_direction
    discarded_weight = isempty(reports) ? 0.0 :
        maximum(report.discarded_weight for report in reports)
    state_preserving = all(report.state_preserving for report in reports)
    preselected_rank = isempty(reports) ? 0 :
        maximum(report.preselected_rank for report in reports)
    selected_rank = isempty(reports) ? 0 :
        maximum(report.selected_rank for report in reports)
    added_rank = isempty(reports) ? 0 :
        maximum(report.expanded_rank - report.rank_before for report in reports)
    ev.last_cbe_info = LGVDCBEInfo(
        selector_name, 0, preselected_rank, selected_rank, added_rank,
        NaN, ev.cbe.preselection_threshold,
        ev.cbe.final_selection_threshold, ev.cbe.trim_threshold,
        discarded_weight, state_preserving, shrewd_applied, true,
        phase_before, ev.lgvd_phase, reports, true, nothing)
    return ψ
end

function _lgvd_shrewd_selector(strategy::LGVDCBE, ev::TDVP1_CBE,
                               psi::TTNS, H::TTNO,
                               current::Int, next::Int)
    strategy.max_add == 0 && return LGVDSelection(
        nothing, 0, 0, 0.0, 0.0, :shrewd_c2_c3)
    cache = ev.cache::EnvCache
    frames = _cbe_edge_frames(psi, current, next)
    source, link = left_orth(frames.active)
    target = frames.partner
    factor_frame = oriented_two_site_factor_frame(
        cache, psi, H, current, next;
        source_tensor=source, target_tensor=target)

    # C2(a): keep the TTNO channel in the source SVD group.
    source_factor = contract_source_factor(cache, factor_frame, link)
    source_scale = Float64(norm(source_factor))
    source_factor -= source * (source' * source_factor)
    source_residual = Float64(norm(source_factor))
    numerical_floor = 1e-14
    source_residual <= numerical_floor * max(source_scale, 1.0) &&
        return LGVDSelection(
            nothing, 0, 0, source_residual, 0.0, :shrewd_c2_c3)
    ns = numout(source_factor)
    source_grouped = permute(
        source_factor,
        ((ntuple(identity, ns)..., ns + 2), (ns + 1,)))
    _, source_s, source_v = split_svd(source_grouped)

    # C2(b): propagate only the compressed bridge, project the target kept
    # frame, and truncate Dbar -> Dprime at epsilon-prime.
    target_factor = contract_target_factor(
        cache, factor_frame, source_s * source_v)
    target_scale = Float64(norm(target_factor))
    target_factor -= target * (target' * target_factor)
    target_residual = Float64(norm(target_factor))
    target_residual <= numerical_floor * max(target_scale, 1.0) &&
        return LGVDSelection(
            nothing, 0, 0, target_residual, 0.0, :shrewd_c2_c3)
    nt = numout(target_factor)
    target_grouped = permute(
        target_factor,
        ((nt + 1,), (ntuple(identity, nt)..., nt + 2)))
    _, target_s, target_v, c2_discarded = split_svd_with_error(
        target_grouped,
        TruncationScheme(rtol=strategy.preselection_threshold))
    dprime = _cbe_rank(target_s)
    dprime == 0 && return LGVDSelection(
        nothing, 0, 0, Float64(c2_discarded), 0.0, :shrewd_c2_c3)

    # C2(c): redirect the target external legs and combine Dprime with the
    # TTNO edge. The numerical cutoff removes only exact-rank debris and is
    # intentionally distinct from epsilon-prime and epsilon-tilde.
    preselection_core = target_s * target_v
    regrouped = permute(
        preselection_core,
        (ntuple(i -> i + 1, nt), (1, nt + 2)))
    preselected, _, _, regroup_discarded = split_svd_with_error(
        regrouped,
        TruncationScheme(
            atol=numerical_floor * max(Float64(norm(regrouped)), 1.0),
            rtol=numerical_floor))
    preselected_rank = _cbe_rank(preselected)
    c2_error = hypot(Float64(c2_discarded),
                     Float64(regroup_discarded))
    preselected_rank == 0 && return LGVDSelection(
        nothing, dprime, 0, c2_error, 0.0, :shrewd_c2_c3)

    # C3(d): close the network only inside the preselected target frame,
    # project the source kept range, and truncate Dhat -> Dtilde at the
    # independent final-selection threshold.
    final_core = contract_projected_two_site(
        cache, factor_frame, link, preselected)
    final_scale = Float64(norm(final_core))
    final_core -= source * (source' * final_core)
    final_residual = Float64(norm(final_core))
    final_residual <= numerical_floor * max(final_scale, 1.0) &&
        return LGVDSelection(
            nothing, preselected_rank, 0, c2_error,
            final_residual, :shrewd_c2_c3)
    _, _, final_v, c3_discarded = split_svd_with_error(
        final_core,
        TruncationScheme(maxdim=strategy.max_add,
                         rtol=strategy.final_selection_threshold))
    selected_factor = transpose(final_v)
    selected_rank = _cbe_rank(selected_factor)
    complement = selected_rank == 0 ? nothing : preselected * selected_factor
    return LGVDSelection(
        complement, preselected_rank, selected_rank,
        c2_error, Float64(c3_discarded), :shrewd_c2_c3)
end

function step!(ev::TDVP1_CBE{LGVDCBE}, ψ::TTNS, H::TTNO, dz::Number)
    ev.enabled || return invoke(
        step!, Tuple{Union{TDVP1,TDVP1_CBE},TTNS,TTNO,Number},
        ev, ψ, H, dz)
    _lgvd_preflight(ev, ψ, H)
    return _lgvd_step_with_selector!(
        ev, ψ, H, dz, _lgvd_shrewd_selector;
        selector_name=:shrewd_c2_c3, shrewd_applied=true)
end

# Strict LGVD never reaches the legacy post-site split seam. Keep a defensive
# fail-closed method in case an internal caller bypasses the specialized step!.
function _split_link_up(::LGVDCBE, ::TDVP1_CBE, ::TTNS, ::TTNO,
                        ::Int, ::Int, ::Number)
    throw(LGVDShrewdSelectionUnavailable(:legacy_split_seam))
end
function _split_link_down(strategy::LGVDCBE, ev::TDVP1_CBE,
                          ψ::TTNS, H::TTNO, u::Int, v::Int, dz::Number)
    return _split_link_up(strategy, ev, ψ, H, u, v, dz)
end
