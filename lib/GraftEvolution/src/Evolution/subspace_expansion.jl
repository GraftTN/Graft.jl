# Subspace-expansion TDVP wrappers. The expansion itself is the shared
# Contractions.expand! primitive, keeping GroundState and Evolution dependent
# on the same lower-layer implementation.

"""
    TDVP1_LSE(; order=2, trunc, max_add=8, mixing=1.0,
              expand_scheme=:exact, rng=nothing, eager=true, verbose=true, ...)

Local-subspace-expansion TDVP: expand bonds before each TDVP1 sweep direction.
This keeps the same one-site projector-splitting skeleton as TDVP1 while
refreshing the manifold locally at every half step. `:rsvd` and `:rangefinder`
share the `rsvd_*` sketch controls and require an explicit `rng`; `:directqr`
does not. With `verbose=true`, emits step and per-direction expansion `@info`
records.
"""
Base.@kwdef mutable struct TDVP1_LSE <: Evolver
    order::Int = 2
    trunc::TruncationScheme = TruncationScheme(; maxdim=100)
    max_add::Int = 8
    mixing::Float64 = 1.0
    expand_scheme::Symbol = :exact
    rng::Union{Nothing,AbstractRNG} = nothing
    rsvd_oversample::Int = 8
    rsvd_poweriter::Int = 0
    rsvd_threaded::Bool = Base.Threads.nthreads() > 1
    rsvd_minbatch::Int = max(2, Base.Threads.nthreads())
    rsvd_memory_cap_bytes::Union{Nothing,Int} = nothing
    rsvd_task_workspace_bytes::Union{Nothing,Int} = nothing
    rsvd_fanout_diagnostics::Union{Nothing,Base.RefValue} = nothing
    enr_rtol::Float64 = 1e-10
    enr_atol::Float64 = 1e-12
    krylovdim::Int = 30
    tol::Float64 = 1e-12
    eager::Bool = true
    contraction_optimize::Bool = true
    contraction_sector_aware::Bool = true
    threaded_channels::Bool = false
    channel_slices::Int = 2
    channel_minbatch::Int = 2
    channel_min_flops::Real = 1_000_000
    channel_memory_cap_bytes::Union{Nothing,Real} = nothing
    distributed::Union{Nothing,AbstractDistributedContext} = nothing
    verbose::Bool = true
    cache::Union{Nothing,EnvCache} = nothing
end

function step!(ev::TDVP1_LSE, ψ::TTNS, H::TTNO, dz::Number)
    cache_reused = ev.cache !== nothing && ev.cache.topo == ψ.topo
    initial_maxbond = ev.verbose ? _tdvp_max_bond_dim(ψ) : 0
    cache = _prepare_subspace_expansion!(ev, ψ, H, dz, "TDVP1_LSE")
    ev.verbose && _log_subspace_tdvp_start("TDVP1_LSE", ev, ψ, H, dz; cache_reused)
    base = TDVP1(; order=1, krylovdim=ev.krylovdim, tol=ev.tol,
                 eager=ev.eager,
                 threaded_channels=ev.threaded_channels,
                 channel_slices=ev.channel_slices,
                 channel_minbatch=ev.channel_minbatch,
                 channel_min_flops=ev.channel_min_flops,
                 channel_memory_cap_bytes=ev.channel_memory_cap_bytes,
                 distributed=ev.distributed,
                 verbose=false, cache)
    if ev.order == 1
        maxbond_before = ev.verbose ? _tdvp_max_bond_dim(ψ) : 0
        _expand_all_bonds!(ev, ψ, H, cache; rev=false)
        ev.verbose && _log_subspace_expansion("TDVP1_LSE", ψ;
                                              rev=false, maxbond_before)
        _tdvp1_sweep!(base, ψ, H, dz; rev=false)
    elseif ev.order == 2
        maxbond_before = ev.verbose ? _tdvp_max_bond_dim(ψ) : 0
        _expand_all_bonds!(ev, ψ, H, cache; rev=false)
        ev.verbose && _log_subspace_expansion("TDVP1_LSE", ψ;
                                              rev=false, maxbond_before)
        _tdvp1_sweep!(base, ψ, H, dz / 2; rev=false)
        maxbond_before = ev.verbose ? _tdvp_max_bond_dim(ψ) : 0
        _expand_all_bonds!(ev, ψ, H, cache; rev=true)
        ev.verbose && _log_subspace_expansion("TDVP1_LSE", ψ;
                                              rev=true, maxbond_before)
        _tdvp1_sweep!(base, ψ, H, dz / 2; rev=true)
    else
        throw(ArgumentError("order must be 1 or 2"))
    end
    ev.cache = base.cache
    ev.verbose && _log_subspace_tdvp_complete("TDVP1_LSE", ψ; initial_maxbond)
    return ψ
end

function _log_subspace_tdvp_start(name::String, ev, ψ::TTNS, H::TTNO,
                                  dz::Number; cache_reused::Bool)
    t = ψ.topo
    nodes = nnodes(t)
    physical_sites = count(identity, ψ.hasphys)
    bonds = nodes - 1
    center_site = _tdvp_center_site(ψ)
    initial_maxbond = _tdvp_max_bond_dim(ψ)
    order = ev.order
    krylovdim = ev.krylovdim
    tol = ev.tol
    eager = ev.eager
    hermitian = ishermitian(H)
    expand_scheme = ev.expand_scheme
    max_add = ev.max_add
    mixing = ev.mixing
    enr_rtol = ev.enr_rtol
    enr_atol = ev.enr_atol
    rsvd_oversample = ev.rsvd_oversample
    rsvd_poweriter = ev.rsvd_poweriter
    rsvd_threaded = ev.rsvd_threaded
    rsvd_minbatch = ev.rsvd_minbatch
    rsvd_memory_cap_bytes = ev.rsvd_memory_cap_bytes
    rsvd_task_workspace_bytes = ev.rsvd_task_workspace_bytes
    trunc_maxdim = ev.trunc.maxdim
    trunc_atol = ev.trunc.atol
    trunc_rtol = ev.trunc.rtol
    trunc_discarded_weight = ev.trunc.discarded_weight
    @info "$name step start" dz order nodes physical_sites bonds center_site initial_maxbond krylovdim tol eager hermitian cache_reused expand_scheme max_add mixing enr_rtol enr_atol rsvd_oversample rsvd_poweriter rsvd_threaded rsvd_minbatch rsvd_memory_cap_bytes rsvd_task_workspace_bytes trunc_maxdim trunc_atol trunc_rtol trunc_discarded_weight
    return nothing
end

function _log_subspace_expansion(name::String, ψ::TTNS;
                                 rev::Bool, maxbond_before::Int)
    direction = rev ? :reverse : :forward
    bonds_visited = nnodes(ψ.topo) - 1
    center_site = _tdvp_center_site(ψ)
    maxbond_after = _tdvp_max_bond_dim(ψ)
    @info "$name expansion complete" direction bonds_visited center_site maxbond_before maxbond_after
    return nothing
end

function _log_subspace_tdvp_complete(name::String, ψ::TTNS;
                                     initial_maxbond::Int)
    center_site = _tdvp_center_site(ψ)
    final_maxbond = _tdvp_max_bond_dim(ψ)
    @info "$name step complete" center_site initial_maxbond final_maxbond
    return nothing
end

function _prepare_subspace_expansion!(ev, ψ::TTNS, H::TTNO, dz::Number, name::String)
    topology(ψ) == topology(H) ||
        throw(ArgumentError("$name: H and ψ have mismatched topologies"))
    ψ.hasphys == H.hasphys ||
        throw(ArgumentError("$name: H and ψ have mismatched physical layout"))
    spacetype(ψ) == spacetype(H) ||
        throw(ArgumentError("$name: H and ψ have mismatched spacetype"))
    if !(eltype(ψ) <: Complex) && dz isa Complex && !isreal(dz)
        throw(ArgumentError("$name complex-step evolution requires a complex-eltype TTNS; convert explicitly"))
    end
    ev.max_add >= 0 || throw(ArgumentError("$name: max_add must be nonnegative"))
    ev.mixing >= 0 || throw(ArgumentError("$name: mixing must be nonnegative"))
    ev.expand_scheme in (:exact, :rsvd, :directqr, :rangefinder) ||
        throw(ArgumentError(
            "$name: expand_scheme must be :exact, :rsvd, :directqr, or :rangefinder"))
    ev.expand_scheme in (:rsvd, :rangefinder) && ev.rng === nothing &&
        throw(ArgumentError(
            "$name: expand_scheme=$(ev.expand_scheme) requires an explicit rng (§9.6)"))
    ev.rsvd_oversample >= 0 ||
        throw(ArgumentError("$name: rsvd_oversample must be nonnegative"))
    ev.rsvd_poweriter >= 0 ||
        throw(ArgumentError("$name: rsvd_poweriter must be nonnegative"))
    ev.rsvd_minbatch >= 1 || throw(ArgumentError("$name: rsvd_minbatch must be positive"))
    ev.rsvd_memory_cap_bytes === nothing ||
        ev.rsvd_memory_cap_bytes >= 0 ||
        throw(ArgumentError("$name: rsvd_memory_cap_bytes must be nonnegative"))
    ev.rsvd_task_workspace_bytes === nothing ||
        ev.rsvd_task_workspace_bytes >= 0 ||
        throw(ArgumentError(
            "$name: rsvd_task_workspace_bytes must be nonnegative"))
    if ev.cache === nothing || ev.cache.topo != ψ.topo
        ev.cache = EnvCache(ψ.topo)
    end
    return ev.cache::EnvCache
end

function _expand_all_bonds!(ev, ψ::TTNS, H::TTNO, cache::EnvCache; rev::Bool)
    t = ψ.topo
    bonds = [n for n in postorder(t) if t.parent[n] != 0]
    for n in (rev ? Iterators.reverse(bonds) : bonds)
        expand!(ψ, H, (n, t.parent[n]); scheme=ev.expand_scheme, cache,
                rng=ev.rng, trunc=ev.trunc, max_add=ev.max_add,
                mixing=ev.mixing, enr_rtol=ev.enr_rtol,
                enr_atol=ev.enr_atol, rsvd_oversample=ev.rsvd_oversample,
                rsvd_poweriter=ev.rsvd_poweriter,
                rsvd_threaded=ev.rsvd_threaded,
                rsvd_minbatch=ev.rsvd_minbatch,
                rsvd_memory_cap_bytes=ev.rsvd_memory_cap_bytes,
                rsvd_task_workspace_bytes=ev.rsvd_task_workspace_bytes,
                rsvd_fanout_diagnostics=ev.rsvd_fanout_diagnostics,
                contraction_optimize=ev.contraction_optimize,
                contraction_sector_aware=ev.contraction_sector_aware,
                threaded_channels=ev.threaded_channels,
                channel_slices=ev.channel_slices,
                channel_minbatch=ev.channel_minbatch,
                channel_min_flops=ev.channel_min_flops,
                channel_memory_cap_bytes=ev.channel_memory_cap_bytes,
                distributed=ev.distributed)
    end
    return ψ
end
