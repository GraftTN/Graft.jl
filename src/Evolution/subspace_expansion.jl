# Subspace-expansion TDVP wrappers. The expansion itself is the shared
# Contractions.expand! primitive, keeping GroundState and Evolution dependent
# on the same lower-layer implementation.

"""
    GSE_TDVP(; order=2, trunc, max_add=8, mixing=1.0,
             expand_scheme=:exact, rng=nothing, ...)

Global-subspace-expansion TDVP: expand every bond once, then run a TDVP1 step
on the enlarged manifold. `expand_scheme=:rsvd` requires an explicit `rng`.
"""
Base.@kwdef mutable struct GSE_TDVP <: Evolver
    order::Int = 2
    trunc::TruncationScheme = TruncationScheme(; maxdim=100)
    max_add::Int = 8
    mixing::Float64 = 1.0
    expand_scheme::Symbol = :exact
    rng::Union{Nothing,AbstractRNG} = nothing
    rsvd_oversample::Int = 8
    rsvd_poweriter::Int = 0
    enr_rtol::Float64 = 1e-10
    enr_atol::Float64 = 1e-12
    krylovdim::Int = 30
    tol::Float64 = 1e-12
    cache::Union{Nothing,EnvCache} = nothing
end

"""
    LSE_TDVP(; order=2, trunc, max_add=8, mixing=1.0,
             expand_scheme=:exact, rng=nothing, ...)

Local-subspace-expansion TDVP: expand bonds before each TDVP1 sweep direction.
This keeps the same one-site projector-splitting skeleton as TDVP1 while
refreshing the manifold locally at every half step.
"""
Base.@kwdef mutable struct LSE_TDVP <: Evolver
    order::Int = 2
    trunc::TruncationScheme = TruncationScheme(; maxdim=100)
    max_add::Int = 8
    mixing::Float64 = 1.0
    expand_scheme::Symbol = :exact
    rng::Union{Nothing,AbstractRNG} = nothing
    rsvd_oversample::Int = 8
    rsvd_poweriter::Int = 0
    enr_rtol::Float64 = 1e-10
    enr_atol::Float64 = 1e-12
    krylovdim::Int = 30
    tol::Float64 = 1e-12
    cache::Union{Nothing,EnvCache} = nothing
end

function step!(ev::GSE_TDVP, ψ::TTNS, H::TTNO, dz::Number)
    cache = _prepare_subspace_expansion!(ev, ψ, H, dz, "GSE_TDVP")
    _expand_all_bonds!(ev, ψ, H, cache; rev=false)
    base = TDVP1(; order=ev.order, krylovdim=ev.krylovdim, tol=ev.tol, cache)
    step!(base, ψ, H, dz)
    ev.cache = base.cache
    return ψ
end

function step!(ev::LSE_TDVP, ψ::TTNS, H::TTNO, dz::Number)
    cache = _prepare_subspace_expansion!(ev, ψ, H, dz, "LSE_TDVP")
    base = TDVP1(; order=1, krylovdim=ev.krylovdim, tol=ev.tol, cache)
    if ev.order == 1
        _expand_all_bonds!(ev, ψ, H, cache; rev=false)
        _tdvp1_sweep!(base, ψ, H, dz; rev=false)
    elseif ev.order == 2
        _expand_all_bonds!(ev, ψ, H, cache; rev=false)
        _tdvp1_sweep!(base, ψ, H, dz / 2; rev=false)
        _expand_all_bonds!(ev, ψ, H, cache; rev=true)
        _tdvp1_sweep!(base, ψ, H, dz / 2; rev=true)
    else
        throw(ArgumentError("order must be 1 or 2"))
    end
    ev.cache = base.cache
    return ψ
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
                rsvd_poweriter=ev.rsvd_poweriter)
    end
    return ψ
end
