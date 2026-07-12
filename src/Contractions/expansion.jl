"""
    expand!(ψ, H, edge; scheme=:exact, cache=nothing, rng=nothing,
            trunc, max_add=8, mixing=1, enr_rtol=1e-10, enr_atol=1e-12,
            rsvd_oversample=8, rsvd_poweriter=0) -> ψ

Shared bond-expansion primitive (§5a/§11.7). `edge` is `(child, parent)` or
`child => parent` using node ids or indices. `scheme=:exact` forms the
predictor basis with a deterministic SVD. `scheme=:rsvd` uses explicit-RNG
blockwise randomized probes on the fused rest space and never touches global
randomness.
"""
function expand!(ψ::TTNS, H::TTNO, edge; scheme::Symbol=:exact,
                 cache::Union{Nothing,EnvCache}=nothing,
                 rng::Union{Nothing,AbstractRNG}=nothing,
                 trunc::TruncationScheme=TruncationScheme(; maxdim=100),
                 max_add::Int=8, mixing::Number=one(Float64),
                 enr_rtol::Float64=1e-10, enr_atol::Float64=1e-12,
                 rsvd_oversample::Int=8, rsvd_poweriter::Int=0)
    scheme in (:exact, :rsvd) ||
        throw(ArgumentError("expand!: scheme must be :exact or :rsvd"))
    max_add >= 0 || throw(ArgumentError("expand!: max_add must be nonnegative"))
    rsvd_oversample >= 0 || throw(ArgumentError("expand!: rsvd_oversample must be nonnegative"))
    rsvd_poweriter >= 0 || throw(ArgumentError("expand!: rsvd_poweriter must be nonnegative"))
    scheme === :rsvd && rng === nothing &&
        throw(ArgumentError("expand!: scheme=:rsvd requires an explicit rng (§9.6)"))
    iszero(mixing) && return ψ
    t = ψ.topo
    n, m = _edge_child_parent(t, edge)
    olddim = dim(virtualspace(ψ, n))
    cap = min(trunc.maxdim, olddim + max_add)
    cap > olddim || return ψ

    c = cache === nothing ? EnvCache(t) : cache
    move_center!(ψ, n; cache=c)
    Θ = two_site_tensor(ψ, n, m)
    h2 = eff_h2(c, ψ, H, n, m)
    PΘ = mixing * h2(Θ)
    P = _child_predictor_basis(ψ, PΘ, n, cap; scheme, rng,
                               rsvd_oversample, rsvd_poweriter)
    U, R = _expand_enrich_split(ψ.tensors[n], P; maxdim=cap,
                                max_add=cap - olddim,
                                enr_rtol, enr_atol)
    dim(domain(U)) == olddim && return ψ
    ψ.tensors[n] = U
    R = Networks._pivotal_link(R)
    ψ.tensors[m] = absorb_on_leg(ψ.tensors[m], R, childslot(t, m, n))
    ψ.center = m
    invalidate_edge!(c, n, m)
    return ψ
end

function _edge_child_parent(t::TreeTopology, edge::Pair)
    a, b = nodeindex(t, edge.first), nodeindex(t, edge.second)
    return _orient_edge(t, a, b)
end
function _edge_child_parent(t::TreeTopology, edge::Tuple{Any,Any})
    a, b = nodeindex(t, edge[1]), nodeindex(t, edge[2])
    return _orient_edge(t, a, b)
end
function _orient_edge(t::TreeTopology, a::Int, b::Int)
    if t.parent[a] == b
        return a, b
    elseif t.parent[b] == a
        return b, a
    else
        throw(ArgumentError("expand!: edge endpoints $(nodeid(t, a)), $(nodeid(t, b)) are not adjacent"))
    end
end

function _child_predictor_basis(ψ::TTNS, PΘ::AbstractTensorMap, n::Int, maxdim::Int;
                                scheme::Symbol=:exact,
                                rng::Union{Nothing,AbstractRNG}=nothing,
                                rsvd_oversample::Int=8,
                                rsvd_poweriter::Int=0)
    pn = numout(ψ.tensors[n])
    NP = numind(PΘ)
    Ps = permute(PΘ, (ntuple(identity, pn), ntuple(j -> pn + j, NP - pn)))
    if scheme === :rsvd
        return _rsvd_predictor_basis(Ps, maxdim; rng, rsvd_oversample,
                                     rsvd_poweriter)
    end
    U, _, _ = split_svd(Ps, TruncationScheme(; maxdim))
    return U
end

function _rsvd_predictor_basis(Ps::AbstractTensorMap, maxdim::Int;
                               rng::AbstractRNG,
                               rsvd_oversample::Int,
                               rsvd_poweriter::Int)
    Vrest = fuse(domain(Ps))
    budget = min(dim(Vrest), maxdim + rsvd_oversample)
    K = _rsvd_probe_space(Vrest, budget)
    Ω = randn(rng, scalartype(Ps), domain(Ps) ← K)
    Y = Ps * Ω
    for _ in 1:rsvd_poweriter
        Y = Ps * (Ps' * Y)
    end
    U, _, _ = split_svd(Y, TruncationScheme(; maxdim))
    return U
end

function _rsvd_probe_space(::ComplexSpace, budget::Int)
    return ℂ^budget
end

function _rsvd_probe_space(Vrest::S, budget::Int) where {S<:ElementarySpace}
    Q = sectortype(Vrest)
    dims = Pair{Q,Int}[]
    for q in sectors(Vrest)
        kq = min(dim(Vrest, q), budget)
        kq > 0 && push!(dims, q => kq)
    end
    isempty(dims) && throw(ArgumentError("expand!: randomized probe space is empty"))
    return Vect[Q](dims...)
end

function _expand_enrich_split(A::AbstractTensorMap, P::AbstractTensorMap;
                              maxdim::Int, max_add::Int,
                              enr_rtol::Float64, enr_atol::Float64)
    expanded = A
    room = min(max_add, maxdim - dim(domain(A)))
    if room > 0 && dim(codomain(A)) > dim(domain(A))
        N = left_null(A)
        if dim(domain(N)) > 0
            M = N' * P
            Um, _, _ = split_svd(M, TruncationScheme(; maxdim=room,
                                                       atol=enr_atol,
                                                       rtol=enr_rtol))
            if dim(domain(Um)) > 0
                E = N * Um
                if isdual(domain(A)[1]) != isdual(domain(E)[1])
                    E = flip(E, numind(E))
                end
                expanded = catdomain(A, E)
            end
        end
    end
    U, _, _ = split_svd(expanded, TruncationScheme(; maxdim))
    R = U' * A
    return U, R
end

"""
    _physless_root_growth_targets(ψ, trunc, max_add) -> Vector{Tuple{Int,Int}}

Per-edge target dimensions for the two children of a physical-leg-free binary
root.  A two-site predictor on either root edge is rank-limited by its sibling
edge: when both start at the same narrow dimension, neither one can provide the
other with a larger predictor basis.  Record their ordinary per-sweep caps
before any one-edge expansion so the paired bootstrap never exceeds `max_add`.
"""
function _physless_root_growth_targets(ψ::TTNS, trunc::TruncationScheme,
                                       max_add::Int)
    t = ψ.topo
    root = t.root
    (!hasphys(ψ, root) && length(t.children[root]) == 2) || return Tuple{Int,Int}[]
    return [(n, min(trunc.maxdim, dim(virtualspace(ψ, n)) + max_add))
            for n in t.children[root]]
end

"""
    _physless_root_two_site_targets(ψ, trunc) -> Vector{Tuple{Int,Int}}

Joint root-edge targets for two-site DMRG.  Unlike 3S, two-site DMRG has no
per-sweep `max_add`: its SVD may retain every Schmidt direction allowed by
`trunc`.  At a physical-leg-free binary root, open both child legs to that
ordinary two-site cap before the first root-edge update.  The common target is
also bounded by both child-side codomain dimensions, so an unbounded
`TruncationScheme()` never requests an artificial infinite virtual space.
"""
function _physless_root_two_site_targets(ψ::TTNS, trunc::TruncationScheme)
    t = ψ.topo
    root = t.root
    children = t.children[root]
    (!hasphys(ψ, root) && length(children) == 2) || return Tuple{Int,Int}[]
    target = min(trunc.maxdim, minimum(dim(codomain(ψ.tensors[n])) for n in children))
    return [(n, target) for n in children]
end

"""
    _null_enrich_split(A; maxdim) -> (U, R)

State-preserving completion of the column space of `A`: `A == U * R` while
`U` contains up to `maxdim` orthonormal columns.  This is only the bootstrap
fallback for the two sibling edges of a physical-leg-free binary root.  The
Hamiltonian-selected predictor remains the primary enrichment everywhere else;
without this paired completion its rank is bounded by the opposite root edge
and cannot start the growth at all.
"""
function _null_enrich_split(A::AbstractTensorMap; maxdim::Int)
    olddim = dim(domain(A))
    room = maxdim - olddim
    room > 0 && dim(codomain(A)) > olddim || return A, nothing

    N = left_null(A)
    dim(domain(N)) > 0 || return A, nothing
    E, _, _ = split_svd(N, TruncationScheme(; maxdim=room))
    if isdual(domain(A)[1]) != isdual(domain(E)[1])
        E = flip(E, numind(E))
    end
    U, _, _ = split_svd(catdomain(A, E), TruncationScheme(; maxdim))
    return U, U' * A
end

"""
    _bootstrap_physless_root!(ψ, cache, targets) -> ψ

Jointly complete the two root-child subspaces recorded in `targets`.  Each
completion is a gauge-preserving zero-weight embedding, so it cannot alter the
TTNS vector.  Performing both before the next one-site root update removes the
otherwise circular rank bound between the siblings.
"""
function _bootstrap_physless_root!(ψ::TTNS, cache::EnvCache,
                                   targets::Vector{Tuple{Int,Int}})
    isempty(targets) && return ψ
    t = ψ.topo
    root = t.root
    for (n, target) in targets
        t.parent[n] == root || throw(ArgumentError("root bootstrap target is not a root child"))
        dim(virtualspace(ψ, n)) >= target && continue
        move_center!(ψ, n; cache)
        U, R = _null_enrich_split(ψ.tensors[n]; maxdim=target)
        R === nothing && continue
        ψ.tensors[n] = U
        R = Networks._pivotal_link(R)
        ψ.tensors[root] = absorb_on_leg(ψ.tensors[root], R, childslot(t, root, n))
        ψ.center = root
        invalidate_edge!(cache, n, root)
    end
    return ψ
end
