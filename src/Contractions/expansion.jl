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
