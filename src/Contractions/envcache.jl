# EnvCache — partial-environment cache keyed by directed edge
# (PyTreeNet: contractions/tree_cach_dict.PartialTreeCachDict, keys
# (node_id, next_node_id) with the same "everything behind node_id, looking
# towards next_node_id" semantics).

"""
    EnvCache(topo::TreeTopology)

Cache of sandwich environments keyed by directed edge `(u, v)`. Holds only the
topology reference and tensors — deliberately serializable/shippable (MPI
subtree dispatch unit, §8; big-but-rebuildable: checkpoints drop it by
default, §7).
"""
struct EnvCache
    topo::TreeTopology
    envs::Dict{Tuple{Int,Int},AbstractTensorMap}
end
EnvCache(topo::TreeTopology) = EnvCache(topo, Dict{Tuple{Int,Int},AbstractTensorMap}())

Base.haskey(c::EnvCache, key::Tuple{Int,Int}) = haskey(c.envs, key)
Base.getindex(c::EnvCache, key::Tuple{Int,Int}) = c.envs[key]
Base.empty!(c::EnvCache) = (empty!(c.envs); c)

# is node `n` on the `u`-side of the directed edge (u, v)?
function _on_side(t::TreeTopology, n::Int, u::Int, v::Int)
    n == u && return true
    n == v && return false
    return u in path_between(t, n, v)
end

"""
    invalidate_node!(cache::EnvCache, n) -> cache

Drop every cached environment whose contracted side contains node `n`. Fired
by `Networks.update_tensor!` (§9.2) — the explicit invalidation event of §3.
"""
function invalidate_node!(c::EnvCache, n::Int)
    filter!(p -> !_on_side(c.topo, n, p.first[1], p.first[2]), c.envs)
    return c
end

"""
    invalidate_edge!(cache::EnvCache, n, m) -> cache

Invalidation event for a gauge move across the edge `(n, m)`: both node
tensors changed, so every environment whose side touches `n` or `m` dies.
"""
function invalidate_edge!(c::EnvCache, n::Int, m::Int)
    filter!(p -> !(_on_side(c.topo, n, p.first[1], p.first[2]) ||
                   _on_side(c.topo, m, p.first[1], p.first[2])), c.envs)
    return c
end

# ---------------------------------------------------------------------------
# generic sandwich contraction around one node
# ---------------------------------------------------------------------------

# Flat-leg index of the leg of node `u` pointing towards neighbour `w`, for a
# TTNS tensor (children slots 1..K, physical K+1 if present, parent last).
_stateleg(t::TreeTopology, hasphys_u::Bool, u::Int, w::Int) =
    t.parent[u] == w ? nchildren(t, u) + (hasphys_u ? 1 : 0) + 1 : childslot(t, u, w)

# Same for a TTNO tensor (children 1..K, P_out K+1, P_in K+2, parent last).
_opleg(t::TreeTopology, hasphys_u::Bool, u::Int, w::Int) =
    t.parent[u] == w ? nchildren(t, u) + (hasphys_u ? 2 : 0) + 1 : childslot(t, u, w)

"""
    build_env(ket, O, bra, u, v, envs) -> AbstractTensorMap

Contract the node sandwich at `u`, leaving the legs towards `v` open, consuming
`envs[(w, u)]` for every other neighbour `w`. `O === nothing` gives the rank-2
transfer environment (ket, bra); otherwise rank-3 (ket, op, bra). This one
function produces both "below" and "above" environments (module docstring).

If `u` is the root, the ket/op/bra parent legs (unit or global-charge spaces)
are closed with a unit cap. With the sentinel `v = 0` nothing stays open and
the fully contracted scalar is returned (used by `inner`/`expect`).
"""
function build_env(ket::TTNS, O::Union{TTNO,Nothing}, bra::TTNS, u::Int, v::Int,
                   envs::Dict{Tuple{Int,Int},AbstractTensorMap})
    t = ket.topo
    hp = hasphys(ket, u)
    A = ket.tensors[u]
    B = bra.tensors[u]
    W = O === nothing ? nothing : O.tensors[u]
    O === nothing || hasphys(O, u) == hp ||
        throw(ArgumentError("TTNO/TTNS physical-leg mismatch at node $(nodeid(t, u))"))

    aidx = zeros(Int, numind(A))
    bidx = zeros(Int, numind(B))
    widx = W === nothing ? Int[] : zeros(Int, numind(W))
    tensors = Any[A]
    indices = Vector{Int}[aidx]
    conj = Bool[false]

    nxt = Ref(0)
    fresh() = (nxt[] += 1; nxt[])

    # physical legs
    if hp
        pk = physleg(ket, u)
        if W === nothing
            lbl = fresh()
            aidx[pk] = lbl
            bidx[pk] = lbl
        else
            K = nchildren(t, u)
            pin = fresh()               # ket P  ↔ W P_in
            pout = fresh()              # W P_out ↔ bra P
            aidx[pk] = pin
            widx[K + 2] = pin
            widx[K + 1] = pout
            bidx[pk] = pout
        end
    end

    # neighbours
    for w in neighbors(t, u)
        la = _stateleg(t, hp, u, w)
        if w == v
            aidx[la] = -1
            W === nothing || (widx[_opleg(t, hp, u, w)] = -2)
            bidx[la] = W === nothing ? -2 : -3
        else
            E = envs[(w, u)]
            eidx = zeros(Int, numind(E))
            eidx[1] = fresh(); aidx[la] = eidx[1]                       # ket
            if W !== nothing
                eidx[2] = fresh(); widx[_opleg(t, hp, u, w)] = eidx[2]  # op
            end
            eidx[end] = fresh(); bidx[la] = eidx[end]                   # bra
            push!(tensors, E); push!(indices, eidx); push!(conj, false)
        end
    end

    # close the root's parent legs (unit / global-charge spaces) with a cap
    if t.parent[u] == 0
        ka, ko, kb = fresh(), fresh(), fresh()
        aidx[end] = ka
        bidx[end] = kb
        Vroot = domain(A)[1]
        if W === nothing
            cap = Backend.ones_tensor(scalartype(A), dual(domain(B)[1]) ⊗ Vroot)
            push!(tensors, cap); push!(indices, [kb, ka]); push!(conj, false)
        else
            widx[end] = ko
            cap = Backend.ones_tensor(scalartype(A), dual(domain(B)[1]) ⊗ domain(W)[numin(W)] ⊗ Vroot)
            push!(tensors, cap); push!(indices, [kb, ko, ka]); push!(conj, false)
        end
    end

    if W !== nothing
        push!(tensors, W); push!(indices, widx); push!(conj, false)
    end
    push!(tensors, B); push!(indices, bidx); push!(conj, true)

    return ncon(tensors, indices, conj)
end

"""
    env!(cache, ket, O, bra, u, v) -> AbstractTensorMap

Memoized recursive environment for the directed edge `(u, v)`; builds (and
caches) all environments of the `u`-side subtree that are missing.
"""
function env!(c::EnvCache, ket::TTNS, O::Union{TTNO,Nothing}, bra::TTNS, u::Int, v::Int)
    return get!(c.envs, (u, v)) do
        for w in neighbors(c.topo, u)
            w == v && continue
            env!(c, ket, O, bra, w, u)
        end
        build_env(ket, O, bra, u, v, c.envs)
    end
end
env!(c::EnvCache, ket::TTNS, O::Union{TTNO,Nothing}, u::Int, v::Int) =
    env!(c, ket, O, ket, u, v)
