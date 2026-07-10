# EnvCache — partial-environment cache keyed by directed edge
# (PyTreeNet: contractions/tree_cach_dict.PartialTreeCachDict, keys
# (node_id, next_node_id) with the same "everything behind node_id, looking
# towards next_node_id" semantics).

"""
    EnvCache(topo::TreeTopology)

Cache of sandwich environments keyed by directed edge `(u, v)`, plus compiled
shape-only effective-Hamiltonian plans. Both classes are deliberately
rebuildable: checkpoints drop the whole cache, while gauge invalidation drops
only value-dependent environments and retains plans whose space signatures
still match (§3).
"""
mutable struct EnvCache
    topo::TreeTopology
    envs::Dict{Tuple{Int,Int},AbstractTensorMap}
    plans::Dict{PlanKey,ContractionPlan}
    rootcaps::Dict{Tuple,AbstractTensorMap}
    plan_hits::Int
    plan_misses::Int
end
EnvCache(topo::TreeTopology) =
    EnvCache(topo, Dict{Tuple{Int,Int},AbstractTensorMap}(),
             Dict{PlanKey,ContractionPlan}(), Dict{Tuple,AbstractTensorMap}(),
             0, 0)
EnvCache(topo::TreeTopology, envs::Dict{Tuple{Int,Int},AbstractTensorMap}) =
    EnvCache(topo, envs, Dict{PlanKey,ContractionPlan}(),
             Dict{Tuple,AbstractTensorMap}(), 0, 0)
# Source-compatible constructor for callers that built an EnvCache with the
# pre-root-cap field layout.
EnvCache(topo::TreeTopology, envs::Dict{Tuple{Int,Int},AbstractTensorMap},
         plans::Dict{PlanKey,ContractionPlan}, plan_hits::Integer,
         plan_misses::Integer) =
    EnvCache(topo, envs, plans, Dict{Tuple,AbstractTensorMap}(),
             Int(plan_hits), Int(plan_misses))

Base.haskey(c::EnvCache, key::Tuple{Int,Int}) = haskey(c.envs, key)
Base.getindex(c::EnvCache, key::Tuple{Int,Int}) = c.envs[key]
Base.empty!(c::EnvCache) = (empty!(c.envs); empty!(c.plans); empty!(c.rootcaps);
                            c.plan_hits = 0; c.plan_misses = 0; c)

"""
Observable EffectiveMap plan-cache state for tests and solver diagnostics.

Environment/value-level plans share `plans` but intentionally do not change
these historical Krylov-map counters; their reuse remains visible through the
shape-only dictionary until the cache-wide accounting policy is added.
"""
plan_cache_stats(c::EnvCache) =
    (hits=c.plan_hits, misses=c.plan_misses, size=length(c.plans))

"""Return a shape-owned, immutable-in-use root cap for a planned network."""
function _root_cap!(c::EnvCache, T::DataType, capspace)
    return get!(c.rootcaps, (T, capspace)) do
        Backend.ones_tensor(T, capspace)
    end
end

"""Look up a shape-only plan without changing EffectiveMap hit/miss counters."""
function _planned_execute!(c::EnvCache, kind::Symbol, spec::ContractionSpec,
                           operands::Tuple, T::DataType)
    plan, _ = Planning.get_or_plan!(c.plans, kind, spec, operands, T)
    return Planning.execute(plan, operands)
end

function _effective_map!(c::EnvCache, kind::Symbol, spec::ContractionSpec,
                         protos, statics::Tuple, T::DataType;
                         optimize::Bool=true, memory_weight::Real=1,
                         sector_aware::Bool=true,
                         memory_cap_bytes::Union{Nothing,Real}=nothing)
    plan, hit = Planning.get_or_plan!(c.plans, kind, spec, protos, T;
                                      optimize=optimize, memory_weight=memory_weight,
                                      sector_aware=sector_aware,
                                      memory_cap_bytes=memory_cap_bytes)
    if hit
        c.plan_hits += 1
    else
        c.plan_misses += 1
    end
    return EffectiveMap(plan, statics)
end

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
    _build_env_ncon_reference(ket, O, bra, u, v, envs)

Contract the node sandwich at `u`, leaving the legs towards `v` open, consuming
`envs[(w, u)]` for every other neighbour `w`. `O === nothing` gives the rank-2
transfer environment (ket, bra); otherwise rank-3 (ket, op, bra). This one
function produces both "below" and "above" environments (module docstring).

If `u` is the root, the ket/op/bra parent legs (unit or global-charge spaces)
are closed with a unit cap. With the sentinel `v = 0` nothing stays open and
the fully contracted scalar is returned (used by `inner`/`expect`).
"""
function _build_env_ncon_reference(ket::TTNS, O::Union{TTNO,Nothing}, bra::TTNS,
                                   u::Int, v::Int,
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
    _build_env_spec(cache, ket, O, bra, u, v) -> (spec, operands)

Lower the legacy sandwich label bookkeeping once into a complete-operand
`ContractionSpec`. Operand order deliberately remains the legacy order
`(ket, child/parent environments..., root cap?, operator?, bra)`. Only the
compiled tree changes; label semantics, TensorKit arrows, and the final open
leg order remain byte-for-byte compatible with `_build_env_ncon_reference`.
"""
function _build_env_spec(c::EnvCache, ket::TTNS, O::Union{TTNO,Nothing},
                         bra::TTNS, u::Int, v::Int)
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
    operands = Any[A]
    labels = Vector{Int}[aidx]
    conjs = Bool[false]
    envslots = Int[]
    caps = Int[]
    nxt = Ref(0)
    fresh() = (nxt[] += 1; nxt[])

    if hp
        pk = physleg(ket, u)
        if W === nothing
            lbl = fresh()
            aidx[pk] = lbl
            bidx[pk] = lbl
        else
            K = nchildren(t, u)
            pin = fresh()
            pout = fresh()
            aidx[pk] = pin
            widx[K + 2] = pin
            widx[K + 1] = pout
            bidx[pk] = pout
        end
    end

    for w in neighbors(t, u)
        la = _stateleg(t, hp, u, w)
        if w == v
            aidx[la] = -1
            W === nothing || (widx[_opleg(t, hp, u, w)] = -2)
            bidx[la] = W === nothing ? -2 : -3
        else
            E = c.envs[(w, u)]
            eidx = zeros(Int, numind(E))
            eidx[1] = fresh(); aidx[la] = eidx[1]
            if W !== nothing
                eidx[2] = fresh(); widx[_opleg(t, hp, u, w)] = eidx[2]
            end
            eidx[end] = fresh(); bidx[la] = eidx[end]
            push!(operands, E); push!(labels, eidx); push!(conjs, false)
            push!(envslots, length(labels))
        end
    end

    if t.parent[u] == 0
        ka, ko, kb = fresh(), fresh(), fresh()
        aidx[end] = ka
        bidx[end] = kb
        Vroot = domain(A)[1]
        if W === nothing
            capspace = dual(domain(B)[1]) ⊗ Vroot
            cap = _root_cap!(c, scalartype(A), capspace)
            push!(operands, cap); push!(labels, [kb, ka]); push!(conjs, false)
        else
            widx[end] = ko
            capspace = dual(domain(B)[1]) ⊗ domain(W)[numin(W)] ⊗ Vroot
            cap = _root_cap!(c, scalartype(A), capspace)
            push!(operands, cap); push!(labels, [kb, ko, ka]); push!(conjs, false)
        end
        push!(caps, length(labels))
    end

    wslot = 0
    if W !== nothing
        push!(operands, W); push!(labels, widx); push!(conjs, false)
        wslot = length(labels)
    end
    push!(operands, B); push!(labels, bidx); push!(conjs, true)
    braslot = length(labels)
    preferred = Int[1]
    append!(preferred, envslots)
    wslot != 0 && push!(preferred, wslot)
    append!(preferred, caps)
    push!(preferred, braslot)
    nopen = v == 0 ? 0 : (W === nothing ? 2 : 3)
    spec = ContractionSpec(labels, conjs, nopen, (nopen, 0), nothing;
                           preferred_slots=preferred)
    return spec, Tuple(operands)
end

"""
    build_env(cache, ket, O, bra, u, v) -> AbstractTensorMap or Number

Cached planned execution of a one-node sandwich. For `v == 0`, the final
rank-zero TensorMap is scalarized to exactly match legacy `ncon` behavior.
"""
function build_env(c::EnvCache, ket::TTNS, O::Union{TTNO,Nothing}, bra::TTNS,
                   u::Int, v::Int)
    spec, operands = _build_env_spec(c, ket, O, bra, u, v)
    kind = O === nothing ? :env_ket_bra : :env_ket_op_bra
    return _planned_execute!(c, kind, spec, operands, scalartype(ket.tensors[u]))
end

"""Compatibility overload that plans the supplied value environments once."""
function build_env(ket::TTNS, O::Union{TTNO,Nothing}, bra::TTNS, u::Int, v::Int,
                   envs::Dict{Tuple{Int,Int},AbstractTensorMap})
    return build_env(EnvCache(ket.topo, envs), ket, O, bra, u, v)
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
        build_env(c, ket, O, bra, u, v)
    end
end
env!(c::EnvCache, ket::TTNS, O::Union{TTNO,Nothing}, u::Int, v::Int) =
    env!(c, ket, O, ket, u, v)
