# EnvCache — partial-environment cache keyed by directed edge
# (PyTreeNet: contractions/tree_cach_dict.PartialTreeCachDict, keys
# (node_id, next_node_id) with the same "everything behind node_id, looking
# towards next_node_id" semantics).

"""
    EnvCache(topo::TreeTopology; max_env_bytes=nothing, eviction=:lru)

Cache of sandwich environments keyed by directed edge `(u, v)`, plus compiled
shape-only effective-Hamiltonian plans. Both classes are deliberately
rebuildable: checkpoints drop the whole cache, while gauge invalidation drops
only value-dependent environments and retains plans whose space signatures
still match (§3). The default `max_env_bytes=nothing` preserves the historical
full-cache behavior. A finite cap enables deterministic LRU eviction of only
value environments; shape-only plans and root caps are never eviction victims.
"""
mutable struct EnvCache
    topo::TreeTopology
    envs::Dict{Tuple{Int,Int},AbstractTensorMap}
    plans::Dict{PlanKey,ContractionPlan}
    rootcaps::Dict{Tuple,AbstractTensorMap}
    plan_hits::Int
    plan_misses::Int
    max_env_bytes::Union{Nothing,Int}
    eviction::Symbol
    env_touches::Dict{Tuple{Int,Int},Int}
    env_clock::Int
    env_hits::Int
    env_misses::Int
    env_rebuilds::Int
    env_evictions::Int
    env_high_water_bytes::Int
    transaction_depth::Int
end

function _env_cache_cap(max_env_bytes)
    max_env_bytes === nothing && return nothing
    max_env_bytes isa Integer ||
        throw(ArgumentError("max_env_bytes must be an integer number of bytes or nothing"))
    max_env_bytes >= 0 ||
        throw(ArgumentError("max_env_bytes must be nonnegative"))
    return Int(max_env_bytes)
end

function _env_cache_policy(eviction::Symbol)
    eviction === :lru ||
        throw(ArgumentError("EnvCache eviction must be :lru"))
    return eviction
end

"""Stored TensorKit block payload bytes for one cached environment."""
function _env_payload_bytes(E::AbstractTensorMap)
    bytes = 0
    for (_, block_) in blocks(E)
        T = eltype(block_)
        bytes += isbitstype(T) ? sizeof(T) * length(block_) : Base.summarysize(block_)
    end
    return bytes
end

function _reconcile_env_metadata!(c::EnvCache)
    filter!(p -> haskey(c.envs, p.first), c.env_touches)
    return c
end

function _env_payload_summary!(c::EnvCache)
    _reconcile_env_metadata!(c)
    total = 0
    largest = 0
    for E in values(c.envs)
        bytes = _env_payload_bytes(E)
        total += bytes
        largest = max(largest, bytes)
    end
    c.env_high_water_bytes = max(c.env_high_water_bytes, total)
    return total, largest
end

function _touch_env!(c::EnvCache, key::Tuple{Int,Int})
    c.env_clock += 1
    c.env_touches[key] = c.env_clock
    return nothing
end

function _seed_env_metadata!(c::EnvCache)
    empty!(c.env_touches)
    c.env_clock = 0
    for key in sort!(collect(keys(c.envs)))
        _touch_env!(c, key)
    end
    _env_payload_summary!(c)
    return c
end

function EnvCache(topo::TreeTopology; max_env_bytes=nothing, eviction::Symbol=:lru)
    return EnvCache(topo, Dict{Tuple{Int,Int},AbstractTensorMap}(),
                    Dict{PlanKey,ContractionPlan}(), Dict{Tuple,AbstractTensorMap}(),
                    0, 0; max_env_bytes, eviction)
end

function EnvCache(topo::TreeTopology, envs::Dict{Tuple{Int,Int},AbstractTensorMap};
                  max_env_bytes=nothing, eviction::Symbol=:lru)
    return EnvCache(topo, envs, Dict{PlanKey,ContractionPlan}(),
                    Dict{Tuple,AbstractTensorMap}(), 0, 0;
                    max_env_bytes, eviction)
end

# Source-compatible constructor for callers that built an EnvCache with the
# pre-root-cap field layout.
EnvCache(topo::TreeTopology, envs::Dict{Tuple{Int,Int},AbstractTensorMap},
         plans::Dict{PlanKey,ContractionPlan}, plan_hits::Integer,
         plan_misses::Integer; kwargs...) =
    EnvCache(topo, envs, plans, Dict{Tuple,AbstractTensorMap}(),
             Int(plan_hits), Int(plan_misses); kwargs...)

function EnvCache(topo::TreeTopology, envs::Dict{Tuple{Int,Int},AbstractTensorMap},
                  plans::Dict{PlanKey,ContractionPlan},
                  rootcaps::Dict{Tuple,AbstractTensorMap},
                  plan_hits::Integer, plan_misses::Integer;
                  max_env_bytes=nothing, eviction::Symbol=:lru)
    cap = _env_cache_cap(max_env_bytes)
    policy = _env_cache_policy(eviction)
    c = EnvCache(topo, envs, plans, rootcaps, Int(plan_hits), Int(plan_misses),
                 cap, policy, Dict{Tuple{Int,Int},Int}(), 0,
                 0, 0, 0, 0, 0, 0)
    _seed_env_metadata!(c)
    cap === nothing || _enforce_env_cap!(c)
    return c
end

Base.haskey(c::EnvCache, key::Tuple{Int,Int}) = haskey(c.envs, key)
function Base.getindex(c::EnvCache, key::Tuple{Int,Int})
    E = c.envs[key]
    _touch_env!(c, key)
    return E
end
Base.empty!(c::EnvCache) = (empty!(c.envs); empty!(c.plans); empty!(c.rootcaps);
                            empty!(c.env_touches); c.env_clock = 0;
                            c.plan_hits = 0; c.plan_misses = 0;
                            c.env_hits = 0; c.env_misses = 0;
                            c.env_rebuilds = 0; c.env_evictions = 0;
                            c.env_high_water_bytes = 0; c.transaction_depth = 0; c)

"""
Observable EffectiveMap plan-cache state for tests and solver diagnostics.

Environment/value-level plans share `plans` but intentionally do not change
these historical Krylov-map counters; their reuse remains visible through the
shape-only dictionary until the cache-wide accounting policy is added.
"""
plan_cache_stats(c::EnvCache) =
    (hits=c.plan_hits, misses=c.plan_misses, size=length(c.plans))

"""
    env_cache_stats(cache)

Observable value-environment payload and policy state. `payload_bytes` counts
actual stored TensorKit block payloads in `envs`; root caps and shape-only
plans are reported separately and are intentionally outside value eviction.
"""
function env_cache_stats(c::EnvCache)
    payload_bytes, largest_entry_bytes = _env_payload_summary!(c)
    return (; payload_bytes, largest_entry_bytes, entry_count=length(c.envs),
            plan_count=length(c.plans), hits=c.env_hits, misses=c.env_misses,
            rebuilds=c.env_rebuilds, high_water_bytes=c.env_high_water_bytes,
            evictions=c.env_evictions, max_env_bytes=c.max_env_bytes,
            eviction=c.max_env_bytes === nothing ? :full : c.eviction)
end

function _lru_victim(c::EnvCache)
    keys_ = sort!(collect(keys(c.envs)))
    isempty(keys_) && return nothing
    victim = first(keys_)
    for key in Iterators.drop(keys_, 1)
        candidate_rank = (get(c.env_touches, key, 0), key)
        victim_rank = (get(c.env_touches, victim, 0), victim)
        candidate_rank < victim_rank && (victim = key)
    end
    return victim
end

function _enforce_env_cap!(c::EnvCache)
    cap = c.max_env_bytes
    cap === nothing && return c
    c.eviction === :lru || throw(ArgumentError("unsupported EnvCache eviction policy"))
    payload_bytes, _ = _env_payload_summary!(c)
    while payload_bytes > cap && !isempty(c.envs)
        victim = _lru_victim(c)
        victim === nothing && break
        delete!(c.envs, victim)
        delete!(c.env_touches, victim)
        c.env_evictions += 1
        payload_bytes, _ = _env_payload_summary!(c)
    end
    return c
end

function _with_env_transaction(f::Function, c::EnvCache)
    c.transaction_depth += 1
    try
        return f()
    finally
        c.transaction_depth -= 1
        c.transaction_depth >= 0 ||
            throw(ArgumentError("EnvCache transaction depth underflow"))
        c.transaction_depth == 0 && _enforce_env_cap!(c)
    end
end

function _store_env!(c::EnvCache, key::Tuple{Int,Int}, E::AbstractTensorMap)
    c.envs[key] = E
    _touch_env!(c, key)
    c.env_rebuilds += 1
    _env_payload_summary!(c)
    return E
end

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
    _reconcile_env_metadata!(c)
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
    _reconcile_env_metadata!(c)
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
            E = _cached_env!(c, (w, u))
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

Ensure every prerequisite environment and then run planned execution of a
one-node sandwich. The whole operation is one eviction transaction, so callers
may safely use this exported low-level entry point with a bounded cache. For
`v == 0`, the final rank-zero TensorMap is scalarized to exactly match legacy
`ncon` behavior.
"""
function _build_env_value(c::EnvCache, ket::TTNS, O::Union{TTNO,Nothing}, bra::TTNS,
                          u::Int, v::Int)
    spec, operands = _build_env_spec(c, ket, O, bra, u, v)
    kind = O === nothing ? :env_ket_bra : :env_ket_op_bra
    return _planned_execute!(c, kind, spec, operands, scalartype(ket.tensors[u]))
end

function build_env(c::EnvCache, ket::TTNS, O::Union{TTNO,Nothing}, bra::TTNS,
                   u::Int, v::Int)
    return _with_env_transaction(c) do
        for w in neighbors(c.topo, u)
            w == v || _env_impl!(c, ket, O, bra, w, u)
        end
        _build_env_value(c, ket, O, bra, u, v)
    end
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
function _cached_env!(c::EnvCache, key::Tuple{Int,Int})
    E = c.envs[key]
    _touch_env!(c, key)
    return E
end

function _env_impl!(c::EnvCache, ket::TTNS, O::Union{TTNO,Nothing}, bra::TTNS,
                    u::Int, v::Int)
    key = (u, v)
    if haskey(c.envs, key)
        c.env_hits += 1
        return _cached_env!(c, key)
    end
    c.env_misses += 1
    for w in neighbors(c.topo, u)
        w == v && continue
        _env_impl!(c, ket, O, bra, w, u)
    end
    return _store_env!(c, key, _build_env_value(c, ket, O, bra, u, v))
end

function env!(c::EnvCache, ket::TTNS, O::Union{TTNO,Nothing}, bra::TTNS, u::Int, v::Int)
    return _with_env_transaction(c) do
        _env_impl!(c, ket, O, bra, u, v)
    end
end
env!(c::EnvCache, ket::TTNS, O::Union{TTNO,Nothing}, u::Int, v::Int) =
    env!(c, ket, O, ket, u, v)
