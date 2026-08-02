# EnvCache — partial-environment cache keyed by directed edge
# (PyTreeNet: contractions/tree_cach_dict.PartialTreeCachDict, keys
# (node_id, next_node_id) with the same "everything behind node_id, looking
# towards next_node_id" semantics).

const _ENV_BUILD_MAX_RETRIES = 8

struct _EnvironmentDependencyStamp
    clear_generation::UInt64
    nodes::Tuple{Vararg{Int}}
    node_generations::Tuple{Vararg{UInt64}}
    ket_tensors::Tuple
    operator_tensors::Tuple
    bra_tensors::Tuple
end

"""
    EnvironmentBuildConflict

Typed failure raised when repeated concurrent mutation prevents a value
environment from reaching a valid commit point within the bounded retry
policy.
"""
struct EnvironmentBuildConflict <: Exception
    key::Tuple{Int,Int}
    attempts::Int
end

struct _StaleEnvironmentDependency <: Exception
    key::Tuple{Int,Int}
end

function Base.showerror(io::IO, err::EnvironmentBuildConflict)
    print(
        io,
        "environment ",
        err.key,
        " remained unstable across ",
        err.attempts,
        " dependency-generation retries",
    )
end

"""
    EnvCache(topo::TreeTopology; max_env_bytes=nothing, eviction=:lru,
             threaded_envs=false, env_staging_minbatch=2,
             env_staging_memory_cap_bytes=nothing)

Cache of sandwich environments keyed by directed edge `(u, v)`, plus compiled
shape-only effective-Hamiltonian plans. Both classes are deliberately
rebuildable: checkpoints drop the whole cache, while gauge invalidation drops
only value-dependent environments and retains plans whose space signatures
still match (§3). The default `max_env_bytes=nothing` preserves the historical
full-cache behavior. A finite cap enables deterministic LRU eviction of only
value environments; shape-only plans and root caps are never eviction victims.
`threaded_envs=true` opts into independent sibling-final staging and requires
an explicit `env_staging_memory_cap_bytes`. Deeper dependencies are prepared
serially, predicted plan live bytes are admitted before tasks start, each task
owns its `PlanWorkspace`, and staged values commit in fixed neighbor order.
"""
mutable struct EnvCache
    topo::TreeTopology
    envs::Dict{Tuple{Int,Int},AbstractTensorMap}
    env_entry_bytes::Dict{Tuple{Int,Int},Int}
    env_payload_size_counts::Dict{Int,Int}
    env_payload_bytes::Int
    env_largest_entry_bytes::Int
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
    threaded_envs::Bool
    env_staging_minbatch::Int
    env_staging_memory_cap_bytes::Union{Nothing,Int}
    cache_generation::Base.RefValue{UInt64}
    cache_lock::ReentrantLock
    node_generations::Vector{UInt64}
    env_stamps::Dict{
        Tuple{Int,Int},
        Union{Nothing,_EnvironmentDependencyStamp},
    }
    lock_contentions::Int
    shape_plan_hits::Int
    shape_plan_misses::Int
    plan_duplicate_builds::Int
    rootcap_hits::Int
    rootcap_misses::Int
    rootcap_duplicate_builds::Int
    stale_build_discards::Int
    env_duplicate_builds::Int
    env_stale_build_discards::Int
    env_waits::Int
    env_retry_exhaustions::Int
    env_staged_batches::Int
    env_staged_tasks::Int
    env_staged_admitted_bytes::Int
    env_serial_fallbacks::Int
    env_last_fallback::Symbol
end

"""
    EnvironmentCacheDiagnostics

Typed value-environment cache counters. `clear_generation` changes only when a
cache-wide clear invalidates every staged dependency; node-local invalidations
advance the corresponding mutation generation without penalizing unrelated
subtrees.
"""
struct EnvironmentCacheDiagnostics
    entries::Int
    hits::Int
    misses::Int
    rebuilds::Int
    duplicate_builds::Int
    stale_build_discards::Int
    evictions::Int
    waits::Int
    retry_exhaustions::Int
    staged_batches::Int
    staged_tasks::Int
    staged_admitted_bytes::Int
    serial_fallbacks::Int
    last_fallback::Symbol
    clear_generation::UInt64
    maximum_node_generation::UInt64
end

"""
    CacheDiagnostics

Typed, point-in-time cache diagnostics. Effective-map plan counters retain
their historical meaning, while shape-plan counters include environment and
local-observable planning. Duplicate counters report work computed outside
the cache lock and discarded because another task committed the same key.
"""
struct CacheDiagnostics
    plan_entries::Int
    rootcap_entries::Int
    effective_plan_hits::Int
    effective_plan_misses::Int
    shape_plan_hits::Int
    shape_plan_misses::Int
    plan_duplicate_builds::Int
    rootcap_hits::Int
    rootcap_misses::Int
    rootcap_duplicate_builds::Int
    lock_contentions::Int
    stale_build_discards::Int
    environments::EnvironmentCacheDiagnostics
end

# Preserve the initial public positional layout from Stage C1.
CacheDiagnostics(
        plan_entries::Int,
        rootcap_entries::Int,
        effective_plan_hits::Int,
        effective_plan_misses::Int,
        shape_plan_hits::Int,
        shape_plan_misses::Int,
        plan_duplicate_builds::Int,
        rootcap_hits::Int,
        rootcap_misses::Int,
        rootcap_duplicate_builds::Int,
        lock_contentions::Int,
        stale_build_discards::Int) =
    CacheDiagnostics(
        plan_entries,
        rootcap_entries,
        effective_plan_hits,
        effective_plan_misses,
        shape_plan_hits,
        shape_plan_misses,
        plan_duplicate_builds,
        rootcap_hits,
        rootcap_misses,
        rootcap_duplicate_builds,
        lock_contentions,
        stale_build_discards,
        EnvironmentCacheDiagnostics(
            0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, :none, UInt64(0), UInt64(0)),
    )

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

function _register_env_payload_locked!(
        c::EnvCache,
        key::Tuple{Int,Int},
        bytes::Int)
    bytes >= 0 || throw(ArgumentError("environment payload bytes must be nonnegative"))
    haskey(c.env_entry_bytes, key) &&
        throw(ArgumentError("environment payload key $key is already registered"))
    c.env_entry_bytes[key] = bytes
    c.env_payload_size_counts[bytes] = get(c.env_payload_size_counts, bytes, 0) + 1
    c.env_payload_bytes += bytes
    c.env_largest_entry_bytes = max(c.env_largest_entry_bytes, bytes)
    c.env_high_water_bytes = max(c.env_high_water_bytes, c.env_payload_bytes)
    return nothing
end

function _unregister_env_payload_locked!(c::EnvCache, key::Tuple{Int,Int})
    bytes = pop!(c.env_entry_bytes, key)
    count = c.env_payload_size_counts[bytes]
    if count == 1
        delete!(c.env_payload_size_counts, bytes)
        if bytes == c.env_largest_entry_bytes
            c.env_largest_entry_bytes =
                isempty(c.env_payload_size_counts) ? 0 :
                maximum(keys(c.env_payload_size_counts))
        end
    else
        c.env_payload_size_counts[bytes] = count - 1
    end
    c.env_payload_bytes -= bytes
    c.env_payload_bytes >= 0 ||
        throw(ArgumentError("environment payload accounting underflow"))
    return bytes
end

function _touch_env!(c::EnvCache, key::Tuple{Int,Int})
    c.env_clock += 1
    c.env_touches[key] = c.env_clock
    return nothing
end

function _seed_env_metadata!(c::EnvCache)
    empty!(c.env_touches)
    empty!(c.env_entry_bytes)
    empty!(c.env_payload_size_counts)
    c.env_payload_bytes = 0
    c.env_largest_entry_bytes = 0
    c.env_high_water_bytes = 0
    c.env_clock = 0
    for key in sort!(collect(keys(c.envs)))
        _touch_env!(c, key)
        _register_env_payload_locked!(c, key, _env_payload_bytes(c.envs[key]))
    end
    return c
end

function EnvCache(
        topo::TreeTopology;
        max_env_bytes=nothing,
        eviction::Symbol=:lru,
        threaded_envs::Bool=false,
        env_staging_minbatch::Integer=2,
        env_staging_memory_cap_bytes=nothing)
    return EnvCache(topo, Dict{Tuple{Int,Int},AbstractTensorMap}(),
                    Dict{PlanKey,ContractionPlan}(), Dict{Tuple,AbstractTensorMap}(),
                    0, 0; max_env_bytes, eviction, threaded_envs,
                    env_staging_minbatch, env_staging_memory_cap_bytes)
end

function EnvCache(
        topo::TreeTopology,
        envs::Dict{Tuple{Int,Int},AbstractTensorMap};
        max_env_bytes=nothing,
        eviction::Symbol=:lru,
        threaded_envs::Bool=false,
        env_staging_minbatch::Integer=2,
        env_staging_memory_cap_bytes=nothing)
    return EnvCache(topo, envs, Dict{PlanKey,ContractionPlan}(),
                    Dict{Tuple,AbstractTensorMap}(), 0, 0;
                    max_env_bytes, eviction, threaded_envs,
                    env_staging_minbatch, env_staging_memory_cap_bytes)
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
                  max_env_bytes=nothing, eviction::Symbol=:lru,
                  threaded_envs::Bool=false,
                  env_staging_minbatch::Integer=2,
                  env_staging_memory_cap_bytes=nothing,
                  cache_generation::Base.RefValue{UInt64}=Ref(UInt64(0)),
                  cache_lock::ReentrantLock=ReentrantLock())
    cap = _env_cache_cap(max_env_bytes)
    policy = _env_cache_policy(eviction)
    env_staging_minbatch >= 1 ||
        throw(ArgumentError("env_staging_minbatch must be positive"))
    staging_cap = _env_cache_cap(env_staging_memory_cap_bytes)
    threaded_envs && staging_cap === nothing &&
        throw(ArgumentError(
            "threaded_envs=true requires env_staging_memory_cap_bytes"))
    copied_envs = copy(envs)
    c = EnvCache(topo, copied_envs,
                 Dict{Tuple{Int,Int},Int}(), Dict{Int,Int}(), 0, 0,
                 plans, rootcaps, Int(plan_hits), Int(plan_misses),
                 cap, policy, Dict{Tuple{Int,Int},Int}(), 0,
                 0, 0, 0, 0, 0, 0,
                 threaded_envs, Int(env_staging_minbatch), staging_cap,
                 cache_generation, cache_lock,
                 fill(UInt64(0), nnodes(topo)),
                 Dict{
                     Tuple{Int,Int},
                     Union{Nothing,_EnvironmentDependencyStamp},
                 }(
                     key => nothing for key in keys(copied_envs)
                 ),
                 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0,
                 0, 0, 0, 0, :none)
    _seed_env_metadata!(c)
    cap === nothing || _enforce_env_cap!(c)
    return c
end

function _with_cache_lock(f::Function, c::EnvCache)
    acquired = trylock(c.cache_lock)
    if !acquired
        lock(c.cache_lock)
        c.lock_contentions += 1
    end
    try
        return f()
    finally
        unlock(c.cache_lock)
    end
end

"""
Create an empty value cache that shares only the source cache's shape plans.

This is the safe reuse boundary for contractions over unrelated tensor values:
environments and root caps remain invocation-local, while exact-shape plans can
be amortized across repeated overlaps.
"""
function _fresh_env_cache_with_plans(source::EnvCache, topo::TreeTopology)
    source.topo == topo ||
        throw(ArgumentError("plan cache topology does not match contraction topology"))
    return EnvCache(
        topo,
        Dict{Tuple{Int,Int},AbstractTensorMap}(),
        source.plans,
        Dict{Tuple,AbstractTensorMap}(),
        0,
        0,
        cache_generation=source.cache_generation,
        cache_lock=source.cache_lock,
        threaded_envs=source.threaded_envs,
        env_staging_minbatch=source.env_staging_minbatch,
        env_staging_memory_cap_bytes=source.env_staging_memory_cap_bytes,
    )
end

function _set_environment_locked!(
        c::EnvCache,
        key::Tuple{Int,Int},
        E::AbstractTensorMap,
        bytes::Int;
        stamp::Union{Nothing,_EnvironmentDependencyStamp}=nothing)
    if haskey(c.envs, key)
        _unregister_env_payload_locked!(c, key)
    elseif haskey(c.env_entry_bytes, key)
        throw(ArgumentError("orphan environment payload metadata for key $key"))
    end
    c.envs[key] = E
    _register_env_payload_locked!(c, key, bytes)
    c.env_stamps[key] = stamp
    _touch_env!(c, key)
    return E
end

function _delete_environment_locked!(c::EnvCache, key::Tuple{Int,Int})
    if haskey(c.envs, key)
        _unregister_env_payload_locked!(c, key)
        delete!(c.envs, key)
    elseif haskey(c.env_entry_bytes, key)
        throw(ArgumentError("orphan environment payload metadata for key $key"))
    end
    delete!(c.env_touches, key)
    delete!(c.env_stamps, key)
    return c
end

function _clear_value_environments_locked!(c::EnvCache)
    empty!(c.envs)
    empty!(c.env_entry_bytes)
    empty!(c.env_payload_size_counts)
    c.env_payload_bytes = 0
    c.env_largest_entry_bytes = 0
    empty!(c.env_touches)
    empty!(c.env_stamps)
    c.env_clock = 0
    return c
end

function _clear_value_environments!(c::EnvCache)
    return _with_cache_lock(c) do
        _clear_value_environments_locked!(c)
        c.cache_generation[] += UInt64(1)
        c
    end
end

Base.haskey(c::EnvCache, key::Tuple{Int,Int}) =
    _with_cache_lock(c) do
        haskey(c.envs, key)
    end
function Base.getindex(c::EnvCache, key::Tuple{Int,Int})
    return _with_cache_lock(c) do
        E = c.envs[key]
        _touch_env!(c, key)
        E
    end
end
function Base.empty!(c::EnvCache)
    return _with_cache_lock(c) do
        _clear_value_environments_locked!(c)
        empty!(c.plans)
        empty!(c.rootcaps)
        c.cache_generation[] += UInt64(1)
        fill!(c.node_generations, UInt64(0))
        c.plan_hits = 0
        c.plan_misses = 0
        c.env_hits = 0
        c.env_misses = 0
        c.env_rebuilds = 0
        c.env_evictions = 0
        c.env_high_water_bytes = 0
        c.lock_contentions = 0
        c.shape_plan_hits = 0
        c.shape_plan_misses = 0
        c.plan_duplicate_builds = 0
        c.rootcap_hits = 0
        c.rootcap_misses = 0
        c.rootcap_duplicate_builds = 0
        c.stale_build_discards = 0
        c.env_duplicate_builds = 0
        c.env_stale_build_discards = 0
        c.env_waits = 0
        c.env_retry_exhaustions = 0
        c.env_staged_batches = 0
        c.env_staged_tasks = 0
        c.env_staged_admitted_bytes = 0
        c.env_serial_fallbacks = 0
        c.env_last_fallback = :none
        c
    end
end

"""
Observable EffectiveMap plan-cache state for tests and solver diagnostics.

Environment/value-level plans share `plans` but intentionally do not change
these historical Krylov-map counters; their reuse remains visible through the
shape-only dictionary until the cache-wide accounting policy is added.
"""
plan_cache_stats(c::EnvCache) =
    _with_cache_lock(c) do
        (hits=c.plan_hits, misses=c.plan_misses, size=length(c.plans))
    end

"""
    cache_diagnostics(cache) -> CacheDiagnostics

Return an atomic typed snapshot of plan/root-cap cache activity.
"""
cache_diagnostics(c::EnvCache) =
    _with_cache_lock(c) do
        CacheDiagnostics(
            length(c.plans),
            length(c.rootcaps),
            c.plan_hits,
            c.plan_misses,
            c.shape_plan_hits,
            c.shape_plan_misses,
            c.plan_duplicate_builds,
            c.rootcap_hits,
            c.rootcap_misses,
            c.rootcap_duplicate_builds,
            c.lock_contentions,
            c.stale_build_discards,
            EnvironmentCacheDiagnostics(
                length(c.envs),
                c.env_hits,
                c.env_misses,
                c.env_rebuilds,
                c.env_duplicate_builds,
                c.env_stale_build_discards,
                c.env_evictions,
                c.env_waits,
                c.env_retry_exhaustions,
                c.env_staged_batches,
                c.env_staged_tasks,
                c.env_staged_admitted_bytes,
                c.env_serial_fallbacks,
                c.env_last_fallback,
                c.cache_generation[],
                maximum(c.node_generations; init=UInt64(0)),
            ),
        )
    end

"""
    env_cache_stats(cache)

Observable value-environment payload and policy state. `payload_bytes` counts
actual stored TensorKit block payloads in `envs`; root caps and shape-only
plans are reported separately and are intentionally outside value eviction.
"""
function env_cache_stats(c::EnvCache)
    return _with_cache_lock(c) do
        payload_bytes = c.env_payload_bytes
        largest_entry_bytes = c.env_largest_entry_bytes
        (; payload_bytes, largest_entry_bytes, entry_count=length(c.envs),
         plan_count=length(c.plans), hits=c.env_hits, misses=c.env_misses,
         rebuilds=c.env_rebuilds, high_water_bytes=c.env_high_water_bytes,
         evictions=c.env_evictions, max_env_bytes=c.max_env_bytes,
         eviction=c.max_env_bytes === nothing ? :full : c.eviction,
         duplicate_builds=c.env_duplicate_builds,
         stale_build_discards=c.env_stale_build_discards,
         waits=c.env_waits,
         retry_exhaustions=c.env_retry_exhaustions,
         threaded=c.threaded_envs,
         staging_minbatch=c.env_staging_minbatch,
         staging_memory_cap_bytes=c.env_staging_memory_cap_bytes,
         staged_batches=c.env_staged_batches,
         staged_tasks=c.env_staged_tasks,
         staged_admitted_bytes=c.env_staged_admitted_bytes,
         serial_fallbacks=c.env_serial_fallbacks,
         last_fallback=c.env_last_fallback,
         clear_generation=c.cache_generation[],
         maximum_node_generation=maximum(
             c.node_generations; init=UInt64(0)))
    end
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
    while c.env_payload_bytes > cap && !isempty(c.envs)
        victim = _lru_victim(c)
        victim === nothing && break
        _delete_environment_locked!(c, victim)
        c.env_evictions += 1
    end
    return c
end

function _with_env_transaction(f::Function, c::EnvCache)
    _with_cache_lock(c) do
        c.transaction_depth += 1
    end
    try
        return f()
    finally
        _with_cache_lock(c) do
            c.transaction_depth -= 1
            c.transaction_depth >= 0 ||
                throw(ArgumentError("EnvCache transaction depth underflow"))
            c.transaction_depth == 0 && _enforce_env_cap!(c)
        end
    end
end

function _store_env!(
        c::EnvCache,
        key::Tuple{Int,Int},
        E::AbstractTensorMap;
        stamp::Union{Nothing,_EnvironmentDependencyStamp}=nothing)
    bytes = _env_payload_bytes(E)
    return _with_cache_lock(c) do
        _set_environment_locked!(c, key, E, bytes; stamp)
        c.env_rebuilds += 1
        E
    end
end

"""Return a shape-owned, immutable-in-use root cap for a planned network."""
function _root_cap!(c::EnvCache, T::DataType, capspace)
    key = (T, capspace)
    while true
        cached = _with_cache_lock(c) do
            generation = c.cache_generation[]
            if haskey(c.rootcaps, key)
                c.rootcap_hits += 1
                return (found=true, cap=c.rootcaps[key], generation)
            end
            c.rootcap_misses += 1
            return (found=false, cap=nothing, generation)
        end
        cached.found && return cached.cap

        candidate = Backend.ones_tensor(T, capspace)
        committed = _with_cache_lock(c) do
            if c.cache_generation[] != cached.generation
                c.stale_build_discards += 1
                return (stale=true, cap=nothing)
            end
            if haskey(c.rootcaps, key)
                c.rootcap_duplicate_builds += 1
                return (stale=false, cap=c.rootcaps[key])
            end
            c.rootcaps[key] = candidate
            return (stale=false, cap=candidate)
        end
        committed.stale || return committed.cap
    end
end

function _cache_get_or_plan!(
        c::EnvCache, kind::Symbol, spec::ContractionSpec, protos, T::DataType;
        effective_lookup::Bool=false, kwargs...)
    optimize = get(kwargs, :optimize, true)
    memory_weight = get(kwargs, :memory_weight, 1)
    sector_aware = get(kwargs, :sector_aware, true)
    memory_cap_bytes = get(kwargs, :memory_cap_bytes, nothing)
    key = Planning.plan_key(
        kind, spec, protos, T;
        optimize, memory_weight, sector_aware, memory_cap_bytes,
    )
    while true
        cached = _with_cache_lock(c) do
            generation = c.cache_generation[]
            if haskey(c.plans, key)
                c.shape_plan_hits += 1
                effective_lookup && (c.plan_hits += 1)
                return (found=true, plan=c.plans[key], generation)
            end
            c.shape_plan_misses += 1
            effective_lookup && (c.plan_misses += 1)
            return (found=false, plan=nothing, generation)
        end
        cached.found && return cached.plan

        candidate = Planning._uncached_plan(spec, protos, T; kwargs...)
        committed = _with_cache_lock(c) do
            if c.cache_generation[] != cached.generation
                c.stale_build_discards += 1
                return (stale=true, plan=nothing)
            end
            if haskey(c.plans, key)
                c.plan_duplicate_builds += 1
                return (stale=false, plan=c.plans[key])
            end
            c.plans[key] = candidate
            return (stale=false, plan=candidate)
        end
        committed.stale || return committed.plan
    end
end

"""Look up a shape-only plan without changing EffectiveMap hit/miss counters."""
function _planned_execute!(c::EnvCache, kind::Symbol, spec::ContractionSpec,
                           operands::Tuple, T::DataType; optimize::Bool=true)
    plan = _cache_get_or_plan!(c, kind, spec, operands, T; optimize)
    return Planning.execute(plan, operands)
end

Base.@noinline function _effective_map!(
        c::EnvCache, kind::Symbol, spec::ContractionSpec,
        protos, statics::Tuple, T::DataType;
        optimize::Bool=true, memory_weight::Real=1,
        sector_aware::Bool=true,
        memory_cap_bytes::Union{Nothing,Real}=nothing,
        input_twists::Tuple=(),
        output_twists::Tuple=())
    plan = _cache_get_or_plan!(
        c, kind, spec, protos, T;
        effective_lookup=true, optimize, memory_weight, sector_aware,
        memory_cap_bytes,
    )
    static_layout_cap_bytes = if memory_cap_bytes === nothing
        Inf
    else
        dense_live = plan.live_peak_bytes
        sector_live = isfinite(plan.sector_live_peak_bytes) ?
                      plan.sector_live_peak_bytes : dense_live
        max(0.0, Float64(memory_cap_bytes) - max(dense_live, sector_live))
    end
    return EffectiveMap(
        plan, statics, input_twists, output_twists;
        static_layout_cap_bytes,
    )
end

# is node `n` on the `u`-side of the directed edge (u, v)?
function _on_side(t::TreeTopology, n::Int, u::Int, v::Int)
    n == u && return true
    n == v && return false
    return u in path_between(t, n, v)
end

function _environment_dependency_nodes(
        c::EnvCache,
        u::Int,
        v::Int)
    if v == 0
        return Tuple(1:nnodes(c.topo))
    end
    return Tuple(sort!(collect(subtree_nodes(c.topo, u, v))))
end

function _capture_environment_stamp(
        c::EnvCache,
        ket::TTNS,
        O::Union{TTNO,Nothing},
        bra::TTNS,
        u::Int,
        v::Int)
    nodes = _environment_dependency_nodes(c, u, v)
    return _with_cache_lock(c) do
        _EnvironmentDependencyStamp(
            c.cache_generation[],
            nodes,
            Tuple(c.node_generations[n] for n in nodes),
            Tuple(ket.tensors[n] for n in nodes),
            O === nothing ? () : Tuple(O.tensors[n] for n in nodes),
            Tuple(bra.tensors[n] for n in nodes),
        )
    end
end

function _environment_stamp_is_current_locked(
        c::EnvCache,
        stamp::_EnvironmentDependencyStamp,
        ket::TTNS,
        O::Union{TTNO,Nothing},
        bra::TTNS)
    c.cache_generation[] == stamp.clear_generation || return false
    length(stamp.nodes) == length(stamp.node_generations) || return false
    length(stamp.nodes) == length(stamp.ket_tensors) || return false
    length(stamp.nodes) == length(stamp.bra_tensors) || return false
    if O === nothing
        isempty(stamp.operator_tensors) || return false
    else
        length(stamp.nodes) == length(stamp.operator_tensors) || return false
    end
    for (i, n) in enumerate(stamp.nodes)
        c.node_generations[n] == stamp.node_generations[i] || return false
        ket.tensors[n] === stamp.ket_tensors[i] || return false
        bra.tensors[n] === stamp.bra_tensors[i] || return false
        O === nothing ||
            O.tensors[n] === stamp.operator_tensors[i] || return false
    end
    return true
end

function _environment_stamp_is_current(
        c::EnvCache,
        stamp::_EnvironmentDependencyStamp,
        ket::TTNS,
        O::Union{TTNO,Nothing},
        bra::TTNS)
    return _with_cache_lock(c) do
        _environment_stamp_is_current_locked(c, stamp, ket, O, bra)
    end
end

"""
    invalidate_node!(cache::EnvCache, n) -> cache

Drop every cached environment whose contracted side contains node `n`. Fired
by `Networks.update_tensor!` (§9.2) — the explicit invalidation event of §3.
"""
function invalidate_node!(c::EnvCache, n::Int)
    return _with_cache_lock(c) do
        checkbounds(c.node_generations, n)
        c.node_generations[n] += UInt64(1)
        for key in collect(keys(c.envs))
            _on_side(c.topo, n, key[1], key[2]) &&
                _delete_environment_locked!(c, key)
        end
        c
    end
end

"""
    invalidate_edge!(cache::EnvCache, n, m) -> cache

Invalidation event for a gauge move across the edge `(n, m)`: both node
tensors changed, so every environment whose side touches `n` or `m` dies.
"""
function invalidate_edge!(c::EnvCache, n::Int, m::Int)
    return _with_cache_lock(c) do
        checkbounds(c.node_generations, n)
        checkbounds(c.node_generations, m)
        c.node_generations[n] += UInt64(1)
        n == m || (c.node_generations[m] += UInt64(1))
        for key in collect(keys(c.envs))
            if _on_side(c.topo, n, key[1], key[2]) ||
               _on_side(c.topo, m, key[1], key[2])
                _delete_environment_locked!(c, key)
            end
        end
        c
    end
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
Return the bra tensor with the pivotal correction required by a Euclidean
state-state inner product. Contracting a conjugated dual physical leg directly
closes a categorical (supertrace) loop; its ribbon twist converts that loop to
the positive Hilbert-space metric. A TTNO sandwich already carries the needed
pivotal structure through its physical input/output pair, so it must not be
twisted again.
"""
function _euclidean_bra_tensor(bra::TTNS, O::Union{TTNO,Nothing}, u::Int)
    B = bra.tensors[u]
    if O === nothing && hasphys(bra, u)
        p = physleg(bra, u)
        isdual(space(B, p)) && return Backend.twist(B, p)
    end
    return B
end

function _component_has_dual_physical(psi::TTNS, u::Int, avoiding::Int)
    return any(subtree_nodes(psi.topo, u, avoiding)) do n
        hasphys(psi, n) && isdual(physspace(psi, n))
    end
end

"""Open legs that need a pivotal twist in a Euclidean one-site result."""
function _euclidean_output_legs(psi::TTNS, n::Int)
    inds = Int[]
    for (k, child) in enumerate(psi.topo.children[n])
        isdual(space(psi.tensors[n], k)) &&
            _component_has_dual_physical(psi, child, n) && push!(inds, k)
    end
    if hasphys(psi, n)
        p = physleg(psi, n)
        isdual(space(psi.tensors[n], p)) && push!(inds, p)
    end
    parent = psi.topo.parent[n]
    if parent != 0
        p = parentleg(psi, n)
        # A parent-side environment is stored with the categorical
        # supertrace residue on its open bra leg.  Closing it through a dual
        # flat parent leg does not cancel that residue, irrespective of
        # whether the parent component contains a dual physical carrier.
        # The latter was a purification-specific proxy and misses ordinary
        # fermionic gauges produced by left_orth/split_two_site!.
        isdual(space(psi.tensors[n], p)) && push!(inds, p)
    end
    return Tuple(inds)
end

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
    B = _euclidean_bra_tensor(bra, O, u)
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
    B = _euclidean_bra_tensor(bra, O, u)
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
            E = _cached_env!(c, (w, u), ket, O, bra)
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
function _prepare_env_execution(
        c::EnvCache,
        ket::TTNS,
        O::Union{TTNO,Nothing},
        bra::TTNS,
        u::Int,
        v::Int;
        optimize::Bool=true)
    spec, operands = _build_env_spec(c, ket, O, bra, u, v)
    kind = O === nothing ? :env_ket_bra : :env_ket_op_bra
    plan = _cache_get_or_plan!(
        c, kind, spec, operands, scalartype(ket.tensors[u]); optimize)
    return (; plan, operands)
end

function _build_env_value(c::EnvCache, ket::TTNS, O::Union{TTNO,Nothing}, bra::TTNS,
                          u::Int, v::Int; optimize::Bool=true)
    prepared = _prepare_env_execution(
        c, ket, O, bra, u, v; optimize)
    return Planning.execute(prepared.plan, prepared.operands)
end

function _environment_plan_live_bytes(plan::ContractionPlan)
    bytes = isfinite(plan.sector_live_peak_bytes) ?
        plan.sector_live_peak_bytes : plan.live_peak_bytes
    isfinite(bytes) && bytes >= 0 ||
        throw(ArgumentError(
            "environment staging requires a finite nonnegative plan live-byte estimate"))
    bytes <= typemax(Int) ||
        throw(ArgumentError(
            "environment staging live-byte estimate exceeds Int capacity"))
    return ceil(Int, bytes)
end

function _record_environment_fallback!(c::EnvCache, reason::Symbol)
    return _with_cache_lock(c) do
        c.env_serial_fallbacks += 1
        c.env_last_fallback = reason
        nothing
    end
end

function _execute_staged_environment_candidate(candidate)
    workspace = PlanWorkspace(candidate.plan)
    value = Planning.execute(
        candidate.plan,
        candidate.operands;
        workspace,
    )
    return (; value, workspace)
end

function _ensure_environment_prerequisites!(
        c::EnvCache,
        ket::TTNS,
        O::Union{TTNO,Nothing},
        bra::TTNS,
        u::Int,
        v::Int;
        optimize::Bool=true,
        allow_staging::Bool=true,
        execute_candidate=_execute_staged_environment_candidate)
    siblings = Int[w for w in neighbors(c.topo, u) if w != v]
    isempty(siblings) && return nothing
    can_stage = allow_staging && c.threaded_envs &&
        Threads.nthreads() > 1 &&
        length(siblings) >= c.env_staging_minbatch
    if !can_stage
        if allow_staging && c.threaded_envs
            reason = Threads.nthreads() > 1 ?
                :insufficient_batch : :single_thread
            _record_environment_fallback!(c, reason)
        end
        for w in siblings
            _env_impl!(
                c, ket, O, bra, w, u;
                optimize, allow_staging=false)
        end
        return nothing
    end

    # Prepare every deeper dependency in deterministic serial order. The
    # sibling-final contractions below then share only immutable input values
    # and synchronized shape plans/root caps.
    for w in siblings
        for z in neighbors(c.topo, w)
            z == u && continue
            _env_impl!(
                c, ket, O, bra, z, w;
                optimize, allow_staging=false)
        end
    end

    staged = Any[]
    for w in siblings
        key = (w, u)
        cached = _lookup_environment!(c, key, ket, O, bra)
        cached.found && continue
        stamp = _capture_environment_stamp(c, ket, O, bra, w, u)
        try
            prepared = _prepare_env_execution(
                c, ket, O, bra, w, u; optimize)
            push!(
                staged,
                (;
                    key,
                    stamp,
                    prepared.plan,
                    prepared.operands,
                    admitted_bytes=_environment_plan_live_bytes(
                        prepared.plan),
                ),
            )
        catch err
            err isa _StaleEnvironmentDependency || rethrow()
            _record_environment_fallback!(c, :stale_preflight)
            for sibling in siblings
                _env_impl!(
                    c, ket, O, bra, sibling, u;
                    optimize, allow_staging=false)
            end
            return nothing
        end
    end

    if length(staged) < c.env_staging_minbatch
        _record_environment_fallback!(c, :insufficient_missing_batch)
        for candidate in staged
            _env_impl!(
                c, ket, O, bra, candidate.key...;
                optimize, allow_staging=false)
        end
        return nothing
    end

    admitted_total = sum(
        BigInt(candidate.admitted_bytes) for candidate in staged;
        init=BigInt(0))
    cap = c.env_staging_memory_cap_bytes
    cap === nothing &&
        throw(ArgumentError(
            "threaded environment staging has no memory cap"))
    if admitted_total > cap
        _record_environment_fallback!(c, :memory_cap)
        for candidate in staged
            _env_impl!(
                c, ket, O, bra, candidate.key...;
                optimize, allow_staging=false)
        end
        return nothing
    end
    admitted_bytes = Int(admitted_total)

    results = Vector{Any}(undef, length(staged))
    try
        threaded_foreach(
                eachindex(staged);
                threaded=true,
                minbatch=c.env_staging_minbatch) do i
            candidate = staged[i]
            results[i] = execute_candidate(candidate)
        end
    catch
        _record_environment_fallback!(c, :worker_failure)
        rethrow()
    end

    _with_cache_lock(c) do
        c.env_staged_batches += 1
        c.env_staged_tasks += length(staged)
        c.env_staged_admitted_bytes += admitted_bytes
        c.env_last_fallback = :none
    end
    for i in eachindex(staged)
        candidate = staged[i]
        committed = _commit_environment_candidate!(
            c,
            candidate.key,
            results[i].value,
            candidate.stamp,
            ket,
            O,
            bra,
        )
        if committed.status === :stale
            _record_environment_fallback!(c, :stale_commit)
            _env_impl!(
                c, ket, O, bra, candidate.key...;
                optimize, allow_staging=false)
        end
    end
    return nothing
end

function _build_env_consistent(
        c::EnvCache,
        ket::TTNS,
        O::Union{TTNO,Nothing},
        bra::TTNS,
        u::Int,
        v::Int;
        optimize::Bool=true)
    key = (u, v)
    for _ in 1:_ENV_BUILD_MAX_RETRIES
        stamp = _capture_environment_stamp(c, ket, O, bra, u, v)
        try
            _ensure_environment_prerequisites!(
                c, ket, O, bra, u, v; optimize)
            candidate = _build_env_value(
                c, ket, O, bra, u, v; optimize)
            _environment_stamp_is_current(c, stamp, ket, O, bra) &&
                return candidate
            _with_cache_lock(c) do
                c.env_stale_build_discards += 1
            end
        catch err
            err isa _StaleEnvironmentDependency || rethrow()
            _with_cache_lock(c) do
                c.env_stale_build_discards += 1
            end
        end
    end
    _with_cache_lock(c) do
        c.env_retry_exhaustions += 1
    end
    throw(EnvironmentBuildConflict(key, _ENV_BUILD_MAX_RETRIES))
end

function build_env(c::EnvCache, ket::TTNS, O::Union{TTNO,Nothing}, bra::TTNS,
                   u::Int, v::Int)
    return _with_env_transaction(c) do
        _build_env_consistent(c, ket, O, bra, u, v)
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
function _cached_env!(
        c::EnvCache,
        key::Tuple{Int,Int},
        ket::TTNS,
        O::Union{TTNO,Nothing},
        bra::TTNS)
    return _with_cache_lock(c) do
        haskey(c.envs, key) || throw(_StaleEnvironmentDependency(key))
        stamp = get(c.env_stamps, key, nothing)
        if stamp !== nothing &&
           !_environment_stamp_is_current_locked(c, stamp, ket, O, bra)
            _delete_environment_locked!(c, key)
            c.env_stale_build_discards += 1
            throw(_StaleEnvironmentDependency(key))
        end
        E = c.envs[key]
        _touch_env!(c, key)
        E
    end
end

function _lookup_environment!(
        c::EnvCache,
        key::Tuple{Int,Int},
        ket::TTNS,
        O::Union{TTNO,Nothing},
        bra::TTNS)
    return _with_cache_lock(c) do
        if !haskey(c.envs, key)
            c.env_misses += 1
            return (found=false, env=nothing)
        end
        stamp = get(c.env_stamps, key, nothing)
        if stamp !== nothing &&
           !_environment_stamp_is_current_locked(c, stamp, ket, O, bra)
            _delete_environment_locked!(c, key)
            c.env_stale_build_discards += 1
            c.env_misses += 1
            return (found=false, env=nothing)
        end
        c.env_hits += 1
        E = c.envs[key]
        _touch_env!(c, key)
        return (found=true, env=E)
    end
end

function _commit_environment_candidate!(
        c::EnvCache,
        key::Tuple{Int,Int},
        candidate::AbstractTensorMap,
        stamp::_EnvironmentDependencyStamp,
        ket::TTNS,
        O::Union{TTNO,Nothing},
        bra::TTNS)
    candidate_bytes = _env_payload_bytes(candidate)
    return _with_cache_lock(c) do
        if !_environment_stamp_is_current_locked(c, stamp, ket, O, bra)
            c.env_stale_build_discards += 1
            return (status=:stale, env=nothing)
        end
        if haskey(c.envs, key)
            existing_stamp = get(c.env_stamps, key, nothing)
            if existing_stamp === nothing ||
               _environment_stamp_is_current_locked(
                   c, existing_stamp, ket, O, bra)
                c.env_duplicate_builds += 1
                E = c.envs[key]
                _touch_env!(c, key)
                return (status=:duplicate, env=E)
            end
            _delete_environment_locked!(c, key)
            c.env_stale_build_discards += 1
        end
        _set_environment_locked!(
            c, key, candidate, candidate_bytes; stamp)
        c.env_rebuilds += 1
        return (status=:committed, env=candidate)
    end
end

function _env_impl!(c::EnvCache, ket::TTNS, O::Union{TTNO,Nothing}, bra::TTNS,
                    u::Int, v::Int; optimize::Bool=true,
                    allow_staging::Bool=true)
    key = (u, v)
    for _ in 1:_ENV_BUILD_MAX_RETRIES
        cached = _lookup_environment!(c, key, ket, O, bra)
        cached.found && return cached.env
        stamp = _capture_environment_stamp(c, ket, O, bra, u, v)
        try
            _ensure_environment_prerequisites!(
                c, ket, O, bra, u, v; optimize, allow_staging)
            candidate = _build_env_value(
                c, ket, O, bra, u, v; optimize)
            committed = _commit_environment_candidate!(
                c, key, candidate, stamp, ket, O, bra)
            committed.status === :stale && continue
            return committed.env
        catch err
            err isa _StaleEnvironmentDependency || rethrow()
            _with_cache_lock(c) do
                c.env_stale_build_discards += 1
            end
        end
    end
    _with_cache_lock(c) do
        c.env_retry_exhaustions += 1
    end
    throw(EnvironmentBuildConflict(key, _ENV_BUILD_MAX_RETRIES))
end

function env!(c::EnvCache, ket::TTNS, O::Union{TTNO,Nothing}, bra::TTNS, u::Int, v::Int)
    return _with_env_transaction(c) do
        _env_impl!(c, ket, O, bra, u, v)
    end
end
env!(c::EnvCache, ket::TTNS, O::Union{TTNO,Nothing}, u::Int, v::Int) =
    env!(c, ket, O, ket, u, v)
