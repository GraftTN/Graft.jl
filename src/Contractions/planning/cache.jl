function _spec_shape(spec::ContractionSpec)
    return (Tuple(Tuple(labels) for labels in spec.labels),
            Tuple(spec.conjs), spec.nopen, spec.out_partition,
            spec.dynamic_slot, Tuple(spec.preferred_slots))
end

"""
Build a collision-safe cache key for one effective-map network.

The optimization objective and cost-model switch are part of the identity too.
A future calibrated memory coefficient, or a caller choosing dense versus
Phase-3 sector-aware selection, must never silently reuse a plan chosen under
a different tradeoff even when the TTN/TTNO spaces themselves are unchanged.
The hard memory cap is likewise part of the identity: a plan admitted under a
larger cap must never be reused for a stricter caller.
"""
function plan_key(kind::Symbol, spec::ContractionSpec, protos, T::DataType;
                  optimize::Bool=true, memory_weight::Real=1,
                  sector_aware::Bool=true,
                  memory_cap_bytes::Union{Nothing,Real}=nothing)
    spaces = Tuple((codomain(p), domain(p)) for p in protos)
    shape = (_spec_shape(spec), spaces)
    cap = _canonical_memory_cap(memory_cap_bytes)
    objective = (optimize, Float64(memory_weight), sector_aware, cap)
    # Keep the inexpensive L0 signature visible for diagnostics while the
    # equality-bearing `shape` field below protects correctness on a collision.
    sigs = Tuple(Backend.space_signature(p) for p in protos)
    return PlanKey(kind, hash((sigs, shape, objective)), shape, T,
                   objective[1], objective[2], objective[3], objective[4])
end

"""
    get_or_plan!(plans, kind, spec, protos, T; kwargs...) -> (plan, hit)

Pure cache helper so `Planning` need not depend upward on `EnvCache`.
"""
function get_or_plan!(plans::Dict{PlanKey,ContractionPlan}, kind::Symbol,
                      spec::ContractionSpec, protos, T::DataType; kwargs...)
    optimize = get(kwargs, :optimize, true)
    memory_weight = get(kwargs, :memory_weight, 1)
    sector_aware = get(kwargs, :sector_aware, true)
    memory_cap_bytes = get(kwargs, :memory_cap_bytes, nothing)
    key = plan_key(kind, spec, protos, T;
                   optimize=optimize, memory_weight=memory_weight,
                   sector_aware=sector_aware,
                   memory_cap_bytes=memory_cap_bytes)
    if haskey(plans, key)
        return plans[key], true
    end
    forwarded = (; kwargs...)
    if haskey(forwarded, :scalar_type)
        forwarded.scalar_type == T ||
            throw(ArgumentError("get_or_plan!: scalar_type=$(forwarded.scalar_type) " *
                                "does not match cache scalar type $T"))
        forwarded = Base.structdiff(forwarded, (; scalar_type=nothing))
    end
    plan = plan_contraction(spec, protos; scalar_type=T, forwarded...)
    plans[key] = plan
    return plan, false
end
