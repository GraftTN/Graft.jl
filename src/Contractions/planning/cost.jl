# Dense, dimensions-only cost model (Phase 2), plus Phase-3 structural
# sector/block profiles.  Both paths operate on tensor *spaces*, never on
# TensorMap payloads.

@inline function _prod_dims(dims)
    p = 1.0
    for d in dims
        p *= Float64(d)
    end
    return p
end

# TensorMap prototypes expose flat leg spaces through `space(t, i)`, whereas a
# no-data TensorMapSpace (used for Θ and link planning) exposes the same view
# through `getindex`. Keeping this distinction here means planning never needs
# to allocate a tensor merely to inspect its shape.
_prototype_leg(proto::AbstractTensorMap, i::Int) = space(proto, i)
_prototype_leg(proto, i::Int) = proto[i]

# `two_site_space` and h0's link prototype are TensorMapSpaces rather than
# data-valued maps.  Keeping their space unchanged means planning does not
# allocate a dummy TensorMap merely to discover its block structure.
_prototype_space(proto::AbstractTensorMap) = space(proto)
_prototype_space(proto) = proto

function _prototype_dims(proto)
    return Int[dim(_prototype_leg(proto, i)) for i in 1:numind(proto)]
end

function _label_dimensions(spec::ContractionSpec, protos)
    length(protos) == length(spec.labels) ||
        throw(ArgumentError("planning prototypes do not match ContractionSpec slots"))
    dims = Vector{Vector{Int}}(undef, length(protos))
    label_dims = Dict{Int,Int}()
    for i in eachindex(protos)
        dims[i] = _prototype_dims(protos[i])
        length(dims[i]) == length(spec.labels[i]) ||
            throw(ArgumentError("prototype $i has $(length(dims[i])) legs; spec has $(length(spec.labels[i]))"))
        for (label, d) in zip(spec.labels[i], dims[i])
            previous = get(label_dims, label, d)
            previous == d || throw(ArgumentError("label $label has incompatible dimensions $previous and $d"))
            label_dims[label] = d
        end
    end
    return dims, label_dims
end

"""
    dense_cost(labelsA, dimsA, labelsB, dimsB) -> NamedTuple

FLOPs and the output/intermediate footprint for one binary contraction. All
shared labels contract together, matching TensorOperations' tree semantics.
"""
function dense_cost(labelsA::Vector{Int}, dimsA::Vector{Int},
                    labelsB::Vector{Int}, dimsB::Vector{Int})
    shared = Set(labelsA) ∩ Set(labelsB)
    oa = Int[i for (i, l) in enumerate(labelsA) if !(l in shared)]
    ca = Int[i for (i, l) in enumerate(labelsA) if l in shared]
    ob = Int[i for (i, l) in enumerate(labelsB) if !(l in shared)]
    # `pA` and `pB` must list contracted legs in the *same label order*.
    # TensorOperations does not infer that correspondence from dimensions;
    # B's native leg order can differ from A's even in a valid ncon network.
    cb = Int[]
    for ia in ca
        ib = findfirst(==(labelsA[ia]), labelsB)
        ib === nothing && throw(ArgumentError("missing contracted label $(labelsA[ia])"))
        dimsA[ia] == dimsB[ib] ||
            throw(ArgumentError("incompatible contracted dimensions for label $(labelsA[ia])"))
        push!(cb, ib)
    end
    out_labels = vcat(labelsA[oa], labelsB[ob])
    out_dims = vcat(dimsA[oa], dimsB[ob])
    # Delegate the actual dense FLOP convention to L0; Planning retains only
    # graph/index bookkeeping and the independently visible output footprint.
    flops = Backend.pair_cost(dimsA, oa, ca, dimsB, ob; memory_weight=0)
    return (oindA=oa, cindA=ca, oindB=ob, cindB=cb,
            labels=out_labels, dims=out_dims,
            flops=flops,
            peak_elements=_prod_dims(out_dims))
end

"""
    _sector_pair_profile(metrics, spaceA, conjA, spaceB, conjB, out)

Lower the index partitions already derived by [`dense_cost`](@ref) into
TensorOperations expert-mode tuples and ask L0 for the exact HomSpace/block
profile of this binary contraction.  Planning stays TensorKit-type-agnostic:
only `Backend.pair_cost` knows about fusion sectors.
"""
function _sector_pair_profile(metrics, spaceA, conjA::Bool, spaceB, conjB::Bool,
                              out::Tuple)
    pA = (Tuple(metrics.oindA), Tuple(metrics.cindA))
    # TensorOperations treats B as `(contracted, open)` for a matrix product.
    pB = (Tuple(metrics.cindB), Tuple(metrics.oindB))
    return Backend.pair_cost(spaceA, pA, conjA, spaceB, pB, conjB, out)
end

function _score(plan::ContractionPlan, memory_weight::Real)
    return plan.flops + Float64(memory_weight) * plan.peak_elements
end

function _sector_score(plan::ContractionPlan, memory_weight::Real)
    return plan.sector_flops + Float64(memory_weight) * plan.sector_peak_elements
end
