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

"""Best scalar type available from data-valued prototypes, else Float64."""
function _planning_scalar_type(protos)
    for proto in protos
        proto isa AbstractTensorMap && return scalartype(proto)
    end
    # Shape-only TensorMapSpace fixtures intentionally carry no scalar type.
    # Float64 preserves the former element-only planner's neutral default;
    # cache-backed production planning passes the actual scalar type explicitly.
    return Float64
end

function _scalar_byte_width(T::DataType)
    isconcretetype(T) || throw(ArgumentError("planning scalar type must be concrete, got $T"))
    return sizeof(T)
end

"""
    _stored_payload_elements(space_, dense_fallback) -> Float64

TensorKit's `dim(::TensorMapSpace)` is the number of stored scalar payload
entries across blocks.  It is therefore the right input model for symmetric
maps; ordinary spaces simply report their dense product.  A few deliberately
minimal dimensions-only fixtures have no structural `dim` implementation, in
which case the dense payload remains the conservative fallback.
"""
function _stored_payload_elements(space_, dense_fallback::Real)
    try
        return Float64(dim(space_))
    catch err
        err isa InterruptException && rethrow()
        return Float64(dense_fallback)
    end
end

@inline function _is_identity_layout(order, n::Int)
    length(order) == n || return false
    for (i, j) in enumerate(order)
        i == j || return false
    end
    return true
end

"""
    _known_transform_payloads(metrics, out, ...)

Return conservative dense and sector-stored payloads for transformation
buffers whose shape is known from TensorOperations' expert-mode partitions.
Each non-identity input/output permutation may require a full extra payload;
conjugated operands are conservatively charged as a transform too.  This does
not invent a BLAS workspace estimate -- unknown backend scratch remains
outside the model -- but makes every known full-payload temporary explicit.
"""
function _known_transform_payloads(metrics, out::Tuple,
                                   dense_a::Real, dense_b::Real, dense_out::Real,
                                   sector_a::Real, sector_b::Real, sector_out::Real,
                                   ninds_a::Int, ninds_b::Int,
                                   conj_a::Bool, conj_b::Bool;
                                   profile=nothing)
    pA = (metrics.oindA..., metrics.cindA...)
    pB = (metrics.cindB..., metrics.oindB...)
    pAB = (out[1]..., out[2]...)
    perm_dense = 0.0
    perm_sector = 0.0
    permuted_a_sector = profile === nothing ? sector_a :
                        profile.left_permuted_stored_elements
    permuted_b_sector = profile === nothing ? sector_b :
                        profile.right_permuted_stored_elements
    product_sector = profile === nothing ? sector_out :
                     profile.product_stored_elements
    if !_is_identity_layout(pA, ninds_a)
        perm_dense += dense_a
        perm_sector += permuted_a_sector
    end
    if !_is_identity_layout(pB, ninds_b)
        perm_dense += dense_b
        perm_sector += permuted_b_sector
    end
    if !_is_identity_layout(pAB, length(metrics.dims))
        perm_dense += dense_out
        perm_sector += product_sector
    end
    # TensorOperations can often fuse conjugation into a block kernel, but a
    # full transformed operand is a known safe upper bound when that is not
    # possible.  Keep it distinct from permutation diagnostics below.
    temporary_dense = perm_dense + (conj_a ? dense_a : 0.0) +
                      (conj_b ? dense_b : 0.0)
    temporary_sector = perm_sector + (conj_a ? sector_a : 0.0) +
                       (conj_b ? sector_b : 0.0)
    return (temporary_dense=temporary_dense,
            permutation_dense=perm_dense,
            temporary_sector=temporary_sector,
            permutation_sector=perm_sector)
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
Base.@noinline function _sector_pair_profile(metrics, spaceA, conjA::Bool,
                                             spaceB, conjB::Bool, out::Tuple)
    pA = (Tuple(metrics.oindA), Tuple(metrics.cindA))
    # TensorOperations treats B as `(contracted, open)` for a matrix product.
    pB = (Tuple(metrics.cindB), Tuple(metrics.oindB))
    return Backend.pair_cost(spaceA, pA, conjA, spaceB, pB, conjB, out)
end

function _score(plan::ContractionPlan, memory_weight::Real)
    return plan.flops + Float64(memory_weight) * plan.live_peak_bytes
end

function _sector_score(plan::ContractionPlan, memory_weight::Real)
    return plan.sector_flops + Float64(memory_weight) * plan.sector_live_peak_bytes
end
