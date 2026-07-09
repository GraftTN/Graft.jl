"""
    ContractionSpec(labels, conjs, nopen, out_partition, dynamic_slot;
                    preferred_slots)

Value-level description of one effective-Hamiltonian network. Labels retain
the ncon convention (`>0` contracted, `<0` output) so the retained reference
path is mechanically identical to the legacy implementation. `preferred_slots`
is semantic input from `effective.jl`: it encodes the Phase-1 env-first fold
without leaking TTN knowledge into the generic planner.
"""
struct ContractionSpec
    labels::Vector{Vector{Int}}
    conjs::Vector{Bool}
    nopen::Int
    out_partition::Tuple{Int,Int}
    dynamic_slot::Int
    preferred_slots::Vector{Int}
end

function ContractionSpec(labels::Vector{<:AbstractVector{<:Integer}},
                         conjs::AbstractVector{Bool}, nopen::Integer,
                         out_partition::Tuple{<:Integer,<:Integer},
                         dynamic_slot::Integer;
                         preferred_slots::AbstractVector{<:Integer}=Int[])
    copied_labels = [Int[label...] for label in labels]
    copied_conjs = Bool[conjs...]
    np = Int(nopen)
    out = (Int(out_partition[1]), Int(out_partition[2]))
    dyn = Int(dynamic_slot)
    order = Int[preferred_slots...]
    length(copied_labels) == length(copied_conjs) ||
        throw(ArgumentError("ContractionSpec: labels/conjs length mismatch"))
    1 <= dyn <= length(copied_labels) ||
        throw(ArgumentError("ContractionSpec: invalid dynamic slot $dyn"))
    dyn == 1 || throw(ArgumentError("ContractionSpec: only dynamic slot 1 is supported"))
    sum(out) == np ||
        throw(ArgumentError("ContractionSpec: output partition $out does not contain $np open legs"))
    isempty(order) && (order = [i for i in eachindex(copied_labels) if i != dyn])
    sort(order) == [i for i in eachindex(copied_labels) if i != dyn] ||
        throw(ArgumentError("ContractionSpec: preferred slots must contain every static slot once"))
    return ContractionSpec(copied_labels, copied_conjs, np, out, dyn, order)
end

"""
One compiled binary contraction. Index partitions are tuples in the exact
TensorOperations expert-mode representation, so execution does not recreate
labels or convert vectors on every Krylov matvec. `out` is the output
permutation and TensorMap codomain/domain partition; internal steps use an
all-codomain partition and the root step uses `ContractionSpec.out_partition`.
"""
struct PairStep
    a::Int
    b::Int
    dst::Int
    oindA::Tuple
    cindA::Tuple
    oindB::Tuple
    cindB::Tuple
    conjA::Bool
    conjB::Bool
    out::Tuple
end

"""
A concrete, cacheable sequence of pair contractions plus both Phase-2 dense and
Phase-3 symmetry-aware metrics.  `peak_elements` remains deliberately dense:
it is the hard memory-safety guard inherited from Phase 1.  The `sector_*`
fields describe TensorKit's symmetry-reduced storage/GEMM layout and are only
an optimization objective for unique-fusion sector types.
"""
struct ContractionPlan
    nslots::Int
    output_slot::Int
    steps::Vector{PairStep}
    strategy::Symbol
    flops::Float64
    peak_elements::Float64
    sector_flops::Float64
    sector_peak_elements::Float64
    sector_peak_block_elements::Float64
end

function ContractionPlan(nslots::Integer, output_slot::Integer,
                         steps::Vector{PairStep}; strategy::Symbol=:heuristic,
                         flops::Real=NaN, peak_elements::Real=NaN,
                         sector_flops::Real=NaN,
                         sector_peak_elements::Real=NaN,
                         sector_peak_block_elements::Real=NaN)
    return ContractionPlan(Int(nslots), Int(output_slot), steps, strategy,
                           Float64(flops), Float64(peak_elements),
                           Float64(sector_flops),
                           Float64(sector_peak_elements),
                           Float64(sector_peak_block_elements))
end

"""
Callable effective Hamiltonian. `statics` owns W, cached environments and an
optional root cap; slot 1 is supplied afresh by KrylovKit on every invocation.
"""
struct EffectiveMap{T<:Tuple}
    plan::ContractionPlan
    statics::T
end

(f::EffectiveMap)(x::AbstractTensorMap) = execute(f.plan, x, f.statics)

function Base.show(io::IO, f::EffectiveMap)
    print(io, "EffectiveMap(strategy=", f.plan.strategy,
          ", steps=", length(f.plan.steps),
          ", dense_peak≈", f.plan.peak_elements,
          ", sector_peak≈", f.plan.sector_peak_elements, " elements)")
end

"""
Structural cache identity. `shape` carries the exact label graph and exact
TensorKit spaces, not merely a hash, so symmetric h2 nodes with different
crossed-child slots can never share an invalid plan.
"""
struct PlanKey
    kind::Symbol
    sig::UInt
    shape::Tuple
    T::DataType
    optimize::Bool
    memory_weight::Float64
    sector_aware::Bool
end

Base.:(==)(a::PlanKey, b::PlanKey) =
    a.kind == b.kind && a.shape == b.shape && a.T == b.T &&
    a.optimize == b.optimize && a.memory_weight == b.memory_weight &&
    a.sector_aware == b.sector_aware
Base.hash(k::PlanKey, h::UInt) =
    hash(k.sector_aware,
         hash(k.memory_weight, hash(k.optimize,
                                    hash(k.T, hash(k.shape, hash(k.kind, h))))))

"""Return all dense and symmetry-aware metrics attached to a compiled plan."""
plan_metrics(plan::ContractionPlan) =
    (flops=plan.flops,
     peak_elements=plan.peak_elements,
     sector_flops=plan.sector_flops,
     sector_peak_elements=plan.sector_peak_elements,
     sector_peak_block_elements=plan.sector_peak_block_elements,
     strategy=plan.strategy)
