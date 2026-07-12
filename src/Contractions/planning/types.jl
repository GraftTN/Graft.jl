"""
    ContractionSpec(labels, conjs, nopen, out_partition, dynamic_slot=nothing;
                    preferred_slots)

Value-level description of one contraction network. Labels retain the ncon
convention (`>0` contracted, `<0` output) so retained reference paths are
mechanically identical to legacy implementations. `dynamic_slot` is slot 1
for Krylov maps and `nothing` for a complete value-level operand tuple.
`preferred_slots` is semantic input from the caller: it encodes an env-first
fold without leaking TTN knowledge into the generic planner.
"""
struct ContractionSpec
    labels::Vector{Vector{Int}}
    conjs::Vector{Bool}
    nopen::Int
    out_partition::Tuple{Int,Int}
    dynamic_slot::Union{Nothing,Int}
    preferred_slots::Vector{Int}
end

function ContractionSpec(labels::Vector{<:AbstractVector{<:Integer}},
                         conjs::AbstractVector{Bool}, nopen::Integer,
                         out_partition::Tuple{<:Integer,<:Integer},
                         dynamic_slot::Union{Nothing,Integer}=nothing;
                         preferred_slots::AbstractVector{<:Integer}=Int[])
    copied_labels = [Int[label...] for label in labels]
    copied_conjs = Bool[conjs...]
    np = Int(nopen)
    out = (Int(out_partition[1]), Int(out_partition[2]))
    dyn = dynamic_slot === nothing ? nothing : Int(dynamic_slot)
    order = Int[preferred_slots...]
    length(copied_labels) == length(copied_conjs) ||
        throw(ArgumentError("ContractionSpec: labels/conjs length mismatch"))
    dyn === nothing || 1 <= dyn <= length(copied_labels) ||
        throw(ArgumentError("ContractionSpec: invalid dynamic slot $dyn"))
    dyn === nothing || dyn == 1 ||
        throw(ArgumentError("ContractionSpec: only dynamic slot 1 is supported"))
    sum(out) == np ||
        throw(ArgumentError("ContractionSpec: output partition $out does not contain $np open legs"))
    input_slots = [i for i in eachindex(copied_labels) if i != dyn]
    isempty(order) && (order = input_slots)
    sort(order) == input_slots ||
        throw(ArgumentError("ContractionSpec: preferred slots must contain every non-dynamic slot once"))
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
A concrete, cacheable sequence of pair contractions plus dense and
symmetry-aware metrics.

`peak_elements` and the existing `sector_*_elements` diagnostics retain their
original meaning: largest individual output/intermediate payload.  The
`*_live_peak_bytes` fields are a separate conservative allocation model. They
charge every original operand (which remains owned by the caller or an
`EffectiveMap`), all simultaneously-live internal outputs, the new pair
output, and known transformation buffers.  `scalar_bytes` makes the model
independent of the scalar type.  The `sector_*` byte fields use TensorKit's
stored block payload where structural planning supports it, and otherwise
fall back to the dense model.
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
    scalar_bytes::Int
    operand_bytes::Float64
    live_peak_bytes::Float64
    known_temporary_peak_bytes::Float64
    known_permutation_peak_bytes::Float64
    sector_operand_bytes::Float64
    sector_live_peak_bytes::Float64
    sector_known_temporary_peak_bytes::Float64
    sector_known_permutation_peak_bytes::Float64
    scalar_output::Bool
end

function ContractionPlan(nslots::Integer, output_slot::Integer,
                         steps::Vector{PairStep}; strategy::Symbol=:heuristic,
                         flops::Real=NaN, peak_elements::Real=NaN,
                         sector_flops::Real=NaN,
                         sector_peak_elements::Real=NaN,
                         sector_peak_block_elements::Real=NaN,
                         scalar_bytes::Integer=sizeof(Float64),
                         operand_bytes::Real=NaN,
                         live_peak_bytes::Real=NaN,
                         known_temporary_peak_bytes::Real=NaN,
                         known_permutation_peak_bytes::Real=NaN,
                         sector_operand_bytes::Real=NaN,
                         sector_live_peak_bytes::Real=NaN,
                         sector_known_temporary_peak_bytes::Real=NaN,
                         sector_known_permutation_peak_bytes::Real=NaN,
                         scalar_output::Bool=false)
    return ContractionPlan(Int(nslots), Int(output_slot), steps, strategy,
                           Float64(flops), Float64(peak_elements),
                           Float64(sector_flops),
                           Float64(sector_peak_elements),
                           Float64(sector_peak_block_elements),
                           Int(scalar_bytes), Float64(operand_bytes),
                           Float64(live_peak_bytes),
                           Float64(known_temporary_peak_bytes),
                           Float64(known_permutation_peak_bytes),
                           Float64(sector_operand_bytes),
                           Float64(sector_live_peak_bytes),
                           Float64(sector_known_temporary_peak_bytes),
                           Float64(sector_known_permutation_peak_bytes),
                           scalar_output)
end

# Preserve the pre-live-model positional constructor for downstream users that
# construct a diagnostic plan directly.  Real plans are compiled through the
# keyword constructor above and therefore always carry byte metrics.
function ContractionPlan(nslots::Integer, output_slot::Integer,
                         steps::Vector{PairStep}, strategy::Symbol,
                         flops::Real, peak_elements::Real,
                         sector_flops::Real, sector_peak_elements::Real,
                         sector_peak_block_elements::Real)
    return ContractionPlan(nslots, output_slot, steps;
                           strategy, flops, peak_elements, sector_flops,
                           sector_peak_elements, sector_peak_block_elements)
end

# Preserve the full positional layout introduced by milestone 1. Scalar-output
# metadata was added later for complete-tuple execution and safely defaults to
# false for manually reconstructed pre-M2 plans.
function ContractionPlan(nslots::Integer, output_slot::Integer,
                         steps::Vector{PairStep}, strategy::Symbol,
                         flops::Real, peak_elements::Real,
                         sector_flops::Real, sector_peak_elements::Real,
                         sector_peak_block_elements::Real,
                         scalar_bytes::Integer, operand_bytes::Real,
                         live_peak_bytes::Real,
                         known_temporary_peak_bytes::Real,
                         known_permutation_peak_bytes::Real,
                         sector_operand_bytes::Real,
                         sector_live_peak_bytes::Real,
                         sector_known_temporary_peak_bytes::Real,
                         sector_known_permutation_peak_bytes::Real)
    return ContractionPlan(Int(nslots), Int(output_slot), steps, strategy,
                           Float64(flops), Float64(peak_elements),
                           Float64(sector_flops), Float64(sector_peak_elements),
                           Float64(sector_peak_block_elements), Int(scalar_bytes),
                           Float64(operand_bytes), Float64(live_peak_bytes),
                           Float64(known_temporary_peak_bytes),
                           Float64(known_permutation_peak_bytes),
                           Float64(sector_operand_bytes),
                           Float64(sector_live_peak_bytes),
                           Float64(sector_known_temporary_peak_bytes),
                           Float64(sector_known_permutation_peak_bytes), false)
end

"""
Callable effective Hamiltonian. `statics` owns W, cached environments and an
optional root cap; slot 1 is supplied afresh by KrylovKit on every invocation.
"""
struct EffectiveMap{T<:Tuple,I<:Tuple}
    plan::ContractionPlan
    statics::T
    output_twists::I
end

EffectiveMap(plan::ContractionPlan, statics::T) where {T<:Tuple} =
    EffectiveMap(plan, statics, ())

function (f::EffectiveMap)(x::AbstractTensorMap)
    y = execute(f.plan, x, f.statics)
    isempty(f.output_twists) || Backend.twist!(y, f.output_twists)
    return y
end

function Base.show(io::IO, f::EffectiveMap)
    print(io, "EffectiveMap(strategy=", f.plan.strategy,
          ", steps=", length(f.plan.steps),
          ", dense_peak≈", f.plan.peak_elements,
          ", live_peak≈", f.plan.live_peak_bytes,
          " B, sector_peak≈", f.plan.sector_peak_elements, " elements)")
end

# Workspace colors are compiled from the postorder plan rather than inferred
# while executing it. A color can be reused only after its prior source has
# been consumed by an earlier binary step; equality would alias a destination
# with an input to `tensorcontract!`, which TensorOperations forbids.
mutable struct _WorkspaceLayout
    colors::Vector{Int}
    births::Vector{Int}
    last_uses::Vector{Int}
    ncolors::Int
    representatives::Vector{Int}
end

function _workspace_layout(plan::ContractionPlan)
    births = zeros(Int, plan.nslots)
    last_uses = zeros(Int, plan.nslots)
    for (i, step) in enumerate(plan.steps)
        births[step.dst] == 0 ||
            throw(ArgumentError("compiled contraction plan produces slot $(step.dst) twice"))
        births[step.dst] = i
    end
    for (i, step) in enumerate(plan.steps), slot in (step.a, step.b)
        births[slot] == 0 && continue
        last_uses[slot] == 0 ||
            throw(ArgumentError("compiled contraction plan consumes intermediate slot $slot twice"))
        last_uses[slot] = i
    end

    colors = zeros(Int, plan.nslots)
    color_last_uses = Int[]
    internal = sort!([step.dst for step in plan.steps if step.dst != plan.output_slot];
                     by=slot -> births[slot])
    for slot in internal
        birth, last_use = births[slot], last_uses[slot]
        last_use > 0 ||
            throw(ArgumentError("compiled contraction plan leaves intermediate slot $slot unused"))
        color = findfirst(last -> last < birth, color_last_uses)
        if color === nothing
            push!(color_last_uses, last_use)
            color = length(color_last_uses)
        else
            color_last_uses[color] = last_use
        end
        colors[slot] = color
    end
    ncolors = length(color_last_uses)
    return _WorkspaceLayout(colors, births, last_uses, ncolors, zeros(Int, ncolors))
end

"""
    PlanWorkspace(plan)

Task-bound mutable storage for the strictly internal outputs of one compiled
plan. It is intentionally separate from `EffectiveMap` and `EnvCache`: a
shared map remains safe to call concurrently, while a solver obtains one
workspace for its own serial Krylov invocation. Root outputs are never kept in
this workspace and are always allocated fresh.
"""
mutable struct PlanWorkspace
    plan::ContractionPlan
    layout::_WorkspaceLayout
    buffers::Vector{Any}
    allocator::TensorOperations.BufferAllocator
    owner::Union{Nothing,Task}
    busy::Bool
    allocations::Int
    reuses::Int
end

function PlanWorkspace(plan::ContractionPlan)
    layout = _workspace_layout(plan)
    return PlanWorkspace(plan, layout, Any[nothing for _ in 1:layout.ncolors],
                         TensorOperations.BufferAllocator(), nothing, false, 0, 0)
end

"""Observable internal allocation state for a task-local plan workspace."""
workspace_stats(workspace::PlanWorkspace) =
    (colors=workspace.layout.ncolors,
     buffers=count(x -> !isnothing(x), workspace.buffers),
     allocations=workspace.allocations,
     reuses=workspace.reuses,
     temporary_buffer_bytes=length(workspace.allocator),
     owner_bound=workspace.owner !== nothing,
     busy=workspace.busy)

function _enter_workspace!(workspace::PlanWorkspace, plan::ContractionPlan)
    workspace.plan === plan ||
        throw(ArgumentError("PlanWorkspace belongs to a different ContractionPlan"))
    task = current_task()
    if workspace.owner === nothing
        workspace.owner = task
    elseif workspace.owner !== task
        throw(ArgumentError("PlanWorkspace is task-local and cannot be used from another Task"))
    end
    workspace.busy &&
        throw(ArgumentError("PlanWorkspace cannot be used reentrantly"))
    workspace.busy = true
    return workspace
end

_leave_workspace!(workspace::PlanWorkspace) = (workspace.busy = false; workspace)

"""Callable task-local workspace wrapper for one otherwise immutable map."""
struct WorkspaceMap{F<:EffectiveMap}
    effective::F
    workspace::PlanWorkspace
end

workspace_map(effective::EffectiveMap) =
    WorkspaceMap(effective, PlanWorkspace(effective.plan))

function (map::WorkspaceMap)(x::AbstractTensorMap)
    y = execute(map.effective.plan, x, map.effective.statics; workspace=map.workspace)
    isempty(map.effective.output_twists) ||
        Backend.twist!(y, map.effective.output_twists)
    return y
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
    memory_cap_bytes::Float64
end

Base.:(==)(a::PlanKey, b::PlanKey) =
    a.kind == b.kind && a.shape == b.shape && a.T == b.T &&
    a.optimize == b.optimize && a.memory_weight == b.memory_weight &&
    a.sector_aware == b.sector_aware &&
    a.memory_cap_bytes == b.memory_cap_bytes
Base.hash(k::PlanKey, h::UInt) =
    hash(k.memory_cap_bytes,
         hash(k.sector_aware,
              hash(k.memory_weight, hash(k.optimize,
                                         hash(k.T, hash(k.shape, hash(k.kind, h)))))))

# Keep the original positional key constructor source-compatible.  `Inf` is
# the canonical cache identity for the absence of a hard memory cap.
PlanKey(kind::Symbol, sig::UInt, shape::Tuple, T::DataType,
        optimize::Bool, memory_weight::Real, sector_aware::Bool) =
    PlanKey(kind, sig, shape, T, optimize, Float64(memory_weight), sector_aware, Inf)

"""Return all dense and symmetry-aware metrics attached to a compiled plan."""
plan_metrics(plan::ContractionPlan) =
    (flops=plan.flops,
     peak_elements=plan.peak_elements,
     sector_flops=plan.sector_flops,
     sector_peak_elements=plan.sector_peak_elements,
     sector_peak_block_elements=plan.sector_peak_block_elements,
     scalar_bytes=plan.scalar_bytes,
     operand_bytes=plan.operand_bytes,
     live_peak_bytes=plan.live_peak_bytes,
     known_temporary_peak_bytes=plan.known_temporary_peak_bytes,
     known_permutation_peak_bytes=plan.known_permutation_peak_bytes,
     sector_operand_bytes=plan.sector_operand_bytes,
     sector_live_peak_bytes=plan.sector_live_peak_bytes,
     sector_known_temporary_peak_bytes=plan.sector_known_temporary_peak_bytes,
     sector_known_permutation_peak_bytes=plan.sector_known_permutation_peak_bytes,
     scalar_output=plan.scalar_output,
     strategy=plan.strategy)
