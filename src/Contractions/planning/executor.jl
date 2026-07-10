@inline _input_slot_count(plan::ContractionPlan) = plan.nslots - length(plan.steps)

"""
    execute(plan, operands) -> TensorMap or Number

Walk a compiled plan from its complete operand tuple. This is the general
value-level execution path used by environments and scalar contractions as
well as the adapter beneath `EffectiveMap`. Every intermediate owns a unique
destination slot; immediately after its only parent use, both source slots are
nulled so previous intermediates are no longer retained by the live slot graph.
TensorOperations' default allocator is GC-managed, so this makes tensors
reclaimable rather than promising an immediate drop in OS RSS. A plan compiled
for zero open legs scalarizes only its final rank-zero TensorMap through L0,
matching `ncon`'s public return convention.
"""
function execute(plan::ContractionPlan, operands::Tuple)
    ninputs = _input_slot_count(plan)
    length(operands) == ninputs ||
        throw(ArgumentError("compiled contraction plan expects $ninputs operands, got $(length(operands))"))
    slots = Vector{Any}(undef, plan.nslots)
    for (i, tensor) in enumerate(operands)
        tensor isa AbstractTensorMap ||
            throw(ArgumentError("compiled contraction plan operand $i is not an AbstractTensorMap"))
        slots[i] = tensor
    end
    for step in plan.steps
        A = slots[step.a]
        B = slots[step.b]
        A isa AbstractTensorMap && B isa AbstractTensorMap ||
            throw(ArgumentError("compiled contraction plan references a released slot"))
        pA = (step.oindA, step.cindA)
        pB = (step.cindB, step.oindB)
        slots[step.dst] = Backend.contract_pair(A, pA, step.conjA,
                                                 B, pB, step.conjB, step.out)
        slots[step.a] = nothing
        slots[step.b] = nothing
        A = nothing
        B = nothing
    end
    y = slots[plan.output_slot]
    y isa AbstractTensorMap || throw(ArgumentError("compiled contraction produced no output"))
    return plan.scalar_output ? Backend.tensor_scalar(y) : y
end

"""Adapter for Krylov maps with slot 1 supplied dynamically."""
execute(plan::ContractionPlan, x::AbstractTensorMap, statics::Tuple) =
    execute(plan, (x, statics...))

"""Retained legacy reference path for A/B tests and benchmark validation."""
function ncon_reference(spec::ContractionSpec, operands::Tuple)
    length(operands) == length(spec.labels) ||
        throw(ArgumentError("ncon reference expects $(length(spec.labels)) operands, got $(length(operands))"))
    tensors = Any[operands...]
    y = ncon(tensors, spec.labels, spec.conjs)
    nout, nin = spec.out_partition
    return nin == 0 ? y : repartition(y, nout, nin)
end

ncon_reference(spec::ContractionSpec, x::AbstractTensorMap, statics::Tuple) =
    ncon_reference(spec, (x, statics...))
