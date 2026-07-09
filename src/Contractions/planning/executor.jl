"""
    execute(plan, x, statics) -> TensorMap

Walk a compiled plan using a fresh dynamic input and persistent static operands.
Every intermediate owns a unique destination slot; immediately after its only
parent use, both source slots are nulled so previous intermediates are no
longer retained by the live slot graph. TensorOperations' default allocator is
GC-managed, so this makes tensors reclaimable rather than promising an
immediate drop in OS RSS.
"""
function execute(plan::ContractionPlan, x::AbstractTensorMap, statics::Tuple)
    length(statics) + 1 <= plan.nslots ||
        throw(ArgumentError("EffectiveMap statics exceed plan input slots"))
    slots = Vector{Any}(undef, plan.nslots)
    slots[1] = x
    for (i, tensor) in enumerate(statics)
        slots[i + 1] = tensor
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
    return y
end

"""Retained legacy reference path for A/B tests and benchmark validation."""
function ncon_reference(spec::ContractionSpec, x::AbstractTensorMap, statics::Tuple)
    tensors = Any[x]
    append!(tensors, statics)
    y = ncon(tensors, spec.labels, spec.conjs)
    nout, nin = spec.out_partition
    return nin == 0 ? y : repartition(y, nout, nin)
end
