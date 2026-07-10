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
function execute(plan::ContractionPlan, operands::Tuple;
                 workspace::Union{Nothing,PlanWorkspace}=nothing)
    ninputs = _input_slot_count(plan)
    length(operands) == ninputs ||
        throw(ArgumentError("compiled contraction plan expects $ninputs operands, got $(length(operands))"))
    workspace === nothing || return _execute_workspace(plan, operands, workspace)
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

"""Fill one internal workspace destination and release per-step temporaries."""
function _workspace_contract!(workspace::PlanWorkspace, C::AbstractTensorMap,
                              A::AbstractTensorMap, pA::Tuple, conjA::Bool,
                              B::AbstractTensorMap, pB::Tuple, conjB::Bool,
                              pAB::Tuple, α::Number=1, β::Number=0)
    (C === A || C === B) &&
        throw(ArgumentError("planned workspace destination aliases a contraction input"))
    checkpoint = TensorOperations.allocator_checkpoint!(workspace.allocator)
    try
        return Backend.contract_pair!(C, A, pA, conjA, B, pB, conjB, pAB, α, β;
                                      allocator=workspace.allocator)
    finally
        TensorOperations.allocator_reset!(workspace.allocator, checkpoint)
    end
end

"""Return one color-owned internal destination, allocating only as needed."""
function _workspace_destination!(workspace::PlanWorkspace, step::PairStep,
                                 A::AbstractTensorMap, pA::Tuple, conjA::Bool,
                                 B::AbstractTensorMap, pB::Tuple, conjB::Bool,
                                 pAB::Tuple)
    color = workspace.layout.colors[step.dst]
    color > 0 || throw(ArgumentError("compiled plan has no workspace color for slot $(step.dst)"))
    C = workspace.buffers[color]
    if !(C isa AbstractTensorMap)
        C = Backend.allocate_contract_pair(A, pA, conjA, B, pB, conjB, pAB)
        workspace.buffers[color] = C
        workspace.layout.representatives[color] = step.dst
        workspace.allocations += 1
    elseif !Backend.contract_pair_compatible(C, A, pA, conjA, B, pB, conjB, pAB)
        # A liveness color can cover two non-overlapping slots with different
        # TensorKit HomSpaces. Specialize that color on first use so each
        # persisted buffer stays exact rather than being repeatedly replaced.
        if workspace.layout.representatives[color] != step.dst
            workspace.layout.ncolors += 1
            color = workspace.layout.ncolors
            workspace.layout.colors[step.dst] = color
            push!(workspace.layout.representatives, step.dst)
            C = Backend.allocate_contract_pair(A, pA, conjA, B, pB, conjB, pAB)
            push!(workspace.buffers, C)
        else
            C = Backend.allocate_contract_pair(A, pA, conjA, B, pB, conjB, pAB)
            workspace.buffers[color] = C
        end
        workspace.allocations += 1
    else
        workspace.reuses += 1
    end
    return C
end

"""Workspace execution with fresh roots and task-local internal buffers."""
function _execute_workspace(plan::ContractionPlan, operands::Tuple,
                            workspace::PlanWorkspace)
    _enter_workspace!(workspace, plan)
    slots = Vector{Any}(undef, plan.nslots)
    try
        for (i, tensor) in enumerate(operands)
            tensor isa AbstractTensorMap ||
                throw(ArgumentError("compiled contraction plan operand $i is not an AbstractTensorMap"))
            slots[i] = tensor
        end
        if isempty(plan.steps)
            y = slots[plan.output_slot]
            y isa AbstractTensorMap ||
                throw(ArgumentError("compiled contraction produced no output"))
            return plan.scalar_output ? Backend.tensor_scalar(y) : copy(y)
        end
        for step in plan.steps
            A = slots[step.a]
            B = slots[step.b]
            A isa AbstractTensorMap && B isa AbstractTensorMap ||
                throw(ArgumentError("compiled contraction plan references a released slot"))
            pA = (step.oindA, step.cindA)
            pB = (step.cindB, step.oindB)
            if step.dst == plan.output_slot
                # KrylovKit can retain every map result, so the root is never
                # color-owned or allocator-backed across executions.
                C = Backend.allocate_contract_pair(A, pA, step.conjA,
                                                    B, pB, step.conjB, step.out)
            else
                C = _workspace_destination!(workspace, step, A, pA, step.conjA,
                                            B, pB, step.conjB, step.out)
            end
            slots[step.dst] = _workspace_contract!(workspace, C, A, pA, step.conjA,
                                                    B, pB, step.conjB, step.out)
            slots[step.a] = nothing
            slots[step.b] = nothing
        end
        y = slots[plan.output_slot]
        y isa AbstractTensorMap ||
            throw(ArgumentError("compiled contraction produced no output"))
        return plan.scalar_output ? Backend.tensor_scalar(y) : y
    finally
        fill!(slots, nothing)
        _leave_workspace!(workspace)
    end
end

"""
    execute_accumulate!(dest, plan, operands; α=1, β=1) -> dest

Walk a complete-tuple plan while sending its final binary contraction straight
to a caller-owned output map.  It is intentionally limited to non-scalar
plans with at least one binary step: internal intermediates retain the normal
fresh-allocation semantics, while a multi-source caller can add each local
projection into one destination without allocating or retaining a full
per-source result.  The destination is never stored in a plan or cache.
"""
function execute_accumulate!(dest::AbstractTensorMap, plan::ContractionPlan,
                             operands::Tuple; α::Number=1, β::Number=1,
                             workspace::Union{Nothing,PlanWorkspace}=nothing)
    plan.scalar_output &&
        throw(ArgumentError("cannot accumulate a scalar contraction plan into a TensorMap"))
    isempty(plan.steps) &&
        throw(ArgumentError("cannot accumulate a zero-step contraction plan"))
    ninputs = _input_slot_count(plan)
    length(operands) == ninputs ||
        throw(ArgumentError("compiled contraction plan expects $ninputs operands, got $(length(operands))"))
    workspace === nothing ||
        return _execute_accumulate_workspace!(dest, plan, operands, workspace; α, β)
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
        if step.dst == plan.output_slot
            Backend.contract_pair!(dest, A, pA, step.conjA,
                                   B, pB, step.conjB, step.out, α, β)
        else
            slots[step.dst] = Backend.contract_pair(A, pA, step.conjA,
                                                     B, pB, step.conjB, step.out)
        end
        slots[step.a] = nothing
        slots[step.b] = nothing
        A = nothing
        B = nothing
    end
    return dest
end

"""Workspace variant of `execute_accumulate!`; only `dest` is caller-owned."""
function _execute_accumulate_workspace!(dest::AbstractTensorMap,
                                        plan::ContractionPlan, operands::Tuple,
                                        workspace::PlanWorkspace;
                                        α::Number=1, β::Number=1)
    _enter_workspace!(workspace, plan)
    slots = Vector{Any}(undef, plan.nslots)
    try
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
            if step.dst == plan.output_slot
                _workspace_contract!(workspace, dest, A, pA, step.conjA,
                                     B, pB, step.conjB, step.out, α, β)
            else
                C = _workspace_destination!(workspace, step, A, pA, step.conjA,
                                            B, pB, step.conjB, step.out)
                slots[step.dst] = _workspace_contract!(workspace, C, A, pA, step.conjA,
                                                        B, pB, step.conjB, step.out)
            end
            slots[step.a] = nothing
            slots[step.b] = nothing
        end
        return dest
    finally
        fill!(slots, nothing)
        _leave_workspace!(workspace)
    end
end

"""Adapter for Krylov maps with slot 1 supplied dynamically."""
execute(plan::ContractionPlan, x::AbstractTensorMap, statics::Tuple; kwargs...) =
    execute(plan, (x, statics...); kwargs...)

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
