import ..Contractions: Planning

"""
    apply(O::TTNO, ψ::TTNS; center=center(ψ), optimize=true) -> TTNS

Exact TTNO-by-TTNS zip application. Each output virtual edge is the fused
state/operator edge space, so the result remains a valid TTNS with one parent
leg per node. The result is canonicalized to `center` before return, preserving
the single-center invariant; variational recompression is the separate
`fit!` primitive.
Set `optimize=false` to use the deterministic env-first contraction order. This
is useful for small exact-action diagnostics where planner latency dominates.
"""
function apply(O::TTNO, ψ::TTNS; center=center(ψ), optimize::Bool=true)
    topology(O) == topology(ψ) || throw(ArgumentError("apply: TTNO and TTNS topologies differ"))
    O.hasphys == ψ.hasphys || throw(ArgumentError("apply: physical-leg layout mismatch"))
    spacetype(O) == spacetype(ψ) || throw(ArgumentError("apply: TTNO and TTNS spacetype mismatch"))
    t = ψ.topo
    S = spacetype(ψ)
    T = promote_type(eltype(O), eltype(ψ))

    # In preorder, a fusion is first consumed by its parent as a child map and
    # then by its own node as the parent map. It can be released immediately
    # after that node contraction; plans themselves retain only structure.
    fusions = _apply_edge_fusions(T, ψ, O)
    plans = Dict{Planning.PlanKey,Planning.ContractionPlan}()
    tensors = Vector{AbstractTensorMap{T,S}}(undef, nnodes(t))
    for n in preorder(t)
        tensors[n] = _apply_node_tensor(plans, O, ψ, n, fusions; optimize)
        delete!(fusions, n)
    end
    out = TTNS(t, tensors, t.root)
    return _canonicalize_apply!(out, center isa Symbol ? nodeindex(t, center) : center)
end

"""Create each state/operator edge fusion once for one exact application."""
function _apply_edge_fusions(::Type{T}, ψ::TTNS{S}, O::TTNO{S}) where
                             {T<:Number,S<:ElementarySpace}
    t = ψ.topo
    fusions = Dict{Int,AbstractTensorMap{T,S}}()
    for n in 1:nnodes(t)
        fusions[n] = _edge_fusion(T, ψ, O, n)
    end
    return fusions
end

"""
    _apply_node_spec(O, ψ, n, fusions) -> (spec, operands)

Lower the existing TTNO-by-TTNS node zip into its legacy operand order
`(state, operator, child fusions..., adjoint(parent fusion))`.  Fusions are
the maps created once by `_apply_edge_fusions`; no child map is rebuilt here.
"""
function _apply_node_spec(O::TTNO{S}, ψ::TTNS{S}, n::Int,
                          fusions::Dict{Int,<:AbstractTensorMap}) where {S<:ElementarySpace}
    t = ψ.topo
    A = ψ.tensors[n]
    W = O.tensors[n]
    K = nchildren(t, n)
    hp = hasphys(ψ, n)
    parent_fusion = get(fusions, n, nothing)
    parent_fusion === nothing &&
        throw(ArgumentError("apply node $(nodeid(t, n)) is missing its edge fusion"))

    operands = Any[A, W]
    labels = Vector{Int}[zeros(Int, numind(A)), zeros(Int, numind(W))]
    conjs = Bool[false, false]
    aidx, widx = labels
    nxt = Ref(0)
    fresh() = (nxt[] += 1; nxt[])
    outleg = Ref(0)
    open() = -(outleg[] += 1)
    # Physical nodes can contract A/W immediately through P. At a physless
    # branching node those maps are disconnected, so fold each child fusion
    # into A first and avoid an avoidable A⊗W outer product.
    preferred = hp ? Int[1, 2] : Int[1]

    for (k, c) in enumerate(t.children[n])
        F = get(fusions, c, nothing)
        F === nothing &&
            throw(ArgumentError("apply node $(nodeid(t, n)) is missing child fusion $(nodeid(t, c))"))
        sleg, oleg = fresh(), fresh()
        aidx[k] = sleg
        widx[k] = oleg
        push!(operands, F); push!(labels, [open(), sleg, oleg]); push!(conjs, false)
        push!(preferred, length(labels))
    end
    if hp
        pin = fresh()
        aidx[K + 1] = pin
        widx[K + 2] = pin
        widx[K + 1] = open()
    end
    ps, po = fresh(), fresh()
    aidx[end] = ps
    widx[end] = po
    push!(operands, adjoint(parent_fusion))
    push!(labels, [ps, po, open()])
    push!(conjs, false)
    parent_slot = length(labels)
    if !hp && K == 0
        # A physless leaf has no child bridge; bind its parent fusion to A
        # before introducing W, again avoiding a disconnected outer product.
        push!(preferred, parent_slot, 2)
    else
        hp || push!(preferred, 2)
        push!(preferred, parent_slot)
    end
    spec = Planning.ContractionSpec(labels, conjs, outleg[], (outleg[] - 1, 1), nothing;
                                    preferred_slots=preferred)
    return spec, Tuple(operands)
end

"""Execute one planned TTNO-by-TTNS node contraction with shape-only reuse."""
function _apply_node_tensor(plans::Dict{Planning.PlanKey,Planning.ContractionPlan},
                            O::TTNO, ψ::TTNS, n::Int,
                            fusions::Dict{Int,<:AbstractTensorMap};
                            optimize::Bool=true)
    spec, operands = _apply_node_spec(O, ψ, n, fusions)
    T = promote_type(eltype(O), eltype(ψ))
    plan, _ = Planning.get_or_plan!(plans, :ttno_state_apply, spec, operands, T;
                                    optimize)
    return Planning.execute(plan, operands)
end

"""
    _apply_ncon_reference(O, ψ; center=center(ψ)) -> TTNS

Retained private exact `ncon` implementation for A/B tests.  It deliberately
rebuilds child fusions through the historical `_edge_fusion` call; production
`apply` above reuses the prebuilt maps instead.
"""
function _apply_ncon_reference(O::TTNO, ψ::TTNS; center=center(ψ))
    topology(O) == topology(ψ) || throw(ArgumentError("apply: TTNO and TTNS topologies differ"))
    O.hasphys == ψ.hasphys || throw(ArgumentError("apply: physical-leg layout mismatch"))
    spacetype(O) == spacetype(ψ) || throw(ArgumentError("apply: TTNO and TTNS spacetype mismatch"))
    t = ψ.topo
    S = spacetype(ψ)
    T = promote_type(eltype(O), eltype(ψ))
    fusions = _apply_edge_fusions(T, ψ, O)
    tensors = Vector{AbstractTensorMap{T,S}}(undef, nnodes(t))
    for n in 1:nnodes(t)
        tensors[n] = _apply_node_tensor_ncon_reference(O, ψ, n, fusions[n])
    end
    out = TTNS(t, tensors, t.root)
    return _canonicalize_apply!(out, center isa Symbol ? nodeindex(t, center) : center)
end

"""Retained direct node `ncon` reference used by `_apply_ncon_reference`."""
function _apply_node_tensor_ncon_reference(O::TTNO{S}, ψ::TTNS{S}, n::Int,
                                            parent_fusion) where {S<:ElementarySpace}
    t = ψ.topo
    A = ψ.tensors[n]
    W = O.tensors[n]
    T = promote_type(eltype(O), eltype(ψ))
    K = nchildren(t, n)
    hp = hasphys(ψ, n)

    tensors = Any[A, W]
    aidx = zeros(Int, numind(A))
    widx = zeros(Int, numind(W))
    indices = Vector{Int}[aidx, widx]
    conjs = Bool[false, false]
    nxt = Ref(0)
    fresh() = (nxt[] += 1; nxt[])
    outleg = Ref(0)
    open() = -(outleg[] += 1)

    for (k, c) in enumerate(t.children[n])
        sleg = fresh()
        oleg = fresh()
        aidx[k] = sleg
        widx[k] = oleg
        F = _edge_fusion(T, ψ, O, c)
        push!(tensors, F); push!(indices, [open(), sleg, oleg]); push!(conjs, false)
    end
    if hp
        pin = fresh()
        aidx[K + 1] = pin
        widx[K + 2] = pin
        widx[K + 1] = open()
    end
    ps = fresh()
    po = fresh()
    aidx[end] = ps
    widx[end] = po
    push!(tensors, adjoint(parent_fusion)); push!(indices, [ps, po, open()]); push!(conjs, false)
    y = ncon(tensors, indices, conjs)
    return repartition(y, outleg[] - 1, 1)
end

function _edge_fusion(::Type{T}, ψ::TTNS, O::TTNO, child::Int) where {T<:Number}
    Vψ = domain(ψ.tensors[child])[1]
    VO = domain(O.tensors[child])[numin(O.tensors[child])]
    return unitary(T, fuse(Vψ ⊗ VO), Vψ ⊗ VO)
end

function _canonicalize_apply!(ψ::TTNS, target::Int)
    t = ψ.topo
    for n in postorder(t)
        n == t.root && continue
        Q, C = left_orth(ψ.tensors[n])
        ψ.tensors[n] = Q
        p = t.parent[n]
        C = _pivotal_link(C)
        ψ.tensors[p] = absorb_on_leg(ψ.tensors[p], C, childslot(t, p, n))
    end
    ψ.center = t.root
    return move_center!(ψ, target)
end
