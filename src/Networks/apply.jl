"""
    apply(O::TTNO, ψ::TTNS; center=center(ψ)) -> TTNS

Exact TTNO-by-TTNS zip application. Each output virtual edge is the fused
state/operator edge space, so the result remains a valid TTNS with one parent
leg per node. The result is canonicalized to `center` before return, preserving
the single-center invariant; variational recompression is the separate
`fit!` primitive.
"""
function apply(O::TTNO, ψ::TTNS; center=center(ψ))
    topology(O) == topology(ψ) || throw(ArgumentError("apply: TTNO and TTNS topologies differ"))
    O.hasphys == ψ.hasphys || throw(ArgumentError("apply: physical-leg layout mismatch"))
    spacetype(O) == spacetype(ψ) || throw(ArgumentError("apply: TTNO and TTNS spacetype mismatch"))
    t = ψ.topo
    S = spacetype(ψ)
    T = promote_type(eltype(O), eltype(ψ))

    fusion = Dict{Int,AbstractTensorMap{T,S}}()
    for n in 1:nnodes(t)
        Vψ = domain(ψ.tensors[n])[1]
        VO = domain(O.tensors[n])[numin(O.tensors[n])]
        fusion[n] = unitary(T, fuse(Vψ ⊗ VO), Vψ ⊗ VO)
    end

    tensors = Vector{AbstractTensorMap{T,S}}(undef, nnodes(t))
    for n in 1:nnodes(t)
        tensors[n] = _apply_node_tensor(O, ψ, n, fusion[n])
    end
    out = TTNS(t, tensors, t.root)
    return _canonicalize_apply!(out, center isa Symbol ? nodeindex(t, center) : center)
end

function _apply_node_tensor(O::TTNO{S}, ψ::TTNS{S}, n::Int,
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
        ψ.tensors[p] = absorb_on_leg(ψ.tensors[p], C, childslot(t, p, n))
    end
    ψ.center = t.root
    return move_center!(ψ, target)
end
