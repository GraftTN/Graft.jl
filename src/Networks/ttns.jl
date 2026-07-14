# TTNS — tree tensor network state (PyTreeNet: core/ttn.py + ttns/ttns.py +
# core/canonical_form.py), on the Backend arrow convention:
#
#   A[n] :: (⊗_{c ∈ children(n)} V_c) ⊗ P_n  ←  V_parent(n)
#
# The root's parent leg is `oneunit(S)` for charge-neutral states or a
# one-dimensional charge space for states in a fixed global sector.
#
# Invariants (architecture §9):
#   (1) exactly one orthogonality center; `move_center!` is the only mutation
#       of the gauge; every node ≠ center is isometric towards the center.
#   (2) tensor updates go through `update_tensor!` so caches can be invalidated.

"""
    TTNS{S,T}

Tree tensor network state. `topo` is an immutable `TreeTopology`; `tensors[i]`
is the node tensor of internal node `i` in the Backend leg convention;
`hasphys[i]` marks nodes carrying a physical leg; `center` is the
orthogonality center (internal index).
"""
mutable struct TTNS{S<:ElementarySpace,T<:Number}
    const topo::TreeTopology
    const tensors::Vector{AbstractTensorMap{T,S}}
    const hasphys::BitVector
    center::Int

    function TTNS(topo::TreeTopology, tensors::Vector{<:AbstractTensorMap},
                  center::Integer)
        isempty(tensors) && throw(ArgumentError("empty TTNS"))
        length(tensors) == nnodes(topo) || throw(ArgumentError("need one tensor per node"))
        S = spacetype(tensors[1])
        T = scalartype(tensors[1])
        hasphys = BitVector(numout(tensors[i]) == nchildren(topo, i) + 1 for i in 1:nnodes(topo))
        ψ = new{S,T}(topo, collect(AbstractTensorMap{T,S}, tensors), hasphys, Int(center))
        check_arrows(ψ)
        return ψ
    end
end

topology(ψ::TTNS) = ψ.topo
Base.eltype(::TTNS{S,T}) where {S,T} = T
Backend.spacetype(::TTNS{S}) where {S} = S
Trees.nnodes(ψ::TTNS) = nnodes(ψ.topo)
hasphys(ψ::TTNS, n::Int) = ψ.hasphys[n]
center(ψ::TTNS) = ψ.center

Base.getindex(ψ::TTNS, n::Int) = ψ.tensors[n]
Base.getindex(ψ::TTNS, s::Symbol) = ψ.tensors[nodeindex(ψ.topo, s)]

Base.copy(ψ::TTNS) = TTNS(ψ.topo, copy.(ψ.tensors), ψ.center)

"""Index of the physical leg of node `n` (throws if the node has none)."""
function physleg(ψ::TTNS, n::Int)
    ψ.hasphys[n] || throw(ArgumentError("node $(nodeid(ψ.topo, n)) has no physical leg"))
    return nchildren(ψ.topo, n) + 1
end

"""Index of the parent (domain) leg of node `n` in the flat leg numbering."""
parentleg(ψ::TTNS, n::Int) = numind(ψ.tensors[n])

physspace(ψ::TTNS, n::Int) = space(ψ.tensors[n], physleg(ψ, n))

"""Virtual space of the edge `(child, parent)` = domain of the child tensor."""
virtualspace(ψ::TTNS, child::Int) = domain(ψ.tensors[child])[1]

"""
    check_arrows(ψ) -> true

Debug/constructor guard for the Backend arrow convention (§9.3): every node has
exactly one domain leg; child-slot codomain factors match the child domains.
"""
function check_arrows(ψ::TTNS)
    t = ψ.topo
    for n in 1:nnodes(t)
        A = ψ.tensors[n]
        numin(A) == 1 || throw(SpaceMismatch("node $(nodeid(t, n)): expected exactly one parent (domain) leg"))
        numout(A) == nchildren(t, n) + ψ.hasphys[n] ||
            throw(SpaceMismatch("node $(nodeid(t, n)): codomain legs ≠ children (+ physical)"))
        for (k, c) in enumerate(t.children[n])
            space(A, k) == domain(ψ.tensors[c])[1] ||
                throw(SpaceMismatch("edge $(nodeid(t, c)) → $(nodeid(t, n)): slot space $(space(A, k)) ≠ child domain $(domain(ψ.tensors[c])[1])"))
        end
    end
    1 <= ψ.center <= nnodes(t) || throw(ArgumentError("invalid orthogonality center"))
    return true
end

Backend.norm(ψ::TTNS) = norm(ψ.tensors[ψ.center])

function normalize!(ψ::TTNS)
    n = norm(ψ)
    ψ.tensors[ψ.center] = ψ.tensors[ψ.center] / n
    return ψ
end

# A QR link factor whose codomain and domain have opposite dual orientation
# carries the ribbon pivotal twist when absorbed into the neighbouring tensor.
# Keep this correction at the network seam: Backend QR/absorption primitives
# remain context-free, while every tree algorithm shares one link convention.
function _pivotal_link(C::AbstractTensorMap)
    needs_twist = isdual(codomain(C)[1]) != isdual(domain(C)[1])
    return needs_twist ? twist(C, 1) : C
end

# ---------------------------------------------------------------------------
# gauge moves
# ---------------------------------------------------------------------------

"""
    move_center!(ψ, target; cache=nothing) -> ψ

Move the orthogonality center to `target` (Symbol or internal index) by QR
sweeps along the connecting path. The **only** sanctioned way to change the
gauge (§9.1). If an `EnvCache` is passed, entries invalidated by the gauge
change are dropped.
"""
move_center!(ψ::TTNS, target::Symbol; kwargs...) = move_center!(ψ, nodeindex(ψ.topo, target); kwargs...)
function move_center!(ψ::TTNS, target::Int; cache=nothing)
    path = path_between(ψ.topo, ψ.center, target)
    for i in 2:length(path)
        _move_center_edge!(ψ, path[i], cache)
    end
    return ψ
end

# one step of the center move, to a node `m` adjacent to the current center
function _move_center_edge!(ψ::TTNS, m::Int, cache)
    t = ψ.topo
    n = ψ.center
    A = ψ.tensors[n]
    if t.parent[n] == m
        # up-move: A = Q ∘ C across the (codomain ← parent) split
        Q, C = left_orth(A)                      # Q :: cod ← V_new, C :: V_new ← V_e
        ψ.tensors[n] = Q
        k = childslot(t, m, n)
        C = _pivotal_link(C)
        ψ.tensors[m] = absorb_on_leg(ψ.tensors[m], C, k)
    else
        # down-move into child slot k
        k = childslot(t, n, m)
        Q, C = orth_factor_leg(A, k)             # Q isometric away from slot k; C :: Y ← dual(V_e)
        ψ.tensors[n] = Q
        Ct = _pivotal_link(transpose(C))
        ψ.tensors[m] = ψ.tensors[m] * Ct              # Ct :: V_e ← dual(Y)
    end
    ψ.center = m
    cache === nothing || invalidate_edge!(cache, n, m)
    return ψ
end

"""
    update_tensor!(ψ, n, A; caches=()) -> ψ

Replace the tensor at node `n`. Must be used instead of writing the container
directly (§9.2): all caches that sandwich `ψ` are notified. `n` must be the
orthogonality center unless `gauge=false` is passed (in which case the caller
takes responsibility for the gauge invariant, e.g. inside a sweep kernel).
"""
function update_tensor!(ψ::TTNS, n::Int, A::AbstractTensorMap; caches=(), gauge::Bool=true)
    if gauge && n != ψ.center
        throw(ArgumentError("updating a non-center tensor breaks the gauge invariant; move_center! first or pass gauge=false"))
    end
    ψ.tensors[n] = A
    for c in caches
        invalidate_node!(c, n)
    end
    return ψ
end

"""
    apply_local(ψ, op, site; cache=nothing) -> TTNS

Return the unnormalized state `op_site * ψ` for a neutral local operator
`op :: P <- P`, or for an abelian charged local operator
`op :: P <- P ⊗ C` with one-dimensional charge leg `C`. The adjoint charged
shape `P ⊗ C <- P` is accepted and canonicalized to the same apply form. The
orthogonality center is moved through [`move_center!`](@ref) and tensor writes
go through [`update_tensor!`](@ref), so any supplied cache receives normal
invalidation events. Charged insertions shift the virtual spaces on the
insertion-to-root path and the root parent sector.
"""
function apply_local(ψ::TTNS, op::AbstractTensorMap, site::Symbol; cache=nothing)
    if numout(op) == 2 && numin(op) == 1
        return _apply_charged_local(ψ, _charged_adjoint_to_apply(op), site; cache)
    end
    numout(op) == 1 || throw(ArgumentError("apply_local expects one physical output leg"))
    numin(op) == 1 && return _apply_neutral_local(ψ, op, site; cache)
    numin(op) == 2 && return _apply_charged_local(ψ, op, site; cache)
    throw(ArgumentError("apply_local expects `P <- P`, charged `P <- P ⊗ C`, or adjoint charged `P ⊗ C <- P`"))
end

function _charged_adjoint_to_apply(op::AbstractTensorMap{T,S}) where {T<:Number,S<:ElementarySpace}
    Pout = codomain(op)[1]
    C = codomain(op)[2]
    Pin = domain(op)[1]
    Pout == Pin || throw(SpaceMismatch("adjoint charged local operator must use one physical space"))
    C isa ElementarySpace || throw(ArgumentError("adjoint charged local operator needs an elementary charge leg"))
    out = zeros(T, Pout ← Pin ⊗ C)

    unitq = one(sectortype(Pout))
    oldcodcoord = _basis_coord(S[Pout, C], unitq)
    olddomcoord = _basis_coord(S[Pin], unitq)
    newcodcoord = _basis_coord(S[Pout], unitq)
    newdomcoord = _basis_coord(S[Pin, C], unitq)

    for pout in 1:dim(Pout), c in 1:dim(C), pin in 1:dim(Pin)
        val = _tensor_entry(op, oldcodcoord, olddomcoord, (pout, c), (pin,), T)
        _add_tensor_entry!(out, newcodcoord, newdomcoord, (pout,), (pin, c), val)
    end
    return out
end

function _apply_neutral_local(ψ::TTNS, op::AbstractTensorMap, site::Symbol; cache=nothing)
    ϕ = copy(ψ)
    n = nodeindex(ϕ.topo, site)
    move_center!(ϕ, n; cache)
    p = physleg(ϕ, n)
    A = ϕ.tensors[n]
    codomain(op)[1] == space(A, p) && domain(op)[1] == space(A, p) ||
        throw(SpaceMismatch("apply_local: operator space does not match physical leg at $site"))
    out = absorb_on_leg(A, op, p)
    update_tensor!(ϕ, n, repartition(out, numout(A), numin(A)); caches=cache === nothing ? () : (cache,))
    return ϕ
end

function _apply_charged_local(ψ::TTNS{S,T}, op::AbstractTensorMap, site::Symbol;
                              cache=nothing) where {S<:ElementarySpace,T<:Number}
    spacetype(codomain(op)[1]) === ComplexSpace &&
        throw(ArgumentError("charged apply_local requires graded physical spaces"))
    C = domain(op)[2]
    q = _single_charge_sector(C)
    ϕ = copy(ψ)
    nsite = nodeindex(ϕ.topo, site)
    move_center!(ϕ, nsite; cache)
    p = physleg(ϕ, nsite)
    A = ϕ.tensors[nsite]
    codomain(op)[1] == space(A, p) && domain(op)[1] == space(A, p) ||
        throw(SpaceMismatch("apply_local: charged operator physical space does not match $site"))

    path = path_to_root(ϕ.topo, nsite)
    shifted = Dict{Int,S}()
    for n in path
        V = domain(ϕ.tensors[n])[1]
        shifted[n] = _shift_space(V, q)
    end

    caches = cache === nothing ? () : (cache,)
    for (i, n) in enumerate(path)
        Aold = ϕ.tensors[n]
        Anew = if n == nsite
            _charged_site_tensor(Aold, op, p, shifted[n], q)
        else
            slot = childslot(ϕ.topo, n, path[i - 1])
            _shift_path_tensor(Aold, slot, shifted[path[i - 1]], shifted[n], q)
        end
        update_tensor!(ϕ, n, Anew; caches, gauge=(n == nsite))
    end
    return ϕ
end

function _single_charge_sector(C::ElementarySpace)
    dim(C) == 1 || throw(ArgumentError("charged apply_local charge leg must be one-dimensional"))
    qs = collect(sectors(C))
    length(qs) == 1 || throw(ArgumentError("charged apply_local charge leg must carry one sector"))
    return only(qs)
end

function _shift_space(V::S, q) where {S<:ElementarySpace}
    Q = sectortype(V)
    q isa Q || throw(ArgumentError("charge sector type $(typeof(q)) does not match virtual sector type $Q"))
    dims = Pair{Q,Int}[]
    for s in sectors(V)
        push!(dims, _fuse_sector(s, q) => dim(V, s))
    end
    return Vect[Q](dims...)
end

function _fuse_sector(a, b)
    fused = a ⊗ b
    length(fused) == 1 ||
        throw(ArgumentError("charged apply_local currently supports abelian one-channel fusion only"))
    return only(fused)
end

function _charged_site_tensor(A::AbstractTensorMap{T,S}, op::AbstractTensorMap,
                              pleg::Int, newdom::S, q) where {T<:Number,S<:ElementarySpace}
    oldcods = _codomain_legs(A)
    olddom = domain(A)[1]
    newcods = copy(oldcods)
    newcods[pleg] = codomain(op)[1]
    Anew = zeros(T, _product_space(newcods) ← newdom)

    unitq = one(sectortype(olddom))
    oldcodcoord = _basis_coord(oldcods, unitq)
    olddomcoord = _basis_coord(S[olddom], unitq)
    newcodcoord = _basis_coord(newcods, unitq)
    newdomcoord = _basis_coord(S[newdom], unitq)
    opcodcoord = _basis_coord(S[codomain(op)[1]], unitq)
    opdomcoord = _basis_coord(S[domain(op)[1], domain(op)[2]], unitq)
    dommap = _shift_index_map(olddom, newdom, q)

    oldcod_ranges = ntuple(i -> 1:dim(oldcods[i]), length(oldcods))
    for I in CartesianIndices(oldcod_ranges), d in 1:dim(olddom)
        oldidx = Tuple(I)
        Aval = _tensor_entry(A, oldcodcoord, olddomcoord, oldidx, (d,), T)
        iszero(Aval) && continue
        pin = oldidx[pleg]
        for pout in 1:dim(newcods[pleg])
            Oval = _tensor_entry(op, opcodcoord, opdomcoord, (pout,), (pin, 1), T)
            iszero(Oval) && continue
            newidx = Base.setindex(oldidx, pout, pleg)
            _add_tensor_entry!(Anew, newcodcoord, newdomcoord, newidx, (dommap[d],), Oval * Aval)
        end
    end
    return Anew
end

function _shift_path_tensor(A::AbstractTensorMap{T,S}, slot::Int, newchild::S,
                            newdom::S, q) where {T<:Number,S<:ElementarySpace}
    oldcods = _codomain_legs(A)
    oldchild = oldcods[slot]
    olddom = domain(A)[1]
    newcods = copy(oldcods)
    newcods[slot] = newchild
    Anew = zeros(T, _product_space(newcods) ← newdom)

    unitq = one(sectortype(olddom))
    oldcodcoord = _basis_coord(oldcods, unitq)
    olddomcoord = _basis_coord(S[olddom], unitq)
    newcodcoord = _basis_coord(newcods, unitq)
    newdomcoord = _basis_coord(S[newdom], unitq)
    childmap = _shift_index_map(oldchild, newchild, q)
    dommap = _shift_index_map(olddom, newdom, q)

    oldcod_ranges = ntuple(i -> 1:dim(oldcods[i]), length(oldcods))
    for I in CartesianIndices(oldcod_ranges), d in 1:dim(olddom)
        oldidx = Tuple(I)
        Aval = _tensor_entry(A, oldcodcoord, olddomcoord, oldidx, (d,), T)
        iszero(Aval) && continue
        newidx = Base.setindex(oldidx, childmap[oldidx[slot]], slot)
        _add_tensor_entry!(Anew, newcodcoord, newdomcoord, newidx, (dommap[d],), Aval)
    end
    return Anew
end

_codomain_legs(A::AbstractTensorMap{T,S}) where {T,S<:ElementarySpace} =
    S[codomain(A)[i] for i in 1:numout(A)]

_product_space(legs::Vector{S}) where {S<:ElementarySpace} =
    isempty(legs) ? oneunit(S) : reduce(⊗, legs)

function _shift_index_map(Vold::ElementarySpace, Vnew::ElementarySpace, q)
    oldbasis = _flat_basis(Vold)
    newbasis = _flat_basis(Vnew)
    newindex = Dict{Tuple{typeof(q),Int},Int}()
    for (i, key) in pairs(newbasis)
        newindex[key] = i
    end
    return [newindex[(_fuse_sector(s, q), i)] for (s, i) in oldbasis]
end

function _flat_basis(V::ElementarySpace)
    Q = sectortype(V)
    out = Tuple{Q,Int}[]
    for s in sectors(V), i in 1:dim(V, s)
        push!(out, (s, i))
    end
    return out
end

function _basis_coord(legs::Vector{S}, unitq) where {S<:ElementarySpace}
    coord = Dict{Tuple,Tuple{typeof(unitq),Int}}()
    if isempty(legs)
        coord[()] = (unitq, 1)
        return coord
    end
    # TensorKit block-row layout: abelian fusion trees (uncoupled sector
    # tuples) iterate first-leg-fastest, each owning one contiguous row range
    # with degeneracy indices column-major inside it. Sweeping basis positions
    # directly coincides with this only for one-dimensional sectors; with
    # sector degeneracy it interleaves trees and permutes degenerate states.
    K = length(legs)
    legsectors = [collect(sectors(V)) for V in legs]
    legoffsets = map(legs) do V
        offsets = Dict{typeof(unitq),Int}()
        offset = 0
        for s in sectors(V)
            offsets[s] = offset
            offset += dim(V, s)
        end
        offsets
    end
    rows = Dict{typeof(unitq),Int}()
    for T in CartesianIndices(Tuple(length.(legsectors)))
        secs = ntuple(j -> legsectors[j][T[j]], K)
        q = unitq
        for s in secs
            q = _fuse_sector(q, s)
        end
        for D in CartesianIndices(ntuple(j -> dim(legs[j], secs[j]), K))
            idx = ntuple(j -> legoffsets[j][secs[j]] + D[j], K)
            row = get(rows, q, 0) + 1
            rows[q] = row
            coord[idx] = (q, row)
        end
    end
    return coord
end

function _tensor_entry(A::AbstractTensorMap, codcoord, domcoord,
                       codidx::Tuple, domidx::Tuple, ::Type{T}) where {T<:Number}
    cq, row = codcoord[codidx]
    dq, col = domcoord[domidx]
    cq == dq || return zero(T)
    for (q, b) in blocks(A)
        q == cq && return b[row, col]
    end
    return zero(T)
end

function _add_tensor_entry!(A::AbstractTensorMap, codcoord, domcoord,
                            codidx::Tuple, domidx::Tuple, val)
    iszero(val) && return A
    cq, row = codcoord[codidx]
    dq, col = domcoord[domidx]
    cq == dq || return A
    for (q, b) in blocks(A)
        if q == cq
            b[row, col] += val
            return A
        end
    end
    throw(SpaceMismatch("internal charged apply_local block $cq is absent"))
end

# Cache-notification hooks; Contractions owns the real implementations for its
# EnvCache. Defined here as generic no-op fallbacks so Networks stays below
# Contractions in the layering.
function invalidate_node! end
function invalidate_edge! end
