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

"""
    truncate_sweep!(ψ::TTNS, ts::TruncationScheme; cache=nothing) -> Float64

Truncate every tree edge of `ψ` exactly once: walking the tree from the
root, each edge is truncated by a `TruncationScheme`-controlled SVD when the
orthogonality center first crosses it downward, and recanonicalized exactly
on the way back up. The state must be canonical at the root on entry; the
center is restored to the root on exit.

Returns an upper bound on the introduced 2-norm error: the sum of the
discarded 2-norms of all edge truncations. Each truncation is performed at
the orthogonality center, so the bound follows from the triangle inequality
over the sequence of locally optimal truncations.
"""
function truncate_sweep!(ψ::TTNS, ts::TruncationScheme; cache=nothing)
    t = ψ.topo
    ψ.center == t.root ||
        throw(ArgumentError("truncate_sweep! requires the center at the root"))
    total = 0.0
    descend!(n::Int) = for m in t.children[n]
        total += _move_center_edge_trunc!(ψ, m, ts, cache)
        descend!(m)
        _move_center_edge!(ψ, n, cache)
    end
    descend!(t.root)
    return total
end

# truncated down-move: like the down branch of `_move_center_edge!` but the
# factorization discards weight under `ts`; returns the discarded 2-norm.
function _move_center_edge_trunc!(ψ::TTNS, m::Int, ts::TruncationScheme,
                                  cache)
    t = ψ.topo
    n = ψ.center
    t.parent[m] == n ||
        throw(ArgumentError("truncated center move must descend to a child"))
    k = childslot(t, n, m)
    Q, C, discarded = svd_factor_leg_with_error(ψ.tensors[n], k, ts)
    ψ.tensors[n] = Q
    Ct = _pivotal_link(transpose(C))
    ψ.tensors[m] = ψ.tensors[m] * Ct
    ψ.center = m
    cache === nothing || invalidate_edge!(cache, n, m)
    return discarded
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
    Cadjoint = codomain(op)[2]
    C = dual(Cadjoint)
    Pin = domain(op)[1]
    Pout == Pin || throw(SpaceMismatch("adjoint charged local operator must use one physical space"))
    Cadjoint isa ElementarySpace ||
        throw(ArgumentError("adjoint charged local operator needs an elementary charge leg"))
    out = zeros(T, Pout ← Pin ⊗ C)

    unitq = one(sectortype(Pout))
    oldcodcoord = _basis_coord(S[Pout, Cadjoint], unitq)
    olddomcoord = _basis_coord(S[Pin], unitq)
    newcodcoord = _basis_coord(S[Pout], unitq)
    newdomcoord = _basis_coord(S[Pin, C], unitq)

    for pout in 1:dim(Pout), c in 1:dim(Cadjoint), pin in 1:dim(Pin)
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
    _apply_canonical_string!(ϕ, nsite, q; caches)
    return ϕ
end

# Charge-wire transport (§9.3 leg convention `(children… ⊗ P) ← V_parent`)
# ---------------------------------------------------------------------------
# A charged insertion runs one charge wire from the physical leg of the
# insertion node to the root's domain leg. Two independent pieces of
# categorical data fix its phase, and neither may be inferred from the depth or
# the parity of the path index:
#
#  (T1) *Tree-native braid word.* Through an ancestor the wire arrives on the
#       child slot `k` carrying it and leaves through the single domain leg, so
#       it braids past exactly the codomain legs `k+1 … numout` — the later
#       sibling bonds and the node's own physical leg. Through the insertion
#       node the wire is born on the physical leg, which is last in the
#       codomain, so it crosses nothing. On top of that word each traversed
#       edge contributes a pivotal bend, and only when the edge is stored dual
#       (see `_shift_path_tensor`) — the flat sector relabelling is not the
#       identity transport. The word is evaluated per basis element from the
#       *stored* leg sectors and orientations, so the result is invariant under
#       root choice, gauge, center position, and which primal/dual orientation
#       an evolver happened to leave on a bond.
#
#  (T2) *Native-to-canonical reordering.* (T1) orders the physical wires the
#       way the tree contracts them (a node's own physical wire, then its child
#       branches in reverse codomain order). Every dense oracle and the TTNO
#       lowerer instead pin the canonical internal-node order — see
#       `TTNOBuild._canonical_crossing_frame`, which performs the same
#       conversion for operator factors. `_apply_canonical_string!` closes the
#       gap with the explicit Jordan–Wigner string on the symmetric difference
#       of the two predecessor sets.
#
# TODO(M3, §4a): non-abelian charges need an oriented fusion route instead of
# these scalar R-symbols; `_fuse_sector` and `_charge_braid` fail closed today.

"""
Scalar braid phase of a charge wire carrying sector `q` crossing a leg that
carries sector `s`. Abelian one-channel fusion only — a non-scalar R-symbol
(non-abelian braiding) is rejected rather than silently truncated.
"""
function _charge_braid(q, s)
    r = Rsymbol(q, s, _fuse_sector(q, s))
    r isa Number || throw(ArgumentError(
        "charged apply_local braid requires scalar abelian R-symbols; got $(typeof(r))"))
    return r
end

"""Sector carried by each flat basis index of `V`, in `_flat_basis` order."""
_flat_sectors(V::ElementarySpace) = [s for (s, _) in _flat_basis(V)]

"""
Physical nodes on which the tree-native charge-wire order and the canonical
internal-node order disagree about crossing the insertion at `nsite`.

The native order visits a node's own physical wire before its child branches
and walks the children in reverse codomain order (the same walk as
`TTNOBuild._physical_node_orders`); the canonical order is increasing internal
node index. A wire crosses everything preceding it, so the correction set is
the symmetric difference of the two predecessor sets.
"""
function _canonical_string_nodes(ψ::TTNS, nsite::Int)
    t = ψ.topo
    native = Int[]
    function visit(n::Int)
        hasphys(ψ, n) && push!(native, n)
        for c in Iterators.reverse(t.children[n])
            visit(c)
        end
        return nothing
    end
    visit(t.root)
    canonical = [n for n in 1:nnodes(t) if hasphys(ψ, n)]
    inative = findfirst(==(nsite), native)
    icanon = findfirst(==(nsite), canonical)
    (inative === nothing || icanon === nothing) &&
        throw(ArgumentError("charged apply_local needs a physical leg at the insertion node"))
    return sort!(collect(symdiff(Set(native[1:inative - 1]), Set(canonical[1:icanon - 1]))))
end

# The string is diagonal and unitary on each physical leg, so it commutes with
# the isometry conditions and needs no gauge move of its own (§9.1/§9.2).
function _apply_canonical_string!(ψ::TTNS, nsite::Int, q; caches=())
    θ = twist(q)
    θ isa Number || throw(ArgumentError(
        "charged apply_local string requires scalar abelian twists; got $(typeof(θ))"))
    θ == 1 && return ψ                       # bosonic charge: no string at all
    θ == -1 || throw(ArgumentError(
        "charged apply_local currently supports fℤ₂×abelian charges only; got twist $θ for $q"))
    for n in _canonical_string_nodes(ψ, nsite)
        # `twist` on the physical leg is exactly R(q, ·) for a fermionic-odd q
        # over an fℤ₂×abelian grading, and is orientation-aware for dual legs.
        update_tensor!(ψ, n, twist(ψ.tensors[n], physleg(ψ, n)); caches, gauge=false)
    end
    return ψ
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
    # `sectors`/`dim` report a dual space through its dual sectors, so the
    # shifted labels are bent back before a dual space is rebuilt: the
    # primal/dual orientation of a bond is part of the representation and a
    # charged insertion must not silently flip it (§9.3).
    isdual(V) || return Vect[Q](dims...)
    return dual(Vect[Q]((dual(s) => d for (s, d) in dims)...))
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

    # (T1) braid word for this node. Relabelling the sector of leg `slot`
    # together with the domain bends the charge wire across that bond. When the
    # bond is stored against the wire — `isdual` on the traversed edge — that
    # bend implicitly crosses the wire with both the pre- and post-shift label
    # of the leg, and `bend` undoes exactly that pair (it collapses to the
    # ribbon twist θ_q of the charge wire for every fℤ₂×abelian grading). A
    # primal bond carries no such twist. This is the same seam convention as
    # `_pivotal_link` for QR link factors, and it is what makes the transport
    # covariant under the orientation an evolver happens to leave on a bond:
    # `move_center!` only ever hands a downward path dual bonds, while a
    # two-site split leaves them primal. What remains is the wire's real
    # crossing word: the codomain legs after `slot`.
    codsectors = [_flat_sectors(V) for V in oldcods]
    bend = [isdual(oldchild) ?
            inv(_charge_braid(q, s)) * inv(_charge_braid(q, _fuse_sector(s, q))) :
            one(_charge_braid(q, s))
            for s in codsectors[slot]]
    crossed = (slot + 1):length(oldcods)

    oldcod_ranges = ntuple(i -> 1:dim(oldcods[i]), length(oldcods))
    for I in CartesianIndices(oldcod_ranges), d in 1:dim(olddom)
        oldidx = Tuple(I)
        Aval = _tensor_entry(A, oldcodcoord, olddomcoord, oldidx, (d,), T)
        iszero(Aval) && continue
        phase = bend[oldidx[slot]]
        for j in crossed
            phase *= _charge_braid(q, codsectors[j][oldidx[j]])
        end
        newidx = Base.setindex(oldidx, childmap[oldidx[slot]], slot)
        _add_tensor_entry!(Anew, newcodcoord, newdomcoord, newidx, (dommap[d],),
                           phase * Aval)
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
