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
        ψ.tensors[m] = absorb_on_leg(ψ.tensors[m], C, k)
    else
        # down-move into child slot k
        k = childslot(t, n, m)
        Q, C = orth_factor_leg(A, k)             # Q isometric away from slot k; C :: Y ← dual(V_e)
        ψ.tensors[n] = Q
        ψ.tensors[m] = ψ.tensors[m] * transpose(C)   # transpose(C) :: V_e ← dual(Y)
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

# Cache-notification hooks; Contractions owns the real implementations for its
# EnvCache. Defined here as generic no-op fallbacks so Networks stays below
# Contractions in the layering.
function invalidate_node! end
function invalidate_edge! end
