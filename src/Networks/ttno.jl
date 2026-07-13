# TTNO — tree tensor network operator (PyTreeNet: ttno/ttno_class.py), leg
# convention (Backend docstring):
#
#   O[n] :: (⊗_{c} U_c) ⊗ P_n  ←  P_n ⊗ U_parent(n)      (physical node)
#   O[n] :: (⊗_{c} U_c)        ←  U_parent(n)            (branching node)
#
# The root's parent virtual leg is `oneunit(S)`. Hermiticity is a *trait*, not
# an assumption (§9.8): nothing here requires O to be self-adjoint, keeping the
# door open for Lindbladians / TTNDO evolution.

"""
    TTNO{S,T}

Tree tensor network operator on the same `TreeTopology` as the states it acts
on. `hasphys` must match the target TTNS node-by-node.
"""
mutable struct TTNO{S<:ElementarySpace,T<:Number}
    const topo::TreeTopology
    const tensors::Vector{AbstractTensorMap{T,S}}
    const hasphys::BitVector
    ishermitian::Bool   # trait (§9.8): wrong `false` costs performance, never correctness

    function TTNO(topo::TreeTopology, tensors::Vector{<:AbstractTensorMap};
                  ishermitian::Bool=false)
        isempty(tensors) && throw(ArgumentError("empty TTNO"))
        length(tensors) == nnodes(topo) || throw(ArgumentError("need one tensor per node"))
        S = spacetype(tensors[1])
        T = scalartype(tensors[1])
        hasphys = BitVector(numin(tensors[i]) == 2 for i in 1:nnodes(topo))
        O = new{S,T}(topo, collect(AbstractTensorMap{T,S}, tensors), hasphys, ishermitian)
        check_arrows(O)
        return O
    end
end

topology(O::TTNO) = O.topo
Base.eltype(::TTNO{S,T}) where {S,T} = T
Backend.spacetype(::TTNO{S}) where {S} = S
Trees.nnodes(O::TTNO) = nnodes(O.topo)
hasphys(O::TTNO, n::Int) = O.hasphys[n]

Base.getindex(O::TTNO, n::Int) = O.tensors[n]
Base.getindex(O::TTNO, s::Symbol) = O.tensors[nodeindex(O.topo, s)]

physleg(O::TTNO, n::Int) = nchildren(O.topo, n) + 1     # codomain (out) physical leg
virtualspace(O::TTNO, child::Int) = domain(O.tensors[child])[numin(O.tensors[child])]

function check_arrows(O::TTNO)
    t = O.topo
    for n in 1:nnodes(t)
        A = O.tensors[n]
        numin(A) in (1, 2) || throw(SpaceMismatch("TTNO node $(nodeid(t, n)): domain must be (P ⊗ U_parent) or (U_parent)"))
        numout(A) == nchildren(t, n) + (numin(A) == 2 ? 1 : 0) ||
            throw(SpaceMismatch("TTNO node $(nodeid(t, n)): codomain legs ≠ children (+ physical)"))
        if numin(A) == 2
            space(A, nchildren(t, n) + 1) == dual(space(A, nchildren(t, n) + 2)) ||
                throw(SpaceMismatch("TTNO node $(nodeid(t, n)): physical in/out spaces differ"))
        end
        for (k, c) in enumerate(t.children[n])
            space(A, k) == domain(O.tensors[c])[numin(O.tensors[c])] ||
                throw(SpaceMismatch("TTNO edge $(nodeid(t, c)) → $(nodeid(t, n)): virtual space mismatch"))
        end
    end
    return true
end

"""
    ishermitian(O::TTNO) -> Bool

Hermiticity *trait* (§9.8). Kernels use it to pick Lanczos vs Arnoldi paths;
a wrong `true` is a correctness bug of the caller, a wrong `false` only costs
performance (defaults are conservative).
"""
ishermitian(O::TTNO) = O.ishermitian

# ---------------------------------------------------------------------------
# `compress!` is implemented in ttno_compression.jl: exact/numerical-rank
# deparallelization, QR canonicalisation, and sector-resolved SVD. Physical
# approximate TTNO truncation remains a separate future opt-in surface.
# TODO: time-dependent TTNO wrapper (PyTreeNet time_dep_ttno counterpart) for
#       driven fields / quenches.
# ---------------------------------------------------------------------------

"""
    TTNDO

Vectorized density-operator tree (PyTreeNet: ttns/ttndo.py). Retained as a
first-class type slot for the quasi-Lindblad / AMEA route (§5, §11.3): the
whole stack below L5 never assumes hermiticity, so evolving a TTNDO with a
non-hermitian Liouvillian TTNO needs no new evolution code.

TODO(post-M2): physical/ancilla leg pairing conventions + trace/expectation
implementations. Nothing implements this yet.
"""
struct TTNDO{S<:ElementarySpace,T<:Number}
    state::TTNS{S,T}    # vectorized representation; pairing convention TODO
end
