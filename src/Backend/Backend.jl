"""
L0 — TensorKit adapter layer (architecture §2).

Upper layers never `import TensorKit` directly (global constraint §9.13): all
spaces, tensors, factorizations and contraction primitives are consumed through
this module, so TensorKit API churn is contained here (this file is written
against TensorKit v0.17 / MatrixAlgebraKit-style factorizations).

# Arrow / dual convention (pinned day one, §2 — guarded by asserts in Networks)

* Every tree edge points towards the root.
* TTNS node tensor:  `A :: (⊗ children spaces) ⊗ P  ←  V_parent`
  (codomain = child legs in topology order, then the physical leg (if any);
  domain = the single parent leg; the root's parent leg is `oneunit(S)` or a
  global-charge space).
* TTNO node tensor:  `O :: (⊗ children spaces) ⊗ P  ←  P ⊗ U_parent`.
* Edge space `V_e` for edge `(child, parent)`: the child's domain is `V_e` and
  the parent's child-slot codomain factor is `V_e`, so contracting an edge is
  map composition (the whole TTNS is one nested composition of node maps).
* Ancilla legs (purification, PP B-sites) are physical legs carrying dual
  representations — nothing in this layer distinguishes them.

Fermionic exchange is handled natively by TensorKit's graded braiding
(`FermionParity`); trees are planar so **no Jordan–Wigner strings** anywhere.
"""
module Backend

using TensorKit
using TensorOperations: TensorOperations, @tensor, ncon

# ---------------------------------------------------------------------------
# re-exports: the vocabulary upper layers are allowed to use
# ---------------------------------------------------------------------------
# spaces & sectors
export ℂ, ComplexSpace, GradedSpace, ElementarySpace, ProductSpace, Vect,
    U1Space, Z2Space, U1Irrep, ZNIrrep, SU2Irrep, FermionParity, Trivial,
    ⊠, ⊗, ←, dual, oneunit, fuse, dim, space, sectortype, spacetype, sectors,
    isdual
# tensors
export AbstractTensorMap, TensorMap, DiagonalTensorMap, id, isometry, unitary,
    permute, repartition, flip, catdomain, catcodomain, numind, numout, numin,
    codomain, domain, blocks, block, norm, dot, tr, scalartype
# factorizations (MatrixAlgebraKit style)
export left_orth, right_orth, left_null, qr_compact, svd_compact, svd_trunc,
    svd_vals, truncrank, trunctol, truncerror, notrunc
# contraction primitives
export @tensor, ncon
# GRAFT-defined
export FermionSector, AbelianSector, TruncationScheme, truncspec, split_svd,
    absorb_on_leg, orth_factor_leg, trivialspace, ones_tensor

# ---------------------------------------------------------------------------
# standard impurity sector types (§2)
# ---------------------------------------------------------------------------
"""Full symmetry for production impurity runs: fℤ₂ ⊠ U(1)_N ⊠ SU(2)_S. (M3+)"""
const FermionSector = FermionParity ⊠ U1Irrep ⊠ SU2Irrep

"""Abelian downgrade: fℤ₂ ⊠ U(1)_N ⊠ U(1)_Sz. First-milestone target."""
const AbelianSector = FermionParity ⊠ U1Irrep ⊠ U1Irrep

"""The trivial (unit) space of the same kind as `V` — used as the root's parent leg."""
trivialspace(V::ElementarySpace) = oneunit(V)
trivialspace(::Type{S}) where {S<:ElementarySpace} = oneunit(S)

"""
    ones_tensor(T, cod::ProductSpace) -> Tensor

Rank-N tensor (empty domain) with every fusion-tree coefficient of the trivial
coupled sector set to one. Used as the root cap closing ket/op/bra unit or
global-charge legs. Exact for abelian sectors (dim-1 trivial blocks).
TODO(M3): audit normalization conventions for non-abelian sectors (SU(2) caps
carry √dim factors in the fusion-tree basis).
"""
function ones_tensor(::Type{T}, cod::ProductSpace) where {T<:Number}
    t = zeros(T, cod)
    for (_, b) in blocks(t)
        fill!(b, one(T))
    end
    return t
end

# ---------------------------------------------------------------------------
# TruncationScheme — the single truncation entry point (§9.5)
# ---------------------------------------------------------------------------
"""
    TruncationScheme(; maxdim, atol, rtol, discarded_weight)

The only sanctioned way to truncate anywhere in GRAFT (global constraint §9.5):
sector-resolved singular value spectra all pass through here, so policy (incl.
future LBO semantics on PP bonds) is defined in one place.

PyTreeNet `util/tensor_splitting.SVDParameters` mapping:
`max_bond_dim → maxdim`, `rel_tol → rtol` (relative to the largest singular
value), `total_tol → atol`. `discarded_weight` bounds the relative 2-norm error
of the discarded spectrum.
"""
Base.@kwdef struct TruncationScheme
    maxdim::Int = typemax(Int)
    atol::Float64 = 0.0
    rtol::Float64 = 0.0
    discarded_weight::Float64 = 0.0
end

const NO_TRUNCATION = TruncationScheme()

"""Convert a `TruncationScheme` into a MatrixAlgebraKit truncation strategy."""
function truncspec(ts::TruncationScheme)
    strat = nothing
    combine(a, b) = a === nothing ? b : a & b
    ts.maxdim < typemax(Int) && (strat = combine(strat, truncrank(ts.maxdim)))
    (ts.atol > 0 || ts.rtol > 0) && (strat = combine(strat, trunctol(; atol=ts.atol, rtol=ts.rtol)))
    ts.discarded_weight > 0 && (strat = combine(strat, truncerror(; rtol=ts.discarded_weight)))
    return strat === nothing ? notrunc() : strat
end

"""
    split_svd(t, ts::TruncationScheme) -> (U, S, Vᴴ)

Truncated SVD across the codomain|domain split of `t` (permute first to choose
the split). `t ≈ U ∘ S ∘ Vᴴ`.
"""
split_svd(t::AbstractTensorMap, ts::TruncationScheme=NO_TRUNCATION) =
    svd_trunc(t; trunc=truncspec(ts))

# ---------------------------------------------------------------------------
# rank-generic leg utilities
#
# These two primitives make gauge moves (and everything else that touches a
# single leg of an arbitrary-rank node tensor) expressible without @tensor
# rank-specialization: permute the target leg to the domain, act, permute back.
# ---------------------------------------------------------------------------

# permutation that restores original index order after (others..., k) reshuffle
function _restore_perm(N::Int, No::Int, k::Int)
    pos(j) = j < k ? j : (j == k ? N : j - 1)
    p1 = ntuple(j -> pos(j), No)
    p2 = ntuple(j -> pos(No + j), N - No)
    return (p1, p2)
end

_others(N::Int, k::Int) = ntuple(j -> j < k ? j : j + 1, N - 1)

"""
    absorb_on_leg(A, C, k) -> A′

Contract `C :: V_new ← V_old` into codomain leg `k` of `A` (which must have
space `V_old`), i.e. `A′ = (id ⊗ … ⊗ C ⊗ … ⊗ id) ∘ A` — but at
permute+GEMM cost instead of materializing the product operator.
Leg `k` of the result has space `V_new`.
"""
function absorb_on_leg(A::AbstractTensorMap, C::AbstractTensorMap, k::Int)
    N, No = numind(A), numout(A)
    1 <= k <= No || throw(ArgumentError("leg $k is not a codomain leg (numout=$No)"))
    domain(C)[1] == space(A, k) ||
        throw(SpaceMismatch("absorb_on_leg: domain of C ($(domain(C)[1])) ≠ leg $k of A ($(space(A, k)))"))
    t = permute(A, (_others(N, k), (k,)))          # :: others ← dual(V_old)
    t = t * transpose(C)                           # :: others ← dual(V_new)
    return permute(t, _restore_perm(N, No, k))
end

"""
    orth_factor_leg(A, k) -> (Q, C)

Orthogonal factorization of `A` "against" codomain leg `k`:
`A = absorb_on_leg(Q, transpose(C)?, k)`-wise, concretely
`permute(A, (others, (k,))) = Q̃ ∘ C` with `Q̃` isometric, `Q` = `Q̃` permuted
back to `A`'s index layout (leg `k` becomes the new bond, space `dual(Y)`),
and `C :: Y ← dual(V_old)`.

Used to move the orthogonality center *into* the direction of leg `k`:
the caller absorbs `transpose(C) :: V_old ← dual(Y)` into the neighbour's
parent leg.
"""
function orth_factor_leg(A::AbstractTensorMap, k::Int)
    N, No = numind(A), numout(A)
    1 <= k <= No || throw(ArgumentError("leg $k is not a codomain leg (numout=$No)"))
    t = permute(A, (_others(N, k), (k,)))          # :: others ← dual(V_old)
    Q̃, C = left_orth(t)                            # Q̃ :: others ← Y isometric, C :: Y ← dual(V_old)
    Q = permute(Q̃, _restore_perm(N, No, k))        # leg k has space dual(Y)
    return Q, C
end

end # module Backend
