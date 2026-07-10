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
export @tensor, ncon, contract_pair, pair_cost, space_signature,
    sector_cost_supported, sector_cost_nontrivial, sector_block_peak,
    tensor_scalar
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
# Compiled-contraction primitives
# ---------------------------------------------------------------------------

"""
    contract_pair(A, pA, conjA, B, pB, conjB, pAB) -> TensorMap

Execute one pre-planned binary tensor contraction. `pA`, `pB`, and `pAB` are
TensorOperations' expert-mode index partitions, so upper layers can compile
leg bookkeeping once and keep the Krylov hot path free of ncon label parsing.

`Backend` deliberately accepts only structural index data here: it must not
depend on `Contractions.Planning.PairStep` (L0 stays below L2).
"""
function contract_pair(A::AbstractTensorMap, pA::Tuple, conjA::Bool,
                       B::AbstractTensorMap, pB::Tuple, conjB::Bool,
                       pAB::Tuple)
    return TensorOperations.tensorcontract(A, pA, conjA, B, pB, conjB, pAB)
end

"""
    tensor_scalar(t::AbstractTensorMap) -> Number

L0 wrapper for TensorOperations' scalar conversion. Planned scalar networks
must use this only after their final rank-zero TensorMap is produced, matching
`ncon`'s public return convention without teaching upper layers TensorKit's
scalar API.
"""
tensor_scalar(t::AbstractTensorMap) = TensorOperations.tensorscalar(t)

"""
    space_signature(x) -> UInt

Hash the immutable codomain/domain structure of a TensorMap or TensorMapSpace.
Compiled contraction plans are shape-only artifacts: tensor values never enter
their cache identity.
"""
space_signature(x) = hash((codomain(x), domain(x)))

"""
    sector_cost_supported(W::TensorMapSpace) -> Bool

Whether the Phase-3 block-GEMM cost model is exact and applicable to the
sector type of `W`.  It is deliberately restricted to `UniqueFusion` **and
`SymmetricBraiding`** (trivial, U(1), Z₂, fermion parity, and their compatible
abelian products): in that regime every sector has quantum dimension one,
permutations are well-defined, and TensorKit's binary `mul!` is precisely a
collection of independent dense GEMMs.  Non-abelian spaces use the Phase-2
dense model.  Planar-only and anyonic spaces require a future planar
TensorOperations executor path: this predicate deliberately does not claim
that the current regular `contract_pair` wrapper can execute them.  The
Phase-3 v1 permutation coefficient is consequently fixed to zero rather than
silently pretending to price anyonic braids.
"""
sector_cost_supported(W::TensorMapSpace) =
    TensorKit.FusionStyle(TensorKit.sectortype(W)) isa TensorKit.UniqueFusion &&
    TensorKit.BraidingStyle(TensorKit.sectortype(W)) isa TensorKit.SymmetricBraiding

"""
    sector_cost_nontrivial(W::TensorMapSpace) -> Bool

Whether `W` has a Phase-3-eligible symmetry *and* an actual nontrivial sector
split.  A U(1)/Z₂ *type* alone is not enough: a charge-constrained map can
still contain exactly one dense block with the same payload as its ordinary
codomain-by-domain matrix.  Planning deliberately avoids structural HomSpace
composition in that dense-equivalent case.  This keeps pure dimension-only
fixtures and single-sector charge maps useful while avoiding planning work on
a cost model that cannot change their order.
"""
function sector_cost_nontrivial(W::TensorMapSpace)
    sector_cost_supported(W) || return false
    # `dim(W)` is TensorKit's stored HomSpace payload (Σ block rows × block
    # columns), while this product is the sector-blind dense payload.  Either
    # a payload reduction or multiple actual blocks can change Phase-3's cost.
    dense_payload = big(dim(codomain(W))) * big(dim(domain(W)))
    nblocks = count(q -> TensorKit.hasblock(W, q), TensorKit.blocksectors(W))
    return nblocks > 1 || big(dim(W)) != dense_payload
end

"""
    sector_block_peak(W::TensorMapSpace) -> Float64

Largest stored matrix block of a TensorKit HomSpace.  This is a structural
query only; no TensorMap data is allocated.  Phase 3 uses it to include input
maps in the per-plan largest-block diagnostic.
"""
function sector_block_peak(W::TensorMapSpace)
    largest = 0.0
    for q in TensorKit.blocksectors(W)
        largest = max(largest,
                      Float64(TensorKit.blockdim(codomain(W), q)) *
                      Float64(TensorKit.blockdim(domain(W), q)))
    end
    return largest
end

function _conjugated_structure(W::TensorMapSpace, p::Tuple, conjW::Bool)
    conjW || return W, p
    # Use TensorKit's index remapping rather than reproducing its flat-leg
    # convention by hand: an input flat leg is represented as dual(domain),
    # and adjoint swaps the codomain/domain index ranges.
    return adjoint(W), TensorKit.adjointtensorindices(W, p)
end

"""
    pair_cost(A::TensorMapSpace, pA, conjA, B::TensorMapSpace, pB, conjB, pAB)

Return exact, allocation-free block metadata for one TensorKit binary
contraction.  `output` is the exact output HomSpace after the requested final
partition.  For unique-fusion sector types, `sector_flops` is the sum of the
actual per-sector GEMM costs `Σ_q 2m_qk_qn_q`; `output_elements` is the exact
symmetry-reduced stored payload of the output; and `largest_block_elements`
is the largest GEMM result block.  `output_largest_block_elements` records the
largest block after `pAB`'s final repartition, while `peak_block_elements` is
their maximum and is the safe live-block diagnostic for Planning.  The three
`*_stored_elements` fields expose the known full-payload transform layouts for
Planning's live-memory model: the two matrix-product operands and the product
before final output repartition.

The block loop intentionally follows the *matrix-product* structure
`compose(permute(A, pA), permute(B, pB))`, not the final repartitioned output:
the latter can fuse those GEMM sectors differently and is not a record of the
work TensorKit executed.
"""
function pair_cost(A::TensorMapSpace, pA::Tuple, conjA::Bool,
                   B::TensorMapSpace, pB::Tuple, conjB::Bool,
                   pAB::Tuple)
    A′, pA′ = _conjugated_structure(A, pA, conjA)
    B′, pB′ = _conjugated_structure(B, pB, conjB)
    LA = TensorKit.permute(A′, pA′)
    LB = TensorKit.permute(B′, pB′)
    product_space = TensorKit.compose(LA, LB)
    output = TensorOperations.tensorcontract(A, pA, conjA, B, pB, conjB, pAB)

    supported = sector_cost_supported(A′) && sector_cost_supported(B′)
    flops = 0.0
    nblocks = 0
    largest = 0.0
    for q in TensorKit.blocksectors(product_space)
        # `product_space` describes exactly the blocks visited by TensorKit's
        # `mul!`; the two guards make a malformed structural call fail as an
        # empty/unsupported profile rather than inventing a block cost.
        TensorKit.hasblock(LA, q) && TensorKit.hasblock(LB, q) || continue
        m = TensorKit.blockdim(codomain(LA), q)
        k = TensorKit.blockdim(domain(LA), q)
        k == TensorKit.blockdim(codomain(LB), q) ||
            throw(ArgumentError("incompatible planned sector block $q"))
        n = TensorKit.blockdim(domain(LB), q)
        flops += 2.0 * m * k * n
        largest = max(largest, Float64(m) * n)
        nblocks += 1
    end
    output_largest = sector_block_peak(output)
    return (output=output,
            sector_flops=supported ? flops : NaN,
            output_elements=Float64(dim(output)),
            left_permuted_stored_elements=Float64(dim(LA)),
            right_permuted_stored_elements=Float64(dim(LB)),
            product_stored_elements=Float64(dim(product_space)),
            block_count=nblocks,
            largest_block_elements=largest,
            output_largest_block_elements=output_largest,
            peak_block_elements=max(largest, output_largest),
            supported=supported)
end

"""
    pair_cost(dimsA, openA, contractA, dimsB, openB; memory_weight=0) -> Float64

Dense estimate for a binary contraction. This is intentionally sector-blind:
Phase 2 uses it for the dimensions-only planner, while sector-aware block costs
remain a Phase-3 concern. `memory_weight` prices the output intermediate in
elements; callers that need the two terms separately retain them in Planning.
"""
function pair_cost(dimsA::AbstractVector{<:Integer}, openA, contractA,
                   dimsB::AbstractVector{<:Integer}, openB;
                   memory_weight::Real=0)
    out_a = _dimprod(dimsA, openA)
    contracted = _dimprod(dimsA, contractA)
    out_b = _dimprod(dimsB, openB)
    flops = out_a * contracted * out_b
    peak = out_a * out_b
    return flops + Float64(memory_weight) * peak
end

@inline function _dimprod(dims::AbstractVector{<:Integer}, inds)
    p = 1.0
    for i in inds
        p *= Float64(dims[i])
    end
    return p
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
