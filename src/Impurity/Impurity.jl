"""
L6 — GRAFT.Impurity: embedded impurity-solver module (architecture §6).
**Backbone types + signatures only; implementations TODO (M0 bath fitting →
M2 thermofield → M5 EDMFT).**

Never referenced by any lower layer (§9.10). Owns *no* private geometry code
(§0.1): geometry builders emit plain `Trees.TreeTopology`.
"""
module Impurity

using ..Trees
using ..Networks
using ..Symbolic

export Partition, audit_partition, BathParametrization, RealPoles, ComplexPoles,
    fit_bath, solve

# ---------------------------------------------------------------------------
# §6.2 Partition: a *user declaration* on the impurity orbitals; H_bath never
# partitions independently — it inherits the block structure through Δ(ω).
# Dependency chain (one-way):
#   physics priors → Partition P → block structure of Δ → blockwise bath fit
#   → modes mounted on the block's branch.
# ---------------------------------------------------------------------------

"""
    Partition(blocks::Vector{Vector{Symbol}})

Immutable grouping of impurity orbitals into blocks (eg/t2g, j_eff, d+ligand…).
First-class *input*: fixed partition ⇒ fixed topology ⇒ warm starts across the
self-consistency loop stay valid (`==`/`hash` are value-based, §9.4/§10.9).
Automatic partitioning is deliberately **not** offered — `audit_partition` is
the diagnostic ("人分区、库验收").
"""
struct Partition
    blocks::Vector{Vector{Symbol}}
    function Partition(blocks::Vector{Vector{Symbol}})
        orbs = reduce(vcat, blocks; init=Symbol[])
        allunique(orbs) || throw(ArgumentError("orbitals appear in more than one block"))
        return new(blocks)
    end
end
Base.:(==)(a::Partition, b::Partition) = a.blocks == b.blocks
Base.hash(p::Partition, h::UInt) = hash(p.blocks, hash(:Partition, h))

# TODO(M5): cross-block vs in-block mutual-information audit; warn when
# cross-block MI ≳ in-block MI ("the partition may be cut wrong"). Rare-event
# handling stays manual: re-partition + loop restart (§6.2).
"""
    audit_partition(ψ_converged, P::Partition) -> report

Partition diagnostic ("人分区、库验收"). TODO(M5) — no methods yet.
"""
function audit_partition end

# ---------------------------------------------------------------------------
# §6.3 bath discretization / hybridization fitting
# ---------------------------------------------------------------------------

abstract type BathParametrization end

"""Real-pole bath (adapol-style: AAA init + refinement). The only planned M0 implementation. TODO."""
struct RealPoles <: BathParametrization
    # TODO(M0): poles::Vector{Float64}, residues (blockwise), fit diagnostics
end

"""
    ComplexPoles

Type slot ONLY (§6.3): quasi-Lindblad pseudomode baths (complex poles). Kept so
the fitter interface doesn't change when the TTNDO route lands; deliberately
unimplemented.
"""
struct ComplexPoles <: BathParametrization end

"""
    fit_bath(Δ, P::Partition; T=0, crossblock=:highmount) -> BathParametrization

TODO(M0): blockwise pole fitting of the hybridization Δ (adapol reference
implementation; single-block partition degenerates to unpartitioned fitting).
The partition argument is in the signature from day one (§6.3).

* `T = 0`: Δ(iωₙ)/Δ(ω) → real-pole fit per block.
* `T > 0` (TODO M2): thermofield star encoding — fit Γf and Γ(1−f) (fermions) /
  absorption & emission parts (bosons) separately; vacuum product initial state.
* `crossblock = :highmount | :rotate` (§6.2): high mounting near the tree
  center, or a pre-fit single-particle rotation (returned with results).

Mandatory self-checks to implement with it (§6.3): (1) β·δε ≪ 1 resolution
check; (2) loop-bath vs final-bath both projected back to Δ(iωₙ) and compared.
Global fitting across the whole Δ matrix while ignoring `P` is a forbidden
path — the interface does not offer it. TODO(M0) — no methods yet.
"""
function fit_bath end

# ---------------------------------------------------------------------------
# §6.1 geometry constructors — thin wrappers over Trees.Geometries that mount
# bath branches according to the Partition (mechanical expansion, §6.2).
# TODO(M0): star/chain/fork assembly from (Partition, BathParametrization).
# Trees already provides: mps/star/binary/fork topologies, is_t3ns predicate.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# §6.4 measurements — G(τ) on sparse IR/DLR grids, G(t), F(t) improved
# estimators (Σ = F·G⁻¹), χ_ch(τ) two-particle (EDMFT W-loop), TRIQS BlockGf
# round-trip. TODO(M0–M2). §6.5 Spectral post-processing (LP/ESPRIT +
# complex-time Krylov Gram matrices) is milestone-1 scope but consumes evolver
# snapshots only — lands as a separate file once Evolution is validated. TODO.
# ---------------------------------------------------------------------------

# TODO(M0+): the single self-consistency-facing entry point (§6.6). GRAFT does
# NOT implement the DMFT/EDMFT loop itself. Contract: `ψ0` warm starts are
# first-class (topology hash validated — refuse silently rebuilt geometry);
# basis rotations `U` are returned with the results, loop side stays oblivious.
"""
    solve(bath, H_loc; partition, T, observables, ψ0=nothing) -> (; G, Σ, χ, ψ, U)

Impurity-solver entry point for self-consistency loops. TODO — no methods yet.
"""
function solve end

end # module Impurity
