# SD1/CAT0 — lowering/merge kernel traits, merge-plan types, and the
# realization-view contract (state-diagram-compiler plan v2, "Ownership and
# kernel separation" and "Plan and realization boundary").
#
# Lowering and merging are independent strategy axes. Lowering owns physical
# correctness, ordered local factors, braided-term certificates, and
# category-correct local morphisms. Merging may only remove redundancy:
# disabling merging never changes the set of Hamiltonians the lowering
# kernel supports.

# ---------------------------------------------------------------------------
# Kernel traits
# ---------------------------------------------------------------------------

"""
Lowering kernel axis. `lower_terms(input, lowering)` has one canonical
output: `Vector{TermTTNOExpansion}`.
"""
abstract type AbstractOperatorLoweringKernel end

"""
    AbelianScalarLowering()

The supported `UniqueFusion + SymmetricBraiding` scalar lowering kernel. It
consumes the SD0 braided-term certificate for every graded term and records
it as lowering provenance; it fails closed on any other category profile.
"""
struct AbelianScalarLowering <: AbstractOperatorLoweringKernel end

"""
Merge kernel axis. `merge_channels(input, terms, merge)` has one canonical
output family: `AbstractTTNOMergePlan`. Merge kernels receive the read-only
input explicitly because category equivalence is input context, not global
state.
"""
abstract type AbstractTTNOMergeKernel end

"""
    DirectSumMerge()

The mechanical no-merge kernel: retains every term channel and produces a
[`DirectSumPlan`](@ref). The direct-sum plan is the primary correctness
oracle for every optimized plan.
"""
struct DirectSumMerge <: AbstractTTNOMergeKernel end

"""
    StateDiagramMerge(optimizer)

The optimizing merge kernel producing a [`StateDiagram`](@ref) plan through
the deterministic postorder optimizer sequence. Every dimension-reducing
step must carry a typed merge proof and an exact log entry.
"""
struct StateDiagramMerge{O} <: AbstractTTNOMergeKernel
    optimizer::O
end

# ---------------------------------------------------------------------------
# Merge proofs and optimizer log
# ---------------------------------------------------------------------------

"""
Exact redundancy witness of one dimension-reducing merge. A basis
isomorphism alone never merges channels: every actual merge additionally
requires an exact structural or exact linear-dependence witness of one of
these kinds.
"""
abstract type AbstractRedundancyWitness end

"""
    StructuralIdentityWitness(kind)

Witness that removed channel copies are structurally identical to the
retained copy: `kind` is `:identical_hyperedge` or `:identical_subtree`.
"""
struct StructuralIdentityWitness <: AbstractRedundancyWitness
    kind::Symbol
end

"""
    GammaCoverWitness

Exact reconstruction log of one raw-Gamma minimum-cover merge on one cut:
the deterministic mixing-block key, the lossless row (below-content) and
column (above-context) keys, the support graph, the deterministic
Hopcroft-Karp maximum matching, the Konig cover, the per-term
row/column assignment, and every coefficient-atom relocation performed by
the reconstruction. The witness replays and reverses the merge exactly.
"""
struct GammaCoverWitness <: AbstractRedundancyWitness
    edge::Int
    block::String
    rows::Vector{String}
    cols::Vector{String}
    support::Vector{Tuple{Int,Int}}
    matching::Vector{Tuple{Int,Int}}
    cover_rows::Vector{Int}
    cover_cols::Vector{Int}
    assignments::Vector{Tuple{Int,Int,Int}}
    moved_atoms::Vector{Tuple{Int,Int,Int}}
    previous::Vector{Tuple{Int,DegeneracyLabel}}
end

"""
    MergeProofStep(kind, edge, span, relation, witness, removed, retained)

One typed, replayable merge proof step: the merge kind, the oriented edge
(child node index) it acts on, the addressed span, the span-level basis
relation used for transport, the exact redundancy witness, and the removed
and retained channel-copy labels. The span convention is opaque (audit
finding F1): a span may address either the channel-copy degeneracy axis or
the fusion-multiplicity axis; in the Abelian degenerate profile every span
addresses the degeneracy axis only.
"""
struct MergeProofStep
    kind::Symbol
    edge::Int
    span::ChannelSpan
    relation::AbstractSpanRelation
    witness::AbstractRedundancyWitness
    removed::Vector{DegeneracyLabel}
    retained::Vector{DegeneracyLabel}
end

"""
    OptimizerLogEntry(step, operation, detail)

One typed exact operation recorded by the deterministic optimizer, in
execution order.
"""
struct OptimizerLogEntry
    step::Int
    operation::Symbol
    detail::String
end

"""
    OptimizerLog(entries)

Complete replayable optimizer log. Deterministic across repeated runs on
equal input; an empty log is the zero-merge state.
"""
struct OptimizerLog
    entries::Vector{OptimizerLogEntry}
end

OptimizerLog() = OptimizerLog(OptimizerLogEntry[])

# ---------------------------------------------------------------------------
# Merge plans
# ---------------------------------------------------------------------------

"""
Producer family of realization plans. `DirectSumPlan` mechanically retains
all term channels; `StateDiagram` contains the optimized graph plus typed
merge proof steps and an optimizer log. Both expose the same canonical
[`RealizationPlanView`](@ref) through [`realization_view`](@ref), and
`realize_ttno` never branches on the concrete producer.
"""
abstract type AbstractTTNOMergePlan end

"""
    DirectSumPlan(topology, provenance, expansions)

The uncompressed direct-sum plan: every term expansion is retained channel
for channel. Primary correctness oracle for all optimized plans. Plans are
self-contained: they carry the topology and the input provenance they were
built from, and realization fails closed on a provenance mismatch.
"""
struct DirectSumPlan{Q,C<:Number} <: AbstractTTNOMergePlan
    topology::TreeTopology
    provenance::BuildInputProvenance
    expansions::Vector{TermTTNOExpansion{Q,C}}
end

"""
    StateDiagram(topology, provenance, expansions, proofs, log)

The optimized merge plan: the term expansions with their channel copies
rewritten by proved merges, the typed merge proof steps, and the exact
optimizer log. With empty proofs and an empty log this is the zero-merge
state diagram, whose realization view must be field-identical to the
direct-sum view over the same input (ADR-0003 decision 5).
"""
struct StateDiagram{Q,C<:Number} <: AbstractTTNOMergePlan
    topology::TreeTopology
    provenance::BuildInputProvenance
    expansions::Vector{TermTTNOExpansion{Q,C}}
    proofs::Vector{MergeProofStep}
    log::OptimizerLog
end

for T in (:StructuralIdentityWitness, :GammaCoverWitness, :MergeProofStep,
          :OptimizerLogEntry, :OptimizerLog, :DirectSumPlan, :StateDiagram)
    @eval begin
        Base.:(==)(a::$T, b::$T) = _ir_isequal(a, b)
        Base.isequal(a::$T, b::$T) = _ir_isequal(a, b)
        Base.hash(x::$T, h::UInt) = _ir_hash(x, h)
    end
end

# ---------------------------------------------------------------------------
# Pipeline contract (generic functions)
# ---------------------------------------------------------------------------

"""
    lower_terms(input::TTNOBuildInput, lowering::AbstractOperatorLoweringKernel)
        -> Vector{TermTTNOExpansion}

Lower every normalized term into its canonical expansion. Lowering owns
physical correctness and fails closed (`MissingCategoryCapability`) before
densification on unsupported category profiles.
"""
function lower_terms end

"""
    merge_channels(input::TTNOBuildInput, terms::Vector{<:TermTTNOExpansion},
                   merge::AbstractTTNOMergeKernel) -> AbstractTTNOMergePlan

Combine term expansions into one realization plan, removing only redundancy
under typed proofs. `DirectSumMerge` is implemented with the SD2 direct-sum
realization; `StateDiagramMerge` with the SD3-SD5 optimizer.
"""
function merge_channels end

"""
    realization_view(plan::AbstractTTNOMergePlan) -> RealizationPlanView

The canonical producer-independent realization view of a merge plan. Views
over a `DirectSumPlan` and over a zero-merge `StateDiagram` of the same
input are field-identical (ADR-0003 decision 5). Implemented with the SD2
direct-sum realization.
"""
function realization_view end

"""
    realize_ttno(input::TTNOBuildInput, plan::AbstractTTNOMergePlan)
        -> (TTNO, TTNOBuildReport)

Producer-independent TTNO realization: immediately consumes
[`realization_view`](@ref) and is the single assembly and correctness gate
for spaces, arrows, duals, the tensor-unit root boundary, deterministic
channel order, coefficient evaluation, Hermiticity contract validation,
merge-proof validation, and reporting. Implemented with SD2.
"""
function realize_ttno end
