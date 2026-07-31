"""Compatibility namespace combining legacy TTNO assembly and typed compilation."""
module TTNOBuild

using ..LegacyTTNOBuild
import ..StateDiagramCompiler

const StateDiagramAPI = StateDiagramCompiler

export ttno_from_opsum, THCFactorization, THCReport, isdf_thc, fit_thc,
    reconstruct_thc,
    compile_ttno, AbstractOperatorLoweringKernel, AbelianScalarLowering,
    AbstractTTNOMergeKernel, DirectSumMerge, StateDiagramMerge,
    StructuralOptimizer, GammaCoverOptimizer, SGEOptimizer,
    MissingCategoryCapability, TTNOBuildReport, TTNOBuildEdgeReport,
    compiler_exact_provenance

# Qualified legacy internals retained as object aliases for existing
# diagnostics; the implementation remains owned by LegacyTTNOBuild.
const _net_u1_charge = LegacyTTNOBuild._net_u1_charge
const _build_braided_term_plan = LegacyTTNOBuild._build_braided_term_plan
const _Euler = LegacyTTNOBuild._Euler
const _input_twist_parity = LegacyTTNOBuild._input_twist_parity

# Public typed compiler bindings are aliases to the owner module's objects.
const compile_ttno = StateDiagramCompiler.compile_ttno
const AbstractOperatorLoweringKernel =
    StateDiagramCompiler.AbstractOperatorLoweringKernel
const AbelianScalarLowering = StateDiagramCompiler.AbelianScalarLowering
const AbstractTTNOMergeKernel = StateDiagramCompiler.AbstractTTNOMergeKernel
const DirectSumMerge = StateDiagramCompiler.DirectSumMerge
const StateDiagramMerge = StateDiagramCompiler.StateDiagramMerge
const StructuralOptimizer = StateDiagramCompiler.StructuralOptimizer
const GammaCoverOptimizer = StateDiagramCompiler.GammaCoverOptimizer
const SGEOptimizer = StateDiagramCompiler.SGEOptimizer
const MissingCategoryCapability = StateDiagramCompiler.MissingCategoryCapability
const TTNOBuildReport = StateDiagramCompiler.TTNOBuildReport
const TTNOBuildEdgeReport = StateDiagramCompiler.TTNOBuildEdgeReport
const compiler_exact_provenance = StateDiagramCompiler.compiler_exact_provenance

# Before the package split these concrete compiler types lived at
# `Graft.TTNOBuild.<name>`.  Keep direct aliases for JLD2's repeated field-type
# lookup when it reconstructs objects written with those qualified paths.
# These are deliberately not wrappers and remain unexported unless they were
# already part of the public compiler facade above.
const AbelianEquivalenceService =
    StateDiagramCompiler.AbelianEquivalenceService
const AbelianFrameCertificate = StateDiagramCompiler.AbelianFrameCertificate
const AnyonicBraidingProfile = StateDiagramCompiler.AnyonicBraidingProfile
const AssertedHermitian = StateDiagramCompiler.AssertedHermitian
const BoundarySignature = StateDiagramCompiler.BoundarySignature
const BuildInputProvenance = StateDiagramCompiler.BuildInputProvenance
const CategoryCapabilityReport = StateDiagramCompiler.CategoryCapabilityReport
const CategorySemantics = StateDiagramCompiler.CategorySemantics
const ChannelClass = StateDiagramCompiler.ChannelClass
const ChannelIdentity = StateDiagramCompiler.ChannelIdentity
const ChannelOrientation = StateDiagramCompiler.ChannelOrientation
const ChannelSpan = StateDiagramCompiler.ChannelSpan
const CoeffAtom = StateDiagramCompiler.CoeffAtom
const CoeffAtomSlot = StateDiagramCompiler.CoeffAtomSlot
const CoeffTable = StateDiagramCompiler.CoeffTable
const DegeneracyLabel = StateDiagramCompiler.DegeneracyLabel
const DirectSumPlan = StateDiagramCompiler.DirectSumPlan
const EliminationCoverWitness = StateDiagramCompiler.EliminationCoverWitness
const ExactScalarSlot = StateDiagramCompiler.ExactScalarSlot
const ExactUnitSlot = StateDiagramCompiler.ExactUnitSlot
const ExplicitLocalTransition = StateDiagramCompiler.ExplicitLocalTransition
const FusionRoute = StateDiagramCompiler.FusionRoute
const GammaCoverWitness = StateDiagramCompiler.GammaCoverWitness
const GammaExpr = StateDiagramCompiler.GammaExpr
const GenericFusionProfile = StateDiagramCompiler.GenericFusionProfile
const IdentitySpanRelation = StateDiagramCompiler.IdentitySpanRelation
const LocalMorphismCertificate = StateDiagramCompiler.LocalMorphismCertificate
const LocalMorphismSpec = StateDiagramCompiler.LocalMorphismSpec
const LocalOpKey = StateDiagramCompiler.LocalOpKey
const LoweringProvenance = StateDiagramCompiler.LoweringProvenance
const MergeProofStep = StateDiagramCompiler.MergeProofStep
const MixingBlockKey = StateDiagramCompiler.MixingBlockKey
const MultiplicityFreeFusionProfile =
    StateDiagramCompiler.MultiplicityFreeFusionProfile
const MultiplicityLabels = StateDiagramCompiler.MultiplicityLabels
const NoBraidingProfile = StateDiagramCompiler.NoBraidingProfile
const NoHermiticityAssertion = StateDiagramCompiler.NoHermiticityAssertion
const NoKnownSpanRelation = StateDiagramCompiler.NoKnownSpanRelation
const NormalizedFactor = StateDiagramCompiler.NormalizedFactor
const NormalizedTerm = StateDiagramCompiler.NormalizedTerm
const OmittedIdentityTransition =
    StateDiagramCompiler.OmittedIdentityTransition
const OptimizerLog = StateDiagramCompiler.OptimizerLog
const OptimizerLogEntry = StateDiagramCompiler.OptimizerLogEntry
const RealizationEntry = StateDiagramCompiler.RealizationEntry
const RealizationPlanView = StateDiagramCompiler.RealizationPlanView
const RootBoundary = StateDiagramCompiler.RootBoundary
const SGEColElimination = StateDiagramCompiler.SGEColElimination
const SGERowElimination = StateDiagramCompiler.SGERowElimination
const SpanBasisTransport = StateDiagramCompiler.SpanBasisTransport
const StateDiagram = StateDiagramCompiler.StateDiagram
const StructuralIdentityWitness =
    StateDiagramCompiler.StructuralIdentityWitness
const SymmetricBraidingProfile = StateDiagramCompiler.SymmetricBraidingProfile
const TTNOBuildInput = StateDiagramCompiler.TTNOBuildInput
const TensorKitMaterializationService =
    StateDiagramCompiler.TensorKitMaterializationService
const TermHyperedge = StateDiagramCompiler.TermHyperedge
const TermTTNOExpansion = StateDiagramCompiler.TermTTNOExpansion
const UniqueFusionProfile = StateDiagramCompiler.UniqueFusionProfile

end # module TTNOBuild
