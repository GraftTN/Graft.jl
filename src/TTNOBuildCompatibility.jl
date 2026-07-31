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

end # module TTNOBuild
