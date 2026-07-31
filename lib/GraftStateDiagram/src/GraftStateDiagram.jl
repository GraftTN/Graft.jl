module GraftStateDiagram

import GraftFoundation
import GraftNetworks
import GraftSymbolic
import GraftTTNOBuild

const Backend = GraftFoundation.Backend
const Trees = GraftFoundation.Trees
const Networks = GraftNetworks.Networks
const Symbolic = GraftSymbolic.Symbolic
const LegacyTTNOBuild = GraftTTNOBuild.LegacyTTNOBuild

include("StateDiagram/StateDiagram.jl")

using .StateDiagramCompiler: compile_ttno, AbstractOperatorLoweringKernel,
    AbelianScalarLowering, AbstractTTNOMergeKernel, DirectSumMerge,
    StateDiagramMerge, StructuralOptimizer, GammaCoverOptimizer, SGEOptimizer,
    MissingCategoryCapability, TTNOBuildReport, TTNOBuildEdgeReport,
    compiler_exact_provenance

export StateDiagramCompiler, compile_ttno, AbstractOperatorLoweringKernel,
    AbelianScalarLowering,
    AbstractTTNOMergeKernel, DirectSumMerge, StateDiagramMerge,
    StructuralOptimizer, GammaCoverOptimizer, SGEOptimizer,
    MissingCategoryCapability, TTNOBuildReport, TTNOBuildEdgeReport,
    compiler_exact_provenance

end # module GraftStateDiagram
