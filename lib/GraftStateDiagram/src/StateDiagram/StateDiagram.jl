"""
Typed, opt-in StateDiagram compiler.

The stable legacy `ttno_from_opsum` builder remains separately owned by
`LegacyTTNOBuild`; this module depends on its explicit lowering interface in
one direction only.
"""
module StateDiagramCompiler

using LinearAlgebra: ColumnNorm, diag, norm, pinv, qr
using ..Backend
using ..Trees
using ..Networks
using ..Symbolic
using ..LegacyTTNOBuild.LegacyLoweringInterface

export compile_ttno, AbstractOperatorLoweringKernel, AbelianScalarLowering,
    AbstractTTNOMergeKernel, DirectSumMerge, StateDiagramMerge,
    StructuralOptimizer, GammaCoverOptimizer, SGEOptimizer,
    MissingCategoryCapability, TTNOBuildReport, TTNOBuildEdgeReport,
    compiler_exact_provenance

include("category_services.jl")
include("build_input.jl")
include("ir.jl")
include("merge_plans.jl")
include("lowering.jl")
include("realize.jl")
include("structural_merge.jl")
include("sge.jl")
include("gamma_cover.jl")
include("facade.jl")
include("compression_provenance.jl")
include("ir_serialization.jl")

end # module StateDiagramCompiler
