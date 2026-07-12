import PrecompileTools
import Random

# Keep each fixture lexically scoped: PrecompileTools 1.2, selected by Julia
# 1.10, does not add that isolation automatically. The workloads use only
# deterministic in-memory data and bounded iteration counts.
include("precompile/dense_core.jl")
include("precompile/dense_solvers.jl")
include("precompile/symmetry.jl")
include("precompile/thermal.jl")

# Checkpoint workloads are deliberately excluded: checkpoint! and resume
# necessarily perform real filesystem I/O. PackageCompiler can add them to a
# deployment-specific sysimage workload when the target path policy is known.
