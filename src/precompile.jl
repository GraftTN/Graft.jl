import PrecompileTools
import Random

# Keep each fixture lexically scoped: PrecompileTools 1.2, selected by Julia
# 1.10, does not add that isolation automatically. The workloads use only
# deterministic in-memory data and bounded iteration counts.
#
# The full workload costs ~6 minutes of `Pkg.precompile()` (dominated by
# TensorKit/TensorOperations/KrylovKit specializing on every solver family in
# dense_solvers.jl) to buy out first-call JIT latency on those exact
# type/algorithm combinations. That trade is worth it for a deployment build
# (e.g. the PackageCompiler sysimage in §10.7) but not for everyday
# `Pkg.precompile()` during development or CI. Opt into it explicitly:
#
#     GRAFT_FULL_PRECOMPILE=true julia --project -e 'using Pkg; Pkg.precompile()'
if get(ENV, "GRAFT_FULL_PRECOMPILE", "false") == "true"
    include("precompile/dense_core.jl")
    include("precompile/dense_solvers.jl")
    include("precompile/symmetry.jl")
    include("precompile/thermal.jl")
end

# Checkpoint workloads are deliberately excluded: checkpoint! and resume
# necessarily perform real filesystem I/O. PackageCompiler can add them to a
# deployment-specific sysimage workload when the target path policy is known.
