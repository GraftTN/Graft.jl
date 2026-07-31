# Owner-local workloads are included by the runtime subpackages that own the
# compiled methods. Importing the umbrella triggers those package workloads
# when the deployment build opts in with `GRAFT_FULL_PRECOMPILE=true`.
#
# No genuinely cross-package umbrella workload is currently needed. Keep this
# file as the orchestration seam for a future workload that cannot belong to a
# single runtime package.

# Checkpoint workloads are deliberately excluded: checkpoint! and resume
# necessarily perform real filesystem I/O. PackageCompiler can add them to a
# deployment-specific sysimage workload when the target path policy is known.
