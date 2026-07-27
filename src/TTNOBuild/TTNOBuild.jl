"""
L4b — TTNO assembly from symbolic Hamiltonians (architecture §4b).

Port of the PyTreeNet pipeline (§4b, the core porting target):
single-term diagrams → **state diagram** (hyperedge/vertex merging) →
per-node tensor assembly; then `compress!` (deparallelization +
sector-resolved SVD).

Planned extensions beyond PyTreeNet (§4b):
* dense four-index Coulomb V_ijkl pre-factorization uses the ISDF-THC and
  fixed-factor fallback in `thc.jl` *before* it ever reaches the diagram;
* abelian sector-aware virtual legs are implemented in `statediagram.jl`;
  non-abelian SU(2) fusion-tree info from the SU2Reduce pass remains TODO(M3);
* bipartite-graph optimization + symbolic Gaussian elimination on the diagram
  — TODO (upstream PyTreeNet has them; port after the baseline is validated).
"""
module TTNOBuild

using LinearAlgebra: ColumnNorm, diag, norm, pinv, qr
using ..Backend
using ..Trees
using ..Networks
using ..Symbolic

export ttno_from_opsum, THCFactorization, THCReport, isdf_thc, fit_thc,
    reconstruct_thc

include("thc.jl")
include("statediagram.jl")

end # module TTNOBuild
