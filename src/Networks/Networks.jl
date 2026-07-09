"""
L3 — Network types: TTNS / TTNO / TTNDO (architecture §3).

Types live *below* Contractions in the include order (the conceptual L2/L3
layering of the architecture document is unchanged: Contractions operates on
these types, nothing here contracts environments).
"""
module Networks

using ..Backend
using ..Backend: SpaceMismatch, scalartype, spacetype
using ..Trees

export TTNS, TTNO, TTNDO, topology, center, hasphys, physleg, parentleg,
    physspace, virtualspace, check_arrows, move_center!, update_tensor!,
    normalize!, ishermitian, invalidate_node!, invalidate_edge!

include("ttns.jl")
include("ttno.jl")

# ---------------------------------------------------------------------------
# TODO(M0/M1, architecture §3 "通用操作"):
#   apply(O::TTNO, ψ::TTNS)  — zip-up / naive apply + recompress
#   fit!(φ, ψ; H=nothing)    — variational fitting/compression primitive
#       (PyTreeNet dmrg/variational_fitting; shared by GK evolver, GSE, METTS
#        collapse compression — deliberately a public primitive, §11.6)
# ---------------------------------------------------------------------------

end # module Networks
