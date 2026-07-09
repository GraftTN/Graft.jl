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
    normalize!, apply, fit!, apply_local, ishermitian, invalidate_node!,
    invalidate_edge!

include("ttns.jl")
include("ttno.jl")
include("apply.jl")
include("fit.jl")

# ---------------------------------------------------------------------------
# TODO(M0/M1, architecture §3 "通用操作"):
#   operator-weighted fitting `fit!(φ, sources; Hs, coeffs)` — PyTreeNet
#       variational_fitting generalizes this state-compression primitive to
#       sums of operator applications for GK/GSE/METTS (§11.6).
# ---------------------------------------------------------------------------

end # module Networks
