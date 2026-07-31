module GraftNetworks

import GraftFoundation
import GraftPlanning

const Backend = GraftFoundation.Backend
const Trees = GraftFoundation.Trees
const Planning = GraftPlanning.Planning

include("Networks/Networks.jl")

using .Networks: TTNS, TTNO, TTNDO, topology, center, hasphys, physleg,
    parentleg, physspace, virtualspace, check_arrows, move_center!,
    update_tensor!, normalize!, apply, fit!, exact_linear_combination,
    truncated_linear_combination, truncate_sweep!, apply_local, ishermitian,
    invalidate_node!, invalidate_edge!, compress!,
    TTNOCompressionSectorReport, TTNOCompressionEdgeReport,
    TTNOCompressionReport, TTNOExactChannelRelation, TTNOExactProvenance,
    TTNOExactWitness

export Backend, Trees, Planning, Networks
export TTNS, TTNO, TTNDO, topology, center, hasphys, physleg,
    parentleg, physspace, virtualspace, check_arrows, move_center!,
    update_tensor!, normalize!, apply, fit!, exact_linear_combination,
    truncated_linear_combination, truncate_sweep!, apply_local, ishermitian,
    invalidate_node!, invalidate_edge!, compress!,
    TTNOCompressionSectorReport, TTNOCompressionEdgeReport,
    TTNOCompressionReport, TTNOExactChannelRelation, TTNOExactProvenance,
    TTNOExactWitness

include("precompile.jl")

end # module GraftNetworks
