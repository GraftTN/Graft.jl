"""
L2 — Environment caching and contraction primitives (architecture §3).

PyTreeNet counterparts: contractions/tree_cach_dict.py (`PartialTreeCachDict`
→ `EnvCache`), sandwich_caching.py, effective_hamiltonians.py,
state_operator_contraction.py.

# Environment storage convention

For the directed edge `u → v` (u, v adjacent), `env(u→v)` is the full
contraction of the bra–TTNO–ket sandwich over the connected component
containing `u` when the edge is cut. It is stored as a rank-3 tensor with all
legs in the codomain, ordered **(ket, op, bra)**; for pure state–state
transfers (no operator) it is rank-2 **(ket, bra)**.

Viewed spaces (with `V_e`/`U_e` the state/operator edge spaces of the edge
between u and v, oriented child→parent):
* u child of v ("below" env):   `(dual(V_e), dual(U_e), V_e)`
* v child of u ("above" env):   `(V_e, U_e, dual(V_e))`

Both directions are produced by ONE recursion: `env(u→v)` contracts u's node
tensors with `env(w→u)` for every neighbour `w ≠ v`.

Two iron rules (§3):
1. invalidation is explicit — `invalidate_node!`/`invalidate_edge!` are events
   fired by `Networks.update_tensor!`/`move_center!`; entries never go stale
   silently;
2. `EnvCache` doubles as the future MPI dispatch unit (§8 level 3): it holds
   topology, per-directed-edge tensors, and small rebuildable compiled plans,
   so a subtree's environments can be shipped wholesale.
"""
module Contractions

using Random: AbstractRNG, randn
using ..Backend
using ..Trees
using ..Networks
import ..Networks: invalidate_node!, invalidate_edge!

export EnvCache, env!, build_env, invalidate_node!, invalidate_edge!,
    EffectiveMap, ContractionPlan, ContractionSpec, PlanKey, plan_cache_stats,
    env_cache_stats,
    PlanWorkspace, workspace_map, workspace_stats, inner, expect, eff_h1,
    eff_h0, eff_h2, two_site_tensor, two_site_space, split_two_site!, expand!

include("planning/Planning.jl")
using .Planning: ContractionSpec, ContractionPlan, EffectiveMap, PlanWorkspace,
    PlanKey, workspace_map, workspace_stats
include("envcache.jl")
include("effective.jl")
include("expansion.jl")
include("expectation.jl")

end # module Contractions
