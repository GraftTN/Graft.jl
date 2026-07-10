"""
    Contractions.Planning

Shape-only planning and binary execution for Krylov-hot effective-Hamiltonian
networks. The parent `Contractions` module describes TTN semantics and supplies
labelled `ContractionSpec`s; this submodule owns only graph lowering, dense
costs, plan selection, caching helpers, and the slot-walk executor.
"""
module Planning

using TensorOperations
using ...Backend

export ContractionSpec, PairStep, ContractionPlan, EffectiveMap, PlanKey,
    execute, execute_accumulate!, plan_contraction, plan_key, get_or_plan!,
    ncon_reference, plan_metrics, dense_cost

include("types.jl")
include("cost.jl")
include("planner.jl")
include("executor.jl")
include("cache.jl")

end # module Planning
