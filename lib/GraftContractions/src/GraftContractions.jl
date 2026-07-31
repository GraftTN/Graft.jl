module GraftContractions

import GraftFoundation
import GraftNetworks
import GraftParallel
import GraftPlanning

const Backend = GraftFoundation.Backend
const Trees = GraftFoundation.Trees
const Networks = GraftNetworks.Networks
const Planning = GraftPlanning.Planning
const Parallel = GraftParallel.Parallel

include("Contractions/Contractions.jl")

using .Contractions: EnvCache, env!, build_env, invalidate_node!,
    invalidate_edge!, EffectiveMap, ChannelSlicedEffectiveMap,
    DistributedChannelEffectiveMap, DistributedChannelAdmissionError,
    ContractionPlan, ContractionSpec, PlannerCandidateFailure,
    PlannerDiagnostics, plan_diagnostics, PlanKey, CacheDiagnostics,
    EnvironmentCacheDiagnostics, EnvironmentBuildConflict, plan_cache_stats,
    cache_diagnostics, env_cache_stats, with_workspace_map, PlanWorkspace,
    workspace_map, workspace_stats, inner, expect, eff_h1, eff_h0, eff_h2,
    two_site_tensor, two_site_space, split_two_site!, expand!

export Backend, Trees, Networks, Planning, Parallel, Contractions
export EnvCache, env!, build_env, invalidate_node!,
    invalidate_edge!, EffectiveMap, ChannelSlicedEffectiveMap,
    DistributedChannelEffectiveMap, DistributedChannelAdmissionError,
    ContractionPlan, ContractionSpec, PlannerCandidateFailure,
    PlannerDiagnostics, plan_diagnostics, PlanKey, CacheDiagnostics,
    EnvironmentCacheDiagnostics, EnvironmentBuildConflict, plan_cache_stats,
    cache_diagnostics, env_cache_stats, with_workspace_map, PlanWorkspace,
    workspace_map, workspace_stats, inner, expect, eff_h1, eff_h0, eff_h2,
    two_site_tensor, two_site_space, split_two_site!, expand!

end # module GraftContractions
