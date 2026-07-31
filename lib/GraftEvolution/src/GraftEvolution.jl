module GraftEvolution

import GraftContractions
import GraftFoundation
import GraftNetworks
import GraftParallel

const Backend = GraftFoundation.Backend
const Trees = GraftFoundation.Trees
const Networks = GraftNetworks.Networks
const Contractions = GraftContractions.Contractions
const Parallel = GraftParallel.Parallel

include("Evolution/Evolution.jl")

using .Evolution

export Evolution
export Evolver, step!, evolve!, CorrelatorSeries, correlator, correlator_series,
    supports_complex_step,
    TDVP1, TDVP2, TDVP1_CBE, GlobalKrylov, DirectKrylovBootstrap,
    DirectKrylovInfo, GSE_TDVP, LSE_TDVP, TEBD, BUG, FixedBUG,
    ImplicitLogScheme, LogBackwardEuler, LogTrapezoid,
    LogGaussLegendre, ImplicitLogTime, logarithmic_time_grid, linsolve!,
    ResidualDrivenExpansion, LinearResidualReport,
    ResidualExpansionEdgeReport, ResidualExpansionReport,
    ResidualDrivenReport, linear_residual, residual_expand!,
    residual_driven_linsolve!,
    TwoSiteLinearPolicy, TwoSiteLinearEdgeReport, TwoSiteLinearReport,
    two_site_linsolve!,
    PairedLinearClassification, classify_linear_pair,
    PairedEdgeSubspaceEvidence, PairedLinearDiagnostic,
    paired_linear_diagnostic,
    evolution_exponentiate_backend

end # module GraftEvolution
