module GraftSpectral

import GraftContractions
import GraftEvolution
import GraftNetworks
import GraftParallel

const Networks = GraftNetworks.Networks
const Contractions = GraftContractions.Contractions
const Evolution = GraftEvolution.Evolution
const Parallel = GraftParallel.Parallel

include("Spectral/Spectral.jl")

using .Spectral

export Networks, Contractions, Evolution, Parallel, Spectral
export UniformSequence,
    SampleReductionPolicy, AllComponents, DeclaredDiagonal, TraceReduction,
    ReductionDiagnostics,
    AbstractZeroPolicy, ExactZero, ToleranceZero,
    AbstractEvidenceRankPolicy, NumericalRank, AbsoluteThresholdRank,
    RelativeThresholdRank, KneeRank,
    AbstractRankPolicy, AutomaticRank, StrictRank, ClampedRank, RankResolution,
    AbstractNodeEstimator, HankelDMD, ESPRIT, LeftSubspaceESPRIT,
    AbstractNodeOutcome, IdentifiedNodes, ZeroSequence, AbstractNodeFailure,
    NodeEstimationFailure, ReductionErasedSignal,
    NodeDiagnostics, HankelDMDDiagnostics, ESPRITDiagnostics,
    LeftSubspaceESPRITDiagnostics, NodeEstimate,
    ExponentialSum, AbstractExponentialFitOutcome, IdentifiedFit, ZeroFit,
    FailedFit, FitDiagnostics, ExponentialFit,
    estimate_nodes, fit_exponential_sum, evaluate,
    LinearPredictionResult, linear_prediction, predict,
    ModePolicy, KeepModes, ProjectUnitCircle, RejectOutsideUnitCircle,
    DropOutsideUnitCircle, ModeModification,
    ARLeastSquares, ARLeastSquaresDiagnostics,
    AbstractPruningPolicy, NoPruning, WeightNormPruning, PrunedMode,
    PruningDiagnostics,
    AbstractAttemptStopRule, FirstControlled, ExhaustiveSearch,
    AbstractAttemptSelectionRule, MinimumTrainingRelativeL2,
    WeightLeastSquaresDiagnostics, ExponentialScore,
    AbstractDescendingRankAttemptOutcome, NodeAttemptFailure,
    ExponentialAttemptFailure, IdentifiedAttempt,
    DescendingRankAttempt, DescendingRankHistory, DescendingRankSearch,
    candidate_estimator, attempt_singular_values,
    ComplexTimeKrylovResult, complex_time_krylov

end # module GraftSpectral
