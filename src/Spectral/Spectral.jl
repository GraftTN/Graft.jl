"""
M1 spectral post-processing.

This module contains only small dense numerical kernels. Time evolution stays
in `Evolution`; bath realization stays in companion packages.
"""
module Spectral

using LinearAlgebra
using ..Networks
using ..Contractions
using ..Evolution: CorrelatorSeries
using ..Parallel: AbstractDistributedContext, distributed_rank,
    distributed_size, distributed_allreduce_sum!

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

include("exponential_sum.jl")
include("linear_prediction.jl")
include("exponential_search.jl")
include("complex_time_krylov.jl")

end # module Spectral
