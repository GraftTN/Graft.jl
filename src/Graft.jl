"""
# Graft.jl

Grafting PyTreeNet's architecture onto TensorKit.jl: a general-purpose TTNS
core library. Impurity-solver workflows live in the companion
`GraftImpurity.jl` package.

Layering (architecture document §1; the conceptual L-numbers are unchanged —
the include order below only reflects that `Contractions` operates on the
`Networks` types):

    Backend        L0   TensorKit adapter: spaces, sectors, arrows, splitting
    Trees          L1   topology, traversal, geometries
    Networks       L3   TTNS / TTNO / TTNDO types, canonical form
    Contractions   L2   EnvCache + effective Hamiltonians + inner/expect
    Symbolic       L4a  OpSum, rewrite passes (PPDress/SU2Reduce/… TODO)
    TTNOBuild      L4b  symbolic Hamiltonian → TTNO assembly
    GroundState    L5a  DMRG family
    Evolution      L5b  complex-step Evolver family (TDVP1 …)
    Thermal        L5c  purification, METTS, correlators, benchmarks
    FreqDomain     L5d  TaSK resolvent kernel (TODO, M6)
    Checkpoints    ✕    JLD2 checkpoint/restart
    Parallel       ✕    threading + optional MPI.jl extension
    TestUtils      ✕    random states, dense/ED references

Dependency direction is monotone (§9.10): L(n) only uses L(<n); companion
packages depend on `Graft`, never the reverse; upper layers never
`import TensorKit` directly (§9.13).
"""
module Graft

include("Backend/Backend.jl")
include("Trees/Trees.jl")
include("Networks/Networks.jl")
include("Parallel/Parallel.jl")
include("Contractions/Contractions.jl")
# `apply` and `fit!` use the generic complete-tuple planner. Load their methods
# after the lower-level Networks data types and Contractions.Planning exist.
Base.include(Networks, joinpath(@__DIR__, "Networks", "apply.jl"))
Base.include(Networks, joinpath(@__DIR__, "Networks", "linear_combination.jl"))
Base.include(Networks, joinpath(@__DIR__, "Networks", "fit.jl"))
include("Symbolic/Symbolic.jl")
include("TTNOBuild/TTNOBuild.jl")
include("GroundState/GroundState.jl")
include("Evolution/Evolution.jl")
include("Spectral/Spectral.jl")
include("Thermal/Thermal.jl")
include("FreqDomain/FreqDomain.jl")
include("IO/Checkpoints.jl")
include("TestUtils/TestUtils.jl")

using .Backend
using .Trees
using .Networks
using .Contractions
using .Symbolic
using .TTNOBuild
using .GroundState
using .Evolution
using .Spectral
using .Thermal
using .FreqDomain
using .Checkpoints
using .Parallel

# public surface (re-exports; TestUtils stays namespaced)
# L0
export TruncationScheme, FermionSector, AbelianSector
# L1
export TreeTopology, nnodes, nodeid, nodeindex, isleaf, leaves, neighbors,
    postorder, preorder, path_between, tdvp_update_path,
    mps_topology, star_topology, binary_topology, fork_topology, mount_chain,
    is_t3ns
# L2/L3
export TTNS, TTNO, TTNDO, topology, center, move_center!, update_tensor!,
    normalize!, check_arrows, physspace, virtualspace, apply, fit!,
    exact_linear_combination, apply_local, compress!,
    TTNOCompressionSectorReport, TTNOCompressionEdgeReport, TTNOCompressionReport,
    EnvCache, EffectiveMap, ChannelSlicedEffectiveMap,
    DistributedChannelEffectiveMap, PlanWorkspace,
    workspace_map, workspace_stats,
    env_cache_stats, inner, expect, eff_h1, eff_h0, eff_h2
# L4
export OpSum, Term, SiteOp, charge, spin_ops, spin_ops_u1,
    boson_ops, boson_ops_u1, boson_ops_pp, fermion_ops_z2,
    boson_modes, BosonCoupling,
    Lindbladian, ppdress, ttno_from_opsum,
    THCFactorization, THCReport, isdf_thc, fit_thc, reconstruct_thc
# L5
export dmrg1!, dmrg2!, dmrg1_3s!, expand!,
    Evolver, step!, evolve!, CorrelatorSeries, correlator, correlator_series,
    supports_complex_step,
    TDVP1, TDVP2, TDVP1_CBE, GlobalKrylov, DirectKrylovBootstrap,
    DirectKrylovInfo, GSE_TDVP, LSE_TDVP, TEBD, BUG,
    ImplicitLogScheme, LogBackwardEuler, LogTrapezoid,
    LogGaussLegendre, ImplicitLogTime, logarithmic_time_grid, linsolve!,
    UniformSequence,
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
    ComplexTimeKrylovResult, complex_time_krylov,
    ThermalRep, Purified, METTS, HybridMETTS, thermalize,
    PurificationProblem, purification_problem, physical_ttno,
    PurifiedState, PurificationTrajectory, ScaledTTNS,
    METTSSample, METTSTrajectory, DistributedMETTSTrajectory,
    METTSStatistics, metts_statistics,
    infinite_temperature_state, thermal_expect, thermal_correlator,
    thermal_realtime_correlator, state_at
# cross-cutting
export checkpoint!, resume, with_checkpoint, checkpoint_mpi!, resume_mpi,
    threaded_foreach,
    configure_parallel_runtime!,
    AbstractDistributedContext, mpi_context, distributed_rank,
    distributed_size, distributed_root, distributed_isroot,
    distributed_barrier, distributed_allreduce_sum!,
    distributed_broadcast!, distributed_allgather,
    distributed_eigsolve, distributed_exponentiate,
    SubtreeOwnership, subtree_owner,
    local_nodes, boundary_edges

include("precompile.jl")

# TODO(§10.7): Graft.build_sysimage() — reuse the bundled workload with
# PackageCompiler for the checkpoint-resume cluster usage pattern.

end # module Graft
