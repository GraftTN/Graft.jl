"""
Pruning policy for weights recovered in the finite-exponential basis.
"""
abstract type AbstractPruningPolicy end

"""Retain every identified node, while still performing the post-policy refit."""
struct NoPruning <: AbstractPruningPolicy end

"""
    WeightNormPruning(tolerance)
    WeightNormPruning(; rtol=sqrt(eps(Float64)), atol=0)

Drop a mode when the norm of its sample-origin weight is no larger than
`max(atol + rtol * maximum(weight_norms),
     64 * eps(Float64) * maximum(weight_norms))`.
"""
struct WeightNormPruning{T<:AbstractFloat} <: AbstractPruningPolicy
    rtol::T
    atol::T
end

function WeightNormPruning(; rtol::Real=sqrt(eps(Float64)), atol::Real=0.0)
    relative = Float64(rtol)
    absolute = Float64(atol)
    isfinite(relative) && relative >= 0 ||
        throw(ArgumentError(
            "WeightNormPruning rtol must be finite and nonnegative",
        ))
    isfinite(absolute) && absolute >= 0 ||
        throw(ArgumentError(
            "WeightNormPruning atol must be finite and nonnegative",
        ))
    return WeightNormPruning(relative, absolute)
end

WeightNormPruning(tolerance::Real) = WeightNormPruning(; rtol=tolerance)

"""One mode removed by a pruning policy."""
struct PrunedMode
    index::Int
    node::ComplexF64
    weight_norm::Float64
    reason::Symbol
end

"""Typed record of one pruning decision."""
struct PruningDiagnostics{P<:AbstractPruningPolicy}
    policy::P
    weight_norms::Vector{Float64}
    threshold::Float64
    retained_indices::Vector{Int}
    rejected_modes::Vector{PrunedMode}
end

abstract type AbstractAttemptStopRule end

"""
Stop at the first successful attempt whose maximum training error is controlled
by the next discarded node-estimation singular value.
"""
struct FirstControlled <: AbstractAttemptStopRule end

"""Run every rank from the resolved initial rank through rank one."""
struct ExhaustiveSearch <: AbstractAttemptStopRule end

abstract type AbstractAttemptSelectionRule end

"""
Select the successful attempt with minimum relative-L2 training error.
Descending attempt order breaks exact ties deterministically.
"""
struct MinimumTrainingRelativeL2 <: AbstractAttemptSelectionRule end

"""Diagnostics for one full-sample exponential-basis least-squares solve."""
struct WeightLeastSquaresDiagnostics
    rank::Int
    condition_number::Float64
    singular_values::Vector{Float64}
end

"""Absolute and relative errors for one training or holdout sample set."""
struct ExponentialScore
    maximum_error::Float64
    l2::Float64
    relative_l2::Float64
    channel_relative_errors::Vector{Float64}
end

abstract type AbstractDescendingRankAttemptOutcome end

"""The node estimator returned a structured numerical failure."""
struct NodeAttemptFailure{F<:AbstractNodeFailure} <:
       AbstractDescendingRankAttemptOutcome
    failure::F
end

"""Node identification succeeded, but a later generic fit stage failed."""
struct ExponentialAttemptFailure <: AbstractDescendingRankAttemptOutcome
    reason::Symbol
    message::String
end

"""One successful, fully refitted exponential candidate."""
struct IdentifiedAttempt{S<:ExponentialSum} <:
       AbstractDescendingRankAttemptOutcome
    value::S
    diagnostics::FitDiagnostics
    controlled::Bool
end

"""
Complete typed record of one attempted estimator rank.

`pruning`, least-squares diagnostics, and scores are `nothing` only when the
attempt failed before that stage.
"""
struct DescendingRankAttempt{N<:NodeEstimate,P,
                             O<:AbstractDescendingRankAttemptOutcome}
    attempted_rank::Int
    node_estimate::N
    pre_pruning_node_count::Int
    pruning::P
    post_pruning_node_count::Int
    least_squares_before::Union{Nothing,WeightLeastSquaresDiagnostics}
    least_squares_after::Union{Nothing,WeightLeastSquaresDiagnostics}
    training_error::Union{Nothing,ExponentialScore}
    holdout_error::Union{Nothing,ExponentialScore}
    control_limit::Union{Nothing,Float64}
    outcome::O
end

"""All descending-rank attempts and the one-based selected-attempt index."""
struct DescendingRankHistory{R<:AbstractAttemptSelectionRule}
    initial_rank::Int
    selection::R
    attempts::Vector{DescendingRankAttempt}
    selected_attempt::Union{Nothing,Int}
end

"""
    DescendingRankSearch(estimator; initial_rank=nothing,
                         pruning=WeightNormPruning(),
                         stop=ExhaustiveSearch(),
                         selection=MinimumTrainingRelativeL2(),
                         holdout_count=0)

Fit finite exponential sums by rebuilding `estimator` as a strict one-shot
candidate at every usable rank, descending to rank one unless `stop` fires.
`holdout_count` reserves that many trailing input samples for scoring only.

This is a numerical exponential-basis driver. It deliberately has no callback
for Fermi-kernel, conformal, positivity, real-energy, bath, or other physical
consumer refits.
"""
struct DescendingRankSearch{
    E<:AbstractNodeEstimator,
    P<:AbstractPruningPolicy,
    S<:AbstractAttemptStopRule,
    R<:AbstractAttemptSelectionRule,
}
    estimator::E
    initial_rank::Union{Nothing,Int}
    pruning::P
    stop::S
    selection::R
    holdout_count::Int
end

function DescendingRankSearch(
    estimator::AbstractNodeEstimator;
    initial_rank::Union{Nothing,Integer}=nothing,
    pruning::AbstractPruningPolicy=WeightNormPruning(),
    stop::AbstractAttemptStopRule=ExhaustiveSearch(),
    selection::AbstractAttemptSelectionRule=MinimumTrainingRelativeL2(),
    holdout_count::Integer=0,
)
    rank_value = initial_rank === nothing ? nothing : Int(initial_rank)
    rank_value === nothing || rank_value > 0 ||
        throw(ArgumentError("initial_rank must be positive"))
    holdout = Int(holdout_count)
    holdout >= 0 ||
        throw(ArgumentError("holdout_count must be nonnegative"))
    return DescendingRankSearch(
        estimator, rank_value, pruning, stop, selection, holdout,
    )
end

"""
    candidate_estimator(estimator, attempted_rank)

Rebuild a node estimator with a strict one-shot rank while preserving its
evidence, sample-reduction, zero, and geometry policies. New
`AbstractNodeEstimator` implementations participate in
`DescendingRankSearch` by extending this typed reconstruction seam.
"""
function candidate_estimator(estimator::AbstractNodeEstimator,
                             attempted_rank::Integer)
    throw(ArgumentError(
        "$(typeof(estimator)) does not implement " *
        "candidate_estimator(estimator, attempted_rank)",
    ))
end

function candidate_estimator(estimator::HankelDMD, attempted_rank::Integer)
    attempted_rank > 0 ||
        throw(ArgumentError("attempted rank must be positive"))
    return HankelDMD(
        rank=StrictRank(
            Int(attempted_rank), _evidence_policy(estimator.rank),
        ),
        reduction=estimator.reduction,
        zero=estimator.zero,
        hankel_rows=estimator.hankel_rows,
    )
end

function candidate_estimator(estimator::ESPRIT, attempted_rank::Integer)
    attempted_rank > 0 ||
        throw(ArgumentError("attempted rank must be positive"))
    return ESPRIT(
        rank=StrictRank(
            Int(attempted_rank), _evidence_policy(estimator.rank),
        ),
        reduction=estimator.reduction,
        zero=estimator.zero,
        hankel_rows=estimator.hankel_rows,
    )
end

function candidate_estimator(estimator::LeftSubspaceESPRIT,
                             attempted_rank::Integer)
    attempted_rank > 0 ||
        throw(ArgumentError("attempted rank must be positive"))
    return LeftSubspaceESPRIT(
        rank=StrictRank(
            Int(attempted_rank), _evidence_policy(estimator.rank),
        ),
        reduction=estimator.reduction,
        zero=estimator.zero,
        hankel_rows=estimator.hankel_rows,
    )
end

function candidate_estimator(estimator::ARLeastSquares,
                             attempted_rank::Integer)
    attempted_rank > 0 ||
        throw(ArgumentError("attempted rank must be positive"))
    return ARLeastSquares(
        rank=StrictRank(
            Int(attempted_rank), _evidence_policy(estimator.rank),
        ),
        reduction=estimator.reduction,
        zero=estimator.zero,
        regularization=estimator.regularization,
        modes=estimator.modes,
    )
end

"""
    attempt_singular_values(estimate)

Return the nonincreasing singular-value evidence used by
`FirstControlled`. Estimators with another backend-diagnostic layout extend
this method.
"""
function attempt_singular_values(estimate::NodeEstimate)
    backend = estimate.backend
    backend === nothing && return Float64[]
    hasproperty(backend, :singular_values) || throw(ArgumentError(
        "$(typeof(backend)) does not expose singular-value evidence; " *
        "extend attempt_singular_values for FirstControlled",
    ))
    values = Float64.(getproperty(backend, :singular_values))
    all(value -> isfinite(value) && value >= 0, values) ||
        throw(ArgumentError(
            "attempt singular values must be finite and nonnegative",
        ))
    return values
end

function attempt_singular_values(
    estimate::NodeEstimate{O,C,B},
) where {O<:AbstractNodeOutcome,C,B<:ARLeastSquaresDiagnostics}
    values = copy(estimate.backend.evidence_singular_values)
    all(value -> isfinite(value) && value >= 0, values) ||
        throw(ArgumentError(
            "attempt singular values must be finite and nonnegative",
        ))
    return values
end

struct _SearchSamples{N}
    times::Vector{Float64}
    values::Matrix{ComplexF64}
    sample_shape::NTuple{N,Int}
end

_search_samples(sequence::UniformSequence) = _SearchSamples(
    sequence.times, sequence.values, sequence.sample_shape,
)

function _split_search_samples(sequence::UniformSequence, holdout_count::Int)
    training_count = length(sequence) - holdout_count
    training_count >= 3 || throw(ArgumentError(
        "holdout_count leaves fewer than three training samples",
    ))
    training = UniformSequence(
        copy(sequence.times[1:training_count]),
        copy(sequence.values[1:training_count, :]),
        sequence.sample_shape,
        sequence.t0,
        sequence.dt,
    )
    holdout = holdout_count == 0 ? nothing : _SearchSamples(
        copy(sequence.times[(training_count + 1):end]),
        copy(sequence.values[(training_count + 1):end, :]),
        sequence.sample_shape,
    )
    return training, holdout
end

struct _SearchWeightFit
    nodes::Vector{ComplexF64}
    poles::Vector{ComplexF64}
    relative_weights::Matrix{ComplexF64}
    absolute_weights::Matrix{ComplexF64}
    diagnostics::WeightLeastSquaresDiagnostics
end

function _search_weight_fit(sequence::UniformSequence,
                            nodes::AbstractVector{<:Number})
    isempty(nodes) && return nothing, ExponentialAttemptFailure(
        :empty_node_set, "exponential weight fitting needs at least one node",
    )
    any(iszero, nodes) && return nothing, ExponentialAttemptFailure(
        :zero_node, "node-to-exponent mapping received a zero node",
    )
    poles = ComplexF64.(-log.(nodes) ./ sequence.dt)
    all(pole -> isfinite(real(pole)) && isfinite(imag(pole)), poles) ||
        return nothing, ExponentialAttemptFailure(
            :nonfinite_poles, "node-to-exponent mapping produced nonfinite poles",
        )
    order = sortperm(poles; by=pole -> (real(pole), imag(pole)))
    ordered_poles = poles[order]
    ordered_nodes = ComplexF64.(nodes[order])
    relative_times = sequence.times .- sequence.t0
    design = exp.(-relative_times * transpose(ordered_poles))
    relative_weights = ComplexF64.(design \ sequence.values)
    all(weight -> isfinite(real(weight)) && isfinite(imag(weight)),
        relative_weights) ||
        return nothing, ExponentialAttemptFailure(
            :nonfinite_weights,
            "exponential least-squares fitting produced nonfinite weights",
        )
    absolute_weights = relative_weights .*
        reshape(exp.(ordered_poles .* sequence.t0), :, 1)
    all(weight -> isfinite(real(weight)) && isfinite(imag(weight)),
        absolute_weights) ||
        return nothing, ExponentialAttemptFailure(
            :nonfinite_weights,
            "shifted-origin weight conversion produced nonfinite weights",
        )
    singular_values = Float64.(svdvals(design))
    threshold = isempty(singular_values) ? 0.0 :
        max(64 * eps(Float64) * first(singular_values), eps(Float64))
    least_squares_rank = count(value -> value > threshold, singular_values)
    condition_number =
        isempty(singular_values) || last(singular_values) <= threshold ?
        Inf : first(singular_values) / last(singular_values)
    diagnostics = WeightLeastSquaresDiagnostics(
        least_squares_rank, condition_number, singular_values,
    )
    return _SearchWeightFit(
        ordered_nodes, ordered_poles, relative_weights, absolute_weights,
        diagnostics,
    ), nothing
end

function _prune_weights(policy::NoPruning, fit::_SearchWeightFit)
    norms = Float64[
        norm(view(fit.relative_weights, index, :))
        for index in axes(fit.relative_weights, 1)
    ]
    retained = collect(eachindex(fit.nodes))
    return PruningDiagnostics(
        policy, norms, 0.0, retained, PrunedMode[],
    )
end

function _prune_weights(policy::WeightNormPruning, fit::_SearchWeightFit)
    norms = Float64[
        norm(view(fit.relative_weights, index, :))
        for index in axes(fit.relative_weights, 1)
    ]
    scale = isempty(norms) ? 0.0 : maximum(norms)
    threshold = max(
        policy.atol + policy.rtol * scale,
        64 * eps(Float64) * scale,
    )
    retained = Int[
        index for index in eachindex(fit.nodes)
        if norms[index] > threshold
    ]
    rejected = PrunedMode[
        PrunedMode(
            index,
            fit.nodes[index],
            norms[index],
            :weight_norm_below_threshold,
        )
        for index in eachindex(fit.nodes)
        if norms[index] <= threshold
    ]
    return PruningDiagnostics(
        policy, norms, threshold, retained, rejected,
    )
end

function _score_search_fit(samples::_SearchSamples,
                           poles::Vector{ComplexF64},
                           relative_weights::Matrix{ComplexF64},
                           training_t0::Float64)
    relative_times = samples.times .- training_t0
    fitted = exp.(-relative_times * transpose(poles)) * relative_weights
    residual = fitted - samples.values
    row_errors = Float64[
        norm(view(residual, index, :))
        for index in axes(residual, 1)
    ]
    maximum_error = isempty(row_errors) ? 0.0 : maximum(row_errors)
    l2 = norm(residual)
    target_norm = norm(samples.values)
    relative_l2 = target_norm == 0 ? l2 : l2 / target_norm
    return ExponentialScore(
        maximum_error,
        l2,
        relative_l2,
        _channel_relative_errors(samples.values, fitted),
    )
end

function _fit_diagnostics(sequence::UniformSequence,
                          fit::_SearchWeightFit,
                          score::ExponentialScore,
                          estimate::NodeEstimate,
                          holdout::Union{Nothing,ExponentialScore})
    aliasing = any(
        abs(imag(pole)) >
        pi / sequence.dt * (1 + 100 * eps(Float64))
        for pole in fit.poles
    )
    unstable = findall(
        pole -> real(pole) < -sqrt(eps(Float64)), fit.poles,
    )
    return FitDiagnostics(
        sequence.t0,
        sequence.dt,
        length(sequence),
        sequence.sample_shape,
        estimate.common.resolved_rank,
        length(fit.poles),
        fit.diagnostics.rank,
        fit.diagnostics.condition_number,
        score.maximum_error,
        score.relative_l2,
        score.channel_relative_errors,
        holdout === nothing ? nothing : holdout.relative_l2,
        aliasing,
        _alias_distance(fit.poles, sequence.dt),
        unstable,
        _minimum_separation(fit.poles),
        _residue_growth(fit.absolute_weights, sequence.values),
    )
end

function _control_limit(::FirstControlled, attempted_rank::Int,
                        singular_values::Vector{Float64})
    isempty(singular_values) && throw(ArgumentError(
        "FirstControlled requires nonempty node-estimation singular values",
    ))
    next_value = attempted_rank < length(singular_values) ?
        singular_values[attempted_rank + 1] : 0.0
    return max(
        10 * next_value,
        64 * eps(Float64) * singular_values[1],
    )
end

_control_limit(::ExhaustiveSearch, attempted_rank::Int,
               singular_values::Vector{Float64}) = nothing

_attempt_control_limit(::ExhaustiveSearch, estimate::NodeEstimate,
                       attempted_rank::Int) = nothing
_attempt_control_limit(rule::FirstControlled, estimate::NodeEstimate,
                       attempted_rank::Int) =
    _control_limit(
        rule, attempted_rank, attempt_singular_values(estimate),
    )

_controlled(::ExhaustiveSearch, score::ExponentialScore, limit) = false
_controlled(::FirstControlled, score::ExponentialScore, limit::Float64) =
    score.maximum_error <= limit

_prefer_attempt(
    ::MinimumTrainingRelativeL2,
    candidate::ExponentialScore,
    incumbent::Union{Nothing,ExponentialScore},
) = incumbent === nothing ||
    candidate.relative_l2 < incumbent.relative_l2

function _failed_attempt(attempted_rank::Int, estimate::NodeEstimate)
    failure = estimate.outcome
    failure isa AbstractNodeFailure ||
        throw(ArgumentError("node attempt did not contain a failure outcome"))
    return DescendingRankAttempt(
        attempted_rank,
        estimate,
        0,
        nothing,
        0,
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        NodeAttemptFailure(failure),
    )
end

function _fit_failed_attempt(
    attempted_rank::Int,
    estimate::NodeEstimate,
    pre_pruning_count::Int,
    pruning,
    post_pruning_count::Int,
    least_squares_before,
    least_squares_after,
    failure::ExponentialAttemptFailure,
)
    return DescendingRankAttempt(
        attempted_rank,
        estimate,
        pre_pruning_count,
        pruning,
        post_pruning_count,
        least_squares_before,
        least_squares_after,
        nothing,
        nothing,
        nothing,
        failure,
    )
end

function _terminal_fit_failure(estimate::NodeEstimate,
                               failure::ExponentialAttemptFailure)
    failed_estimate = NodeEstimate(
        NodeEstimationFailure(failure.reason, failure.message),
        estimate.common,
        estimate.backend,
    )
    return failed_estimate
end

function _fit_exponential_search(
    search::DescendingRankSearch,
    training::UniformSequence,
    holdout::Union{Nothing,_SearchSamples},
)
    probe = estimate_nodes(search.estimator, training)
    if probe.outcome isa ZeroSequence
        history = DescendingRankHistory(
            0, search.selection, DescendingRankAttempt[], nothing,
        )
        if holdout !== nothing && !all(iszero, holdout.values)
            failed_estimate = NodeEstimate(
                NodeEstimationFailure(
                    :zero_training_nonzero_holdout,
                    "zero training samples cannot validate nonzero holdout samples",
                ),
                probe.common,
                probe.backend,
            )
            return ExponentialFit(
                FailedFit(failed_estimate),
                _empty_fit_diagnostics(training),
                history,
            )
        end
        outcome = ZeroFit(training.sample_shape, ComplexF64, probe)
        return ExponentialFit(
            outcome, _empty_fit_diagnostics(training), history,
        )
    end

    usable_rank = min(
        probe.common.evidence_rank, probe.common.geometry_rank,
    )
    requested_cap = probe.common.requested_rank
    initial_rank = if search.initial_rank !== nothing
        min(search.initial_rank, usable_rank)
    elseif requested_cap !== nothing
        min(requested_cap, usable_rank)
    else
        usable_rank
    end
    if initial_rank < 1
        probe.outcome isa AbstractNodeFailure || begin
            probe = NodeEstimate(
                NodeEstimationFailure(
                    :rank_unavailable,
                    "descending search found no usable positive candidate rank",
                ),
                probe.common,
                probe.backend,
            )
        end
        history = DescendingRankHistory(
            0, search.selection, DescendingRankAttempt[], nothing,
        )
        return ExponentialFit(
            FailedFit(probe), _empty_fit_diagnostics(training), history,
        )
    end

    attempts = DescendingRankAttempt[]
    selected_attempt = nothing
    selected_score = nothing
    terminal_estimate = probe
    for attempted_rank in initial_rank:-1:1
        estimator = candidate_estimator(search.estimator, attempted_rank)
        estimate = estimate_nodes(estimator, training)
        terminal_estimate = estimate
        if estimate.outcome isa AbstractNodeFailure
            push!(attempts, _failed_attempt(attempted_rank, estimate))
            continue
        elseif estimate.outcome isa ZeroSequence
            failure = ExponentialAttemptFailure(
                :unexpected_zero_sequence,
                "a rank candidate returned zero after a nonzero initial probe",
            )
            terminal_estimate = _terminal_fit_failure(estimate, failure)
            push!(
                attempts,
                _fit_failed_attempt(
                    attempted_rank, estimate, 0, nothing, 0,
                    nothing, nothing, failure,
                ),
            )
            continue
        end

        nodes = estimate.outcome.nodes
        before_fit, before_failure = _search_weight_fit(training, nodes)
        if before_failure !== nothing
            terminal_estimate = _terminal_fit_failure(
                estimate, before_failure,
            )
            push!(
                attempts,
                _fit_failed_attempt(
                    attempted_rank, estimate, length(nodes), nothing, 0,
                    nothing, nothing, before_failure,
                ),
            )
            continue
        end

        pruning = _prune_weights(search.pruning, before_fit)
        if isempty(pruning.retained_indices)
            failure = ExponentialAttemptFailure(
                :all_modes_pruned,
                "weight-norm pruning removed every identified mode",
            )
            terminal_estimate = _terminal_fit_failure(estimate, failure)
            push!(
                attempts,
                _fit_failed_attempt(
                    attempted_rank, estimate, length(before_fit.nodes),
                    pruning, 0, before_fit.diagnostics, nothing, failure,
                ),
            )
            continue
        end

        retained_nodes = before_fit.nodes[pruning.retained_indices]
        after_fit, after_failure = _search_weight_fit(
            training, retained_nodes,
        )
        if after_failure !== nothing
            terminal_estimate = _terminal_fit_failure(
                estimate, after_failure,
            )
            push!(
                attempts,
                _fit_failed_attempt(
                    attempted_rank, estimate, length(before_fit.nodes),
                    pruning, length(retained_nodes), before_fit.diagnostics,
                    nothing, after_failure,
                ),
            )
            continue
        end

        training_score = _score_search_fit(
            _search_samples(training),
            after_fit.poles,
            after_fit.relative_weights,
            training.t0,
        )
        holdout_score = holdout === nothing ? nothing :
            _score_search_fit(
                holdout,
                after_fit.poles,
                after_fit.relative_weights,
                training.t0,
            )
        control_limit = _attempt_control_limit(
            search.stop, estimate, attempted_rank,
        )
        controlled = _controlled(
            search.stop, training_score, control_limit,
        )
        weights = _restore_weights(
            after_fit.absolute_weights, training.sample_shape,
        )
        model = ExponentialSum(after_fit.poles, weights)
        diagnostics = _fit_diagnostics(
            training, after_fit, training_score, estimate, holdout_score,
        )
        outcome = IdentifiedAttempt(model, diagnostics, controlled)
        attempt = DescendingRankAttempt(
            attempted_rank,
            estimate,
            length(before_fit.nodes),
            pruning,
            length(after_fit.nodes),
            before_fit.diagnostics,
            after_fit.diagnostics,
            training_score,
            holdout_score,
            control_limit,
            outcome,
        )
        push!(attempts, attempt)
        if _prefer_attempt(
            search.selection, training_score, selected_score,
        )
            selected_attempt = length(attempts)
            selected_score = training_score
        end
        controlled && break
    end

    history = DescendingRankHistory(
        initial_rank, search.selection, attempts, selected_attempt,
    )
    if selected_attempt === nothing
        return ExponentialFit(
            FailedFit(terminal_estimate),
            _empty_fit_diagnostics(training),
            history,
        )
    end
    selected = attempts[selected_attempt]
    identified = selected.outcome
    identified isa IdentifiedAttempt ||
        error("selected descending-rank attempt is not identified")
    outcome = IdentifiedFit(identified.value, selected.node_estimate)
    return ExponentialFit(outcome, identified.diagnostics, history)
end

function fit_exponential_sum(search::DescendingRankSearch,
                             sequence::UniformSequence)
    training, holdout = _split_search_samples(
        sequence, search.holdout_count,
    )
    return _fit_exponential_search(search, training, holdout)
end

function fit_exponential_sum(search::DescendingRankSearch,
                             training::UniformSequence,
                             holdout::UniformSequence)
    search.holdout_count == 0 || throw(ArgumentError(
        "explicit holdout cannot be combined with holdout_count",
    ))
    training.sample_shape == holdout.sample_shape || throw(ArgumentError(
        "training and holdout samples must have the same shape",
    ))
    return _fit_exponential_search(
        search, training, _search_samples(holdout),
    )
end

fit_exponential_sum(search::DescendingRankSearch,
                    times::AbstractVector{<:Real}, samples) =
    fit_exponential_sum(search, UniformSequence(times, samples))

fit_exponential_sum(search::DescendingRankSearch,
                    series::CorrelatorSeries) =
    fit_exponential_sum(search, series.times, series.values)
