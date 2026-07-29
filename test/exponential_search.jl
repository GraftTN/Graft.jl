using Test
using Graft
using LinearAlgebra: norm
using Random: Xoshiro

@testset "M1 descending-rank exponential search" begin
    times = collect(0.0:0.1:2.0)
    nodes = ComplexF64[0.91 + 0.08im, 0.76 - 0.17im]
    weights = ComplexF64[1.1 - 0.2im, -0.35 + 0.6im]
    samples = ComplexF64[
        sum(weights[index] * nodes[index]^power
            for index in eachindex(nodes))
        for power in 0:(length(times) - 1)
    ]

    exhaustive = DescendingRankSearch(
        ESPRIT(
            rank=ClampedRank(4, RelativeThresholdRank(1e-10)),
        );
        pruning=NoPruning(),
        stop=ExhaustiveSearch(),
        selection=MinimumTrainingRelativeL2(),
        holdout_count=3,
    )
    fit = fit_exponential_sum(exhaustive, times, samples)
    @test fit.outcome isa IdentifiedFit
    @test DescendingRankSearch(
        ESPRIT(rank=ClampedRank(2)),
    ).stop isa ExhaustiveSearch
    @test fit.attempts isa DescendingRankHistory
    @test fit.attempts.selection isa MinimumTrainingRelativeL2
    @test [attempt.attempted_rank for attempt in fit.attempts.attempts] ==
          [2, 1]
    @test fit.attempts.selected_attempt == 1
    @test fit.attempts.attempts[1].holdout_error !== nothing
    @test fit.attempts.attempts[1].holdout_error.relative_l2 < 1e-10
    @test fit.diagnostics.holdout_relative_l2 < 1e-10
    @test evaluate(fit, times) ≈ samples atol=1e-10

    controlled_search = DescendingRankSearch(
        ESPRIT(
            rank=ClampedRank(4, RelativeThresholdRank(1e-10)),
        );
        pruning=NoPruning(),
        stop=FirstControlled(),
    )
    controlled_fit = fit_exponential_sum(controlled_search, times, samples)
    first_attempt = first(controlled_fit.attempts.attempts)
    singular_values = attempt_singular_values(first_attempt.node_estimate)
    next_value = first_attempt.attempted_rank < length(singular_values) ?
        singular_values[first_attempt.attempted_rank + 1] : 0.0
    @test first_attempt.control_limit ==
          max(10 * next_value,
              64 * eps(Float64) * first(singular_values))
    @test length(controlled_fit.attempts.attempts) == 1
    @test first_attempt.outcome.controlled
    @test controlled_fit.attempts.selected_attempt == 1

    small_weight = 1e-10 + 0im
    three_nodes = ComplexF64[nodes..., 0.53 + 0.11im]
    three_weights = ComplexF64[weights..., small_weight]
    three_samples = ComplexF64[
        sum(three_weights[index] * three_nodes[index]^power
            for index in eachindex(three_nodes))
        for power in 0:(length(times) - 1)
    ]
    pruned_fit = fit_exponential_sum(
        DescendingRankSearch(
            ESPRIT(rank=StrictRank(3));
            initial_rank=3,
            pruning=WeightNormPruning(rtol=1e-8),
            stop=ExhaustiveSearch(),
        ),
        times,
        three_samples,
    )
    @test pruned_fit.outcome isa IdentifiedFit
    rank_three = first(pruned_fit.attempts.attempts)
    @test rank_three.pre_pruning_node_count == 3
    @test rank_three.post_pruning_node_count == 2
    @test length(rank_three.pruning.rejected_modes) == 1
    @test norm(evaluate(pruned_fit, times) - three_samples) <
          1e-8

    zero_then_holdout = zeros(ComplexF64, 8)
    zero_then_holdout[end] = 1
    invalid_holdout = fit_exponential_sum(
        DescendingRankSearch(
            ESPRIT(rank=ClampedRank(2));
            holdout_count=2,
        ),
        collect(0.0:1.0:7.0),
        zero_then_holdout,
    )
    @test invalid_holdout.outcome isa FailedFit
    @test invalid_holdout.outcome.node_estimate.outcome.reason ==
          :zero_training_nonzero_holdout
    @test isempty(invalid_holdout.attempts.attempts)

    zero_fit = fit_exponential_sum(
        DescendingRankSearch(
            ARLeastSquares(rank=ClampedRank(3));
            holdout_count=2,
        ),
        collect(0.0:1.0:7.0),
        zeros(ComplexF64, 8),
    )
    @test zero_fit.outcome isa ZeroFit
    @test isempty(zero_fit.attempts.attempts)
    @test evaluate(zero_fit, 10.0) == 0

    ar_candidate = candidate_estimator(
        ARLeastSquares(
            rank=ClampedRank(4, RelativeThresholdRank(1e-8)),
            regularization=1e-12,
            modes=ProjectUnitCircle(),
        ),
        2,
    )
    @test ar_candidate.rank isa StrictRank
    @test ar_candidate.rank.requested == 2
    @test ar_candidate.regularization == 1e-12
    @test ar_candidate.modes isa ProjectUnitCircle

    generator = Xoshiro(20260729)
    noisy_samples = samples .+
        1e-8 .* randn(generator, ComplexF64, length(samples))
    noisy_ar = estimate_nodes(
        ARLeastSquares(rank=StrictRank(2)), times, noisy_samples,
    )
    @test noisy_ar.outcome isa IdentifiedNodes
    @test length(noisy_ar.backend.evidence_singular_values) > 2
    @test noisy_ar.backend.evidence_singular_values[3] > 0
    @test attempt_singular_values(noisy_ar)[3] ==
          noisy_ar.backend.evidence_singular_values[3]

    unstable_samples = ComplexF64[1.2^(index - 1)
                                  for index in eachindex(times)]
    total_failure = fit_exponential_sum(
        DescendingRankSearch(
            ARLeastSquares(
                rank=ClampedRank(1),
                modes=DropOutsideUnitCircle(),
            );
            initial_rank=1,
            stop=ExhaustiveSearch(),
        ),
        times,
        unstable_samples,
    )
    @test total_failure.outcome isa FailedFit
    @test total_failure.attempts.selected_attempt === nothing
    @test length(total_failure.attempts.attempts) == 1
    @test first(total_failure.attempts.attempts).outcome isa
          NodeAttemptFailure
end
