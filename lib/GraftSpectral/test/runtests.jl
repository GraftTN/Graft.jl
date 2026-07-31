using GraftSpectral
using Test

@testset "descending-rank exponential fit" begin
    times = collect(0.0:0.1:2.0)
    nodes = ComplexF64[0.91 + 0.08im, 0.76 - 0.17im]
    weights = ComplexF64[1.1 - 0.2im, -0.35 + 0.6im]
    samples = ComplexF64[
        sum(weights[index] * nodes[index]^power for index in eachindex(nodes))
        for power in 0:(length(times) - 1)
    ]
    search = DescendingRankSearch(
        ESPRIT(rank=ClampedRank(4, RelativeThresholdRank(1e-10)));
        pruning=NoPruning(),
        stop=ExhaustiveSearch(),
        selection=MinimumTrainingRelativeL2(),
        holdout_count=3,
    )

    fit = fit_exponential_sum(search, times, samples)

    @test fit.outcome isa IdentifiedFit
    @test fit.attempts isa DescendingRankHistory
    @test [attempt.attempted_rank for attempt in fit.attempts.attempts] == [2, 1]
    @test fit.attempts.selected_attempt == 1
    @test fit.diagnostics.holdout_relative_l2 < 1e-10
    @test maximum(abs, evaluate(fit, times) - samples) < 1e-10
end
