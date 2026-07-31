using GraftThermal: METTS, metts_statistics
using Random: Xoshiro
using Test

@testset "METTS sampling and statistics contract" begin
    representation = METTS(
        rng=Xoshiro(20260803), collapse_basis=:alternating,
        burnin=2, nsamples=8, thin=3)
    @test representation.burnin == 2
    @test representation.nsamples == 8
    @test representation.thin == 3
    @test_throws ArgumentError METTS(
        rng=Xoshiro(1), burnin=-1, nsamples=1, thin=1)
    @test_throws ArgumentError METTS(
        rng=Xoshiro(1), burnin=0, nsamples=0, thin=1)

    constant = metts_statistics(fill(2.0, 8))
    @test constant.mean == 2.0
    @test constant.variance == 0.0
    @test constant.stderr == 0.0
    @test constant.effective_samples == 8.0

    correlated = metts_statistics([0.0, 0.0, 1.0, 1.0, 2.0, 2.0]; maxlag=2)
    @test correlated.mean == 1.0
    @test correlated.variance ≈ 0.8
    @test correlated.tau_int > 0.5
    @test 1.0 <= correlated.effective_samples < correlated.nsamples
    @test correlated.stderr > 0.0
end
