@graft_testset "M2 Matsubara transforms" begin
    beta = 3.0
    epsilon = 0.7
    times = collect(range(0.0, beta; length=4001))
    green_tau = -exp.(-epsilon .* times) ./ (1 + exp(-beta * epsilon))
    green = CorrelatorSeries(times, ComplexF64.(green_tau), (; beta))
    transformed = matsubara_transform(
        green; statistics=:fermionic, indices=0:5)
    exact = [inv(im * omega - epsilon) for omega in transformed.frequencies]
    @test maximum(abs.(transformed.values .- exact)) < 1e-6
    @test transformed.metadata.quadrature == :trapezoid
    @test transformed.indices == collect(0:5)

    constant = CorrelatorSeries(times, ones(ComplexF64, length(times)), (; beta))
    bosonic = matsubara_transform(
        constant; statistics=:bosonic, indices=0:4)
    @test bosonic.values[1] ≈ beta atol=1e-12
    @test maximum(abs, bosonic.values[2:end]) < 1e-12

    nodes = [0.2, 0.8, 1.6, 2.7]
    explicit = matsubara_transform(
        CorrelatorSeries(nodes, ones(ComplexF64, 4), (;));
        statistics=:bosonic, indices=[0], beta,
        weights=[0.4, 0.7, 1.1, 0.8])
    @test explicit.values == ComplexF64[beta]
    @test explicit.metadata.quadrature == :explicit
    @test_throws ArgumentError matsubara_transform(
        green; statistics=:classical)
end
