using Test
using Graft
using Graft.Spectral
using Graft.TestUtils
using Graft.Backend: ℂ
using LinearAlgebra: I, dot, norm
using Random: Xoshiro

isdefined(@__MODULE__, Symbol("@graft_testset")) || include("test_harness.jl")

@graft_testset "M1 exponential sums and ESPRIT" begin
    times = collect(range(0.0, 4.0; length=41))
    poles = ComplexF64[0.2 + 0.7im, 0.55 - 1.1im]
    weights = ComplexF64[1.2 - 0.3im, -0.4 + 0.8im]
    values = [sum(weights .* exp.(-poles .* t)) for t in times]

    fit = esprit(times, values; M=2)
    @test fit.diagnostics.chosen_rank == 2
    @test fit.diagnostics.l2err < 1e-10
    @test maximum(abs.(evaluate(fit, times) .- values)) < 1e-10
    @test sort(fit.poles; by=z -> (real(z), imag(z))) ≈
          sort(poles; by=z -> (real(z), imag(z))) atol=1e-9
    @test rank_from_svals([10.0, 1.0, 1e-12]; err=1e-8) == 2
    @test rank_from_svals([10.0, 1.0, 1e-12]; M=1) == 1
    @test_throws ArgumentError esprit([0.0, 0.1, 0.25], values[1:3]; M=1)

    matrix_weights = [
        ComplexF64[1.0 0.2im; -0.3im 0.7],
        ComplexF64[0.1 + 0.2im -0.4; 0.6 0.3im],
    ]
    matrix_values = [
        sum(matrix_weights[k] * exp(-poles[k] * t) for k in eachindex(poles))
        for t in times
    ]
    matrix_fit = esprit(times, matrix_values; M=2, matrix_mode=:stacked)
    @test matrix_fit.diagnostics.nchannels == 4
    @test maximum(norm.(evaluate(matrix_fit, times) .- matrix_values)) < 1e-9

    trace_fit = esprit(times, matrix_values; M=2, matrix_mode=:trace)
    @test maximum(norm.(evaluate(trace_fit, times) .- matrix_values)) < 1e-8

    shifted_times = times .+ 3.7
    shifted_values = [
        sum(weights .* exp.(-poles .* t)) for t in shifted_times
    ]
    shifted_fit = esprit(shifted_times, shifted_values; M=2)
    @test evaluate(shifted_fit, shifted_times) ≈ shifted_values atol=1e-9
    @test_throws ArgumentError rank_from_svals([1.0, 2.0])
end

@graft_testset "M1 linear prediction" begin
    times = collect(0.0:0.1:2.0)
    lambda = 0.96 * exp(0.23im)
    values = ComplexF64[lambda^(i - 1) for i in eachindex(times)]
    model = linear_prediction(times, values; order=1)
    future = predict(model, 8)
    expected = ComplexF64[lambda^(length(values) + i - 1) for i in 1:8]
    @test future ≈ expected atol=1e-11
    @test model.diagnostics.l2err < 1e-12
    @test isempty(predict(model, 0))
    @test_throws ArgumentError linear_prediction(times, values; order=0)

    lambda2 = 0.91 * exp(-0.41im)
    multiexponential = ComplexF64[
        1.3 * lambda^(i - 1) - 0.4im * lambda2^(i - 1)
        for i in eachindex(times)
    ]
    multimodel = linear_prediction(times, multiexponential; order=2)
    poles = exponential_sum(multimodel)
    @test poles.diagnostics.pole_fit_error < 1e-10
    @test evaluate(poles, times) ≈ multiexponential atol=1e-10
    @test isempty(poles.diagnostics.unstable_roots)

    shifted_times = times .+ 2.5
    shifted_values = ComplexF64[
        1.3 * exp(-0.2 * t) - 0.4im * exp(-(0.1 + 0.3im) * t)
        for t in shifted_times
    ]
    shifted_model = linear_prediction(
        shifted_times, shifted_values; order=2)
    shifted_sum = exponential_sum(shifted_model)
    @test shifted_model.t0 == first(shifted_times)
    @test evaluate(shifted_sum, shifted_times) ≈ shifted_values atol=1e-10

    real_model = linear_prediction(
        times, exp.(-0.3 .* times); order=1)
    @test predict(real_model, 2) ≈
        exp.(-0.3 .* [times[end] + 0.1, times[end] + 0.2])
end

@graft_testset "M1 complex-time Krylov Gram solve" begin
    H = ComplexF64[0.0 0.0; 0.0 2.0]
    snapshots = [
        ComplexF64[1.0, 1.0] / sqrt(2),
        ComplexF64[1.0, exp(-0.3im)] / sqrt(2),
        ComplexF64[1.0, exp(-0.6im)] / sqrt(2),
    ]
    S = ComplexF64[dot(a, b) for a in snapshots, b in snapshots]
    HM = ComplexF64[dot(a, H * b) for a in snapshots, b in snapshots]
    result = complex_time_krylov(S, HM; rtol=1e-12)
    @test result.diagnostics.retained_rank == 2
    @test real.(result.values) ≈ [0.0, 2.0] atol=1e-10
    @test sum(result.weights) ≈ 1.0
    @test maximum(result.residuals) < 1e-10

    duplicate = complex_time_krylov(S[[1, 1], [1, 1]], HM[[1, 1], [1, 1]])
    @test duplicate.diagnostics.retained_rank == 1
    @test_throws ArgumentError complex_time_krylov(
        ComplexF64[1 2; 2 1], Matrix{ComplexF64}(I, 2, 2))

    spin = spin_ops()
    topo = mps_topology(1)
    phys = Dict(:site1 => spin.P)
    operator = ttno_from_opsum(
        OpSum() + Term(1.0, SiteOp(:site1, :Z, spin.Z)),
        topo, phys; hermitian=true)
    psi = random_ttns(Xoshiro(26060729), ComplexF64, topo, phys, ℂ^1)
    flipped = apply_local(psi, spin.X, :site1)
    network_result = complex_time_krylov([psi, flipped], operator)
    @test real.(network_result.values) ≈ [-1.0, 1.0] atol=1e-10
    @test maximum(network_result.residuals) < 1e-10
end
