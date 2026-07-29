using Test
using Graft
using Graft.Spectral
using Graft.TestUtils
using Graft.Backend: ℂ
using LinearAlgebra: I, dot, norm
using Random: Xoshiro

isdefined(@__MODULE__, Symbol("@graft_testset")) || include("test_harness.jl")

@graft_testset "M1 exponential-sum node estimators" begin
    times = collect(range(0.0, 4.0; length=41))
    dt = times[2] - times[1]
    nodes = ComplexF64[0.82 + 0.17im, 0.67 - 0.31im]
    weights = ComplexF64[1.2 - 0.3im, -0.4 + 0.8im]
    values = ComplexF64[
        sum(weights[k] * nodes[k]^n for k in eachindex(nodes))
        for n in 0:(length(times) - 1)
    ]
    node_order(values) = sort(values; by=value -> (real(value), imag(value)))

    @test nodes[1] != conj(nodes[2])
    for estimator in (
        HankelDMD(rank=StrictRank(2)),
        ESPRIT(rank=StrictRank(2)),
        LeftSubspaceESPRIT(rank=StrictRank(2)),
    )
        estimate = estimate_nodes(estimator, times, values)
        @test estimate.outcome isa IdentifiedNodes
        @test estimate.common.resolved_rank == 2
        @test node_order(estimate.outcome.nodes) ≈ node_order(nodes) atol=1e-10
    end
    left_estimate = estimate_nodes(
        LeftSubspaceESPRIT(rank=StrictRank(2)), times, values,
    )
    @test left_estimate.backend.sample_layout === :time_major_block_rows
    @test left_estimate.backend.subspace_layout ===
          :left_time_shift_by_channel_block
    @test_throws ArgumentError estimate_nodes(
        HankelDMD(rank=StrictRank(1)),
        [0.0, 0.1, 0.25],
        values[1:3],
    )

    matrix_weights = [
        ComplexF64[1.0 0.2im; -0.3im 0.7],
        ComplexF64[0.1 + 0.2im -0.4; 0.6 0.3im],
    ]
    matrix_values = [
        sum(matrix_weights[k] * nodes[k]^n for k in eachindex(nodes))
        for n in 0:(length(times) - 1)
    ]
    matrix_fit = fit_exponential_sum(
        HankelDMD(rank=StrictRank(2), reduction=AllComponents()),
        times,
        matrix_values,
    )
    @test matrix_fit.outcome isa IdentifiedFit
    matrix_reduction = matrix_fit.outcome.node_estimate.common.reduction
    @test matrix_reduction.policy isa AllComponents
    @test matrix_reduction.original_channels == 4
    @test matrix_reduction.reduced_channels == 4
    @test maximum(norm.(evaluate(matrix_fit, times) .- matrix_values)) < 1e-9

    diagonal_weights = [
        ComplexF64[1.0 0.0; 0.0 0.2im],
        ComplexF64[0.3 - 0.1im 0.0; 0.0 0.7],
    ]
    diagonal_values = [
        sum(diagonal_weights[k] * nodes[k]^n for k in eachindex(nodes))
        for n in 0:(length(times) - 1)
    ]
    diagonal_estimate = estimate_nodes(
        ESPRIT(rank=StrictRank(2), reduction=DeclaredDiagonal()),
        times,
        diagonal_values,
    )
    @test diagonal_estimate.outcome isa IdentifiedNodes
    @test diagonal_estimate.common.reduction.original_channels == 4
    @test diagonal_estimate.common.reduction.reduced_channels == 2
    @test diagonal_estimate.common.reduction.discarded_norm == 0.0
    @test node_order(diagonal_estimate.outcome.nodes) ≈
          node_order(nodes) atol=1e-10

    invalid_diagonal = estimate_nodes(
        HankelDMD(rank=StrictRank(2), reduction=DeclaredDiagonal()),
        times,
        matrix_values,
    )
    @test invalid_diagonal.outcome isa NodeEstimationFailure
    @test invalid_diagonal.outcome.reason == :diagonal_declaration_violated

    trace_values = [
        ComplexF64[value 0.0; 0.0 -value] for value in values
    ]
    erased = estimate_nodes(
        ESPRIT(rank=StrictRank(1), reduction=TraceReduction()),
        times,
        trace_values,
    )
    @test erased.outcome isa ReductionErasedSignal
    @test erased.outcome.original_norm > 0
    @test erased.outcome.reduced_norm == 0
    @test !(erased.outcome isa ZeroSequence)

    rank_one_values = ComplexF64[nodes[1]^n for n in 0:(length(times) - 1)]
    evidence_policy = AbsoluteThresholdRank(1e-10)
    strict = estimate_nodes(
        HankelDMD(rank=StrictRank(2, evidence_policy)),
        times,
        rank_one_values,
    )
    clamped = estimate_nodes(
        HankelDMD(rank=ClampedRank(2, evidence_policy)),
        times,
        rank_one_values,
    )
    @test strict.outcome isa NodeEstimationFailure
    @test strict.outcome.reason == :rank_exceeds_evidence
    @test strict.common.requested_rank == 2
    @test strict.common.evidence_rank == 1
    @test clamped.outcome isa IdentifiedNodes
    @test clamped.common.requested_rank == 2
    @test clamped.common.resolved_rank == 1
    @test clamped.common.clamped

    zero_values = [zeros(ComplexF64, 2, 2) for _ in times]
    zero_estimate = estimate_nodes(
        HankelDMD(rank=StrictRank(2)),
        times,
        zero_values,
    )
    zero_fit = fit_exponential_sum(
        HankelDMD(rank=StrictRank(2)),
        times,
        zero_values,
    )
    @test zero_estimate.outcome isa ZeroSequence
    @test zero_estimate.common.resolved_rank == 0
    @test zero_fit.outcome isa ZeroFit
    @test evaluate(zero_fit, first(times)) == zeros(ComplexF64, 2, 2)

    poles = -log.(nodes) ./ dt
    pure_sum = ExponentialSum(poles, weights)
    @test !hasproperty(pure_sum, :diagnostics)
    @test evaluate(pure_sum, times) ≈ values atol=1e-10

    shifted_times = times .+ 3.7
    matrix_sum = ExponentialSum(poles, matrix_weights)
    shifted_values = evaluate(matrix_sum, shifted_times)
    shifted_fit = fit_exponential_sum(
        HankelDMD(rank=StrictRank(2), reduction=TraceReduction()),
        shifted_times,
        shifted_values,
    )
    @test shifted_fit.outcome isa IdentifiedFit
    @test shifted_fit.diagnostics.t0 == first(shifted_times)
    shifted_reduction = shifted_fit.outcome.node_estimate.common.reduction
    @test shifted_reduction.original_channels == 4
    @test shifted_reduction.reduced_channels == 1
    @test node_order(shifted_fit.outcome.value.poles) ≈
          node_order(poles) atol=1e-9
    expected_order = sortperm(poles; by=pole -> (real(pole), imag(pole)))
    @test maximum(norm.(
        shifted_fit.outcome.value.weights .- matrix_weights[expected_order],
    )) < 1e-9
    @test maximum(norm.(
        evaluate(shifted_fit, shifted_times) .- shifted_values,
    )) < 1e-9
    @test maximum(norm.(
        evaluate(shifted_fit, times) .- matrix_values,
    )) < 1e-9

    vector_weights = [
        ComplexF64[1.0 + 0.2im, -0.4],
        ComplexF64[0.3im, 0.7 - 0.1im],
    ]
    vector_values = [
        sum(vector_weights[index] * nodes[index]^power
            for index in eachindex(nodes))
        for power in 0:(length(times) - 1)
    ]
    vector_fit = fit_exponential_sum(
        ESPRIT(rank=StrictRank(2)), times, vector_values,
    )
    @test vector_fit.outcome.value.weights isa Vector{Vector{ComplexF64}}
    @test maximum(norm.(evaluate(vector_fit, times) .- vector_values)) <
          1e-9

    tolerated_values = copy(matrix_values)
    tolerated_fit = fit_exponential_sum(
        ESPRIT(
            rank=StrictRank(2),
            reduction=DeclaredDiagonal(
                atol=sqrt(sum(sum(abs2, value) for value in tolerated_values)),
            ),
        ),
        times,
        tolerated_values,
    )
    @test tolerated_fit.outcome isa IdentifiedFit
    @test maximum(norm.(evaluate(tolerated_fit, times) .- tolerated_values)) <
          1e-9
    @test tolerated_fit.outcome.node_estimate.common.reduction.reduced_channels == 2
    @test tolerated_fit.outcome.node_estimate.common.reduction.flattening ===
          :column_major

    geometry_times = collect(0.0:1.0:3.0)
    geometry_nodes = ComplexF64[0.9, 0.7, 0.5]
    geometry_values = [
        ComplexF64[
            geometry_nodes[1]^power,
            geometry_nodes[2]^power,
            geometry_nodes[3]^power,
        ]
        for power in 0:3
    ]
    geometry_strict = estimate_nodes(
        ESPRIT(rank=StrictRank(3)), geometry_times, geometry_values,
    )
    geometry_clamped = estimate_nodes(
        ESPRIT(rank=ClampedRank(3)), geometry_times, geometry_values,
    )
    @test geometry_strict.outcome.reason == :rank_exceeds_geometry
    @test geometry_strict.common.evidence_rank == 3
    @test geometry_strict.common.geometry_rank == 2
    @test geometry_clamped.common.clamp_reason == :geometry
    @test geometry_clamped.common.resolved_rank == 2

    tiny_values = fill(ComplexF64(1e-13), length(times))
    for estimator in (
        HankelDMD(zero=ToleranceZero(atol=1e-12, reference_scale=1.0)),
        ESPRIT(zero=ToleranceZero(atol=1e-12, reference_scale=1.0)),
        ARLeastSquares(zero=ToleranceZero(atol=1e-12, reference_scale=1.0)),
    )
        @test estimate_nodes(estimator, times, tiny_values).outcome isa
              ZeroSequence
    end

    large_origin = 1e16
    @test_throws ArgumentError UniformSequence(
        [large_origin, nextfloat(large_origin), nextfloat(large_origin)],
        ComplexF64[1, 2, 3],
    )
    @test_throws ArgumentError UniformSequence(
        [1e12, 1e12 + 0.1, 1e12 + 0.3],
        ComplexF64[1, 2, 3],
    )
    @test_throws ArgumentError UniformSequence(
        Float64[0, 1, 1], ones(ComplexF64, 3, 1), (), 0.0, 1.0,
    )
    @test_throws ArgumentError UniformSequence(
        Float64[0, 1, 2],
        reshape(ComplexF64[1, NaN, 1], 3, 1),
        (),
        0.0,
        1.0,
    )
    @test_throws ArgumentError UniformSequence(
        Float64[0, 1, 2], ones(ComplexF64, 3, 1), (2,), 0.0, 1.0,
    )
    @test_throws ArgumentError UniformSequence(
        Float64[0, 1, 2], ones(ComplexF64, 3, 1), (), 1.0, 1.0,
    )

    overflowing_times = collect(1000.0:0.1:1002.0)
    overflowing_values = exp.(-(overflowing_times .- first(overflowing_times)))
    overflowing_fit = fit_exponential_sum(
        ESPRIT(rank=StrictRank(1)),
        overflowing_times,
        overflowing_values,
    )
    @test overflowing_fit.outcome isa FailedFit
    @test overflowing_fit.outcome.node_estimate.outcome.reason ==
          :nonfinite_weights

    @test matrix_fit.diagnostics.sample_shape == (2, 2)
    @test matrix_fit.diagnostics.estimator_rank == 2
    @test matrix_fit.diagnostics.retained_order == 2
    @test isfinite(matrix_fit.diagnostics.minimum_pole_separation)
    @test isfinite(matrix_fit.diagnostics.residue_growth)
    @test matrix_fit.outcome.node_estimate.backend.sample_layout ===
          :time_major_block_rows
    @test matrix_fit.outcome.node_estimate.backend.shift_residual < 1e-10
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
    ar_fit = fit_exponential_sum(
        ARLeastSquares(rank=StrictRank(2)),
        times,
        multiexponential,
    )
    @test ar_fit.outcome isa IdentifiedFit
    @test ar_fit.diagnostics.relative_l2 < 1e-10
    @test evaluate(ar_fit, times) ≈ multiexponential atol=1e-10
    @test ar_fit.outcome.node_estimate.backend.coefficients ≈
          multimodel.coefficients atol=1e-12
    @test isempty(ar_fit.outcome.node_estimate.backend.unstable_nodes)
    @test isempty(ar_fit.outcome.node_estimate.backend.modifications)

    shifted_times = times .+ 2.5
    shifted_values = ComplexF64[
        1.3 * exp(-0.2 * t) - 0.4im * exp(-(0.1 + 0.3im) * t)
        for t in shifted_times
    ]
    shifted_model = linear_prediction(
        shifted_times, shifted_values; order=2)
    shifted_fit = fit_exponential_sum(
        ARLeastSquares(rank=StrictRank(2)),
        shifted_times,
        shifted_values,
    )
    @test shifted_model.t0 == first(shifted_times)
    @test shifted_fit.diagnostics.t0 == first(shifted_times)
    @test shifted_fit.outcome.node_estimate.backend.coefficients ≈
          shifted_model.coefficients atol=1e-12
    @test evaluate(shifted_fit, shifted_times) ≈ shifted_values atol=1e-10

    unstable_values = ComplexF64[1.2^(index - 1) + 0.2 * 0.8^(index - 1)
                                 for index in eachindex(times)]
    projected_fit = fit_exponential_sum(
        ARLeastSquares(
            rank=StrictRank(2), modes=ProjectUnitCircle(),
        ),
        times,
        unstable_values,
    )
    rejected = estimate_nodes(
        ARLeastSquares(
            rank=StrictRank(2), modes=RejectOutsideUnitCircle(),
        ),
        times,
        unstable_values,
    )
    dropped_fit = fit_exponential_sum(
        ARLeastSquares(
            rank=StrictRank(2), modes=DropOutsideUnitCircle(),
        ),
        times,
        unstable_values,
    )
    @test projected_fit.outcome isa IdentifiedFit
    @test only(
        projected_fit.outcome.node_estimate.backend.modifications,
    ).action === :projected
    @test rejected.outcome.reason == :mode_rejected
    @test only(rejected.backend.modifications).result === nothing
    @test dropped_fit.outcome isa IdentifiedFit
    @test dropped_fit.diagnostics.retained_order == 1
    @test only(dropped_fit.outcome.node_estimate.backend.modifications).action ===
          :dropped

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
