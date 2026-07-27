using Graft
using LinearAlgebra: norm

isdefined(@__MODULE__, Symbol("@graft_testset")) || include("test_harness.jl")

function _grid_interaction(orbitals, kernel)
    density = Matrix{promote_type(eltype(orbitals), eltype(kernel))}(
        undef, size(orbitals, 1), size(orbitals, 2)^2)
    norb = size(orbitals, 2)
    for j in 1:norb, i in 1:norb
        @views density[:, (j - 1) * norb + i] .=
            conj.(orbitals[:, i]) .* orbitals[:, j]
    end
    return reshape(density' * kernel * density, norb, norb, norb, norb)
end

@graft_testset "M5 ISDF-THC interaction pre-factorization" begin
    orbitals = [
        1.0  0.2
        0.4 -0.7
       -0.3  0.9
        0.8  0.5
    ]
    A = [
        1.0  0.2 -0.1  0.0
        0.0  0.9  0.3 -0.2
        0.2 -0.1  0.8  0.1
        0.1  0.0  0.2  0.7
    ]
    kernel = A * A'
    reference = _grid_interaction(orbitals, kernel)

    exact = isdf_thc(orbitals, kernel; rtol=1e-12, assess=true)
    @test exact.report.selected_rank == 3
    @test exact.report.pair_density_rank == 3
    @test length(exact.report.interpolation_points) == 3
    @test allunique(exact.report.interpolation_points)
    @test exact.report.pair_density_relative_error < 1e-12
    @test exact.report.interaction_relative_error < 1e-12
    @test exact.report.interaction_max_abs_error < 1e-12
    @test reconstruct_thc(exact) ≈ reference atol=1e-12 rtol=1e-12
    @test reshape(reconstruct_thc(exact; tensor=false), size(reference)) ≈
          reference atol=1e-12 rtol=1e-12
    @test exact.coupling ≈ exact.coupling'

    compact = isdf_thc(orbitals, kernel; rank=2, rtol=1e-12, assess=true)
    @test compact.report.selected_rank == 2
    @test compact.report.pair_density_relative_error > 0
    @test compact.report.interaction_relative_error > 0
    @test norm(reconstruct_thc(compact) - reference) / norm(reference) ≈
          compact.report.interaction_relative_error

    unchecked = isdf_thc(orbitals, kernel; rank=2)
    @test isnothing(unchecked.report.interaction_relative_error)
    @test isnothing(unchecked.report.interaction_max_abs_error)

    X = [
        1.0  0.2  0.7
        0.3 -0.8  0.5
       -0.4  0.6  0.9
    ]
    seed_coupling = [
        1.2  0.1 -0.2
        0.1  0.8  0.3
       -0.2  0.3  1.5
    ]
    seed_report = THCReport(
        0, 3, 3, 3, Int[], 0.0, nothing, nothing)
    seed = THCFactorization(X, seed_coupling, seed_report)
    dense = reconstruct_thc(seed)
    fitted = fit_thc(dense, X; rtol=1e-12)
    @test isnothing(fitted.report.pair_density_relative_error)
    @test fitted.report.interaction_relative_error < 1e-12
    @test reconstruct_thc(fitted) ≈ dense atol=1e-12 rtol=1e-12

    complex_orbitals = complex.(orbitals, reverse(orbitals; dims=1) ./ 7)
    complex_kernel = complex.(kernel)
    complex_reference = _grid_interaction(complex_orbitals, complex_kernel)
    complex_fit = isdf_thc(
        complex_orbitals, complex_kernel; rtol=1e-12, assess=true)
    @test reconstruct_thc(complex_fit) ≈
          complex_reference atol=2e-12 rtol=2e-12

    @test_throws ArgumentError isdf_thc(orbitals, kernel; rank=0)
    @test_throws ArgumentError isdf_thc(orbitals, kernel; rtol=1.0)
    @test_throws DimensionMismatch isdf_thc(orbitals, kernel[1:3, 1:3])
    nonsymmetric = copy(kernel)
    nonsymmetric[1, 2] += 0.1
    @test_throws ArgumentError isdf_thc(orbitals, nonsymmetric)
    zero_orbitals = zeros(size(orbitals))
    @test_throws ArgumentError isdf_thc(zero_orbitals, kernel)
    @test_throws DimensionMismatch fit_thc(zeros(2, 2, 2, 3), X)
    nonhermitian = copy(dense)
    nonhermitian[1, 1, 1, 2] += 0.1
    @test_throws ArgumentError fit_thc(nonhermitian, X)
    bad_report = THCReport{Float64}(
        0, 2, 3, 3, Int[], nothing, nothing, nothing)
    @test_throws DimensionMismatch THCFactorization(
        X, seed_coupling, bad_report)
    @test_throws DimensionMismatch THCFactorization(
        X, seed_coupling[1:2, 1:2], seed_report)
end
