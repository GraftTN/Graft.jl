using LinearAlgebra: norm

function grid_interaction(orbitals, kernel)
    norb = size(orbitals, 2)
    density = Matrix{promote_type(eltype(orbitals), eltype(kernel))}(
        undef, size(orbitals, 1), norb^2)
    for j in 1:norb, i in 1:norb
        @views density[:, (j - 1) * norb + i] .=
            conj.(orbitals[:, i]) .* orbitals[:, j]
    end
    return reshape(density' * kernel * density, norb, norb, norb, norb)
end

@testset "ISDF-THC reconstruction" begin
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
    reference = grid_interaction(orbitals, kernel)

    factorization = GraftTTNOBuild.isdf_thc(
        orbitals, kernel; rtol=1e-12, assess=true)
    @test factorization isa GraftTTNOBuild.THCFactorization
    @test factorization.report.selected_rank == 3
    @test factorization.report.pair_density_relative_error < 1e-12
    @test factorization.report.interaction_relative_error < 1e-12
    @test GraftTTNOBuild.reconstruct_thc(factorization) ≈
        reference atol=1e-12 rtol=1e-12

    compact = GraftTTNOBuild.isdf_thc(
        orbitals, kernel; rank=2, rtol=1e-12, assess=true)
    @test compact.report.selected_rank == 2
    @test norm(GraftTTNOBuild.reconstruct_thc(compact) - reference) /
        norm(reference) ≈ compact.report.interaction_relative_error

    @test_throws ArgumentError GraftTTNOBuild.isdf_thc(
        orbitals, kernel; rank=0)
    @test_throws DimensionMismatch GraftTTNOBuild.isdf_thc(
        orbitals, kernel[1:3, 1:3])
end
