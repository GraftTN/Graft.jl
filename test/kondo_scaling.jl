@graft_testset "M1 Kondo scaling analysis" begin
    interactions = [8.0, 10.0, 12.0, 15.0]
    temperatures = [1 / 1024, 1 / 512, 1 / 256, 1 / 128]
    expected_exponent = 0.21
    scales = 0.057 .* exp.(-expected_exponent .* interactions)
    ground = [0.08, 0.065, 0.052, 0.039]
    curves = [
        ground[i] .* (1 .- (temperatures ./ scales[i]).^2)
        for i in eachindex(interactions)
    ]
    result = fit_kondo_scaling(
        interactions, temperatures, curves, ground; low_points=3)
    @test result.exponent ≈ expected_exponent atol=1e-12
    @test result.r2 > 1 - 1e-12
    @test result.diagnostics.low_points == 3
    @test maximum(result.diagnostics.curve_relative_errors) < 1e-12
    @test_throws ArgumentError fit_kondo_scaling(
        interactions, temperatures, curves, ground; low_points=1)
end

@graft_testset "M1 semicircular retarded branch" begin
    for x in (-0.9, -0.4, -0.05, 0.0, 0.05, 0.4, 0.9)
        G = semicircular_hybridization(x)
        @test real(G) ≈ 2x atol=1e-12
        @test imag(G) ≈ -2 * sqrt(1 - x^2) atol=1e-12
    end
    @test semicircular_hybridization(2.0) ≈ 2 * (2 - sqrt(3))
    @test semicircular_hybridization(-2.0) ≈ 2 * (-2 + sqrt(3))
    @test semicircular_hybridization(1.0) ≈ 2
    @test semicircular_hybridization(-1.0) ≈ -2
end

@graft_testset "M1 Kondo semicircular bath validation" begin
    @test semicircular_hybridization(100im) ≈ inv(100im) rtol=1e-4
    @test semicircular_hybridization(-100im) ≈ inv(-100im) rtol=1e-4

    nsites = 29
    angles = [j * pi / (nsites + 1) for j in 1:nsites]
    energies = cos.(angles)
    couplings = sqrt.(2 .* sin.(angles).^2 ./ (nsites + 1))
    @test discrete_bath_hybridization(
        1.2im, energies, couplings) ≈
        semicircular_hybridization(1.2im) atol=1e-12

    report = validate_semicircular_bath(
        energies, couplings;
        omega_min=pi / 1024, omega_max=10,
        npoints=256, tolerance=1e-6)
    @test !report.accepted
    @test report.max_error > 1
    @test report.nsites == 29
    @test report.worst_frequency <= 0.01

    artifact_path = normpath(joinpath(
        @__DIR__, "..", "examples", "data",
        "kondo_semicircular_bath_29.csv"))
    fitted_energies, fitted_couplings = read_bath_csv(artifact_path)
    fitted_report = validate_semicircular_bath(
        fitted_energies, fitted_couplings;
        omega_min=pi / 1024, omega_max=100,
        npoints=4096, tolerance=1e-6)
    @test length(fitted_energies) == 29
    @test fitted_report.accepted
    @test fitted_report.max_error < 3e-7
    @test fitted_report.nsites == 29

    mktempdir() do directory
        path = joinpath(directory, "bath.csv")
        open(path, "w") do io
            println(io, "energy,coupling_re,coupling_im")
            println(io, "-0.5,0.2,0.0")
            println(io, "0.5,0.2,0.0")
        end
        epsilons, values = read_bath_csv(path)
        @test epsilons == [-0.5, 0.5]
        @test values == ComplexF64[0.2, 0.2]
    end
end
