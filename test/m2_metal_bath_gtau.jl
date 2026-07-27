using LinearAlgebra: Diagonal, Symmetric, eigen

function _metal_bath_gtau(nsites, beta, times)
    energies, couplings = gauss_semicircular_bath(nsites)
    one_body = zeros(Float64, nsites + 1, nsites + 1)
    one_body[2:end, 2:end] .= Diagonal(energies)
    one_body[1, 2:end] .= couplings
    one_body[2:end, 1] .= couplings
    decomposition = eigen(Symmetric(one_body))
    impurity_weights = abs2.(decomposition.vectors[1, :])
    values = ComplexF64[
        -sum(
            impurity_weights .* exp.(-decomposition.values .* tau) ./
            (1 .+ exp.(-beta .* decomposition.values)))
        for tau in times
    ]
    return values, energies, couplings
end

@graft_testset "M2 metallic-bath finite-temperature G(tau)" begin
    beta = 10.0
    comparison_times = collect(range(0.0, beta; length=101))
    coarse, _, _ = _metal_bath_gtau(32, beta, comparison_times)
    converged, energies, couplings =
        _metal_bath_gtau(64, beta, comparison_times)
    @test maximum(abs.(coarse .- converged)) < 5e-10
    @test real(first(converged)) ≈ -0.5 atol=1e-12
    @test real(last(converged)) ≈ -0.5 atol=1e-12
    @test maximum(abs, imag.(converged)) < 1e-14

    times = collect(range(0.0, beta; length=4001))
    values, _, _ = _metal_bath_gtau(64, beta, times)
    series = CorrelatorSeries(times, values, (; beta))
    matsubara = matsubara_transform(
        series; statistics=:fermionic, indices=0:5)
    exact = ComplexF64[
        inv(im * frequency -
            discrete_bath_hybridization(
                im * frequency, energies, couplings))
        for frequency in matsubara.frequencies
    ]
    @test maximum(abs.(matsubara.values .- exact)) < 2e-6

    bath_report = validate_semicircular_bath(
        energies, couplings;
        omega_min=pi / beta, omega_max=10,
        npoints=512, tolerance=1e-10)
    @test bath_report.accepted
end
