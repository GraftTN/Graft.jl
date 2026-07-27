using LinearAlgebra: Diagonal, I, diag, eigen, Hermitian, kron, norm, svd

function _p4_pp_expect(amplitudes, fermion_values, boson_values)
    normalization = sum(abs2, amplitudes)
    value = 0.0
    for f in axes(amplitudes, 1), p in axes(amplitudes, 2),
        b in axes(amplitudes, 3), af in axes(amplitudes, 4),
        ap in axes(amplitudes, 5)
        value += abs2(amplitudes[f, p, b, af, ap]) *
            fermion_values[f] * boson_values[p]
    end
    return value / normalization
end

@graft_testset "P4 finite-temperature PP LBO error budget" begin
    nmax = 5
    d = nmax + 1
    beta = 2.0
    epsilon, omega, coupling = -0.2, 0.7, 0.25
    identity_fermion = Matrix{Float64}(I, 2, 2)
    identity_boson = Matrix{Float64}(I, d, d)
    number_fermion = Matrix(Diagonal([0.0, 1.0]))
    number_boson = Matrix(Diagonal(collect(0.0:nmax)))
    position_boson = zeros(Float64, d, d)
    for occupation in 0:(nmax - 1)
        position_boson[occupation + 1, occupation + 2] =
            sqrt(occupation + 1)
        position_boson[occupation + 2, occupation + 1] =
            sqrt(occupation + 1)
    end
    hamiltonian =
        epsilon * kron(identity_boson, number_fermion) +
        omega * kron(number_boson, identity_fermion) +
        coupling * kron(position_boson, number_fermion)
    half_density = exp(-beta * hamiltonian / 2)

    # Logical physical index is (fermion, boson); PP embeds |p> as
    # |p>_P |p>_BPP and keeps a distinct thermal ancilla index.
    logical_index(f, p) = f + 2 * (p - 1)
    amplitudes = zeros(ComplexF64, 2, d, d, 2, d)
    for f in 1:2, p in 1:d, af in 1:2, ap in 1:d
        amplitudes[f, p, p, af, ap] =
            half_density[logical_index(f, p), logical_index(af, ap)]
    end
    amplitudes ./= norm(amplitudes)

    density_matrix = half_density * half_density'
    partition = real(sum(diag(density_matrix)))
    density_exact = real(sum(
        density_matrix[logical_index(f, p), logical_index(f, p)] * (f - 1)
        for f in 1:2, p in 1:d) / partition)
    boson_exact = real(sum(
        density_matrix[logical_index(f, p), logical_index(f, p)] * (p - 1)
        for f in 1:2, p in 1:d) / partition)
    @test _p4_pp_expect(
        amplitudes, [0.0, 1.0], ones(d)) ≈ density_exact atol=1e-13
    @test _p4_pp_expect(
        amplitudes, ones(2), collect(0.0:nmax)) ≈ boson_exact atol=1e-13

    matrix = reshape(
        permutedims(amplitudes, (3, 1, 2, 4, 5)), d, :)
    decomposition = svd(matrix)
    discarded_weights = Float64[]
    density_errors = Float64[]
    boson_errors = Float64[]
    for kept in 1:d
        truncated = decomposition.U[:, 1:kept] *
            Diagonal(decomposition.S[1:kept]) *
            decomposition.Vt[1:kept, :]
        truncated_amplitudes = permutedims(
            reshape(truncated, d, 2, d, 2, d),
            (2, 3, 1, 4, 5))
        truncated_amplitudes ./= norm(truncated_amplitudes)
        discarded = sum(abs2, decomposition.S[(kept + 1):end]) /
            sum(abs2, decomposition.S)
        density_error = abs(
            _p4_pp_expect(
                truncated_amplitudes, [0.0, 1.0], ones(d)) -
            density_exact)
        boson_error = abs(
            _p4_pp_expect(
                truncated_amplitudes, ones(2), collect(0.0:nmax)) -
            boson_exact)
        push!(discarded_weights, discarded)
        push!(density_errors, density_error)
        push!(boson_errors, boson_error)
        @test density_error <= 2sqrt(discarded) + 1e-12
        @test boson_error <= 2nmax * sqrt(discarded) + 1e-12
    end
    @test all(diff(discarded_weights) .<= 100eps(Float64))
    @test discarded_weights[end] < 1e-28
    @test density_errors[end] < 1e-12
    @test boson_errors[end] < 1e-12

    recorded = ThermalBenchmarkDatum(
        :boson_occupation, :scalar, 0.0,
        boson_exact + boson_errors[2];
        lbo_error=boson_errors[2])
    @test recorded.stderr == 0
    @test recorded.lbo_error == boson_errors[2]
end
