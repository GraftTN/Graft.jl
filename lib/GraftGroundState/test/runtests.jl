import Random
using GraftFoundation: TruncationScheme, dim, domain, mps_topology, ℂ
using GraftGroundState: dmrg1!, dmrg1_3s!
using GraftSymbolic: OpSum, SiteOp, Term, spin_ops
using GraftTestUtils: dense_hamiltonian, exact_groundstate, random_ttns, to_dense
using GraftTTNOBuild: ttno_from_opsum
using LinearAlgebra: norm
using Random: Xoshiro
using Test

@testset "one-site DMRG solver contract" begin
    topology = mps_topology(1)
    spin = spin_ops()
    physical_spaces = Dict(:site1 => spin.P)
    hamiltonian = OpSum() +
        Term(0.37, SiteOp(:site1, :Z, spin.Z)) +
        Term(-0.61, SiteOp(:site1, :X, spin.X))
    operator = ttno_from_opsum(
        hamiltonian, topology, physical_spaces; hermitian=true)
    state = random_ttns(
        Xoshiro(20260801), ComplexF64, topology, physical_spaces, ℂ^1)

    dense_operator = dense_hamiltonian(hamiltonian, state)
    exact_energy, _ = exact_groundstate(dense_operator)
    solved, energies = dmrg1!(
        state, operator; nsweeps=1, krylovdim=4, verbose=false)
    vector = to_dense(solved)

    @test length(energies) == 1
    @test only(energies) ≈ exact_energy atol=1e-12
    @test norm(dense_operator * vector - exact_energy * vector) /
          norm(vector) < 1e-11

    @test_throws ArgumentError dmrg1_3s!(
        copy(state), operator;
        nsweeps=0, expand_scheme=:unknown, verbose=false,
    )
    default_state, default_energies = dmrg1_3s!(
        copy(state), operator; nsweeps=0, verbose=false)
    @test isempty(default_energies)
    @test to_dense(default_state) == to_dense(state)
    @test_throws ArgumentError dmrg1_3s!(
        copy(state), operator;
        nsweeps=0, expand_scheme=:rangefinder, rng=nothing, verbose=false,
    )
    direct_state, direct_energies = dmrg1_3s!(
        copy(state), operator;
        nsweeps=0, expand_scheme=:directqr, verbose=false,
    )
    @test isempty(direct_energies)
    @test to_dense(direct_state) == to_dense(state)
end


@testset "DMRG3S default factorized range finder" begin
    topology = mps_topology(2)
    spin = spin_ops()
    physical_spaces = Dict(:site1 => spin.P, :site2 => spin.P)
    hamiltonian = OpSum() +
        Term(0.41, SiteOp(:site1, :Z, spin.Z)) +
        Term(-0.23, SiteOp(:site2, :X, spin.X)) +
        Term(0.67, SiteOp(:site1, :X, spin.X),
                   SiteOp(:site2, :Z, spin.Z))
    operator = ttno_from_opsum(
        hamiltonian, topology, physical_spaces; hermitian=true)
    initial = random_ttns(
        Xoshiro(2026080712), ComplexF64, topology, physical_spaces, ℂ^1)

    global_rng = Random.default_rng()
    saved_global_rng = copy(global_rng)
    try
        expected = rand(global_rng, UInt64)
        copy!(global_rng, saved_global_rng)
        first_state, first_energies = dmrg1_3s!(
            copy(initial), operator;
            trunc=TruncationScheme(maxdim=2), nsweeps=1, max_add=1,
            rsvd_oversample=1, rsvd_poweriter=1,
            rsvd_threaded=false, verbose=false)
        second_state, second_energies = dmrg1_3s!(
            copy(initial), operator;
            trunc=TruncationScheme(maxdim=2), nsweeps=1, max_add=1,
            rsvd_oversample=1, rsvd_poweriter=1,
            rsvd_threaded=false, verbose=false)

        @test first_energies ≈ second_energies rtol=1e-13 atol=1e-13
        @test norm(to_dense(first_state) - to_dense(second_state)) < 1e-12
        @test dim(domain(first_state.tensors[2])) == 2
        @test rand(global_rng, UInt64) == expected
    finally
        copy!(global_rng, saved_global_rng)
    end
end
