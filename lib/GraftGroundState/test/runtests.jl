using GraftFoundation: mps_topology, ℂ
using GraftGroundState: dmrg1!
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
end
