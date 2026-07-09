include(joinpath(@__DIR__, "hubbard_2x2_common.jl"))

using .Hubbard2x2Common
using GRAFT
using GRAFT.TestUtils
using GRAFT.Backend: ComplexSpace
using LinearAlgebra: dot
using Printf
using Random

function main()
    t = 1.0
    U = 4.0
    topo, phys, orbitals = hubbard_2x2_binary()
    H = hubbard_2x2_opsum(; t, U, shifted=true)
    O = ttno_from_opsum(H, topo, phys; hermitian=true)

    psi = random_ttns(Xoshiro(20260710), ComplexF64, topo, phys, ComplexSpace(16))
    Hd = dense_hamiltonian(H, topo, phys)
    E_exact, v_exact = exact_groundstate(Hd)

    _, energies = dmrg2!(psi, O;
                         trunc=TruncationScheme(maxdim=32, atol=1e-12),
                         nsweeps=4, krylovdim=48)
    E_dmrg = real(expect(psi, O))

    Nop = dense_hamiltonian(electron_number_opsum(), topo, phys)
    Dop = dense_hamiltonian(double_occupancy_opsum(), topo, phys)
    N_exact = real(dot(v_exact, Nop * v_exact))
    D_exact = real(dot(v_exact, Dop * v_exact))
    Nttno = ttno_from_opsum(electron_number_opsum(), topo, phys; hermitian=true)
    Dttno = ttno_from_opsum(double_occupancy_opsum(), topo, phys; hermitian=true)
    N_dmrg = real(expect(psi, Nttno))
    D_dmrg = real(expect(psi, Dttno))

    println("2x2 Hubbard model on a balanced binary tree")
    println("boundary = open")
    println("open nearest-neighbor bonds = ", open_boundary_bonds_2x2())
    println("spin-orbital leaf order = ", orbitals)
    @printf("parameters: t = %.6g, U = %.6g, onsite = U*(n_up-1/2)*(n_dn-1/2)\n", t, U)
    println("tree nodes = ", nnodes(topo), ", physical leaves = ", length(orbitals))
    @printf("ED ground energy      = %.12f\n", E_exact)
    @printf("DMRG ground energy    = %.12f\n", E_dmrg)
    @printf("abs(DMRG - ED)        = %.3e\n", abs(E_dmrg - E_exact))
    @printf("<N_e> ED              = %.8f\n", N_exact)
    @printf("<N_e> TTNS            = %.8f\n", N_dmrg)
    @printf("<double occupancy> ED = %.8f\n", D_exact)
    @printf("<double occupancy> TTNS = %.8f\n", D_dmrg)
    println("DMRG sweep energies   = ", energies)
    println("max DMRG bond dim     = ", max_bond_dim(psi))
    return nothing
end

main()
