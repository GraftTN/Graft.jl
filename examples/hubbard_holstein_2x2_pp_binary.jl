include(joinpath(@__DIR__, "hubbard_2x2_common.jl"))

using .Hubbard2x2Common
using Graft
using Graft.TestUtils
using Graft.Backend: U1Space
using LinearAlgebra: dot
using Printf
using Random

function main()
    t = 1.0
    U = 4.0
    omega = 0.8
    g = 0.35
    nmax = 1

    topo, phys, orbitals, phonons = hubbard_holstein_2x2_binary(; nmax)
    H = hubbard_holstein_2x2_opsum(; t, U, omega, g, nmax,
                                   shifted=true, centered_density=true)
    Hd = dense_hamiltonian(H, topo, phys)
    E_ed, v_ed = exact_groundstate(Hd)
    Ne_dense = dense_hamiltonian(electron_number_opsum(), topo, phys)
    D_dense = dense_hamiltonian(double_occupancy_opsum(), topo, phys)
    Nph_dense = dense_hamiltonian(phonon_number_opsum(nmax), topo, phys)
    Ne_ed = real(dot(v_ed, Ne_dense * v_ed))
    D_ed = real(dot(v_ed, D_dense * v_ed))
    Nph_ed = real(dot(v_ed, Nph_dense * v_ed))

    Hp, topop, physp = dress_like_holstein(H, topo, phys, phonons; nmax)
    physp = Dict{Symbol,typeof(first(values(physp)))}(site => P for (site, P) in physp)
    O = ttno_from_opsum(Hp, topop, physp; hermitian=true)

    pp_bond = U1Space(-1 => 8, 0 => 16, 1 => 8)
    psi = random_ttns(Xoshiro(20260710), ComplexF64, topop, physp, pp_bond)
    _, energies = dmrg2!(psi, O;
                         trunc=TruncationScheme(maxdim=24, atol=1e-11),
                         nsweeps=6, krylovdim=32)
    E_dmrg = real(expect(psi, O))

    Np, topoN, physN = dress_like_holstein(electron_number_opsum(),
                                           topo, phys, phonons; nmax)
    Dp, topoD, physD = dress_like_holstein(double_occupancy_opsum(),
                                           topo, phys, phonons; nmax)
    Bp, topoB, physB = dress_like_holstein(phonon_number_opsum(nmax),
                                           topo, phys, phonons; nmax)
    @assert topoN == topop && topoD == topop && topoB == topop
    @assert physN == physp && physD == physp && physB == physp

    Nop = ttno_from_opsum(Np, topop, physp; hermitian=true)
    Dop = ttno_from_opsum(Dp, topop, physp; hermitian=true)
    Bop = ttno_from_opsum(Bp, topop, physp; hermitian=true)
    Ne_dmrg = real(expect(psi, Nop))
    D_dmrg = real(expect(psi, Dop))
    Nph_dmrg = real(expect(psi, Bop))

    println("2x2 Hubbard-Holstein model with projected purification")
    println("boundary = open")
    println("open nearest-neighbor bonds = ", open_boundary_bonds_2x2())
    println("spin-orbital leaf order = ", orbitals)
    println("phonon P leaves = ", phonons)
    println("phonon B ancillas = ", [pp_ancilla_site(ph) for ph in phonons])
    @printf("parameters: t = %.6g, U = %.6g, omega = %.6g, g = %.6g, nmax = %d\n",
            t, U, omega, g, nmax)
    println("Holstein coupling = g*(n_up+n_dn-1)*(b+bdagger)")
    println("undressed tree nodes = ", nnodes(topo), ", PP tree nodes = ", nnodes(topop))
    @printf("cutoff ED ground energy = %.12f\n", E_ed)
    @printf("PP DMRG ground energy = %.12f\n", E_dmrg)
    @printf("abs(PP DMRG - ED)    = %.3e\n", abs(E_dmrg - E_ed))
    @printf("<N_e> ED              = %.8f\n", Ne_ed)
    @printf("<N_e> PP TTNS         = %.8f\n", Ne_dmrg)
    @printf("<double occupancy> ED = %.8f\n", D_ed)
    @printf("<double occupancy> PP TTNS = %.8f\n", D_dmrg)
    @printf("<total phonon number> ED = %.8f\n", Nph_ed)
    @printf("<total phonon number> PP TTNS = %.8f\n", Nph_dmrg)
    println("DMRG sweep energies   = ", energies)
    println("max DMRG bond dim     = ", max_bond_dim(psi))
    return nothing
end

main()
