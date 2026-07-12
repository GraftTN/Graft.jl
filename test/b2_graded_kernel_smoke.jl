using Test
using Graft
using Graft.TestUtils
using Graft.Backend
using LinearAlgebra: dot, norm

isdefined(@__MODULE__, Symbol("@graft_testset")) || include("test_harness.jl")

function _b2_fz2_hopping()
    F = fermion_ops_z2()
    topo = mps_topology(2)
    phys = Dict(nodeid(topo, i) => F.P for i in 1:nnodes(topo))
    H = OpSum()
    H += Term(-1.0, SiteOp(:site1, :Cd, F.Cd), SiteOp(:site2, :C, F.C))
    H += Term(-1.0, SiteOp(:site1, :C, F.C), SiteOp(:site2, :Cd, F.Cd))
    return F, topo, phys, H, ttno_from_opsum(H, topo, phys; hermitian=true)
end

@graft_testset "B2 graded kernel smoke" begin
    F, topo, phys, H, O = _b2_fz2_hopping()
    E0, _ = exact_groundstate(dense_hamiltonian(H, topo, phys))

    ψd = product_ttns(ComplexF64, topo, phys,
                      Dict(:site1 => FermionParity(1), :site2 => FermionParity(0)))
    _, Es = dmrg2!(ψd, O; trunc=TruncationScheme(maxdim=4, atol=1e-12),
                   nsweeps=4, verbose=TEST_VERBOSE)
    @test Es[end] ≈ E0 atol = 1e-10
    @test collect(sectors(domain(ψd[topo.root])[1])) == [FermionParity(1)]

    ψt = product_ttns(ComplexF64, topo, phys,
                      Dict(:site1 => FermionParity(1), :site2 => FermionParity(0)))
    Hd = dense_hamiltonian(H, topo, phys)
    dt = 0.04
    nsteps = 2
    vex = exact_evolve(Hd, to_dense(ψt), -im * dt * nsteps)
    ev = TDVP2(trunc=TruncationScheme(maxdim=4, atol=1e-12),
               verbose=TEST_VERBOSE)
    for _ in 1:nsteps
        step!(ev, ψt, O, -im * dt)
    end
    @test abs(1 - abs(dot(to_dense(ψt), vex))) < 1e-8
end
