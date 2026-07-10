using Test
using GRAFT
using GRAFT.TestUtils
using GRAFT.Backend
using LinearAlgebra: I, dot, norm

@testset "B5 fZ2 hopping Green smoke" begin
    F = fermion_ops_z2()
    topo = mps_topology(2)
    phys = Dict(nodeid(topo, i) => F.P for i in 1:nnodes(topo))

    H = OpSum()
    H += Term(-1.0, SiteOp(:site1, :Cd, F.Cd), SiteOp(:site2, :C, F.C))
    H += Term(-1.0, SiteOp(:site1, :C, F.C), SiteOp(:site2, :Cd, F.Cd))
    O = ttno_from_opsum(H, topo, phys; hermitian=true)

    ψ0 = product_ttns(ComplexF64, topo, phys,
                      Dict(:site1 => FermionParity(0), :site2 => FermionParity(0)))
    ts = [0.0, 0.04, 0.08]
    vals = correlator(ψ0, 0.0, :site1 => F.C, :site1 => F.Cd, ts;
                      H=O, evolver=TDVP2(trunc=TruncationScheme(maxdim=4, atol=1e-12),
                                         verbose=TEST_VERBOSE))

    Hd = dense_hamiltonian(H, topo, phys)
    v0 = to_dense(ψ0)
    cd = reshape(convert(Array, F.Cd), 2, 2, 1)[:, :, 1]
    Cd1 = kron(cd, Matrix{ComplexF64}(I, 2, 2))
    ref = [dot(Cd1 * v0, exact_evolve(Hd, Cd1 * v0, -im * t)) for t in ts]

    @test norm(vals - ref) < 1e-8
end
