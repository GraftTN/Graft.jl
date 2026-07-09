using Test
using GRAFT
using GRAFT.TestUtils
using GRAFT.Backend: ℂ
using LinearAlgebra: dot, norm
using Random

@testset "B4 bath TDVP end-to-end" begin
    S = spin_ops()
    B = boson_ops(1)
    P = Partition([[:imp]])
    topo = TreeTopology(:imp, Pair{Symbol,Symbol}[])
    J(ω) = 0.2 * ω

    bb = BosonBath(J; partition=P, topology=topo, matter_ops=S, boson_ops=B,
                   nmodes=2, ωmin=0.5, ωmax=1.5, prefix=:bfit, density=:Z)
    phys = merge(Dict(:imp => S.P), bb.phys)
    O = ttno_from_opsum(bb.H, bb.topology, phys; hermitian=true)
    ψ = random_ttns(Xoshiro(20260709), ComplexF64, bb.topology, phys, ℂ^4)
    Hd = dense_hamiltonian(bb.H, ψ)

    dt = 0.015
    nsteps = 2
    vex = exact_evolve(Hd, to_dense(ψ), -im * dt * nsteps)
    ev = TDVP2(trunc=TruncationScheme(maxdim=8, atol=1e-12))
    for _ in 1:nsteps
        step!(ev, ψ, O, -im * dt)
    end

    Z = dense_hamiltonian(OpSum() + Term(1.0, SiteOp(:imp, :Z, S.Z)), ψ)
    @test abs(1 - abs(dot(to_dense(ψ), vex))) < 1e-10
    @test abs(expect(ψ, S.Z, :imp) - dot(vex, Z * vex)) < 1e-8
end
