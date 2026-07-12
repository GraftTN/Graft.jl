using Test
using Graft
using Graft.TestUtils
using Graft.Backend: ℂ
using LinearAlgebra: dot, norm
using Random

isdefined(@__MODULE__, Symbol("@graft_testset")) || include("test_harness.jl")

function _b5_holstein_trimer()
    S = spin_ops()
    B = boson_ops(1)
    topo = mps_topology(3)
    for i in 1:3
        topo = mount_chain(topo, Symbol(:site, i), 1; prefix=Symbol(:ph, i, :_))
    end

    phys = Dict{Symbol,typeof(S.P)}()
    for i in 1:3
        phys[Symbol(:site, i)] = S.P
        phys[Symbol(:ph, i, :_1)] = B.P
    end

    H = boson_modes([Symbol(:ph, i, :_1) => 0.65 for i in 1:3]; ops=B)
    for i in 1:3
        H += Term(0.0, SiteOp(Symbol(:site, i), :N, S.N))
        H += BosonCoupling([(Symbol(:site, i), Symbol(:ph, i, :_1)) => 0.25],
                            :density; matter_ops=S, boson_ops=B, density=:N)
    end
    for i in 1:2
        a = Symbol(:site, i)
        b = Symbol(:site, i + 1)
        H += Term(-0.25, SiteOp(a, :Sp, S.Sp), SiteOp(b, :Sm, S.Sm))
        H += Term(-0.25, SiteOp(a, :Sm, S.Sm), SiteOp(b, :Sp, S.Sp))
        H += Term(0.02, SiteOp(a, :Z, S.Z), SiteOp(b, :Z, S.Z))
    end

    return S, topo, phys, H
end

@graft_testset "B5 Holstein TDVP chi milestone" begin
    S, topo, phys, H = _b5_holstein_trimer()
    O = ttno_from_opsum(H, topo, phys; hermitian=true)
    ψ = random_ttns(Xoshiro(20260709), ComplexF64, topo, phys, ℂ^4)
    Hd = dense_hamiltonian(H, ψ)
    E0, v0 = exact_groundstate(Hd)

    _, Es = dmrg2!(ψ, O; trunc=TruncationScheme(maxdim=16, atol=1e-12),
                   nsweeps=8, verbose=TEST_VERBOSE)
    @test abs(Es[end] - E0) < 1e-8

    Nop = S.N
    Nmat = dense_hamiltonian(OpSum() + Term(1.0, SiteOp(:site2, :N, Nop)), ψ)
    nbar = expect(ψ, Nop, :site2)
    nbar_ref = real(dot(v0, Nmat * v0))
    ts = [0.0, 0.05, 0.1]

    vals = correlator(ψ, E0, :site2 => Nop, :site2 => Nop, ts;
                      H=O, evolver=TDVP2(trunc=TruncationScheme(maxdim=16, atol=1e-12),
                                         verbose=TEST_VERBOSE)) .- nbar^2
    ref = [exp(im * E0 * t) * dot(Nmat * v0, exact_evolve(Hd, Nmat * v0, -im * t)) - nbar_ref^2
           for t in ts]

    @test maximum(abs.(ref .- ref[1])) > 1e-4
    @test norm(vals - ref) < 5e-5
end
