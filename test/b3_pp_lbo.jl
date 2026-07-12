using Test
using Graft
using Graft.TestUtils
using Graft.Backend
using LinearAlgebra: Diagonal, diag, dot, norm

isdefined(@__MODULE__, Symbol("@graft_testset")) || include("test_harness.jl")

function _b3_well_model(nmax; n0=4, g=0.1)
    B = boson_ops(nmax)
    vals = [(n - n0)^2 for n in 0:nmax]
    F = TensorMap(Matrix(Diagonal(Float64.(vals))), B.P ← B.P)
    topo = mps_topology(1)
    phys = Dict(:site1 => B.P)
    H = OpSum()
    H += Term(1.0, SiteOp(:site1, :well, F))
    H += Term(g, SiteOp(:site1, :X, B.X))
    Hd = dense_hamiltonian(H, topo, phys)
    E, v = exact_groundstate(Hd)
    return B, topo, phys, H, Hd, E, v
end

@graft_testset "B3 PP LBO semantics" begin
    nmax = 8
    B, topo, phys, H, Hd, E, v = _b3_well_model(nmax)
    Hp, topop, physp = ppdress(H, topo, phys; nmax, boson_sites=[:site1])
    Hpp = dense_hamiltonian(Hp, topop, physp)
    d = nmax + 1
    pp_subspace = [n + 1 + d * n for n in 0:nmax]
    @test norm(Hpp[pp_subspace, pp_subspace] - dense_hamiltonian(H, topo, phys)) < 1e-12

    PP = boson_ops_pp(nmax)
    coeffs = TensorMap(Matrix(Diagonal(v)), PP.P ← dual(PP.Bspace))
    prev = Inf
    for chi in 1:4
        U, S, Vh = split_svd(coeffs, TruncationScheme(maxdim=chi))
        vchi = diag(convert(Array, U * S * Vh))
        vchi ./= norm(vchi)
        err = abs(real(dot(vchi, Hd * vchi)) - E)
        _, _, _, _, _, Enaive, _ = _b3_well_model(chi - 1)
        naive_err = abs(Enaive - E)
        @test err <= prev + 1e-12
        @test err < naive_err
        prev = err
    end
end
