# GRAFT.jl test suite — every kernel is cross-validated against exact
# diagonalization / exact propagation on small trees, plus gauge-invariance
# property tests (architecture §9.11: this is a merge requirement, CI-enforced).
using Test
using GRAFT
using GRAFT.TestUtils
using GRAFT.Backend: ℂ, ⊗, ←, dim, domain, dual, space, id, numind, numout, numin,
    U1Space, U1Irrep, FermionParity, TensorMap, oneunit, sectors
using TensorOperations
using GRAFT.Trees: edges
using GRAFT.Contractions: two_site_tensor, two_site_space, split_two_site!
using Random
using LinearAlgebra: I, dot, norm

const RNG = Xoshiro(20260709)

"Transverse-field Ising OpSum on all edges/nodes of a topology."
function tfi(topo; J=1.0, g=0.7)
    S = spin_ops()
    H = OpSum()
    for (c, p) in edges(topo)
        H += Term(-J, SiteOp(nodeid(topo, c), :Z, S.Z), SiteOp(nodeid(topo, p), :Z, S.Z))
    end
    for i in 1:nnodes(topo)
        H += Term(-g, SiteOp(nodeid(topo, i), :X, S.X))
    end
    return H
end

allspin(topo) = Dict(nodeid(topo, i) => spin_ops().P for i in 1:nnodes(topo))
bonddims(ψ) = [dim(domain(ψ.tensors[c])[1]) for c in 1:nnodes(ψ.topo) if ψ.topo.parent[c] != 0]
function fused_u1_charge(ops)
    q = U1Irrep(0)
    for op in ops
        q = only(q ⊗ charge(op))
    end
    return q
end

function dense_two_site_ttno(O)
    topo = topology(O)
    @assert nnodes(topo) == 2
    root = topo.root
    child = only(topo.children[root])
    Wr = convert(Array, O[root])
    Wc = convert(Array, O[child])
    χ = size(Wc, 3)
    droot = size(Wr, 2)
    dchild = size(Wc, 1)
    M = zeros(eltype(Wr), droot, dchild, droot, dchild)
    for a in 1:χ, ro in 1:droot, co in 1:dchild, ri in 1:droot, ci in 1:dchild
        M[ro, co, ri, ci] += Wr[a, ro, ri, 1] * Wc[co, ci, a]
    end
    return reshape(M, droot * dchild, droot * dchild)
end

function dense_two_leaf_star_ttno(O)
    topo = topology(O)
    @assert nnodes(topo) == 3
    root = topo.root
    children_ = topo.children[root]
    @assert length(children_) == 2
    W0 = convert(Array, O[root])
    W1 = convert(Array, O[children_[1]])
    W2 = convert(Array, O[children_[2]])
    χ1, χ2 = size(W1, 3), size(W2, 3)
    d0, d1, d2 = size(W0, 3), size(W1, 1), size(W2, 1)
    M = zeros(eltype(W0), d0, d1, d2, d0, d1, d2)
    for a in 1:χ1, b in 1:χ2,
        o0 in 1:d0, o1 in 1:d1, o2 in 1:d2,
        i0 in 1:d0, i1 in 1:d1, i2 in 1:d2
        M[o0, o1, o2, i0, i1, i2] += W0[a, b, o0, i0, 1] * W1[o1, i1, a] * W2[o2, i2, b]
    end
    return reshape(M, d0 * d1 * d2, d0 * d1 * d2)
end

struct LocalZEvolver <: Evolver
    site::Symbol
    omega::Float64
end

function GRAFT.step!(ev::LocalZEvolver, ψ::TTNS, ::TTNO, dz::Number)
    P = physspace(ψ, nodeindex(topology(ψ), ev.site))
    U = TensorMap(ComplexF64[exp(dz * ev.omega) 0; 0 exp(-dz * ev.omega)], P ← P)
    ϕ = apply_local(ψ, U, ev.site)
    ψ.tensors .= ϕ.tensors
    ψ.center = center(ϕ)
    return ψ
end

struct LocalXEvolver <: Evolver
    site::Symbol
    omega::Float64
end

function GRAFT.step!(ev::LocalXEvolver, ψ::TTNS, ::TTNO, dz::Number)
    P = physspace(ψ, nodeindex(topology(ψ), ev.site))
    a = dz * ev.omega
    U = TensorMap(ComplexF64[cosh(a) sinh(a); sinh(a) cosh(a)], P ← P)
    ϕ = apply_local(ψ, U, ev.site)
    ψ.tensors .= ϕ.tensors
    ψ.center = center(ϕ)
    return ψ
end

struct NoOpEvolver <: Evolver end
GRAFT.step!(::NoOpEvolver, ψ::TTNS, ::TTNO, ::Number) = ψ

@testset "GRAFT.jl" begin

@testset "Trees: topology & paths" begin
    t = star_topology(3, 2)
    @test nnodes(t) == 7
    @test isleaf(t, nodeindex(t, :b1_2))
    @test is_t3ns(t)                                        # bare junction: degree 3
    @test !is_t3ns(t; physical=[:center])                   # degree 3 + physical leg
    @test is_t3ns(mps_topology(5); physical=[Symbol(:site, i) for i in 1:5])
    po = postorder(t)
    @test po[end] == t.root
    @test length(po) == 7 && allunique(po)
    p = path_between(t, nodeindex(t, :b1_2), nodeindex(t, :b3_2))
    @test nodeid(t, p[1]) == :b1_2 && nodeid(t, p[end]) == :b3_2
    @test length(p) == 5                    # b1_2 b1_1 center b3_1 b3_2
    # value semantics (§9.4): equal trees hash equal
    t2 = star_topology(3, 2)
    @test t == t2 && hash(t) == hash(t2)
    @test binary_topology(2) != t2
end

@testset "Parallel helpers" begin
    out = zeros(Int, 8)
    threaded_foreach(1:8; threaded=false) do i
        out[i] = i^2
    end
    @test out == [i^2 for i in 1:8]

    out2 = zeros(Int, 8)
    threaded_foreach((i for i in 1:8); threaded=true, minbatch=1) do i
        out2[i] = 2 * i
    end
    @test out2 == [2 * i for i in 1:8]
    @test_throws ArgumentError threaded_foreach(identity, [1]; minbatch=0)
end

@testset "canonical form & gauge invariance" begin
    for topo in (mps_topology(6), star_topology(3, 2), binary_topology(2))
        phys = allspin(topo)
        ψ = random_ttns(RNG, ComplexF64, topo, phys, ℂ^3)
        @test norm(ψ) ≈ 1 atol = 1e-12
        v = to_dense(ψ)
        @test norm(v) ≈ 1 atol = 1e-12
        # property test: moving the center anywhere leaves the state invariant
        for target in (leaves(topo)[end], topo.root)
            w = to_dense(move_center!(copy(ψ), target))
            @test norm(v - w) < 1e-12
        end
    end
end

@testset "graded spaces (U(1)) gauge moves" begin
    topo = star_topology(2, 2)
    P = U1Space(0 => 1, 1 => 1)
    V = U1Space(-1 => 1, 0 => 2, 1 => 1)
    phys = Dict(nodeid(topo, i) => P for i in 1:nnodes(topo))
    ψ = random_ttns(RNG, ComplexF64, topo, phys, V)
    v = to_dense(ψ)
    w = to_dense(move_center!(copy(ψ), leaves(topo)[1]))
    @test norm(v - w) < 1e-12
    φ = random_ttns(RNG, ComplexF64, topo, phys, V)
    @test abs(inner(φ, ψ) - dot(to_dense(φ), v)) < 1e-12
end

@testset "TTNO builder vs dense" begin
    S = spin_ops()
    for topo in (mps_topology(5), star_topology(3, 2), binary_topology(2))
        phys = allspin(topo)
        H = tfi(topo)
        # add a long-range term and a 3-site term to exercise pass-through /
        # merge channels (postorder first node, a middle node, and the root
        # are always three distinct nodes)
        po = postorder(topo)
        a, b, c = po[1], po[max(2, length(po) ÷ 2)], topo.root
        H += Term(0.37, SiteOp(nodeid(topo, a), :Z, S.Z),
                  SiteOp(nodeid(topo, c), :Z, S.Z))
        H += Term(-0.21, SiteOp(nodeid(topo, a), :X, S.X),
                  SiteOp(nodeid(topo, b), :Z, S.Z),
                  SiteOp(nodeid(topo, c), :X, S.X))
        O = ttno_from_opsum(H, topo, phys; hermitian=true)
        ψ = random_ttns(RNG, ComplexF64, topo, phys, ℂ^2)
        @test norm(to_dense(O) - dense_hamiltonian(H, ψ)) < 1e-12
    end
    # branching (physless) node: hang two chains off a bare junction
    topo = TreeTopology(:j, [:j => :a1, :a1 => :a2, :j => :b1, :b1 => :b2])
    phys = Dict(s => S.P for s in (:a1, :a2, :b1, :b2))
    H = OpSum()
    H += Term(0.8, SiteOp(:a1, :Z, S.Z), SiteOp(:b1, :Z, S.Z))
    H += Term(-0.5, SiteOp(:a2, :X, S.X))
    O = ttno_from_opsum(H, topo, phys; hermitian=true)
    ψ = random_ttns(RNG, ComplexF64, topo, phys, ℂ^2)
    @test norm(to_dense(O) - dense_hamiltonian(H, ψ)) < 1e-12
end

@testset "TTNO apply" begin
    topo = mps_topology(2)
    S = spin_ops()
    phys = Dict(:site1 => S.P, :site2 => S.P)
    H = OpSum()
    H += Term(0.7, SiteOp(:site1, :X, S.X), SiteOp(:site2, :Z, S.Z))
    H += Term(0.2, SiteOp(:site2, :X, S.X))
    O = ttno_from_opsum(H, topo, phys; hermitian=false)
    ψ = random_ttns(RNG, ComplexF64, topo, phys, ℂ^2)
    move_center!(ψ, :site1)
    ϕ = apply(O, ψ)
    @test check_arrows(ϕ)
    @test center(ϕ) == center(ψ)
    @test norm(to_dense(ϕ) - dense_hamiltonian(H, ψ) * to_dense(ψ)) < 1e-10

    B = boson_ops_u1(2)
    phys_b = Dict(:site1 => B.P, :site2 => B.P)
    Hb = OpSum()
    Hb += Term(0.5, SiteOp(:site1, :Bd, B.Bd), SiteOp(:site2, :B, B.B))
    Hb += Term(0.5, SiteOp(:site1, :B, B.B), SiteOp(:site2, :Bd, B.Bd))
    Ob = ttno_from_opsum(Hb, topo, phys_b; hermitian=true)
    ψb = product_ttns(ComplexF64, topo, phys_b,
                      Dict(:site1 => U1Irrep(1), :site2 => U1Irrep(0)))
    ϕb = apply(Ob, ψb)
    @test check_arrows(ϕb)
    @test norm(to_dense(ϕb) - dense_hamiltonian(Hb, topo, phys_b) * to_dense(ψb)) < 1e-10
end

@testset "variational fit" begin
    topo = mps_topology(2)
    S = spin_ops()
    phys = Dict(:site1 => S.P, :site2 => S.P)
    H = OpSum()
    H += Term(0.7, SiteOp(:site1, :X, S.X), SiteOp(:site2, :Z, S.Z))
    H += Term(0.2, SiteOp(:site2, :X, S.X))
    O = ttno_from_opsum(H, topo, phys; hermitian=false)
    ψ = random_ttns(RNG, ComplexF64, topo, phys, ℂ^2)
    target = apply(O, ψ)

    φ = random_ttns(RNG, ComplexF64, topo, phys, virtualspace(target, nodeindex(topo, :site1)))
    _, errs = fit!(φ, target; nsweeps=4)
    @test check_arrows(φ)
    @test !isempty(errs)
    @test norm(to_dense(φ) - to_dense(target)) < 1e-8

    φlow = random_ttns(RNG, ComplexF64, topo, phys, ℂ^1)
    initial = norm(to_dense(φlow) - to_dense(target))
    _, errslow = fit!(φlow, target; nsweeps=4)
    @test errslow[end] < initial

    target2 = apply(ttno_from_opsum(OpSum() + Term(0.3, SiteOp(:site1, :Z, S.Z)),
                                    topo, phys; hermitian=true), ψ)
    coeffs = ComplexF64[1.0, -0.4im]
    φsum = random_ttns(RNG, ComplexF64, topo, phys, ℂ^4)
    _, errsum = fit!(φsum, (target, target2); coeffs, nsweeps=4)
    refsum = coeffs[1] * to_dense(target) + coeffs[2] * to_dense(target2)
    @test errsum[end] < 1e-8
    @test norm(to_dense(φsum) - refsum) < 1e-8

    O2 = ttno_from_opsum(OpSum() + Term(0.3, SiteOp(:site1, :Z, S.Z)),
                         topo, phys; hermitian=true)
    φop = random_ttns(RNG, ComplexF64, topo, phys, ℂ^4)
    _, errop = fit!(φop, (ψ, ψ); Hs=(O, O2), coeffs, nsweeps=4)
    refop = coeffs[1] * to_dense(apply(O, ψ)) + coeffs[2] * to_dense(apply(O2, ψ))
    @test errop[end] < 1e-8
    @test norm(to_dense(φop) - refop) < 1e-8
end

@testset "charged TTNO builder vs dense" begin
    U = spin_ops_u1()
    @test charge(SiteOp(:s, :Sp, U.Sp)) == U1Irrep(1)
    @test charge(SiteOp(:s, :Z, U.Z)) == U1Irrep(0)
    @test_throws ArgumentError Term(1.0, SiteOp(:s, :Sp, U.Sp))

    topo = mps_topology(2)
    phys = Dict(nodeid(topo, i) => U.P for i in 1:nnodes(topo))
    H = OpSum()
    for (c, p) in edges(topo)
        a, b = nodeid(topo, c), nodeid(topo, p)
        H += Term(0.5, SiteOp(a, :Sp, U.Sp), SiteOp(b, :Sm, U.Sm))
        H += Term(0.5, SiteOp(a, :Sm, U.Sm), SiteOp(b, :Sp, U.Sp))
        H += Term(0.2, SiteOp(a, :Z, U.Z), SiteOp(b, :Z, U.Z))
    end
    O = ttno_from_opsum(H, topo, phys; hermitian=true)
    @test norm(dense_two_site_ttno(O) - dense_hamiltonian(H, topo, phys)) < 1e-12

    topo_star = star_topology(2, 1)
    phys_star = Dict(nodeid(topo_star, i) => U.P for i in 1:nnodes(topo_star))
    Hstar = OpSum()
    for (c, p) in edges(topo_star)
        a, b = nodeid(topo_star, c), nodeid(topo_star, p)
        Hstar += Term(0.5, SiteOp(a, :Sp, U.Sp), SiteOp(b, :Sm, U.Sm))
        Hstar += Term(0.5, SiteOp(a, :Sm, U.Sm), SiteOp(b, :Sp, U.Sp))
    end
    Ostar = ttno_from_opsum(Hstar, topo_star, phys_star; hermitian=true)
    @test check_arrows(Ostar)
    @test norm(dense_two_leaf_star_ttno(Ostar) - dense_hamiltonian(Hstar, topo_star, phys_star)) < 1e-12

    B = boson_ops_u1(2)
    topo_b = mps_topology(2)
    phys_b = Dict(nodeid(topo_b, i) => B.P for i in 1:nnodes(topo_b))
    Hb = OpSum()
    Hb += Term(0.7, SiteOp(:site1, :Bd, B.Bd), SiteOp(:site2, :B, B.B))
    Hb += Term(0.7, SiteOp(:site1, :B, B.B), SiteOp(:site2, :Bd, B.Bd))
    Ob = ttno_from_opsum(Hb, topo_b, phys_b; hermitian=true)
    @test norm(dense_two_site_ttno(Ob) - dense_hamiltonian(Hb, topo_b, phys_b)) < 1e-12

    F = fermion_ops_z2()
    phys_f = Dict(nodeid(topo_b, i) => F.P for i in 1:nnodes(topo_b))
    Hf = OpSum()
    Hf += Term(-1.0, SiteOp(:site1, :Cd, F.Cd), SiteOp(:site2, :C, F.C))
    Hf += Term(-1.0, SiteOp(:site1, :C, F.C), SiteOp(:site2, :Cd, F.Cd))
    Of = ttno_from_opsum(Hf, topo_b, phys_f; hermitian=true)
    @test check_arrows(Of)
    @test collect(sectors(virtualspace(Of, nodeindex(topo_b, :site1)))) == [FermionParity(1)]
    @test_logs (:warn, r"FermionParity Arrays") begin
        @test norm(dense_two_site_ttno(Of) - dense_hamiltonian(Hf, topo_b, phys_f)) < 1e-12
    end

    ψprod = product_ttns(ComplexF64, topo, phys,
                         Dict(:site1 => U1Irrep(1), :site2 => U1Irrep(0)))
    @test check_arrows(ψprod)
    @test collect(sectors(virtualspace(ψprod, nodeindex(topo, :site1)))) == [U1Irrep(1)]
    @test collect(sectors(domain(ψprod[topo.root])[1])) == [U1Irrep(1)]

    ψvac = product_ttns(ComplexF64, topo_b, phys_f,
                        Dict(:site1 => FermionParity(0), :site2 => FermionParity(0)))
    ϕ1 = apply_local(ψvac, F.Cd, :site1)
    ϕ2 = apply_local(ψvac, F.Cd, :site2)
    @test check_arrows(ϕ1)
    @test check_arrows(ϕ2)
    @test inner(ϕ1, ϕ1) ≈ 1
    @test inner(ϕ2, ϕ2) ≈ 1
    @test collect(sectors(domain(ϕ1[topo_b.root])[1])) == [FermionParity(1)]
    @test norm(to_dense(ψvac) - ComplexF64[1, 0, 0, 0]) < 1e-12
    @test_logs (:warn, r"FermionParity Arrays") begin
        @test norm(to_dense(ϕ1) - ComplexF64[0, 0, 1, 0]) < 1e-12
        @test norm(to_dense(ϕ2) - ComplexF64[0, 1, 0, 0]) < 1e-12
    end

    topo_one = mps_topology(1)
    phys_one = Dict(:site1 => F.P)
    ψeven = product_ttns(ComplexF64, topo_one, phys_one, Dict(:site1 => FermionParity(0)))
    ψodd = product_ttns(ComplexF64, topo_one, phys_one, Dict(:site1 => FermionParity(1)))
    @test inner(ψodd, ψodd) ≈ 1
    Hzero = OpSum() + Term(0.0, SiteOp(:site1, :I, F.I))
    Ozero = ttno_from_opsum(Hzero, topo_one, phys_one; hermitian=true)
    G = correlator(ψeven, 0.0, :site1 => F.C, :site1 => F.Cd, [0.0, 0.2];
                   H=Ozero, evolver=NoOpEvolver())
    @test G ≈ ComplexF64[1, 1]

    topo_fstar = star_topology(2, 1)
    phys_fstar = Dict(nodeid(topo_fstar, i) => F.P for i in 1:nnodes(topo_fstar))
    Hfs = OpSum()
    Hfs += Term(-1.0, SiteOp(:b1_1, :Cd, F.Cd), SiteOp(:b2_1, :C, F.C))
    Hfs += Term(-1.0, SiteOp(:b1_1, :C, F.C), SiteOp(:b2_1, :Cd, F.Cd))
    Ofs = ttno_from_opsum(Hfs, topo_fstar, phys_fstar; hermitian=true)
    @test check_arrows(Ofs)
    @test_logs (:warn, r"FermionParity Arrays") begin
        @test norm(dense_two_leaf_star_ttno(Ofs) - dense_hamiltonian(Hfs, topo_fstar, phys_fstar)) < 1e-12
    end
end

@testset "projected purification rewrite" begin
    B = boson_ops(2)
    PP = boson_ops_pp(2)
    @test charge(SiteOp(:p, :Bpd, PP.Bpd)) == U1Irrep(1)
    @test charge(SiteOp(:b, :Bbd, PP.Bbd)) == U1Irrep(-1)
    @test charge(SiteOp(:p, :Bp, PP.Bp)) == U1Irrep(-1)
    @test charge(SiteOp(:b, :Bb, PP.Bb)) == U1Irrep(1)
    @test reshape(convert(Array, PP.Bb), 3, 3, 1)[:, :, 1][1, 2] == 1
    @test reshape(convert(Array, PP.Bbd), 3, 3, 1)[:, :, 1][2, 1] == 1

    topo = mps_topology(1)
    phys = Dict(:site1 => B.P)
    H = OpSum()
    H += Term(0.3, SiteOp(:site1, :X, B.X))
    H += Term(0.7, SiteOp(:site1, :N, B.N))
    Hp, topop, physp = ppdress(H, topo, phys; nmax=2, boson_sites=[:site1])
    @test nnodes(topop) == 2
    @test haskey(physp, :site1_B1)
    @test length(Hp) == 3
    @test all(t -> fused_u1_charge(t.ops) == U1Irrep(0), Hp)
    Op = ttno_from_opsum(Hp, topop, physp; hermitian=true)
    @test check_arrows(Op)
    H0 = dense_hamiltonian(H, topo, phys)
    Hpp = dense_hamiltonian(Hp, topop, physp)
    d = 3
    # B leaves carry the dual representation, so fixed PP charge is nB_index == nP.
    pp_subspace = [n + 1 + d * n for n in 0:2]
    @test norm(Hpp[pp_subspace, pp_subspace] - H0) < 1e-12

    topo_m = star_topology(1, 1; center=:spin, prefix=:b)
    phys_m = Dict(:spin => spin_ops().P, :b1_1 => B.P)
    Hm = boson_modes([:b1_1 => 0.7]; ops=B)
    Hm += BosonCoupling([(:spin, :b1_1) => 0.2],
                        :density; matter_ops=spin_ops(), boson_ops=B, density=:Z)
    Hmp, topomp, physmp = ppdress(Hm, topo_m, phys_m; nmax=2, boson_sites=[:b1_1])
    @test all(P -> typeof(first(sectors(P))) == U1Irrep, values(physmp))
    @test physmp[:spin] == U1Space(0 => 2)
    Omp = ttno_from_opsum(Hmp, topomp, physmp; hermitian=true)
    @test check_arrows(Omp)
end

@testset "expectation values & overlaps" begin
    topo = star_topology(3, 2)
    phys = allspin(topo)
    S = spin_ops()
    H = tfi(topo)
    O = ttno_from_opsum(H, topo, phys; hermitian=true)
    ψ = random_ttns(RNG, ComplexF64, topo, phys, ℂ^4)
    v = to_dense(ψ)
    Hd = dense_hamiltonian(H, ψ)
    @test abs(expect(ψ, O) - v' * Hd * v) < 1e-12
    @test abs(expect(ψ, S.Z, :b2_1) -
              v' * dense_hamiltonian(OpSum() + Term(1.0, SiteOp(:b2_1, :Z, S.Z)), ψ) * v) < 1e-12
    φ = random_ttns(RNG, ComplexF64, topo, phys, ℂ^3)
    @test abs(inner(φ, ψ) - dot(to_dense(φ), v)) < 1e-12
    # eff_h1 closes to the expectation value at the center
    cache = EnvCache(topo)
    h1 = GRAFT.eff_h1(cache, ψ, O, ψ.center)
    @test abs(dot(ψ.tensors[ψ.center], h1(ψ.tensors[ψ.center])) - expect(ψ, O)) < 1e-12
end

@testset "two-site merge/split roundtrip" begin
    topo = star_topology(3, 2)
    phys = allspin(topo)
    ψ = random_ttns(RNG, ComplexF64, topo, phys, ℂ^3)
    v = to_dense(ψ)
    n = nodeindex(topo, :b2_2); m = nodeindex(topo, :b2_1)
    move_center!(ψ, n)
    Θ = two_site_tensor(ψ, n, m)
    split_two_site!(ψ, Θ, n, m; center_on=:m)
    @test norm(to_dense(ψ) - v) < 1e-12
    @test check_arrows(ψ)
end

@testset "DMRG vs ED" begin
    topo = star_topology(3, 1)
    phys = allspin(topo)
    H = tfi(topo; g=0.9)
    O = ttno_from_opsum(H, topo, phys; hermitian=true)
    ψ = random_ttns(RNG, ComplexF64, topo, phys, ℂ^2)
    E0, _ = exact_groundstate(dense_hamiltonian(H, ψ))
    _, Es = dmrg2!(ψ, O; trunc=TruncationScheme(maxdim=16), nsweeps=8)
    @test Es[end] ≈ E0 atol = 1e-10
    _, Es1 = dmrg1!(ψ, O; nsweeps=4)
    @test Es1[end] ≈ E0 atol = 1e-10

    ψx = random_ttns(RNG, ComplexF64, topo, phys, ℂ^1)
    vx = to_dense(ψx)
    leaf = leaves(topo)[1]
    expand!(ψx, O, (leaf, topo.parent[leaf]);
            trunc=TruncationScheme(maxdim=4), max_add=3)
    @test norm(to_dense(ψx) - vx) < 1e-10
    @test maximum(bonddims(ψx)) > 1
    ψr = random_ttns(MersenneTwister(1101), ComplexF64, topo, phys, ℂ^1)
    @test_throws ArgumentError expand!(copy(ψr), O, (leaf, topo.parent[leaf]);
                                       scheme=:rsvd,
                                       trunc=TruncationScheme(maxdim=4), max_add=1)
    ψr1, ψr2 = copy(ψr), copy(ψr)
    expand!(ψr1, O, (leaf, topo.parent[leaf]); scheme=:rsvd,
            rng=MersenneTwister(2202), trunc=TruncationScheme(maxdim=4),
            max_add=3, rsvd_oversample=2)
    expand!(ψr2, O, (leaf, topo.parent[leaf]); scheme=:rsvd,
            rng=MersenneTwister(2202), trunc=TruncationScheme(maxdim=4),
            max_add=3, rsvd_oversample=2)
    @test norm(to_dense(ψr1) - to_dense(ψr)) < 1e-10
    @test norm(to_dense(ψr1) - to_dense(ψr2)) < 1e-12
    @test maximum(bonddims(ψr1)) > 1

    ψ3 = random_ttns(RNG, ComplexF64, topo, phys, ℂ^1)
    _, Es3 = dmrg1_3s!(ψ3, O; trunc=TruncationScheme(maxdim=16),
                       nsweeps=5, max_add=4)
    @test Es3[end] ≈ E0 atol = 1e-8
    @test maximum(bonddims(ψ3)) > 1
end

@testset "bosons (trivial sector)" begin
    S = spin_ops()
    B = boson_ops(2)
    @test norm(convert(Array, B.X) - (convert(Array, B.B) + convert(Array, B.Bd))) < 1e-14
    @test eltype(convert(Array, B.X)) == Float64

    topo = star_topology(2, 1; center=:spin, prefix=:b)
    phys = Dict(:spin => S.P, :b1_1 => B.P, :b2_1 => B.P)
    H = boson_modes([:b1_1 => 0.7, :b2_1 => 1.1]; ops=B)
    H += Term(-0.35, SiteOp(:spin, :X, S.X))
    H += BosonCoupling([(:spin, :b1_1) => 0.22, (:spin, :b2_1) => -0.18],
                       :density; matter_ops=S, boson_ops=B, density=:Z)
    O = ttno_from_opsum(H, topo, phys; hermitian=true)
    ψ0 = random_ttns(RNG, ComplexF64, topo, phys, ℂ^6)
    Hd = dense_hamiltonian(H, ψ0)
    E0, _ = exact_groundstate(Hd)
    ψd = copy(ψ0)
    _, Es = dmrg2!(ψd, O; trunc=TruncationScheme(maxdim=12, atol=1e-12), nsweeps=6)
    @test Es[end] ≈ E0 atol = 1e-9

    dt = 0.01
    nsteps = 2
    ψt = random_ttns(RNG, ComplexF64, topo, phys, ℂ^8)
    vex = exact_evolve(Hd, to_dense(ψt), -im * dt * nsteps)
    for ev in (TDVP1(order=2),
               TDVP2(trunc=TruncationScheme(maxdim=16, atol=1e-12)),
               TDVP1_CBE(trunc=TruncationScheme(maxdim=16, atol=1e-12),
                         d_tilde_max=4, enr_rtol=1e-12, enr_atol=1e-12))
        ψe = copy(ψt)
        for _ in 1:nsteps
            step!(ev, ψe, O, -im * dt)
        end
        @test abs(1 - abs(dot(to_dense(ψe), vex))) < 1e-7
    end

    ψim = random_ttns(RNG, ComplexF64, topo, phys, ℂ^4)
    evi = TDVP2(trunc=TruncationScheme(maxdim=12, atol=1e-10))
    for _ in 1:30
        step!(evi, ψim, O, -0.12)
        normalize!(ψim)
    end
    @test real(expect(ψim, O)) ≈ E0 atol = 5e-3

    base = mps_topology(3)
    hol = base
    for i in 1:3
        hol = mount_chain(hol, Symbol(:site, i), 1; prefix=Symbol(:ph, i, :_))
    end
    @test nnodes(base) == 3
    @test nnodes(hol) == 6
    @test all(nodeindex(hol, Symbol(:ph, i, :_1)) > 0 for i in 1:3)

    Bh = boson_ops(1)
    phys_h = Dict{Symbol,typeof(S.P)}()
    for i in 1:3
        phys_h[Symbol(:site, i)] = S.P
        phys_h[Symbol(:ph, i, :_1)] = Bh.P
    end
    Hh = boson_modes([Symbol(:ph, i, :_1) => 0.65 for i in 1:3]; ops=Bh)
    for i in 1:3
        Hh += Term(-0.45, SiteOp(Symbol(:site, i), :N, S.N))
        Hh += BosonCoupling([(Symbol(:site, i), Symbol(:ph, i, :_1)) => 0.35],
                            :density; matter_ops=S, boson_ops=Bh, density=:N)
    end
    for i in 1:2
        a = Symbol(:site, i)
        b = Symbol(:site, i + 1)
        Hh += Term(-0.25, SiteOp(a, :Sp, S.Sp), SiteOp(b, :Sm, S.Sm))
        Hh += Term(-0.25, SiteOp(a, :Sm, S.Sm), SiteOp(b, :Sp, S.Sp))
        Hh += Term(0.08, SiteOp(a, :Z, S.Z), SiteOp(b, :Z, S.Z))
    end
    Oh = ttno_from_opsum(Hh, hol, phys_h; hermitian=true)
    ψh = random_ttns(RNG, ComplexF64, hol, phys_h, ℂ^4)
    Hdh = dense_hamiltonian(Hh, ψh)
    E0h, v0h = exact_groundstate(Hdh)
    _, Esh = dmrg2!(ψh, Oh; trunc=TruncationScheme(maxdim=16, atol=1e-12), nsweeps=6)
    @test Esh[end] ≈ E0h atol = 1e-9
    Nb = boson_modes([Symbol(:ph, i, :_1) => 1.0 for i in 1:3]; ops=Bh)
    @test real(v0h' * dense_hamiltonian(Nb, ψh) * v0h) > 1e-4
end

@testset "TDVP vs exact propagation" begin
    topo = star_topology(3, 2)
    phys = allspin(topo)
    H = tfi(topo; g=0.9)
    O = ttno_from_opsum(H, topo, phys; hermitian=true)
    dt = 0.02; nsteps = 5

    # full-rank manifold: TDVP1 ≡ exact evolution
    ψ = random_ttns(RNG, ComplexF64, topo, phys, ℂ^8)
    Hd = dense_hamiltonian(H, ψ)
    vex = exact_evolve(Hd, to_dense(ψ), -im * dt * nsteps)
    ev = TDVP1(order=2)
    for _ in 1:nsteps
        step!(ev, ψ, O, -im * dt)
    end
    @test abs(1 - abs(dot(to_dense(ψ), vex))) < 1e-10
    @test norm(ψ) ≈ 1 atol = 1e-10
    @test abs(expect(ψ, O) - dot(vex, Hd * vex)) < 1e-8     # energy conservation

    # bond growth from χ=1: TDVP2 exact, CBE close, TDVP1 stuck
    ψ1 = random_ttns(RNG, ComplexF64, topo, phys, ℂ^1)
    Hd = dense_hamiltonian(H, ψ1)
    vex = exact_evolve(Hd, to_dense(ψ1), -im * dt * 10)
    ψ2 = copy(ψ1); ψ3 = copy(ψ1)

    ev2 = TDVP2(trunc=TruncationScheme(maxdim=16, atol=1e-12))
    for _ in 1:10
        step!(ev2, ψ2, O, -im * dt)
    end
    @test abs(1 - abs(dot(to_dense(ψ2), vex))) < 1e-10
    @test maximum(bonddims(ψ2)) > 1

    evc = TDVP1_CBE(trunc=TruncationScheme(maxdim=16, atol=1e-12),
                    d_tilde_max=8, enr_rtol=1e-10, enr_atol=1e-12)
    for _ in 1:10
        step!(evc, ψ3, O, -im * dt)
    end
    @test abs(1 - abs(dot(to_dense(ψ3), vex))) < 1e-6
    @test maximum(bonddims(ψ3)) > 1

    # behavioral contract from the PyTreeNet fork's test suite:
    # disabled enrichment reproduces TDVP1 exactly
    ψa = copy(ψ1); ψb = copy(ψ1)
    eva = TDVP1(order=2)
    evb = TDVP1_CBE(order=2, enabled=false)
    for _ in 1:3
        step!(eva, ψa, O, -im * dt)
        step!(evb, ψb, O, -im * dt)
    end
    @test norm(to_dense(ψa) - to_dense(ψb)) < 1e-14
end

@testset "GlobalKrylov vs exact propagation" begin
    topo = mps_topology(2)
    phys = allspin(topo)
    H = tfi(topo; g=0.4)
    O = ttno_from_opsum(H, topo, phys; hermitian=true)
    ψ = random_ttns(RNG, ComplexF64, topo, phys, ℂ^4)
    Hd = dense_hamiltonian(H, ψ)
    v0 = to_dense(ψ)
    dz = -0.03im
    ev = GlobalKrylov(krylovdim=8, maxiter=4, fit_nsweeps=5,
                      fit_tol=1e-11, tol=1e-10)
    step!(ev, ψ, O, dz)
    @test ev.last_info !== nothing
    @test ev.last_info.converged == 1
    @test norm(to_dense(ψ) - exact_evolve(Hd, v0, dz)) < 1e-7

    ψr = random_ttns(RNG, Float64, topo, phys, ℂ^4)
    @test_throws ArgumentError step!(GlobalKrylov(), ψr, O, dz)
end

@testset "linear solve and implicit log time" begin
    topo = mps_topology(2)
    phys = allspin(topo)
    H = tfi(topo; g=0.4)
    O = ttno_from_opsum(H, topo, phys; hermitian=true)
    rhs = random_ttns(RNG, ComplexF64, topo, phys, ℂ^4)
    Hd = dense_hamiltonian(H, rhs)
    v = to_dense(rhs)
    A = Matrix{ComplexF64}(I, length(v), length(v)) + 0.05 * Hd

    ψ = copy(rhs)
    _, info = linsolve!(ψ, O, rhs; a0=1.0, a1=0.05,
                        krylovdim=8, maxiter=4, tol=1e-10,
                        fit_nsweeps=6, fit_tol=1e-11)
    @test info.converged == 1
    @test norm(to_dense(ψ) - (A \ v)) < 1e-7

    ψτ = copy(rhs)
    ev = ImplicitLogTime(krylovdim=8, maxiter=4, tol=1e-10,
                         fit_nsweeps=6, fit_tol=1e-11)
    step!(ev, ψτ, O, -0.05)
    @test ev.last_info !== nothing
    @test ev.last_info.converged == 1
    @test norm(to_dense(ψτ) - (A \ v)) < 1e-7
    @test_throws ArgumentError step!(ev, copy(rhs), O, -0.05im)
end

@testset "subspace expansion TDVP vs exact propagation" begin
    topo = mps_topology(2)
    phys = allspin(topo)
    H = tfi(topo; g=0.4)
    O = ttno_from_opsum(H, topo, phys; hermitian=true)
    ψ0 = random_ttns(RNG, ComplexF64, topo, phys, ℂ^1)
    Hd = dense_hamiltonian(H, ψ0)
    v0 = to_dense(ψ0)
    dz = -0.02im
    vex = exact_evolve(Hd, v0, dz)

    for ev in (GSE_TDVP(order=2, trunc=TruncationScheme(maxdim=4, atol=1e-12),
                        max_add=3, krylovdim=8, tol=1e-10),
               LSE_TDVP(order=2, trunc=TruncationScheme(maxdim=4, atol=1e-12),
                        max_add=3, krylovdim=8, tol=1e-10))
        ψ = copy(ψ0)
        step!(ev, ψ, O, dz)
        @test check_arrows(ψ)
        @test maximum(bonddims(ψ)) > 1
        @test abs(1 - abs(dot(to_dense(ψ), vex))) < 1e-5
    end

    ψr = random_ttns(RNG, Float64, topo, phys, ℂ^1)
    @test_throws ArgumentError step!(GSE_TDVP(), ψr, O, dz)
end

@testset "imaginary time (complex-step contract §5b)" begin
    topo = mps_topology(6)
    phys = allspin(topo)
    H = tfi(topo; g=1.1)
    O = ttno_from_opsum(H, topo, phys; hermitian=true)
    ψ = random_ttns(RNG, ComplexF64, topo, phys, ℂ^2)
    E0, _ = exact_groundstate(dense_hamiltonian(H, ψ))
    ev = TDVP2(trunc=TruncationScheme(maxdim=16, atol=1e-10))
    for _ in 1:80
        step!(ev, ψ, O, -0.1)          # dz = -δτ: e^{-τH}
        normalize!(ψ)
    end
    @test real(expect(ψ, O)) ≈ E0 atol = 1e-3
    @test supports_complex_step(TDVP1)
    @test !supports_complex_step(ImplicitLogTime)
    @test !isempty(methods(linsolve!))
end

@testset "local insertions and zero-temperature correlators" begin
    S = spin_ops()
    topo = mps_topology(2)
    phys = allspin(topo)
    ψ0 = product_ttns(ComplexF64, topo, Dict(:site1 => [1.0, 0.0], :site2 => [0.0, 1.0]))
    ω = 0.4
    H = OpSum()
    H += Term(ω, SiteOp(:site1, :Z, S.Z))
    O = ttno_from_opsum(H, topo, phys; hermitian=true)
    v0 = to_dense(ψ0)
    E0 = real(dot(v0, dense_hamiltonian(H, ψ0) * v0))

    ϕ = apply_local(ψ0, S.X, :site1)
    @test center(ϕ) == nodeindex(topo, :site1)
    @test check_arrows(ϕ)
    @test norm(to_dense(ϕ) - dense_hamiltonian(OpSum() + Term(1.0, SiteOp(:site1, :X, S.X)), ψ0) * v0) < 1e-12
    @test norm(to_dense(ψ0) - v0) < 1e-12

    ts = [0.0, 0.05, 0.11]
    vals = correlator(ψ0, E0, :site1 => S.X, :site1 => S.X, ts;
                      H=O, evolver=LocalZEvolver(:site1, ω))
    series = correlator_series(ψ0, E0, :site1 => S.X, :site1 => S.X, ts;
                               H=O, evolver=LocalZEvolver(:site1, ω),
                               metadata=(; kind=:test))
    @test series isa CorrelatorSeries
    @test collect(series) == collect(zip(ts, vals))
    @test series.metadata.kind == :test
    @test series.metadata.Asite == :site1
    X1 = dense_hamiltonian(OpSum() + Term(1.0, SiteOp(:site1, :X, S.X)), ψ0)
    Hd = dense_hamiltonian(H, ψ0)
    ref = [exp(im * E0 * t) * dot(X1 * v0, exact_evolve(Hd, X1 * v0, -im * t)) for t in ts]
    @test norm(vals - ref) < 1e-10

    topoχ = mps_topology(1)
    physχ = allspin(topoχ)
    ψχ = product_ttns(ComplexF64, topoχ, Dict(:site1 => ComplexF64[1, 1] / sqrt(2)))
    Hχ = OpSum() + Term(ω, SiteOp(:site1, :X, S.X))
    Oχ = ttno_from_opsum(Hχ, topoχ, physχ; hermitian=true)
    vχ = to_dense(ψχ)
    Eχ = real(dot(vχ, dense_hamiltonian(Hχ, ψχ) * vχ))
    Nχ = dense_hamiltonian(OpSum() + Term(1.0, SiteOp(:site1, :N, S.N)), ψχ)
    nbar = expect(ψχ, S.N, :site1)
    valsχ = correlator(ψχ, Eχ, :site1 => S.N, :site1 => S.N, ts;
                       H=Oχ, evolver=LocalXEvolver(:site1, ω)) .- nbar^2
    refχ = [exp(im * Eχ * t) * dot(Nχ * vχ, exact_evolve(dense_hamiltonian(Hχ, ψχ), Nχ * vχ, -im * t)) - nbar^2
            for t in ts]
    @test norm(valsχ - refχ) < 1e-10
    @test maximum(abs.(valsχ .- valsχ[1])) > 1e-4
end

@testset "boson bath fitting and mounting" begin
    S = spin_ops()
    B = boson_ops(2)
    P = Partition([[:imp]])
    J(ω) = 0.2 * ω
    bath = fit_bath(J, P; nmodes=3, ωmin=0.5, ωmax=2.0)
    @test bath isa RealPoles
    @test bath.diagnostics.domain == :real_axis
    @test bath.blocks == P.blocks
    @test bath.block_ranges == [1:3]
    @test bath.poles ≈ [0.75, 1.25, 1.75]
    @test bath.residues ≈ 0.2 .* bath.poles .* 0.5
    @test couplings(bath) ≈ sqrt.(bath.residues)
    @test bath.diagnostics.block_diagnostics[1].rel_weight_change < 0.2
    bathT = fit_bath(J, P; T=1.0, nmodes=2, ωmin=0.1, ωmax=1.0)
    @test bathT isa ThermofieldRealPoles
    @test bathT.diagnostics.representation == :thermofield_star

    νs = [0.0, 0.4, 1.0, 2.0, 4.0, 8.0]
    exact_poles = [0.75, 1.25, 1.75]
    exact_residues = [0.04, 0.02, 0.01]
    Uν = [sum(2 * ω * r / (ν^2 + ω^2)
              for (ω, r) in zip(exact_poles, exact_residues)) for ν in νs]
    bath_m = fit_bath((; frequencies=im .* νs, values=Uν), P;
                      domain=:matsubara, nmodes=3, ωmin=0.5, ωmax=2.0)
    @test bath_m isa RealPoles
    @test bath_m.diagnostics.domain == :matsubara
    @test bath_m.poles ≈ exact_poles
    @test bath_m.residues ≈ exact_residues atol = 1e-10
    @test matsubara_reconstruct(bath_m, im .* νs) ≈ Uν atol = 1e-10
    @test bath_m.diagnostics.block_diagnostics[1].relative_residual < 1e-10

    P2 = Partition([[:a], [:b]])
    Uν2 = 0.5 .* Uν
    bath_m2 = fit_bath((; frequencies=νs, values=hcat(Uν, Uν2)), P2;
                       domain=:matsubara, nmodes=3, ωmin=0.5, ωmax=2.0)
    @test bath_m2.block_ranges == [1:3, 4:6]
    @test matsubara_reconstruct(bath_m2, νs; block=1) ≈ Uν atol = 1e-10
    @test matsubara_reconstruct(bath_m2, νs; block=2) ≈ Uν2 atol = 1e-10

    topo = TreeTopology(:imp, Pair{Symbol,Symbol}[])
    mounted = mount_bath(topo, bath, P; prefix=:ph)
    @test nnodes(mounted.topology) == 4
    @test mounted.sites == [:ph1_1_1, :ph1_2_1, :ph1_3_1]
    @test all(==(:imp), mounted.anchors)
    mounted_chain = mount_bath(topo, bath, P; mode=:chain, prefix=:ch)
    @test nnodes(mounted_chain.topology) == 4
    @test mounted_chain.sites == [:ch1_1, :ch1_2, :ch1_3]
    @test mounted_chain.block_sites == [mounted_chain.sites]

    bb = BosonBath(J; partition=P, topology=topo, matter_ops=S, boson_ops=B,
                   nmodes=2, ωmin=0.5, ωmax=1.5, prefix=:bfit, density=:Z)
    @test bb.bath.poles ≈ [0.75, 1.25]
    @test bb.sites == [:bfit1_1_1, :bfit1_2_1]
    @test Set(keys(bb.phys)) == Set(bb.sites)
    phys = merge(Dict(:imp => S.P), bb.phys)
    O = ttno_from_opsum(bb.H, bb.topology, phys; hermitian=true)
    ψ = random_ttns(RNG, ComplexF64, bb.topology, phys, ℂ^3)
    @test norm(to_dense(O) - dense_hamiltonian(bb.H, ψ)) < 1e-12
end

include("b4_finite_temperature_bath.jl")

if lowercase(get(ENV, "GRAFT_LONG_B5", "false")) in ("1", "true", "yes")
    include("b5_holstein_tdvp_chi.jl")
    include("b5_fz2_hopping_green.jl")
end

if lowercase(get(ENV, "GRAFT_LONG_B4", "false")) in ("1", "true", "yes")
    include("b4_bath_tdvp_e2e.jl")
end

if lowercase(get(ENV, "GRAFT_LONG_B2", "false")) in ("1", "true", "yes")
    include("b2_graded_kernel_smoke.jl")
end

if lowercase(get(ENV, "GRAFT_LONG_B3", "false")) in ("1", "true", "yes")
    include("b3_pp_lbo.jl")
end

@testset "checkpoint / resume" begin
    topo = mps_topology(4)
    phys = allspin(topo)
    ψ = random_ttns(RNG, ComplexF64, topo, phys, ℂ^3)
    path = joinpath(mktempdir(), "state.jld2")
    checkpoint!((; ψ, step=7, rng=copy(RNG)), path; metadata=(; bathhash=0x1234))
    st = resume(path)
    @test st.state.step == 7
    @test st.metadata.bathhash == 0x1234
    @test norm(to_dense(st.state.ψ) - to_dense(ψ)) < 1e-14
    # rotation: another write keeps the previous file as .1
    checkpoint!((; ψ, step=8), path)
    @test isfile(path * ".1")
    @test resume(path).state.step == 8

    # mixed spin/boson trivial-sector states are the B1/B3 checkpoint
    # boundary: future PP state must remain a serializable solver-state value.
    b = boson_ops(2)
    topo_b = mps_topology(2)
    phys_b = Dict(:site1 => spin_ops().P, :site2 => b.P)
    ψb = random_ttns(RNG, ComplexF64, topo_b, phys_b, ℂ^2)
    path_b = joinpath(mktempdir(), "boson_state.jld2")
    checkpoint!((; ψ=ψb, step=1, trunc=TruncationScheme(maxdim=8)), path_b)
    @test norm(to_dense(resume(path_b).state.ψ) - to_dense(ψb)) < 1e-14

    # JLD2 must also preserve TensorKit graded spaces, because B2/B3 introduce
    # charged virtual/physical spaces before any specialized checkpoint schema.
    topo_g = mps_topology(3)
    Pg = U1Space(0 => 1, 1 => 1)
    Vg = U1Space(-1 => 1, 0 => 2, 1 => 1)
    phys_g = Dict(nodeid(topo_g, i) => Pg for i in 1:nnodes(topo_g))
    ψg = random_ttns(RNG, ComplexF64, topo_g, phys_g, Vg)
    path_g = joinpath(mktempdir(), "graded_state.jld2")
    checkpoint!((; ψ=ψg, step=2), path_g; metadata=(; sector=:U1))
    rg = resume(path_g)
    @test rg.metadata.sector == :U1
    @test norm(to_dense(rg.state.ψ) - to_dense(ψg)) < 1e-14

    path_iter = joinpath(mktempdir(), "iter_state.jld2")
    states = ((; step=i, value=i^2) for i in 1:5)
    collected = collect(with_checkpoint(states; every=2, path=path_iter, keep=2,
                                        metadata=(value, count) -> (; count),
                                        statefn=value -> value))
    @test length(collected) == 5
    @test resume(path_iter).state.step == 4
    @test resume(path_iter).metadata.count == 4
    @test resume(path_iter * ".1").state.step == 2
end

include("contraction_planning.jl")

end
