# GRAFT.jl test suite — every kernel is cross-validated against exact
# diagonalization / exact propagation on small trees, plus gauge-invariance
# property tests (architecture §9.11: this is a merge requirement, CI-enforced).
using Test
using GRAFT
using GRAFT.TestUtils
using GRAFT.Backend: ℂ, ⊗, ←, dim, domain, U1Space
using GRAFT.Trees: edges
using GRAFT.Contractions: two_site_tensor, split_two_site!
using Random
using LinearAlgebra: dot, norm

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
    topo = star_topology(3, 2)
    phys = allspin(topo)
    H = tfi(topo; g=0.9)
    O = ttno_from_opsum(H, topo, phys; hermitian=true)
    ψ = random_ttns(RNG, ComplexF64, topo, phys, ℂ^2)
    E0, _ = exact_groundstate(dense_hamiltonian(H, ψ))
    _, Es = dmrg2!(ψ, O; trunc=TruncationScheme(maxdim=16), nsweeps=8)
    @test Es[end] ≈ E0 atol = 1e-10
    _, Es1 = dmrg1!(ψ, O; nsweeps=4)
    @test Es1[end] ≈ E0 atol = 1e-10
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

end
