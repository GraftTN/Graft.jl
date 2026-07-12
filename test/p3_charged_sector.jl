using Test
using Graft
using Graft.TestUtils
using Graft.Backend
using LinearAlgebra: I, dot, kron, norm
using Random

isdefined(@__MODULE__, Symbol("@graft_testset")) || include("test_harness.jl")

@graft_testset "P3 charged-root fermion propagation" begin
    F = fermion_ops_z2()
    topo = mps_topology(2)
    phys = Dict(nodeid(topo, i) => F.P for i in 1:2)
    H = OpSum()
    H += Term(-0.5, SiteOp(:site1, :N, F.N))
    H += Term(-1.0, SiteOp(:site1, :Cd, F.Cd), SiteOp(:site2, :C, F.C))
    H += Term(-1.0, SiteOp(:site1, :C, F.C), SiteOp(:site2, :Cd, F.Cd))
    prob = purification_problem(H, topo, phys; hermitian=true)

    neutral = random_ttns(
        Xoshiro(20260712), ComplexF64,
        prob.topo_doubled, prob.phys_doubled, F.P)

    @testset "neutral doubled fermion gauge and TDVP1 identity" begin
        root = neutral.topo.root
        reference = to_dense(neutral)
        @test collect(sectors(domain(neutral[root])[1])) == [FermionParity(0)]

        # Both directions of every center move must be a pure gauge change.
        # The dual thermal legs make the pivotal correction observable.
        for target in 1:nnodes(neutral.topo)
            centered = move_center!(copy(neutral), target)
            @test norm(to_dense(centered) - reference) < 1e-11
            move_center!(centered, root)
            @test norm(to_dense(centered) - reference) < 1e-11
            @test check_arrows(centered)
        end

        recanonicalized = canonicalize!(copy(neutral), root)
        @test norm(to_dense(recanonicalized) - reference) < 1e-11

        for order in (1, 2), mode in (:tdvp1, :cbe_off, :cbe_on)
            ev = if mode === :tdvp1
                TDVP1(; order, verbose=false)
            else
                TDVP1_CBE(
                    ; order, enabled=mode === :cbe_on,
                    trunc=TruncationScheme(maxdim=16),
                    d_tilde_max=8, verbose=false)
            end
            evolved = copy(neutral)
            step!(ev, evolved, prob.K, 0.0)
            @test norm(to_dense(evolved) - reference) < 1e-11
            @test norm(evolved) ≈ norm(neutral) atol = 1e-12 rtol = 0
            @test collect(sectors(domain(evolved[root])[1])) == [FermionParity(0)]
            @test check_arrows(evolved)
        end
    end

    psi = apply_local(neutral, F.Cd, :site1)
    root = psi.topo.root
    @test collect(sectors(domain(psi[root])[1])) == [FermionParity(1)]
    @test check_arrows(psi)

    expected = expect(psi, prob.K)
    for n in 1:nnodes(psi.topo)
        centered = move_center!(copy(psi), n)
        h1 = eff_h1(
            EnvCache(psi.topo), centered, prob.K, n;
            optimize=false, sector_aware=false)
        @test dot(centered[n], h1(centered[n])) ≈ expected atol = 1e-11
    end
    for n in 1:nnodes(psi.topo)
        m = psi.topo.parent[n]
        m == 0 && continue
        centered = move_center!(copy(psi), n)
        theta = Graft.Contractions.two_site_tensor(centered, n, m)
        h2 = eff_h2(
            EnvCache(psi.topo), centered, prob.K, n, m;
            optimize=false, sector_aware=false)
        @test dot(theta, h2(theta)) ≈ expected atol = 1e-11
    end

    initial = to_dense(psi)
    dz = -1e-3
    exact = exact_evolve(dense_hamiltonian(H, psi), initial, dz)

    for ev in (
            TDVP1(order=1, verbose=false),
            TDVP1_CBE(order=1, enabled=false, verbose=false),
            TDVP1_CBE(
                order=1, enabled=true,
                trunc=TruncationScheme(maxdim=16, atol=1e-12),
                d_tilde_max=8, verbose=false))
        trial = copy(psi)
        step!(ev, trial, prob.K, dz)
        evolved = to_dense(trial)
        fidelity = abs(dot(evolved, exact)) / (norm(evolved) * norm(exact))
        @test 1 - fidelity < 1e-6
        @test collect(sectors(domain(trial[root])[1])) == [FermionParity(1)]
        @test check_arrows(trial)
    end

    step!(TDVP2(
        order=1,
        trunc=TruncationScheme(maxdim=16, atol=1e-12),
        verbose=false), psi, prob.K, dz)
    evolved = to_dense(psi)
    fidelity = abs(dot(evolved, exact)) / (norm(evolved) * norm(exact))
    @test 1 - fidelity < 1e-7
    @test collect(sectors(domain(psi[root])[1])) == [FermionParity(1)]
    @test check_arrows(psi)

    beta = 0.04
    taus = [0.02]
    series = thermal_correlator(Purified(), prob,
        :site1 => F.C, :site1 => F.Cd, beta, taus;
        evolver=TDVP2(
            trunc=TruncationScheme(maxdim=16, atol=1e-12),
            verbose=false),
        prep_nsteps=2, prop_nsteps=2)
    Hd = dense_hamiltonian(H, topo, phys)
    identity2 = Matrix{ComplexF64}(I, 2, 2)
    C1 = kron(reshape(convert(Array, F.C), 2, 2), identity2)
    Cd1 = kron(reshape(convert(Array, F.Cd), 2, 2), identity2)
    reference = exact_thermal_correlator(Hd, C1, Cd1, beta, taus)
    @test maximum(abs.(series.values .- reference)) < 1e-7
end
