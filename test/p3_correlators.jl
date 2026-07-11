using Test
using GRAFT
using GRAFT.TestUtils
using GRAFT.Backend
using LinearAlgebra: norm, tr, I
using Random

const RNG = Xoshiro(20260712)
const QUIET = (verbose=false,)

@testset "P3 thermal correlators" begin
    @testset "neutral C_ZZ(tau) vs dense trace" begin
        S = spin_ops()
        topo = mps_topology(1)
        phys = Dict(:site1 => S.P)
        H = OpSum() + Term(0.5, SiteOp(:site1, :Z, S.Z))
        prob = purification_problem(H, topo, phys; hermitian=true)
        Hd = dense_hamiltonian(H, topo, phys)
        beta = 2.0
        taus = [0.0, 0.5, 1.0, 1.5, 2.0]
        series = thermal_correlator(Purified(), prob,
            :site1 => S.Z, :site1 => S.Z, beta, taus;
            evolver=TDVP2(trunc=TruncationScheme(maxdim=4); QUIET...),
            prep_nsteps=40, prop_nsteps=40)
        Zd = dense_hamiltonian(OpSum() + Term(1.0, SiteOp(:site1, :Z, S.Z)), topo, phys)
        ref = exact_thermal_correlator(Hd, Zd, Zd, beta, taus)
        @test maximum(abs.(series.values .- ref)) < 1e-10
    end

    @testset "connected chi_nn(tau) vs dense trace" begin
        S = spin_ops()
        topo = mps_topology(2)
        phys = Dict(nodeid(topo, i) => S.P for i in 1:2)
        H = OpSum()
        for (c, p) in GRAFT.Trees.edges(topo)
            H += Term(-1.0, SiteOp(nodeid(topo, c), :Z, S.Z),
                       SiteOp(nodeid(topo, p), :Z, S.Z))
        end
        H += Term(-0.3, SiteOp(:site1, :X, S.X))
        prob = purification_problem(H, topo, phys; hermitian=true)
        Hd = dense_hamiltonian(H, topo, phys)
        beta = 1.0
        taus = [0.0, 0.25, 0.5, 0.75, 1.0]
        series = thermal_correlator(Purified(), prob,
            :site1 => S.N, :site1 => S.N, beta, taus;
            evolver=TDVP2(trunc=TruncationScheme(maxdim=8); QUIET...),
            prep_nsteps=30, prop_nsteps=30, connected=true)
        Nd = dense_hamiltonian(OpSum() + Term(1.0, SiteOp(:site1, :N, S.N)), topo, phys)
        ref = exact_thermal_correlator(Hd, Nd, Nd, beta, taus)
        nbar = real(exact_thermal_expect(Hd, Nd, beta))
        ref_c = ref .- nbar^2
        @test maximum(abs.(series.values .- ref_c)) < 1e-9
    end

    @testset "KMS endpoint convention" begin
        S = spin_ops()
        topo = mps_topology(1)
        phys = Dict(:site1 => S.P)
        H = OpSum() + Term(0.5, SiteOp(:site1, :Z, S.Z)) +
             Term(0.3, SiteOp(:site1, :X, S.X))
        prob = purification_problem(H, topo, phys; hermitian=true)
        Hd = dense_hamiltonian(H, topo, phys)
        beta = 1.0
        Zd = dense_hamiltonian(OpSum() + Term(1.0, SiteOp(:site1, :Z, S.Z)), topo, phys)
        Xd = dense_hamiltonian(OpSum() + Term(1.0, SiteOp(:site1, :X, S.X)), topo, phys)

        # bosonic periodicity: C(tau=0) = C(tau=beta)
        series_Z = thermal_correlator(Purified(), prob,
            :site1 => S.Z, :site1 => S.Z, beta, [0.0, beta];
            evolver=TDVP2(trunc=TruncationScheme(maxdim=4); QUIET...),
            prep_nsteps=40, prop_nsteps=40)
        ref_Z = exact_thermal_correlator(Hd, Zd, Zd, beta, [0.0, beta])
        @test abs(series_Z.values[1] - series_Z.values[2]) < 1e-8
        @test abs(ref_Z[1] - ref_Z[2]) < 1e-12
    end

    @testset "correlator via shared trajectory" begin
        S = spin_ops()
        topo = mps_topology(1)
        phys = Dict(:site1 => S.P)
        H = OpSum() + Term(0.5, SiteOp(:site1, :Z, S.Z)) +
             Term(0.3, SiteOp(:site1, :X, S.X))
        prob = purification_problem(H, topo, phys; hermitian=true)
        beta = 1.0
        taus = [0.0, 0.25, 0.5, 0.75, 1.0]
        save_betas = sort(unique(vcat(beta .- taus, [beta])))
        traj = thermalize(Purified(), prob, beta;
            evolver=TDVP2(trunc=TruncationScheme(maxdim=4); QUIET...),
            nsteps=40, save_betas=save_betas)

        s1 = thermal_correlator(Purified(), prob,
            :site1 => S.Z, :site1 => S.Z, beta, taus;
            evolver=TDVP2(trunc=TruncationScheme(maxdim=4); QUIET...),
            trajectory=traj, prop_nsteps=40)
        s2 = thermal_correlator(Purified(), prob,
            :site1 => S.Z, :site1 => S.Z, beta, taus;
            evolver=TDVP2(trunc=TruncationScheme(maxdim=4); QUIET...),
            prep_nsteps=40, prop_nsteps=40)
        @test maximum(abs.(s1.values .- s2.values)) < 1e-10
    end

    @testset "trajectory mismatch rejected" begin
        S = spin_ops()
        topo = mps_topology(1)
        phys = Dict(:site1 => S.P)
        H = OpSum() + Term(0.5, SiteOp(:site1, :Z, S.Z))
        prob = purification_problem(H, topo, phys; hermitian=true)
        traj = thermalize(Purified(), prob, 1.0;
            evolver=TDVP2(; QUIET...), nsteps=10)
        @test_throws ArgumentError thermal_correlator(Purified(), prob,
            :site1 => S.Z, :site1 => S.Z, 2.0, [0.0, 1.0];
            evolver=TDVP2(; QUIET...), trajectory=traj, prop_nsteps=10)
    end

    @testset "U1 spin correlator vs dense" begin
        U = spin_ops_u1()
        topo = mps_topology(1)
        phys = Dict(:site1 => U.P)
        H = OpSum() + Term(0.5, SiteOp(:site1, :Z, U.Z))
        prob = purification_problem(H, topo, phys; hermitian=true)
        Hd = dense_hamiltonian(H, topo, phys)
        beta = 1.0
        taus = [0.0, 0.25, 0.5, 0.75, 1.0]
        series = thermal_correlator(Purified(), prob,
            :site1 => U.Z, :site1 => U.Z, beta, taus;
            evolver=TDVP2(trunc=TruncationScheme(maxdim=4); QUIET...),
            prep_nsteps=40, prop_nsteps=40)
        Zd = dense_hamiltonian(OpSum() + Term(1.0, SiteOp(:site1, :Z, U.Z)), topo, phys)
        ref = exact_thermal_correlator(Hd, Zd, Zd, beta, taus)
        @test maximum(abs.(series.values .- ref)) < 1e-8
    end

    @testset "two-site Holstein density correlator vs ED" begin
        S = spin_ops()
        B = boson_ops(1)
        topo = mps_topology(1)
        topo = mount_chain(topo, :site1, 1; prefix=:ph)
        phys = Dict(:site1 => S.P, :ph1 => B.P)
        H = boson_modes([:ph1 => 0.5]; ops=B)
        H += Term(-0.3, SiteOp(:site1, :N, S.N))
        H += BosonCoupling([(:site1, :ph1) => 0.2], :density;
            matter_ops=S, boson_ops=B, density=:N)
        prob = purification_problem(H, topo, phys; hermitian=true)
        Hd = dense_hamiltonian(H, topo, phys)
        beta = 1.0
        taus = [0.0, 0.25, 0.5, 0.75, 1.0]
        series = thermal_correlator(Purified(), prob,
            :site1 => S.N, :site1 => S.N, beta, taus;
            evolver=TDVP2(trunc=TruncationScheme(maxdim=8); QUIET...),
            prep_nsteps=40, prop_nsteps=40)
        Nd = dense_hamiltonian(OpSum() + Term(1.0, SiteOp(:site1, :N, S.N)), topo, phys)
        ref = exact_thermal_correlator(Hd, Nd, Nd, beta, taus)
        @test maximum(abs.(series.values .- ref)) < 1e-7
    end

    @testset "fermion correlator at beta=0 via to_dense" begin
        F = fermion_ops_z2()
        topo = mps_topology(2)
        phys = Dict(nodeid(topo, i) => F.P for i in 1:2)
        H = OpSum() + Term(0.0, SiteOp(:site1, :I, F.I))
        prob = purification_problem(H, topo, phys; hermitian=true)
        state0 = infinite_temperature_state(prob)

        bra = apply_local(state0.psi, adjoint(F.C), :site1)
        ket = apply_local(state0.psi, F.Cd, :site1)
        vb = to_dense(bra)
        vk = to_dense(ket)
        overlap = dot(vb, vk)
        @test real(overlap) ≈ 0.5 atol = 1e-12
    end
end
