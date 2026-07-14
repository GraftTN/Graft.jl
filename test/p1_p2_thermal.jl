using Test
using Graft
using Graft.TestUtils
using Graft.Backend
using LinearAlgebra: norm, tr
using Random

isdefined(@__MODULE__, Symbol("@graft_testset")) || include("test_harness.jl")

const RNG = Xoshiro(20260712)
const QUIET = (verbose=false,)

@graft_testset "P1/P2 thermal purification" begin
    @testset "P1: binary tree beta=0" begin
        S = spin_ops()
        topo = binary_topology(2)
        phys = Dict(nodeid(topo, i) => S.P for i in 1:nnodes(topo))
        H = OpSum()
        for (c, p) in Graft.Trees.edges(topo)
            H += Term(-1.0, SiteOp(nodeid(topo, c), :Z, S.Z),
                       SiteOp(nodeid(topo, p), :Z, S.Z))
        end
        prob = purification_problem(H, topo, phys; hermitian=true)
        state0 = infinite_temperature_state(prob)
        @test norm(state0.psi) ≈ 1 atol = 1e-12
        @test state0.logZ ≈ log(2^nnodes(topo)) atol = 1e-12
        @test real(thermal_expect(state0,
            physical_ttno(prob, OpSum() + Term(1.0,
                SiteOp(nodeid(topo, 1), :N, S.N))))) ≈ 0.5 atol = 1e-12
    end

    if GRAFT_EXTENDED_TESTS
        @testset "P1: star topology beta=0" begin
            S = spin_ops()
            topo = star_topology(3, 1; center=:imp)
            phys = Dict(nodeid(topo, i) => S.P for i in 1:nnodes(topo))
            H = OpSum()
            for (c, p) in Graft.Trees.edges(topo)
                H += Term(-1.0, SiteOp(nodeid(topo, c), :Z, S.Z),
                           SiteOp(nodeid(topo, p), :Z, S.Z))
            end
            prob = purification_problem(H, topo, phys; hermitian=true)
            state0 = infinite_temperature_state(prob)
            @test norm(state0.psi) ≈ 1 atol = 1e-12
            @test state0.logZ ≈ log(2^nnodes(topo)) atol = 1e-12
        end

        @testset "P1: fermion+bath pair beta=0" begin
            F = fermion_ops_z2()
            topo = mps_topology(2)
            phys = Dict(nodeid(topo, i) => F.P for i in 1:2)
            H = OpSum()
            H += Term(-1.0, SiteOp(:site1, :Cd, F.Cd), SiteOp(:site2, :C, F.C))
            H += Term(-1.0, SiteOp(:site1, :C, F.C), SiteOp(:site2, :Cd, F.Cd))
            prob = purification_problem(H, topo, phys; hermitian=true)
            state0 = infinite_temperature_state(prob)
            @test norm(state0.psi) ≈ 1 atol = 1e-12
            @test state0.logZ ≈ log(4) atol = 1e-12
            @test real(thermal_expect(state0,
                physical_ttno(prob, OpSum() + Term(1.0,
                    SiteOp(:site1, :N, F.N))))) ≈ 0.5 atol = 1e-12
        end
    end

    @testset "P1: PP boson beta=0 (nmax=2)" begin
        B = boson_ops(2)
        PP = boson_ops_pp(2)
        topo = mps_topology(1)
        phys = Dict(:site1 => B.P)
        H = OpSum() + Term(0.7, SiteOp(:site1, :N, B.N))
        Hp, topop, physp = ppdress(H, topo, phys; nmax=2, boson_sites=[:site1])
        pp_pairs = Dict(:site1 => :site1_B1)
        prob = purification_problem(Hp, topop, physp;
                                    hermitian=true, pp_pairs=pp_pairs)
        state0 = infinite_temperature_state(prob)
        @test norm(state0.psi) ≈ 1 atol = 1e-12
        @test state0.logZ ≈ log(3) atol = 1e-12
        O_N = physical_ttno(prob, OpSum() + Term(1.0, SiteOp(:site1, :N, PP.N)))
        @test real(thermal_expect(state0, O_N)) ≈ 1.0 atol = 1e-12
    end

    @testset "P1: malformed pp_pairs rejected" begin
        B = boson_ops(2)
        topo = mps_topology(1)
        phys = Dict(:site1 => B.P)
        H = OpSum() + Term(1.0, SiteOp(:site1, :N, B.N))
        @test_throws ArgumentError purification_problem(H, topo, phys;
            pp_pairs=Dict(:nonexistent => :also_nonexistent))
    end

    @testset "P1: topology immutability" begin
        S = spin_ops()
        topo = mps_topology(2)
        phys = Dict(nodeid(topo, i) => S.P for i in 1:2)
        H = OpSum()
        for (c, p) in Graft.Trees.edges(topo)
            H += Term(-1.0, SiteOp(nodeid(topo, c), :Z, S.Z),
                       SiteOp(nodeid(topo, p), :Z, S.Z))
        end
        topo_copy = TreeTopology(topo.ids, copy(topo.index), copy(topo.parent),
                                   copy(topo.children), topo.root, copy(topo.depth))
        prob = purification_problem(H, topo, phys; hermitian=true)
        @test topo == topo_copy
    end

    @testset "P2: two-level system logZ and energy" begin
        S = spin_ops()
        topo = mps_topology(1)
        phys = Dict(:site1 => S.P)
        H = OpSum() + Term(0.5, SiteOp(:site1, :Z, S.Z))
        prob = purification_problem(H, topo, phys; hermitian=true)
        Hd = dense_hamiltonian(H, topo, phys)
        betas = GRAFT_EXTENDED_TESTS ? (0.5, 1.0, 2.0, 4.0) : (1.0,)
        for beta in betas
            traj = thermalize(Purified(), prob, beta;
                evolver=TDVP2(; QUIET...), nsteps=40)
            @test abs(traj.final.logZ - exact_thermal_logZ(Hd, beta)) < 1e-10
            @test abs(real(thermal_expect(traj.final, prob.K)) -
                      real(exact_thermal_expect(Hd, Hd, beta))) < 1e-10
        end
    end

    @testset "P2: two-site interacting vs ED" begin
        S = spin_ops()
        topo = mps_topology(2)
        phys = Dict(nodeid(topo, i) => S.P for i in 1:2)
        H = OpSum()
        for (c, p) in Graft.Trees.edges(topo)
            H += Term(-1.0, SiteOp(nodeid(topo, c), :Z, S.Z),
                       SiteOp(nodeid(topo, p), :Z, S.Z))
        end
        H += Term(-0.3, SiteOp(:site1, :X, S.X))
        prob = purification_problem(H, topo, phys; hermitian=true)
        Hd = dense_hamiltonian(H, topo, phys)
        beta = 1.0
        traj = thermalize(Purified(), prob, beta;
            evolver=TDVP2(trunc=TruncationScheme(maxdim=8); QUIET...), nsteps=30)
        @test abs(traj.final.logZ - exact_thermal_logZ(Hd, beta)) < 1e-9
        @test abs(real(thermal_expect(traj.final, prob.K)) -
                  real(exact_thermal_expect(Hd, Hd, beta))) < 1e-9
        O_N = physical_ttno(prob, OpSum() + Term(1.0, SiteOp(:site1, :N, S.N)))
        @test abs(real(thermal_expect(traj.final, O_N)) -
                  real(exact_thermal_expect(Hd,
                      dense_hamiltonian(OpSum() + Term(1.0,
                          SiteOp(:site1, :N, S.N)), topo, phys), beta))) < 1e-9
    end

    @testset "P2: truncated oscillator partition sum" begin
        B = boson_ops(1)
        topo = mps_topology(1)
        phys = Dict(:site1 => B.P)
        omega = 0.7
        H = OpSum() + Term(omega, SiteOp(:site1, :N, B.N))
        prob = purification_problem(H, topo, phys; hermitian=true)
        Hd = dense_hamiltonian(H, topo, phys)
        beta = 2.0
        traj = thermalize(Purified(), prob, beta;
            evolver=TDVP2(; QUIET...), nsteps=30)
        @test abs(traj.final.logZ - exact_thermal_logZ(Hd, beta)) < 1e-10
        Z_exact = 1 + exp(-beta * omega)
        @test abs(exp(traj.final.logZ) - Z_exact) < 1e-9
    end

    @testset "P2: uniform vs explicit grid" begin
        S = spin_ops()
        topo = mps_topology(1)
        phys = Dict(:site1 => S.P)
        H = OpSum() + Term(0.5, SiteOp(:site1, :Z, S.Z))
        prob = purification_problem(H, topo, phys; hermitian=true)
        beta = 1.0
        traj_u = thermalize(Purified(), prob, beta;
            evolver=TDVP2(; QUIET...), nsteps=20)
        grid = collect(range(0.0, beta / 2; length=21))
        traj_e = thermalize(Purified(), prob, beta;
            evolver=TDVP2(; QUIET...), tau_grid=grid)
        @test abs(traj_u.final.logZ - traj_e.final.logZ) < 1e-12
    end

    @testset "P2: ImplicitLogTime(normalize=true) rejected" begin
        S = spin_ops()
        topo = mps_topology(1)
        phys = Dict(:site1 => S.P)
        H = OpSum() + Term(0.5, SiteOp(:site1, :Z, S.Z))
        prob = purification_problem(H, topo, phys; hermitian=true)
        ev = ImplicitLogTime(normalize=true)
        @test_throws ArgumentError thermalize(Purified(), prob, 1.0;
            evolver=ev, nsteps=10)
    end

    @testset "P2: checkpoint state_at" begin
        S = spin_ops()
        topo = mps_topology(1)
        phys = Dict(:site1 => S.P)
        H = OpSum() + Term(0.5, SiteOp(:site1, :Z, S.Z))
        prob = purification_problem(H, topo, phys; hermitian=true)
        beta = 2.0
        save_betas = GRAFT_EXTENDED_TESTS ? [0.5, 1.0, 1.5] : [1.0]
        traj = thermalize(Purified(), prob, beta;
            evolver=TDVP2(; QUIET...), nsteps=40,
            save_betas=save_betas)
        @test haskey(traj.checkpoints, 0.0)
        @test haskey(traj.checkpoints, beta)
        for b in save_betas
            st = state_at(traj, b; atol=1e-10)
            @test abs(st.beta - b) < 1e-10
        end
        Hd = dense_hamiltonian(H, topo, phys)
        for b in [0.0, save_betas..., beta]
            st = state_at(traj, b; atol=1e-10)
            @test abs(st.logZ - exact_thermal_logZ(Hd, b)) < 1e-8
        end
    end

    @testset "P2: one-mode Holstein plain vs ED" begin
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
        betas = GRAFT_EXTENDED_TESTS ? (0.5, 1.0) : (1.0,)
        for beta in betas
            traj = thermalize(Purified(), prob, beta;
                evolver=TDVP2(trunc=TruncationScheme(maxdim=8); QUIET...),
                nsteps=40)
            @test abs(traj.final.logZ - exact_thermal_logZ(Hd, beta)) < 1e-8
            @test abs(real(thermal_expect(traj.final, prob.K)) -
                      real(exact_thermal_expect(Hd, Hd, beta))) < 1e-7
        end
    end

    @testset "P2: aux_evolution != :none rejected" begin
        S = spin_ops()
        topo = mps_topology(1)
        phys = Dict(:site1 => S.P)
        H = OpSum() + Term(0.5, SiteOp(:site1, :Z, S.Z))
        prob = purification_problem(H, topo, phys; hermitian=true)
        @test_throws ArgumentError thermalize(Purified(aux_evolution=:backward),
            prob, 1.0; evolver=TDVP2(; QUIET...), nsteps=10)
    end
end
