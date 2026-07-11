using Test
using GRAFT
using GRAFT.TestUtils
using GRAFT.Backend
using LinearAlgebra: norm, tr
using Random

const RNG = Xoshiro(20260712)

@testset "P0 thermal conventions" begin
    @testset "single-site trivial beta=0 trace" begin
        S = spin_ops()
        topo = mps_topology(1)
        phys = Dict(:site1 => S.P)
        H = OpSum() + Term(1.0, SiteOp(:site1, :Z, S.Z))
        prob = purification_problem(H, topo, phys; hermitian=true)
        state0 = infinite_temperature_state(prob)
        @test norm(state0.psi) ≈ 1 atol = 1e-12
        @test state0.logZ ≈ log(2) atol = 1e-12
        @test real(thermal_expect(state0,
            physical_ttno(prob, OpSum() + Term(1.0, SiteOp(:site1, :Z, S.Z))))) ≈ 0 atol = 1e-12
        @test real(thermal_expect(state0,
            physical_ttno(prob, OpSum() + Term(1.0, SiteOp(:site1, :N, S.N))))) ≈ 0.5 atol = 1e-12
    end

    @testset "fermion trace vs supertrace" begin
        F = fermion_ops_z2()
        topo = mps_topology(1)
        phys = Dict(:site1 => F.P)
        H = OpSum() + Term(0.0, SiteOp(:site1, :I, F.I))
        prob = purification_problem(H, topo, phys; hermitian=true)
        state0 = infinite_temperature_state(prob)
        @test norm(state0.psi) ≈ 1 atol = 1e-12
        @test state0.logZ ≈ log(2) atol = 1e-12
        val = real(thermal_expect(state0,
            physical_ttno(prob, OpSum() + Term(1.0, SiteOp(:site1, :N, F.N)))))
        @test val ≈ 0.5 atol = 1e-12
    end

    @testset "two-site trivial beta=0" begin
        S = spin_ops()
        topo = mps_topology(2)
        phys = Dict(nodeid(topo, i) => S.P for i in 1:2)
        H = OpSum()
        for (c, p) in GRAFT.Trees.edges(topo)
            H += Term(-1.0, SiteOp(nodeid(topo, c), :Z, S.Z),
                       SiteOp(nodeid(topo, p), :Z, S.Z))
        end
        prob = purification_problem(H, topo, phys; hermitian=true)
        state0 = infinite_temperature_state(prob)
        @test norm(state0.psi) ≈ 1 atol = 1e-12
        @test state0.logZ ≈ log(4) atol = 1e-12
        @test real(thermal_expect(state0,
            physical_ttno(prob, OpSum() + Term(1.0, SiteOp(:site1, :Z, S.Z))))) ≈ 0 atol = 1e-12
        @test real(thermal_expect(state0,
            physical_ttno(prob, OpSum() + Term(1.0, SiteOp(:site1, :N, S.N))))) ≈ 0.5 atol = 1e-12
        for n in 1:nnodes(state0.psi.topo)
            p = state0.psi.topo.parent[n]
            p == 0 && continue
            child_sym = nodeid(state0.psi.topo, n)
            if !haskey(prob.physical_of, child_sym) &&
               child_sym ∉ values(prob.pp_ancilla_of)
                @test dim(domain(state0.psi.tensors[n])[1]) == 1
            end
        end
    end

    @testset "U1 spin beta=0" begin
        U = spin_ops_u1()
        topo = mps_topology(1)
        phys = Dict(:site1 => U.P)
        H = OpSum() + Term(1.0, SiteOp(:site1, :Z, U.Z))
        prob = purification_problem(H, topo, phys; hermitian=true)
        state0 = infinite_temperature_state(prob)
        @test norm(state0.psi) ≈ 1 atol = 1e-12
        @test state0.logZ ≈ log(2) atol = 1e-12
        @test real(thermal_expect(state0,
            physical_ttno(prob, OpSum() + Term(1.0, SiteOp(:site1, :N, U.N))))) ≈ 0.5 atol = 1e-12
    end

    @testset "ancilla name collision" begin
        S = spin_ops()
        topo = TreeTopology(:site1, [:site1 => :site1_thermal])
        phys = Dict(:site1 => S.P, :site1_thermal => S.P)
        H = OpSum() + Term(1.0, SiteOp(:site1, :Z, S.Z))
        @test_throws ArgumentError purification_problem(H, topo, phys; hermitian=true)
    end

    @testset "dense thermal references" begin
        S = spin_ops()
        topo = mps_topology(1)
        phys = Dict(:site1 => S.P)
        H = OpSum() + Term(1.0, SiteOp(:site1, :Z, S.Z))
        Hd = dense_hamiltonian(H, topo, phys)
        Z0 = exact_thermal_Z(Hd, 0.0)
        @test real(Z0) ≈ 2.0 atol = 1e-12
        @test exact_thermal_logZ(Hd, 0.0) ≈ log(2) atol = 1e-12
        @test real(exact_thermal_expect(Hd, dense_hamiltonian(
            OpSum() + Term(1.0, SiteOp(:site1, :N, S.N)), topo, phys), 0.0)) ≈ 0.5 atol = 1e-12
        beta = 1.0
        Zb = exact_thermal_Z(Hd, beta)
        Zref = tr(exp(-beta * Hd))
        @test real(Zb) ≈ real(Zref) atol = 1e-10
        @test exact_thermal_logZ(Hd, beta) ≈ log(real(Zref)) atol = 1e-10
    end
end
