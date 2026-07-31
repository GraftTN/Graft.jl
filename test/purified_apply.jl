using Test
using LinearAlgebra: dot, norm
using Random: Xoshiro
using Graft
using Graft.Backend
using GraftTestUtils
using Graft.Contractions: eff_h0

isdefined(@__MODULE__, Symbol("@graft_testset")) || include("test_harness.jl")

@graft_testset "apply/to_dense on dual physical legs (purification)" begin
    F = fermion_ops_z2()

    # --- 2-node minimal: identity axiom ---
    H1 = OpSum()
    H1 += Term(-0.5, SiteOp(:site1, :N, F.N))
    p1 = purification_problem(
        H1, mps_topology(1), Dict(:site1 => F.P); hermitian=true)
    ψ1 = random_ttns(
        Xoshiro(7), ComplexF64, p1.topo_doubled, p1.phys_doubled,
        Dict{Symbol,Any}(:site1_thermal => F.P))
    idP = id(ComplexF64, F.P)
    Hid = OpSum()
    Hid += Term(1.0, SiteOp(:site1, :Id, idP))
    Id1 = ttno_from_opsum(
        Hid, p1.topo_doubled, p1.phys_doubled; hermitian=true)
    @test inner(ψ1, apply(Id1, ψ1; optimize=false)) ≈ inner(ψ1, ψ1) atol=1e-12
    @test dot(
        categorical_coordinates(ψ1),
        to_dense(p1.K) * categorical_coordinates(ψ1)) ≈
        expect(ψ1, p1.K) atol=1e-12

    # --- doubled 2-site with hopping: oracle == JW/kron matrix ---
    H = OpSum()
    H += Term(-0.5, SiteOp(:site1, :N, F.N))
    H += Term(-1.0, SiteOp(:site1, :Cd, F.Cd), SiteOp(:site2, :C, F.C))
    H += Term(-1.0, SiteOp(:site1, :C, F.C), SiteOp(:site2, :Cd, F.Cd))
    prob = purification_problem(
        H, mps_topology(2), Dict(:site1 => F.P, :site2 => F.P); hermitian=true)
    td = prob.topo_doubled
    phys_primal = Dict{Symbol,valtype(prob.phys_doubled)}(
        s => F.P for s in keys(prob.phys_doubled))
    @test maximum(abs.(
        to_dense(prob.K) - dense_hamiltonian(H, td, phys_primal))) < 1e-12

    bond = Dict{Symbol,Any}()
    for n in 1:nnodes(td)
        td.parent[n] == 0 && continue
        idn = nodeid(td, n)
        bond[idn] = endswith(String(idn), "_thermal") ? F.P :
            Vect[FermionParity](FermionParity(0) => 2, FermionParity(1) => 2)
    end
    ψ = move_center!(
        random_ttns(Xoshiro(20260728), ComplexF64, td, prob.phys_doubled, bond),
        td.root)
    c = categorical_coordinates(ψ)
    @test dot(c, to_dense(prob.K) * c) ≈ expect(ψ, prob.K) atol=1e-12

    # --- TDVP1 down-split h0 vs dense reference (the original phenomenon) ---
    denseK = to_dense(prob.K)
    for v in td.children[td.root]
        s = copy(ψ)
        C = Graft.Evolution._split_link_down(
            TDVP1(), s, prob.K, td.root, v, -0.01)
        h0 = eff_h0(EnvCache(td), s, prob.K, v, td.root;
                    optimize=false, sector_aware=false)
        tensors = copy(s.tensors)
        tensors[v] = s.tensors[v] * Graft.Networks.pivotal_link(C)
        embedded = TTNS(td, tensors, v)
        ce = categorical_coordinates(embedded)
        @test dot(C, h0(C)) ≈ dot(ce, denseK * ce) atol=1e-12
    end

    # --- full-rank TDVP1 propagation on the doubled state vs exact ---
    dz = -0.025
    nsteps = 8
    exact = exact_evolve(denseK, c, nsteps * dz)
    for order in (1, 2)
        state = copy(ψ)
        evolver = TDVP1(order=order, verbose=false)
        for _ in 1:nsteps
            step!(evolver, state, prob.K, dz)
            Graft.normalize!(state)
        end
        evolved = categorical_coordinates(state)
        infidelity =
            1 - abs(dot(evolved, exact)) / (norm(evolved) * norm(exact))
        @test infidelity < (order == 1 ? 2e-5 : 1e-9)
    end
end
