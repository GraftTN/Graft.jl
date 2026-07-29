using Test
using Graft
using Graft.TestUtils
using Graft.Backend: FermionParity, Vect, domain, dual, isdual, oneunit, ℂ,
    ⊗, ←
using LinearAlgebra: norm
using Random: Xoshiro

isdefined(@__MODULE__, Symbol("@graft_testset")) || include("test_harness.jl")

function _residual_orientation_state(
        rng, topo, phys, bond, dual_edges;
        rootspace=oneunit(typeof(bond)))
    S = typeof(bond)
    edge_space(n) = n in dual_edges ? dual(bond) : bond
    tensors = map(1:nnodes(topo)) do n
        codomain_spaces = S[edge_space(child) for child in topo.children[n]]
        haskey(phys, nodeid(topo, n)) &&
            push!(codomain_spaces, phys[nodeid(topo, n)])
        parent_space =
            topo.parent[n] == 0 ? rootspace : edge_space(n)
        randn(
            rng, ComplexF64,
            reduce(⊗, codomain_spaces) ← parent_space)
    end
    return TTNS(topo, tensors, topo.root)
end

function _residual_orientation_signature(state)
    topo = topology(state)
    return [
        isdual(virtualspace(state, child))
        for child in 1:nnodes(topo) if topo.parent[child] != 0
    ]
end

@graft_testset "exact sums align root-centered virtual orientations" begin
    topo = mps_topology(3)
    phys = Dict(
        nodeid(topo, n) => spin_ops().P for n in 1:nnodes(topo))
    edges = [
        child for child in 1:nnodes(topo)
        if topo.parent[child] != 0
    ]
    primal = _residual_orientation_state(
        Xoshiro(2026073001), topo, phys, ℂ^2, Int[])
    mixed = _residual_orientation_state(
        Xoshiro(2026073002), topo, phys, ℂ^2, [last(edges)])
    @test center(primal) == center(mixed) == topo.root
    @test _residual_orientation_signature(primal) !=
        _residual_orientation_signature(mixed)
    @test check_arrows(primal)
    @test check_arrows(mixed)

    sources_before = to_dense.((primal, mixed))
    coefficients = ComplexF64[0.37 - 0.11im, -0.29 + 0.41im]
    result = exact_linear_combination([primal, mixed], coefficients)
    reference =
        coefficients[1] * sources_before[1] +
        coefficients[2] * sources_before[2]
    @test norm(to_dense(result) - reference) < 1e-12
    @test to_dense(primal) == sources_before[1]
    @test to_dense(mixed) == sources_before[2]
    @test check_arrows(result)

    fermion = fermion_ops_z2()
    graded_topo = mps_topology(2)
    graded_phys = Dict(
        nodeid(graded_topo, n) => fermion.P
        for n in 1:nnodes(graded_topo))
    graded_bond = Vect[FermionParity](
        FermionParity(0) => 2, FermionParity(1) => 2)
    graded_edges = [
        child for child in 1:nnodes(graded_topo)
        if graded_topo.parent[child] != 0
    ]
    graded_primal = _residual_orientation_state(
        Xoshiro(2026073003), graded_topo, graded_phys, graded_bond,
        Int[])
    graded_dual = _residual_orientation_state(
        Xoshiro(2026073004), graded_topo, graded_phys, graded_bond,
        graded_edges)
    graded_before =
        categorical_coordinates.((graded_primal, graded_dual))
    graded_coefficients = ComplexF64[0.23 + 0.17im, -0.31im]
    graded_result = exact_linear_combination(
        [graded_primal, graded_dual], graded_coefficients)
    graded_reference =
        graded_coefficients[1] * graded_before[1] +
        graded_coefficients[2] * graded_before[2]
    @test center(graded_primal) == center(graded_dual) ==
        graded_topo.root
    @test _residual_orientation_signature(graded_primal) !=
        _residual_orientation_signature(graded_dual)
    @test norm(
        categorical_coordinates(graded_result) - graded_reference) < 1e-12
    @test categorical_coordinates(graded_primal) == graded_before[1]
    @test categorical_coordinates(graded_dual) == graded_before[2]
    @test check_arrows(graded_result)
end

@graft_testset "charged purification exact sum and physical residual" begin
    fermion = fermion_ops_z2()
    topo = mps_topology(2)
    phys = Dict(:site1 => fermion.P, :site2 => fermion.P)
    hamiltonian = OpSum() +
        Term(-0.5, SiteOp(:site1, :N, fermion.N)) +
        Term(-1.0, SiteOp(:site1, :Cd, fermion.Cd),
             SiteOp(:site2, :C, fermion.C)) +
        Term(-1.0, SiteOp(:site1, :C, fermion.C),
             SiteOp(:site2, :Cd, fermion.Cd))
    problem =
        purification_problem(hamiltonian, topo, phys; hermitian=true)
    neutral = random_ttns(
        Xoshiro(2026073005), ComplexF64, problem.topo_doubled,
        problem.phys_doubled, fermion.P)
    charged_reference = apply_local(neutral, fermion.Cd, :site2)
    root = topology(charged_reference).root
    rootspace = domain(charged_reference.tensors[root])[1]
    charged_bond = Vect[FermionParity](
        FermionParity(0) => 2, FermionParity(1) => 2)
    doubled_edges = [
        child for child in 1:nnodes(problem.topo_doubled)
        if problem.topo_doubled.parent[child] != 0
    ]
    charged = _residual_orientation_state(
        Xoshiro(2026073006), problem.topo_doubled,
        problem.phys_doubled, charged_bond, [last(doubled_edges)];
        rootspace)
    acted = apply(problem.K, charged; center=root)
    @test center(charged) == center(acted) == root
    @test _residual_orientation_signature(charged) !=
        _residual_orientation_signature(acted)
    @test domain(charged.tensors[root])[1] ==
        domain(acted.tensors[root])[1]

    charged_coordinates = categorical_coordinates(charged)
    acted_coordinates = categorical_coordinates(acted)
    h = 0.01
    rhs = exact_linear_combination(
        [charged, acted], ComplexF64[1, -h / 2])
    @test norm(
        categorical_coordinates(rhs) -
        (charged_coordinates - h * acted_coordinates / 2)) < 2e-12

    a0 = 0.91
    a1 = -0.17
    exact_rhs = exact_linear_combination(
        [charged, acted], ComplexF64[a0, a1])
    residual = exact_linear_combination(
        [exact_rhs, charged, acted], ComplexF64[1, -a0, -a1])
    @test norm(categorical_coordinates(residual)) < 5e-12
    @test Graft.Evolution._linear_physical_residual(
        charged, problem.K, exact_rhs, a0, a1) < 5e-12
    @test categorical_coordinates(charged) == charged_coordinates
    @test check_arrows(rhs)
    @test check_arrows(residual)
end
