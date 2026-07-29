# Focused tests for the production Residual Driven Expansion path.
using Test
using Random: Xoshiro
using LinearAlgebra: norm
using Graft
using Graft.TestUtils: product_ttns, random_ttns, to_dense
using Graft.Backend: ℂ, dim, domain
using Graft.Trees: edges

const _RDE_TEST_RNG = Xoshiro(0x20260730)
const _RDE = Graft.Evolution

function _rde_test_operator(topo, phys)
    spin = spin_ops()
    hamiltonian = OpSum()
    for n in 1:nnodes(topo)
        site = nodeid(topo, n)
        haskey(phys, site) || continue
        hamiltonian += Term(
            0.17 + 0.03n,
            SiteOp(site, :X, spin.X),
        )
    end
    return ttno_from_opsum(hamiltonian, topo, phys; hermitian=true)
end

_rde_bond_ranks(ψ) = [
    dim(domain(ψ.tensors[child])[1])
    for (child, _) in edges(topology(ψ))
]

@testset "residual-driven policy and exact residual" begin
    @test_throws ArgumentError _RDE.ResidualDrivenExpansion(max_add=-1)
    @test_throws ArgumentError _RDE.ResidualDrivenExpansion(max_total_add=-1)
    @test_throws ArgumentError _RDE.ResidualDrivenExpansion(max_edges=-1)
    @test_throws ArgumentError _RDE.ResidualDrivenExpansion(max_rounds=-1)
    @test_throws ArgumentError _RDE.ResidualDrivenExpansion(
        schedule=:unsupported,
    )
    @test_throws ArgumentError _RDE.ResidualDrivenExpansion(weight_atol=Inf)
    @test_throws ArgumentError _RDE.ResidualDrivenExpansion(enrichment_rtol=NaN)
    @test_throws ArgumentError _RDE.ResidualDrivenExpansion(
        compression_atol=Inf,
    )
    @test_throws ArgumentError _RDE.ResidualDrivenExpansion(
        trunc=TruncationScheme(maxdim=0),
    )
    @test_throws ArgumentError _RDE.ResidualDrivenExpansion(
        residual_trunc=TruncationScheme(maxdim=0),
    )

    rank_cap_topology = mps_topology(2)
    rank_cap_physical = Dict(
        nodeid(rank_cap_topology, n) => spin_ops().P
        for n in 1:nnodes(rank_cap_topology)
    )
    rank_cap_state = random_ttns(
        Xoshiro(0x7a11ca9), ComplexF64, rank_cap_topology,
        rank_cap_physical, ℂ^2)
    rank_cap_before = to_dense(rank_cap_state)
    @test_throws ArgumentError _RDE._rde_check_state_rank_cap(
        rank_cap_state,
        _RDE.ResidualDrivenExpansion(
            trunc=TruncationScheme(maxdim=1)),
    )
    @test to_dense(rank_cap_state) == rank_cap_before

    topo = mps_topology(2)
    spin = spin_ops()
    phys = Dict(nodeid(topo, n) => spin.P for n in 1:nnodes(topo))
    operator = _rde_test_operator(topo, phys)
    rhs = random_ttns(_RDE_TEST_RNG, ComplexF64, topo, phys, ℂ^2)
    ψ = random_ttns(_RDE_TEST_RNG, ComplexF64, topo, phys, ℂ^1)
    residual, report = _RDE.linear_residual(
        ψ,
        operator,
        rhs;
        a0=1,
        a1=0,
    )
    @test report.compression_error == 0.0
    @test report.normres ≈ norm(to_dense(rhs) - to_dense(ψ)) atol=1e-12
    @test length(report.edge_ranks) == 1
    @test only(report.edge_ranks).second ==
        only(_rde_bond_ranks(residual))
end

@testset "residual-driven zero weight and chain enrichment" begin
    topo = mps_topology(2)
    spin = spin_ops()
    phys = Dict(nodeid(topo, n) => spin.P for n in 1:nnodes(topo))

    ψzero = random_ttns(_RDE_TEST_RNG, ComplexF64, topo, phys, ℂ^1)
    zero_before = to_dense(ψzero)
    zero_policy = _RDE.ResidualDrivenExpansion(
        trunc=TruncationScheme(maxdim=2),
        max_add=1,
        max_total_add=1,
        max_edges=1,
        max_rounds=1,
        weight_atol=1e-10,
        weight_rtol=0,
        enrichment_atol=1e-12,
        enrichment_rtol=0,
    )
    _, zero_report = _RDE.residual_expand!(
        ψzero,
        copy(ψzero),
        zero_policy,
    )
    @test zero_report.total_added == 0
    @test all(!edge.selected for edge in zero_report.edges)
    @test maximum(
        (edge.uncovered_weight for edge in zero_report.edges);
        init=0.0,
    ) <= zero_policy.weight_atol
    @test to_dense(ψzero) == zero_before

    ψ = random_ttns(_RDE_TEST_RNG, ComplexF64, topo, phys, ℂ^1)
    residual = random_ttns(
        _RDE_TEST_RNG,
        ComplexF64,
        topo,
        phys,
        ℂ^2,
    )
    before = to_dense(ψ)
    policy = _RDE.ResidualDrivenExpansion(
        trunc=TruncationScheme(maxdim=2),
        max_add=1,
        max_total_add=1,
        max_edges=1,
        max_rounds=1,
        weight_atol=1e-13,
        weight_rtol=0,
        enrichment_atol=1e-13,
        enrichment_rtol=0,
    )
    _, report = _RDE.residual_expand!(ψ, residual, policy)
    @test report.stop_reason == :expanded
    @test report.selected_edges == 1
    @test report.total_added == 1
    @test only(report.edges).rank_before == 1
    @test only(report.edges).rank_after == 2
    @test only(report.edges).requested_rank == 1
    @test only(report.edges).added_rank == 1
    @test report.embedding_error <= 1e-12
    @test norm(to_dense(ψ) - before) <= 1e-12
    @test check_arrows(ψ)
end

@testset "residual-driven branching selection, preservation, and caps" begin
    topo = TreeTopology(:hub, [:hub => :left, :hub => :right])
    spin = spin_ops()
    phys = Dict(:left => spin.P, :right => spin.P)
    ψ = random_ttns(_RDE_TEST_RNG, ComplexF64, topo, phys, ℂ^1)
    residual = random_ttns(
        _RDE_TEST_RNG,
        ComplexF64,
        topo,
        phys,
        ℂ^2,
    )
    before = to_dense(ψ)
    before_ranks = _rde_bond_ranks(ψ)
    policy = _RDE.ResidualDrivenExpansion(
        trunc=TruncationScheme(maxdim=3),
        max_add=1,
        max_total_add=1,
        max_edges=1,
        max_rounds=2,
        weight_atol=1e-13,
        weight_rtol=0,
        enrichment_atol=1e-13,
        enrichment_rtol=0,
    )
    _, report = _RDE.residual_expand!(ψ, residual, policy)
    after_ranks = _rde_bond_ranks(ψ)

    @test report.total_added == 1
    @test report.selected_edges == 1
    @test sum(after_ranks .- before_ranks) == 1
    @test all(edge.added_rank <= policy.max_add for edge in report.edges)
    @test report.total_added <= policy.max_total_add
    @test norm(to_dense(ψ) - before) <= 1e-12
    @test report.embedding_error <= 1e-12
    @test center(ψ) == topo.root
    @test check_arrows(ψ)
end

@testset "residual-driven adjacent edges rescore within one round" begin
    topo = mps_topology(3)
    ψ = product_ttns(
        ComplexF64,
        topo,
        Dict(
            :site1 => ComplexF64[1, 0],
            :site2 => ComplexF64[1, 0],
            :site3 => ComplexF64[1, 0],
        ),
    )
    residual = product_ttns(
        ComplexF64,
        topo,
        Dict(
            :site1 => ComplexF64[0, 1],
            :site2 => ComplexF64[0, 1],
            :site3 => ComplexF64[1, 0],
        ),
    )
    policy = _RDE.ResidualDrivenExpansion(
        trunc=TruncationScheme(maxdim=2),
        max_add=1,
        max_total_add=2,
        max_edges=2,
        max_rounds=1,
        weight_atol=1e-13,
        weight_rtol=0,
        enrichment_atol=1e-13,
        enrichment_rtol=0,
    )

    initial_candidates = _RDE._rde_score_edges!(
        copy(ψ), residual, policy)
    initial_by_child = Dict(
        nodeid(topo, candidate.child) => candidate
        for candidate in initial_candidates
    )
    @test initial_by_child[:site1].possible_add == 1
    @test initial_by_child[:site2].possible_add == 0
    @test initial_by_child[:site2].weight <= policy.weight_atol

    before = to_dense(ψ)
    before_center = center(ψ)
    _, report = _RDE.residual_expand!(ψ, residual, policy)
    edge_reports = Dict(edge.edge.first => edge for edge in report.edges)

    @test report.stop_reason == :expanded
    @test report.selected_edges == 2
    @test report.total_added == 2
    @test edge_reports[:site1].added_rank == 1
    @test edge_reports[:site2].added_rank == 1
    @test edge_reports[:site1].requested_rank == 1
    @test edge_reports[:site2].requested_rank == 1
    @test edge_reports[:site2].uncovered_weight > policy.weight_atol
    @test all(edge.added_rank <= policy.max_add for edge in report.edges)
    @test all(edge.rank_after <= policy.trunc.maxdim for edge in report.edges)
    @test norm(to_dense(ψ) - before) <= 1e-12
    @test report.embedding_error <= 1e-12
    @test center(ψ) == before_center
    @test check_arrows(ψ)
end

@testset "residual complement rank honors discarded weight" begin
    topo = mps_topology(2)
    boson = boson_ops(3)
    phys = Dict(nodeid(topo, n) => boson.P for n in 1:nnodes(topo))
    ψ = random_ttns(_RDE_TEST_RNG, ComplexF64, topo, phys, ℂ^1)
    residual = random_ttns(
        _RDE_TEST_RNG,
        ComplexF64,
        topo,
        phys,
        ℂ^4,
    )
    full = copy(ψ)
    full_policy = _RDE.ResidualDrivenExpansion(
        trunc=TruncationScheme(maxdim=4),
        max_add=3,
        max_total_add=3,
        max_edges=1,
        max_rounds=1,
        weight_atol=0,
        weight_rtol=0,
        enrichment_atol=0,
        enrichment_rtol=0,
    )
    _, full_report = _RDE.residual_expand!(
        full, residual, full_policy)

    controlled = copy(ψ)
    controlled_policy = _RDE.ResidualDrivenExpansion(
        trunc=TruncationScheme(maxdim=4, discarded_weight=0.9),
        max_add=3,
        max_total_add=3,
        max_edges=1,
        max_rounds=1,
        weight_atol=0,
        weight_rtol=0,
        enrichment_atol=0,
        enrichment_rtol=0,
    )
    _, controlled_report = _RDE.residual_expand!(
        controlled, residual, controlled_policy)

    @test full_report.total_added > 0
    @test 0 < controlled_report.total_added <= full_report.total_added
    @test norm(to_dense(full) - to_dense(ψ)) <= 1e-12
    @test norm(to_dense(controlled) - to_dense(ψ)) <= 1e-12
end

@testset "residual-driven solve loop reduction and truthful stops" begin
    topo = mps_topology(2)
    spin = spin_ops()
    phys = Dict(nodeid(topo, n) => spin.P for n in 1:nnodes(topo))
    operator = _rde_test_operator(topo, phys)
    rhs = random_ttns(_RDE_TEST_RNG, ComplexF64, topo, phys, ℂ^2)

    ψ = random_ttns(_RDE_TEST_RNG, ComplexF64, topo, phys, ℂ^1)
    policy = _RDE.ResidualDrivenExpansion(
        trunc=TruncationScheme(maxdim=2),
        max_add=1,
        max_total_add=1,
        max_edges=1,
        max_rounds=1,
        weight_atol=1e-13,
        weight_rtol=0,
        enrichment_atol=1e-13,
        enrichment_rtol=0,
    )
    _, report = _RDE.residual_driven_linsolve!(
        ψ,
        operator,
        rhs,
        policy;
        a0=1,
        a1=0,
        krylovdim=4,
        maxiter=2,
        tol=1e-10,
        fit_nsweeps=4,
        fit_tol=0.0,
    )
    @test report.converged
    @test report.stop_reason == :converged
    @test report.committed
    @test report.total_added == 1
    @test length(report.physical_residuals) == 2
    @test report.physical_residuals[end] < report.physical_residuals[1]
    @test report.physical_residuals[end] <= 1e-10
    @test norm(to_dense(ψ) - to_dense(rhs)) <= 1e-10

    disabled = random_ttns(
        _RDE_TEST_RNG,
        ComplexF64,
        topo,
        phys,
        ℂ^1,
    )
    disabled_policy = _RDE.ResidualDrivenExpansion(
        trunc=TruncationScheme(maxdim=2),
        max_add=0,
        max_total_add=1,
        max_edges=1,
        max_rounds=1,
    )
    _, disabled_report = _RDE.residual_driven_linsolve!(
        disabled,
        operator,
        rhs,
        disabled_policy;
        a0=1,
        a1=0,
        krylovdim=4,
        maxiter=2,
        tol=1e-10,
        fit_nsweeps=2,
        fit_tol=0.0,
    )
    @test !disabled_report.converged
    @test disabled_report.stop_reason == :expansion_disabled

    exhausted = random_ttns(
        _RDE_TEST_RNG,
        ComplexF64,
        topo,
        phys,
        ℂ^1,
    )
    exhausted_policy = _RDE.ResidualDrivenExpansion(
        trunc=TruncationScheme(maxdim=2),
        max_add=1,
        max_total_add=0,
        max_edges=1,
        max_rounds=1,
    )
    _, exhausted_report = _RDE.residual_driven_linsolve!(
        exhausted,
        operator,
        rhs,
        exhausted_policy;
        a0=1,
        a1=0,
        krylovdim=4,
        maxiter=2,
        tol=1e-10,
        fit_nsweeps=2,
        fit_tol=0.0,
    )
    @test !exhausted_report.converged
    @test exhausted_report.stop_reason == :global_budget_exhausted
end

@testset "typed local solver failures are uncommitted reports" begin
    topo = mps_topology(2)
    spin = spin_ops()
    phys = Dict(nodeid(topo, n) => spin.P for n in 1:nnodes(topo))
    operator = _rde_test_operator(topo, phys)
    rhs = random_ttns(_RDE_TEST_RNG, ComplexF64, topo, phys, ℂ^2)
    ψ = random_ttns(_RDE_TEST_RNG, ComplexF64, topo, phys, ℂ^1)
    before = to_dense(ψ)
    policy = _RDE.ResidualDrivenExpansion(
        trunc=TruncationScheme(maxdim=2),
        max_add=1,
        max_total_add=1,
        max_edges=1,
        max_rounds=1,
    )
    _, report = _RDE.residual_driven_linsolve!(
        ψ,
        operator,
        rhs,
        policy;
        a0=1,
        a1=0.17,
        krylovdim=2,
        maxiter=1,
        tol=1e-30,
        fit_nsweeps=1,
        fit_tol=0.0,
    )
    @test !report.converged
    @test report.stop_reason == :local_solver_failed
    @test !report.committed
    @test report.exception_message !== nothing
    @test occursin("local Krylov solve failed", report.exception_message)
    @test to_dense(ψ) == before

    nonfinite = _RDE._LocalLinearSolveFailure(
        :site1, :nonfinite_local_solution)
    @test _RDE._rde_local_failure_reason(nonfinite) ==
        :nonfinite_local_solution
end

@testset "residual compression is audited and fail-closed" begin
    topo = mps_topology(2)
    boson = boson_ops(3)
    phys = Dict(nodeid(topo, n) => boson.P for n in 1:nnodes(topo))
    hamiltonian = OpSum()
    for n in 1:nnodes(topo)
        site = nodeid(topo, n)
        hamiltonian += Term(
            0.1n,
            SiteOp(site, :N, boson.N),
        )
    end
    operator = ttno_from_opsum(
        hamiltonian, topo, phys; hermitian=true)

    exact = random_ttns(
        _RDE_TEST_RNG, ComplexF64, topo, phys, ℂ^2)
    exact_report = _RDE.LinearResidualReport(
        Float64(norm(exact)),
        0.0,
        _RDE._rde_residual_ranks(exact),
    )
    compression_policy = _RDE.ResidualDrivenExpansion(
        residual_trunc=TruncationScheme(maxdim=1),
        compression_atol=0,
        compression_rtol=0,
    )
    surrogate, compression_report = _RDE._rde_compress_residual(
        exact, exact_report, compression_policy)
    @test only(exact_report.edge_ranks).second == 2
    @test only(compression_report.edge_ranks).second == 1
    @test compression_report.normres == exact_report.normres
    @test compression_report.compression_error ≈
        norm(to_dense(exact) - to_dense(surrogate)) atol=1e-12
    @test compression_report.compression_error > 0

    rhs = random_ttns(_RDE_TEST_RNG, ComplexF64, topo, phys, ℂ^3)
    ψ = random_ttns(_RDE_TEST_RNG, ComplexF64, topo, phys, ℂ^1)
    before = to_dense(ψ)
    policy = _RDE.ResidualDrivenExpansion(
        trunc=TruncationScheme(maxdim=2),
        residual_trunc=TruncationScheme(maxdim=1),
        max_add=1,
        max_total_add=1,
        max_edges=1,
        max_rounds=1,
        compression_atol=0,
        compression_rtol=0,
    )
    _, report = _RDE.residual_driven_linsolve!(
        ψ,
        operator,
        rhs,
        policy;
        a0=1,
        a1=0,
        krylovdim=4,
        maxiter=2,
        tol=1e-10,
        fit_nsweeps=2,
        fit_tol=0.0,
    )
    @test !report.converged
    @test report.stop_reason == :compression_inaccurate
    @test !report.committed
    @test only(report.residuals).compression_error > 0
    @test all(rank.second <= 1 for rank in only(report.residuals).edge_ranks)
    @test to_dense(ψ) == before
end

@testset "residual-driven linear solve exception is transactional" begin
    topo = mps_topology(2)
    spin = spin_ops()
    phys = Dict(nodeid(topo, n) => spin.P for n in 1:nnodes(topo))
    operator = _rde_test_operator(topo, phys)
    rhs = random_ttns(_RDE_TEST_RNG, ComplexF64, topo, phys, ℂ^2)
    ψ = random_ttns(_RDE_TEST_RNG, ComplexF64, topo, phys, ℂ^1)
    before = to_dense(ψ)
    before_center = center(ψ)
    guarded_policy = _RDE.ResidualDrivenExpansion(
        trunc=TruncationScheme(maxdim=2),
        max_add=1,
        max_total_add=1,
        max_edges=1,
        max_rounds=1,
        residual_max_bond=1,
    )
    @test_throws ArgumentError _RDE.residual_driven_linsolve!(
        ψ,
        operator,
        rhs,
        guarded_policy;
        a0=1,
        a1=0,
        krylovdim=4,
        maxiter=2,
        tol=1e-10,
        fit_nsweeps=2,
        fit_tol=0.0,
    )
    @test to_dense(ψ) == before
    @test center(ψ) == before_center
end
