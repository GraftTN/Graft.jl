# Focused tests for the production Residual Driven Expansion path.
using Test
using Random: Xoshiro
using LinearAlgebra: norm, rank
using Graft
using GraftTestUtils:
    categorical_coordinates, product_ttns, random_ttns, to_dense
using Graft.Backend: FermionParity, oneunit, ℂ, dim, domain
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

function _rde_parity_schmidt_ranks(
        coordinates::AbstractVector{<:Number}, nsites::Int)
    return map(1:(nsites - 1)) do cut
        coefficients = reshape(
            coordinates, 2^cut, 2^(nsites - cut))
        left_even = [
            index for index in axes(coefficients, 1)
            if iseven(count_ones(index - 1))
        ]
        left_odd = [
            index for index in axes(coefficients, 1)
            if isodd(count_ones(index - 1))
        ]
        right_even = [
            index for index in axes(coefficients, 2)
            if iseven(count_ones(index - 1))
        ]
        right_odd = [
            index for index in axes(coefficients, 2)
            if isodd(count_ones(index - 1))
        ]
        return (
            even=rank(
                coefficients[left_even, right_odd];
                atol=1e-12, rtol=0),
            odd=rank(
                coefficients[left_odd, right_even];
                atol=1e-12, rtol=0),
            total=rank(coefficients; atol=1e-12, rtol=0),
        )
    end
end

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

@testset "graded matching expansion advances in a two-round wavefront" begin
    fermion = fermion_ops_z2()
    topo = mps_topology(4)
    phys = Dict(
        nodeid(topo, n) => fermion.P for n in 1:nnodes(topo))
    product(bits) = product_ttns(
        ComplexF64,
        topo,
        phys,
        Dict(
            Symbol(:site, site) => FermionParity(bit)
            for (site, bit) in enumerate(bits)
        ),
    )
    ψ = product((1, 0, 0, 0))
    edge_components = (
        product((0, 1, 0, 0)),
        product((1, 1, 1, 0)),
        product((1, 0, 1, 1)),
        product((0, 1, 1, 1)),
    )
    rhs = exact_linear_combination(
        [ψ, edge_components...],
        ComplexF64[1, 0.05, -0.04, 0.03, 0.0015],
    )
    rhs_schmidt_ranks = _rde_parity_schmidt_ranks(
        categorical_coordinates(rhs), 4)
    @test rhs_schmidt_ranks == fill(
        (even=1, odd=1, total=2), 3)
    @test domain(ψ.tensors[topo.root])[1] ==
        domain(rhs.tensors[topo.root])[1]
    @test domain(ψ.tensors[topo.root])[1] !=
        oneunit(typeof(fermion.P))
    hamiltonian = OpSum() +
        Term(0.3, SiteOp(:site1, :N, fermion.N))
    operator = ttno_from_opsum(
        hamiltonian, topo, phys; hermitian=true)
    residual, _ = _RDE.linear_residual(
        ψ, operator, rhs; a0=1, a1=0)
    policy = _RDE.ResidualDrivenExpansion(
        trunc=TruncationScheme(maxdim=2),
        max_add=1,
        max_total_add=3,
        max_edges=3,
        max_rounds=2,
        weight_atol=1e-13,
        weight_rtol=0,
        enrichment_atol=1e-13,
        enrichment_rtol=0,
    )

    before = categorical_coordinates(ψ)
    residual_before = norm(
        categorical_coordinates(rhs) - before)
    _, first_expansion = _RDE.residual_expand!(ψ, residual, policy)
    first_grown_edges = Set(
        edge.edge.first
        for edge in first_expansion.edges
        if edge.added_rank > 0
    )

    @test first_expansion.stop_reason == :expanded
    @test first_expansion.selected_edges == 2
    @test first_expansion.selected_edges <= policy.max_edges
    @test first_expansion.total_added == 2
    @test first_expansion.remaining_add == 1
    @test first_grown_edges == Set((:site1, :site3))
    @test all(
        edge.added_rank <= policy.max_add
        for edge in first_expansion.edges
    )
    @test all(
        edge.rank_after <= policy.trunc.maxdim
        for edge in first_expansion.edges
    )
    @test sum(edge.added_rank for edge in first_expansion.edges) ==
        first_expansion.total_added
    @test categorical_coordinates(ψ) ≈ before atol=2e-12
    @test first_expansion.embedding_error <= 1e-12
    @test center(ψ) == topo.root
    @test check_arrows(ψ)

    _, first_solve = _RDE.linsolve!(
        ψ,
        operator,
        rhs;
        a0=1,
        a1=0,
        krylovdim=6,
        maxiter=4,
        tol=1e-12,
        fit_nsweeps=2,
        fit_tol=0.0,
        _root_first=true,
    )
    first_difference =
        categorical_coordinates(rhs) - categorical_coordinates(ψ)
    expected_middle =
        -0.04 * categorical_coordinates(edge_components[2])
    @test first_difference ≈ expected_middle atol=2e-10
    @test norm(first_difference) < residual_before
    @test norm(first_difference) > 1e-10
    @test first_solve.normres ≈ norm(first_difference) atol=2e-10
    @test center(ψ) == topo.root
    @test check_arrows(ψ)

    middle_residual, _ = _RDE.linear_residual(
        ψ, operator, rhs; a0=1, a1=0)
    before_middle_expansion = categorical_coordinates(ψ)
    _, second_expansion = _RDE.residual_expand!(
        ψ,
        middle_residual,
        policy;
        remaining_add=first_expansion.remaining_add,
    )
    second_grown_edges = Set(
        edge.edge.first
        for edge in second_expansion.edges
        if edge.added_rank > 0
    )

    @test second_expansion.stop_reason == :expanded
    @test second_expansion.selected_edges == 1
    @test second_expansion.selected_edges <= policy.max_edges
    @test second_expansion.total_added == 1
    @test second_expansion.remaining_add == 0
    @test second_grown_edges == Set((:site2,))
    @test categorical_coordinates(ψ) ≈
        before_middle_expansion atol=2e-12
    @test second_expansion.embedding_error <= 1e-12
    @test center(ψ) == topo.root
    @test check_arrows(ψ)

    _, second_solve = _RDE.linsolve!(
        ψ,
        operator,
        rhs;
        a0=1,
        a1=0,
        krylovdim=6,
        maxiter=4,
        tol=1e-12,
        fit_nsweeps=2,
        fit_tol=0.0,
        _root_first=true,
    )
    final_difference =
        categorical_coordinates(rhs) - categorical_coordinates(ψ)
    final_schmidt_ranks = _rde_parity_schmidt_ranks(
        categorical_coordinates(ψ), 4)
    @test norm(final_difference) <= 1e-10
    @test second_solve.normres <= 1e-10
    @test final_schmidt_ranks == fill(
        (even=1, odd=1, total=2), 3)
    @test center(ψ) == topo.root
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
