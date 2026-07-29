using Test
using Graft
using Graft.TestUtils
using Graft.Backend: ℂ, dim
using Graft.Trees: edges
using LinearAlgebra: I, norm
using Random: Xoshiro

isdefined(@__MODULE__, Symbol("@graft_testset")) || include("test_harness.jl")

function _two_site_linear_hamiltonian(topo; coupling=0.73, field=0.41)
    spins = spin_ops()
    hamiltonian = OpSum()
    for (child, parent) in edges(topo)
        hamiltonian += Term(
            -coupling,
            SiteOp(nodeid(topo, child), :Z, spins.Z),
            SiteOp(nodeid(topo, parent), :Z, spins.Z))
    end
    for node in 1:nnodes(topo)
        hamiltonian +=
            Term(-field, SiteOp(nodeid(topo, node), :X, spins.X))
    end
    return hamiltonian
end

_two_site_linear_phys(topo) =
    Dict(nodeid(topo, node) => spin_ops().P for node in 1:nnodes(topo))

@graft_testset "two-site linear solve: dense two-node oracle and reports" begin
    topo = mps_topology(2)
    phys = _two_site_linear_phys(topo)
    hamiltonian = _two_site_linear_hamiltonian(topo)
    operator = ttno_from_opsum(hamiltonian, topo, phys; hermitian=true)
    rhs = random_ttns(Xoshiro(0x2a5117e), ComplexF64, topo, phys, ℂ^2)
    initial = random_ttns(Xoshiro(0x51ec7ed), ComplexF64, topo, phys, ℂ^1)
    dense_h = dense_hamiltonian(hamiltonian, rhs)
    dense_rhs = to_dense(rhs)
    a1 = 0.08
    reference =
        (Matrix{ComplexF64}(I, length(dense_rhs), length(dense_rhs)) +
         a1 * dense_h) \ dense_rhs

    policy = TwoSiteLinearPolicy(
        trunc=TruncationScheme(maxdim=2),
        sweeps=3, krylovdim=8, maxiter=16,
        local_tol=1e-12, residual_tol=1e-10)
    solved = copy(initial)
    returned, report =
        two_site_linsolve!(solved, operator, rhs, policy; a0=1.0, a1)

    @test returned === solved
    @test report isa TwoSiteLinearReport
    @test report.transaction_committed
    @test report.converged
    @test report.stop_reason == :converged
    @test report.physical_residuals[end] <= policy.residual_tol
    @test norm(to_dense(solved) - reference) <= 1e-9 * norm(reference)
    @test !isempty(report.edge_reports)
    @test all(edge -> edge.solver_converged, report.edge_reports)
    @test all(edge -> edge.local_residual_before_truncation <= 1e-10,
              report.edge_reports)
    @test all(edge -> edge.retained_rank <= policy.trunc.maxdim,
              report.edge_reports)
    @test all(edge -> edge.discarded_norm >= 0 &&
                      edge.discarded_weight >= 0,
              report.edge_reports)

    @test_throws ArgumentError TwoSiteLinearPolicy(sweeps=0)
    @test_throws ArgumentError TwoSiteLinearPolicy(
        trunc=TruncationScheme(maxdim=0))
    @test_throws ArgumentError TwoSiteLinearPolicy(
        trunc=TruncationScheme(rtol=1.1))
end

@graft_testset "two-site linear solve: truncation cap and classifier matrix" begin
    topo = mps_topology(2)
    phys = _two_site_linear_phys(topo)
    hamiltonian = _two_site_linear_hamiltonian(topo; coupling=1.0, field=0.63)
    operator = ttno_from_opsum(hamiltonian, topo, phys; hermitian=true)
    rhs = random_ttns(Xoshiro(0xc4a551f1), ComplexF64, topo, phys, ℂ^2)
    initial = random_ttns(Xoshiro(0xbadca9), ComplexF64, topo, phys, ℂ^1)
    before = to_dense(initial)
    capped_policy = TwoSiteLinearPolicy(
        trunc=TruncationScheme(maxdim=1),
        sweeps=2, krylovdim=8, maxiter=16,
        local_tol=1e-12, residual_tol=1e-12)
    capped = copy(initial)
    _, capped_report = two_site_linsolve!(
        capped, operator, rhs, capped_policy; a0=1.0, a1=0.19)

    @test capped_report.transaction_committed
    @test !capped_report.converged
    @test capped_report.stop_reason == :rank_cap_exhausted
    @test capped_report.physical_residuals[end] > capped_policy.residual_tol
    @test any(edge -> edge.discarded_norm > 0, capped_report.edge_reports)
    @test any(edge -> edge.discarded_weight > 0,
              capped_report.edge_reports)
    @test any(edge ->
                  edge.local_residual_after_truncation >
                  edge.local_residual_before_truncation,
              capped_report.edge_reports)
    @test norm(to_dense(capped) - before) > 1e-10

    exact_policy = TwoSiteLinearPolicy(
        trunc=TruncationScheme(maxdim=2),
        sweeps=2, krylovdim=8, maxiter=16,
        local_tol=1e-12, residual_tol=1e-10)
    exact = copy(initial)
    _, exact_report = two_site_linsolve!(
        exact, operator, rhs, exact_policy; a0=1.0, a1=0.19)
    @test exact_report.converged

    residual_driven_stalled = (; physical_residuals=[1e-3])
    residual_driven_converged = (; physical_residuals=[1e-12])
    @test classify_linear_pair(
        residual_driven_stalled, exact_report;
        tolerance=1e-10).classification ==
        :two_site_converged_residual_driven_stalled
    @test classify_linear_pair(
        residual_driven_stalled, capped_report;
        tolerance=1e-10).classification ==
        :both_stalled
    @test classify_linear_pair(
        residual_driven_converged, exact_report;
        tolerance=1e-10).classification ==
        :matched
    @test classify_linear_pair(
        residual_driven_converged, capped_report;
        tolerance=1e-10).classification ==
        :inconsistent

    classification = classify_linear_pair(
        residual_driven_stalled, exact_report; tolerance=1e-10)
    @test !classification.residual_driven_converged
    @test classification.two_site_converged
    @test classification.residual_driven_residual == 1e-3
    @test classification.two_site_residual ==
          exact_report.physical_residuals[end]

    failed_residual_driven = (;
        physical_residuals=Float64[],
        committed=false,
    )
    inconclusive = classify_linear_pair(
        failed_residual_driven, exact_report; tolerance=1e-10)
    @test inconclusive.classification == :inconclusive_failure
    @test !inconclusive.residual_driven_converged
    @test inconclusive.residual_driven_residual == Inf
end

@graft_testset "two-site linear solve: genuine branching dense oracle" begin
    topo = star_topology(2, 1)
    phys = _two_site_linear_phys(topo)
    hamiltonian =
        _two_site_linear_hamiltonian(topo; coupling=0.37, field=0.29)
    operator = ttno_from_opsum(hamiltonian, topo, phys; hermitian=true)
    rhs = random_ttns(Xoshiro(0x719b12), ComplexF64, topo, phys, ℂ^2)
    solved = random_ttns(Xoshiro(0x331fd4), ComplexF64, topo, phys, ℂ^1)
    dense_h = dense_hamiltonian(hamiltonian, rhs)
    dense_rhs = to_dense(rhs)
    a1 = 0.035
    reference =
        (Matrix{ComplexF64}(I, length(dense_rhs), length(dense_rhs)) +
         a1 * dense_h) \ dense_rhs
    policy = TwoSiteLinearPolicy(
        trunc=TruncationScheme(maxdim=2),
        sweeps=8, krylovdim=16, maxiter=24,
        local_tol=1e-12, residual_tol=1e-9)

    _, report =
        two_site_linsolve!(solved, operator, rhs, policy; a0=1.0, a1)

    @test report.transaction_committed
    @test report.converged
    @test report.physical_residuals[end] <= policy.residual_tol
    @test norm(to_dense(solved) - reference) <= 2e-8 * norm(reference)
    @test check_arrows(solved)
    @test topology(solved) == topo
    @test Set((edge.child, edge.parent) for edge in report.edge_reports) ==
          Set((child, topo.parent[child])
              for child in 1:nnodes(topo) if topo.parent[child] != 0)
end

@graft_testset "two-site linear solve: physless binary-root bootstrap" begin
    topo = binary_topology(2; prefix=:linear2s)
    sites = [nodeid(topo, node) for node in leaves(topo)]
    phys = Dict(site => spin_ops().P for site in sites)
    hamiltonian =
        OpSum() + Term(0.31, SiteOp(first(sites), :X, spin_ops().X))
    operator = ttno_from_opsum(hamiltonian, topo, phys; hermitian=true)
    rhs = random_ttns(
        Xoshiro(0xb00757a9), ComplexF64, topo, phys, ℂ^4)
    solved = random_ttns(
        Xoshiro(0x57a7e0ff), ComplexF64, topo, phys, ℂ^2)
    policy = TwoSiteLinearPolicy(
        trunc=TruncationScheme(maxdim=4),
        sweeps=2,
        krylovdim=8,
        maxiter=8,
        local_tol=1e-12,
        residual_tol=1e-10,
    )

    _, report = two_site_linsolve!(
        solved, operator, rhs, policy; a0=1.0, a1=0.0)

    root_children = topo.children[topo.root]
    @test report.transaction_committed
    @test report.converged
    @test report.physical_residuals[end] <= policy.residual_tol
    @test [dim(virtualspace(solved, child))
           for child in root_children] == [4, 4]
    @test all(
        child -> any(
            edge -> edge.child == child && edge.retained_rank == 4,
            report.edge_reports,
        ),
        root_children,
    )
    @test norm(to_dense(solved) - to_dense(rhs)) <= 1e-9 * norm(to_dense(rhs))
    @test check_arrows(solved)
end

@graft_testset "paired linear diagnostic: matched budgets and immutability" begin
    topo = mps_topology(2)
    phys = _two_site_linear_phys(topo)
    hamiltonian =
        _two_site_linear_hamiltonian(topo; coupling=0.52, field=0.27)
    operator = ttno_from_opsum(hamiltonian, topo, phys; hermitian=true)
    rhs = random_ttns(
        Xoshiro(0xd1a6a057), ComplexF64, topo, phys, ℂ^2)
    initial = random_ttns(
        Xoshiro(0x1d3a71ca), ComplexF64, topo, phys, ℂ^1)
    before = to_dense(initial)
    before_center = center(initial)
    residual_policy = ResidualDrivenExpansion(
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
    two_site_policy = TwoSiteLinearPolicy(
        trunc=TruncationScheme(maxdim=2),
        sweeps=8,
        krylovdim=4,
        maxiter=2,
        local_tol=1e-10,
        residual_tol=1e-10,
    )

    diagnostic = paired_linear_diagnostic(
        initial,
        operator,
        rhs,
        residual_policy,
        two_site_policy;
        a0=1.0,
        a1=0.0,
        krylovdim=4,
        maxiter=2,
        tol=1e-10,
        fit_nsweeps=4,
        fit_tol=0.0,
    )

    @test diagnostic isa PairedLinearDiagnostic
    @test diagnostic.residual_driven_report isa ResidualDrivenReport
    @test diagnostic.two_site_report isa TwoSiteLinearReport
    @test diagnostic.classification isa PairedLinearClassification
    @test diagnostic.edge_subspace_evidence isa
          Vector{PairedEdgeSubspaceEvidence}
    @test diagnostic.classification.residual_driven_residual ==
          diagnostic.residual_driven_report.physical_residuals[end]
    @test diagnostic.classification.two_site_residual ==
          diagnostic.two_site_report.physical_residuals[end]
    subspace = only(diagnostic.edge_subspace_evidence)
    @test subspace.available
    @test subspace.stop_reason == :available
    @test subspace.initial_rank == 1
    @test subspace.residual_driven_rank == 2
    @test subspace.two_site_rank == 2
    @test subspace.residual_driven_novel_rank == 1
    @test subspace.two_site_novel_rank == 1
    @test only(subspace.principal_cosines) >= 1 - 1e-10
    @test subspace.residual_driven_to_two_site_projection_error <= 1e-10
    @test subspace.two_site_to_residual_driven_projection_error <= 1e-10
    @test to_dense(initial) == before
    @test center(initial) == before_center

    failure_residual_policy = ResidualDrivenExpansion(
        trunc=TruncationScheme(maxdim=2),
        max_add=1,
        max_total_add=1,
        max_edges=1,
        max_rounds=1,
    )
    failure_two_site_policy = TwoSiteLinearPolicy(
        trunc=TruncationScheme(maxdim=2),
        sweeps=2,
        krylovdim=2,
        maxiter=1,
        local_tol=1e-30,
        residual_tol=1e-30,
    )
    failed_diagnostic = paired_linear_diagnostic(
        initial,
        operator,
        rhs,
        failure_residual_policy,
        failure_two_site_policy;
        a0=1.0,
        a1=0.0,
        krylovdim=2,
        maxiter=1,
        tol=1e-30,
        fit_nsweeps=1,
        fit_tol=0.0,
    )
    @test failed_diagnostic.classification.classification ==
          :inconclusive_failure
    @test !failed_diagnostic.residual_driven_report.committed
    @test isempty(
        failed_diagnostic.residual_driven_report.physical_residuals)
    @test to_dense(initial) == before

    incompatible_topology = mps_topology(3; prefix=:paired_external)
    incompatible_phys = _two_site_linear_phys(incompatible_topology)
    incompatible_initial = random_ttns(
        Xoshiro(0xe71e4a1), ComplexF64,
        incompatible_topology, incompatible_phys, ℂ^1)
    incompatible_residual_driven = random_ttns(
        Xoshiro(0xe71e4a2), ComplexF64,
        incompatible_topology, incompatible_phys, ℂ^2)
    incompatible_evidence =
        Graft.Evolution._paired_edge_subspace_evidence(
            incompatible_initial,
            incompatible_residual_driven,
            copy(incompatible_initial),
        )
    unavailable = only(filter(edge -> !edge.available, incompatible_evidence))
    @test unavailable.stop_reason == :incompatible_external_space
    @test unavailable.initial_rank == 1
    @test unavailable.residual_driven_rank == 2
    @test unavailable.two_site_rank == 1
    @test unavailable.residual_driven_novel_rank == 0
    @test unavailable.two_site_novel_rank == 0
    @test isempty(unavailable.principal_cosines)
    @test isnan(unavailable.residual_driven_to_two_site_projection_error)
    @test isnan(unavailable.two_site_to_residual_driven_projection_error)

    function rejects_without_mutation(
            residual_candidate, two_site_candidate)
        @test_throws ArgumentError paired_linear_diagnostic(
            initial,
            operator,
            rhs,
            residual_candidate,
            two_site_candidate;
            a0=1.0,
            a1=0.0,
            krylovdim=4,
            maxiter=2,
            tol=1e-10,
            fit_nsweeps=4,
            fit_tol=0.0,
        )
        @test to_dense(initial) == before
        @test center(initial) == before_center
    end

    rejects_without_mutation(
        ResidualDrivenExpansion(trunc=TruncationScheme(maxdim=1)),
        two_site_policy,
    )
    rejects_without_mutation(
        ResidualDrivenExpansion(
            trunc=TruncationScheme(
                maxdim=2,
                discarded_weight=0.9,
            ),
        ),
        two_site_policy,
    )
    nonzero_truncation = TruncationScheme(maxdim=2, atol=1e-12)
    rejects_without_mutation(
        ResidualDrivenExpansion(
            trunc=nonzero_truncation,
            max_add=1,
            max_total_add=1,
            max_edges=1,
            max_rounds=1,
        ),
        TwoSiteLinearPolicy(
            trunc=nonzero_truncation,
            sweeps=8,
            krylovdim=4,
            maxiter=2,
            local_tol=1e-10,
            residual_tol=1e-10,
        ),
    )
    rejects_without_mutation(
        ResidualDrivenExpansion(
            trunc=TruncationScheme(maxdim=2),
            max_add=1,
            max_total_add=1,
            max_edges=1,
            max_rounds=typemax(Int),
        ),
        two_site_policy,
    )
    rejects_without_mutation(
        residual_policy,
        TwoSiteLinearPolicy(
            trunc=TruncationScheme(maxdim=2),
            sweeps=7,
            krylovdim=4,
            maxiter=2,
            local_tol=1e-10,
            residual_tol=1e-10,
        ),
    )
    rejects_without_mutation(
        residual_policy,
        TwoSiteLinearPolicy(
            trunc=TruncationScheme(maxdim=2),
            sweeps=8,
            krylovdim=5,
            maxiter=2,
            local_tol=1e-10,
            residual_tol=1e-10,
        ),
    )
    rejects_without_mutation(
        residual_policy,
        TwoSiteLinearPolicy(
            trunc=TruncationScheme(maxdim=2),
            sweeps=8,
            krylovdim=4,
            maxiter=3,
            local_tol=1e-10,
            residual_tol=1e-10,
        ),
    )
    rejects_without_mutation(
        residual_policy,
        TwoSiteLinearPolicy(
            trunc=TruncationScheme(maxdim=2),
            sweeps=8,
            krylovdim=4,
            maxiter=2,
            local_tol=1e-9,
            residual_tol=1e-10,
        ),
    )
    rejects_without_mutation(
        residual_policy,
        TwoSiteLinearPolicy(
            trunc=TruncationScheme(maxdim=2),
            sweeps=8,
            krylovdim=4,
            maxiter=2,
            local_tol=1e-10,
            residual_tol=1e-9,
        ),
    )

    multi_topology = mps_topology(3; prefix=:paired_budget)
    multi_phys = _two_site_linear_phys(multi_topology)
    multi_hamiltonian = _two_site_linear_hamiltonian(multi_topology)
    multi_operator =
        ttno_from_opsum(multi_hamiltonian, multi_topology, multi_phys;
                        hermitian=true)
    multi_initial = random_ttns(
        Xoshiro(0xbad6e701), ComplexF64,
        multi_topology, multi_phys, ℂ^1)
    multi_rhs = random_ttns(
        Xoshiro(0xbad6e702), ComplexF64,
        multi_topology, multi_phys, ℂ^2)
    multi_before = to_dense(multi_initial)
    requirements = Graft.Evolution._paired_rank_growth_requirements(
        multi_initial, 2)
    @test requirements == (per_edge=1, total=2, edges=2)
    insufficient_growth = ResidualDrivenExpansion(
        trunc=TruncationScheme(maxdim=2),
        max_add=1,
        max_total_add=1,
        max_edges=2,
        max_rounds=1,
    )
    multi_two_site = TwoSiteLinearPolicy(
        trunc=TruncationScheme(maxdim=2),
        sweeps=2,
        krylovdim=4,
        maxiter=2,
        local_tol=1e-10,
        residual_tol=1e-10,
    )
    @test_throws ArgumentError paired_linear_diagnostic(
        multi_initial,
        multi_operator,
        multi_rhs,
        insufficient_growth,
        multi_two_site;
        a0=1.0,
        a1=0.0,
        krylovdim=4,
        maxiter=2,
        tol=1e-10,
        fit_nsweeps=1,
        fit_tol=0.0,
    )
    @test to_dense(multi_initial) == multi_before
end
