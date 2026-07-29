using Test
using Graft
using Graft.TestUtils
using Graft.Backend: ℂ
using LinearAlgebra: norm
using Random: Xoshiro

isdefined(@__MODULE__, Symbol("@graft_testset")) || include("test_harness.jl")

function _rdi_fixture(; sites::Int, identity_only::Bool=false)
    topo = mps_topology(sites)
    S = spin_ops()
    phys = Dict(nodeid(topo, n) => S.P for n in 1:nnodes(topo))
    H = OpSum()
    if identity_only
        H += Term(0.23, SiteOp(nodeid(topo, 1), :I, S.I))
    else
        for n in 1:nnodes(topo)
            H += Term(0.17 * n, SiteOp(nodeid(topo, n), :Z, S.Z))
            H += Term(-0.11, SiteOp(nodeid(topo, n), :X, S.X))
        end
    end
    O = ttno_from_opsum(H, topo, phys; hermitian=true)
    ψ = random_ttns(
        Xoshiro(2606073000 + sites), ComplexF64, topo, phys, ℂ^1)
    return ψ, O
end

function _rdi_expansion_policy()
    return ResidualDrivenExpansion(
        trunc=TruncationScheme(maxdim=2),
        residual_trunc=TruncationScheme(),
        max_add=0,
        max_total_add=0,
        max_edges=0,
        max_rounds=0,
    )
end

function _rdi_two_site_policy()
    return TwoSiteLinearPolicy(
        trunc=TruncationScheme(maxdim=2),
        sweeps=1,
        krylovdim=4,
        maxiter=4,
        local_tol=1e-8,
        residual_tol=1e-8,
    )
end

@graft_testset "ImplicitLogTime residual-driven and two-site opt-in" begin
    ψ0, O = _rdi_fixture(; sites=1)
    h = 0.01
    common = (;
        krylovdim=4,
        maxiter=4,
        tol=1e-8,
        fit_nsweeps=1,
        fit_tol=0.0,
    )

    baseline_state = copy(ψ0)
    explicit_default_state = copy(ψ0)
    baseline = ImplicitLogTime(; common...)
    explicit_default =
        ImplicitLogTime(; common..., expansion=nothing, two_site=nothing)
    step!(baseline, baseline_state, O, -h)
    step!(explicit_default, explicit_default_state, O, -h)
    @test norm(to_dense(baseline_state) - to_dense(explicit_default_state)) <
          1e-13
    @test baseline.last_info == explicit_default.last_info
    @test isempty(baseline.last_residual_driven_reports)
    @test isempty(baseline.last_two_site_reports)

    expansion = _rdi_expansion_policy()
    for scheme in (LogBackwardEuler(), LogTrapezoid())
        ψ = copy(ψ0)
        ev = ImplicitLogTime(; scheme, expansion, common...)
        step!(ev, ψ, O, -h)
        @test length(ev.last_residual_driven_reports) == 1
        @test isempty(ev.last_two_site_reports)
        @test all(report -> report isa ResidualDrivenReport,
                  ev.last_residual_driven_reports)
        @test ev.last_info.normres ==
              maximum(last(report.physical_residuals)
                      for report in ev.last_residual_driven_reports)
        @test ev.last_info.converged ==
              Int(all(report.converged
                      for report in ev.last_residual_driven_reports))
    end

    two_site_state, two_site_operator =
        _rdi_fixture(; sites=2, identity_only=true)
    two_site = _rdi_two_site_policy()
    two_site_ev =
        ImplicitLogTime(; scheme=LogTrapezoid(), two_site, common...)
    step!(two_site_ev, two_site_state, two_site_operator, -h)
    @test isempty(two_site_ev.last_residual_driven_reports)
    @test length(two_site_ev.last_two_site_reports) == 1
    @test only(two_site_ev.last_two_site_reports) isa TwoSiteLinearReport
    @test two_site_ev.last_info.normres ==
          last(only(two_site_ev.last_two_site_reports).physical_residuals)
    @test two_site_ev.last_info.converged ==
          Int(only(two_site_ev.last_two_site_reports).converged)

    @test_throws ArgumentError ImplicitLogTime(;
        expansion, two_site, common...)
    @test_throws ArgumentError ImplicitLogTime(;
        scheme=LogGaussLegendre(2), expansion, common...)
    @test_throws ArgumentError ImplicitLogTime(;
        scheme=LogGaussLegendre(2), two_site, common...)
    @test ImplicitLogTime(
        ; scheme=LogGaussLegendre(2), common...).scheme isa LogGaussLegendre
    @test_throws ArgumentError ImplicitLogTime(;
        two_site=TwoSiteLinearPolicy(
            ; sweeps=1, krylovdim=5, maxiter=4,
            local_tol=1e-8, residual_tol=1e-8),
        common...)
    @test_throws ArgumentError ImplicitLogTime(;
        two_site=TwoSiteLinearPolicy(
            ; sweeps=1, krylovdim=4, maxiter=5,
            local_tol=1e-8, residual_tol=1e-8),
        common...)
    @test_throws ArgumentError ImplicitLogTime(;
        two_site=TwoSiteLinearPolicy(
            ; sweeps=1, krylovdim=4, maxiter=4,
            local_tol=2e-8, residual_tol=1e-8),
        common...)
    @test_throws ArgumentError ImplicitLogTime(;
        two_site=TwoSiteLinearPolicy(
            ; sweeps=1, krylovdim=4, maxiter=4,
            local_tol=1e-8, residual_tol=2e-8),
        common...)
    @test_throws ArgumentError ImplicitLogTime(;
        two_site=TwoSiteLinearPolicy(
            ; sweeps=2, krylovdim=4, maxiter=4,
            local_tol=1e-8, residual_tol=1e-8),
        common...)

    rollback_state, rollback_operator = _rdi_fixture(; sites=2)
    rollback_before = to_dense(rollback_state)
    failing_two_site = TwoSiteLinearPolicy(
        trunc=TruncationScheme(maxdim=2),
        sweeps=1,
        krylovdim=2,
        maxiter=1,
        local_tol=eps(Float64),
        residual_tol=eps(Float64),
    )
    rollback_ev = ImplicitLogTime(
        ; scheme=LogTrapezoid(), two_site=failing_two_site,
        normalize=true, krylovdim=2, maxiter=1, tol=eps(Float64),
        fit_nsweeps=1, fit_tol=0.0)
    step!(rollback_ev, rollback_state, rollback_operator, -h)
    @test !only(rollback_ev.last_two_site_reports).transaction_committed
    @test norm(to_dense(rollback_state) - rollback_before) == 0
end
