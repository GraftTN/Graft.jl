import GraftEvolution
import GraftFoundation
using GraftEvolution: LGVDCBE, LGVDCBEInfo, TDVP1_CBE, step!
using GraftFoundation: FermionParity, TruncationScheme, domain, dim, left_null,
    mps_topology, star_topology, ℂ
using GraftNetworks: center, check_arrows
using GraftSymbolic: OpSum, SiteOp, Term, fermion_ops_z2, spin_ops
using GraftTestUtils: product_ttns, random_ttns, to_dense
using GraftTTNOBuild: ttno_from_opsum
using Random: Xoshiro
using Test

const _LGVD = GraftEvolution.Evolution

function _lgvd_fixture(; seed=2026080207)
    topo = mps_topology(2)
    spin = spin_ops()
    physical = Dict(:site1 => spin.P, :site2 => spin.P)
    terms = OpSum() +
        Term(0.37, SiteOp(:site1, :Z, spin.Z)) +
        Term(-0.19, SiteOp(:site2, :X, spin.X)) +
        Term(0.23, SiteOp(:site1, :X, spin.X),
                   SiteOp(:site2, :Z, spin.Z))
    operator = ttno_from_opsum(terms, topo, physical; hermitian=true)
    state = random_ttns(
        Xoshiro(seed), ComplexF64, topo, physical, ℂ^1;
        center=topo.root)
    return state, operator
end

function _lgvd_graded_fixture()
    topo = mps_topology(2)
    fermion = fermion_ops_z2()
    physical = Dict(:site1 => fermion.P, :site2 => fermion.P)
    basis = Dict(:site1 => FermionParity(0),
                 :site2 => FermionParity(1))
    terms = OpSum() +
        Term(-0.5, SiteOp(:site1, :N, fermion.N)) +
        Term(-1.0, SiteOp(:site1, :Cd, fermion.Cd),
                   SiteOp(:site2, :C, fermion.C)) +
        Term(-1.0, SiteOp(:site1, :C, fermion.C),
                   SiteOp(:site2, :Cd, fermion.Cd))
    operator = ttno_from_opsum(terms, topo, physical; hermitian=true)
    state = product_ttns(ComplexF64, topo, physical, basis)
    return state, operator
end

function _synthetic_lgvd_selector(strategy, _, state, _, current, next)
    frame = _LGVD._lgvd_next_frame(state, current, next)
    complement = left_null(frame.tensor)
    available = dim(domain(complement))
    if available == 0 || strategy.max_add == 0
        return _LGVD.LGVDSelection(nothing, 0, 0, 0.0, 0.0,
                                   :synthetic_test)
    end
    selected, _, _ = GraftFoundation.split_svd(
        complement, TruncationScheme(maxdim=min(strategy.max_add, available)))
    rank = dim(domain(selected))
    return _LGVD.LGVDSelection(
        selected, rank, rank, 0.0, 0.0, :synthetic_test)
end

@testset "strict LGVD production selector completes public step" begin
    state, operator = _lgvd_fixture()
    evolver = TDVP1_CBE(
        cbe=LGVDCBE(max_add=1),
        trunc=TruncationScheme(maxdim=1),
        krylovdim=8,
        tol=1e-12,
        verbose=false)

    step!(evolver, state, operator, -1e-3im)

    @test evolver.cache !== nothing
    @test evolver.lgvd_phase === _LGVD.LGVDRightToLeft
    @test evolver.last_cbe_info isa LGVDCBEInfo
    @test evolver.last_cbe_info.completed
    @test evolver.last_cbe_info.blocking_reason === nothing
    @test evolver.last_cbe_info.shrewd_applied
    @test evolver.last_cbe_info.strict_schedule_applied
    @test all(edge.state_preserving for edge in
              evolver.last_cbe_info.edge_reports)
    @test all(isfinite, to_dense(state))
end

@testset "strict LGVD synthetic sweep exercises LRL/RLR and cap rotation" begin
    state, operator = _lgvd_fixture(seed=2026080208)
    evolver = TDVP1_CBE(
        cbe=LGVDCBE(max_add=1),
        trunc=TruncationScheme(maxdim=1),
        krylovdim=8,
        tol=1e-12,
        verbose=false)

    _LGVD._lgvd_step_with_selector!(
        evolver, state, operator, -0.01im, _synthetic_lgvd_selector;
        selector_name=:synthetic_test, shrewd_applied=false)
    first_info = evolver.last_cbe_info

    @test first_info.completed
    @test first_info.strict_schedule_applied
    @test !first_info.shrewd_applied
    @test first_info.phase_before === _LGVD.LGVDLeftToRight
    @test first_info.phase_after === _LGVD.LGVDRightToLeft
    @test [edge.direction for edge in first_info.edge_reports] ==
          [_LGVD.LGVDLeftToRight, _LGVD.LGVDRightToLeft,
           _LGVD.LGVDLeftToRight]
    @test all(edge.rank_before == 1 for edge in first_info.edge_reports)
    @test all(edge.expanded_rank == 2 for edge in first_info.edge_reports)
    @test all(edge.final_rank == 1 for edge in first_info.edge_reports)
    @test all(edge.state_preserving for edge in first_info.edge_reports)
    @test all(edge.embedding_error < 1e-12 for edge in first_info.edge_reports)
    @test all(edge.saturated_rotation_enabled for edge in first_info.edge_reports)
    @test all(isfinite(edge.discarded_weight) && edge.discarded_weight >= 0
              for edge in first_info.edge_reports)

    _LGVD._lgvd_step_with_selector!(
        evolver, state, operator, -0.01im, _synthetic_lgvd_selector;
        selector_name=:synthetic_test, shrewd_applied=false)
    second_info = evolver.last_cbe_info
    @test second_info.phase_before === _LGVD.LGVDRightToLeft
    @test second_info.phase_after === _LGVD.LGVDLeftToRight
    @test [edge.direction for edge in second_info.edge_reports] ==
          [_LGVD.LGVDRightToLeft, _LGVD.LGVDLeftToRight,
           _LGVD.LGVDRightToLeft]
    @test all(isfinite, to_dense(state))
end

@testset "strict LGVD chain guard is pre-mutating" begin
    @test length(_LGVD._lgvd_chain_path(mps_topology(4))) == 4
    @test_throws ArgumentError _LGVD._lgvd_chain_path(star_topology(3, 1))
    @test_throws ArgumentError _LGVD._lgvd_chain_path(mps_topology(1))
end

@testset "strict LGVD zero padding is fermion-parity safe" begin
    state, operator = _lgvd_graded_fixture()
    evolver = TDVP1_CBE(
        cbe=LGVDCBE(max_add=1),
        trunc=TruncationScheme(maxdim=1),
        krylovdim=8,
        tol=1e-12,
        verbose=false)

    _LGVD._lgvd_step_with_selector!(
        evolver, state, operator, -1e-3im, _synthetic_lgvd_selector;
        selector_name=:synthetic_graded_test, shrewd_applied=false)

    @test check_arrows(state)
    @test all(edge.state_preserving for edge in
              evolver.last_cbe_info.edge_reports)
    @test all(edge.embedding_error < 1e-11 for edge in
              evolver.last_cbe_info.edge_reports)
    @test all(isfinite, to_dense(state))
end
