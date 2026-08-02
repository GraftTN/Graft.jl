using GraftFoundation: TruncationScheme, domain, id, norm, split_svd
using GraftNetworks: check_arrows, move_center!
using Test

function _lgvd_c1_target_reference(evolver, state, operator, current, next;
                                   maxrank=typemax(Int))
    factors = _LGVD._cbe_projected_core(
        evolver, state, operator, current, next)
    _, _, right = split_svd(
        factors.M, TruncationScheme(maxdim=maxrank))
    return factors.Np * transpose(right)
end

@testset "strict LGVD dense factorized C2/C3 matches C1 both directions" begin
    state, operator = _lgvd_fixture(seed=2026080212)
    strategy = LGVDCBE(
        max_add=2,
        preselection_threshold=1e-4,
        final_selection_threshold=1e-6)
    evolver = TDVP1_CBE(
        cbe=strategy,
        trunc=TruncationScheme(maxdim=2),
        verbose=false,
        cache=_LGVD.EnvCache(state.topo))

    for (current, next) in ((1, 2), (2, 1))
        move_center!(state, current; cache=evolver.cache)
        selection = _LGVD._lgvd_shrewd_selector(
            strategy, evolver, state, operator, current, next)
        reference = _lgvd_c1_target_reference(
            evolver, state, operator, current, next)
        target = _LGVD._lgvd_next_frame(state, current, next).tensor

        @test selection.selector === :shrewd_c2_c3
        @test selection.preselected_rank >= selection.selected_rank > 0
        @test selection.selected_rank <= strategy.max_add
        @test selection.c2_discarded_norm >= 0
        @test selection.c3_discarded_norm >= 0
        @test norm(target' * selection.complement) < 1e-11
        @test norm(selection.complement' * selection.complement -
                   id(domain(selection.complement))) < 1e-11
        @test norm(selection.complement * selection.complement' -
                   reference * reference') < 1e-10
    end
end

@testset "strict LGVD final V factor truncates Dhat to Dtilde" begin
    topo = mps_topology(4)
    spin = spin_ops()
    physical = Dict(Symbol("site$i") => spin.P for i in 1:4)
    terms = OpSum() +
        Term(0.2, SiteOp(:site1, :X, spin.X),
                  SiteOp(:site2, :Z, spin.Z)) +
        Term(0.3, SiteOp(:site2, :X, spin.X),
                  SiteOp(:site3, :Z, spin.Z)) +
        Term(-0.17, SiteOp(:site3, :X, spin.X),
                    SiteOp(:site4, :Z, spin.Z))
    operator = ttno_from_opsum(terms, topo, physical; hermitian=true)
    state = random_ttns(
        Xoshiro(2026080213), ComplexF64, topo, physical, ℂ^2; center=2)
    strategy = LGVDCBE(max_add=1)
    evolver = TDVP1_CBE(
        cbe=strategy,
        trunc=TruncationScheme(maxdim=3),
        verbose=false,
        cache=_LGVD.EnvCache(topo))

    selection = _LGVD._lgvd_shrewd_selector(
        strategy, evolver, state, operator, 2, 3)
    reference = _lgvd_c1_target_reference(
        evolver, state, operator, 2, 3; maxrank=1)
    target = _LGVD._lgvd_next_frame(state, 2, 3).tensor
    @test selection.preselected_rank == 2
    @test selection.selected_rank == 1
    @test norm(target' * selection.complement) < 1e-11
    @test norm(selection.complement' * selection.complement -
               id(domain(selection.complement))) < 1e-11
    @test norm(selection.complement * selection.complement' -
               reference * reference') < 1e-10

    move_center!(state, 1; cache=evolver.cache)
    numerically_empty = _LGVD._lgvd_shrewd_selector(
        strategy, evolver, state, operator, 1, 2)
    @test numerically_empty.selected_rank == 0
    @test numerically_empty.complement === nothing
end

@testset "strict LGVD graded factorized selector is safe" begin
    state, operator = _lgvd_graded_fixture()
    strategy = LGVDCBE(max_add=2)
    evolver = TDVP1_CBE(
        cbe=strategy,
        trunc=TruncationScheme(maxdim=2),
        verbose=false,
        cache=_LGVD.EnvCache(state.topo))

    for (current, next) in ((1, 2), (2, 1))
        move_center!(state, current; cache=evolver.cache)
        selection = _LGVD._lgvd_shrewd_selector(
            strategy, evolver, state, operator, current, next)
        target = _LGVD._lgvd_next_frame(state, current, next).tensor
        @test selection.selector === :shrewd_c2_c3
        @test selection.selected_rank >= 0
        if selection.complement !== nothing
            @test norm(target' * selection.complement) < 1e-10
            @test norm(selection.complement' * selection.complement -
                       id(domain(selection.complement))) < 1e-10
        end
        @test check_arrows(state)
    end
end
