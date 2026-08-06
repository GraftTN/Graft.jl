using GraftFoundation: TruncationScheme, domain, id, norm, permute, split_svd
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

function _explicit_cbe_core(evolver, state, operator, current, next)
    core = _LGVD._cbe_projected_core(
        evolver, state, operator, current, next)
    frames = core.frames
    theta = _LGVD.two_site_tensor(state, frames.n, frames.m)
    action = _LGVD.eff_h2(
        evolver.cache, state, operator, frames.n, frames.m)(theta)
    grouped = permute(action, (frames.active_legs, frames.partner_legs))
    reference = core.Na' * grouped * _LGVD._cbe_right_injection(core.Np)
    return core, reference, theta
end

@testset "factorized double-complement matches explicit two-site action" begin
    for (state, operator) in (_lgvd_fixture(seed=2026080711),
                              _lgvd_graded_fixture())
        evolver = TDVP1_CBE(
            cbe=LGVDCBE(max_add=2),
            trunc=TruncationScheme(maxdim=2),
            verbose=false,
            threaded_channels=true,
            channel_slices=2,
            channel_minbatch=1,
            channel_min_flops=0,
            channel_memory_cap_bytes=1_000_000_000,
            cache=_LGVD.EnvCache(state.topo))
        for (current, next) in ((1, 2), (2, 1))
            move_center!(state, current; cache=evolver.cache)
            core, reference, theta = _explicit_cbe_core(
                evolver, state, operator, current, next)
            @test !hasproperty(
                _LGVD._cbe_projected_factors(
                    evolver, state, operator, current, next), :Θ)
            @test norm(core.M - reference) <=
                1e-11 * max(norm(reference), 1.0)
            @test core.core_norm ≈ norm(theta) rtol=1e-12 atol=1e-12
        end
    end
end

@testset "factorized Naive power iteration matches explicit projected core" begin
    state, operator = _lgvd_fixture(seed=2026080715)
    move_center!(state, 1)
    function policy(seed)
        return _LGVD.NaiveCBE(
            rng=Xoshiro(seed), max_add=1, oversample=1, poweriter=1,
            enr_atol=0, enr_rtol=0)
    end
    evolver = TDVP1_CBE(
        cbe=policy(0x51eed),
        trunc=TruncationScheme(maxdim=2),
        verbose=false,
        cache=_LGVD.EnvCache(state.topo))
    factors = _LGVD._cbe_projected_factors(
        evolver, state, operator, 1, 2)
    explicit = _LGVD._cbe_projected_materialize(factors)
    factorized, _ = _LGVD._cbe_implicit_rsvd_directions(
        policy(0x51eed), factors, 1)
    reference, _ = _LGVD._cbe_implicit_rsvd_directions(
        policy(0x51eed), explicit, 1)

    @test factorized !== nothing
    @test reference !== nothing
    @test norm(factorized * factorized' - reference * reference') < 1e-11
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
