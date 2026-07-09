const _CP = GRAFT.Contractions
const _Planning = GRAFT.Contractions.Planning

function _assert_planned_matches_ncon(spec, statics, planned, x;
                                      rtol::Real=1e-13, atol::Real=1e-13)
    got = planned(x)
    ref = _CP._ncon_effective_reference(spec, x, statics)
    @test numout(got) == spec.out_partition[1]
    @test numin(got) == spec.out_partition[2]
    @test norm(got - ref) <= atol + rtol * max(norm(ref), 1)
    return got
end

"""
Exercise a one-site planned map on the center tensor and independent Krylov-like
vectors. This deliberately goes beyond the state-tensor-only A/B checks: TDVP2
passes arbitrary Krylov combinations through its root backward site update.
"""
function _assert_h1_family_matches_ncon!(cache, ψ, O, n, rng;
                                         rtol::Real=1e-11,
                                         atol::Real=1e-11)
    spec, statics, protos = _CP._h1_spec(cache, ψ, O, n)
    planned = eff_h1(cache, ψ, O, n)
    envfirst = _Planning.plan_contraction(spec, protos; optimize=false)
    envfirst_map = _CP.EffectiveMap(envfirst, statics)

    x0 = ψ.tensors[n]
    x1 = randn(rng, ComplexF64,
               GRAFT.Backend.codomain(x0) ← domain(x0))
    x2 = randn(rng, ComplexF64,
               GRAFT.Backend.codomain(x0) ← domain(x0))
    inputs = (x0, x1, x2)

    for x in inputs
        ref = _CP._ncon_effective_reference(spec, x, statics)
        got = planned(x)
        got_envfirst = envfirst_map(x)
        scale = max(norm(ref), one(real(norm(ref))))
        @test norm(got - ref) <= atol + rtol * scale
        @test norm(got_envfirst - ref) <= atol + rtol * scale
    end

    # `ishermitian=true` in TDVP2 selects the Lanczos path, so test the
    # operator property itself after the same post-split gauge configuration.
    a, b = x1, x2
    @test dot(a, planned(b)) ≈ dot(planned(a), b) rtol=rtol atol=atol
    @test dot(a, envfirst_map(b)) ≈ dot(envfirst_map(a), b) rtol=rtol atol=atol

    # Values are never stored in `cache.plans`; forcing a recompilation while
    # retaining the current environments distinguishes a stale-plan bug from a
    # plan-executor bug without rebuilding the physical state.
    empty!(cache.plans)
    spec_fresh, statics_fresh, _ = _CP._h1_spec(cache, ψ, O, n)
    planned_fresh = eff_h1(cache, ψ, O, n)
    for x in inputs
        ref_fresh = _CP._ncon_effective_reference(spec_fresh, x, statics_fresh)
        scale = max(norm(ref_fresh), one(real(norm(ref_fresh))))
        @test norm(planned_fresh(x) - ref_fresh) <= atol + rtol * scale
    end

    return (; spec, statics, planned, inputs)
end

function _exercise_effective_maps!(ψ, O)
    topo = topology(ψ)
    for n in 1:nnodes(topo)
        move_center!(ψ, n)
        cache = EnvCache(topo)
        spec, statics, protos = _CP._h1_spec(cache, ψ, O, n)
        planned = eff_h1(cache, ψ, O, n)
        @test planned isa _CP.EffectiveMap
        _assert_planned_matches_ncon(spec, statics, planned, ψ.tensors[n])

        # Phase 2 may choose the dense candidate, but it must never violate the
        # Phase-1 env-first peak-memory floor.  Phase 3 adds a second,
        # symmetry-reduced stored-payload guard without weakening that dense
        # first-class memory ceiling.
        envfirst = _Planning.plan_contraction(spec, protos; optimize=false)
        @test planned.plan.peak_elements <= envfirst.peak_elements
        @test isfinite(planned.plan.sector_peak_elements)
        @test planned.plan.sector_peak_elements <= envfirst.sector_peak_elements
    end

    for (n, m) in edges(topo)
        move_center!(ψ, n)
        Θ = two_site_tensor(ψ, n, m)
        @test space(Θ) == two_site_space(ψ, n, m)

        cache2 = EnvCache(topo)
        spec2, statics2, _ = _CP._h2_spec(cache2, ψ, O, n, m)
        planned2 = eff_h2(cache2, ψ, O, n, m)
        got2 = _assert_planned_matches_ncon(spec2, statics2, planned2, Θ)
        @test numout(got2) == numind(Θ)
        @test numin(got2) == 0

        # The normal same-space link is a valid h0 input and keeps this small
        # TDVP map on the same planned/reference path as h1/h2.
        C = id(virtualspace(ψ, n))
        cache0 = EnvCache(topo)
        spec0, statics0, _ = _CP._h0_spec(cache0, ψ, O, n, m)
        planned0 = eff_h0(cache0, ψ, O, n, m)
        _assert_planned_matches_ncon(spec0, statics0, planned0, C)
    end
    return nothing
end

@testset "compiled contraction plans: A/B effective maps" begin
    for topo in (mps_topology(4), star_topology(2, 1), binary_topology(2),
                 fork_topology(2, 1))
        phys = allspin(topo)
        O = ttno_from_opsum(tfi(topo; g=0.41), topo, phys; hermitian=true)
        ψ = random_ttns(MersenneTwister(1300 + nnodes(topo)), ComplexF64,
                        topo, phys, ℂ^2)
        _exercise_effective_maps!(ψ, O)
    end

    # Neutral U(1) TTNO/state: the plan uses exact TensorKit spaces, not just
    # dense array dimensions, so this is the symmetric counterpart of the
    # trivial-sector checks above.
    U = spin_ops_u1()
    topo = star_topology(2, 1)
    phys = Dict(nodeid(topo, i) => U.P for i in 1:nnodes(topo))
    H = OpSum()
    for n in 1:nnodes(topo)
        H += Term(0.17 + 0.03 * n, SiteOp(nodeid(topo, n), :Z, U.Z))
    end
    O = ttno_from_opsum(H, topo, phys; hermitian=true)
    V = U1Space(-1 => 1, 0 => 2, 1 => 1)
    ψ = random_ttns(MersenneTwister(1401), ComplexF64, topo, phys, V)
    _exercise_effective_maps!(ψ, O)
end

@testset "compiled contraction plans: sector-aware structural planning" begin
    # Pure HomSpace metadata: this must not allocate TensorMap payloads.  It
    # pins TensorKit's flat-domain-leg convention and the exact two-sector
    # GEMM accounting used by the Phase-3 planner.
    P = U1Space(0 => 1, 1 => 1)
    V = U1Space(-1 => 1, 0 => 2, 1 => 1)
    A0 = (P ⊗ V) ← V
    B0 = V ← P
    pA = ((1, 2), (3,))
    pB = ((1,), (2,))
    pAB = ((1, 2, 3), ())
    profile = GRAFT.Backend.pair_cost(A0, pA, false, B0, pB, false, pAB)
    @test A0[3] == dual(domain(A0)[1])
    @test profile.supported
    @test profile.block_count == 2
    @test profile.sector_flops == 18
    @test profile.output_elements == 6
    @test profile.largest_block_elements == 3
    @test profile.output == TensorOperations.tensorcontract(A0, pA, false,
                                                             B0, pB, false, pAB)

    # A final codomain/domain repartition can fuse matrix-product sectors into
    # a larger stored output block.  Keep both diagnostics so the plan-level
    # peak-block metric never underreports the live root output.
    Ar = (P ⊗ P) ← P
    Br = (P ⊗ V) ← V
    root_profile = GRAFT.Backend.pair_cost(
        Ar, ((1, 2), (3,)), false,
        Br, ((1,), (2, 3)), false,
        ((1, 3), (2, 4)),
    )
    @test root_profile.largest_block_elements == 8
    @test root_profile.output_largest_block_elements == 9
    @test root_profile.peak_block_elements == 9

    # The dense model prefers (A*B)*C (60 < 63), but the exact U(1) block
    # model prefers A*(B*C) (28 < 30).  This deterministic three-map fixture
    # proves that Phase 3 is a real order choice, not only diagnostics.
    A = U1Space(0 => 1, 1 => 1) ← U1Space(0 => 1, 1 => 2)
    B = U1Space(0 => 1, 1 => 2) ← U1Space(0 => 1, 1 => 4)
    C = U1Space(0 => 1, 1 => 4) ← U1Space(0 => 2, 1 => 1)
    spec = _Planning.ContractionSpec(
        Vector{Int}[[-1, 1], [1, 2], [2, -2]],
        Bool[false, false, false], 2, (1, 1), 1;
        preferred_slots=[2, 3],
    )
    dense = _Planning.plan_contraction(spec, (A, B, C);
                                         sector_aware=false, memory_weight=0)
    sector = _Planning.plan_contraction(spec, (A, B, C);
                                          sector_aware=true, memory_weight=0)
    envfirst = _Planning.plan_contraction(spec, (A, B, C); optimize=false)
    @test dense.strategy == :env_first
    @test dense.flops == 60
    @test dense.sector_flops == 30
    @test sector.strategy == :sector_bounded
    @test sector.flops == 63
    @test sector.sector_flops == 28
    @test (sector.steps[1].a, sector.steps[1].b) == (2, 3)
    @test sector.peak_elements <= envfirst.peak_elements
    @test sector.sector_peak_elements <= envfirst.sector_peak_elements

    key_dense = _Planning.plan_key(:sector_fixture, spec, (A, B, C), ComplexF64;
                                   sector_aware=false, memory_weight=0)
    key_sector = _Planning.plan_key(:sector_fixture, spec, (A, B, C), ComplexF64;
                                    sector_aware=true, memory_weight=0)
    @test key_dense != key_sector
end

@testset "compiled contraction plans: cache, graph identity, and gauge" begin
    # Two-site arms give each root-child virtual leg a physical support of four,
    # so `ℂ^2 → ℂ^3` is a genuine shape change. A one-site arm would cap both
    # random states at dimension two and make this cache-key assertion vacuous.
    topo = star_topology(2, 2)
    phys = allspin(topo)
    O = ttno_from_opsum(tfi(topo; g=0.63), topo, phys; hermitian=true)
    ψ = random_ttns(MersenneTwister(1501), ComplexF64, topo, phys, ℂ^2)
    n = topo.root
    move_center!(ψ, n)
    cache = EnvCache(topo)

    # The cache identity includes the optimization objective as well as the
    # network shape. This keeps a future benchmark-calibrated memory weight
    # from silently reusing a plan selected under the default objective.
    spec_key, _, protos_key = _CP._h1_spec(cache, ψ, O, n)
    key_default = _Planning.plan_key(:h1, spec_key, protos_key, ComplexF64)
    key_envfirst = _Planning.plan_key(:h1, spec_key, protos_key, ComplexF64;
                                      optimize=false)
    key_memheavy = _Planning.plan_key(:h1, spec_key, protos_key, ComplexF64;
                                      memory_weight=2)
    @test key_default != key_envfirst
    @test key_default != key_memheavy

    h1a = eff_h1(cache, ψ, O, n)
    after_miss = _CP.plan_cache_stats(cache)
    @test after_miss.misses == 1
    @test after_miss.hits == 0
    h1b = eff_h1(cache, ψ, O, n)
    after_hit = _CP.plan_cache_stats(cache)
    @test after_hit.misses == 1
    @test after_hit.hits == 1
    @test h1a.plan.steps === h1b.plan.steps

    # Public effective-map keywords feed the cache objective: even when the
    # selected dense tree happens to equal env-first for this small network,
    # a caller requesting Phase-1-only planning gets a distinct cache entry.
    plans_after_default = length(cache.plans)
    h1_envfirst = eff_h1(cache, ψ, O, n; optimize=false)
    @test h1_envfirst.plan.strategy == :env_first
    @test length(cache.plans) == plans_after_default + 1

    plans_before_invalidation = length(cache.plans)
    _CP.invalidate_node!(cache, n)
    @test length(cache.plans) == plans_before_invalidation
    _ = eff_h1(cache, ψ, O, n)
    @test _CP.plan_cache_stats(cache).hits == 2

    # Same topology but a different state-bond shape must miss rather than
    # reuse a plan; clearing only environments preserves the old plan for this
    # exact cache-key check.
    ψwide = random_ttns(MersenneTwister(1502), ComplexF64, topo, phys, ℂ^3)
    empty!(cache.envs)
    _ = eff_h1(cache, ψwide, O, n)
    @test length(cache.plans) > plans_before_invalidation

    # H2 on two symmetric root children has equal dense dimensions but a
    # different Wm crossed-child leg. The label graph is therefore part of the
    # key; reusing one plan here would be a silent dimension-valid bug.
    c1, c2 = topo.children[topo.root]
    cache_h2 = EnvCache(topo)
    move_center!(ψ, c1)
    h2a = eff_h2(cache_h2, ψ, O, c1, topo.root)
    move_center!(ψ, c2)
    h2b = eff_h2(cache_h2, ψ, O, c2, topo.root)
    @test length(cache_h2.plans) == 2
    @test h2a.plan.steps !== h2b.plan.steps

    # A closed gauge excursion preserves the local effective observable.
    move_center!(ψ, n)
    e0 = dot(ψ.tensors[n], eff_h1(EnvCache(topo), ψ, O, n)(ψ.tensors[n]))
    move_center!(ψ, c1)
    move_center!(ψ, n)
    e1 = dot(ψ.tensors[n], eff_h1(EnvCache(topo), ψ, O, n)(ψ.tensors[n]))
    @test e0 ≈ e1 rtol=1e-12 atol=1e-12
end

@testset "compiled contraction plans: mixed-boson post-TDVP2 root h1" begin
    # This is the previously failing physical shape: boson leaves have d = 3,
    # while the spin root has d = 2. The all-spin A/B set above has d = 2 on
    # every physical leg and cannot expose an index permutation masked by equal
    # dimensions. The step below mirrors TDVP2's bond-forward sequence through
    # the exact post-split root gauge, immediately before `_site_backward!`.
    S = spin_ops()
    B = boson_ops(2)
    topo = star_topology(2, 1; center=:spin, prefix=:b)
    root = topo.root
    leaf = first(topo.children[root])
    phys = Dict(:spin => S.P, :b1_1 => B.P, :b2_1 => B.P)
    H = boson_modes([:b1_1 => 0.7, :b2_1 => 1.1]; ops=B)
    H += Term(-0.35, SiteOp(:spin, :X, S.X))
    H += BosonCoupling([(:spin, :b1_1) => 0.22, (:spin, :b2_1) => -0.18],
                       :density; matter_ops=S, boson_ops=B, density=:Z)
    O = ttno_from_opsum(H, topo, phys; hermitian=true)
    ψ = random_ttns(MersenneTwister(1601), ComplexF64, topo, phys, ℂ^4)
    cache = EnvCache(topo)

    move_center!(ψ, leaf; cache=cache)
    Θ = two_site_tensor(ψ, leaf, root)
    spec2, statics2, _ = _CP._h2_spec(cache, ψ, O, leaf, root)
    h2 = eff_h2(cache, ψ, O, leaf, root)
    _assert_planned_matches_ncon(spec2, statics2, h2, Θ)

    # This is `_bond_forward!` without calling the private TDVP helper: evolve
    # the two-site center, invalidate the affected environments, and split
    # back with the center at the root. The local dimension is only 18 here.
    Θnext, _ = GRAFT.Evolution.exponentiate(h2, -0.06, Θ;
                                              ishermitian=true,
                                              krylovdim=12,
                                              tol=1e-12)
    _CP.invalidate_edge!(cache, leaf, root)
    split_two_site!(ψ, Θnext, leaf, root;
                    trunc=TruncationScheme(maxdim=12, atol=1e-12),
                    center_on=:m)
    @test ψ.center == root
    @test check_arrows(ψ)

    checked = _assert_h1_family_matches_ncon!(cache, ψ, O, root,
                                               MersenneTwister(1602))

    # Match the actual backward-site Krylov primitive as well as individual
    # matvecs. If this fails while the matvecs pass, the issue is specifically
    # in the planned map's linear-operator/partition behavior under Lanczos.
    x = checked.inputs[1]
    reference_h1 = z -> _CP._ncon_effective_reference(checked.spec, z,
                                                        checked.statics)
    y_planned, _ = GRAFT.Evolution.exponentiate(checked.planned, 0.06, x;
                                                  ishermitian=true,
                                                  krylovdim=12,
                                                  tol=1e-12)
    y_reference, _ = GRAFT.Evolution.exponentiate(reference_h1, 0.06, x;
                                                    ishermitian=true,
                                                    krylovdim=12,
                                                    tol=1e-12)
    @test norm(y_planned - y_reference) <= 1e-10 * max(norm(y_reference), 1)
end

@testset "compiled contraction plans: non-square TDVP h0 link" begin
    # Build the actual TDVP1 child→parent QR seam with an intentionally wide
    # old bond. `left_orth` reduces P (dim 2) ← V_old (dim 4) to a compact
    # link C :: V_new ← V_old, so h0 must accept a genuinely non-square
    # TensorMap rather than only the square identity used by the generic A/B
    # loop above.
    S = spin_ops()
    topo = mps_topology(2)
    root = topo.root
    child = only(topo.children[root])
    P, Vold = S.P, ℂ^4
    rng = MersenneTwister(1701)
    Achild = randn(rng, ComplexF64, P ← Vold)
    # A root still has one *unit* parent leg. `one(P)` is a rank-zero
    # ProductSpace, whereas `oneunit(P)` is the required one-leg ℂ¹ space.
    Aroot = randn(rng, ComplexF64, Vold ⊗ P ← oneunit(P))
    tensors = Vector{GRAFT.Backend.AbstractTensorMap}(undef, nnodes(topo))
    tensors[child] = Achild
    tensors[root] = Aroot
    ψ = TTNS(topo, tensors, child)

    phys = Dict(nodeid(topo, i) => P for i in 1:nnodes(topo))
    O = ttno_from_opsum(tfi(topo; g=0.37), topo, phys; hermitian=true)
    cache = EnvCache(topo)

    # The private helper is deliberately used here because this is the exact
    # TDVP seam whose returned C is passed to eff_h0 immediately afterward.
    C = GRAFT.Evolution._split_link_up(TDVP1(), ψ, O, child, root, -0.1)
    _CP.invalidate_node!(cache, child)
    spec0, statics0, protos0 = _CP._h0_spec(cache, ψ, O, child, root)

    @test space(C) == protos0[1]
    @test numout(C) == 1 && numin(C) == 1
    @test dim(space(C, 1)) < dim(domain(C)[1])

    h0 = eff_h0(cache, ψ, O, child, root)
    _assert_planned_matches_ncon(spec0, statics0, h0, C)

    # Krylov will probe more than the QR factor itself, so cover an independent
    # non-square tensor in exactly the h0 input space as well.
    Cprobe = randn(MersenneTwister(1702), ComplexF64, protos0[1])
    _assert_planned_matches_ncon(spec0, statics0, h0, Cprobe)
end

@testset "compiled contraction plans: fork-spine memory objective" begin
    # Shape-only replica of plan §0.3 at s0_1. No TensorMap data is allocated:
    # this lets the test pin the baseline headline even on memory-constrained
    # CI runners. The first physical x×W pair has 445,644,800 Float64 elements
    # (3.3203125 GiB), whereas the env-first plan stays below the 1.5 GiB gate.
    χ, d = 32, 2
    x = (ℂ^χ ⊗ ℂ^χ ⊗ ℂ^d) ← ℂ^χ
    W = (ℂ^20 ⊗ ℂ^17 ⊗ ℂ^d) ← (ℂ^d ⊗ ℂ^20)
    env(ω) = (ℂ^χ ⊗ ℂ^ω ⊗ ℂ^χ) ← one(ℂ^χ)
    spec = _CP.ContractionSpec(
        Vector{Int}[
            [2, 4, 1, 6],
            [3, 5, -3, 1, 7],
            [2, 3, -1],
            [4, 5, -2],
            [6, 7, -4],
        ],
        Bool[false, false, false, false, false], 4, (3, 1), 1;
        preferred_slots=[3, 4, 2, 5],
    )
    physical_first = Float64(χ)^3 * 20 * 17 * d * 20
    @test physical_first == 445_644_800
    @test physical_first * 8 / 1024^3 ≈ 3.3203125

    envfirst = _Planning.plan_contraction(spec, (x, W, env(20), env(17), env(20));
                                            optimize=false)
    selected = _Planning.plan_contraction(spec, (x, W, env(20), env(17), env(20)))
    gate = 1.5 * 1024^3 / 8
    # These no-payload ℂ HomSpaces carry dimensions but intentionally do not
    # encode a composable arrow layout.  There is no sector split to optimize,
    # so Phase 3 must retain the dense-equivalent metadata path rather than
    # invoking TensorKit structural composition just to rediscover Phase 2.
    @test !GRAFT.Backend.sector_cost_nontrivial(x)
    @test envfirst.sector_peak_elements == envfirst.peak_elements
    @test envfirst.sector_flops == 2 * envfirst.flops
    @test envfirst.peak_elements < gate
    @test selected.peak_elements <= envfirst.peak_elements
end
