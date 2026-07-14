# Targeted P0/P1 contraction-memory checks.  This file is intentionally
# self-contained so the constrained host can invoke it directly. `runtests.jl`
# includes it for stronger-host validation too; keep fixtures shape-only or
# two/three-map small so it never becomes a benchmark.
using Test
using Random
using LinearAlgebra
using Graft
using Graft.TestUtils: random_ttns, to_dense
using Graft.Backend: ℂ, ⊗, ←, U1Space, norm, numout, numin
using Graft.Contractions: env!

isdefined(@__MODULE__, Symbol("@graft_testset")) || include("test_harness.jl")

const _P0P1Planning = Graft.Contractions.Planning
const _P0P1_TARGET = lowercase(get(ENV, "GRAFT_P0P1_TARGET", "all"))
_P0P1_TARGET in ("all", "m1", "m2", "m3", "m4", "m5", "m6", "m7") ||
    throw(ArgumentError("unknown GRAFT_P0P1_TARGET=$_P0P1_TARGET"))
_p0p1_enabled(group::Symbol) = _P0P1_TARGET in ("all", String(group))
_p0p1_explicit(group::Symbol) = _P0P1_TARGET == String(group)

@info "Graft P0/P1 targeted BLAS configuration" config=LinearAlgebra.BLAS.get_config() threads=LinearAlgebra.BLAS.get_num_threads()

function _p0p1_permuted_two_map(rng)
    # Both operands need a non-identity expert-mode layout before their
    # contraction.  This gives the byte model a tiny deterministic known
    # permutation buffer without allocating more than a pair of small maps.
    Aspace = ℂ^3 ← ℂ^2
    Bspace = ℂ^4 ← ℂ^3
    spec = _P0P1Planning.ContractionSpec(
        Vector{Int}[[1, -1], [-2, 1]], Bool[false, false], 2, (1, 1), 1;
        preferred_slots=[2],
    )
    A = randn(rng, ComplexF64, Aspace)
    B = randn(rng, ComplexF64, Bspace)
    return spec, A, B
end

function _p0p1_sandwich_fixture(rng; u1::Bool=false)
    topo = mps_topology(2)
    ops = u1 ? spin_ops_u1() : spin_ops()
    phys = Dict(nodeid(topo, i) => ops.P for i in 1:nnodes(topo))
    H = OpSum()
    for i in 1:nnodes(topo)
        H += Term(0.13 + 0.07i, SiteOp(nodeid(topo, i), :Z, ops.Z))
    end
    O = ttno_from_opsum(H, topo, phys; hermitian=true)
    bond = u1 ? U1Space(-1 => 1, 0 => 2, 1 => 1) : ℂ^2
    ket = random_ttns(rng, ComplexF64, topo, phys, bond)
    bra = random_ttns(rng, ComplexF64, topo, phys, bond)
    return topo, ops, O, ket, bra
end

function _p0p1_env_matches_ncon!(cache, ket, O, bra, u, v)
    for w in neighbors(topology(ket), u)
        w == v || env!(cache, ket, O, bra, w, u)
    end
    got = Graft.Contractions.build_env(cache, ket, O, bra, u, v)
    ref = Graft.Contractions._build_env_ncon_reference(ket, O, bra, u, v,
                                                        cache.envs)
    if got isa Number
        @test got ≈ ref rtol=1e-12 atol=1e-12
    else
        @test norm(got - ref) <= 1e-12 * max(norm(ref), 1)
    end
    return got, ref
end

"""Independent retained-ncon recursion for planned fit A/B checks."""
function _p0p1_fit_ref_env!(φ, ψ, u, v,
                             envs::Dict{Tuple{Int,Int},Graft.Backend.AbstractTensorMap})
    return get!(envs, (u, v)) do
        for w in neighbors(topology(φ), u)
            w == v || _p0p1_fit_ref_env!(φ, ψ, w, u, envs)
        end
        Graft.Networks._fit_build_env_ncon_reference(φ, ψ, u, v, envs)
    end
end

function _p0p1_fit_maps_match(got, ref; rtol=1e-12, atol=1e-12)
    @test norm(got - ref) <= atol + rtol * max(norm(ref), 1)
    return nothing
end

"""Independent retained-ncon recursion for operator-aware fit A/B checks."""
function _p0p1_fit_operator_ref_env!(φ, ψ, O, u, v,
                                      envs::Dict{Tuple{Int,Int},Graft.Backend.AbstractTensorMap})
    return get!(envs, (u, v)) do
        for w in neighbors(topology(φ), u)
            w == v || _p0p1_fit_operator_ref_env!(φ, ψ, O, w, u, envs)
        end
        Graft.Networks._fit_operator_build_env_ncon_reference(φ, ψ, O, u, v, envs)
    end
end

if _p0p1_enabled(:m1)
@graft_extended_testset _p0p1_explicit(:m1) "P0/P1: live contraction memory model" begin
    rng = MersenneTwister(20260710)
    spec, A, B = _p0p1_permuted_two_map(rng)
    plan = _P0P1Planning.plan_contraction(spec, (A, B);
                                            optimize=false,
                                            scalar_type=ComplexF64)
    planned = _P0P1Planning.EffectiveMap(plan, (B,))
    reference = _P0P1Planning.ncon_reference(spec, A, (B,))
    @test norm(planned(A) - reference) <= 1e-13 * max(norm(reference), 1)

    metrics = _P0P1Planning.plan_metrics(plan)
    @test plan.peak_elements == metrics.peak_elements
    @test plan.scalar_bytes == sizeof(ComplexF64)
    @test plan.operand_bytes > 0
    @test plan.live_peak_bytes > plan.operand_bytes
    @test plan.known_permutation_peak_bytes > 0
    @test plan.known_temporary_peak_bytes >= plan.known_permutation_peak_bytes
    @test metrics.live_peak_bytes == plan.live_peak_bytes
    @test metrics.sector_live_peak_bytes == plan.sector_live_peak_bytes

    # The hard cap applies even when optimization is disabled.  This is the
    # smallest deterministic rejection path and guards against an over-cap
    # env-first fallback being returned silently.
    capped = _P0P1Planning.plan_contraction(spec, (A, B);
                                              optimize=false,
                                              scalar_type=ComplexF64,
                                              memory_cap_bytes=plan.live_peak_bytes)
    @test capped.live_peak_bytes <= plan.live_peak_bytes
    @test_throws ArgumentError _P0P1Planning.plan_contraction(
        spec, (A, B); optimize=false, scalar_type=ComplexF64,
        memory_cap_bytes=plan.live_peak_bytes - 1,
    )
    key_uncapped = _P0P1Planning.plan_key(:p0p1_memory, spec, (A, B), ComplexF64)
    key_capped = _P0P1Planning.plan_key(:p0p1_memory, spec, (A, B), ComplexF64;
                                         memory_cap_bytes=plan.live_peak_bytes)
    @test key_uncapped != key_capped
    legacy_key = _P0P1Planning.PlanKey(
        key_uncapped.kind, key_uncapped.sig, key_uncapped.shape, key_uncapped.T,
        true, 1, true,
    )
    @test legacy_key.memory_weight == 1.0
    @test_throws ArgumentError _P0P1Planning.plan_contraction(
        spec, (A, B); scalar_type=BigFloat, memory_cap_bytes=plan.live_peak_bytes,
    )

    plans = Dict{_P0P1Planning.PlanKey,_P0P1Planning.ContractionPlan}()
    cached, hit = _P0P1Planning.get_or_plan!(
        plans, :p0p1_memory, spec, (A, B), ComplexF64;
        optimize=false, memory_cap_bytes=plan.live_peak_bytes,
    )
    @test !hit
    @test cached.scalar_bytes == sizeof(ComplexF64)
    _, hit_again = _P0P1Planning.get_or_plan!(
        plans, :p0p1_memory, spec, (A, B), ComplexF64;
        optimize=false, memory_cap_bytes=plan.live_peak_bytes,
    )
    @test hit_again
    explicit_scalar_plans = Dict{_P0P1Planning.PlanKey,_P0P1Planning.ContractionPlan}()
    _, explicit_hit = _P0P1Planning.get_or_plan!(
        explicit_scalar_plans, :p0p1_memory, spec, (A, B), ComplexF64;
        optimize=false, memory_cap_bytes=plan.live_peak_bytes,
        scalar_type=ComplexF64,
    )
    @test !explicit_hit
    mismatch_scalar_plans = Dict{_P0P1Planning.PlanKey,_P0P1Planning.ContractionPlan}()
    @test_throws ArgumentError _P0P1Planning.get_or_plan!(
        mismatch_scalar_plans, :p0p1_memory, spec, (A, B), ComplexF64;
        scalar_type=Float64,
    )
end

@graft_extended_testset _p0p1_explicit(:m1) "P0/P1: sector-stored live memory" begin
    # This three-map U(1) chain is small enough for a direct A/B comparison,
    # while its block payload differs from the dense product.  It verifies
    # that the live-byte fields use stored sector payload rather than merely
    # copying dense element diagnostics.
    Aspace = U1Space(0 => 1, 1 => 1) ← U1Space(0 => 1, 1 => 2)
    Bspace = U1Space(0 => 1, 1 => 2) ← U1Space(0 => 1, 1 => 4)
    Cspace = U1Space(0 => 1, 1 => 4) ← U1Space(0 => 2, 1 => 1)
    spec = _P0P1Planning.ContractionSpec(
        Vector{Int}[[-1, 1], [1, 2], [2, -2]],
        Bool[false, false, false], 2, (1, 1), 1;
        preferred_slots=[2, 3],
    )
    plan = _P0P1Planning.plan_contraction(spec, (Aspace, Bspace, Cspace);
                                            scalar_type=ComplexF64,
                                            memory_weight=0,
                                            sector_aware=true)
    @test plan.sector_operand_bytes < plan.operand_bytes
    @test plan.sector_live_peak_bytes <= plan.live_peak_bytes
    @test plan.sector_known_temporary_peak_bytes <= plan.known_temporary_peak_bytes

    rng = MersenneTwister(20260711)
    A = randn(rng, ComplexF64, Aspace)
    B = randn(rng, ComplexF64, Bspace)
    C = randn(rng, ComplexF64, Cspace)
    result = _P0P1Planning.EffectiveMap(plan, (B, C))(A)
    reference = _P0P1Planning.ncon_reference(spec, A, (B, C))
    @test norm(result - reference) <= 1e-12 * max(norm(reference), 1)
end

@graft_testset "P0/P1: public effective-map hard cap" begin
    topo = mps_topology(2)
    S = spin_ops()
    phys = Dict(nodeid(topo, i) => S.P for i in 1:nnodes(topo))
    H = OpSum()
    for i in 1:nnodes(topo)
        H += Term(0.2 + 0.1i, SiteOp(nodeid(topo, i), :Z, S.Z))
    end
    O = ttno_from_opsum(H, topo, phys; hermitian=true)
    ψ = random_ttns(MersenneTwister(20260712), ComplexF64, topo, phys, ℂ^2)
    move_center!(ψ, topo.root)
    cache = EnvCache(topo)
    uncapped = eff_h1(cache, ψ, O, topo.root)
    capped = eff_h1(cache, ψ, O, topo.root;
                     memory_cap_bytes=uncapped.plan.live_peak_bytes)
    @test capped.plan.live_peak_bytes <= uncapped.plan.live_peak_bytes
    @test capped.plan.scalar_bytes == sizeof(ComplexF64)
    spec, statics, _ = Graft.Contractions._h1_spec(cache, ψ, O, topo.root)
    reference = _P0P1Planning.ncon_reference(spec, ψ.tensors[topo.root], statics)
    @test norm(capped(ψ.tensors[topo.root]) - reference) <=
          1e-12 * max(norm(reference), 1)
end

@graft_extended_testset _p0p1_explicit(:m1) "P0/P1: bounded hard-cap fallback" begin
    # Eleven tensors exceeds the exact-DP limit while remaining shape-only.
    # The intentionally reversed semantic order outer-products its two chain
    # ends; a cap below that env-first live peak must select an actual bounded
    # beam candidate rather than return the old one-tree greedy fallback.
    n = 11
    labels = Vector{Vector{Int}}([[-1, 1]])
    for i in 2:(n - 1)
        push!(labels, [i - 1, i])
    end
    push!(labels, [n - 1, -2])
    spec = _P0P1Planning.ContractionSpec(labels, falses(n), 2, (1, 1), 1;
                                          preferred_slots=collect(n:-1:2))
    protos = (ℂ^2 ← ℂ^8, ntuple(_ -> ℂ^8 ← ℂ^8, n - 2)..., ℂ^8 ← ℂ^2)
    envfirst = _P0P1Planning.plan_contraction(spec, protos;
                                                optimize=false,
                                                scalar_type=ComplexF64)
    dims, _ = _P0P1Planning._label_dimensions(spec, protos)
    beam_trees = _P0P1Planning._memory_beam_trees(spec, dims)
    @test !isempty(beam_trees)
    beam_live = minimum(
        max(plan.live_peak_bytes, plan.sector_live_peak_bytes)
        for plan in (_P0P1Planning._compile_plan(
            tree, spec, dims, protos;
            strategy=:memory_beam, structural_metrics=false,
            scalar_type=ComplexF64,
        ) for tree in beam_trees)
    )
    envfirst_live = max(envfirst.live_peak_bytes, envfirst.sector_live_peak_bytes)
    @test beam_live < envfirst_live
    cap = (beam_live + envfirst_live) / 2
    capped = _P0P1Planning.plan_contraction(
        spec, protos; scalar_type=ComplexF64,
        memory_cap_bytes=cap,
    )
    @test capped.strategy == :memory_beam
    @test max(capped.live_peak_bytes, capped.sector_live_peak_bytes) <= cap
    @test_throws ArgumentError _P0P1Planning.plan_contraction(
        spec, protos; scalar_type=ComplexF64, memory_cap_bytes=beam_live - 1,
    )
end
end # :m1 target

if _p0p1_enabled(:m2)
@graft_testset "P0/P1: default complete-tuple environment contracts" begin
    topo, _, O, ket, bra = _p0p1_sandwich_fixture(MersenneTwister(2026071301))
    root = topo.root
    child = only(topo.children[root])

    # Keep one exact edge layout, complete-tuple scalar closure, and immutable
    # root-cap reuse in the default tier: failures here are correctness or
    # retained-resource regressions rather than extended diagnostics.
    cache = EnvCache(topo)
    edge, _ = _p0p1_env_matches_ncon!(cache, ket, O, bra, root, child)
    @test numout(edge) == 3 && numin(edge) == 0
    @test length(cache.rootcaps) == 1
    rootcap = only(values(cache.rootcaps))
    env!(cache, ket, O, bra, child, root)
    scalar_spec, scalar_operands =
        Graft.Contractions._build_env_spec(cache, ket, O, bra, root, 0)
    scalar_plan = _P0P1Planning.plan_contraction(
        scalar_spec, scalar_operands;
        scalar_type=Graft.Backend.scalartype(ket.tensors[root]),
    )
    scalar_ref = Graft.Contractions._build_env_ncon_reference(
        ket, O, bra, root, 0, cache.envs,
    )
    @test scalar_plan.scalar_output
    @test _P0P1Planning.execute(scalar_plan, scalar_operands) ≈ scalar_ref
    @test_throws ArgumentError _P0P1Planning.execute(
        scalar_plan, scalar_operands[1:(end - 1)],
    )
    scalar, _ = _p0p1_env_matches_ncon!(cache, ket, O, bra, root, 0)
    @test scalar isa Number
    @test only(values(cache.rootcaps)) === rootcap

    effcache = EnvCache(topo)
    move_center!(ket, root; cache=effcache)
    h1a = eff_h1(effcache, ket, O, root)
    h1b = eff_h1(effcache, ket, O, root)
    @test h1a.statics[end] === h1b.statics[end]
    @test length(effcache.rootcaps) == 1
end

@graft_extended_testset _p0p1_explicit(:m2) "P0/P1: planned environments and scalar contractions" begin
    topo, S, O, ket, bra = _p0p1_sandwich_fixture(MersenneTwister(20260713))
    root = topo.root
    child = only(topo.children[root])

    # Ket-bra edge environments retain their two all-codomain legs in both
    # directions. The private reference is the unmodified ncon bookkeeping.
    c0 = EnvCache(topo)
    below0, _ = _p0p1_env_matches_ncon!(c0, ket, nothing, bra, child, root)
    @test numout(below0) == 2 && numin(below0) == 0
    above0, _ = _p0p1_env_matches_ncon!(c0, ket, nothing, bra, root, child)
    @test numout(above0) == 2 && numin(above0) == 0

    # Ket-TTNO-bra environments preserve the positional (ket, op, bra) leg
    # order, including the root cap. The cap is shape-owned and reused.
    cH = EnvCache(topo)
    belowH, _ = _p0p1_env_matches_ncon!(cH, ket, O, bra, child, root)
    @test numout(belowH) == 3 && numin(belowH) == 0
    aboveH, _ = _p0p1_env_matches_ncon!(cH, ket, O, bra, root, child)
    @test numout(aboveH) == 3 && numin(aboveH) == 0
    @test length(cH.rootcaps) == 1
    rootcap = only(values(cH.rootcaps))

    # Root caps are also hoisted for Krylov maps. Reconstructing the same
    # root h1 closure must reuse its immutable cap by exact shape.
    effψ = copy(ket)
    effcache = EnvCache(topo)
    move_center!(effψ, root; cache=effcache)
    h1a = eff_h1(effcache, effψ, O, root)
    h1b = eff_h1(effcache, effψ, O, root)
    @test h1a.statics[end] === h1b.statics[end]
    @test length(effcache.rootcaps) == 1

    # A complete tuple plan scalarizes only its final rank-zero TensorMap;
    # exact operand count is part of the generic executor contract.
    env!(cH, ket, O, bra, child, root)
    scalar_spec, scalar_operands =
        Graft.Contractions._build_env_spec(cH, ket, O, bra, root, 0)
    scalar_plan = _P0P1Planning.plan_contraction(
        scalar_spec, scalar_operands;
        scalar_type=Graft.Backend.scalartype(ket.tensors[root]),
    )
    scalar_ref = Graft.Contractions._build_env_ncon_reference(ket, O, bra,
                                                               root, 0, cH.envs)
    @test scalar_plan.scalar_output
    @test _P0P1Planning.execute(scalar_plan, scalar_operands) ≈ scalar_ref
    @test_throws ArgumentError _P0P1Planning.execute(
        scalar_plan, scalar_operands[1:(end - 1)],
    )
    legacy_layout = _P0P1Planning.ContractionPlan(
        scalar_plan.nslots, scalar_plan.output_slot, scalar_plan.steps,
        scalar_plan.strategy, scalar_plan.flops, scalar_plan.peak_elements,
        scalar_plan.sector_flops, scalar_plan.sector_peak_elements,
        scalar_plan.sector_peak_block_elements, scalar_plan.scalar_bytes,
        scalar_plan.operand_bytes, scalar_plan.live_peak_bytes,
        scalar_plan.known_temporary_peak_bytes,
        scalar_plan.known_permutation_peak_bytes,
        scalar_plan.sector_operand_bytes, scalar_plan.sector_live_peak_bytes,
        scalar_plan.sector_known_temporary_peak_bytes,
        scalar_plan.sector_known_permutation_peak_bytes,
    )
    @test !legacy_layout.scalar_output
    root_scalar, _ = _p0p1_env_matches_ncon!(cH, ket, O, bra, root, 0)
    @test root_scalar isa Number
    @test only(values(cH.rootcaps)) === rootcap

    # Closing at a non-root center has no cap and must still scalarize.
    env!(cH, ket, O, bra, root, child)
    child_scalar, _ = _p0p1_env_matches_ncon!(cH, ket, O, bra, child, 0)
    @test child_scalar isa Number

    # Changing a same-shape center tensor invalidates only value environments;
    # shape plans and caps remain reusable and rebuild against fresh values.
    move_center!(ket, child; cache=cH)
    plan_count = length(cH.plans)
    cap_count = length(cH.rootcaps)
    update_tensor!(ket, child, 1.01 * ket.tensors[child]; caches=(cH,))
    @test length(cH.plans) == plan_count
    @test length(cH.rootcaps) == cap_count
    env!(cH, ket, O, bra, child, root)
    rebuilt, _ = _p0p1_env_matches_ncon!(cH, ket, O, bra, root, 0)
    @test rebuilt isa Number

    # `inner`, TTNO `expect`, and local operator expectation all execute the
    # same complete-tuple planner path. Gauge movement exercises v=0 at both
    # the root and a non-root orthogonality center.
    cinner = EnvCache(topo)
    for w in neighbors(topo, root)
        env!(cinner, ket, nothing, bra, w, root)
    end
    inner_ref = Graft.Contractions._build_env_ncon_reference(ket, nothing, bra,
                                                              root, 0, cinner.envs)
    @test inner(bra, ket) ≈ inner_ref rtol=1e-12 atol=1e-12

    # Repeated unrelated overlaps may share shape plans, but never value
    # environments. The low-latency diagnostic path uses only env-first plans
    # and must not mutate the lending cache except for its plan dictionary.
    cached_envs = copy(cinner.envs)
    cached_rootcaps = copy(cinner.rootcaps)
    value_counters = (cinner.env_hits, cinner.env_misses, cinner.env_rebuilds,
                      cinner.env_evictions)
    pooled_inner = inner(bra, ket; plan_cache=cinner, optimize=false)
    @test pooled_inner ≈ inner_ref rtol=1e-12 atol=1e-12
    @test keys(cinner.envs) == keys(cached_envs)
    @test all(cinner.envs[key] === value for (key, value) in cached_envs)
    @test keys(cinner.rootcaps) == keys(cached_rootcaps)
    @test all(cinner.rootcaps[key] === value for (key, value) in cached_rootcaps)
    @test (cinner.env_hits, cinner.env_misses, cinner.env_rebuilds,
           cinner.env_evictions) == value_counters

    envfirst_keys = Set(key for key in keys(cinner.plans)
                        if key.kind === :env_ket_bra && !key.optimize)
    @test !isempty(envfirst_keys)
    @test all(cinner.plans[key].strategy === :env_first for key in envfirst_keys)
    envfirst_plans = Dict(key => cinner.plans[key] for key in envfirst_keys)

    scale = 1.01 + 0.02im
    scaled_ket = copy(ket)
    update_tensor!(scaled_ket, root, scale * scaled_ket.tensors[root]; gauge=false)
    @test inner(bra, scaled_ket; plan_cache=cinner, optimize=false) ≈
          scale * inner_ref rtol=1e-12 atol=1e-12
    @test Set(key for key in keys(cinner.plans)
              if key.kind === :env_ket_bra && !key.optimize) == envfirst_keys
    @test all(cinner.plans[key] === plan for (key, plan) in envfirst_plans)
    @test_throws ArgumentError inner(
        bra, ket; plan_cache=EnvCache(mps_topology(3)), optimize=false,
    )

    ψ = copy(ket)
    cexpect = EnvCache(topo)
    move_center!(ψ, root; cache=cexpect)
    e_root = expect(ψ, O; cache=cexpect)
    local_root = expect(ψ, S.Z, nodeid(topo, root); cache=cexpect)
    local_ref = Graft.Contractions._local_expect_ncon_reference(ψ, S.Z, root)
    @test local_root ≈ local_ref rtol=1e-12 atol=1e-12
    move_center!(ψ, child; cache=cexpect)
    e_child = expect(ψ, O; cache=cexpect)
    @test e_child ≈ e_root rtol=1e-12 atol=1e-12

    # A four-node fork has a branching root and a non-root internal tensor
    # with both a parent and child. Cover both directed cuts and its closed
    # scalar contraction against the independent legacy ncon path.
    ftopo = fork_topology(2, 1)
    fphys = Dict(nodeid(ftopo, i) => S.P for i in 1:nnodes(ftopo))
    fH = OpSum()
    for i in 1:nnodes(ftopo)
        fH += Term(0.09 + 0.02i, SiteOp(nodeid(ftopo, i), :Z, S.Z))
    end
    fO = ttno_from_opsum(fH, ftopo, fphys; hermitian=true)
    fket = random_ttns(MersenneTwister(20260715), ComplexF64, ftopo, fphys, ℂ^2)
    fbra = random_ttns(MersenneTwister(20260716), ComplexF64, ftopo, fphys, ℂ^2)
    finternal = only(filter(i -> !isempty(ftopo.children[i]), ftopo.children[ftopo.root]))
    fleaf = only(ftopo.children[finternal])
    fcache = EnvCache(ftopo)
    fup, _ = _p0p1_env_matches_ncon!(fcache, fket, fO, fbra,
                                      finternal, ftopo.root)
    fdown, _ = _p0p1_env_matches_ncon!(fcache, fket, fO, fbra,
                                        finternal, fleaf)
    fscalar, _ = _p0p1_env_matches_ncon!(fcache, fket, fO, fbra,
                                          finternal, 0)
    @test numout(fup) == 3 && numout(fdown) == 3
    @test fscalar isa Number

    # A small abelian sector fixture covers stored-block plans and exact
    # scalarization without materializing any dense reference.
    utopo, _, uO, uket, ubra = _p0p1_sandwich_fixture(MersenneTwister(20260714);
                                                       u1=true)
    uroot = utopo.root
    uchild = only(utopo.children[uroot])
    ucache = EnvCache(utopo)
    uedge, _ = _p0p1_env_matches_ncon!(ucache, uket, uO, ubra, uchild, uroot)
    @test numout(uedge) == 3 && numin(uedge) == 0
    uscalar, _ = _p0p1_env_matches_ncon!(ucache, uket, uO, ubra, uroot, 0)
    @test uscalar isa Number
end
end # :m2 target

if _p0p1_enabled(:m3)
@graft_extended_testset _p0p1_explicit(:m3) "P0/P1: planned variational fit contractions" begin
    topo, _, _, ψ, φ = _p0p1_sandwich_fixture(MersenneTwister(20260717))
    root = topo.root
    child = only(topo.children[root])
    F = Graft.Networks

    # Both directions of the ket-bra fit environment use a fresh retained
    # ncon recursion, so this does not compare a planned value with itself.
    cache = F._FitCache(topo)
    for (u, v) in ((child, root), (root, child))
        got = F._fit_env!(cache, φ, ψ, u, v)
        refenvs = Dict{Tuple{Int,Int},Graft.Backend.AbstractTensorMap}()
        ref = _p0p1_fit_ref_env!(φ, ψ, u, v, refenvs)
        _p0p1_fit_maps_match(got, ref)
    end

    # Local projection and the scalar overlap are complete-operand plans;
    # their named direct paths remain private A/B references.
    got_project = F._fit_local_tensor(cache, φ, ψ, root)
    refenvs = Dict{Tuple{Int,Int},Graft.Backend.AbstractTensorMap}()
    for w in neighbors(topo, root)
        _p0p1_fit_ref_env!(φ, ψ, w, root, refenvs)
    end
    ref_project = F._fit_project_tensor_ncon_reference(φ, ψ, root, refenvs)
    _p0p1_fit_maps_match(got_project, ref_project)
    ref_overlap = F._fit_scalar_ncon_reference(φ, ψ, root, refenvs)
    @test F._fit_overlap(φ, ψ) ≈ ref_overlap rtol=1e-12 atol=1e-12

    # Gauge moves must not alter the planned overlap.  This covers root and
    # non-root scalar closure layouts without materializing a dense state.
    φg, ψg = copy(φ), copy(ψ)
    move_center!(φg, child)
    move_center!(ψg, child)
    @test F._fit_overlap(φg, ψg) ≈ ref_overlap rtol=1e-12 atol=1e-12

    # Value invalidation preserves independent shape plans and immutable caps.
    # Rebuilding after a same-shape target update must use fresh values.
    plan_count = length(cache.plans)
    cap_count = length(cache.rootcaps)
    cap = only(values(cache.rootcaps))
    @test haskey(cache.envs, (root, child))
    update_tensor!(φ, root, 1.01 * φ.tensors[root])
    F._invalidate_fit_node!(cache, root)
    @test !haskey(cache.envs, (root, child))
    @test length(cache.plans) == plan_count
    @test length(cache.rootcaps) == cap_count
    @test only(values(cache.rootcaps)) === cap
    rebuilt = F._fit_env!(cache, φ, ψ, root, child)
    rebuilt_refs = Dict{Tuple{Int,Int},Graft.Backend.AbstractTensorMap}()
    rebuilt_ref = _p0p1_fit_ref_env!(φ, ψ, root, child, rebuilt_refs)
    _p0p1_fit_maps_match(rebuilt, rebuilt_ref)
    @test length(cache.plans) == plan_count

    # Two source projections accumulate through one caller-owned destination;
    # the result agrees with the retained sum while not aliasing either source.
    target = copy(φ)
    src2 = copy(φ)
    sources = (ψ, src2)
    coeffs = ComplexF64[0.7 - 0.2im, -0.3 + 0.5im]
    got_sum = F._fit_local_tensor([F._FitCache(topo), F._FitCache(topo)],
                                  target, sources, coeffs, root)
    refsum = nothing
    for (src, α) in zip(sources, coeffs)
        srcenvs = Dict{Tuple{Int,Int},Graft.Backend.AbstractTensorMap}()
        for w in neighbors(topo, root)
            _p0p1_fit_ref_env!(target, src, w, root, srcenvs)
        end
        localref = F._fit_project_tensor_ncon_reference(target, src, root, srcenvs)
        refsum = refsum === nothing ? α * localref : refsum + α * localref
    end
    _p0p1_fit_maps_match(got_sum, refsum)
    @test got_sum !== ψ.tensors[root]
    @test got_sum !== src2.tensors[root]

    # A one-sweep public fit touches planned environments, local projections,
    # and repeated scalar error evaluations together.
    fitted = copy(target)
    _, errors = fit!(fitted, sources; coeffs, nsweeps=1)
    @test length(errors) == 1 && isfinite(only(errors))
    @test check_arrows(fitted)
    @test center(fitted) == center(target)

    # Small abelian-sector A/Bs retain TensorKit's stored block structure.
    utopo, _, _, uψ, uφ = _p0p1_sandwich_fixture(MersenneTwister(20260718);
                                                  u1=true)
    uroot = utopo.root
    uchild = only(utopo.children[uroot])
    ucache = F._FitCache(utopo)
    uenv = F._fit_env!(ucache, uφ, uψ, uchild, uroot)
    urefs = Dict{Tuple{Int,Int},Graft.Backend.AbstractTensorMap}()
    uenv_ref = _p0p1_fit_ref_env!(uφ, uψ, uchild, uroot, urefs)
    _p0p1_fit_maps_match(uenv, uenv_ref)
    uproject = F._fit_local_tensor(ucache, uφ, uψ, uroot)
    upro_refs = Dict{Tuple{Int,Int},Graft.Backend.AbstractTensorMap}()
    for w in neighbors(utopo, uroot)
        _p0p1_fit_ref_env!(uφ, uψ, w, uroot, upro_refs)
    end
    uproject_ref = F._fit_project_tensor_ncon_reference(uφ, uψ, uroot, upro_refs)
    _p0p1_fit_maps_match(uproject, uproject_ref)
    uscalar_ref = F._fit_scalar_ncon_reference(uφ, uψ, uroot, upro_refs)
    @test F._fit_overlap(uφ, uψ) ≈ uscalar_ref rtol=1e-12 atol=1e-12
end
end # :m3 target

if _p0p1_enabled(:m4)
@graft_testset "P0/P1: default direct operator-fit contracts" begin
    topo, S, O, ψ, φ = _p0p1_sandwich_fixture(MersenneTwister(2026071901))
    root = topo.root
    F = Graft.Networks

    # Non-Hermitian left/right order and mixed identity/operator actions feed
    # the exact residual. Keep these unique correctness paths in the default
    # tier even though extended mode covers additional gauges, sectors, and
    # multi-operator permutations.
    Hnh = OpSum()
    Hnh += Term(0.31 + 0.17im, SiteOp(nodeid(topo, root), :X, S.X))
    Onh = ttno_from_opsum(
        Hnh, topo, Dict(nodeid(topo, i) => S.P for i in 1:nnodes(topo));
        hermitian=false,
    )
    direct_double = F._fit_double_overlap(ψ, O, φ, Onh)
    legacy_double = inner(apply(O, ψ), apply(Onh, φ))
    @test direct_double ≈ legacy_double rtol=1e-12 atol=1e-12

    src2 = copy(φ)
    coeffs = ComplexF64[0.6 - 0.1im, -0.2 + 0.4im]
    direct, legacy = copy(φ), copy(φ)
    legacy_sources = (ψ, apply(Onh, src2; center=center(legacy)))
    _, legacy_errors = fit!(legacy, legacy_sources; coeffs, nsweeps=1, tol=0.0)
    _, direct_errors = fit!(direct, (ψ, src2);
                            Hs=(nothing, Onh), coeffs, nsweeps=1, tol=0.0)
    @test direct_errors ≈ legacy_errors rtol=1e-10 atol=2e-8
    @test norm(to_dense(direct) - to_dense(legacy)) <= 1e-10
    @test check_arrows(direct) && center(direct) == center(φ)
end

@graft_extended_testset _p0p1_explicit(:m4) "P0/P1: direct operator-aware fit contractions" begin
    topo, S, O, ψ, φ = _p0p1_sandwich_fixture(MersenneTwister(20260719))
    root = topo.root
    child = only(topo.children[root])
    F = Graft.Networks

    # The planned rank-three cache keeps exact (source ket, operator, target
    # bra) edge order and agrees with an independent retained ncon recursion.
    cache = F._FitCache(topo, O)
    for (u, v) in ((child, root), (root, child))
        got = F._fit_env!(cache, φ, ψ, u, v)
        refs = Dict{Tuple{Int,Int},Graft.Backend.AbstractTensorMap}()
        ref = _p0p1_fit_operator_ref_env!(φ, ψ, O, u, v, refs)
        _p0p1_fit_maps_match(got, ref)
    end
    got_project = F._fit_local_tensor(cache, φ, ψ, root)
    project_refs = Dict{Tuple{Int,Int},Graft.Backend.AbstractTensorMap}()
    for w in neighbors(topo, root)
        _p0p1_fit_operator_ref_env!(φ, ψ, O, w, root, project_refs)
    end
    project_ref = F._fit_operator_project_tensor_ncon_reference(φ, ψ, O, root,
                                                                  project_refs)
    _p0p1_fit_maps_match(got_project, project_ref)
    scalar_ref = F._fit_operator_scalar_ncon_reference(φ, ψ, O, root, project_refs)
    direct_scalar = F._fit_operator_overlap(φ, O, ψ)
    @test direct_scalar ≈ scalar_ref rtol=1e-12 atol=1e-12
    @test direct_scalar ≈ inner(φ, apply(O, ψ; center=center(φ))) rtol=1e-12 atol=1e-12

    # Target gauge changes leave the direct sandwich invariant, while a
    # same-shape update drops only value environments and retains its plans.
    φg, ψg = copy(φ), copy(ψ)
    move_center!(φg, child)
    move_center!(ψg, child)
    @test F._fit_operator_overlap(φg, O, ψg) ≈ direct_scalar rtol=1e-12 atol=1e-12
    cache_target = copy(φ)
    cache_check = F._FitCache(topo, O)
    F._fit_env!(cache_check, cache_target, ψ, root, child)
    plan_count = length(cache_check.plans)
    cap_count = length(cache_check.rootcaps)
    cap = only(values(cache_check.rootcaps))
    update_tensor!(cache_target, root, 1.01 * cache_target.tensors[root])
    F._invalidate_fit_node!(cache_check, root)
    @test !haskey(cache_check.envs, (root, child))
    @test length(cache_check.plans) == plan_count
    @test length(cache_check.rootcaps) == cap_count
    @test only(values(cache_check.rootcaps)) === cap
    rebuilt = F._fit_env!(cache_check, cache_target, ψ, root, child)
    rebuilt_refs = Dict{Tuple{Int,Int},Graft.Backend.AbstractTensorMap}()
    rebuilt_ref = _p0p1_fit_operator_ref_env!(cache_target, ψ, O, root, child,
                                               rebuilt_refs)
    _p0p1_fit_maps_match(rebuilt, rebuilt_ref)

    # A rank-four direct residual term preserves non-Hermitian left/right
    # operator order instead of replacing H†H with H².
    Hnh = OpSum()
    Hnh += Term(0.31 + 0.17im, SiteOp(nodeid(topo, root), :X, S.X))
    Onh = ttno_from_opsum(Hnh, topo,
                           Dict(nodeid(topo, i) => S.P for i in 1:nnodes(topo));
                           hermitian=false)
    direct_double = F._fit_double_overlap(ψ, O, φ, Onh)
    legacy_double = inner(apply(O, ψ), apply(Onh, φ))
    @test direct_double ≈ legacy_double rtol=1e-12 atol=1e-12

    # The public Hs= form has the same result and default exact residual as
    # materialize-then-fit on this tiny tree, without retaining that target.
    direct_fit, legacy_fit = copy(φ), copy(φ)
    legacy_target = apply(O, ψ; center=center(legacy_fit))
    _, legacy_errors = fit!(legacy_fit, legacy_target; nsweeps=1, tol=0.0)
    _, direct_errors = fit!(direct_fit, (ψ,); Hs=(O,), nsweeps=1, tol=0.0)
    # Independent squared-norm accumulations can differ at sqrt(eps) near zero.
    @test direct_errors ≈ legacy_errors rtol=1e-10 atol=2e-8
    @test norm(to_dense(direct_fit) - to_dense(legacy_fit)) <= 1e-10
    @test check_arrows(direct_fit) && center(direct_fit) == center(φ)

    # A two-source non-Hermitian target exercises every double-layer pair in
    # the default error metric, including i != j operator order.
    src2 = copy(φ)
    coeffs = ComplexF64[0.6 - 0.1im, -0.2 + 0.4im]
    direct_multi, legacy_multi = copy(φ), copy(φ)
    legacy_sources = (apply(O, ψ; center=center(legacy_multi)),
                      apply(Onh, src2; center=center(legacy_multi)))
    _, legacy_multi_errors = fit!(legacy_multi, legacy_sources; coeffs, nsweeps=1, tol=0.0)
    _, direct_multi_errors = fit!(direct_multi, (ψ, src2); Hs=(O, Onh), coeffs,
                                  nsweeps=1, tol=0.0)
    @test direct_multi_errors ≈ legacy_multi_errors rtol=1e-10 atol=2e-8
    @test norm(to_dense(direct_multi) - to_dense(legacy_multi)) <= 1e-10

    # `nothing` remains a first-class optional action: mixed identity/operator
    # sources agree with the historical materialize-then-fit formulation.
    direct_optional, legacy_optional = copy(φ), copy(φ)
    legacy_optional_sources = (ψ, apply(O, src2; center=center(legacy_optional)))
    _, legacy_optional_errors = fit!(legacy_optional, legacy_optional_sources;
                                     coeffs, nsweeps=1, tol=0.0)
    _, direct_optional_errors = fit!(direct_optional, (ψ, src2);
                                     Hs=(nothing, O), coeffs, nsweeps=1, tol=0.0)
    # Both formulas evaluate the exact residual; independent direct
    # double-layer accumulation can leave a roundoff-scale positive remainder
    # where materializing the same tiny target cancels to zero.
    @test direct_optional_errors ≈ legacy_optional_errors rtol=1e-10 atol=2e-8
    @test norm(to_dense(direct_optional) - to_dense(legacy_optional)) <= 1e-10

    # A small abelian sector fixture checks the operator-aware labels/caps
    # without using a dense reference.
    utopo, _, uO, uψ, uφ = _p0p1_sandwich_fixture(MersenneTwister(20260720);
                                                   u1=true)
    uroot = utopo.root
    uchild = only(utopo.children[uroot])
    ucache = F._FitCache(utopo, uO)
    uenv = F._fit_env!(ucache, uφ, uψ, uchild, uroot)
    urefs = Dict{Tuple{Int,Int},Graft.Backend.AbstractTensorMap}()
    uenv_ref = _p0p1_fit_operator_ref_env!(uφ, uψ, uO, uchild, uroot, urefs)
    _p0p1_fit_maps_match(uenv, uenv_ref)
    uproject = F._fit_local_tensor(ucache, uφ, uψ, uroot)
    upro_refs = Dict{Tuple{Int,Int},Graft.Backend.AbstractTensorMap}()
    for w in neighbors(utopo, uroot)
        _p0p1_fit_operator_ref_env!(uφ, uψ, uO, w, uroot, upro_refs)
    end
    uproject_ref = F._fit_operator_project_tensor_ncon_reference(uφ, uψ, uO,
                                                                   uroot, upro_refs)
    _p0p1_fit_maps_match(uproject, uproject_ref)
    uscalar_ref = F._fit_operator_scalar_ncon_reference(uφ, uψ, uO, uroot,
                                                         upro_refs)
    @test F._fit_operator_overlap(uφ, uO, uψ) ≈ uscalar_ref rtol=1e-12 atol=1e-12
    @test F._fit_operator_overlap(uφ, uO, uψ) ≈
          inner(uφ, apply(uO, uψ; center=center(uφ))) rtol=1e-12 atol=1e-12

    # GlobalKrylov and linsolve! both use the same _GKOperator matvec.  Check
    # it directly against the retained legacy materialize-then-fit route, then
    # execute one deliberately tiny shifted solve smoke path.
    E = Graft.Evolution
    gktemplate = copy(φ)
    gkop = E._GKOperator(O, gktemplate, 1, 0.0, false)
    gkx = E._GKState(copy(ψ), gktemplate, 1, 0.0, false)
    gk_direct = gkop(gkx).ψ
    gk_legacy = copy(gktemplate)
    fit!(gk_legacy, apply(O, ψ; center=center(gk_legacy)); nsweeps=1, tol=0.0)
    @test norm(to_dense(gk_direct) - to_dense(gk_legacy)) <= 1e-10
    solve_state = copy(φ)
    _, solve_info = linsolve!(solve_state, O, copy(ψ); a0=1.0, a1=0.1,
                               krylovdim=2, maxiter=1, tol=1e-6,
                               fit_nsweeps=1, fit_tol=0.0)
    @test check_arrows(solve_state)
    @test solve_info.numops >= 1
end
end # :m4 target

if _p0p1_enabled(:m5)
@graft_testset "P0/P1: default exact planned-apply contracts" begin
    topo, S, O, ψ, _ = _p0p1_sandwich_fixture(MersenneTwister(2026072101))
    child = only(topo.children[topo.root])
    F = Graft.Networks

    planned = apply(O, ψ; center=child)
    envfirst = apply(O, ψ; center=child, optimize=false)
    reference = F._apply_ncon_reference(O, ψ; center=child)
    @test check_arrows(planned) && center(planned) == child
    @test norm(to_dense(planned) - to_dense(reference)) <= 1e-12
    @test check_arrows(envfirst) && center(envfirst) == child
    @test norm(to_dense(envfirst) - to_dense(reference)) <= 1e-12

    # Retain a stored-sector application and the non-root physical-leg-free
    # junction in default coverage: their fusion/arrow layouts have no generic
    # dense fallback that would make a silent planner regression harmless.
    utopo, _, uO, uψ, _ = _p0p1_sandwich_fixture(MersenneTwister(2026072201);
                                                  u1=true)
    uchild = only(utopo.children[utopo.root])
    uplanned = apply(uO, uψ; center=uchild)
    ureference = F._apply_ncon_reference(uO, uψ; center=uchild)
    @test check_arrows(uplanned) && center(uplanned) == uchild
    @test norm(to_dense(uplanned) - to_dense(ureference)) <= 1e-12

    jtopo = TreeTopology(:root, [:root => :junction, :junction => :left,
                                  :junction => :right])
    jphys = Dict(:root => S.P, :left => S.P, :right => S.P)
    jH = OpSum()
    for site in (:root, :left, :right)
        jH += Term(0.11 + 0.02im, SiteOp(site, :Z, S.Z))
    end
    jO = ttno_from_opsum(jH, jtopo, jphys; hermitian=false)
    jψ = random_ttns(MersenneTwister(2026072401), ComplexF64, jtopo, jphys, ℂ^2)
    jfusions = F._apply_edge_fusions(promote_type(eltype(jO), eltype(jψ)), jψ, jO)
    junction = nodeindex(jtopo, :junction)
    jspec, _ = F._apply_node_spec(jO, jψ, junction, jfusions)
    @test jspec.preferred_slots == [1, 3, 4, 2, 5]
    jout = apply(jO, jψ; center=junction)
    jreference = F._apply_ncon_reference(jO, jψ; center=junction)
    @test check_arrows(jout) && center(jout) == junction
    @test norm(to_dense(jout) - to_dense(jreference)) <= 1e-12
end

@graft_extended_testset _p0p1_explicit(:m5) "P0/P1: planned exact TTNO application" begin
    topo, S, O, ψ, _ = _p0p1_sandwich_fixture(MersenneTwister(20260721))
    root = topo.root
    child = only(topo.children[root])
    F = Graft.Networks
    T = promote_type(eltype(O), eltype(ψ))

    # A node plan preserves the historical operand/open-leg layout while using
    # the exact prebuilt child fusion object rather than rebuilding it.
    fusions = F._apply_edge_fusions(T, ψ, O)
    plans = Dict{_P0P1Planning.PlanKey,_P0P1Planning.ContractionPlan}()
    spec, operands = F._apply_node_spec(O, ψ, root, fusions)
    @test operands[3] === fusions[child]
    planned_node = F._apply_node_tensor(plans, O, ψ, root, fusions)
    legacy_node = F._apply_node_tensor_ncon_reference(O, ψ, root, fusions[root])
    _p0p1_fit_maps_match(planned_node, legacy_node)
    plan_count = length(plans)
    repeat_node = F._apply_node_tensor(plans, O, ψ, root, fusions)
    _p0p1_fit_maps_match(repeat_node, legacy_node)
    @test length(plans) == plan_count

    # Full application retains exact output and requested canonical center.
    planned = apply(O, ψ; center=child)
    legacy = F._apply_ncon_reference(O, ψ; center=child)
    @test check_arrows(planned) && center(planned) == child
    @test norm(to_dense(planned) - to_dense(legacy)) <= 1e-12

    # A small U(1) fixture validates fusion and arrow conventions in sectors.
    utopo, _, uO, uψ, _ = _p0p1_sandwich_fixture(MersenneTwister(20260722);
                                                  u1=true)
    uchild = only(utopo.children[utopo.root])
    uplanned = apply(uO, uψ; center=uchild)
    ulegacy = F._apply_ncon_reference(uO, uψ; center=uchild)
    @test check_arrows(uplanned) && center(uplanned) == uchild
    @test norm(to_dense(uplanned) - to_dense(ulegacy)) <= 1e-12

    # A three-node star exercises two distinct child fusion maps at one node
    # and mirrors the preorder lifetime release used by public apply.
    stopo = star_topology(2, 1)
    sphys = Dict(nodeid(stopo, i) => S.P for i in 1:nnodes(stopo))
    sH = OpSum()
    for i in 1:nnodes(stopo)
        sH += Term(0.08 + 0.03im * i, SiteOp(nodeid(stopo, i), :Z, S.Z))
    end
    sO = ttno_from_opsum(sH, stopo, sphys; hermitian=false)
    sψ = random_ttns(MersenneTwister(20260723), ComplexF64, stopo, sphys, ℂ^2)
    sT = promote_type(eltype(sO), eltype(sψ))
    sfusions = F._apply_edge_fusions(sT, sψ, sO)
    sroot = stopo.root
    sspec, soperands = F._apply_node_spec(sO, sψ, sroot, sfusions)
    schildren = stopo.children[sroot]
    @test soperands[3] === sfusions[schildren[1]]
    @test soperands[4] === sfusions[schildren[2]]
    splans = Dict{_P0P1Planning.PlanKey,_P0P1Planning.ContractionPlan}()
    sgot = F._apply_node_tensor(splans, sO, sψ, sroot, sfusions)
    sref = F._apply_node_tensor_ncon_reference(sO, sψ, sroot, sfusions[sroot])
    _p0p1_fit_maps_match(sgot, sref)
    stensors = Vector{Graft.Backend.AbstractTensorMap}(undef, nnodes(stopo))
    for n in preorder(stopo)
        stensors[n] = F._apply_node_tensor(splans, sO, sψ, n, sfusions)
        delete!(sfusions, n)
    end
    @test isempty(sfusions)
    sout = TTNS(stopo, stensors, stopo.root)
    F._canonicalize_apply!(sout, schildren[1])
    slegacy = F._apply_ncon_reference(sO, sψ; center=schildren[1])
    @test check_arrows(sout) && center(sout) == schildren[1]
    @test norm(to_dense(sout) - to_dense(slegacy)) <= 1e-12

    # A non-root physless junction takes the special preferred fold through
    # child fusions before W, avoiding a disconnected state/operator product.
    jtopo = TreeTopology(:root, [:root => :junction, :junction => :left,
                                  :junction => :right])
    jphys = Dict(:root => S.P, :left => S.P, :right => S.P)
    jH = OpSum()
    for site in (:root, :left, :right)
        jH += Term(0.11 + 0.02im, SiteOp(site, :Z, S.Z))
    end
    jO = ttno_from_opsum(jH, jtopo, jphys; hermitian=false)
    jψ = random_ttns(MersenneTwister(20260724), ComplexF64, jtopo, jphys, ℂ^2)
    jT = promote_type(eltype(jO), eltype(jψ))
    jfusions = F._apply_edge_fusions(jT, jψ, jO)
    junction = nodeindex(jtopo, :junction)
    jspec, _ = F._apply_node_spec(jO, jψ, junction, jfusions)
    @test jspec.preferred_slots == [1, 3, 4, 2, 5]
    jplans = Dict{_P0P1Planning.PlanKey,_P0P1Planning.ContractionPlan}()
    jnode = F._apply_node_tensor(jplans, jO, jψ, junction, jfusions)
    jnode_ref = F._apply_node_tensor_ncon_reference(jO, jψ, junction,
                                                     jfusions[junction])
    _p0p1_fit_maps_match(jnode, jnode_ref)
    jout = apply(jO, jψ; center=junction)
    jlegacy = F._apply_ncon_reference(jO, jψ; center=junction)
    @test check_arrows(jout) && center(jout) == junction
    @test norm(to_dense(jout) - to_dense(jlegacy)) <= 1e-12
end
end # :m5 target

if _p0p1_enabled(:m6)
@graft_testset "P0/P1: default workspace ownership contracts" begin
    rng = MersenneTwister(2026072501)
    V = ℂ^2 ← ℂ^2
    spec = _P0P1Planning.ContractionSpec(
        Vector{Int}[[-1, 1], [1, 2], [2, 3], [3, 4], [4, -2]],
        falses(5), 2, (1, 1), 1; preferred_slots=[2, 3, 4, 5],
    )
    operands = ntuple(_ -> randn(rng, ComplexF64, V), 5)
    plan = _P0P1Planning.plan_contraction(
        spec, operands; optimize=false, scalar_type=ComplexF64,
    )
    workspace = _P0P1Planning.PlanWorkspace(plan)
    got = _P0P1Planning.execute(plan, operands; workspace)
    _p0p1_fit_maps_match(got, _P0P1Planning.ncon_reference(spec, operands))

    # An in-place destination may never alias a source TensorMap.
    step = plan.steps[1]
    pA = (step.oindA, step.cindA)
    pB = (step.cindB, step.oindB)
    @test_throws ArgumentError _P0P1Planning._workspace_contract!(
        workspace, operands[step.a], operands[step.a], pA, step.conjA,
        operands[step.b], pB, step.conjB, step.out,
    )

    # Mutable buffers bind to the first calling task and must reject reuse by
    # another task rather than racing on shared scratch storage.
    topo, _, O, ket, _ = _p0p1_sandwich_fixture(MersenneTwister(2026072601))
    root = topo.root
    cache = EnvCache(topo)
    move_center!(ket, root; cache)
    h1 = eff_h1(cache, ket, O, root)
    wrapped = _P0P1Planning.workspace_map(h1)
    x = ket.tensors[root]
    _p0p1_fit_maps_match(wrapped(x), h1(x))
    @test _P0P1Planning.workspace_stats(wrapped.workspace).owner_bound
    cross_task_error = fetch(@async begin
        try
            wrapped(x)
            nothing
        catch err
            err
        end
    end)
    @test cross_task_error isa ArgumentError
end

@graft_extended_testset _p0p1_explicit(:m6) "P0/P1: planned contraction workspaces" begin
    rng = MersenneTwister(20260725)
    V = ℂ^2 ← ℂ^2
    # The env-first fold has three internal outputs. Its first and third
    # lifetimes do not overlap and have the same HomSpace, so one color-owned
    # destination can safely serve both after the consuming step finishes.
    spec = _P0P1Planning.ContractionSpec(
        Vector{Int}[[-1, 1], [1, 2], [2, 3], [3, 4], [4, -2]],
        falses(5), 2, (1, 1), 1; preferred_slots=[2, 3, 4, 5],
    )
    operands = ntuple(_ -> randn(rng, ComplexF64, V), 5)
    plan = _P0P1Planning.plan_contraction(spec, operands;
                                            optimize=false,
                                            scalar_type=ComplexF64)
    @test length(plan.steps) == 4
    workspace = _P0P1Planning.PlanWorkspace(plan)
    first_slot, third_slot = plan.steps[1].dst, plan.steps[3].dst
    @test workspace.layout.colors[first_slot] == workspace.layout.colors[third_slot]
    @test workspace.layout.ncolors == 2

    reference = _P0P1Planning.ncon_reference(spec, operands)
    y1 = _P0P1Planning.execute(plan, operands; workspace)
    _p0p1_fit_maps_match(y1, reference)
    y1_snapshot = copy(y1)
    stats_first = _P0P1Planning.workspace_stats(workspace)
    @test stats_first.buffers == 2
    @test stats_first.allocations == 2
    @test stats_first.reuses >= 1

    changed = (1.07 * operands[1], operands[2:end]...)
    y2 = _P0P1Planning.execute(plan, changed; workspace)
    _p0p1_fit_maps_match(y2, _P0P1Planning.ncon_reference(spec, changed))
    @test y1 !== y2
    _p0p1_fit_maps_match(y1, y1_snapshot)
    stats_second = _P0P1Planning.workspace_stats(workspace)
    @test stats_second.allocations == stats_first.allocations
    @test stats_second.reuses > stats_first.reuses

    # Accumulation keeps its public destination caller-owned while reusing the
    # same strictly internal workspace destinations.
    dest = zero(y2)
    _P0P1Planning.execute_accumulate!(dest, plan, changed;
                                       α=1, β=0, workspace)
    _p0p1_fit_maps_match(dest, y2)
    @test dest !== y2

    # `tensorcontract!` must never receive an aliased source/destination.
    step = plan.steps[1]
    pA = (step.oindA, step.cindA)
    pB = (step.cindB, step.oindB)
    @test_throws ArgumentError _P0P1Planning._workspace_contract!(
        workspace, operands[step.a], operands[step.a], pA, step.conjA,
        operands[step.b], pB, step.conjB, step.out,
    )

    # EffectiveMap remains immutable/shared by default; the wrapper owns a
    # task-local workspace and nevertheless returns a fresh Krylov-safe root.
    topo, _, O, ket, _ = _p0p1_sandwich_fixture(MersenneTwister(20260726))
    root = topo.root
    cache = EnvCache(topo)
    move_center!(ket, root; cache)
    h1 = eff_h1(cache, ket, O, root)
    wrapped = _P0P1Planning.workspace_map(h1)
    x = ket.tensors[root]
    hx_ref = h1(x)
    hx1 = wrapped(x)
    hx1_snapshot = copy(hx1)
    hx2 = wrapped(0.93 * x)
    _p0p1_fit_maps_match(hx1, hx_ref)
    _p0p1_fit_maps_match(hx2, h1(0.93 * x))
    @test hx1 !== hx2
    _p0p1_fit_maps_match(hx1, hx1_snapshot)
    @test _P0P1Planning.workspace_stats(wrapped.workspace).owner_bound
    cross_task_error = fetch(@async begin
        try
            wrapped(x)
            nothing
        catch err
            err
        end
    end)
    @test cross_task_error isa ArgumentError

    # Ground-state local Krylov calls construct their own wrapper, proving the
    # solver integration keeps mutable workspace state out of EffectiveMap.
    solver_state = copy(ket)
    solved, energies = dmrg1!(solver_state, O; nsweeps=1, krylovdim=2,
                               verbose=false)
    @test solved === solver_state
    @test length(energies) == 1 && check_arrows(solver_state)

    # A small abelian effective-map fixture exercises TensorKit's block-backed
    # output storage through the same in-place workspace path.
    utopo, _, uO, uψ, _ = _p0p1_sandwich_fixture(MersenneTwister(20260727);
                                                  u1=true)
    uroot = utopo.root
    ucache = EnvCache(utopo)
    move_center!(uψ, uroot; cache=ucache)
    uh1 = eff_h1(ucache, uψ, uO, uroot)
    uwrapped = _P0P1Planning.workspace_map(uh1)
    ux = uψ.tensors[uroot]
    uref = uh1(ux)
    ugot = uwrapped(ux)
    _p0p1_fit_maps_match(ugot, uref)
    @test _P0P1Planning.workspace_stats(uwrapped.workspace).owner_bound
end
end # :m6 target

_p0p1_env_block_bytes(E) =
    sum(length(block_) * sizeof(eltype(block_)) for (_, block_) in Graft.Backend.blocks(E))

if _p0p1_enabled(:m7)
@graft_testset "P0/P1: bounded environment-cache memory" begin
    topo, S, O, ket, bra = _p0p1_sandwich_fixture(MersenneTwister(20260728))
    root = topo.root
    child = only(topo.children[root])
    below_key, above_key = (child, root), (root, child)

    # The default is still a full value cache, with payload based on actual
    # TensorKit blocks rather than dense leg products.
    full = EnvCache(topo)
    below = env!(full, ket, O, bra, child, root)
    above = env!(full, ket, O, bra, root, child)
    full_stats = env_cache_stats(full)
    full_bytes = sum(_p0p1_env_block_bytes(E) for E in values(full.envs))
    @test full_stats.payload_bytes == full_bytes
    @test full_stats.largest_entry_bytes ==
          maximum(_p0p1_env_block_bytes(E) for E in values(full.envs))
    @test full_stats.entry_count == 2
    @test full_stats.plan_count >= 1
    @test full_stats.hits == 0 && full_stats.misses == 2 && full_stats.rebuilds == 2
    @test full_stats.high_water_bytes == full_bytes
    @test full_stats.evictions == 0 && full_stats.max_env_bytes === nothing
    @test full_stats.eviction == :full
    @test env!(full, ket, O, bra, child, root) === below
    @test env_cache_stats(full).hits == 1
    _p0p1_fit_maps_match(below,
                          Graft.Contractions._build_env_ncon_reference(
                              ket, O, bra, child, root, full.envs))
    _p0p1_fit_maps_match(above,
                          Graft.Contractions._build_env_ncon_reference(
                              ket, O, bra, root, child, full.envs))

    # One-entry capacity makes the LRU victim deterministic: the older edge
    # falls out after the newer directed environment is returned to its caller.
    cap = full_stats.largest_entry_bytes
    bounded = EnvCache(topo; max_env_bytes=cap)
    bounded_below = env!(bounded, ket, O, bra, child, root)
    bounded_above = env!(bounded, ket, O, bra, root, child)
    _p0p1_fit_maps_match(bounded_below, below)
    _p0p1_fit_maps_match(bounded_above, above)
    bounded_stats = env_cache_stats(bounded)
    @test bounded_stats.payload_bytes <= cap
    @test bounded_stats.entry_count == 1 && bounded_stats.evictions == 1
    @test !haskey(bounded.envs, below_key) && haskey(bounded.envs, above_key)
    rebuilt_below = env!(bounded, ket, O, bra, child, root)
    _p0p1_fit_maps_match(rebuilt_below, below)
    @test haskey(bounded.envs, below_key) && !haskey(bounded.envs, above_key)
    @test env_cache_stats(bounded).rebuilds == 3
    @test env_cache_stats(bounded).misses == 3

    # A direct read refreshes recency: after inserting one more payload, the
    # untouched opposite edge is the deterministic LRU victim, not the key
    # just accessed again.
    lru = EnvCache(topo; max_env_bytes=full_stats.payload_bytes)
    env!(lru, ket, O, bra, child, root)
    env!(lru, ket, O, bra, root, child)
    env!(lru, ket, O, bra, child, root)
    synthetic_key = (99, 100)
    Graft.Contractions._with_env_transaction(lru) do
        Graft.Contractions._store_env!(lru, synthetic_key, copy(below))
    end
    @test haskey(lru.envs, below_key) && !haskey(lru.envs, above_key)
    @test haskey(lru.envs, synthetic_key)

    # Value invalidation releases environments but leaves shape-only plans and
    # root caps reusable across same-shape tensor changes and explicit edges.
    plans_before = collect(keys(bounded.plans))
    plan_count_before = length(bounded.plans)
    rootcap_count_before = length(bounded.rootcaps)
    @test rootcap_count_before >= 1
    update_tensor!(ket, child, 1.01 * ket.tensors[child]; gauge=false,
                   caches=(bounded,))
    @test isempty(bounded.envs)
    @test length(bounded.plans) == plan_count_before
    @test Set(keys(bounded.plans)) == Set(plans_before)
    @test length(bounded.rootcaps) == rootcap_count_before
    rebuilds_before = env_cache_stats(bounded).rebuilds
    updated_below = env!(bounded, ket, O, bra, child, root)
    updated_ref = Graft.Contractions._build_env_ncon_reference(
        ket, O, bra, child, root, bounded.envs,
    )
    _p0p1_fit_maps_match(updated_below, updated_ref)
    @test env_cache_stats(bounded).rebuilds == rebuilds_before + 1
    Graft.Contractions.invalidate_edge!(bounded, child, root)
    @test isempty(bounded.envs)
    @test length(bounded.plans) == plan_count_before
    @test length(bounded.rootcaps) == rootcap_count_before

    # A cap is enforced only after a recursive outer transaction. The root
    # needs its other leaf environment while it builds, so early eviction would
    # fail this cold branching contraction.
    stopo = star_topology(2, 1)
    sphys = Dict(nodeid(stopo, i) => S.P for i in 1:nnodes(stopo))
    sH = OpSum()
    for i in 1:nnodes(stopo)
        sH += Term(0.19 + 0.04im * i, SiteOp(nodeid(stopo, i), :Z, S.Z))
    end
    sO = ttno_from_opsum(sH, stopo, sphys; hermitian=false)
    sket = random_ttns(MersenneTwister(20260729), ComplexF64, stopo, sphys, ℂ^2)
    sbra = random_ttns(MersenneTwister(20260730), ComplexF64, stopo, sphys, ℂ^2)
    sroot = stopo.root
    sleaf = first(stopo.children[sroot])
    sfull = EnvCache(stopo)
    sref = env!(sfull, sket, sO, sbra, sroot, sleaf)
    scap = env_cache_stats(sfull).largest_entry_bytes
    sbounded = EnvCache(stopo; max_env_bytes=scap)
    sgot = env!(sbounded, sket, sO, sbra, sroot, sleaf)
    _p0p1_fit_maps_match(sgot, sref)
    @test env_cache_stats(sbounded).payload_bytes <= scap
    @test env_cache_stats(sbounded).evictions >= 1
    direct_bounded = EnvCache(stopo; max_env_bytes=scap)
    direct_got = Graft.Contractions.build_env(direct_bounded, sket, sO, sbra,
                                               sroot, sleaf)
    _p0p1_fit_maps_match(direct_got, sref)
    @test env_cache_stats(direct_bounded).payload_bytes <= scap
    full_expect = expect(sket, sO; cache=EnvCache(stopo))
    bounded_expect = expect(sket, sO; cache=EnvCache(stopo; max_env_bytes=scap))
    @test bounded_expect ≈ full_expect rtol=1e-12 atol=1e-12

    # Sector payload stats follow stored U(1) blocks, not a dense surrogate.
    utopo, _, uO, uket, ubra = _p0p1_sandwich_fixture(MersenneTwister(20260731);
                                                       u1=true)
    uroot = utopo.root
    uchild = only(utopo.children[uroot])
    ucache = EnvCache(utopo)
    uenv = env!(ucache, uket, uO, ubra, uchild, uroot)
    ubytes = _p0p1_env_block_bytes(uenv)
    ustats = env_cache_stats(ucache)
    @test ustats.payload_bytes == ubytes == ustats.largest_entry_bytes
    ubounded = EnvCache(utopo; max_env_bytes=ubytes)
    ubounded_env = env!(ubounded, uket, uO, ubra, uchild, uroot)
    _p0p1_fit_maps_match(ubounded_env, uenv)
    @test env_cache_stats(ubounded).payload_bytes <= ubytes
end
end # :m7 target
