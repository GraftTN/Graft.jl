# Targeted P0/P1 contraction-memory checks.  This file is intentionally
# self-contained so the constrained host can invoke it directly. `runtests.jl`
# includes it for stronger-host validation too; keep fixtures shape-only or
# two/three-map small so it never becomes a benchmark.
using Test
using Random
using LinearAlgebra
using GRAFT
using GRAFT.TestUtils: random_ttns
using GRAFT.Backend: ℂ, ⊗, ←, U1Space, norm, numout, numin
using GRAFT.Contractions: env!

const _P0P1Planning = GRAFT.Contractions.Planning
const _P0P1_TARGET = lowercase(get(ENV, "GRAFT_P0P1_TARGET", "all"))
_P0P1_TARGET in ("all", "m1", "m2", "m3", "m4", "m5", "m6", "m7") ||
    throw(ArgumentError("unknown GRAFT_P0P1_TARGET=$_P0P1_TARGET"))
_p0p1_enabled(group::Symbol) = _P0P1_TARGET in ("all", String(group))

@info "GRAFT P0/P1 targeted BLAS configuration" config=LinearAlgebra.BLAS.get_config() threads=LinearAlgebra.BLAS.get_num_threads()

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
    got = GRAFT.Contractions.build_env(cache, ket, O, bra, u, v)
    ref = GRAFT.Contractions._build_env_ncon_reference(ket, O, bra, u, v,
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
                             envs::Dict{Tuple{Int,Int},GRAFT.Backend.AbstractTensorMap})
    return get!(envs, (u, v)) do
        for w in neighbors(topology(φ), u)
            w == v || _p0p1_fit_ref_env!(φ, ψ, w, u, envs)
        end
        GRAFT.Networks._fit_build_env_ncon_reference(φ, ψ, u, v, envs)
    end
end

function _p0p1_fit_maps_match(got, ref; rtol=1e-12, atol=1e-12)
    @test norm(got - ref) <= atol + rtol * max(norm(ref), 1)
    return nothing
end

if _p0p1_enabled(:m1)
@testset "P0/P1: live contraction memory model" begin
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

@testset "P0/P1: sector-stored live memory" begin
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

@testset "P0/P1: public effective-map hard cap" begin
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
    spec, statics, _ = GRAFT.Contractions._h1_spec(cache, ψ, O, topo.root)
    reference = _P0P1Planning.ncon_reference(spec, ψ.tensors[topo.root], statics)
    @test norm(capped(ψ.tensors[topo.root]) - reference) <=
          1e-12 * max(norm(reference), 1)
end

@testset "P0/P1: bounded hard-cap fallback" begin
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
@testset "P0/P1: planned environments and scalar contractions" begin
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
        GRAFT.Contractions._build_env_spec(cH, ket, O, bra, root, 0)
    scalar_plan = _P0P1Planning.plan_contraction(
        scalar_spec, scalar_operands;
        scalar_type=GRAFT.Backend.scalartype(ket.tensors[root]),
    )
    scalar_ref = GRAFT.Contractions._build_env_ncon_reference(ket, O, bra,
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
    inner_ref = GRAFT.Contractions._build_env_ncon_reference(ket, nothing, bra,
                                                              root, 0, cinner.envs)
    @test inner(bra, ket) ≈ inner_ref rtol=1e-12 atol=1e-12

    ψ = copy(ket)
    cexpect = EnvCache(topo)
    move_center!(ψ, root; cache=cexpect)
    e_root = expect(ψ, O; cache=cexpect)
    local_root = expect(ψ, S.Z, nodeid(topo, root); cache=cexpect)
    local_ref = GRAFT.Contractions._local_expect_ncon_reference(ψ, S.Z, root)
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
@testset "P0/P1: planned variational fit contractions" begin
    topo, _, _, ψ, φ = _p0p1_sandwich_fixture(MersenneTwister(20260717))
    root = topo.root
    child = only(topo.children[root])
    F = GRAFT.Networks

    # Both directions of the ket-bra fit environment use a fresh retained
    # ncon recursion, so this does not compare a planned value with itself.
    cache = F._FitCache(topo)
    for (u, v) in ((child, root), (root, child))
        got = F._fit_env!(cache, φ, ψ, u, v)
        refenvs = Dict{Tuple{Int,Int},GRAFT.Backend.AbstractTensorMap}()
        ref = _p0p1_fit_ref_env!(φ, ψ, u, v, refenvs)
        _p0p1_fit_maps_match(got, ref)
    end

    # Local projection and the scalar overlap are complete-operand plans;
    # their named direct paths remain private A/B references.
    got_project = F._fit_local_tensor(cache, φ, ψ, root)
    refenvs = Dict{Tuple{Int,Int},GRAFT.Backend.AbstractTensorMap}()
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
    rebuilt_refs = Dict{Tuple{Int,Int},GRAFT.Backend.AbstractTensorMap}()
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
        srcenvs = Dict{Tuple{Int,Int},GRAFT.Backend.AbstractTensorMap}()
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
    urefs = Dict{Tuple{Int,Int},GRAFT.Backend.AbstractTensorMap}()
    uenv_ref = _p0p1_fit_ref_env!(uφ, uψ, uchild, uroot, urefs)
    _p0p1_fit_maps_match(uenv, uenv_ref)
    uproject = F._fit_local_tensor(ucache, uφ, uψ, uroot)
    upro_refs = Dict{Tuple{Int,Int},GRAFT.Backend.AbstractTensorMap}()
    for w in neighbors(utopo, uroot)
        _p0p1_fit_ref_env!(uφ, uψ, w, uroot, upro_refs)
    end
    uproject_ref = F._fit_project_tensor_ncon_reference(uφ, uψ, uroot, upro_refs)
    _p0p1_fit_maps_match(uproject, uproject_ref)
    uscalar_ref = F._fit_scalar_ncon_reference(uφ, uψ, uroot, upro_refs)
    @test F._fit_overlap(uφ, uψ) ≈ uscalar_ref rtol=1e-12 atol=1e-12
end
end # :m3 target
