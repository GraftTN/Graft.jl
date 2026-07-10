# Targeted P0/P1 contraction-memory checks.  This file is intentionally
# self-contained so the constrained host can invoke it directly. `runtests.jl`
# includes it for stronger-host validation too; keep fixtures shape-only or
# two/three-map small so it never becomes a benchmark.
using Test
using Random
using LinearAlgebra
using GRAFT
using GRAFT.TestUtils: random_ttns
using GRAFT.Backend: ℂ, ⊗, ←, U1Space, norm

const _P0P1Planning = GRAFT.Contractions.Planning

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
