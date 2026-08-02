using GraftPlanning
using GraftPlanning.Backend: FermionParity, U1Space, Vect, ⊗, ←, ℂ,
    norm, ones_tensor
using Random: Xoshiro, randn
using Test

function _static_layout_fixture(rng, P)
    Aspace = P ← P
    Bspace = P ⊗ P
    Cspace = P ← P
    spec = ContractionSpec(
        Vector{Int}[[-1, 1], [2, 1], [-2, 2]],
        Bool[false, false, false],
        2,
        (1, 1),
        1;
        preferred_slots=[2, 3],
    )
    A = randn(rng, ComplexF64, Aspace)
    B = ones_tensor(ComplexF64, Bspace)
    C = randn(rng, ComplexF64, Cspace)
    plan = plan_contraction(spec, (A, B, C); optimize=false)
    return spec, plan, A, (B, C)
end

function _assert_static_layout(spec, plan, A, statics; atol)
    original_steps = copy(plan.steps)
    effective = EffectiveMap(plan, statics)
    stats = static_layout_stats(effective)
    reference = ncon_reference(spec, A, statics)

    @test stats.admitted
    @test stats.prepared_slots == (2, 3)
    @test stats.retained_bytes > 0
    @test effective.plan === plan
    @test effective.statics === statics
    @test effective.layout.execution_plan !== plan
    @test effective.layout.execution_plan.steps !== plan.steps
    @test plan.steps == original_steps

    # Each cached-plan static partition remains non-identity, while the
    # private plan consumes the once-prepared leaf through an identity
    # codomain/domain partition. Conjugation and output semantics are copied.
    for slot in stats.prepared_slots
        raw_index = only(i for i in eachindex(plan.steps)
                         if slot == plan.steps[i].a || slot == plan.steps[i].b)
        raw = plan.steps[raw_index]
        prepared = effective.layout.execution_plan.steps[raw_index]
        if slot == raw.a
            @test (raw.oindA..., raw.cindA...) != Tuple(1:2)
            @test (prepared.oindA..., prepared.cindA...) == Tuple(1:2)
        else
            @test (raw.cindB..., raw.oindB...) != Tuple(1:2)
            @test (prepared.cindB..., prepared.oindB...) == Tuple(1:2)
        end
        @test prepared.conjA == raw.conjA
        @test prepared.conjB == raw.conjB
        @test prepared.out == raw.out
    end

    prepared_ids = objectid.(effective.layout.prepared)
    for _ in 1:3
        @test norm(effective(A) - reference) <= atol * max(norm(reference), 1)
        @test objectid.(effective.layout.prepared) == prepared_ids
        @test plan.steps == original_steps
    end

    workspace = workspace_map(effective)
    @test workspace.workspace.plan === effective.layout.execution_plan
    @test norm(workspace(A) - reference) <= atol * max(norm(reference), 1)
    allocations = workspace_stats(workspace.workspace).allocations
    @test norm(workspace(A) - reference) <= atol * max(norm(reference), 1)
    after = workspace_stats(workspace.workspace)
    @test after.allocations == allocations
    @test after.reuses > 0
    @test objectid.(effective.layout.prepared) == prepared_ids

    rejected = EffectiveMap(
        plan, statics; static_layout_cap_bytes=stats.retained_bytes - 1)
    rejected_stats = static_layout_stats(rejected)
    @test !rejected_stats.admitted
    @test isempty(rejected_stats.prepared_slots)
    @test rejected_stats.retained_bytes == 0
    @test rejected.layout.execution_plan === plan
    @test rejected.layout.prepared === statics
    @test norm(rejected(A) - reference) <= atol * max(norm(reference), 1)
    @test plan.steps == original_steps
    return nothing
end

@testset "GraftPlanning structural specification" begin
    labels = Vector{Int}[[-1, 1], [1, 2], [2, -2]]
    conjugations = Bool[false, false, false]
    spec = ContractionSpec(
        labels,
        conjugations,
        2,
        (1, 1),
        1;
        preferred_slots=[2, 3],
    )

    labels[1][1] = -99
    conjugations[1] = true
    @test spec.labels[1] == [-1, 1]
    @test spec.conjs == falses(3)
    @test spec.dynamic_slot == 1
    @test spec.preferred_slots == [2, 3]

    @test_throws ArgumentError ContractionSpec(
        Vector{Int}[[-1, 1]], Bool[], 1, (1, 0), 1)
    @test_throws ArgumentError ContractionSpec(
        Vector{Int}[[-1, 1], [1, -2]], falses(2), 2, (1, 1), 2)
    @test_throws ArgumentError ContractionSpec(
        Vector{Int}[[-1, 1], [1, -2]], falses(2), 2, (2, 1), 1)
    @test_throws ArgumentError ContractionSpec(
        Vector{Int}[[-1, 1], [1, -2]], falses(2), 2, (1, 1), 1;
        preferred_slots=[1])
end

@testset "GraftPlanning dense cost and cache-free plan" begin
    metrics = dense_cost([-1, 1], [2, 3], [1, -2], [3, 4])
    @test metrics.oindA == [1]
    @test metrics.cindA == [2]
    @test metrics.oindB == [2]
    @test metrics.cindB == [1]
    @test metrics.labels == [-1, -2]
    @test metrics.dims == [2, 4]
    @test metrics.peak_elements == 8
    @test metrics.flops > 0
    @test_throws ArgumentError dense_cost(
        [-1, 1], [2, 3], [1, -2], [4, 5])

    spec = ContractionSpec(
        Vector{Int}[[-1, 1], [1, 2], [2, -2]],
        falses(3),
        2,
        (1, 1),
        1;
        preferred_slots=[2, 3],
    )
    prototypes = (ℂ^2 ← ℂ^3, ℂ^3 ← ℂ^4, ℂ^4 ← ℂ^5)
    plan = plan_contraction(
        spec,
        prototypes;
        optimize=false,
        scalar_type=ComplexF64,
    )

    @test plan.strategy == :env_first
    @test length(plan.steps) == 2
    @test plan.scalar_bytes == sizeof(ComplexF64)
    @test plan.peak_elements >= 10
    @test plan.live_peak_bytes >= plan.operand_bytes
    @test plan_diagnostics(plan).classification == :selected
    @test plan_metrics(plan).strategy == :env_first

    @test_throws ArgumentError plan_contraction(
        spec,
        prototypes;
        optimize=false,
        scalar_type=ComplexF64,
        memory_cap_bytes=plan.live_peak_bytes - 1,
    )
end

@testset "GraftPlanning sector-aware structural cost" begin
    left = U1Space(0 => 1, 1 => 1) ← U1Space(0 => 1, 1 => 2)
    middle = U1Space(0 => 1, 1 => 2) ← U1Space(0 => 1, 1 => 4)
    right = U1Space(0 => 1, 1 => 4) ← U1Space(0 => 2, 1 => 1)
    spec = ContractionSpec(
        Vector{Int}[[-1, 1], [1, 2], [2, -2]],
        falses(3),
        2,
        (1, 1),
        1;
        preferred_slots=[2, 3],
    )

    dense = plan_contraction(
        spec,
        (left, middle, right);
        sector_aware=false,
        memory_weight=0,
        scalar_type=ComplexF64,
    )
    sector = plan_contraction(
        spec,
        (left, middle, right);
        sector_aware=true,
        memory_weight=0,
        scalar_type=ComplexF64,
    )

    @test dense.strategy == :env_first
    @test dense.flops == 60
    @test dense.sector_flops == 30
    @test sector.strategy == :sector_exact
    @test sector.flops == 63
    @test sector.sector_flops == 28
    @test sort([sector.steps[1].a, sector.steps[1].b]) == [2, 3]
    @test sector.sector_operand_bytes < sector.operand_bytes
    @test sector.sector_live_peak_bytes <= sector.live_peak_bytes
end

@testset "GraftPlanning dense static leaf prelayout" begin
    fixture = _static_layout_fixture(Xoshiro(0x51a71c), ℂ^2)
    _assert_static_layout(fixture...; atol=1e-12)

    # A two-input scalar contraction exercises root scalarization with a
    # non-identity static leaf partition.
    scalar_spec = ContractionSpec(
        Vector{Int}[[1, 2], [2, 1]],
        Bool[false, false],
        0,
        (0, 0),
        1;
        preferred_slots=[2],
    )
    Aspace = ℂ^2 ← ℂ^3
    Bspace = ℂ^3 ← ℂ^2
    rng = Xoshiro(0x5ca1a2)
    A = randn(rng, ComplexF64, Aspace)
    B = randn(rng, ComplexF64, Bspace)
    plan = plan_contraction(scalar_spec, (Aspace, Bspace); optimize=false)
    effective = EffectiveMap(plan, (B,))
    reference = ncon_reference(scalar_spec, A, (B,))
    @test effective(A) ≈ reference rtol=1e-13 atol=1e-13
    @test workspace_map(effective)(A) ≈ reference rtol=1e-13 atol=1e-13
    @test static_layout_stats(effective).prepared_slots == (2,)

    # Conjugation is still applied by the derived PairStep after preparation;
    # it is not folded into or lost from the static tensor layout.
    conjugated_spec = ContractionSpec(
        Vector{Int}[[-1, 1], [-2, 1]],
        Bool[false, true],
        2,
        (1, 1),
        1;
        preferred_slots=[2],
    )
    D = randn(rng, ComplexF64, ℂ^2 ← ℂ^2)
    E = randn(rng, ComplexF64, ℂ^2 ← ℂ^2)
    conjugated_plan = plan_contraction(
        conjugated_spec, (D, E); optimize=false)
    conjugated = EffectiveMap(conjugated_plan, (E,))
    conjugated_reference = ncon_reference(conjugated_spec, D, (E,))
    @test static_layout_stats(conjugated).prepared_slots == (2,)
    @test only(conjugated.layout.execution_plan.steps).conjB
    @test norm(conjugated(D) - conjugated_reference) <=
          1e-13 * max(norm(conjugated_reference), 1)

    # Identity-only leaves and zero-step maps retain the original plan rather
    # than manufacturing an execution-plan copy with no prepared work.
    identity_spec = ContractionSpec(
        Vector{Int}[[-1, 1], [1, -2]],
        Bool[false, false],
        2,
        (1, 1),
        1;
        preferred_slots=[2],
    )
    identity_plan = plan_contraction(
        identity_spec, (ℂ^2 ← ℂ^3, ℂ^3 ← ℂ^4);
        optimize=false,
    )
    identity_static = randn(rng, ComplexF64, ℂ^3 ← ℂ^4)
    identity_map = EffectiveMap(identity_plan, (identity_static,))
    @test !static_layout_stats(identity_map).admitted
    @test identity_map.layout.execution_plan === identity_plan
    @test identity_map.layout.prepared === identity_map.statics

    zero_step_plan = ContractionPlan(1, 1, PairStep[])
    zero_step_map = EffectiveMap(zero_step_plan, ())
    @test !static_layout_stats(zero_step_map).admitted
    @test zero_step_map.layout.execution_plan === zero_step_plan
end

@testset "GraftPlanning FermionParity static leaf prelayout" begin
    even, odd = FermionParity(0), FermionParity(1)
    P = Vect[FermionParity](even => 1, odd => 1)
    fixture = _static_layout_fixture(Xoshiro(0xfe2210), P)
    _assert_static_layout(fixture...; atol=2e-12)

    rng = Xoshiro(0xfe2211)
    A = randn(rng, ComplexF64, P ← P)
    B = randn(rng, ComplexF64, P ← P)
    spec = ContractionSpec(
        Vector{Int}[[-1, 1], [-2, 1]],
        Bool[false, true], 2, (1, 1), 1;
        preferred_slots=[2],
    )
    plan = plan_contraction(spec, (A, B); optimize=false)
    effective = EffectiveMap(plan, (B,))
    reference = ncon_reference(spec, A, (B,))
    @test static_layout_stats(effective).prepared_slots == (2,)
    @test only(effective.layout.execution_plan.steps).conjB
    @test norm(effective(A) - reference) <= 2e-12 * max(norm(reference), 1)
end

@testset "GraftPlanning owner-local load graph" begin
    loaded = Set(nameof(module_) for module_ in values(Base.loaded_modules))
    @test :GraftPlanning in loaded
    for forbidden in (
        :Graft,
        :GraftNetworks,
        :GraftContractions,
        :GraftTestUtils,
    )
        @test forbidden ∉ loaded
    end
end
