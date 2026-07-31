using GraftPlanning
using GraftPlanning.Backend: U1Space, ←, ℂ
using Test

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
