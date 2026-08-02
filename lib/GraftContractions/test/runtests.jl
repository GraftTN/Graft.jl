using Test
import GraftContractions
using GraftFoundation: ℂ, FermionParity, TensorMap, Vect, dim, domain, id,
    mps_topology, norm, sectors, ←, ⊗
using GraftNetworks: TTNS, check_arrows
using GraftContractions: EnvCache, cache_diagnostics, env!, env_cache_stats,
    inner, invalidate_node!
using Random: Xoshiro, randn

const _C = GraftContractions.Contractions

function two_site_product_state(
        child_entries::AbstractVector{<:Number},
        root_entries::AbstractVector{<:Number})
    length(child_entries) == length(root_entries) == 2 ||
        throw(ArgumentError("expected two two-level states"))
    topo = mps_topology(2)
    physical = ℂ^2
    bond = ℂ^1
    root = TensorMap(
        reshape(ComplexF64.(root_entries), 1, 2, 1),
        bond ⊗ physical ← bond,
    )
    child = TensorMap(
        reshape(ComplexF64.(child_entries), 2, 1), physical ← bond)
    return TTNS(topo, [root, child], topo.root)
end

include("two_site_factor_frame.jl")

@testset "subspace predictor range finders" begin
    dense = randn(Xoshiro(2026080201), ComplexF64, ℂ^10 ← ℂ^18)

    direct = _C._directqr_predictor_basis(dense)
    @test norm(direct' * direct - id(domain(direct))) < 1e-12
    @test dim(domain(direct)) == 10

    range_serial = _C._rangefinder_predictor_basis(
        dense, 4;
        rng=Xoshiro(2026080202), rsvd_oversample=2, rsvd_poweriter=1,
        threaded=false,
    )
    range_parallel = _C._rangefinder_predictor_basis(
        dense, 4;
        rng=Xoshiro(2026080202), rsvd_oversample=2, rsvd_poweriter=1,
        threaded=true, minbatch=1, memory_cap_bytes=1_000_000,
        task_workspace_memory_bytes=0,
    )
    @test norm(range_serial' * range_serial - id(domain(range_serial))) < 1e-12
    @test range_serial == range_parallel

    rsvd = _C._rsvd_predictor_basis(
        dense, 4;
        rng=Xoshiro(2026080203), rsvd_oversample=2, rsvd_poweriter=0,
        threaded=false,
    )
    @test norm(rsvd' * rsvd - id(domain(rsvd))) < 1e-12
    @test dim(domain(rsvd)) == 4

    even, odd = FermionParity(0), FermionParity(1)
    graded_codomain = Vect[FermionParity](even => 3, odd => 2)
    graded_domain = Vect[FermionParity](even => 5, odd => 4)
    graded = randn(
        Xoshiro(2026080204), ComplexF64, graded_codomain ← graded_domain)
    graded_direct = _C._directqr_predictor_basis(graded)
    graded_range = _C._rangefinder_predictor_basis(
        graded, 2;
        rng=Xoshiro(2026080205), rsvd_oversample=1, rsvd_poweriter=1,
        threaded=false,
    )
    @test norm(graded_direct' * graded_direct - id(domain(graded_direct))) < 1e-12
    @test norm(graded_range' * graded_range - id(domain(graded_range))) < 1e-12
    @test collect(sectors(domain(graded_direct)[1])) == [even, odd]
    @test collect(sectors(domain(graded_range)[1])) == [even, odd]
end

@testset "GraftContractions overlap environments and plan pool" begin
    zero_zero = two_site_product_state([1, 0], [1, 0])
    one_zero = two_site_product_state([0, 1], [1, 0])
    @test check_arrows(zero_zero)
    @test check_arrows(one_zero)

    plan_cache = EnvCache(zero_zero.topo)
    @test inner(zero_zero, zero_zero; plan_cache, optimize=false) ≈ 1
    after_first = cache_diagnostics(plan_cache)
    @test after_first.plan_entries > 0
    @test after_first.environments.entries == 0

    @test inner(zero_zero, one_zero; plan_cache, optimize=false) ≈ 0 atol=1.0e-14
    after_second = cache_diagnostics(plan_cache)
    @test after_second.plan_entries == after_first.plan_entries
    @test after_second.environments.entries == 0

    environment_cache = EnvCache(zero_zero.topo)
    root = zero_zero.topo.root
    child = only(zero_zero.topo.children[root])
    first_environment = env!(
        environment_cache, zero_zero, nothing, zero_zero, child, root)
    first_stats = env_cache_stats(environment_cache)
    @test first_stats.entry_count == 1
    @test first_stats.misses == first_stats.rebuilds == 1

    second_environment = env!(
        environment_cache, zero_zero, nothing, zero_zero, child, root)
    second_stats = env_cache_stats(environment_cache)
    @test second_environment === first_environment
    @test second_stats.hits == 1
    @test second_stats.rebuilds == first_stats.rebuilds

    invalidate_node!(environment_cache, child)
    invalidated_stats = env_cache_stats(environment_cache)
    @test invalidated_stats.entry_count == 0
    @test invalidated_stats.maximum_node_generation == 1

    wrong_topology_cache = EnvCache(mps_topology(1))
    @test_throws ArgumentError inner(
        zero_zero, zero_zero; plan_cache=wrong_topology_cache, optimize=false)
end
