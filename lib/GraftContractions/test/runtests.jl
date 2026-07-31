using Test
using GraftFoundation: ℂ, TensorMap, mps_topology, ←, ⊗
using GraftNetworks: TTNS, check_arrows
using GraftContractions: EnvCache, cache_diagnostics, env!, env_cache_stats,
    inner, invalidate_node!

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
