using Test
import GraftContractions
using GraftFoundation: AbstractTensorMap, ℂ, FermionParity, TensorMap, Vect,
    dim, domain, id, mps_topology, norm, ones_tensor, sectors, ←, ⊗
using GraftNetworks: TTNS, check_arrows
using GraftContractions: EnvCache, cache_diagnostics, env!, env_cache_stats,
    inner, invalidate_node!
using Random: Xoshiro, randn

const _C = GraftContractions.Contractions

function assert_incremental_env_payload(cache::EnvCache)
    scan = _C._with_cache_lock(cache) do
        entry_bytes = Dict(
            key => _C._env_payload_bytes(E) for (key, E) in cache.envs)
        size_counts = Dict{Int,Int}()
        for bytes in values(entry_bytes)
            size_counts[bytes] = get(size_counts, bytes, 0) + 1
        end
        total = sum(values(entry_bytes); init=0)
        largest = maximum(values(entry_bytes); init=0)
        (; entry_bytes, size_counts, total, largest,
         cached_entry_bytes=copy(cache.env_entry_bytes),
         cached_size_counts=copy(cache.env_payload_size_counts),
         cached_total=cache.env_payload_bytes,
         cached_largest=cache.env_largest_entry_bytes)
    end
    stats = env_cache_stats(cache)
    @test scan.cached_entry_bytes == scan.entry_bytes
    @test scan.cached_size_counts == scan.size_counts
    @test scan.cached_total == scan.total == stats.payload_bytes
    @test scan.cached_largest == scan.largest == stats.largest_entry_bytes
    return nothing
end

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

@testset "effective-map static layout memory admission" begin
    rng = Xoshiro(2026080301)
    P = ℂ^2
    A = randn(rng, ComplexF64, P ← P)
    B = ones_tensor(ComplexF64, P ⊗ P)
    C = randn(rng, ComplexF64, P ← P)
    spec = GraftContractions.ContractionSpec(
        Vector{Int}[[-1, 1], [2, 1], [-2, 2]],
        falses(3), 2, (1, 1), 1;
        preferred_slots=[2, 3],
    )
    cache = EnvCache(mps_topology(2))
    uncapped = _C._effective_map!(
        cache, :static_layout_uncapped, spec, (A, B, C), (B, C), ComplexF64;
        optimize=false,
    )
    uncapped_layout = GraftContractions.Planning.static_layout_stats(uncapped)
    @test uncapped_layout.admitted
    @test uncapped_layout.prepared_slots == (2, 3)
    @test uncapped_layout.retained_bytes > 0

    dense_live = uncapped.plan.live_peak_bytes
    sector_live = isfinite(uncapped.plan.sector_live_peak_bytes) ?
                  uncapped.plan.sector_live_peak_bytes : dense_live
    plan_cap = max(dense_live, sector_live)
    capped = _C._effective_map!(
        cache, :static_layout_capped, spec, (A, B, C), (B, C), ComplexF64;
        optimize=false, memory_cap_bytes=plan_cap,
    )
    capped_layout = GraftContractions.Planning.static_layout_stats(capped)
    @test !capped_layout.admitted
    @test isempty(capped_layout.prepared_slots)
    @test capped_layout.retained_bytes == 0
    @test norm(capped(A) - uncapped(A)) <= 1e-12 * max(norm(uncapped(A)), 1)

    plan_live = _C._slice_live_bytes(uncapped.plan)
    @test _C._slice_live_bytes(uncapped) ==
          plan_live + uncapped_layout.retained_bytes
    @test _C._concurrent_slice_bytes((uncapped, capped)) ==
          ceil(Int, _C._slice_live_bytes(uncapped) +
                    _C._slice_live_bytes(capped))
end

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

@testset "EnvCache incremental payload accounting" begin
    state = two_site_product_state([1, 0], [1, 0])
    topo = state.topo
    root = topo.root
    child = only(topo.children[root])
    edge_key = (child, root)

    cache = EnvCache(topo)
    edge_env = env!(cache, state, nothing, state, edge_key...)
    assert_incremental_env_payload(cache)

    large = randn(Xoshiro(2026080301), ComplexF64, ℂ^3 ← ℂ^4)
    large_bytes = _C._env_payload_bytes(large)
    synthetic_a = (91, 92)
    synthetic_b = (93, 94)
    _C._store_env!(cache, synthetic_a, large)
    _C._store_env!(cache, synthetic_b, copy(large))
    assert_incremental_env_payload(cache)
    @test env_cache_stats(cache).largest_entry_bytes == large_bytes

    # Replacing one maximum-sized entry retains the maximum while its peer is
    # live; deleting the last peer recomputes the maximum from the size counts.
    high_water = env_cache_stats(cache).high_water_bytes
    _C._store_env!(cache, synthetic_a, copy(edge_env))
    assert_incremental_env_payload(cache)
    @test env_cache_stats(cache).largest_entry_bytes == large_bytes
    _C._with_cache_lock(cache) do
        _C._delete_environment_locked!(cache, synthetic_b)
    end
    assert_incremental_env_payload(cache)
    @test env_cache_stats(cache).largest_entry_bytes ==
          _C._env_payload_bytes(edge_env)
    @test env_cache_stats(cache).high_water_bytes == high_water
    _C._with_cache_lock(cache) do
        _C._delete_environment_locked!(cache, synthetic_a)
    end
    assert_incremental_env_payload(cache)

    # An unannounced tensor replacement makes the existing stamp stale. The
    # stale lookup deletes through the same accounting path before rebuilding.
    state.tensors[child] = copy(state.tensors[child])
    lookup = _C._lookup_environment!(
        cache, edge_key, state, nothing, state)
    @test !lookup.found
    assert_incremental_env_payload(cache)
    env!(cache, state, nothing, state, edge_key...)
    assert_incremental_env_payload(cache)

    invalidate_node!(cache, child)
    assert_incremental_env_payload(cache)

    # Seeded construction owns a copy of the supplied dictionary and performs
    # the only production full scan needed to initialize incremental metadata.
    seeded_envs = Dict{Tuple{Int,Int},AbstractTensorMap}(
        edge_key => edge_env,
        synthetic_b => large,
    )
    seeded = EnvCache(topo, seeded_envs)
    @test seeded.envs !== seeded_envs
    empty!(seeded_envs)
    @test length(seeded.envs) == 2
    assert_incremental_env_payload(seeded)
    generation = seeded.cache_generation[]
    _C._clear_value_environments!(seeded)
    @test seeded.cache_generation[] == generation + UInt64(1)
    @test isempty(seeded.envs)
    assert_incremental_env_payload(seeded)

    # Cap eviction removes both value and payload metadata, while high-water
    # remains the maximum pre-eviction live payload until cache-wide empty!.
    capped = EnvCache(topo; max_env_bytes=0)
    _C._with_env_transaction(capped) do
        _C._store_env!(capped, synthetic_a, large)
    end
    @test isempty(capped.envs)
    @test env_cache_stats(capped).high_water_bytes == large_bytes
    assert_incremental_env_payload(capped)
    empty!(capped)
    @test env_cache_stats(capped).high_water_bytes == 0
    assert_incremental_env_payload(capped)
end
