const _CP = Graft.Contractions
const _Planning = Graft.Contractions.Planning

function _assert_env_payload_accounting(cache::EnvCache)
    scan = _CP._with_cache_lock(cache) do
        entry_bytes = Dict(
            key => _CP._env_payload_bytes(E) for (key, E) in cache.envs)
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

struct _PlanningDistributedContext <: AbstractDistributedContext
    rank::Int
    size::Int
end

Graft.Parallel.distributed_rank(context::_PlanningDistributedContext) =
    context.rank
Graft.Parallel.distributed_size(context::_PlanningDistributedContext) =
    context.size

"""
Return every *ordered* binary tree for a very small label graph, including
early outer products.  This deliberately does not call Phase 3's DP: it is a
compact independent oracle for the connected-DP normal form.
"""
function _all_ordered_trees(spec, protos)
    dims, _ = _Planning._label_dimensions(spec, protos)
    memo = Dict{Int,Vector{Any}}()
    function build(mask::Int)
        haskey(memo, mask) && return memo[mask]
        if count_ones(mask) == 1
            slot = trailing_zeros(mask) + 1
            return memo[mask] = Any[(tree=slot,
                                     labels=copy(spec.labels[slot]),
                                     dims=copy(dims[slot]))]
        end
        states = Any[]
        leftmask = (mask - 1) & mask
        while leftmask != 0
            rightmask = mask ⊻ leftmask
            if rightmask != 0
                for left in build(leftmask), right in build(rightmask)
                    metrics = _Planning.dense_cost(left.labels, left.dims,
                                                    right.labels, right.dims)
                    push!(states, (tree=Any[left.tree, right.tree],
                                   labels=metrics.labels, dims=metrics.dims))
                end
            end
            leftmask = (leftmask - 1) & mask
        end
        return memo[mask] = states
    end
    fullmask = (1 << length(spec.labels)) - 1
    return dims, Any[state.tree for state in build(fullmask)]
end

function _sector_oracle_score(spec, protos, envfirst; memory_weight::Real)
    dims, trees = _all_ordered_trees(spec, protos)
    scores = Float64[]
    for tree in trees
        plan = _Planning._compile_plan(tree, spec, dims, protos;
                                       strategy=:oracle, structural_metrics=true,
                                       canonical_intermediates=true)
        plan.peak_elements <= envfirst.peak_elements || continue
        plan.sector_peak_elements <= envfirst.sector_peak_elements || continue
        push!(scores, _Planning._sector_score(plan, memory_weight))
    end
    isempty(scores) && error("sector oracle produced no memory-safe tree")
    return minimum(scores)
end

function _sector_dp_score(spec, protos, envfirst; memory_weight::Real)
    dims, _ = _Planning._label_dimensions(spec, protos)
    scores = Float64[]
    for tree in _Planning._sector_dp_trees(spec, dims, protos)
        plan = _Planning._compile_plan(tree, spec, dims, protos;
                                       strategy=:oracle, structural_metrics=true,
                                       canonical_intermediates=true)
        plan.peak_elements <= envfirst.peak_elements || continue
        plan.sector_peak_elements <= envfirst.sector_peak_elements || continue
        push!(scores, _Planning._sector_score(plan, memory_weight))
    end
    isempty(scores) && error("sector DP produced no memory-safe tree")
    return minimum(scores)
end

function _sector_three_map_fixture()
    A = U1Space(0 => 1, 1 => 1) ← U1Space(0 => 1, 1 => 2)
    B = U1Space(0 => 1, 1 => 2) ← U1Space(0 => 1, 1 => 4)
    C = U1Space(0 => 1, 1 => 4) ← U1Space(0 => 2, 1 => 1)
    spec = _Planning.ContractionSpec(
        Vector{Int}[[-1, 1], [1, 2], [2, -2]],
        Bool[false, false, false], 2, (1, 1), 1;
        preferred_slots=[2, 3],
    )
    return A, B, C, spec
end

function _assert_planned_matches_ncon(spec, statics, planned, x;
                                      rtol::Real=1e-13, atol::Real=1e-13)
    got = planned(x)
    ref = _CP._ncon_effective_reference(spec, x, statics)
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
               Graft.Backend.codomain(x0) ← domain(x0))
    x2 = randn(rng, ComplexF64,
               Graft.Backend.codomain(x0) ← domain(x0))
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

function _exercise_effective_maps!(ψ, O; extended::Bool=GRAFT_EXTENDED_TESTS)
    topo = topology(ψ)
    selected_nodes = extended ? (1:nnodes(topo)) : (topo.root,)
    for n in selected_nodes
        move_center!(ψ, n)
        cache = EnvCache(topo)
        spec, statics, protos = _CP._h1_spec(cache, ψ, O, n)
        planned = eff_h1(cache, ψ, O, n)
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

    root_edge = (first(topo.children[topo.root]), topo.root)
    selected_edges = extended ? edges(topo) : (root_edge,)
    for (n, m) in selected_edges
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

@graft_testset "compiled contraction plans: A/B effective maps" begin
    # MPS and star coverage is delegated to their dedicated regression stages below.
    for topo in (binary_topology(2), fork_topology(2, 1))
        phys = allspin(topo)
        O = ttno_from_opsum(tfi(topo; g=0.41), topo, phys; hermitian=true)
        ψ = random_ttns(MersenneTwister(1300 + nnodes(topo)), ComplexF64,
                        topo, phys, ℂ^2)
        _exercise_effective_maps!(ψ, O)
    end

    # Tier 3 is an explicit large-map optimization. The selected TTNO child
    # channel is projected to smaller exact spaces, each partial executes in
    # task-local storage, and the result is reduced in slice order.
    topo = star_topology(3, 1)
    phys = allspin(topo)
    O = ttno_from_opsum(tfi(topo; g=0.41), topo, phys; hermitian=true)
    ψ = random_ttns(MersenneTwister(1391), ComplexF64, topo, phys, ℂ^3)
    root = topo.root
    full = eff_h1(EnvCache(topo), ψ, O, root)
    if Base.Threads.nthreads() > 1
        gated = eff_h1(
            EnvCache(topo), ψ, O, root; threaded_channels=true,
            channel_slices=3, channel_min_flops=1e300,
        )
        @test gated isa EffectiveMap
        @test_throws ArgumentError eff_h1(
            EnvCache(topo), ψ, O, root; threaded_channels=true,
            channel_slices=3, channel_min_flops=-1,
        )
        @test_throws ArgumentError eff_h1(
            EnvCache(topo), ψ, O, root; threaded_channels=true,
            channel_slices=3, channel_min_flops=Inf,
            channel_memory_cap_bytes=1_000_000_000,
        )
        @test_throws ArgumentError eff_h1(
            EnvCache(topo), ψ, O, root; threaded_channels=true,
            channel_slices=3, channel_memory_cap_bytes=0,
        )
        sliced = eff_h1(
            EnvCache(topo), ψ, O, root; threaded_channels=true,
            channel_slices=3, channel_memory_cap_bytes=1_000_000_000,
        )
        @test sliced isa ChannelSlicedEffectiveMap
        @test workspace_map(sliced) === sliced
        excess_context = _PlanningDistributedContext(
            0, length(sliced.maps) + 1)
        admission_error = try
            DistributedChannelEffectiveMap(sliced, excess_context)
            nothing
        catch err
            err
        end
        @test admission_error isa DistributedChannelAdmissionError
        @test admission_error.reason == :empty_rank
        @test admission_error.ranks == length(sliced.maps) + 1
        @test admission_error.nonempty_slices == length(sliced.maps)
        @test occursin(
            "relaunch with at most $(length(sliced.maps)) ranks",
            sprint(showerror, admission_error),
        )

        # Admission occurs after exact grouping but before any sliced plan is
        # compiled or published into the cache.
        rejected_cache = EnvCache(topo)
        rejected_context = _PlanningDistributedContext(0, 100)
        rejected = try
            eff_h1(
                rejected_cache, ψ, O, root;
                distributed=rejected_context,
                channel_slices=3,
                channel_minbatch=1,
                channel_memory_cap_bytes=1_000_000_000,
            )
            nothing
        catch err
            err
        end
        @test rejected isa DistributedChannelAdmissionError
        @test rejected.reason == :empty_rank
        @test rejected.ranks == 100
        @test length(sliced.maps) <= rejected.nonempty_slices < 100
        @test all(
            key -> !startswith(String(key.kind), "h1_channel_"),
            keys(rejected_cache.plans),
        )

        reference = full(ψ.tensors[root])
        got1 = sliced(ψ.tensors[root])
        got2 = sliced(ψ.tensors[root])
        scale = max(norm(reference), 1.0)
        @test norm(got1 - reference) <= 1e-12 * scale
        @test norm(got2 - got1) == 0

        pooled = _CP._channel_workspace_map(sliced)
        worker_tasks = copy(pooled.tasks)
        try
            pooled1 = pooled(ψ.tensors[root])
            pooled2 = pooled(ψ.tensors[root])
            @test norm(pooled1 - reference) <= 1e-12 * scale
            @test norm(pooled2 - pooled1) == 0
            @test all(!istaskdone(task) for task in worker_tasks)
            cross_task_error = fetch(@async begin
                try
                    pooled(ψ.tensors[root])
                    nothing
                catch err
                    err
                end
            end)
            @test cross_task_error isa ArgumentError
        finally
            close(pooled)
        end
        @test all(istaskdone, worker_tasks)
        @test_throws ArgumentError pooled(ψ.tensors[root])
        @test isnothing(close(pooled))

        scoped = _CP.with_workspace_map(sliced) do h1map
            h1map(ψ.tensors[root])
        end
        @test norm(scoped - reference) <= 1e-12 * scale

        serial_sliced = ChannelSlicedEffectiveMap(
            sliced.maps, length(sliced.maps) + 1, sliced.concurrent_live_bytes)
        serial_workspace = _CP._channel_workspace_map(serial_sliced)
        @test serial_workspace isa _CP._SerialChannelSlicedWorkspaceMap
        serial_got = serial_workspace(ψ.tensors[root])
        @test norm(serial_got - reference) <= 1e-12 * scale
        @test isnothing(close(serial_workspace))

        # The two-site map may split either an external TTNO edge or the
        # internal child-parent TTNO bond. Candidate selection minimizes the
        # predicted maximum slice cost before building the actual maps.
        child = first(topo.children[root])
        move_center!(ψ, child)
        Θ = two_site_tensor(ψ, child, root)
        h2full = eff_h2(EnvCache(topo), ψ, O, child, root)
        @test_throws ArgumentError eff_h2(
            EnvCache(topo), ψ, O, child, root; threaded_channels=true,
            channel_slices=3, channel_memory_cap_bytes=0,
        )
        h2sliced = eff_h2(
            EnvCache(topo), ψ, O, child, root; threaded_channels=true,
            channel_slices=3, channel_memory_cap_bytes=1_000_000_000,
        )
        @test h2sliced isa ChannelSlicedEffectiveMap
        rejected_h2_cache = EnvCache(topo)
        rejected_h2 = try
            eff_h2(
                rejected_h2_cache, ψ, O, child, root;
                distributed=rejected_context,
                channel_slices=3,
                channel_minbatch=1,
                channel_memory_cap_bytes=1_000_000_000,
            )
            nothing
        catch err
            err
        end
        @test rejected_h2 isa DistributedChannelAdmissionError
        @test rejected_h2.reason == :empty_rank
        @test rejected_h2.ranks == 100
        @test all(
            key -> !startswith(String(key.kind), "h2_channel_"),
            keys(rejected_h2_cache.plans),
        )
        h2reference = h2full(Θ)
        h2scale = max(norm(h2reference), 1.0)
        h2pooled = _CP._channel_workspace_map(h2sliced)
        h2tasks = copy(h2pooled.tasks)
        try
            h2got1 = h2pooled(Θ)
            h2got2 = h2pooled(Θ)
            @test norm(h2got1 - h2reference) <= 1e-12 * h2scale
            @test norm(h2got2 - h2got1) == 0
        finally
            close(h2pooled)
        end
        @test all(istaskdone, h2tasks)

        # A leaf has only its parent TTNO leg, which is a domain flat leg.
        # This covers the generic transform-leg path and parent-edge slicing.
        leaf = first(leaves(topo))
        leaf_full = eff_h1(EnvCache(topo), ψ, O, leaf)
        leaf_sliced = eff_h1(
            EnvCache(topo), ψ, O, leaf; threaded_channels=true,
            channel_slices=3, channel_memory_cap_bytes=1_000_000_000,
        )
        @test leaf_sliced isa ChannelSlicedEffectiveMap
        leaf_reference = leaf_full(ψ.tensors[leaf])
        leaf_got = _CP.with_workspace_map(leaf_sliced) do h1map
            h1map(ψ.tensors[leaf])
        end
        @test norm(leaf_got - leaf_reference) <=
            1e-12 * max(norm(leaf_reference), 1.0)
    else
        fallback = eff_h1(
            EnvCache(topo), ψ, O, root; threaded_channels=true,
            channel_slices=3,
        )
        @test fallback isa EffectiveMap
        child = first(topo.children[root])
        h2fallback = eff_h2(
            EnvCache(topo), ψ, O, child, root; threaded_channels=true,
            channel_slices=3,
        )
        @test h2fallback isa EffectiveMap
    end

    # Neutral U(1) TTNO/state: the plan uses exact TensorKit spaces, not just
    # dense array dimensions, so this is the symmetric counterpart of the
    # trivial-sector checks above.
    U = spin_ops_u1()
    topo = star_topology(2, 1)
    phys = Dict(nodeid(topo, i) => U.P for i in 1:nnodes(topo))
    H = OpSum()
    for (c, p) in edges(topo)
        H += Term(0.11, SiteOp(nodeid(topo, c), :Z, U.Z),
                  SiteOp(nodeid(topo, p), :Z, U.Z))
        H += Term(0.07, SiteOp(nodeid(topo, c), :Sp, U.Sp),
                  SiteOp(nodeid(topo, p), :Sm, U.Sm))
        H += Term(0.07, SiteOp(nodeid(topo, c), :Sm, U.Sm),
                  SiteOp(nodeid(topo, p), :Sp, U.Sp))
    end
    for n in 1:nnodes(topo)
        H += Term(0.17 + 0.03 * n, SiteOp(nodeid(topo, n), :Z, U.Z))
    end
    O = ttno_from_opsum(H, topo, phys; hermitian=true)
    V = U1Space(-1 => 1, 0 => 2, 1 => 1)
    ψ = random_ttns(MersenneTwister(1401), ComplexF64, topo, phys, V)
    _exercise_effective_maps!(ψ, O)
    if Base.Threads.nthreads() > 1
        full = eff_h1(EnvCache(topo), ψ, O, topo.root)
        expected_gate_flops = isfinite(full.plan.sector_flops) ?
            full.plan.sector_flops : full.plan.flops
        @test _CP._channel_plan_flops(full.plan) == expected_gate_flops
        sliced = eff_h1(
            EnvCache(topo), ψ, O, topo.root; threaded_channels=true,
            channel_slices=2, channel_memory_cap_bytes=1_000_000_000,
        )
        reference = full(ψ.tensors[topo.root])
        got = sliced(ψ.tensors[topo.root])
        @test norm(got - reference) <= 1e-12 * max(norm(reference), 1.0)

        child = first(topo.children[topo.root])
        move_center!(ψ, child)
        Θ = two_site_tensor(ψ, child, topo.root)
        h2full = eff_h2(EnvCache(topo), ψ, O, child, topo.root)
        h2sliced = eff_h2(
            EnvCache(topo), ψ, O, child, topo.root; threaded_channels=true,
            channel_slices=2, channel_memory_cap_bytes=1_000_000_000,
        )
        @test h2sliced isa ChannelSlicedEffectiveMap
        h2reference = h2full(Θ)
        h2got = h2sliced(Θ)
        @test norm(h2got - h2reference) <=
            1e-12 * max(norm(h2reference), 1.0)
    end
end

@graft_testset "compiled contraction plans: sector-aware 3-map action gate" begin
    # The dense model prefers (A*B)*C (60 < 63), but the exact U(1) block
    # model prefers A*(B*C) (28 < 30). Keep this action-level regression in
    # the default tier because a wrong sector order can silently remain
    # dimension-valid.
    A, B, C, spec = _sector_three_map_fixture()
    dense = _Planning.plan_contraction(spec, (A, B, C);
                                         sector_aware=false, memory_weight=0)
    sector = _Planning.plan_contraction(spec, (A, B, C);
                                          sector_aware=true, memory_weight=0)
    envfirst = _Planning.plan_contraction(spec, (A, B, C); optimize=false)
    @test dense.strategy == :env_first
    @test dense.flops == 60
    @test dense.sector_flops == 30
    @test sector.strategy == :sector_exact
    @test sector.flops == 63
    @test sector.sector_flops == 28
    @test sort([sector.steps[1].a, sector.steps[1].b]) == [2, 3]
    @test sector.peak_elements <= envfirst.peak_elements
    @test sector.sector_peak_elements <= envfirst.sector_peak_elements

    rng = MersenneTwister(1801)
    At = randn(rng, ComplexF64, A)
    Bt = randn(rng, ComplexF64, B)
    Ct = randn(rng, ComplexF64, C)
    exact_map = _Planning.EffectiveMap(sector, (Bt, Ct))
    exact_reference = _Planning.ncon_reference(spec, At, (Bt, Ct))
    @test norm(exact_map(At) - exact_reference) <= 1e-12 * max(norm(exact_reference), 1)
end

@graft_extended_testset "compiled contraction plans: sector-aware structural planning" begin
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
    profile = Graft.Backend.pair_cost(A0, pA, false, B0, pB, false, pAB)
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
    root_profile = Graft.Backend.pair_cost(
        Ar, ((1, 2), (3,)), false,
        Br, ((1,), (2, 3)), false,
        ((1, 3), (2, 4)),
    )
    @test root_profile.largest_block_elements == 8
    @test root_profile.output_largest_block_elements == 9
    @test root_profile.peak_block_elements == 9

    A, B, C, spec = _sector_three_map_fixture()

    # In the supported symmetric-braiding, λ_perm=0 model, canonicalizing an
    # intermediate output makes A×B and B×A the same structural cost state.
    # This is what removes high-degree leaf-order factorials without dropping
    # a physical contraction: PairStep.out materializes that canonical order.
    dimsA, dimsB = _Planning._prototype_dims(A), _Planning._prototype_dims(B)
    metricsAB = _Planning.dense_cost([-1, 1], dimsA, [1, -2], dimsB)
    metricsBA = _Planning.dense_cost([1, -2], dimsB, [-1, 1], dimsA)
    profileAB = _Planning._sector_pair_profile(
        metricsAB, A, false, B, false,
        _Planning._canonical_intermediate_partition(metricsAB.labels),
    )
    profileBA = _Planning._sector_pair_profile(
        metricsBA, B, false, A, false,
        _Planning._canonical_intermediate_partition(metricsBA.labels),
    )
    @test profileAB.output == profileBA.output
    @test profileAB.sector_flops == profileBA.sector_flops

    # Compare connected exact-DP candidates with an independent enumeration
    # of *all* ordered four-map trees, including early outer products.  The
    # star supplies the important bridge structure that makes a leaf-only
    # outer product possible.  Equality certifies the connected normal form
    # for this supported cost model rather than merely repeating the DP's
    # own shared-label restriction in the oracle.
    Vstar = U1Space(0 => 1, 1 => 1)
    center4 = Vstar ← reduce(⊗, ntuple(_ -> Vstar, 3))
    leaf4 = Vstar ← Vstar
    protos4 = (center4, leaf4, leaf4, leaf4)
    spec4 = _Planning.ContractionSpec(
        Vector{Int}[[-1, 1, 2, 3], [1, -2], [2, -3], [3, -4]],
        Bool[false, false, false, false], 4, (4, 0), 1;
        preferred_slots=[2, 3, 4],
    )
    envfirst4 = _Planning.plan_contraction(spec4, protos4; optimize=false)
    dp4 = _sector_dp_score(spec4, protos4, envfirst4; memory_weight=1)
    oracle4 = _sector_oracle_score(spec4, protos4, envfirst4; memory_weight=1)
    selected4 = _Planning.plan_contraction(spec4, protos4;
                                            sector_aware=true, memory_weight=1)
    @test dp4 ≈ oracle4
    @test _Planning._sector_score(selected4, 1) ≈ oracle4

    # Phase 3's exact scope is the same local ≤10-tensor scope as the dense
    # optimizer.  This is a high-degree 8-tensor U(1) star, rather than an
    # easy chain: canonical intermediate legs make the formerly factorial
    # leaf-order search collapse to a small exact frontier with no data tensor.
    nleaves8 = 7
    center8 = Vstar ← reduce(⊗, ntuple(_ -> Vstar, nleaves8))
    protos8 = (center8, ntuple(_ -> Vstar ← Vstar, nleaves8)...)
    labels8 = Vector{Int}[vcat([-1], collect(1:nleaves8))]
    for i in 1:nleaves8
        push!(labels8, [i, -(i + 1)])
    end
    spec8 = _Planning.ContractionSpec(labels8, falses(nleaves8 + 1),
                                       nleaves8 + 1, (nleaves8 + 1, 0), 1;
                                       preferred_slots=collect(2:(nleaves8 + 1)))
    dims8, _ = _Planning._label_dimensions(spec8, protos8)
    @test _Planning._SECTOR_EXACT_TENSOR_LIMIT == 10
    @test !isempty(_Planning._sector_dp_trees(spec8, dims8, protos8))

    # A sector type without an actual split must retain Phase 2 instead of
    # spending an exact-DP miss on a single dense charge-0 block.
    V0 = U1Space(0 => 1)
    W0 = V0 ← V0
    @test Graft.Backend.sector_cost_supported(W0)
    @test !Graft.Backend.sector_cost_nontrivial(W0)
    spec0 = _Planning.ContractionSpec(
        Vector{Int}[[-1, 1], [1, 2], [2, -2]],
        Bool[false, false, false], 2, (1, 1), 1;
        preferred_slots=[2, 3],
    )
    @test _Planning.plan_contraction(spec0, (W0, W0, W0);
                                     sector_aware=true).strategy != :sector_exact

    # Exact Pareto retention has no arbitrary frontier-length cap.  These
    # nine otherwise identical structural states trade sector FLOPs against
    # both dense and sector peak, so none is mathematically dominated.
    frontier = _Planning._SectorDPState[]
    for i in 1:9
        state = _Planning._SectorDPState(
            i, [-1, 1], [2, 2], Vstar ← Vstar, false,
            Float64(i), Float64(i), Float64(i), Float64(10 - i), Float64(10 - i),
        )
        _Planning._insert_sector_state!(frontier, state)
    end
    @test length(frontier) == 9

    key_dense = _Planning.plan_key(:sector_fixture, spec, (A, B, C), ComplexF64;
                                   sector_aware=false, memory_weight=0)
    key_sector = _Planning.plan_key(:sector_fixture, spec, (A, B, C), ComplexF64;
                                    sector_aware=true, memory_weight=0)
    @test key_dense != key_sector
end

@graft_testset "compiled contraction plans: cache, graph identity, and gauge" begin
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
    @test key_envfirst != key_memheavy

    h1_default = eff_h1(cache, ψ, O, n)
    if GRAFT_EXTENDED_TESTS
        after_miss = _CP.plan_cache_stats(cache)
        @test after_miss.misses == 1
        @test after_miss.hits == 0
        h1_repeat = eff_h1(cache, ψ, O, n)
        after_hit = _CP.plan_cache_stats(cache)
        @test after_hit.misses == 1
        @test after_hit.hits == 1
        @test h1_default.plan.steps === h1_repeat.plan.steps
    end

    # Public effective-map keywords feed the cache objective: even when the
    # selected dense tree happens to equal env-first for this small network,
    # a caller requesting Phase-1-only planning gets a distinct cache entry.
    h1_plans_after_default = count(key -> key.kind === :h1, keys(cache.plans))
    h1_envfirst = eff_h1(cache, ψ, O, n; optimize=false)
    @test h1_envfirst.plan.strategy == :env_first
    @test count(key -> key.kind === :h1, keys(cache.plans)) ==
          h1_plans_after_default + 1

    # A same-shape tensor-value update invalidates environments but keeps the
    # compiled shape plan reusable.
    h1_plans_before_update = count(key -> key.kind === :h1, keys(cache.plans))
    update_tensor!(ψ, n, 1.01 * ψ.tensors[n]; caches=(cache,))
    @test count(key -> key.kind === :h1, keys(cache.plans)) == h1_plans_before_update
    h1_reused = eff_h1(cache, ψ, O, n)
    @test h1_reused.plan.steps === h1_default.plan.steps
    @test count(key -> key.kind === :h1, keys(cache.plans)) == h1_plans_before_update

    # Same topology but a different state-bond shape must miss rather than
    # reuse a plan; clearing only environments preserves the old plan for this
    # exact cache-key check.
    ψwide = random_ttns(MersenneTwister(1502), ComplexF64, topo, phys, ℂ^3)
    h1_plans_before_shape_change = count(key -> key.kind === :h1, keys(cache.plans))
    _CP._clear_value_environments!(cache)
    _assert_env_payload_accounting(cache)
    _ = eff_h1(cache, ψwide, O, n)
    @test count(key -> key.kind === :h1, keys(cache.plans)) ==
          h1_plans_before_shape_change + 1

    if GRAFT_EXTENDED_TESTS
        after_shape_change = _CP.plan_cache_stats(cache)
        @test after_shape_change.misses == 3
        @test after_shape_change.hits == 2
        @test after_shape_change.size == length(cache.plans)
    end

    # H2 on two symmetric root children has equal dense dimensions but a
    # different Wm crossed-child leg. The label graph is therefore part of the
    # key; reusing one plan here would be a silent dimension-valid bug.
    c1, c2 = topo.children[topo.root]
    cache_h2 = EnvCache(topo)
    move_center!(ψ, c1; cache=cache_h2)
    h2a = eff_h2(cache_h2, ψ, O, c1, topo.root)
    move_center!(ψ, c2; cache=cache_h2)
    h2b = eff_h2(cache_h2, ψ, O, c2, topo.root)
    @test count(key -> key.kind === :h2, keys(cache_h2.plans)) == 2
    @test h2a.plan.steps !== h2b.plan.steps

    if GRAFT_EXTENDED_TESTS
        @test _CP.plan_cache_stats(cache_h2).misses == 2

        # A closed gauge excursion preserves the local effective observable.
        move_center!(ψ, n)
        e0 = dot(ψ.tensors[n], eff_h1(EnvCache(topo), ψ, O, n)(ψ.tensors[n]))
        move_center!(ψ, c1)
        move_center!(ψ, n)
        e1 = dot(ψ.tensors[n], eff_h1(EnvCache(topo), ψ, O, n)(ψ.tensors[n]))
        @test e0 ≈ e1 rtol=1e-12 atol=1e-12
    end
end

function _concurrent_cache_requests(request, count::Int)
    gate = Channel{Nothing}(count)
    results = Vector{Any}(undef, count)
    @sync begin
        for i in 1:count
            Base.Threads.@spawn begin
                take!(gate)
                results[i] = request(i)
            end
        end
        for _ in 1:count
            put!(gate, nothing)
        end
    end
    return results
end

function _repeated_map_allocations(map_, input, repetitions::Int)
    output = Ref{Any}()
    bytes = @allocated begin
        for _ in 1:repetitions
            output[] = map_(input)
        end
    end
    return output[], bytes
end

@graft_testset "compiled contraction plans: concurrent plan and root-cap cache" begin
    A, B, C, spec = _sector_three_map_fixture()
    cache = EnvCache(star_topology(2, 1))
    request_count = max(8, 2Base.Threads.nthreads())

    maps = _concurrent_cache_requests(request_count) do _
        _CP._effective_map!(
            cache, :concurrent_equal_key, spec, (A, B, C), (B, C),
            ComplexF64,
        )
    end
    diagnostics = cache_diagnostics(cache)
    @test diagnostics isa CacheDiagnostics
    @test diagnostics.plan_entries == 1
    @test diagnostics.shape_plan_hits + diagnostics.shape_plan_misses ==
          request_count
    @test diagnostics.effective_plan_hits +
          diagnostics.effective_plan_misses == request_count
    @test diagnostics.plan_duplicate_builds <=
          max(diagnostics.shape_plan_misses - 1, 0)
    @test all(map -> map.plan.steps === maps[1].plan.steps, maps)
    planner_diagnostics = plan_diagnostics(maps[1].plan)
    @test planner_diagnostics isa PlannerDiagnostics
    @test planner_diagnostics.strategy == maps[1].plan.strategy
    @test planner_diagnostics.classification == :selected
    @test isempty(planner_diagnostics.candidate_failures)

    empty!(cache)
    distinct_count = 8
    distinct_maps = _concurrent_cache_requests(distinct_count) do i
        _CP._effective_map!(
            cache, Symbol(:concurrent_distinct_key_, i), spec,
            (A, B, C), (B, C), ComplexF64,
        )
    end
    diagnostics = cache_diagnostics(cache)
    @test diagnostics.plan_entries == distinct_count
    @test diagnostics.shape_plan_hits == 0
    @test diagnostics.shape_plan_misses == distinct_count
    @test diagnostics.plan_duplicate_builds == 0
    @test all(map -> map isa EffectiveMap, distinct_maps)

    empty!(cache)
    capspace = ℂ^1 ⊗ ℂ^1
    caps = _concurrent_cache_requests(request_count) do _
        _CP._root_cap!(cache, ComplexF64, capspace)
    end
    diagnostics = cache_diagnostics(cache)
    @test diagnostics.rootcap_entries == 1
    @test diagnostics.rootcap_hits + diagnostics.rootcap_misses == request_count
    @test diagnostics.rootcap_duplicate_builds <=
          max(diagnostics.rootcap_misses - 1, 0)
    @test all(cap -> cap === caps[1], caps)

    empty!(cache)
    cleared = cache_diagnostics(cache)
    @test cleared.plan_entries == 0
    @test cleared.rootcap_entries == 0
    @test cleared.effective_plan_hits == 0
    @test cleared.effective_plan_misses == 0
    @test cleared.shape_plan_hits == 0
    @test cleared.shape_plan_misses == 0
    @test cleared.plan_duplicate_builds == 0
    @test cleared.rootcap_hits == 0
    @test cleared.rootcap_misses == 0
    @test cleared.rootcap_duplicate_builds == 0
    @test cleared.lock_contentions == 0
    @test cleared.stale_build_discards == 0
    @test cleared.environments.entries == 0
    @test cleared.environments.clear_generation > 0
    legacy = CacheDiagnostics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    @test legacy.environments ==
          EnvironmentCacheDiagnostics(
              0, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 0, 0, 0, :none, UInt64(0), UInt64(0))
end

@graft_testset "compiled contraction plans: environment dependency generations" begin
    topo = star_topology(3, 2)
    phys = allspin(topo)
    O = ttno_from_opsum(tfi(topo; g=0.37), topo, phys; hermitian=true)
    ket = random_ttns(MersenneTwister(1701), ComplexF64, topo, phys, ℂ^3)
    bra = random_ttns(MersenneTwister(1702), ComplexF64, topo, phys, ℂ^3)
    root = topo.root
    excluded_child = first(topo.children[root])
    key = (root, excluded_child)

    serial_cache = EnvCache(topo)
    serial = _CP.env!(serial_cache, ket, O, bra, key...)

    cache = EnvCache(topo)
    request_count = max(8, 2Base.Threads.nthreads())
    concurrent = _concurrent_cache_requests(request_count) do _
        _CP.env!(cache, ket, O, bra, key...)
    end
    @test all(E -> E === concurrent[1], concurrent)
    @test norm(concurrent[1] - serial) <= 1e-12 * max(norm(serial), 1)
    diagnostics = cache_diagnostics(cache).environments
    @test diagnostics isa EnvironmentCacheDiagnostics
    @test diagnostics.entries == length(cache.envs)
    @test diagnostics.rebuilds == diagnostics.entries
    @test diagnostics.hits + diagnostics.misses >= request_count
    @test diagnostics.duplicate_builds <=
          max(diagnostics.misses - diagnostics.entries, 0)
    @test diagnostics.retry_exhaustions == 0

    # Sibling-final contractions stage into task-local workspaces only after a
    # complete predicted-live-byte admission check. Results merge in the
    # topology's fixed neighbor order.
    @test_throws ArgumentError EnvCache(topo; threaded_envs=true)
    staged_cache = EnvCache(
        topo;
        threaded_envs=true,
        env_staging_minbatch=2,
        env_staging_memory_cap_bytes=typemax(Int),
    )
    staged = _CP.env!(staged_cache, ket, O, bra, key...)
    @test norm(staged - serial) <= 1e-12 * max(norm(serial), 1)
    _assert_env_payload_accounting(staged_cache)
    staged_diag = cache_diagnostics(staged_cache).environments
    if Base.Threads.nthreads() > 1
        @test staged_diag.staged_batches >= 1
        @test staged_diag.staged_tasks >= 2
        @test staged_diag.staged_admitted_bytes > 0
        @test staged_diag.last_fallback == :none
    else
        @test staged_diag.staged_batches == 0
        @test staged_diag.serial_fallbacks >= 1
        @test staged_diag.last_fallback == :single_thread
    end

    # A worker exception happens before the deterministic commit phase, so no
    # sibling-final candidate can leak into the cache. The classified failure
    # remains observable and the same cache is immediately reusable.
    if Base.Threads.nthreads() > 1
        failure_cache = EnvCache(
            topo;
            threaded_envs=true,
            env_staging_minbatch=2,
            env_staging_memory_cap_bytes=typemax(Int),
        )
        sibling_keys = Tuple{Int,Int}[
            (w, root) for w in neighbors(topo, root)
            if w != excluded_child
        ]
        @test_throws CompositeException _CP._ensure_environment_prerequisites!(
            failure_cache,
            ket,
            O,
            bra,
            key...;
            execute_candidate=_ -> error("injected staging worker failure"),
        )
        @test all(k -> !haskey(failure_cache, k), sibling_keys)
        failure_diag = cache_diagnostics(failure_cache).environments
        @test failure_diag.staged_batches == 0
        @test failure_diag.staged_tasks == 0
        @test failure_diag.serial_fallbacks == 1
        @test failure_diag.last_fallback == :worker_failure
        _assert_env_payload_accounting(failure_cache)
        recovered = _CP.env!(failure_cache, ket, O, bra, key...)
        @test norm(recovered - serial) <= 1e-12 * max(norm(serial), 1)
        _assert_env_payload_accounting(failure_cache)
    end

    capped_cache = EnvCache(
        topo;
        threaded_envs=true,
        env_staging_minbatch=2,
        env_staging_memory_cap_bytes=0,
    )
    capped = _CP.env!(capped_cache, ket, O, bra, key...)
    @test norm(capped - serial) <= 1e-12 * max(norm(serial), 1)
    capped_diag = cache_diagnostics(capped_cache).environments
    @test capped_diag.staged_batches == 0
    @test capped_diag.serial_fallbacks >= 1
    @test capped_diag.last_fallback in (:memory_cap, :single_thread)

    # Bounded high-contention invalidation stress: all readers either hit a
    # valid generation or retry, and every returned value equals the serial
    # reference. The invalidator changes no tensors, so numerical equality is
    # exact up to the ordinary contraction tolerance.
    stress_cache = EnvCache(topo)
    stress_count = max(8, 2Base.Threads.nthreads())
    stress_u, stress_v = key
    stress_results = _concurrent_cache_requests(stress_count) do i
        if i <= 4
            _CP.invalidate_node!(stress_cache, root)
            yield()
        end
        _CP.env!(stress_cache, ket, O, bra, stress_u, stress_v)
    end
    @test all(
        E -> norm(E - serial) <= 1e-12 * max(norm(serial), 1),
        stress_results,
    )
    stress_diag = cache_diagnostics(stress_cache).environments
    @test stress_diag.retry_exhaustions == 0
    @test stress_diag.entries == length(stress_cache.envs)
    _assert_env_payload_accounting(stress_cache)

    # An invalidation outside the dependency component does not invalidate its
    # captured stamp; a relevant node update does.
    leaves_ = collect(Graft.leaves(topo))
    dependent_leaf = first(leaves_)
    unrelated_leaf = first(filter(!=(dependent_leaf), leaves_))
    leaf_key = (dependent_leaf, topo.parent[dependent_leaf])
    leaf_cache = EnvCache(topo)
    leaf_env = _CP.env!(leaf_cache, ket, O, bra, leaf_key...)
    leaf_stamp = leaf_cache.env_stamps[leaf_key]
    @test leaf_stamp !== nothing
    _CP.invalidate_node!(leaf_cache, unrelated_leaf)
    @test _CP._environment_stamp_is_current(
        leaf_cache, leaf_stamp, ket, O, bra)

    update_tensor!(
        ket,
        dependent_leaf,
        1.01 * ket.tensors[dependent_leaf];
        gauge=false,
        caches=(leaf_cache,),
    )
    @test !_CP._environment_stamp_is_current(
        leaf_cache, leaf_stamp, ket, O, bra)
    rejected = _CP._commit_environment_candidate!(
        leaf_cache, leaf_key, copy(leaf_env), leaf_stamp, ket, O, bra)
    @test rejected.status == :stale
    @test !haskey(leaf_cache, leaf_key)
    _assert_env_payload_accounting(leaf_cache)
    rebuilt = _CP.env!(leaf_cache, ket, O, bra, leaf_key...)
    rebuilt_ref = _CP._build_env_ncon_reference(
        ket, O, bra, leaf_key..., leaf_cache.envs)
    @test norm(rebuilt - rebuilt_ref) <=
          1e-12 * max(norm(rebuilt_ref), 1)
    _assert_env_payload_accounting(leaf_cache)

    # Tensor identity is checked in addition to explicit generations, closing
    # the write-before-invalidate window and rejecting unannounced replacement.
    identity_cache = EnvCache(topo)
    identity_old = _CP.env!(identity_cache, ket, O, bra, leaf_key...)
    ket.tensors[dependent_leaf] = 0.99 * ket.tensors[dependent_leaf]
    identity_new = _CP.env!(identity_cache, ket, O, bra, leaf_key...)
    @test identity_new !== identity_old
    identity_diag = cache_diagnostics(identity_cache).environments
    @test identity_diag.stale_build_discards >= 1
    _assert_env_payload_accounting(identity_cache)

    # Cache-wide clear generations reject a candidate captured before clear.
    clear_cache = EnvCache(topo)
    clear_candidate = _CP.env!(clear_cache, ket, O, bra, leaf_key...)
    clear_stamp = clear_cache.env_stamps[leaf_key]
    empty!(clear_cache)
    clear_rejected = _CP._commit_environment_candidate!(
        clear_cache, leaf_key, copy(clear_candidate), clear_stamp,
        ket, O, bra)
    @test clear_rejected.status == :stale
    @test isempty(clear_cache.envs)
    clear_diag = cache_diagnostics(clear_cache).environments
    @test clear_diag.stale_build_discards == 1
    @test clear_diag.retry_exhaustions == 0
    _assert_env_payload_accounting(clear_cache)

    # Clearing inside an overlapping outer transaction invalidates staged work
    # without corrupting the active-transaction count.
    transaction_cache = EnvCache(topo)
    entered = Channel{Nothing}(1)
    release = Channel{Nothing}(1)
    transaction = Base.Threads.@spawn begin
        _CP._with_env_transaction(transaction_cache) do
            put!(entered, nothing)
            take!(release)
        end
    end
    take!(entered)
    empty!(transaction_cache)
    put!(release, nothing)
    fetch(transaction)
    @test transaction_cache.transaction_depth == 0
    @test isempty(transaction_cache.envs)
    _assert_env_payload_accounting(transaction_cache)
end

@graft_testset "compiled contraction plans: repeated matvec allocation flatness" begin
    topo = mps_topology(4)
    phys = allspin(topo)
    O = ttno_from_opsum(tfi(topo; g=0.43), topo, phys; hermitian=true)
    ψ = random_ttns(MersenneTwister(1751), ComplexF64, topo, phys, ℂ^3)
    target = first(topo.children[topo.root])
    move_center!(ψ, target)
    cache = EnvCache(topo)
    map_ = eff_h1(cache, ψ, O, target)
    input = ψ.tensors[target]
    map_(input)
    map_(input)

    cache_before = cache_diagnostics(cache)
    GC.gc()
    y4, allocated4 = _repeated_map_allocations(map_, input, 4)
    GC.gc()
    y8, allocated8 = _repeated_map_allocations(map_, input, 8)
    cache_after = cache_diagnostics(cache)
    @test norm(y8 - y4) <= 1e-12 * max(norm(y4), 1)
    @test allocated4 > 0
    @test allocated8 <= 2allocated4 + 65_536
    @test cache_after.plan_entries == cache_before.plan_entries
    @test cache_after.rootcap_entries == cache_before.rootcap_entries
    @test cache_after.shape_plan_hits == cache_before.shape_plan_hits
    @test cache_after.shape_plan_misses == cache_before.shape_plan_misses

    workspace = workspace_map(map_)
    workspace(input)
    workspace(input)
    workspace_before = workspace_stats(workspace.workspace)
    GC.gc()
    wy4, workspace_allocated4 =
        _repeated_map_allocations(workspace, input, 4)
    GC.gc()
    wy8, workspace_allocated8 =
        _repeated_map_allocations(workspace, input, 8)
    workspace_after = workspace_stats(workspace.workspace)
    @test norm(wy8 - y8) <= 1e-12 * max(norm(y8), 1)
    @test norm(wy4 - y4) <= 1e-12 * max(norm(y4), 1)
    @test workspace_allocated8 <= 2workspace_allocated4 + 65_536
    @test workspace_after.allocations == workspace_before.allocations
    @test workspace_after.buffers == workspace_before.buffers
end

@graft_testset "compiled contraction plans: mixed-boson post-TDVP2 root h1" begin
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
    Θnext, _ = Graft.Evolution.exponentiate(h2, -0.06, Θ;
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
    y_planned, _ = Graft.Evolution.exponentiate(checked.planned, 0.06, x;
                                                  ishermitian=true,
                                                  krylovdim=12,
                                                  tol=1e-12)
    y_reference, _ = Graft.Evolution.exponentiate(reference_h1, 0.06, x;
                                                    ishermitian=true,
                                                    krylovdim=12,
                                                    tol=1e-12)
    @test norm(y_planned - y_reference) <= 1e-10 * max(norm(y_reference), 1)
end

@graft_testset "compiled contraction plans: non-square TDVP h0 link" begin
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
    tensors = Vector{Graft.Backend.AbstractTensorMap}(undef, nnodes(topo))
    tensors[child] = Achild
    tensors[root] = Aroot
    ψ = TTNS(topo, tensors, child)

    phys = Dict(nodeid(topo, i) => P for i in 1:nnodes(topo))
    O = ttno_from_opsum(tfi(topo; g=0.37), topo, phys; hermitian=true)
    cache = EnvCache(topo)

    # The private helper is deliberately used here because this is the exact
    # TDVP seam whose returned C is passed to eff_h0 immediately afterward.
    C = Graft.Evolution._split_link_up(TDVP1(), ψ, O, child, root, -0.1)
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

@graft_testset "compiled contraction plans: fork-spine memory objective" begin
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
    @test !Graft.Backend.sector_cost_nontrivial(x)
    @test envfirst.sector_peak_elements == envfirst.peak_elements
    @test envfirst.sector_flops == 2 * envfirst.flops
    @test envfirst.peak_elements < gate
    @test selected.peak_elements <= envfirst.peak_elements
end
