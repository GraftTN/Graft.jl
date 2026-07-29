using Graft

using Graft.Backend: ℂ, pair_workload_profile
using Graft.TestUtils: random_ttns
using LinearAlgebra: BLAS, norm
using Random: MersenneTwister
using Statistics: median

const SAMPLES = parse(
    Int, get(ENV, "GRAFT_CONTRACTION_BASELINE_SAMPLES", "5"))
const BOND_DIMENSION = parse(
    Int, get(ENV, "GRAFT_CONTRACTION_BASELINE_BOND_DIM", "8"))
const REPEATED_MATVECS = parse(
    Int, get(ENV, "GRAFT_CONTRACTION_BASELINE_MATVECS", "16"))
const STAGING_CAP_BYTES = parse(
    Int, get(ENV, "GRAFT_CONTRACTION_STAGING_CAP_BYTES", "20000000000"))
const SELECTED_CASE = get(ENV, "GRAFT_CONTRACTION_CASE", "all")

SAMPLES >= 1 || error("baseline samples must be positive")
BOND_DIMENSION >= 1 || error("baseline bond dimension must be positive")
REPEATED_MATVECS >= 2 || error("baseline matvec count must be at least two")
STAGING_CAP_BYTES >= 0 || error("staging cap must be nonnegative")

function fixture(topology, seed)
    spin = spin_ops()
    physical_spaces = Dict(
        nodeid(topology, i) => spin.P for i in 1:nnodes(topology))
    terms = OpSum()
    for node in 1:nnodes(topology)
        terms += Term(
            0.23 + 0.003node,
            SiteOp(nodeid(topology, node), :X, spin.X),
        )
        terms += Term(
            -0.09,
            SiteOp(nodeid(topology, node), :Z, spin.Z),
        )
    end
    for (child, parent) in Graft.Trees.edges(topology)
        terms += Term(
            0.31 + 0.002child,
            SiteOp(nodeid(topology, child), :Z, spin.Z),
            SiteOp(nodeid(topology, parent), :Z, spin.Z),
        )
    end
    operator = ttno_from_opsum(
        terms, topology, physical_spaces; hermitian=true)
    state = random_ttns(
        MersenneTwister(seed),
        ComplexF64,
        topology,
        physical_spaces,
        ℂ^BOND_DIMENSION,
    )
    normalize!(state)
    target = isempty(topology.children[topology.root]) ?
        topology.root : first(topology.children[topology.root])
    move_center!(state, target)
    return state, operator, target
end

function measured_value(f)
    value = Ref{Any}()
    bytes = @allocated value[] = f()
    return value[], bytes
end

function repeated_allocations(f, input, repetitions)
    output = Ref{Any}()
    bytes = @allocated begin
        for _ in 1:repetitions
            output[] = f(input)
        end
    end
    return output[], bytes
end

function repeated_timed(f, input, repetitions)
    output = Ref{Any}()
    measurement = @timed begin
        for _ in 1:repetitions
            output[] = f(input)
        end
    end
    return output[], measurement
end

function _histogram_increment!(histogram, key)
    histogram[key] = get(histogram, key, 0) + 1
    return histogram
end

function _histogram_text(histogram)
    isempty(histogram) && return "none"
    return join(
        ("$(key):$(histogram[key])" for key in sort!(collect(keys(histogram)))),
        ",",
    )
end

function workload_histograms(plan, operands)
    ninputs = plan.nslots - length(plan.steps)
    length(operands) == ninputs ||
        error("workload histogram operand count mismatch")
    slots = Vector{Any}(undef, plan.nslots)
    for (index, operand) in enumerate(operands)
        slots[index] = operand
    end
    gemms = Dict{String,Int}()
    permutations = Dict{String,Int}()
    for step in plan.steps
        profile = pair_workload_profile(
            slots[step.a],
            (step.oindA, step.cindA),
            step.conjA,
            slots[step.b],
            (step.cindB, step.oindB),
            step.conjB,
            step.out,
        )
        for gemm in profile.gemms
            _histogram_increment!(gemms, "$(gemm.m)x$(gemm.k)x$(gemm.n)")
        end
        for (role, elements) in pairs(profile.transforms)
            _histogram_increment!(permutations, "$(role)=$(elements)")
        end
        slots[step.dst] = profile.output
        slots[step.a] = nothing
        slots[step.b] = nothing
    end
    return (;
        gemm_mkn=_histogram_text(gemms),
        permutation_elements=_histogram_text(permutations),
    )
end

function construction_sample(state, operator, target; threaded)
    cache = EnvCache(
        topology(state);
        threaded_envs=threaded,
        env_staging_minbatch=2,
        env_staging_memory_cap_bytes=
            threaded ? STAGING_CAP_BYTES : nothing,
    )
    start = time_ns()
    map_, allocation_bytes = measured_value() do
        eff_h1(cache, state, operator, target)
    end
    elapsed_seconds = (time_ns() - start) / 1e9
    return (; cache, map_, allocation_bytes, elapsed_seconds)
end

function baseline(name, topology, seed)
    state, operator, target = fixture(topology, seed)
    fixture_peak_rss_bytes = Int(Sys.maxrss())

    # Exercise both paths before the sampled construction cells, then
    # alternate their order so clock/thermal drift cannot masquerade as a
    # staging benefit.
    warm = construction_sample(state, operator, target; threaded=false)
    construction_sample(state, operator, target; threaded=true)
    input = state.tensors[target]
    warm.map_(input)

    serial_samples = Any[]
    staged_samples = Any[]
    for sample in 1:SAMPLES
        first_threaded = iseven(sample)
        first = construction_sample(
            state, operator, target; threaded=first_threaded)
        second = construction_sample(
            state, operator, target; threaded=!first_threaded)
        if first_threaded
            push!(staged_samples, first)
            push!(serial_samples, second)
        else
            push!(serial_samples, first)
            push!(staged_samples, second)
        end
    end
    serial_times = getproperty.(serial_samples, :elapsed_seconds)
    staged_times = getproperty.(staged_samples, :elapsed_seconds)
    serial_median = median(serial_times)
    staged_median = median(staged_times)
    serial = serial_samples[argmin(
        abs(sample.elapsed_seconds -
            serial_median)
        for sample in serial_samples)]
    staged = staged_samples[argmin(
        abs(sample.elapsed_seconds -
            staged_median)
        for sample in staged_samples)]

    map_ = serial.map_
    map_(input)
    GC.gc()
    one_result, one_matvec_bytes = measured_value() do
        map_(input)
    end
    GC.gc()
    repeated_result, repeated_measurement = repeated_timed(
        map_, input, REPEATED_MATVECS)
    workspace = workspace_map(map_)
    workspace(input)
    workspace(input)
    workspace_before = workspace_stats(workspace.workspace)
    GC.gc()
    workspace_result, workspace_measurement = repeated_timed(
        workspace, input, REPEATED_MATVECS)
    workspace_after = workspace_stats(workspace.workspace)

    scale = max(norm(one_result), 1.0)
    norm(one_result - repeated_result) <= 1e-12scale ||
        error("$name repeated matvec result changed")
    norm(one_result - workspace_result) <= 1e-12scale ||
        error("$name workspace matvec result changed")
    workspace_after.allocations == workspace_before.allocations ||
        error("$name workspace allocated new persistent buffers after warm-up")

    plan = plan_diagnostics(map_.plan)
    histograms = workload_histograms(
        map_.plan, (input, map_.statics...))
    serial_cache = cache_diagnostics(serial.cache)
    staged_cache = cache_diagnostics(staged.cache)
    runtime = parallel_runtime_config()
    peak_rss_bytes = Int(Sys.maxrss())
    println(
        "GRAFT_CONTRACTION_BASELINE ",
        "topology=$name ",
        "nodes=$(nnodes(topology)) ",
        "target=$(nodeid(topology, target)) ",
        "bond_dimension=$BOND_DIMENSION ",
        "julia_threads=$(Threads.nthreads()) ",
        "blas_vendor=$(runtime.blas_vendor) ",
        "blas_threads=$(runtime.blas_threads) ",
        "strided_threads=$(runtime.strided_threads) ",
        "julia_version=$(runtime.julia_version) ",
        "machine=$(runtime.machine) ",
        "heap_size_hint_bytes=$(Int(Base.JLOptions().heap_size_hint)) ",
        "samples=$SAMPLES ",
        "repeated_matvecs=$REPEATED_MATVECS ",
        "serial_construction_median_seconds=$(repr(serial_median)) ",
        "staged_construction_median_seconds=$(repr(staged_median)) ",
        "staged_speedup=$(repr(serial_median / staged_median)) ",
        "serial_seconds=$(join(repr.(serial_times), ',')) ",
        "staged_seconds=$(join(repr.(staged_times), ',')) ",
        "serial_construction_alloc_bytes=$(serial.allocation_bytes) ",
        "staged_construction_alloc_bytes=$(staged.allocation_bytes) ",
        "one_matvec_alloc_bytes=$one_matvec_bytes ",
        "repeated_matvec_seconds=$(repr(repeated_measurement.time)) ",
        "repeated_matvec_alloc_bytes=$(repeated_measurement.bytes) ",
        "repeated_matvec_gc_seconds=$(repr(repeated_measurement.gctime)) ",
        "repeated_matvec_gc_pauses=$(repeated_measurement.gcstats.pause) ",
        "repeated_matvec_gc_full_sweeps=$(repeated_measurement.gcstats.full_sweep) ",
        "workspace_repeated_seconds=$(repr(workspace_measurement.time)) ",
        "workspace_repeated_alloc_bytes=$(workspace_measurement.bytes) ",
        "workspace_repeated_gc_seconds=$(repr(workspace_measurement.gctime)) ",
        "workspace_repeated_gc_pauses=$(workspace_measurement.gcstats.pause) ",
        "workspace_repeated_gc_full_sweeps=$(workspace_measurement.gcstats.full_sweep) ",
        "workspace_persistent_allocations=$(workspace_after.allocations) ",
        "fixture_peak_rss_bytes=$fixture_peak_rss_bytes ",
        "peak_rss_bytes=$peak_rss_bytes ",
        "post_fixture_peak_rss_growth_bytes=$(max(0, peak_rss_bytes - fixture_peak_rss_bytes)) ",
        "gemm_mkn_histogram=$(histograms.gemm_mkn) ",
        "permutation_element_histogram=$(histograms.permutation_elements) ",
        "plan_strategy=$(plan.strategy) ",
        "plan_classification=$(plan.classification) ",
        "plan_live_peak_bytes=$(repr(plan.live_peak_bytes)) ",
        "plan_sector_live_peak_bytes=$(repr(plan.sector_live_peak_bytes)) ",
        "serial_env_entries=$(serial_cache.environments.entries) ",
        "serial_env_hits=$(serial_cache.environments.hits) ",
        "serial_env_misses=$(serial_cache.environments.misses) ",
        "staged_env_batches=$(staged_cache.environments.staged_batches) ",
        "staged_env_tasks=$(staged_cache.environments.staged_tasks) ",
        "staged_admitted_bytes=$(staged_cache.environments.staged_admitted_bytes) ",
        "staged_serial_fallbacks=$(staged_cache.environments.serial_fallbacks) ",
        "staged_last_fallback=$(staged_cache.environments.last_fallback)",
    )
end

configure_parallel_runtime!(; blas_threads=1, strided_threads=1)
cases = (
    ("chain", mps_topology(8), 1801),
    ("star", star_topology(4, 2), 1802),
    ("fork", fork_topology(3, 2), 1803),
    ("balanced", binary_topology(3), 1804),
)
for (name, topology, seed) in cases
    SELECTED_CASE == "all" || SELECTED_CASE == name || continue
    baseline(name, topology, seed)
end
