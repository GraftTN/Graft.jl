using Graft
using Graft.Backend: U1Space, blocks, ←
using Graft.Contractions: _rsvd_random_probe
using Random: Xoshiro, rand
using SHA
using Statistics: median

const WORKLOAD = Symbol(get(ENV, "GRAFT_P1_WORKLOAD", "thermal"))
const MODE = Symbol(get(ENV, "GRAFT_P1_MODE", "serial"))
const SAMPLES = parse(Int, get(ENV, "GRAFT_P1_SAMPLES", "5"))
const MEMORY_CAP_BYTES = parse(
    Int, get(ENV, "GRAFT_P1_MEMORY_CAP_BYTES", "2147483648"))
const TASK_WORKSPACE_BYTES = parse(
    Int, get(ENV, "GRAFT_P1_TASK_WORKSPACE_BYTES",
             WORKLOAD === :thermal ? "25165824" : "1048576"))

WORKLOAD in (:thermal, :rsvd) ||
    error("GRAFT_P1_WORKLOAD must be thermal or rsvd")
MODE in (:serial, :threaded) ||
    error("GRAFT_P1_MODE must be serial or threaded")
SAMPLES >= 3 || error("GRAFT_P1_SAMPLES must be at least three")
MEMORY_CAP_BYTES >= 0 ||
    error("GRAFT_P1_MEMORY_CAP_BYTES must be nonnegative")
TASK_WORKSPACE_BYTES >= 0 ||
    error("GRAFT_P1_TASK_WORKSPACE_BYTES must be nonnegative")

function current_rss_bytes()
    for line in eachline("/proc/self/status")
        startswith(line, "VmRSS:") || continue
        fields = split(line)
        return parse(Int, fields[2]) * 1024
    end
    error("VmRSS is unavailable in /proc/self/status")
end

function ready!(workload)
    GC.gc(true)
    try
        ccall(:malloc_trim, Cint, (Cint,), 0)
    catch
        nothing
    end
    baseline = current_rss_bytes()
    println(
        "GRAFT_P1_READY ",
        "workload=$workload ",
        "mode=$MODE ",
        "julia_threads=$(Threads.nthreads()) ",
        "baseline_rss_bytes=$baseline",
    )
    flush(stdout)
    readline(stdin)
    return baseline
end

function digest_tensor(tensor)
    context = SHA.SHA256_CTX()
    tensor_blocks = collect(blocks(tensor))
    sort!(tensor_blocks; by=item -> repr(item[1]))
    for (sector, block_) in tensor_blocks
        SHA.update!(context, collect(codeunits(repr(sector))))
        SHA.update!(context, (0x00,))
        SHA.update!(context, reinterpret(UInt8, vec(block_)))
    end
    return bytes2hex(SHA.digest!(context))
end

function digest_tensor_blocks(tensor)
    tensor_blocks = collect(blocks(tensor))
    sort!(tensor_blocks; by=item -> repr(item[1]))
    return [
        string(repr(sector), '=',
               bytes2hex(SHA.sha256(reinterpret(UInt8, vec(block_)))))
        for (sector, block_) in tensor_blocks
    ]
end

function digest_values(values)
    return bytes2hex(SHA.sha256(
        collect(reinterpret(UInt8, values))))
end

function thermal_fixture()
    spin = spin_ops()
    topology = mps_topology(2)
    physical = Dict(nodeid(topology, index) => spin.P
                    for index in 1:nnodes(topology))
    generator = OpSum() +
        Term(0.7, SiteOp(:site1, :Z, spin.Z),
                  SiteOp(:site2, :Z, spin.Z)) +
        Term(0.35, SiteOp(:site1, :X, spin.X)) +
        Term(-0.25, SiteOp(:site2, :X, spin.X))
    problem = purification_problem(
        generator, topology, physical; hermitian=true)
    beta = 1.0
    tau_count = parse(Int, get(ENV, "GRAFT_P1_THERMAL_TAUS", "17"))
    tau_count >= 5 || error("GRAFT_P1_THERMAL_TAUS must be at least five")
    taus = collect(range(0.0, beta; length=tau_count))
    evolver() = TDVP2(
        order=1,
        trunc=TruncationScheme(maxdim=8, rtol=1e-12),
        krylovdim=8,
        tol=1e-9,
        verbose=false,
    )
    trajectory = thermalize(
        Purified(),
        problem,
        beta;
        evolver=evolver(),
        nsteps=8,
        save_betas=sort(unique(beta .- taus)),
    )
    return (; spin, problem, beta, taus, evolver, trajectory)
end

function thermal_call(fixture)
    return thermal_correlator(
        Purified(),
        fixture.problem,
        :site1 => fixture.spin.Z,
        :site2 => fixture.spin.Z,
        fixture.beta,
        fixture.taus;
        evolver=fixture.evolver(),
        trajectory=fixture.trajectory,
        prop_nsteps=8,
        threaded=MODE === :threaded,
        minbatch=2,
        task_memory_cap_bytes=MEMORY_CAP_BYTES,
        task_workspace_memory_bytes=TASK_WORKSPACE_BYTES,
    )
end

function run_thermal()
    fixture = thermal_fixture()

    # Compile the propagation path without warming the full fan-out live set.
    thermal_correlator(
        Purified(),
        fixture.problem,
        :site1 => fixture.spin.Z,
        :site2 => fixture.spin.Z,
        fixture.beta,
        [fixture.beta / 2];
        evolver=fixture.evolver(),
        trajectory=fixture.trajectory,
        prop_nsteps=2,
        threaded=false,
        task_memory_cap_bytes=MEMORY_CAP_BYTES,
        task_workspace_memory_bytes=TASK_WORKSPACE_BYTES,
    )
    # Compile the full batching/evolution path before the measured handshake.
    # The external monitor therefore records steady-state live memory, not JIT
    # code pages.
    thermal_call(fixture)
    baseline = ready!(:thermal)

    times = Float64[]
    allocations = Int[]
    gc_times = Float64[]
    digests = String[]
    sample_values = Vector{Vector{ComplexF64}}()
    diagnostics = nothing
    for _ in 1:SAMPLES
        GC.gc()
        measurement = @timed thermal_call(fixture)
        series = measurement.value
        push!(times, measurement.time)
        push!(allocations, measurement.bytes)
        push!(gc_times, measurement.gctime)
        push!(digests, digest_values(series.values))
        push!(sample_values, copy(series.values))
        diagnostics = series.metadata.fanout
    end
    bitwise_identical = all(==(first(digests)), digests)
    max_sample_abs_error = maximum(
        maximum(abs, values .- first(sample_values); init=0.0)
        for values in sample_values;
        init=0.0,
    )
    tolerance = 1e-10
    if max_sample_abs_error > tolerance
        println(
            stderr,
            "GRAFT_P1_THERMAL_MISMATCH digests=$(join(digests, ',')) ",
            "values=$(join((join(repr.(value), ':') for value in sample_values), ','))",
        )
        error("thermal benchmark sample difference $max_sample_abs_error " *
              "exceeds tolerance $tolerance")
    end
    runtime = parallel_runtime_config()
    println(
        "GRAFT_P1_RESULT ",
        "workload=thermal ",
        "mode=$MODE ",
        "julia_threads=$(runtime.julia_threads) ",
        "blas_vendor=$(runtime.blas_vendor) ",
        "blas_threads=$(runtime.blas_threads) ",
        "strided_threads=$(runtime.strided_threads) ",
        "samples=$SAMPLES ",
        "times_seconds=$(join(repr.(times), ',')) ",
        "median_seconds=$(repr(median(times))) ",
        "alloc_bytes=$(join(allocations, ',')) ",
        "gc_seconds=$(join(repr.(gc_times), ',')) ",
        "result_sha256=$(first(digests)) ",
        "bitwise_identical=$bitwise_identical ",
        "max_sample_abs_error=$(repr(max_sample_abs_error)) ",
        "tolerance=$(repr(tolerance)) ",
        "fanout_mode=$(diagnostics.mode) ",
        "fanout_fallback=$(diagnostics.fallback) ",
        "item_count=$(diagnostics.item_count) ",
        "worker_limit=$(diagnostics.worker_limit) ",
        "batch_count=$(diagnostics.batch_count) ",
        "max_batch_items=$(diagnostics.max_batch_items) ",
        "retained_memory_bytes=$(diagnostics.retained_memory_bytes) ",
        "peak_admitted_bytes=$(diagnostics.peak_admitted_bytes) ",
        "memory_cap_bytes=$(diagnostics.memory_cap_bytes) ",
        "baseline_rss_bytes=$baseline",
    )
end

function rsvd_target(block_dimension)
    sectors = parse(Int, get(ENV, "GRAFT_P1_RSVD_BLOCKS", "8"))
    sectors >= 8 || error("GRAFT_P1_RSVD_BLOCKS must be at least eight")
    space = U1Space([charge => block_dimension
                     for charge in 0:(sectors - 1)]...)
    return space ← space
end

function rsvd_call(target)
    diagnostics = Ref{Any}(nothing)
    rng = Xoshiro(0x9e3779b97f4a7c15)
    probe = _rsvd_random_probe(
        rng,
        ComplexF64,
        target;
        threaded=MODE === :threaded,
        minbatch=2,
        memory_cap_bytes=MEMORY_CAP_BYTES,
        task_workspace_memory_bytes=TASK_WORKSPACE_BYTES,
        fanout_diagnostics=diagnostics,
    )
    return probe, diagnostics[], rand(rng, UInt64)
end

@noinline function measure_rsvd_sample(target)
    measurement = @timed rsvd_call(target)
    probe, diagnostics, rng_next = measurement.value
    digest = digest_tensor(probe)
    block_digest = digest_tensor_blocks(probe)
    return (;
        time=measurement.time,
        bytes=measurement.bytes,
        gctime=measurement.gctime,
        digest,
        block_digest,
        rng_next,
        diagnostics,
    )
end

function run_rsvd()
    block_dimension = parse(
        Int, get(ENV, "GRAFT_P1_RSVD_BLOCK_DIM", "1024"))
    block_dimension >= 64 ||
        error("GRAFT_P1_RSVD_BLOCK_DIM must be at least 64")

    # Compile the same sector count and scalar path with a small payload.
    rsvd_call(rsvd_target(8))
    target = rsvd_target(block_dimension)
    baseline = ready!(:rsvd)

    times = Float64[]
    allocations = Int[]
    gc_times = Float64[]
    digests = String[]
    block_digests = Vector{Vector{String}}()
    rng_next_values = UInt64[]
    diagnostics = nothing
    for _ in 1:SAMPLES
        GC.gc()
        sample = measure_rsvd_sample(target)
        push!(times, sample.time)
        push!(allocations, sample.bytes)
        push!(gc_times, sample.gctime)
        push!(digests, sample.digest)
        push!(block_digests, sample.block_digest)
        push!(rng_next_values, sample.rng_next)
        diagnostics = sample.diagnostics
        sample = nothing
        GC.gc(true)
        try
            ccall(:malloc_trim, Cint, (Cint,), 0)
        catch
            nothing
        end
    end
    if !all(==(first(digests)), digests)
        println(
            stderr,
            "GRAFT_P1_RSVD_MISMATCH digests=$(join(digests, ',')) ",
            "block_digests=$(join((join(value, ':') for value in block_digests), ','))",
        )
        error("RSVD benchmark result changed across samples")
    end
    all(==(first(rng_next_values)), rng_next_values) ||
        error("RSVD caller RNG continuation changed across samples")
    runtime = parallel_runtime_config()
    println(
        "GRAFT_P1_RESULT ",
        "workload=rsvd ",
        "mode=$MODE ",
        "julia_threads=$(runtime.julia_threads) ",
        "blas_vendor=$(runtime.blas_vendor) ",
        "blas_threads=$(runtime.blas_threads) ",
        "strided_threads=$(runtime.strided_threads) ",
        "samples=$SAMPLES ",
        "times_seconds=$(join(repr.(times), ',')) ",
        "median_seconds=$(repr(median(times))) ",
        "alloc_bytes=$(join(allocations, ',')) ",
        "gc_seconds=$(join(repr.(gc_times), ',')) ",
        "result_sha256=$(first(digests)) ",
        "caller_rng_next=$(first(rng_next_values)) ",
        "fanout_mode=$(diagnostics.mode) ",
        "fanout_fallback=$(diagnostics.fallback) ",
        "item_count=$(diagnostics.item_count) ",
        "worker_limit=$(diagnostics.worker_limit) ",
        "batch_count=$(diagnostics.batch_count) ",
        "max_batch_items=$(diagnostics.max_batch_items) ",
        "retained_memory_bytes=$(diagnostics.retained_memory_bytes) ",
        "peak_admitted_bytes=$(diagnostics.peak_admitted_bytes) ",
        "memory_cap_bytes=$(diagnostics.memory_cap_bytes) ",
        "baseline_rss_bytes=$baseline",
    )
end

configure_parallel_runtime!(; blas_threads=1, strided_threads=1)
WORKLOAD === :thermal ? run_thermal() : run_rsvd()
