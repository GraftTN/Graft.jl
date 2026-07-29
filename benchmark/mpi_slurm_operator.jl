using Graft
using MPI

using Graft.Backend: ℂ, blocks
using Graft.TestUtils: random_ttns
using LinearAlgebra: BLAS, norm
using Random: MersenneTwister
using Statistics: median

const EXPECTED_RANKS = parse(Int, get(ENV, "GRAFT_MPI_EXPECTED_SIZE", "1"))
const BLAS_THREADS = parse(Int, get(ENV, "GRAFT_MPI_BLAS_THREADS", "4"))
const ARM_COUNT = parse(Int, get(ENV, "GRAFT_MPI_OPERATOR_ARMS", "4"))
const ARM_DEPTH = parse(Int, get(ENV, "GRAFT_MPI_OPERATOR_ARM_DEPTH", "4"))
const BOND_DIMENSION = parse(Int, get(ENV, "GRAFT_MPI_OPERATOR_BOND_DIM", "12"))
const REPETITIONS = parse(Int, get(ENV, "GRAFT_MPI_OPERATOR_REPETITIONS", "5"))
const SAMPLES = parse(Int, get(ENV, "GRAFT_MPI_BENCH_SAMPLES", "5"))
const MEMORY_CAP_BYTES = parse(
    Int, get(ENV, "GRAFT_MPI_OPERATOR_MEMORY_CAP_BYTES", "20000000000"))
const EXPECT_ADMISSION_REJECTION = lowercase(get(
    ENV, "GRAFT_MPI_OPERATOR_EXPECT_ADMISSION_REJECTION", "false")) in
    ("1", "true", "yes", "on")

context = mpi_context(; blas_threads=BLAS_THREADS, strided_threads=1)
rank = distributed_rank(context)
nranks = distributed_size(context)
root = distributed_root(context)

nranks == EXPECTED_RANKS ||
    error("expected $EXPECTED_RANKS MPI ranks, got $nranks")
BLAS.get_num_threads() == BLAS_THREADS ||
    error("expected $BLAS_THREADS BLAS threads, got $(BLAS.get_num_threads())")
Threads.nthreads() == 1 ||
    error("operator benchmark requires one Julia thread per rank")
ARM_COUNT >= 3 || error("operator benchmark requires at least three arms")
ARM_DEPTH >= 1 || error("operator benchmark arm depth must be positive")
BOND_DIMENSION >= 1 || error("operator benchmark bond dimension must be positive")
REPETITIONS >= 1 || error("operator benchmark repetitions must be positive")
SAMPLES >= 1 || error("operator benchmark samples must be positive")

function benchmark_fixture()
    topology = star_topology(ARM_COUNT, ARM_DEPTH)
    spin = spin_ops()
    physical_spaces = Dict(
        nodeid(topology, i) => spin.P for i in 1:nnodes(topology))
    terms = OpSum()
    for (child, parent) in Graft.Trees.edges(topology)
        terms += Term(
            0.17 + 0.001child,
            SiteOp(nodeid(topology, child), :Z, spin.Z),
            SiteOp(nodeid(topology, parent), :Z, spin.Z),
        )
        terms += Term(
            -0.11,
            SiteOp(nodeid(topology, child), :X, spin.X),
            SiteOp(nodeid(topology, parent), :X, spin.X),
        )
    end
    leaves = Graft.leaves(topology)
    for (pair_index, (left, right)) in enumerate(
            (pair for pair in Iterators.product(leaves, leaves)
             if pair[1] < pair[2]))
        terms += Term(
            0.03 + 0.0002pair_index,
            SiteOp(nodeid(topology, left), :Z, spin.Z),
            SiteOp(nodeid(topology, right), :X, spin.X),
        )
        terms += Term(
            -0.02 - 0.0001pair_index,
            SiteOp(nodeid(topology, left), :X, spin.X),
            SiteOp(nodeid(topology, right), :Z, spin.Z),
        )
    end
    for node in 1:nnodes(topology)
        terms += Term(
            0.29 + 0.005node,
            SiteOp(nodeid(topology, node), :X, spin.X),
        )
        terms += Term(
            -0.07,
            SiteOp(nodeid(topology, node), :Z, spin.Z),
        )
    end
    operator = ttno_from_opsum(
        terms, topology, physical_spaces; hermitian=true)
    state = random_ttns(
        MersenneTwister(260729),
        ComplexF64,
        topology,
        physical_spaces,
        ℂ^BOND_DIMENSION,
    )
    normalize!(state)
    move_center!(state, topology.root)
    return state, operator, topology
end

function repeated_map(map_, input)
    result = map_(input)
    for _ in 2:REPETITIONS
        result = map_(input)
    end
    return result
end

function timed_run(map_, input)
    GC.gc()
    distributed_barrier(context)
    start = time_ns()
    result = repeated_map(map_, input)
    rank_seconds = (time_ns() - start) / 1e9
    maximum_seconds = [rank_seconds]
    MPI.Allreduce!(maximum_seconds, MPI.MAX, context.comm)
    return result, only(maximum_seconds), rank_seconds
end

function tensor_payload_bytes(tensor)
    return sum(
        sizeof(eltype(payload)) * length(payload)
        for (_, payload) in blocks(tensor);
        init=0,
    )
end

state, operator, topology = benchmark_fixture()
input = state.tensors[topology.root]
serial_map = eff_h1(EnvCache(topology), state, operator, topology.root)
active_map = try
    if nranks == 1
        serial_map
    else
        eff_h1(
            EnvCache(topology),
            state,
            operator,
            topology.root;
            distributed=context,
            channel_slices=nranks,
            channel_minbatch=1,
            channel_min_flops=0,
            channel_memory_cap_bytes=MEMORY_CAP_BYTES,
        )
    end
catch err
    if EXPECT_ADMISSION_REJECTION &&
       err isa DistributedChannelAdmissionError
        if rank == root
            println(
                "GRAFT_SLURM_ADMISSION ",
                "kind=operator_channel ",
                "result=rejected ",
                "reason=$(err.reason) ",
                "ranks=$(err.ranks) ",
                "nonempty_slices=$(err.nonempty_slices)",
            )
        end
        distributed_barrier(context)
        exit(0)
    end
    rethrow()
end
EXPECT_ADMISSION_REJECTION &&
    error("expected distributed operator-channel admission rejection")

serial_reference = rank == root ? repeated_map(serial_map, input) : nothing
distributed_barrier(context)
warmup, _, _ = timed_run(active_map, input)
reference_error = rank == root ?
    norm(warmup - serial_reference) / max(norm(serial_reference), 1.0) : 0.0
reference_errors = [reference_error]
distributed_broadcast!(context, reference_errors)
only(reference_errors) <= 1e-10 ||
    error("distributed operator map relative error $(only(reference_errors))")

times = Float64[]
rank_times = Float64[]
result_ref = Ref(warmup)
for _ in 1:SAMPLES
    sample_result, seconds, rank_seconds = timed_run(active_map, input)
    result_ref[] = sample_result
    rank == root && push!(times, seconds)
    push!(rank_times, rank_seconds)
end
result = result_ref[]

rank_medians = distributed_allgather(context, median(rank_times))
rank_rss_bytes = distributed_allgather(context, Int(Sys.maxrss()))
rank_hosts = distributed_allgather(context, gethostname())
rank_norms = distributed_allgather(context, norm(result))
rank_local_slices = nranks == 1 ? [1] :
    distributed_allgather(context, length(active_map.local_indices))
total_slices = nranks == 1 ? 1 : length(active_map.effective.maps)
payload_bytes = tensor_payload_bytes(result)

if rank == root
    elapsed = median(times)
    fastest_rank_seconds = minimum(rank_medians)
    slowest_rank_seconds = maximum(rank_medians)
    println(
        "GRAFT_SLURM_RESULT ",
        "kind=operator_channel ",
        "ranks=$nranks ",
        "blas_threads=$(BLAS.get_num_threads()) ",
        "julia_threads=$(Threads.nthreads()) ",
        "nodes=$(nnodes(topology)) ",
        "arms=$ARM_COUNT ",
        "arm_depth=$ARM_DEPTH ",
        "bond_dimension=$BOND_DIMENSION ",
        "repetitions=$REPETITIONS ",
        "samples=$SAMPLES ",
        "total_slices=$total_slices ",
        "local_slices=$(join(rank_local_slices, ',')) ",
        "median_seconds=$(repr(elapsed)) ",
        "minimum_seconds=$(repr(minimum(times))) ",
        "maximum_seconds=$(repr(maximum(times))) ",
        "times_seconds=$(join(repr.(times), ',')) ",
        "rank_median_min_seconds=$(repr(fastest_rank_seconds)) ",
        "rank_median_max_seconds=$(repr(slowest_rank_seconds)) ",
        "load_imbalance=$(repr(slowest_rank_seconds / fastest_rank_seconds)) ",
        "peak_rss_min_bytes=$(minimum(rank_rss_bytes)) ",
        "peak_rss_max_bytes=$(maximum(rank_rss_bytes)) ",
        "peak_rss_aggregate_bytes=$(sum(rank_rss_bytes)) ",
        "reduction_payload_bytes=$payload_bytes ",
        "reference_relative_error=$(repr(only(reference_errors))) ",
        "rank_norm_spread=$(repr(maximum(rank_norms) - minimum(rank_norms))) ",
        "hosts=$(join(rank_hosts, ','))",
    )
end
