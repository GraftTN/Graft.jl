using Graft
using MPI

using Graft.Backend: ℂ
using Graft.TestUtils: random_ttns
using LinearAlgebra: BLAS, norm
using Random: MersenneTwister
using Statistics: median

const EXPECTED_RANKS = parse(Int, get(ENV, "GRAFT_MPI_EXPECTED_SIZE", "1"))
const BLAS_THREADS = parse(Int, get(ENV, "GRAFT_MPI_BLAS_THREADS", "4"))
const SNAPSHOT_COUNT = parse(Int, get(ENV, "GRAFT_MPI_BENCH_SNAPSHOTS", "36"))
const BOND_DIMENSION = parse(Int, get(ENV, "GRAFT_MPI_BENCH_BOND_DIM", "12"))
const ARM_DEPTH = parse(Int, get(ENV, "GRAFT_MPI_BENCH_ARM_DEPTH", "3"))
const SAMPLES = parse(Int, get(ENV, "GRAFT_MPI_BENCH_SAMPLES", "3"))

context = mpi_context(;
    blas_threads=BLAS_THREADS,
    strided_threads=1,
)
rank = distributed_rank(context)
nranks = distributed_size(context)
root = distributed_root(context)

nranks == EXPECTED_RANKS ||
    error("expected $EXPECTED_RANKS MPI ranks, got $nranks")
BLAS.get_num_threads() == BLAS_THREADS ||
    error("expected $BLAS_THREADS BLAS threads, got $(BLAS.get_num_threads())")
Threads.nthreads() == 1 ||
    error("this benchmark isolates BLAS threading and requires one Julia thread")
SNAPSHOT_COUNT >= nranks ||
    error("snapshot count must be at least the MPI rank count")
SAMPLES >= 1 || error("sample count must be positive")

function benchmark_fixture()
    topology = star_topology(3, ARM_DEPTH)
    spin = spin_ops()
    physical_spaces = Dict(
        nodeid(topology, i) => spin.P for i in 1:nnodes(topology))

    terms = OpSum()
    for (child, parent) in Graft.Trees.edges(topology)
        terms += Term(
            0.19,
            SiteOp(nodeid(topology, child), :Z, spin.Z),
            SiteOp(nodeid(topology, parent), :Z, spin.Z),
        )
    end
    for node in 1:nnodes(topology)
        terms += Term(
            0.37 + 0.01node,
            SiteOp(nodeid(topology, node), :X, spin.X),
        )
    end
    operator = ttno_from_opsum(
        terms, topology, physical_spaces; hermitian=true)
    snapshots = [
        random_ttns(
            MersenneTwister(260728 + i),
            ComplexF64,
            topology,
            physical_spaces,
            ℂ^BOND_DIMENSION,
        )
        for i in 1:SNAPSHOT_COUNT
    ]
    foreach(normalize!, snapshots)
    return snapshots, operator, topology
end

function timed_run(f)
    GC.gc()
    distributed_barrier(context)
    start = time_ns()
    result = f()
    rank_seconds = (time_ns() - start) / 1e9
    maximum_seconds = [rank_seconds]
    MPI.Allreduce!(maximum_seconds, MPI.MAX, context.comm)
    return result, only(maximum_seconds), rank_seconds
end

snapshots, operator, topology = benchmark_fixture()
run_benchmark() =
    complex_time_krylov(snapshots, operator; distributed=context)

# Compile and populate contraction-plan caches before measuring wall time.
warmup, _, _ = timed_run(run_benchmark)
times = Float64[]
rank_times = Float64[]
result_ref = Ref(warmup)
for _ in 1:SAMPLES
    sample_result, seconds, rank_seconds = timed_run(run_benchmark)
    result_ref[] = sample_result
    rank == root && push!(times, seconds)
    push!(rank_times, rank_seconds)
end
result = result_ref[]
rank_medians = distributed_allgather(context, median(rank_times))
rank_rss_bytes = distributed_allgather(context, Int(Sys.maxrss()))
rank_hosts = distributed_allgather(context, gethostname())

if rank == root
    elapsed = median(times)
    slowest_rank_seconds = maximum(rank_medians)
    fastest_rank_seconds = minimum(rank_medians)
    load_imbalance = slowest_rank_seconds / fastest_rank_seconds
    println(
        "GRAFT_SLURM_RESULT ",
        "kind=sample_parallel ",
        "ranks=$nranks ",
        "blas_threads=$(BLAS.get_num_threads()) ",
        "julia_threads=$(Threads.nthreads()) ",
        "nodes=$(nnodes(topology)) ",
        "snapshots=$SNAPSHOT_COUNT ",
        "bond_dimension=$BOND_DIMENSION ",
        "samples=$SAMPLES ",
        "median_seconds=$(repr(elapsed)) ",
        "minimum_seconds=$(repr(minimum(times))) ",
        "maximum_seconds=$(repr(maximum(times))) ",
        "rank_median_min_seconds=$(repr(fastest_rank_seconds)) ",
        "rank_median_max_seconds=$(repr(slowest_rank_seconds)) ",
        "load_imbalance=$(repr(load_imbalance)) ",
        "peak_rss_min_bytes=$(minimum(rank_rss_bytes)) ",
        "peak_rss_max_bytes=$(maximum(rank_rss_bytes)) ",
        "peak_rss_aggregate_bytes=$(sum(rank_rss_bytes)) ",
        "hosts=$(join(rank_hosts, ',')) ",
        "overlap_norm=$(repr(norm(result.overlap))) ",
        "hamiltonian_norm=$(repr(norm(result.hamiltonian))) ",
        "energy_sum_real=$(repr(sum(real, result.values))) ",
        "energy_sum_imag=$(repr(sum(imag, result.values))) ",
        "retained_rank=$(result.diagnostics.retained_rank) ",
        "max_residual=$(repr(maximum(result.residuals)))",
    )
end
