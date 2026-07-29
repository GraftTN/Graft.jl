using Test
using Graft
using MPI

using Graft.TestUtils: to_dense
using LinearAlgebra: BLAS, norm
using Random: Xoshiro, rand

const EXPECTED_RANKS = parse(
    Int, get(ENV, "GRAFT_MPI_EXPECTED_SIZE", "2"))
const EXPECTED_THREADS = parse(
    Int, get(ENV, "GRAFT_MPI_EXPECTED_THREADS", "1"))
const SAMPLES_PER_RANK = parse(
    Int, get(ENV, "GRAFT_MPI_CHECKPOINT_SAMPLES_PER_RANK", "4"))

iszero(SAMPLES_PER_RANK % 2) ||
    error("checkpoint continuation requires an even sample count per rank")
SAMPLES_PER_RANK >= 2 ||
    error("checkpoint continuation requires at least two samples per rank")

context = mpi_context(; blas_threads=1, strided_threads=1)
rank = distributed_rank(context)
nranks = distributed_size(context)
root = distributed_root(context)

nranks == EXPECTED_RANKS ||
    error("expected $EXPECTED_RANKS MPI ranks, got $nranks")
Threads.nthreads() == EXPECTED_THREADS ||
    error("expected $EXPECTED_THREADS Julia threads, got $(Threads.nthreads())")
BLAS.get_num_threads() == 1 ||
    error("checkpoint continuation requires one BLAS thread per rank")

struct MPICheckpointIdentityEvolver <: Evolver end

function distributed_maximum_seconds(f)
    distributed_barrier(context)
    start = time_ns()
    value = f()
    local_seconds = (time_ns() - start) / 1e9
    maximum_seconds = [local_seconds]
    MPI.Allreduce!(maximum_seconds, MPI.MAX, context.comm)
    return value, only(maximum_seconds)
end

function trajectory(
        problem, nsamples::Int;
        seed::UInt64,
        resume_from=nothing)
    return thermalize(
        METTS(;
            rng=Xoshiro(seed),
            collapse_basis=:computational,
            burnin=1,
            nsamples,
            thin=1,
        ),
        problem,
        0.0;
        evolver=MPICheckpointIdentityEvolver(),
        nsteps=1,
        distributed=context,
        resume_from,
    )
end

spin = spin_ops()
topology = mps_topology(2)
physical_spaces = Dict(
    nodeid(topology, i) => spin.P for i in 1:nnodes(topology))
terms = OpSum() +
    Term(0.37, SiteOp(:site1, :Z, spin.Z)) +
    Term(-0.21, SiteOp(:site2, :X, spin.X)) +
    Term(
        0.13,
        SiteOp(:site1, :Z, spin.Z),
        SiteOp(:site2, :Z, spin.Z),
    )
problem = purification_problem(
    terms, topology, physical_spaces; hermitian=true)

total_samples = SAMPLES_PER_RANK * nranks
half_samples = div(total_samples, 2)
seed = UInt64(0x20260729)

uninterrupted = trajectory(problem, total_samples; seed)
first_half = trajectory(problem, half_samples; seed)

checkpoint_path = MPI.bcast(
    rank == root ? tempname() : "",
    root,
    context.comm,
)
_, checkpoint_seconds = distributed_maximum_seconds() do
    checkpoint_mpi!(
        first_half,
        checkpoint_path;
        keep=0,
        metadata=(; purpose=:continuation_equivalence),
    )
end
restored, restart_seconds = distributed_maximum_seconds() do
    resume_mpi(checkpoint_path, context)
end
resumed = trajectory(
    problem,
    half_samples;
    seed=UInt64(0xdeadbeef),
    resume_from=restored.state,
)

function sample_error(left, right)
    return norm(to_dense(left.state) - to_dense(right.state))
end

local_sample_errors = [
    sample_error(left, right)
    for (left, right) in zip(
        uninterrupted.local_chain.samples,
        resumed.local_chain.samples,
    )
]
local_maximum_error = maximum(local_sample_errors; init=0.0)
local_final_error = norm(
    to_dense(uninterrupted.local_chain.final_product) -
    to_dense(resumed.local_chain.final_product),
)
local_rng_match = rand(
    copy(uninterrupted.local_chain.rng), UInt64, 4) ==
    rand(copy(resumed.local_chain.rng), UInt64, 4)
rank_maximum_errors = distributed_allgather(
    context, max(local_maximum_error, local_final_error))
rank_rng_matches = distributed_allgather(context, local_rng_match)

@testset "distributed checkpoint continuation equivalence" begin
    @test restored.metadata.purpose === :continuation_equivalence
    @test uninterrupted.global_nsamples == total_samples
    @test resumed.global_nsamples == total_samples
    @test uninterrupted.samples_per_rank ==
          fill(SAMPLES_PER_RANK, nranks)
    @test resumed.samples_per_rank == uninterrupted.samples_per_rank
    @test length(uninterrupted.local_chain.samples) == SAMPLES_PER_RANK
    @test length(resumed.local_chain.samples) == SAMPLES_PER_RANK
    @test uninterrupted.local_chain.total_steps ==
          resumed.local_chain.total_steps
    @test uninterrupted.local_chain.final_outcomes ==
          resumed.local_chain.final_outcomes
    @test [
        sample.chain_step for sample in uninterrupted.local_chain.samples
    ] == [
        sample.chain_step for sample in resumed.local_chain.samples
    ]
    @test [
        sample.outcomes for sample in uninterrupted.local_chain.samples
    ] == [
        sample.outcomes for sample in resumed.local_chain.samples
    ]
    @test [
        sample.log_weight for sample in uninterrupted.local_chain.samples
    ] == [
        sample.log_weight for sample in resumed.local_chain.samples
    ]
    @test maximum(rank_maximum_errors) <= 1e-14
    @test all(rank_rng_matches)
end

distributed_barrier(context)
if rank == root
    for source in 0:(nranks - 1)
        rm(
            string(
                checkpoint_path,
                ".rank",
                lpad(string(source), 4, '0'),
                ".jld2",
            );
            force=true,
        )
    end
    rm(checkpoint_path; force=true)
    println(
        "GRAFT_MPI_CHECKPOINT_RESULT ",
        "ranks=$nranks ",
        "samples_per_rank=$SAMPLES_PER_RANK ",
        "checkpoint_seconds=$(repr(checkpoint_seconds)) ",
        "restart_seconds=$(repr(restart_seconds)) ",
        "maximum_continuation_error=$(repr(maximum(rank_maximum_errors))) ",
        "rng_continuation_match=$(all(rank_rng_matches))",
    )
end
