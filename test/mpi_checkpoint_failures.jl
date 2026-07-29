using Test
using Graft
using MPI
using SHA: sha256

using LinearAlgebra: BLAS
using Random: Xoshiro

const EXPECTED_RANKS = parse(
    Int, get(ENV, "GRAFT_MPI_EXPECTED_SIZE", "2"))
const EXPECTED_THREADS = parse(
    Int, get(ENV, "GRAFT_MPI_EXPECTED_THREADS", "1"))

context = mpi_context(; blas_threads=1, strided_threads=1)
rank = distributed_rank(context)
nranks = distributed_size(context)
root = distributed_root(context)
extension_module = Base.get_extension(Graft, :GraftMPIExt)

nranks == EXPECTED_RANKS ||
    error("expected $EXPECTED_RANKS MPI ranks, got $nranks")
Threads.nthreads() == EXPECTED_THREADS ||
    error("expected $EXPECTED_THREADS Julia threads, got $(Threads.nthreads())")
BLAS.get_num_threads() == 1 ||
    error("checkpoint failure tests require one BLAS thread per rank")
extension_module === nothing &&
    error("GraftMPIExt did not load")

struct MPICheckpointFailureIdentityEvolver <: Evolver end

function shared_tempname()
    return MPI.bcast(
        rank == root ? tempname() : "",
        root,
        context.comm,
    )
end

function root_mutation(operation)
    rank == root && operation()
    distributed_barrier(context)
    return nothing
end

function expect_collective_error(operation, stage::Symbol, pattern)
    caught = try
        operation()
        nothing
    catch err
        err
    end
    @test caught isa DistributedCheckpointError
    message = sprint(showerror, caught)
    messages = distributed_allgather(context, message)
    @test length(unique(messages)) == 1
    @test occursin(pattern, message)
    if caught isa DistributedCheckpointError
        @test caught.stage === stage
        @test !isempty(caught.failures)
    end
    return caught
end

function fixture_trajectory(nsamples::Int; resume_from=nothing)
    spin = spin_ops()
    topology = mps_topology(1)
    physical_spaces = Dict(:site1 => spin.P)
    terms = OpSum() +
        Term(0.37, SiteOp(:site1, :Z, spin.Z))
    problem = purification_problem(
        terms, topology, physical_spaces; hermitian=true)
    trajectory = thermalize(
        METTS(;
            rng=Xoshiro(0x20260729),
            collapse_basis=:computational,
            burnin=1,
            nsamples,
            thin=1,
        ),
        problem,
        0.0;
        evolver=MPICheckpointFailureIdentityEvolver(),
        nsteps=1,
        distributed=context,
        resume_from,
    )
    return trajectory
end

function shard_path(manifest_path, shard_name)
    return joinpath(dirname(manifest_path), shard_name)
end

function file_sha256(path)
    return bytes2hex(open(sha256, path))
end

function remove_checkpoint_family(path)
    directory = dirname(path)
    prefix = basename(path)
    for name in readdir(directory)
        startswith(name, prefix) ||
            continue
        rm(joinpath(directory, name); force=true)
    end
    return nothing
end

samples = 2nranks
trajectory = fixture_trajectory(samples)
model_identity = trajectory.local_chain.metadata.problem_hash
checkpoint_path = shared_tempname()
cleanup_paths = String[checkpoint_path]
checkpoint_mpi!(
    trajectory,
    checkpoint_path;
    keep=0,
    metadata=(; purpose=:checkpoint_failure_matrix),
)
manifest_record = resume(checkpoint_path)
manifest = manifest_record.state
bad_rank = min(1, nranks - 1)

@testset "distributed checkpoint v2 manifest" begin
    @test manifest.format === :graft_mpi_metts_v2
    @test manifest.nranks == nranks
    @test manifest.model_identity == model_identity
    @test length(manifest.shards) == nranks
    @test length(manifest.shard_sha256) == nranks
    @test length(manifest.shard_sizes) == nranks
    @test all(name -> basename(name) == name, manifest.shards)
    restored = resume_mpi(
        checkpoint_path,
        context;
        expected_model_identity=model_identity,
    )
    @test restored.metadata.purpose === :checkpoint_failure_matrix
    @test restored.state.global_nsamples == samples
end

@testset "legacy v1 checkpoint remains readable" begin
    legacy_path = shared_tempname()
    push!(cleanup_paths, legacy_path)
    legacy_shard = string(
        legacy_path,
        ".rank",
        lpad(string(rank), 4, '0'),
        ".jld2",
    )
    checkpoint!(
        trajectory.local_chain,
        legacy_shard;
        keep=0,
        metadata=(;
            format=:graft_mpi_metts_shard_v1,
            rank,
            nranks,
            total_steps=trajectory.local_chain.total_steps,
            trajectory_metadata=trajectory.local_chain.metadata,
        ),
    )
    chain_steps = distributed_allgather(
        context, trajectory.local_chain.total_steps)
    distributed_barrier(context)
    if rank == root
        legacy_manifest = (;
            format=:graft_mpi_metts_v1,
            nranks,
            global_nsamples=trajectory.global_nsamples,
            samples_per_rank=copy(trajectory.samples_per_rank),
            chain_steps=Int.(chain_steps),
            shards=[
                string(
                    legacy_path,
                    ".rank",
                    lpad(string(source), 4, '0'),
                    ".jld2",
                )
                for source in 0:(nranks - 1)
            ],
        )
        checkpoint!(legacy_manifest, legacy_path; keep=0)
    end
    distributed_barrier(context)
    restored = resume_mpi(
        legacy_path,
        context;
        expected_model_identity=model_identity,
    )
    @test restored.state.global_nsamples == trajectory.global_nsamples
    @test restored.state.samples_per_rank == trajectory.samples_per_rank
    @test restored.state.local_chain.total_steps ==
          trajectory.local_chain.total_steps
end

@testset "wrong model and wrong rank count fail collectively" begin
    expect_collective_error(:manifest_validation, "different model") do
        resume_mpi(
            checkpoint_path,
            context;
            expected_model_identity=(:wrong_model, model_identity),
        )
    end

    wrong_rank_path = shared_tempname()
    push!(cleanup_paths, wrong_rank_path)
    root_mutation() do
        wrong_manifest = merge(manifest, (; nranks=nranks + 1))
        checkpoint!(
            wrong_manifest,
            wrong_rank_path;
            keep=0,
            metadata=manifest_record.metadata,
        )
    end
    expect_collective_error(:manifest_validation, "ranks") do
        resume_mpi(wrong_rank_path, context)
    end
end

@testset "missing and corrupt shards fail collectively" begin
    missing_path = shard_path(
        checkpoint_path, manifest.shards[bad_rank + 1])
    missing_backup = missing_path * ".missing-test"
    root_mutation() do
        mv(missing_path, missing_backup; force=true)
    end
    expect_collective_error(:shard_read, "missing") do
        resume_mpi(checkpoint_path, context)
    end
    root_mutation() do
        mv(missing_backup, missing_path; force=true)
    end

    corrupt_path = shard_path(
        checkpoint_path, manifest.shards[bad_rank + 1])
    original_bytes = rank == root ? read(corrupt_path) : UInt8[]
    root_mutation() do
        open(corrupt_path, "a") do io
            write(io, UInt8(0xff))
        end
    end
    expect_collective_error(:shard_read, "size mismatch") do
        resume_mpi(checkpoint_path, context)
    end
    root_mutation() do
        write(corrupt_path, original_bytes)
    end
end

@testset "mixed-step shard fails before state return" begin
    advanced = fixture_trajectory(samples; resume_from=trajectory)
    advanced_path = shared_tempname()
    push!(cleanup_paths, advanced_path)
    checkpoint_mpi!(advanced, advanced_path; keep=0)
    advanced_manifest = resume(advanced_path).state
    mixed_path = shared_tempname()
    push!(cleanup_paths, mixed_path)

    root_mutation() do
        source_path = shard_path(
            advanced_path, advanced_manifest.shards[bad_rank + 1])
        source_record = resume(source_path)
        synthetic_name = string(
            basename(mixed_path),
            ".synthetic.rank",
            lpad(string(bad_rank), 4, '0'),
            ".jld2",
        )
        synthetic_path = shard_path(mixed_path, synthetic_name)
        synthetic_metadata = merge(
            source_record.metadata,
            (;
                generation=manifest.generation,
                model_identity=manifest.model_identity,
            ),
        )
        checkpoint!(
            source_record.state,
            synthetic_path;
            keep=0,
            metadata=synthetic_metadata,
        )
        mixed_shards = copy(manifest.shards)
        mixed_digests = copy(manifest.shard_sha256)
        mixed_sizes = copy(manifest.shard_sizes)
        mixed_shards[bad_rank + 1] = synthetic_name
        mixed_digests[bad_rank + 1] = file_sha256(synthetic_path)
        mixed_sizes[bad_rank + 1] = filesize(synthetic_path)
        mixed_manifest = merge(
            manifest,
            (;
                shards=mixed_shards,
                shard_sha256=mixed_digests,
                shard_sizes=mixed_sizes,
            ),
        )
        checkpoint!(mixed_manifest, mixed_path; keep=0)
    end

    expect_collective_error(:shard_validation, "mixes chain steps") do
        resume_mpi(mixed_path, context)
    end
end

@testset "write stages fail collectively without manifest publication" begin
    for (failure_stage, reported_stage, causal_rank) in (
            (:serialization, :shard_write, bad_rank),
            (:shard_write, :shard_write, bad_rank),
            (:before_manifest, :before_manifest, bad_rank),
            (:manifest_write, :manifest_write, root))
        failure_path = shared_tempname()
        push!(cleanup_paths, failure_path)
        caught = expect_collective_error(
                reported_stage, string("injected ", failure_stage)) do
            extension_module._checkpoint_mpi!(
                trajectory,
                failure_path;
                keep=0,
                failure_stage,
                failure_rank=causal_rank,
            )
        end
        if caught isa DistributedCheckpointError
            @test caught.failures == [
                (causal_rank, "injected $failure_stage failure")]
        end
        manifest_exists = distributed_allgather(
            context, isfile(failure_path))
        @test !any(manifest_exists)
    end
end

@testset "failed update preserves the prior manifest" begin
    stable_path = shared_tempname()
    push!(cleanup_paths, stable_path)
    checkpoint_mpi!(trajectory, stable_path; keep=0)
    expect_collective_error(:before_manifest, "injected before_manifest") do
        extension_module._checkpoint_mpi!(
            fixture_trajectory(samples; resume_from=trajectory),
            stable_path;
            keep=0,
            failure_stage=:before_manifest,
            failure_rank=bad_rank,
        )
    end
    restored = resume_mpi(stable_path, context)
    @test restored.state.local_chain.total_steps ==
          trajectory.local_chain.total_steps
    @test restored.state.global_nsamples == trajectory.global_nsamples
end

distributed_barrier(context)
if rank == root
    for path in cleanup_paths
        remove_checkpoint_family(path)
    end
    println(
        "GRAFT_MPI_CHECKPOINT_FAILURE_RESULT ",
        "ranks=$nranks ",
        "result=passed ",
        "causal_rank=$bad_rank ",
        "format=graft_mpi_metts_v2",
    )
end
