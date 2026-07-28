module GraftMPIExt

using Graft
using MPI

using Graft.Backend: AbstractTensorMap, blocks
import Graft.Parallel: mpi_context, distributed_rank, distributed_size,
    distributed_root, distributed_barrier, distributed_allreduce_sum!,
    distributed_broadcast!, distributed_allgather, distributed_eigsolve,
    distributed_exponentiate
import Graft.Checkpoints: checkpoint!, resume, checkpoint_mpi!, resume_mpi
using Graft.Thermal: DistributedMETTSTrajectory

"""
    MPIContext

Explicit MPI execution context used by Graft's distributed algorithms. The
communicator is never stored in global package state and is not serializable.
"""
struct MPIContext{C} <: Graft.AbstractDistributedContext
    comm::C
    root::Int
    threadlevel::MPI.ThreadLevel
end

"""
    mpi_context(; comm=MPI.COMM_WORLD, root=0, initialize=true,
                configure_runtime=true, blas_threads=1, strided_threads=1)

Create a distributed context. MPI is initialized with `THREAD_FUNNELED` when
needed: Julia worker threads may perform local contractions, while collectives
run only after those workers have joined.
"""
function mpi_context(; comm=MPI.COMM_WORLD, root::Integer=0,
                     initialize::Bool=true,
                     configure_runtime::Bool=true,
                     blas_threads::Integer=1,
                     strided_threads::Integer=1)
    if !MPI.Initialized()
        initialize || throw(ArgumentError(
            "MPI is not initialized; call MPI.Init or pass initialize=true"))
        MPI.Init(; threadlevel=:funneled)
    end
    MPI.Finalized() &&
        throw(ArgumentError("MPI has already been finalized"))
    threadlevel = MPI.Query_thread()
    threadlevel >= MPI.THREAD_FUNNELED || throw(ArgumentError(
        "Graft requires at least MPI_THREAD_FUNNELED"))
    size = MPI.Comm_size(comm)
    0 <= root < size ||
        throw(ArgumentError("root must lie in 0:$(size - 1)"))
    configure_runtime && Graft.configure_parallel_runtime!(
        ; blas_threads, strided_threads)
    return MPIContext(comm, Int(root), threadlevel)
end

distributed_rank(context::MPIContext) = MPI.Comm_rank(context.comm)
distributed_size(context::MPIContext) = MPI.Comm_size(context.comm)
distributed_root(context::MPIContext) = context.root

function distributed_barrier(context::MPIContext)
    MPI.Barrier(context.comm)
    return nothing
end

function distributed_allreduce_sum!(context::MPIContext,
                                    value::Union{AbstractArray,Base.RefValue})
    MPI.Allreduce!(value, MPI.SUM, context.comm)
    return value
end

function distributed_allreduce_sum!(context::MPIContext,
                                    tensor::AbstractTensorMap)
    for (_, data) in blocks(tensor)
        buffer = collect(data)
        MPI.Allreduce!(buffer, MPI.SUM, context.comm)
        copyto!(data, buffer)
    end
    return tensor
end

function distributed_broadcast!(context::MPIContext,
                                value::Union{AbstractArray,Base.RefValue};
                                root::Integer=context.root)
    MPI.Bcast!(value, root, context.comm)
    return value
end

function distributed_broadcast!(context::MPIContext,
                                tensor::AbstractTensorMap;
                                root::Integer=context.root)
    for (_, data) in blocks(tensor)
        buffer = collect(data)
        MPI.Bcast!(buffer, root, context.comm)
        copyto!(data, buffer)
    end
    return tensor
end

function distributed_allgather(context::MPIContext, value)
    return [
        MPI.bcast(value, source, context.comm)
        for source in 0:(distributed_size(context) - 1)
    ]
end

struct _RootDrivenMap{C,W}
    context::C
    workspace::W
end

function (map::_RootDrivenMap)(x::AbstractTensorMap)
    command = Int[1]
    distributed_broadcast!(map.context, command)
    input = MPI.bcast(
        x, distributed_root(map.context), map.context.comm)
    return map.workspace(input)
end

function _root_driven_solver(solver, context::MPIContext, effective, x)
    root = distributed_root(context)
    rank = distributed_rank(context)
    return Graft.Contractions._with_workspace_map(effective) do workspace
        payload = if rank == root
            driven = _RootDrivenMap(context, workspace)
            result = try
                (; ok=true, value=solver(driven), message="")
            catch err
                (; ok=false, value=nothing, message=sprint(showerror, err))
            end
            distributed_broadcast!(context, Int[0])
            result
        else
            while true
                command = Int[0]
                distributed_broadcast!(context, command)
                iszero(only(command)) && break
                only(command) == 1 ||
                    error("invalid root-driven solver command")
                input = MPI.bcast(nothing, root, context.comm)
                workspace(input)
            end
            nothing
        end
        payload = MPI.bcast(payload, root, context.comm)
        payload.ok || error("distributed solver failed on root: " *
                            payload.message)
        return payload.value
    end
end

function distributed_eigsolve(
        context::MPIContext, effective, x, howmany::Integer, which;
        kwargs...)
    return _root_driven_solver(context, effective, x) do driven
        Graft.GroundState.eigsolve(
            driven, x, howmany, which; kwargs...)
    end
end

function distributed_exponentiate(
        context::MPIContext, effective, x, time;
        kwargs...)
    return _root_driven_solver(context, effective, x) do driven
        Graft.Evolution.exponentiate(driven, time, x; kwargs...)
    end
end

_rank_shard(path, rank) =
    string(path, ".rank", lpad(string(rank), 4, '0'), ".jld2")

function checkpoint_mpi!(
        trajectory::DistributedMETTSTrajectory,
        path::AbstractString;
        keep::Int=3,
        metadata=NamedTuple())
    context = trajectory.context
    rank = distributed_rank(context)
    size = distributed_size(context)
    size == length(trajectory.samples_per_rank) || throw(ArgumentError(
        "distributed trajectory rank count is inconsistent"))
    chain_steps = distributed_allgather(
        context, trajectory.local_chain.total_steps)
    shard = _rank_shard(path, rank)
    shard_metadata = (;
        format=:graft_mpi_metts_shard_v1,
        rank,
        nranks=size,
        total_steps=trajectory.local_chain.total_steps,
        trajectory_metadata=trajectory.local_chain.metadata,
    )
    checkpoint!(
        trajectory.local_chain, shard; keep,
        metadata=merge(metadata, shard_metadata))
    distributed_barrier(context)
    if rank == distributed_root(context)
        shards = [_rank_shard(path, source) for source in 0:(size - 1)]
        manifest = (;
            format=:graft_mpi_metts_v1,
            nranks=size,
            global_nsamples=trajectory.global_nsamples,
            samples_per_rank=copy(trajectory.samples_per_rank),
            chain_steps=Int.(chain_steps),
            shards,
        )
        checkpoint!(manifest, path; keep, metadata)
    end
    distributed_barrier(context)
    return path
end

checkpoint!(
        trajectory::DistributedMETTSTrajectory,
        path::AbstractString;
        kwargs...) = checkpoint_mpi!(trajectory, path; kwargs...)

function resume_mpi(path::AbstractString, context::MPIContext)
    manifest_record = resume(path)
    manifest = manifest_record.state
    hasproperty(manifest, :format) &&
        manifest.format === :graft_mpi_metts_v1 || throw(ArgumentError(
            "checkpoint is not a distributed METTS manifest"))
    size = distributed_size(context)
    manifest.nranks == size || throw(ArgumentError(
        "checkpoint needs $(manifest.nranks) ranks, got $size"))
    rank = distributed_rank(context)
    shard_path = manifest.shards[rank + 1]
    isfile(shard_path) || throw(ArgumentError(
        "distributed checkpoint shard is missing: $shard_path"))
    shard_record = resume(shard_path)
    shard_metadata = shard_record.metadata
    shard_metadata.format === :graft_mpi_metts_shard_v1 ||
        throw(ArgumentError("invalid distributed METTS shard format"))
    shard_metadata.rank == rank ||
        throw(ArgumentError("distributed METTS shard rank mismatch"))
    shard_metadata.nranks == size ||
        throw(ArgumentError("distributed METTS shard communicator mismatch"))
    shard_record.state.total_steps == manifest.chain_steps[rank + 1] ||
        throw(ArgumentError("distributed METTS checkpoint mixes chain steps"))
    trajectory = Graft.Thermal._distributed_trajectory(
        shard_record.state, context)
    trajectory.samples_per_rank == manifest.samples_per_rank ||
        throw(ArgumentError("distributed METTS shard sample counts mismatch"))
    trajectory.global_nsamples == manifest.global_nsamples ||
        throw(ArgumentError("distributed METTS global sample count mismatch"))
    return (; state=trajectory, metadata=manifest_record.metadata)
end

end # module GraftMPIExt
