module GraftMPIExt

using Graft
using MPI
using SHA: sha256

using Graft.Backend: AbstractTensorMap, blocks
import Graft.Parallel: mpi_context, distributed_rank, distributed_size,
    distributed_root, distributed_barrier, distributed_allreduce_sum!,
    distributed_broadcast!, distributed_allgather, distributed_eigsolve,
    distributed_exponentiate
import Graft.Checkpoints: DistributedCheckpointError,
    checkpoint!, resume, checkpoint_mpi!, resume_mpi
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

function _collective_workspace_call(context::MPIContext, workspace, input)
    local_result = try
        (; ok=true, value=workspace(input), message="")
    catch err
        (; ok=false, value=nothing, message=sprint(showerror, err))
    end
    failures = [local_result.ok ? 0 : 1]
    MPI.Allreduce!(failures, MPI.SUM, context.comm)
    if !iszero(only(failures))
        messages = distributed_allgather(context, local_result.message)
        details = join(
            ("rank $(rank - 1): $message"
             for (rank, message) in enumerate(messages) if !isempty(message)),
            "; ")
        error("distributed map evaluation failed: $details")
    end
    return local_result.value
end

function (map::_RootDrivenMap)(x::AbstractTensorMap)
    command = Int[1]
    distributed_broadcast!(map.context, command)
    input = MPI.bcast(
        x, distributed_root(map.context), map.context.comm)
    return _collective_workspace_call(map.context, map.workspace, input)
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
                try
                    _collective_workspace_call(context, workspace, input)
                catch
                    # The root sees the same collective failure, terminates the
                    # adaptive solve, and sends the stop command and payload.
                    # Workers must remain in the command loop to receive them.
                end
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

_legacy_rank_shard(path, rank) =
    string(path, ".rank", lpad(string(rank), 4, '0'), ".jld2")

function _checkpoint_generation(path::AbstractString)
    payload = Vector{UInt8}(codeunits(
        string(path, '\0', time_ns(), '\0', getpid())))
    return first(bytes2hex(sha256(payload)), 24)
end

function _rank_shard_name(path::AbstractString, generation, rank::Integer)
    return string(
        basename(path),
        ".",
        generation,
        ".rank",
        lpad(string(rank), 4, '0'),
        ".jld2",
    )
end

_rank_shard_path(path::AbstractString, shard_name::AbstractString) =
    joinpath(dirname(path), shard_name)

_file_sha256(path::AbstractString) = bytes2hex(open(sha256, path))

function _collective_checkpoint_stage(
        operation, context::MPIContext, stage::Symbol)
    local_result = try
        (; ok=true, value=operation(), message="")
    catch err
        (; ok=false, value=nothing, message=sprint(showerror, err))
    end
    statuses = distributed_allgather(
        context,
        (; ok=local_result.ok, message=local_result.message),
    )
    failures = Tuple{Int,String}[
        (source - 1, status.message)
        for (source, status) in enumerate(statuses)
        if !status.ok
    ]
    isempty(failures) ||
        throw(DistributedCheckpointError(stage, failures))
    return local_result.value
end

function _injected_checkpoint_failure(
        stage::Symbol, selected_stage, selected_rank::Integer, rank::Integer)
    selected_stage === stage && selected_rank == rank ||
        return nothing
    error("injected $stage failure")
end

function _trajectory_model_identity(trajectory::DistributedMETTSTrajectory)
    metadata = trajectory.local_chain.metadata
    hasproperty(metadata, :problem_hash) || throw(ArgumentError(
        "distributed METTS trajectory has no problem model identity"))
    return metadata.problem_hash
end

function _checkpoint_mpi!(
        trajectory::DistributedMETTSTrajectory,
        path::AbstractString;
        keep::Int=3,
        metadata=NamedTuple(),
        failure_stage::Union{Nothing,Symbol}=nothing,
        failure_rank::Integer=-1)
    context = trajectory.context
    rank = distributed_rank(context)
    size = distributed_size(context)
    keep >= 0 ||
        throw(ArgumentError("checkpoint rotation count must be nonnegative"))
    size == length(trajectory.samples_per_rank) || throw(ArgumentError(
        "distributed trajectory rank count is inconsistent"))
    failure_stage in (
        nothing,
        :serialization,
        :shard_write,
        :before_manifest,
        :manifest_write,
    ) || throw(ArgumentError("unknown injected checkpoint failure stage"))
    failure_stage === nothing ||
        0 <= failure_rank < size || throw(ArgumentError(
            "injected checkpoint failure rank must lie in 0:$(size - 1)"))

    model_identity = _trajectory_model_identity(trajectory)
    model_identities = distributed_allgather(context, model_identity)
    if !all(
            identity -> isequal(identity, first(model_identities)),
            model_identities)
        failures = Tuple{Int,String}[
            (source - 1, "model identity differs across rank shards")
            for (source, identity) in enumerate(model_identities)
            if !isequal(identity, first(model_identities))
        ]
        throw(DistributedCheckpointError(:metadata, failures))
    end

    chain_steps = distributed_allgather(
        context, trajectory.local_chain.total_steps)
    generation = MPI.bcast(
        rank == distributed_root(context) ?
        _checkpoint_generation(path) : "",
        distributed_root(context),
        context.comm,
    )
    shard_name = _rank_shard_name(path, generation, rank)
    shard_path = _rank_shard_path(path, shard_name)
    local_shard = _collective_checkpoint_stage(
            context, :shard_write) do
        _injected_checkpoint_failure(
            :serialization, failure_stage, failure_rank, rank)
        shard_metadata = (;
            format=:graft_mpi_metts_shard_v2,
            generation,
            rank,
            nranks=size,
            total_steps=trajectory.local_chain.total_steps,
            model_identity,
            trajectory_metadata=trajectory.local_chain.metadata,
        )
        _injected_checkpoint_failure(
            :shard_write, failure_stage, failure_rank, rank)
        checkpoint!(
            trajectory.local_chain,
            shard_path;
            keep=0,
            metadata=merge(metadata, shard_metadata),
        )
        return (;
            name=shard_name,
            sha256=_file_sha256(shard_path),
            size_bytes=filesize(shard_path),
            total_steps=trajectory.local_chain.total_steps,
            model_identity,
        )
    end
    shards = distributed_allgather(context, local_shard)

    _collective_checkpoint_stage(context, :before_manifest) do
        _injected_checkpoint_failure(
            :before_manifest, failure_stage, failure_rank, rank)
    end

    if rank == distributed_root(context)
        manifest = (;
            format=:graft_mpi_metts_v2,
            generation,
            nranks=size,
            global_nsamples=trajectory.global_nsamples,
            samples_per_rank=copy(trajectory.samples_per_rank),
            chain_steps=Int.(chain_steps),
            model_identity,
            shards=getproperty.(shards, :name),
            shard_sha256=getproperty.(shards, :sha256),
            shard_sizes=Int.(getproperty.(shards, :size_bytes)),
        )
    end
    _collective_checkpoint_stage(context, :manifest_write) do
        rank == distributed_root(context) || return nothing
        _injected_checkpoint_failure(
            :manifest_write, failure_stage, failure_rank, rank)
        checkpoint!(manifest, path; keep, metadata)
    end
    return path
end

function checkpoint_mpi!(
        trajectory::DistributedMETTSTrajectory,
        path::AbstractString;
        keep::Int=3,
        metadata=NamedTuple())
    return _checkpoint_mpi!(trajectory, path; keep, metadata)
end

checkpoint!(
        trajectory::DistributedMETTSTrajectory,
        path::AbstractString;
        kwargs...) = checkpoint_mpi!(trajectory, path; kwargs...)

function _validate_v2_manifest(
        manifest,
        size::Int,
        expected_model_identity)
    required = (
        :generation,
        :nranks,
        :global_nsamples,
        :samples_per_rank,
        :chain_steps,
        :model_identity,
        :shards,
        :shard_sha256,
        :shard_sizes,
    )
    all(field -> hasproperty(manifest, field), required) ||
        throw(ArgumentError(
            "distributed METTS manifest is missing required fields"))
    manifest.nranks == size || throw(ArgumentError(
        "checkpoint needs $(manifest.nranks) ranks, got $size"))
    for field in (
            :samples_per_rank,
            :chain_steps,
            :shards,
            :shard_sha256,
            :shard_sizes)
        length(getproperty(manifest, field)) == size ||
            throw(ArgumentError(
                "distributed METTS manifest $field length mismatch"))
    end
    all(name -> name isa AbstractString &&
                !isabspath(name) &&
                basename(name) == name,
        manifest.shards) || throw(ArgumentError(
            "distributed METTS manifest contains a nonportable shard name"))
    expected_model_identity === nothing ||
        isequal(manifest.model_identity, expected_model_identity) ||
        throw(ArgumentError(
            "distributed METTS checkpoint belongs to a different model"))
    return nothing
end

function _validated_shard_state(
        shard_record,
        manifest,
        rank::Int,
        size::Int;
        expected_model_identity=nothing,
        legacy::Bool=false)
    shard_metadata = shard_record.metadata
    expected_format = legacy ?
        :graft_mpi_metts_shard_v1 : :graft_mpi_metts_shard_v2
    shard_metadata.format === expected_format ||
        throw(ArgumentError("invalid distributed METTS shard format"))
    shard_metadata.rank == rank ||
        throw(ArgumentError("distributed METTS shard rank mismatch"))
    shard_metadata.nranks == size ||
        throw(ArgumentError("distributed METTS shard communicator mismatch"))
    shard_record.state.total_steps == manifest.chain_steps[rank + 1] ||
        throw(ArgumentError("distributed METTS checkpoint mixes chain steps"))
    if !legacy
        shard_metadata.generation == manifest.generation ||
            throw(ArgumentError(
                "distributed METTS checkpoint mixes shard generations"))
        isequal(shard_metadata.model_identity, manifest.model_identity) ||
            throw(ArgumentError(
                "distributed METTS checkpoint mixes model identities"))
        hasproperty(shard_record.state.metadata, :problem_hash) &&
            isequal(
                shard_record.state.metadata.problem_hash,
                manifest.model_identity,
            ) || throw(ArgumentError(
                "distributed METTS shard has the wrong model identity"))
    elseif expected_model_identity !== nothing
        hasproperty(shard_record.state.metadata, :problem_hash) &&
            isequal(
                shard_record.state.metadata.problem_hash,
                expected_model_identity,
            ) || throw(ArgumentError(
                "distributed METTS checkpoint belongs to a different model"))
    end
    return shard_record.state
end

function _resume_mpi_v1(
        path::AbstractString,
        manifest,
        context::MPIContext;
        expected_model_identity=nothing)
    size = distributed_size(context)
    manifest.nranks == size || throw(ArgumentError(
        "checkpoint needs $(manifest.nranks) ranks, got $size"))
    rank = distributed_rank(context)
    shard_path = manifest.shards[rank + 1]
    shard_record = _collective_checkpoint_stage(
            context, :shard_read) do
        isfile(shard_path) || throw(ArgumentError(
            "distributed checkpoint shard is missing: $shard_path"))
        resume(shard_path)
    end
    state = _collective_checkpoint_stage(
            context, :shard_validation) do
        _validated_shard_state(
            shard_record,
            manifest,
            rank,
            size;
            expected_model_identity,
            legacy=true,
        )
    end
    return Graft.Thermal._distributed_trajectory(state, context)
end

function resume_mpi(
        path::AbstractString,
        context::MPIContext;
        expected_model_identity=nothing)
    manifest_record = _collective_checkpoint_stage(
            context, :manifest_read) do
        resume(path)
    end
    manifest = manifest_record.state
    hasproperty(manifest, :format) || throw(ArgumentError(
            "checkpoint is not a distributed METTS manifest"))
    if manifest.format === :graft_mpi_metts_v1
        trajectory = _resume_mpi_v1(
            path,
            manifest,
            context;
            expected_model_identity,
        )
        trajectory.samples_per_rank == manifest.samples_per_rank ||
            throw(ArgumentError(
                "distributed METTS shard sample counts mismatch"))
        trajectory.global_nsamples == manifest.global_nsamples ||
            throw(ArgumentError(
                "distributed METTS global sample count mismatch"))
        return (; state=trajectory, metadata=manifest_record.metadata)
    end
    manifest.format === :graft_mpi_metts_v2 || throw(ArgumentError(
        "checkpoint is not a supported distributed METTS manifest"))

    size = distributed_size(context)
    _collective_checkpoint_stage(context, :manifest_validation) do
        _validate_v2_manifest(
            manifest, size, expected_model_identity)
    end
    rank = distributed_rank(context)
    shard_path = _rank_shard_path(path, manifest.shards[rank + 1])
    shard_record = _collective_checkpoint_stage(
            context, :shard_read) do
        isfile(shard_path) || throw(ArgumentError(
            "distributed checkpoint shard is missing: $shard_path"))
        filesize(shard_path) == manifest.shard_sizes[rank + 1] ||
            throw(ArgumentError(
                "distributed checkpoint shard size mismatch: $shard_path"))
        _file_sha256(shard_path) == manifest.shard_sha256[rank + 1] ||
            throw(ArgumentError(
                "distributed checkpoint shard digest mismatch: $shard_path"))
        resume(shard_path)
    end
    state = _collective_checkpoint_stage(
            context, :shard_validation) do
        _validated_shard_state(
            shard_record,
            manifest,
            rank,
            size;
            expected_model_identity,
        )
    end
    trajectory = Graft.Thermal._distributed_trajectory(state, context)
    trajectory.samples_per_rank == manifest.samples_per_rank ||
        throw(ArgumentError("distributed METTS shard sample counts mismatch"))
    trajectory.global_nsamples == manifest.global_nsamples ||
        throw(ArgumentError("distributed METTS global sample count mismatch"))
    return (; state=trajectory, metadata=manifest_record.metadata)
end

end # module GraftMPIExt
