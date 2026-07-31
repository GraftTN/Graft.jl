module GraftParallelMPIExt

using MPI
using GraftFoundation.Backend: AbstractTensorMap, blocks
import GraftParallel: AbstractRootDrivenContext, configure_parallel_runtime!,
    mpi_context, distributed_rank, distributed_size, distributed_root,
    distributed_barrier, distributed_allreduce_sum!, distributed_broadcast!,
    distributed_allgather, root_driven_solver

"""
    MPIContext

Explicit MPI execution context used by Graft's distributed algorithms. The
communicator is never stored in global package state and is not serializable.
"""
struct MPIContext{C} <: AbstractRootDrivenContext
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
    configure_runtime && configure_parallel_runtime!(
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

struct _RootDrivenMap{C<:MPIContext,W}
    context::C
    workspace::W
end

function _collective_workspace_call(
        context::MPIContext, workspace, input::AbstractTensorMap)
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

function root_driven_solver(solver, context::MPIContext, workspace)
    root = distributed_root(context)
    rank = distributed_rank(context)
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

end # module GraftParallelMPIExt
