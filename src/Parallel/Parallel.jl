"""
Cross-cutting — parallelization (architecture §8).

Roll-out order (per-milestone plan §12):
1. sector-block threading — free via TensorKit block sparsity;
2. operator-channel MPI reduction of H_eff·ψ — implemented in `GraftMPIExt`
   and shared by DMRG and TDVP through root-driven adaptive solvers;
3. subtree-environment-level MPI — ownership validation is implemented;
   distributed `EnvCache` transport and token scheduling remain future work;
4. same-depth node-level parallel updates — convergence risk, only after 3
   proves communication isn't the bottleneck (unscheduled).

MPI lives in a package extension (`GraftMPIExt`, §10.6) — the core stays free
of heavy deps. Data-structure obligations that are already honored: EnvCache
and checkpoints are subtree-dispatchable, no global implicit state (§9.9).
"""
module Parallel

import Base.Threads
using LinearAlgebra: BLAS
using TensorOperations: TensorOperations
using ..Trees: TreeTopology, nnodes, neighbors

export ParallelRuntimeConfig, ParallelRuntimeConfigurationError,
    BoundedFanoutDiagnostics, BoundedFanoutAdmissionError,
    BoundedFanoutItemError,
    parallel_runtime_config, threaded_foreach, bounded_threaded_foreach,
    configure_parallel_runtime!,
    AbstractDistributedContext, mpi_context, distributed_rank,
    distributed_size, distributed_root, distributed_isroot,
    distributed_barrier, distributed_allreduce_sum!,
    distributed_broadcast!, distributed_allgather,
    distributed_eigsolve, distributed_exponentiate,
    SubtreeOwnership, subtree_owner,
    local_nodes, boundary_edges

"""
    AbstractDistributedContext

Core-side marker for an explicitly supplied distributed execution context.
Concrete MPI state lives in `GraftMPIExt`; the core package never imports
MPI.jl and never consults a global communicator.
"""
abstract type AbstractDistributedContext end

"""Construct an MPI context. Methods are supplied by the MPI package extension."""
function mpi_context end

"""Zero-based rank of a distributed context."""
function distributed_rank end

"""Number of ranks in a distributed context."""
function distributed_size end

"""Zero-based root rank selected for root-only operations."""
function distributed_root end

distributed_isroot(context::AbstractDistributedContext) =
    distributed_rank(context) == distributed_root(context)

"""Collective barrier for a distributed context."""
function distributed_barrier end

"""In-place sum over every rank. MPI methods are supplied by `GraftMPIExt`."""
function distributed_allreduce_sum! end

"""Broadcast a mutable value from the context root or an explicit root."""
function distributed_broadcast! end

"""Collect one arbitrary Julia value from every rank, in rank order."""
function distributed_allgather end

"""Root-driven distributed eigensolve supplied by a distributed extension."""
function distributed_eigsolve end

"""Root-driven distributed exponential action supplied by an extension."""
function distributed_exponentiate end

"""
    SubtreeOwnership(topo, owners; nranks=maximum(owners)+1)

Validated zero-based node ownership for distributed tree work. Every nonempty
rank must own a connected induced subtree. Boundary edges are recorded in
canonical child-parent orientation. This value is communicator-independent
and can therefore be checkpointed without serializing an MPI handle.
"""
struct SubtreeOwnership
    topo::TreeTopology
    owners::Vector{Int}
    nranks::Int
    boundaries::Vector{Tuple{Int,Int}}
    function SubtreeOwnership(topo::TreeTopology, owners;
                              nranks::Integer=isempty(owners) ? 0 :
                                  maximum(owners) + 1)
        assigned = Int.(collect(owners))
        length(assigned) == nnodes(topo) ||
            throw(DimensionMismatch(
                "subtree ownership needs one owner per topology node"))
        nranks >= 1 || throw(ArgumentError("nranks must be positive"))
        all(rank -> 0 <= rank < nranks, assigned) ||
            throw(ArgumentError("subtree owners must lie in 0:$(nranks - 1)"))
        for rank in 0:(nranks - 1)
            nodes = findall(==(rank), assigned)
            isempty(nodes) && continue
            _connected_owned_subtree(topo, assigned, rank, first(nodes)) ==
                Set(nodes) ||
                throw(ArgumentError(
                    "nodes owned by rank $rank do not form a connected subtree"))
        end
        boundaries = Tuple{Int,Int}[]
        for child in 1:nnodes(topo)
            parent = topo.parent[child]
            parent == 0 && continue
            assigned[child] == assigned[parent] ||
                push!(boundaries, (child, parent))
        end
        return new(topo, assigned, Int(nranks), boundaries)
    end
end

function _connected_owned_subtree(topo, owners, rank, start)
    visited = Set{Int}()
    pending = Int[start]
    while !isempty(pending)
        node = pop!(pending)
        node in visited && continue
        owners[node] == rank || continue
        push!(visited, node)
        append!(pending, neighbors(topo, node))
    end
    return visited
end

subtree_owner(ownership::SubtreeOwnership, node::Integer) =
    ownership.owners[Int(node)]
local_nodes(ownership::SubtreeOwnership, rank::Integer) =
    findall(==(Int(rank)), ownership.owners)
boundary_edges(ownership::SubtreeOwnership) = copy(ownership.boundaries)

"""
    ParallelRuntimeConfig

Machine-readable snapshot of the process-global parallel runtime. `generation`
changes only when explicit configuration first takes ownership of the runtime
or changes an effective backend thread count. `active_regions` counts outer
Graft task fan-out regions that have started but not yet joined.
"""
struct ParallelRuntimeConfig
    julia_version::VersionNumber
    machine::String
    julia_threads::Int
    blas_vendor::Symbol
    blas_threads::Int
    strided_threads::Int
    configured::Bool
    generation::UInt64
    active_regions::Int
end

Base.:(==)(left::ParallelRuntimeConfig, right::ParallelRuntimeConfig) =
    all(
        field -> getfield(left, field) == getfield(right, field),
        fieldnames(ParallelRuntimeConfig),
    )

Base.hash(config::ParallelRuntimeConfig, seed::UInt) = foldl(
    (value, field) -> hash(getfield(config, field), value),
    fieldnames(ParallelRuntimeConfig);
    init=seed,
)

"""
    ParallelRuntimeConfigurationError

Raised before process-global backend mutation when a different configuration
is requested while Graft task fan-out is active.
"""
struct ParallelRuntimeConfigurationError <: Exception
    active_regions::Int
    current::NamedTuple
    requested::NamedTuple
end

"""
    BoundedFanoutDiagnostics

Machine-readable result of [`bounded_threaded_foreach`](@ref). `mode` is
`:threaded` only when at least one batch actually used more than one Julia
thread. `fallback` explains a serial disposition. Memory accounting is an
admission estimate: retained bytes are charged once and per-item bytes are
charged for every concurrently admitted item.
"""
struct BoundedFanoutDiagnostics
    mode::Symbol
    fallback::Symbol
    item_count::Int
    worker_limit::Int
    batch_count::Int
    max_batch_items::Int
    retained_memory_bytes::Int
    peak_admitted_bytes::Int
    memory_cap_bytes::Union{Nothing,Int}
    completed_items::Int
    cancelled_items::Int
end

"""
    BoundedFanoutAdmissionError

Raised before task launch when one item plus retained state cannot fit beneath
the explicit memory cap.
"""
struct BoundedFanoutAdmissionError <: Exception
    item_index::Int
    item_id::Any
    item_memory_bytes::Int
    retained_memory_bytes::Int
    memory_cap_bytes::Int
end

function Base.showerror(io::IO, err::BoundedFanoutAdmissionError)
    print(
        io,
        "bounded fan-out admission rejected item ",
        err.item_index,
        " (id=",
        repr(err.item_id),
        "): retained ",
        err.retained_memory_bytes,
        " + item ",
        err.item_memory_bytes,
        " bytes exceeds cap ",
        err.memory_cap_bytes,
    )
end

"""
    BoundedFanoutItemError

Deterministic item-indexed wrapper for a task failure. When several tasks in
one admitted batch fail, the lowest logical item index is reported. Later
batches are never admitted; `cancelled_items` counts those items.
"""
struct BoundedFanoutItemError <: Exception
    item_index::Int
    item_id::Any
    cause::Any
    backtrace::Any
    completed_items::Int
    attempted_items::Int
    cancelled_items::Int
end

function Base.showerror(io::IO, err::BoundedFanoutItemError)
    print(
        io,
        "bounded fan-out item ",
        err.item_index,
        " (id=",
        repr(err.item_id),
        ") failed after ",
        err.completed_items,
        " completed; ",
        err.cancelled_items,
        " later item",
        err.cancelled_items == 1 ? " was" : "s were",
        " not admitted: ",
    )
    showerror(io, err.cause)
end

function Base.showerror(io::IO, err::ParallelRuntimeConfigurationError)
    print(
        io,
        "cannot change process-global parallel runtime while ",
        err.active_regions,
        " Graft fan-out region",
        err.active_regions == 1 ? " is" : "s are",
        " active (current=",
        err.current,
        ", requested=",
        err.requested,
        ")",
    )
end

const _PARALLEL_RUNTIME_LOCK = ReentrantLock()
const _PARALLEL_RUNTIME_CONFIGURED = Ref(false)
const _PARALLEL_RUNTIME_GENERATION = Ref{UInt64}(0)
const _PARALLEL_RUNTIME_ACTIVE_REGIONS = Ref(0)

function _parallel_runtime_config_unlocked()
    return ParallelRuntimeConfig(
        VERSION,
        String(Sys.MACHINE),
        Threads.nthreads(),
        BLAS.vendor(),
        BLAS.get_num_threads(),
        TensorOperations.Strided.get_num_threads(),
        _PARALLEL_RUNTIME_CONFIGURED[],
        _PARALLEL_RUNTIME_GENERATION[],
        _PARALLEL_RUNTIME_ACTIVE_REGIONS[],
    )
end

"""
    parallel_runtime_config() -> ParallelRuntimeConfig

Return the effective Julia, BLAS, and Strided thread counts together with the
backend vendor, process identity, explicit-configuration generation, and
active Graft fan-out count. This function never mutates runtime state.
"""
function parallel_runtime_config()
    lock(_PARALLEL_RUNTIME_LOCK)
    try
        return _parallel_runtime_config_unlocked()
    finally
        unlock(_PARALLEL_RUNTIME_LOCK)
    end
end

"""
    configure_parallel_runtime!(; blas_threads=1, strided_threads=1)
        -> ParallelRuntimeConfig

Configure process-global backend thread pools before entering Graft's outer
Julia-thread fan-out regions. The defaults prevent BLAS and Strided from
nested threading inside each Graft task. Equal settings are idempotent and do
not advance the configuration generation. A different setting is rejected
before mutation while a Graft task fan-out region is active.

This function changes global runtime state and should be called once during
process setup, before concurrent work starts.
"""
function configure_parallel_runtime!(; blas_threads::Integer=1,
                                     strided_threads::Integer=1)
    blas_threads >= 1 || throw(ArgumentError("blas_threads must be positive"))
    strided_threads >= 1 || throw(ArgumentError("strided_threads must be positive"))
    requested = (;
        blas_threads=Int(blas_threads),
        strided_threads=Int(strided_threads),
    )
    lock(_PARALLEL_RUNTIME_LOCK)
    try
        current = (;
            blas_threads=BLAS.get_num_threads(),
            strided_threads=TensorOperations.Strided.get_num_threads(),
        )
        changed = current != requested
        if changed && _PARALLEL_RUNTIME_ACTIVE_REGIONS[] > 0
            throw(ParallelRuntimeConfigurationError(
                _PARALLEL_RUNTIME_ACTIVE_REGIONS[],
                current,
                requested,
            ))
        end
        first_configuration = !_PARALLEL_RUNTIME_CONFIGURED[]
        if changed
            try
                BLAS.set_num_threads(requested.blas_threads)
                TensorOperations.Strided.set_num_threads(
                    requested.strided_threads)
            catch
                BLAS.set_num_threads(current.blas_threads)
                TensorOperations.Strided.set_num_threads(
                    current.strided_threads)
                rethrow()
            end
        end
        if first_configuration || changed
            _PARALLEL_RUNTIME_GENERATION[] += 1
        end
        _PARALLEL_RUNTIME_CONFIGURED[] = true
        return _parallel_runtime_config_unlocked()
    finally
        unlock(_PARALLEL_RUNTIME_LOCK)
    end
end

function _with_parallel_runtime_region(operation)
    lock(_PARALLEL_RUNTIME_LOCK)
    try
        _PARALLEL_RUNTIME_ACTIVE_REGIONS[] += 1
    finally
        unlock(_PARALLEL_RUNTIME_LOCK)
    end
    try
        return operation()
    finally
        lock(_PARALLEL_RUNTIME_LOCK)
        try
            _PARALLEL_RUNTIME_ACTIVE_REGIONS[] -= 1
            _PARALLEL_RUNTIME_ACTIVE_REGIONS[] >= 0 ||
                error("parallel runtime active-region count underflow")
        finally
            unlock(_PARALLEL_RUNTIME_LOCK)
        end
    end
end

"""
    threaded_foreach(f, items; threaded=Threads.nthreads() > 1, minbatch=2) -> nothing

Run `f(item)` for every element of `items`, optionally using Julia threads.
This is the shared M1 block-loop primitive (§10.4): the per-item call goes
through a function barrier, kernels opt in explicitly with the `threaded`
keyword, and the serial fallback is deterministic. `items` may be any iterable;
non-indexable iterables are collected once before dispatch.
"""
function threaded_foreach(f, items; threaded::Bool=Threads.nthreads() > 1,
                          minbatch::Integer=2)
    minbatch >= 1 || throw(ArgumentError("minbatch must be positive"))
    xs = _indexable_items(items)
    if threaded && Threads.nthreads() > 1 && length(xs) >= minbatch
        _with_parallel_runtime_region() do
            Threads.@threads for i in eachindex(xs)
                _threaded_call(f, xs[i])
            end
        end
    else
        for x in xs
            _threaded_call(f, x)
        end
    end
    return nothing
end

"""
    bounded_threaded_foreach(f, items;
        threaded=Threads.nthreads() > 1,
        minbatch=2,
        item_memory_bytes=0,
        retained_memory_bytes=0,
        memory_cap_bytes=nothing,
        item_id=(index, item) -> index) -> BoundedFanoutDiagnostics

Run an item fan-out under an explicit P1 memory-admission and failure contract.
`item_memory_bytes` may be a nonnegative integer or `(index, item) -> bytes`;
`item_id` supplies an observable identifier for typed errors.

When threading is requested without `memory_cap_bytes`, execution falls back
to deterministic serial order and reports `fallback=:missing_memory_cap`.
With a cap, consecutive batches contain at most `Threads.nthreads()` items and
are greedily bounded by retained plus per-item admission bytes. A single item
that cannot fit is rejected before any item starts. A task failure joins its
already-admitted batch, reports the lowest failing item index, and prevents all
later batches from starting.
"""
function bounded_threaded_foreach(
    f,
    items;
    threaded::Bool=Threads.nthreads() > 1,
    minbatch::Integer=2,
    item_memory_bytes=0,
    retained_memory_bytes::Integer=0,
    memory_cap_bytes::Union{Nothing,Integer}=nothing,
    item_id=(index, item) -> index,
)
    minbatch >= 1 || throw(ArgumentError("minbatch must be positive"))
    retained_memory_bytes >= 0 ||
        throw(ArgumentError("retained_memory_bytes must be nonnegative"))
    memory_cap_bytes === nothing || memory_cap_bytes >= 0 ||
        throw(ArgumentError("memory_cap_bytes must be nonnegative"))

    xs = collect(items)
    nitems = length(xs)
    retained = Int(retained_memory_bytes)
    cap = memory_cap_bytes === nothing ? nothing : Int(memory_cap_bytes)
    ids = Any[item_id(index, item) for (index, item) in enumerate(xs)]
    costs = Vector{Int}(undef, nitems)
    for (index, item) in enumerate(xs)
        cost = item_memory_bytes isa Integer ?
            item_memory_bytes : item_memory_bytes(index, item)
        cost isa Integer ||
            throw(ArgumentError("item_memory_bytes must return an integer"))
        cost >= 0 ||
            throw(ArgumentError("item_memory_bytes must be nonnegative"))
        costs[index] = Int(cost)
    end

    if cap !== nothing
        retained <= cap ||
            throw(BoundedFanoutAdmissionError(
                0, :retained_state, 0, retained, cap))
        for index in eachindex(xs)
            costs[index] <= cap - retained ||
                throw(BoundedFanoutAdmissionError(
                    index, ids[index], costs[index], retained, cap))
        end
    end

    requested_threading = threaded && nitems >= minbatch
    candidate_threading = requested_threading && Threads.nthreads() > 1 &&
        cap !== nothing
    batches = UnitRange{Int}[]
    if candidate_threading
        start = 1
        while start <= nitems
            admitted = retained
            stop = start - 1
            while stop < nitems &&
                  stop - start + 1 < Threads.nthreads() &&
                  costs[stop + 1] <= cap - admitted
                stop += 1
                admitted += costs[stop]
            end
            # Per-item admission above guarantees progress.
            push!(batches, start:stop)
            start = stop + 1
        end
    end

    max_batch_items = isempty(batches) ? min(nitems, 1) :
        maximum(length, batches)
    use_threads = candidate_threading && max_batch_items > 1
    fallback = if use_threads
        :none
    elseif !threaded
        :threading_disabled
    elseif Threads.nthreads() == 1
        :single_thread
    elseif nitems < minbatch
        :below_minbatch
    elseif cap === nothing
        :missing_memory_cap
    else
        :memory_limited
    end

    if !use_threads
        completed = 0
        for index in eachindex(xs)
            try
                _threaded_call(f, xs[index])
                completed += 1
            catch err
                throw(BoundedFanoutItemError(
                    index,
                    ids[index],
                    err,
                    catch_backtrace(),
                    completed,
                    index,
                    nitems - index,
                ))
            end
        end
        peak = retained + (isempty(costs) ? 0 : maximum(costs))
        return BoundedFanoutDiagnostics(
            :serial,
            fallback,
            nitems,
            1,
            nitems,
            min(nitems, 1),
            retained,
            peak,
            cap,
            completed,
            0,
        )
    end

    completed = 0
    peak = retained
    for batch in batches
        failures = Vector{Any}(nothing, length(batch))
        # A BitVector packs multiple slots into one machine word, so workers
        # writing distinct logical indices would still race on shared storage.
        # Byte-sized slots preserve independent ownership per admitted item.
        successes = zeros(UInt8, length(batch))
        batch_admitted = retained
        for index in batch
            batch_admitted += costs[index]
        end
        peak = max(peak, batch_admitted)
        # Hand workers only the current batch. Julia scheduler tasks may retain
        # their most recent closure arguments briefly after synchronization;
        # passing the complete item vector here would retain every payload
        # beyond the admitted batch and invalidate the live-set model.
        batch_items = Any[xs[index] for index in batch]
        try
            _with_parallel_runtime_region() do
                _run_bounded_batch!(f, batch_items, failures, successes)
            end
        finally
            # Completed scheduler tasks may retain their argument container.
            # Clear payload references after join and before admitting another
            # batch so such task retention cannot extend the live tensor set.
            fill!(batch_items, nothing)
        end
        completed += count(!iszero, successes)
        failed_slot = findfirst(!isnothing, failures)
        if failed_slot !== nothing
            index = first(batch) + failed_slot - 1
            err, bt = failures[failed_slot]
            throw(BoundedFanoutItemError(
                index,
                ids[index],
                err,
                bt,
                completed,
                last(batch),
                nitems - last(batch),
            ))
        end
    end
    return BoundedFanoutDiagnostics(
        :threaded,
        :none,
        nitems,
        min(Threads.nthreads(), max_batch_items),
        length(batches),
        max_batch_items,
        retained,
        peak,
        cap,
        completed,
        0,
    )
end

_indexable_items(xs::AbstractArray) = xs
_indexable_items(xs) = collect(xs)

@noinline _threaded_call(f, x) = f(x)

@noinline function _run_bounded_batch!(
    f,
    batch_items::Vector{Any},
    failures::Vector{Any},
    successes::Vector{UInt8},
)
    Threads.@threads for slot in eachindex(successes)
        try
            _threaded_call(f, batch_items[slot])
            successes[slot] = 0x01
        catch err
            failures[slot] = (err, catch_backtrace())
        end
    end
    return nothing
end

end # module Parallel
