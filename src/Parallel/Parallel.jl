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

export threaded_foreach, configure_parallel_runtime!,
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
    configure_parallel_runtime!(; blas_threads=1, strided_threads=1) -> NamedTuple

Configure process-global backend thread pools before entering Graft's outer
Julia-thread fan-out regions. The defaults prevent BLAS and Strided from
nested threading inside each Graft task. Returns the effective Julia, BLAS,
and Strided thread counts.

This function changes global runtime state and should be called once during
process setup, before concurrent work starts.
"""
function configure_parallel_runtime!(; blas_threads::Integer=1,
                                     strided_threads::Integer=1)
    blas_threads >= 1 || throw(ArgumentError("blas_threads must be positive"))
    strided_threads >= 1 || throw(ArgumentError("strided_threads must be positive"))
    BLAS.set_num_threads(blas_threads)
    TensorOperations.Strided.set_num_threads(strided_threads)
    return (; julia_threads=Threads.nthreads(),
            blas_threads=BLAS.get_num_threads(),
            strided_threads=TensorOperations.Strided.get_num_threads())
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
        Threads.@threads for i in eachindex(xs)
            _threaded_call(f, xs[i])
        end
    else
        for x in xs
            _threaded_call(f, x)
        end
    end
    return nothing
end

_indexable_items(xs::AbstractArray) = xs
_indexable_items(xs) = collect(xs)

@noinline _threaded_call(f, x) = f(x)

end # module Parallel
