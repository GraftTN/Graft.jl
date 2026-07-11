"""
Cross-cutting — parallelization (architecture §8).

Roll-out order (per-milestone plan §12):
1. sector-block threading — free via TensorKit block sparsity (M0: the only level);
2. operator-term-level MPI Allreduce of H_eff·ψ (future MPI extension; DMRG
   and TDVP share it);
3. subtree-environment-level MPI — `EnvCache` is the communication unit; a rank
   owns a subtree (M5; partition-induced branches are natural ownership bounds);
4. same-depth node-level parallel updates — convergence risk, only after 3
   proves communication isn't the bottleneck (unscheduled).

MPI lands as a package extension (`GraftMPIExt`, §10.6) — the core stays free
of heavy deps. Data-structure obligations that are already honored: EnvCache
and checkpoints are subtree-dispatchable, no global implicit state (§9.9).
"""
module Parallel

import Base.Threads

export threaded_foreach

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

# TODO(M5): subtree ownership map type for EnvCache distribution

end # module Parallel
