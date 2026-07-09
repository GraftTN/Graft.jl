"""
Cross-cutting — parallelization (architecture §8). **All TODO.**

Roll-out order (per-milestone plan §12):
1. sector-block threading — free via TensorKit block sparsity (M0: the only level);
2. operator-term-level MPI Allreduce of H_eff·ψ (M1; DMRG and TDVP share it);
3. subtree-environment-level MPI — `EnvCache` is the communication unit; a rank
   owns a subtree (M5; partition-induced branches are natural ownership bounds);
4. same-depth node-level parallel updates — convergence risk, only after 3
   proves communication isn't the bottleneck (unscheduled).

MPI lands as a package extension (`GRAFTMPIExt`, §10.6) — the core stays free
of heavy deps. Data-structure obligations that are already honored: EnvCache
and checkpoints are subtree-dispatchable, no global implicit state (§9.9).
"""
module Parallel

# TODO(M1): threaded block loops helper (function barrier per block, §10.4)
# TODO(M5): subtree ownership map type for EnvCache distribution

end # module Parallel
