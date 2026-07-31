# Graft.jl

<p align="center">
  <img src="https://cdn.jsdelivr.net/gh/GraftTN/Graft.jl@main/assets/graftjl-logo.png" alt="Graft.jl logo" width="240">
</p>

A general-purpose tree tensor network core library. DMFT/EDMFT impurity-solver workflows are provided by the companion `GraftImpurity.jl` package, which depends on Graft rather than being embedded in it.

The architecture is largely inspired by [PyTreeNet](https://github.com/Drachier/PyTreeNet), and its tensor network foundation is built on [TensorKit.jl](https://github.com/QuantumKitHub/TensorKit.jl), which lets us exploit the abelian and non-abelian symmetries. It is designed to flexibly adopt new algorithms from papers for experimentation and verification — i.e., tree *graft*ing.


## Quick example

```julia
using Graft, GraftTestUtils, Random
using Graft.Backend: ℂ

topo = star_topology(3, 2)                       # generic star geometry
S = spin_ops()
phys = Dict(nodeid(topo, i) => S.P for i in 1:nnodes(topo))

H = OpSum()
for (c, p) in Graft.Trees.edges(topo)
    H += Term(-1.0, SiteOp(nodeid(topo, c), :Z, S.Z), SiteOp(nodeid(topo, p), :Z, S.Z))
end
for i in 1:nnodes(topo)
    H += Term(-0.9, SiteOp(nodeid(topo, i), :X, S.X))
end
O = ttno_from_opsum(H, topo, phys; hermitian=true)

ψ = random_ttns(Xoshiro(1), ComplexF64, topo, phys, ℂ^2)
ψ, energies = dmrg2!(ψ, O; trunc=TruncationScheme(maxdim=32))

ev = TDVP1_CBE(trunc=TruncationScheme(maxdim=64), d_tilde_max=16)
evolve!(ev, ψ, O, -0.05im, 100)                  # real-time evolution, bond-adaptive
```

## Tests

Use focused tests for the affected code by default. Run the full suite only for
broad or shared low-level changes; full acceptance uses four parallel shards.

```bash
# Focused test (replace with the affected test file)
julia --project=test --startup-file=no test/ttno_compression.jl

# Full acceptance: four shards in parallel
printf '%s\n' 1 2 3 4 | xargs -P4 -I{} env \
  GRAFT_TEST_SHARD_COUNT=4 GRAFT_TEST_SHARD_INDEX={} \
  JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  julia --project=test --startup-file=no test/runtests.jl
```

M1/M2 focused entrypoints:

```bash
julia --project=test --startup-file=no test/spectral.jl
julia --project=test --startup-file=no test/implicit_log_time.jl
julia --project=test --startup-file=no test/metts.jl
```

Impurity acceptance workflows (finite-mode Anderson-Holstein benchmarks,
Kondo scaling with the checked-in adapol bath artifact, Matsubara transforms,
and the CTSEG cross-check against committed reference data) live in the
companion [GraftImpurity.jl](https://github.com/GraftTN/GraftImpurity.jl)
package.

## Spectral and thermal analysis

`HankelDMD` (the matrix-pencil family), right-subspace `ESPRIT`,
`ARLeastSquares`/`linear_prediction`,
`DescendingRankSearch`, and `complex_time_krylov` provide the M1 spectral
post-processing layer. Scalar and array-valued exponential fits use explicit
rank, zero, channel-reduction, pruning, and mode policies; fit diagnostics are
kept separate from the pure `ExponentialSum` value. Complex-time Krylov accepts
either dense Gram matrices or TTNS snapshots plus a Hermitian TTNO.
`LeftSubspaceESPRIT` is an explicit time-major compatibility backend for
consumers with demonstrated noisy-parity requirements; it is not the default
`ESPRIT`.

The M2 layer includes deterministic purification, `METTS`, `HybridMETTS`,
and real-time purification with auxiliary `:backward`/custom evolution.

## Parallel runtime

When using Julia-level fan-out, launch Julia with one BLAS thread and configure
the Strided backend once before starting work:

```bash
JULIA_NUM_THREADS=24 OPENBLAS_NUM_THREADS=1 julia --project=.
```

```julia
using Graft
configure_parallel_runtime!()  # BLAS = 1, Strided = 1
```

This avoids nested BLAS/Strided threading inside each Graft task. Benchmark a
different backend thread count explicitly for workloads dominated by one large
contraction; sector-rich workloads with many small blocks should keep both at
one. On multi-socket hosts, choose thread count and CPU affinity so work is
balanced across NUMA domains. Thermal correlators and dense thermal references
keep fan-out off by default because each active item retains its own state and
environments; opt in with `threaded=true` after budgeting that per-item memory.
They and RSVD probe generation accept `threaded`/`minbatch` or
`rsvd_threaded`/`rsvd_minbatch` controls for serial A/B runs and
workload-specific granularity.

Tier-3 one- and two-site TTNO channel slicing is experimental and remains off
by default. Large Krylov maps can opt in through `eff_h1`, `eff_h2`, the DMRG
drivers, or the TDVP evolvers, with an explicit contraction live-memory budget:

```julia
ev = TDVP1(threaded_channels=true, channel_slices=3,
           channel_min_flops=1_000_000,
           channel_memory_cap_bytes=2_000_000_000)
```

Solver/evolver entry points default `channel_min_flops` to `1_000_000`, so
small maps fall back before slice construction or memory-cap enforcement.
Direct `eff_h1`/`eff_h2` calls default the gate to zero to preserve explicit
opt-in behavior. The gate uses stored-sector FLOPs when the planner can model
them and otherwise falls back to the dense estimate. On the current 24-thread
star benchmarks, one-site maps cross break-even near χ=32 and reach about
1.25× at χ=64. Two-site maps use a cost model to choose between internal and
external TTNO edges; the 3×1 star is 0.63–0.67× at χ=16 but 1.22–1.53× at χ=64.
Keep BLAS and Strided at one thread when this outer fan-out is active.

### MPI extension

MPI support is loaded only when MPI.jl is present in the application
environment. Construct one explicit context and pass it to the operation that
should be distributed; Graft does not consult a global communicator:

```julia
using Graft, MPI

context = mpi_context() # COMM_WORLD; also sets BLAS/Strided threads to 1

ev = TDVP1(
    distributed=context,
    channel_slices=24,
    channel_min_flops=1_000_000,
    channel_memory_cap_bytes=2_000_000_000,
)
result = complex_time_krylov(snapshots, H; distributed=context)
```

`eff_h1` and `eff_h2` replicate the state, assign TTNO-channel slices across
ranks and Julia threads, and sum partial TensorMaps block by block. DMRG and
TDVP use a root-driven Krylov protocol: rank zero owns adaptive solver control,
while every matvec broadcasts its input and all ranks evaluate their assigned
channels. This avoids collective-order divergence when local floating-point
convergence decisions differ.

METTS and HybridMETTS distribute independent Markov chains:

```julia
trajectory = thermalize(
    METTS(; rng, nsamples=600), problem, beta;
    evolver, distributed=context,
)
checkpoint!(trajectory, "metts-mpi.jld2")
trajectory = resume_mpi("metts-mpi.jld2", context).state
```

The global sample count must provide at least one sample per rank. RNG streams
are deterministically separated by rank. Checkpoints contain one atomic shard
per rank plus a root-written manifest that validates rank count, sample count,
and chain step before resume.

Launch Julia with the MPI implementation selected by MPI.jl and four Julia
threads per rank, for example:

```bash
mpiexecjl -n 6 julia --project=/path/to/app --threads=4 mpi_job.jl
```

The repository keeps functional, adaptive-solver, and performance checks
separate:

```bash
mpiexecjl -n 6 julia --project=/path/to/mpi-test-env --threads=4 test/mpi_smoke.jl
mpiexecjl -n 6 julia --project=/path/to/mpi-test-env --threads=4 test/mpi_solver_smoke.jl
mpiexecjl -n 6 julia --project=/path/to/mpi-test-env --threads=4 test/mpi_tdvp.jl
mpiexecjl -n 6 julia --project=/path/to/mpi-test-env --threads=4 test/mpi_speedup.jl
```

On the current 6-rank, 4-threads-per-rank test host, the warmed
ComplexTimeKrylov benchmark reports 0.274 s serial versus 0.059 s MPI, or
4.646x. `GRAFT_MPI_MIN_SPEEDUP`, `GRAFT_MPI_BENCH_SAMPLES`,
`GRAFT_MPI_BENCH_SNAPSHOTS`, and `GRAFT_MPI_BENCH_BOND_DIM` control the
acceptance workload.

The larger single-node SLURM benchmark in `benchmark/mpi_slurm_large.jl` uses
ten tree nodes, 36 snapshots, and bond dimension 12. On the `h3c` partition's
48-core `h10` node (job 71949), three warmed samples gave 2.70044 s with one
MPI rank and four BLAS threads versus 0.289441 s with 12 MPI ranks and four
BLAS threads per rank. This is a 9.3298x speedup and 77.75% parallel
efficiency. Both configurations produced identical overlap and Hamiltonian
norms, energy sums, retained rank, and maximum residual. Reproduce both
configurations in one exclusive-node allocation with:

```bash
sbatch benchmark/run_mpi_slurm_h3c.sh
```

## Algorithmic References and Provenance

References are grouped by the Graft functionality they inform. Each entry states
whether it is an implementation basis or a design reference; citing a method
does not imply that every variant in the paper is implemented.

### Tree Operators and Software Architecture

1. **TTNO state diagrams** — *implemented; algorithmic basis*

   R. M. Milbradt, Q. Huang, and C. B. Mendl, “State Diagrams to determine Tree Tensor Network Operators,” *SciPost Physics Core* **7**, 036 (2024).
   [DOI](https://doi.org/10.21468/SciPostPhysCore.7.2.036) ·
   [arXiv](https://arxiv.org/abs/2311.13433)

   **Provenance:** Basis for constructing TTNOs from operator sums through state diagrams.

2. **Tree-network architecture** — *implemented; PyTreeNet design reference*

   R. M. Milbradt, Q. Huang, and C. B. Mendl, “PyTreeNet: A Python Library for easy Utilisation of Tree Tensor Networks,” arXiv:2407.13249 (2024).
   [arXiv](https://arxiv.org/abs/2407.13249)

   **Provenance:** Informs the package's tree-network organization, terminology, and parts of its TDVP implementation lineage.

### Ground-State and Time-Evolution Algorithms

1. **Tree TDVP / ForkTPS** — *implemented; algorithmic basis*

   D. Bauernfeind and M. Aichhorn, “Time Dependent Variational Principle for Tree Tensor Networks,” *SciPost Physics* **8**, 024 (2020).
   [DOI](https://doi.org/10.21468/SciPostPhys.8.2.024) ·
   [arXiv](https://arxiv.org/abs/1908.03090)

   **Provenance:** Basis for TDVP sweeps and local time evolution on tree tensor networks.

2. **CBE-TDVP** — *implemented; adapted algorithmic basis*

   J.-W. Li, A. Gleis, and J. von Delft, “Time-dependent variational principle with controlled bond expansion for matrix product states,” *Physical Review Letters* **133**, 026401 (2024).
   [DOI](https://doi.org/10.1103/PhysRevLett.133.026401) ·
   [arXiv](https://arxiv.org/abs/2208.10972)

   **Provenance:** Basis for controlled bond expansion, adapted from chains to tree tensor networks.

3. **DMRG3S** — *implemented; algorithmic basis*

   C. Hubig, I. P. McCulloch, U. Schollwöck, and F. A. Wolf, “A Strictly Single-Site DMRG Algorithm with Subspace Expansion,” *Physical Review B*
   **91**, 155115 (2015).
   [DOI](https://doi.org/10.1103/PhysRevB.91.155115) ·
   [arXiv](https://arxiv.org/abs/1501.05504)

   **Provenance:** Design reference for single-site DMRG with subspace expansion.

4. **RSVD post-expansion** — *implemented; algorithmic basis*

   I. P. McCulloch and J. J. Osborne, “Comment on ‘Controlled Bond Expansion for Density Matrix Renormalization Group Ground State Search at Single-Site Costs’ (Extended Version),” arXiv:2403.00562 (2024).
   [arXiv](https://arxiv.org/abs/2403.00562)

   **Provenance:** Design reference for randomized-SVD post-expansion choices.

5. **Global Krylov** — *design reference*

   S. Paeckel, T. Köhler, A. Swoboda, S. R. Manmana, U. Schollwöck, and C. Hubig, “Time-evolution methods for matrix-product states,” *Annals of
   Physics* **411**, 167998 (2019).
   [DOI](https://doi.org/10.1016/j.aop.2019.167998) ·
   [arXiv](https://arxiv.org/abs/1901.05824)

   **Provenance:** Design reference for global Krylov time evolution.

6. **GSE/LSE TDVP** — *algorithmic basis*

   M. Yang and S. R. White, “Time Dependent Variational Principle with Ancillary Krylov Subspace,” *Physical Review B* **102**, 094315 (2020).
   [DOI](https://doi.org/10.1103/PhysRevB.102.094315) ·
   [arXiv](https://arxiv.org/abs/2005.06104)

   **Provenance:** Global ancillary-Krylov foundation for the planned GSE/LSE expansion family.

7. **Implicit logarithmic-time evolution** — *implemented; algorithmic basis*

   J. P. Zima, E. M. Stoudenmire, S. R. White, O. Parcollet, and J. Kaye, “Fast Tensor Network Imaginary Time Evolution by Implicit Stepping on Logarithmic Grids,” arXiv:2606.02930 (2026).
   [arXiv](https://arxiv.org/abs/2606.02930)

   **Provenance:** Design reference for implicit imaginary-time stepping on logarithmic grids.

### Thermal-State Algorithms

1. **Projected purification for bosons** — *implemented; algorithmic basis*

   T. Köhler, J. Stolpp, and S. Paeckel, “Efficient and Flexible Approach to Simulate Low-Dimensional Quantum Lattice Models with Large Local Hilbert Spaces,” *SciPost Physics* **10**, 058 (2021).
   [DOI](https://doi.org/10.21468/SciPostPhys.10.3.058) ·
   [arXiv](https://arxiv.org/abs/2008.08466)

   **Provenance:** Algorithmic basis for the planned projected-purification treatment of large bosonic local spaces.

Additional annotated literature notes and working reference material live in the
sibling `GraftHarness` checkout.

## License

Graft.jl is licensed under the [Apache License 2.0](LICENSE).
