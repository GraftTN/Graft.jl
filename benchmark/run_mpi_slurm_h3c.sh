#!/usr/bin/env bash
#SBATCH --job-name=graft-mpi-large
#SBATCH --partition=h3c
#SBATCH --nodes=1
#SBATCH --ntasks=12
#SBATCH --cpus-per-task=4
#SBATCH --exclusive
#SBATCH --time=00:40:00
#SBATCH --output=benchmark/graft-mpi-large-%j.out

set -euo pipefail

repo="${SLURM_SUBMIT_DIR}"
mpiexec="/public/home/chenlj/bin/mpiexecjl"
julia="/public/home/chenlj/.local/bin/julia"

export JULIA_DEPOT_PATH="/tmp/graft-julia-depot:/public/home/chenlj/.julia"
export JULIA_NUM_THREADS=1
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OMP_DYNAMIC=false
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export GRAFT_MPI_BLAS_THREADS=4
export GRAFT_MPI_BENCH_ARM_DEPTH="${GRAFT_MPI_BENCH_ARM_DEPTH:-3}"
export GRAFT_MPI_BENCH_SNAPSHOTS="${GRAFT_MPI_BENCH_SNAPSHOTS:-36}"
export GRAFT_MPI_BENCH_BOND_DIM="${GRAFT_MPI_BENCH_BOND_DIM:-12}"
export GRAFT_MPI_BENCH_SAMPLES="${GRAFT_MPI_BENCH_SAMPLES:-3}"

cd "${repo}"

echo "GRAFT_SLURM_METADATA job=${SLURM_JOB_ID} node=${SLURMD_NODENAME} started=$(date --iso-8601=seconds)"
echo "GRAFT_SLURM_METADATA snapshots=${GRAFT_MPI_BENCH_SNAPSHOTS} bond_dimension=${GRAFT_MPI_BENCH_BOND_DIM} arm_depth=${GRAFT_MPI_BENCH_ARM_DEPTH}"

for ranks in 1 12; do
    export GRAFT_MPI_EXPECTED_SIZE="${ranks}"
    echo "GRAFT_SLURM_START ranks=${ranks} blas_threads=${GRAFT_MPI_BLAS_THREADS}"
    "${mpiexec}" --project=benchmark \
        -launcher fork -n "${ranks}" \
        -bind-to core:4 -map-by socket \
        "${julia}" --project=benchmark --startup-file=no \
        benchmark/mpi_slurm_large.jl
done

echo "GRAFT_SLURM_METADATA finished=$(date --iso-8601=seconds)"
