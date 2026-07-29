#!/usr/bin/env bash
#SBATCH --job-name=graft-mpi-p3
#SBATCH --partition=h3c
#SBATCH --nodes=4
#SBATCH --ntasks=32
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=4
#SBATCH --exclusive
#SBATCH --time=03:00:00
#SBATCH --output=benchmark/graft-mpi-p3-%j.out

set -euo pipefail

repo="${SLURM_SUBMIT_DIR}"
mpiexec="/public/home/chenlj/bin/mpiexecjl"
julia="/public/home/chenlj/.local/bin/julia"

export JULIA_DEPOT_PATH="/tmp/graft-julia-depot:/public/home/chenlj/.julia"
export OMP_DYNAMIC=false
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export GRAFT_MPI_BENCH_SAMPLES="${GRAFT_MPI_BENCH_SAMPLES:-5}"
export GRAFT_MPI_BENCH_ARM_DEPTH="${GRAFT_MPI_BENCH_ARM_DEPTH:-3}"
export GRAFT_MPI_BENCH_SNAPSHOTS="${GRAFT_MPI_BENCH_SNAPSHOTS:-36}"
export GRAFT_MPI_BENCH_BOND_DIM="${GRAFT_MPI_BENCH_BOND_DIM:-12}"
export GRAFT_MPI_OPERATOR_ARMS="${GRAFT_MPI_OPERATOR_ARMS:-4}"
export GRAFT_MPI_OPERATOR_ARM_DEPTH="${GRAFT_MPI_OPERATOR_ARM_DEPTH:-4}"
export GRAFT_MPI_OPERATOR_BOND_DIM="${GRAFT_MPI_OPERATOR_BOND_DIM:-12}"
export GRAFT_MPI_OPERATOR_REPETITIONS="${GRAFT_MPI_OPERATOR_REPETITIONS:-5}"
export GRAFT_MPI_OPERATOR_MEMORY_CAP_BYTES="${GRAFT_MPI_OPERATOR_MEMORY_CAP_BYTES:-20000000000}"

cd "${repo}"

run_mpi() {
    local ranks="$1"
    local threads="$2"
    local blas_threads="$3"
    local program="$4"

    export GRAFT_MPI_EXPECTED_SIZE="${ranks}"
    export GRAFT_MPI_EXPECTED_THREADS="${threads}"
    export GRAFT_MPI_BLAS_THREADS="${blas_threads}"
    export JULIA_NUM_THREADS="${threads}"
    export OMP_NUM_THREADS="${blas_threads}"
    export OPENBLAS_NUM_THREADS="${blas_threads}"
    export MKL_NUM_THREADS="${blas_threads}"

    echo "GRAFT_SLURM_START ranks=${ranks} threads=${threads} \
blas_threads=${blas_threads} program=${program}"
    "${mpiexec}" --project=benchmark \
        -launcher slurm -n "${ranks}" -ppn 8 \
        -bind-to core:4 \
        "${julia}" --project=benchmark --startup-file=no "${program}"
}

echo "GRAFT_SLURM_METADATA job=${SLURM_JOB_ID} \
nodes=${SLURM_JOB_NUM_NODES} nodelist=${SLURM_JOB_NODELIST} \
started=$(date --iso-8601=seconds)"
echo "GRAFT_SLURM_METADATA head=$(git rev-parse HEAD)"
git status --short

for ranks in 1 2 4 8 16 32; do
    run_mpi "${ranks}" 1 4 benchmark/mpi_slurm_large.jl
done

for ranks in 1 2 4 6; do
    export GRAFT_MPI_OPERATOR_EXPECT_ADMISSION_REJECTION=false
    run_mpi "${ranks}" 1 4 benchmark/mpi_slurm_operator.jl
done

export GRAFT_MPI_OPERATOR_EXPECT_ADMISSION_REJECTION=true
run_mpi 8 1 4 benchmark/mpi_slurm_operator.jl
unset GRAFT_MPI_OPERATOR_EXPECT_ADMISSION_REJECTION

for ranks in 2 4 8; do
    run_mpi "${ranks}" 4 1 test/mpi_tdvp.jl
done

echo "GRAFT_SLURM_METADATA finished=$(date --iso-8601=seconds)"
