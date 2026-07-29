#!/usr/bin/env bash
#SBATCH --job-name=graft-mpi-admission
#SBATCH --partition=h3c
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=4
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --output=benchmark/graft-mpi-admission-%j.out

set -euo pipefail

repo="${SLURM_SUBMIT_DIR}"
mpiexec="/public/home/chenlj/bin/mpiexecjl"
julia="/public/home/chenlj/.local/bin/julia"

export JULIA_DEPOT_PATH="/tmp/graft-julia-depot:/public/home/chenlj/.julia"
export OMP_DYNAMIC=false
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export GRAFT_MPI_BENCH_SAMPLES="${GRAFT_MPI_BENCH_SAMPLES:-9}"
export GRAFT_MPI_OPERATOR_ARMS="${GRAFT_MPI_OPERATOR_ARMS:-4}"
export GRAFT_MPI_OPERATOR_ARM_DEPTH="${GRAFT_MPI_OPERATOR_ARM_DEPTH:-4}"
export GRAFT_MPI_OPERATOR_BOND_DIM="${GRAFT_MPI_OPERATOR_BOND_DIM:-12}"
export GRAFT_MPI_OPERATOR_REPETITIONS="${GRAFT_MPI_OPERATOR_REPETITIONS:-5}"
export GRAFT_MPI_OPERATOR_MEMORY_CAP_BYTES="${GRAFT_MPI_OPERATOR_MEMORY_CAP_BYTES:-20000000000}"

cd "${repo}"

run_operator() {
    local ranks="$1"
    local expect_rejection="$2"

    export GRAFT_MPI_EXPECTED_SIZE="${ranks}"
    export GRAFT_MPI_EXPECTED_THREADS=1
    export GRAFT_MPI_BLAS_THREADS=4
    export GRAFT_MPI_OPERATOR_EXPECT_ADMISSION_REJECTION="${expect_rejection}"
    export JULIA_NUM_THREADS=1
    export OMP_NUM_THREADS=4
    export OPENBLAS_NUM_THREADS=4
    export MKL_NUM_THREADS=4

    echo "GRAFT_SLURM_START ranks=${ranks} threads=1 \
blas_threads=4 expected_admission_rejection=${expect_rejection} \
program=benchmark/mpi_slurm_operator.jl"
    "${mpiexec}" --project=benchmark \
        -launcher slurm -n "${ranks}" -ppn 8 \
        -bind-to core:4 \
        "${julia}" --project=benchmark --startup-file=no \
        benchmark/mpi_slurm_operator.jl
}

echo "GRAFT_SLURM_METADATA job=${SLURM_JOB_ID} \
nodes=${SLURM_JOB_NUM_NODES} nodelist=${SLURM_JOB_NODELIST} \
started=$(date --iso-8601=seconds)"
echo "GRAFT_SLURM_METADATA head=$(git rev-parse HEAD)"
git status --short

for ranks in 1 4 6; do
    run_operator "${ranks}" false
done
run_operator 8 true

echo "GRAFT_SLURM_METADATA finished=$(date --iso-8601=seconds)"
