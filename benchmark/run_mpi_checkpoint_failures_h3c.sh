#!/usr/bin/env bash
#SBATCH --job-name=graft-mpi-ckpt-fail
#SBATCH --partition=h3c
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --output=benchmark/graft-mpi-ckpt-fail-%j.out

set -euo pipefail

repo="${SLURM_SUBMIT_DIR}"
mpiexec="/public/home/chenlj/bin/mpiexecjl"
julia="/public/home/chenlj/.local/bin/julia"

export JULIA_DEPOT_PATH="/tmp/graft-julia-depot:/public/home/chenlj/.julia"
export JULIA_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export GRAFT_MPI_EXPECTED_THREADS=1

cd "${repo}"

echo "GRAFT_SLURM_METADATA job=${SLURM_JOB_ID} \
nodes=${SLURM_JOB_NUM_NODES} nodelist=${SLURM_JOB_NODELIST} \
started=$(date --iso-8601=seconds)"
echo "GRAFT_SLURM_METADATA head=$(git rev-parse HEAD)"
git status --short

for ranks in 2 4 8; do
    export GRAFT_MPI_EXPECTED_SIZE="${ranks}"
    echo "GRAFT_SLURM_START ranks=${ranks} threads=1 blas_threads=1 \
program=test/mpi_checkpoint_failures.jl"
    "${mpiexec}" --project=benchmark \
        -launcher slurm -n "${ranks}" -ppn 8 \
        -bind-to core:1 \
        "${julia}" --project=benchmark --startup-file=no \
        test/mpi_checkpoint_failures.jl
done

echo "GRAFT_SLURM_METADATA finished=$(date --iso-8601=seconds)"
