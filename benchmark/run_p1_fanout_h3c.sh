#!/usr/bin/env bash
#SBATCH --job-name=graft-p1-fanout
#SBATCH --partition=h3c
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --array=0-15%4
#SBATCH --output=benchmark/graft-p1-fanout-%A_%a.out

set -euo pipefail

repo="${SLURM_SUBMIT_DIR}"
task="${SLURM_ARRAY_TASK_ID}"
cell=$((task % 8))
threads=(1 1 2 2 4 4 8 8)
modes=(serial threaded serial threaded serial threaded serial threaded)

if (( task < 8 )); then
    workload=thermal
else
    workload=rsvd
fi
julia_threads="${threads[cell]}"
mode="${modes[cell]}"

export PATH="/public/home/chenlj/.local/bin:${PATH}"
export JULIA_DEPOT_PATH="/tmp/graft-p1-depot-${SLURM_ARRAY_JOB_ID}-${task}:/public/home/chenlj/.julia"
export OMP_DYNAMIC=false
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export GRAFT_P1_THERMAL_TAUS="${GRAFT_P1_THERMAL_TAUS:-9}"
export GRAFT_P1_RSVD_BLOCKS="${GRAFT_P1_RSVD_BLOCKS:-8}"
export GRAFT_P1_RSVD_BLOCK_DIM="${GRAFT_P1_RSVD_BLOCK_DIM:-1024}"

cd "${repo}"

echo "GRAFT_SLURM_METADATA job=${SLURM_ARRAY_JOB_ID} task=${task} \
node=${SLURMD_NODENAME} started=$(date --iso-8601=seconds)"
echo "GRAFT_SLURM_METADATA head=$(git rev-parse HEAD)"
git status --short
echo "GRAFT_SLURM_START workload=${workload} mode=${mode} \
julia_threads=${julia_threads} blas_threads=1 strided_threads=1"

python3 benchmark/monitor_p1_fanout.py \
    --workload "${workload}" \
    --mode "${mode}" \
    --threads "${julia_threads}" \
    --samples "${GRAFT_P1_SAMPLES:-5}" \
    --memory-cap-bytes "${GRAFT_P1_MEMORY_CAP_BYTES:-2147483648}"

echo "GRAFT_SLURM_METADATA finished=$(date --iso-8601=seconds)"
