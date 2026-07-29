#!/usr/bin/env bash
set -euo pipefail

for workload in thermal rsvd; do
    for julia_threads in 1 2 4 8; do
        for mode in serial threaded; do
            python3 benchmark/monitor_p1_fanout.py \
                --workload "${workload}" \
                --mode "${mode}" \
                --threads "${julia_threads}" \
                --samples "${GRAFT_P1_SAMPLES:-5}" \
                --memory-cap-bytes "${GRAFT_P1_MEMORY_CAP_BYTES:-2147483648}"
        done
    done
done
