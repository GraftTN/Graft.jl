#!/usr/bin/env bash
set -euo pipefail

samples="${GRAFT_CONTRACTION_BASELINE_SAMPLES:-3}"
bond_dimension="${GRAFT_CONTRACTION_BASELINE_BOND_DIM:-4}"
matvecs="${GRAFT_CONTRACTION_BASELINE_MATVECS:-8}"

for topology in chain star fork balanced; do
    for julia_threads in 1 2 4 8; do
        env \
            GRAFT_CONTRACTION_CASE="${topology}" \
            GRAFT_CONTRACTION_BASELINE_SAMPLES="${samples}" \
            GRAFT_CONTRACTION_BASELINE_BOND_DIM="${bond_dimension}" \
            GRAFT_CONTRACTION_BASELINE_MATVECS="${matvecs}" \
            julia --startup-file=no --project=. \
                -t"${julia_threads}" benchmark/contraction_runtime.jl
    done
done

# A bounded heap hint is a comparison cell, not a recommended default. The
# report compares its allocations, GC time, RSS, and result checks against the
# stock 4-thread chain cell above.
env \
    GRAFT_CONTRACTION_CASE=chain \
    GRAFT_CONTRACTION_BASELINE_SAMPLES="${samples}" \
    GRAFT_CONTRACTION_BASELINE_BOND_DIM="${bond_dimension}" \
    GRAFT_CONTRACTION_BASELINE_MATVECS="${matvecs}" \
    julia --startup-file=no --project=. --heap-size-hint=512M \
        -t4 benchmark/contraction_runtime.jl
