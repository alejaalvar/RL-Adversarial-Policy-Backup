#!/usr/bin/env bash
# Re-run the round-1 attack sweeps (Sumo Ants + Sumo Humans) exactly as committed, and draw
# the figure, so it can be compared with figures/attack_sweep.png.
#
# Usage: ./reproduce_attack.sh        (~25 minutes on an 8-core CPU; resumes if interrupted)
#
# Each sweep is re-run from the config snapshot saved next to its committed results
# (results/<run>/config.json), not from configs/, so later edits to configs/ can't change what
# is being reproduced. Output goes to results/scratch/reproduce/ (gitignored); the committed
# results are never touched.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

RUNS=(results/ants_zoo1_original results/humans_zoo1_original)
OUT=results/scratch/reproduce

for run in "${RUNS[@]}"; do
    # The commit whose code produced the committed results (first sweep invocation).
    commit=$(./run.sh python -c \
        "import json; print(json.load(open('${run}/run_info.json'))['invocations'][0]['git_commit'])")
    # Code or noise scales changed since then (committed or not)? Then differences may come from
    # that change rather than from the machine.
    if ! git diff --quiet "${commit}" -- noisebackup results/obs_stats; then
        echo "WARNING: ${run} was produced at commit ${commit:0:7}, and noisebackup/ or"
        echo "         results/obs_stats/ has changed since. Differences may come from that"
        echo "         change, not just the machine. For an exact reproduction, first run:"
        echo "             git checkout ${commit:0:7}"
    fi
    echo "Reproducing ${run} (code from commit ${commit:0:7}) ..."
    ./run.sh python -m noisebackup.sweep --config "${run}/config.json" \
        --out "${OUT}/$(basename "${run}")"
done

./run.sh python -m noisebackup.plot \
    "${OUT}/ants_zoo1_original/episodes.csv" \
    "${OUT}/humans_zoo1_original/episodes.csv" \
    --out "${OUT}/attack_sweep.png"

echo
echo "Done. Compare:"
echo "  reproduced: ${OUT}/attack_sweep.png  (numbers: ${OUT}/attack_sweep.csv)"
echo "  committed:  figures/attack_sweep.png  (numbers: figures/attack_sweep.csv)"
