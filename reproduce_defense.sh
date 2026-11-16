#!/usr/bin/env bash
# Re-run the round-2 defense sweeps (noise-trained and clean-fine-tuned victims, Sumo Ants +
# Sumo Humans) from the committed weights in models/, and redraw both round-2 figures, so they
# can be compared with figures/defense_sweep.png and figures/training_curves.png.
#
# Usage: ./reproduce_defense.sh       (~45 minutes on an 8-core CPU; resumes if interrupted)
#
# This does NOT re-train the victims (~5 hours; see the README for that command). The committed
# weights are the canonical defended victims.
#
# Each sweep is re-run from the config snapshot saved next to its committed results
# (results/<run>/config.json), not from configs/. Output goes to results/scratch/reproduce/
# (gitignored); the committed results are never touched.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

RUNS=(results/ants_defended results/ants_finetuned_clean
      results/humans_defended results/humans_finetuned_clean)
OUT=results/scratch/reproduce

for run in "${RUNS[@]}"; do
    # The commit whose code produced the committed results (first sweep invocation).
    commit=$(./run.sh python -c \
        "import json; print(json.load(open('${run}/run_info.json'))['invocations'][0]['git_commit'])")
    # Code, noise scales or victim weights changed since then (committed or not)? Then
    # differences may come from that change rather than from the machine.
    if ! git diff --quiet "${commit}" -- noisebackup results/obs_stats models; then
        echo "WARNING: ${run} was produced at commit ${commit:0:7}, and noisebackup/,"
        echo "         results/obs_stats/ or models/ has changed since. Differences may come"
        echo "         from that change, not just the machine. For an exact reproduction, first run:"
        echo "             git checkout ${commit:0:7}"
    fi
    echo "Reproducing ${run} (code from commit ${commit:0:7}) ..."
    ./run.sh python -m noisebackup.sweep --config "${run}/config.json" \
        --out "${OUT}/$(basename "${run}")"
done

# The original Zoo1 lines come from the committed round-1 results (./reproduce_attack.sh
# reproduces those).
./run.sh python -m noisebackup.plot \
    results/ants_zoo1_original/episodes.csv \
    "${OUT}/ants_defended/episodes.csv" \
    "${OUT}/ants_finetuned_clean/episodes.csv" \
    results/humans_zoo1_original/episodes.csv \
    "${OUT}/humans_defended/episodes.csv" \
    "${OUT}/humans_finetuned_clean/episodes.csv" \
    --out "${OUT}/defense_sweep.png"

# Training curves come straight from the committed training logs in models/.
./run.sh python -m noisebackup.training_curves \
    models/ants_defended models/ants_finetuned_clean \
    models/humans_defended models/humans_finetuned_clean \
    --out "${OUT}/training_curves.png"

echo
echo "Done. Compare:"
echo "  reproduced: ${OUT}/defense_sweep.png    (numbers: ${OUT}/defense_sweep.csv)"
echo "  committed:  figures/defense_sweep.png   (numbers: figures/defense_sweep.csv)"
echo "  reproduced: ${OUT}/training_curves.png  (numbers: ${OUT}/training_curves.csv)"
echo "  committed:  figures/training_curves.png (numbers: figures/training_curves.csv)"
