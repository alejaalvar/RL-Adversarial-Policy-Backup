#!/usr/bin/env bash
# Follow-up experiments (queued 2026-11-21), run unattended, one after another:
#   1. held-out opponents: the six committed sweeps against Zoo2 and Zoo3 (12 sweeps)
#   2. extra training seed: the four fine-tuning runs with seed 2000 / noise seed 2001,
#      each then swept against Zoo1
#   3. longer training: the four fine-tuning runs for 20M steps, each then swept against Zoo1
# then results/followup_summary.md (followup/summarize.py).
#
# Start it so it survives the terminal closing:
#   setsid nohup followup/run_followups.sh < /dev/null > /dev/null 2>&1 &
# Follow it:  tail -f results/followup/run.log      Steps so far:  cat results/followup/status.tsv
#
# Writes only new folders (results/<run>_opp2|_opp3|_seed2|_20M/, models/<run>_seed2|_20M/,
# results/followup/) and never touches committed models/, results/ or figures/: sweeps refuse a
# folder made with a different config, and fine-tuning refuses any existing folder. Safe to
# restart: finished sweeps and trained victims are skipped, interrupted sweeps resume.
# Threat model: direct observation (noise added straight to the victim's observations).
set -uo pipefail
cd "$(dirname "$0")/.."

OUT=results/followup
mkdir -p "${OUT}"
STATUS="${OUT}/status.tsv"
exec >>"${OUT}/run.log" 2>&1

START_COMMIT=$(git rev-parse HEAD)
# Everything that decides which games are played. If any of it changes while the queue runs,
# the remaining steps are skipped rather than mixing two versions of the experiment.
GUARDED=(noisebackup results/obs_stats models)
RUNS=(ants_defended humans_defended ants_finetuned_clean humans_finetuned_clean)

record() {  # record <step> <OK|FAILED|SKIPPED> <detail>
    printf '%s\t%s\t%s\t%s\n' "$(date -Is)" "$1" "$2" "$3" | tee -a "${STATUS}"
}

step() {  # step <name> <command...>
    local name=$1
    shift
    if ! git diff --quiet "${START_COMMIT}" -- "${GUARDED[@]}"; then
        record "${name}" SKIPPED "${GUARDED[*]} changed since ${START_COMMIT:0:7}"
        return 1
    fi
    echo "=== $(date -Is) START ${name}: $*"
    local t0=${SECONDS}
    "$@" </dev/null
    local rc=$?
    if [[ ${rc} -eq 0 ]]; then
        record "${name}" OK "$(((SECONDS - t0) / 60)) min"
    else
        record "${name}" FAILED "exit ${rc} after $(((SECONDS - t0) / 60)) min"
    fi
    return ${rc}
}

sweep() {  # sweep <run_name>: configs/followup/<run_name>.json -> results/<run_name>/
    local run=$1 config="configs/followup/$1.json"
    local victim
    victim=$(python3 -c "import json,sys; v=json.load(open(sys.argv[1]))['victim']; \
print(v['path'] if v['type'] == 'finetuned' else '')" "${config}")
    if [[ -n ${victim} && ! -f ${victim} ]]; then
        record "sweep ${run}" SKIPPED "victim ${victim} missing (training failed or skipped)"
        return 1
    fi
    step "sweep ${run}" ./run.sh python -m noisebackup.sweep --config "${config}"
}

train() {  # train <name> <config> <out_dir> [extra finetune args]
    local name=$1 config=$2 out=$3
    shift 3
    if [[ -f ${out}/weights.pkl ]]; then
        record "train ${name}" OK "already trained (${out}/weights.pkl exists)"
        return 0
    fi
    step "train ${name}" ./run.sh python -m noisebackup.finetune --config "${config}" --out "${out}" "$@"
}

echo "=== $(date -Is) follow-up queue started at commit ${START_COMMIT}"
echo "--- uncommitted files at start (the runs record git_dirty: true because of these):"
git status --porcelain --untracked-files=all | grep -v -E '^\?\? (models|results)/'

# 1. Held-out opponents.
for opp in 2 3; do
    for base in ants_zoo1_original humans_zoo1_original \
        ants_defended humans_defended ants_finetuned_clean humans_finetuned_clean; do
        sweep "${base}_opp${opp}"
    done
done

# 2. Extra training seed.
for run in "${RUNS[@]}"; do
    train "${run}_seed2" "configs/followup/finetune/${run}_seed2.json" "models/${run}_seed2"
done
for run in "${RUNS[@]}"; do
    sweep "${run}_seed2"
done

# 3. Twice as long (20M steps), committed training configs otherwise unchanged.
for run in "${RUNS[@]}"; do
    train "${run}_20M" "configs/finetune/${run}.json" "models/${run}_20M" --timesteps 20000000
done
for run in "${RUNS[@]}"; do
    sweep "${run}_20M"
done

# The summary only reads results, so it runs even if the guarded folders changed.
echo "=== $(date -Is) START summary"
if ./run.sh python followup/summarize.py </dev/null; then
    record "summary" OK "results/followup_summary.md"
else
    record "summary" FAILED "exit $?; rerun: ./run.sh python followup/summarize.py"
fi
echo "=== $(date -Is) follow-up queue finished"
