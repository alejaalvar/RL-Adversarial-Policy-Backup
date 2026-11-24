# RL Adversarial Policy: backup plan

CS446 term project backup: how robust are pretrained "zoo" victims (Bansal et al., 2018) to
**Gaussian noise on their observations**, and does one noise-augmented fine-tuning defense help?

Builds on [HumanCompatibleAI/adversarial-policies](https://github.com/HumanCompatibleAI/adversarial-policies)
(Gleave et al., ICLR 2020), imported from its Docker image rather than copied.

> **Threat model: direct observation.** Noise is added straight to the victim's observations.
> This differs from the paper's *natural-observation* attack, where an adversary can only act
> in the shared environment.

## Setup

Everything runs inside the original project's 2019 container image (Python 3.7, TensorFlow 1.13,
MuJoCo), so the only things you install are podman and git.

1. **Install podman** with your distribution's package manager
   ([install guide](https://podman.io/docs/installation)). For example:

   ```bash
   sudo apt install podman   # Debian / Ubuntu / Linux Mint
   sudo dnf install podman   # Fedora
   ```

   Podman (not Docker) is required: it runs the container as you, so the results it writes
   stay owned by you. The image can't run as a normal user under Docker.

2. **Clone this repo:**

   ```bash
   git clone git@github.com:alejaalvar/RL-Adversarial-Policy-Backup.git
   cd RL-Adversarial-Policy-Backup
   ```

3. **Get the free MuJoCo key** (once):

   ```bash
   mkdir -p ~/.mujoco && curl -o ~/.mujoco/mjkey.txt https://www.roboti.us/file/mjkey.txt
   ```

4. **Download the image** (~3.2 GB, once). It's pinned to an exact build so everyone runs
   identical software:

   ```bash
   podman pull docker.io/humancompatibleai/adversarial_policies@sha256:1df16d7a129f55d997aa9faba55407f76a8ee1b6305461e5faa56ea599602b8c
   ```

5. **Check that it works** (a few seconds):

   ```bash
   ./run.sh test -m "not slow"
   ```

## Usage

Run every command through `run.sh`. It starts the container, mounts this repo and your MuJoCo
key into it, and runs the command inside. Nothing to mount or configure by hand.

```bash
./run.sh test                 # all tests (the slow ones play real games; ~1 min)
./run.sh test -m "not slow"   # fast tests only
./run.sh lint                 # black + isort + flake8 checks; run before committing
./run.sh fmt                  # auto-format
./run.sh shell                # shell inside the container
./run.sh <any command>        # e.g. ./run.sh python -m noisebackup.sweep --config configs/ants.json
```

## Reproduce our results

```bash
./reproduce_attack.sh    # round 1: the original Zoo1 victims under noise (~25 min)
./reproduce_defense.sh   # round 2: the fine-tuned victims under noise (~45 min)
```

- **`reproduce_attack.sh`** re-runs both attack sweeps (7 noise levels × 1,000 games each, for
  Sumo Ants and Sumo Humans) and draws the figure. Compare
  `results/scratch/reproduce/attack_sweep.png` with the committed `figures/attack_sweep.png`.
- **`reproduce_defense.sh`** re-runs the four defense sweeps (noise-trained and
  clean-fine-tuned victims, both games) from the **committed weights** in `models/`, and redraws
  both round-2 figures. Compare `results/scratch/reproduce/defense_sweep.png` and
  `training_curves.png` with the committed ones in `figures/`. It does not re-train the
  victims (about 5 hours); see [Re-train the defended victims](#re-train-the-defended-victims-optional).

Times are for an 8-core CPU (longer with fewer cores). If a script is interrupted, run it
again and it continues where it stopped. TensorFlow prints a few
`WARNING:tensorflow: ... is deprecated` lines along the way; they're harmless. The exact
numbers are in the `.csv` next to each figure.

What the scripts do:
- **Uses the committed settings.** Each sweep re-runs from the config saved next to the
  committed results (`results/<run>/config.json`), not from `configs/`, so later edits to
  `configs/` can't change what is being reproduced.
- **Never touches the committed results.** Its output goes to `results/scratch/reproduce/`,
  which git ignores.
- **Warns about code changes.** If the experiment code (`noisebackup/`) or the noise scales
  (`results/obs_stats/`), or for the defense the victim weights (`models/`), have changed
  since the commit that produced the committed results, it prints a warning listing the
  changed files. If only plotting code changed (`plot.py`, `training_curves.py`), the games
  are unaffected and the numbers should still match exactly. Otherwise the warning prints
  three commands for an exact reproduction: roll those folders back to the old commit, re-run
  that one sweep into `results/scratch/exact/`, and restore them. Compare the resulting
  `episodes.csv` with the committed one. Commit or stash any uncommitted changes in those
  folders first: the restore step brings back the committed versions.

**Expect exactly the same games on the same machine, and the same results within error on a
different one.** Every source of randomness is seeded: starting positions, the agents' action
sampling, and the noise. So the same machine replays exactly the same games. Another machine
may not:
- Seeds fix the *random numbers*, not the *arithmetic*. Computers round every calculation to
  about 16 digits, and different CPUs (or core counts) can do the same math in a slightly
  different order or with different instructions, so the last digit can come out differently.
- The games are chaotic. A difference in the 16th digit of one action slightly changes a
  contact, which changes the next observation, and within a few hundred steps it's a
  different game.
- The individual games may therefore differ, but they come from the same distribution. The win
  rates should agree within the shaded 95% bands, and the curves should look the same.

## Run your own sweep

A sweep is defined by a JSON config in `configs/`:

```bash
./run.sh python -m noisebackup.sweep --config configs/ants.json
```

This writes `results/<run_name>/`:
- `episodes.csv`: one row per game
- `config.json`: a snapshot of the config
- `run_info.json`: when it ran, at which git commit, and whether there were uncommitted changes

It refuses to write into a folder made with a different config, and after a crash it resumes
by skipping the noise levels already finished.

Config fields: `run_name`, `env_name`, `victim` / `opponent`, `victim_index`, `obs_stats`
(noise scales), `sigmas` (noise levels), `episodes` (games per noise level), `num_envs` (games
played in parallel), `seed` (games and action sampling), `noise_seed`. For a different
experiment, copy a config, give it a new `run_name`, and edit it.

A `victim` or `opponent` is either a zoo agent, `{"type": "zoo", "path": "1"}` (Zoo1), or a
fine-tuned victim from `models/`,
`{"type": "finetuned", "path": "models/ants_defended/weights.pkl"}`. The victim also has a
`label` (`zoo1`, `defended` or `finetuned_clean`), which names its line in the figures.

Quick overrides for trying things out:

```bash
./run.sh python -m noisebackup.sweep --config configs/ants.json --episodes 40 --num-envs 4
```

- `--episodes N`: games per noise level
- `--num-envs N`: games in parallel (use about one per CPU core)
- `--out DIR`: output folder

Using `--episodes` or `--num-envs` changes which games are played, so that output goes to
`results/scratch/<run_name>/` and never mixes with real results.

Then draw a figure from one or more result files:

```bash
./run.sh python -m noisebackup.plot results/ants_zoo1_original/episodes.csv --out results/scratch/ants.png
```

`run_info.json` records `git_dirty: false` when the results came from committed code: no
change to any tracked file, and no new file outside `models/` and `results/` (new files in
those two folders are other runs' outputs, so sweeps can run back to back without a commit in
between).

## Re-train the defended victims (optional)

The committed weights in `models/` are the defended victims we evaluated. To train them again
(about 1 h 10 min per Ants run and 1 h 25 min per Humans run on our desktop):

```bash
./run.sh python -m noisebackup.finetune --config configs/finetune/humans_defended.json
```

The four training configs are in `configs/finetune/` (`{ants,humans}_{defended,finetuned_clean}`).
Each run writes `models/<run_name>/`:
- `weights.pkl`: the fine-tuned policy, in the same format as the zoo's own parameter files
- `training_info.json`: config, git commit, `git_dirty`, timing
- `training_episodes.csv`: one row per training game (when it ended, its σ, winner, length)

It refuses to write into an existing folder, so move or delete the committed one first (or
use `--out DIR`). For a quick test, `--timesteps 100000` trains for fewer steps and writes to
`results/scratch/models/` instead.

Training is seeded on a best-effort basis, so a re-trained victim will be similar but not
identical to the committed one, and its sweep results will differ within error.

## Experiment design (round 1: attack)

| Setting | Value |
|---|---|
| Games | `multicomp/SumoAnts-v0`, `multicomp/SumoHumans-v0` |
| Victim / opponent | Zoo1 (agent 0) vs. Zoo1 (agent 1) |
| Noise | victim only, every input, fresh each step: `σ × stdᵢ × N(0, 1)` |
| `stdᵢ` | per-input std from 100 clean games (`results/obs_stats/<env>.json`) |
| σ grid | 0, 0.05, 0.1, 0.25, 0.5, 1, 2 |
| Games per point | 1,000 (8 parallel games × 125 each) |
| Metric | victim win rate = wins ÷ all games, Wilson 95% intervals; tie rate shown separately |

## Experiment design (round 2: defense)

| Setting | Value |
|---|---|
| Defended victim | Zoo1 (agent 0) fine-tuned from its own weights, against the frozen Zoo1 opponent |
| Noise during training | a new σ ~ Uniform[0, 1] for every training game, same noise formula and `stdᵢ` as round 1 |
| Control victim | fine-tuned identically with σ = 0, to separate "noise helped" from "extra training helped" |
| Reward | win/loss only (+10 win; −10 loss, timeout, or falling before contact; 0 if the opponent falls before contact), with PPO's reward scaling |
| Training | PPO (Stable Baselines PPO2), 10M steps, the paper's fine-tuning settings: 16 parallel games, batch 16,384, lr 3e-4, 4 minibatches, 4 epochs |
| Evaluation | the round-1 sweep, unchanged, on the final weights |

**A bug in `gym_compete`'s LSTM policy, fixed here.** When PPO trains on several games at once,
it hands the network a batch laid out game by game. The zoo LSTM policy reads it that way but
returns its outputs time step by time step, so each action was scored against a different
game's output. PPO's `approxkl` was ~1,000 before any learning (it should be ~0), and training
wrecked the victims. `noisebackup/finetune.py` reorders the outputs while building the training
model, and a test checks that with learning rate 0 the first update has `approxkl` ≈ 0. Playing
one step at a time (sweeps, opponents) is unaffected. The paper's code uses the same policy
class and settings.

## Follow-up experiments

Three checks on the round-2 caveats, run after the main results (Nov 21–22). They were not
part of `PREDICTIONS.md`, so treat them as follow-up, not as tested predictions.

| Experiment | What changes | Runs |
|---|---|---|
| Held-out opponents | the six committed victims (Zoo1, noise-trained, control) vs. Zoo2 and Zoo3 instead of Zoo1 | 12 sweeps |
| Second training seed | training seed 2000 / noise seed 2001 instead of 1000 / 1001, then swept vs. Zoo1 | 4 trainings + 4 sweeps |
| Longer training | 20M PPO steps instead of 10M (from Zoo1, same seeds), then swept vs. Zoo1 | 4 trainings + 4 sweeps |

Everything else matches rounds 1 and 2: same σ grid, 1,000 games per point, same sweep seeds.
The outputs use the committed run name plus a suffix (`_opp2`, `_opp3`, `_seed2`, `_20M`), in
`models/` and `results/`.

What they showed, and what they change in the earlier results, is in the dated addendum
"2026-11-24: Follow-up experiments" in `PREDICTIONS.md`. The full tables, with a Holm
correction over all 182 comparisons, are in `results/followup_summary.md`.
`figures/followup_checks.png` shows the noise-trained − control gap at σ = 0 and σ = 1 in every
condition.

**Re-running.** All 20 sweeps and 8 trainings, in order, then the summary (about 19 hours):

```bash
setsid nohup followup/run_followups.sh < /dev/null > /dev/null 2>&1 &
tail -f results/followup/run.log          # progress
cat results/followup/status.tsv           # one line per finished step: OK, FAILED or SKIPPED
```

- Like the other scripts, it refuses to overwrite: sweeps resume, and trainings with a
  `weights.pkl` are skipped. So to re-run, move the committed follow-up folders aside first.
- If `noisebackup/`, `results/obs_stats/` or `models/` change while it runs, the remaining steps
  are skipped rather than mixing code versions.
- Stay logged in (locking the screen is fine): logging out completely may stop rootless
  podman, unless lingering is on (`loginctl enable-linger`).

Afterwards, redraw the summary and the figure (seconds each):

```bash
./run.sh python followup/summarize.py      # -> results/followup_summary.md
./run.sh python followup/plot_checks.py    # -> figures/followup_checks.png
```

The committed follow-up runs record `git_dirty: true`: the follow-up configs and scripts were
not committed yet while they ran. `noisebackup/` was unchanged at commit 3452ab8, and
`results/followup/run.log` lists exactly which files were uncommitted.

## Layout

```
noisebackup/          package: obs_stats, noise, evaluate, sweep, plot, weights,
                      finetune, training_curves
tests/                pytest; slow tests (real games) marked @pytest.mark.slow
configs/              one JSON config per sweep; configs/finetune/ one per training run;
                      configs/followup/ the follow-up runs
followup/             follow-up runner, summary and figure scripts
models/               fine-tuned victims: weights, training record and log per run, committed
results/              raw results (one row per game) and observation stats, committed;
                      results/followup/ the follow-up step log, followup_summary.md its summary
figures/              generated from results/ and models/
run.sh                runs any command inside the container
reproduce_attack.sh   re-runs the round-1 attack sweeps and draws the figure
reproduce_defense.sh  re-runs the round-2 defense sweeps from models/ and draws the figures
```
