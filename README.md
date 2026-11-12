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
./reproduce_attack.sh
```

This re-runs both attack sweeps (7 noise levels × 1,000 games each, for Sumo Ants and Sumo
Humans) and draws the figure. It takes about 25 minutes on an 8-core CPU (longer with fewer
cores). If it's interrupted, run it again and it continues where it stopped. TensorFlow prints
a few `WARNING:tensorflow: ... is deprecated` lines along the way; they're harmless.

Then compare `results/scratch/reproduce/attack_sweep.png` with the committed
`figures/attack_sweep.png`. The exact numbers are in the `.csv` next to each figure.

What it does:
- **Uses the committed settings.** Each sweep re-runs from the config saved next to the
  committed results (`results/<run>/config.json`), not from `configs/`, so later edits to
  `configs/` can't change what is being reproduced.
- **Never touches the committed results.** Its output goes to `results/scratch/reproduce/`,
  which git ignores.
- **Warns about code changes.** If the experiment code (`noisebackup/`) or the noise scales
  (`results/obs_stats/`) have changed since the commit that produced the committed results,
  it prints a warning and the `git checkout` command for an exact reproduction.

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

Config fields: `run_name`, `env_name`, `victim` / `opponent` (zoo agent versions),
`victim_index`, `obs_stats` (noise scales), `sigmas` (noise levels), `episodes` (games per
noise level), `num_envs` (games played in parallel), `seed` (games and action sampling),
`noise_seed`. For a different experiment, copy a config, give it a new `run_name`, and edit it.

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

Tip: commit finished results before starting the next sweep. `run_info.json` then records
`git_dirty: false`, meaning the results came from committed code.

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

## Layout

```
noisebackup/          package: obs_stats, noise, evaluate, sweep, plot
tests/                pytest; slow tests (real games) marked @pytest.mark.slow
configs/              one JSON config per sweep
results/              raw results (one row per game) and observation stats, committed
figures/              generated from results/
run.sh                runs any command inside the container
reproduce_attack.sh   re-runs the round-1 attack sweeps and draws the figure
```
