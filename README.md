# RL Adversarial Policy: backup plan

CS446 term project backup: how robust are pretrained "zoo" victims (Bansal et al., 2018) to
**Gaussian noise on their observations**, and does one noise-augmented fine-tuning defense help?

Builds on [HumanCompatibleAI/adversarial-policies](https://github.com/HumanCompatibleAI/adversarial-policies)
(Gleave et al., ICLR 2020), imported from its Docker image rather than copied.

> **Threat model: direct observation.** Noise is added straight to the victim's observations.
> This differs from the paper's *natural-observation* attack, where an adversary can only act
> in the shared environment.

## Setup

Requirements: `podman` (or edit `run.sh` for `docker`) and the free MuJoCo key.

```bash
podman pull docker.io/humancompatibleai/adversarial_policies:latest
mkdir -p ~/.mujoco && curl -o ~/.mujoco/mjkey.txt https://www.roboti.us/file/mjkey.txt
```

## Usage

Everything runs inside the container through `run.sh`:

```bash
./run.sh test                 # all tests
./run.sh test -m "not slow"   # fast tests only (no MuJoCo games)
./run.sh lint                 # black + isort + flake8 checks; run before committing
./run.sh fmt                  # auto-format
./run.sh shell                # shell inside the container
```

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
noisebackup/   package: obs_stats, noise, evaluate, sweep, plot
tests/         pytest; slow tests (real games) marked @pytest.mark.slow
configs/       one JSON config per sweep
results/       raw results (one row per game) and observation stats, committed
figures/       generated from results/
```
