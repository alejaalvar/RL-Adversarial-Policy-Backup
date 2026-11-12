# CLAUDE.md: backup plan for the Adversarial Policies project

## Who you're working with

Two CS446 students (Alejandro and Berny) in their first reinforcement learning class, with solid
Python backgrounds and no prior RL experience. Explain RL-specific decisions briefly, and prefer
simple, readable code over clever code.

## The assignment

1. Understand the problem and the main RL method well enough to explain them clearly.
2. Reproduce a course-sized version of one core result or experiment.
3. Design one meaningful extension, make a prediction **before** running it, and analyze it.
4. Deliver an IEEE-style report, code, and a 10–15 minute presentation (Nov 30 / Dec 2).

Data collection for the backup is planned for the week of Nov 23–29.

## The paper

Gleave et al., "Adversarial Policies: Attacking Deep Reinforcement Learning," ICLR 2020.
Victims are pretrained zoo policies (Bansal et al., 2018). With the victim frozen, the game
becomes a single-agent problem, and the attacker trains with PPO. Adversaries win by creating
*natural* observations that push the victim's network off-distribution.

## This repo: the backup plan

Professor's wording: "Use Gaussian noise and one noise-augmented training defense."

**Threat model: direct observation.** Noise is added straight to the victim's observations. This
is NOT the paper's natural-observation attack. Keep the distinction explicit in code comments,
results, and figures. There is no adversary training here: the noise is the attack, and the
opponent is a normal zoo agent.

### Settled design (grilling session, 2026-11-11)

- **Games:** `multicomp/SumoAnts-v0` (137 obs inputs) and `multicomp/SumoHumans-v0` (395).
  Both are symmetric; victims are LSTM policies.
- **Victim / opponent:** Zoo1 vs. Zoo1. The victim is agent 0 (true for all Sumo games;
  only YouShallNotPass has the victim at index 1).
- **Clean baselines are not 50/50** (measured 2026-11-11, no noise): Sumo Humans agent 0 wins
  ~80–84% (confirmed with the original `aprl.score_agent`: 168/200), so it's a property of the
  game/agents, not our code. Sumo Ants ≈ 42% victim / 36% opponent / 22% ties. Decision: keep the
  victim as agent 0 anyway (paper convention) and report the baseline as measured.
- **Noise:** victim only, every input, fresh every step, independent per input.
- **Relative σ:** noise on input *i* = σ × stdᵢ × N(0,1). stdᵢ comes from the victim's
  observations in 100 clean games, saved in `results/obs_stats/<env>.json`. Inputs with zero
  spread get no noise. The defended victim reuses the same stats file, so σ means the same thing
  for both victims. Raw input spreads range over ~4–5 orders of magnitude, which is why σ is
  relative. **Frozen stats (measured 2026-11-11, 100 games, seed 100, commit a86585c):**
  Ants 137 inputs, std 0.0038–127 (median 0.29), none constant; Humans 395 inputs,
  std 0–109 (median 0.30), 6 constant (these get no noise). Never re-measure without deciding
  to redo every sweep that used them; the CLI refuses to overwrite.
- **σ grid:** {0, 0.05, 0.1, 0.25, 0.5, 1, 2}. Fixed before any results; no pilot sweep.
  Any added points must be labeled as follow-up.
- **Games per point:** 1,000. Each of 8 parallel games plays a fixed share (125) so long games
  aren't dropped at the cutoff. Game seeds are the same at every σ; noise has its own seed.
  Zoo agents *sample* their actions (as in the paper); that sampling is seeded too
  (`seed` + agent index, via `evaluate.load_policies(seed=...)`), so a config replays the
  exact same games. Found 2026-11-11: before this, two identical Ants sweeps differed game by
  game (but agreed within sampling error); those results were discarded and re-run.
- **Metric:** victim win rate = wins ÷ all games (ties are non-wins), Wilson 95% intervals,
  plus a separate tie-rate panel.
- **Results:** `results/<run_name>/episodes.csv`, one row per game, plus the config and a
  settings record (config, seeds, git commit, timestamp). Committed. Summaries and figures are
  always regenerated from it. Sweeps resume by skipping finished σ points and refuse to reuse
  a results folder written with a different config.

### Order of work

Round 1 (attack): scaffold → obs_stats → noise → evaluate → sweep → plot → measure real obs
stats → predictions → full sweeps (Ants, then Humans) → figure.

Round 2 (defense), after its own design session: fine-tune a copy of the Zoo1 victim
(not from scratch) with noise on its observations, re-run the same sweep including σ = 0,
plot original vs. defended.

## Predictions

Must be written before running the full experiments, in `PREDICTIONS.md`, dated, and never
edited after results come in. Claude drafts a fill-in template with no numbers; the students
fill it in. If it isn't filled in before a full sweep, stop and remind them.

## How to work in this repo

- **Everything runs in Docker** via `./run.sh` (image
  `docker.io/humancompatibleai/adversarial_policies:latest`, Python 3.7). Use the image's
  *installed* `aprl`; do not `pip install -e` a clone of the original repo, which differs
  from the image's copy in 28 files. Read the image's copy before relying on an interface.
- **No Sacred.** Our scripts use JSON configs plus a few CLI flags.
- **Reuse from aprl** (verified): `aprl.envs.wrappers.make_env`, `aprl.envs.wrappers.simulate`,
  `aprl.envs.multi_agent.make_dummy_vec_multi_env` / `make_subproc_vec_multi_env` /
  `VecMultiWrapper`, `aprl.policies.loader.load_policy`, `aprl.envs.gym_compete.game_outcome`.
- Note: `aprl.score_agent`'s `noisy_agent_index` adds noise to **actions**, not observations,
  so it is not used here.
- **TDD:** red → green, one slice at a time, tests only at the agreed seams:
  `noise.ObservationNoise`, `obs_stats.compute` + save/load, `evaluate.run_episodes`,
  `sweep.run_sweep`, `plot.summarize`. Real-game tests are marked `@pytest.mark.slow`.
- **Style:** black + isort + flake8 at 100 columns; `./run.sh lint` must be clean.
- **Commits:** Claude never commits. Stop after each slice with a summary and test output;
  the students review and commit. No PRs and no CI; tests run locally before each commit.
- Compute is a CPU-only desktop. Don't set up GPU support (TF 1.13 can't use the RTX 4070).
- Test with tiny episode counts before launching long runs.
