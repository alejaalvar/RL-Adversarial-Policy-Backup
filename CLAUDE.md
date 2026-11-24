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

Data collection for the backup was planned for the week of Nov 23–29 and finished early, on
Nov 16 (round 1: Nov 11; round 2: Nov 16).

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

Round 1 is done (2026-11-11): `results/{ants,humans}_zoo1_original/`, `figures/attack_sweep.png`.

### Round 2: defense (grilling session, 2026-11-12)

- **What:** fine-tune a copy of Zoo1 (agent 0, starting from its weights) in both games against
  the frozen Zoo1 opponent, which **samples** its actions as in the sweep (aprl.train makes it
  deterministic; we don't use aprl.train). Own script `noisebackup/finetune.py`, no Sacred.
- **Variants (4 runs):** `defended`: σ ~ Uniform[0, 1] drawn per game, per parallel game, with the
  same frozen obs stats; `finetuned_clean` (control): σ = 0, otherwise identical.
- **Reward:** win/loss only (paper's sparse reward 0.01 × ±1000), reward normalization
  (VecNormalize norm_obs=False; the zoo policy normalizes observations internally, frozen).
- **PPO2:** paper preset `finetune_defense` but 10M steps: 16 envs, batch 16,384, lr 3e-4,
  ent_coef 0, nminibatches 4, noptepochs 4, clip 0.2, γ 0.99, λ 0.95.
- **Seeds:** training games 1000, training noise 1001 (sweep: 0 / 1). Best-effort training
  seeding; training need not be bit-identical; **committed weights are canonical**.
- **Weights:** zoo format (flat float32 vector like `agent_parameters-v1.pkl`), in
  `models/<run_name>/weights.pkl` + `training_info.json` + `training_episodes.csv` (per-game
  step, σ, winner, length). Evaluated through aprl's zoo loader path (victim type `"finetuned"`),
  so the round-1 seeding hook and any num_envs work. Only the final 10M-step weights are
  evaluated; checkpoints go to `results/scratch/` and are never evaluated.
- **Evaluation:** the round-1 sweep unchanged; labels `defended` / `finetuned_clean`. Figures:
  `figures/defense_sweep.png` (Zoo1 + defended + control) and `figures/training_curves.png`.
  `reproduce_defense.sh` re-runs the sweeps from committed weights (re-training is optional).
- **TDD seams (added):** ObservationNoise per-game σ range; `weights` save/load round trip;
  finetune tiny real run (lr = 0 → weights equal Zoo1 exactly); `load_policies` with a
  `"finetuned"` victim (Zoo1 saved as finetuned reproduces Zoo1's games); learning-curve summary.
- **Order:** predictions Part 2 → noise σ range → weights → evaluate finetuned → finetune
  script + configs → timing run → 4 trainings → 4 sweeps + figures → reproduce script + docs.

Round 2 is done (2026-11-16): `models/<run>/` trained at 3d84a9e, sweeps in
`results/{ants,humans}_{defended,finetuned_clean}/` at 36c52dc (all `git_dirty: false`),
`figures/defense_sweep.png`, `figures/training_curves.png`, `./reproduce_defense.sh`.

**Found during round 2 (2026-11-12/13):**
- **gym_compete LSTM output-order bug.** `gym_compete.policy.LSTMPolicy` reads PPO's batch game
  by game but returns outputs time-step-major (`outputs_ta.stack()` → `seq_to_batch`, no
  transpose), so with more than one game per PPO minibatch every action was scored against
  another game's output: approxkl ~1,000 and clipfrac 0.998 from update 1, and the first 10M-step
  trainings collapsed (Humans victim 80% → 0% wins). Fixed by `finetune._game_major_lstm_outputs`
  (patches `seq_to_batch` while PPO2 is built); guarded by the lr = 0 test that the first update
  has approxkl ≈ 0 with 2 games per minibatch. Acting (n_steps = 1) is unaffected, so round 1
  and all opponents are fine. The broken models are kept, untracked, in
  `results/scratch/models_lstm_order_bug/`. When fine-tuning any zoo LSTM, always watch
  approxkl/clipfrac on the first updates.
- **`git_dirty` rule.** Dirty = any change to a tracked file, or any new untracked file outside
  `models/` and `results/` (`records.OUTPUT_FOLDERS`); a run's own output folder is ignored
  entirely. Before this, back-to-back runs saw each other's new output folders and recorded
  dirty with unchanged code.

### Follow-ups (2026-11-21/22)

Not in PREDICTIONS.md; always label them "follow-up". Configs in `configs/followup/`, scripts
in `followup/` (`run_followups.sh` runner, `summarize.py` → `results/followup_summary.md`,
`plot_checks.py` → `figures/followup_checks.png`). Committed in 077868b (figure) and c79d075.
- **What:** (1) the six committed victims vs. held-out opponents Zoo2 and Zoo3 (both exist for
  both games; `_opp2` / `_opp3`); (2) second training seed 2000 / 2001 (`_seed2`); (3) 20M steps
  from Zoo1 with the committed training configs (`--timesteps 20000000`, `_20M`). New victims
  swept vs. Zoo1 with the round-1 sweep unchanged. Run at 3452ab8; all runs record
  `git_dirty: true` only because the follow-up configs and scripts were uncommitted (listed in
  `results/followup/run.log`).
- **Statistics:** each new sweep vs. its committed run (pooled two-proportion z test, as in the
  addendum); Holm over 182 comparisons = the addendum's 42 (noise-trained vs. control,
  noise-trained vs. Zoo1, control vs. Zoo1 × 7 σ × 2 games) + 140 new. The noise-trained −
  control gap *within* each new condition is not in that family (report it as uncorrected).
- **Findings:**
  - The Humans σ = 1 gain over the control replicates vs. Zoo1 (seed 2: +5.8; 20M: +10.8) and
    survives Holm (+10.5, Holm p 3e-9).
  - It mostly vanishes vs. held-out opponents (Zoo2 −0.4, Zoo3 +4.1). Zoo2 and Zoo3 beat
    every Humans victim: clean win rate ~82% → 13–22%. Cross-checked with
    `aprl.score_agent` (Zoo1 vs Zoo2: 29/200 wins), so the 82% baseline is specific to the
    Zoo1-vs-Zoo1 matchup.
  - Ants: no consistent benefit anywhere; the fine-tuning drop replicates with seed 2.
  - After Holm, the addendum's Humans clean cost (−4.7) and σ = 0.5 gain (+5.3) are not
    significant; don't present them as findings.
- **Lessons:**
  - `aprl.score_agent` (Sacred) writes `data/sacred/` in the repo. Delete it after a cross-check:
    it is outside `models/` and `results/`, so it makes later runs dirty.
  - `followup/` is not in `run.sh lint`'s paths (only `noisebackup` and `tests`); lint it by
    hand. Its scripts have no tests; `summarize.py` was checked by reproducing the addendum's
    p-values exactly.

## Predictions

Must be written before running the full experiments, in `PREDICTIONS.md`, dated, and never
edited after results come in. Claude drafts a fill-in template with no numbers; the students
fill it in. If it isn't filled in before a full sweep, stop and remind them.

## How to work in this repo

- **Everything runs in the container** via `./run.sh` (podman required, not Docker: the image
  can't run as non-root and Docker would leave root-owned files; image
  `docker.io/humancompatibleai/adversarial_policies`, pinned by digest in `run.sh`, Python 3.7).
  `./reproduce_attack.sh` and `./reproduce_defense.sh` re-run the round-1 and round-2 sweeps
  from their saved configs (round 2 from the committed weights, no re-training). Use the image's
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
  `sweep.run_sweep`, `plot.summarize`, `records.git_info`, `weights`, `evaluate.load_policies`
  (finetuned), `finetune` (tiny real runs, lr = 0, approxkl), `training_curves.summarize` /
  `load_run`. Real-game tests are marked `@pytest.mark.slow`.
- **Style:** black + isort + flake8 at 100 columns; `./run.sh lint` must be clean.
- **Commits:** Claude never commits. Stop after each slice with a summary and test output;
  the students review and commit. No PRs and no CI; tests run locally before each commit.
- Compute is a CPU-only desktop. Don't set up GPU support (TF 1.13 can't use the RTX 4070).
- Test with tiny episode counts before launching long runs.
