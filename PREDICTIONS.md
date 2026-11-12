# Predictions

**Rules (course requirement):**

- Fill this in **before** running the full sweeps, then commit it. The commit timestamp is the proof.
- **Never edit a prediction after results exist.** If you change your mind before running,
  edit freely and re-commit. After results exist, add a dated note under _Addenda_ instead.
- Wrong predictions are fine and expected; the report discusses _why_ they were wrong.

|                                          |                        |
| ---------------------------------------- | ---------------------- |
| Written by                               | **Alejandro Alvarado** |
| Date                                     | **11/11/2026**         |
| Commit of the code being predicted about | **706cc8a**            |

---

## Part 1: Attack (round 1)

### What is being tested

- **Victim:** pretrained zoo agent Zoo1 (Bansal et al., 2018), an LSTM policy, frozen.
  **Opponent:** the same Zoo1 policy, with clean observations.
- **Attack:** every timestep, Gaussian noise is added to _every_ input of the victim's
  observation: `noise_i = σ × std_i × N(0, 1)`, where `std_i` is how much input _i_ normally
  varies in clean play. So σ = 0.5 means "noise half the size of each input's normal
  variation". **Direct-observation threat model**: this is not the paper's attack, where an
  adversary can only act in the world.
- **Games:** Sumo Ants (137 inputs) and Sumo Humans (395 inputs). 1,000 games per σ.
- **σ values:** 0, 0.05, 0.1, 0.25, 0.5, 1, 2.
- **Win rate** = victim wins ÷ all games (ties count as non-wins).

### Already known before predicting (σ = 0, seen during development)

| Game        | Victim wins | Opponent wins | Ties  | Source                                       |
| ----------- | ----------- | ------------- | ----- | -------------------------------------------- |
| Sumo Ants   | ~42%        | ~36%          | ~22%  | 64 test games                                |
| Sumo Humans | ~80–84%     | ~16–19%       | ~1–2% | 64 test games + 200 with the original scorer |

These are small samples; the sweep's σ = 0 point will measure them properly.

### P1. Victim win rate at each σ

Your best single guess for each cell, in %.

| σ           | 0    | 0.05 | 0.1  | 0.25 | 0.5  | 1    | 2    |
| ----------- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| Sumo Ants   | ~42% | ~42% | ~41% | ~39% | ~34% | ~25% | ~15% |
| Sumo Humans | ~82% | ~82% | ~81% | ~77% | ~65% | ~35% | ~10% |

### P2. Where does it start to hurt?

The smallest σ at which the victim's win rate is **at least 10 percentage points** below its
own σ = 0 value (or "none" if it never drops that much by σ = 2):

- Sumo Ants: **When σ=1, assuming it is only about 8 points down by σ=0.5.**
- Sumo Humans: **When σ=0.5.**

### P3. Which victim is more fragile, and why?

Which game's victim loses more of its win rate as σ grows? (Ants / Humans / about the same)

Answer: **The humanoid will be more fragile.**

Why (1–3 sentences): **This is because it has more inputs to corrupt, and a body that must constantly balance**

>

### P4. What happens to ties?

As σ grows, does the **tie rate** go up, go down, or stay about the same? Answer for each game.

- Sumo Ants: **The ties will go up**, because **a confused ant moves erratically but stays upright, so fewer games end decisively and more run to the time limit.**
- Sumo Humans: **The ties will go down**, because **a confused humanoid will tend to fall down, which ends the game as a loss rather than a tie.**

### P5. The reasoning behind your curve

In 2–4 sentences: what do you think happens inside the victim as noise grows, and why does
that produce the curve you predicted in P1?

**At small σ, the noise sits inside the range of variation the network saw in training, and normalized inputs barely move, so behavior is essentially unchanged. As σ approaches 1, the victim's normalized inputs drift well outside anything it trained on, especially its own joint and contact readings, so it issues motor commands based on a wrong picture of its own body. For a humanoid that quickly means falling; for an ant it means aimless, ineffective movement. That gives a flat start, then a steep drop, with humanoids dropping harder.**

>

### P6. Confidence (optional)

How confident are you in P1 overall? (low / medium / high) **I am fairly confident (medium) about the overall shape. I am less sure about where the drop starts, especially because LSTMs can partly smooth out noise using their memory, and the networks own normalization or clipping could make it more robust than I expect, which would push the drop to a larger σ value.**

---

## Part 2: Defense (round 2)

**Rules:** same as Part 1. Fill this in and commit it **before any training starts** (including
the short timing run), so no training output can influence it.

|                                          |        |
| ---------------------------------------- | ------ |
| Written by                               | Alejandro Alvarado |
| Date                                     | 11/12/2026 |
| Commit of the code being predicted about | 7a6a6a7 |

### What is being tested

- **Defended victim:** a copy of Zoo1 fine-tuned for 10M steps (PPO, the paper's own victim
  fine-tuning settings) against the same Zoo1 opponent, with observation noise during training:
  at the start of every training game a new σ is drawn uniformly from **[0, 1]**.
- **Control victim:** fine-tuned identically, but with **no noise** (σ = 0). It separates "the
  noise helped" from "any extra training against this opponent helped".
- **Reward during training:** win/loss only (a timeout or falling over before contact counts
  against the victim).
- **Evaluation:** exactly the round-1 sweep (same σ grid, same 1,000 games per σ, same seeds).
  Note that σ = 2 is **outside** the training range.

### Already known before predicting (round 1, original Zoo1 victim)

| σ | 0 | 0.05 | 0.1 | 0.25 | 0.5 | 1 | 2 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Sumo Ants: Zoo1 win rate | 46.2% | 42.1% | 44.5% | 42.1% | 33.8% | 15.2% | 8.1% |
| Sumo Humans: Zoo1 win rate | 82.3% | 81.2% | 80.2% | 70.7% | 46.1% | 9.9% | 0.5% |

(±~3 points at 95%; see `figures/attack_sweep.png`.)

### P7. Defended victim's win rate at each σ

Your best single guess for each cell, in %.

| σ | 0 | 0.05 | 0.1 | 0.25 | 0.5 | 1 | 2 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Sumo Ants (defended) | 45%| 45%|| 45%| 44%| 41%| 30%|
| Sumo Humans (defended) | 80%| 80%| 79%| 76%| 68%| 35%| 4%|

### P8. Control victim's win rate at each σ

| σ | 0 | 0.05 | 0.1 | 0.25 | 0.5 | 1 | 2 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Sumo Ants (control) | 50%| 48%| 48%| 45%| 36%| 16%| 8%|
| Sumo Humans (control) | 85%| 84%| 83%| 73%| 47%| 10%| 1%|

### P9. Does the defense cost clean performance?

At σ = 0, is the defended victim's win rate higher than, about the same as (within ~3 points),
or lower than the original Zoo1's? Answer for each game.

- Sumo Ants: about the same. Any cost from noisy training roughly cancels the gain from extra practice.
- Sumo Humans: slightly lower (a few points). A more cautious, balance-first style should cost a little against a clean opponent.

### P10. Was it the noise, or just the extra training?

At σ = 0.5 and σ = 1, does the defended victim beat the control by a clear margin (more than
the ~3-point error)? Answer for each game.

- Sumo Ants: yes at σ = 1, barely at σ = 0.5.
- Sumo Humans: yes to both

### P11. Which game benefits more from the defense?

Compare the improvement over Zoo1 at σ = 0.5 and σ = 1. (Ants / Humans / about the same)

Answer: humans

Why (1–3 sentences):

> Its round-1 failure mode was concrete and fixable: the victim fell over before contact. Training directly punishes that, so the victim can learn steadier behavior. The ants’ failure is a more diffuse loss of effectiveness, which is harder to train away.

### P12. Beyond the training range

At σ = 2 (never seen in training), does the defended victim still clearly beat Zoo1?

- Sumo Ants: yes, modestly (about 14% vs. 8.1%).
- Sumo Humans: no, not clearly. At σ = 2 the body-state information is so corrupted that a humanoid probably can’t stay up regardless; any gain would likely fall inside the error bars.

### P13. Ties

For Sumo Humans, round 1's ties came from the noisy victim falling before contact. Does the
defended victim's tie rate at σ = 1 and σ = 2 go down compared with Zoo1's (31.5% and 81.1%)?
What happens to the Sumo Ants tie rate?

- Sumo Ants: slightly down, because because training counts timeouts against the victim, so it’s pushed to finish games rather than stall.
- Sumo Humans: clearly down at σ = 1 (from 31.5% to roughly 15%), because falling before contact is exactly what the training reward penalizes. Still high at σ = 2, because that’s outside the training range and falls are likely unavoidable.

### P14. The reasoning behind your predictions

In 2–4 sentences: what do you think the victim learns during noisy fine-tuning, and why does
that produce the numbers you predicted in P7–P13?

> Since the noise is fresh and independent each step, the LSTM can learn to rely on its memory, effectively averaging recent observations to filter out noise, instead of reacting to each noisy snapshot. The victim should also learn more conservative, stable movements that tolerate a wrong sense of its own body. Both help most where noise is large but still within training (σ = 0.5 to 1), cost a little at σ = 0, and generalize only partially to σ = 2.

### P15. Confidence (optional)

How confident are you in P7–P8 overall? (low / medium / high) low

---

## Addenda

Dated notes added **after** results exist. Never edit the predictions above.

-
