# Predictions

**Rules (course requirement):**

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

Our best single guess for each cell, in %.

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

### P5. The reasoning behind our curve

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

Our best single guess for each cell, in %.

| σ | 0 | 0.05 | 0.1 | 0.25 | 0.5 | 1 | 2 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Sumo Ants (defended) | 45%| 45%|45%| 44%| 41%| 30%| 14%|
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

### P14. The reasoning behind our predictions

In 2–4 sentences: what do you think the victim learns during noisy fine-tuning, and why does
that produce the numbers you predicted in P7–P13?

> Since the noise is fresh and independent each step, the LSTM can learn to rely on its memory, effectively averaging recent observations to filter out noise, instead of reacting to each noisy snapshot. The victim should also learn more conservative, stable movements that tolerate a wrong sense of its own body. Both help most where noise is large but still within training (σ = 0.5 to 1), cost a little at σ = 0, and generalize only partially to σ = 2.

### P15. Confidence (optional)

How confident are you in P7–P8 overall? (low / medium / high) _low_

---

## Addenda

Dated notes added **after** results exist. Never edit the predictions above.

### 2026-11-16: Results vs. predictions (Parts 1 and 2)

**How to read this.**
- Every number comes from the committed sweeps: `results/<run>/episodes.csv`, summarized in
  `figures/attack_sweep.csv` and `figures/defense_sweep.csv`. That is 1,000 games per point, so
  a single win rate is ±~3 points (95%).
- "Significant" means a two-proportion z-test between two victims at the same σ, p < 0.05. A
  difference needs about 4–5 points to reach that.
- **Evidence** marks something the data shows directly. **Hypothesis** marks an explanation that
  fits the data but that we have not tested.
- Verdicts: ✅ right · ◐ partly right · ❌ wrong.
- **Terms.** *Fine-tuning* means continuing to train an already-trained policy from its
  weights; by itself it says nothing about noise. Both round-2 victims are fine-tuned: Zoo1
  plus 10M PPO steps against the frozen Zoo1 opponent, with a win/loss-only reward. They
  differ in one thing. The **noise-trained** victim (`defended` in the code and figures) got
  a random σ ~ U[0, 1] on its observations each training game. The **clean-trained control**
  (`finetuned_clean`) got σ = 0. So noise-trained − Zoo1 mixes the effects of extra training
  and noise, control − Zoo1 is extra training alone, and noise-trained − control is noise
  alone.
- **σ and win rate** are as defined in Part 1: σ is the noise size relative to each input's
  normal spread in clean play, and win rate = victim wins ÷ all games (ties count as non-wins).
- **Points, not percent.** Differences between win rates are in percentage points: 86.2% →
  81.5% is "−4.7 points", not "−4.7%".
- **Ties come in two kinds**, and they behave differently:
  - *Timeout:* 500 steps pass with no winner.
  - *Early fall:* one agent falls or leaves the ring before the two have touched. The game's
    scorer gives the other agent no win, so it counts as a tie, not a loss. In training,
    though, the victim's reward for falling early is −10, the same as a loss (and 0 if the
    opponent falls early).

#### 1. Headline findings

In case you cannot be bothered to read the entire addenda, here is the TL;DR containing our high-level findings. Limitations not discussed.

1. **Noise breaks both victims, and Humans far more.** Win rate at σ = 2: Ants 46% → 8%,
   Humans 82% → 0.5%. Humans collapse because the victim falls over before touching the
   opponent, which the game scores as a tie.
2. **The defense helped only Humans, and mainly at σ = 1, the edge of the training range.** At
   σ = 1 the noise-trained Humans victim won 19.6% vs 9.1% for the clean-trained control
   (p ≈ 10⁻¹¹). Its ties (falls or ring-outs before contact) dropped from 41% (control) to 14%.
3. **Part of the apparent defense at moderate noise is just extra training.** At σ = 0.25–0.5
   the clean-trained control also beat Zoo1 by 8–9 points. Only the remainder (+5 at σ = 0.5,
   +10 at σ = 1) is due to the noise.
4. **Noise training costs some clean performance in Humans, but only relative to the control.**
   At σ = 0 it matches Zoo1 (81.5% vs 82.3%) but trails the control (86.2%; −4.7 points,
   p = 0.004).
5. **In Ants, fine-tuning made both victims worse, and the defense did nothing.** Both fine-tuned
   Ants victims lost 8–10 clean points against Zoo1 (p < 0.001), and noise vs. control never
   differed significantly at any σ.
6. **Nothing helps at σ = 2, outside the training range.** All victims are within 2 points of
   each other.

#### 2. Part 1: attack (original Zoo1 victim)

**P1. Win rate at each σ** ◐

| σ | 0 | 0.05 | 0.1 | 0.25 | 0.5 | 1 | 2 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Ants predicted | 42 | 42 | 41 | 39 | 34 | 25 | 15 |
| Ants measured | 46.2 | 42.1 | 44.5 | 42.1 | **33.8** | 15.2 | 8.1 |
| Humans predicted | 82 | 82 | 81 | 77 | 65 | 35 | 10 |
| Humans measured | 82.3 | 81.2 | 80.2 | 70.7 | 46.1 | 9.9 | 0.5 |

- **The shape was right:** flat at small σ, then a steep drop.
- **Ants was close up to σ = 0.5** (34 predicted, 33.8 measured). The tail was overestimated:
  25 → 15.2 at σ = 1, and 15 → 8.1 at σ = 2.
- **Humans was exact up to σ = 0.1.** From σ = 0.25 it was overestimated, and badly at σ = 0.5–1
  (65 → 46, 35 → 10). The collapse came about one grid step earlier and went deeper than
  predicted.

**P2. Where it starts to hurt (first point ≥ 10 points below σ = 0)** ❌ (both one step late)
- Ants: predicted σ = 1; measured **σ = 0.5** (−12.4 points).
- Humans: predicted σ = 0.5; measured **σ = 0.25** (−11.6 points).
- **Why:** both networks are more sensitive than "within normal variation" reasoning
  suggests (Hypothesis). σ = 0.25 noise is a quarter of each input's clean spread, but it hits
  all inputs at once (137 or 395 of them), independently, every step. The combination is a
  pattern the policy never saw, even when each input on its own looks plausible.

**P3. Which is more fragile** ✅ Humans
- **Evidence:** Humans lost 81.8 points (82.3 → 0.5) vs Ants 38.1 (46.2 → 8.1). In relative
  terms, Humans kept 0.6% of their clean win rate vs Ants 18%.
- **Why:** the "body that must constantly balance" half of our reasoning is supported (see P4).
  The "more inputs" half is not tested. Noise is scaled per input, and we have no experiment
  that separates input count from body type.

**P4. Ties** ❌ both, but for an instructive reason
- **Ants, predicted up; measured flat** (15–20% at every σ).
  - Evidence: Ants ties are mostly timeouts, 70–88% of ties reach the 500-step limit, and that
    share doesn't grow with noise. What grows is **opponent wins: 37% → 77%**.
  - So a confused ant doesn't stall. It keeps engaging and loses.
- **Humans, predicted down; measured sharply up** (1.7% → 31.5% at σ = 1 → 81.1% at σ = 2).
  - Our mechanism was right: the noisy humanoid falls.
  - _But_ the game's scoring rule turns it into a tie. In `gym_compete` Sumo, a fall or ring-out
    _before the agents have touched_ is a tie, not a loss.
  - Evidence: Humans ties are short. At σ = 2, 91% of ties end within 100 steps (mean 72 steps),
    and no Humans tie at σ ≥ 1 is a timeout.

**P5. Reasoning** ◐
- The "flat start, then steep drop, humanoids dropping harder" picture matches.
- "For a humanoid that quickly means falling" is supported by the short ties above.
- "For an ant it means aimless, ineffective movement" is half right. Ants don't wander into
  timeouts; they lose contests (rising opponent wins).

**P6. Confidence (medium)** ✅ reasonable
- We flagged uncertainty about *where* the drop starts, and that is exactly where the
  predictions missed (P2).
- Whether the LSTM's memory partly averages out the noise remains untested.

#### 3. Part 2: defense (noise-trained vs. clean-trained control)

**P7. Noise-trained win rate** ◐ (Humans good at low σ; both overestimated at high σ)

| σ | 0 | 0.05 | 0.1 | 0.25 | 0.5 | 1 | 2 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Ants predicted | 45 | 45 | 45 | 44 | 41 | 30 | 14 |
| Ants measured | 37.8 | 39.0 | 34.4 | 35.0 | 30.2 | 15.9 | 6.4 |
| Humans predicted | 80 | 80 | 79 | 76 | 68 | 35 | 4 |
| Humans measured | 81.5 | 79.5 | 81.5 | 76.9 | 59.9 | 19.6 | 0.6 |

**P8. Control win rate** ◐ (Humans very close; Ants clean performance far overestimated)

| σ | 0 | 0.05 | 0.1 | 0.25 | 0.5 | 1 | 2 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Ants predicted | 50 | 48 | 48 | 45 | 36 | 16 | 8 |
| Ants measured | 36.4 | 36.3 | 35.9 | 34.9 | 28.6 | **17.0** | **7.5** |
| Humans predicted | 85 | 84 | 83 | 73 | 47 | 10 | 1 |
| Humans measured | 86.2 | 84.1 | 84.4 | 78.6 | 54.6 | **9.1** | 0.4 |

- The prediction that extra clean training slightly *helps* was right for Humans: +3.9 points
  at σ = 0 (p = 0.017), and +8.5 at σ = 0.5.
- For Ants it went the other way (see P9).
- The high-σ control values were predicted almost exactly in both games.

**P9. Clean-performance cost (σ = 0, vs. Zoo1)**
- **Ants, predicted about the same:** ❌ Measured 37.8% vs 46.2%, a −8.4 point drop
  (p < 0.001).
  - The control dropped just as much (−9.8), so this is a cost of **fine-tuning itself**, not
    of the noise.
  - The training curves show it was gradual (`figures/training_curves.png`: control
    45% → 39% over 10M steps), not a sudden break.
  - **Hypothesis:** the win/loss-only reward gives too weak a learning signal in Ants.
    - Ants games are long: wins take ~220 steps, and ties run to 500.
    - With PPO's discount (γ = 0.99, an effective horizon of ~100 steps), a reward at the end
      of a 450-step game is worth ~1% of its value at the start.
    - So most updates are driven by noise in the advantage estimates, and a well-tuned policy
      drifts downhill.
    - Humans games are shorter (~150 steps), so the signal is stronger. That fits the Humans
      control improving instead.
  - Untested. It would be checked by: more training seeds, γ = 0.995, or the dense-then-sparse
    reward Bansal et al. trained with.
- **Humans, predicted slightly lower:** ◐ Against Zoo1, about the same (81.5 vs 82.3,
  not significant). Against the control, **−4.7 points (p = 0.004)**.
  - So our intuition, "a more cautious, balance-first style should cost a little", holds
    when the comparison is fair. Extra training gained ~4 points, and the noise gave them back.
  - Evidence for the cautious style: the noise-trained victim's games are longer.
    - Its wins take 173 steps at σ = 0 vs 150 for the control and 152 for Zoo1.
    - At σ = 0.5 the gap widens, 193 vs 155.

**P10. Noise vs. control at σ = 0.5 and 1**
- **Ants, predicted yes at σ = 1:** ❌ No difference at any σ. At σ = 0.5 it is +1.6 (p = 0.43);
  at σ = 1 it is −1.1 (p = 0.51).
- **Humans, predicted yes to both:** ✅ σ = 1: +10.5 points (p ≈ 10⁻¹¹), robust. σ = 0.5: +5.3
  points (p = 0.017).
  - That counts as significant, but only modestly. Across 42 comparisons, one p ≈ 0.02 could
    be chance, so the σ = 0.5 result is weaker evidence than σ = 1.

**P11. Which game benefits more** ✅ Humans
- Gain over Zoo1 at σ = 0.5 / 1: Humans +13.8 / +9.7; Ants −3.6 / +0.7.
- Our reason, a concrete failure (falling) that the reward directly punishes, fits the
  evidence.
  - In training, the victim falling before contact scores −10, like a loss.
  - The improvement at σ = 1 is mostly fewer falls: ties dropped 31.5% → 14.2% vs Zoo1, and
    40.6% → 14.2% vs the control.
- **Why Ants didn't benefit** (Hypothesis): the simplest explanation is the same weak signal as
  in P9. If fine-tuning couldn't even hold on to clean performance, it had little chance of
  learning robustness. Our alternative, a more diffuse failure that's harder to train away,
  also fits and isn't ruled out.

**P12. Beyond the training range (σ = 2)**
- **Ants, predicted yes, modestly:** ❌ 6.4% vs 8.1% (n.s.).
- **Humans, predicted no:** ✅ 0.6% vs 0.5%.
- **Why** (Hypothesis): at σ = 2 the noise is twice each input's clean spread, beyond anything
  seen in training (max σ = 1). For Humans, 74–84% of games still end in an early fall,
  whatever the training.

**P13. Ties**
- **Ants, predicted slightly down:** ❌ Slightly up, 2–5 points at most σ (e.g. 19.3% vs 16.8%
  at σ = 0).
  - Ties remain mostly timeouts (~90% at σ ≤ 0.5), so "push it to finish games" did not happen.
    That is consistent with the weak-signal hypothesis in P9.
- **Humans, predicted ~15% at σ = 1, still high at σ = 2:** ✅ Measured **14.2%** at σ = 1
  (Zoo1 31.5%) and 73.6% at σ = 2. This was the most precise prediction in Part 2.
  - A nuance: fewer falls did not all become wins. At σ = 1, opponent wins *rose*: 66.2% vs
    58.6% for Zoo1.
  - The noise-trained victim stays upright more often, but then still loses most of the
    contests it reaches.

**P14. Reasoning** ◐
- "More conservative, stable movements" is supported indirectly by three pieces of evidence:
  - longer games (P9);
  - fewer pre-contact falls at σ = 1 (P13);
  - a clean cost vs the control (P9).
- "The LSTM learns to average recent observations" is **untested**. Checking it needs a look
  inside the network, e.g. whether the noise-trained LSTM's hidden state changes less
  step-to-step under noise than Zoo1's. That's the deferred activation analysis.
- "Partial generalization to σ = 2" did not happen.

**P15. Confidence (low)** ✅ appropriate. The predictions were right about Humans' direction
and mechanism, but wrong about Ants and about the size of the effects.

#### 4. Caveats that limit these conclusions

- **One training run per variant.** Each victim is a single 10M-step PPO run. We can't tell how
  much of a difference comes from training luck rather than the method.
  - Both Ants runs declining together argues against pure luck there.
  - The Humans +10.5 at σ = 1 is large, but we haven't measured run-to-run spread. The paper
    used 3 seeds.
- **Same opponent in training and evaluation.** Both fine-tuned victims trained against the Zoo1
  opponent they are evaluated against, so any improvement may be specific to that opponent.
- **Many comparisons.** About 42 victim-vs-victim tests; at p < 0.05, two could be false
  positives by chance. The robust results have p < 0.001: Humans σ = 1, the Ants clean drop,
  and the Humans σ = 0.5 gain over Zoo1.
- **10M steps may be too short.** The noise-trained Humans victim's training win rate was still
  rising at the end (≈ 49% → 55% over the last ~3M steps), so a longer run might help more.
- **Direct-observation threat model.** Noise is injected into the victim's inputs; this is not
  the paper's natural-observation attack, and robustness to random noise need not transfer to
  an adversarial opponent.

#### 5. Method note: the first training attempt failed (2026-11-12)

- The first four 10M-step trainings collapsed. The Humans control went from 80% to 0% wins
  during training.
- The cause was a bug in `gym_compete`'s LSTM policy, not in the method. It returned PPO's
  training outputs in the wrong order whenever a minibatch held more than one game.
- We fixed it, added a test, and re-trained from scratch (2026-11-13). Every result above
  comes from the re-trained victims. The predictions were committed before either training.
  Details: README, "A bug in `gym_compete`'s LSTM policy".

### 2026-11-24: Follow-up experiments (held-out opponents, second seed, 20M steps)

**What and why.** Three checks on the caveats in the 2026-11-16 note, run Nov 21–22. None of
them was predicted, so they are follow-up results, not tests of P1–P15.
- **Held-out opponents:** the six committed victims against Zoo2 and Zoo3 instead of Zoo1.
  Tests the caveat "same opponent in training and evaluation".
- **Second training seed:** all four fine-tuning runs again, with training seed 2000 / noise
  seed 2001. Tests "one training run per variant".
- **20M steps:** all four fine-tuning runs for 20M steps instead of 10M (from Zoo1, same
  seeds). Tests "10M steps may be too short".

Everything else is unchanged: σ grid, 1,000 games per point, sweep seeds. Raw tables:
`results/followup_summary.md`. Figure: `figures/followup_checks.png`.

**How to read this.** Same conventions as the 2026-11-16 note, plus:
- **Gap** = noise-trained − control win rate in the same condition, in points. Only the
  original gaps (first row of each table) belong to a Holm family (below). Every other
  p-value here is **uncorrected**.
- **Holm:** one correction over 182 comparisons: the 2026-11-16 note's 42 (noise-trained vs.
  control, noise-trained vs. Zoo1, control vs. Zoo1, at each σ in both games) plus 140 new
  ones (each follow-up sweep vs. its committed run, at each σ).

#### 1. Headline findings

1. **The Humans defense effect is real against the training opponent.** At σ = 1 the
   noise-trained victim beats the control with a second seed (+5.8) and with 20M steps
   (+10.8), and the original +10.5 survives the Holm correction (Holm p = 3e-9).
2. **It mostly does not transfer to new opponents.** Against Zoo2 the σ = 1 gap is −0.4;
   against Zoo3 it is +4.1, near the floor (every victim wins under 7%).
3. **Zoo2 and Zoo3 are much stronger than Zoo1 in Sumo Humans.** Every Humans victim's clean
   win rate falls from ~82% to 13–22%. So the 82% baseline belonged to the Zoo1-vs-Zoo1
   matchup, not to Sumo Humans in general.
4. **The defense still does nothing consistent in Sumo Ants**, and the fine-tuning drop below
   Zoo1 replicates with the second seed.
5. **Two results from the 2026-11-16 note are not robust** (see section 3): the Humans
   clean-performance cost (headline 4, P9) and the Humans σ = 0.5 gain (P10).

#### 2. Results

**Sumo Humans: gap at σ = 0, 0.5 and 1** (win rates: noise-trained / control)

| Condition | σ = 0 | σ = 0.5 | σ = 1 |
|---|---|---|---|
| Original (seed 1, 10M, vs Zoo1) | −4.7 (81.5 / 86.2) | +5.3 (59.9 / 54.6) | **+10.5** (19.6 / 9.1) |
| Second seed, vs Zoo1 | −1.1 (81.8 / 82.9) | +6.6, p = 0.003 | **+5.8** (17.7 / 11.9), p = 3e-4 |
| 20M steps, vs Zoo1 | −2.8 (86.8 / 89.6), p = 0.05 | +9.9, p = 7e-6 | **+10.8** (20.3 / 9.5), p = 1e-11 |
| Held-out Zoo2 | −6.9 (14.6 / 21.5), p = 6e-5 | −3.4, p = 0.02 | −0.4 (4.3 / 4.7), p = 0.67 |
| Held-out Zoo3 | −8.7 (13.2 / 21.9), p = 3e-7 | −2.9, p = 0.05 | +4.1 (6.9 / 2.8), p = 2e-5 |

- **Against Zoo1, the σ = 1 gain replicates.** With the second seed it is about half the
  size, so training luck moves it by several points, but both seeds agree on the direction.
  Longer training doesn't change it.
- **The σ = 0.5 gain grows with training** (+5.3 → +9.9) and replicates with the second seed.
- **Against Zoo2 and Zoo3, the noise-trained victim is worse with no noise** (−7 to −9) and
  at σ = 0.5. Its gain at σ = 1 is gone (Zoo2) or small (Zoo3).
  - **Hypothesis:** noise training tuned the victim to the one opponent it trained against.
    It learned a more cautious style that pays off against Zoo1 under noise, and that style
    costs it against stronger, unfamiliar opponents. Untested.
- **Cross-check for the Zoo2 result.** The original `aprl.score_agent` gives Zoo1 vs Zoo2
  29/200 wins (14.5%), matching our 18.4% within error. So the drop isn't a bug in our
  pipeline.

**Sumo Ants: gap at σ = 0, 0.5 and 1**

| Condition | σ = 0 | σ = 0.5 | σ = 1 |
|---|---|---|---|
| Original (seed 1, 10M, vs Zoo1) | +1.4 | +1.6 | −1.1 |
| Second seed, vs Zoo1 | −3.0 | −0.7 | −1.4 |
| 20M steps, vs Zoo1 | −6.8, p = 0.002 | −2.9 | −0.1 |
| Held-out Zoo2 | +10.2, p = 3e-6 | +4.4, p = 0.04 | 0.0 |
| Held-out Zoo3 | +4.0 | +4.3, p = 0.04 | +4.8, p = 0.004 |

- **No consistent pattern.** The signs change from condition to condition, and only one σ = 1
  gap differs from zero (Zoo3, uncorrected p = 0.004).
- **The fine-tuning drop replicates.** Clean win rates with the second seed: 35.9%
  (noise-trained) and 38.9% (control), vs. 46.2% for Zoo1.
- **20M steps didn't fix it.** The control recovered partly (42.1%); the noise-trained victim
  didn't (35.3%).
  - This doesn't fit the 2026-11-16 hypothesis well, a weak learning signal making both
    victims drift downhill. The hypothesis stays untested.
- Zoo2 and Zoo3 are about as strong as Zoo1 in Ants (Zoo1 victim, clean: 47.8% and 44.3% vs.
  46.2%).

**What survives the Holm correction**
- From the 2026-11-16 note's 42 comparisons, 9 survive:
  - Humans noise-trained − control at σ = 1: +10.5.
  - Humans noise-trained − Zoo1 at σ = 0.5 (+13.8) and σ = 1 (+9.7).
  - Humans control − Zoo1 at σ = 0.25 (+7.9) and σ = 0.5 (+8.5).
  - Ants noise-trained − Zoo1 at σ = 0 (−8.4) and σ = 0.1 (−10.1).
  - Ants control − Zoo1 at σ = 0 (−9.8) and σ = 0.1 (−8.6).
- From the 140 new ones, only the 36 Humans held-out-opponent drops survive (all six victims,
  both opponents, σ ≤ 1). They measure how strong Zoo2 and Zoo3 are, not the defense.
- No second-seed or 20M comparison with its committed run survives. The largest is the 20M
  Humans noise-trained victim at σ = 0: +5.3 over the 10M one, raw p = 0.001, Holm p = 0.16.

#### 3. Corrections to the 2026-11-16 note

That note is unchanged above; these points supersede it.
- **Headline 4 and P9 (Humans): the clean-performance cost is not established.** The −4.7
  vs. the control (p = 0.004) has Holm p = 0.55. It shrinks to −1.1 with the second seed and
  −2.8 with 20M steps. The longer games of the noise-trained victim (P9, P14) are still
  measured, but they no longer come with a clear clean cost.
- **P10 (Humans, σ = 0.5): weaker than "✅".** +5.3 has Holm p = 1, so the original run alone
  doesn't establish it. The same direction appears with the second seed (+6.6) and 20M steps
  (+9.9), so it is probably real but modest.
- **P11 (Humans benefits more): ✅ against Zoo1 only.** Against the held-out opponents neither
  game shows a dependable benefit.
- **Caveats (section 4 of that note):**
  - "One training run per variant": now two seeds. The Humans σ = 1 gain holds, but its size
    varies (+5.8 to +10.5).
  - "Same opponent in training and evaluation": confirmed as a real limitation (finding 2).
  - "10M steps may be too short": doubling it left the σ = 1 gain unchanged and enlarged the
    σ = 0.5 gain.

#### 4. What this means for the claim

Noise-augmented fine-tuning made the Sumo Humans victim more robust to Gaussian observation
noise **against the opponent it was trained with**. That held across a second seed and
longer training. It did not help Sumo Ants, and the Humans benefit largely did not carry over
to opponents it had not trained against. Limits:
- The threat model is still direct observation.
- Two seeds are better than one, but still few.
- The within-condition gaps after the first row of each table are uncorrected.

### 2026-11-24 (later): Holm correction extended to the within-condition comparisons

**What changed.** The note above corrected 182 comparisons. That left out the comparisons it
leans on most: noise-trained vs. control *within* each follow-up condition (the "gap" tables in
its section 2). Those 56 (4 conditions × 7 σ × 2 games) are now in the same Holm correction,
which covers **238 comparisons**.
- They were added after the results were in. Adding comparisons only makes the bar stricter
  for everything already in the family, so this can't make an earlier result look better.
- Every raw p-value is unchanged. Only the Holm p-values moved.
- Figures are unchanged. Full tables: `results/followup_summary.md`, section 4.

**Within-condition gaps that survive** (13 of 56):

| Game | Condition | σ | Gap (points) | Holm p |
|---|---|---|---|---|
| Humans | 20M steps, vs Zoo1 | 1 | +10.8 | 2e-9 |
| Humans | 20M steps, vs Zoo1 | 0.5 | +9.9 | 0.001 |
| Humans | second seed, vs Zoo1 | 1 | +5.8 | 0.047 |
| Humans | held-out Zoo3 | 1 | +4.1 | 0.004 |
| Humans | held-out Zoo2 | 0, 0.05, 0.1 | −6.9 to −9.7 | ≤ 0.011 |
| Humans | held-out Zoo3 | 0, 0.05, 0.1, 0.25 | −8.7 to −12.4 | ≤ 6e-5 |
| Ants | held-out Zoo2 | 0, 0.05 | +10.2, +9.6 | ≤ 0.002 |

**What this changes in the note above.**
- **Humans σ = 1 now holds after correction in all three runs against Zoo1:** the original
  +10.5 (Holm p 4e-9), the second seed +5.8 (Holm p 0.047, only just), and 20M steps +10.8.
  Finding 1 is now corrected evidence, not uncorrected.
- **Against Zoo3, the σ = 1 gain (+4.1) also survives.** Against Zoo2 there is still no gain
  (−0.4). So "mostly does not transfer" (finding 2) stands, but against Zoo3 a small gain
  remains.
- **Against held-out opponents, the noise-trained Humans victim is worse than the control at
  low noise** (σ ≤ 0.1 vs Zoo2, σ ≤ 0.25 vs Zoo3), and this survives correction. That's
  stronger support for the "tuned to its training opponent" hypothesis, which is still
  untested as an explanation.
- **The σ = 0.5 Humans gain survives at 20M steps** (+9.9), but not with the second seed (+6.6,
  Holm p 0.51) or in the original run (+5.3). "Probably real but modest" still fits.
- **Ants: one new surviving result.** Against Zoo2 with no or little noise (σ = 0, 0.05), the
  noise-trained victim beats the control by about 10 points. It doesn't appear against Zoo3
  (+4.0, Holm p 1) or at higher noise, so it's a single-condition result, not evidence of a
  defense. The Zoo3 σ = 1 Ants gap (+4.8) does not survive (Holm p 0.65). "No consistent
  benefit in Ants" stands.
- **Unchanged:** the 9 surviving comparisons from the 2026-11-16 note, the 36 surviving
  held-out-opponent drops, and the corrections in section 3 of the note above (the Humans
  clean cost and the original σ = 0.5 gain still don't survive).
