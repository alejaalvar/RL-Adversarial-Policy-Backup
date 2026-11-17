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
