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

_Write this section before the noise-augmented fine-tuning runs, once round 2's design is
settled._ It will cover: the defended victim's win rate at each σ, whether it loses any
clean (σ = 0) performance, and which game benefits more.

---

## Addenda

Dated notes added **after** results exist. Never edit the predictions above.

-
