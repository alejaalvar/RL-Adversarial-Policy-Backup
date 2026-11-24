# Follow-up experiments: summary

Generated 2026-11-24 at commit 7aa5837 by `followup/summarize.py` from the raw `episodes.csv` files. **Threat model: direct observation** (noise added straight to the victim's observations).

**How to read this.**
- Win rate = victim wins ÷ all games (ties count as non-wins), with a Wilson 95% interval.
- *Committed* is the matching committed run: same victim and σ, trained with seed 1000 for 10M steps, against opponent Zoo1 (`results/<run>/` without the suffix).
- *Difference* = follow-up − committed, in percentage points. *p*: two-sided pooled two-proportion z test, as in the PREDICTIONS.md addendum.
- *Holm p*: Holm-corrected over **238 comparisons**: the addendum's 42 (noise-trained vs. control, noise-trained vs. Zoo1, control vs. Zoo1, at each σ in both games), 140 follow-up sweeps vs. their committed runs, and 56 noise-trained vs. control within each follow-up condition (section 4; added to the correction on 2026-11-24, after the results were in). ✱ = significant at 0.05 after correction.
- For the held-out opponents, the difference measures the *opponent* change (the victim is the same), so a drop means Zoo2/Zoo3 is a harder opponent for that victim.

## Failed, skipped or missing runs

None: every step finished and every sweep is complete.

## 1. Held-out opponent Zoo2

The victim is unchanged; only the opponent differs.

### ants_defended_opp2 (vs. ants_defended)

| σ | wins / games | win rate [95% CI] | committed | difference (points) | p | Holm p |
|---|---|---|---|---|---|---|
| 0 | 431 / 1000 | 43.1% [40.1, 46.2] | 37.8% | +5.3 | 0.016 | 1 |
| 0.05 | 430 / 1000 | 43.0% [40.0, 46.1] | 39.0% | +4.0 | 0.069 | 1 |
| 0.1 | 403 / 1000 | 40.3% [37.3, 43.4] | 34.4% | +5.9 | 0.0064 | 1 |
| 0.25 | 423 / 1000 | 42.3% [39.3, 45.4] | 35.0% | +7.3 | 8.0e-04 | 0.14 |
| 0.5 | 347 / 1000 | 34.7% [31.8, 37.7] | 30.2% | +4.5 | 0.032 | 1 |
| 1 | 148 / 1000 | 14.8% [12.7, 17.1] | 15.9% | -1.1 | 0.5 | 1 |
| 2 | 61 / 1000 | 6.1% [4.8, 7.8] | 6.4% | -0.3 | 0.78 | 1 |

### ants_finetuned_clean_opp2 (vs. ants_finetuned_clean)

| σ | wins / games | win rate [95% CI] | committed | difference (points) | p | Holm p |
|---|---|---|---|---|---|---|
| 0 | 329 / 1000 | 32.9% [30.1, 35.9] | 36.4% | -3.5 | 0.1 | 1 |
| 0.05 | 334 / 1000 | 33.4% [30.5, 36.4] | 36.3% | -2.9 | 0.17 | 1 |
| 0.1 | 355 / 1000 | 35.5% [32.6, 38.5] | 35.9% | -0.4 | 0.85 | 1 |
| 0.25 | 371 / 1000 | 37.1% [34.2, 40.1] | 34.9% | +2.2 | 0.31 | 1 |
| 0.5 | 303 / 1000 | 30.3% [27.5, 33.2] | 28.6% | +1.7 | 0.4 | 1 |
| 1 | 148 / 1000 | 14.8% [12.7, 17.1] | 17.0% | -2.2 | 0.18 | 1 |
| 2 | 59 / 1000 | 5.9% [4.6, 7.5] | 7.5% | -1.6 | 0.15 | 1 |

### ants_zoo1_original_opp2 (vs. ants_zoo1_original)

| σ | wins / games | win rate [95% CI] | committed | difference (points) | p | Holm p |
|---|---|---|---|---|---|---|
| 0 | 478 / 1000 | 47.8% [44.7, 50.9] | 46.2% | +1.6 | 0.47 | 1 |
| 0.05 | 485 / 1000 | 48.5% [45.4, 51.6] | 42.1% | +6.4 | 0.004 | 0.67 |
| 0.1 | 452 / 1000 | 45.2% [42.1, 48.3] | 44.5% | +0.7 | 0.75 | 1 |
| 0.25 | 430 / 1000 | 43.0% [40.0, 46.1] | 42.1% | +0.9 | 0.68 | 1 |
| 0.5 | 360 / 1000 | 36.0% [33.1, 39.0] | 33.8% | +2.2 | 0.3 | 1 |
| 1 | 157 / 1000 | 15.7% [13.6, 18.1] | 15.2% | +0.5 | 0.76 | 1 |
| 2 | 65 / 1000 | 6.5% [5.1, 8.2] | 8.1% | -1.6 | 0.17 | 1 |

### humans_defended_opp2 (vs. humans_defended)

| σ | wins / games | win rate [95% CI] | committed | difference (points) | p | Holm p |
|---|---|---|---|---|---|---|
| 0 | 146 / 1000 | 14.6% [12.5, 16.9] | 81.5% | -66.9 | 5.7e-197 | 1.3e-194 ✱ |
| 0.05 | 106 / 1000 | 10.6% [8.8, 12.7] | 79.5% | -68.9 | 1.6e-210 | 3.8e-208 ✱ |
| 0.1 | 123 / 1000 | 12.3% [10.4, 14.5] | 81.5% | -69.2 | 4.4e-211 | 1.0e-208 ✱ |
| 0.25 | 133 / 1000 | 13.3% [11.3, 15.5] | 76.9% | -63.6 | 1.2e-179 | 2.7e-177 ✱ |
| 0.5 | 110 / 1000 | 11.0% [9.2, 13.1] | 59.9% | -48.9 | 1.2e-115 | 2.6e-113 ✱ |
| 1 | 43 / 1000 | 4.3% [3.2, 5.7] | 19.6% | -15.3 | 5.2e-26 | 1.1e-23 ✱ |
| 2 | 9 / 1000 | 0.9% [0.5, 1.7] | 0.6% | +0.3 | 0.44 | 1 |

### humans_finetuned_clean_opp2 (vs. humans_finetuned_clean)

| σ | wins / games | win rate [95% CI] | committed | difference (points) | p | Holm p |
|---|---|---|---|---|---|---|
| 0 | 215 / 1000 | 21.5% [19.1, 24.2] | 86.2% | -64.7 | 3.6e-185 | 8.2e-183 ✱ |
| 0.05 | 187 / 1000 | 18.7% [16.4, 21.2] | 84.1% | -65.4 | 3.4e-188 | 7.8e-186 ✱ |
| 0.1 | 220 / 1000 | 22.0% [19.5, 24.7] | 84.4% | -62.4 | 4.5e-172 | 1.0e-169 ✱ |
| 0.25 | 180 / 1000 | 18.0% [15.7, 20.5] | 78.6% | -60.6 | 6.2e-162 | 1.4e-159 ✱ |
| 0.5 | 144 / 1000 | 14.4% [12.4, 16.7] | 54.6% | -40.2 | 9.5e-80 | 2.0e-77 ✱ |
| 1 | 47 / 1000 | 4.7% [3.6, 6.2] | 9.1% | -4.4 | 1.0e-04 | 0.019 ✱ |
| 2 | 7 / 1000 | 0.7% [0.3, 1.4] | 0.4% | +0.3 | 0.36 | 1 |

### humans_zoo1_original_opp2 (vs. humans_zoo1_original)

| σ | wins / games | win rate [95% CI] | committed | difference (points) | p | Holm p |
|---|---|---|---|---|---|---|
| 0 | 184 / 1000 | 18.4% [16.1, 20.9] | 82.3% | -63.9 | 1.3e-179 | 2.9e-177 ✱ |
| 0.05 | 175 / 1000 | 17.5% [15.3, 20.0] | 81.2% | -63.7 | 1.6e-178 | 3.5e-176 ✱ |
| 0.1 | 175 / 1000 | 17.5% [15.3, 20.0] | 80.2% | -62.7 | 4.3e-173 | 9.4e-171 ✱ |
| 0.25 | 139 / 1000 | 13.9% [11.9, 16.2] | 70.7% | -56.8 | 9.4e-146 | 2.0e-143 ✱ |
| 0.5 | 114 / 1000 | 11.4% [9.6, 13.5] | 46.1% | -34.7 | 7.0e-66 | 1.5e-63 ✱ |
| 1 | 49 / 1000 | 4.9% [3.7, 6.4] | 9.9% | -5.0 | 1.9e-05 | 0.0037 ✱ |
| 2 | 4 / 1000 | 0.4% [0.2, 1.0] | 0.5% | -0.1 | 0.74 | 1 |

## 1. Held-out opponent Zoo3

The victim is unchanged; only the opponent differs.

### ants_defended_opp3 (vs. ants_defended)

| σ | wins / games | win rate [95% CI] | committed | difference (points) | p | Holm p |
|---|---|---|---|---|---|---|
| 0 | 407 / 1000 | 40.7% [37.7, 43.8] | 37.8% | +2.9 | 0.18 | 1 |
| 0.05 | 411 / 1000 | 41.1% [38.1, 44.2] | 39.0% | +2.1 | 0.34 | 1 |
| 0.1 | 396 / 1000 | 39.6% [36.6, 42.7] | 34.4% | +5.2 | 0.016 | 1 |
| 0.25 | 380 / 1000 | 38.0% [35.0, 41.0] | 35.0% | +3.0 | 0.16 | 1 |
| 0.5 | 350 / 1000 | 35.0% [32.1, 38.0] | 30.2% | +4.8 | 0.022 | 1 |
| 1 | 189 / 1000 | 18.9% [16.6, 21.4] | 15.9% | +3.0 | 0.077 | 1 |
| 2 | 77 / 1000 | 7.7% [6.2, 9.5] | 6.4% | +1.3 | 0.26 | 1 |

### ants_finetuned_clean_opp3 (vs. ants_finetuned_clean)

| σ | wins / games | win rate [95% CI] | committed | difference (points) | p | Holm p |
|---|---|---|---|---|---|---|
| 0 | 367 / 1000 | 36.7% [33.8, 39.7] | 36.4% | +0.3 | 0.89 | 1 |
| 0.05 | 372 / 1000 | 37.2% [34.3, 40.2] | 36.3% | +0.9 | 0.68 | 1 |
| 0.1 | 343 / 1000 | 34.3% [31.4, 37.3] | 35.9% | -1.6 | 0.45 | 1 |
| 0.25 | 359 / 1000 | 35.9% [33.0, 38.9] | 34.9% | +1.0 | 0.64 | 1 |
| 0.5 | 307 / 1000 | 30.7% [27.9, 33.6] | 28.6% | +2.1 | 0.3 | 1 |
| 1 | 141 / 1000 | 14.1% [12.1, 16.4] | 17.0% | -2.9 | 0.074 | 1 |
| 2 | 84 / 1000 | 8.4% [6.8, 10.3] | 7.5% | +0.9 | 0.46 | 1 |

### ants_zoo1_original_opp3 (vs. ants_zoo1_original)

| σ | wins / games | win rate [95% CI] | committed | difference (points) | p | Holm p |
|---|---|---|---|---|---|---|
| 0 | 443 / 1000 | 44.3% [41.2, 47.4] | 46.2% | -1.9 | 0.39 | 1 |
| 0.05 | 436 / 1000 | 43.6% [40.6, 46.7] | 42.1% | +1.5 | 0.5 | 1 |
| 0.1 | 451 / 1000 | 45.1% [42.0, 48.2] | 44.5% | +0.6 | 0.79 | 1 |
| 0.25 | 451 / 1000 | 45.1% [42.0, 48.2] | 42.1% | +3.0 | 0.18 | 1 |
| 0.5 | 350 / 1000 | 35.0% [32.1, 38.0] | 33.8% | +1.2 | 0.57 | 1 |
| 1 | 204 / 1000 | 20.4% [18.0, 23.0] | 15.2% | +5.2 | 0.0024 | 0.41 |
| 2 | 85 / 1000 | 8.5% [6.9, 10.4] | 8.1% | +0.4 | 0.75 | 1 |

### humans_defended_opp3 (vs. humans_defended)

| σ | wins / games | win rate [95% CI] | committed | difference (points) | p | Holm p |
|---|---|---|---|---|---|---|
| 0 | 132 / 1000 | 13.2% [11.2, 15.4] | 81.5% | -68.3 | 1.8e-205 | 4.2e-203 ✱ |
| 0.05 | 119 / 1000 | 11.9% [10.0, 14.1] | 79.5% | -67.6 | 3.0e-202 | 7.0e-200 ✱ |
| 0.1 | 118 / 1000 | 11.8% [9.9, 13.9] | 81.5% | -69.7 | 3.0e-214 | 7.0e-212 ✱ |
| 0.25 | 88 / 1000 | 8.8% [7.2, 10.7] | 76.9% | -68.1 | 6.3e-208 | 1.5e-205 ✱ |
| 0.5 | 107 / 1000 | 10.7% [8.9, 12.8] | 59.9% | -49.2 | 2.9e-117 | 6.2e-115 ✱ |
| 1 | 69 / 1000 | 6.9% [5.5, 8.6] | 19.6% | -12.7 | 5.5e-17 | 1.1e-14 ✱ |
| 2 | 12 / 1000 | 1.2% [0.7, 2.1] | 0.6% | +0.6 | 0.16 | 1 |

### humans_finetuned_clean_opp3 (vs. humans_finetuned_clean)

| σ | wins / games | win rate [95% CI] | committed | difference (points) | p | Holm p |
|---|---|---|---|---|---|---|
| 0 | 219 / 1000 | 21.9% [19.4, 24.6] | 86.2% | -64.3 | 5.0e-183 | 1.1e-180 ✱ |
| 0.05 | 206 / 1000 | 20.6% [18.2, 23.2] | 84.1% | -63.5 | 8.7e-178 | 1.9e-175 ✱ |
| 0.1 | 226 / 1000 | 22.6% [20.1, 25.3] | 84.4% | -61.8 | 6.0e-169 | 1.3e-166 ✱ |
| 0.25 | 212 / 1000 | 21.2% [18.8, 23.8] | 78.6% | -57.4 | 2.5e-145 | 5.4e-143 ✱ |
| 0.5 | 136 / 1000 | 13.6% [11.6, 15.9] | 54.6% | -41.0 | 2.5e-83 | 5.3e-81 ✱ |
| 1 | 28 / 1000 | 2.8% [1.9, 4.0] | 9.1% | -6.3 | 2.6e-09 | 5.2e-07 ✱ |
| 2 | 5 / 1000 | 0.5% [0.2, 1.2] | 0.4% | +0.1 | 0.74 | 1 |

### humans_zoo1_original_opp3 (vs. humans_zoo1_original)

| σ | wins / games | win rate [95% CI] | committed | difference (points) | p | Holm p |
|---|---|---|---|---|---|---|
| 0 | 167 / 1000 | 16.7% [14.5, 19.1] | 82.3% | -65.6 | 3.3e-189 | 7.7e-187 ✱ |
| 0.05 | 147 / 1000 | 14.7% [12.6, 17.0] | 81.2% | -66.5 | 1.1e-194 | 2.6e-192 ✱ |
| 0.1 | 161 / 1000 | 16.1% [14.0, 18.5] | 80.2% | -64.1 | 5.7e-181 | 1.3e-178 ✱ |
| 0.25 | 143 / 1000 | 14.3% [12.3, 16.6] | 70.7% | -56.4 | 1.5e-143 | 3.2e-141 ✱ |
| 0.5 | 107 / 1000 | 10.7% [8.9, 12.8] | 46.1% | -35.4 | 5.6e-69 | 1.2e-66 ✱ |
| 1 | 47 / 1000 | 4.7% [3.6, 6.2] | 9.9% | -5.2 | 7.8e-06 | 0.0015 ✱ |
| 2 | 3 / 1000 | 0.3% [0.1, 0.9] | 0.5% | -0.2 | 0.48 | 1 |

## 2. Extra training seed

Training seed 2000 / noise seed 2001 instead of 1000 / 1001.

### ants_defended_seed2 (vs. ants_defended)

| σ | wins / games | win rate [95% CI] | committed | difference (points) | p | Holm p |
|---|---|---|---|---|---|---|
| 0 | 359 / 1000 | 35.9% [33.0, 38.9] | 37.8% | -1.9 | 0.38 | 1 |
| 0.05 | 358 / 1000 | 35.8% [32.9, 38.8] | 39.0% | -3.2 | 0.14 | 1 |
| 0.1 | 342 / 1000 | 34.2% [31.3, 37.2] | 34.4% | -0.2 | 0.92 | 1 |
| 0.25 | 338 / 1000 | 33.8% [30.9, 36.8] | 35.0% | -1.2 | 0.57 | 1 |
| 0.5 | 303 / 1000 | 30.3% [27.5, 33.2] | 30.2% | +0.1 | 0.96 | 1 |
| 1 | 159 / 1000 | 15.9% [13.8, 18.3] | 15.9% | +0.0 | 1 | 1 |
| 2 | 77 / 1000 | 7.7% [6.2, 9.5] | 6.4% | +1.3 | 0.26 | 1 |

### ants_finetuned_clean_seed2 (vs. ants_finetuned_clean)

| σ | wins / games | win rate [95% CI] | committed | difference (points) | p | Holm p |
|---|---|---|---|---|---|---|
| 0 | 389 / 1000 | 38.9% [35.9, 42.0] | 36.4% | +2.5 | 0.25 | 1 |
| 0.05 | 377 / 1000 | 37.7% [34.7, 40.7] | 36.3% | +1.4 | 0.52 | 1 |
| 0.1 | 378 / 1000 | 37.8% [34.8, 40.8] | 35.9% | +1.9 | 0.38 | 1 |
| 0.25 | 375 / 1000 | 37.5% [34.6, 40.5] | 34.9% | +2.6 | 0.23 | 1 |
| 0.5 | 310 / 1000 | 31.0% [28.2, 33.9] | 28.6% | +2.4 | 0.24 | 1 |
| 1 | 173 / 1000 | 17.3% [15.1, 19.8] | 17.0% | +0.3 | 0.86 | 1 |
| 2 | 77 / 1000 | 7.7% [6.2, 9.5] | 7.5% | +0.2 | 0.87 | 1 |

### humans_defended_seed2 (vs. humans_defended)

| σ | wins / games | win rate [95% CI] | committed | difference (points) | p | Holm p |
|---|---|---|---|---|---|---|
| 0 | 818 / 1000 | 81.8% [79.3, 84.1] | 81.5% | +0.3 | 0.86 | 1 |
| 0.05 | 793 / 1000 | 79.3% [76.7, 81.7] | 79.5% | -0.2 | 0.91 | 1 |
| 0.1 | 795 / 1000 | 79.5% [76.9, 81.9] | 81.5% | -2.0 | 0.26 | 1 |
| 0.25 | 755 / 1000 | 75.5% [72.7, 78.1] | 76.9% | -1.4 | 0.46 | 1 |
| 0.5 | 583 / 1000 | 58.3% [55.2, 61.3] | 59.9% | -1.6 | 0.47 | 1 |
| 1 | 177 / 1000 | 17.7% [15.5, 20.2] | 19.6% | -1.9 | 0.28 | 1 |
| 2 | 4 / 1000 | 0.4% [0.2, 1.0] | 0.6% | -0.2 | 0.53 | 1 |

### humans_finetuned_clean_seed2 (vs. humans_finetuned_clean)

| σ | wins / games | win rate [95% CI] | committed | difference (points) | p | Holm p |
|---|---|---|---|---|---|---|
| 0 | 829 / 1000 | 82.9% [80.4, 85.1] | 86.2% | -3.3 | 0.041 | 1 |
| 0.05 | 835 / 1000 | 83.5% [81.1, 85.7] | 84.1% | -0.6 | 0.72 | 1 |
| 0.1 | 846 / 1000 | 84.6% [82.2, 86.7] | 84.4% | +0.2 | 0.9 | 1 |
| 0.25 | 760 / 1000 | 76.0% [73.3, 78.5] | 78.6% | -2.6 | 0.17 | 1 |
| 0.5 | 517 / 1000 | 51.7% [48.6, 54.8] | 54.6% | -2.9 | 0.19 | 1 |
| 1 | 119 / 1000 | 11.9% [10.0, 14.1] | 9.1% | +2.8 | 0.041 | 1 |
| 2 | 7 / 1000 | 0.7% [0.3, 1.4] | 0.4% | +0.3 | 0.36 | 1 |

## 3. Twice as long (20M steps)

Trained from Zoo1 for 20M steps instead of 10M.

### ants_defended_20M (vs. ants_defended)

| σ | wins / games | win rate [95% CI] | committed | difference (points) | p | Holm p |
|---|---|---|---|---|---|---|
| 0 | 353 / 1000 | 35.3% [32.4, 38.3] | 37.8% | -2.5 | 0.25 | 1 |
| 0.05 | 343 / 1000 | 34.3% [31.4, 37.3] | 39.0% | -4.7 | 0.029 | 1 |
| 0.1 | 358 / 1000 | 35.8% [32.9, 38.8] | 34.4% | +1.4 | 0.51 | 1 |
| 0.25 | 335 / 1000 | 33.5% [30.6, 36.5] | 35.0% | -1.5 | 0.48 | 1 |
| 0.5 | 299 / 1000 | 29.9% [27.1, 32.8] | 30.2% | -0.3 | 0.88 | 1 |
| 1 | 170 / 1000 | 17.0% [14.8, 19.5] | 15.9% | +1.1 | 0.51 | 1 |
| 2 | 72 / 1000 | 7.2% [5.8, 9.0] | 6.4% | +0.8 | 0.48 | 1 |

### ants_finetuned_clean_20M (vs. ants_finetuned_clean)

| σ | wins / games | win rate [95% CI] | committed | difference (points) | p | Holm p |
|---|---|---|---|---|---|---|
| 0 | 421 / 1000 | 42.1% [39.1, 45.2] | 36.4% | +5.7 | 0.0091 | 1 |
| 0.05 | 400 / 1000 | 40.0% [37.0, 43.1] | 36.3% | +3.7 | 0.089 | 1 |
| 0.1 | 421 / 1000 | 42.1% [39.1, 45.2] | 35.9% | +6.2 | 0.0045 | 0.73 |
| 0.25 | 392 / 1000 | 39.2% [36.2, 42.3] | 34.9% | +4.3 | 0.046 | 1 |
| 0.5 | 328 / 1000 | 32.8% [30.0, 35.8] | 28.6% | +4.2 | 0.042 | 1 |
| 1 | 171 / 1000 | 17.1% [14.9, 19.6] | 17.0% | +0.1 | 0.95 | 1 |
| 2 | 73 / 1000 | 7.3% [5.8, 9.1] | 7.5% | -0.2 | 0.86 | 1 |

### humans_defended_20M (vs. humans_defended)

| σ | wins / games | win rate [95% CI] | committed | difference (points) | p | Holm p |
|---|---|---|---|---|---|---|
| 0 | 868 / 1000 | 86.8% [84.6, 88.8] | 81.5% | +5.3 | 0.0012 | 0.21 |
| 0.05 | 852 / 1000 | 85.2% [82.9, 87.3] | 79.5% | +5.7 | 8.3e-04 | 0.15 |
| 0.1 | 838 / 1000 | 83.8% [81.4, 86.0] | 81.5% | +2.3 | 0.17 | 1 |
| 0.25 | 814 / 1000 | 81.4% [78.9, 83.7] | 76.9% | +4.5 | 0.013 | 1 |
| 0.5 | 630 / 1000 | 63.0% [60.0, 65.9] | 59.9% | +3.1 | 0.15 | 1 |
| 1 | 203 / 1000 | 20.3% [17.9, 22.9] | 19.6% | +0.7 | 0.7 | 1 |
| 2 | 10 / 1000 | 1.0% [0.5, 1.8] | 0.6% | +0.4 | 0.32 | 1 |

### humans_finetuned_clean_20M (vs. humans_finetuned_clean)

| σ | wins / games | win rate [95% CI] | committed | difference (points) | p | Holm p |
|---|---|---|---|---|---|---|
| 0 | 896 / 1000 | 89.6% [87.6, 91.3] | 86.2% | +3.4 | 0.02 | 1 |
| 0.05 | 876 / 1000 | 87.6% [85.4, 89.5] | 84.1% | +3.5 | 0.025 | 1 |
| 0.1 | 880 / 1000 | 88.0% [85.8, 89.9] | 84.4% | +3.6 | 0.02 | 1 |
| 0.25 | 839 / 1000 | 83.9% [81.5, 86.0] | 78.6% | +5.3 | 0.0024 | 0.41 |
| 0.5 | 531 / 1000 | 53.1% [50.0, 56.2] | 54.6% | -1.5 | 0.5 | 1 |
| 1 | 95 / 1000 | 9.5% [7.8, 11.5] | 9.1% | +0.4 | 0.76 | 1 |
| 2 | 4 / 1000 | 0.4% [0.2, 1.0] | 0.4% | +0.0 | 1 | 1 |

## 4. Noise-trained vs. control within each condition

Does the defense still beat the control? Both victims from the same condition, at each σ.

### Held-out opponent Zoo2: Sumo Ants

| σ | noise-trained | control | gap (points) | p | Holm p |
|---|---|---|---|---|---|
| 0 | 43.1% | 32.9% | +10.2 | 2.6e-06 | 5.1e-04 ✱ |
| 0.05 | 43.0% | 33.4% | +9.6 | 1.0e-05 | 0.0019 ✱ |
| 0.1 | 40.3% | 35.5% | +4.8 | 0.027 | 1 |
| 0.25 | 42.3% | 37.1% | +5.2 | 0.017 | 1 |
| 0.5 | 34.7% | 30.3% | +4.4 | 0.036 | 1 |
| 1 | 14.8% | 14.8% | +0.0 | 1 | 1 |
| 2 | 6.1% | 5.9% | +0.2 | 0.85 | 1 |

### Held-out opponent Zoo2: Sumo Humans

| σ | noise-trained | control | gap (points) | p | Holm p |
|---|---|---|---|---|---|
| 0 | 14.6% | 21.5% | -6.9 | 6.0e-05 | 0.011 ✱ |
| 0.05 | 10.6% | 18.7% | -8.1 | 3.0e-07 | 6.0e-05 ✱ |
| 0.1 | 12.3% | 22.0% | -9.7 | 8.7e-09 | 1.7e-06 ✱ |
| 0.25 | 13.3% | 18.0% | -4.7 | 0.0038 | 0.65 |
| 0.5 | 11.0% | 14.4% | -3.4 | 0.022 | 1 |
| 1 | 4.3% | 4.7% | -0.4 | 0.67 | 1 |
| 2 | 0.9% | 0.7% | +0.2 | 0.62 | 1 |

### Held-out opponent Zoo3: Sumo Ants

| σ | noise-trained | control | gap (points) | p | Holm p |
|---|---|---|---|---|---|
| 0 | 40.7% | 36.7% | +4.0 | 0.066 | 1 |
| 0.05 | 41.1% | 37.2% | +3.9 | 0.074 | 1 |
| 0.1 | 39.6% | 34.3% | +5.3 | 0.014 | 1 |
| 0.25 | 38.0% | 35.9% | +2.1 | 0.33 | 1 |
| 0.5 | 35.0% | 30.7% | +4.3 | 0.041 | 1 |
| 1 | 18.9% | 14.1% | +4.8 | 0.0038 | 0.65 |
| 2 | 7.7% | 8.4% | -0.7 | 0.57 | 1 |

### Held-out opponent Zoo3: Sumo Humans

| σ | noise-trained | control | gap (points) | p | Holm p |
|---|---|---|---|---|---|
| 0 | 13.2% | 21.9% | -8.7 | 3.2e-07 | 6.2e-05 ✱ |
| 0.05 | 11.9% | 20.6% | -8.7 | 1.3e-07 | 2.7e-05 ✱ |
| 0.1 | 11.8% | 22.6% | -10.8 | 1.6e-10 | 3.2e-08 ✱ |
| 0.25 | 8.8% | 21.2% | -12.4 | 8.2e-15 | 1.7e-12 ✱ |
| 0.5 | 10.7% | 13.6% | -2.9 | 0.047 | 1 |
| 1 | 6.9% | 2.8% | +4.1 | 2.0e-05 | 0.0037 ✱ |
| 2 | 1.2% | 0.5% | +0.7 | 0.088 | 1 |

### Extra training seed: Sumo Ants

| σ | noise-trained | control | gap (points) | p | Holm p |
|---|---|---|---|---|---|
| 0 | 35.9% | 38.9% | -3.0 | 0.17 | 1 |
| 0.05 | 35.8% | 37.7% | -1.9 | 0.38 | 1 |
| 0.1 | 34.2% | 37.8% | -3.6 | 0.094 | 1 |
| 0.25 | 33.8% | 37.5% | -3.7 | 0.084 | 1 |
| 0.5 | 30.3% | 31.0% | -0.7 | 0.73 | 1 |
| 1 | 15.9% | 17.3% | -1.4 | 0.4 | 1 |
| 2 | 7.7% | 7.7% | +0.0 | 1 | 1 |

### Extra training seed: Sumo Humans

| σ | noise-trained | control | gap (points) | p | Holm p |
|---|---|---|---|---|---|
| 0 | 81.8% | 82.9% | -1.1 | 0.52 | 1 |
| 0.05 | 79.3% | 83.5% | -4.2 | 0.016 | 1 |
| 0.1 | 79.5% | 84.6% | -5.1 | 0.003 | 0.51 |
| 0.25 | 75.5% | 76.0% | -0.5 | 0.79 | 1 |
| 0.5 | 58.3% | 51.7% | +6.6 | 0.003 | 0.51 |
| 1 | 17.7% | 11.9% | +5.8 | 2.6e-04 | 0.047 ✱ |
| 2 | 0.4% | 0.7% | -0.3 | 0.36 | 1 |

### Twice as long (20M steps): Sumo Ants

| σ | noise-trained | control | gap (points) | p | Holm p |
|---|---|---|---|---|---|
| 0 | 35.3% | 42.1% | -6.8 | 0.0018 | 0.31 |
| 0.05 | 34.3% | 40.0% | -5.7 | 0.0083 | 1 |
| 0.1 | 35.8% | 42.1% | -6.3 | 0.0039 | 0.65 |
| 0.25 | 33.5% | 39.2% | -5.7 | 0.0081 | 1 |
| 0.5 | 29.9% | 32.8% | -2.9 | 0.16 | 1 |
| 1 | 17.0% | 17.1% | -0.1 | 0.95 | 1 |
| 2 | 7.2% | 7.3% | -0.1 | 0.93 | 1 |

### Twice as long (20M steps): Sumo Humans

| σ | noise-trained | control | gap (points) | p | Holm p |
|---|---|---|---|---|---|
| 0 | 86.8% | 89.6% | -2.8 | 0.052 | 1 |
| 0.05 | 85.2% | 87.6% | -2.4 | 0.12 | 1 |
| 0.1 | 83.8% | 88.0% | -4.2 | 0.007 | 1 |
| 0.25 | 81.4% | 83.9% | -2.5 | 0.14 | 1 |
| 0.5 | 63.0% | 53.1% | +9.9 | 7.3e-06 | 0.0014 ✱ |
| 1 | 20.3% | 9.5% | +10.8 | 1.2e-11 | 2.4e-09 ✱ |
| 2 | 1.0% | 0.4% | +0.6 | 0.11 | 1 |

## Holm-corrected list: all 238 comparisons

Sorted by raw p. *A − B* is the difference in victim win rate.

| # | source | A − B | σ | difference (points) | p | Holm p | significant |
|---|---|---|---|---|---|---|---|
| 1 | vs. committed | humans_defended_opp3 − humans_defended | 0.1 | -69.7 | 3.0e-214 | 7.0e-212 | yes |
| 2 | vs. committed | humans_defended_opp2 − humans_defended | 0.1 | -69.2 | 4.4e-211 | 1.0e-208 | yes |
| 3 | vs. committed | humans_defended_opp2 − humans_defended | 0.05 | -68.9 | 1.6e-210 | 3.8e-208 | yes |
| 4 | vs. committed | humans_defended_opp3 − humans_defended | 0.25 | -68.1 | 6.3e-208 | 1.5e-205 | yes |
| 5 | vs. committed | humans_defended_opp3 − humans_defended | 0 | -68.3 | 1.8e-205 | 4.2e-203 | yes |
| 6 | vs. committed | humans_defended_opp3 − humans_defended | 0.05 | -67.6 | 3.0e-202 | 7.0e-200 | yes |
| 7 | vs. committed | humans_defended_opp2 − humans_defended | 0 | -66.9 | 5.7e-197 | 1.3e-194 | yes |
| 8 | vs. committed | humans_zoo1_original_opp3 − humans_zoo1_original | 0.05 | -66.5 | 1.1e-194 | 2.6e-192 | yes |
| 9 | vs. committed | humans_zoo1_original_opp3 − humans_zoo1_original | 0 | -65.6 | 3.3e-189 | 7.7e-187 | yes |
| 10 | vs. committed | humans_finetuned_clean_opp2 − humans_finetuned_clean | 0.05 | -65.4 | 3.4e-188 | 7.8e-186 | yes |
| 11 | vs. committed | humans_finetuned_clean_opp2 − humans_finetuned_clean | 0 | -64.7 | 3.6e-185 | 8.2e-183 | yes |
| 12 | vs. committed | humans_finetuned_clean_opp3 − humans_finetuned_clean | 0 | -64.3 | 5.0e-183 | 1.1e-180 | yes |
| 13 | vs. committed | humans_zoo1_original_opp3 − humans_zoo1_original | 0.1 | -64.1 | 5.7e-181 | 1.3e-178 | yes |
| 14 | vs. committed | humans_defended_opp2 − humans_defended | 0.25 | -63.6 | 1.2e-179 | 2.7e-177 | yes |
| 15 | vs. committed | humans_zoo1_original_opp2 − humans_zoo1_original | 0 | -63.9 | 1.3e-179 | 2.9e-177 | yes |
| 16 | vs. committed | humans_zoo1_original_opp2 − humans_zoo1_original | 0.05 | -63.7 | 1.6e-178 | 3.5e-176 | yes |
| 17 | vs. committed | humans_finetuned_clean_opp3 − humans_finetuned_clean | 0.05 | -63.5 | 8.7e-178 | 1.9e-175 | yes |
| 18 | vs. committed | humans_zoo1_original_opp2 − humans_zoo1_original | 0.1 | -62.7 | 4.3e-173 | 9.4e-171 | yes |
| 19 | vs. committed | humans_finetuned_clean_opp2 − humans_finetuned_clean | 0.1 | -62.4 | 4.5e-172 | 1.0e-169 | yes |
| 20 | vs. committed | humans_finetuned_clean_opp3 − humans_finetuned_clean | 0.1 | -61.8 | 6.0e-169 | 1.3e-166 | yes |
| 21 | vs. committed | humans_finetuned_clean_opp2 − humans_finetuned_clean | 0.25 | -60.6 | 6.2e-162 | 1.4e-159 | yes |
| 22 | vs. committed | humans_zoo1_original_opp2 − humans_zoo1_original | 0.25 | -56.8 | 9.4e-146 | 2.0e-143 | yes |
| 23 | vs. committed | humans_finetuned_clean_opp3 − humans_finetuned_clean | 0.25 | -57.4 | 2.5e-145 | 5.4e-143 | yes |
| 24 | vs. committed | humans_zoo1_original_opp3 − humans_zoo1_original | 0.25 | -56.4 | 1.5e-143 | 3.2e-141 | yes |
| 25 | vs. committed | humans_defended_opp3 − humans_defended | 0.5 | -49.2 | 2.9e-117 | 6.2e-115 | yes |
| 26 | vs. committed | humans_defended_opp2 − humans_defended | 0.5 | -48.9 | 1.2e-115 | 2.6e-113 | yes |
| 27 | vs. committed | humans_finetuned_clean_opp3 − humans_finetuned_clean | 0.5 | -41.0 | 2.5e-83 | 5.3e-81 | yes |
| 28 | vs. committed | humans_finetuned_clean_opp2 − humans_finetuned_clean | 0.5 | -40.2 | 9.5e-80 | 2.0e-77 | yes |
| 29 | vs. committed | humans_zoo1_original_opp3 − humans_zoo1_original | 0.5 | -35.4 | 5.6e-69 | 1.2e-66 | yes |
| 30 | vs. committed | humans_zoo1_original_opp2 − humans_zoo1_original | 0.5 | -34.7 | 7.0e-66 | 1.5e-63 | yes |
| 31 | vs. committed | humans_defended_opp2 − humans_defended | 1 | -15.3 | 5.2e-26 | 1.1e-23 | yes |
| 32 | vs. committed | humans_defended_opp3 − humans_defended | 1 | -12.7 | 5.5e-17 | 1.1e-14 | yes |
| 33 | within condition | humans_defended_opp3 − humans_finetuned_clean_opp3 | 0.25 | -12.4 | 8.2e-15 | 1.7e-12 | yes |
| 34 | within condition | humans_defended_20M − humans_finetuned_clean_20M | 1 | +10.8 | 1.2e-11 | 2.4e-09 | yes |
| 35 | addendum | humans_defended − humans_finetuned_clean | 1 | +10.5 | 2.1e-11 | 4.3e-09 | yes |
| 36 | within condition | humans_defended_opp3 − humans_finetuned_clean_opp3 | 0.1 | -10.8 | 1.6e-10 | 3.2e-08 | yes |
| 37 | addendum | humans_defended − humans_zoo1_original | 0.5 | +13.8 | 6.3e-10 | 1.3e-07 | yes |
| 38 | addendum | humans_defended − humans_zoo1_original | 1 | +9.7 | 9.6e-10 | 1.9e-07 | yes |
| 39 | vs. committed | humans_finetuned_clean_opp3 − humans_finetuned_clean | 1 | -6.3 | 2.6e-09 | 5.2e-07 | yes |
| 40 | within condition | humans_defended_opp2 − humans_finetuned_clean_opp2 | 0.1 | -9.7 | 8.7e-09 | 1.7e-06 | yes |
| 41 | within condition | humans_defended_opp3 − humans_finetuned_clean_opp3 | 0.05 | -8.7 | 1.3e-07 | 2.7e-05 | yes |
| 42 | within condition | humans_defended_opp2 − humans_finetuned_clean_opp2 | 0.05 | -8.1 | 3.0e-07 | 6.0e-05 | yes |
| 43 | within condition | humans_defended_opp3 − humans_finetuned_clean_opp3 | 0 | -8.7 | 3.2e-07 | 6.2e-05 | yes |
| 44 | within condition | ants_defended_opp2 − ants_finetuned_clean_opp2 | 0 | +10.2 | 2.6e-06 | 5.1e-04 | yes |
| 45 | addendum | ants_defended − ants_zoo1_original | 0.1 | -10.1 | 3.8e-06 | 7.4e-04 | yes |
| 46 | within condition | humans_defended_20M − humans_finetuned_clean_20M | 0.5 | +9.9 | 7.3e-06 | 0.0014 | yes |
| 47 | vs. committed | humans_zoo1_original_opp3 − humans_zoo1_original | 1 | -5.2 | 7.8e-06 | 0.0015 | yes |
| 48 | addendum | ants_finetuned_clean − ants_zoo1_original | 0 | -9.8 | 8.6e-06 | 0.0016 | yes |
| 49 | within condition | ants_defended_opp2 − ants_finetuned_clean_opp2 | 0.05 | +9.6 | 1.0e-05 | 0.0019 | yes |
| 50 | vs. committed | humans_zoo1_original_opp2 − humans_zoo1_original | 1 | -5.0 | 1.9e-05 | 0.0037 | yes |
| 51 | within condition | humans_defended_opp3 − humans_finetuned_clean_opp3 | 1 | +4.1 | 2.0e-05 | 0.0037 | yes |
| 52 | addendum | humans_finetuned_clean − humans_zoo1_original | 0.25 | +7.9 | 4.9e-05 | 0.0091 | yes |
| 53 | within condition | humans_defended_opp2 − humans_finetuned_clean_opp2 | 0 | -6.9 | 6.0e-05 | 0.011 | yes |
| 54 | addendum | ants_finetuned_clean − ants_zoo1_original | 0.1 | -8.6 | 8.8e-05 | 0.016 | yes |
| 55 | vs. committed | humans_finetuned_clean_opp2 − humans_finetuned_clean | 1 | -4.4 | 1.0e-04 | 0.019 | yes |
| 56 | addendum | ants_defended − ants_zoo1_original | 0 | -8.4 | 1.4e-04 | 0.026 | yes |
| 57 | addendum | humans_finetuned_clean − humans_zoo1_original | 0.5 | +8.5 | 1.4e-04 | 0.026 | yes |
| 58 | within condition | humans_defended_seed2 − humans_finetuned_clean_seed2 | 1 | +5.8 | 2.6e-04 | 0.047 | yes |
| 59 | vs. committed | ants_defended_opp2 − ants_defended | 0.25 | +7.3 | 8.0e-04 | 0.14 | no |
| 60 | vs. committed | humans_defended_20M − humans_defended | 0.05 | +5.7 | 8.3e-04 | 0.15 | no |
| 61 | addendum | ants_finetuned_clean − ants_zoo1_original | 0.25 | -7.2 | 9.4e-04 | 0.17 | no |
| 62 | addendum | ants_defended − ants_zoo1_original | 0.25 | -7.1 | 0.0011 | 0.2 | no |
| 63 | vs. committed | humans_defended_20M − humans_defended | 0 | +5.3 | 0.0012 | 0.21 | no |
| 64 | addendum | humans_defended − humans_zoo1_original | 0.25 | +6.2 | 0.0016 | 0.28 | no |
| 65 | within condition | ants_defended_20M − ants_finetuned_clean_20M | 0 | -6.8 | 0.0018 | 0.31 | no |
| 66 | vs. committed | ants_zoo1_original_opp3 − ants_zoo1_original | 1 | +5.2 | 0.0024 | 0.41 | no |
| 67 | vs. committed | humans_finetuned_clean_20M − humans_finetuned_clean | 0.25 | +5.3 | 0.0024 | 0.41 | no |
| 68 | within condition | humans_defended_seed2 − humans_finetuned_clean_seed2 | 0.1 | -5.1 | 0.003 | 0.51 | no |
| 69 | within condition | humans_defended_seed2 − humans_finetuned_clean_seed2 | 0.5 | +6.6 | 0.003 | 0.51 | no |
| 70 | within condition | humans_defended_opp2 − humans_finetuned_clean_opp2 | 0.25 | -4.7 | 0.0038 | 0.65 | no |
| 71 | within condition | ants_defended_opp3 − ants_finetuned_clean_opp3 | 1 | +4.8 | 0.0038 | 0.65 | no |
| 72 | within condition | ants_defended_20M − ants_finetuned_clean_20M | 0.1 | -6.3 | 0.0039 | 0.65 | no |
| 73 | vs. committed | ants_zoo1_original_opp2 − ants_zoo1_original | 0.05 | +6.4 | 0.004 | 0.67 | no |
| 74 | addendum | humans_defended − humans_finetuned_clean | 0 | -4.7 | 0.0043 | 0.71 | no |
| 75 | vs. committed | ants_finetuned_clean_20M − ants_finetuned_clean | 0.1 | +6.2 | 0.0045 | 0.73 | no |
| 76 | vs. committed | ants_defended_opp2 − ants_defended | 0.1 | +5.9 | 0.0064 | 1 | no |
| 77 | within condition | humans_defended_20M − humans_finetuned_clean_20M | 0.1 | -4.2 | 0.007 | 1 | no |
| 78 | addendum | humans_defended − humans_finetuned_clean | 0.05 | -4.6 | 0.0077 | 1 | no |
| 79 | addendum | ants_finetuned_clean − ants_zoo1_original | 0.05 | -5.8 | 0.0079 | 1 | no |
| 80 | within condition | ants_defended_20M − ants_finetuned_clean_20M | 0.25 | -5.7 | 0.0081 | 1 | no |
| 81 | within condition | ants_defended_20M − ants_finetuned_clean_20M | 0.05 | -5.7 | 0.0083 | 1 | no |
| 82 | vs. committed | ants_finetuned_clean_20M − ants_finetuned_clean | 0 | +5.7 | 0.0091 | 1 | no |
| 83 | addendum | ants_finetuned_clean − ants_zoo1_original | 0.5 | -5.2 | 0.012 | 1 | no |
| 84 | vs. committed | humans_defended_20M − humans_defended | 0.25 | +4.5 | 0.013 | 1 | no |
| 85 | addendum | humans_finetuned_clean − humans_zoo1_original | 0.1 | +4.2 | 0.014 | 1 | no |
| 86 | within condition | ants_defended_opp3 − ants_finetuned_clean_opp3 | 0.1 | +5.3 | 0.014 | 1 | no |
| 87 | vs. committed | ants_defended_opp2 − ants_defended | 0 | +5.3 | 0.016 | 1 | no |
| 88 | within condition | humans_defended_seed2 − humans_finetuned_clean_seed2 | 0.05 | -4.2 | 0.016 | 1 | no |
| 89 | vs. committed | ants_defended_opp3 − ants_defended | 0.1 | +5.2 | 0.016 | 1 | no |
| 90 | addendum | humans_defended − humans_finetuned_clean | 0.5 | +5.3 | 0.017 | 1 | no |
| 91 | addendum | humans_finetuned_clean − humans_zoo1_original | 0 | +3.9 | 0.017 | 1 | no |
| 92 | within condition | ants_defended_opp2 − ants_finetuned_clean_opp2 | 0.25 | +5.2 | 0.017 | 1 | no |
| 93 | vs. committed | humans_finetuned_clean_20M − humans_finetuned_clean | 0.1 | +3.6 | 0.02 | 1 | no |
| 94 | vs. committed | humans_finetuned_clean_20M − humans_finetuned_clean | 0 | +3.4 | 0.02 | 1 | no |
| 95 | vs. committed | ants_defended_opp3 − ants_defended | 0.5 | +4.8 | 0.022 | 1 | no |
| 96 | within condition | humans_defended_opp2 − humans_finetuned_clean_opp2 | 0.5 | -3.4 | 0.022 | 1 | no |
| 97 | vs. committed | humans_finetuned_clean_20M − humans_finetuned_clean | 0.05 | +3.5 | 0.025 | 1 | no |
| 98 | within condition | ants_defended_opp2 − ants_finetuned_clean_opp2 | 0.1 | +4.8 | 0.027 | 1 | no |
| 99 | vs. committed | ants_defended_20M − ants_defended | 0.05 | -4.7 | 0.029 | 1 | no |
| 100 | vs. committed | ants_defended_opp2 − ants_defended | 0.5 | +4.5 | 0.032 | 1 | no |
| 101 | within condition | ants_defended_opp2 − ants_finetuned_clean_opp2 | 0.5 | +4.4 | 0.036 | 1 | no |
| 102 | within condition | ants_defended_opp3 − ants_finetuned_clean_opp3 | 0.5 | +4.3 | 0.041 | 1 | no |
| 103 | vs. committed | humans_finetuned_clean_seed2 − humans_finetuned_clean | 1 | +2.8 | 0.041 | 1 | no |
| 104 | vs. committed | humans_finetuned_clean_seed2 − humans_finetuned_clean | 0 | -3.3 | 0.041 | 1 | no |
| 105 | vs. committed | ants_finetuned_clean_20M − ants_finetuned_clean | 0.5 | +4.2 | 0.042 | 1 | no |
| 106 | vs. committed | ants_finetuned_clean_20M − ants_finetuned_clean | 0.25 | +4.3 | 0.046 | 1 | no |
| 107 | within condition | humans_defended_opp3 − humans_finetuned_clean_opp3 | 0.5 | -2.9 | 0.047 | 1 | no |
| 108 | within condition | humans_defended_20M − humans_finetuned_clean_20M | 0 | -2.8 | 0.052 | 1 | no |
| 109 | within condition | ants_defended_opp3 − ants_finetuned_clean_opp3 | 0 | +4.0 | 0.066 | 1 | no |
| 110 | vs. committed | ants_defended_opp2 − ants_defended | 0.05 | +4.0 | 0.069 | 1 | no |
| 111 | vs. committed | ants_finetuned_clean_opp3 − ants_finetuned_clean | 1 | -2.9 | 0.074 | 1 | no |
| 112 | within condition | ants_defended_opp3 − ants_finetuned_clean_opp3 | 0.05 | +3.9 | 0.074 | 1 | no |
| 113 | vs. committed | ants_defended_opp3 − ants_defended | 1 | +3.0 | 0.077 | 1 | no |
| 114 | within condition | ants_defended_seed2 − ants_finetuned_clean_seed2 | 0.25 | -3.7 | 0.084 | 1 | no |
| 115 | addendum | ants_defended − ants_zoo1_original | 0.5 | -3.6 | 0.084 | 1 | no |
| 116 | addendum | humans_defended − humans_finetuned_clean | 0.1 | -2.9 | 0.085 | 1 | no |
| 117 | addendum | humans_finetuned_clean − humans_zoo1_original | 0.05 | +2.9 | 0.087 | 1 | no |
| 118 | within condition | humans_defended_opp3 − humans_finetuned_clean_opp3 | 2 | +0.7 | 0.088 | 1 | no |
| 119 | vs. committed | ants_finetuned_clean_20M − ants_finetuned_clean | 0.05 | +3.7 | 0.089 | 1 | no |
| 120 | within condition | ants_defended_seed2 − ants_finetuned_clean_seed2 | 0.1 | -3.6 | 0.094 | 1 | no |
| 121 | vs. committed | ants_finetuned_clean_opp2 − ants_finetuned_clean | 0 | -3.5 | 0.1 | 1 | no |
| 122 | within condition | humans_defended_20M − humans_finetuned_clean_20M | 2 | +0.6 | 0.11 | 1 | no |
| 123 | within condition | humans_defended_20M − humans_finetuned_clean_20M | 0.05 | -2.4 | 0.12 | 1 | no |
| 124 | vs. committed | ants_defended_seed2 − ants_defended | 0.05 | -3.2 | 0.14 | 1 | no |
| 125 | within condition | humans_defended_20M − humans_finetuned_clean_20M | 0.25 | -2.5 | 0.14 | 1 | no |
| 126 | addendum | ants_defended − ants_zoo1_original | 2 | -1.7 | 0.14 | 1 | no |
| 127 | vs. committed | ants_finetuned_clean_opp2 − ants_finetuned_clean | 2 | -1.6 | 0.15 | 1 | no |
| 128 | vs. committed | humans_defended_20M − humans_defended | 0.5 | +3.1 | 0.15 | 1 | no |
| 129 | vs. committed | humans_defended_opp3 − humans_defended | 2 | +0.6 | 0.16 | 1 | no |
| 130 | addendum | ants_defended − ants_zoo1_original | 0.05 | -3.1 | 0.16 | 1 | no |
| 131 | within condition | ants_defended_20M − ants_finetuned_clean_20M | 0.5 | -2.9 | 0.16 | 1 | no |
| 132 | vs. committed | ants_defended_opp3 − ants_defended | 0.25 | +3.0 | 0.16 | 1 | no |
| 133 | vs. committed | humans_finetuned_clean_seed2 − humans_finetuned_clean | 0.25 | -2.6 | 0.17 | 1 | no |
| 134 | within condition | ants_defended_seed2 − ants_finetuned_clean_seed2 | 0 | -3.0 | 0.17 | 1 | no |
| 135 | vs. committed | ants_zoo1_original_opp2 − ants_zoo1_original | 2 | -1.6 | 0.17 | 1 | no |
| 136 | vs. committed | ants_finetuned_clean_opp2 − ants_finetuned_clean | 0.05 | -2.9 | 0.17 | 1 | no |
| 137 | vs. committed | humans_defended_20M − humans_defended | 0.1 | +2.3 | 0.17 | 1 | no |
| 138 | vs. committed | ants_zoo1_original_opp3 − ants_zoo1_original | 0.25 | +3.0 | 0.18 | 1 | no |
| 139 | vs. committed | ants_finetuned_clean_opp2 − ants_finetuned_clean | 1 | -2.2 | 0.18 | 1 | no |
| 140 | vs. committed | ants_defended_opp3 − ants_defended | 0 | +2.9 | 0.18 | 1 | no |
| 141 | vs. committed | humans_finetuned_clean_seed2 − humans_finetuned_clean | 0.5 | -2.9 | 0.19 | 1 | no |
| 142 | addendum | ants_defended − ants_finetuned_clean | 0.05 | +2.7 | 0.21 | 1 | no |
| 143 | vs. committed | ants_finetuned_clean_seed2 − ants_finetuned_clean | 0.25 | +2.6 | 0.23 | 1 | no |
| 144 | vs. committed | ants_finetuned_clean_seed2 − ants_finetuned_clean | 0.5 | +2.4 | 0.24 | 1 | no |
| 145 | vs. committed | ants_defended_20M − ants_defended | 0 | -2.5 | 0.25 | 1 | no |
| 146 | vs. committed | ants_finetuned_clean_seed2 − ants_finetuned_clean | 0 | +2.5 | 0.25 | 1 | no |
| 147 | vs. committed | ants_defended_opp3 − ants_defended | 2 | +1.3 | 0.26 | 1 | no |
| 148 | vs. committed | ants_defended_seed2 − ants_defended | 2 | +1.3 | 0.26 | 1 | no |
| 149 | vs. committed | humans_defended_seed2 − humans_defended | 0.1 | -2.0 | 0.26 | 1 | no |
| 150 | addendum | ants_finetuned_clean − ants_zoo1_original | 1 | +1.8 | 0.27 | 1 | no |
| 151 | vs. committed | humans_defended_seed2 − humans_defended | 1 | -1.9 | 0.28 | 1 | no |
| 152 | vs. committed | ants_zoo1_original_opp2 − ants_zoo1_original | 0.5 | +2.2 | 0.3 | 1 | no |
| 153 | vs. committed | ants_finetuned_clean_opp3 − ants_finetuned_clean | 0.5 | +2.1 | 0.3 | 1 | no |
| 154 | vs. committed | ants_finetuned_clean_opp2 − ants_finetuned_clean | 0.25 | +2.2 | 0.31 | 1 | no |
| 155 | vs. committed | humans_defended_20M − humans_defended | 2 | +0.4 | 0.32 | 1 | no |
| 156 | within condition | ants_defended_opp3 − ants_finetuned_clean_opp3 | 0.25 | +2.1 | 0.33 | 1 | no |
| 157 | addendum | ants_defended − ants_finetuned_clean | 2 | -1.1 | 0.33 | 1 | no |
| 158 | vs. committed | ants_defended_opp3 − ants_defended | 0.05 | +2.1 | 0.34 | 1 | no |
| 159 | addendum | humans_defended − humans_zoo1_original | 0.05 | -1.7 | 0.34 | 1 | no |
| 160 | addendum | humans_defended − humans_finetuned_clean | 0.25 | -1.7 | 0.36 | 1 | no |
| 161 | vs. committed | humans_finetuned_clean_opp2 − humans_finetuned_clean | 2 | +0.3 | 0.36 | 1 | no |
| 162 | vs. committed | humans_finetuned_clean_seed2 − humans_finetuned_clean | 2 | +0.3 | 0.36 | 1 | no |
| 163 | within condition | humans_defended_seed2 − humans_finetuned_clean_seed2 | 2 | -0.3 | 0.36 | 1 | no |
| 164 | within condition | ants_defended_seed2 − ants_finetuned_clean_seed2 | 0.05 | -1.9 | 0.38 | 1 | no |
| 165 | vs. committed | ants_defended_seed2 − ants_defended | 0 | -1.9 | 0.38 | 1 | no |
| 166 | vs. committed | ants_finetuned_clean_seed2 − ants_finetuned_clean | 0.1 | +1.9 | 0.38 | 1 | no |
| 167 | vs. committed | ants_zoo1_original_opp3 − ants_zoo1_original | 0 | -1.9 | 0.39 | 1 | no |
| 168 | within condition | ants_defended_seed2 − ants_finetuned_clean_seed2 | 1 | -1.4 | 0.4 | 1 | no |
| 169 | vs. committed | ants_finetuned_clean_opp2 − ants_finetuned_clean | 0.5 | +1.7 | 0.4 | 1 | no |
| 170 | addendum | ants_defended − ants_finetuned_clean | 0.5 | +1.6 | 0.43 | 1 | no |
| 171 | vs. committed | humans_defended_opp2 − humans_defended | 2 | +0.3 | 0.44 | 1 | no |
| 172 | vs. committed | ants_finetuned_clean_opp3 − ants_finetuned_clean | 0.1 | -1.6 | 0.45 | 1 | no |
| 173 | vs. committed | ants_finetuned_clean_opp3 − ants_finetuned_clean | 2 | +0.9 | 0.46 | 1 | no |
| 174 | addendum | humans_defended − humans_zoo1_original | 0.1 | +1.3 | 0.46 | 1 | no |
| 175 | vs. committed | humans_defended_seed2 − humans_defended | 0.25 | -1.4 | 0.46 | 1 | no |
| 176 | vs. committed | humans_defended_seed2 − humans_defended | 0.5 | -1.6 | 0.47 | 1 | no |
| 177 | vs. committed | ants_zoo1_original_opp2 − ants_zoo1_original | 0 | +1.6 | 0.47 | 1 | no |
| 178 | vs. committed | ants_defended_20M − ants_defended | 2 | +0.8 | 0.48 | 1 | no |
| 179 | vs. committed | humans_zoo1_original_opp3 − humans_zoo1_original | 2 | -0.2 | 0.48 | 1 | no |
| 180 | vs. committed | ants_defended_20M − ants_defended | 0.25 | -1.5 | 0.48 | 1 | no |
| 181 | addendum | ants_defended − ants_finetuned_clean | 0.1 | -1.5 | 0.48 | 1 | no |
| 182 | vs. committed | ants_defended_opp2 − ants_defended | 1 | -1.1 | 0.5 | 1 | no |
| 183 | vs. committed | ants_zoo1_original_opp3 − ants_zoo1_original | 0.05 | +1.5 | 0.5 | 1 | no |
| 184 | vs. committed | humans_finetuned_clean_20M − humans_finetuned_clean | 0.5 | -1.5 | 0.5 | 1 | no |
| 185 | addendum | ants_defended − ants_finetuned_clean | 1 | -1.1 | 0.51 | 1 | no |
| 186 | vs. committed | ants_defended_20M − ants_defended | 1 | +1.1 | 0.51 | 1 | no |
| 187 | vs. committed | ants_defended_20M − ants_defended | 0.1 | +1.4 | 0.51 | 1 | no |
| 188 | vs. committed | ants_finetuned_clean_seed2 − ants_finetuned_clean | 0.05 | +1.4 | 0.52 | 1 | no |
| 189 | addendum | ants_defended − ants_finetuned_clean | 0 | +1.4 | 0.52 | 1 | no |
| 190 | within condition | humans_defended_seed2 − humans_finetuned_clean_seed2 | 0 | -1.1 | 0.52 | 1 | no |
| 191 | addendum | humans_defended − humans_finetuned_clean | 2 | +0.2 | 0.53 | 1 | no |
| 192 | vs. committed | humans_defended_seed2 − humans_defended | 2 | -0.2 | 0.53 | 1 | no |
| 193 | addendum | humans_finetuned_clean − humans_zoo1_original | 1 | -0.8 | 0.54 | 1 | no |
| 194 | within condition | ants_defended_opp3 − ants_finetuned_clean_opp3 | 2 | -0.7 | 0.57 | 1 | no |
| 195 | vs. committed | ants_zoo1_original_opp3 − ants_zoo1_original | 0.5 | +1.2 | 0.57 | 1 | no |
| 196 | vs. committed | ants_defended_seed2 − ants_defended | 0.25 | -1.2 | 0.57 | 1 | no |
| 197 | within condition | humans_defended_opp2 − humans_finetuned_clean_opp2 | 2 | +0.2 | 0.62 | 1 | no |
| 198 | addendum | ants_finetuned_clean − ants_zoo1_original | 2 | -0.6 | 0.62 | 1 | no |
| 199 | vs. committed | ants_finetuned_clean_opp3 − ants_finetuned_clean | 0.25 | +1.0 | 0.64 | 1 | no |
| 200 | addendum | humans_defended − humans_zoo1_original | 0 | -0.8 | 0.64 | 1 | no |
| 201 | addendum | ants_defended − ants_zoo1_original | 1 | +0.7 | 0.67 | 1 | no |
| 202 | within condition | humans_defended_opp2 − humans_finetuned_clean_opp2 | 1 | -0.4 | 0.67 | 1 | no |
| 203 | vs. committed | ants_finetuned_clean_opp3 − ants_finetuned_clean | 0.05 | +0.9 | 0.68 | 1 | no |
| 204 | vs. committed | ants_zoo1_original_opp2 − ants_zoo1_original | 0.25 | +0.9 | 0.68 | 1 | no |
| 205 | vs. committed | humans_defended_20M − humans_defended | 1 | +0.7 | 0.7 | 1 | no |
| 206 | vs. committed | humans_finetuned_clean_seed2 − humans_finetuned_clean | 0.05 | -0.6 | 0.72 | 1 | no |
| 207 | within condition | ants_defended_seed2 − ants_finetuned_clean_seed2 | 0.5 | -0.7 | 0.73 | 1 | no |
| 208 | addendum | humans_finetuned_clean − humans_zoo1_original | 2 | -0.1 | 0.74 | 1 | no |
| 209 | vs. committed | humans_zoo1_original_opp2 − humans_zoo1_original | 2 | -0.1 | 0.74 | 1 | no |
| 210 | vs. committed | humans_finetuned_clean_opp3 − humans_finetuned_clean | 2 | +0.1 | 0.74 | 1 | no |
| 211 | vs. committed | ants_zoo1_original_opp3 − ants_zoo1_original | 2 | +0.4 | 0.75 | 1 | no |
| 212 | vs. committed | ants_zoo1_original_opp2 − ants_zoo1_original | 0.1 | +0.7 | 0.75 | 1 | no |
| 213 | vs. committed | ants_zoo1_original_opp2 − ants_zoo1_original | 1 | +0.5 | 0.76 | 1 | no |
| 214 | vs. committed | humans_finetuned_clean_20M − humans_finetuned_clean | 1 | +0.4 | 0.76 | 1 | no |
| 215 | addendum | humans_defended − humans_zoo1_original | 2 | +0.1 | 0.76 | 1 | no |
| 216 | vs. committed | ants_defended_opp2 − ants_defended | 2 | -0.3 | 0.78 | 1 | no |
| 217 | vs. committed | ants_zoo1_original_opp3 − ants_zoo1_original | 0.1 | +0.6 | 0.79 | 1 | no |
| 218 | within condition | humans_defended_seed2 − humans_finetuned_clean_seed2 | 0.25 | -0.5 | 0.79 | 1 | no |
| 219 | within condition | ants_defended_opp2 − ants_finetuned_clean_opp2 | 2 | +0.2 | 0.85 | 1 | no |
| 220 | vs. committed | ants_finetuned_clean_opp2 − ants_finetuned_clean | 0.1 | -0.4 | 0.85 | 1 | no |
| 221 | vs. committed | ants_finetuned_clean_seed2 − ants_finetuned_clean | 1 | +0.3 | 0.86 | 1 | no |
| 222 | vs. committed | humans_defended_seed2 − humans_defended | 0 | +0.3 | 0.86 | 1 | no |
| 223 | vs. committed | ants_finetuned_clean_20M − ants_finetuned_clean | 2 | -0.2 | 0.86 | 1 | no |
| 224 | vs. committed | ants_finetuned_clean_seed2 − ants_finetuned_clean | 2 | +0.2 | 0.87 | 1 | no |
| 225 | vs. committed | ants_defended_20M − ants_defended | 0.5 | -0.3 | 0.88 | 1 | no |
| 226 | vs. committed | ants_finetuned_clean_opp3 − ants_finetuned_clean | 0 | +0.3 | 0.89 | 1 | no |
| 227 | vs. committed | humans_finetuned_clean_seed2 − humans_finetuned_clean | 0.1 | +0.2 | 0.9 | 1 | no |
| 228 | vs. committed | humans_defended_seed2 − humans_defended | 0.05 | -0.2 | 0.91 | 1 | no |
| 229 | vs. committed | ants_defended_seed2 − ants_defended | 0.1 | -0.2 | 0.92 | 1 | no |
| 230 | within condition | ants_defended_20M − ants_finetuned_clean_20M | 2 | -0.1 | 0.93 | 1 | no |
| 231 | vs. committed | ants_finetuned_clean_20M − ants_finetuned_clean | 1 | +0.1 | 0.95 | 1 | no |
| 232 | within condition | ants_defended_20M − ants_finetuned_clean_20M | 1 | -0.1 | 0.95 | 1 | no |
| 233 | vs. committed | ants_defended_seed2 − ants_defended | 0.5 | +0.1 | 0.96 | 1 | no |
| 234 | addendum | ants_defended − ants_finetuned_clean | 0.25 | +0.1 | 0.96 | 1 | no |
| 235 | vs. committed | ants_defended_seed2 − ants_defended | 1 | +0.0 | 1 | 1 | no |
| 236 | vs. committed | humans_finetuned_clean_20M − humans_finetuned_clean | 2 | +0.0 | 1 | 1 | no |
| 237 | within condition | ants_defended_opp2 − ants_finetuned_clean_opp2 | 1 | +0.0 | 1 | 1 | no |
| 238 | within condition | ants_defended_seed2 − ants_finetuned_clean_seed2 | 2 | +0.0 | 1 | 1 | no |

## Provenance

| run | trained (steps, hours, git_dirty) | swept at commit | sweep git_dirty |
|---|---|---|---|
| ants_defended_opp2 | 9,994,240, 1.2 h, false | 3452ab8 | true |
| ants_finetuned_clean_opp2 | 9,994,240, 1.2 h, false | 3452ab8 | true |
| ants_zoo1_original_opp2 |  | 3452ab8 | true |
| humans_defended_opp2 | 9,994,240, 1.4 h, false | 3452ab8 | true |
| humans_finetuned_clean_opp2 | 9,994,240, 1.4 h, false | 3452ab8 | true |
| humans_zoo1_original_opp2 |  | 3452ab8 | true |
| ants_defended_opp3 | 9,994,240, 1.2 h, false | 3452ab8 | true |
| ants_finetuned_clean_opp3 | 9,994,240, 1.2 h, false | 3452ab8 | true |
| ants_zoo1_original_opp3 |  | 3452ab8 | true |
| humans_defended_opp3 | 9,994,240, 1.4 h, false | 3452ab8 | true |
| humans_finetuned_clean_opp3 | 9,994,240, 1.4 h, false | 3452ab8 | true |
| humans_zoo1_original_opp3 |  | 3452ab8 | true |
| ants_defended_seed2 | 9,994,240, 1.1 h, true | 3452ab8 | true |
| ants_finetuned_clean_seed2 | 9,994,240, 1.1 h, true | 3452ab8 | true |
| humans_defended_seed2 | 9,994,240, 1.4 h, true | 3452ab8 | true |
| humans_finetuned_clean_seed2 | 9,994,240, 1.4 h, true | 3452ab8 | true |
| ants_defended_20M | 19,988,480, 2.3 h, true | 3452ab8 | true |
| ants_finetuned_clean_20M | 19,988,480, 2.3 h, true | 3452ab8 | true |
| humans_defended_20M | 19,988,480, 2.8 h, true | 3452ab8 | true |
| humans_finetuned_clean_20M | 19,988,480, 2.8 h, true | 3452ab8 | true |
