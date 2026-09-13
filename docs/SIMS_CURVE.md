# The simulation curve: where more search stops paying

`docs/ABLATION_RESULTS.md` left the sims question open. At a matched evaluation
budget it measured 64 → 160 → 320 simulations at 0.4531 / 0.4865 / 0.5171 and
noted that the two increments were the same size (+23.3 and +21.3 Elo), so the
curve had not flattened and "whatever the real knee is, it is above 320". Its
closing recommendation was the experiment run here:
`--factor sims --levels 320,640,1280 --seeds 5`.

**The answer is not above 320. There is no knee above 320, because the curve is
already falling at 320.** At equal compute on the current binary, strength
decreases monotonically across the whole range asked about, and a second
experiment shows it stops declining below 160:

| sims | generations | external score | vs 160 | |
| ---: | ---: | ---: | ---: | --- |
| 80 | 128 | 0.6240 | +12.0 Elo | indistinguishable (perm p 0.29) |
| **160** | 64 | **0.6072 / 0.6076** | — | measured twice, independently |
| 320 | 32 | 0.5466 | **−43.2 Elo** | |
| 640 | 16 | 0.5243 | −58.7 Elo | |
| 1280 | 8 | 0.5182 | −63.0 Elo | |
| 2560 | 4 | 0.4934 | −80.2 Elo | |

Across 160–2560: spread 0.1137 score = **79.5 Elo = 4.63× the pooled run-to-run
SD**; omnibus permutation F = 15.377, **p < 1e-4**; a log-linear trend of
**−17.9 Elo per doubling of simulations** with permutation p = 1e-5. Across
80–160: nothing, 0.74× the seed SD. **The curve is a plateau with a cliff on the
high side, and 160 sits at the top edge of the plateau.**

The previous experiment's ordering is reversed, and the reversal has a single,
identified cause.

**The cause is one line of `src/mcts.c`.** Commit `9f6936d` is the only commit
touching `src/` between the two experiments, it changes only `mcts.c`, and of
its 9 insertions and 6 deletions exactly one is code:

```c
     int32_t nsum = nd->N - 1;                   /* sum_b N(s,b) */
     if (nsum < 0) nsum = 0;
-    const float sq  = sqrtf((float)nsum);
+    const float sq  = sqrtf((float)nsum + 1.0f);
```

With `sqrt(nsum)` the PUCT exploration term is exactly 0 on a node's first
descent, so the node takes whatever `gen_legal()` listed first and **the policy
is never consulted**; the commit message records that this wasted the only visit
22–42% of expanded nodes ever received. AlphaZero's own pseudocode uses
`sqrt(sum + 1)`, which keeps the priors in play from the very first simulation.

Re-running the identical plan on the fixed binary is worth **+84.6 Elo at 160
simulations** and **+20.5 Elo at 320** — four times as much at the smaller
budget, exactly as the mechanism predicts, because a wasted first visit is a
larger fraction of a smaller search.

| sims | previous binary (`95676a9`) | this binary (`9f6936d`) | gain |
| ---: | ---: | ---: | ---: |
| 160 | 0.4865 ± 0.0058 | **0.6072** ± 0.0364 | **+0.1207 = +84.6 Elo** (Welch p 0.0015, exact perm p 0.0079) |
| 320 | 0.5171 ± 0.0157 | 0.5466 ± 0.0309 | +0.0295 = +20.5 Elo (Welch p 0.1067, perm p 0.0794) |

Every conclusion in `docs/ABLATION_RESULTS.md` §3 about simulation count was
measured on a search that threw away its first simulation at every node. Those
numbers are not wrong about the binary they were run on; they no longer describe
the binary in the tree.

One caveat on that attribution, stated because it cannot be verified after the
fact: `docs/ABLATION_RESULTS.md` records that its binary was built at 20:12 on
2026-09-12 from `e84c8c6` **plus uncommitted changes**, and that those changes
had become `95676a9` by the time it was written — `95676a9` was committed at
20:18 and its runs started at 21:43. The `src/`-only diff from `95676a9` to
`9f6936d` is the one line above. The identification therefore rests on that
provenance note being accurate about an uncommitted tree; if the 20:12 build
differed from `95676a9` in some other way, part of the +84.6 Elo belongs
elsewhere. Everything else in this file is independent of that question.

---

## 1. Method

Exactly `py/ablate.py` as it stands at `HEAD` — nothing in `src/` was touched,
and no fix to the harness was needed or made. One modelling error in the harness
was found and is documented in §2.2 rather than patched, because it does not
affect any comparison.

**Compute matching.** `sims` changes the cost of a game, so the harness matches
on **total network evaluations**, not games: the generation count is derived from
a fixed evaluation budget of **117,637,120 planned evaluations per run**, which
is byte-for-byte the same budget the previous sims experiment used, so its 160
and 320 cells are directly comparable to these. Doubling `sims` therefore halves
the generations, the games and the optimiser steps.

**External evaluation.** Internal Elo is anchored to each run's own hall of fame
and is not comparable across runs, so it is ignored. Every finished
`checkpoint.crl` — the model after the full budget, not the internally
early-stopped `best.crl` — plays **2,000 games as its raw policy head (argmax,
no search) against `random`**, colour-reversed pairs from a fixed opening set,
under one evaluation seed (20260911) common to every model, so the comparison is
paired and the opening-set component of the error cancels. Evaluation SE per
match is **0.0035, 14% of the seed SD** — the variance being tested is training
variance, not measurement noise.

**Seeds.** Five independent training seeds (1001–1005) per level, run seed-major.
Training is not reproducible at a fixed seed — threaded gradient reduction
reorders floating-point sums — so a "seed" is a replicate and its SD is the full
run-to-run variance.

```sh
# the command that produced everything below
python3 py/ablate.py --factor sims --levels 160,320,640,1280,2560 --seeds 5 \
        --budget-games 6144 --threads 8 --eval-games 2000 \
        --out runs/ablation_sims_curve.json
```

(Run from a snapshot copy of `build/`, `py/` and `src/` in a temp directory, so
that a concurrent relink of `build/chessrl` by another agent could not kill a
training run mid-flight. `--budget-games 6144` only *sets* the evaluation
budget; the games each level actually plays are derived from it.)

### 1.1 The machine was not idle, and this time it was severe

The machine **was** idle at launch — 01:13, one-minute load average 2.09, `ps`
showing nothing but the desktop UI, and a solo calibration of this exact binary
minutes earlier reaching 2.15–2.23M evaluations/second. It stopped being idle
about seven minutes in, when two sibling agents started their own 8-thread
`chessrl` training jobs and a third started a diagnostic. For most of the
experiment four 8-thread jobs were competing for 8 cores.

Load average, sampled every 20–30 s for the duration (n = 530):

| min | p25 | median | p75 | max |
| ---: | ---: | ---: | ---: | ---: |
| 4.7 | 33.8 | **99.2** | 151.2 | **224.7** |

The experiment realised 3.119e9 network evaluations in 6,061 s of training —
**514,549 evaluations/second aggregate, a 4.2× slowdown** against the 2.17M/s
measured solo on the same binary immediately before launch. Planned wall clock
was 35m28s; actual was **113 minutes**.

**Why this does not invalidate the comparison, and what it does cost.**

* The budget is matched on **evaluations, not wall clock**. A slow run is not a
  smaller run: every run at a level performed the same work. Per-run training
  time within one level ranged 134 s to 410 s purely from contention.
* Runs are interleaved **seed-major**, so every level appears once in every
  contention regime rather than one level absorbing it.
* The extra scheduling non-determinism reorders the gradient reduction, which
  *is* the run-to-run variance this design tests against. It inflates the
  yardstick and makes the test conservative.
* What it does corrupt is **throughput per level**, so the per-level
  evaluations/second in §2.2 are reported with that warning and no conclusion
  rests on them.

---

## 2. What each level actually did

### 2.1 The plan

| sims | gens | games/gen | planned games | steps/gen | total steps | buffer spans | hof snapshots |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 160 | 64 | 96 | 6,144 | 200 | 12,800 | 4.4 gens | 7 |
| 320 | 32 | 96 | 3,072 | 200 | 6,400 | 4.4 gens | 4 |
| 640 | 16 | 96 | 1,536 | 200 | 3,200 | 4.4 gens | 2 |
| 1280 | 8 | 96 | 768 | 200 | 1,600 | 4.4 gens | 1 |
| 2560 | 4 | 96 | 384 | 200 | 800 | 4.4 gens | 1 |

### 2.2 What it cost in reality

Planned budgets are not realised budgets: `ablate.py` derives generations from a
single assumed plies-per-game constant (119.7, the median of this machine's past
telemetry), and real games get longer as the search deepens. Read from each
run's own `telemetry.jsonl`, means over five seeds:

| sims | self-play games | anchor games | positions | optimiser steps | positions/step | realised evals | vs plan | train s | evals/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 160 | 4,374 | 1,770 | 663,096 | 12,800 | 51.8 | 124,843,855 | 1.06× | 267.0 | 726,122 |
| 320 | 2,166 | 906 | 338,495 | 6,400 | 52.9 | 131,030,332 | 1.11× | 243.7 | 781,132 |
| 640 | 1,062 | 474 | 173,337 | 3,200 | 54.2 | 131,907,197 | 1.12× | 239.9 | 700,492 |
| 1280 | 510 | 258 | 85,034 | 1,600 | 53.2 | 124,972,413 | 1.06× | 247.3 | 602,304 |
| 2560 | 234 | 150 | 38,594 | 800 | 48.2 | 111,024,583 | 0.94× | 214.3 | 602,578 |

Four things in that table matter and are not in the plan.

1. **Evaluation matching held to ±9%.** Realised evaluations span 111.0M to
   131.9M, 0.94× to 1.12× of plan. The level that scored *worst* (2560) received
   the *least* compute, by 13% against 640 — real, and far too small to produce
   an 80-Elo gap, but it is a mild bias in the direction of the result and is
   stated rather than hidden.
2. **Wall clock roughly followed, but this table cannot prove it.** Mean training
   time per run runs 214–267 s with no ordering that tracks the score, and the
   *winning* level is the slowest — which, if taken at face value, would mean an
   equal-wall-clock comparison is slightly *harsher* on 160 than this one. Under
   a load average that ranged 4.7 to 224.7 these numbers are not trustworthy at
   that resolution, so nothing rests on them; §7.2 measures throughput properly
   on a quiet machine instead.
3. **The sample-reuse ratio is matched.** Positions per optimiser step is
   48–54 at every level, so with `--batch 256` each position is drawn about 4.9
   times at every level. The levels differ in how much data they see, not in how
   hard they squeeze it.
4. **`ablate.py`'s games-per-generation model is wrong, harmlessly.** It models
   `agents × games-per-agent / 2` = 96 games per generation; `src/az.c` plays 96
   *pairings*, of which 27 are rating-anchor games against `random` and `gen0`
   (48/48 in the first two generations), leaving 69 self-play games that enter
   the replay buffer. The planned "total games" column therefore overstates
   training data by about 30%. It does not affect the comparison — the same
   model is applied at every level, so it cancels out of the generation
   derivation — but it does mean the anchor share is **not** constant: it runs
   28.8% of pairings at 160 sims up to **39.1% at 2560**, because the
   anchor-heavy first two generations are half of a four-generation run. High
   simulation counts spend a larger slice of their budget on the ruler.

---

## 3. Results

### 3.1 Raw numbers

Score is out of 2,000 games vs `random`, policy head only, identical openings
for every model.

| sims | seed | score | W | D | L | train |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 160 | 1001 | 0.5883 | 365 | 1623 | 12 | 2m14s |
| 160 | 1002 | 0.5680 | 274 | 1724 | 2 | 6m50s |
| 160 | 1003 | 0.5970 | 389 | 1610 | 1 | 6m46s |
| 160 | 1004 | 0.6190 | 494 | 1488 | 18 | 3m13s |
| 160 | 1005 | 0.6635 | 654 | 1346 | 0 | 3m12s |
| 320 | 1001 | 0.5580 | 250 | 1732 | 18 | 1m42s |
| 320 | 1002 | 0.5333 | 141 | 1851 | 8 | 6m23s |
| 320 | 1003 | 0.5955 | 391 | 1600 | 9 | 6m24s |
| 320 | 1004 | 0.5248 | 127 | 1845 | 28 | 2m58s |
| 320 | 1005 | 0.5212 | 101 | 1883 | 16 | 2m51s |
| 640 | 1001 | 0.5228 | 124 | 1843 | 33 | 2m25s |
| 640 | 1002 | 0.5295 | 128 | 1862 | 10 | 6m16s |
| 640 | 1003 | 0.5240 | 118 | 1860 | 22 | 5m39s |
| 640 | 1004 | 0.5282 | 185 | 1743 | 72 | 2m46s |
| 640 | 1005 | 0.5170 | 97 | 1874 | 29 | 2m54s |
| 1280 | 1001 | 0.5238 | 133 | 1829 | 38 | 4m01s |
| 1280 | 1002 | 0.5475 | 207 | 1776 | 17 | 6m11s |
| 1280 | 1003 | 0.5085 | 72 | 1890 | 38 | 5m19s |
| 1280 | 1004 | 0.4978 | 61 | 1869 | 70 | 2m34s |
| 1280 | 1005 | 0.5135 | 86 | 1882 | 32 | 2m32s |
| 2560 | 1001 | 0.4780 | 50 | 1812 | 138 | 5m02s |
| 2560 | 1002 | 0.4748 | 55 | 1789 | 156 | 5m28s |
| 2560 | 1003 | 0.5195 | 112 | 1854 | 34 | 2m32s |
| 2560 | 1004 | 0.5050 | 97 | 1826 | 77 | 2m19s |
| 2560 | 1005 | 0.4900 | 45 | 1870 | 85 | 2m31s |

The wins and losses are worth reading directly, because every cell is
draw-dominated. Per 2,000 games, on average:

| sims | wins | losses | draws |
| ---: | ---: | ---: | ---: |
| 160 | **435** | **7** | 1,558 |
| 320 | 202 | 16 | 1,782 |
| 640 | 130 | 33 | 1,836 |
| 1280 | 112 | 39 | 1,849 |
| 2560 | 72 | 98 | 1,830 |

A 160-sim model beats a random mover 435 times in 2,000 and loses 7. A 2,560-sim
model wins 72 and loses 98 — it loses to a random player *more often than it
beats one*. The ordering is not an artefact of the draw rate.

### 3.2 Per level

| sims | gens | mean | SD | SE | 95% CI | min | max |
| ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| 160 | 64 | **0.6072** | 0.0364 | 0.0163 | [0.5619, 0.6524] | 0.5680 | 0.6635 |
| 320 | 32 | **0.5466** | 0.0309 | 0.0138 | [0.5082, 0.5849] | 0.5212 | 0.5955 |
| 640 | 16 | **0.5243** | 0.0050 | 0.0022 | [0.5181, 0.5305] | 0.5170 | 0.5295 |
| 1280 | 8 | **0.5182** | 0.0189 | 0.0084 | [0.4948, 0.5416] | 0.4978 | 0.5475 |
| 2560 | 4 | **0.4934** | 0.0188 | 0.0084 | [0.4701, 0.5168] | 0.4748 | 0.5195 |

```
within-level (seed) SD, pooled : 0.0246 score  = 17.2 Elo
between-level spread (max-min) : 0.1137 score  = 79.5 Elo
spread / seed SD               : 4.63 x
evaluation SE per match        : 0.0035  (14% of the seed SD)
minimum detectable effect      : 0.0496 score = 34.7 Elo  (n=5, alpha 0.05, power 0.80)
```

Monotone decreasing, with no flat region anywhere.

### 3.3 Tests

| comparison | diff | diff Elo | Cohen d | Welch p | perm p (exact) | Holm p (perm) | Holm p (Welch) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 160 vs 320 | 0.0606 | **+43.2** | +2.47 | 0.0225 | 0.0238 | 0.1190 | 0.1150 |
| 160 vs 640 | 0.0829 | **+58.7** | +3.37 | 0.0066 | 0.0079 | 0.0794 | 0.0529 |
| **160 vs 1280** | 0.0890 | **+63.0** | +3.62 | 0.0029 | 0.0079 | 0.0794 | **0.0257** |
| **160 vs 2560** | 0.1137 | **+80.2** | +4.63 | 0.0008 | 0.0079 | 0.0794 | **0.0082** |
| 320 vs 640 | 0.0223 | +15.5 | +0.91 | 0.1836 | 0.1349 | 0.3571 | 0.3777 |
| 320 vs 1280 | 0.0284 | +19.8 | +1.15 | 0.1259 | 0.1190 | 0.3571 | 0.3777 |
| 320 vs 2560 | 0.0531 | +37.0 | +2.16 | 0.0146 | 0.0079 | 0.0794 | 0.1022 |
| 640 vs 1280 | 0.0061 | +4.2 | +0.25 | 0.5185 | 0.5238 | 0.5238 | 0.5185 |
| 640 vs 2560 | 0.0308 | +21.5 | +1.26 | 0.0192 | 0.0159 | 0.0952 | 0.1150 |
| 1280 vs 2560 | 0.0247 | +17.2 | +1.01 | 0.0713 | 0.0794 | 0.3175 | 0.2852 |

Omnibus permutation F-test across all five levels: **F = 15.377, p < 1e-4**
(20,000 draws). Pairwise permutation tests are exact — all 252 relabellings
enumerated.

**A design fault worth stating plainly: with five levels, no permutation
comparison can survive Holm, by construction.** At 5 vs 5 the exact two-sided
permutation p-value has a floor of 2/252 = **0.00794**, and Holm's most
stringent threshold in a ten-member family is 0.05/10 = 0.005. The floor is
above the threshold, so a Holm-adjusted permutation p can never fall below
0.0794 no matter how complete the separation. The harness's
"0 of 10 pairwise comparisons survive Holm" is therefore **not** evidence of a
weak effect — it is arithmetic. Adding the fifth level (2560) to the three the
previous experiment asked for is what did it; with four levels the family is six
and 0.0476 is reachable, which is how `docs/ABLATION_RESULTS.md` §2.4 got a
significant pair.

Two things fill the gap, and both are in the table above. Welch's t-test is
continuous and has no such floor: **two comparisons survive Holm on Welch**
(160 vs 2560 at 0.0082, 160 vs 1280 at 0.0257), at the cost of the approximate
normality the permutation test does not assume. And the omnibus test is
Monte-Carlo, not enumerated, so it is unaffected: p < 1e-4.

### 3.4 The trend, as one test

The pairwise family is the wrong instrument for a question about a *curve*. The
directional hypothesis is a single test with no multiplicity. Regressing score
on log2(sims) across all 25 runs:

```
slope      = -0.02557 score per doubling of sims  =  -17.9 Elo per doubling
Pearson r  = -0.8156   (r^2 = 0.665)
permutation p (100,000 shuffles, two-sided, Phipson-Smyth correction) = 1e-5
```

Two thirds of the variance in external strength across these 25 runs is
explained by one number: how many simulations per move the run was trained with,
and the sign is negative.

### 3.5 Verdict

**At equal compute, more search per move during training makes the network
weaker, monotonically, across a 16× range. The curve does not turn over anywhere
above 160 because it is already on the way down at 160.** Spread 4.63× the
pooled seed SD, omnibus p < 1e-4, trend p = 1e-5, −17.9 Elo per doubling.

What cannot be claimed: that any *adjacent* pair is individually resolved. The
MDE is 34.7 Elo and the adjacent gaps are 43, 16, 4 and 17 Elo, so only
160 → 320 clears it. "640 and 1280 are indistinguishable" means *no further apart
than 35 Elo*, not equal. The evidence for the ordering is the monotone trend plus
the omnibus test, not any single pair.

---

## 4. Why: the telemetry

Same 25 runs, re-analysed on last-generation telemetry with the same machinery
(`--metric`), at zero additional compute.

| outcome | 160 | 320 | 640 | 1280 | 2560 | pooled seed SD | spread / SD | omnibus p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| threefold-repetition draws | 0.1275 | 0.0261 | 0.0116 | 0.0058 | 0.0000 | 0.0143 | 8.89× | <1e-4 |
| self-play draw rate | 0.4522 | 0.4696 | 0.5043 | 0.5913 | 0.6812 | 0.0621 | 3.69× | 0.0001 |
| average game length (plies) | 135.4 | 152.0 | 155.9 | 161.4 | 174.4 | 6.42 | 6.07× | <1e-4 |
| captures per game | 23.20 | 22.11 | 20.55 | 20.52 | 21.54 | 0.737 | 3.64× | <1e-4 |
| policy top-1 agreement | 0.4930 | 0.3457 | 0.2671 | 0.2638 | 0.3199 | 0.0089 | 25.64× | <1e-4 |
| KL(search visits ‖ policy) | 0.3732 | 0.4668 | 0.4192 | 0.3679 | 0.3562 | 0.0149 | 7.44× | <1e-4 |

And how the last generation's self-play games *ended*:

| sims | checkmate | 200-ply cap | threefold | fifty-move | resign |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 160 | **0.5333** | 0.2551 | 0.1275 | 0.0116 | 0.0145 |
| 320 | 0.5275 | 0.4348 | 0.0261 | 0.0000 | 0.0029 |
| 640 | 0.4928 | 0.4870 | 0.0116 | 0.0000 | 0.0029 |
| 1280 | 0.4058 | 0.5855 | 0.0058 | 0.0000 | 0.0029 |
| 2560 | **0.3188** | **0.6812** | 0.0000 | 0.0000 | 0.0000 |

**This is the mechanism, and it is not the one the project expected.**

* **Deeper search still kills the repetition collapse.** Threefold draws fall
  0.1275 → 0.0000 across the range, 8.9× the seed SD, p < 1e-4. The
  `docs/ALGORITHM.md` claim that shuffling is a search failure survives
  compute matching for a third time.
* **But killing repetition no longer buys strength.** The level with the *most*
  repetition draws is the strongest by 80 Elo. Repetition rate and external
  strength, which moved together in every previous sweep, are now
  **anti-correlated**. Repetition was a symptom of a search that ignored its
  priors; with `9f6936d` in place, a 160-sim search resolves games on its own
  and the residual repetition is no longer diagnostic.
* **What replaces repetition is worse.** As simulations rise, games do not
  resolve — they run out of room. Checkmates fall from 53% to 32% of self-play
  games while the **200-ply cap rises from 26% to 68%**. At 2,560 simulations
  two out of three training games end in an artificial truncation labelled with
  the draw penalty. The value head's target is then dominated by a label that
  says nothing about the position, which is precisely the pathology
  `docs/TACTICS.md` diagnosed from the other direction: *when neither side can
  be made to lose, the result stops carrying information about the position.*
  Two evenly-matched deep searches are very good at not losing.
* **Captures barely move** (23.2 → 20.5, a 12% decline), unlike the pre-fix
  sweep where they tripled from 6.6 to 18.5 across 64 → 320. On the fixed
  binary even 160 simulations already produces capture-rich, non-shuffling
  chess, so that lever is spent.
* **`policy_top1` and `policy_kl` remain uninterpretable across configurations.**
  Top-1 agreement is non-monotone (0.49 / 0.35 / 0.27 / 0.26 / 0.32) at 25.6×
  the seed SD, and its ordering is *opposite* to strength over most of the
  range; KL is non-monotone with its worst cell at 320. Both are scored against
  the MCTS visit distribution, which the factor under test changes. The caution
  in `docs/ABLATION_RESULTS.md` §5, point 4 stands and is now reinforced.

The 80-sim level from §7 extends the pattern in the same direction and is worth
reading next to that table (last generation, 5 seeds, separate experiment):
repetition **0.2087**, ply cap 0.2203, checkmate 0.4058, captures 24.90, length
140.2. It has *twice* the repetition rate of 160 and the *lowest* checkmate rate
of any level measured, and it is nominally the strongest configuration in this
whole file. Repetition rate and checkmate rate are no longer telling you
anything about strength.

One thing worth separating, because it is easy to conflate.
`docs/TACTICS.md` §2 shows that raising simulations **at play time, on one fixed
trained network**, monotonically improves it (0.035 → 0.237 vs `material-1` from
160 to 6400). That is a different knob from the one measured here, which is
simulations **during training, at a fixed total compute budget**. Both results
are true at once: search is worth buying when you already have the network, and
not worth buying with the experience that produces it.

---

## 5. The data-starvation tension, quantified

The brief anticipated a level that wins on score while producing too few games
to trust. That did not happen — the winner is the level with the **most** data —
but the mirror image of it did, and it is the main limitation of this result.

| sims | gens | self-play games | positions | optimiser steps | positions/step |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 160 | 64 | 4,374 | 663,096 | 12,800 | 51.8 |
| 320 | 32 | 2,166 | 338,495 | 6,400 | 52.9 |
| 640 | 16 | 1,062 | 173,337 | 3,200 | 54.2 |
| 1280 | 8 | 510 | 85,034 | 1,600 | 53.2 |
| 2560 | 4 | 234 | 38,594 | 800 | 48.2 |

At 2,560 simulations a run is **four generations, 234 self-play games, 38,594
positions and 800 optimiser steps** — 17.2× less data and 16× fewer gradient
steps than the 160-sim level, at equal compute. The sample-reuse ratio is
matched (48–54 positions per step throughout), so this is a difference in
absolute experience, not in how hard the data is squeezed.

**Therefore this experiment cannot separate "deep search produces worse training
data" from "four generations is not enough training", and it should not pretend
to.** At a matched evaluation budget the two are reciprocals by construction:
`gens = budget / (games_per_gen × plies × sims)`. What it does establish — and
this is the decision-relevant claim — is that **buying search with experience is
a losing trade at this budget**, by 18 Elo per doubling.

That claim is explicitly budget-dependent. At 100× the compute, 2,560
simulations would run 400 generations rather than 4, and nothing here says how
the ranking would look then. The termination mix in §4 is the reason to be
cautious in *both* directions: the ply-cap pathology at high simulation counts
is a property of the games, not of the run length, and it would not be cured by
a longer run.

---

## 6. Recommendation for the one-hour run

**Use `--sims 160`.**

```sh
./build/chessrl az --run runs/<name> \
    --sims 160 \           # the best of 160/320/640/1280/2560 at equal compute:
                           #   +43 Elo over 320, +80 over 2560, monotone,
                           #   -17.9 Elo per doubling (trend p = 1e-5)
    --agents 16 \          # from docs/ABLATION_RESULTS.md §2.7 -- see the caveat
    --games-per-agent 6 \
    --steps 100 \
    --start mixed \        # az_default_cfg() zeroes its own start_mode assignment
    --ema-decay 0
```

Two caveats on the rest of that line. **`sims` was ablated at 32 agents**, which
is the harness's base configuration, so `--agents 16 --sims 160` is a cell
neither experiment ran — the same warning `docs/ABLATION_RESULTS.md` §7 gives
about its own recommendation. And **the `agents` result itself was measured on
the broken search** (§9, item 1), so `--agents 16` is the weaker half of this
recommendation by a wide margin. If only one flag can be trusted here, it is
`--sims 160`.

**The evidence, in order of weight.**

1. **Everything above 160 is worse, and the decline is steep and clean**: −43 Elo
   at 320, −80 Elo at 2560, monotone across a 16× range with no flat region,
   4.63× the pooled seed SD, omnibus p < 1e-4, and a log-linear trend of −17.9
   Elo per doubling at p = 1e-5.
2. **Below 160 the decline stops.** 80 simulations is +12 Elo — 0.74× the seed
   SD, 0.37× the 33-Elo MDE, exact permutation p = 0.29 (§7). The top of this
   curve is a plateau spanning at least 80–160 with a cliff above it, and 160 is
   the top end of the plateau.
3. **On a wall-clock budget — which is what "one hour" means — 160 is the better
   half of that tie.** At 80 simulations the same evaluation budget is spread
   over twice the generations and twice the optimiser steps, and the
   per-generation serial cost drops throughput from 871k to 671k evaluations per
   second of wall clock (§7.2). One hour at 160 sims buys 30% more compute than
   one hour at 80.
4. **The number has been measured twice, independently, and replicated to four
   decimal places** (0.6072 and 0.6076 over ten runs in two experiments hours
   apart under very different machine load, §7.1).
5. It is the operating point the project already uses, so nothing has to change —
   but for a completely different reason than the one previously on file, and
   `docs/ABLATION_RESULTS.md` §6's recommendation of 320 should be treated as
   superseded.

**What could still move it.** 40 and 64 remain untested on the fixed search
(§7.3). Since the curve has already flattened by 80, a large gain below it would
be surprising, and any such gain would have to overcome the same
generations-per-hour penalty that costs 80 its tie-break. The pre-fix
measurement of 64 (0.4531, 23 Elo below 160) cannot settle it, because the PUCT
fix is worth four times as much at 160 as at 320 — its benefit grows as the
budget shrinks, so it plausibly rewrote the low end hardest of all.

---

## 7. Bracketing the peak from below

A second, self-contained experiment, same harness, same budget, same seeds, same
binary: **`--levels 80,160 --seeds 5`**. Two levels, so the pairwise family is
one and the exact permutation test can reach significance if there is anything
to find. It re-runs the 160 cell from scratch rather than reusing the one in §3,
which makes it an internally consistent comparison *and* an independent
replication.

```sh
python3 py/ablate.py --factor sims --levels 80,160 --seeds 5 \
        --budget-games 6144 --threads 8 --eval-games 2000 \
        --out runs/ablation_sims_curve_80.json
```

| sims | gens | mean | SD | SE | 95% CI | min | max |
| ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| 80 | 128 | **0.6240** | 0.0192 | 0.0086 | [0.6002, 0.6478] | 0.5948 | 0.6445 |
| 160 | 64 | **0.6076** | 0.0249 | 0.0111 | [0.5767, 0.6385] | 0.5637 | 0.6248 |

```
within-level (seed) SD, pooled : 0.0222 score  = 16.3 Elo   (cf. 0.0246 in the main design)
80 - 160                       : 0.0164 score  = +12.0 Elo, Cohen d +0.74
exact permutation p            : 0.2937      Welch p 0.2792
omnibus permutation F          : 1.360,  p = 0.2902
minimum detectable effect      : 0.0449 score = 33.0 Elo
VERDICT                        : INDISTINGUISHABLE FROM SEED NOISE
```

**The curve has a flat top, and 160 is on it.** Below 160 the decline stops: 80
is nominally 12 Elo better, 0.74× the seed SD and 0.37× the MDE. Pooling the two
independent 160 measurements (n = 10) against 80 (n = 5) gives the same answer —
+0.0166 = +12.2 Elo, Cohen *d* = 0.62, exact permutation p = 0.2704, MDE 32.5
Elo. So the shape is a plateau spanning at least 80–160, with a cliff above it.

### 7.1 The replication, which is the most reassuring number here

The 160 cell was measured twice, in two separate experiments, hours apart, under
different machine load:

| | n | mean | SD | W per 2,000 | L per 2,000 |
| --- | ---: | ---: | ---: | ---: | ---: |
| §3, main design | 5 | 0.6072 | 0.0364 | 435 | 7 |
| §7, bracket | 5 | 0.6076 | 0.0249 | 434 | 3 |

**Difference: +0.0004 score** (Welch p 0.98, exact permutation p 0.976). Ten runs
of the same configuration, split into two groups by nothing but when they ran,
agree to four decimal places in the mean. The pooled seed SD is 0.0294 at that
cell, so this is not a small-variance artefact — it is the training variance
averaging out exactly as it should. It also retires the worry that the heavy
contention in §1.1 biased anything: the contended group and the quiet group
produced the same number.

### 7.2 Wall clock is the tie-break, and it favours 160

At a tie on strength the decision goes to throughput, and the bracket ran quietly
enough to measure it (load 10–25, not 99):

| sims | gens | realised evals | train s | **evals per second of wall clock** |
| ---: | ---: | ---: | ---: | ---: |
| 80 | 128 | 115,176,010 | 171.7 | **670,798** |
| 160 | 64 | 124,701,506 | 143.2 | **870,821** |

Self-play throughput is identical (1.15M vs 1.19M evals/s inside the generation
loop). The 30% gap is per-generation serial cost: at 80 simulations the same
evaluation budget is spread over **twice the generations and twice the optimiser
steps** (25,600 vs 12,800), and each generation pays for 27 rating-anchor games,
an Elo fit and a hall-of-fame check. **In one hour of wall clock, 160 simulations
buys 30% more network evaluations than 80 does** — which is worth more than a
12-Elo point estimate that sits inside a 33-Elo MDE.

### 7.3 What is still not known below 80

40 and 64 remain untested on the fixed search. The trend has clearly stopped —
80 and 160 are a wash — so there is no reason to expect a large gain further
down, and the pre-fix data (64 sims at 0.4531, 23 Elo below 160) is the only
evidence there is, however compromised. If the plateau extends to 64 it changes
nothing, because 64 would be slower still in generations-per-hour. The one
scenario that would matter is a *second* peak below 64, and nothing in the shape
of this curve suggests one.

---

## 8. Limitations

* **The machine was not idle** during the main experiment (§1.1): median load 99
  on 8 cores, peak 224.7, 113 minutes of wall clock against a 35-minute plan.
  This changes wall clock, not the budget, and the seed-major interleaving
  spreads it across levels. The replication in §7.1 is the direct evidence that
  it did not bias the result — the contended 160 cell and the quiet one agree to
  +0.0004. Every wall-clock claim in this file is taken from the quiet bracket
  (§7.2), never from the contended main run.
* **The outcome is the raw policy head without search.** A hyper-parameter could
  improve the searched agent without improving the raw policy. `--eval-mode
  search` would measure that, at the cost of putting `search.c`'s hand-written
  material evaluation in front of the network, which `docs/FROM_SCRATCH.md`
  excludes from the shipped agent. For this factor the risk is pointed:
  training-time simulations and play-time simulations are different knobs
  (§4), and this measures the network the training produced, not the agent that
  ships with search in front of it.
* **`sims` confounds search depth with total games, total optimiser steps and
  generation count** (§5). Unavoidably — that is the trade being priced.
* **The ranking is budget-specific.** Every run here is 111–132M evaluations.
  `runs/az_hour/best.crl` is roughly 79× that. That the ranking persists at 79×
  the budget is an assumption, not a result.
* **Five seeds is a small sample.** MDE 34.7 Elo; effects smaller than that were
  missed more often than not.
* **Five levels made the Holm-corrected permutation column unusable** (§3.3).
  That is a design fault, not a finding. A future `sims` experiment should use
  at most four levels, or accept the omnibus and trend tests as the primary
  evidence and say so in advance.
* **One opponent, one scale.** `random` was the harness default, fixed before
  the run. `docs/ABLATION_RESULTS.md` §4 found that a blunder-punishing opponent
  (`randomplus-0.5`) amplified the `agents` ordering three-fold; this experiment
  has not been re-scored that way, and it should be.
* **Hang rate was not measured.** It is not in the telemetry stream — the
  instrument is `tools/diag_tactics.c`, which is not part of `make` and is built
  by hand. Re-scoring these 35 checkpoints with it would test the mechanism in
  §4 directly and is the cheapest follow-up in this list.

---

## 9. Follow-ups, in order of expected value

1. **Re-run every conclusion in `docs/ABLATION_RESULTS.md` that involved
   search.** The `agents` experiment was run at 160 sims on the broken search,
   which cost that configuration ~85 Elo. Its ordering may or may not survive.
2. **`--factor cpuct`.** The PUCT fix changed what the exploration constant
   does — priors now act from the first descent — and `cpuct 1.4` was tuned
   before that. It is the single most likely stale hyper-parameter in the base
   configuration.
3. **The ply cap.** Two thirds of training games at high simulation counts end
   at `--max-plies 200` (§4). `--max-plies` is not a factor in `ablate.py`, but
   it is a one-line addition to `FACTORS`, and the termination mix says it is
   now a first-order influence on what the value head is trained to predict.
4. **Re-score these 35 checkpoints against `randomplus-0.5`** and with
   `tools/diag_tactics.c`, for a blunder-rate reading of the same models. The
   whole point of `docs/TACTICS.md` is that this agent loses to anything that
   punishes a hanging piece, and `random` cannot punish one. §4 argues that high
   simulation counts corrupt the value target by truncating games; a hang-rate
   reading of the same checkpoints would test that directly.
5. **`--factor sims --levels 40,80,160`** if the low end ever becomes worth
   another 45 minutes (§7.3). Low priority: the plateau is already established
   at its upper edge, and anything below 80 pays a worsening
   generations-per-hour penalty for whatever it might gain.

---

## Provenance

| | |
| --- | --- |
| harness | `py/ablate.py` and `py/benchmark.py` **at `HEAD`**, byte-identical to `git show HEAD:` (verified); nothing in the harness was changed or needed changing. Another agent modified the working-tree copy of `py/ablate.py` after these runs started; that copy was never used here |
| sources | `git HEAD` = `9f6936d`; the working tree carried unrelated uncommitted changes from another agent (`src/az.c`, `src/az.h`, `src/main.c`) which are **not** in the snapshot — the snapshot's `src/` was taken from `git archive HEAD` and verified file-by-file against `HEAD` |
| binary | `build/chessrl` md5 `51ed65710c75485eef816b0bae4416fa`, `build/libchessrl.dylib` md5 `707f15b0e394eaafb6b5ac0d3c012921`, both built from those sources and snapshot-copied to a temp path before the runs |
| difference from `docs/ABLATION_RESULTS.md` | `src/mcts.c` only: commit `9f6936d`, `sqrt(nsum)` → `sqrt(nsum + 1.0f)` (9 insertions, 6 deletions) |
| verification | clean out-of-tree rebuild of all 9 sources at the project's flags: **0 warnings, 0 errors**; `make test` **7 checks / 0 failures**; `make audit` shipped play path clean. Run twice — once on the snapshot's `HEAD` sources and once on the working tree as it stood at the end |
| machine | Apple M3, 8 cores, 8 GB, macOS 23.6.0 (Darwin), Python 3.14.5 — **not idle**, see §1.1 |
| raw output | `runs/ablation_sims_curve.json` (25 runs), `runs/ablation_sims_curve_80.json` (10 runs) |
| cost | 35 training runs, 4.32e9 realised network evaluations, 70,000 evaluation games, 2h21m wall clock (113 min contended + 28 min quiet) |
| dates | 2026-09-13, 01:13–03:06 local (main design), 03:09–03:37 (bracket) |
