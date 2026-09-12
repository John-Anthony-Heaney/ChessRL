# Ablation results: does the population earn its keep, and is 160 sims the knee?

Two compute-matched, five-seed ablations run with `py/ablate.py` on 2026-09-12.
Both answer a question the project had previously settled with one run per
configuration, which — as `docs/ABLATION.md` argues at length — settles nothing.

**Headline.**

| factor | levels | verdict | size of the effect |
| --- | --- | --- | --- |
| `agents` | 4, 16, 64, 128 | **moves the result by more than seed noise** (omnibus permutation p = 0.0190) | 128 -> 4 agents is worth **+0.0478 score = +33.2 Elo**, 2.22x the pooled seed SD |
| `sims` | 64, 160, 320 | **moves the result by more than seed noise** (omnibus permutation p < 1e-4) | 64 -> 320 sims is worth **+0.0640 score = +44.5 Elo**, 6.20x the pooled seed SD, every pair significant after Holm |

Both effects point the same way: **at a fixed compute budget this trainer wants
fewer agents and more search per move.** Neither is the direction the shipped
configuration points: the 1,900-generation production run used 128 agents and
160 simulations, and the compiled-in defaults in `az_default_cfg()` are 32
agents and **64 simulations**, which is the worst level measured here — 45 Elo
below 320 at equal compute.

The single most reusable number produced here is the **pooled run-to-run
standard deviation**: `0.0215` score = **14.9 Elo** in the `agents` design and
`0.0103` score = **7.2 Elo** in the `sims` design. Every single-run comparison
in this repository, past and future, should be read against that yardstick.

---

## 1. Method

Exactly the harness described in `docs/ABLATION.md`; nothing in `src/` was
modified, and no bug in `py/ablate.py` had to be fixed — it ran as written.

**Compute matching.** The generation count is derived from a fixed budget, never
held equal. `agents` changes games per generation but not the cost of a game, so
it is matched on **total self-play games** (9,216 per run) with `--steps` scaled
proportionally to games/generation, which also equalises total optimiser steps
(19,200) and the sample-reuse ratio. `sims` changes the cost of a game, so it is
matched on **total network evaluations** (117,637,120 per run). What is *not*
matched is listed with each experiment; that block is not boilerplate and is the
first place to look before believing a verdict.

**External evaluation.** Internal Elo is anchored to each run's own hall of fame
and is not comparable across runs (`docs/RATING.md`), so it is ignored entirely.
Every finished `checkpoint.crl` — the model after the full budget, not the
internally early-stopped `best.crl` — plays **2,000 games as its raw policy head
(argmax, no search) against `random`**, in colour-reversed pairs from a fixed
opening set, under one evaluation seed (20260911) common to every model, so the
comparison is paired and the opening-set component of the error cancels out of
the differences. The resulting measurement noise is small relative to what is
being measured: the evaluation SE per match is 0.0029, i.e. **14% of the seed SD**
in the `agents` experiment and 29% in `sims`. The seed SD is training variance,
not measurement variance.

**Where this score scale sits.** On the identical protocol (2,000 games, same
seed, policy head only, vs `random`), the project's strongest existing model,
`runs/az_hour/best.crl` — 1,900 generations, 729,600 games, roughly 79x the
budget of any run here — scores **0.6695** (679W 1320D 1L). A coin-flip 0.5000 is
roughly what a short-budget model achieves. So the interval 0.45 to 0.55 that
these experiments live in is not a compressed corner of the scale; it is the
bottom half of the range this network architecture reaches at 79x the compute.

**Seeds.** Five independent training seeds (1001-1005) per level, run
seed-major so that every level always has the same number of replicates.
Training is **not reproducible at a fixed seed** — the per-thread trunk-gradient
buffers are reduced in whatever order threads finish and floating-point addition
is not associative — so a "seed" here is a replicate, and the SD it produces is
the full run-to-run variance including the thread scheduler. That is the correct
thing to test a factor against, and it cannot be made smaller by fixing a seed.

**Statistics.** Effect size before p-value, always: raw difference, the same
difference in Elo, and Cohen's *d* against the pooled seed SD. Two tests, because
each covers the other's weakness — Welch's t (unequal variances) and a
permutation test which at 5-vs-5 is computed **exactly** (all 252 relabellings
enumerated, no Monte-Carlo error), with Holm-Bonferroni across the pairwise
family and an omnibus permutation F-test so that significance cannot be
manufactured by picking the smallest pairwise p. The **minimum detectable
effect** (alpha = 0.05, power = 0.80, n = 5 per level) is reported so that a null
is interpretable.

**Machine state — the machine was NOT idle.** This is the one honest defect in
the run. `ps` and `uptime` at launch (21:41) showed load 2.5 with only the
desktop UI active, but from roughly 21:50 to 22:40 sibling measurement jobs
belonging to other agents in the same workflow (`conv_acc`, `diag_tactics`) ran
concurrently, and the one-minute load average peaked at **30.4** on 8 cores. It
settled to 5-9 for the second half of the `agents` experiment and about 7 for
`sims`. Observed effect: the first `agents` runs achieved 361k-780k network
evaluations/second where the last achieved 1.08M-1.30M, and per-run training
time ran 4m43s-7m48s early against 3m10s-3m46s late.

Why this does not invalidate the comparisons:

* the budget is matched on **games and evaluations, not wall clock**, so a slow
  run is not a smaller run — every run at a level played exactly the same number
  of games;
* runs are interleaved **seed-major**, so every level appears once in every
  contention regime; contention is spread across levels rather than confounded
  with one of them;
* the extra non-determinism contention introduces (thread scheduling reorders
  the gradient reduction) is precisely the run-to-run variance the design
  measures and tests against, so it inflates the yardstick rather than biasing
  the comparison.

The one number contention does corrupt is **throughput**, so throughput is
reported separately below from the quiet tail of the experiment only, and is
flagged where it feeds a recommendation.

**Cost.** `agents`: 20 training runs, 184,320 self-play games, 3.53e9 network
evaluations, 40,000 evaluation games, 88.9 minutes wall clock. `sims`: 15
training runs, 122,880 self-play games, 1.76e9 network evaluations, 30,000
evaluation games, 36.8 minutes.

```sh
# the two commands that produced everything below
python3 py/ablate.py --factor agents --levels 4,16,64,128 --seeds 5 \
        --budget-games 9216 --threads 8 --eval-games 2000
python3 py/ablate.py --factor sims   --levels 64,160,320   --seeds 5 \
        --budget-games 6144 --threads 8 --eval-games 2000
```

(Run from a snapshot copy of `build/` and `py/` so that a concurrent relink of
`build/chessrl` by another agent could not kill a training run mid-flight.
`--budget-games 9216` divides into whole generations at all four `agents`
levels; for `sims` the games budget only sets the evaluation budget, 117.6M
evaluations per run.)

---

## 2. Experiment 1 — `agents`: the population does not earn its keep

### 2.1 The plan

Every level plays the same 9,216 games and takes the same 19,200 optimiser
steps. What changes is how those games are grouped.

| agents | games/gen | generations | steps/gen | total steps | replay buffer spans | hof snapshots |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 12 | 768 | 25 | 19,200 | 34.8 generations | 10 |
| 16 | 48 | 192 | 100 | 19,200 | 8.7 generations | 10 |
| 64 | 192 | 48 | 400 | 19,200 | 2.2 generations | 10 |
| 128 | 384 | 24 | 800 | 19,200 | 1.1 generations | 13 |

### 2.2 Raw numbers

Score is out of 2,000 games vs `random`, policy head only, same openings for
every model.

| agents | seed | score | W | D | L | train |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 1001 | 0.5455 | 197 | 1788 | 15 | 4m43s |
| 4 | 1002 | 0.5473 | 196 | 1797 | 7 | 6m41s |
| 4 | 1003 | 0.5485 | 227 | 1740 | 33 | 4m58s |
| 4 | 1004 | 0.5240 | 109 | 1878 | 13 | 4m05s |
| 4 | 1005 | 0.5210 | 138 | 1808 | 54 | 4m10s |
| 16 | 1001 | 0.5295 | 133 | 1852 | 15 | 7m48s |
| 16 | 1002 | 0.5010 | 40 | 1924 | 36 | 4m17s |
| 16 | 1003 | 0.5265 | 117 | 1872 | 11 | 3m25s |
| 16 | 1004 | 0.5433 | 188 | 1797 | 15 | 3m27s |
| 16 | 1005 | 0.5298 | 122 | 1875 | 3 | 3m27s |
| 64 | 1001 | 0.4955 | 29 | 1924 | 47 | 5m12s |
| 64 | 1002 | 0.4935 | 33 | 1908 | 59 | 3m20s |
| 64 | 1003 | 0.4890 | 19 | 1918 | 63 | 3m12s |
| 64 | 1004 | 0.4835 | 31 | 1872 | 97 | 3m14s |
| 64 | 1005 | 0.5710 | 301 | 1682 | 17 | 3m10s |
| 128 | 1001 | 0.4753 | 29 | 1843 | 128 | 4m34s |
| 128 | 1002 | 0.4945 | 50 | 1878 | 72 | 3m37s |
| 128 | 1003 | 0.4875 | 36 | 1878 | 86 | 3m17s |
| 128 | 1004 | 0.5030 | 55 | 1902 | 43 | 3m20s |
| 128 | 1005 | 0.4873 | 31 | 1887 | 82 | 3m46s |

The win and loss columns are worth reading directly, because the score is
dominated by draws in every cell: a 4-agent model beats a random player 173
times in 2,000 on average and loses 24, while a 128-agent model wins 40 and
loses 82. The ordering is not an artifact of the draw rate.

### 2.3 Per level

| agents | gens | mean | SD | SE | 95% CI | min | max |
| ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| 4 | 768 | **0.5373** | 0.0135 | 0.0061 | [0.5204, 0.5541] | 0.5210 | 0.5485 |
| 16 | 192 | **0.5260** | 0.0154 | 0.0069 | [0.5069, 0.5451] | 0.5010 | 0.5433 |
| 64 | 48 | **0.5065** | 0.0364 | 0.0163 | [0.4614, 0.5516] | 0.4835 | 0.5710 |
| 128 | 24 | **0.4895** | 0.0102 | 0.0046 | [0.4768, 0.5022] | 0.4753 | 0.5030 |

Monotone decreasing in the number of agents.

```
within-level (seed) SD, pooled : 0.0215 score  = 14.9 Elo
between-level spread (max-min) : 0.0478 score  = 33.2 Elo
spread / seed SD               : 2.22 x
evaluation SE per match        : 0.0029  (14% of the seed SD)
```

### 2.4 Tests

| comparison | diff | diff Elo | Cohen d | Welch p | perm p (exact) | Holm p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 vs 16 | 0.0112 | +7.8 | +0.52 | 0.2555 | 0.3175 | 0.7857 |
| 4 vs 64 | 0.0308 | +21.4 | +1.43 | 0.1355 | 0.0873 | 0.3492 |
| **4 vs 128** | **0.0478** | **+33.2** | **+2.22** | **0.0003** | **0.0079** | **0.0476** |
| 16 vs 64 | 0.0195 | +13.6 | +0.91 | 0.3162 | 0.2619 | 0.7857 |
| 16 vs 128 | 0.0365 | +25.4 | +1.70 | 0.0032 | 0.0159 | 0.0794 |
| 64 vs 128 | 0.0170 | +11.8 | +0.79 | 0.3638 | 0.4286 | 0.7857 |

Omnibus permutation F-test across all four levels: **F = 4.831, p = 0.0190**
(20,000 draws). Pairwise permutation tests are exact (252 relabellings).

**Minimum detectable effect**, n = 5 per level, alpha = 0.05, power = 0.80:
**0.0434 score points = 30.2 Elo.**

### 2.5 Verdict

**The factor moves strength by more than run-to-run noise, and it moves it
against the population.** The spread across levels is 2.22x the pooled seed SD;
the omnibus permutation test rejects at p = 0.0190; the extreme pair, 4 agents
vs the shipped default of 128, survives Holm correction at p = 0.0476 with an
effect of **+33 Elo in favour of 4 agents** and Cohen's *d* = 2.22.

What cannot be claimed: no adjacent pair is individually significant after
correction. With an MDE of 30.2 Elo, this design can only resolve gaps of about
that size between two individual levels, so "4 and 16 are indistinguishable"
means *no further apart than 30 Elo*, not *equal*. The evidence for the ordering
rests on the monotone trend plus the omnibus test, not on any one pair.

The question posed was whether the population contributes nothing. The answer is
worse than nothing: **at a fixed game budget the population is actively
expensive.** Collapsing 128 agents to 4 does not merely free compute to spend
elsewhere, it buys about 33 Elo on its own.

### 2.6 What this does and does not identify

At a fixed game budget, population size is *tied to* generation count — 4 agents
means 768 generations, 128 agents means 24. Three things therefore change
together, and this experiment cannot separate them:

1. **Learning cycles.** Total optimiser steps are matched (19,200) but their
   interleaving with fresh data is not: the 4-agent run alternates 25 steps with
   12 new games, 768 times; the 128-agent run takes 800 steps against 384 games,
   24 times. More, smaller, fresher cycles win.
2. **Replay staleness.** A 50,000-position buffer spans 34.8 generations at 4
   agents and **1.1** at 128 — at the top level the buffer is not decorrelating
   at all, it is barely holding one generation.
3. **Evolution.** At 4 agents, `cull 0.10` of 4 culls zero, so **PBT is inert at
   the strongest level**. The best cell in the experiment ran with no evolution
   whatsoever. That is not a confound to apologise for; it is direct evidence
   that PBT is not paying for itself at this budget — and it agrees with a
   comment already in `src/az.c` beside `cull_frac`, which records that at 32
   agents "cull 0 was better still" and keeps culling only because the
   population mechanism is part of the contract.

For the practical decision — how many agents should the next run use — this does
not matter: population size is the knob, and turning it down is what produces
the gain. For the scientific question "what is the population for", it matters a
great deal, and the follow-up is `--factor games-per-agent`, which moves
games/generation *without* moving the number of agents.

Also untested: fewer than 4 agents. Pairing needs a population, and at 1-2
agents the Swiss pairing and hall-of-fame machinery degenerate. "Collapse to one
network" is not what was measured; "collapse to a handful" is.

### 2.7 The wall-clock caveat that changes the recommendation

Compute was matched on games, and the levels do **not** convert games into wall
clock equally. Mean self-play throughput over the quiet tail of the experiment
(seeds 1003-1005 only, to exclude the contended period):

| agents | self-play throughput | wall clock for 9,216 games | relative |
| ---: | ---: | ---: | ---: |
| 4 | 744,823 evals/s | 4m24s | **1.38x** |
| 16 | 1,074,503 evals/s | 3m26s | 1.08x |
| 64 | 1,249,197 evals/s | 3m11s | 1.00x |
| 128 | 1,202,970 evals/s | 3m27s | 1.08x |

(Throughput is the final generation's `evals_per_sec`; wall clock is the whole
run. Both are means over seeds 1003-1005 only.)

Twelve pairings per generation cannot fill 8 threads, and 768 generations pay
the per-generation serial cost 32 times more often than 24 do, so the 4-agent
level buys its +33 Elo at **38% more wall clock for the same 9,216 games**. The
16-agent level costs only 8% more than the fastest level and is statistically
indistinguishable from 4 (diff 7.8 Elo, d = 0.52, exact permutation p = 0.3175;
the MDE says any true gap is under 30 Elo). On an equal-wall-clock budget rather
than an equal-games budget, 16 is the defensible operating point, not 4.

### 2.8 Secondary outcomes, from the same runs

`ablate.py` stores each run's last-generation telemetry, so the same statistics
can be re-run on other outcome variables at zero additional compute. Every
number below is the same 20 runs, 5 seeds per level.

| outcome | 4 | 16 | 64 | 128 | spread / seed SD | omnibus p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| threefold-repetition draws | 0.067 | 0.115 | 0.184 | 0.244 | 2.75x | 0.0042 |
| captures per game | 19.67 | 17.27 | 13.23 | 14.56 | 3.31x | 0.0007 |
| average game length (plies) | 182.9 | 150.9 | 141.5 | 139.3 | 4.57x | 0.0001 |
| self-play draw rate | 0.667 | 0.624 | 0.612 | 0.646 | 0.39x | 0.9259 |
| policy top-1 agreement | 0.588 | 0.527 | 0.572 | 0.595 | 5.03x | <1e-4 |
| KL(search visits \|\| policy) | 0.341 | 0.291 | 0.359 | 0.395 | 4.64x | <1e-4 |

The first three corroborate the strength result with a mechanism: the
configurations that play better also shuffle less (repetition draws fall by a
factor of 3.7 from 128 agents to 4), trade more, and play longer games. The
overall draw rate does not move at all (p = 0.93) — what changes is the
*composition* of the draws, not their number.

Two warnings attach to this table.

* **These telemetry means are not equally precise across levels.** A generation
  at 4 agents is 12 games; at 128 it is 384. The last-generation repetition rate
  at the 4-agent level is a fraction of 12 games, which is why its SD (0.091) is
  five times the 128-agent level's (0.016), and why one 4-agent run reports a
  final-generation draw rate of exactly 1.000. The external score does not have
  this problem — every model plays the same 2,000 evaluation games.
* **`policy_top1` and `policy_kl` do not track strength.** Both are non-monotone
  here, with 16 agents an outlier in both, and see section 3.5 for the `sims`
  experiment, where top-1 agreement runs *exactly opposite* to external strength.
  They are measured against a target the factor itself moves, and they are not
  comparable across configurations.

---

## 3. Experiment 2 — `sims`: 160 is not the knee

`docs/ALGORITHM.md` reads a knee at 160 simulations off a sweep in which every
cell ran the same 20 generations, so more simulations also meant more total
compute: the x-axis was a compute axis. `docs/ABLATION.md` flags this. This
experiment fixes the total network evaluations per run instead.

### 3.1 The plan

Every level performs the same 117,637,120 network evaluations. What changes is
how they are spent: on more moves, or on deeper search per move.

| sims | gens | games/gen | total games | steps/gen | total steps | evals/run |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 160 | 96 | 15,360 | 200 | 32,000 | 117,637,120 |
| 160 | 64 | 96 | 6,144 | 200 | 12,800 | 117,637,120 |
| 320 | 32 | 96 | 3,072 | 200 | 6,400 | 117,637,120 |

The 320-sim level therefore sees **one fifth of the games and one fifth of the
optimiser steps** of the 64-sim level. That is the price of matching on
evaluations, and it is the honest way to state the trade: quality of search per
move against quantity of experience.

### 3.2 Raw numbers

| sims | seed | score | W | D | L | train |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 1001 | 0.4457 | 6 | 1771 | 223 | 2m35s |
| 64 | 1002 | 0.4532 | 7 | 1799 | 194 | 2m37s |
| 64 | 1003 | 0.4537 | 4 | 1807 | 189 | 2m10s |
| 64 | 1004 | 0.4497 | 2 | 1795 | 203 | 2m08s |
| 64 | 1005 | 0.4627 | 8 | 1835 | 157 | 2m26s |
| 160 | 1001 | 0.4880 | 32 | 1888 | 80 | 2m17s |
| 160 | 1002 | 0.4803 | 21 | 1879 | 100 | 2m10s |
| 160 | 1003 | 0.4878 | 28 | 1895 | 77 | 2m00s |
| 160 | 1004 | 0.4948 | 56 | 1867 | 77 | 2m03s |
| 160 | 1005 | 0.4818 | 21 | 1885 | 94 | 2m23s |
| 320 | 1001 | 0.5208 | 128 | 1827 | 45 | 2m10s |
| 320 | 1002 | 0.5028 | 39 | 1933 | 28 | 1m56s |
| 320 | 1003 | 0.5415 | 190 | 1786 | 24 | 1m54s |
| 320 | 1004 | 0.5165 | 110 | 1846 | 44 | 2m14s |
| 320 | 1005 | 0.5040 | 58 | 1900 | 42 | 2m16s |

A 64-sim model loses to a random player about 190 times in 2,000 games and wins
5; a 320-sim model loses 37 and wins 105. The separation is visible without
statistics.

### 3.3 Per level

| sims | gens | mean | SD | SE | 95% CI | min | max |
| ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| 64 | 160 | **0.4531** | 0.0063 | 0.0028 | [0.4452, 0.4609] | 0.4457 | 0.4627 |
| 160 | 64 | **0.4865** | 0.0058 | 0.0026 | [0.4793, 0.4937] | 0.4803 | 0.4948 |
| 320 | 32 | **0.5171** | 0.0157 | 0.0070 | [0.4976, 0.5366] | 0.5028 | 0.5415 |

```
within-level (seed) SD, pooled : 0.0103 score  = 7.2 Elo
between-level spread (max-min) : 0.0640 score  = 44.5 Elo
spread / seed SD               : 6.20 x
evaluation SE per match        : 0.0030  (29% of the seed SD)
```

### 3.4 Tests

| comparison | diff | diff Elo | Cohen d | Welch p | perm p (exact) | Holm p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **64 vs 160** | -0.0335 | **-23.3** | -3.24 | <1e-4 | 0.0079 | **0.0238** |
| **64 vs 320** | -0.0640 | **-44.6** | -6.20 | 0.0003 | 0.0079 | **0.0238** |
| **160 vs 320** | -0.0306 | **-21.3** | -2.96 | 0.0092 | 0.0079 | **0.0238** |

Omnibus permutation F-test: **F = 48.149, p < 1e-4** (20,000 draws).

**Minimum detectable effect**, n = 5 per level: **0.0209 score points = 14.5 Elo.**

Throughput was equal across levels — 1.16M, 1.22M, 1.21M evaluations/second, and
mean training times of 2.4, 2.2 and 2.1 minutes per run — so for this factor the
evaluation-matched comparison is also a **wall-clock-matched** comparison. There
is no hidden cost in the 320-sim cell.

### 3.5 Verdict

**More search per move wins decisively at equal compute, and 160 is not a knee.**
All three pairwise comparisons survive Holm correction; the spread is 6.20x the
pooled seed SD; the omnibus test rejects at p < 1e-4. Going from 160 to 320
simulations is worth **+21 Elo** at identical total evaluations and identical
wall clock — and the increment from 160 to 320 (+21.3 Elo) is essentially the
same size as the increment from 64 to 160 (+23.3 Elo). **The curve has not
flattened at the top level tested.** Whatever the real knee is, it is above 320.

This is the more surprising of the two results, because the 320-sim runs got
there on 3,072 games against the 64-sim runs' 15,360 — one fifth of the
experience and one fifth of the gradient steps — and still won by 44 Elo.

The secondary telemetry reproduces the `docs/ALGORITHM.md` sweep exactly, now
with 5 seeds and compute matching:

| outcome | 64 | 160 | 320 | spread / seed SD | omnibus p |
| --- | ---: | ---: | ---: | ---: | ---: |
| threefold-repetition draws | 0.603 | 0.183 | 0.058 | 11.07x | <1e-4 |
| captures per game | 6.58 | 13.65 | 18.46 | 17.14x | <1e-4 |
| average game length (plies) | 110.7 | 145.5 | 156.5 | 6.13x | <1e-4 |
| self-play draw rate | 0.815 | 0.641 | 0.586 | 3.94x | 0.0001 |
| policy top-1 agreement | 0.619 | 0.559 | 0.422 | 15.73x | <1e-4 |
| KL(search visits \|\| policy) | 0.412 | 0.382 | 0.408 | 2.10x | 0.0156 |

The original single-seed claim — that the shuffling equilibrium is a search
failure and not a reward failure — was one of the few conclusions
`docs/ABLATION.md` rated as adequately supported even at one seed. It now has
five seeds, an 11x-the-noise effect and p < 1e-4, and it survives compute
matching: 64 sims sits at 60% repetition draws and 320 sims at 5.8%, *at equal
total evaluations*.

**And a caution the project needs.** `policy_top1` runs exactly backwards:
0.619 at 64 sims, 0.422 at 320 — the configuration with the best external
strength has the *worst* agreement with its own search, at 15.7x the seed SD.
`policy_kl` is non-monotone and its 160-sim cell is the lowest. Both are measured
against the MCTS visit distribution, which the factor under test changes: more
simulations produce a sharper, harder target, so agreement falls while the
network gets stronger. The plateau of top-1 agreement at ~0.49 and the drifting
KL reported for the 1,900-generation run are therefore **not** evidence that
learning stalled; neither statistic is comparable across configurations, and
across the one factor where both were measured against external strength they
were anti-correlated with it.

---

## 4. Robustness check: a second opponent for the `agents` result

The primary opponent was fixed before the experiment ran (`random`, the harness
default, chosen in `docs/ABLATION.md` because it keeps short-budget models away
from both bounds). Because the whole project's weakness is that it loses to
anything that can punish a hanging piece, the `agents` result was re-measured
against an opponent that does exactly that: **`randomplus-0.5`**, which plays the
1-ply material-best move half the time and a random legal move otherwise.

This is the *same 20 trained models* re-scored with a different measuring stick —
a robustness check on the outcome variable, not an independent replication. 400
games per model, same evaluation seed, policy head only. Anchor on this scale:
`runs/az_hour/best.crl` scores 0.6238 (131W 237D 32L).

| agents | mean | SD | 95% CI | min | max |
| ---: | ---: | ---: | --- | ---: | ---: |
| 4 | 0.2638 | 0.0836 | [0.1600, 0.3675] | 0.1525 | 0.3663 |
| 16 | **0.3103** | 0.0737 | [0.2188, 0.4017] | 0.1787 | 0.3500 |
| 64 | 0.1447 | 0.0488 | [0.0842, 0.2053] | 0.0875 | 0.2200 |
| 128 | 0.1015 | 0.0320 | [0.0618, 0.1412] | 0.0638 | 0.1525 |

```
within-level (seed) SD, pooled : 0.0629 score  = 67.0 Elo
between-level spread (max-min) : 0.2088 score  = 222.5 Elo
spread / seed SD               : 3.32 x
omnibus permutation F-test     : F = 12.177, p = 0.0007
minimum detectable effect      : 0.1270 score points = 135.4 Elo
```

| comparison | diff | diff Elo | Cohen d | perm p (exact) | Holm p |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4 vs 16 | -0.0465 | -39.5 | -0.74 | 0.3651 | 0.3651 |
| 4 vs 128 | +0.1623 | +200.5 | +2.58 | 0.0159 | 0.0794 |
| **16 vs 128** | **+0.2088** | **+240.0** | **+3.32** | **0.0079** | **0.0476** |
| 16 vs 64 | +0.1655 | +169.8 | +2.63 | 0.0159 | 0.0794 |
| 4 vs 64 | +0.1190 | +130.3 | +1.89 | 0.0159 | 0.0794 |
| 64 vs 128 | +0.0432 | +70.2 | +0.69 | 0.1429 | 0.2857 |

**The ordering survives, and the gap is much larger against an opponent that
punishes blunders.** Small populations {4, 16} score two to three times what
large populations {64, 128} do on this scale. Within each group the difference
is not resolvable — and the top two swap places: 16 is nominally ahead of 4
here, by 39 Elo, which is well inside this instrument's 135-Elo MDE.

Note that this scale is much coarser: the pooled seed SD is 67 Elo against 15
Elo on the `random` scale, partly because a 400-game match is a noisier
instrument (evaluation SE 0.0114) and partly because a blunder-punishing
opponent amplifies genuine differences in blunder rate between runs. It is the
right scale for *which direction*, and the wrong scale for *how much*.

---

## 5. What the two results say together

1. **The population does not pay for itself.** Its stated purpose
   (`docs/ALGORITHM.md`) is to give evolution something real to select on and to
   supply varied opponents. At a fixed game budget, 128 agents score 33 Elo
   *below* 4 agents against `random` and 240 Elo below 16 agents against
   `randomplus-0.5`, and the strongest cell in the whole experiment ran with PBT
   structurally disabled (`cull 0.10` of 4 agents culls nobody). Nothing here
   says the population *cannot* help; it says that at this budget it costs more
   than it returns, and that the compute is better spent on generation count and
   search.

2. **Search per move is the lever, and it has not been turned far enough.**
   Doubling simulations from 160 to 320 is worth +21 Elo at identical total
   evaluations and identical wall clock, on half the games; going from the
   binary's default of 64 to 320 is worth +45 Elo on one fifth of the games. The project's
   own intuition ("sims per move is the lever that actually matters") is correct,
   and the reason it looked like a knee at 160 is that the sweep which found the
   knee was not compute-matched.

3. **The measured run-to-run SD gives the repo its missing yardstick.** 14.9 Elo
   (`agents` design, 9,216 games/run) and 7.2 Elo (`sims` design, 117.6M
   evals/run). The SD is not a constant of the trainer — it depends on the
   configuration, and it was three times larger at 64 agents (SD 0.0364, driven
   by one run at 0.5710) than at 128 (0.0102). Any single-run difference in this
   repository smaller than roughly 15 Elo is not evidence of anything.

4. **Two of the three telemetry series the project watches are not comparable
   across configurations.** `policy_top1` moved *opposite* to external strength
   across the `sims` factor at 15.7x the seed SD, and `policy_kl` was
   non-monotone in both experiments. Both are scored against the MCTS visit
   distribution, which the factors under test change. Within one fixed
   configuration they are still a legitimate learning curve; across
   configurations they are not.

---

## 6. Recommendation for the next training run

**Change two flags, and expect the gain to come mostly from the second.**

```sh
./build/chessrl az --run runs/<name> \
    --agents 16 \          # production used 128:  +25 Elo at equal games (point
                           #   estimate; Holm p 0.079, so the pair alone is not
                           #   significant -- see 2.5), and the same wall clock
    --sims 320 \           # production used 160 (binary default 64):  +21 Elo
                           #   over 160 and +45 over 64, at equal evaluations
                           #   AND equal wall clock
    --games-per-agent 6 \  # 48 games/generation
    --steps 100 \          # keeps steps-per-game (and so sample reuse) at the tested ratio
    --start mixed \        # az_default_cfg() zeroes its own start_mode assignment; pass it
    --ema-decay 0
```

Why 16 and not 4, when 4 scored highest against `random`: 4 agents needs 38%
more wall clock for the same number of games (12 pairings cannot fill 8 threads,
and 768 generations pay the per-generation serial cost 32 times over), its
advantage over 16 is 7.8 Elo with an exact permutation p of 0.32 — far inside
the 30-Elo MDE — and against the blunder-punishing opponent 16 was nominally
*ahead*. 16 agents takes the win at 8% over the fastest level's wall clock.

Why 320 and not higher: 320 is simply the largest value tested. The result says
the knee is **above** 320, not that it is **at** 320. The immediate follow-up is
`--factor sims --levels 320,640,1280 --seeds 5`, matched on evaluations, which
is about 45 minutes on a quiet machine and is the highest-value experiment left.

Other things worth doing, in order of expected value:

* **`--factor games-per-agent`** (2, 6, 18) at fixed `--agents`. This separates
  "more generations" from "fewer agents", which the present experiment
  deliberately confounds, and tells you whether to shrink the population or just
  shrink the generation.
* **A head-to-head confirmation run.** The two factors were each measured around
  a common base (`agents` at 160 sims, `sims` at 32 agents); their effects are
  not guaranteed to add. Train the recommended configuration and the current
  default at one equal budget, 3 seeds each, and compare externally.
* **Re-test the `buffer` claim.** At 16 agents with 48 games/generation, a
  50,000-position buffer spans 8.7 generations. The staleness reasoning in
  `src/az.c` is sound but its 18-point number came from one run per setting, and
  the buffer's span in generations is exactly what the `agents` factor was
  silently varying by 32x here.
* **Leave the draw penalty alone until it is re-tested.** `docs/ABLATION.md` is
  right that its supporting evidence is the shape of a seed, and the repetition
  rate — the outcome that claim was made on — has a pooled seed SD of 0.049 to
  0.064 in these two experiments, which is comparable to the entire effect the
  original sweep reported.

---

## 7. Limitations

* **The machine was not idle.** Sibling agent jobs pushed the load average to
  30.4 on 8 cores for roughly the first half of the `agents` experiment. This
  changes wall clock, not the budget (which is matched on games and
  evaluations), and it is spread across levels by the seed-major interleaving;
  see section 1. Throughput numbers are taken from the quiet tail only.
* **The outcome is the raw policy head without search.** A hyper-parameter could
  in principle improve the searched agent without improving the raw policy.
  `--eval-mode search` would measure that, at the cost of putting the
  hand-written material evaluation of `search.c` in front of the network, which
  `docs/FROM_SCRATCH.md` excludes from the shipped agent.
* **Five seeds is a small sample.** The MDEs are 30.2 Elo (`agents`) and 14.5
  Elo (`sims`); effects smaller than those would have been missed more often
  than not. That is why the MDE is printed next to every null.
* **`agents` confounds population size with generation count, replay staleness
  and PBT activity** (section 2.6). `sims` confounds search depth with total
  games and total optimiser steps (section 3.1) — unavoidably, since that is the
  trade being priced.
* **The two factors were measured independently, around a common base.**
  `agents` was varied at 160 sims and 32-agent-derived defaults; `sims` was
  varied at 32 agents. Nothing here shows their effects add, and the
  recommended configuration (16 agents, 320 sims) is a cell neither experiment
  actually ran. It needs the confirmation run described in section 6.
* **Two outcome scales were examined for `agents`** (`random`, then
  `randomplus-0.5`). The first was fixed in advance and is the primary result;
  the second is reported as a robustness check, with its own statistics, rather
  than folded into the headline.
* **Absolute strength is low everywhere.** Every model in both experiments sits
  between 0.45 and 0.55 against a random mover, where the project's best
  existing model sits at 0.6695. These are 9,216- and 3,072-to-15,360-game runs
  against a 729,600-game one. The *ranking* of configurations at this budget is
  what was measured; that the ranking persists at 100x the budget is an
  assumption, not a result.

---

## Provenance

| | |
| --- | --- |
| harness | `py/ablate.py` (unmodified), `py/benchmark.py` for every score |
| sources | the working tree as built at 2026-09-12 20:12; `git HEAD` was `e84c8c6` plus uncommitted changes when the runs started, and `95676a9` by the time this was written |
| binary | `build/chessrl` (md5 `cfcad16095bbc685c6a36c6d5fdd1b48`) + `build/libchessrl.dylib` (md5 `36d04e835f350246917f753274c04c91`), built from the same sources and snapshot-copied to a temp path before the runs, so that a concurrent relink could not kill a run |
| verification | `make -j8` clean (all 10 sources recompiled out of tree at the project's flags: 0 warnings), `make test` 7 checks / 0 failures, `make audit` shipped play path clean |
| machine | Apple M-series, 8 cores, macOS 23.6.0, Python 3.14.5 |
| raw output | `runs/ablation_agents.json`, `runs/ablation_sims.json`, `runs/ablation_agents_randomplus05.json` |
| dates | 2026-09-12 21:43 to 2026-09-13 00:09 local |
