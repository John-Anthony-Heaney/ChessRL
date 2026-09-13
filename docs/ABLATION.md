# Ablation: measuring hyper-parameters against seed noise

`py/ablate.py` answers one question properly: **does changing this
hyper-parameter do anything, or is the difference you saw the seed?**

```sh
python3 py/ablate.py --list-factors
python3 py/ablate.py --factor agents --levels 8,32,128 --seeds 5 --dry-run
python3 py/ablate.py --factor agents --levels 8,32,128 --seeds 5 \
                     --budget-games 12288 --out runs/ablation_agents.json
```

Standard library only, like everything else in `py/`.

---

## 1. Why the earlier sweeps do not settle anything

Henderson, Islam, Bachman, Pineau, Precup & Meger, *Deep Reinforcement Learning
that Matters* (AAAI 2018, [arXiv:1709.06560](https://arxiv.org/abs/1709.06560))
ran the same algorithm with different random seeds and found the spread between
seed groups exceeded the gap between the algorithms people were publishing at
the time. Their conclusion is uncomfortable and correct: **with one run per
configuration you cannot tell an algorithmic effect from a seed.**

Every hyper-parameter comparison in this project so far is one run per
configuration. The sweep in `docs/ALGORITHM.md` is a single 8-cell grid, one
run per cell. The comments in `src/az.c` that say `MEASURED, not guessed` are
each a single 30- or 40-generation run. Measured is better than guessed, and
those comments are honest about their evidence — but a single run is a sample of
size one, and a sample of size one has no standard deviation, so there is
nothing to compare a difference against.

This is worse here than in a typical RL paper, because **the trainer is not even
reproducible at a fixed seed.** Two identical invocations of

```sh
./build/chessrl az --agents 8 --gens 6 --sims 16 --games-per-agent 6 \
    --threads 4 --steps 20 --seed 4242 --run <dir>
```

produce different checkpoints (different MD5, `elo_best` 1534.5 vs 1526.5,
`policy_kl` 0.705 vs 0.679). The per-thread trunk-gradient buffers are reduced
in whatever order the threads finish, and floating-point addition is not
associative. So the run-to-run variance the harness measures contains both the
seed and the scheduler; you cannot escape it by fixing the seed, and a
single-run number cannot be reproduced even by its own author.

---

## 2. The three rules

### 2.1 Compute-matched comparisons, never equal generations

`src/az.c` plays `n_agents * (games_per_agent / 2)` games per generation. At
6 games per agent that is **24** games per generation with 8 agents and **384**
with 128. Running both for the same number of generations compares two runs
that differ 16x in how much chess they have seen. Whatever you find is a
difference in experience, not in population size.

So `--budget-games` is fixed and the **generation count is derived from it**:

| level | games/gen | generations at a 12,288-game budget |
| ---: | ---: | ---: |
| 8 | 24 | 512 |
| 32 | 96 | 128 |
| 128 | 384 | 32 |

The default budget of 12,288 is chosen so it divides exactly into whole
generations at all three default levels; other budgets are rounded to whole
generations and the harness warns when the rounding moves the total by more
than 2%.

**When the factor changes the cost of a game, games are the wrong unit.**
Doubling `--sims` doubles the network evaluations per move, so equal games means
double the compute. For `sims` and `cap-frac` the harness matches on **total
network evaluations** instead (`--match evals`, chosen automatically). The basis
is printed at the top of every plan and stored in the JSON, because a
compute-matched result and an experience-matched result are different claims and
must not be confused.

There is no basis that matches everything. Matching evaluations makes total
games differ; matching games makes generations differ, which makes replay
staleness in generations differ. The harness prints a **`WHAT IS NOT MATCHED`**
block listing every derived quantity that varies more than 5% across levels, and
what that costs you. Read it before you believe the verdict.

Three specific interactions, handled explicitly:

* **Optimiser steps.** At equal games, an 8-agent run with a fixed
  `--steps 200` per generation takes 16x as many gradient steps as a 128-agent
  run. `--match-steps proportional` (the default when the factor changes
  games/generation) scales `--steps` with games/generation, so total steps *and*
  the sample-reuse ratio match too. `--match-steps fixed` keeps the per-
  generation count and lets the totals diverge. Both are defensible; only
  silence is not.
* **Replay buffer.** `--buffer` is a capacity in *positions*. Held fixed, 50,000
  positions is ~14.5 generations of retention at 8 agents and ~0.9 at 128 — the
  buffer is not even decorrelating at the top level. `--scale-buffer` scales
  capacity with games/generation instead, holding staleness constant in
  generations and letting capacity vary. Pick one and say which.
* **Hall of fame.** A snapshot every 10 generations is 51 frozen opponents over
  512 generations and 3 over 32. `--scale-hof` (**on by default**) scales
  `--hof-every` so snapshots per game are constant; `--no-scale-hof` leaves it.

### 2.2 Evaluation must be external

The trainer's Elo is seeded at 1500 and anchored by each run's **own** hall of
fame (`docs/ALGORITHM.md`). Two runs' internal Elos are therefore two different
scales that happen to share a unit, and ranking configurations by internal Elo
measures how fast a population pulled away from its own past, not how well it
plays chess. Using it would invalidate the entire experiment.

Every finished model is instead scored by `py/benchmark.py` against a **fixed
common opponent**, colour-reversed pairs, over a fixed opening set.

* **Opponent: `random` by default.** Chosen because the score has to land away
  from 0% and 100% to carry information, and it does. Measured on two existing
  checkpoints with the policy head, 60 games each:

  | model | generations | vs `material` | vs `random` |
  | --- | ---: | ---: | ---: |
  | `runs/smoke/best.crl` | 16 | 5.0% | 44.2% |
  | `runs/az_v2/best.crl` | 95 | 25.8% | 63.3% |

  `material` compresses short-budget models against the floor, where the
  variance is bounded and the measurement loses power. `random` spans 44% to
  63% over the same pair of models with plenty of headroom left. Use
  `--opponent material` for long budgets and
  `--opponent 'uci:/opt/homebrew/bin/stockfish' --opp-go nodes=100` once models
  are strong enough to need it. The harness warns if every level lands under 3%
  or over 97%, which means the opponent is the wrong one and the comparison is
  worthless.
* **Our side: the raw policy head, argmax, no search** (`--eval-mode policy`).
  The factors being ablated are training hyper-parameters, and the thing they
  affect is the network. Putting the alpha-beta searcher in front of it adds a
  hand-written material evaluation (`docs/FROM_SCRATCH.md` keeps `search.c` only
  as a baseline) and, on the evidence in `docs/BENCHMARK.md`, several hundred
  Elo of strength that has nothing to do with what was learned. It is also ~40
  games/second, so 800 games per model costs about 20 seconds.
* **Model file: `checkpoint.crl`**, which is the model after the full budget.
  `best.crl` is early-stopped on the run's **internal** Elo, so which generation
  it comes from is decided by a quantity that is not comparable across runs.
  `--ema-decay 0` is part of the base configuration for the same reason: with
  the EMA on, `best.crl` is whichever of the raw and averaged weights won an
  8-game head-to-head, which is a coin flip inserted between training and
  measurement.
* **One evaluation seed for every model** (`--eval-seed`, default 20260911).
  Every model plays the same openings against the same opponent, so the
  comparison is *paired*: the part of the evaluation error that comes from the
  opening set is common to all models and cancels out of the differences.

### 2.3 Seed variance is the headline

Each level is run with `--seeds` independent training seeds. The report leads
with the pooled **within-level standard deviation**, and expresses everything
else relative to it:

* per-level mean, SD, standard error and a Student-t 95% confidence interval,
  truncated at the 0/1 bounds of a score;
* the **between-level spread as a multiple of the within-level SD** — the single
  most informative number in the output. Below about 1x, the factor is doing
  less than the seed;
* **effect size before p-value**, always: the raw difference, the same
  difference in Elo, and Cohen's *d* against the pooled seed SD. A p-value
  without an effect size is not reported;
* two tests, because each covers the other's weakness. **Welch's t-test**
  (unequal variances, Welch–Satterthwaite df) assumes approximate normality of
  the means; the **permutation test** assumes nothing at all and, at 5 seeds per
  level, is computed *exactly* — all 252 relabellings are enumerated, so there
  is no Monte-Carlo error in the p-value. An omnibus permutation F-test across
  all levels at once is reported alongside, so a "significant" result cannot be
  manufactured by testing every pair and keeping the smallest p. Pairwise
  p-values carry a Holm–Bonferroni adjustment;
* the **minimum detectable effect** at the sample size used
  (α = 0.05, power = 0.80, two-sample): this is what makes a null result
  interpretable. "No significant difference" means "no difference larger than
  the MDE", not "no difference";
* the **evaluation SE per match** as a percentage of the seed SD. If that ratio
  approaches 100%, the "seed variance" is mostly measurement noise and the fix
  is more evaluation games, not more seeds.

The verdict is stated in words, one of:

* *moves the result by more than seed noise*, or
* *indistinguishable from seed noise at this sample size* — always accompanied
  by the MDE, so the reader knows what size of effect was ruled out, or
* *inconclusive* — fewer than two successful seeds at some level.

Fewer than 3 seeds triggers a warning; 5 is the minimum worth reporting.

---

## 3. An honest audit of the existing single-seed conclusions

Not all of them are equally weak. The distinction that matters is not "was it
one seed" (they all were) but **how large the effect was relative to plausible
noise, whether it was monotone across several levels, and whether there is a
mechanism.**

### Adequately supported, even at one seed

**Simulation count drives the repetition collapse** (`docs/ALGORITHM.md`).
Repetition draws fall 0.599 → 0.275 → 0.053 → 0.027 as sims go 48 → 96 → 160 →
256, with captures per game rising 7.6 → 10.8 → 17.9 → 18.7 and policy KL
falling in step. That is a **10x change in the outcome**, *monotone across four
levels*, corroborated by three other telemetry series moving coherently, with a
mechanism that predicts it: a policy that cannot see a plan repeats. Seed noise
would have to be implausibly large and implausibly ordered to produce that.
Treat it as real.

The caveat that remains is not about seeds: **that sweep was not
compute-matched.** All cells ran 20 generations at the same games/generation, so
the 256-sim cell used ~5x the compute of the 48-sim cell. The conclusion "more
search reduces repetition" survives, because repetition is a within-game
property of the search; the implied conclusion "160 sims is the right operating
point" does not follow from it, because the knee was read off a chart where the
x-axis is also a compute axis. `--factor sims` matches on evaluations and
re-tests it.

### Not supported

**The draw penalty.** At 48 sims, tripling the penalty from −0.1 to −0.6 moved
repetition from 0.599 to 0.516 — one run each. Nobody has measured the
run-to-run SD of the repetition rate under this trainer, so there is no
yardstick to hold that 0.083 against, and the `-0.3` cell in between sits at
0.616, i.e. **above** the −0.1 cell, which is not what a real monotone effect
looks like. This is exactly the shape of a difference that is a seed. Re-test:

```sh
python3 py/ablate.py --factor draw-penalty --levels -0.6,-0.3,-0.1,0.0 \
                     --seeds 5 --metric repetition
```

(The harness's primary outcome is external playing strength, but `--metric`
re-tests any last-generation telemetry statistic with the same machinery, so the
original claim can be re-run on the original outcome variable.)

**The replay-buffer size.** `src/az.c` records that raising `--buffer` from
50,000 to 400,000 "cost 18 points of policy top-1 agreement with the search over
a 30-generation run". One run per setting, two settings, no intermediate points,
and top-1 agreement is a quantity that drifts as the search sharpens. The
direction may well be right — the reasoning in the comment about staleness is
sound — but the 18 points are not established.

**The value coefficient.** `value_coef` 1 → 4 is recorded as improving value R²
from 0.54 to 0.56 and policy top-1 from 55% to 60%, over a single 40-generation
run. A 0.02 change in R² is well inside what one run tells you nothing about.

**Everything else in `az_default_cfg` marked MEASURED.** Same pattern, same
caveat. The defaults are not wrong — they are simply not yet distinguished from
the seed.

### The honest summary for the user's original question

Nobody in this project has yet measured whether the number of agents per
generation matters, because nobody has measured what a run-to-run standard
deviation looks like. That is the first thing `--factor agents` produces, and
the pooled seed SD is worth having even if the factor turns out to do nothing:
**every past and future single-run comparison in this repo should be read
against it.**

---

## 4. Practicalities

* **Runs are sequential.** Each training run uses every core, so running two at
  once would only make both slower and make the throughput numbers meaningless.
* **Levels are interleaved seed-major** — seed 1 at every level, then seed 2,
  and so on — so an experiment stopped early still has every level at the same
  number of seeds, which is the only state from which a partial result can be
  analysed.
* **Checkpointed after every single run.** The output JSON is rewritten
  atomically after each training run and each evaluation, with the statistics
  recomputed. Ctrl-C is safe; re-running the identical command resumes and
  reuses the completed runs. Resume is keyed on a SHA-1 of the factor, levels,
  base configuration, derived plan and evaluation settings, so changing the
  experiment starts a new one rather than silently mixing results.
* **A failed run is recorded and the experiment continues.** The exit code,
  stage and last lines of stderr go into the JSON, the run is excluded from the
  statistics, and the count of failures is printed with the verdict.
* **`--dry-run` prints the whole plan and costs nothing.** Always run it first.
  The wall-clock estimate is calibrated from this machine's own
  `runs/*/telemetry.jsonl` — median plies per game, median seconds per optimiser
  step and the distribution of `evals_per_sec` — and is quoted as a bracket
  between an idle machine and a contended one, not as a single number.
* **Output**: a terminal table, a JSON file (default
  `runs/ablation_<factor>.json`) carrying the plan, every run's command, score,
  benchmark detail and final telemetry, and an appended section in this file.
* **The binary and the shared library must be built from the same sources.**
  `build/chessrl` trains the models and `build/libchessrl.dylib` evaluates them;
  a model written under one net layout and read under another is silently
  meaningless. The harness refuses to start if either is older than `src/`.

### What the headline experiment costs

`--factor agents --levels 8,32,128 --seeds 5` at the default 12,288-game budget
is 15 training runs, 184,320 self-play games and 4.24 billion network
evaluations, plus 12,000 evaluation games:

| machine state | evals/s | wall clock |
| --- | ---: | ---: |
| idle (p90 of past telemetry) | 792,000 | ~1h34m |
| typical (median) | 729,000 | **~1h53m** |
| contended (p5) | 285,000 | ~4h13m |

Every other factor at 5 seeds costs about the same: 1h50m–2h00m for a 3-level
factor, ~2h30m for the 4-level `draw-penalty`. Re-run `--dry-run` on the day;
the estimate re-calibrates itself from whatever telemetry exists by then.

### Base configuration

Every level varies exactly one flag; all of these are held fixed and recorded in
the JSON. It departs from `az_default_cfg` in four places, each deliberate:

| setting | value | why not the binary default |
| --- | --- | --- |
| `--sims` | 160 | 64 sits in the shuffling regime the sweep above diagnosed; 160 is the operating point the project actually uses |
| `--games-per-agent` | 6 | makes games/generation `agents * 3`: 24 / 96 / 384 |
| `--ema-decay` | 0 | removes the raw-vs-EMA head-to-head coin flip from between training and measurement |
| `--start` | mixed (explicit) | see below |

**A bug worth knowing about.** `az_default_cfg()` in `src/az.c` assigns
`c->start_mode = AZ_START_MIXED` *before* `memset(c, 0, sizeof *c)`, so the
assignment is immediately zeroed and the compiled-in default is actually
`AZ_START_CLASSICAL` (and the write happens before the `if (!c) return;` null
check). `ablate.py` passes `--start mixed` explicitly so the ablation is not
silently classical-only. This file does not fix it — `src/` is owned elsewhere.

### Known limitations

* Training is not reproducible at a fixed seed (§1), so "seed variance" is
  really run-to-run variance. This makes the multi-seed design more necessary,
  not less, but it means a "seed" here is a replicate rather than a label.
* The outcome is the **policy head without search**. A hyper-parameter could in
  principle improve the searched agent without improving the raw policy;
  `--eval-mode search` measures that instead, at the cost of mixing in the
  hand-written evaluation that `docs/FROM_SCRATCH.md` excludes from the shipped
  agent.
* At small populations `cull_frac 0.10` of 8 agents culls zero agents, so PBT is
  inert at that level. The harness warns; it is arguably part of what "fewer
  agents" means, but it is a confound if you thought you were varying only the
  number of opponents.
* Five seeds is a small sample. The MDE is printed precisely so that the size of
  what you cannot see is on the page next to what you can.

---

# Results

<!-- ablate.py appends results below this line -->

### `asym-frac` -- 2026-09-13 02:23:16Z

fraction of self-play games played with unequal simulation budgets.  3 levels x 5 seeds = 15 runs, compute-matched at 238,110,720 network evaluations per run.
Evaluation: 800 games against `random`, raw policy head, argmax, colour-reversed pairs, evaluation seed 20260911 common to every model.
Outcome variable: external score vs the reference opponent (`--metric score`).
Command: `python3 py/ablate.py --factor asym-frac --levels 0,0.25,0.5 --seeds 5 --out runs/ablation_asym_frac.json`

| level | generations | games/run | n seeds | mean | seed SD | 95% CI | min | max |
| --- | ---: | ---: | ---: | ---: | ---: | :---: | ---: | ---: |
| 0.0 | 128 | 12,288 | 5 | 0.6238 | 0.0263 | 0.5911 - 0.6564 | 0.5869 | 0.6569 |
| 0.25 | 141 | 13,536 | 5 | 0.6181 | 0.0218 | 0.5911 - 0.6451 | 0.5938 | 0.6438 |
| 0.5 | 158 | 15,168 | 5 | 0.6040 | 0.0253 | 0.5726 - 0.6354 | 0.5769 | 0.6400 |

| comparison | difference | in Elo | Cohen's d | Welch p | permutation p | Holm p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.0 vs 0.25 | +0.0056 | +4 | +0.23 | 0.7226 | 0.7222 | 0.8333 |
| 0.0 vs 0.5 | +0.0198 | +14 | +0.80 | 0.2611 | 0.2778 | 0.8333 |
| 0.25 vs 0.5 | +0.0141 | +10 | +0.58 | 0.3721 | 0.3413 | 0.8333 |

* **Pooled within-level (seed) SD: 0.0245 score points = 18.0 Elo.**
* Between-level spread: 0.0198 = 14.5 Elo -- 0.80x the seed SD.
* Omnibus permutation test across all levels: p = 0.4433.
* Minimum detectable effect at n=5, alpha=0.05, power=0.80: **0.0496 score points = 36 Elo**.
* Evaluation SE per match: 0.0075, i.e. 31% of the seed SD.
* Not matched: total self-play games varies 1.2x across levels -- levels see different amounts of EXPERIENCE (the price of matching on evaluations).
* Not matched: total optimiser steps varies 1.2x across levels -- levels take different numbers of gradient steps on the same data.
* Not matched: hall-of-fame snapshots varies 1.2x across levels -- the frozen-opponent pool differs in size.
* Not matched: generations varies 1.2x across levels -- the cosine LR schedule is stretched over a different number of generations (its SHAPE in fraction-of-training is identical).

**Verdict: `asym-frac` indistinguishable from seed noise at this sample size.**  omnibus permutation p = 0.4433, no pairwise comparison survives Holm correction.  The largest between-level difference is 0.0198, which is 0.80x the pooled seed SD (0.0245) and 0.40x the minimum detectable effect (0.0496).  An effect smaller than the MDE would have been missed more often than not.
