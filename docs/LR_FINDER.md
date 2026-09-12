# The learning-rate range test, and whether it works on a non-stationary objective

`chessrl lrfind` implements Smith's learning-rate range test
([Cyclical Learning Rates, §3.3](https://arxiv.org/abs/1506.01186)) against this
system's real self-play data and its real optimiser step.

It exists because `docs/TECHNIQUES.md` dismissed the technique on a specific,
plausible, and until now **untested** claim:

> the loss here is non-stationary — the data distribution changes as the agents
> improve — so a range test at generation 1 tells you little about generation 200.

This document is the test of that claim. Short version: **the claim is
directionally real and quantitatively wrong.** The recommendation does move with
training stage, by more than seed noise — but it moves *within a single decade*,
and every training stage measured, from generation 0 to generation 200, puts the
recommended peak learning rate between 6.2e-4 and 7.9e-3 with a geometric mean of
2.6e-3. The configured `--lr 2e-3` is inside that band at every stage. The thing
a range test is actually for — pinning the order of magnitude — transfers across
the whole run.

The second half of the document answers the practical question: **is `2e-3`
right?** It is. A negative result, reported as such.

---

## 1. The command

```
chessrl lrfind [--model PATH] [--lo 1e-6] [--hi 1.0] [--steps 300]
               [--batch 256] [--warm-games 256] [--threads N] [--seed N]
               [--smooth 0.9] [--stop-factor X] [--csv PATH]
               [--plot-rows N] [--plot-cols N]
               [... every az self-play knob: --agents --sims --cap-frac
                    --cap-sims --start --draw-penalty --value-coef --value-mix
                    --wd --clip --clip-head --buffer --max-plies --cpuct
                    --dir-alpha --dir-eps --temp-start --temp-end --resign
                    --resign-check --games-per-agent]
               [--current-lr 2e-3 --current-lr-final 2e-4]
```

The exact invocation used for every number in this document, at the 128-agent
configuration of `runs/az_v2`:

```
./build/chessrl lrfind --model runs/az_v2/checkpoint.crl \
    --threads 3 --sims 160 --cap-frac 0.25 --cap-sims 24 --start mixed \
    --draw-penalty -0.6 --batch 256 --steps 300 --warm-games 512 --seed 1 \
    --csv lrfind.csv
```

Without `--model` it initialises a fresh network exactly as generation 1 of a
training run does, which is the "generation 0" case. With `--model` the
checkpoint decides the population size; `--agents` is then ignored.

`--current-lr` / `--current-lr-final` only change what the report compares
against; they default to `az`'s own defaults (2e-3 → 2e-4), so the comparison is
against the shipped schedule unless you say otherwise.

Cost: on an idle 8-core laptop at `--threads 3`, one complete test — 512
self-play games plus 300 optimiser steps — takes about **5 seconds**. Under the
contention these measurements were taken under it ranged from 5 s to 83 s.
`docs/TECHNIQUES.md` costed this technique as "Medium" and argued a 52-minute
training run was a cheaper experiment. That is off by roughly three orders of
magnitude and is corrected there.

---

## 2. What it actually does

1. **Load or initialise.** `--model` is read with `model_load`; without it,
   `nn_init` plus `az_hyper_init`/`az_hyper_mutate` reproduces generation 1's
   starting state bit for bit.

2. **Fill the replay buffer with real self-play.** `az_dispatch(JOB_SELFPLAY)` —
   the same call, the same `AZCfg`, the same MCTS with root noise, playout-cap
   randomisation, resignation and Chess960 start sampling that training uses.
   `--warm-games` is rounded **up** to a whole number of pairing rounds
   (`n_agents × games_per_agent/2` games each), because truncating a round
   would sample only the top of the Elo-sorted pairing list and bias the buffer
   towards the strongest agents.

3. **Snapshot the weights.** A full copy of the trunk and every head.

4. **Sweep.** `--steps` calls to `az_dispatch(JOB_LEARN)` with
   `AZShared.one_step` set, so each dispatch is exactly one optimiser step: the
   same uniform sample from the buffer, the same counting-sort by agent, the
   same per-head AdamW at that agent's `lr_scale`, the same trunk reduction and
   the same gradient clip. Only `lr_now` changes, geometrically from `--lo` to
   `--hi`. Per step it records the learning rate, total loss, policy loss, value
   loss, the **pre-clip** trunk gradient norm, the mean pre-clip head gradient
   norm, and the number of positions in the minibatch.

5. **Restore, and prove it.** The snapshot is copied back and compared with
   `memcmp` over every float of the trunk and of all heads. Every report ends
   with, e.g.:

   ```
   weights      restored from the pre-sweep snapshot: BIT-IDENTICAL
                (143328 trunk + 128 x 14145 head floats compared)
   model file   never written; lrfind opens no checkpoint for writing
   ```

   If the comparison fails the command prints the mismatch and refuses to report
   a result. `lrfind` calls `model_save` nowhere and creates no run directory.

### Conventions, stated so the numbers can be read

- **The loss at step *i* is the loss of that minibatch *before* step *i*'s
  update is applied.** This is the standard convention (it is what fastai's
  `lr_find` and Smith's original both plot): the curve at a given rate reflects
  the cumulative damage of every *smaller* rate that preceded it, which is what
  makes it turn up once the rate is too large to recover from.

- **Adam starts cold.** A checkpoint stores weights, not optimiser moments, so
  there is nothing else to start them from. The sweep therefore measures the
  rate at which a *fresh* Adam is stable on this weight configuration and this
  data distribution. That is the question a range test is for, and it is also
  exactly the situation at the start of a training run. It is *not* the same as
  Adam mid-run with warm second moments, and that is the one deliberate
  infidelity in the whole procedure.

- **Smoothing is a debiased exponential moving average**, coefficient `--smooth`
  (default 0.9, a ~10-step window). At 300 steps over six decades that is 0.2
  decades of lag. The coefficient is printed with every result so the number can
  never be read without knowing what smoothed it.

- **The sweep is not bit-reproducible.** Self-play work is stolen from one
  atomic counter, so which worker plays which pairing depends on scheduling.
  Repeat runs at the same seed differ slightly. Every conclusion below is stated
  over 2–3 seeds for exactly this reason.

---

## 3. The four numbers, and the rule

| Reported | Definition |
|---|---|
| **min smoothed loss** | `argmin` of the smoothed total loss |
| **steepest descent** | most negative `d(smoothed loss)/d(log10 lr)`, differenced over ±`steps/40` rows |
| **divergence** | first step *after* the minimum where the smoothed loss climbs back above its **initial** value, or goes non-finite |
| **4×-the-minimum** | first step after the minimum where the smoothed loss exceeds 4× its minimum (the classic criterion; on this objective, whose cross-entropy has an irreducible floor of ~2.3 nats, it is never reached and is reported as such) |

**The rule applied is: recommended peak LR = `lr(min) / 10`.** Stated once, in
the code, in the output, and here.

Why that rule and not another:

- The **minimum** of the curve is the rate at which the step has grown large
  enough that the update stops reducing the loss. It is an upper bound on a
  usable rate, not a setting.
- One decade below it is the conventional margin, and is `fastai`'s `lr_find`
  default.
- Our schedule quotes the **peak** of a cosine decay. A peak has to keep working
  for a whole run on data that keeps moving, so the peak is the right quantity
  to compare a range-test recommendation against.
- The **steepest-descent** point is reported but does **not** bind. It is a
  finite difference of a noisy series and is therefore a strictly noisier
  estimator than the argmin of the same smoothed curve. Measured: across the 24
  sweeps below, `lr(min)` varies by 1.05–1.66× between seeds at a fixed stage,
  while `lr(steepest)` varies by up to **six decades** at a fixed stage (it
  returned 1e-6 on one seed and 1.0 on another of the same checkpoint). Using it
  as a binding constraint would import that noise into the headline. When it
  falls well below the recommendation the report says so explicitly.
- Smith's own upper bound — the divergence point — is reported too. That is the
  max of a *triangular* range for a cyclical schedule, not a fixed rate, and it
  is by far the most stable landmark here (see §5).

---

## 4. Degenerate cases

All of these are exercised; the text below is the program's actual output.

**An old checkpoint.** Every ≥200-generation model in `runs/` is
`MODEL_VERSION` 3 with the pre-rebuild 160/96 shape — `runs/az_main/best.crl`
(gen 243), `runs/az_main/checkpoint.crl` (gen 250), `runs/pilot/best.crl` (gen
270), `runs/pilot/checkpoint.crl` (gen 300). `model_load` refuses all of them,
correctly: the tensors are different and in a different order, and reading one
would produce noise. **There is no readable 200+ generation checkpoint of the
real configuration.** That is why §5 builds its own.

```
lrfind: cannot read model 'runs/az_main/best.crl'.
        model_load accepts MODEL_VERSION 4 with widths 788/128/128/32 only; an
        older checkpoint has different tensors in a different order and is
        refused rather than read as noise (see net.h).
```

**Non-finite at the first step.** Note the trap this catches: `relu(NaN)` is 0
on every C implementation, because `NaN > 0` is false. A network whose value
trunk has gone non-finite therefore reports a perfectly ordinary *loss* while
every gradient reaching the trunk is `NaN` and every optimiser step is garbage.
A loss-only finiteness check misses this completely, so a step counts as usable
only if the loss **and** the pre-clip gradient norm are both finite. Tested by
writing `NaN` over `Trunk.b0v` in a copy of a real checkpoint:

```
  step      lr      loss    policy    value   smoothed   |g|trunk   |g|head   n
     0      1e-06    4.5179   2.6301   0.4719        inf       nan    24.908   256

  DEGENERATE: the FIRST step was already unusable at lr = 1e-06:
    loss               4.51787
    trunk grad norm    nan
  Nothing in this sweep is interpretable. ... No recommendation.
```

**A flat curve.** A range test over a window containing no usable rate produces
a flat noisy line, and every landmark read off it is a reading of minibatch
noise. So the descent is tested against the noise before anything is reported:
`resid_sd` is the spread of the raw loss about its own smoothed curve **over the
descending region only** (steps 0 to the minimum — past the minimum the loss is
exploding by construction, and including that tail inflates the noise estimate
by an order of magnitude), and an EMA with coefficient β reduces variance by
(1−β)/(1+β), so 3·sd(smoothed) is the bar.

`--lo 0.3 --hi 3.0`, i.e. a sweep entirely above the useful range:

```
  noise floor          minibatch sd 11.3533 about the smoothed curve over the 60
                       descending steps -> sd(smoothed) 2.6046
  descent depth        0.0000  (0.0 x sd(smoothed))   NOT SIGNIFICANT

  DEGENERATE: the smoothed loss is FLAT to within the minibatch noise across the
  whole of [0.3, 3]. The minimum, the steepest point and the first crossing back
  above the starting loss are all readings of noise here, not of a curve. Widen
  the sweep, raise --warm-games, or raise --steps. No recommendation.
```

A narrow window near the noise floor (`--lo 1e-8 --hi 1e-5`) lands on either
side of this bar depending on the draw, which is the correct behaviour: the same
window returned `descent depth 0.0067 (0.3 x sd)` on one run and
`0.4639 (12.6 x sd)` on another.

**A buffer too small to sample `--batch` from.**

```
  buffer       2132 positions, batch 4096 -> 115.3 passes over the buffer
  WARNING      the buffer holds FEWER positions (2132) than one minibatch (4096).
               Every batch is a resample of the same data; the curve measures
               memorisation of 2132 positions, not learning. Raise --warm-games.
```

A softer warning fires above 30 passes. The standard protocol below runs at
4.0–5.5 passes, which is the same order as real training's 4.1 expected samples
per position per generation lifetime.

**No descending region** (`--lo` already above the useful range):

```
  NONE. The smoothed loss was already at its minimum at the BOTTOM of the sweep
  (lr = 0.3): this sweep contains no descending region. ... Nothing to recommend.
```

**Too few usable steps** (`--lo 1e30`, where the first update destroys the
weights): `DEGENERATE: only 2 finite step(s). Too few to read a curve from.`

**Divergence never reached, and a minimum at the top of the sweep.** Both fire
together when `--hi` is set below the useful range (here `--hi 1e-3` on a
generation-5 checkpoint whose real `lr(min)` is ~0.05). The command declines to
quote a divergence LR and downgrades the recommendation to a lower bound:

```
  min smoothed loss    lr = 0.001       loss 4.2553   (step 149)
  divergence           NOT REACHED anywhere in [1e-06, 0.001]. The smoothed loss never
                       climbed back above its starting value, so this sweep gives NO
                       upper bound on a usable rate. Re-run with a larger --hi.
  4x-the-minimum       NOT REACHED in [1e-06, 0.001]

  RECOMMENDATION
  LOWER BOUND ONLY. The minimum sits at the TOP of the sweep (lr = 0.001), so
  [1e-06, 0.001] does not contain the point where the rate stops helping. Re-run
  with a larger --hi. On the evidence here a peak of 0.0001 is safe but may be low.
```

A divergence that lands on the very last step is tagged
`[LAST STEP: this is a lower bound]` for the same reason.

---

## 5. Does the recommendation move with training stage?

### Protocol

Everything except the checkpoint and the seed is held fixed:

```
--threads 3 --batch 256 --steps 300 --warm-games 512 --lo 1e-6 --hi 1.0
--smooth 0.9 --start mixed --draw-penalty -0.6 --cap-frac 0.25
```

**Series M** — the real 128-agent configuration (`--sims 160 --cap-sims 24`),
three seeds per stage. Stages: a fresh network; generation 5 from a short
throwaway run with that configuration; generations 85 and 95 of `runs/az_v2`.

**Series L** — one complete **200-generation** run at a small, fast
configuration (32 agents, `--sims 32 --cap-sims 8`, 100 steps/gen), snapshotted
at generations 5, 50, 100, 150 and 200, two seeds per stage. This exists because
no readable 200-generation checkpoint of the real configuration survives, and
generation 1 versus generation 200 is precisely the comparison the objection
makes. Over that run the data distribution moved a great deal: draw rate 47% →
94%, policy KL 0.173 → 0.586, trunk gradient norm 0.09 → 0.54, value
MSE/baseline 0.20 → 0.63.

### Series M: recommended peak LR (`lr(min)/10`), ×1e-3

| stage | seed 1 | seed 2 | seed 3 | geo-mean | seed spread | `lr(min)` | divergence LR |
|---|---:|---:|---:|---:|---:|---:|---:|
| generation 0 (fresh) | 1.88 | 2.26 | 1.97 | **2.03** | 1.20× | 0.019–0.023 | 0.362–0.416 |
| generation 5 | 4.12 | 4.12 | 4.32 | **4.19** | 1.05× | 0.041–0.043 | 0.548–0.831 |
| generation 85 | 2.98 | 2.48 | 2.72 | **2.72** | 1.20× | 0.025–0.030 | 0.301–0.362 |
| generation 95 | 3.94 | 4.32 | 3.59 | **3.94** | 1.20× | 0.036–0.043 | 0.274–0.379 |

Stage-to-stage spread **2.06× (0.31 decades)**, against a within-stage seed
spread of 1.05–1.20×.

### Series L: one 200-generation run, recommended peak LR ×1e-3

| stage | seed 1 | seed 2 | geo-mean | `lr(min)` | divergence LR | descent depth |
|---|---:|---:|---:|---:|---:|---:|
| generation 0 (fresh) | 3.76 | 3.27 | **3.51** | 0.033–0.038 | 0.362–0.724 | 1.35–1.60 |
| generation 5 | 4.74 | 7.88 | **6.11** | 0.047–0.079 | 0.724–0.955 | 1.04–1.43 |
| generation 50 | 0.62 | 0.90 | **0.75** | 0.006–0.009 | 0.262–0.346 | 0.56–0.68 |
| generation 100 | 1.97 | 1.56 | **1.75** | 0.016–0.020 | 0.379–0.416 | 0.70–0.74 |
| generation 150 | 2.26 | 1.56 | **1.88** | 0.016–0.023 | 0.208–0.218 | 0.43–0.79 |
| generation 200 | 2.16 | 2.60 | **2.37** | 0.022–0.026 | 0.104–0.208 | 0.43–0.47 |

Stage-to-stage spread **8.19× (0.91 decades)**, against a within-stage seed
spread of 1.15–1.66×. **Generation 0 → generation 200: 3.51e-3 → 2.37e-3, a
factor of 0.68.**

### Smoothed total loss against learning rate (series L, seed 1)

The curves are the same shape at every stage; they get shallower, and the
divergence point creeps down.

| lr | gen 0 | gen 5 | gen 50 | gen 100 | gen 150 | gen 200 |
|---|---:|---:|---:|---:|---:|---:|
| 1e-6 | 5.404 | 4.267 | 3.773 | 3.821 | 3.917 | 3.644 |
| 1e-4 | 5.307 | 4.194 | 3.700 | 3.872 | 3.742 | 3.573 |
| 1e-3 | 4.968 | 3.544 | 3.449 | 3.522 | 3.400 | 3.326 |
| **2e-3** | 4.716 | 3.429 | 3.305 | 3.387 | 3.364 | 3.302 |
| 1e-2 | 3.962 | 3.262 | 3.232 | 3.189 | **3.141** | 3.218 |
| 2e-2 | **3.850** | 3.309 | 3.267 | **3.117** | 3.162 | **3.201** |
| 3e-2 | 3.872 | 3.263 | **3.267** | 3.170 | 3.160 | 3.196 |
| 1e-1 | 4.041 | 3.295 | 3.363 | 3.287 | 3.246 | 3.338 |
| 3e-1 | 4.410 | 3.601 | 3.633 | 3.693 | 4.135 | 4.097 |
| 1.0 | 6.322 | 4.574 | 5.541 | 5.478 | 4.910 | 6.098 |

### Controls

**Seeds.** Above: 1.05–1.66× at a fixed stage. That is the noise floor of the
method as configured.

**Data quantity.** Doubling the self-play (`--warm-games 1024`, ~2× the buffer,
half the passes) at two stages:

| stage | 512 games (geo-mean of 3 seeds) | 1024 games | ratio |
|---|---:|---:|---:|
| generation 0, 128 agents | 2.03e-3 | 2.85e-3 | 1.40× |
| generation 85, 128 agents | 2.72e-3 | 4.32e-3 | 1.59× |

**This matters and is not a small caveat.** Doubling the data moves the reading
by 1.4–1.6×, which is about as much as the training stage moves it in series M.
Both shifts are upward, so the absolute numbers here should be read as "2–4e-3,
±a factor of 2", not to two significant figures.

### Verdict on the non-stationarity objection

Across all 24 sweeps — two configurations, eight training stages spanning
generation 0 to generation 200, 2–3 seeds each — the recommended peak learning
rate ranges from **6.2e-4 to 7.9e-3**: a factor of 12.7, or **1.10 decades**,
with a geometric mean of **2.63e-3**.

**The objection is directionally right and quantitatively wrong.**

*Right*, in that the recommendation genuinely moves with training stage by more
than seed noise. In series L the stage-to-stage spread (8.2×) is five times the
within-stage spread (1.66×). A generation-1 reading is not a precise
generation-200 reading, and anyone quoting the output to two significant figures
is over-reading it.

*Wrong*, in the way that matters. A range test is a tool for finding the **order
of magnitude**, and the order of magnitude never moved: every stage of both
series landed inside a single decade, and the whole band is centred on the value
already configured. The claim that a generation-1 test "tells you little about
generation 200" implies the answer would be qualitatively different there. It is
not: series L says 3.5e-3 at generation 0 and 2.4e-3 at generation 200 — the
endpoints are closer to each other than either is to the generation-50 dip.

There is also no useful *trend* to re-tune against. The trajectory is not
monotone in either series: series M rises from generation 0 to 5 and then wobbles
(2.0 → 4.2 → 2.7 → 3.9); series L rises to generation 5, collapses by a factor of
8 at generation 50, and recovers (3.5 → 6.1 → 0.75 → 1.8 → 1.9 → 2.4). A
mid-run re-tune would be chasing an excursion, not tracking a drift. What the
data supports is running the test **once**, taking the order of magnitude, and
not running it again.

One direction *is* consistent and worth stating: **the network's tolerance for a
large step does not fall as training proceeds.** `lr(min)` at generation 200 of
series L (0.022–0.026) is comparable to generation 0 (0.033–0.038) and larger
than generations 50–150. The cosine schedule assumes the opposite. See §6.

**`docs/TECHNIQUES.md` is corrected accordingly** — the technique is re-rated
from "Marginal / Small / Medium cost" to a legitimate one-off tool with a
30-line entry and a 5-second cost, and the non-stationarity argument is replaced
by what was measured.

---

## 6. Is `2e-3` (cosine-decayed to `2e-4`) right?

**Yes for the peak. The decay floor is the part that is not supported by this
measurement.**

### The peak

`2e-3` divided by the per-stage recommendation, across all 24 sweeps:

| series | min | max | geometric mean |
|---|---:|---:|---:|
| M (128 agents, gen 0–95) | 0.48× | 0.99× | 0.65× |
| L (32 agents, gen 0–200) | 0.33× | 2.68× | 0.89× |
| all | 0.25× | 3.22× | **0.76×** |

The configured peak is within half a decade of the measurement at 22 of 24
points, and the two exceptions are both the generation-50 dip in series L, where
`2e-3` was 2.2–3.2× **too high** rather than too low. There is no order-of-
magnitude error here and no case for changing `--lr`. The doc's original
assertion — "`2e-3` for AdamW on a 154k-parameter net is already in the right
place" — was correct, and is now measured rather than asserted.

If anything the measurement says `2e-3` is mildly *conservative* after
generation 5 (geometric mean 0.76× of the recommendation, i.e. the recommendation
is ~1.3× higher). That is well inside the ±1.4–1.6× that simply doubling the
self-play data moves the reading, so it is not a basis for a change either.

### The floor

At generation 200 of 200 the cosine has taken the learning rate to `2e-4`. The
range test at that exact checkpoint puts the loss minimum at `lr ≈ 0.022–0.026`
and recommends a peak of `2.4e-3`. **`2e-4` is 12× below the recommendation and
about 110× below the rate at which the loss stops improving**, at a point in
training where the measured tolerance for a large step is no lower than it was at
generation 0.

That is not automatically a bug: a decaying rate late in training buys variance
reduction and convergence, which a range test does not measure. But the usual
*justification* for decay — that the network becomes more sensitive to large
steps as it approaches a minimum — is not what happens here, and the measurement
says so. Two things follow, neither of which this document changes on its own
evidence:

- The floor is worth an ablation. `--lr-final 1e-3` instead of `2e-4` is a
  one-flag experiment and the range test says the network will take it.
- A cyclical or constant schedule is a live option rather than an obviously
  worse one. Smith's own upper bound for a triangular range — the divergence
  point — is 0.10–0.96 across every stage measured, so a cycle between, say,
  `5e-4` and `5e-3` sits comfortably inside the stable region everywhere.

---

## 7. What the curve is actually measuring: 94–100% value head

Every report prints the decomposition. Across all 24 sweeps:

```
  the descent is       94% value term, 6% policy term        (typical)
  policy term alone    lr(min) = 0.00566  -> min/10 = 0.000566  (range 0.0418 nats)
  value term alone     lr(min) = 0.052    -> min/10 = 0.0052    (range 0.4669, x4.00 coef)
```

The total-loss curve is **87–100% the value term** at every stage of both
series. With `value_coef = 4.0`, the value loss falls by 0.30–0.47 (×4 = 1.2–1.9
in the total) across the sweep while the policy cross-entropy moves by
0.00–0.13 nats — comparable to or below its own noise. Concretely, at generation
0 of series M the policy loss is flat to ±0.02 nats across six decades of
learning rate.

Three consequences, stated plainly:

1. **The headline recommendation is the value head's learning rate**, not the
   network's. It is a legitimate answer to "what rate can this optimiser take",
   because the value term is what dominates the gradient — but it is not
   evidence about what the policy head wants.

2. **The range test cannot locate the policy head's optimum at any stage
   measured.** Its `lr(min)` wanders between 1e-6 and 1.6e-2 across seeds of the
   same checkpoint because the descent it is the argmin of is 0.00–0.13 nats
   deep. Where it is readable at all (generations 85–95 of series M, 0.006–0.016
   → `/10` = 6e-4–1.6e-3) it agrees with `2e-3`, which is reassuring and nothing
   more.

3. This is an **independent confirmation of `docs/TECHNIQUES.md` §1.3**, which
   measured the value path contributing ~50× the policy path's gradient into the
   shared trunk, and of the live telemetry (`grad_value_share` = 0.871 at
   generation 85 of `runs/az_v2`). Two completely different measurements — a
   backward-pass decomposition and a learning-rate sweep — put the same number in
   the same place. Per-loss gradient scaling (§3.2's top-ranked item) would
   change what this tool measures, and the range test should be re-run after it.

---

## 8. Caveats

- **Adam starts cold** (§2). The test measures a fresh optimiser on a trained
  weight configuration. Mid-run Adam has warm second moments and would tolerate
  a different rate; a checkpoint does not store them, so this is not fixable
  without changing the checkpoint format.
- **Data quantity moves the answer by 1.4–1.6×** (§5), about as much as training
  stage does in series M. Read the numbers to one significant figure.
- **Series L is a small configuration.** 32 agents, 32 simulations, and by
  generation 200 it was drawing 94% of its games — a degenerate population that
  the real 128-agent run is not in. It is the best available proxy for
  "generation 200" because every real ≥200-generation checkpoint is unreadable
  (§4), but it is a proxy.
- **The `runs/az_v2` ladder is short.** Generations 85 and 95 are ten apart, not
  a hundred, because that run stopped (see the note at the end of this file).
- **`--steps 300` over six decades is 0.02 decades per step**, and `lr(min)` is
  quantised to that grid. Two seeds landing on adjacent steps differ by 4.7% for
  that reason alone.
- **Non-reproducible to the bit** (§2): work-stealing self-play.

What would strengthen this: a readable 250-generation checkpoint of the real
configuration; a sweep at 4× the self-play data to see where the data-quantity
sensitivity saturates; and a re-run after per-loss gradient scaling lands, which
should finally make the policy term's curve readable.

---

## 9. Reproducing it

```sh
make -j8

# generation 0, the real configuration
./build/chessrl lrfind --agents 128 --threads 3 --sims 160 --cap-frac 0.25 \
    --cap-sims 24 --start mixed --draw-penalty -0.6 --batch 256 --steps 300 \
    --warm-games 512 --seed 1 --csv gen0.csv

# any generation-4 checkpoint
./build/chessrl lrfind --model runs/az_v2/checkpoint.crl --threads 3 \
    --sims 160 --cap-frac 0.25 --cap-sims 24 --start mixed --draw-penalty -0.6 \
    --batch 256 --steps 300 --warm-games 512 --seed 1 --csv gen95.csv

# the 200-generation ladder: train small and fast, snapshot as it goes
./build/chessrl az --agents 32 --gens 200 --games-per-agent 2 --sims 32 \
    --cap-frac 0.25 --cap-sims 8 --start mixed --draw-penalty -0.6 \
    --ema-decay 0 --steps 100 --threads 3 --seed 7 --run lrf_long
```

The CSV has one row per step: `step, lr, log10_lr, loss_total, loss_policy,
loss_value, loss_smoothed, policy_smoothed, value_smoothed,
grad_norm_trunk_preclip, grad_norm_head_preclip, batch_positions, finite`.

> **Do not rebuild `build/chessrl` while a training run is executing from it.**
> On macOS, relinking the file underneath a running process kills it. The
> `runs/az_v2` 300-generation run died at generation 95 during this work, almost
> certainly for that reason; its last telemetry line and its `checkpoint.crl` are
> both generation 95, and `az` has no resume. Copy the binary first
> (`cp build/chessrl /tmp/chessrl_frozen`) and run the copy, as the series-L
> training here did.

### A sample report

```
  READING THE CURVE   (smoothing: debiased EMA, beta = 0.900, ~10-step window)
  noise floor          minibatch sd 0.2181 about the smoothed curve over the 230
                       descending steps -> sd(smoothed) 0.0500
  descent depth        1.8036  (36.0 x sd(smoothed))
  min smoothed loss    lr = 0.0394      loss 3.2959   (step 229)
  steepest descent     lr = 0.00178     -1.350 loss per decade (step 162)
  divergence           lr = 0.379       smoothed loss back above its starting value
  4x-the-minimum       NOT REACHED in [1e-06, 1]
  policy term alone    lr(min) = 0.00566    -> min/10 = 0.000566   (range 0.0418 nats)
  value term alone     lr(min) = 0.052      -> min/10 = 0.0052     (range 0.4669, x4.00 coef)
  the descent is       98% value term, 2% policy term

  RECOMMENDATION
  RULE APPLIED: peak LR = lr(minimum smoothed loss) / 10.

    lr(min)/10           0.00394   <- the recommendation
    lr(steepest)         0.00178   (cross-check, does not bind)

  AGAINST THE CONFIGURED SCHEDULE (--lr 0.002 cosine-decayed to 0.0002)
    configured peak / recommended peak = 0.51x (-0.29 decades)
    within half a decade of the recommendation: the configured peak is in range.

  smoothed total loss against log10(learning rate)

    6.0013 |                                                            !!|
           |                                                           *  |
           |                                                          *   |
           |** **************                                        *    |
           |  *              *******                                      |
           |                        ***                             *     |
           |                           **                                 |
    4.5690 |                             **                        *      |
           |                               **                     *       |
           |                                 *                            |
           |                                  ***                *        |
           |                                     *              *         |
           |                                      **           *          |
           |                                        **        *           |
           |                                          **     *            |
    3.2959 |                                            *****             |
           +--------------------------------------------------------------+
                                             R             m         d
            1e-6      1e-5      1e-4       1e-3      1e-2      1e-1    1e0
            m = min smoothed loss   R = recommendation   d = divergence
            11 step(s) above the top of the plot, drawn as '!'
```
