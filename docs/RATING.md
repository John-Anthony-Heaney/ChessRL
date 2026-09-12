# RATING -- the internal Elo was measuring nothing, and what replaced it

> **Every internal Elo figure this project has ever reported -- 2157, 2629,
> 3348 -- is inflated. None of them is a strength measurement.** They are the
> output of a rating scheme that adds points at a fixed rate whether or not the
> agent is learning anything, and the proof is below: a population whose
> weights are frozen, which therefore *cannot* improve, gains rating linearly
> for five hundred generations.
>
> Only the external benchmarks -- `py/benchmark.py` against Stockfish, and the
> ladder in `docs/BENCHMARK.md` -- were ever measurements of strength. Nothing
> in this document changes them. What it changes is that there is now an
> internal number that is also honest.

---

## 1. The symptom

Over the first 1841 generations of `runs/az_hour` -- the project's longest
run, still going -- the reported best-agent Elo rose from 1554 to 4041. Fitted against generation number:

| fit | R^2 |
|---|---|
| straight line, slope **+1.220 Elo/generation** | **0.996** |
| logarithm | 0.780 |

A learning curve is not a straight line. Learning saturates; this did not. And
over the same generations every measure of actual learning had stopped moving:

| gens | elo_best | elo_mean | policy KL | policy top-1 | value MSE / baseline |
|---:|---:|---:|---:|---:|---:|
| 1-230      | 1884 | 1690 | 0.363 | 0.398 | 0.496 |
| 231-460    | 2217 | 2024 | 0.310 | 0.422 | 0.576 |
| 461-690    | 2434 | 2234 | 0.316 | 0.472 | 0.595 |
| 691-920    | 2680 | 2475 | 0.326 | 0.495 | 0.637 |
| 921-1150   | 3007 | 2783 | 0.330 | 0.491 | 0.643 |
| 1151-1380  | 3280 | 3050 | 0.333 | 0.492 | 0.658 |
| 1381-1610  | 3570 | 3342 | 0.329 | 0.494 | 0.655 |
| 1611-1840  | 3870 | 3645 | 0.323 | 0.502 | 0.654 |

The policy KL -- the actual learning curve, see `docs/ALGORITHM.md` -- bottomed
out around generation 350 and slowly got worse. Top-1 agreement with the search
plateaued near 0.49 by generation 700. The value head degraded monotonically
against its own constant-predictor baseline. The rating climbed at a constant
rate through all of it, and `elo_mean` and `elo_p10` climbed at the same rate
(+1.193 and +1.173 per generation, R^2 0.997 each), so this was not one lucky
agent pulling away -- the whole population was floating upward together.

That last observation is the thread to pull. A rating scheme in which everyone
drifts up together is a scheme that is creating points.

---

## 2. The mechanism, measured

### 2.1 Between live agents, Elo is exactly zero-sum

`apply_elo()` (`src/az.c`, the legacy-rating section) is textbook incremental Elo with `K = 24`. For
a game between two live agents the two updates are

```
white:  +K (s  - e)
black:  +K ((1 - s) - (1 - e))  =  -K (s - e)
```

which sum to zero, exactly, for every game and every result. The population's
*total* rating cannot change through games it plays against itself. It can only
be redistributed.

This is not an argument, it is an invariant, and the code now measures it. Two
new telemetry fields:

* `elo_sum_delta` -- the actual change in `sum(elo[])` across `apply_elo()`.
* `elo_inject` -- the part of that change contributed by games against an
  opponent whose rating is *frozen*.

Over every generation of every run measured here the two agree to float
precision (the residual is the float rating array being summed in double):

```
gen  1  inject=  0.0000  sumdelta=  0.0001
gen  2  inject= 19.4676  sumdelta= 19.4674
gen  3  inject=-16.4751  sumdelta=-16.4752
gen  4  inject= -6.5028  sumdelta= -6.5030
...
```

Frozen opponents are therefore not *a* source of drift. They are the **only**
source of drift.

### 2.2 A frozen opponent makes the scheme non-zero-sum

The hall of fame holds snapshots of past champions, and `apply_elo()`
deliberately does not update their ratings -- correctly, since they are meant
to be fixed reference points. But that is precisely what breaks the
conservation law. When a live agent beats a frozen one, the live agent's rating
goes up and *nothing goes down*. Rating enters the population from nowhere.

By itself that would be self-limiting. A fixed set of frozen opponents at fixed
ratings is a ladder with a top: as the population climbs past them its expected
score approaches 1, `s - e` approaches 0, and the injection dies away. The
rating would saturate -- which, note, is exactly the shape a real learning
curve has.

It does not saturate. Something keeps refreshing the ladder.

### 2.3 The ratchet

In the hall-of-fame step of `az_run()`:

```c
e->elo = elo[gs.best_i];
```

A new snapshot is stamped with the **current best agent's rating** -- a number
that has already drifted. Two things follow, and they compound:

1. The snapshot's rating is frozen at the drifted value, so the drift is now
   permanent and can never be corrected.
2. The snapshot is a *fresh* opponent rated at the top of the population but
   beatable by it (the population goes on training; the snapshot does not). So
   `s - e` is positive again, and the injection restarts at full strength.

Every `hof_every` generations the ladder gets a new top rung, positioned at
whatever height the population has already drifted to. That is a ratchet: each
turn locks in the previous drift and sets up the next.

**The control experiment.** If the ratchet is what prevents saturation, then
taking exactly one snapshot -- at generation 1, when the rating is still the
honest 1500 seed -- should make the drift saturate. Run with
`--hof-every 100000` so the only snapshot is the generation-1 one, weights
frozen, everything else identical:

```
                                  legacy elo_mean        cumulative
                              slope /gen      R^2        injection
  hall of fame every 5 gens   +0.1756      0.983             2368
  ONE snapshot, ever          +0.0160      0.445              352
```

and the injection dies out exactly as predicted, within the first fifty
generations, as the population climbs past the one fixed rung:

| gens | injection/gen | gens | injection/gen |
|---:|---:|---:|---:|
| 1-50    | **+4.48** | 251-300 | +1.15 |
| 51-100  | +0.34 | 301-350 | +1.26 |
| 101-150 | -1.41 | 351-400 | +1.07 |
| 151-200 | -0.23 | 401-450 | +1.48 |
| 201-250 | -0.62 | 451-500 | -0.48 |

`elo_mean` over those ten bands: 1503, 1510, 1509, 1506, 1506, 1507, 1508,
1509, 1513, 1514. Flat after the first band, R^2 against a straight line 0.445
instead of 0.983.

So the frozen opponent by itself produces a bounded, saturating offset -- an
error, but a *bounded* error. It is the restamping that turns it into an
unbounded ramp. **The ratchet is the bug.**

### 2.4 What the cull/clone step does

Claim under test: a culled agent keeps its own (low) rating while receiving an
elite's head, and that *dampens* rather than inflates. **Verified, with a
caveat.**

*Direct channel: exactly zero, and now measured.* The evolution step copies
`heads`, `hypers` and the Adam moments; it never touches `elo[]`. The new
`elo_sum_carry` field is the change in `sum(elo[])` between the end of one
generation's rating step and the start of the next -- i.e. everything evolution
did -- and it is identically 0.0 in every generation of every run here. The
alternative, giving the clone the elite's rating, would add
`n_cull * (elite - victim)` points to the population total every generation,
straight into the number being reported. Keeping the victim's rating is the
non-inflating choice. That part of the diagnosis is right.

*Indirect channel: small but positive, i.e. it inflates a little.* An agent
holding an elite's weights at a culled agent's rating is **underrated**.
Against a frozen opponent it therefore scores better than its rating predicts,
`s - e > 0`, and the injection of section 2.2 gets larger. Against live
opponents this does nothing to the total (zero-sum), it only re-rates the clone
quickly, which is the intended behaviour.

Measured, with weights frozen (`--steps 0`) and clones made exact
(`--mutate-sigma 0`) so that cloning changes no agent's actual strength and the
only difference is the rating bookkeeping:

```
                   cumulative   injection   legacy elo_mean   anchored elo_mean
                    injection       /gen       slope /gen         slope /gen
  --cull 0               2368      +4.736     +0.1756           +0.0020 +/- 0.0024
  --cull 0.20            2805      +5.611     +0.2152           +0.0050 +/- 0.0028
```

(The two anchored columns are from the pair of runs made before the pin gate
was tightened -- see 5.3. Both were flat then; with the shipped gate the same
configuration is flatter still, -0.0064 +/- 0.0076. The legacy columns are
unaffected by that change.)

Cloning raises the injection rate by about **19%** and the legacy drift by
about **23%**. Part of that is the underrating effect above; part of it is
real, because selection genuinely does make the population slightly stronger
even with no gradient at all -- the raw score against the random mover rises
from 0.5212 to 0.5426 (about +15 Elo), which the anchored rating also picks up
as a higher *level*. What neither version does is give the anchored rating a
*slope*: both stay flat.

So: the cull/clone step is not the cause. It contributes a second-order
increase to the injection that the hall of fame is already producing. The
original diagnosis was right about the direction of the direct effect and
missed a smaller effect of the opposite sign; neither changes the conclusion
that the hall-of-fame ratchet is the mechanism.

---

## 3. The fix

The scheme has two defects and both have to go:

* it has **no fixed point** -- every rating is defined relative to other
  ratings that are themselves moving;
* it is a **running total** -- an error, once added, is never revisited.

### 3.1 Permanent anchors

Two players are added to the tournament whose strength is constant for the life
of the run and whose rating is pinned and never updated:

| anchor | what it is | rating |
|---|---|---|
| `random` | a uniform random legal mover | **0, by definition** |
| `gen0` | the generation-0 randomly initialised network, copied before the first optimiser step and never trained | measured once against `random`, then pinned |

Neither holds any chess knowledge. The random mover is `gen_legal()` plus a
coin; the gen-0 network is untrained weights. `docs/FROM_SCRATCH.md` is
untouched -- indeed `tests/test_scratch.c` already relies on both objects for
exactly this reason, as the strongest empirical check in that document is
"untrained weights must play no better than random".

Implementation notes that matter:

* The gen-0 anchor keeps its own frozen copy of the **trunk** as well as its
  head, because the trunk is shared and trained. It costs one extra `Trunk` +
  `Head` for the run.
* It plays exactly as a live agent plays: full simulation budget (never the
  playout cap, so its strength does not move with `--cap-frac`), the same root
  Dirichlet noise, the same temperature schedule. An earlier version searched
  it cleanly while the population searched under noise, and measured a
  ~300-point gap that was the noise rather than the training.
* Anchor games are **measurement games**: nothing they produce enters the
  replay buffer, and nothing they produce enters the population statistics
  (draw rate, average length, opening counts, KL, ...). Measuring the
  population must not change the population, and must not change any number a
  previous run was reported with.
* Neither side resigns in an anchor game. Letting only the live side resign
  would bias every anchor game toward the anchor.
* Anchor games *replace* population pairings rather than adding to the
  generation, so the measurement buys no extra games and the generation count
  is unchanged. It is not quite free: an anchor's move is searched
  synchronously rather than through the batched stepped search (the gen-0
  anchor does not share the population's trunk, and `mcts_batch_run()`
  evaluates one trunk per call), and with resignation off its games run longer.
  At the shipped 24 games out of a 384-game generation that is a few percent of
  the self-play time.

Only the random mover fixes the gauge. `gen0`'s rating is a *measurement*, made
once against `random` at the start of the run and pinned thereafter, so the two
pins are consistent by construction rather than two independent assertions
about a scale. Where the match is 100% one way the adjusted score
`(w + 0.5) / (n + 1)` is used and the result is reported as
`"bounded": 1` -- a lower bound, not an estimate.

### 3.2 Rate by global fit, not by increments

Each generation, every rating is re-estimated **from scratch** by
Bradley-Terry / logistic maximum likelihood over a sliding window of recent
results, with the anchors held fixed (`elofit_run` in `src/az.c`):

```
P(i beats j) = gamma_i / (gamma_i + gamma_j),    gamma_i = 10^(rating_i / 400)
```

which is standard logistic Elo written multiplicatively. Draws count as half a
win and half a loss. The solver is minorisation-maximisation (Zermelo 1929;
Hunter 2004), one sweep of

```
gamma_i  <-  W_i / sum over i's games of 1 / (gamma_i + gamma_opponent)
```

over the free players. Every sweep increases the log-likelihood, there is no
step size to tune, and a few hundred sweeps over a few thousand games costs
microseconds against a self-play generation costing seconds. `--elo-iters`
(default 400) caps the sweeps; convergence usually happens in 100-300.

Because it is a fit and not a total, **there is nowhere for an error to
accumulate**. Last generation's number has no authority over this one's; it is
used only as a warm start for the iteration.

The degenerate cases all occur in practice and none of them is allowed to emit
an infinity:

| case | what happens | reported as |
|---|---|---|
| no games in the window | not rated; the previous value is held | `elo_fit_unrated` |
| won or lost **every** game | the likelihood has no interior maximum; the estimate is clamped to `[-4000, 12000]` | `elo_fit_bounded` -- **a bound, not a rating** |
| no path of games to any pinned player | the component's ratings are determined only up to an additive constant, which is the drift being fixed; the component is left alone | `elo_fit_unanchored` |

The all-wins/all-losses case is decided **from the data** -- `W_i <= 0` or
`W_i >= n_i` -- before a single sweep runs, not by comparing the iterate to the
clamp afterwards. The first version did the latter, and `400 * log10(0)`
landing exactly *on* the clamp slipped through a `<` comparison and froze
hall-of-fame entries at -4000. Connectivity is decided by union-find over the
window's games.

### 3.3 Hall-of-fame ratings come from the fit, and freeze late

A snapshot's rating is now the fitted one, and once assigned it is frozen and
the fit treats it as data -- which is correct, and was never the problem. The
problem was the *value*.

But taking the fitted value **at the moment of the snapshot** would freeze a
different bias. The snapshot is the *best* agent, chosen as the maximum of `n`
noisy ratings, so its rating is biased upward by the winner's curse; and a bias
that compounds across snapshots is a ratchet of exactly the kind being removed.
So an entry stays a **free parameter** for `hof_pin_lag` generations (default:
the window length, so that by the time it freezes the window contains only
games played *after* the snapshot was selected) and must have at least
`HOF_PIN_MIN_GAMES` = 48 games in the window -- about +/-50 Elo of standard
error -- and a non-degenerate estimate; section 5.3 is the measurement that
set that number. An
entry that never qualifies simply stays free forever, which is harmless: free
players cannot inject drift, they merely do not extend the range the anchors
can measure.

Recycling a hall-of-fame slot -- which happens after `HOF_CAP * hof_every`
generations -- drops that slot's games from the window, so a new snapshot never
inherits its predecessor's record.

### 3.4 Pairing

`--anchor-games` pairing slots per generation (default 24) go to an anchor.
They are chosen by systematic sampling across the pairing list, which is sorted
strongest-first before the window shuffle, so the anchor games are spread over
the whole population rather than drawn from one end of it. Half go to the
anchor **nearest the population's current fitted strength**, the rest are split
evenly so that every anchor is linked to the population in every window. A
match that is 100% one way carries almost no information about a rating
difference -- the same problem the external Stockfish ladder has, and for the
same reason (`docs/BENCHMARK.md`, `docs/RATING_SCALES.md`).

A trickle of `gen0`-versus-`random` games keeps running after the calibration
quota is filled. Two players who cannot change must score the same against each
other forever; a number that is supposed to be constant is worth continuing to
check.

### 3.5 Telemetry

Everything old is still written, unchanged, so past runs stay comparable and
the drift stays visible. New fields, per generation:

| field | meaning |
|---|---|
| `elo_anchored_best` / `_mean` / `_p10` | the fitted rating -- **the number to believe** |
| `elo_anchor[]` | per anchor: `name`, pinned `elo`, `bounded`, `games`, and the raw `score` the population took |
| `anchor_cal_score` / `anchor_cal_total` | `gen0` vs `random`: two frozen players, so this can never move |
| `elo_fit_games` / `_iters` / `_delta` | window size, sweeps taken, final convergence |
| `elo_fit_unrated` / `_bounded` / `_unanchored` | the degenerate cases, counted, never hidden |
| `hof_n` / `hof_pinned` / `hof_fit_min` / `_mean` / `_max` | the fitted reference set |
| `elo_inject` / `elo_sum_delta` / `elo_sum` / `elo_sum_carry` | the legacy scheme, measured |
| `hof_elo_min` / `hof_elo_max` | the legacy hall-of-fame ratings, i.e. the ratchet |

**The single most honest signal in the file is `elo_anchor[].score`.** It is a
raw fraction of a point against a player whose strength is a constant of
nature. No model, no fit, no scale, nothing to drift. If it is not moving, the
population is not improving, whatever any rating says.

---

## 4. The acid test

A population that is not improving must not gain rating. So: freeze the weights
and see.

```
chessrl az --agents 24 --gens 500 --games-per-agent 8 --sims 8 --max-plies 80 \
           --steps 0 --cull 0 --ema-decay 0 --hof-every 5 --seed 11
```

`--steps 0` means no optimiser step is ever taken; `--cull 0` means no clone
and no mutation. Not "learning slowly" -- the weights are bit-for-bit constant
for five hundred generations. Anything the rating does is drift.

A note on the error bars below. The fit window is 8 generations long, so
consecutive generations share most of their data and an ordinary-least-squares
standard error, which assumes independent points, is optimistic by roughly
sqrt(8). Both the "+/-" figures quoted and the sigma counts have that factor
applied. Also, the first `--elo-window` generations are a warm-up: at
generation 1 the window holds one generation of games instead of eight, so the
rating is not comparable until it has filled. The slopes below are unchanged
whether they are fitted from generation 1 or from generation 9.

### The old rating

```
                first     last   slope /gen            R^2
  elo_best     1521.6   1630.7   +0.1864 +/- 0.0091   0.869
  elo_mean     1500.0   1598.7   +0.1756 +/- 0.0028   0.983
  elo_p10      1488.4   1576.4   +0.1770 +/- 0.0062   0.930
```

The bug reproduces. A population that cannot change gains 99 points of mean
rating in 500 generations, in a straight line, R^2 = 0.983, 62 standard errors
from zero. Run without the anchor games at all (`--no-anchor-elo`, so the
generation is exactly what it always was) the slope is +0.2152, R^2 = 0.968 --
the same behaviour; the anchor games neither create it nor hide it.

The hall-of-fame legacy ratings climb with the population, from 1521.6 at the
first snapshot to 1664.8 at the last. That is the ratchet doing its work.

### The new rating

```
                         first     last   slope /gen             R^2
  elo_anchored_best       78.4     51.5   -0.0002 +/- 0.0141   0.000
  elo_anchored_mean      -17.5     22.0   -0.0064 +/- 0.0076   0.011
  elo_anchored_p10       -77.4      1.4   -0.0019 +/- 0.0096   0.001
```

**Flat.** Every slope is under one standard error from zero, R^2 against a
straight line is 0.011 or less, and the largest of them is 27 times smaller
than the legacy slope and points the other way: -3 Elo over 500 generations
against the legacy scheme's +88. Banded:

| gens | elo_best | elo_mean | **anchored best** | **anchored mean** |
|---:|---:|---:|---:|---:|
| 1-100   | 1556.1 | 1515.9 | 54.7 | 12.6 |
| 101-200 | 1584.0 | 1536.9 | 60.2 | 15.7 |
| 201-300 | 1600.3 | 1555.4 | 64.5 | 17.3 |
| 301-400 | 1612.7 | 1568.4 | 54.1 | 11.6 |
| 401-500 | 1635.3 | 1587.7 | 59.4 | 11.0 |

The left two columns rise by 79 and 72 points. The right two end where they
started. In this run no hall-of-fame entry ever reached the 48-game pin
threshold, so the entire scale is carried by the two anchors -- which is the
configuration to trust, and the one the shipped defaults produce.

### And the raw scores, which have no model in them at all

```
                        games    mean score   slope /gen
  population vs random   3732      0.5263     +0.000003 +/- 0.000045
  population vs gen0     8268      0.5092     -0.000019 +/- 0.000042
  gen0        vs random  1542      0.5074     -0.000016 +/- 0.000065
```

All three constant over 500 generations and 13,542 games. The last line is the
strongest single check in the file: two players who cannot change, scoring the
same against each other from the first generation to the five-hundredth. The
middle line is the second strongest -- a frozen population scoring 0.509
against a frozen network, i.e. dead level, which is what it is.

One incidental result worth recording: at the pinning generation the
generation-0 network had scored exactly 24.0 / 48 against a uniform random
legal mover, so its pin came out at exactly 0.0 -- at 8 simulations per move an
untrained network is precisely as strong as random. That is
`docs/FROM_SCRATCH.md`'s strongest empirical check falling out of the rating
machinery for free, and it agrees with `tests/test_scratch.c`, which measures
0.54 at 100 simulations.

### The same test at a larger search budget

120 generations, 32 simulations per move, weights frozen. Here the two anchors
are genuinely far apart -- the generation-0 network scores 0.682 against the
random mover, so it pins near +150 -- and the population sits between them:

```
                        games    mean score   slope /gen
  population vs random    744      0.6595     +0.00045 +/- 0.00087
  population vs gen0     2136      0.4928     +0.00007 +/- 0.00065
  gen0        vs random   402      0.6823     +0.00010 +/- 0.00111
```

`elo_anchored_mean` runs 67 at generation 1 (window one-eighth full) to about
120 from generation 20 onward; fitted from generation 21, once the window has
filled, the slope is **-0.014 +/- 0.142** per generation. Flat. The legacy
`elo_mean` over the same run: **+0.342 +/- 0.021**, R^2 = 0.949.

The 0.4928 in the middle line is worth pausing on. A frozen population scoring
0.4928 +/- 0.011 against a frozen network drawn from the same distribution is
the measurement harness certifying itself: whatever asymmetries exist between
how an anchor's move is computed (a blocking `mcts_search`) and how a live
agent's is (the batched stepped search), they are worth less than one
standard error.

---

## 5. A population that IS improving

The other half of the requirement: the anchored rating has to rise when
something real is happening.

### 5.1 Improvement with no gradient at all

The cleanest version of the question, because there is nothing else going on.
Weights frozen (`--steps 0`), but `--cull 0.20 --mutate-sigma 0`: every
generation the worst fifth of the population is replaced by an exact copy of a
randomly chosen elite. No optimiser step is ever taken and no weight is ever
perturbed. The only thing that changes is *which* of a fixed set of random
networks the population is made of -- and that is genuine improvement, because
some of those networks really are better than others. It also has a natural
end: after enough generations there is nothing left to select.

| gens | score vs random | **anchored mean** | legacy `elo_mean` |
|---:|---:|---:|---:|
| 1-50    | 0.5356 | **18.1** | 1510.4 |
| 51-100  | 0.5472 | **29.9** | 1532.3 |
| 101-150 | 0.5450 | **27.1** | 1546.7 |
| 151-200 | 0.5439 | **26.5** | 1558.4 |
| 201-250 | 0.5389 | **26.7** | 1573.7 |
| 251-300 | 0.5422 | **29.5** | 1583.2 |
| 301-350 | 0.5489 | **31.9** | 1590.4 |
| 351-400 | 0.5450 | **28.3** | 1592.7 |
| 401-450 | 0.5394 | **21.1** | 1604.1 |
| 451-500 | 0.5400 | **26.6** | 1614.4 |

The anchored rating rises by about 12 points over the first hundred
generations, in step with the raw score against the random mover, and then
stops -- because the improvement stopped. The band means have a standard error
of about 3 (the generation-to-generation sd of `elo_anchored_mean` is 8.0 and
the 8-generation window makes roughly six of every fifty generations
independent), so that rise is real and the subsequent flatness is real.

The legacy rating over the same 500 generations: +104 points, still going, at
the same rate at the end as at the beginning.

That is the shape asked for: **rises while something real is happening, levels
off when it stops.**

### 5.2 Gradient training, and what it actually showed

Three runs at increasing budget, each the largest of its kind that fits the
time available:

| run | gens | sims | policy KL | policy top-1 | value MSE / base | score vs random | **anchored mean** | legacy `elo_mean` |
|---|---:|---:|---|---|---|---|---|---|
| 80 gens  | 80 | 32 | 0.72 -> 0.50 | 0.34 -> 0.65 | 0.39 -> 0.82 | 0.544 -> 0.511 | **43.7 -> 7.3** | 1503.7 -> 1516.2 |
| 40 gens  | 40 | 96 | 0.53 -> 0.40 | 0.41 -> 0.58 | 0.20 -> 0.56 | 0.633 -> 0.646 | **133 -> 120** | 1494.3 -> 1490.7 |
| 30 gens  | 30 | 96 | 0.52 -> 0.32 | 0.42 -> 0.55 | 0.41 -> 0.76 | 0.662 -> 0.667 | **116 -> 103** | 1495.9 -> 1484.6 |

In all three the policy head learned: KL fell and top-1 agreement with the
search rose by 15-30 points. In none of them did the agent get *stronger*. The
raw score against a uniform random mover -- which has no model in it, no fit,
and nothing that can drift -- was flat in two and fell in the third. The value
head collapsed in all three, which is the likely reason: a search whose leaf
evaluations are collapsing toward the mean produces worse targets, whatever the
policy head is doing about matching them.

**I could not produce, within this compute budget, a short gradient run in
which the agent genuinely got stronger, so the "anchored rating rises with real
learning" half of the acid test rests on 5.1 rather than on 5.2.** That is
worth stating plainly rather than dressing up. It is also not a surprise: in
the project's own 1841-generation reference run, top-1 plateaued at generation
700 and the value head degraded monotonically for the eleven hundred
generations after that, while the reported Elo added another 1400 points.

What 5.2 *does* show is the discrimination the old number could not make. In
the 80-generation run the population genuinely got weaker -- 0.544 to 0.511
against the random mover, and an untrained network measured at 0.670 against
the same opponent at the same search budget in a frozen control, so the
direction is not in doubt. The anchored rating fell from 44 to 7. The legacy
rating rose. One of those two numbers was tracking the agent.

### 5.3 The noise floor, and a defect this found

With the shipped `--anchor-games 24` and an 8-generation window, about 190
anchor games back each generation's estimate and `elo_anchored_mean` has a
generation-to-generation standard deviation of **8 Elo** (`elo_anchored_best`,
being a maximum over the population, about 13). A trend has to clear that.
`--anchor-games` buys precision directly, at the cost of population games.

The first version of the hall-of-fame pin gate required only 8 games in the
window. Measured on a frozen population -- where every snapshot is by
construction the same strength -- the pinned set spread from **+61 to +404**,
and as those first entries landed they moved the population's own rating by
about 50 points over thirty generations before settling (slope over generations
1-120: +0.19 +/- 0.04 per generation; over 41-120, after the pins were in:
-0.04 +/- 0.06, i.e. flat). A bounded one-off level shift rather than a
ratchet -- but it is still wrong, and it is exactly the failure mode this
document exists to catch. The gate is now 48 games, about +/-50 Elo of standard
error, which on the shipped `--hof-pct` means few or no entries pin and the
scale is carried by the anchors alone. Less range, no invented precision.

---

## 6. What this does and does not measure

* **It is not a FIDE, chess.com or Lichess rating**, and it is not comparable
  to the Stockfish-ladder numbers in `docs/BENCHMARK.md`. It is a rating on a
  scale whose zero is a uniform random legal mover playing this trainer's
  configuration. Read `docs/RATING_SCALES.md` before converting anything.
* **The scale depends on the configuration.** `--sims`, `--dir-eps`,
  `--max-plies`, `--start` and the temperature schedule all change how strong
  any given network is. Two runs with different settings are not on the same
  scale, even though both are anchored. What is guaranteed is that *within* a
  run the scale cannot move.
* **It saturates, and says so.** Once the population beats both anchors ~100%
  of the time, the anchor link carries no information and the rating is
  determined only through the hall of fame. `elo_fit_bounded` and the raw
  `elo_anchor[].score` values make that state visible. At that point the number
  is a lower bound, and the way to extend it is more hall-of-fame games
  (`--hof-pct`) so its entries can be rated precisely enough to pin, or
  stronger anchors.
* **Frozen-versus-frozen games are valid evidence forever**, and this
  implementation does not yet exploit that. A hall-of-fame entry's games
  against another frozen entry or against an anchor never go stale, so they
  could be kept far outside the 8-generation window. With the shipped
  `--hof-pct` and the 48-game pin threshold, usually **no** snapshot ever
  accumulates enough games to be pinned, so the scale rests entirely on the two
  anchors -- safe, but limited to the range they span. Keeping frozen-vs-frozen
  results permanently, and routing a few games between snapshots, is the
  obvious next improvement and the thing that would let the anchored scale
  extend cleanly past the anchors.
* **The first `--elo-window` generations are a warm-up.** At generation 1 the
  window holds one generation of games instead of eight, and the gen-0 anchor
  is not pinned until its calibration quota is filled. `elo_fit_games` says how
  full the window is; do not read a trend across the warm-up.
* **Selection is unaffected.** Pairing, culling and `best.crl` still use the
  incremental `elo[]`, because all any of them needs is a *within-generation
  ranking* -- a relative comparison, which a drifting scale does not damage.
  Keeping it that way means the acid test above compares two measurements over
  one identical set of games rather than two different runs.
* **What anchors ON does change** is that the anchor games replace population
  pairings, so about 6% of a generation's games produce no training data.
  `--no-anchor-elo` turns the mechanism off entirely and reproduces the
  previous trainer **bit for bit**: built at HEAD and at this change, eight
  generations, one thread, same seed, the two produce identical
  `checkpoint.crl` and `best.crl` (same MD5) and telemetry that agrees on every
  non-timing field to the last digit.

  Getting there needed one real care. Writing the Elo update as
  `const float d = ELO_K * (sw - ew); elo[i] += d;` instead of
  `elo[i] += ELO_K * (sw - ew);` rounds the product to float before the add and
  so blocks the fused multiply-add that this build's `-ffp-contract=fast`
  allows. That is a one-ULP difference in one agent's rating -- and one ULP was
  enough to change a checkpoint's MD5 and send the run down a different path
  from the second generation on. The injection accounting is therefore kept in
  `double` and deliberately shares no subexpression with the update.

---

## 7. What the published numbers mean now

Every run in `runs/` carries a final internal Elo, and every one of them is a
reading of the same broken scale:

| run | generations | final `elo_best` |
|---|---:|---:|
| `runs/smoke`    |  224 | 1547 |
| `runs/pilot`    |  300 | 1748 |
| `runs/az`       |  201 | 1569 |
| `runs/az_v2`    |   95 | 1881 |
| `runs/az_main`  |  250 | 2103 |
| `runs/az_v3`    |  300 | 2179 |
| `runs/az_hour`  | 1900 | 4086 |

**None of these is a strength measurement**, and neither are the figures this
project has quoted along the way. 2157, 2629 and 3348 are `runs/az_hour` read
at generations **242, 762 and 1322** -- one run at three points on a ramp that
adds about 1.2 points per generation whether or not anything is being learned.
By generation 762 the policy KL had already bottomed out and the value head was
already degrading; the 1191 points added between 2157 and 3348 bought nothing
that any other instrument in the telemetry can see. The ordering among them is not entirely
meaningless -- a later network from the same run probably *is* stronger than an
earlier one -- but **the magnitude of every gap is manufactured**, the gaps
between different runs are not even on the same scale
(`docs/ABLATION.md` section on cross-run comparison already says this), and
none of them is on any recognisable rating scale.

`docs/BENCHMARK.md` reports `runs/pilot/best.crl` as "internal Elo 1750" in
every one of its result blocks. That label should be read as a run identifier,
not a rating.

The only strength measurements this project has ever made are the external
ones: `py/benchmark.py` against a weakened Stockfish, reported in
`docs/BENCHMARK.md`, with the caveats in `docs/RATING_SCALES.md` (measurement
good to about +/-80 Elo; pool conversion good to about +/-450). Those are
unaffected by any of this.

From this change onward, `elo_anchored_best` and the raw anchor scores are
internal numbers that can be believed -- within the limits of section 6.
`elo_best` is still written to the telemetry, and it still inflates; it is kept
so that runs from before and after this change can be plotted on the same axes,
and so that the drift stays visible rather than being quietly deleted.

One consequence worth stating explicitly: **`docs/ALGORITHM.md`'s "## Elo"
section is wrong.** It says that never updating hall-of-fame ratings "anchors
the scale so the population's rating shows real absolute progress rather than
drifting with the mean". The opposite is true, and this document is the
measurement that shows it. That file is not edited here only because it was out
of scope for this change.
