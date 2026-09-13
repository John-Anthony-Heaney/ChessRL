# Asymmetric search budgets: breaking the self-play distribution trap

`docs/TACTICS.md` diagnosed the agent's piece-hanging as a **data** problem, not
a capacity problem. Its central number:

> In the self-play that trained this network, being two pawns down is worth an
> actual score of **−0.036**. Material does not begin to predict the result
> below about six pawns, because the opponent is the same network and hands it
> straight back.

The value head is not broken. It is correctly fitted to a distribution in which
material does not predict the outcome, and it cannot learn what its labels do
not contain. The trap is self-reinforcing: both sides blunder → blunders do not
predict outcomes → the network never learns blunders are costly → both sides
keep blundering.

This file tests one direct attack on that trap. **If one side of a self-play
game searched a larger budget, it should win material AND KEEP it**, so that
within those games material would start predicting the result.

**Verdict, in one line: the mechanism is real and the intervention does not
work, so it ships OFF.**

Asymmetric budgets do to the training data exactly what the diagnosis says they
should — in generation-1894 self-play they turn a distribution where two pawns
down is worth −0.020 into one where it is worth −0.182, lift
corr(material, result) from +0.44 to +0.65, cut the draw rate from 0.71 to
0.26, and take the value head's R² against the outcome from −0.27 to **+0.51**.
Trained with, at five seeds and a matched compute budget, they produce a
network that is **no better by any measure the exercise was aimed at**: the
external score is 0.6238 → 0.6040 (the wrong direction, 0.80× the seed SD,
omnibus p = 0.44), the implied queen-to-pawn ratio stays at ~2.4 against the
~9 it should be, the hang rate does not move and the score against the
material-1 oracle does not move.

The flags are implemented, tested, documented and defaulted to zero. §3 and §6
say why the story was not enough.

---

## 0. What was built

Two flags on `chessrl az`, default off:

```
--asym-frac F     fraction of self-play games played with unequal budgets
--asym-ratio R    the weak side searches sims / R   (R = 4 => 160 vs 40)
--asym-record S   both | strong -- whose moves become training data there
```

In an asymmetric game one side searches the full `--sims` budget and the other
searches `sims / R`. Everything else is unchanged: the same shared trunk, the
same per-agent heads, the same reward (the game result and nothing else), the
same policy target (the MCTS visit distribution), the same temperatures, the
same Dirichlet noise.

Three properties are enforced in `build_pairings()` (`src/az.c`) rather than
left to chance, because each of them is a way the experiment could have lied:

* **Exactly** `round(F × eligible)` games are asymmetric — a Bresenham walk over
  the pairing list, not a per-game coin, so the achieved fraction is not a
  binomial draw around the requested one. The telemetry reports
  `asym_game_frac` so this is checked rather than asserted.
* **The colours are balanced exactly.** The full budget alternates White/Black
  on the asymmetric ordinal, so White holds it in ⌈n/2⌉ of the games and Black
  in ⌊n/2⌋ — exactly balanced to within one game, not "half on average".
  Telemetry: `asym_white_strong_share`, which reads 0.5000 whenever the number
  of asymmetric games in a generation is even and 18/35 = 0.514 when it is odd.
* **Anchor games are never asymmetric.** The anchored rating in
  `docs/RATING.md` works by playing a uniform random mover and a frozen gen-0
  network whose strength is a constant of nature. Handicapping either side of
  one of those games would move the ruler that measures everything else. They
  are excluded from eligibility in `build_pairings()` and again in
  `az_slot_newgame()`.

The assignment is made once per generation on the finished pairing list, with a
random rotation phase drawn from the generation's master stream, because the
list is rank-ordered before the window shuffle and a fixed stride would keep
landing on the same part of the population.

**With `--asym-frac 0` the trainer is the trainer that existed before this
change, exactly.** The rotation phase is the only new draw from any random
stream and it sits inside the `asym_frac > 0` guard, so a run with the feature
off consumes the identical sequence of random numbers; nothing else on the
self-play path is conditional on anything but `pr.strong`, which is `-1`
throughout. That is what makes the `asym-frac = 0` level of §6 a legitimate
control rather than a fourth new configuration.

### This is inside `docs/FROM_SCRATCH.md`, and here is the argument

`docs/FROM_SCRATCH.md` forbids **every form of chess evaluation** in the
learning or play path: piece values, piece-square tables, mobility, king
safety, MVV-LVA, opening books, and "reward shaping of any kind derived from
the position rather than the result".

A simulation budget is none of those things.

* **It never looks at the board.** The code that decides an asymmetric game
  reads the pairing index and a random number. It cannot read the position; it
  runs before the first move is generated.
* **It expresses no opinion about chess.** It says how much compute one player
  gets, not what any piece or square is worth. The identical code would run
  unchanged on Go, Hex or Nim.
* **The reward is untouched.** `z` is still +1 / −1 / `draw_penalty` and
  nothing else. No shaping term, no material term, no potential function.
* **The targets are untouched.** The policy target is still the MCTS visit
  distribution; the value target is still the outcome.
* **It is the same argument the file already accepts for the draw penalty and
  the playout cap.** The draw penalty is permitted because it is symmetric and
  "a statement about learning dynamics in symmetric self-play, not about the
  game". Playout-cap randomisation is permitted because it is "a statement
  about where simulation budget buys training signal". An asymmetric budget is
  a statement of the same kind: about the distribution self-play generates, not
  about how chess should be played.
* **A stronger-than-me opponent is not knowledge.** `docs/FROM_SCRATCH.md`
  already permits the hall of fame and the two anchors — opponents of different
  strength drawn from the agent's own history and from a random mover. A
  larger search budget is a weaker intervention than either: it is the *same*
  network, at the *same* generation, thinking for longer.

`tools/audit_knowledge.sh` passes on the changed files and `make test` reports
7 checks, 0 failures, "shipped play path: clean" (§8).

The honest counter-argument, stated so the reader can weigh it: the handicap
does inject **information about who is likely to win**, and §3 shows that most
of the new material-to-result correlation comes from exactly that. That is a
statement about the sampling distribution, not about the position — but it is
the reason the result has to be judged on the external score in §6 and not on
the correlation in §3.

---

## 1. The instrument, and the proof it measures the right thing

`tools/diag_asym.c`. It plays self-play games under the training settings
(Dirichlet root noise, temperature 1.0 for 20 plies then 0.25), makes
`--asym-frac` of them unequal, and reports the material-to-result relationship
**separately for the symmetric and the asymmetric games of the same
invocation** — same model, same opening distribution, same rng stream. The
comparison is therefore paired and the only difference is the handicap.

It contains piece values (`py/baselines.py`'s: P=1 N=3 B=3.25 R=5 Q=9) so its
numbers can be read next to `docs/TACTICS.md`. `docs/FROM_SCRATCH.md` permits
that in a measuring instrument and forbids it in the learning path; the file is
linked into nothing and `tools/audit_knowledge.sh` scans `src/*.c` only.

```sh
cc -O3 -std=c11 -D_DARWIN_C_SOURCE -Wall -Wextra -Wno-unused-parameter \
   -funroll-loops -fno-math-errno -ffp-contract=fast -mcpu=native -Isrc \
   tools/diag_asym.c build/chess.o build/net.o build/mcts.o \
   -o build/diag_asym -lm -lpthread
```

**It reproduces the existing instrument exactly.** With the handicap off it
draws the same random numbers as `tools/diag_tactics.c`'s `calib`, and every
statistic agrees to the last digit:

```sh
build/diag_asym     --model runs/az_hour/best.crl --games 400 --sims 160 \
                    --asym-frac 0 --threads 8 --start classical --max-plies 300
build/diag_tactics  calib --model runs/az_hour/best.crl --games 400 --sims 160 \
                    --threads 8 --train-mode
```

| | `diag_asym --asym-frac 0` | `diag_tactics calib --train-mode` |
|---|---:|---:|
| positions | 69 099 | 69 099 |
| draw rate | 0.615 | 0.615 (246 of 400) |
| MSE(v, z) | 0.3883 | 0.3883 |
| Var(z) | 0.3718 | 0.3718 |
| MSE/Var | 1.0443 | 1.0443 |
| R² | −0.0443 | −0.0443 |
| actual score at −2 pawns | **−0.020** (n = 4 599) | **−0.020** (n = 4 599) |

**The baseline has moved slightly since `docs/TACTICS.md` was written**, and it
is worth saying why: HEAD now contains the PUCT first-descent fix
(`sqrt(sum + 1)`, commit `9f6936d`), which `docs/TACTICS.md` §5 recommended.
Two pawns down in symmetric self-play is now worth **−0.020** rather than
−0.036. Both numbers are "indistinguishable from zero"; the diagnosis is
unchanged and the baseline used throughout this file is the current one.

---

## 2. THE HEADLINE — does material start predicting the result?

Model: `runs/az_hour/best.crl` (generation 1894), the same checkpoint
`docs/TACTICS.md` dissected. 400 games, half of them asymmetric, 160 vs 40
simulations.

```sh
build/diag_asym --model runs/az_hour/best.crl --games 400 --sims 160 \
                --asym-frac 0.5 --asym-ratio 4 --threads 8 \
                --start classical --max-plies 300 --seed 20260912
```

| | symmetric (200 games) | **asymmetric (200 games)** |
|---|---:|---:|
| positions | 34 218 | 31 904 |
| draw rate | 0.705 | **0.260** |
| **corr(material, result)** | **+0.4447** | **+0.6537** |
| Var(z) — how decisive the games are | 0.2803 | **0.7182** |
| value head R² against the outcome | **−0.269** | **+0.5105** |
| corr(v, result) | +0.3015 | +0.7208 |
| mean predicted v | +0.0216 | +0.0772 |
| mean outcome z | +0.0008 | +0.0025 |
| **the full-budget side scored** | — | **0.8550** |
| White held the full budget | — | 0.520 of games |

**And the table that is the whole point** — actual score by material balance,
from the side to move's view:

| material (pawns) | symmetric: n / **actual score** | asymmetric: n / **actual score** |
|---:|---|---|
| ≤ −6 | 4 859 / −0.381 | 8 420 / **−0.774** |
| −5 | 1 069 / −0.158 | 786 / **−0.439** |
| −4 | 1 113 / −0.069 | 989 / **−0.405** |
| −3 | 1 724 / **+0.040** | 1 272 / **−0.257** |
| **−2** | 2 476 / **−0.004** | 1 476 / **−0.182** |
| −1 | 2 576 / −0.039 | 1 332 / −0.126 |
| 0 | 7 619 / +0.002 | 4 044 / +0.018 |
| +1 | 2 363 / +0.044 | 1 291 / +0.176 |
| +2 | 2 373 / +0.011 | 1 404 / +0.218 |
| +3 | 1 618 / **−0.036** | 1 182 / **+0.304** |
| ≥ +6 | 4 457 / +0.406 | 8 023 / +0.792 |

Read the two middle columns. In symmetric self-play the relationship between
material and the result is *flat and noisy* between −3 and +3 pawns: being
three pawns **down** scores +0.040 and being three pawns **up** scores −0.036,
which is not a signal, it is a sign error made of noise. In the asymmetric
games the same range is monotone and steep: −0.257, −0.182, −0.126, +0.018,
+0.176, +0.218, +0.304.

**The equivalent of `docs/TACTICS.md`'s −0.036: two pawns down in asymmetric
self-play is worth −0.182.** The symmetric comparison is −0.004 in the paired
arm of the same invocation and −0.020 in the 400-game symmetric baseline of §1
— a factor of **9 to 45**, and the honest way to say it is that the symmetric
number is indistinguishable from zero and the asymmetric one is not.

And under the settings training actually uses (mixed Chess960 starts, 200-ply
cap) the same comparison holds, a little smaller:

```sh
build/diag_asym --model runs/az_hour/best.crl --games 400 --sims 160 \
                --asym-frac 0.5 --asym-ratio 4 --threads 8 \
                --start mixed --max-plies 200 --seed 20260912
```

| | symmetric | asymmetric |
|---|---:|---:|
| draw rate | 0.730 | 0.315 |
| corr(material, result) | +0.4438 | +0.5790 |
| value head R² | −0.455 | +0.387 |
| actual score at −2 pawns | −0.065 | **−0.178** |
| the full-budget side scored | — | 0.8125 |

### Correlation by ply — material becomes informative much earlier

corr(material, result) within a ply bucket, classical starts:

| ply | 0–9 | 10–19 | 20–39 | 40–59 | 60–99 | 100+ |
|---|---:|---:|---:|---:|---:|---:|
| symmetric | −0.036 | +0.074 | +0.148 | +0.194 | +0.381 | +0.620 |
| **asymmetric** | **+0.166** | **+0.444** | **+0.487** | **+0.618** | **+0.707** | **+0.796** |

In symmetric self-play material carries essentially no information about the
result until the game is 60 plies old. In asymmetric games it carries more
information at ply 10 than symmetric play ever reaches before ply 60.

---

## 3. The part that undermines the story, measured

The correlation above is real, but **most of it is a between-player effect, not
a within-player one**. Split the asymmetric positions by whose turn it is:

| ratio 4, classical | all asym positions | strong side to move | weak side to move |
|---|---:|---:|---:|
| positions | 31 904 | 15 987 | 15 917 |
| corr(material, result) | **+0.654** | **+0.380** | **+0.384** |
| mean outcome z | +0.003 | **+0.689** | **−0.687** |
| mean predicted v | +0.077 | +0.510 | −0.357 |

Conditioning on which player is to move removes more than half of the
correlation, and the residual (+0.38) is *below* the symmetric arm's +0.44.
The mechanism is therefore not "a pawn now matters more per pawn"; it is
"material tells you which of the two players you are, and that player wins".

The network cannot see its own simulation budget, so material is the only cue
it has for that — which means it will in fact learn a steeper material→value
map, which is the desired effect. But it is a **proxy relationship**, and a
proxy is exactly the kind of thing that can fit the training distribution and
fail to transfer. That is why §6 is the decisive experiment and §2 is not.

### The ratio sweep makes the trade-off explicit

Same model, same 400 games, `--asym-frac 0.5`, classical starts:

| ratio | weak sims | strong side scored | asym draw rate | corr(mat, result) | value R² | **within-side corr** |
|---:|---:|---:|---:|---:|---:|---:|
| — (symmetric) | 160 | 0.500 | 0.62–0.70 | +0.44 … +0.52 | −0.27 … −0.02 | — |
| 2 | 80 | 0.645 | 0.590 | +0.511 | +0.032 | +0.459 |
| **4** | **40** | **0.855** | **0.260** | **+0.654** | **+0.511** | **+0.379** |
| 8 | 20 | 0.968 | 0.055 | +0.705 | +0.688 | **+0.104** |
| 16 | 10 | 0.983 | 0.035 | +0.766 | +0.742 | **+0.108** |

The headline correlation is monotone in the handicap, and the within-side
correlation **collapses** past ratio 4: at 8 and 16 the strong side wins 97–98%
of games regardless of what happens on the board, so the label is almost
entirely "which player am I" and almost none of it is "what is the position".
At ratio 2 the handicap is too weak to change anything (+0.511 against a
symmetric +0.502 measured in the same invocation).

Ratio 4 is the only setting on this evidence that buys a large distributional
change while leaving real within-side signal, so it is the default and the
level the `asym-frac` ablation in §6 uses. Ratio 16 is tested there too, as a
separate arm, precisely because at the strength a 12,288-game run reaches the
trade-off sits in a different place — see below.

*(The symmetric arm's correlation varies between +0.444 and +0.516 across the
four invocations — 200 games each, same model. That spread is the measurement
noise this table has to be read against; the ratio-4 gap is roughly six times
it.)*

### And it only works if the model is already good

The handicap's whole effect runs through "the stronger searcher wins". How much
a 4x budget is actually worth depends on how good the network is, and that
turns out to be the single most important caveat in this file. Same command,
same settings (`--start mixed --max-plies 200`, ratio 4), three checkpoints:

| model | generation | **the full-budget side scored** | symmetric corr(mat, result) | asymmetric | gain |
|---|---:|---:|---:|---:|---:|
| `runs/smoke/best.crl` | 4 | **0.515** | +0.048 | +0.128 | +0.080 |
| `runs/az_v3/best.crl` | 295 | **0.675** | +0.331 | +0.349 | **+0.018** |
| `runs/az_hour/best.crl` | 1894 | **0.813** | +0.444 | **+0.579** | **+0.135** |

A four-fold search advantage is worth 0.515 to a network four generations old
and 0.813 to one 1894 generations old, because search only compounds once the
priors and the value head are worth searching over. Around 0.675 the handicap
barely moves the material-to-result relationship at all: at generation 295 the
correlation goes +0.331 -> +0.349, which is nothing.

**This sets up §6 before it happens.** The compute-matched ablation trains for
128-167 generations at 32 agents, and its own telemetry (§5) reports the
full-budget side finishing at **0.689** at ratio 4 — the generation-295 regime,
not the generation-1894 one. Ratio 16 gets it to **0.754** in training (0.823
when the finished checkpoints are re-measured cleanly), which is why that arm
exists: it is the only one in §6 where the intervention demonstrably changes
the data as much as §2 does.

So §6 is a fair test of "does this help a run of this size". It is **not** a
test of the mechanism at the strength where the mechanism bites hardest, and
§7 says so again where it matters.

---

## 4. The value-bias hazard, and which recording rule is right

`docs/TACTICS.md` §3c shows a value head that is **systematically, one-sidedly
optimistic**. Asymmetry is a plausible way to make that worse, because in an
asymmetric game one side wins most of the time. Two things were measured.

**First: is the label distribution still balanced?** Yes, as long as both
sides' positions are recorded. The strong side's positions carry a mean outcome
of **+0.689** and the weak side's **−0.687**; recorded together they cancel to
**+0.0025** over the asymmetric games, against +0.0008 in the symmetric ones.
The asymmetry adds no first-order bias to the value target.

**Second: what happens if only the strong side's moves are recorded?** That is
the obvious way to answer the "the weak side's policy targets come out of a
smaller search, so they are worse data" objection. It is also, on the same
measurement, a trap:

| recording rule | mean z of the recorded labels | corr(material, result) in the recorded set |
|---|---:|---:|
| `both` (default) | **+0.003** | **+0.654** |
| `strong` | **+0.689** | +0.380 |

Recording only the strong side hands the value head a target whose mean is
+0.69 rather than 0: every gradient step now carries a common positive
component that has nothing to do with the position, which is the exact shape of
the one-sided optimism `docs/TACTICS.md` §3c already blames for 37% of
positions reading `v > 0.8` while losing 94% of them. It also *lowers* the
material correlation in the recorded set, for the reason in §3: conditioning on
"I am the strong player" removes most of the signal.

Both effects are smaller in the regime an affordable training run actually
reaches — §5 measures the realised label mean at **+0.026** rather than +0.69,
because at that strength the full-budget side only scores 0.63 — but both have
the predicted sign, and §6 tests the recording rule with five paired training
runs rather than leaving it as a prediction.

---

## 5. Training with it: what the trainer's own telemetry says

Five seeds per arm, last generation, `mean ± sd`. Every arm is matched to the
same **238,110,720 network evaluations** (§6), which is why the generation
counts differ.

| | frac 0.0 (baseline) | frac 0.25, r4 | frac 0.5, r4, `both` | frac 0.5, r4, `strong` | frac 0.5, **r16**, `both` |
|---|---:|---:|---:|---:|---:|
| generations | 128 | 141 | 158 | 158 | 167 |
| `asym_game_frac` | 0.0000 | **0.2464** | **0.5072** | 0.5072 | 0.5072 |
| `asym_white_strong_share` | — | **0.5294** | **0.5143** | 0.5143 | 0.5143 |
| `asym_mean_sims` | 160 | 100.09 | 100.08 | 100.07 | **85.11** |
| `asym_recorded_strong_share` | — | 0.5032 | 0.5003 | **1.0000** | 0.5056 |
| **`asym_strong_score`** | — | **0.671 ± 0.025** | **0.689 ± 0.067** | 0.631 ± 0.072 | **0.754 ± 0.057** |
| self-play draw rate | 0.481 ± 0.103 | 0.559 ± 0.073 | 0.530 ± 0.048 | 0.533 ± 0.042 | 0.510 ± 0.120 |
| captures / game | 23.70 | 24.48 | 24.39 | 23.73 | 24.13 |
| positions recorded / game | 131.4 | 135.9 | 134.6 | **102.1** | 130.0 |
| value target mean | −0.023 | −0.035 | −0.038 | **+0.026** | −0.033 |
| value prediction mean | −0.026 | −0.038 | −0.039 | **+0.019** | −0.036 |
| policy top-1 vs the search | 0.530 | 0.544 | 0.541 | 0.527 | **0.557** |
| policy KL | 0.334 | 0.329 | 0.335 | **0.394** | **0.419** |

Four things worth reading off it.

1. **The knob does exactly what it says.** The achieved fraction is exactly the
   requested fraction of eligible games, the colour split is exact to within
   one game, and the mean simulation count per move is exactly
   (160 + 40)/2 = 100 at ratio 4 and (160 + 10)/2 = 85 at ratio 16 — which is
   the number `py/ablate.py` uses to match compute.
2. **The handicap bites, but not as hard as §2.** By the end of training the
   full-budget side takes 0.689 of a point at ratio 4 and 0.754 at ratio 16,
   against 0.813 for the generation-1894 checkpoint. §3's strength-dependence
   table said this would happen.
3. **The value-target bias is real, small, and confined to `--asym-record
   strong`.** The mean label goes from −0.023 (baseline) to **+0.026** when
   only the strong side's moves are recorded, and the head's mean prediction
   follows it to +0.019. It is a 0.05 shift rather than the +0.69 that §4's
   static probe implies, because the strong side only scores 0.63 here — but
   it is the predicted sign, it is outside the seed spread, and it is a cost
   with no measured benefit.
4. **`--asym-record strong` throws away a quarter of the training data**
   (134.6 → 102.1 positions per game, exactly the half of the asymmetric
   half) and its policy fits the search *worse*, not better (KL 0.335 →
   0.394), despite every recorded target now coming from a full 160-simulation
   search. Fewer positions cost more than better targets bought.

### The measurement of §2, repeated on the trained models

`build/diag_asym --games 200 --sims 160 --asym-frac 0.5 --start mixed
--max-plies 200`, run on each arm's five final checkpoints (ratio 4 for the
first two arms, ratio 16 for the third, matching how each was trained):

| arm (n=5) | strong side scored | draw rate sym → asym | corr(mat, result) sym → asym | value R² sym → asym |
|---|---:|---|---|---|
| baseline, frac 0 | 0.689 ± 0.042 | 0.646 → 0.552 | +0.398 → **+0.491** | +0.047 → **+0.192** |
| trained frac 0.5, r4 | 0.667 ± 0.041 | 0.688 → 0.615 | +0.338 → **+0.470** | **−0.324** → +0.027 |
| trained frac 0.5, r16 | 0.823 ± 0.068 | 0.619 → **0.346** | +0.422 → **+0.609** | −0.012 → **+0.441** |

The distributional effect reproduces on every arm and every seed: asymmetric
games always carry more material-to-result correlation than symmetric ones.

And the cost shows up in the second row. **The model trained with asymmetry is
markedly *worse* at predicting the outcome of its own symmetric games**
(R² +0.047 → −0.324). That is straightforward distribution shift: a head fitted
to a 50/50 mixture cannot tell which kind of game it is in, so it hedges, and
the hedge is wrong in both halves. It is the first concrete sign that the
intervention is not free.

---

## 6. THE DECISIVE EXPERIMENT — compute-matched, multi-seed, external

```sh
python3 py/ablate.py --factor asym-frac --levels 0,0.25,0.5 --seeds 5 \
                     --out runs/ablation_asym_frac.json
```

`py/ablate.py`'s rules apply unchanged (`docs/ABLATION.md`): matched on **total
network evaluations** because asymmetry changes the cost of a game, five
independent training seeds per level, evaluated **externally** by
`py/benchmark.py` — 800 colour-reversed games against a uniform random mover,
raw policy head at argmax with no search, the same evaluation seed for every
model so the comparison is paired, scoring `checkpoint.crl` rather than the
internally early-stopped `best.crl`.

| level | gens | total games | mean | sd | se | 95% CI | min | max |
|---:|---:|---:|---:|---:|---:|---|---:|---:|
| **0.0** | 128 | 12 288 | **0.6238** | 0.0263 | 0.0118 | [0.5911, 0.6564] | 0.5869 | 0.6569 |
| 0.25 | 141 | 13 536 | 0.6181 | 0.0218 | 0.0097 | [0.5911, 0.6451] | 0.5938 | 0.6438 |
| 0.5 | 158 | 15 168 | 0.6040 | 0.0253 | 0.0113 | [0.5726, 0.6354] | 0.5769 | 0.6400 |

| | |
|---|---|
| within-level (seed) SD, pooled | **0.0245 score = 18.0 Elo** |
| between-level spread (max − min) | 0.0198 score = 14.5 Elo |
| **spread / seed SD** | **0.80×** |
| evaluation SE per match | 0.0075 (31% of the seed SD) |

| comparison | diff | diff Elo | Cohen *d* | Welch p | perm p (exact) | Holm p |
|---|---:|---:|---:|---:|---:|---:|
| 0.0 vs 0.25 | 0.0056 | +4.2 | +0.23 | 0.7226 | 0.7222 | 0.8333 |
| 0.0 vs 0.5 | 0.0198 | +14.5 | +0.80 | 0.2611 | 0.2778 | 0.8333 |
| 0.25 vs 0.5 | 0.0141 | +10.3 | +0.58 | 0.3721 | 0.3413 | 0.8333 |

Omnibus permutation F-test across all levels: **F = 0.860, p = 0.4433**
(20 000 draws). Pairwise permutation tests are exact — all 252 relabellings
enumerated. Minimum detectable effect at n = 5 per level, α = 0.05,
power = 0.80: **0.0496 score = 36.4 Elo**.

> **VERDICT: `asym-frac` is indistinguishable from seed noise at this sample
> size.** The largest between-level difference is 0.0198, which is 0.80× the
> pooled seed SD and 0.40× the minimum detectable effect.

And note the **direction**. The three level means are monotone *downward* in
the amount of asymmetry — 0.6238, 0.6181, 0.6040. That is not significant
either, but it is the opposite of what the mechanism predicts, and the honest
reading is "no evidence of benefit, weak and non-significant evidence of harm",
not "it might be helping a little".

### The two extra arms, on the same yardstick

Both use the identical configuration `ablate.py` generated for its
`asym-frac = 0.5` level, the identical five seeds, the identical external
evaluation, and generation counts recomputed from the same evaluation budget.

| arm (n = 5) | mean | sd | diff vs baseline | Elo | Cohen *d* | Welch p | perm p |
|---|---:|---:|---:|---:|---:|---:|---:|
| **frac 0.0 (baseline)** | **0.6238** | 0.0263 | — | — | — | — | — |
| frac 0.25, ratio 4, `both` | 0.6181 | 0.0218 | −0.0056 | −4.1 | −0.22 | 0.7226 | 0.7222 |
| frac 0.5, ratio 4, `both` | 0.6040 | 0.0253 | −0.0198 | −14.6 | −0.79 | 0.2611 | 0.2778 |
| frac 0.5, ratio 4, **`strong`** | 0.6170 | 0.0247 | −0.0068 | −5.0 | −0.27 | 0.6872 | 0.7063 |
| frac 0.5, **ratio 16**, `both` | **0.6345** | 0.0273 | **+0.0107** | +7.9 | +0.43 | 0.5439 | 0.5159 |

Pooled within-arm SD across all five arms: **0.0252**; MDE at 5 vs 5:
**0.0508 score = 37 Elo**.

* **The recording rule does not matter.** `strong` (0.6170) sits between
  `both` (0.6040) and the baseline (0.6238), well inside one seed SD of each.
  §4 predicted `strong` would be worse than `both` and §5 measured the two
  costs it carries (a quarter of the data thrown away, a +0.05 shift in the
  value target); neither shows up as a strength difference, and neither shows
  up as a benefit. **Recording both sides stays the default**, on the grounds
  that it is the rule with no measured downside, not on the grounds that it won.
* **The handicap size does not rescue it either.** Ratio 16 is the only arm
  whose point estimate is above the baseline, and it is the arm where the
  intervention demonstrably changed the training data the most (§5: the strong
  side scores 0.823, the asymmetric draw rate falls 0.619 → 0.346, the
  material correlation rises +0.422 → +0.609, the value head's R² on that data
  rises −0.012 → +0.441). **All of that bought +0.0107 of score, d = 0.43,
  p = 0.52.** If anything is there, it is well under the 0.0508 this design
  can see.

### And the probes the whole exercise was aimed at

`tools/diag_tactics.c` on every final checkpoint: the piece-deletion probe on a
**fixed** corpus (`--source random`, uniformly random legal games, so the
positions do not depend on the model being probed), 60 complete games against
the material-1 oracle, and the two calibration runs. Five seeds per arm.

| arm (n=5) | pawn Δv | queen Δv | **queen/pawn** | HANG rate | mean drop | score vs material-1 | R² on own self-play |
|---|---:|---:|---:|---:|---:|---:|---:|
| frac 0.0 (baseline) | −0.066 ± 0.019 | −0.184 ± 0.046 | **2.82 ± 0.25** | 0.0906 ± 0.0140 | +0.379 | 0.0284 ± 0.0245 | +0.128 ± 0.062 |
| frac 0.5, r4, `both` | −0.091 ± 0.017 | −0.219 ± 0.075 | **2.36 ± 0.56** | 0.0945 ± 0.0087 | +0.386 | 0.0280 ± 0.0295 | −0.151 ± 0.208 |
| frac 0.5, r4, `strong` | −0.053 ± 0.019 | −0.127 ± 0.086 | **2.21 ± 1.56** | 0.1059 ± 0.0096 | +0.458 | 0.0168 ± 0.0158 | −0.014 ± 0.082 |
| frac 0.5, r16, `both` | −0.069 ± 0.031 | −0.198 ± 0.106 | **2.83 ± 0.80** | 0.0972 ± 0.0096 | +0.413 | 0.0282 ± 0.0194 | +0.009 ± 0.205 |

`docs/TACTICS.md` called the implied queen-to-pawn ratio "the clearest single
indicator of whether the fix worked": a correctly scaled head would put it near
9, and the generation-1894 checkpoint sits at **1.9**. Every arm here sits
between **2.2 and 2.8**, and the baseline is at the top of that range. The hang
rate does not move (0.091 → 0.095 / 0.106 / 0.097, against a pooled sd of
~0.010). The score against the material-1 oracle does not move
(0.028 → 0.028 / 0.017 / 0.028).

**Nothing the intervention was built to move, moved.**

---

## 7. What this rules out, and what it does not

**Ruled out, at this budget:** that turning on asymmetric search budgets at
ratio 4 or 16, with either recording rule, improves the network by more than
**0.0508 of score (≈ 37 Elo)** at 32 agents and a 238-million-evaluation
budget (12 288 to 16 032 self-play games, 128 to 167 generations). It
does not improve the external score, the implied piece values, the hang rate or
the score against the material-1 oracle, and three of the four point estimates
point the wrong way.

**Not ruled out, and worth saying clearly:**

* **The long-run regime.** §3 measured that a 4× budget is worth 0.515 of a
  point at generation 4, 0.675 at 295 and 0.813 at 1894 — the intervention gets
  *stronger* as the network does. The ablation trains for 128–167 generations,
  where the strong side takes 0.69 (ratio 4) or 0.75 (ratio 16). A null at that
  scale does not prove a null at 1894 generations. It is equally true that
  nothing here is evidence *for* an effect at 1894 generations, and the cost of
  finding out is a pair of multi-day runs.
* **Effects smaller than the MDE.** Five seeds is a small sample. 0.0508 is a
  large effect to require; `docs/ABLATION.md`'s own headline factor, population
  size, was measured against the same wall.
* **A ratio between 4 and 16, or a fraction above 0.5.** Only three points on
  each axis were tried. §3's sweep says the useful window is narrow — below
  ratio 4 the handicap does nothing, above ratio 8 the within-side signal
  collapses — and the two arms tested sit on either side of it.

**Two reasons to think the null is not an accident of budget**, both measured
rather than argued:

1. **The correlation asymmetry creates is mostly a proxy.** §3: conditioning on
   which player is to move removes more than half of it, and the residual
   (+0.38) is *below* the symmetric arm's +0.44. The network has no input
   feature for its own simulation budget, so the only way it can exploit the
   new correlation is through material — but the thing material is now a proxy
   *for* is "which player am I", which is not a property of the position and
   does not exist at play time.
2. **It costs calibration on the half of the games that stay symmetric.** §5:
   the frac-0.5 model's value R² on its own symmetric self-play is **−0.324**
   against the baseline's **+0.047**. A head fitted to a mixture it cannot
   identify hedges, and the hedge is wrong in both halves. Whatever the
   asymmetric half buys, this is what it is paid for with.

**What this points at instead.** `docs/TACTICS.md`'s ranking survives intact:
the binding constraint is the value target, and the way to fix it is to make
the label depend on the position rather than on which player generated it.
Asymmetric budgets change *who* wins, not *why*. The two candidates still
untested that change the *why* are `AZCfg.value_mix` (bootstrapping the target
from the search's own root value at that exact position, which `az.h` already
argues is inside the contract) and raising the simulation count itself, where
`docs/ABLATION_RESULTS.md` found 64/160/320 scoring 0.4531/0.4865/0.5171 on a
curve that had **not** flattened at the top level tested.

---

## 8. Reproducing everything

```sh
# ---- build ----------------------------------------------------------------
make -j8 && make -j8 lib          # warning-free
make test                         # 7 checks, 0 failures; audit: shipped play path clean
tools/audit_knowledge.sh -v src/az.c src/main.c

# the instrument is not in the Makefile; it is built by hand and linked into
# nothing, which is what lets it contain piece values
cc -O3 -std=c11 -D_DARWIN_C_SOURCE -Wall -Wextra -Wno-unused-parameter \
   -funroll-loops -fno-math-errno -ffp-contract=fast -mcpu=native -Isrc \
   tools/diag_asym.c build/chess.o build/net.o build/mcts.o \
   -o build/diag_asym -lm -lpthread
cc ... tools/diag_tactics.c ... -o build/diag_tactics -lm -lpthread

# ---- section 1: the instrument agrees with tools/diag_tactics.c ------------
build/diag_asym    --model runs/az_hour/best.crl --games 400 --sims 160 \
                   --asym-frac 0 --threads 8 --start classical --max-plies 300
build/diag_tactics calib --model runs/az_hour/best.crl --games 400 --sims 160 \
                   --threads 8 --train-mode

# ---- section 2: the headline, and the training-settings variant ------------
build/diag_asym --model runs/az_hour/best.crl --games 400 --sims 160 \
                --asym-frac 0.5 --asym-ratio 4 --threads 8 \
                --start classical --max-plies 300 --seed 20260912
build/diag_asym --model runs/az_hour/best.crl --games 400 --sims 160 \
                --asym-frac 0.5 --asym-ratio 4 --threads 8 \
                --start mixed --max-plies 200 --seed 20260912

# ---- section 3: the ratio sweep and the strength dependence ----------------
for R in 2 4 8 16; do
  build/diag_asym --model runs/az_hour/best.crl --games 400 --sims 160 \
                  --asym-frac 0.5 --asym-ratio $R --threads 8 \
                  --start classical --max-plies 300 --seed 20260912
done
for M in runs/smoke/best.crl runs/az_v3/best.crl runs/az_hour/best.crl; do
  build/diag_asym --model $M --games 200 --sims 160 --asym-frac 0.5 \
                  --asym-ratio 4 --threads 4 --start mixed --max-plies 200 \
                  --seed 20260912
done

# ---- section 6: the compute-matched ablation ------------------------------
python3 py/ablate.py --factor asym-frac --levels 0,0.25,0.5 --seeds 5 \
                     --out runs/ablation_asym_frac.json
```

The two extra arms in sections 5 and 6 are run outside `py/ablate.py`
because they vary a factor it does not carry (`--asym-record`) and a level of
`--asym-ratio` that has no matching control in the `asym-frac` sweep. They use
the **identical** configuration `ablate.py` generated for its `asym-frac = 0.5`
level, the identical five seeds, the identical external evaluation
(`py/benchmark.py --opponent random --games 800 --seed 20260911 --policy-only`),
and generation counts recomputed from the same 238,110,720-evaluation budget
(158 generations at ratio 4, 167 at ratio 16). The scripts are
`record_arm.sh` and `ratio16_arm.sh` in this experiment's scratch directory and
the exact commands are recorded in the run directories' telemetry headers.

### Files touched

| file | change |
|---|---|
| `src/az.h` | three `AZCfg` fields and the `AZ_ASYM_REC_*` enum |
| `src/az.c` | the assignment in `build_pairings()`, the budget in `az_slot_begin_move()`, the recording gate and the counters, the telemetry block, the default and the clamp |
| `src/main.c` | `--asym-frac`, `--asym-ratio`, `--asym-record` and their usage text |
| `tools/diag_asym.c` | new; the instrument of §1, linked into nothing |
| `docs/ASYMMETRY.md` | this file |
| `py/ablate.py` | **two additive entries** so `--factor asym-frac` exists: `asym-frac`/`asym-ratio` in `BASE_CONFIG` and in `FACTORS`, plus the asymmetric term in `effective_sims()` without which the plan would not be compute-matched. Nothing else in that file changed. |
| `docs/ABLATION.md` | the results section `ablate.py` appends by itself at the end of every experiment |

### A note on the machine

Every wall-clock number in this file was produced on an 8-core Apple M3 that
was **not** idle: other agents were training in parallel throughout. That
affects throughput and nothing else — every statistic here is over games and
positions, not seconds — but it is why no timing claims are made.
