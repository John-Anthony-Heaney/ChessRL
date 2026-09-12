# Why the agent hangs pieces

A mechanistic diagnosis of `runs/az_hour/best.crl` (generation 1894, 128 agents,
best agent 65). Every number below was produced by `tools/diag_tactics.c`; the
command that produced it is printed with it.

**Verdict, in one line.** The binding constraint is the **value head**, and the
reason the value head is wrong is that **the self-play data it is trained on
contains almost no relationship between material and the game result** — in
training self-play, being two pawns down is worth an actual score of **−0.036**,
because the opponent gives the material straight back. The policy priors are a clear
second: after 1894 generations the policy head still puts prior mass on
piece-dropping moves at **0.93×** the rate a uniform prior would. Search budget
is third and is not binding: the tree already reaches the refutation, searches
it 93% of the time, and plays the losing move anyway.

---

## 0. The instrument, and the proof that it measures the right thing

Everything here is graded by one ruler: depth-1 negamax over material with a
6-ply quiescence search and delta pruning — i.e. `material-1` from
`py/baselines.py`, reimplemented in C so it can run tens of millions of times.
Piece values are `py/baselines.py`'s (P=1, N=3, B=3.25, R=5, Q=9). The C copy
reproduces the two details that decide games: a drawn result scores 0.0 at the
root (so it will not shuffle while ahead) and ties are broken uniformly at
random.

Two quantities, kept apart on purpose, because "hangs pieces" is a claim about
the first one:

```
drop(m) = material_before − oracle_value(m)     what the move GIVES AWAY
loss(m) = best_value      − oracle_value(m)     that, PLUS what it fails to win

HANG = drop ≥ 2 pawns        HOLD = drop ≤ 0.25 pawns
```

This file contains piece values and a quiescence search. `docs/FROM_SCRATCH.md`
permits that in a measuring instrument and forbids it in the learning or play
path; `tools/diag_tactics.c` is linked into nothing — not `build/chessrl`, not
`libchessrl.dylib`, not the app — and `make test`'s audit is unaffected.

### It reproduces the external result

```
diag_tactics game --games 200 --sims 160 --threads 8
```

| | score vs material-1 |
|---|---|
| external benchmark (30 games) | 0.050 |
| this harness, 200 games, temp 1 for 12 plies | **0.035** ± 0.035 |
| this harness, 200 games, argmax from move 1 (`--temp0`) | **0.030** ± 0.035 |

Same opponent, same result. Every number below is therefore about the agent
that actually scored 5%.

### Build and run

Nothing in the `Makefile` references this tool; it is built by hand with the
project's own flags:

```sh
cc -O3 -std=c11 -D_DARWIN_C_SOURCE -Wall -Wextra -Wno-unused-parameter \
   -funroll-loops -fno-math-errno -ffp-contract=fast -mcpu=native -Isrc \
   tools/diag_tactics.c build/chess.o build/net.o build/mcts.o \
   -o build/diag_tactics -lm -lpthread
```

Warning-free. `make -j8` and `make test` are untouched and still pass
(7 checks, 0 failures; audit clean).

### Three position corpora

All experiments run against positions generated three ways, because the answer
depends sharply on which one you look at — and that dependence is itself the
finding.

| corpus | how | mean legal moves | of which HANG | mean ply | mean material |
|---|---|---|---|---|---|
| `--source self` | the agent's own games, 160 sims | 30.7 | 18.7% | 55 | +0.49 |
| `--source vs` | the agent's games **against material-1** | 24.0 | 20.7% | 47 | −5.72 |
| `--source random` | uniformly random legal games | 28.2 | 41.7% | 131 | +0.84 |

A position enters a corpus only if it is not in check, at least one legal move
hangs ≥2 pawns, at least two legal moves hold their material, and the side to
move **can** hold what it owns (the oracle's best move is within 0.5 pawns of the
current balance). That last filter matters: without it the corpus fills with
positions where the piece was already gone before the agent moved, and "it gave
a piece away" stops being the event being measured. Positions are sampled
uniformly from complete games, not taken as they come, so the corpus is not
crushed into the first ten plies.

---

## 1. The end-to-end audit: every move of real games

This experiment conditions on nothing at all. It plays complete games and grades
every move the agent makes.

```
diag_tactics game --games 200 --sims 160 --threads 8            # vs material-1
diag_tactics game --games 200 --sims 160 --threads 8 --self     # vs itself
```

| | vs material-1 | vs itself |
|---|---|---|
| result | 2 W / 188 L / 10 D, score **0.035** | 5 W / 4 L / 191 D, score 0.502 |
| agent moves graded | 11 948 | 10 221 |
| **HANG rate** | **0.0715** ± 0.0024 | **0.0248** ± 0.0015 |
| HOLD rate | 0.847 | 0.951 |
| mean drop | **+0.274 pawns/move** | −0.046 pawns/move |
| opponent HANG rate | 0.0012 | 0.0266 |
| games with ≥1 hang | **1.000** | 0.460 |
| modal hangs per game | 5 | 0 |

Material balance from the agent's own view, by ply:

| ply | 0 | 10 | 20 | 30 | 40 | 50 | 60 | 70 |
|---|---|---|---|---|---|---|---|---|
| vs material-1 | −0.20 | −1.60 | −2.90 | −4.26 | −6.04 | −7.66 | −8.86 | −10.40 |
| vs itself | −0.06 | −0.26 | −0.28 | −0.23 | −0.12 | −0.22 | −0.27 | +0.25 |

The arithmetic of the 5% is right there: 7.2% of moves × ~60 agent moves per
game = every game contains a hang, usually five of them, and the opponent keeps
all of it (its own hang rate is 0.12%). Against itself the same agent is nearly
clean — 2.5% — and the material never moves, because the opponent hands
everything back.

**The 100% → 5% cliff between the greedy bot and material-1 is not that
quiescence finds moves the agent cannot see. It is that quiescence *keeps* what
it takes.**

---

## 2. KEY EXPERIMENT — blunder rate against simulation count

```
diag_tactics sims --n 800 --source {vs,self,random} --threads 8
diag_tactics sims --n 400 --source {vs,self,random} --threads 8 --high
```

HANG rate of the move MCTS actually selects (`root_noise` off, fresh tree per
probe, so `sims` is the whole budget):

| sims | `vs` corpus (n=800) | `self` (n=800) | `random` (n=800) |
|---:|---:|---:|---:|
| policy argmax, no search | **0.240** | 0.193 | 0.212 |
| 20 | 0.166 | 0.122 | 0.214 |
| 80 | 0.106 | 0.025 | 0.199 |
| **160** (shipped) | **0.095** | 0.021 | 0.193 |
| 400 | 0.098 | 0.024 | 0.179 |
| 1600 | 0.095 | 0.016 | 0.175 |
| 6400 | **0.070** | 0.014 | 0.152 |
| 25600 (n=400) | **0.052** | 0.018 | 0.113 |
| *uniform random legal mover* | *0.206* | *0.185* | *0.399* |

And end-to-end, in real games, where the score is the thing that matters:

```
diag_tactics game --games {200,200,120,120} --sims {160,400,1600,6400}
```

| sims | score vs material-1 | HANG rate | mean drop/move |
|---:|---:|---:|---:|
| 160 | 0.035 | 0.0715 ± 0.0024 | +0.274 |
| 400 | 0.085 | 0.0620 ± 0.0021 | +0.227 |
| 1600 | 0.142 | 0.0568 ± 0.0025 | +0.182 |
| 6400 | **0.237** | **0.0510** ± 0.0023 | +0.131 |

**Reading.** Search is doing real work — it more than halves the policy's own
blunder rate (0.240 → 0.095) — and more search keeps helping, monotonically.
But the return is logarithmic and the curve does not approach zero: **160× the
budget buys a 2× reduction** (0.105 → 0.052 on the corpus, 0.0715 → 0.0510 in
games) and still loses the match 0.237. A 40× search costs 40× the wall clock;
at 6400 sims the agent is still four standard errors away from not hanging
pieces. Search budget is not the binding constraint, and it is not a route to
one either.

A methodological note that matters for reading the curve: **the `sims = 1` row
is not "the policy".** `mcts_select()` uses `sqrt(N_parent − 1)`, which is
exactly 0 on a node's first descent, so the first descent takes the *first legal
move* and the priors only take over from the second (`src/mcts.c:576-592`, whose
comment already says AlphaZero's `sqrt(sum + 1)` "would cost nothing to switch
to"). The proof that this is what is happening: at `sims = 1` the trained net,
the gen-295 checkpoint and a randomly-initialised net all score **exactly
0.074** on the `vs` corpus. The `POLICY` row above is the real policy argmax,
computed directly from the priors.

---

## 3. Is it the value head?

### 3a. Can it tell a hanging move from a safe one?

```
diag_tactics value --n 800 --source {self,vs,random} --threads 8
```

`safe` is a material-neutral reference move (holds its material and wins none),
`hang` is the largest-drop legal move. Values are from the root mover's view.

| | `self` | `vs` | `random` |
|---|---:|---:|---:|
| mean V after safe move | −0.154 | −0.090 | −0.066 |
| mean V after hanging move | −0.473 | −0.269 | −0.095 |
| **separation** | **+0.319** | **+0.179** | **+0.029** |
| P[V(safe) > V(hang)] | 0.900 | 0.767 | **0.531** |
| after the forced recapture: separation | +0.223 | +0.338 | +0.193 |
| after the recapture: P[correct] | 0.821 | 0.868 | 0.727 |
| corr(V, oracle value of the move) | +0.593 | +0.345 | +0.592 |
| mean worst hang, pawns | 4.95 | 5.59 | 5.36 |

The head is **not blind** — 0.767 on the distribution it actually plays in is
well above chance. It is **badly scaled**. A ~5.6-pawn loss of material moves it
by 0.34 of a 2.0-wide output range, and on the `vs` corpus the position *after
the piece is gone* is still evaluated at **+0.156** — "I am winning" — against
+0.493 for the safe line. On random-play positions it is a coin flip (0.531).

### 3b. Delete one of my own pieces and re-evaluate

The most direct question there is, and it needs no search at all.

```
diag_tactics ablate --n 800 --source self --threads 8
```

| piece removed | best.crl (g1894) | az_v3 (g295) | random weights |
|---|---:|---:|---:|
| pawn (1.00) | −0.2685 | −0.0886 | +0.0001 |
| knight (3.00) | −0.3138 | −0.0379 | +0.0001 |
| bishop (3.25) | −0.2781 | −0.0779 | −0.0002 |
| rook (5.00) | −0.3946 | −0.1829 | −0.0004 |
| queen (9.00) | **−0.5364** | −0.2503 | +0.0001 |
| *mean v before* | *+0.157* | *−0.331* | *−0.001* |

Randomly-initialised weights move by **zero** for every piece — the right
control, and it confirms the probe measures something learned rather than
something architectural.

The trained head has the *order* roughly right and the *scale* badly wrong:
**losing a queen costs only 2× what losing a pawn costs.** From a base of +0.157,
deleting a knight leaves +0.157 − 0.314 = −0.157, i.e. "about equal" while a
piece down. On the `vs` corpus, where the base is +0.552, deleting a knight
leaves **+0.347** — still comfortably winning, a piece down.

### 3c. Calibration — and the reason for all of it

```
diag_tactics calib --games 400 --sims 160 --threads 8 --train-mode   # training settings
diag_tactics calib --games 300 --sims 160 --threads 8                # play settings
diag_tactics calib --games 200 --sims 160 --threads 8 --vs-oracle
```

`--train-mode` reproduces what self-play actually looked like during the run:
Dirichlet root noise on, temperature 1.0 for 20 plies then 0.25. That is the
distribution the value head's labels came from, so it is the one the claim below
has to be made on.

| | **training-settings self-play** | play-settings self-play | **vs material-1** |
|---|---:|---:|---:|
| positions | 67 128 (400 games) | 34 162 (300 games) | 13 051 (200 games) |
| draw rate | 61% | 92% | 7.5% |
| MSE(v, z) / Var(z) | **0.947** | 1.497 | **11.31** |
| R² | **+0.053** | −0.497 | **−10.31** |
| corr(v, material) | +0.678 | +0.600 | +0.865 |

On its own training distribution the value head explains **5% of the variance of
the outcome**. On the distribution it is measured in, it is eleven times worse
than predicting the mean.

**v bucketed, against material-1** (the shape of being wrong):

| v bucket | n | mean v | actual score | gap |
|---|---:|---:|---:|---:|
| [−1.0,−0.8) | 729 | −0.836 | −0.923 | +0.087 |
| [−0.8,−0.6) | 2 442 | −0.706 | −0.760 | +0.054 |
| [+0.2,+0.4) | 347 | +0.311 | −0.827 | +1.138 |
| [+0.6,+0.8) | 1 764 | +0.720 | −0.849 | +1.569 |
| [+0.8,+1.0) | **4 780** | +0.913 | **−0.883** | **+1.796** |

Thirty-seven per cent of all positions in games against material-1 get
`v > 0.8` while the agent goes on to lose 94% of them. This is not an imprecise
head; it is a **systematically wrong** one, and the error is one-sided.

**Now bucket by material instead of by v. This is the whole diagnosis:**

| material balance (pawns) | training self-play: n / mean v / **actual score** | vs material-1: n / mean v / **actual score** |
|---:|---|---|
| ≤ −6 | 10 498 / −0.607 / **−0.535** | 7 193 / −0.247 / **−0.816** |
| −5 | 2 021 / −0.330 / **−0.172** | 547 / +0.693 / **−0.898** |
| −3 | 3 448 / −0.155 / **−0.091** | 545 / +0.791 / **−0.853** |
| **−2** | 4 687 / −0.018 / **−0.036** | 1 837 / +0.855 / **−0.882** |
| −1 | 5 069 / −0.015 / **−0.040** | 72 / +0.824 / −0.958 |
| 0 | 13 794 / +0.163 / **+0.003** | 1 878 / +0.821 / **−0.848** |
| +2 | 4 521 / +0.170 / **+0.052** | 32 / +0.895 / −0.312 |
| +3 | 3 091 / +0.229 / **+0.105** | 37 / +0.717 / +0.000 |
| ≥ +6 | 9 834 / +0.592 / **+0.562** | — |

Read the middle column. **In the self-play that trained this network, being two
pawns down is worth an actual score of −0.036.** One pawn down: −0.040. Three
pawns down: −0.091. Two pawns *up*: +0.052. Material does not begin to predict
the result until the imbalance passes about six pawns, because the opponent is
the same network and hands it straight back: both sides hang at ~2.5% of moves,
the training telemetry logs 23 captures per game, a 48% draw rate and 15% of
games ending in insufficient material at generation 1900.

And now read the value head against it: at −2 pawns it outputs **−0.018** where
the truth is −0.036; at −3, −0.155 against −0.091; at 0, +0.163 against +0.003.
**On its own training distribution the value head is close to correctly
calibrated for material.** The right column is what the same relationship looks
like against an opponent that keeps what it takes: −2 pawns is worth −0.882, and
the head — trained on the left column — says **+0.855**.

The value head is not failing to learn. It is learning a label that is right
about self-play and useless about chess.

## 4. Is it the priors?

```
diag_tactics prior --n 800 --source {self,vs,random} --threads 8
```

| | `self` | `vs` | `random` |
|---|---:|---:|---:|
| prior mass on hanging moves | 0.1451 | 0.1918 | 0.2724 |
| the same mass under a **uniform** prior | 0.1847 | 0.2056 | 0.3991 |
| **ratio (1.0 = blind)** | **0.786** | **0.933** | 0.683 |
| prior on the worst hanging move | 0.0157 | 0.0342 | 0.0228 |
| prior on the oracle-best move | 0.1018 | 0.1010 | 0.2396 |
| mean rank of the worst hanging move | 24.7 | 15.7 | 17.1 |
| **policy argmax hangs material** | **0.193** | **0.240** | 0.212 |
| uniform random legal mover hangs | 0.185 | 0.206 | 0.399 |

On the distribution that decides games — positions from games against
material-1 — the policy head's top move drops ≥2 pawns **24.0% of the time,
against a random-mover baseline of 20.6%**. It is not merely blind to hanging
material; on that distribution it is slightly *worse* than a coin. Its prior
mass on hanging moves is 0.93× uniform.

The direct cost to the search: **19% of the root's prior mass, and therefore
19% of the visit budget at the start of every search, is spent on moves that
drop at least two pawns** (the `visit share on hangs` column of §2 confirms this:
0.126 of visits at 160 sims, decaying only to 0.073 at 6400).

---

## 5. How deep does the tree actually go?

```
diag_tactics tree --n 800 --source vs --sims 160 --threads 8
```

At the shipped 160 simulations, on `vs`-corpus positions (mean 24.0 legal moves):

| | |
|---|---|
| root moves receiving >0 visits | 12.7 of 24.0 (53%) |
| >1 visit | 6.4 |
| >5 visits | 4.8 |
| >20 visits | 2.1 |
| visit share, top move / runner-up | 0.469 / 0.212 |
| nodes allocated per search | 4 602 |

Simulation-path (leaf) depth, plies below the root:

| depth | 0 | 1 | **2** | 3 | 4 | 5 | 6 | 7+ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `vs` | 0.6% | 7.9% | **42.0%** | 23.3% | 20.6% | 3.8% | 1.4% | 0.4% |
| `self` | 0.6% | 15.4% | 36.3% | 31.1% | 12.5% | 2.7% | 0.9% | 0.4% |
| `random` | 0.6% | 9.5% | 25.8% | 27.7% | 22.7% | 7.7% | 4.4% | 1.6% |

Mean leaf depth **2.74** plies (`vs`), deepest 9. **Depth 2 is exactly where a
recapture lives, and 42% of all simulations terminate there.** The arithmetic
does not settle it against the search: the tree is deep enough.

### And it does look at the punishment

For the worst hanging move at the root:

| | `vs` | `self` |
|---|---:|---:|
| visits that edge receives | 3.16 of 160 | 1.13 of 160 |
| its subtree was expanded | 47.5% | 68.8% |
| the refuting capture was searched at all | 19.9% | 6.9% |
| **the child policy head's prior on that refutation** | **0.575** | **0.675** |
| its rank in the child's policy | 1.4 | 1.1 |
| refutation is the child's top-1 prior | 83.6% | 92.5% |

So the *child's* policy head is excellent at spotting the free capture — it
ranks the refutation first 84% of the time with prior 0.57. The blunder edge
simply is not visited enough to use that, because it is not the move the search
is considering. Which brings us to the case that matters:

### When the search actually plays a hanging move (76 of 800 positions)

| | |
|---|---|
| visits it gave that move | **83.0 of 160** |
| Q the search settled on | **+0.2540** (root mover's view) |
| the refutation's prior at the child | 0.341, mean rank 1.6 |
| **the refutation was searched** | **93.4% of the time, 31.1 visits** |
| **Q the search gave the refutation** | **−0.2937** (correctly negative) |

Read that carefully. When the agent hangs a piece it has spent **half its whole
budget** on that move, it **has** examined the refutation in 93% of cases with
31 visits, and it **has** correctly concluded that the refutation is bad for it
(−0.294). It plays the move anyway, because the edge's mean Q is +0.254: the
31 visits at −0.29 are averaged against ~51 visits on the ~23 replies the
opponent will never play, which the value head scores at about **+0.6**.

The search is not failing to see. It is averaging a correct −0.29 against a
wrong +0.6, and the wrong number comes from the value head. If the value head
said −0.9 for "a piece down" instead of −0.29, 31 visits would be enough to sink
the edge.

### A cheap, separate search inefficiency

The same walk shows the `sqrt(N − 1)` first-descent rule costing real work:

| | `vs` | `self` | `random` |
|---|---:|---:|---:|
| expanded nodes with ≥1 child visited, per search | 29 | 34 | 49 |
| of those, only ONE child was ever looked at | 27.3% | 37.6% | 46.9% |
| **the policy's own top move was never visited** | **21.9%** | **32.6%** | **41.6%** |
| …because visit #1 went to move-generation-order child 0 | 21.9% | 32.6% | 41.6% |

At a fifth to two fifths of the expanded nodes in the tree, the single look the
node ever gets is spent on whichever move the generator happened to emit first,
and the policy head is never consulted. This is worth fixing and is nearly free
(`sqrt(sum + 1)`), but it is a second-order effect: it is not what makes the
6400-simulation curve flatten out at 5%.

---

## 6. Phases, and Chess960 vs classical

```
diag_tactics phase --n 800 --source {self,vs,random} --sims 160 --threads 8
```

| corpus | phase | n | HANG @160 | random baseline | prior on hangs | V separation | P[V correct] |
|---|---|---:|---:|---:|---:|---:|---:|
| `self` | opening | 620 | 0.010 | 0.179 | 0.138 | +0.351 | 0.947 |
| | middlegame | 120 | 0.075 | 0.213 | 0.172 | +0.242 | 0.767 |
| | endgame | 60 | 0.033 | 0.191 | 0.168 | +0.148 | 0.683 |
| `vs` | opening | 483 | 0.079 | 0.169 | 0.162 | +0.225 | 0.810 |
| | middlegame | 293 | 0.113 | 0.259 | 0.234 | +0.110 | 0.717 |
| | endgame | 24 | 0.208 | 0.286 | 0.283 | +0.073 | 0.542 |
| `random` | opening | 190 | 0.137 | 0.402 | 0.229 | +0.110 | 0.679 |
| | middlegame | 292 | 0.178 | 0.452 | 0.276 | +0.012 | 0.473 |
| | endgame | 318 | 0.239 | 0.349 | 0.294 | −0.005 | **0.497** |

Phase is classified by material left on the board (≥26 pieces opening, 13–25
middlegame, ≤12 endgame) — an instrument's convention.

The degradation is monotonic and it tracks the value head exactly: as the
position empties, `P[V correct]` falls from 0.95 to chance and the blunder rate
rises with it. On random-play endgames the value head is worth nothing at all
(0.497, separation −0.005).

### Chess960 is not the problem

```
diag_tactics phase --n 800 --source {self,vs,random} --sims 160   # second table
```

| corpus | starts | n | HANG @160 | random baseline | V separation |
|---|---|---:|---:|---:|---:|
| `self` | classical | 800 | **0.022** | 0.180 | +0.326 |
| | chess960 | 800 | 0.048 | 0.186 | +0.182 |
| `vs` | classical | 800 | **0.105** | 0.191 | +0.161 |
| | chess960 | 800 | 0.130 | 0.192 | +0.156 |
| `random` | classical | 800 | **0.142** | 0.414 | +0.046 |
| | chess960 | 800 | 0.142 | 0.406 | +0.037 |

The corpora are equally difficult (identical random baselines), and the agent is
**equal or better on classical starts in all three**. Training on mixed starts
did not hurt classical play; measuring on classical is, if anything, the
favourable case. **There is no cheap fix here.**

---

## 7. Did training improve tactics at all?

```
diag_tactics compare --n 800 --source {self,vs,random} --threads 8
```

Same corpus, three models: the 1894-generation checkpoint, the earlier
295-generation checkpoint, and a randomly-initialised network.

**`vs` corpus (n=800), uniform-random baseline HANG = 0.206:**

| model | HANG @160 | HANG @1600 | prior mass on hangs | V separation |
|---|---:|---:|---:|---:|
| `runs/az_hour/best.crl` (gen 1894) | **0.095** | 0.095 | 0.1918 | **+0.1785** |
| `runs/az_v3/best.crl` (gen 295) | 0.181 | 0.168 | 0.1979 | +0.0430 |
| random weights | 0.287 | 0.287 | 0.2181 | −0.0001 |
| *uniform random legal mover* | *0.206* | *0.206* | *0.2056* | — |

**`self` corpus (n=800), baseline 0.185:**

| model | HANG @160 | HANG @1600 | prior on hangs | V separation |
|---|---:|---:|---:|---:|
| gen 1894 | **0.021** | 0.016 | 0.1451 | **+0.3194** |
| gen 295 | 0.229 | 0.133 | 0.2068 | +0.0137 |
| random weights | 0.463 | 0.463 | 0.2362 | +0.0002 |

**`random` corpus (n=800), baseline 0.399:**

| model | HANG @160 | HANG @1600 | prior on hangs | V separation |
|---|---:|---:|---:|---:|
| gen 1894 | **0.193** | 0.175 | 0.2724 | +0.0285 |
| gen 295 | 0.280 | 0.233 | 0.3521 | +0.0112 |
| random weights | 0.355 | 0.354 | 0.3951 | +0.0000 |

Training **is** working, and the gains against `random` did not come from
somewhere else: the blunder rate fell by a factor of 3 from initialisation, and
by a factor of 2 from generation 295 to 1894.

But look at **which head** improved. Over 1599 generations:

* value separation: 0.043 → 0.179 (`vs`), 0.014 → 0.319 (`self`) — a 4× to 23×
  improvement, and the piece-deletion ablation in §3b agrees (a queen went from
  −0.250 to −0.536 of value).
* prior mass on hanging moves: 0.198 → 0.192, against a uniform value of
  **0.206**. Essentially nothing.

**Almost all of the tactical progress in 1894 generations is in the value head.
The policy head has learned close to nothing about not giving pieces away**,
which is exactly the shape you would expect: the policy target is the MCTS visit
distribution, the visit distribution is driven by the value head, and the value
head only barely distinguishes a hung piece.

---

## 8. Verdict

Ranked, with the evidence.

### 1 — The value head. Binding, and it is a *data* problem, not a capacity problem.

* It is systematically, one-sidedly wrong on the distribution it plays in:
  MSE/Var = **11.3**, R² = **−10.3** against material-1; 37% of positions get
  `v > 0.8` while losing 94% of them. On its own training distribution it is
  MSE/Var = 0.947, R² = **+0.053** (§3c).
* Its material exchange rate is nearly flat: a queen is worth **2×** a pawn to it,
  not 9× (§3b). A knight down reads as "about equal" from the self-play base and
  "clearly winning" from the `vs` base.
* **The root cause is in the training labels.** Measured under the actual
  training settings (root noise, training temperatures, 67 128 positions), being
  two pawns down in self-play is worth an actual score of **−0.036**, one pawn
  down **−0.040**, three pawns down **−0.091** (§3c). Material does not predict
  the result below about six pawns, because the opponent — itself — gives it
  back: 2.5% hang rate on both sides, 23 captures per game, 48% draws, 15% of
  games ending in insufficient material.
* And against that label the head is **nearly correct**: −0.018 predicted where
  the truth is −0.036 at −2 pawns; −0.155 against −0.091 at −3. It is fitting
  its data, and the data has almost nothing in it to fit — hence R² = +0.053.
  This is consistent with the training curves quoted in the brief (value MSE
  against the variance baseline degrading monotonically 0.485 -> 0.655 while the
  policy KL bottomed out at generation 350): the head is not diverging, it is
  sitting near the ceiling its target allows.
* And it is what stops the search: when the agent hangs a piece, the search has
  found the refutation, given it 31 visits and scored it correctly at −0.29 —
  and still plays the move, because ~23 replies the opponent will never make are
  scored at about **+0.6** and the mean comes out at +0.254 (§5).

### 2 — The policy priors. Clearly binding, and barely touched by training.

* The policy argmax hangs ≥2 pawns **24.0%** of the time on `vs`-corpus
  positions, against a uniform-random baseline of **20.6%** (§4).
* Its prior mass on hanging moves is **0.933×** uniform, and 1599 generations
  moved it from 0.218 to 0.192 (§4, §7).
* This costs the search directly: 19% of the visit budget starts out pointed at
  moves that drop a piece, and 12.6% of visits are still there when the search
  ends (§2, §4).
* It is second, not first, because the search does repair most of it
  (0.240 → 0.095) — and because the child-node policy already ranks the
  refutation first 84% of the time (§5), so the priors are not what stops the
  punishment being seen.

### 3 — Search budget. Real, monotonic, and not the constraint.

* The tree already reaches the relevant depth: mean leaf depth 2.74 plies with
  **42% of simulations ending at exactly depth 2**, the ply a recapture lives on
  (§5).
* It already examines the refutation of the move it is about to play, 93.4% of
  the time (§5).
* **160× the budget buys 2×**: corpus HANG 0.105 → 0.052, in-game HANG
  0.0715 → 0.0510, match score 0.035 → 0.237 (§2). Extrapolating that slope, the
  budget needed to stop hanging pieces by search alone is not reachable.
* One free improvement exists inside the search and is worth taking on its own
  merits: `mcts_select()`'s `sqrt(N_parent − 1)` spends the only visit that 22–42%
  of expanded nodes ever get on move-generation order instead of on the policy's
  own top move (§5). `src/mcts.c:576-592` already notes that `sqrt(sum + 1)`
  "would cost nothing to switch to".

### Not a cause

* **Chess960 training.** The agent is equal or better on classical starts in all
  three corpora, on corpora of identical difficulty (§6). Removing 960 from
  training is not the cheap fix it might have been.
* **Tree depth / branching arithmetic.** 160 simulations over ~24 legal moves is
  enough to see a recapture, and it does (§5).

### What this implies for what gets built next

The ranking says the work is on the **value target and the data that produces
it**, not on the network's size and not on the simulation count. Anything that
makes material predictive of the result *in the training data* attacks the root
cause; anything that only makes the search bigger buys a constant factor.

Two things worth noting about the constraint envelope. `docs/FROM_SCRATCH.md`
forbids putting piece values into the learning path, so the fix cannot be
material shaping — that is the exact mistake the A2C trainer already made and
the rebuild already removed. It does **not** forbid changing the *distribution*
the value head is trained on, nor bootstrapping the value target from the
agent's own search: `AZCfg.value_mix` (currently 0.0) mixes `q_search` into the
target and `az.h` already argues that this is inside the contract on the same
grounds as the MCTS visit distribution being the policy target.

The measurable properties to move are in §3c's middle column: how much the
outcome in the training data actually depends on material (currently −0.036 per
two pawns), the symmetric give-and-take that flattens it, and the self-play draw
rate (48% logged during the run, 61% reproduced here at training settings, 95.5%
at play settings). `tools/diag_tactics.c calib --train-mode` and
`tools/diag_tactics.c game` are the two commands that say whether a change
moved either one.

---

## Appendix: reproducing everything

```sh
# build (nothing in the Makefile references the tool)
cc -O3 -std=c11 -D_DARWIN_C_SOURCE -Wall -Wextra -Wno-unused-parameter \
   -funroll-loops -fno-math-errno -ffp-contract=fast -mcpu=native -Isrc \
   tools/diag_tactics.c build/chess.o build/net.o build/mcts.o \
   -o build/diag_tactics -lm -lpthread

B=build/diag_tactics

# §0/§1  end-to-end audits, and the instrument's agreement with the benchmark
for S in 160 400 1600 6400; do $B game --games 200 --sims $S --threads 8; done
$B game --games 200 --sims 160 --threads 8 --self
$B game --games 200 --sims 160 --threads 8 --temp0

# §2..§7  the full battery on each corpus
for SRC in vs self random; do $B all --n 800 --source $SRC --threads 8; done
for SRC in vs self random; do $B sims --n 400 --source $SRC --threads 8 --high; done

# §3b  the piece-deletion ablation (also runs the earlier checkpoint and random weights)
$B ablate --n 800 --source self --threads 8
$B ablate --n 800 --source vs   --threads 8

# §3c  calibration -- training settings, play settings, and vs the oracle
$B calib --games 400 --sims 160 --threads 8 --train-mode
$B calib --games 300 --sims 160 --threads 8
$B calib --games 200 --sims 160 --threads 8 --vs-oracle

# look at the corpus itself
$B corpus --n 40 --source vs --threads 8
```

Default seed 20260912, 8 threads throughout. Corpus generation is deterministic
given `(seed, threads)`: each thread owns a fixed contiguous slice of the output,
so the corpus is reproducible but changes if the thread count changes. Total
runtime for the whole battery is about 12 minutes on an 8-core M3.

All searches in the probes run with `reuse = 0` and a cleared evaluation cache,
so `sims` is always the entire budget for that position and no measurement
inherits a tree from the previous one. The shipped play path leaves `reuse` on,
which makes a move cheaper rather than the tree bigger (`mcts.h`), so this is the
conservative choice — and the end-to-end audit reproduces the external match
result either way.
