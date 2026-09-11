# BENCHMARK -- how strong is the champion, really?

`runs/pilot/best.crl` reports an internal Elo of about 1750.  That number is
**self-referential**: it is a rating computed inside one population, anchored
only to frozen snapshots of that population's own past.  It measures "stronger
than it used to be" and says nothing about absolute strength.

`py/benchmark.py` measures the champion against opponents from outside the
population and reports the result with an honest error bar.

## Running it

```sh
make -j8 lib
python3 py/benchmark.py --model runs/pilot/best.crl --games 200            # one match
python3 py/benchmark.py --ladder --ladder-games 40                         # the ladder
python3 py/benchmark.py --ladder --opponent 'uci:/opt/homebrew/bin/stockfish'
python3 py/benchmark.py --self-ladder --self-games 60                      # search vs policy
```

Every run appends a section to this file.  `--no-md` suppresses that,
`--json PATH` writes the complete result including every game's move list.

## Opponents

| name | what it is |
| --- | --- |
| `random` | a uniformly random legal move.  Needs nothing installed. |
| `material` | 1-ply greedy on material with uniform random tie-breaking; a mate scores 10000, any drawn result scores 0.  It grabs hanging pieces and takes mate in 1, and hangs everything it owns. |
| `uci:<cmd>` | any external UCI engine, driven over stdin/stdout pipes.  `--opp-option "Skill Level=0"` passes options through, `--opp-go "nodes=100"` controls its search. |

## Method

**Colour balance and pairing.**  Games are played in pairs.  Both games of a
pair start from the identical position; the champion has White in the first and
Black in the second.  Both engines therefore get both sides of every opening,
which removes the first-move advantage from the comparison.

**Opening diversity without a book.**  Each pair's start position is produced by
playing `--opening-plies` (default 4) uniformly random legal plies from the
initial position, drawn from a seeded RNG.  Lines are de-duplicated.  Without
this every game between two near-deterministic engines would be the same game.

**Reproducibility.**  `--seed` fixes the openings, and every game additionally
gets its own RNG seeded from `(seed, game index)`, so the random and material
opponents make identical choices whatever `--threads` is set to.  Two runs with
the same seed and the same flags produce the same games.

**Score and Elo.**  A win is 1, a draw 0.5, a loss 0.  Elo difference is the
standard logistic inverse

```
elo = -400 * log10(1/score - 1)
```

so 50% is exactly 0 Elo and 75% is +191.

**Confidence interval.**  The interval is computed on the *pair* scores, not the
game scores.  The two games of a pair share an opening and are played by
engines that are close to deterministic, so they are strongly correlated;
treating them as two independent observations would understate the error.  The
standard error is the empirical standard error of the mean of the pair scores,
which accounts for draws automatically (an all-draws match has less spread than
a match split evenly between wins and losses).  The 95% interval on the score is
transformed through the Elo formula to give the Elo interval.

Degenerate cases are reported, not papered over:

* If every game has the identical result the empirical variance is zero.  The
  harness falls back on the rule of three -- with `n` games and no game showing
  a different outcome, the 95% one-sided bound on the rate of that unseen
  outcome is `1 - 0.05**(1/n)` -- and reports which method it used.
* If the score is exactly 0% or 100% the Elo difference is infinite.  The
  harness says so and prints a **one-sided bound** instead of a number: with 0
  points in `n` games the true score is below `1 - 0.05**(1/n)` with 95%
  confidence, which transforms to an Elo ceiling.

**Likelihood of superiority** is the usual match formula
`0.5*(1 + erf((W-L)/sqrt(2(W+L))))`; draws carry no information about which side
is better and do not enter it.

**Adjudication.**  A game is a draw once it reaches `--ply-cap` plies (default
300); those games are counted separately in the report.  `--resign-cp N` (off by
default) additionally adjudicates a game when the *opponent* reports an
evaluation of at least N centipawns, in a consistent direction, for
`--resign-plies` consecutive plies of its own.

It is off by default for a measured reason.  Adjudicating on one side's own
evaluation hands that side the benefit of the doubt: in a 6-game probe against
Stockfish at `go nodes 100`, `--resign-cp 400 --resign-plies 6` turned a
1-2-3 (33.3%) result into 0-0-6 (0.0%), because Stockfish repeatedly reached
+400 in positions our engine went on to hold or win.  Use it to save wall time
on long shutouts, not to produce a headline number.

**The ladder.**  A single match against one strong opponent tells you nothing
when the score is 0%.  `--ladder` plays a short match against a graded series --
`random`, `material`, then, if an external engine is given, `go nodes 1`, `10`,
`100`, `1000`, then its lowest `Skill Level`, then `UCI_LimitStrength` at its
`UCI_Elo` floor.  Rungs whose options the engine does not advertise are skipped.
The ladder stops early once the champion drops below `--ladder-stop` (default
5%), because further rungs would only produce more 0%s.

Two details matter:

* A rung that carries a real **rating anchor** (the `UCI_Elo` rung does) is
  played *even if the ladder has already stopped* -- marked `*` in the table.
  A shutout against a calibrated opponent still converts into an absolute
  bound, and an absolute number is the entire point.  `--no-anchor-rung` turns
  that off.
* The 50% crossing is computed over the rungs sorted by **measured** score, not
  by the order they were configured in.  The configured order is only a guess at
  the strength ordering -- Stockfish at `nodes 1000` and Stockfish at
  `UCI_Elo 1320` are not obviously ranked against each other -- and the match
  results are the evidence.  When the guess and the measurement disagree, the
  report says so.

**Anchoring, and what the anchored number is worth.**  When a rung has an
`anchor_elo`, the measured Elo difference is added to it: `UCI_Elo 1320` plus a
measured `-58` gives `~1262`.  That number inherits every assumption in the
anchor:

1. Stockfish's own `UCI_Elo` calibration is approximate, and is calibrated at
   normal time controls, not at the `--opp-go movetime` used here.
2. The logistic Elo model is only trustworthy over modest differences.  An
   anchor the champion scores 3% against implies a ~600-point extrapolation from
   two wins, and should be given very little weight.  Prefer anchors where the
   score is near 50%.
3. It is a *relative* rating against one engine's scale, not a FIDE rating, and
   not a rating against humans.

The right way to use it is to run several anchors and check that the estimates
agree.  If they do not, the disagreement is the honest error bar.

**The self-ladder.**  `--self-ladder` plays the champion's raw policy head
(argmax over the policy, no search at all) against the same champion using
alpha-beta at depths 1, 2, 4 and 6.  Each side runs its own engine handle, and a
fresh one per game, so no transposition table is shared or carried over.  The
result separates what the *network* learned from what the hand-written material
evaluation inside the search adds.

**Effort accounting.**  Every match reports moves, ms/move and nodes/move for
both sides, plus how many searches returned a depth below the target.  If that
last number is large the movetime cap is binding and the "depth N" label is a
lie -- raise `--movetime`.  (At depths 1-6 on this model the default 1000ms cap
never binds; the worst observed depth-6 search was 570ms.)

## Headline: the 2026-09-09 calibration

Every number below is measured, and every run that produced it is recorded in
`## Runs`.  Model: `runs/pilot/best.crl`, generation 270, champion agent 143.
External opponent: **Stockfish 19** (`/opt/homebrew/bin/stockfish`), driven over
pipes at `go movetime 100`.  Seed 20260909, 4 random opening plies, colour-
reversed pairs, 300-ply cap.

### The champion as shipped (alpha-beta depth 4)

| opponent | games | W-D-L | score | Elo difference [95% CI] |
| --- | ---: | ---: | ---: | ---: |
| `random` | 60 | 60-0-0 | 100.0% | > +516 (one-sided) |
| `material` (1-ply greedy) | 60 | 60-0-0 | 100.0% | > +516 (one-sided) |
| Stockfish `go nodes 1` | 60 | 6-32-22 | 36.7% | -95 [-159, -37] |
| Stockfish `go nodes 10` | 60 | 5-34-21 | 36.7% | -95 [-156, -39] |
| Stockfish `go nodes 100` | 60 | 3-13-44 | 15.8% | -290 [-444, -196] |
| Stockfish `go nodes 1000` | 60 | 0-0-60 | 0.0% | < -516 (one-sided) |
| **Stockfish `UCI_Elo 1320`** | 60 | 23-4-33 | **41.7%** | **-58 [-140, +18]** |

**Anchored estimate: ~1262 Elo on Stockfish's `UCI_Elo` scale, 95% CI
[1180, 1338].**

Three further anchors were run as a consistency check, because a single anchor
proves nothing:

| anchor | W-D-L | score | implied champion Elo |
| ---: | ---: | ---: | ---: |
| `UCI_Elo 1320` | 23-4-33 | 41.7% | **1262** [1180, 1338] |
| `UCI_Elo 1500` | 15-2-43 | 26.7% | **1324** [1228, 1400] |
| `UCI_Elo 1700` | 1-3-56 | 4.2% | 1155 -- a 545-point extrapolation from one win, distrust it |
| `UCI_Elo 1900` | 2-0-58 | 3.3% | 1315 -- likewise |

The two anchors close enough to 50% to be trustworthy agree to within ~60 Elo.
**Call it 1250-1350 on Stockfish's scale**, with the caveats in the Method
section above: that is one engine's approximate self-calibration, at a 100ms
move time, not a FIDE rating.

### What the network actually learned, as opposed to what the search adds

The champion's policy head, played on its own with no search at all:

| opponent | games | W-D-L | score | Elo difference [95% CI] |
| --- | ---: | ---: | ---: | ---: |
| `random` | 60 | 18-42-0 | 65.0% | +108 [+70, +148] |
| `material` (1-ply greedy) | 60 | 2-57-1 | **50.8%** | **+6 [-14, +26]** |
| Stockfish `go nodes 1` | 60 | 0-3-57 | 2.5% | -636 [-inf, -503] |
| Stockfish `UCI_Elo 1320` | 60 | 0-1-59 | 0.8% | -830 [-inf, -639] |

**The raw policy is statistically indistinguishable from a one-ply greedy
material grabber** (+6 Elo, CI straddling zero, LOS 72%, 95% of games drawn).
Its 50% crossing sits 2% of the way from `material` towards Stockfish's weakest
possible setting.

And directly, champion-against-itself, raw policy versus the same network with
search on top:

| search depth | policy's W-D-L | policy score | Elo the search adds |
| --- | ---: | ---: | ---: |
| 1 | 0-17-43 | 14.2% | +313 [+243, +411] |
| 2 | 0-3-57 | 2.5% | +636 [+503, +inf] |
| 4 | 0-0-60 | 0.0% | > +516 (one-sided) |
| 6 | 0-0-60 | 0.0% | > +516 (one-sided) |

The policy head does not win a single game out of 240 against its own search,
and loses 43-17-0 to a search only **one ply** deep.

### What this means for the reported 1750

* The trainer's 1750 is not merely uncalibrated, it is **~490 Elo above** the
  externally anchored estimate of the full engine (1750 - 1262; at least 412
  even at the friendly end of the CI).
* Against the network alone the gap is far worse.  The policy scored 0.8%
  against `UCI_Elo 1320`, which bounds it **below ~680 Elo** on that scale with
  95% confidence -- so the reported 1750 sits more than **1000 points** above
  where the learned policy actually plays.
* Worse, it credits the *network* with strength that comes from the
  hand-written alpha-beta and its material evaluation.  Strip the search out and
  the learned policy plays at the level of "take the piece if it is free".
* The degenerate training equilibrium shows up directly in the games: raw policy
  versus `random` is 70% draws, and 32 of those 60 games ended in **stalemate**
  -- the network reaches winning material and cannot convert.

### Reproducing

```sh
make -j8 lib && make -j8
python3 py/benchmark.py --ladder --ladder-games 60 --threads 6 --seed 20260909 \
    --opponent 'uci:/opt/homebrew/bin/stockfish' --opp-go movetime=100
python3 py/benchmark.py --ladder --policy-only --ladder-games 60 --threads 6 --seed 20260909 \
    --opponent 'uci:/opt/homebrew/bin/stockfish' --opp-go movetime=100
python3 py/benchmark.py --self-ladder --self-games 60 --self-depths 1,2,4,6 --threads 6 --seed 20260909
python3 py/benchmark.py --games 60 --opponent random --threads 6 --seed 20260909
for E in 1500 1700 1900; do
  python3 py/benchmark.py --games 60 --threads 6 --seed 20260909 \
    --opponent 'uci:/opt/homebrew/bin/stockfish' \
    --opp-option UCI_LimitStrength=true --opp-option "UCI_Elo=$E" --opp-go movetime=100
done
```

Results are identical at any `--threads` value: openings come from the seed, each
game gets its own RNG seeded from `(seed, game index)`, and the champion reloads
its engine handle per game so no transposition table leaks between games.

## Runs

### 2026-09-09 20:07:22Z

```
python3 py/benchmark.py --ladder --ladder-games 60 --threads 6 --seed 20260909 --opponent uci:/opt/homebrew/bin/stockfish --opp-go movetime=100 --json /private/tmp/claude-501/-Users-johnanthonyheaney-Desktop-ChessRL-ChessRL/48288563-0aad-4064-8914-b56df27a204e/scratchpad/ladder_final_d4.json --no-md --quiet
```

* model `/Users/johnanthonyheaney/Desktop/ChessRL/ChessRL/runs/pilot/best.crl`, generation 270, champion agent 143 (internal Elo 1750)
* seed 20260909, 4 random opening plies, 300-ply cap, 6 thread(s), commit `e84c8c6`

**Ladder** -- 60 games per rung

| rung | n | W-D-L | score | Elo diff [95% CI] | LOS |
| --- | ---: | ---: | ---: | ---: | ---: |
| random | 60 | 60-0-0 | 100.0% | > +516 | 100% |
| material | 60 | 60-0-0 | 100.0% | > +516 | 100% |
| Stockfish 19 nodes=1 | 60 | 6-32-22 | 36.7% | -95 [-159, -37] | 0% |
| Stockfish 19 nodes=10 | 60 | 5-34-21 | 36.7% | -95 [-156, -39] | 0% |
| Stockfish 19 nodes=100 | 60 | 3-13-44 | 15.8% | -290 [-444, -196] | 0% |
| Stockfish 19 nodes=1000 | 60 | 0-0-60 | 0.0% | < -516 | 0% |
| Stockfish 19 Skill=0 | - | - | - | - | *the champion scored below 5% on 'Stockfish 19 nodes=1000'* |
| Stockfish 19 UCI_Elo=1320 | 60 | 23-4-33 | 41.7% | -58 [-140, +18] | 9% |

50% crossing: 50% falls between 'material' (100.0%) and 'Stockfish 19 UCI_Elo=1320' (41.7%), 86% of the way up

Anchored estimate via Stockfish 19 UCI_Elo=1320 (anchor 1320): **1262 Elo** [1180, 1338]

<details><summary>full terminal report</summary>

```
==============================================================================
ChessRL external benchmark
==============================================================================
model      : /Users/johnanthonyheaney/Desktop/ChessRL/ChessRL/runs/pilot/best.crl
generation : 270   agents: 256   champion: agent 143 (internal Elo 1750)
seed       : 20260909   opening plies: 4   ply cap: 300   threads: 6

------------------------------------------------------------------------------
LADDER  (60 games per rung, stop below 5%)
------------------------------------------------------------------------------
rung                          n    W-D-L   score  Elo diff [95% CI]   LOS  notes
---------------------------  --  -------  ------  -----------------  ----  ------------------------------------------------------------------
random                       60   60-0-0  100.0%             > +516  100%  0% draws, 35 plies avg
material                     60   60-0-0  100.0%             > +516  100%  0% draws, 35 plies avg
Stockfish 19 nodes=1         60  6-32-22   36.7%    -95 [-159, -37]    0%  53% draws, 217 plies avg
Stockfish 19 nodes=10        60  5-34-21   36.7%    -95 [-156, -39]    0%  57% draws, 222 plies avg
Stockfish 19 nodes=100       60  3-13-44   15.8%  -290 [-444, -196]    0%  22% draws, 166 plies avg
Stockfish 19 nodes=1000      60   0-0-60    0.0%             < -516    0%  0% draws, 91 plies avg
Stockfish 19 Skill=0          -        -       -                  -     -  skipped: the champion scored below 5% on 'Stockfish 19 nodes=1000'
Stockfish 19 UCI_Elo=1320 *  60  23-4-33   41.7%    -58 [-140, +18]    9%  7% draws, 135 plies avg
  * played after the ladder had already stopped, because this rung carries a real
    rating anchor and even a shutout against it gives an absolute bound.

50% crossing: 50% falls between 'material' (100.0%) and 'Stockfish 19 UCI_Elo=1320' (41.7%), 86% of the way up
  (the configured rung order did NOT match the measured one; the
   crossing above uses the measured order: Stockfish 19 nodes=1000 < Stockfish 19 nodes=100 < Stockfish 19 nodes=10 < Stockfish 19 nodes=1 < Stockfish 19 UCI_Elo=1320 < material < random)
ANCHORED ESTIMATE via Stockfish 19 UCI_Elo=1320 (anchor 1320 Elo): champion ~ 1262 Elo [1180, 1338]
```

</details>


### 2026-09-09 20:24:00Z

```
python3 py/benchmark.py --ladder --policy-only --ladder-games 60 --threads 6 --seed 20260909 --opponent uci:/opt/homebrew/bin/stockfish --opp-go movetime=100 --json /private/tmp/claude-501/-Users-johnanthonyheaney-Desktop-ChessRL-ChessRL/48288563-0aad-4064-8914-b56df27a204e/scratchpad/ladder_final_policy.json --no-md --quiet
```

* model `/Users/johnanthonyheaney/Desktop/ChessRL/ChessRL/runs/pilot/best.crl`, generation 270, champion agent 143 (internal Elo 1750)
* seed 20260909, 4 random opening plies, 300-ply cap, 6 thread(s), commit `e84c8c6`

**Ladder** -- 60 games per rung

| rung | n | W-D-L | score | Elo diff [95% CI] | LOS |
| --- | ---: | ---: | ---: | ---: | ---: |
| random | 60 | 18-42-0 | 65.0% | +108 [+70, +148] | 100% |
| material | 60 | 2-57-1 | 50.8% | +6 [-14, +26] | 72% |
| Stockfish 19 nodes=1 | 60 | 0-3-57 | 2.5% | -636 [-inf, -503] | 0% |
| Stockfish 19 nodes=10 | - | - | - | - | *the champion scored below 5% on 'Stockfish 19 nodes=1'* |
| Stockfish 19 nodes=100 | - | - | - | - | *the champion scored below 5% on 'Stockfish 19 nodes=1'* |
| Stockfish 19 nodes=1000 | - | - | - | - | *the champion scored below 5% on 'Stockfish 19 nodes=1'* |
| Stockfish 19 Skill=0 | - | - | - | - | *the champion scored below 5% on 'Stockfish 19 nodes=1'* |
| Stockfish 19 UCI_Elo=1320 | 60 | 0-1-59 | 0.8% | -830 [-inf, -639] | 0% |

50% crossing: 50% falls between 'material' (50.8%) and 'Stockfish 19 nodes=1' (2.5%), 2% of the way up

Anchored estimate via Stockfish 19 UCI_Elo=1320 (anchor 1320): **490 Elo** [-inf, 681]

<details><summary>full terminal report</summary>

```
==============================================================================
ChessRL external benchmark
==============================================================================
model      : /Users/johnanthonyheaney/Desktop/ChessRL/ChessRL/runs/pilot/best.crl
generation : 270   agents: 256   champion: agent 143 (internal Elo 1750)
seed       : 20260909   opening plies: 4   ply cap: 300   threads: 6

------------------------------------------------------------------------------
LADDER  (60 games per rung, stop below 5%)
------------------------------------------------------------------------------
rung                          n    W-D-L  score  Elo diff [95% CI]   LOS  notes
---------------------------  --  -------  -----  -----------------  ----  ---------------------------------------------------------------
random                       60  18-42-0  65.0%   +108 [+70, +148]  100%  70% draws, 74 plies avg
material                     60   2-57-1  50.8%      +6 [-14, +26]   72%  95% draws, 99 plies avg
Stockfish 19 nodes=1         60   0-3-57   2.5%  -636 [-inf, -503]    0%  5% draws, 78 plies avg
Stockfish 19 nodes=10         -        -      -                  -     -  skipped: the champion scored below 5% on 'Stockfish 19 nodes=1'
Stockfish 19 nodes=100        -        -      -                  -     -  skipped: the champion scored below 5% on 'Stockfish 19 nodes=1'
Stockfish 19 nodes=1000       -        -      -                  -     -  skipped: the champion scored below 5% on 'Stockfish 19 nodes=1'
Stockfish 19 Skill=0          -        -      -                  -     -  skipped: the champion scored below 5% on 'Stockfish 19 nodes=1'
Stockfish 19 UCI_Elo=1320 *  60   0-1-59   0.8%  -830 [-inf, -639]    0%  2% draws, 62 plies avg
  * played after the ladder had already stopped, because this rung carries a real
    rating anchor and even a shutout against it gives an absolute bound.

50% crossing: 50% falls between 'material' (50.8%) and 'Stockfish 19 nodes=1' (2.5%), 2% of the way up
ANCHORED ESTIMATE via Stockfish 19 UCI_Elo=1320 (anchor 1320 Elo): champion ~ 490 Elo [-inf, 681]
```

</details>


### 2026-09-09 20:21:08Z

```
python3 py/benchmark.py --self-ladder --self-games 60 --self-depths 1,2,4,6 --threads 6 --seed 20260909 --json /private/tmp/claude-501/-Users-johnanthonyheaney-Desktop-ChessRL-ChessRL/48288563-0aad-4064-8914-b56df27a204e/scratchpad/selfladder.json --no-md --quiet
```

* model `/Users/johnanthonyheaney/Desktop/ChessRL/ChessRL/runs/pilot/best.crl`, generation 270, champion agent 143 (internal Elo 1750)
* seed 20260909, 4 random opening plies, 300-ply cap, 6 thread(s), commit `e84c8c6`

**Self-ladder** -- champion raw policy (argmax, no search), 60 games per rung

| search | n | policy W-D-L | policy score | Elo the search adds | draws |
| --- | ---: | ---: | ---: | ---: | ---: |
| depth 1 | 60 | 0-17-43 | 14.2% | +313 [+243, +411] | 28% |
| depth 2 | 60 | 0-3-57 | 2.5% | +636 [+503, +inf] | 5% |
| depth 4 | 60 | 0-0-60 | 0.0% | *infinite (policy scored 0%)* | 0% |
| depth 6 | 60 | 0-0-60 | 0.0% | *infinite (policy scored 0%)* | 0% |

<details><summary>full terminal report</summary>

```
==============================================================================
ChessRL external benchmark
==============================================================================
model      : /Users/johnanthonyheaney/Desktop/ChessRL/ChessRL/runs/pilot/best.crl
generation : 270   agents: 256   champion: agent 143 (internal Elo 1750)
seed       : 20260909   opening plies: 4   ply cap: 300   threads: 6

------------------------------------------------------------------------------
SELF-LADDER  (champion raw policy (argmax, no search), 60 games per rung)
------------------------------------------------------------------------------
search    n  policy W-D-L  policy score  Elo the search adds [95% CI]  draws  plies
-------  --  ------------  ------------  ----------------------------  -----  -----
depth 1  60       0-17-43         14.2%             +313 [+243, +411]    28%     91
depth 2  60        0-3-57          2.5%             +636 [+503, +inf]     5%     73
depth 4  60        0-0-60          0.0%                        > +516     0%     48
depth 6  60        0-0-60          0.0%                        > +516     0%     46

Read this as: how much of the champion's playing strength comes from the
network's policy head alone, and how much is added by the hand-written
alpha-beta search sitting on top of it.
```

</details>


### 2026-09-09 20:01:56Z

```
python3 py/benchmark.py --games 60 --opponent random --threads 6 --seed 20260909 --json /private/tmp/claude-501/-Users-johnanthonyheaney-Desktop-ChessRL-ChessRL/48288563-0aad-4064-8914-b56df27a204e/scratchpad/vs_random_d4.json --quiet
```

* model `/Users/johnanthonyheaney/Desktop/ChessRL/ChessRL/runs/pilot/best.crl`, generation 270, champion agent 143 (internal Elo 1750)
* seed 20260909, 4 random opening plies, 300-ply cap, 6 thread(s), commit `e84c8c6`

**Match** -- champion d4 vs random

| games | W-D-L | score | Elo diff [95% CI] | LOS | draws | avg plies | ply-cap draws |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 60 | 60-0-0 | 100.0% | > +516 | 100% | 0% | 35 | 0 |

> scored 60/60 -- the Elo difference is positive infinity; reporting a one-sided 95% bound instead

<details><summary>full terminal report</summary>

```
==============================================================================
ChessRL external benchmark
==============================================================================
model      : /Users/johnanthonyheaney/Desktop/ChessRL/ChessRL/runs/pilot/best.crl
generation : 270   agents: 256   champion: agent 143 (internal Elo 1750)
seed       : 20260909   opening plies: 4   ply cap: 300   threads: 6

------------------------------------------------------------------------------
MATCH
------------------------------------------------------------------------------
champion d4  vs  random
  alpha-beta depth 4, movetime cap 1000ms
  uniformly random legal move
games 60   W 60  D 0  L 0   score 100.0%  (95% CI 90.5%-100.0%)
Elo diff (champion d4 POV): > +516
  note: scored 60/60 -- the Elo difference is positive infinity; reporting a one-sided 95% bound instead
CI basis: pairs (30 complete pairs), rule-of-three (all outcomes identical)
LOS (P that champion d4 is genuinely stronger): 100.0%
pairs: swept 30, split 0, both drawn 0, swept against 0
as white 100.0% (30 games), as black 100.0% (30 games)
draw rate 0.0%   avg length 35.0 plies (median 33, max 80)   hit the 300-ply cap: 0
terminations: checkmate 60
4.4s elapsed
```

</details>


### 2026-09-09 20:13:32Z

```
python3 py/benchmark.py --games 60 --threads 6 --seed 20260909 --opponent uci:/opt/homebrew/bin/stockfish --opp-option UCI_LimitStrength=true --opp-option UCI_Elo=1500 --opp-go movetime=100 --json /private/tmp/claude-501/-Users-johnanthonyheaney-Desktop-ChessRL-ChessRL/48288563-0aad-4064-8914-b56df27a204e/scratchpad/anchor_1500.json --no-md --quiet
```

* model `/Users/johnanthonyheaney/Desktop/ChessRL/ChessRL/runs/pilot/best.crl`, generation 270, champion agent 143 (internal Elo 1750)
* seed 20260909, 4 random opening plies, 300-ply cap, 6 thread(s), commit `e84c8c6`

**Match** -- champion d4 vs uci:stockfish

| games | W-D-L | score | Elo diff [95% CI] | LOS | draws | avg plies | ply-cap draws |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 60 | 15-2-43 | 26.7% | -176 [-272, -100] | 0% | 3% | 106 | 1 |

<details><summary>full terminal report</summary>

```
==============================================================================
ChessRL external benchmark
==============================================================================
model      : /Users/johnanthonyheaney/Desktop/ChessRL/ChessRL/runs/pilot/best.crl
generation : 270   agents: 256   champion: agent 143 (internal Elo 1750)
seed       : 20260909   opening plies: 4   ply cap: 300   threads: 6

------------------------------------------------------------------------------
MATCH
------------------------------------------------------------------------------
champion d4  vs  uci:stockfish
  alpha-beta depth 4, movetime cap 1000ms
  /opt/homebrew/bin/stockfish; go movetime 100; UCI_Elo=1500, UCI_LimitStrength=true
games 60   W 15  D 2  L 43   score 26.7%  (95% CI 17.3%-36.0%)
Elo diff (champion d4 POV): -176 [-272, -100]
CI basis: pairs (30 complete pairs), normal approximation on the mean
LOS (P that champion d4 is genuinely stronger): 0.0%
pairs: swept 0, split 16, both drawn 0, swept against 14
as white 33.3% (30 games), as black 20.0% (30 games)
draw rate 3.3%   avg length 106.0 plies (median 88, max 300)   hit the 300-ply cap: 1
terminations: checkmate 58, ply cap 1, insufficient material 1
champion d4: 3052 moves, 34.0 ms/move, 4782 nodes/move, 166 searches (5%) reported a depth below the target (movetime cap, forced move, or mate found)
uci:stockfish: 3066 moves, 97.1 ms/move
77.2s elapsed
```

</details>


### 2026-09-09 20:14:49Z

```
python3 py/benchmark.py --games 60 --threads 6 --seed 20260909 --opponent uci:/opt/homebrew/bin/stockfish --opp-option UCI_LimitStrength=true --opp-option UCI_Elo=1700 --opp-go movetime=100 --json /private/tmp/claude-501/-Users-johnanthonyheaney-Desktop-ChessRL-ChessRL/48288563-0aad-4064-8914-b56df27a204e/scratchpad/anchor_1700.json --no-md --quiet
```

* model `/Users/johnanthonyheaney/Desktop/ChessRL/ChessRL/runs/pilot/best.crl`, generation 270, champion agent 143 (internal Elo 1750)
* seed 20260909, 4 random opening plies, 300-ply cap, 6 thread(s), commit `e84c8c6`

**Match** -- champion d4 vs uci:stockfish

| games | W-D-L | score | Elo diff [95% CI] | LOS | draws | avg plies | ply-cap draws |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 60 | 1-3-56 | 4.2% | -545 [-1353, -418] | 0% | 5% | 101 | 0 |

<details><summary>full terminal report</summary>

```
==============================================================================
ChessRL external benchmark
==============================================================================
model      : /Users/johnanthonyheaney/Desktop/ChessRL/ChessRL/runs/pilot/best.crl
generation : 270   agents: 256   champion: agent 143 (internal Elo 1750)
seed       : 20260909   opening plies: 4   ply cap: 300   threads: 6

------------------------------------------------------------------------------
MATCH
------------------------------------------------------------------------------
champion d4  vs  uci:stockfish
  alpha-beta depth 4, movetime cap 1000ms
  /opt/homebrew/bin/stockfish; go movetime 100; UCI_Elo=1700, UCI_LimitStrength=true
games 60   W 1  D 3  L 56   score 4.2%  (95% CI 0.0%-8.3%)
Elo diff (champion d4 POV): -545 [-1353, -418]
CI basis: pairs (30 complete pairs), normal approximation on the mean
LOS (P that champion d4 is genuinely stronger): 0.0%
pairs: swept 0, split 4, both drawn 0, swept against 26
as white 6.7% (30 games), as black 1.7% (30 games)
draw rate 5.0%   avg length 101.4 plies (median 88, max 266)   hit the 300-ply cap: 0
terminations: checkmate 57, fifty-move 2, insufficient material 1
champion d4: 2907 moves, 38.7 ms/move, 4979 nodes/move, 177 searches (6%) reported a depth below the target (movetime cap, forced move, or mate found)
uci:stockfish: 2935 moves, 98.0 ms/move
72.1s elapsed
```

</details>


### 2026-09-09 20:16:01Z

```
python3 py/benchmark.py --games 60 --threads 6 --seed 20260909 --opponent uci:/opt/homebrew/bin/stockfish --opp-option UCI_LimitStrength=true --opp-option UCI_Elo=1900 --opp-go movetime=100 --json /private/tmp/claude-501/-Users-johnanthonyheaney-Desktop-ChessRL-ChessRL/48288563-0aad-4064-8914-b56df27a204e/scratchpad/anchor_1900.json --no-md --quiet
```

* model `/Users/johnanthonyheaney/Desktop/ChessRL/ChessRL/runs/pilot/best.crl`, generation 270, champion agent 143 (internal Elo 1750)
* seed 20260909, 4 random opening plies, 300-ply cap, 6 thread(s), commit `e84c8c6`

**Match** -- champion d4 vs uci:stockfish

| games | W-D-L | score | Elo diff [95% CI] | LOS | draws | avg plies | ply-cap draws |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 60 | 2-0-58 | 3.3% | -585 [-inf, -427] | 0% | 0% | 99 | 0 |

> lower end of the score interval reaches 0%, so the lower Elo bound is -inf

<details><summary>full terminal report</summary>

```
==============================================================================
ChessRL external benchmark
==============================================================================
model      : /Users/johnanthonyheaney/Desktop/ChessRL/ChessRL/runs/pilot/best.crl
generation : 270   agents: 256   champion: agent 143 (internal Elo 1750)
seed       : 20260909   opening plies: 4   ply cap: 300   threads: 6

------------------------------------------------------------------------------
MATCH
------------------------------------------------------------------------------
champion d4  vs  uci:stockfish
  alpha-beta depth 4, movetime cap 1000ms
  /opt/homebrew/bin/stockfish; go movetime 100; UCI_Elo=1900, UCI_LimitStrength=true
games 60   W 2  D 0  L 58   score 3.3%  (95% CI 0.0%-7.9%)
Elo diff (champion d4 POV): -585 [-inf, -427]
  note: lower end of the score interval reaches 0%, so the lower Elo bound is -inf
CI basis: pairs (30 complete pairs), normal approximation on the mean
LOS (P that champion d4 is genuinely stronger): 0.0%
pairs: swept 0, split 2, both drawn 0, swept against 28
as white 6.7% (30 games), as black 0.0% (30 games)
draw rate 0.0%   avg length 98.9 plies (median 91, max 199)   hit the 300-ply cap: 0
terminations: checkmate 60
champion d4: 2833 moves, 38.7 ms/move, 4749 nodes/move, 205 searches (7%) reported a depth below the target (movetime cap, forced move, or mate found)
uci:stockfish: 2861 moves, 96.6 ms/move
68.8s elapsed
```

</details>

