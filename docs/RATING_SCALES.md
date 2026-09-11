# RATING_SCALES -- turning "1262 UCI_Elo" into a number a person recognises

`py/benchmark.py` can report the champion's strength on the **chess.com**,
**FIDE** or **Lichess** rating scale instead of Stockfish's own. This document
explains what that conversion is, where every number in it comes from, and --
most importantly -- how wrong it can be.

Short version, before anything else:

> Our **measurement** is good to about +/-80 Elo.
> The **conversion between rating pools** is good to about +/-450 points, at
> best. The conversion error is roughly **five times larger** than the
> measurement error, and playing more games does nothing to shrink it.
> Any converted figure should be read as a wide range, never as a number.

---

## 1. What we actually measure, and what we do not

The harness plays the champion a fixed number of colour-balanced games against
Stockfish, with Stockfish deliberately weakened:

```
setoption name UCI_LimitStrength value true
setoption name UCI_Elo value 1320
```

From the match score it computes an Elo *difference* with the standard
logistic, `elo = -400 * log10(1/score - 1)`, and adds that difference to the
number we set `UCI_Elo` to. A 1320-rated opponent and a measured `-58` gives
`~1262`.

That number is:

* **a rating difference against one specific engine at one specific setting,
  expressed on that engine's own scale.**

It is **not**:

* a FIDE rating,
* a chess.com rating,
* a rating against humans of any kind,
* or even a rating against other *engines* in general, since the opponent is a
  strong engine that has been handicapped rather than a genuinely weak one.

Two further caveats that belong to the measurement itself, before any pool
conversion enters the picture:

1. **We do not run Stockfish at the time control it was calibrated at.** The
   calibration was done at 120s+1s (see section 2). The harness runs the
   opponent on a node or movetime budget -- `nodes=20000`, `movetime=100`, and
   so on. The `UCI_Elo` label is only meaningful at the calibration time
   control, so our anchor is already off-label.
2. **Extrapolating from a lopsided score is unreliable.** An anchor the
   champion scores 3% against implies a ~600-point extrapolation from a couple
   of wins. `docs/BENCHMARK.md` already says this; it is repeated here because
   the converted number inherits it. Prefer an anchor the champion scores near
   50% against.

## 2. What Stockfish's `UCI_Elo` scale actually is

This matters more than anything else in this document, and it is the part most
people skip.

Stockfish's own source says, in `src/search.h`:

> Skill structure is used to implement strength limit. If we have a UCI_Elo, we
> convert it to an appropriate skill level, anchored to the Stash engine. This
> method is based on a fit of the Elo results for games played between
> Stockfish at various skill levels and various versions of the Stash engine.
> Skill 0 .. 19 now covers CCRL Blitz Elo from 1320 to 3190, approximately

-- <https://github.com/official-stockfish/Stockfish/blob/master/src/search.h>

The commit that `search.h` cites for the fit gives the rest of the detail: the
games were played at **120 seconds + 1 second increment**, against versions of
the **Stash** engine that are ranked on **CCRL Blitz**, and the resulting scale
is accurate to roughly **+/-100 Elo** -- within that engine pool.

-- <https://github.com/vondele/Stockfish/commit/a08b8d4e9711c2>

So `UCI_Elo` is an **engine-versus-engine** rating, on the
[CCRL Blitz (40 moves / 4 minutes) list](https://www.computerchess.org.uk/ccrl/404/).
CCRL is a list of computer programs playing computer programs. It has no
players in it. Its numbers are meaningful *relative to other engines on that
list* and nothing else.

Two more facts, both verifiable locally with `echo uci | stockfish`:

* `option name UCI_Elo type spin default 1320 min 1320 max 3190` -- **1320 is
  the floor.** You cannot ask Stockfish to play at 1200. Anything we report
  below 1320 is our own extrapolation off the bottom of Stockfish's scale.
* The handicap works by *picking a deliberately suboptimal move* at
  `depth = 1 + Skill Level`
  ([Stockfish FAQ](https://official-stockfish.github.io/docs/stockfish-wiki/Stockfish-FAQ.html)).
  A handicapped Stockfish therefore plays a stream of strong moves punctuated
  by occasional nonsense. A human of the equivalent rating makes a different
  *kind* of mistake -- consistently imprecise rather than mostly perfect with
  sudden lapses. Two players with the same average error rate but different
  error *profiles* do not score the same against a third player, which is
  exactly why a mapping from one to the other cannot be exact.

Stockfish's own FAQ is blunt about the general problem:

> Rating Stockfish against a human scale, such as FIDE Elo, has become
> virtually impossible

## 3. Why any pool conversion is approximate

Four independent reasons, all of which apply here.

**(a) A rating only means something inside its pool.** Elo and Glicko measure
you *against the people you played*. Two pools with different populations will
happily assign very different numbers to identical playing strength. There is
no physical unit of chess strength that a rating is measuring.

**(b) chess.com's pools differ from each other.** chess.com uses Glicko, with
separate pools for Bullet, Blitz, Rapid and Daily. Its own comparison article
puts a 1000 Blitz player at 880 Bullet and 1456 Turn-Based -- a ~575-point
spread for the same person. Quoting "a chess.com rating" without naming the
pool is meaningless, which is why every converted number this tool prints says
**chess.com Rapid**.
-- <https://www.chess.com/article/view/chesscom-rating-comparisons>

**(c) The gap to FIDE is not a constant, and does not even move in one
direction.** Using the ChessGoals dataset (see section 4), the chess.com Rapid
figure sits 85 points *below* the paired FIDE figure at the bottom of their
range, 30 *above* it in the middle, 15 above, then 55 *below* again at the top.
A single additive offset would be wrong at one end or the other, which is why
the implemented map is piecewise linear rather than "add N".

**(d) The pools move.** chess.com Rapid has deflated sharply as beginners have
joined: a 1000 Rapid rating was around the 49th percentile five years ago and
is around the 81st today. FIDE reset its own scale in March 2024 -- the floor
went from 1000 to 1400 and *every* player rated 1000-2000 got a one-time
increase -- so pre-2024 chess.com-to-FIDE comparisons (including chess.com's
own 2014 article) are on a different FIDE scale than today's.
-- <https://www.chess.com/forum/view/general/til-5-years-ago-a-1000-rapid-rating-was-49th-percentile-today-its-the-81st>,
<https://qc.fide.com/2024/03/01/new-rating-regulations-come-into-force/>

### An illustration worth internalising

The Maia engines are neural networks trained to *imitate* Lichess players at a
specific rating -- maia1 on games between 1100-rated players, maia5 on 1500s,
maia9 on 1900s. They then play the Lichess pool themselves. Their live Lichess
Rapid ratings, read from the Lichess API on 2026-09-11:

| bot | trained to imitate | actual Lichess Rapid | games |
| --- | ---: | ---: | ---: |
| maia1 | 1100 | **1466** | 616,371 |
| maia5 | 1500 | **1682** | 227,155 |
| maia9 | 1900 | **1756** | 212,478 |

An 800-point spread in intended strength came out as a 290-point spread in
earned rating, and every one of them is rated well above what it was built to
imitate. These are *human-imitating* engines with hundreds of thousands of
games each, playing the very pool they were trained on -- the friendliest
possible conversion problem -- and the numbers still do not line up. Our
problem (a handicapped alpha-beta engine, measured on an engine-versus-engine
list, converted to a human Glicko pool it has never played in) is far harder.

-- <https://lichess.org/@/maia1>, <https://lichess.org/@/maia5>,
<https://lichess.org/@/maia9>,
<https://lichess.org/@/lichess/blog/introducing-maia-a-human-like-neural-network-chess-engine/X9PUixUA>

## 4. The conversion this tool implements

The map lives in **one table at the top of `convert_rating()` in
`py/benchmark.py`**. It is deliberately in one place so that a reader can see
the assumptions and change them. It is built in two steps.

### Step 1 -- `UCI_Elo` to a FIDE-equivalent strength. **This is an assumption.**

Write the assumption as `fide_equivalent = UCI_Elo + OFFSET`. There is no
published study that fixes `OFFSET`. What exists are two arguments that point
in *opposite* directions:

* **CCRL's low-end numbers understate human-equivalent strength**, which would
  make `OFFSET` **positive**. The best-known illustration on TalkChess is an
  engine rated 1712 on CCRL 40/4 that is judged to play "not that far from
  FIDE 2000" -- implying `OFFSET` of roughly **+300**.
* **A handicapped strong engine is weaker than its label against people**,
  which would make `OFFSET` **negative**. Play reports of engines at their
  lowest settings consistently put them below what the label claims, and
  community estimates of the equivalent ratings of Lichess's own Stockfish
  levels vary by 400 points between posts -- implying `OFFSET` of roughly
  **-400**.

Neither is a measurement. Together they bracket `OFFSET` somewhere in the
region of **-400 to +300**. **The table uses `OFFSET = 0`** -- near the
midpoint of that bracket. That is a placeholder chosen for being unbiased
between two guesses, not a finding, and the conversion error bar in section 5
is sized to cover the width of the whole bracket.

-- <https://talkchess.com/viewtopic.php?t=83285>,
<https://talkchess.com/viewtopic.php?t=59332>,
<https://lichess.org/forum/general-chess-discussion/what-elo-are-the-various-stockfish-levels>
(all community discussion, not data)

### Step 2 -- FIDE to chess.com Rapid and Lichess Rapid. **This is sourced.**

From the ChessGoals rating comparison, updated July 2026, built from roughly
20,000 active player profiles (9,300+ chess.com Bullet, 10,100+ chess.com
Rapid, 1,760 active USCF and 1,923 active FIDE players), with Lichess
comparisons restricted to established ratings (RD under 150).

-- <https://chessgoals.com/rating-comparison/>

Their published rows, keyed on chess.com Blitz:

| cc Blitz | cc Rapid | FIDE | Lichess Rapid |
| ---: | ---: | ---: | ---: |
| 500 | 815 | -- | 1290 |
| 1000 | 1255 | -- | 1615 |
| 1500 | 1655 | 1740 | 1905 |
| 2000 | 1995 | 1965 | 2165 |
| 2500 | 2260 | 2245 | 2400 |
| 3000 | 2430 | 2485 | 2615 |

Note the empty FIDE cells. The authors state plainly that there is **"no
accurate data for players under about 1550 FIDE"**. That is directly relevant
to us: the strength we are actually measuring sits at or below the bottom of
their FIDE data.

### The resulting anchor table

Composing the two steps gives the table that is in the code. Columns are
`UCI_Elo`, chess.com Rapid, FIDE, Lichess Rapid:

| UCI_Elo | chess.com Rapid | FIDE | Lichess Rapid | provenance |
| ---: | ---: | ---: | ---: | --- |
| 1320 | 1020 | 1320 | 1440 | extrapolated |
| 1500 | 1290 | 1500 | 1640 | extrapolated |
| 1740 | 1655 | 1740 | 1905 | sourced |
| 1965 | 1995 | 1965 | 2165 | sourced |
| 2245 | 2260 | 2245 | 2400 | sourced |
| 2485 | 2430 | 2485 | 2615 | sourced |

Values between anchors are linearly interpolated. The map is monotonic over
its whole range (verified in section 7).

* The bottom row is **1320 because that is Stockfish's floor for `UCI_Elo`**;
  below it Stockfish itself has not calibrated anything.
* The two "extrapolated" rows lie below ChessGoals' lowest FIDE data point.
  They continue the slope of the lowest sourced segment, which is a stated
  rule, not evidence. They carry extra uncertainty in the code for that reason.
* The top row is **2485 because that is where ChessGoals' data ends.**

**Outside 1320-2485 the tool refuses to convert.** It reports the nearest
anchor, says which side the measurement fell on, and prints a warning. It does
not silently extrapolate. For the run that produced ~1262, that refusal *is*
the answer: 1262 is below Stockfish's own floor, and the only honest statement
is "somewhere below the bottom anchor".

## 5. The error bar

Two terms, combined in quadrature:

| term | size | shrinks with more games? |
| --- | ---: | --- |
| match measurement (95% CI) | ~+/-80 Elo | yes |
| pool conversion -> chess.com Rapid | ~+/-450 pts | **no** |
| pool conversion -> FIDE | ~+/-350 pts | **no** |
| pool conversion -> Lichess Rapid | ~+/-400 pts | **no** |
| extra, on the extrapolated rows | +/-150 pts added in quadrature | **no** |

The conversion figures are not published intervals -- **no such interval
exists**. They are the section-4 bracket on the step-1 assumption (-400 to
+300, so about +/-350 around the midpoint) carried through the local slope of
the map, which is roughly 1.5 at the bottom and roughly 1.0 higher up.

**Combined: about +/-460 points on chess.com Rapid**, i.e. a converted result
is a band roughly 900 points wide. Stated the way it matters:

> The conversion error is about 5x the measurement error. A converted number
> quoted to four significant figures would be false precision by a factor of
> about a hundred. This is why the tool rounds to the nearest 50 and prints a
> range.

If you want a tighter number, the thing to improve is **not** the number of
games. It is the anchor: measure against an opponent whose rating is known in
the *target* pool. A bot with a real, established chess.com Rapid Glicko
rating, played over enough games, would collapse the conversion term entirely,
because there would be no conversion left to do.

## 6. Percentiles, which are usually more useful than the number

**Lichess Rapid** percentiles in the tool are computed from the histogram
Lichess itself publishes -- 477,455 active rapid players, read 2026-09-11.
This is real primary data:

| Lichess Rapid | stronger than |
| ---: | ---: |
| 1000 | 7% |
| 1200 | 18% |
| 1400 | 34% |
| 1600 | 52% |
| 1800 | 70% |
| 2000 | 85% |
| 2200 | 95% |

-- <https://lichess.org/stat/rating/distribution/rapid>

**chess.com Rapid** percentiles are a **community estimate**. chess.com does
not publish a rating distribution. The table in the code (1000 -> ~80th, 1200
-> ~90th, 1500 -> ~96th, 2000 -> ~99th) comes from a community blog hosted on
chess.com, not from chess.com itself, and as section 3(d) notes it drifts by
tens of percentile points over a few years.
-- <https://www.chess.com/blog/SolarGurke/what-percentage-of-chess-com-players-are-at-each-rating>

Also note what the two percentiles are measuring, because the denominators are
not the same. The chess.com figure is over *all rated accounts*, which includes
every account that played two games and quit -- community reports put the
median chess.com Rapid rating somewhere in the 600s for that reason, so
"stronger than 80% of rated chess.com Rapid players" is a weaker claim than it
sounds. The Lichess figure is computed over players *active that week*, a much
more demanding denominator. The same player will score a higher percentile on
chess.com than on Lichess partly for this reason alone.

**FIDE** publishes no rating distribution, so no percentile is offered for
`--scale fide`.

## 7. What could not be sourced, and what was rejected

Stated plainly, because the gaps matter more than the fills:

* **There is no published measurement of Stockfish's `UCI_Elo` against any
  human rating pool.** This is the single weakest link in the chain and it is
  entirely an assumption. Searching for controlled experiments pitting
  `UCI_LimitStrength` settings against rated human play turned up nothing but
  forum opinion.
* **chess.com publishes no rating distribution and no official conversion
  table.** Its own comparison article is from 2014, is based on 265 survey
  responses, is anchored on Blitz, and predates the 2024 FIDE rescaling. It
  also produces obviously self-selected results -- a 1000 Blitz player mapping
  to FIDE 1338 is a survey artifact (only people who *have* a FIDE rating
  answered), not a conversion. It was not used.
* **chess.com's bot ratings were rejected as an anchor.** They looked like a
  promising sanity check, but chess.com does not document how they are
  assigned, they are not derived from rated play, forum reports say they are
  stale, and community measurement finds them increasingly overrated at higher
  levels (a 2200-rated bot judged to play around 2000).
  -- <https://www.chess.com/blog/AdviceCabinet/are-chess-com-bots-ratings-accurate>
* **No data exists below ~1550 FIDE / ~1250 chess.com Rapid.** This is exactly
  where our champion sits.

Given all that, a defensible answer to "what is this in chess.com terms?" at
the strength we are currently measuring is: **we cannot say, and the tool
declines to guess.** What it will give you instead is the nearest anchor and
the direction, which is honest and still useful.

## 8. Using it

```sh
# default: unchanged, everything on Stockfish's UCI_Elo scale
python3 py/benchmark.py --ladder --opponent 'uci:/opt/homebrew/bin/stockfish'

# also report on chess.com Rapid (adds a section; the UCI_Elo report is intact)
python3 py/benchmark.py --ladder --opponent 'uci:/opt/homebrew/bin/stockfish' \
    --scale chesscom

# a single anchored match, which is the cleanest measurement
python3 py/benchmark.py --opponent 'uci:/opt/homebrew/bin/stockfish' \
    --opp-option UCI_LimitStrength=true --opp-option UCI_Elo=1320 \
    --games 200 --scale chesscom

# the Stockfish sweep
python3 py/sf_ladder.py --groups elo --scale lichess
```

`--scale uci` is the default and is the identity: it produces exactly the
output the tool produced before this feature existed. A non-`uci` scale only
ever *adds* a section; it never replaces or rewrites the measured figures.

The converted section always prints the measured `UCI_Elo` figure with its 95%
CI directly above the converted band, so the two can never be confused. There
is no code path that prints a converted number on its own.

Against `random` or `material` there is no absolute rating to convert -- those
opponents have no rating -- and the tool says so rather than inventing one.

### Changing the assumptions

Edit `ANCHORS` and `CONV95` at the top of `convert_rating()` in
`py/benchmark.py`. The table is the whole model; nothing else needs touching.
If you find a real measurement of handicapped Stockfish against a human pool,
that is the number to put in step 1, and the conversion error bar should shrink
accordingly.

### How the implementation was checked

* `--scale uci` output diffed byte-for-byte against the pre-change output
  (identical apart from the wall-clock line).
* All six anchor rows hand-checked against the table above.
* Monotonicity verified for all three target scales over 1000-2900.
* Interpolation checked at the midpoint of a segment.
* Out-of-range inputs on both sides verified to refuse and report the nearest
  bound.
* `python3 -m py_compile` on `py/benchmark.py` and `py/sf_ladder.py`.

## 9. Sources

Primary:

* Stockfish `src/search.h`, the `Skill` struct and its calibration comment --
  <https://github.com/official-stockfish/Stockfish/blob/master/src/search.h>
* The calibration commit it references (Stash engine, 120s+1s, CCRL Blitz,
  ~+/-100 Elo) -- <https://github.com/vondele/Stockfish/commit/a08b8d4e9711c2>
* Stockfish FAQ, how Skill Level and UCI_Elo work --
  <https://official-stockfish.github.io/docs/stockfish-wiki/Stockfish-FAQ.html>
* Stockfish UCI option list -- `echo uci | stockfish`, verified locally on
  Stockfish 19: `UCI_Elo type spin default 1320 min 1320 max 3190`
* CCRL Blitz (40/4) rating list --
  <https://www.computerchess.org.uk/ccrl/404/>
* Lichess rapid rating distribution (official, 477,455 active players) --
  <https://lichess.org/stat/rating/distribution/rapid>
* Lichess API, live Maia bot ratings --
  <https://lichess.org/@/maia1>, <https://lichess.org/@/maia5>,
  <https://lichess.org/@/maia9>
* FIDE Qualification Commission, March 2024 rating regulations (floor raised
  to 1400; one-time increase for 1000-2000) --
  <https://qc.fide.com/2024/03/01/new-rating-regulations-come-into-force/>,
  <https://handbook.fide.com/chapter/B022024>
* chess.com's own rating comparison article (2014, 265 survey responses,
  Blitz-anchored; **not used**, listed so the decision is auditable) --
  <https://www.chess.com/article/view/chesscom-rating-comparisons>

Third-party dataset, transparent about method, **used for step 2**:

* ChessGoals rating comparison, updated July 2026, ~20,000 profiles --
  <https://chessgoals.com/rating-comparison/>

Community estimate, **explicitly labelled as such wherever it is used**:

* chess.com Rapid percentiles (community blog hosted on chess.com, Aug 2026) --
  <https://www.chess.com/blog/SolarGurke/what-percentage-of-chess-com-players-are-at-each-rating>
* chess.com Rapid percentile drift over five years --
  <https://www.chess.com/forum/view/general/til-5-years-ago-a-1000-rapid-rating-was-49th-percentile-today-its-the-81st>
* chess.com bot rating accuracy (**rejected as an anchor**) --
  <https://www.chess.com/blog/AdviceCabinet/are-chess-com-bots-ratings-accurate>

Forum discussion only, no data, used solely to bracket the step-1 assumption:

* CCRL versus FIDE human ratings --
  <https://talkchess.com/viewtopic.php?t=83285>,
  <https://talkchess.com/viewtopic.php?t=59332>,
  <https://talkchess.com/viewtopic.php?t=71053>
* Estimated ratings of Lichess's Stockfish levels (estimates vary by ~400
  points between posts) --
  <https://lichess.org/forum/general-chess-discussion/what-elo-are-the-various-stockfish-levels>,
  <https://lichess.org/forum/general-chess-discussion/equivalence-between-glicko-2-ratings-and-the-stockfish-levels>
* Lichess versus chess.com rating gap (corroborates ChessGoals' direction and
  rough size, but is not itself data) --
  <https://medium.com/elo-chess/how-do-lichess-ratings-compare-with-chess-com-ratings-3e091dd0ff23>

Data read on 2026-09-11.
