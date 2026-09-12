# Architecture plan: is a convolutional network affordable here?

**Short answer: no, at any size measured, and not by a small margin.** The
cheapest convolutional network in the brief — 2 residual blocks of 16 filters,
a seventh the parameters of our current MLP — costs **6x** per evaluation and
would leave us **27 simulations per move** at the same wall-clock. Our own
measurements say the search stops finding plans below about 96. The largest,
10 blocks of 64 filters, costs **80x** and leaves **2 simulations per move**.

The cheaper middle ground *is* affordable and is the recommendation: adding
**attack maps, a repetition count and the opponent's last move as extra sparse
features** costs 157 -> 143 simulations per move, which is above the floor with
room to spare, and supplies exactly the spatial fact that a small convolutional
network would be worst at computing for itself.

Everything below is a number produced by `tools/conv_proto.c`, with the command
that produced it. Nothing in `src/` was modified; this is a costed feasibility
study, not an integration.

---

## 1. Reproducing this

`tools/conv_proto.c` is deliberately **not** in the Makefile, so `make -j8` and
`make test` are untouched by it (both verified warning-free and passing at the
end of this work). Build it by hand:

```sh
# the fast build, the one every number below comes from
cc -O3 -std=c11 -D_DARWIN_C_SOURCE -Wall -Wextra -Wno-unused-parameter \
   -funroll-loops -fno-math-errno -ffp-contract=fast -mcpu=native -Isrc \
   -DUSE_ACCELERATE -DACCELERATE_NEW_LAPACK \
   tools/conv_proto.c src/chess.c src/net.c -o build/conv_proto \
   -lm -lpthread -framework Accelerate

# the portable build (no Accelerate), for the AMX comparison in section 5
cc -O3 -std=c11 -D_DARWIN_C_SOURCE -Wall -Wextra -Wno-unused-parameter \
   -funroll-loops -fno-math-errno -ffp-contract=fast -mcpu=native -Isrc \
   tools/conv_proto.c src/chess.c src/net.c -o build/conv_proto_neon -lm -lpthread

build/conv_proto params | check | fwd | bwd | feat | sims | all
```

Machine: Apple M3, 4 performance + 4 efficiency cores, 8 GB.

---

## 2. What was built, and how it was checked

A standalone residual convolutional network over 8x8 planes, forward **and**
backward:

```
x0                      20 x 8 x 8    input planes
a  = conv3x3(x0) + b                  C x 64
h  = relu(norm(a))                    C x 64        <- block input
repeat N times:
    t1 = conv3x3(h)  + b1 ;  u1 = relu(norm(t1))
    t2 = conv3x3(u1) + b2 ;  h  = relu(norm(t2) + h)      <- residual
policy: pp = conv1x1(h, C->4) ; q = Wq . relu(norm(pp))   -> 32-d query
value:  vp = conv1x1(h, C->1) ; v = tanh(Wv2 . relu(Wv1 . relu(norm(vp))))
```

Convolutions are **im2col + sgemm** (Accelerate's `cblas_sgemm`, with a
hand-written NEON kernel underneath it for the portable build). The brief warns
that a naive 7-loop convolution would mislead; it would have, by two orders of
magnitude:

```
one 3x3 conv, 32->32 filters, one position:
    naive 7-loop      711,459 ns
    im2col + sgemm      6,098 ns      117x faster
```

Three independent correctness gates, all passing (`build/conv_proto check`):

| gate | result |
|---|---|
| im2col+sgemm vs the naive 7-loop convolution | max abs diff **0** (bit-identical) |
| folded inference trunk vs a naive reference trunk | max abs diff **9.2e-07** on values up to 2.35 |
| backward pass vs central finite differences, 13 tensors x 6 samples | worst relative error **7.2e-03** |

The gradient check caught two real defects that would otherwise have been
reported as measurements: the two head LayerNorms were writing their reciprocal
standard deviations into a buffer the value head later overwrote, and the saved
im2col buffer was strided at `9*C` per layer when the stem needs
`9*20` — a heap overflow at C = 16, and silent corruption of the stem's saved
columns. Both are fixed. **A benchmark of an incorrect network is not a
measurement**, which is why the check runs before any timing.

### Four ways this study was deliberately made favourable to convolutions

The conclusion is negative, so the study was biased *toward* the thing being
rejected wherever there was a choice:

1. **BatchNorm, folded.** AlphaZero used BatchNorm, which at inference collapses
   into the preceding convolution and costs nothing. The headline conv numbers
   use that free form. LayerNorm — which this codebase otherwise prefers, and
   which does not fold — is measured separately and costs the conv net a further
   **2.2x to 2.8x**. Charging it would have made the verdict worse.
2. **The cheap policy head.** AlphaZero's 8x8x73 = 4672-way move plane needs a
   128 x 4672 fully-connected layer: 598,016 MACs, **22x the arithmetic of our
   entire current network on its own**, and measured at 2.05x slower than the
   32-d factored query head this codebase already has. The conv net is given
   the cheap head.
3. **The conv net's best batch.** Each size was swept over batch 1/4/8/16/32/64
   and measured at whichever was fastest; the MLP is measured at batch 1, the
   way `nn_eval` is actually called from `src/mcts.c`.
4. **A conservative MLP baseline.** `tools/profile.c` reports 613,169 evals/sec
   for `nn_eval`, but it compiles `src/net.c` into one translation unit with the
   benchmark so the call inlines. `conv_proto` links `net.o` separately, exactly
   as `src/mcts.c` does, and measures ~255,000–300,000 evals/sec of thread CPU
   time. That is the rate the real search sees, and it is the **smaller**
   number, so every slowdown quoted below is the smallest defensible one.

### Measurement conditions, stated plainly

This laptop was **not quiet** while these numbers were taken — load averages of
21 to 34 on 8 cores, from unrelated jobs. That is fatal to naive benchmarking:
the same `nn_eval` loop timed on the wall clock gave 613k, 453k and 240k
evals/sec in three runs an hour apart. Two things fix it:

- **Every rate is evals per second of `CLOCK_THREAD_CPUTIME_ID`**, not wall
  clock, so a descheduled benchmark reports the right rate rather than a slow
  one. Multi-thread figures are the sum over threads.
- **Every ratio the conclusion rests on is measured by alternating the two
  candidates**, six repetitions each sized to the same ~4 GMAC of work, taking
  the best repetition of each. Both candidates then see the same weather.

Residual noise is core type (a P-core is ~1.5x an E-core) and frequency. The
stability achieved is shown by three independent repeats of the decisive table
in section 6: the slowdowns came out 6/14/47/80, 6/14/45/88 and 6/14/44/72.
Absolute rates wander; the ratios do not.

---

## 3. The arithmetic

`build/conv_proto params`

| network | parameters | MACs/eval | x MLP params | x MLP MACs |
|---|---:|---:|---:|---:|
| **MLP (current)** | **157,473** | **27,136** | 1.00 | 1.00 |
| conv 2 x 16 | 22,768 | 789,536 | 0.14 | **29x** |
| conv 4 x 32 | 90,864 | 5,107,744 | 0.58 | **188x** |
| conv 6 x 64 | 467,056 | 29,079,584 | 2.97 | **1,072x** |
| conv 10 x 64 | 763,504 | 47,953,952 | 4.85 | **1,767x** |
| *AlphaZero (19 x 256)* | *~40M* | *~1.43e9* | *~254* | *~52,900* |

**This table is the whole story in one line: convolutions are
parameter-cheap and compute-expensive; our sparse MLP is parameter-expensive
and compute-cheap.** A 2 x 16 conv net has a *seventh* of our parameters and
does *29 times* the arithmetic, because every weight is reused on all 64
squares. On a GPU that trade is free — it is the reason AlphaZero is shaped
that way. On 8 CPU cores with no GPU, arithmetic is the scarce resource and
parameters are nearly free, so the trade runs exactly backwards.

Our MLP's first layer is the extreme case: 100,864 weights that cost 35 row
gathers, because only ~35 of 788 inputs are ever 1.

---

## 4. Forward throughput

`build/conv_proto fwd` (Accelerate build; evals/sec of thread CPU time; MLP and
conv alternated, best of 6)

| conv | best batch | MLP 1 thr | conv 1 thr | **slowdown** | MLP 8 thr | conv 8 thr | slowdown 8 thr | conv GFLOP/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 x 16 | 16 | 255,389 | 44,303 | **6x** | 2,097,045 | 322,075 | **7x** | 70.0 |
| 4 x 32 | 32 | 253,908 | 16,100 | **16x** | 2,089,648 | 94,726 | **22x** | 164.5 |
| 6 x 64 | 8 | 290,112 | 5,317 | **55x** | 1,973,677 | 24,558 | **80x** | 309.2 |
| 10 x 64 | 16 | 299,343 | 3,366 | **89x** | 1,993,535 | 14,610 | **136x** | 322.8 |

Two things in that table are worth dwelling on.

**The conv net uses the hardware far better and still loses.** Our MLP achieves
about **14 GFLOP/s** (27,136 MACs x 2 x 255,389 evals/s) — it is a sparse
gather and a 128x128 matrix-*vector* product, neither of which can fill a wide
machine. The conv net reaches **70 to 323 GFLOP/s**, five to twenty-three times
better utilisation. It loses anyway, by 6x to 89x, because it is doing 29x to
1767x more work. Better kernels cannot close a gap of that shape.

**The conv net scales worse across cores, not better.** The MLP goes from
255,389 to 2,097,045 evals/sec on 8 threads — **8.2x, essentially perfect**,
because its NEON kernels are per-core and share nothing. The conv net manages
7.3x, 5.9x, 4.6x and 4.3x, so its 8-thread slowdown (7x, 22x, 80x, 136x) is
*worse* than its 1-thread slowdown at every size. Accelerate's `sgemm` runs on
the AMX coprocessor, which is **shared per core cluster**. Self-play is eight
independent threads of games; that is precisely the workload AMX sharing
punishes.

### Where a conv evaluation goes, and the implementation-independent floor

`build/conv_proto fwd`, ns per evaluation, 1 thread, each size at its best batch:

| conv | im2col | **sgemm** | bias+relu | heads | total | (MLP total = 3,385 ns) |
|---|---:|---:|---:|---:|---:|---|
| 2 x 16 | 7,683 | **4,782** | 929 | 9,419 | 22,813 | sgemm alone = 1.4x the whole MLP |
| 4 x 32 | 27,798 | **27,075** | 3,735 | 9,840 | 68,448 | sgemm alone = 8.0x |
| 6 x 64 | 74,351 | **131,554** | 10,261 | 10,972 | 227,137 | sgemm alone = 38.9x |
| 10 x 64 | 130,058 | **263,490** | 17,711 | 10,699 | 421,959 | sgemm alone = 77.8x |

The `sgemm` column is the irreducible arithmetic — Accelerate's hand-tuned AMX
kernels, already at 70–323 GFLOP/s. **Even if im2col, the pointwise work and the
heads all cost zero**, 4 x 32 and above remain 8x to 78x slower than our entire
current evaluation. No amount of engineering on my part changes the verdict for
those sizes; only 2 x 16 is close enough that implementation quality matters,
and 2 x 16 fails on a different ground (section 8).

### Without Accelerate

The project's hard rules allow Accelerate *with a scalar fallback*. Here is what
the fallback costs (`build/conv_proto_neon fwd`, hand-written NEON sgemm):

| conv | conv 1 thr | slowdown vs MLP | GFLOP/s |
|---|---:|---:|---:|
| 2 x 16 | 11,162 | **25x** | 17.6 |
| 4 x 32 | 2,056 | **135x** | 21.0 |
| 6 x 64 | 344 | **834x** | 20.0 |
| 10 x 64 | 195 | **1,419x** | 18.7 |

Accelerate's AMX kernels are doing **4.0x, 7.8x, 15.5x and 17.3x** of the work
at the four sizes. Strip them out and the convolutional network is **25x to
1,419x** slower than our MLP — which needs nothing but NEON and is already
within 2x of its own ceiling. A convolutional engine would be entirely
dependent on one Apple system framework and on one coprocessor shared per core
cluster. That is a portability argument on top of the speed one.

---

## 5. The learning cost

`build/conv_proto bwd` — the backward pass is implemented and finite-difference
checked, so this is measured rather than estimated. Comparing like with like
(LayerNorm forward against LayerNorm forward+backward, since a backward pass
needs a normaliser that is actually present):

| conv | fwd folded | fwd LN | fwd+bwd LN | bwd/fwd |
|---|---:|---:|---:|---:|
| 2 x 16 | 53,289 | 25,391 | 7,122 | 3.6x |
| 4 x 32 | 21,575 | 8,401 | 2,003 | 4.2x |
| 6 x 64 | 5,931 | 2,576 | 611 | 4.2x |
| 10 x 64 | 3,808 | 1,679 | 356 | 4.7x |

About 4x at every size — a little above the textbook 2–3x because `col2im` is a
scalar scatter while everything else is `sgemm`. Nothing pathological: **the
conv net's learning cost is its forward cost multiplied again**, and the forward
cost is already the problem.

One asymmetry the table does not show. A training step also pays the optimiser
update over every parameter. The MLP's dominant `W0` (788 x 128) is a *sparse*
layer — only ~35 rows are touched per position — so it is cheap to train as well
as cheap to evaluate. Every convolutional weight is dense and is updated in
full, on every minibatch.

---

## 6. The arithmetic that decides it

`build/conv_proto sims`

Self-play runs **160 MCTS simulations per move** (confirmed from
`runs/az_hour/telemetry.jsonl`: `sims_per_move = 160`, 1900 generations, 384
games each = 729,600 games in 69 minutes). Throughput falls roughly linearly in
the simulation count, so a network K times slower per evaluation buys 160/K
simulations at the same wall-clock — the same games/second, the same
generations/hour.

The floor is not a guess. `docs/ALGORITHM.md` measured it by sweeping the
simulation count at our draw penalty (-0.6):

| sims | repetition draws | captures/game |
|---:|---:|---:|
| 48 | 51.6% | 8.5 |
| 96 | 17.7% | 14.4 |
| 160 | 5.3% | 17.9 |

Tripling the draw *penalty* at 48 sims barely moved repetition (0.599 -> 0.516);
tripling the *simulations* nearly eliminated it. **Shuffling is the search
failing to find a plan, and no network fixes that.** The useful floor is about
96.

| conv | slowdown (median of 3) | **simulations/move affordable** | verdict |
|---|---:|---:|---|
| 2 x 16 | 6x | **27** | below floor — below even the 48-sim shuffling configuration |
| 4 x 32 | 14x | **11** | far below floor |
| 6 x 64 | 45x | **3.6** | far below floor |
| 10 x 64 | 80x | **2.0** | far below floor |

**Every size fails, and the smallest fails by a factor of nearly four.** A 2 x 16
conv net at 27 simulations per move would be searching *less* deeply than the
48-simulation configuration that produced a 60%-repetition shuffling population
— and a 2-block, 16-filter network is not a plausible candidate to be so much
better per evaluation that it overcomes that.

Read the other way: to keep 160 simulations *and* pay for a convolutional net,
the 69-minute run becomes

| conv | same 729,600 games at 160 sims |
|---|---:|
| 2 x 16 | **6.9 hours** |
| 4 x 32 | **16.1 hours** |
| 6 x 64 | **51.8 hours** |
| 10 x 64 | **92 hours** |

against a budget of roughly an hour.

---

## 7. The cheap middle ground — and where the contract line falls

### The test I applied

`docs/FROM_SCRATCH.md` permits the rules of chess and forbids every form of
chess evaluation. The operational question for a candidate feature:

> Could a player who knows **only the rules**, and holds no opinion whatever
> about which positions are good, compute this number?

If yes, it is a projection of the rules; the network still has to learn from
game results alone whether it matters and in which direction. If computing it
requires **ranking** pieces, squares or moves, it carries an opinion and it is
forbidden however useful it would be.

**Allowed.** *Attack maps* — "is square s attacked by side c" is a function of
the move rules and the occupancy and nothing else; a uniform random mover could
compute it, and `src/chess.c` already exports `attacks_rook`, `attacks_bishop`
and the rest for exactly this. *Per-piece-type mobility counts* — a count of
legal moves. The claim "mobility is good" is an opinion; the count is not, and
the network is free to learn that it means nothing. *Repetition count and the
opponent's last move* — facts about the game so far.

**Forbidden, and dropped.** *"Squares attacked by a lower-valued piece"* —
"lower-valued" requires a piece-value ordering, which is the **first item** on
the forbidden list in `docs/FROM_SCRATCH.md`. It is dropped even though it is
probably the single most useful feature on the list, and that is the point: the
line is drawn by the rule, not by the payoff. *Static exchange evaluation*,
*"hanging piece"*, *"defended by a pawn"* — same reason, every one of them ranks
pieces.

**The borderline case, argued explicitly.** *"Attacked by them AND not attacked
by us"* needs no piece values at all — it is set arithmetic on two allowed
bitboards — so it passes the contract test. It is still **not included**, on a
different ground: it is the conclusion we hope the network reaches,
hand-computed. Given both attack maps, the first layer can form that
conjunction itself in one weight. Supplying it adds no information and only
narrows what the network is permitted to notice. **Supply the facts, not the
inference.** That distinction — between a rule-derived fact and a hand-authored
inference over facts — is worth keeping even where the contract does not force
it, because it is the difference between giving the network better eyes and
giving it our opinions.

### What each bundle costs

`build/conv_proto feat`. The MLP's first layer is a gather of the active rows
out of the input table, so both the **active count** and the **table size**
matter — a bigger `W0` is a colder `W0`. Both are measured directly, at the
true enlarged table size. Median of three runs:

| bundle | slots | active | extract | evals/sec | **sims/move** |
|---|---:|---:|---:|---:|---:|
| **A** baseline (today) | 788 | 35 | — | 235,660 | **157** |
| **B** A + repetition + last move | 920 | 38 | 22 ns | 227,697 | **153** |
| **C** B + *their* attack map | 984 | 70 | 109 ns | 187,393 | **126** |
| **D** B + *both* attack maps | 1,048 | 102 | 198 ns | 160,999 | **108** |
| **E** D + mobility counts | 1,096 | 108 | 198 ns | 160,972 | **109** |
| **F** E + 8 history positions | 7,240 | 315 | 1,127 ns | 68,594 | **47** |
| **C'** = C with attacks in a 32-wide side accumulator | 984 | 70 | 109 ns | 209,450 | **143** |
| **D'** = D with attacks in a 32-wide side accumulator | 1,048 | 102 | 198 ns | 198,760 | **132** |

Every cell is the median of three independent runs; the three agreed to within
3% on `sims/move` at every bundle (A came out 156.8 / 157.8 / 157.6, D' 132.3 /
131.9 / 131.6).

Extraction costs on their own: both attack maps **156 ns/position**; mobility
counts 185 ns but **~0 marginal inside MCTS**, because the search has already
generated the legal move list at every node it expands; last move + repetition
**15 ns**; eight history positions 783 ns.

**C' and D' are the engineering point.** The gather is linear in the
accumulator *width* as well as the row count, so attack features need not be
paid for at 128 wide. `src/net.h` already contains the pattern — the value head
owns its own 32-wide sparse trunk `W0v` for unrelated reasons. A third
accumulator, 32 wide, fed by the attack maps, recovers most of the cost:
**108 -> 132** simulations for both maps, **126 -> 143** for one.

### History planes (item 5), honestly

Eight previous positions as sparse features is **not affordable**: 207 extra
active features and a 7,240-row input table drop us to **47 simulations per
move**, below the floor, for a benefit I cannot demonstrate. AlphaZero needed
history partly because its convolutional input had no other way to express
repetition, and partly to see the opponent's last move.

We can have both of those things for almost nothing. The **repetition count and
the opponent's last move cost 3 active features, 15 ns, and 4 simulations per
move** (bundle B). That is where essentially all of the realisable benefit is,
and it addresses a defect we can name precisely: `src/mcts.c` keeps a repetition
window and adjudicates threefold repetition as terminal, but **the network
cannot see it**. The value head is being asked to score positions whose true
value depends on history that is not in its input. The value head's monotonic
degradation across the run (MSE 0.485 -> 0.655 against the variance baseline) is
*consistent* with that, though it is not proof of it.

---

## 8. An architectural argument, clearly labelled as an argument

The measurements above are the case. This section is reasoning, not data, and
should be weighted accordingly — but it explains *why* the cheap option is not
merely cheaper but may be strictly better here.

The brief's mechanistic story is that convolutions share weights across the
board, so "this piece is attacked" is learned once instead of once per
piece-square combination. That is right, and it is the correct reason AlphaZero
is convolutional. But consider what a 3x3 convolution can actually compute.

**A 3x3 kernel propagates information exactly one square per layer.** A rook on
a1 attacking h1 is a fact about squares 7 apart, and establishing it requires
resolving "no piece strictly between" — a 7-term conjunction along the ray.
Counting layers (stem + 2 convolutions per block):

| conv | conv layers | max propagation distance |
|---|---:|---:|
| 2 x 16 | 5 | **5 squares — cannot see a full-file rook attack at all** |
| 4 x 32 | 9 | 9 squares |
| 6 x 64 | 13 | 13 |
| AlphaZero 19 x 256 | 39 | 39 |

The two sizes we could come closest to affording are the two that are *worst*
at exactly the pattern chess needs most. A 2 x 16 network physically cannot
represent a rook's attack along an open file — the information cannot travel
that far. A 4 x 32 network can reach, but must compute occlusion conjunctions
for 8 ray directions, 2 colours and 64 origins out of 32 shared channels.
AlphaZero had 39 layers and 256 channels and 44 million games to learn it in.

So the single most valuable thing the convolution would buy — sliding-piece
attack patterns — is the thing the affordable convolutions are least able to
learn. And that exact fact is available **exactly, from the rules, for 156 ns**,
via `attacks_rook` and `attacks_bishop` in `src/chess.c`. Convolutions would
spend their capacity approximating a function the move generator already
computes perfectly.

Where a convolution genuinely would help is the *local* patterns — knight,
pawn and king attacks are fixed small offsets, exactly what a 3x3 kernel shares
well. Those are in the attack map too.

---

## 9. Ranked recommendation

Given 8 CPU cores, no GPU, and roughly an hour of training. Each item lists its
measured cost in simulations-per-move, since that is the currency that matters.

**1. Repetition count + the opponent's last move.** 132 extra input slots, 3
extra active features, 15 ns extraction. **Cost: 157 -> 153 sims/move (3%).**
Do this first because it is nearly free, because it closes a defect we can name
exactly (the search adjudicates repetitions; the network cannot see them), and
because it is the cheapest possible probe of whether the value head's
degradation is an information problem. If it does nothing, we have lost 3% of
throughput and learned something.

**2. The opponent's attack map, into a separate 32-wide accumulator.** 64 extra
slots, ~32 extra active features, ~93 ns extraction (78 ns for the map, 15
for the repetition features it rides with). **Cost: 153 -> 143
sims/move**, still 47 clear of the floor. This is the direct mechanistic
response to the 100% -> 5% cliff: "my piece stands on a square they attack"
becomes one feature rather than a piece-square conjunction the MLP must learn
768 times over. Start one-sided; it is the half that carries the danger signal,
and it is half the cost.

**3. Our own attack map as well.** **Marginal cost: 143 -> 132 sims/move.** Only
if (2) pays. Together the two maps let the first layer form
"attacked-and-undefended" itself, which is the conjunction we actually want and
which section 7 argues we should not hand-write.

**4. Per-piece-type mobility counts.** 48 slots, 6 active, **~0 marginal
extraction cost** inside MCTS because `gen_legal` has already run at every
expanded node. **Cost: ~0 sims/move.** The weakest prior of the four, but it is
so nearly free that it should ride along with (3) rather than be tested alone.

Ordering rationale: strictly ascending in cost, and each step is independently
measurable at fixed wall-clock. Nothing here requires a new training regime, a
new optimiser, or a change to `src/mcts.c` — only `nn_features`, `NF_INPUT`, and
one new accumulator in the trunk.

---

## 10. What I would NOT do, and why

**Do not adopt convolutions at any size measured.** 27, 11, 3.6 and 2.0
simulations per move against a floor of 96. Even with a perfect sgemm and zero
overhead everywhere else, 4 x 32 and above remain 8x to 78x slower than our
whole current evaluation (section 4). This is not close.

**Do not adopt AlphaZero's 8x8x73 policy head.** 602,818 parameters and 598,016
MACs — 22x the arithmetic of our entire network for the policy output alone,
measured at 2.05x slower than the factored query head. The factored move
embedding in `src/net.h` is strictly the better design on this hardware and
should be kept even if a conv trunk were ever adopted.

**Do not add 8 history positions.** 47 simulations per move, below the floor,
for a benefit that the 3-feature repetition count already captures most of.

**Do not "just use a smaller conv net" than 2 x 16.** Section 8: below ~9 conv
layers the network cannot propagate a sliding-piece attack across the board at
all, so shrinking it further removes the only thing convolutions were being
bought for.

**Do not widen the MLP as a consolation.** `src/net.h` already records the
sweep: the accumulator was measured at 128/160/192/224 and is "on the flat part
of the quality curve and the steep part of the cost curve" at 128. Spending
throughput on width has already been tried and rejected on evidence.

**Do not add any feature that ranks pieces**, however well it would work.
Section 7.

---

## 11. What this study does not show, and the honest alternative hypothesis

**This is a cost study. It measures price, not value.** I have not shown that
attack maps fix hanging pieces; I have shown they are affordable and that
convolutions are not. The recommendation is "the cheap thing is worth trying
because it is cheap and mechanistically targeted", not "the cheap thing will
work".

The falsifiable prediction, and the experiment: adopt bundles B then C', retrain
at **fixed wall-clock** (not fixed generations — the point is that the extra
features are paid for out of throughput), and measure against `material-1` with
quiescence, the opponent that currently beats us 27-0-3. If the hanging-piece
explanation is right, that score should move. If it does not move, the
representation hypothesis is wrong and the extra features should be reverted.

**And there is a competing explanation I cannot rule out, which I think is at
least as likely.** The cliff in the brief is between a 1-ply greedy material bot
(we beat it 100%) and *the same bot with a quiescence search* (we score 5%).
Those two differ **only in whether they resolve capture sequences**. That is a
difference in *search*, not in *evaluation*. It is entirely possible that our
network already represents danger adequately and that 160 PUCT simulations,
guided by a policy prior with no particular pull toward recaptures, simply
cannot resolve an exchange the way even a shallow quiescence search can. If
that is the real defect, better input features will not fix it, and the work
belongs in `src/mcts.c` — in how simulations are allocated down forcing lines —
rather than in `src/net.h`.

That is outside what I was asked to cost, and the two hypotheses are not
mutually exclusive. But the architecture plan would be dishonest if it presented
"add attack maps" as *the* answer when the measured evidence points at the
search as firmly as it points at the representation. Both should be tested, and
the feature work is worth doing first only because it is far cheaper to try.

---

## Appendix: the finding in one paragraph

Convolutions trade parameters for arithmetic. On a GPU that trade is free and
AlphaZero's shape is correct. On 8 CPU cores it runs backwards: the smallest
convolutional network in the brief has one seventh of our parameters and does
29 times the arithmetic, and even with Accelerate's AMX kernels running at 5–23x
better hardware utilisation than our sparse MLP achieves, it is 6x slower per
evaluation and 25x slower without Accelerate. At our fixed wall-clock that is 27
MCTS simulations per move against a measured floor of 96, so the better network
would be searched into uselessness. The spatial information convolutions were
being bought for — which squares each side attacks — is available exactly from
the move generator for 156 nanoseconds, and costs 14 simulations per move rather
than the 133 a 2 x 16 conv net costs.
