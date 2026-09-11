# Techniques we use, techniques we don't, and which ones would actually help

A survey of the standard AlphaZero-lineage tricks against **this** system: a
~154k-parameter MLP over a sparse 788-feature input, 160 MCTS simulations per
move, 8 CPU cores, no GPU, zero dependencies.

Every performance or quality number below is one I measured. The probe sources
and the exact commands are in the last section so anything here can be checked.

---

## 0. What was read, and when

Read at commit `ff485f4` ("Track the champion model and its telemetry"), with
the following files **dirty in the working tree** because other agents were
editing them while this was written:

| File | State when read | Note |
|---|---|---|
| `src/net.h`, `src/net.c` | clean at `ff485f4` | the architecture is **unchanged**; every finding below still applies |
| `src/mcts.h`, `src/mcts.c` | **modified, uncommitted** | subtree reuse, an evaluation cache and FPU reduction have just landed |
| `src/az.h`, `src/az.c` | **modified, uncommitted** | `value_mix`, `ema_decay`, `cap_frac`, `warmup_gens`, `grad_clip_head` have just landed |
| `docs/ALGORITHM.md`, `docs/FROM_SCRATCH.md` | clean at `ff485f4` | |

Measurements come from `runs/az_main/telemetry.jsonl` (250 generations) and
`runs/az_main/best.crl` (generation 243, 128 agents, best agent index 42,
internal population Elo 2130 — an internal scale, not a FIDE or CCRL rating).

Because several techniques listed as "missing" landed in the working tree
**during** this survey, each entry in section 3 is tagged `LANDING` where that
is the case. Nothing in `net.h` changed, so every architecture finding is live.

---

## 1. The measured diagnosis (read this before the survey)

The brief's hypothesis was: the value head is starved because the trunk's
gradients are dominated by the policy loss. **That is measurably false, and the
real cause is worse.**

### 1.1 The hidden layer is 93% dead

On 2,929 positions from real MCTS self-play with the champion agent:

| Layer | Width | Mean active units per position | Permanently dead units | rms |
|---|---:|---:|---:|---:|
| `h1 = relu(acc)` | 160 | 47.2 (29.5%) | 13 | 0.812 |
| `h2 = relu(W1·h1 + b1)` | 96 | **1.83 (1.9%)** | **89** | 0.056 |

89 of the 96 second-layer units never fire on any position. Of the 7 that ever
fire, only two fire regularly — `h2[28]` on 85.3% of positions and `h2[37]` on
93.2%; the other five fire on under 1%. Reproduced on a second agent (index 0,
Elo 1930): 84 dead, 1.68 active per position. `W1`/`b1` live in the shared
`Trunk`, so this is a property of the population's shared representation, not of
one head.

**Both heads read an effectively two-dimensional vector.** That is the whole
story. The value head is `Wv·h2 + bv` over a space with two live directions;
the policy head's `q = Wp·h2` is confined to a 2-D manifold inside its 32-D
embedding space.

This is textbook [dying ReLU](https://arxiv.org/pdf/1806.06068): the
pre-activation is negative for essentially every input, so the output and the
gradient are permanently zero and the unit can never recover. The two standard
causes are exactly what this trainer has — no normalisation anywhere, and a
large, badly conditioned gradient (see 1.3). The dead units' `W1` rows have
frozen at `‖row‖ = 4.31` against `‖row‖ = 1.41` at initialisation, i.e. they
grew 3x and then died; live rows sit at 3.08.

I cannot prove the units died *during* the run rather than early, because only
the final checkpoint is kept. At initialisation `b1 = 0` and
`W1 ~ N(0, sqrt(2/160))`, so about half of `h2` fires at step 0 — the layer
started healthy. **Log the per-layer dead-unit fraction in telemetry;** it is
four lines and it would have caught this at generation 20.

### 1.2 No read-out of this trunk can predict the outcome

Least-squares probes from each internal layer to the game result `z`, fitted on
150 fresh self-play games (20,774 positions, 160 sims, root noise on, the
training temperature schedule), split 50/50 **by game**:

| Read-out | R² train | R² test | Meaning |
|---|---:|---:|---|
| the actual value head `v` | — | **−0.11** | worse than predicting the mean |
| best linear `h2` (96) → z, λ=1 | +0.066 | **+0.039** | the ceiling for *any* linear value head on this trunk |
| best linear `h1` (160) → z, λ=1 | +0.259 | −0.117 | more signal, none of it generalises |
| best linear `h1` (160) → z, λ=1e4 | +0.114 | **+0.035** | best achievable after heavy regularisation |
| MCTS root value `q`, affine-rescaled | +0.069 | +0.036 | the search is no better |

A ridge sweep over λ ∈ [0.1, 1e5] never gets a linear read-out of any layer past
**R²_test ≈ 0.04**. Three conclusions follow, and they eliminate most of the
obvious fixes:

1. **Making the value head deeper or wider cannot help much yet.** Its ceiling
   on today's trunk is R² ≈ 0.04. Fix the representation first.
2. **The `h1` row is the variance story.** 0.26 in-sample collapsing to −0.12
   out-of-sample is not ordinary overfitting: every position in a game carries
   the *same* label, so 20,774 positions are ~150 independent labels. The
   Monte-Carlo target's effective sample size is the number of games. This is
   the brief's hypothesis 3, and it is real and large.
3. Telemetry's `value_mse_baseline` ratio is computed on the training
   minibatch, so the headline "R² fell from 0.65 to 0.16" is an *in-sample*
   number. Out of sample on fresh games it is ≈ 0. It is worse than reported.

Two independent 150-game runs gave R²(v,z) of −0.027 and −0.114 — the estimate
is noisy at this sample size, but the sign and the correlation (0.23–0.25) are
stable.

### 1.3 The value loss dominates the trunk gradient 50:1 — the opposite of the brief

On a real training batch (4,096 buffered positions, real MCTS visit targets,
`value_coef = 4.0`, `dv = 2·value_coef·(v − z)`):

| Gradient into | policy path | value path | ratio |
|---|---:|---:|---:|
| shared trunk | 1,233 | 70,280 | **0.02 : 1** |
| agent head | 146 | 6,842 | **0.02 : 1** |

The comparison that motivated the brief — "policy loss ~2.6 vs value loss ~0.35"
— compares a cross-entropy against a squared error. It is not a gradient. The
policy CE has an irreducible floor equal to the target's own entropy
(`target_entropy` ≈ 2.30 at the end of the run), so ~90% of that 2.56 is a
constant that produces no gradient at all.

I also verified from the telemetry that **the 250-generation run already used
`value_coef = 4.0`** (`loss.total = loss.policy + 4·loss.value` holds exactly at
generations 1 and 250). The value head collapsed *with* the fix that was
supposed to prevent it already applied.

This reframes the whole problem, and it implicates the fix itself:

> `value_coef = 4.0` makes the value loss the dominant gradient into a trunk
> with no normalisation. The value path into `h2` is `dh2[j] = dv·(1−v²)·Wv[j]`,
> and `Wv[37] = −2.45` — one huge, consistently-signed coefficient pushing one
> unit's pre-activation while everything else is driven negative. **The
> `value_coef` change is the prime suspect for killing the hidden layer.**

The comment in `az_default_cfg` says the 1 → 4 change was measured over 40
generations and improved both heads. It did — at generation 40 the run still had
R² ≈ 0.60. The collapse is not visible until generation ~100. **A 40-generation
tuning run cannot see a pathology whose signature takes 100 generations to
appear**, and the full run is only 52 minutes. Re-tune anything load-bearing at
full length.

This also explains the 16x rise in `grad_norm` (0.348 → 5.559 over the run,
which I confirmed from telemetry). It is the value term growing, and because the
clip is a single global norm over the whole trunk, a diverging value loss
**throttles the learning rate of the policy too**. `grad_clip_head` (landing
now) splits head from trunk but not value from policy.

### 1.4 A scale bug: the search values a draw at 0, training values it at −0.6

`src/mcts.c` sets `nd->tval = 0.0f` for stalemate, insufficient material,
fifty-move, repetition and the depth cap. `src/az.c` labels every drawn game
with `cfg->draw_penalty`, which was **−0.6** for this run. The search and the
trainer disagree about what a draw is worth by 0.6.

Measured consequence, same 150 games: `mean(z) = −0.486`, `mean(v) = −0.381`
(the net is roughly calibrated to its label), `mean(q) = −0.008` — the search
value sits 0.48 above the target scale. Raw `R²(q,z) = −0.91`; after affine
rescaling it becomes +0.036. The signal is there, the scale is wrong.

Two things follow:

- Both sides prefer a *terminal* draw (0.0) to a position the network calls
  drawish (−0.6). That is a shuffling incentive, and repetition draws rose from
  2.1% (gens 1–10) to 15.9% (gens 241–250) even at 160 simulations.
- **`value_mix` is about to inherit this.** `target = (1−mix)·z + mix·q` with
  `mean(q) − mean(z) = +0.48` injects a large systematic bias into the value
  target. KataGo-style target mixing is a good idea (see 3.3) but it must not be
  switched on until `tval` and `draw_penalty` agree. Either give terminal draws
  `draw_penalty`, or set `draw_penalty = 0` and express the anti-shuffling
  pressure some other way.

### 1.5 A third of the policy is a position-independent lookup table

Decomposing the logit `q·(Efrom+Eto+Epc+Epromo+Ecap) + Bft[from,to]` over the
legal moves of 1,524 self-play positions, per-position standard deviation:

| Term | sd |
|---|---:|
| trunk-dependent embedding term | 0.671 |
| `Bft[from,to]`, a static 64×64 table | 0.325 |
| total logit | 0.787 |

**32.6% of the policy's logit spread does not depend on the position at all.**
That is a consequence of 1.1: with a 2-D trunk output there is not much for the
position-dependent term to say, so a static move-ordering table picks up the
slack. (It is contract-legal — `Bft` is learned from self-play, not authored —
but it is a symptom.)

### 1.6 Where the time goes

| Quantity | Measured |
|---|---|
| 250 generations, 96,000 games, 2.25e9 evaluations | **52 minutes** wall clock |
| fraction of wall clock in self-play | **97.3%** |
| fraction in learning | 2.7% |
| `nn_eval` throughput, single thread | 207,047 / s (4.83 µs) |
| mean top-move visit share at 160 sims | 31.3% |
| mean search depth reached | 5.12 ply |
| mean branching factor | 23.6 |
| total decoupled weight-decay shrinkage over the run | **0.55%** |

Self-play is 97% of the budget, so throughput work converts almost 1:1 into more
simulations, and `docs/ALGORITHM.md`'s own sweep shows simulations are the
single lever that already works here (48 → 256 sims cut repetition draws from
60% to 3%). Weight decay at `1e-4` against a mean lr of ~1e-3 over 50,000 steps
shrinks weights by 0.55% in total: **weight decay is switched off in all but
name.**

---

## 2. What we already do

Credit where it is due — this is more of the AlphaZero recipe than most
from-scratch attempts get to.

**Architecture.** Sparse NNUE-style first layer (788 binary inputs, ~35 active,
so `W0·x` is a gather-and-sum of 35 rows). Side-to-move-relative encoding, which
correctly exploits chess's colour-flip symmetry *by canonicalisation* — the
vertical mirror, the our/their plane swap, and the castling-bit swap are all
handled, and the en-passant file is mirror-invariant. One trunk shared across
128 agents with per-agent heads and a per-agent style vector `z` added into the
accumulator. A factored policy head (from/to/piece/promo/capture embeddings plus
a from→to bias) instead of a 4672-way flat output — a genuinely good fit for a
tiny network.

**Optimisation.** AdamW with decoupled weight decay
([Loshchilov & Hutter](https://arxiv.org/abs/1711.05101)), bias correction,
global gradient-norm clipping, cosine learning-rate decay from `lr` to
`lr_final` over the run ([SGDR](https://arxiv.org/abs/1608.03983)), a
loss-weighting term (`value_coef`), a replay buffer with uniform sampling and
many minibatch steps per generation.

**RL and self-play.** MCTS visit distribution as the policy target. Result-only
reward. Resignation with a validated false-positive rate (`resign_check_frac`
plays a slice out anyway and scores the threshold — this is done properly).
Dirichlet root noise at AlphaZero's chess settings (α=0.3, ε=0.25). A two-step
temperature schedule (1.0 for 20 plies, then 0.25). Opening diversity via
Chess960 — all 960 arrays, mixed 10% classical, which is a stronger
anti-memorisation measure than AlphaZero used. Population-based training with
Elo-adjacent pairings, a frozen hall of fame anchoring the Elo scale, and
elite cloning with per-tensor-RMS-scaled mutation.

**Search.** PUCT with correct sign conventions, terminal-state caching, exact
in-tree threefold-repetition detection over the game history plus the selection
path (many hobby implementations get this wrong), incremental make/unmake with
one `Position` copy per search, graceful degradation on pool exhaustion, and
root noise disabled outside self-play. As of the working tree: **subtree reuse,
an evaluation cache keyed on the network input, and FPU reduction.**

**Measurement.** `policy_kl` rather than raw cross-entropy as the learning
curve; `value_accuracy_decisive` reported alongside the headline number
precisely because a negative draw penalty makes the headline flattering. That
scepticism is right and it is why the collapse was caught at all.

---

## 3. What we are missing

For each: what it is, whether it applies **here**, expected benefit, cost, and
whether `docs/FROM_SCRATCH.md` permits it.

Contract note applied throughout: the contract forbids *chess evaluation*, not
machine learning. Normalisation, skip connections, optimiser choices, auxiliary
targets derived from the game result, and bandit arithmetic are all domain-
independent and legal. The things that would violate it are called out
explicitly.

### 3.1 Architecture

| Technique | Applies here? | Expected benefit | Cost | Contract |
|---|---|---|---|---|
| **Normalisation (Layer/RMSNorm)** | **Yes — the single highest-value item** | Large | ~40 lines | Legal |
| **Leaky ReLU** | **Yes** | Large | ~8 lines | Legal |
| **Skip connections** | Yes, but as a shortcut not a depth fix | Medium | ~25 lines | Legal |
| Wider / deeper trunk | Only after the above | Medium | Small | Legal |
| Separate policy / value trunks | Yes, worth testing | Medium | ~60 lines | Legal |
| Squeeze-excitation | No | ~Zero | High | Legal |
| Dropout | **No — actively harmful here** | Negative | Small | Legal |
| 8-position history + repetition plane | Partly — the repetition bit, yes | Medium | Medium | Legal |

**Normalisation.** LayerNorm ([Ba et al.](https://arxiv.org/abs/1607.06450)) or
RMSNorm ([Zhang & Sennrich](https://arxiv.org/abs/1910.07467)) rescales a
layer's pre-activations to a fixed statistic before the nonlinearity. This is
*the* standard prevention for a layer whose pre-activations drift until they are
all negative, and it is exactly the pathology measured in 1.1. **Batch norm is
the wrong choice here** even though AlphaZero and KataGo both use it: it needs
minibatch statistics at training time and a separate inference path, and
`nn_eval` is a single-position call on the hot path at 207k/s. RMSNorm costs one
pass over 160 floats plus a reciprocal square root, needs no batch, and behaves
identically in training and play. Apply it to `acc` before `relu` and to `z2`
before `relu`. Expected benefit: this is the binding constraint on both heads;
nothing else in this document matters as much.

**Leaky ReLU.** `f(x) = x` for `x>0`, `0.01x` otherwise. A dead unit keeps a
small gradient and can come back. It is the cheapest possible fix — about eight
lines across `nn_eval` and `nn_backward`, no new parameters, no change to the
model file format or the ABI. Do this *first*, measure, then add normalisation.
A full 250-generation run is 52 minutes, so this is a same-afternoon experiment.

**Skip connections.** [He et al.](https://arxiv.org/abs/1512.03385). At two
hidden layers this network is far too shallow for residual connections to matter
for gradient flow — the usual argument does not apply. But a skip has a
*different* and directly relevant use here: a connection from `h1` (160 wide,
29.5% alive) past `h2` to the heads routes around the dead layer entirely.
Caveat from the measurement: the `h1` probe reaches R²_test only ≈ 0.035 and
only under heavy regularisation, so a raw `h1 → value` shortcut would overfit
game identity. Do it after normalisation, not instead of it.

**Width and depth.** `NF_ACC=160`, `NF_HID=96`. Right now 93% of the second
layer is unused, so the effective width is ~2. Adding width before fixing that
adds dead units. Afterwards, `NF_HID` is the obvious first knob, and the cost is
cheap: the `W1` matmul is 160×96 = 15,360 MACs against the sparse first layer's
~5,600, and evaluation is 97% of wall clock, so doubling `NF_HID` costs roughly
25% throughput. Measure the trade against simulation count.

**Separate policy and value trunks.** AlphaZero shares one residual tower and
splits only in the heads. Given 1.3 — the value gradient is 50x the policy
gradient into a shared trunk — a split is more attractive here than it normally
would be, because it removes the interference mechanism entirely rather than
trying to balance it. Against it: the shared trunk is the thing that makes
population training work at all (`net.h`'s design note is right about that), and
duplicating it doubles `W0`, the largest tensor. Middle path: keep `W0` shared
and give the value head its own small `W1v`. That is cheap and testable.

**Squeeze-excitation.** [Hu et al.](https://arxiv.org/abs/1709.01507);
[Lc0 adopted it in T35 and it is now standard there](https://lczero.org/dev/backend/nn/).
It reweights *channels* using globally pooled *spatial* context. This network
has no channels and no spatial structure — it is a plain MLP over a flat feature
vector. There is nothing to squeeze. **Skip it**; the analogous idea for a flat
net is just a gating layer, which is not where the problem is.

**Dropout.** [Srivastava et al.](https://jmlr.org/papers/v15/srivastava14a.html).
**Do not.** Dropout fights overfitting by removing capacity. This model has 89 of
96 hidden units permanently removed already and a linear-probe ceiling of
R² ≈ 0.04; it is massively *under*fitting its input. Dropping units from a layer
with 1.83 live units per position is close to deleting the network. The one
place where something dropout-shaped is defensible is the value target's
overfitting to game identity (1.2) — but the right tool for that is target-
variance reduction (3.3), not activation noise.

**Input history and the repetition plane.** AlphaZero's chess input is
[8 positions of history plus repetition and no-progress counts](https://arxiv.org/abs/2304.14918);
we use one position plus castling, en-passant file and a halfmove-clock bucket.

Split the question honestly:
- *History planes*: in chess, unlike Go, the position is Markov — the legal
  moves and the result depend only on the current state plus castling/ep/clock,
  all of which we already encode. AlphaZero's 8-step history exists mostly to
  make repetition visible and for architectural uniformity with Go. 8 positions
  would multiply the first layer's cost by ~8, which at 97% self-play time is
  unaffordable. **Verdict: skip.**
- *A repetition feature*: **this one matters, and its absence is a real bug in
  the representation.** The search knows about repetitions (`mcts_rep_count`),
  the network cannot see them at all. So the network cannot learn that a
  position is one repetition from a draw, cannot value the difference, and its
  value head is being trained on labels (draws) it has no feature to explain.
  With repetition draws at 15.9% of games, this is not marginal. Two extra input
  bits — "this position has occurred once before", "twice before" — cost two
  rows of `W0` and a `Position` key lookup that `Game` already maintains.
  **Highest benefit-to-cost ratio of anything in this table after 3.1's first
  two rows.**

### 3.2 Optimisation

| Technique | Applies here? | Expected benefit | Cost | Contract |
|---|---|---|---|---|
| **Per-loss gradient scaling** | **Yes** | Large | ~20 lines | Legal |
| LR range test ("lr_finder") | Marginal | Small | Medium | Legal |
| LR warmup | Yes, small | Small | ~5 lines | Legal (`LANDING`) |
| Cosine vs step decay | Already cosine | Zero | — | — |
| SGD+momentum instead of AdamW | No | Negative | — | Legal |
| Weight EMA / Polyak | Yes | Small–medium | ~40 lines | Legal (`LANDING`) |
| Gradient clipping | Present but mis-scoped | Medium | ~15 lines | Legal (partly `LANDING`) |
| Weight decay | Present but inert | Small | 1 line | Legal |
| Label smoothing | No | ~Zero | — | Legal |
| Mixed precision | **No** | Negative | — | Legal |

**Per-loss gradient scaling (not in any list, and the one that matters).** The
measured 50:1 imbalance (1.3) is not fixable by tuning `value_coef`, because
`value_coef` is the thing that *created* it. What is missing is normalising each
loss's gradient contribution to the trunk — compute the policy and value trunk
gradients separately, rescale each to a target norm, then sum. `az.c` already
has the machinery: the `w->probe` block runs the backward pass twice more
precisely to decompose the two. Turning that decomposition from a diagnostic
into a control loop is maybe 20 lines and directly targets the mechanism that
killed the hidden layer.

**LR range test.** [Smith, Cyclical Learning Rates](https://arxiv.org/abs/1506.01186):
ramp the learning rate exponentially over one short run and read the maximum
stable value off the loss curve. Honest assessment for this system: it is a tool
for when you do not know the right order of magnitude, and `2e-3` for AdamW on a
154k-parameter net is already in the right place. More importantly, the loss
here is non-stationary — the data distribution changes as the agents improve —
so a range test at generation 1 tells you little about generation 200. **Low
priority**, and a full 250-generation run at 52 minutes is a *better* experiment
than an LR finder.

**Warmup.** [Goyal et al.](https://arxiv.org/abs/1706.02677). Adam's second
moment is badly estimated for the first few dozen steps, so the first updates
are effectively enormous. With a value gradient this large that is a plausible
contributor to early damage to `W1`. Cheap, low risk, `LANDING` as
`warmup_gens`.

**SGD with momentum vs AdamW.** AlphaZero used SGD with momentum 0.9 and a
stepped schedule. Do not switch. Adam's per-parameter scaling is exactly right
for a sparse first layer where feature rows are updated at wildly different
frequencies (a king on e1 fires constantly, an en-passant file rarely). SGD
would need per-feature learning rates to match. **Keep AdamW.**

**Weight EMA / Polyak averaging.** [Polyak & Juditsky, 1992]; keep
`θ_ema ← d·θ_ema + (1−d)·θ` and play with the average. Standard, cheap,
usually worth a small but free gain, and it also damps the generation-to-
generation churn that PBT cloning introduces. `LANDING` as `ema_decay` with an
`ema_h2h_games` head-to-head gate before promotion, which is the right way to do
it — measure, don't assume it helps.

**Gradient clipping.** Present, but it is one global norm over the whole trunk,
so a diverging value loss silently throttles the policy's learning rate too. The
logged `grad_norm` is measured **before** clipping and reaches 5.56 by
generation 250. The clip value used for that run is not in the telemetry — it is
not a logged field, which is itself worth fixing — so the scale-down has to be
computed for each candidate: at the `grad_clip = 1.0` the brief reports, the
trunk update was divided by ~5.6 through the last third of the run; at today's
default of 4.0 it would be ~1.4. Either way it is an unintentional, undocumented
second learning-rate decay stacked on top of the cosine schedule, and it tightens
as training proceeds.
`grad_clip_head` (`LANDING`) splits head from trunk. It should also split value
from policy, and the clip-hit rate should be in telemetry (`az.c` now counts
`gclip_n` — surface it).

**Weight decay.** Measured total shrinkage over the whole run: **0.55%**. It
does nothing at `1e-4`. Either raise it by ~100x and see if it regularises
anything, or set it to 0 and stop pretending. Given the model is underfitting,
0 is the better default.

**Label smoothing.** [Szegedy et al.](https://arxiv.org/abs/1512.00567) /
[Müller et al.](https://arxiv.org/abs/1906.02629). The policy target is already
a soft visit distribution with entropy 2.30 nats over ~24 moves — it is smoothed
by construction. Smoothing further just adds bias. **Skip.**

**Mixed precision.** On Apple Silicon CPU there is no fp16 matmul throughput
advantage of the kind a GPU tensor core gives, the hot path is a *sparse gather*
of 35 rows (memory-bound, and fp16 would halve the traffic — the only real
argument), and every gradient here is already marginal. NEON fp16 arithmetic
would need `__fp16` intrinsics, breaking the "C11 + libc" simplicity for a
speedup that is at best 1.3x on one layer. **Not worth it.** If you want
throughput, section 3.4 has cheaper wins. (Honest caveat: int8 quantisation of
`W0` in the *inference* path, NNUE-style, is a real 2–4x — but that is a
shipping optimisation, not a training one.)

### 3.3 RL and self-play

| Technique | Applies here? | Expected benefit | Cost | Contract |
|---|---|---|---|---|
| **Value-target variance reduction (mix `z` with `q`)** | **Yes — after fixing 1.4** | **Large** | ~20 lines | Legal (`LANDING`) |
| **WDL (win/draw/loss) value head** | **Yes** | Large | ~80 lines | Legal |
| Playout cap randomisation | Yes | Medium (throughput) | ~30 lines | Legal (`LANDING`) |
| Forced playouts + policy target pruning | Yes | Medium | ~50 lines | Legal |
| Auxiliary targets (KataGo ownership/score) | **No chess analogue** | ~Zero | High | Mostly illegal |
| Auxiliary target: opponent's reply policy | Yes | Small–medium | ~40 lines | Legal |
| TD(λ) value targets | Yes, as a generalisation of mixing | Medium | ~40 lines | Legal |
| Resignation | Already done, and done well | — | — | — |
| Replay buffer sizing / sampling | Already measured; prioritised sampling — no | ~Zero | — | Legal |
| Left-right mirror augmentation | Valid 75% of the time; low value | Small | ~40 lines | Legal |
| Opening diversity | Already best-in-class (960) | — | — | — |
| Population-based training | Already done | — | — | — |

**Value-target variance reduction.** This is the direct attack on 1.2. Instead
of giving all 138 positions of a game the same coin-flip label, blend in the
search value computed at that exact position:
`target = (1−λ)·z + λ·q_search`. Bias for variance. It is not injected knowledge
— `q` is this network's own search over its own value head, bootstrapped from
self-play, exactly as the MCTS visit distribution is for the policy. The
literature on
[value targets in AlphaZero](https://link.springer.com/article/10.1007/s00521-021-05928-5)
finds soft-Z and related targets train faster than pure `z` on Connect-Four and
Breakthrough. **`LANDING` as `value_mix`, defaulting to 0.** Two warnings:
1. **Do not enable it until the draw-value scale mismatch in 1.4 is fixed.**
   Measured `mean(q) − mean(z) = +0.48`.
2. `q` is computed *with* Dirichlet root noise during self-play, so it is biased
   by exploration. KataGo's playout cap randomisation exists partly to give
   clean, noise-free, high-playout values for exactly this purpose — the two
   features compose, and `cap_frac` is landing alongside `value_mix`.

**A WDL value head.** Predict `P(win), P(draw), P(loss)` with a 3-way softmax
and cross-entropy instead of a scalar `tanh` with MSE.
[Lc0 moved to this in 2019](https://lczero.org/blog/2020/04/wdl-head/) because a
single scalar cannot distinguish "certain draw" from "50% win / 50% loss".
Unusually strong case here:
- It **removes `draw_penalty` from the value scale entirely.** The draw penalty
  currently contaminates everything — it is why `mean(z) = −0.49`, why the
  `value_accuracy` headline number is misleading, and why the search/train scale
  mismatch in 1.4 exists at all. With WDL, the draw preference becomes a
  *search-time* contempt parameter applied to a proper probability, which is
  where it belongs and where Lc0 puts it.
- Cross-entropy on a softmax gives a bounded, well-conditioned gradient, instead
  of `dv = 2·value_coef·(v−z)` with `value_coef = 4`. That directly attacks 1.3.
- With 52% draws, a 3-way target carries strictly more information per label
  than a scalar that squashes draws onto an arbitrary point.
Cost: three outputs instead of one (`Wv` becomes 96×3), a softmax and CE in the
loss, and `mcts.c` must convert WDL to a scalar for backup. `MODEL_VERSION`
bumps. ~80 lines. **Strongly recommended, but after 3.1's first two rows** —
on a 2-D trunk it will not have anything to predict with.

**Playout cap randomisation.** [KataGo](https://arxiv.org/abs/1902.10565): run
the full budget on only a fraction `p` of moves and record only those for
training; run a cheap search on the rest and play them without recording.
KataGo uses N=600, n=100, p=0.25 and measures a **1.37x** acceleration. The
tension it resolves is exactly ours: policy targets need many playouts, value
targets need many *games*, and at 97.3% self-play time this is throughput for
free. `LANDING` as `cap_frac` / `cap_sims`.

**Forced playouts and policy target pruning.** Also KataGo, measured **1.25x**:
force each root child to at least `n_forced(c) = sqrt(k·P(c)·ΣN)` playouts with
`k=2`, then *subtract* those forced playouts back out of the training target so
the target is not polluted by forced exploration. Directly relevant: at 160 sims
over 23.6 legal moves, ~6.8 playouts per move on average, so the visit
distribution is a coarse estimate and Dirichlet noise leaks straight into the
policy target. Legal — it is bandit arithmetic over priors and visit counts,
with no reference to the position. **Good value for ~50 lines.**

**KataGo's auxiliary heads — be precise about which transfer.** KataGo's biggest
single ablation win is ownership and score targets (**1.65x**), and it is
tempting to reach for. Honestly:
- *Ownership* (which player ends up owning each point) has **no chess analogue**.
  Chess has no territory. The nearest things anyone proposes — "which squares
  are controlled at the end", "material at the end" — are not derivable from the
  result and are precisely the hand-authored positional evaluations that
  `docs/FROM_SCRATCH.md` forbids. **Do not.**
- *Score* (final margin) likewise: the chess analogue is final material, which
  is piece values. **Forbidden, explicitly** — it is the `avg_final_material`
  shaping that was already removed once.
- *Opponent-reply policy* (predict the opponent's next move, loss weight 0.15,
  measured **1.30x**) **does transfer cleanly**. It is derived from the game
  record alone, contains no evaluation, and is a pure representation-learning
  regulariser — it forces the trunk to encode something about the position
  beyond the current mover's options. Given that the trunk representation is the
  measured bottleneck, this is the auxiliary target worth having.
- *Global pooling* (**1.60x**) is spatial and does not apply to a flat MLP,
  same as squeeze-excitation.

**TD(λ).** The general form of which `value_mix` is the λ=const special case:
bootstrap the value target from the search value `n` plies ahead, geometrically
weighted. More principled than a fixed blend and a natural follow-up once
`value_mix` is measured. Legal — it bootstraps from the agent's own search.

**Replay buffer.** Already measured (the comment in `az_default_cfg` is right:
capacity sets staleness, not sample count, and 400k cost 18 points of policy
top-1). One temptation to resist: **prioritised replay by value error would be
actively wrong here.** It would preferentially resample positions where `v−z` is
large, which with Monte-Carlo labels means preferentially resampling the *most
mislabelled* positions — surprising outcomes, not informative ones. Skip.

**Data augmentation — the mirror question, worked out.** Chess has exactly two
candidate symmetries and the answer differs for each:

1. **Colour flip** (vertical mirror + swap the sides): a true symmetry of the
   rules. **Already exploited**, by canonicalisation rather than augmentation —
   the side-to-move-relative encoding in `nn_features` halves the input space
   rather than doubling the data, which is strictly better. Nothing to add.
2. **Left-right mirror** (`sq ^ 7`): **not a symmetry whenever castling rights
   exist.** Under a file reflection e1 ↔ d1, so a king on its castling square
   maps to a square from which standard castling is undefined; kingside and
   queenside swap. Precisely:
   - *Standard chess, castling rights empty*: a strict symmetry. Mirror the
     piece squares, mirror the en-passant file (`file → 7−file`; the ep square
     itself is not mirror-invariant even though the *file index* encoding means
     you just re-index it), leave the halfmove clock, and mirror every move's
     from/to and the `Bft` entry.
   - *Standard chess, any castling right present*: **invalid**, and the position
     produced is not reachable.
   - *Chess960*: the 960 arrays are closed under file reflection, so the mirror
     of a legal 960 position with rights is another legal 960 position —
     provided you also swap the kingside/queenside castling bits, which for our
     4-bit encoding is a bit swap. So it is valid here more often than in
     standard chess.
   Measured: **75.1% of self-play positions have no castling rights at all**, so
   the easy case covers three quarters of the data. 4.6% have an en-passant
   square.
   **But: recommended low priority.** Augmentation buys effective data, and 1.2
   shows data volume is not the binding constraint — 12.8M sample-updates were
   applied over the run and the model still underfits catastrophically. Fix the
   representation first; revisit augmentation when the network can actually use
   more data.

### 3.4 Search

| Technique | Applies here? | Expected benefit | Cost | Contract |
|---|---|---|---|---|
| **Subtree reuse** | Yes — 31.3% of sims recoverable | Medium | ~120 lines | Legal (`LANDING`) |
| **Network evaluation cache** | Yes | Small–medium | ~80 lines | Legal (`LANDING`) |
| In-tree transposition table | Marginal | Small | High | Legal |
| FPU reduction | Yes | Small–medium | ~10 lines | Legal (`LANDING`) |
| c_puct scheduling (the log formula) | Yes | Small | ~5 lines | Legal |
| Virtual loss | **No** | Zero | — | Legal |
| Dirichlet parameters | Fine as-is; consider scaling α | ~Zero | ~5 lines | Legal |
| Temperature schedule | Fine; a smooth decay is marginally better | Small | ~10 lines | Legal |
| Gumbel AlphaZero root selection | Partly | Small here | High | Legal |

**Subtree reuse.** Standard in
[Lc0](https://github.com/LeelaChessZero/lc0/wiki/Technical-Explanation-of-Leela-Chess-Zero)
("the chosen move is made the new root of the tree"). `mcts.c` at `ff485f4`
rebuilt from scratch every move. Measured: **the played move holds 31.3% of the
160 simulations on average**, so reuse inherits about 50 visits — roughly a 1.45x
effective budget, or the same strength for ~30% less compute. `LANDING`. The
new header's contract ("`sims` is a total, not an increment") makes reuse a pure
cost saving rather than a bigger tree, which is the conservative and correct
choice. Note the extra win in the *play* path: `uci.c` ramps sim budgets
(32, 64, 128, …) and comments that "the tree is thrown away", costing up to 2x.

**Evaluation cache.** Memoise `nn_eval` by input. Measured transposition density
by full enumeration from real self-play positions: **0.0% duplicates at depth 2,
35.8% at depth 3** (416,065 paths → 267,221 distinct). Depth 2 is 0% by
construction — after one move each, the position determines the path. Caveat
against over-claiming: a 160-node PUCT tree is sparse, so it will realise far
less than 35.8%; the real win is *across* moves in combination with reuse. At
4.83 µs per evaluation a hit is cheap to be worth having. `LANDING`, and keyed
on the full network input rather than the zobrist key, which is the right call —
the input includes the halfmove bucket, so two positions with the same zobrist
key are not interchangeable for the network.

**In-tree transposition table** (sharing node statistics between tree paths that
reach the same position, as opposed to just caching the evaluation) is a much
larger change, breaks the tree invariants that the repetition logic depends on,
and given the depth-2/depth-3 numbers above buys little at 5-ply search depth.
**Not worth it here.**

**FPU reduction.** The value assigned to an unvisited edge. We used a flat
`fpu = 0.0` ("assume a draw"), which at 160 sims over 23.6 moves is an
optimistic constant that encourages breadth. Lc0 uses parent-Q minus a reduction
scaled by explored policy mass, [default `fpu-reduction` 0.9](https://lczero.org/play/flags/).
`LANDING` as `fpu_reduction` with the flat behaviour preserved at 0, which is
the right way to add it. One interaction worth flagging: with the value head at
R² ≈ 0, `fpu = 0` was accidentally *close to optimal* — a parent-Q-relative FPU
propagates a meaningless parent value. Retune FPU **after** the value head works,
not before.

**c_puct scheduling.** We use a constant 1.4. AlphaZero's own formula grows it
with visits: `C(s) = log((1+N(s)+c_base)/c_base) + c_init` with
`c_base = 19652`, `c_init = 1.25`
([pseudocode](https://gist.github.com/erenon/cb42f6656e5e04e854e6f44a7ac54023)).
At `N = 160` the log term contributes `log(19813/19652) ≈ 0.008` — i.e. **the
formula is indistinguishable from a constant 1.25 at our simulation count.** It
only matters at tens of thousands of visits. Implementing it faithfully is five
lines and harmless, but the honest expected benefit at 160 sims is ~zero.
Sweeping the *constant* is worth more: `mcts_defaults` has never been tuned
against this network, and Lc0 found 2.5–3.0 optimal for theirs.

**Virtual loss.** A temporary visit penalty so that parallel threads in **one**
tree explore different branches. We parallelise across *games*, one tree per
thread, with no shared tree. **There is nothing for virtual loss to do.** It
would be pure overhead. Skip — and this is a good example of a canonical
AlphaZero technique that is simply irrelevant to this architecture.

**Dirichlet noise.** α=0.3, ε=0.25 are AlphaZero's chess values and the sampler
is correct (Marsaglia-Tsang with the α<1 boost — a detail many implementations
get wrong). KataGo scales α inversely with the average number of legal moves;
chess branching is stable enough at ~24 that this matters little. One thing that
*would* matter: with subtree reuse landing, root noise must be re-drawn on the
inherited root rather than compounding across moves — the new header's
`tree_noised` flag suggests this was thought about.

**Temperature.** A step from 1.0 to 0.25 at ply 20. AlphaZero used 1.0 for 30
moves then →0. A smooth decay is marginally better but this is a second-order
knob. Leave it.

**Gumbel AlphaZero.** [Danihelka et al., ICLR 2022](https://iclr.cc/virtual/2022/poster/6418):
sample root actions without replacement via Gumbel-top-k, allocate the budget by
sequential halving, and complete unvisited action values by interpolation. It
comes with a **guaranteed** policy improvement, whereas plain AlphaZero can fail
to improve the policy when it does not visit every root action. Tempting at
first glance because we run only 160 simulations. But the benefit is largest
when `sims < number of actions`, and here 160 sims over 23.6 legal moves means
roughly 6.8 visits per move — **most root actions do get visited**, so the
failure mode Gumbel fixes is not our failure mode. It is also a substantial
rewrite of the root of `mcts_search`. **Interesting, not a priority.** It would
become a priority if playout cap randomisation drops the fast-search budget to
~30 sims, where the action count *does* exceed the budget.

---

## 4. Ranked shortlist: the five changes most likely to move this agent

Ranked by expected strength gained per unit of implementation effort, given the
measured diagnosis. The economics that shape this list: **a full 250-generation
run costs 52 minutes**, so anything that can be tested by retraining is cheap to
evaluate, and there is no excuse for tuning a load-bearing parameter over 40
generations when the pathology it causes appears at 100.

### 1. Leaky ReLU in the trunk, then RMSNorm before both nonlinearities

*~8 lines, then ~40.* Directly targets the measured cause: 89 of 96 hidden units
permanently dead, both heads reading an effectively 2-dimensional vector, and a
linear-probe ceiling of R² ≈ 0.04 on the outcome. Nothing else in this document
can beat that ceiling, because every other item reads from this representation.
Do Leaky ReLU first because it is eight lines and costs nothing at inference,
measure, then add normalisation. Expect this to move the value head, the policy
head and the shuffling rate together — they all trace to the same bottleneck.

**Add the diagnostic at the same time**: per-layer dead-unit fraction in
telemetry. It would have caught this 230 generations earlier.

### 2. Rebalance the value and policy gradients, and stop `value_coef = 4.0` from doing damage

*~20 lines.* The value path contributes **50x** the policy path's gradient norm
into the shared trunk, with one coefficient (`Wv[37] = −2.45`) dominating. That
is the most likely mechanism for how the layer died, and it is the direct result
of a parameter that was tuned over 40 generations to fix the symptom. Normalise
each loss's trunk-gradient contribution to a target norm before summing, rather
than picking a scalar `value_coef` and hoping. `az.c` already computes the
decomposition for telemetry; promote it to a control. Split the gradient clip by
loss as well as by trunk/head, and log both the clip value and the clip-hit rate
— neither is in today's telemetry, so the size of this undocumented second
learning-rate schedule cannot be read off a finished run.

### 3. Make the search and the trainer agree about the value of a draw — before `value_mix` ships

*~5 lines for the fix; it gates a much larger feature.* `mcts.c` backs up
`0.0` for every drawn terminal; `az.c` labels drawn games `−0.6`. Measured
consequence: `mean(q) = −0.008` against `mean(z) = −0.486`, and
`R²(q,z) = −0.91` that becomes `+0.036` after affine rescaling. Both sides
currently prefer a forced draw to a drawish position, and repetition draws rose
2.1% → 15.9% over the run. **`value_mix` is landing right now and will blend
`q` into the value target; switched on as-is it injects a +0.48 systematic
bias.** This is five lines of work standing in front of the single best
variance-reduction technique available, which is why it ranks this high.

### 4. A WDL (win / draw / loss) value head

*~80 lines, bumps `MODEL_VERSION`.* Three-way softmax with cross-entropy instead
of scalar `tanh` with MSE. It attacks three measured problems at once: it gives
the value loss a bounded, well-conditioned gradient (fixing #2 structurally
rather than by tuning); it removes `draw_penalty` from the value scale entirely,
which is what created #3; and with 52% of games drawn it carries strictly more
information per label than a scalar that collapses "certain draw" and "50/50
win-loss" onto the same number. Lc0 made this change for exactly these reasons.
Sequenced after #1 because on a 2-D trunk there is nothing for a better head to
predict with.

### 5. Feed the network the repetition count, and take the throughput wins

*~10 lines for repetition; the rest is `LANDING`.* Two input bits for "this
position has occurred once / twice before". The search knows about repetitions;
the network is blind to them and is being trained on draw labels it has no
feature to explain, with 15.9% of games ending that way. Trivial cost, and it is
the one representational gap versus AlphaZero (which encodes repetition planes)
that genuinely matters for chess — unlike the 8-position history, which is
redundant for a Markov game and would cost 8x the first layer.

Alongside it, bank the throughput: subtree reuse (**measured 31.3% of
simulations recoverable**), the evaluation cache, and playout cap randomisation
(KataGo: **1.37x**). Self-play is **97.3%** of wall clock, and this repo's own
sweep shows simulation count is the lever that already works — 48 → 256 sims
took repetition draws from 60% to 3%. Every 1.4x of throughput is a free
increase in search quality.

**Deliberately just off the list**, in order: forced playouts with policy target
pruning (KataGo 1.25x, ~50 lines, a real win once #1 lands); the opponent-reply
auxiliary head (KataGo 1.30x, the one KataGo auxiliary target with a legal chess
analogue, and it pressures exactly the representation that is broken); and
sweeping `c_puct`, which has never been tuned against this network.

---

## 5. Things that sound good and would not help here

| Technique | Why not |
|---|---|
| **Dropout** | The model has 89 of 96 hidden units already permanently removed and a linear-probe ceiling of R² ≈ 0.04. It is underfitting by a mile. Dropout removes capacity to fight overfitting that is not happening. |
| **A deeper or wider value head** | Measured ceiling for *any* linear read-out of any layer of this trunk is R²_test ≈ 0.04, across a ridge sweep from λ=0.1 to 1e5. A better head on the same trunk cannot beat 0.04. Fix the trunk first. |
| **Raising `value_coef` further** | It is already 4.0 and already produces 50x the policy's gradient. It is a suspect in the collapse, not a cure. |
| **Squeeze-excitation, global pooling** | Both reweight channels using pooled *spatial* context. This is a flat MLP with no channels and no board geometry. KataGo's global pooling is its second-largest ablation win (1.60x) and it transfers to us not at all. |
| **Virtual loss** | Parallelism is across games with one tree per thread. There is no shared tree to de-conflict. |
| **KataGo's ownership and score heads** | Their biggest single win (1.65x), and both are forbidden. Chess has no territory; the score analogue is final material, i.e. piece values, which `docs/FROM_SCRATCH.md` bans by name and which was already removed from this codebase once. |
| **8 positions of input history** | Chess is Markov given castling/ep/clock, which we already encode. AlphaZero's history planes are mostly there to expose repetition — take the two repetition bits instead, at 1/8th the cost of the first layer. |
| **Label smoothing** | The policy target is a visit distribution with 2.30 nats of entropy. It is already smooth; smoothing adds bias and nothing else. |
| **Mixed precision** | No CPU tensor cores. The hot path is a sparse 35-row gather, and halving memory traffic on one layer is at most ~1.3x for real complexity and a dependency on `__fp16`. Subtree reuse gives more, for free, and is already landing. |
| **Prioritised replay by value error** | With Monte-Carlo labels, large `\|v−z\|` means "this game ended surprisingly", not "this position is informative". It would preferentially resample the most mislabelled data. |
| **An LR range test** | A tool for when the order of magnitude is unknown; 2e-3 for AdamW on 154k parameters is not. The objective is non-stationary, so a generation-1 result does not describe generation 200, and a full run costs 52 minutes anyway. |
| **c_puct's log schedule** | At N=160 the log term adds 0.008 to c_puct. It is a constant at our simulation count. Sweep the constant instead. |
| **Gumbel AlphaZero** | Fixes the case where simulations < root actions. We run 160 simulations over 23.6 legal moves, so nearly every root action is visited. Revisit if playout cap randomisation pushes the fast budget below ~25. |
| **Left-right mirror augmentation** | Valid (75.1% of positions have no castling rights, and Chess960 arrays are closed under reflection if the K/Q bits swap). But it buys effective *data*, and the measured bottleneck is representation, not data — 12.8M sample-updates already went in and the model still underfits. |
| **Switching AdamW → SGD+momentum** | Adam's per-parameter scaling is what makes a sparse first layer trainable when feature rows fire at wildly different rates. AlphaZero used SGD on a dense convnet; that is not this. |

---

## 6. Reproducing the measurements

All probes are read-only, link against the repo's own objects
(`build/chess.o build/net.o build/mcts.o`), and modify nothing. They live in the
session scratchpad, not the repo:

```
/private/tmp/claude-501/-Users-johnanthonyheaney-Desktop-ChessRL-ChessRL/\
48288563-0aad-4064-8914-b56df27a204e/scratchpad/
```

| Probe | Produces |
|---|---|
| `probe.c` | parameter counts; value distribution; first-pass gradient balance |
| `probe2.c` | out-of-sample R²(v,z), R²(q,z), the ridge-swept linear probes from `h1`/`h2` to `z` |
| `probe3.c` | gradient balance with real MCTS targets; visit share; transposition density; `nn_eval` throughput |
| `probe4.c` | weight and activation rms against initialisation; weight-decay arithmetic |
| `probe5.c` | dead-unit counts per layer on the self-play distribution (the headline finding) |
| `probe6.c` | which units are alive and why; the policy-logit decomposition |
| `probe7.c` | fraction of positions where a left-right mirror is legal |

Build and run, from the repo root:

```sh
make -j8
cc -O3 -std=c11 -D_DARWIN_C_SOURCE -Wall -Isrc <probe>.c \
   build/chess.o build/net.o build/mcts.o -o <probe> -lm
./<probe> runs/az_main/best.crl 160 150
```

Telemetry aggregates come from `runs/az_main/telemetry.jsonl` with the standard
library only. Every probe calls `chess_init()` first — without it `gen_moves`
segfaults, which is worth knowing.

---

## Sources

- David J. Wu, *Accelerating Self-Play Learning in Go* (KataGo) — playout cap randomisation, forced playouts and policy target pruning, global pooling, auxiliary heads, and the ablation table quoted above: <https://arxiv.org/abs/1902.10565> and <https://arxiv.org/html/1902.10565v5>
- KataGo methods notes: <https://github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md>
- Silver et al., *A general reinforcement learning algorithm that masters chess, shogi and Go through self-play*: <https://gwern.net/doc/reinforcement-learning/model/alphago/2018-silver.pdf>
- AlphaZero pseudocode (`pb_c_base = 19652`, `pb_c_init = 1.25`, Dirichlet α=0.3, ε=0.25): <https://gist.github.com/erenon/cb42f6656e5e04e854e6f44a7ac54023>
- Czech et al., *Representation Matters for Mastering Chess* — AlphaZero's chess input planes and history: <https://arxiv.org/abs/2304.14918>
- Lc0, *Win-Draw-Loss evaluation*: <https://lczero.org/blog/2020/04/wdl-head/>
- Lc0, technical explanation (tree reuse, NNCache): <https://github.com/LeelaChessZero/lc0/wiki/Technical-Explanation-of-Leela-Chess-Zero>
- Lc0, network topology (SE blocks): <https://lczero.org/dev/backend/nn/>
- Lc0, engine parameters (`cpuct`, `cpuct-base`, `fpu-reduction`): <https://lczero.org/play/flags/>
- Danihelka et al., *Policy improvement by planning with Gumbel*, ICLR 2022: <https://iclr.cc/virtual/2022/poster/6418>
- Willemsen et al., *Value targets in off-policy AlphaZero: a new greedy backup*: <https://link.springer.com/article/10.1007/s00521-021-05928-5>
- He et al., *Deep Residual Learning*: <https://arxiv.org/abs/1512.03385>
- Ba, Kiros & Hinton, *Layer Normalization*: <https://arxiv.org/abs/1607.06450>
- Zhang & Sennrich, *Root Mean Square Layer Normalization*: <https://arxiv.org/abs/1910.07467>
- Hu, Shen & Sun, *Squeeze-and-Excitation Networks*: <https://arxiv.org/abs/1709.01507>
- Srivastava et al., *Dropout*: <https://jmlr.org/papers/v15/srivastava14a.html>
- Loshchilov & Hutter, *Decoupled Weight Decay Regularization* (AdamW): <https://arxiv.org/abs/1711.05101>
- Loshchilov & Hutter, *SGDR: Stochastic Gradient Descent with Warm Restarts* (cosine schedule): <https://arxiv.org/abs/1608.03983>
- Goyal et al., *Accurate, Large Minibatch SGD* (learning-rate warmup): <https://arxiv.org/abs/1706.02677>
- Smith, *Cyclical Learning Rates for Training Neural Networks* (the LR range test): <https://arxiv.org/abs/1506.01186>
- Szegedy et al., *Rethinking the Inception Architecture* (label smoothing): <https://arxiv.org/abs/1512.00567>
- Müller, Kornblith & Hinton, *When Does Label Smoothing Help?*: <https://arxiv.org/abs/1906.02629>
- Tan, Le & Shokoufandeh, *Detecting Dead Weights and Units in Neural Networks*: <https://arxiv.org/pdf/1806.06068>
