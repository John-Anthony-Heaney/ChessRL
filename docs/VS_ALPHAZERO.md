# VS_ALPHAZERO — what they did, what we do, and what to change

A cited comparison between AlphaZero (and Leela Chess Zero, KataGo, and the
credible small-scale reimplementations) and **this** system, written to answer
one question: **given 8 CPU cores and about an hour, what should we change?**

Every number attributed to us is one I produced; the command is in §9. Every
number attributed to someone else has a URL. Where a source does not give a
number I say so rather than inventing one.

Read alongside `docs/TECHNIQUES.md` (the technique-by-technique survey, written
before LayerNorm, the residual block and the split value trunk landed — its
top-2 items are now **done**, which is why this document does not repeat them)
and `docs/RATING.md` (why internal Elo is ignored throughout).

---

## 0. The state of play, restated in numbers

`runs/az_hour/best.crl`, generation 1894 of a 1900-generation run:

| | value |
|---|---|
| wall clock | 4183.5 s (69.7 min) |
| generations / games / plies | 1900 / 729,600 / 97,875,636 |
| network evaluations | 5.14 × 10⁹ (mean 1.60 × 10⁶ /s across 8 threads) |
| positions recorded for training | 24,489,651 (33.4 per game) |
| optimiser steps × batch | 380,000 × 256 = 97,280,000 sample-updates |
| self-play / learning share of wall clock | 78.1% / 20.4% |
| parameters (shared trunk + one head) | 157,473 (trunk 143,328, head 14,145) |

External result (`docs/RATING.md`, `runs/sf_ladder_hour.json`): 100% against a
1-ply greedy material bot, **5.0%** against `material-1` — the *same* bot with a
6-ply quiescence search bolted on — and 0/40 against Stockfish's `UCI_Elo 1320`
floor.

---

## 1. Side by side

`material-1` is the opponent that breaks us: negamax depth 1 over material,
alpha-beta, 6-ply quiescence over captures and promotions, delta pruning,
P=1 N=3 B=3.25 R=5 Q=9 (`py/baselines.py`).

| | **AlphaZero (chess)** | **Leela Chess Zero** | **KataGo (Go)** | **ChessRL (`az_hour`)** |
|---|---|---|---|---|
| net | 1 conv + **19 residual blocks × 256 filters, 3×3** ([CPW](https://chessprogramming.org/AlphaZero)) | 64×6, **128×10**, 192×15, 256×20 ([lczero](https://lczero.org/dev/wiki/training-runs/)) | 6×96 → 10×128 → 15×192 → **20×256** ([§KataGo](https://arxiv.org/html/1902.10565v5)) | flat MLP: sparse 788→128, **one** residual block 128→128, separate 788→32→32 value trunk |
| parameters | ≈2.3 × 10⁷ in the tower (my arithmetic: 19 × 2 × 3×3×256×256) | 1M–2.5 × 10⁷ | ≈2.3 × 10⁷ at 20×256 | **1.57 × 10⁵** |
| input | **119 planes**, 8-position history, **3 repetition planes**, castling, no-progress, colour, move count ([ar5iv](https://ar5iv.labs.arxiv.org/html/1712.01815)) | same family | 22 planes + global inputs | **788 sparse bits, ~35 active, ONE position, no history, no repetition** |
| policy encoding | 8×8×73 = **4,672** flat outputs | same | 19×19+1 | factored embeddings (from/to/piece/promo/capture + from→to bias), 32-dim |
| simulations / move (self-play) | **800** ([pseudocode](https://gist.github.com/erenon/cb42f6656e5e04e854e6f44a7ac54023)) | 800 | **600 full / 100 fast, p = 0.25**, annealed to 1000/200 | **160 full / 24 fast, p = 0.25 → 58 nominal, 52.4 measured evals/move** |
| games | **44,000,000** | test10 alone spanned **games 15M–67M** ([lczero blog](https://lczero.org/blog/2018/10/lc0-training/)) | **4.2M** over 19 days | **729,600** |
| steps × batch | **700,000 × 4096** = 2.9 × 10⁹ samples | — | 256 batch, ~2.41 × 10⁸ samples | **380,000 × 256** = 9.7 × 10⁷ samples |
| optimiser | SGD, momentum 0.9, wd 1e-4 | — | SGD, momentum 0.9 | **AdamW**, wd 1e-4 |
| LR schedule | **0.2 → 0.02 → 0.002 → 0.0002** at steps 0/100k/300k/500k (1000× total) | high, then "occasional LR drops" | 6e-5/sample → 6e-6 (10×) | cosine **2e-3 → 1e-3** — a **2× decay** (measured, §4.4) |
| replay window | last **10⁶ games** | — | **grows** 250k → ~22M samples | **fixed 50,000 positions** (≈4 generations, ≈1,500 games) |
| value target | game outcome *z* ∈ {−1,0,+1}, MSE | outcome blended with search *q*; **WDL head since July 2019** ([lczero](https://lczero.org/blog/2020/04/wdl-head/)) | **3-way W/D/L softmax**, c_value = 1.5, + score/ownership heads | outcome *z*, **draw = −0.6**, scalar tanh + MSE, `value_mix` implemented but **0** |
| draws | scored 0; games over the step cap adjudicated drawn | WDL | komi randomised N(7, 1) | draw = `draw_penalty` (−0.6 here), symmetric |
| Dirichlet | **α = 0.3, ε = 0.25** | same | shaped | **α = 0.3, ε = 0.25** ✅ |
| temperature | τ = 1 for the first **30 moves** (pseudocode `num_sampling_moves`), greedy after | — | — | τ = 1.0 for **20 plies**, then **0.25** (never greedy) |
| resignation | self-play pseudocode has none (`max_moves` 512); **5% win-probability** resignation in the evaluation matches | — | — | v < −0.90, **10%** played out to validate (16.9% of games resigned at gen 1900) |
| hardware / time | **5,000 TPUv1 + 64 TPUv2, 9 h** | thousands of volunteer GPUs, years | **≤28 V100, 19 days** (~1.4 GPU-years) | **8 CPU cores, 70 min** |

Two rows are worth staring at. Our **simulation budget is 13.8× below AlphaZero's**
and 10× below KataGo's full-search cap, and our **replay window is fixed at
50,000** where KataGo's grows to 22 million. Both are one-line changes.

---

## 2. The compute gap, as arithmetic

This is the number every proposal below has to be measured against.

**Per evaluation.** AlphaZero: 19 blocks × 2 convs × (3×3×256×256) × 64 squares
= **1.43 × 10⁹ multiply-adds**. Ours: the 35-row gather into 128 (4,480) + the
128×128 residual matmul (16,384) + the value trunk (1,120 + 1,024) + the policy
query (4,096) + ~28 move logits ≈ **2.8 × 10⁴ multiply-adds**. Ratio **≈52,000×**.

**Number of evaluations.** AlphaZero: 44 × 10⁶ games × ~135 plies × 800 sims
≈ **4.75 × 10¹²**. Ours: **5.14 × 10⁹** (measured, telemetry `evals`). Ratio **≈924×**.

**Total self-play arithmetic** (counting a multiply-add as two operations, both
sides the same way). AlphaZero ≈ **1.4 × 10²²**; ChessRL ≈ **2.9 × 10¹⁴**.
Ratio **≈5 × 10⁷**.

Cross-check, independently: 5,000 TPUv1 × 9 h × 92 × 10¹² ops/s = **1.49 × 10²²**.
The two estimates agree to 10%, so the 5 × 10⁷ figure is sound. (Training adds
~6% on our side — 380,000 steps × 256 × 2.8 × 10⁴ × 3 for forward+backward — and
is negligible on theirs.)

Put the other way: at our measured 1.6 × 10⁶ evaluations/second, **matching
AlphaZero's number of network evaluations would take 34 days of this laptop —
with a network 52,000 times smaller**. No item in §7 moves a factor of 10⁷. Every
item in §7 is about spending 2.9 × 10¹⁴ operations better.

---

## 3. What KataGo changed, and what we already have

KataGo is the right prior art because it is explicitly about efficiency at small
scale: it passed ELF OpenGo in 19 days on fewer than 30 GPUs, a claimed **50×
reduction in computation**
([abstract](https://arxiv.org/abs/1902.10565)). Its ablation table, measured at
2.5 G equivalent queries
([Table 2, §5.2](https://arxiv.org/html/1902.10565v5)):

| KataGo technique | speedup | do we have it? |
|---|---|---|
| Auxiliary ownership & score targets | **1.65×** | **Forbidden.** Go's score analogue in chess is final material. `docs/FROM_SCRATCH.md` bans piece values by name. |
| Global pooling | **1.60×** | **Does not transfer.** It reweights channels using pooled *spatial* context. We are a flat MLP with no channels and no board geometry. |
| Game-specific features/optimisations | **1.55×** | Mostly forbidden (they are Go evaluation knowledge). |
| **Playout cap randomisation** (p = 0.25, N/n = 600/100) | **1.37×** | ✅ **shipped and on** — `cap_frac` 0.25, `cap_sims` 24 (`src/az.h`, telemetry `full_search_frac` 0.2507). Our N and n are 3.75× and 4× smaller than KataGo's. |
| Auxiliary policy targets (opponent reply) | **1.30×** | ❌ not implemented. Legal — it is a self-play-derived target. |
| **Forced playouts + policy target pruning** (k = 2) | **1.25×** | ❌ not implemented. Legal. |

Other things KataGo does that we already do: uniform-window replay
(`buffer_positions`), a self-play-derived value target with a mixing parameter
(`value_mix`, implemented, default 0), Polyak/EMA weights (`ema_decay` 0.999 with
a head-to-head promotion gate — **KataGo does not do this; it is ours**), and a
validated resignation threshold.

Things KataGo does that we do **not**, beyond the table:

* **A growing replay window**: `N_window = c(1 + β((N_total/c)^α − 1)/α)` with
  `c = 250,000, α = 0.75, β = 0.4` ([Appendix C](https://arxiv.org/html/1902.10565v5)).
  At our final data volume (24.5M recorded positions) that formula gives a
  **4.3M-sample window — 85× ours**.
* **Branching to off-policy moves**: "in 2.5% of positions, the game is branched
  to try an alternative move drawn randomly from the policy… a full search is
  performed to produce a policy training sample", and "in 5% of games… between 3
  and 10 moves are chosen uniformly at random". This is deliberate coverage of
  *bad* play. We have Chess960 opening diversity but nothing at the move level.
* A three-way W/D/L value head with cross-entropy (c_value = 1.5).
* Later KataGo work (not in the paper) reports: subtree value bias correction
  **+30…60 Elo**, optimistic policy **+40…90 Elo**, uncertainty-weighted playouts
  with variance-scaled cPUCT **≈+75 Elo**
  ([KataGoMethods.md](https://github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md)).

**Verdict on KataGo**: of its six ablated techniques, three are forbidden or
inapplicable to us, one we already ship, and two (1.30× and 1.25×) are available.
Its two *unablated* ideas — the growing window and off-policy branching — are
cheaper than either and are in §7.

---

## 4. Where we differ from AlphaZero, item by item

### 4.1 Network size and shape

AlphaZero: 19 residual blocks of 256 3×3 filters over an 8×8 board. Depth is
doing work: 38 stacked 3×3 convolutions give a receptive field far larger than
the board, which is how the net computes long-range relations like "the bishop on
g2 attacks d5".

Ours: **two** nonlinearities on the policy path (and two more on the separate
value trunk). `acc = W0·x + b0 + z` is *linear in the input* — the accumulator is
a plain **sum** of per-(piece, square) vectors. Any "piece A attacks piece B"
relation is a *product* of two input bits, and every such product has to be
manufactured by 128 hidden units shared across all **294,528** unordered pairs of
the 768 piece-square features. NNUE solves exactly this by pairing features in
the *input* (HalfKP/HalfKA) and by being 2–8× wider. We do neither.

`src/net.h` records a measured sweep of the accumulator at 128/160/192/224 and
concludes 128 is "on the flat part of the quality curve and the steep part of the
cost curve". That is real evidence, but it spans a **1.75× range** at fixed data,
and it was taken to fix a throughput/quality trade, not to test whether 4× wider
helps. It is weak evidence against a 512-wide accumulator.

Against it stands the only scaling-law study of AlphaZero agents I could find:
Neumann & Gros, *Scaling Laws for a Multi-Agent Reinforcement Learning Model*
([arXiv:2210.00849](https://arxiv.org/abs/2210.00849)), on Connect Four and
Pentago, who report power laws for strength vs parameters and vs compute, that
**"previously published state-of-the-art game-playing models are significantly
smaller than their optimal size, given the respective compute budgets"**, and that
**"large AlphaZero models are more sample efficient, performing better than
smaller models with the same amount of training data."**

### 4.2 Input representation

AlphaZero's chess input is 119 planes: 8 positions of history, **repetition
counts**, castling, no-progress counter, colour, total move count
([Table S1](https://ar5iv.labs.arxiv.org/html/1712.01815)). We encode one
position: pieces, 4 castling bits, the en-passant file, and a halfmove-clock
bucket — **no repetition information at all**, while 16.4% of our games end by
repetition and a further 15.4% by insufficient material.

The best-measured representation result at hobbyist scale is Czech et al.,
*Representation Matters: The Game of Chess Poses a Challenge to Vision Transformers*
([arXiv:2304.14918](https://arxiv.org/abs/2304.14918)): moving from a 39-plane
AlphaZero-style input to a 52-plane input is worth **+96.7 ± 30.4 Elo**, and a
WDLP value head a further **+33.2 ± 19.0 Elo** (RTX 2070, KingBase Lite 2019,
batch 1024, 7 epochs, three seeds).

**Read the fine print before getting excited**: their extra planes include
"relative material difference (5 planes)" and "material counts (5 planes)". The
single largest documented representation win for a small chess network is
**exactly what `docs/FROM_SCRATCH.md` forbids by name**. That is not an argument
for breaking the contract; it is an argument for being honest that the contract
has a measured price, and that price is on the order of 100 Elo.

### 4.3 Simulations per move, and how the budget was chosen

AlphaZero: 800, fixed, for every self-play move. KataGo introduced playout cap
randomisation precisely because that number is a compromise between two different
jobs, and stated the trade-off explicitly:

> "ideal numbers of playouts for policy learning are much larger, not far from
> AlphaZero's choice of 800 playouts per move" — while value training benefits
> from more games at fewer playouts
> ([§ Playout Cap Randomization](https://arxiv.org/html/1902.10565v5))

We already run the split. We run it at **160/24**, against KataGo's **600/100**
rising to **1000/200**. Our *policy* targets are therefore produced by a search
that is 3.75–6× weaker than the one KataGo judged adequate and 5× weaker than
AlphaZero's.

What that looks like from inside the tree, measured on 600 positions at 160 sims
(`diag_tactics tree`):

| | |
|---|---|
| mean legal moves | 28.3 |
| root moves receiving **any** visit | 16.5 of 28.3 (58%) |
| root moves receiving **more than one** visit | **3.2** |
| visit share of the top move | 0.636 |
| **mean leaf depth** | **3.10 plies** (deepest 12) |

A policy target built from 3.2 meaningfully-visited moves and a 3-ply lookahead is
barely an improvement on the policy that generated it. That is what a policy KL
bottoming out at generation ~350 and a top-1 agreement plateauing at 0.49–0.51
look like from the outside.

The counter-argument, and it is a real one: Wang, Emmerich, Preuss & Plaat,
*Analysis of Hyper-Parameters for Small Games: Iterations or Epochs in Self-Play?*
([arXiv:2003.05988](https://arxiv.org/pdf/2003.05988)) conclude that the outer
loop of self-play iterations "subsumes" MCTS simulations and epochs, and should
be maximised while the inner-loop parameters are set *low*. Their games are 6×6
boards where the search saturates almost immediately; at 3.10 plies of lookahead
over 28 legal moves we are nowhere near saturation. And this repo has already
measured the opposite once: `docs/ALGORITHM.md`'s sweep at fixed generations took
repetition draws from 0.599 at 48 sims to 0.053 at 160 and 0.027 at 256, and
policy KL from 0.527 to 0.379.

### 4.4 Steps, batch, optimiser, learning rate

We are **not** short of optimiser steps. AlphaZero took 700,000; we took 380,000.
What we are short of is *samples per step* (256 vs 4096, 16×) and *distinct data*
(9.7 × 10⁷ sample-updates over 2.4 × 10⁷ recorded positions — each position is
re-used **≈4 times**; AlphaZero's 2.9 × 10⁹ samples over ≈5.9 × 10⁹ positions
(44M games × ~135 plies) means each position was used **less than once**).

Optimiser: AlphaZero used SGD + momentum 0.9 + weight decay 1e-4; we use AdamW.
`docs/TECHNIQUES.md` is right that Adam is the correct choice for a sparse first
layer whose feature rows fire at wildly different rates — do not change it.

**Learning rate — a measured finding.** AlphaZero drops the LR **1000×** across
the run (0.2 → 0.02 → 0.002 → 0.0002 at steps 0/100k/300k/500k), and the Lc0
project reports that "improvements accelerate after these reductions"
([lczero blog](https://lczero.org/blog/2018/10/lc0-training/)). Fitting the
cosine in `src/az.c:2895–2905` to the telemetry's `lr` column recovers
`lr = 2.0e-3`, `lr_final = 1.0e-3`, `generations = 1900` exactly (predicted
0.001895 at gen 400 against a logged 0.001895). **The `az_hour` run decayed its
learning rate by 2× in total**, against `az_default_cfg`'s own default of 10×
(2e-3 → 2e-4) and AlphaZero's 1000×. That is a flag, not a code change.

### 4.5 The value target, and draws

AlphaZero regresses the game outcome *z* with MSE and scores a draw 0. We
regress *z* with MSE and score a draw `draw_penalty` = **−0.6** in this run — a
symmetric choice the contract explicitly permits, and one made to stop the
shuffling equilibrium.

**But `src/mcts.c` backs up 0.0 for every drawn terminal** (lines 671, 676, 679,
682, 690 — stalemate, insufficient material, fifty-move, repetition and the ply
cap). So the search and the trainer disagree about the value of a draw by 0.6,
and the disagreement is live in the shipped configuration.

Measured consequence over the last 200 generations of `az_hour`:

| | |
|---|---|
| mean MCTS root value *q* | **+0.020** |
| mean value target *z* | **−0.305** |
| draw rate | 0.455 |
| Var(z) baseline | 0.5454 |
| MSE(*q*, *z*) / Var(z) | **0.993** |
| MSE(*v*, *z*) / Var(z) | 0.658 |
| MSE(*q*, *z*) / Var(z) after removing the constant +0.325 offset | 0.800 |
| generations in which the **search** beat the bare value head at predicting the result | **0 of 1900** |

Two conclusions, and the second is the more interesting one.

1. The search currently prefers a *forced* draw (backed up as 0.0) to a *drawish*
   position (evaluated by a head trained to say −0.6). It is manufacturing the
   very terminations the draw penalty exists to suppress: 16.4% repetition +
   15.4% insufficient material + 12.5% ply-cap at generation 1900.
2. **`value_mix` is not justified by this run's evidence, even after the scale
   fix.** `docs/TECHNIQUES.md` item 3 argued the draw mismatch "gates" value
   mixing. It does — but the telemetry's own ceiling test (`search_mse_outcome`,
   put there for exactly this purpose) says *q* is worse than *v* at tracking *z*
   **after** the offset is removed (0.800 vs 0.658). The comparison is not
   perfectly fair — *v*'s error is measured near-in-sample on the minibatches it
   was just trained on — but "turn on value_mix" is currently a hypothesis, not a
   conclusion, and it is a 2 × 7-minute A/B to settle.

### 4.6 Resignation, temperature, noise

* **Dirichlet noise**: α = 0.3, ε = 0.25 — identical to AlphaZero. Nothing to do.
* **Temperature**: AlphaZero samples proportional to visits for the first 30
  *moves* (60 plies) then plays greedily. We sample at τ = 1.0 for 20 *plies*,
  then at τ = 0.25 forever. Our exploration window is 3× shorter and we never
  become greedy. Untested here; cheap to sweep; low expected value.
* **Resignation**: we resign at v < −0.90 with `resign_check_frac` = 0.10 played
  out and scored — this is done properly, exactly as AlphaGo Zero specified, and
  16.9% of games at generation 1900 ended that way. AlphaZero's published
  self-play pseudocode has *no* resignation (games are cut at `max_moves` 512);
  the 5% win-probability resignation is described for the evaluation matches.
  The cost of resigning is that the value head never sees the conversion phase of
  a won game. Leave it; the throughput is worth more.

---

## 5. The failure we actually measured: 100% → 5%

The greedy bot and `material-1` search to the *same nominal depth* with the *same*
evaluation. The only difference is that `material-1` resolves capture sequences.
We beat the first 40–0 and lose 95% to the second.

### 5.1 The literature has a name for this, and it is 16 years old

Ramanujan, Sabharwal & Selman, *On Adversarial Search Spaces and Sampling-Based
Planning*, ICAPS 2010
([PDF](https://www.cs.cornell.edu/~raghu/Raghuram_Ramanujan_files/icaps10.pdf)),
defines a **level-k search trap**: a move after which the opponent has a
guaranteed win within k plies. Their measured findings, on 200 grandmaster games
and 200 semi-random games:

* Shallow traps are **everywhere in chess** and essentially absent in Go. At
  47-ply depth, **as many as 15%** of semi-random boards and **roughly 12%** of
  grandmaster boards contain a level-1…7 trap; at 63 ply, **18.44%** and **17.00%**.
* **UCT reliably identifies level-3 traps and reliably misses level-5 and
  level-7 traps.** Table 1: on a level-5 trap board, UCT's utility for the trap
  move is −0.066 against −0.056 for its own best move — indistinguishable.
* **More simulations do not fix it.** "Even if UCT is allowed to sample as many as
  10 times the number of nodes minimax needs to identify the trap, the utility
  assigned by UCT to the trap state remains high, over −0.15, as long as the trap
  is at level 5 or higher."
* Fig. 4: with a level-5 trap present, **UCT spends nearly 70% of its iterations
  below depth 5**; with a level-7 trap, **95% below depth 7** (as deep as level 20),
  while minimax searches only to depth 3.

That is our agent, described in 2010, before AlphaZero existed.

### 5.2 The same pathology, measured on `runs/az_hour/best.crl`

Using `tools/diag_tactics` (a measuring instrument — it contains piece values and
a quiescence search on purpose, is never linked into the agent, and defines a
**blunder** as "the chosen move loses ≥ 2.0 pawns to `material-1`"):

**The value head is the ceiling, not the search.**

| | |
|---|---|
| mean V after a safe move / after a blunder | −0.4691 / −0.7372 |
| **P[V(safe) > V(blunder)]** | **0.872** (0.500 = blind) |
| P[V(safe) > V(blunder)] *after* the forced recapture | 0.862 |
| corr(V after move, oracle value of move), 16,976 moves | r = **+0.239** |
| prior mass on blunder moves / under a uniform prior | 0.219 / 0.322 (ratio 0.681) |
| policy argmax **is** a blunder | **0.222** |
| **blunder rate at 160 sims** | **0.105** |

The blunder rate (0.105) is almost exactly the value head's pairwise error rate
(1 − 0.872 = 0.128). **The search cannot rank correctly what its own leaf
evaluator ranks wrongly.** And the mechanism is Ramanujan's: at 160 sims the
worst hanging move at the root receives 1.48 visits, its subtree is expanded 54%
of the time, and **the refuting capture is searched at all only 5.3% of the time**.

**Does more search fix it? Partly — and it matters which positions you ask about.**

Blunder rate vs simulation count, three corpora, 600 positions each:

| sims | random-play openings | agent self-play | **positions from games vs `material-1`** |
|---:|---:|---:|---:|
| *blunders among the legal moves* | *33.5%* | *18.8%* | *12.5%* |
| 1 | 0.303 | 0.057 | 0.012 |
| 8 | 0.187 | 0.043 | 0.303 |
| 32 | 0.163 | 0.033 | 0.158 |
| 80 | 0.103 | 0.030 | 0.108 |
| 160 | **0.105** | 0.028 | **0.108** |
| 400 | 0.112 | 0.025 | 0.103 |
| 1600 | 0.093 | 0.023 | 0.055 |
| 6400 | **0.100** | 0.005 | **0.023** |
| *random legal mover* | 0.322 | 0.183 | 0.121 |

On the random-play corpus the rate is flat from 80 sims on — 40× more search buys
0.005. On the corpus that actually arises *against the opponent we lose to*, it
falls **4.7×** from 160 to 6400. (The 1-simulation entry in the third column is
not a typo and not a result: at 1 simulation the move is the raw policy argmax on
a corpus where only 12.5% of the legal moves are blunders at all; the search
starts *worse* than the prior at 8 simulations and has to climb back. Read that
column from 32 sims up.) The end-to-end match confirms the second reading
(60 games per rung, same openings and seed, colours reversed within each pair):

| champion search budget | score vs `material-1` | Elo | ms/move |
|---:|---:|---:|---:|
| 64 sims | 2.5% (1W 1D 58L) | −636 | 1.4 |
| **256 sims (the shipped/benchmarked setting)** | **5.8%** (1W 5D 54L) | −483 | 2.2 |
| 1024 sims | 7.5% (1W 7D 52L) | −436 | 6.4 |
| **4096 sims (`API_MAX_SIMS`, the engine's cap)** | **20.0%** (6W 12D 42L) | **−241** | 42.2 |

**+396 Elo across six doublings (66 Elo/doubling), and the curve is steepest at
the top: 1024 → 4096 is +196 Elo, i.e. 98 Elo per doubling.** `src/api.c:157` caps
`sims` at 4096, so this is where the measurement stops, not where the curve does.
(The 60-game rungs have ±~9-point score error bars, so read the shape, not the
individual steps.)

**Is the blunder rate falling with training, and how fast?** Same corpus, same
seed, across the checkpoints in `runs/` (different runs with different configs,
so this is a trend, not a controlled scaling law):

| checkpoint | cumulative games | blunder @160 | V-separation |
|---|---:|---:|---:|
| random weights | 0 | 0.337 | +0.0003 |
| `az/best.crl` g27 | 1,792 | 0.292 | −0.0072 |
| `az_v2/best.crl` g86 | 33,024 | 0.293 | +0.0139 |
| `az_v3/best.crl` g295 | 113,280 | 0.240 | +0.1101 |
| **`az_hour/best.crl` g1894** | **727,296** | **0.105** | **+0.2681** |

A log-log fit over all four points gives `blunder ≈ 1.15 × games^−0.156`; a fit
over the last two, where the slope is much steeper, gives `−0.445`. Extrapolated
to a 5% blunder rate at 174 games/second, those two fits say **875 hours** and
**6 hours** respectively.

**That two-orders-of-magnitude spread is the finding, not the endpoints.** The
curve is still falling at generation 1894 and nobody can say from four
cross-run points how fast. What it does rule out is "one more hour of the same
thing"; what it makes cheap is the experiment that settles it — **run the
identical configuration for 8 hours instead of 1 and re-measure** (§7 item 3b).
If the optimistic slope holds, that alone roughly halves the blunder rate.

### 5.3 What the literature says the cure is, and why most of it is closed to us

**MCTS-minimax hybrids without an evaluation function.** Baier & Winands,
*MCTS-Minimax Hybrids*, IEEE TCIAIG
([PDF](https://dke.maastrichtuniversity.nl/m.winands/documents/mcts-minimax_hybrids_final.pdf)),
embed shallow minimax into MCTS's selection/expansion (MCTS-MS), rollout
(MCTS-MR) or backpropagation (MCTS-MB) phases, **using only terminal values and
no domain knowledge**. Best-setting win rates against plain MCTS-Solver: Catch
the Lion ≈78%, Breakthrough ≈69%, Connect-4 ≈54%, Othello ≈50% (MCTS-MS), and
"both MCTS-MS and MCTS-MB were effective up to a branching factor of at least
50" — chess's ~28–35 is inside that range.

**Why it will not save us.** The hybrid works by finding *terminal* states within
the minimax horizon. In Catch the Lion and Breakthrough, losing a piece often *is*
losing. **In chess, hanging a queen is not a terminal state.** A knowledge-free
shallow minimax under our MCTS would find forced mates and nothing else — and our
losses to `material-1` are material losses, not mates. The authors list chess as
future work and say so: extending to "larger games of similar type such as Shogi
and chess" is proposed, not done. The version of this idea that *would* work —
minimax over the **learned value head** at a shallow depth, or a capture-only
extension — is discussed in §7 items 11 and 12, where the contract question is.

**Graph search / transpositions.** Czech, Korus & Kersting,
*Monte-Carlo Graph Search for AlphaZero*
([arXiv:2012.11045](https://arxiv.org/abs/2012.11045)), generalise the tree to a
DAG so transpositions share statistics: **+310 Elo in crazyhouse but only +69 Elo
in chess**, with 30–70% memory saved, at 5 s/move and ~17,000 evaluations/second.
Chess is the weak case, and +69 Elo at 17k nps is not a cure for tactical
blindness.

**Bigger network, no search.** The cleanest answer to "how big must a chess
network be before it stops hanging pieces" is Ruoss et al., *Grandmaster-Level
Chess Without Search* / *Amortized Planning with Large-Scale Transformers*
([arXiv:2402.04494](https://arxiv.org/abs/2402.04494)), Table 1:

| agent | tournament Elo | puzzle accuracy |
|---|---:|---:|
| **AlphaZero policy network alone** (no search) | **1777** | **56.1%** |
| AlphaZero value network alone (1-ply greedy) | 1992 | 82.0% |
| AlphaZero + 400 MCTS simulations | 2470 | 95.6% |
| 9M-param transformer, supervised on 15B Stockfish-labelled points | 2025 | 88.9% |
| 136M-param | 2259 | 94.5% |
| 270M-param | 2299 (2895 vs humans) | 95.4% |

Read the first row again. **AlphaZero's own fully trained network, with no search,
solves 56.1% of puzzles and plays at 1777.** Search is worth ~700 Elo to it. The
same gap in Go is larger still: AlphaGo Zero's raw 40-block network scored
**3055 Elo against 5185 with search**
([AGZ, Nature 2017](https://discovery.ucl.ac.uk/10045895/1/agz_unformatted_nature.pdf)).

So "the network should just see the hanging piece" is not how this works even at
DeepMind scale. What a good network buys you is a leaf evaluation accurate enough
that a *modest* search finds the refutation. Their 9M-parameter model — **57×
ours, trained supervised on Stockfish labels, not self-play** — reaches 88.9%
puzzle accuracy. Our 157k-parameter model reaches 87.2% on the far easier task of
pairwise-ranking a safe move above a 2-pawn blunder.

### 5.4 Has anyone made this work at our scale?

**No, and I could not find anyone who claims to have.**

* **Zeta36/chess-alpha-zero**, the best-known single-machine AlphaZero chess
  repo, reports "a guesstimate of 1200 Elo with 1200 sims/move" from *supervised*
  learning on ~10,000 games with a 7-block net, states "the self-play is too much
  costed for an only machine", and redirects users to the distributed Leela Chess
  project ([README](https://github.com/Zeta36/chess-alpha-zero/blob/master/readme.md)).
  **They never ran self-play to convergence.**
* A current single-GPU AlphaZero chess implementation
  ([yurit04/alpha_chess](https://github.com/yurit04/alpha_chess)) targets
  "roughly 1500–2000 Elo" with a **128-channel, 10-block** net (≈3 × 10⁶
  parameters, ~20× ours) at 200/50 simulations on an RTX 3090, budgets
  "realistically a couple of days", and publishes **no verified match result** —
  only instructions for the reader to bracket the strength themselves.
* Lc0's own first high-quality networks (test10, 256×20) were trained on **games
  15 million through 67 million** ([lczero blog](https://lczero.org/blog/2018/10/lc0-training/)).
  Early Lc0 was publicly noted for "very strange tactical oversights, even in
  fairly simple positions", attributed to using 10×128 instead of 20×256.

The honest statement: **every from-scratch chess self-play run that has produced
a verified result used a network of 3 × 10⁶ to 2.3 × 10⁷ parameters (20× to 150×
ours) and 10⁷–10⁸ games (10× to 100× ours).** Lc0 does ship 64×6 networks at
≈5 × 10⁵ parameters, only 3× ours — but those are distilled from the main run's
data, not trained from scratch. Nothing in §7 changes that. §7 is about extracting
more from the hour, not about reaching 1500 Elo in it.

---

## 6. What NOT to do

| | why not |
|---|---|
| 8 positions of input history | Chess is Markov given castling/ep/clock, which we encode. 8× the cost of the largest tensor. AlphaZero's history planes exist mostly to expose repetition — take the two repetition bits instead (§7 item 10). |
| Squeeze-excitation / global pooling | KataGo's second-largest win (1.60×) and Lc0 standard since T35. Both reweight *channels* using pooled *spatial* context. We have neither. |
| Ownership / score auxiliary heads | KataGo's largest win (1.65×). The chess analogue of score is final material. Banned by name in `docs/FROM_SCRATCH.md`, and removed from this codebase once already. |
| Derived material-difference input planes | Czech et al.'s +96.7 Elo. Same ban. Stated here so the price of the contract is on the record. |
| Turning on `value_mix` today | §4.5: `search_mse_outcome_ratio` ≥ `value_mse_outcome_ratio` in **1900 of 1900 generations**, and still worse after the draw offset is removed. It is gated on the draw fix *and* on a measurement, not just the draw fix. |
| Raising `value_coef` above 4.0 | `grad_value_share` is already **0.905** at generation 1900 — the value loss supplies 90.5% of the shared trunk's gradient. It is a suspect, not a cure. |
| Gumbel AlphaZero | Fixes the case where simulations < root actions, for the searches that *produce training targets*. Our full searches run 160 over 28.3 legal moves. The 24-sim fast searches are below the action count but are never recorded, so Gumbel has nothing to improve there. |
| Switching AdamW → SGD | AlphaZero used SGD on a dense convnet. Adam's per-parameter scaling is what makes a sparse first layer trainable. |
| Dropout, label smoothing, mixed precision | Unchanged from `docs/TECHNIQUES.md` §5; all still wrong here. |

---

## 7. The ranked, costed plan

Ranked by **expected strength per (implementation hour + compute hour)**. "Impl"
is my estimate of hands-on time; "compute" is what it costs inside the 70-minute
budget. Every item says how it would be *measured* — external blunder rate and
the `material-1` score, never internal Elo.

### Tier 0 — no code at all. Run these as A/B pairs tonight.

**1. Raise the training full-search cap from 160 to ~512, funded by fewer games.**
*Impl: 0 (flags). Compute: 2.6× per move ⇒ ~840 generations and ~320k games instead of 1900 and 730k.*
`--sims 512 --cap-sims 64 --cap-frac 0.25` gives an effective 152 evaluations per
move against today's 58. Arithmetic from the measured per-generation split
(self-play 1.719 s, learning 0.449 s): generations in 4183 s ≈ 4183 / (1.719k + 0.449):

| config | eff. sims/move | k | generations/hour | games/hour | positions recorded/hour |
|---|---:|---:|---:|---:|---:|
| **today** 160/24 @ 0.25 | 58 | 1.00 | 1900 | 730k | 24.3M |
| 512/32 @ 0.15 | 104 | 1.79 | ~1185 | ~455k | ~9.1M |
| **512/64 @ 0.25** | **152** | **2.62** | **~840** | **~320k** | **~10.6M** |
| 600/100 @ 0.25 (KataGo's own) | 225 | 3.88 | ~575 | ~220k | ~7.3M |
| 800/40 @ 0.25 (AlphaZero's N) | 230 | 3.97 | ~575 | ~220k | ~7.3M |

**State the trade plainly**: this buys a **3.2× deeper policy target** at the cost
of **2.3× fewer recorded positions**. That is exactly the tension playout cap
randomisation exists to manage, and KataGo resolved it in favour of the deeper
search — but at 4.2M games, not 0.7M.

*Evidence here*: our own sims sweep (`docs/ALGORITHM.md`) moved every behavioural
metric at constant generations; the tree is 3.10 plies deep with 3.2
meaningfully-visited root moves (§4.3); play-time strength is still rising at
98 Elo/doubling at 4096 sims (§5.2). *Evidence there*: KataGo states policy
learning wants "not far from AlphaZero's 800"; AlphaZero used 800.
*Risk*: Wang et al. argue the opposite for small games (§4.3); the value target is
a game result and does **not** get better with more simulations per move, so the
2.3× data cut lands entirely on the head that §5.2 identifies as the binding
constraint; and 840 generations is 840 PBT cycles instead of 1900. This is why it
is an A/B and not an assertion. *Contract*: untouched.

**2. Let the learning rate actually decay.** *Impl: 0. Compute: 0.*
The run finished at `lr = 1.0e-3`, a **2× total decay** (§4.4), against
`az_default_cfg`'s own 10× default and AlphaZero's 1000×. `--lr-final 2e-4`
(or 5e-5). Lc0 reports gains accelerate after each drop. This is free.

**3. Stop shipping and benchmarking the agent at 256 simulations.** *Impl: 0. Compute: 0 (it is play time, not training time).*
Measured: **5.8% → 20.0% against `material-1`** going from 256 to 4096 sims,
**+242 Elo**, at 42 ms/move. `src/api.c:157` caps it at 4096 and the curve is
steepest there (98 Elo per doubling over the last two). Separately: **put the node
count next to every published number.** "2.5% against Stockfish at one node"
compares our 256-node search against a 1-node search and reads as a much worse
result than it is.

**3b. Run the identical configuration for 8 hours instead of 1.** *Impl: 0 (`--generations`). Compute: 8 h.*
The blunder rate is still falling at generation 1894 and the two power-law fits
through the checkpoint series disagree by two orders of magnitude about how fast
(§5.2). One overnight run resolves that, and if the steeper fit is right it
roughly halves the blunder rate on its own. It is the cheapest way to find out
how much of §8 is really scale. Keep `--lr-final` proportional so the schedule
still completes.

### Tier 1 — a few lines, no compute cost.

**4. Make the search and the trainer agree about the value of a draw.**
*Impl: ~5 lines in `src/mcts.c` (a `draw_value` field, plumbed from `AZCfg.draw_penalty`). Compute: 0.*
Today `mcts.c` backs up 0.0 at every drawn terminal while the trainer labels the
game −0.6; mean *q* = +0.020 against mean *z* = −0.305 (§4.5). The search is
steering into the terminations the draw penalty exists to suppress: 16.4%
repetition + 15.4% insufficient + 12.5% ply-cap at generation 1900. *Contract*:
untouched — the penalty is symmetric and already sanctioned by
`docs/FROM_SCRATCH.md`; this makes the search obey the same symmetric rule.
*Measure*: repetition/insufficient termination rates and the `material-1` score,
not the draw rate alone.

**5. A growing replay window.** *Impl: ~15 lines. Compute: 0. Memory is the real constraint: `AZPos` is 104 B plus ≈28.3 × (`MoveKey` 5 B + target 4 B) ≈ 255 B in the two pools ≈ **360 B/position**, so 4.3M positions ≈ **1.55 GB** against `docs/ALGORITHM.md`'s 8 GB budget. Cap the formula at ~2M positions (≈0.7 GB) and it still holds 40× today's window.*
Ours is fixed at 50,000 ≈ 4 generations ≈ 1,500 games ≈ **0.2% of the run's data**.
KataGo: `N_window = c(1 + β((N_total/c)^α − 1)/α)`, `c = 250,000, α = 0.75,
β = 0.4` → **4.3M at our final data volume**. *The counter-evidence in
`az_default_cfg` deserves a direct answer*: a flat 400k buffer was measured to
cost "18 points of policy top-1 agreement" over 30 generations. Two problems with
that experiment. At 30 generations the whole run is 387k positions, so a 400k
buffer retains **everything, including generation-1 data from a random network** —
it is not KataGo's schedule, which would have used 301k of those 387k and grown
from there. And top-1 *agreement with the current search* mechanically rewards a
fresh buffer; it measures staleness, not strength. Re-run it against the blunder
rate and the `material-1` score.

**6. Normalise the two losses' contributions to the shared trunk.**
*Impl: ~20 lines; `az.c` already computes the decomposition for telemetry. Compute: 0.*
`grad_value_share` went from 0.337 at generation 1 to **0.905 at generation 1900**.
One scalar (`value_coef = 4.0`) is acting as a second, undeclared learning-rate
schedule on a trunk that is now 90% driven by the value loss. Target a fixed
share instead of a fixed coefficient, and log the clip-hit rate for both paths.

### Tier 2 — real work, worth it in this order.

**7. A WDL value head.** *Impl: ~80 lines + `MODEL_VERSION` bump. Compute: negligible.*
Three-way softmax + cross-entropy instead of scalar tanh + MSE. *Evidence*:
Czech et al. measured **+33.2 ± 19.0 Elo** for a WDLP head on a chess network;
Lc0 shipped it in July 2019 because the scalar head "had to assume a draw is ½ of
winning probability"; KataGo's value head has always been three-way. *Why it
matters here specifically*: 47.7% of our games are draws, and a scalar collapses
"certain draw" and "50/50 win-or-loss" onto the same number; it also removes
`draw_penalty` from the value scale entirely, which structurally fixes item 4
rather than patching it; and it gives the value loss a bounded, well-conditioned
gradient, which structurally fixes item 6. Sequenced *after* 4 and 6 because
those are 25 lines between them and tell you whether 7 is even needed.

**8. KataGo-style branching to off-policy moves.** *Impl: ~60 lines. Compute: ~5%.*
2.5% of positions branched to a random policy move with a full search recorded;
5% of games branched after r turns with 3–10 uniformly random moves. This is the
only item on the list that attacks the *coverage* cause of §5.2 rather than the
capacity cause: two agents that blunder at the same rate generate almost no data
in which a blunder is on the board and is punished. *Contract*: untouched —
random legal moves contain no chess opinion. *Measure*: the blunder rate on the
`--source vs` corpus, which is the one that tracks the match result.

**9. Forced playouts + policy target pruning.** *Impl: ~50 lines. Compute: 0.*
KataGo **1.25×**: force `n_forced(c) = (kP(c)ΣN)^½` playouts with k = 2, then
subtract them back out of the *target* before training. Sharpens the policy target
and increases exploration of low-prior moves — which is exactly the 1.48-visits-out-
of-160 problem in §5.2. Pairs naturally with item 1.

**10. Two repetition bits in the input.** *Impl: ~10 lines. Compute: ~0.*
The only representational gap against AlphaZero that is both legal and cheap.
16.4% of our games end by repetition and the network cannot see a repetition at
all. Small, certain, and it stops the value head being trained on draw labels it
has no feature to explain.

### Tier 3 — expensive, uncertain, or contract questions.

**11. More parameters.** *Impl: small (constants) — but it invalidates every checkpoint. Compute: quadratic in the accumulator width.*
The literature is unambiguous that we are far below compute-optimal (§4.1,
Neumann & Gros). `net.h`'s own 128/160/192/224 sweep found the quality curve flat
— over a 1.75× range, at fixed data, chosen to defend throughput. If item 1 is
run, run a 256-wide arm at the same time; they trade against the same budget and
you will learn which side of the frontier you are on. Do **not** do this before
items 1–6, because a wider trunk on the current gradient balance (90.5% value)
will just be more of the same trunk.

**12. Minimax or quiescence inside the search — the contract question, stated not decided.**
This is the direct fix for the measured failure, and I am not going to pretend
otherwise. Three versions, in increasing order of contract risk:

* **(a) A shallow full-width minimax over the *learned* value head** at selected
  nodes (Baier & Winands's MCTS-MS, with the network's *v* replacing their
  terminal-only backup). Contains no piece values, no move ordering, no
  chess-specific evaluation — it is the same class of object as PUCT. By the
  letter of `docs/FROM_SCRATCH.md` this is legal: the forbidden list is *chess
  evaluation*, and the evaluation here is the network's. Cost: ~120 lines, and
  `b` nodes per expansion instead of 1.
* **(b) A capture-only extension (quiescence) over the learned value head.**
  "Which moves are captures" is a fact of the rules, not an opinion — but *deciding
  that captures are the moves worth extending* is a hand-authored notion of which
  positions are noisy. That is the same species of judgement as MVV-LVA, which the
  contract bans by name. **Contract risk: high. Do not do this silently.**
* **(c) Attack maps as input features** ("which squares each side attacks", a pure
  rules predicate `chess.c` already computes for check detection). Cheap and
  powerful; but the useful version — "my piece stands on a square they attack" — is
  the literal definition of *hanging*, and that is evaluation. **Contract risk: high.**

My recommendation: **(a) is worth a prototype and (b)/(c) need an explicit,
documented amendment to `docs/FROM_SCRATCH.md` or nothing at all.** The contract's
value is that it is not negotiated retroactively when a measurement is
disappointing.

### The one change

**Item 1: raise the training-time full-search budget from 160 to 512 and pay for
it with fewer games — `--sims 512 --cap-sims 64 --cap-frac 0.25`.**

Because it costs *nothing to implement*, it is the lever this repository has
already measured to change behaviour more than any other (`docs/ALGORITHM.md`'s
48 → 256 sweep), it is the one place where our configuration is furthest from
both AlphaZero's and KataGo's considered choice (13.8× and 3.75×), the tree
statistics say the policy-improvement operator is barely improving anything
(3.10 plies, 3.2 visited root moves), and the play-time strength curve is still
climbing at 98 Elo per doubling where the engine's cap stops it. It is a
70-minute experiment with a 70-minute control and a single flag between them —
and if the 2.3× data cut wins instead, that result is just as useful, because it
says the value head is data-starved and points straight at items 5, 7 and 8.

Run item 2 (`--lr-final 2e-4`) in the same pair — it is also free, and orthogonal.

---

## 8. The honest possibility that this is just scale

It is, largely. The arithmetic in §2 is not close: **5 × 10⁷**. Three independent
readings agree:

1. **Extrapolation of our own curve** (§5.2): the blunder rate is falling as
   `games^−0.156` over all four checkpoints and `games^−0.445` over the last two.
   A 5% blunder rate is 875 hours of this laptop on the first fit and 6 hours on
   the second. The honest reading is that the exponent is unknown to within two
   orders of magnitude — **but both fits agree that one more hour of the same
   configuration is not enough**, and the cheaper of the two answers is an
   overnight run away (§7 item 3b).
2. **The best comparable evidence from DeepMind** (§5.3): AlphaZero's own
   ~2.3 × 10⁷-parameter network, trained on 44M games, solves **56.1%** of puzzles
   with no search and plays at 1777. The 9M-parameter supervised transformer —
   still 57× ours — reaches 88.9%. Nobody has demonstrated that a 1.6 × 10⁵-
   parameter flat MLP can do this job, and the scaling laws
   ([Neumann & Gros](https://arxiv.org/abs/2210.00849)) say larger models are
   *more* sample-efficient at equal data, not less.
3. **Nobody has published a from-scratch chess self-play result at our scale**
   (§5.4). The closest single-machine projects either never ran self-play to
   convergence, or budget "a couple of days" on an RTX 3090 evaluating a
   20×-larger network at ~10⁵ positions/second — and publish no verified number.

What is *not* scale, and is worth fixing regardless, is everything in Tiers 0 and
1: a search budget 13.8× below the two systems that studied the question, a replay
window holding 0.2% of the run's data, a learning rate that decayed 2× instead of
10×, a search and a trainer that disagree about what a draw is worth, and a shared
trunk whose gradient is 90.5% value loss. Those are not scale. Those are 40 lines
of code and three flags, and they are the difference between measuring the scale
limit and measuring our configuration.

The realistic target for the next hour is not beating `material-1`. It is moving
the blunder rate from 10.5% toward 7%, which on the measured ladder is worth on
the order of 100–200 Elo — and knowing, from a controlled A/B rather than an
argument, which of those items produced it.

---

## 9. Reproducing every number in this document

Build (nothing here modifies the repo):

```sh
make -j8
cc -O3 -std=c11 -D_DARWIN_C_SOURCE -Wall -Wextra -Wno-unused-parameter \
   -funroll-loops -fno-math-errno -ffp-contract=fast -mcpu=native -Isrc \
   tools/diag_tactics.c build/chess.o build/net.o build/mcts.o \
   -o build/diag_tactics -lm -lpthread
cp build/diag_tactics /tmp/dt        # macOS kills a running, relinked binary
```

| number | command |
|---|---|
| §0 run totals, §4.3–4.5 telemetry, the LR fit | `python3 - < runs/az_hour/telemetry.jsonl` — sum `sec`/`games`/`plies`/`evals`/`steps`, and fit `lr = lr_f + ½(lr₀−lr_f)(1+cos(π(g−1)/(G−1)))` |
| §0 parameter counts | compile a 3-line program that prints `TRUNK_NPARAM`, `HEAD_NPARAM` from `src/net.h` |
| §5.2 value / prior / sims / tree tables | `/tmp/dt all` ; `/tmp/dt sims --source self` ; `/tmp/dt sims --source vs` |
| §5.2 checkpoint trend | `for m in runs/{az,az_v2,az_v3,az_hour}/best.crl; do /tmp/dt compare --model $m --old runs/az_hour/best.crl; done` |
| §5.2 search-budget ladder | `for d in 1 4 16 64; do python3 py/benchmark.py --model runs/az_hour/best.crl --opponent material-1 --games 60 --depth $d --threads 8 --seed 4242 --no-md --quiet; done` (sims = depth × 64, clamped to 4096 by `API_MAX_SIMS`) |
| §4.5 draw-scale mismatch | mean of `mcts_root_value_mean`, `value_target_mean`, `search_mse_outcome_ratio`, `value_mse_outcome_ratio` over the last 200 telemetry lines |
| §2 FLOP estimate | 19 × 2 × 3×3×256×256 × 64 for AlphaZero; 35×128 + 128×128 + 35×32 + 32×32 + 128×32 + 28×32 for us; cross-check against 5000 TPUv1 × 9 h × 92 TOPS |

`--depth 128` and `--depth 64` return identical results: `src/api.c:139,157` clamp
`sims` to `[16, 4096]`, so 4096 is the highest budget the current play path can be
measured at.

---

## 10. Sources

**AlphaZero**
* Silver et al., *Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm*: <https://arxiv.org/abs/1712.01815> · full text <https://ar5iv.labs.arxiv.org/html/1712.01815> (700,000 steps × 4096; LR 0.2→0.02→0.002→0.0002; 800 simulations; 44M games; 119 planes, T=8, repetition planes; 4,672 policy outputs; α=0.3; 5,000 TPUv1 + 64 TPUv2; 9 h for chess; "does not augment the training data")
* Silver et al., *A general reinforcement learning algorithm that masters chess, shogi and Go through self-play*, Science 362 (2018): <https://gwern.net/doc/reinforcement-learning/model/alphago/2018-silver.pdf>
* AlphaZero pseudocode (`num_simulations` 800, `num_sampling_moves` 30, `root_dirichlet_alpha` 0.3, `root_exploration_fraction` 0.25, `pb_c_base` 19652, `pb_c_init` 1.25, `window_size` 1e6, `batch_size` 4096, `momentum` 0.9, `weight_decay` 1e-4, `max_moves` 512): <https://gist.github.com/erenon/cb42f6656e5e04e854e6f44a7ac54023>
* Silver et al., *Mastering the game of Go without human knowledge* (AlphaGo Zero; raw network 3055 Elo vs 5185 with search): <https://discovery.ucl.ac.uk/10045895/1/agz_unformatted_nature.pdf>
* Chess Programming Wiki, *AlphaZero* (19 residual blocks, 256 filters 3×3, 73 policy planes): <https://chessprogramming.org/AlphaZero>
* McGrath et al., *Acquisition of Chess Knowledge in AlphaZero*: <https://arxiv.org/pdf/2111.09259>

**KataGo**
* Wu, *Accelerating Self-Play Learning in Go*: <https://arxiv.org/abs/1902.10565> · full text <https://arxiv.org/html/1902.10565v5> (Table 2 ablations 1.37×/1.25×/1.60×/1.30×/1.65×/1.55×; playout cap randomisation p=0.25, (N,n)=(600,100)→(1000,200); forced playouts `n_forced=(kP·ΣN)^½`, k=2; window formula c=250,000, α=0.75, β=0.4; 6×96→10×128→15×192→20×256; batch 256; SGD momentum 0.9; 4.2M games, 241M samples, ≤28 V100, 19 days)
* *KataGo methods* (subtree value bias +30…60 Elo; optimistic policy +40…90 Elo; uncertainty-weighted playouts ≈+75 Elo): <https://github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md>

**Leela Chess Zero**
* Training runs (old-main 192×15, test10 256×20, test20, test30): <https://lczero.org/dev/wiki/training-runs/>
* *Lc0 training* (test10 trained on games 15M–67M; LR drops; self-play Elo vs CCRL): <https://lczero.org/blog/2018/10/lc0-training/>
* *Win-Draw-Loss evaluation* (WDL head, networks from July 2019): <https://lczero.org/blog/2020/04/wdl-head/>
* Network topology / SE blocks: <https://lczero.org/dev/backend/nn/>
* Technical explanation (tree reuse, NNCache): <https://lczero.org/dev/wiki/technical-explanation-of-leela-chess-zero/>

**Tactical blindness and search traps**
* Ramanujan, Sabharwal & Selman, *On Adversarial Search Spaces and Sampling-Based Planning*, ICAPS 2010: <https://www.cs.cornell.edu/~raghu/Raghuram_Ramanujan_files/icaps10.pdf>
* Baier & Winands, *MCTS-Minimax Hybrids*, IEEE TCIAIG: <https://dke.maastrichtuniversity.nl/m.winands/documents/mcts-minimax_hybrids_final.pdf>
* Baier & Winands, *MCTS-Minimax Hybrids with State Evaluations*, JAIR 2018: <https://www.ijcai.org/proceedings/2018/0782.pdf>
* Czech, Korus & Kersting, *Monte-Carlo Graph Search for AlphaZero* (+69 Elo chess, +310 crazyhouse): <https://arxiv.org/abs/2012.11045>

**Scale, representation and small-scale reimplementations**
* Ruoss et al., *Grandmaster-Level Chess Without Search* / *Amortized Planning with Large-Scale Transformers*: <https://arxiv.org/abs/2402.04494>
* Neumann & Gros, *Scaling Laws for a Multi-Agent Reinforcement Learning Model*: <https://arxiv.org/abs/2210.00849>
* Neumann & Gros, *AlphaZero Neural Scaling and Zipf's Law*: <https://arxiv.org/pdf/2412.11979>
* Czech et al., *Representation Matters* (+96.7 ± 30.4 Elo input, +33.2 ± 19.0 Elo WDLP): <https://arxiv.org/abs/2304.14918>
* Wang, Emmerich, Preuss & Plaat, *Analysis of Hyper-Parameters for Small Games: Iterations or Epochs in Self-Play?*: <https://arxiv.org/pdf/2003.05988>
* Wu et al., *Accelerating and Improving AlphaZero Using Population Based Training* (PBT on LR and value-loss ratio; 47% → 74% vs ELF OpenGo on 19×19): <https://arxiv.org/pdf/2003.06212>
* Zeta36, *chess-alpha-zero* README: <https://github.com/Zeta36/chess-alpha-zero/blob/master/readme.md>
* yurit04, *alpha_chess* README: <https://github.com/yurit04/alpha_chess>
* Willemsen, Baier & Kaisers, *Value targets in off-policy AlphaZero: a new greedy backup*: <https://link.springer.com/article/10.1007/s00521-021-05928-5>
