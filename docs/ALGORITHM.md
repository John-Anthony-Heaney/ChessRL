# Training algorithm

## Why this shape

The hard constraint is throughput: **>1000 complete games/second on an 8-core M3
with 8 GB of RAM**, ~120k position evaluations/second. Everything below follows
from that budget.

* **Sparse first layer.** 788 binary inputs, ~35 ever active, so `W0 @ x` is a
  gather-and-sum of ~35 rows (5.6k MACs) instead of a 126k-MAC matmul.
* **No tree search during training.** Moves are sampled straight from the policy
  head. Search is reserved for the human-vs-champion match, where it costs
  nothing to train.
* **One shared trunk, 256 private heads.** A fully independent 190k-parameter
  network per agent would give each agent only ~1600 positions per generation
  and would need ~600 MB of optimiser state. Sharing the trunk gives it every
  position played by the whole population (tens of millions), while each agent
  still owns its policy head, value head and a style vector `z` added into the
  accumulator — enough for agents to be genuinely different opponents and for
  evolution to have something real to select on.

## Per-generation loop

```
for g in 1..G:
    1. PAIR      Swiss-like pairing by Elo + a slice of hall-of-fame opponents
    2. PLAY      threads play the pairings, recording trajectories
    3. LEARN     A2C update: per-agent heads + the shared trunk
    4. RATE      update Elo from results (hall-of-fame ratings are frozen)
    5. EVOLVE    cull the bottom, clone+mutate from the elite, perturb hypers
    6. LOG       append one JSON line of telemetry; checkpoint periodically
```

## Two-player credit assignment

Each player's decisions form its own MDP. For agent playing colour `c`, its
decision points are the plies `t` where `side_to_move == c`; between its
decision at `t` and its next one at `t'` (usually `t+2`) the opponent moved.

* `Φ(s)` = potential = `tanh(material_balance_white(s) / 5)`, from **White's**
  view; `Φ_c = Φ` for White, `-Φ` for Black.
* Shaping reward for step `t -> t'`:  `r_t = shaping * (γ·Φ_c(s_t') - Φ_c(s_t))`.
  This is potential-based shaping (Ng, Harada & Russell 1999): it provably does
  not change the optimal policy, it only removes the sparsity of the win/loss
  signal. Without it, nothing learns within the compute budget.
* Terminal reward: `+1` win, `-1` loss, `0` draw, from `c`'s view, added at the
  last decision of the game. A draw by the ply cap or by repetition when
  material is clearly winning gets a small negative nudge (`-0.05`) so agents do
  not learn to shuffle.
* The value head `v(s)` predicts the discounted return for the **side to move**.
* Advantages via GAE(λ) along the player's own subsequence.

## Loss (per decision point)

```
L = -A_t · log π(a_t | s_t)  -  β · H(π(·|s_t))  +  c_v · (v(s_t) - R_t)^2
```
`β` = `entropy_coef`, `c_v` = `value_coef`, both per-agent PBT hyper-parameters.
Advantages are normalised (zero mean, unit variance) per generation.

Gradients accumulate into per-agent head buffers (spin-lock protected) and a
per-thread trunk buffer that is reduced at the end of the generation. One Adam
step per generation for the trunk, one per agent for each head, with
`lr = base_lr * agent.lr_scale`. Global grad-norm clipping.

## Elo

Standard logistic Elo, `K = 24`, seeded at 1500. Hall-of-fame agents (frozen
snapshots taken every `hof_every` generations) never have their rating updated,
which anchors the scale so the population's rating shows real absolute progress
rather than drifting with the mean.

## Evolution (PBT)

Sort by Elo. The bottom `cull_frac` are each replaced by a copy of a uniformly
chosen agent from the top `elite_frac`, with:
* head weights perturbed by `N(0, mutate_sigma)` scaled per-tensor by that
  tensor's RMS,
* every hyper-parameter multiplied by `exp(N(0, 0.2))` and clamped to its range.

The trunk is never mutated — it is the population's shared, purely
gradient-trained representation.

## Telemetry (one JSON line per generation, `runs/<name>/telemetry.jsonl`)

`gen, games, plies, sec, gps (games/sec), elo_best, elo_mean, elo_p10,
white_win, black_win, draw, avg_len, captures_per_game, checks_per_game,
castle_rate, promo_rate, ep_rate, term{checkmate,stalemate,fifty,repetition,
insufficient,maxplies}, opening_top[8], piece_dest_entropy, avg_final_material,
best_agent{i,elo,temperature,entropy_coef,lr_scale,shaping}, loss{policy,value,
entropy}, grad_norm`

`py/report.py` turns this file into the strategy analysis.

---

# AlphaZero rebuild (supersedes the A2C sections above)

The A2C trainer and the hand-evaluated alpha-beta search were removed because
they violated the from-scratch requirement in `docs/FROM_SCRATCH.md`. See
`src/mcts.c` and `src/az.c`. The reward is now only the game result; the policy
target is the MCTS visit distribution; positions go through a replay buffer with
hundreds of minibatch steps per generation instead of one step per generation.

## Tuning: repetition draws are a search problem, not a reward problem

A 40-generation pilot drifted into 74.5% threefold-repetition draws, with
captures collapsing from 14.2 to 5.2 per game. The agents had learned to shuffle.

The natural reading is that a draw is too cheap, so the draw penalty was swept
against the simulation count (64 agents, 20 generations, mixed 960 starts):

| sims | draw penalty | checkmate | repetition | captures/game | policy KL |
|---:|---:|---:|---:|---:|---:|
|  48 | -0.1 | 0.201 | 0.599 |  7.6 | 0.527 |
|  48 | -0.3 | 0.191 | 0.616 |  7.4 | 0.523 |
|  48 | -0.6 | 0.227 | 0.516 |  8.5 | 0.540 |
|  96 | -0.1 | 0.286 | 0.275 | 10.8 | 0.478 |
|  96 | -0.3 | 0.325 | 0.244 | 11.7 | 0.459 |
|  96 | -0.6 | 0.347 | 0.177 | 14.4 | 0.421 |
| 160 | -0.6 | 0.423 | 0.053 | 17.9 | 0.427 |
| 256 | -0.6 | 0.451 | 0.027 | 18.7 | 0.379 |

Tripling the draw penalty at 48 sims barely moves repetition (0.599 -> 0.516).
Tripling the simulation count at a fixed penalty nearly eliminates it
(0.599 -> 0.053). **The shuffling was the search failing to find a plan, not the
reward failing to discourage a draw.** A policy that cannot see progress repeats,
and no reward shaping fixes that -- only more search does.

This is worth remembering because the first instinct, both times the population
drifted to draws, was to change the reward. The first time (material shaping in
the A2C trainer) that instinct also produced an agent that had learned nothing.

Throughput falls roughly linearly in the simulation count, so 160 sits at the
knee: repetition is already down to 5% and the marginal gain to 256 costs a
third of the games.

## Telemetry to watch

`policy_kl` is the learning curve, NOT the raw policy cross-entropy. CE is
`H(pi) + KL(pi || p)` and the MCTS target's own entropy `H(pi)` drifts as the
search sharpens, so raw CE can rise while the network is improving.
`policy_top1` (how often the network's argmax matches the search's choice) and
`value_accuracy_decisive` are the other two that matter.
