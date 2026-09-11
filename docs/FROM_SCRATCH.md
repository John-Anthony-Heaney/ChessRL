# From scratch: what is learned and what is not

The agent must derive its chess ability entirely from self-play. This document
is the contract, and the audit checklist that enforces it.

## Allowed: the rules of chess

`src/chess.c` encodes the rules — which moves are legal, and whether a position
is checkmate, stalemate, a threefold repetition, a fifty-move draw, or a dead
position. These are the definition of the game, not an opinion about how to play
it. A random-move player uses exactly the same facts.

## Allowed: a general-purpose search algorithm

PUCT Monte-Carlo tree search (`src/mcts.c`) is domain-independent. It contains
no chess: it takes priors and leaf values from the network and does bandit
arithmetic over them. Swap in a different game's rules and network and it works
unchanged. This is the same sense in which AlphaZero is "from scratch".

## Forbidden: every form of chess evaluation

Nothing outside `chess.c` may contain, and no learning signal may depend on:

- piece values (Q=9, R=5, …) or any material count
- piece-square tables, centre-control bonuses, development or tempo terms
- mobility, king safety, pawn-structure or passed-pawn terms
- MVV-LVA, static exchange evaluation, killer moves, history heuristics
- opening books, endgame tablebases, or any human-authored opening preference
- reward shaping of any kind derived from the position rather than the result

## What the agent is trained on

The only reward is the game result: `+1` win, `-1` loss, `draw_penalty` for a
draw. The value head regresses that outcome. The policy head is trained to match
the MCTS visit distribution, which is the search's improvement on the policy.

The **draw penalty** is symmetric — both sides receive it — so it expresses no
view about chess. It only prevents the population from settling into the
shuffling equilibrium that a zero-valued draw permits. It is a statement about
learning dynamics in symmetric self-play, not about the game.

## Explicitly removed in this rebuild

| Removed | Where it was | Why it violated the contract |
|---|---|---|
| `Φ = tanh(material/5)` potential shaping | `train.c`, `arena.c` | injected piece values into the reward |
| material blend in the leaf evaluation | `search.c` | the hand-written term, not the network, chose moves |
| MVV-LVA / SEE move ordering | `search.c` | hand-authored notion of a good capture |
| the alpha-beta searcher as the shipped agent | `search.c` | replaced by MCTS over learned priors alone |

`search.c` is retained ONLY as a measurement baseline, never as the agent, and
is excluded from the shipped play path.

## How the contract is enforced

`tests/test_scratch.c` and `tools/audit_knowledge.sh` fail the build if the
learning or play path references material values or any of the forbidden terms.
The strongest empirical check: a network with **randomly initialised weights**
must play no better than random. If untrained weights produce a competent
player, knowledge has leaked in from somewhere other than training.
