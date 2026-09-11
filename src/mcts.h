/* mcts.h -- PUCT Monte-Carlo tree search, AlphaZero style.
 *
 * ZERO CHESS KNOWLEDGE. This file and its implementation must not contain
 * piece values, piece-square tables, mobility terms, MVV-LVA, SEE, killer
 * moves, or any other chess heuristic. The ONLY sources of evaluation are:
 *   - the learned policy head, which supplies the priors P(s,a), and
 *   - the learned value head, which supplies the leaf value V(s).
 * The only non-learned facts it may use are the rules of chess themselves:
 * which moves are legal, and whether a position is checkmate, stalemate or a
 * drawn position. Those come from chess.c and are rules, not evaluation.
 *
 * The search itself is a general-purpose algorithm. That is what makes the
 * agent "learned from scratch": the knowledge is learned, the search is not
 * domain-specific.
 */
#ifndef MCTS_H
#define MCTS_H

#include <stdint.h>
#include "chess.h"
#include "net.h"

/* One node per (state, action) edge, AlphaZero style: the edge statistics live
 * on the child. Children of a node are contiguous in the pool. */
typedef struct {
    float   P;        /* prior probability from the policy head              */
    float   W;        /* summed value, from THIS node's side-to-move view    */
    int32_t N;        /* visit count                                         */
    int32_t first;    /* pool index of the first child, -1 while unexpanded  */
    int16_t nchild;
    int8_t  terminal; /* 0 = not terminal, else TR_* reason                  */
    float   tval;     /* terminal value from this node's mover's view        */
    Move    move;     /* the move that reached this node                     */
} MctsNode;

typedef struct {
    MctsNode *pool;
    int       cap;
    int       used;

    /* search parameters -- all generic, none of them chess-specific */
    float     c_puct;           /* exploration constant, default 1.4        */
    float     fpu;              /* first-play urgency for unvisited edges   */
    float     dirichlet_alpha;  /* root noise concentration, default 0.3    */
    float     dirichlet_eps;    /* root noise weight, default 0.25          */

    /* stats for telemetry */
    uint64_t  evals;            /* network evaluations performed            */
    int       max_depth_seen;
} Mcts;

void mcts_init(Mcts *m, int max_nodes);
void mcts_free(Mcts *m);
void mcts_defaults(Mcts *m);

/* Runs `sims` simulations from g->pos and writes the root children's visit
 * counts into visits[] (parallel to the legal move list gen_legal() produces
 * for that position). Returns the number of legal moves, or 0 if the game is
 * already over. `root_value` receives the root's mean value, side-to-move view.
 * When root_noise is non-zero, Dirichlet noise is mixed into the root priors,
 * which is the exploration mechanism during self-play. */
int mcts_search(Mcts *m, const Trunk *t, const Head *h, const Game *g,
                int sims, int root_noise, uint64_t *rng,
                int32_t *visits, float *root_value);

/* Samples a move index from visit counts at temperature `temp`.
 * temp <= 0 selects the argmax. */
int mcts_pick(const int32_t *visits, int n, float temp, uint64_t *rng);

/* The policy TARGET used for training: visit counts normalised to sum to 1. */
void mcts_target(const int32_t *visits, int n, float *out);

#endif /* MCTS_H */
