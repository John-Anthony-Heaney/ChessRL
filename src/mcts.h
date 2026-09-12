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
 *
 * Everything added for speed -- subtree reuse, the evaluation cache, first-play
 * urgency -- is bookkeeping over the same two sources. None of it can express a
 * preference for a chess position that the network did not already express.
 */
#ifndef MCTS_H
#define MCTS_H

#include <stdint.h>
#include "chess.h"
#include "net.h"

/* Hard cap on the length of a selection path.  A tree this deep is already
 * pathological; the cap exists so the path arrays are fixed-size and the search
 * provably terminates even if repetition cycles are searched.  Overridable at
 * compile time only so that the cap's own code path can be exercised: real
 * searches run 3 to 11 plies deep, so 128 is never reached. */
#ifndef MCTS_MAX_DEPTH
#define MCTS_MAX_DEPTH   128
#endif
/* How much game history is kept for repetition detection.  The fifty-move rule
 * bounds the useful window at 100 plies. */
#define MCTS_HIST_KEEP   128

/* One node per (state, action) edge, AlphaZero style: the edge statistics live
 * on the child. Children of a node are contiguous in the pool, and a node's
 * children always have HIGHER pool indices than the node itself (they are
 * allocated when it is expanded, which is after it was allocated). Subtree
 * reuse depends on that ordering; see mcts.c. */
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

/* The network-evaluation cache. Opaque: its layout is mcts.c's business. */
struct MctsCache;
/* The resumable state of one in-progress search. Opaque for the same reason:
 * it is the search's own program counter and stack, and nothing outside
 * mcts.c has any business reading it. */
struct MctsRun;

typedef struct {
    MctsNode *pool;
    int       cap;
    int       used;

    /* search parameters -- all generic, none of them chess-specific */
    float     c_puct;           /* exploration constant, default 1.4        */
    float     fpu;              /* first-play urgency for unvisited edges   */
    /* First-play urgency REDUCTION (Leela/KataGo).  When > 0 an unvisited edge
     * is valued at  Q(parent) - fpu_reduction * sqrt(visited policy mass)
     * instead of the flat `fpu`, so a node whose explored moves already look
     * good is less eager to try the rest.  0 (the default) reproduces the flat
     * `fpu` exactly.  Generic bandit arithmetic: it reads visit counts and
     * priors, never the position. */
    float     fpu_reduction;
    float     dirichlet_alpha;  /* root noise concentration, default 0.3    */
    float     dirichlet_eps;    /* root noise weight, default 0.25          */

    /* SUBTREE REUSE.  1 (default) lets a search inherit the subtree the
     * previous search built under the move that was actually played, instead of
     * rebuilding from scratch.  It is safe to leave on: the search verifies the
     * full position, the legal move list and the repetition window before it
     * will touch a stale tree, and falls back to a fresh tree whenever any of
     * those disagree.  Set to 0 to force a fresh tree every move. */
    int       reuse;
    /* EVALUATION CACHE.  1 (default) serves a leaf evaluation from the cache
     * when the same network input has been seen under the same weights.  A hit
     * is bit-for-bit what the network would have returned, so turning this off
     * changes speed and nothing else.  Set to 0 to bypass it. */
    int       cache;
    /* Allocated cache size, as a power of two entries; 0 when there is none.
     * Chosen from the node pool at mcts_init() and read-only afterwards. */
    int       cache_bits;

    /* stats for telemetry (plain fields: zero them when you want per-move
     * numbers -- nothing inside the search reads them) */
    uint64_t  evals;            /* network forward passes actually performed */
    int       max_depth_seen;
    uint64_t  pool_exhausted;   /* expansions refused, pool was full         */
    uint64_t  cache_hits;       /* evaluations served from the cache         */
    uint64_t  cache_misses;
    uint64_t  cache_flushes;    /* invalidations because the weights changed */
    uint64_t  reuse_hits;       /* searches that inherited a subtree         */
    uint64_t  reuse_misses;     /* searches that had a tree but rebuilt      */
    uint64_t  reuse_nodes;      /* nodes carried over, cumulative            */

    /* ---------------- private: owned by mcts.c, do not poke ------------- */
    struct MctsCache *ecache;   /* the cache itself; `cache` above switches it */
    int8_t    owns_cache;       /* 0 when ecache is shared and borrowed       */
    struct MctsRun   *run;      /* the stepped search's resumable state       */
    int32_t  *remap;            /* cap ints, scratch for subtree compaction  */
    uint64_t  wstamp;           /* fingerprint of (trunk, head) + weights    */
    Position  rpos;             /* position the standing tree is rooted at   */
    uint64_t  rkeys[MCTS_HIST_KEEP + 4];  /* its repetition window           */
    int       nrkeys;
    int       rply;             /* Game.ply it was rooted at                 */
    int8_t    tree_valid;       /* a standing tree is present                */
    int8_t    tree_noised;      /* its root priors carry Dirichlet noise     */
    int8_t    tree_capped;      /* it contains a node adjudicated at the cap */
} Mcts;

void mcts_init(Mcts *m, int max_nodes);
void mcts_free(Mcts *m);
void mcts_defaults(Mcts *m);

/* Runs `sims` simulations from g->pos and writes the root children's visit
 * counts into visits[] (parallel to the legal move list gen_legal() produces
 * for that position). Returns the number of legal moves, or 0 if the game is
 * already over. `root_value` receives the root's mean value, side-to-move view.
 * When root_noise is non-zero, Dirichlet noise is mixed into the root priors,
 * which is the exploration mechanism during self-play.
 *
 * `sims` is a TOTAL, not an increment: whether the tree is fresh or inherited,
 * the search returns with sum(visits) == sims and root.N == sims + 1. Reuse
 * therefore makes a move cheaper rather than making the tree bigger, and every
 * caller's arithmetic over visits[] is unchanged by it. */
int mcts_search(Mcts *m, const Trunk *t, const Head *h, const Game *g,
                int sims, int root_noise, uint64_t *rng,
                int32_t *visits, float *root_value);

/* ==========================================================================
 *              STEPPED SEARCH -- many independent searches at once
 * ==========================================================================
 * mcts_search() is one blocking call because MCTS is strictly sequential: a
 * simulation cannot choose its next path until the previous leaf's value is
 * known.  Every network evaluation is therefore a batch of ONE, which for a
 * 128x128 trunk is a matrix-VECTOR product and cannot use the machine's width.
 *
 * The cure is not to parallelise one search -- that needs virtual loss and
 * changes what is computed -- but to have many INDEPENDENT searches in flight
 * and evaluate their pending leaves together.  Independent searches share
 * nothing, so there is NO approximation here at all: each search performs
 * exactly the simulations it would have performed alone, in exactly the same
 * order, off exactly the same values.
 *
 * The API turns one search inside out:
 *
 *     if (!mcts_begin(m, t, h, g, sims, noise, rng)) -> nothing to search
 *     while (mcts_step(m)) {                 1 = a leaf is waiting for the net
 *         const uint16_t *f = mcts_pending(m, &nf);
 *         ... evaluate f/nf against h, in a batch with other searches ...
 *         mcts_deliver(m, &fw);
 *     }
 *     n = mcts_end(m, visits, &root_value);
 *
 * and mcts_search() is exactly that loop with a batch size of one, so the two
 * cannot drift apart.  `m` may not be touched by anything else between begin
 * and end; each concurrent search needs its own Mcts (its own node pool, its
 * own repetition window, its own standing tree).  The evaluation CACHE may be
 * shared -- see mcts_cache_attach(). */

/* 1 when a search was started, 0 when there is nothing to search (bad
 * arguments, or the game is already over).  Does not evaluate anything. */
int  mcts_begin(Mcts *m, const Trunk *t, const Head *h, const Game *g,
                int sims, int root_noise, uint64_t *rng);
/* Runs the search until it needs a network evaluation or its budget is spent.
 * 1 = a leaf is pending (call mcts_pending then mcts_deliver); 0 = finished. */
int  mcts_step(Mcts *m);
/* The pending leaf's active features.  Valid only between a mcts_step() that
 * returned 1 and the matching mcts_deliver().  The head to evaluate against is
 * the one passed to mcts_begin(). */
const uint16_t *mcts_pending(const Mcts *m, int *nf);
/* Hands back the evaluation of the pending leaf.  Only fw->q and fw->v are
 * read, which is exactly what the non-stepped search uses. */
void mcts_deliver(Mcts *m, const Fwd *fw);
/* Ends the search: same return value and same outputs as mcts_search(). */
int  mcts_end(Mcts *m, int32_t *visits, float *root_value);

/* ------------------------------------------------------ the batch driver */
/* Scratch for driving several stepped searches together: the pointer arrays
 * nn_eval_batch() wants, plus the Fwd rows it writes.  One per THREAD -- it is
 * not itself thread-safe and does not need to be. */
typedef struct MctsBatch MctsBatch;

MctsBatch *mcts_batch_create(int cap);
void       mcts_batch_free(MctsBatch *b);
int        mcts_batch_cap(const MctsBatch *b);
int        mcts_batch_count(const MctsBatch *b);
/* Queues the leaf `m` is waiting on.  `m` must have just returned 1 from
 * mcts_step(), and `h` must be the head it was begun with. */
void       mcts_batch_add(MctsBatch *b, Mcts *m, const Head *h);
/* Evaluates every queued leaf in ONE nn_eval_batch() call, delivers each value
 * to the search that asked for it, and empties the queue.  Returns how many
 * rows were evaluated. */
int        mcts_batch_run(MctsBatch *b, const Trunk *t);

/* Convenience wrapper: runs `n` independent searches to completion together,
 * one batched evaluation per round.  Every array is indexed by search.
 * `noise[i]`, `rng[i]`, `visits[i]` and `nout[i]` may each be NULL as a whole;
 * `rng[i]` may also be NULL per search when that search takes no noise.
 * nout[i] receives what mcts_search() would have returned for search i. */
void mcts_search_many(int n, Mcts *const *ms, const Trunk *t,
                      const Head *const *hs, const Game *const *gs,
                      const int *sims, const int *noise, uint64_t *const *rng,
                      int32_t *const *visits, float *root_value, int *nout,
                      MctsBatch *b);

/* -------------------------------------------------------- shared cache */
/* The evaluation cache is a pure function of (network input, trunk, head,
 * weights) -- all four are in the key -- so several searches may share one,
 * and a hit is bit-for-bit what the network would have returned for THAT
 * search.  Sharing is what makes G concurrent games per thread affordable: one
 * table per thread instead of G of them.  Every Mcts attached to a cache must
 * be driven by ONE thread; the table takes no lock. */
struct MctsCache *mcts_cache_create(int bits);
void              mcts_cache_destroy(struct MctsCache *c);
/* Drops m's own cache and borrows `c` (which m will not free).  Pass NULL to
 * leave m with no cache at all. */
void              mcts_cache_attach(Mcts *m, struct MctsCache *c);

/* Number of expansions that could not be performed because the node pool was
 * full, cumulative since mcts_init(). Non-zero means the search was degraded
 * (it kept evaluating but stopped growing the tree): raise max_nodes. */
uint64_t mcts_pool_exhausted(const Mcts *m);

/* Drops every cached evaluation and the standing tree. The search does this by
 * itself whenever it detects that the weights or the agent changed, so callers
 * need it only when they mutate weights through some path this cannot see. */
void mcts_cache_clear(Mcts *m);

/* Samples a move index from visit counts at temperature `temp`.
 * temp <= 0 selects the argmax. */
int mcts_pick(const int32_t *visits, int n, float temp, uint64_t *rng);

/* The policy TARGET used for training: visit counts normalised to sum to 1. */
void mcts_target(const int32_t *visits, int n, float *out);

#endif /* MCTS_H */
