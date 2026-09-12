/* mcts.c -- PUCT Monte-Carlo tree search, AlphaZero style.
 *
 * ZERO CHESS KNOWLEDGE.  Grep this file for a piece value, a square table, a
 * mobility term, MVV-LVA, SEE, a killer move or an opening book and you will
 * find none.  Every number that influences a decision comes from exactly two
 * places:
 *
 *   1. the learned network  -- P(s,a) from the policy head, V(s) from the value
 *      head.  These are the ONLY evaluations in the file.
 *   2. the rules of chess   -- gen_legal(), in_check(), insufficient_material(),
 *      the fifty-move counter and repetition counting.  These are the definition
 *      of the game (a random mover uses the same facts), not an opinion about
 *      how to play it.
 *
 * The arithmetic in between -- selection, expansion, backup -- is the ordinary
 * domain-independent PUCT bandit recursion.  Swap chess.c for another game's
 * rules and net.c for another game's network and this file is unchanged.  The
 * three speed features below are bookkeeping over the same two sources: reuse
 * keeps statistics the network already produced, the cache remembers what the
 * network already said, and first-play urgency is bandit arithmetic over visit
 * counts and priors.  None of them can prefer a position the network did not.
 *
 * ---------------------------------------------------------------- conventions
 *
 * SIGN.  Every node's W is accumulated from the perspective of the side to move
 * AT THAT NODE.  A child node is the opponent's node, so the parent's Q for the
 * edge into it is  Q(s,a) = -child.W / child.N.  Backup negates the value at
 * every ply.  Getting this wrong produces an agent that plays for its opponent,
 * so tests/test_mcts.c checks it from both colours.
 *
 * VISIT CONVENTION.  The root is expanded once, outside the simulation loop,
 * and that expansion sets root.N = 1.  Each simulation then descends from the
 * root through exactly one root child.  Therefore
 *
 *      sum over root children of N  ==  sims        (exactly)
 *      root.N                       ==  sims + 1
 *
 * `sims` is a TOTAL, and that is what makes subtree reuse invisible to callers:
 * an inherited subtree arrives with N visits already in it and the search tops
 * it up to sims + 1 rather than adding sims more.  A move gets cheaper; the
 * visit distribution keeps meaning exactly what it meant before.
 *
 * REPETITION.  Threefold repetition needs the game history AND the moves made
 * inside the search.  One array holds both: the tail of g->hist back to the last
 * irreversible move (exactly the window game_repetitions() uses -- nothing older
 * can repeat, because a capture or pawn move can never be undone) followed by
 * the key of every position pushed along the current selection path.  The count
 * steps back two plies at a time, like game_repetitions(), since only positions
 * with the same side to move can be repetitions.
 *
 * ALLOCATION.  The hot path performs no malloc.  Nodes come from a preallocated
 * pool; scratch buffers are stack locals.  One Position is copied per search --
 * never per node -- and it is maintained incrementally with make_move /
 * unmake_move along the selection path.  gen_legal() is called exactly once per
 * expanded node; the resulting moves live in the child nodes from then on.
 *
 * ------------------------------------------------------------- subtree reuse
 *
 * After a move is played the subtree under it is already a tree for the new
 * position, so the next search inherits it instead of paying for it again.  The
 * danger is using a stale tree for the wrong position, so a reuse happens only
 * when ALL of these agree:
 *
 *   - the standing tree was built by the same trunk AND the same head, with the
 *     same weights (mc_weight_stamp);
 *   - the caller's Game advanced by 0, 1 or 2 plies and every one of those moves
 *     is an edge of the standing tree;
 *   - replaying those edges reproduces g->pos EXACTLY -- not just the zobrist
 *     key but the side, castling rights, en-passant square, halfmove clock,
 *     piece placement and the Chess960 rook files (mc_same_pos);
 *   - the repetition window the subtree was built under, extended by the moves
 *     played, is the window the new search would compute.  Otherwise a cached
 *     TR_REPETITION could be a fact about a different game;
 *   - gen_legal() for the new position is exactly the child list, in order,
 *     because the caller's visits[] is parallel to it;
 *   - the subtree holds no more than sims + 1 visits and its children's visits
 *     still add up to its own minus one, so sum(visits) == sims holds exactly
 *     (an exhausted pool can break that second part: a node visited while the
 *     pool was full is counted but descends nowhere);
 *   - no node in it was adjudicated at the ply cap, which is measured from the
 *     root and so would not survive a re-rooting.
 *
 * Together those make a strong guarantee: WITH THE ROOT NOISE OFF an inherited
 * tree is bit-for-bit the tree a fresh search of the same budget would have
 * built -- not approximately, exactly.  A simulation that descends into child C
 * and carries on applies the same PUCT rule to C's own statistics that a search
 * rooted at C would, so after N visits C's subtree IS a root-at-C tree of N-1
 * simulations; `sims` is a total, so topping up finishes that same sequence, and
 * no simulation reads the rng.  With the noise ON that is not true and is not
 * meant to be: a fresh search noises the root priors before its first
 * simulation and an inherited subtree was built without them.  That is what
 * every AlphaZero implementation does, but a caller who wants the strict
 * guarantee during self-play sets reuse = 0.
 *
 * Any disagreement rebuilds from scratch.  Dirichlet noise never survives: it
 * is applied only to the root's direct children, and the new root is a child or
 * grandchild whose own children were never touched by it.  Re-rooting at the
 * SAME position (what a time-capped caller's 32/64/128 ramp does) is allowed
 * only when no noise is involved in either direction, and is then exactly
 * equivalent to a fresh search of the larger budget -- MCTS is incremental, and
 * with the noise off nothing in a simulation reads the rng.
 *
 * -------------------------------------------------------- evaluation cache
 *
 * Positions repeat inside one tree (transpositions) and across moves, so a
 * small direct-mapped cache holds what the network said about them.  It stores
 * the policy query vector q and the value v -- everything nn_logits() and the
 * backup need -- which is 33 floats rather than a whole Fwd, and is bit-for-bit
 * what a fresh nn_eval() would have produced.  The key is NOT the bare zobrist
 * key: that covers the pieces, the castling mask, the en-passant file and the
 * side, but the network's input also has a halfmove-clock bucket, and in
 * Chess960 the legal move list depends on the rook origin files.  Both are
 * folded in -- and so is the WEIGHT STAMP, so an entry names (input, trunk,
 * head, weights) rather than a position.  That is what lets several concurrent
 * searches share one table while playing different agents: an entry made under
 * other weights has a different tag and simply cannot be hit, which is a
 * stronger guarantee than the flush it replaces and costs no bookkeeping at
 * all.  mcts_cache_clear() remains for a caller who mutates weights by a route
 * the stamp cannot see; it is a generation bump, so it is still O(1).
 *
 * ------------------------------------------------- many searches at once
 *
 * MCTS is strictly sequential -- a simulation cannot choose its next path until
 * the previous leaf has been evaluated -- so a lone search hands the network
 * one position at a time and every evaluation is a batch of one.  The search is
 * therefore written as a resumable state machine (mcts_begin / mcts_step /
 * mcts_deliver / mcts_end) that STOPS at a leaf instead of blocking on it, so a
 * driver can collect one pending leaf from each of many INDEPENDENT searches
 * and evaluate them all in one nn_eval_batch().  Independent searches share no
 * tree and no pool, so this needs no virtual loss and introduces no
 * approximation: each performs exactly the simulations it would have performed
 * alone.  mcts_search() is that machine driven one leaf at a time, so there is
 * only one implementation of the search and the two paths cannot drift.
 */

#include "mcts.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* The pool floor: enough for the root plus a full legal move list, so the root
 * always expands and a search always returns a usable move distribution. */
#define MCTS_MIN_NODES   (1 + MAX_MOVES)

/* Evaluation-cache size, as a power of two entries, derived from the node pool
 * so that a big search gets a big cache and a self-play worker does not pay for
 * one.  Each entry is 144 bytes. */
#define MCTS_CACHE_MIN_BITS  10
#define MCTS_CACHE_MAX_BITS  16

/* ------------------------------------------------------------------- rng */
/* xoshiro256** -- bit-identical to arena.c's rng_next(), so the caller's
 * `uint64_t *rng` (a 4-word xoshiro state, as in arena.h / PlayCfg) may be
 * shared with the rest of the system.  Duplicated rather than linked so that
 * mcts.c depends only on chess.c and net.c. */

static inline uint64_t mc_rotl(uint64_t x, int k)
{
    return (x << k) | (x >> (64 - k));
}

static uint64_t mc_rng_next(uint64_t *s)
{
    const uint64_t result = mc_rotl(s[1] * 5, 7) * 9;
    const uint64_t t = s[1] << 17;

    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];
    s[2] ^= t;
    s[3] = mc_rotl(s[3], 45);

    return result;
}

/* Uniform in the OPEN interval (0,1): never 0, so logf() of it is finite. */
static float mc_rng_u01(uint64_t *s)
{
    return (float)((mc_rng_next(s) >> 40) + 1u) * (1.0f / 16777217.0f);
}

static double mc_rng_u01d(uint64_t *s)
{
    return (double)(mc_rng_next(s) >> 11) * (1.0 / 9007199254740992.0);
}

/* Marsaglia polar method; fully reentrant. */
static float mc_rng_normal(uint64_t *s)
{
    float u, v, r;
    do {
        u = 2.0f * mc_rng_u01(s) - 1.0f;
        v = 2.0f * mc_rng_u01(s) - 1.0f;
        r = u * u + v * v;
    } while (r >= 1.0f || r == 0.0f);
    return u * sqrtf(-2.0f * logf(r) / r);
}

/* Marsaglia-Tsang (2000) gamma sampler.  alpha < 1 uses the standard boost
 * Gamma(a) = Gamma(a+1) * U^(1/a), which matters here because the AlphaZero
 * Dirichlet concentration is 0.3. */
static float mc_rng_gamma(uint64_t *s, float alpha)
{
    float boost = 1.0f;

    if (!(alpha > 0.0f)) return 0.0f;
    if (alpha < 1.0f) {
        boost = powf(mc_rng_u01(s), 1.0f / alpha);
        alpha += 1.0f;
    }

    const float d = alpha - 1.0f / 3.0f;
    const float c = 1.0f / sqrtf(9.0f * d);

    for (;;) {
        float x, v;
        do {
            x = mc_rng_normal(s);
            v = 1.0f + c * x;
        } while (v <= 0.0f);
        v = v * v * v;

        const float u  = mc_rng_u01(s);
        const float x2 = x * x;
        if (u < 1.0f - 0.0331f * x2 * x2)                      return boost * d * v;
        if (logf(u) < 0.5f * x2 + d * (1.0f - v + logf(v)))    return boost * d * v;
    }
}

/* ------------------------------------------------------------- hashing */

static inline uint64_t mc_mix64(uint64_t x)
{
    x ^= x >> 30; x *= 0xBF58476D1CE4E5B9ULL;
    x ^= x >> 27; x *= 0x94D049BB133111EBULL;
    x ^= x >> 31;
    return x;
}

/* A strided sample of a weight block, treated as the flat float array it is.
 * Deliberately layout-independent: the network's shape is net.h's business and
 * has changed before, and a fingerprint that names tensors would read off the
 * end of the next one.  Sampled in short contiguous runs rather than one float
 * at a time so that 128 samples cost ~16 cache lines instead of 128. */
static uint64_t mc_hash_sample(uint64_t s, const void *base, size_t n, int nrun)
{
    const float *v = (const float *)base;
    const size_t run  = 8;
    const size_t step = (n > run * (size_t)nrun) ? n / (size_t)nrun : run;

    for (size_t o = 0; o < n; o += step) {
        for (size_t i = 0; i < run && o + i < n; i++) {
            uint32_t u;
            memcpy(&u, &v[o + i], sizeof u);
            s = (s ^ (uint64_t)u) * 0x100000001B3ULL;
        }
    }
    return s ^ (uint64_t)n;          /* a reshape is a change too */
}

/* A fingerprint of "which network is this".  A standing tree and every cached
 * evaluation belong to the weights that produced them: in self-play the head
 * changes between plies because the two sides are different agents, and every
 * parameter moves on a training step.  Reusing across either would quietly mix
 * one agent's evaluations into another's search and into its training targets.
 *
 * The pointers catch a change of agent; the sampled weights catch a change of
 * weights (a training step moves every parameter, so a sample is enough).  It
 * costs well under a microsecond, once per search -- see the report. */
static uint64_t mc_weight_stamp(const Trunk *t, const Head *h)
{
    uint64_t s = mc_mix64((uint64_t)(uintptr_t)t ^ 0x9E3779B97F4A7C15ULL);
    s = mc_mix64(s ^ (uint64_t)(uintptr_t)h);
    s = mc_hash_sample(s, t, TRUNK_NPARAM, 16);
    s = mc_hash_sample(s, h, HEAD_NPARAM,  16);
    s = mc_mix64(s);
    return s ? s : 1u;               /* 0 is the "nothing stamped yet" value */
}

/* --------------------------------------------------------- evaluation cache */

typedef struct {
    uint64_t tag;                 /* full key; 0 with gen != live is empty    */
    uint32_t gen;                 /* generation stamp, for O(1) invalidation  */
    float    v;                   /* value head, side-to-move view            */
    float    q[NF_PDIM];          /* policy query; nn_logits() needs only this */
} McEntry;

struct MctsCache {
    McEntry *e;
    uint32_t mask;
    uint32_t gen;
};

/* ------------------------------------------------- the stepped search's state
 *
 * Everything mcts_search() used to hold in locals and in its two nested loops.
 * It is here, and not on the stack, because the search must be able to STOP at
 * a leaf, hand the leaf out to be evaluated in a batch with other searches, and
 * resume exactly where it was.  `phase` is the program counter; `resume` is
 * where a delivery lands.
 *
 * One per Mcts, allocated once in mcts_init(): the hot path still never calls
 * malloc, and a concurrent search still copies exactly one Position. */
enum {
    MC_PH_ROOT = 0,    /* expand the root                                     */
    MC_PH_ROOT_EVAL,   /* ... resuming, the root's evaluation has arrived     */
    MC_PH_ROOT_DONE,   /* the root is expanded: check it, seed N and W        */
    MC_PH_PREP,        /* root noise, the budget, the repetition base         */
    MC_PH_SIM,         /* start one simulation                                */
    MC_PH_DESCEND,     /* select down to a leaf                               */
    MC_PH_LEAF_EVAL,   /* ... resuming, the leaf's evaluation has arrived     */
    MC_PH_BACKUP,      /* back the value up and unwind the position           */
    MC_PH_DONE
};

struct MctsRun {
    const Trunk *t;
    const Head  *h;
    const Game  *g;
    uint64_t    *rng;
    int          sims, root_noise;
    int          phase, resume;
    int          started;          /* between mcts_begin() and mcts_end()     */

    /* the ONE Position copy of the whole search, maintained incrementally */
    Position     pos;
    /* [ game history tail | selection path ], keys[nkeys-1] == current key */
    uint64_t     keys[MCTS_HIST_KEEP + MCTS_MAX_DEPTH + 2];
    int          nkeys, base_keys;

    int          path[MCTS_MAX_DEPTH + 1];
    Undo         undo[MCTS_MAX_DEPTH + 1];
    int          depth;

    int          todo, s;          /* simulation budget and counter           */
    int          nroot;            /* legal moves at the root, 0 = no search  */
    float        value;            /* the value being backed up               */
    float        rootv0;           /* the root's own evaluation               */

    /* the leaf being expanded, held across a suspension */
    int          node;             /* its pool index                          */
    Move         list[MAX_MOVES];  /* its legal moves, from gen_legal()       */
    int          nlist;
    McEntry     *slot;             /* cache slot to fill on delivery, or NULL */
    uint64_t     tag;
    uint16_t     fidx[NF_MAXACTIVE];
    int          nf;
    Fwd          fw;               /* only q and v are ever meaningful        */
};

/* The cache key is the whole network input, not just the zobrist key.  See the
 * file header: the halfmove-clock bucket is a network feature the zobrist key
 * does not carry, and the Chess960 rook files change the legal move list. */
static uint64_t mc_pos_key(const Position *p)
{
    int bucket = (int)p->halfmove / 13;
    if (bucket > 7) bucket = 7;

    uint64_t x = (uint64_t)(unsigned)bucket
               | ((uint64_t)p->chess960      << 4)
               | ((uint64_t)p->crook[0][0]   << 8)
               | ((uint64_t)p->crook[0][1]   << 16)
               | ((uint64_t)p->crook[1][0]   << 24)
               | ((uint64_t)p->crook[1][1]   << 32);

    return mc_mix64(p->key ^ mc_mix64(x));
}

/* The tag folds the WEIGHT STAMP in beside the network input, so an entry names
 * (input, trunk, head, weights) rather than merely the position.  Two
 * consequences, and the second is the whole point:
 *
 *   - a search never has to flush the table when the agent or the weights
 *     change.  An entry written under other weights simply has a different tag
 *     and can never be hit again.  That is a STRICTLY STRONGER guarantee than
 *     the flush it replaces, which relied on noticing the change in time.
 *   - several searches can therefore SHARE one table even while they play
 *     different agents.  Before this the stamp was compared against a per-search
 *     scalar and any disagreement wiped the whole table, so two concurrent games
 *     on different heads would have spent their lives flushing each other.
 *     Sharing is what makes G concurrent games per thread affordable: one table
 *     per thread instead of G of them.
 *
 * A hit is still bit-for-bit what nn_eval() would have returned for THAT
 * search, which is the only property anything downstream depends on.
 * tests/test_mcts.c checks it by running whole searches with the cache on and
 * with it off and comparing visit counts exactly. */
/* The tag carries the stamp; the INDEX does not.  A slot is chosen from the
 * position alone, so which slot a lookup lands in is a function of the game and
 * of nothing else.  That matters because the stamp contains the trunk and head
 * POINTERS -- that is how a change of agent is detected -- and those move with
 * ASLR: indexing by the tag would make the hit/miss pattern differ between two
 * runs of the same seed, and with G games in flight the number of evaluations a
 * game needs decides the round it finishes in, which decides the order finished
 * games enter the replay buffer.  The VALUES would still be right (a hit needs
 * a full tag match), but the run would stop being reproducible.  Indexing on
 * the position costs only that two heads evaluating the SAME position now share
 * one slot instead of two. */
typedef struct { uint64_t tag; uint32_t idx; } McKey;

static inline McKey mc_key(const Mcts *m, const Position *p)
{
    const uint64_t pk = mc_pos_key(p);
    McKey k;
    k.tag = mc_mix64(pk ^ m->wstamp);
    k.idx = (uint32_t)pk;
    return k;
}

/* Invalidate the standing tree only.  The tree is statistics ABOUT particular
 * weights and cannot be re-tagged, so unlike the cache it really must be
 * dropped when they change. */
static void mc_tree_drop(Mcts *m) { m->tree_valid = 0; }

void mcts_cache_clear(Mcts *m)
{
    if (!m) return;
    m->tree_valid = 0;
    m->cache_flushes++;
    if (!m->ecache) return;
    if (++m->ecache->gen == 0) {       /* wrapped after 2^32 flushes: reset hard */
        memset(m->ecache->e, 0, ((size_t)m->ecache->mask + 1) * sizeof(McEntry));
        m->ecache->gen = 1;
    }
}

/* ------------------------------------------------------------ shared cache */

static int mc_clamp_bits(int bits)
{
    if (bits < 8)  bits = 8;
    if (bits > 20) bits = 20;
    return bits;
}

struct MctsCache *mcts_cache_create(int bits)
{
    bits = mc_clamp_bits(bits);
    struct MctsCache *c = (struct MctsCache *)malloc(sizeof *c);
    if (!c) return NULL;
    c->e = (McEntry *)calloc((size_t)1 << bits, sizeof(McEntry));
    if (!c->e) { free(c); return NULL; }
    c->mask = (uint32_t)(((size_t)1 << bits) - 1);
    c->gen  = 1;
    return c;
}

void mcts_cache_destroy(struct MctsCache *c)
{
    if (!c) return;
    free(c->e);
    free(c);
}

void mcts_cache_attach(Mcts *m, struct MctsCache *c)
{
    if (!m) return;
    if (m->owns_cache && m->ecache) mcts_cache_destroy(m->ecache);
    m->ecache     = c;
    m->owns_cache = 0;
    m->tree_valid = 0;
    if (!c) { m->cache_bits = 0; return; }
    int bits = 0;
    while (((uint32_t)1 << bits) <= c->mask) bits++;
    m->cache_bits = bits;
}

/* ------------------------------------------------------------- node pool */

uint64_t mcts_pool_exhausted(const Mcts *m)
{
    return m ? m->pool_exhausted : 0;
}

void mcts_defaults(Mcts *m)
{
    if (!m) return;
    m->c_puct          = 1.4f;
    /* First-play urgency: the value assumed for an edge that has never been
     * visited.  0 is the neutral "unknown, assume a draw" choice and, unlike a
     * parent-derived FPU, introduces nothing that has to be tuned per game. */
    m->fpu             = 0.0f;
    m->fpu_reduction   = 0.0f;        /* 0 reproduces the flat fpu exactly */
    m->dirichlet_alpha = 0.3f;
    m->dirichlet_eps   = 0.25f;
    m->reuse           = 1;
    m->cache           = 1;
    /* cache_bits is NOT reset here: it reports the size of an allocation made
     * once, in mcts_init(), and this function may be called at any time. */
}

static int mc_cache_bits_for(int max_nodes)
{
    int bits = 0;
    while ((1 << (bits + 1)) <= max_nodes) bits++;   /* floor(log2(max_nodes)) */
    bits -= 2;
    if (bits < MCTS_CACHE_MIN_BITS) bits = MCTS_CACHE_MIN_BITS;
    if (bits > MCTS_CACHE_MAX_BITS) bits = MCTS_CACHE_MAX_BITS;
    return bits;
}

void mcts_init(Mcts *m, int max_nodes)
{
    if (!m) return;
    memset(m, 0, sizeof *m);
    mcts_defaults(m);

    if (max_nodes < MCTS_MIN_NODES) max_nodes = MCTS_MIN_NODES;

    m->pool = (MctsNode *)malloc((size_t)max_nodes * sizeof(MctsNode));
    if (!m->pool) return;             /* pool == NULL: mcts_search returns 0 */

    /* Scratch for subtree compaction: one index per node.  Allocated here so
     * that the search itself never calls malloc. */
    m->remap = (int32_t *)malloc((size_t)max_nodes * sizeof(int32_t));
    if (!m->remap) { free(m->pool); m->pool = NULL; return; }

    m->cap  = max_nodes;
    m->used = 0;

    /* The stepped search's resumable state.  Allocated once, here, so that the
     * hot path still never calls malloc even though the search can now suspend
     * itself in the middle of a simulation. */
    m->run = (struct MctsRun *)calloc(1, sizeof(struct MctsRun));
    if (!m->run) { free(m->pool); free(m->remap); m->pool = NULL; m->remap = NULL; return; }

    m->cache_bits = mc_cache_bits_for(max_nodes);
    {
        struct MctsCache *ca = mcts_cache_create(m->cache_bits);
        m->ecache     = ca;            /* NULL simply means "no cache" */
        m->owns_cache = (ca != NULL);
        if (!ca) m->cache_bits = 0;
    }
    /* m->evals and m->max_depth_seen accumulate across searches (a running
     * total and a running maximum).  They are plain fields, so a caller that
     * wants per-move telemetry just zeroes them before each search. */
}

void mcts_free(Mcts *m)
{
    if (!m) return;
    free(m->pool);
    free(m->remap);
    free(m->run);
    if (m->owns_cache) mcts_cache_destroy(m->ecache);
    m->pool       = NULL;
    m->remap      = NULL;
    m->run        = NULL;
    m->ecache     = NULL;
    m->owns_cache = 0;
    m->cap        = 0;
    m->used       = 0;
    m->tree_valid = 0;
}

/* ------------------------------------------------------------ the search */

/* Threefold counting over [game history tail | selection path].  Identical in
 * form to game_repetitions(): walk back at most `halfmove` plies (nothing older
 * can match, an irreversible move lies in between) two at a time. */
static int mcts_rep_count(const uint64_t *keys, int nkeys, uint64_t key, int halfmove)
{
    int lo = nkeys - 1 - halfmove;
    if (lo < 0) lo = 0;

    int c = 0;
    for (int i = nkeys - 1; i >= lo; i -= 2)
        if (keys[i] == key) c++;
    return c;
}

/* PUCT:  argmax_a  Q(s,a) + c_puct * P(s,a) * sqrt(sum_b N(s,b)) / (1 + N(s,a))
 *
 * Q is expressed from the CURRENT node's point of view.  The child accumulates
 * from the opposite point of view, hence the negation. */
static int mcts_select(const Mcts *m, const MctsNode *nd)
{
    const MctsNode *ch = m->pool + nd->first;
    const int n = nd->nchild;

    /* Every visit to this node after its own expansion descended into exactly
     * one child, so sum_b N(s,b) == N(s) - 1.  On the first descent that sum is
     * 0, so the exploration term vanishes and the first child is taken; the
     * formula is applied literally, and the priors take over from the second
     * descent onwards.  (AlphaZero's own pseudocode uses sqrt(sum + 1) here,
     * which would cost nothing to switch to.) */
    int32_t nsum = nd->N - 1;
    if (nsum < 0) nsum = 0;

    const float sq  = sqrtf((float)nsum);
    const float c   = m->c_puct;

    /* First-play urgency.  Flat by default; with fpu_reduction > 0 it becomes
     * the parent's own estimate discounted by how much of the prior mass has
     * already been tried, which is the Leela/KataGo rule.  Both inputs are
     * search statistics -- a visit count and a prior -- so this stays inside
     * the "bandit arithmetic over network output" contract. */
    float fpu = m->fpu;
    if (m->fpu_reduction > 0.0f) {
        float pvis = 0.0f;
        for (int i = 0; i < n; i++) if (ch[i].N > 0) pvis += ch[i].P;
        fpu = ((nd->N > 0) ? (nd->W / (float)nd->N) : 0.0f)
            - m->fpu_reduction * sqrtf(pvis);
        if      (fpu < -1.0f) fpu = -1.0f;
        else if (fpu >  1.0f) fpu =  1.0f;
    }

    int   best  = 0;
    float bestv = -3.0e38f;

    for (int i = 0; i < n; i++) {
        const int32_t cn = ch[i].N;
        const float q = (cn > 0) ? -(ch[i].W / (float)cn) : fpu;
        const float u = c * ch[i].P * sq / (float)(1 + cn);
        const float s = q + u;
        if (s > bestv) { bestv = s; best = i; }
    }
    return best;
}

/* ---------------------------------------------------------------- expansion
 *
 * mcts_expand() used to be one function: generate the moves, test the terminal
 * conditions, evaluate the network, expand.  It is now two, split exactly where
 * the network call was, because that call is the only thing a search ever has
 * to wait for.
 *
 *   mc_expand_begin  does everything up to and including the CACHE LOOKUP and
 *                    returns what happened.  On a miss it leaves the active
 *                    features in r->fidx/r->nf for the caller to evaluate,
 *                    which it may do in a batch with other searches' leaves.
 *   mc_expand_end    consumes r->fw (q and v, from the cache or from the
 *                    network) and expands the node.
 *
 * The order of operations is unchanged, including the two that are observable:
 * the evaluation still happens BEFORE the pool-capacity test, so an exhausted
 * pool still costs an evaluation and still returns v without expanding; and the
 * terminal tests still happen before the evaluation, so a terminal leaf still
 * never touches the network.
 *
 * ONLY q and v are ever read out of a cached entry.  nn_logits() reads fw->q
 * and nothing else, and the backup uses fw->v; the rest of Fwd is the training
 * path's business.  A cached result is therefore bit-for-bit what nn_eval()
 * would have produced for everything this file goes on to compute. */
enum { MC_EX_TERMINAL = 0,   /* nd->tval is the value; no network needed      */
       MC_EX_READY,          /* r->fw is filled from the cache                */
       MC_EX_NEEDEVAL };     /* r->fidx/r->nf need the network                */

static int mc_expand_begin(Mcts *m, int ni, const Position *pos,
                           const uint64_t *keys, int nkeys, int depth)
{
    struct MctsRun *restrict r = m->run;
    MctsNode *nd = &m->pool[ni];

    r->node  = ni;
    r->slot  = NULL;
    r->nlist = gen_legal(pos, r->list);          /* once per expanded node */
    const int n = r->nlist;

    /* ---- terminal states.  Rules of chess, never the network. ---------- */
    if (n == 0) {
        if (in_check(pos, pos->side)) {
            nd->terminal = (int8_t)TR_CHECKMATE;
            nd->tval     = -1.0f;                /* the side to move is mated */
        } else {
            nd->terminal = (int8_t)TR_STALEMATE;
            nd->tval     = 0.0f;                 /* a draw, NOT a win        */
        }
        return MC_EX_TERMINAL;
    }
    if (insufficient_material(pos)) {
        nd->terminal = (int8_t)TR_INSUFFICIENT; nd->tval = 0.0f; return MC_EX_TERMINAL;
    }
    if (pos->halfmove >= 100) {
        nd->terminal = (int8_t)TR_FIFTY;        nd->tval = 0.0f; return MC_EX_TERMINAL;
    }
    if (mcts_rep_count(keys, nkeys, pos->key, (int)pos->halfmove) >= 3) {
        nd->terminal = (int8_t)TR_REPETITION;   nd->tval = 0.0f; return MC_EX_TERMINAL;
    }
    if (depth >= MCTS_MAX_DEPTH) {
        /* The ply cap is measured from the ROOT, so this adjudication would not
         * survive a re-rooting: the same node sits a ply or two higher in the
         * next search, where the cap does not apply.  Rather than carry a stale
         * adjudication, mark the tree and refuse to inherit it -- which keeps
         * "an inherited tree is exactly a fresh one" true without exception. */
        nd->terminal = (int8_t)TR_MAX_PLIES;    nd->tval = 0.0f;
        m->tree_capped = 1;
        return MC_EX_TERMINAL;
    }

    /* ---- the network: the only evaluation in this file ----------------- */
    struct MctsCache *ca = m->cache ? m->ecache : NULL;
    if (ca) {
        const McKey k = mc_key(m, pos);
        McEntry *slot = &ca->e[k.idx & ca->mask];
        if (slot->gen == ca->gen && slot->tag == k.tag) {
            memcpy(r->fw.q, slot->q, sizeof slot->q);
            r->fw.v = slot->v;
            m->cache_hits++;
            return MC_EX_READY;
        }
        m->cache_misses++;
        r->slot = slot;
        r->tag  = k.tag;
    }
    r->nf = nn_features(pos, r->fidx);
    return MC_EX_NEEDEVAL;
}

/* Expands the node mc_expand_begin() left pending and returns its value from
 * its OWN side-to-move point of view. */
static float mc_expand_end(Mcts *m, const Head *h, const Position *pos)
{
    struct MctsRun *restrict r = m->run;
    MctsNode *nd = &m->pool[r->node];
    const int n = r->nlist;

    /* Out of nodes: degrade gracefully.  Keep evaluating, stop growing.  The
     * node stays unexpanded, so it is re-evaluated on every later visit and the
     * search behaves like a 1-ply rollout from here down.  Never crashes, never
     * corrupts the tree. */
    if (m->used + n > m->cap) {
        m->pool_exhausted++;
        return r->fw.v;
    }

    MoveKey mk[MAX_MOVES];
    float   logits[MAX_MOVES];
    float   pri[MAX_MOVES];

    for (int i = 0; i < n; i++) nn_move_key(pos, r->list[i], &mk[i]);
    nn_logits(h, &r->fw, mk, n, logits);
    softmax_t(logits, n, 1.0f, pri);             /* priors over the LEGAL moves */

    const int base = m->used;
    m->used += n;
    nd->first  = base;
    nd->nchild = (int16_t)n;

    for (int i = 0; i < n; i++) {
        MctsNode *c = &m->pool[base + i];
        c->P        = pri[i];
        c->W        = 0.0f;
        c->N        = 0;
        c->first    = -1;
        c->nchild   = 0;
        c->terminal = 0;
        c->tval     = 0.0f;
        c->move     = r->list[i];
    }
    return r->fw.v;
}

/* P <- (1-eps) * P + eps * Dir(alpha), the AlphaZero root exploration noise.
 * Self-play only: it must be off for evaluation and for play against a human. */
static void mcts_root_noise(Mcts *m, MctsNode *root, uint64_t *rng)
{
    const int n = root->nchild;
    if (n <= 0) return;

    float d[MAX_MOVES];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        d[i] = mc_rng_gamma(rng, m->dirichlet_alpha);
        sum += d[i];
    }

    if (!(sum > 0.0f)) {                         /* astronomically unlikely */
        const float u = 1.0f / (float)n;
        for (int i = 0; i < n; i++) d[i] = u;
    } else {
        const float r = 1.0f / sum;
        for (int i = 0; i < n; i++) d[i] *= r;
    }

    const float e = m->dirichlet_eps;
    MctsNode *ch = m->pool + root->first;
    for (int i = 0; i < n; i++)
        ch[i].P = (1.0f - e) * ch[i].P + e * d[i];
}

/* ------------------------------------------------------------ subtree reuse */

/* Same position in every way the search or the network can tell apart.  The
 * zobrist key alone is not enough: the halfmove clock is a network input and is
 * not in the key, and the Chess960 rook files decide what gen_legal() emits. */
static int mc_same_pos(const Position *a, const Position *b)
{
    return a->key      == b->key
        && a->side     == b->side
        && a->castling == b->castling
        && a->ep       == b->ep
        && a->halfmove == b->halfmove
        && a->chess960 == b->chess960
        && memcmp(a->crook, b->crook, sizeof a->crook) == 0
        && memcmp(a->piece, b->piece, sizeof a->piece) == 0;
}

/* Moves the subtree rooted at pool[r] to pool[0] and compacts it to the front
 * of the pool.  Returns the new node count.
 *
 * The one fact that makes this safe in place: a node's children are allocated
 * when it is expanded, which is always AFTER the node itself, so every child
 * index exceeds its parent's.  Two consequences:
 *
 *   - one ASCENDING pass marks the whole subtree, because a node is always
 *     reached before its children are scanned;
 *   - numbering the marked nodes in ascending order gives the k-th of them the
 *     destination k, and since marked indices are distinct and increasing the
 *     k-th is at least k.  Every write therefore lands on an index at or below
 *     the one being read, and never on a node that has not been copied yet.
 *
 * Contiguity of a child block survives because the block is a run of
 * consecutive indices and no other subtree node can fall inside it. */
static int mcts_compact(Mcts *m, int r)
{
    int32_t *map = m->remap;
    const int used = m->used;
    int dst = 0;

    for (int i = r; i < used; i++) map[i] = -1;
    map[r] = -2;                                   /* -2 = in the subtree */

    for (int i = r; i < used; i++) {
        if (map[i] != -2) continue;
        map[i] = dst++;
        const MctsNode *nd = &m->pool[i];
        if (nd->first >= 0)
            for (int k = 0; k < nd->nchild; k++) map[nd->first + k] = -2;
    }

    for (int i = r; i < used; i++) {
        if (map[i] < 0) continue;
        MctsNode nd = m->pool[i];
        if (nd.first >= 0) nd.first = map[nd.first];
        m->pool[map[i]] = nd;
    }

    m->used = dst;
    return dst;
}

/* Tries to re-root the standing tree on g->pos.  Returns 1 when pool[0] is now
 * a valid, verified tree for that position; 0 when the caller must build one.
 * Every early return is a refusal to use a tree that might not be the right
 * one, and refusing is always safe. */
static int mcts_try_reuse(Mcts *m, const Game *g, const uint64_t *keys, int nkeys,
                          int sims, int root_noise)
{
    if (!m->reuse || !m->tree_valid || m->used <= 0) return 0;
    if (m->tree_capped) return 0;     /* holds a root-relative adjudication */

    const int d = g->ply - m->rply;
    if (d < 0 || d > 2) return 0;
    /* Re-rooting at the SAME position is what a time-capped caller's 32/64/128
     * ramp does, and it is then exactly a fresh search of the larger budget --
     * but only with the noise out of the picture in both directions.  A second
     * helping of Dirichlet noise on top of the first is not a fresh search. */
    if (d == 0 && (root_noise || m->tree_noised)) return 0;

    Position p = m->rpos;
    uint64_t pk[2];
    int idx = 0;

    for (int k = 0; k < d; k++) {
        const MctsNode *nd = &m->pool[idx];
        if (nd->terminal || nd->first < 0 || nd->nchild <= 0) return 0;

        const Move mv = g->moves[m->rply + k];
        int ci = -1;
        for (int i = 0; i < nd->nchild; i++)
            if (m->pool[nd->first + i].move == mv) { ci = nd->first + i; break; }
        if (ci < 0) return 0;         /* not an edge of this tree: cannot follow */

        Undo u;
        make_move(&p, mv, &u);        /* safe: tree edges are legal moves of p */
        pk[k] = p.key;
        idx   = ci;
    }

    const MctsNode *R = &m->pool[idx];
    if (R->terminal || R->first < 0 || R->nchild <= 0 || R->N <= 0) return 0;
    if (R->N > sims + 1) return 0;    /* would overshoot sum(visits) == sims */
    if (!mc_same_pos(&p, &g->pos))    return 0;

    /* sum(children N) == N - 1 is what makes `sims` a total the caller can rely
     * on, and an exhausted pool can break it: a node visited while the pool was
     * full is counted but descends nowhere, and it may be expanded later once a
     * re-rooting frees space.  Inheriting such a node would hand the caller a
     * visits[] that sums to less than sims, so check rather than assume. */
    {
        int32_t sum = 0;
        for (int i = 0; i < R->nchild; i++) sum += m->pool[R->first + i].N;
        if (sum != R->N - 1) return 0;
    }

    /* The repetition window must be the one this subtree was built under, or a
     * cached TR_REPETITION would be a fact about a different game's history. */
    {
        uint64_t pred[MCTS_HIST_KEEP + 4];
        int np = m->nrkeys;
        if (np < 0 || np > MCTS_HIST_KEEP) return 0;

        memcpy(pred, m->rkeys, (size_t)np * sizeof pred[0]);
        for (int k = 0; k < d; k++) pred[np++] = pk[k];

        int want = (int)g->pos.halfmove + 1;
        if (want > np) want = np;
        if (want != nkeys) return 0;
        if (memcmp(pred + (np - want), keys, (size_t)want * sizeof pred[0]) != 0)
            return 0;
    }

    /* The caller's visits[] is parallel to gen_legal(), so the children have to
     * BE that list, in that order.  They will be -- gen_legal() is a function of
     * the position, and the position has just been verified -- but this is the
     * one guarantee the caller cannot check for itself, so it is checked here. */
    {
        Move ml[MAX_MOVES];
        const int nl = gen_legal(&g->pos, ml);
        if (nl != (int)R->nchild) return 0;
        for (int i = 0; i < nl; i++)
            if (m->pool[R->first + i].move != ml[i]) return 0;
    }

    if (idx != 0) mcts_compact(m, idx);

    m->reuse_hits++;
    m->reuse_nodes += (uint64_t)m->used;
    return 1;
}

/* ------------------------------------------------------------------ search */
/*
 * The search is written as a RESUMABLE STATE MACHINE rather than as two nested
 * loops, because MCTS is strictly sequential and that is exactly the problem.
 * A simulation cannot pick its next path until the previous leaf's value is
 * known, so a lone search can never present the network with more than one
 * position at a time -- a 128x128 matrix-VECTOR product, which cannot use the
 * machine's width.
 *
 * So the search stops at the leaf instead of blocking on it.  mcts_step() runs
 * until a leaf needs the network and hands it out; the caller batches that leaf
 * with the pending leaves of OTHER, INDEPENDENT searches, evaluates them all in
 * one nn_eval_batch(), and gives each value back with mcts_deliver().
 *
 * Nothing about the search changes.  Independent searches share no tree, no
 * node pool and no repetition window, so each one performs exactly the
 * simulations it would have performed alone, in exactly the same order, off
 * exactly the same values -- there is no virtual loss here and no approximation
 * of any kind.  That is the advantage over leaf-parallelising ONE search, which
 * has to invent a penalty to stop every thread walking the same path.
 *
 * mcts_search() is this machine driven with a batch size of one, so the
 * sequential and the concurrent paths cannot drift apart: there is only one
 * implementation.
 */

int mcts_begin(Mcts *m, const Trunk *t, const Head *h, const Game *g,
               int sims, int root_noise, uint64_t *rng)
{
    if (!m || !m->pool || !m->run || m->cap <= 0 || !t || !h || !g) return 0;

    struct MctsRun *r = m->run;
    r->started = 0;                              /* nothing to end if we bail */
    if (g->result != GR_ONGOING) return 0;       /* the game is already over */

    r->t = t; r->h = h; r->g = g; r->rng = rng;
    r->sims = sims; r->root_noise = root_noise;
    r->nroot = 0; r->base_keys = 0; r->slot = NULL;

    /* A standing TREE is only valid for the weights that made it, so drop it
     * the moment the agent or the weights change.  The evaluation cache needs
     * no such thing: the stamp is part of every cache tag (see mc_tag). */
    {
        const uint64_t st = mc_weight_stamp(t, h);
        if (st != m->wstamp) { mc_tree_drop(m); m->wstamp = st; }
    }

    /* The ONE Position copy of the whole search.  From here on the position is
     * maintained incrementally by make_move / unmake_move. */
    r->pos = g->pos;

    /* [ game history tail | selection path ], keys[nkeys-1] == current key. */
    r->nkeys = 0;
    if (g->hist_len > 0) {
        int keep = g->hist_len;
        if (keep > (int)r->pos.halfmove + 1) keep = (int)r->pos.halfmove + 1;
        if (keep > MCTS_HIST_KEEP)           keep = MCTS_HIST_KEEP;
        if (keep < 1)                        keep = 1;
        for (int i = g->hist_len - keep; i < g->hist_len; i++) r->keys[r->nkeys++] = g->hist[i];
    }
    if (r->nkeys == 0 || r->keys[r->nkeys - 1] != r->pos.key) {  /* Game never pushed */
        r->nkeys = 0;
        r->keys[r->nkeys++] = r->pos.key;
    }

    /* Inherit the subtree under the moves that were played, if there provably
     * is one; otherwise a fresh tree, exactly as before. */
    const int reused = mcts_try_reuse(m, g, r->keys, r->nkeys, sims, root_noise);
    if (!reused && m->tree_valid) m->reuse_misses++;

    MctsNode *root = &m->pool[0];
    if (!reused) {
        m->used = 1;
        m->tree_capped = 0;
        root->P        = 1.0f;
        root->W        = 0.0f;
        root->N        = 0;
        root->first    = -1;
        root->nchild   = 0;
        root->terminal = 0;
        root->tval     = 0.0f;
        root->move     = MV_NONE;
        r->phase = MC_PH_ROOT;
    } else {
        /* pool[0] is a former child, and its EDGE fields -- the prior for the
         * move into it, and the move itself -- belong to a parent that is no
         * longer there.  Nothing reads them at a root, but leave a root looking
         * like a root for anyone who walks the pool. */
        root->P    = 1.0f;
        root->move = MV_NONE;
        r->phase = MC_PH_PREP;
    }

    r->started = 1;
    return 1;
}

/* Selects down from the root to a leaf.  Returns 1 when that leaf needs the
 * network (r->fidx / r->nf are filled), 0 when r->value is already the leaf's
 * value and the simulation can be backed up. */
static int mc_descend(Mcts *m)
{
    struct MctsRun *restrict r = m->run;
    /* depth and nkeys are stepped once per ply, so they are kept in registers
     * and written back at the ends -- they live in the run state only because
     * the descent has to survive a suspension, not because the loop wants them
     * there. */
    int depth = r->depth;
    int nkeys = r->nkeys;

    for (;;) {
        const int ni = r->path[depth];
        MctsNode *nd = &m->pool[ni];

        if (nd->terminal) {                  /* cached: no network, no rules */
            r->value = nd->tval;
            r->depth = depth; r->nkeys = nkeys;
            return 0;
        }
        if (nd->first < 0) {                 /* unexpanded leaf */
            r->depth = depth; r->nkeys = nkeys;
            const int rc = mc_expand_begin(m, ni, &r->pos, r->keys, nkeys, depth);
            if (rc == MC_EX_NEEDEVAL) return 1;
            r->value = (rc == MC_EX_TERMINAL) ? m->pool[ni].tval
                                              : mc_expand_end(m, r->h, &r->pos);
            return 0;
        }

        const int ci = nd->first + mcts_select(m, nd);
        make_move(&r->pos, m->pool[ci].move, &r->undo[depth]);
        r->keys[nkeys++] = r->pos.key;
        r->path[++depth] = ci;
    }
}

int mcts_step(Mcts *m)
{
    if (!m || !m->run || !m->run->started) return 0;
    struct MctsRun *restrict r = m->run;

    for (;;) {
        switch (r->phase) {

        case MC_PH_ROOT: {
            const int rc = mc_expand_begin(m, 0, &r->pos, r->keys, r->nkeys, 0);
            if (rc == MC_EX_NEEDEVAL) { r->resume = MC_PH_ROOT_EVAL; return 1; }
            r->rootv0 = (rc == MC_EX_TERMINAL) ? m->pool[0].tval
                                               : mc_expand_end(m, r->h, &r->pos);
            r->phase = MC_PH_ROOT_DONE;
            continue;
        }

        case MC_PH_ROOT_EVAL:
            r->rootv0 = mc_expand_end(m, r->h, &r->pos);
            r->phase  = MC_PH_ROOT_DONE;
            continue;

        case MC_PH_ROOT_DONE: {
            MctsNode *root = &m->pool[0];
            /* checkmate/stalemate/draw at the root, or a pool too small to hold
             * even one move list: either way there is nothing to search. */
            if (root->terminal || root->nchild <= 0 || root->first < 0) {
                m->tree_valid = 0;
                r->nroot = 0;
                r->phase = MC_PH_DONE;
                return 0;
            }
            root->N = 1;
            root->W = r->rootv0;
            r->phase = MC_PH_PREP;
            continue;
        }

        case MC_PH_PREP: {
            MctsNode *root = &m->pool[0];
            r->nroot = root->nchild;

            int noised = 0;
            if (r->root_noise && r->rng &&
                m->dirichlet_eps > 0.0f && m->dirichlet_alpha > 0.0f) {
                mcts_root_noise(m, root, r->rng);
                noised = 1;
            }
            /* Whether the STANDING tree's root priors carry noise.  A reused
             * root is a former child or grandchild, whose own children the
             * noise never reached, so this is simply what was just applied. */
            m->tree_noised = (int8_t)noised;

            r->base_keys = r->nkeys;
            /* `sims` is the TOTAL budget, so an inherited subtree is topped up
             * rather than added to: sum(child visits) == sims and
             * root.N == sims + 1 either way. */
            r->todo = r->sims + 1 - root->N;
            if (r->todo < 0) r->todo = 0;
            r->s = 0;
            r->phase = MC_PH_SIM;
            continue;
        }

        case MC_PH_SIM:
            if (r->s >= r->todo) { r->phase = MC_PH_DONE; return 0; }
            r->depth   = 0;
            r->path[0] = 0;
            r->phase   = MC_PH_DESCEND;
            continue;

        case MC_PH_DESCEND:
            if (mc_descend(m)) { r->resume = MC_PH_LEAF_EVAL; return 1; }
            r->phase = MC_PH_BACKUP;
            continue;

        case MC_PH_LEAF_EVAL:
            r->value = mc_expand_end(m, r->h, &r->pos);
            r->phase = MC_PH_BACKUP;
            continue;

        case MC_PH_BACKUP: {
            const int depth = r->depth;
            if (depth > m->max_depth_seen) m->max_depth_seen = depth;

            /* ---- BACKUP: negate at every ply, the players alternate ----- */
            float value = r->value;
            for (int d = depth; d >= 0; d--) {
                MctsNode *nd = &m->pool[r->path[d]];
                nd->N += 1;
                nd->W += value;
                value  = -value;
            }

            /* ---- unwind the position back to the root ------------------- */
            for (int d = depth; d >= 1; d--)
                unmake_move(&r->pos, m->pool[r->path[d]].move, &r->undo[d - 1]);
            r->nkeys = r->base_keys;

            /* The next simulation starts here rather than through MC_PH_SIM:
             * this is the hot loop, and it is the same two assignments. */
            if (++r->s >= r->todo) { r->phase = MC_PH_DONE; return 0; }
            r->depth   = 0;
            r->path[0] = 0;
            r->phase   = MC_PH_DESCEND;
            continue;
        }

        case MC_PH_DONE:
        default:
            return 0;
        }
    }
}

const uint16_t *mcts_pending(const Mcts *m, int *nf)
{
    if (nf) *nf = 0;
    if (!m || !m->run) return NULL;
    if (nf) *nf = m->run->nf;
    return m->run->fidx;
}

/* r->fw now holds the pending leaf's evaluation: count it, cache it, resume. */
static void mc_commit(Mcts *m)
{
    struct MctsRun *restrict r = m->run;

    m->evals++;
    if (r->slot) {                       /* direct-mapped: the newest wins */
        r->slot->tag = r->tag;
        r->slot->gen = m->ecache->gen;
        r->slot->v   = r->fw.v;
        memcpy(r->slot->q, r->fw.q, sizeof r->slot->q);
        r->slot = NULL;
    }
    r->phase = r->resume;
}

void mcts_deliver(Mcts *m, const Fwd *fw)
{
    if (!m || !m->run || !fw) return;
    struct MctsRun *restrict r = m->run;

    /* Only q and v are read out of an evaluation, so only q and v are copied --
     * 33 floats, not a whole Fwd.  A batched caller owns the Fwd rows (they are
     * nn_eval_batch's working storage as well as its output) and cannot lend
     * them to the search, so this copy is what a batch costs; the one-at-a-time
     * path below skips it by evaluating straight into r->fw. */
    memcpy(r->fw.q, fw->q, sizeof r->fw.q);
    r->fw.v = fw->v;
    mc_commit(m);
}

int mcts_end(Mcts *m, int32_t *visits, float *root_value)
{
    if (root_value) *root_value = 0.0f;
    if (!m || !m->run || !m->run->started) return 0;

    struct MctsRun *r = m->run;
    r->started = 0;
    if (r->nroot <= 0) return 0;         /* terminal root, or pool too small */

    const MctsNode *root = &m->pool[0];
    if (visits) {
        const MctsNode *ch = m->pool + root->first;
        for (int i = 0; i < r->nroot; i++) visits[i] = ch[i].N;
    }
    if (root_value) *root_value = (root->N > 0) ? (root->W / (float)root->N) : 0.0f;

    /* Hand this tree to the next search, together with everything it needs to
     * prove the tree belongs to the position it is then given. */
    m->rpos   = r->g->pos;
    m->nrkeys = r->base_keys;
    memcpy(m->rkeys, r->keys, (size_t)r->base_keys * sizeof r->keys[0]);
    m->rply       = r->g->ply;
    m->tree_valid = 1;

    return r->nroot;
}

/* The sequential search: the same machine, driven one leaf at a time.  There is
 * no second implementation to keep in step -- this IS the batched search with a
 * batch size of one. */
int mcts_search(Mcts *m, const Trunk *t, const Head *h, const Game *g,
                int sims, int root_noise, uint64_t *rng,
                int32_t *visits, float *root_value)
{
    if (root_value) *root_value = 0.0f;
    if (!mcts_begin(m, t, h, g, sims, root_noise, rng)) return 0;

    struct MctsRun *restrict r = m->run;
    while (mcts_step(m)) {
        nn_eval(t, h, r->fidx, r->nf, &r->fw);   /* straight into place */
        mc_commit(m);
    }
    return mcts_end(m, visits, root_value);
}

/* ------------------------------------------------------- the batch driver */

typedef struct {
    Mcts           *m;
    const Head     *h;
    const uint16_t *f;
    int             nf;
} McQueued;

struct MctsBatch {
    int              cap, n;
    McQueued        *q;
    const Head     **heads;
    const uint16_t **fidx;
    int             *nf;
    Fwd             *out;
};

MctsBatch *mcts_batch_create(int cap)
{
    if (cap < 1) cap = 1;
    MctsBatch *b = (MctsBatch *)calloc(1, sizeof *b);
    if (!b) return NULL;
    b->cap   = cap;
    b->q     = (McQueued *)       calloc((size_t)cap, sizeof(McQueued));
    b->heads = (const Head **)    calloc((size_t)cap, sizeof(const Head *));
    b->fidx  = (const uint16_t **)calloc((size_t)cap, sizeof(const uint16_t *));
    b->nf    = (int *)            calloc((size_t)cap, sizeof(int));
    b->out   = (Fwd *)            calloc((size_t)cap, sizeof(Fwd));
    if (!b->q || !b->heads || !b->fidx || !b->nf || !b->out) { mcts_batch_free(b); return NULL; }
    return b;
}

void mcts_batch_free(MctsBatch *b)
{
    if (!b) return;
    free(b->q); free(b->heads); free(b->fidx); free(b->nf); free(b->out);
    free(b);
}

int mcts_batch_cap(const MctsBatch *b)   { return b ? b->cap : 0; }
int mcts_batch_count(const MctsBatch *b) { return b ? b->n   : 0; }

void mcts_batch_add(MctsBatch *b, Mcts *m, const Head *h)
{
    if (!b || !m || b->n >= b->cap) return;
    McQueued *e = &b->q[b->n++];
    e->m  = m;
    e->h  = h;
    e->f  = mcts_pending(m, &e->nf);
}

int mcts_batch_run(MctsBatch *b, const Trunk *t)
{
    if (!b) return 0;
    const int n = b->n;
    if (n <= 0 || !t) { b->n = 0; return 0; }

    /* Group rows that share a head.  nn_eval_batch() fuses the per-agent matrix
     * of ADJACENT rows that name the same head, so this is worth real time when
     * concurrent games are played by different agents -- which in self-play they
     * almost always are.  It cannot change a row's result: every row is an
     * independent position and is evaluated against its own head either way.
     * Insertion sort: n is the games-in-flight count, tens at most. */
    for (int i = 1; i < n; i++) {
        const McQueued key = b->q[i];
        int j = i - 1;
        while (j >= 0 && (uintptr_t)b->q[j].h > (uintptr_t)key.h) { b->q[j + 1] = b->q[j]; j--; }
        b->q[j + 1] = key;
    }

    for (int i = 0; i < n; i++) {
        b->heads[i] = b->q[i].h;
        b->fidx[i]  = b->q[i].f;
        b->nf[i]    = b->q[i].nf;
    }

    nn_eval_batch(t, b->heads, n, b->fidx, b->nf, b->out);

    for (int i = 0; i < n; i++) mcts_deliver(b->q[i].m, &b->out[i]);

    b->n = 0;
    return n;
}

void mcts_search_many(int n, Mcts *const *ms, const Trunk *t,
                      const Head *const *hs, const Game *const *gs,
                      const int *sims, const int *noise, uint64_t *const *rng,
                      int32_t *const *visits, float *root_value, int *nout,
                      MctsBatch *b)
{
    if (n <= 0 || !ms || !t || !hs || !gs || !sims || !b) return;
    const int cap = b->cap;

    for (int lo = 0; lo < n; lo += cap) {
        int hi = lo + cap;
        if (hi > n) hi = n;

        for (int i = lo; i < hi; i++) {
            const int ok = mcts_begin(ms[i], t, hs[i], gs[i], sims[i],
                                      noise ? noise[i] : 0, rng ? rng[i] : NULL);
            if (!ok && nout) nout[i] = 0;
        }

        for (;;) {
            for (int i = lo; i < hi; i++) {
                Mcts *m = ms[i];
                if (!m->run || !m->run->started) continue;
                if (m->run->phase == MC_PH_DONE) continue;
                if (mcts_step(m)) mcts_batch_add(b, m, hs[i]);
            }
            if (mcts_batch_count(b) == 0) break;
            mcts_batch_run(b, t);
        }

        for (int i = lo; i < hi; i++) {
            const int nr = mcts_end(ms[i], visits ? visits[i] : NULL,
                                    root_value ? &root_value[i] : NULL);
            if (nout) nout[i] = nr;
        }
    }
}

/* --------------------------------------------------------------- picking */

/* Sample proportionally to N^(1/temp); temp <= 0 is the argmax.  Weights are
 * normalised by the largest visit count before exponentiating, so nothing
 * overflows and the maximum always has weight 1.  Two passes, no scratch array,
 * so the move count is not capped by this function. */
int mcts_pick(const int32_t *visits, int n, float temp, uint64_t *rng)
{
    if (n <= 0 || !visits) return -1;

    int best = 0;
    for (int i = 1; i < n; i++) if (visits[i] > visits[best]) best = i;

    if (!(temp > 0.0f) || !rng) return best;

    const double mx = (double)visits[best];
    if (!(mx > 0.0)) {                    /* nothing visited: uniform choice */
        return (int)(mc_rng_next(rng) % (uint64_t)n);
    }

    const double inv = 1.0 / (double)temp;

    double total = 0.0;
    for (int i = 0; i < n; i++)
        if (visits[i] > 0) total += pow((double)visits[i] / mx, inv);
    if (!(total > 0.0)) return best;

    double r = mc_rng_u01d(rng) * total;
    for (int i = 0; i < n; i++) {
        if (visits[i] <= 0) continue;
        r -= pow((double)visits[i] / mx, inv);
        if (r <= 0.0) return i;
    }
    return best;                          /* rounding fell off the end */
}

void mcts_target(const int32_t *visits, int n, float *out)
{
    if (n <= 0 || !visits || !out) return;

    double sum = 0.0;
    for (int i = 0; i < n; i++) if (visits[i] > 0) sum += (double)visits[i];

    if (!(sum > 0.0)) {
        const float u = 1.0f / (float)n;
        for (int i = 0; i < n; i++) out[i] = u;
        return;
    }
    const double r = 1.0 / sum;
    for (int i = 0; i < n; i++)
        out[i] = (visits[i] > 0) ? (float)((double)visits[i] * r) : 0.0f;
}
