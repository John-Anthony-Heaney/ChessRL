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
 * folded in.  Invalidation is a generation counter, so a flush is O(1) and
 * happens on any change of trunk, head or weights.
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

/* Fills fw->q and fw->v, from the cache when the position has been seen under
 * these weights and from the network otherwise.
 *
 * ONLY q and v are written.  nn_logits() reads fw->q and nothing else, and the
 * search uses fw->v; the rest of Fwd (acc, h1, z2, h2, raw_v) is the training
 * path's business and is left alone here.  A cached result is therefore
 * bit-for-bit what nn_eval() would have produced for everything this file
 * goes on to compute. */
static void mc_evaluate(Mcts *m, const Trunk *t, const Head *h,
                        const Position *pos, Fwd *fw)
{
    struct MctsCache *ca = m->cache ? m->ecache : NULL;
    uint64_t  tag  = 0;
    McEntry  *slot = NULL;

    if (ca) {
        tag  = mc_pos_key(pos);
        slot = &ca->e[(uint32_t)tag & ca->mask];
        if (slot->gen == ca->gen && slot->tag == tag) {
            memcpy(fw->q, slot->q, sizeof slot->q);
            fw->v = slot->v;
            m->cache_hits++;
            return;
        }
        m->cache_misses++;
    }

    uint16_t fidx[NF_MAXACTIVE];
    const int nf = nn_features(pos, fidx);
    nn_eval(t, h, fidx, nf, fw);
    m->evals++;

    if (slot) {                       /* direct-mapped: the newest wins */
        slot->tag = tag;
        slot->gen = ca->gen;
        slot->v   = fw->v;
        memcpy(slot->q, fw->q, sizeof slot->q);
    }
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

    m->cache_bits = mc_cache_bits_for(max_nodes);
    {
        const size_t n = (size_t)1 << m->cache_bits;
        struct MctsCache *ca = (struct MctsCache *)malloc(sizeof *ca);
        if (ca) {
            ca->e = (McEntry *)calloc(n, sizeof(McEntry));
            if (!ca->e) { free(ca); ca = NULL; }
            else { ca->mask = (uint32_t)(n - 1); ca->gen = 1; }
        }
        m->ecache = ca;                /* NULL simply means "no cache" */
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
    if (m->ecache) { free(m->ecache->e); free(m->ecache); }
    m->pool       = NULL;
    m->remap      = NULL;
    m->ecache     = NULL;
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

/* Evaluates a leaf and, unless it is terminal, expands it.
 *
 * Returns the node's value from ITS OWN side-to-move point of view:
 *   - terminal: +/-1 for a decided result, 0 for any draw, cached in tval so
 *     later visits cost nothing and never touch the network;
 *   - otherwise: the value head's v.
 */
static float mcts_expand(Mcts *m, const Trunk *t, const Head *h, int ni,
                         const Position *pos, const uint64_t *keys, int nkeys,
                         int depth)
{
    MctsNode *nd = &m->pool[ni];

    Move list[MAX_MOVES];
    const int n = gen_legal(pos, list);          /* once per expanded node */

    /* ---- terminal states.  Rules of chess, never the network. ---------- */
    if (n == 0) {
        if (in_check(pos, pos->side)) {
            nd->terminal = (int8_t)TR_CHECKMATE;
            nd->tval     = -1.0f;                /* the side to move is mated */
        } else {
            nd->terminal = (int8_t)TR_STALEMATE;
            nd->tval     = 0.0f;                 /* a draw, NOT a win        */
        }
        return nd->tval;
    }
    if (insufficient_material(pos)) {
        nd->terminal = (int8_t)TR_INSUFFICIENT; nd->tval = 0.0f; return 0.0f;
    }
    if (pos->halfmove >= 100) {
        nd->terminal = (int8_t)TR_FIFTY;        nd->tval = 0.0f; return 0.0f;
    }
    if (mcts_rep_count(keys, nkeys, pos->key, (int)pos->halfmove) >= 3) {
        nd->terminal = (int8_t)TR_REPETITION;   nd->tval = 0.0f; return 0.0f;
    }
    if (depth >= MCTS_MAX_DEPTH) {
        /* The ply cap is measured from the ROOT, so this adjudication would not
         * survive a re-rooting: the same node sits a ply or two higher in the
         * next search, where the cap does not apply.  Rather than carry a stale
         * adjudication, mark the tree and refuse to inherit it -- which keeps
         * "an inherited tree is exactly a fresh one" true without exception. */
        nd->terminal = (int8_t)TR_MAX_PLIES;    nd->tval = 0.0f;
        m->tree_capped = 1;
        return 0.0f;
    }

    /* ---- the network: the only evaluation in this file ----------------- */
    Fwd fw;
    mc_evaluate(m, t, h, pos, &fw);              /* q and v, cached */

    /* Out of nodes: degrade gracefully.  Keep evaluating, stop growing.  The
     * node stays unexpanded, so it is re-evaluated on every later visit and the
     * search behaves like a 1-ply rollout from here down.  Never crashes, never
     * corrupts the tree. */
    if (m->used + n > m->cap) {
        m->pool_exhausted++;
        return fw.v;
    }

    MoveKey mk[MAX_MOVES];
    float   logits[MAX_MOVES];
    float   pri[MAX_MOVES];

    for (int i = 0; i < n; i++) nn_move_key(pos, list[i], &mk[i]);
    nn_logits(h, &fw, mk, n, logits);
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
        c->move     = list[i];
    }
    return fw.v;
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

int mcts_search(Mcts *m, const Trunk *t, const Head *h, const Game *g,
                int sims, int root_noise, uint64_t *rng,
                int32_t *visits, float *root_value)
{
    if (root_value) *root_value = 0.0f;
    if (!m || !m->pool || m->cap <= 0 || !t || !h || !g) return 0;
    if (g->result != GR_ONGOING) return 0;       /* the game is already over */

    /* A tree and a cached evaluation are only valid for the weights that made
     * them.  Drop both the moment the agent or the weights change. */
    {
        const uint64_t st = mc_weight_stamp(t, h);
        if (st != m->wstamp) { mcts_cache_clear(m); m->wstamp = st; }
    }

    /* The ONE Position copy of the whole search.  From here on the position is
     * maintained incrementally by make_move / unmake_move. */
    Position pos = g->pos;

    /* [ game history tail | selection path ], keys[nkeys-1] == current key. */
    uint64_t keys[MCTS_HIST_KEEP + MCTS_MAX_DEPTH + 2];
    int nkeys = 0;
    if (g->hist_len > 0) {
        int keep = g->hist_len;
        if (keep > (int)pos.halfmove + 1) keep = (int)pos.halfmove + 1;
        if (keep > MCTS_HIST_KEEP)        keep = MCTS_HIST_KEEP;
        if (keep < 1)                     keep = 1;
        for (int i = g->hist_len - keep; i < g->hist_len; i++) keys[nkeys++] = g->hist[i];
    }
    if (nkeys == 0 || keys[nkeys - 1] != pos.key) {   /* Game never pushed */
        nkeys = 0;
        keys[nkeys++] = pos.key;
    }

    /* Inherit the subtree under the moves that were played, if there provably
     * is one; otherwise a fresh tree, exactly as before. */
    const int reused = mcts_try_reuse(m, g, keys, nkeys, sims, root_noise);
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

        const float v0 = mcts_expand(m, t, h, 0, &pos, keys, nkeys, 0);
        if (root->terminal) {                    /* checkmate/stalemate/draw */
            m->tree_valid = 0;
            return 0;
        }
        if (root->nchild <= 0 || root->first < 0) {   /* pool too small       */
            m->tree_valid = 0;
            return 0;
        }
        root->N = 1;
        root->W = v0;
    } else {
        /* pool[0] is a former child, and its EDGE fields -- the prior for the
         * move into it, and the move itself -- belong to a parent that is no
         * longer there.  Nothing reads them at a root, but leave a root looking
         * like a root for anyone who walks the pool. */
        root->P    = 1.0f;
        root->move = MV_NONE;
    }

    const int nroot = root->nchild;

    int noised = 0;
    if (root_noise && rng && m->dirichlet_eps > 0.0f && m->dirichlet_alpha > 0.0f) {
        mcts_root_noise(m, root, rng);
        noised = 1;
    }
    /* Whether the STANDING tree's root priors carry noise.  A reused root is a
     * former child or grandchild, whose own children the noise never reached,
     * so this is simply what was just applied. */
    m->tree_noised = (int8_t)noised;

    int  path[MCTS_MAX_DEPTH + 1];
    Undo undo[MCTS_MAX_DEPTH + 1];
    const int base_keys = nkeys;

    /* `sims` is the TOTAL budget, so an inherited subtree is topped up rather
     * than added to: sum(child visits) == sims and root.N == sims + 1 either
     * way, and nothing downstream can tell that reuse happened. */
    int todo = sims + 1 - root->N;
    if (todo < 0) todo = 0;

    for (int s = 0; s < todo; s++) {
        int   depth = 0;
        float value;
        path[0] = 0;

        /* ---- SELECT, then EXPAND + EVALUATE at the leaf ---------------- */
        for (;;) {
            MctsNode *nd = &m->pool[path[depth]];

            if (nd->terminal) {                  /* cached: no network, no rules */
                value = nd->tval;
                break;
            }
            if (nd->first < 0) {                 /* unexpanded leaf */
                value = mcts_expand(m, t, h, path[depth], &pos, keys, nkeys, depth);
                break;
            }

            const int ci = nd->first + mcts_select(m, nd);
            make_move(&pos, m->pool[ci].move, &undo[depth]);
            keys[nkeys++] = pos.key;
            path[++depth] = ci;
        }

        if (depth > m->max_depth_seen) m->max_depth_seen = depth;

        /* ---- BACKUP: negate at every ply, the players alternate -------- */
        for (int d = depth; d >= 0; d--) {
            MctsNode *nd = &m->pool[path[d]];
            nd->N += 1;
            nd->W += value;
            value  = -value;
        }

        /* ---- unwind the position back to the root ---------------------- */
        for (int d = depth; d >= 1; d--)
            unmake_move(&pos, m->pool[path[d]].move, &undo[d - 1]);
        nkeys = base_keys;
    }

    if (visits) {
        const MctsNode *ch = m->pool + root->first;
        for (int i = 0; i < nroot; i++) visits[i] = ch[i].N;
    }
    if (root_value) *root_value = (root->N > 0) ? (root->W / (float)root->N) : 0.0f;

    /* Hand this tree to the next search, together with everything it needs to
     * prove the tree belongs to the position it is then given. */
    m->rpos   = g->pos;
    m->nrkeys = base_keys;
    memcpy(m->rkeys, keys, (size_t)base_keys * sizeof keys[0]);
    m->rply       = g->ply;
    m->tree_valid = 1;

    return nroot;
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
