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
 * rules and net.c for another game's network and this file is unchanged.
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
 * and that expansion sets root.N = 1.  Each of the `sims` simulations then
 * descends from the root through exactly one root child.  Therefore
 *
 *      sum over root children of N  ==  sims        (exactly)
 *      root.N                       ==  sims + 1
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
 */

#include "mcts.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* Hard cap on the length of a selection path.  A tree this deep is already
 * pathological; the cap exists so the path arrays are fixed-size and the search
 * provably terminates even if repetition cycles are searched.  A node at the cap
 * is adjudicated as a draw (value 0), which is the same ply-cap adjudication the
 * game level applies and, like it, expresses no view about chess. */
#define MCTS_MAX_DEPTH   128
/* How much game history is kept for repetition detection.  The fifty-move rule
 * bounds the useful window at 100 plies. */
#define MCTS_HIST_KEEP   128
/* The pool floor: enough for the root plus a full legal move list, so the root
 * always expands and a search always returns a usable move distribution. */
#define MCTS_MIN_NODES   (1 + MAX_MOVES)

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

/* ------------------------------------------------------------- node pool */
/* mcts.h has no field for the pool-exhaustion counter and headers are fixed, so
 * the counter lives in a hidden block immediately BEFORE the node array.  The
 * Mcts struct, its layout and its ABI are untouched; mcts_pool_exhausted()
 * below reads the block.  See the report: this wants to be a field in Mcts. */

#define MCTS_POOL_MAGIC 0x4D435453504F4F4CULL   /* "MCTSPOOL" */

typedef struct {
    uint64_t magic;
    uint64_t exhausted;   /* expansions refused because the pool was full */
    uint64_t reserved;
} PoolHdr;

static PoolHdr *pool_hdr(const Mcts *m)
{
    if (!m || !m->pool) return NULL;
    PoolHdr *hdr = ((PoolHdr *)(void *)m->pool) - 1;
    return (hdr->magic == MCTS_POOL_MAGIC) ? hdr : NULL;
}

/* Number of expansions that could not be performed because the node pool was
 * full, cumulative since mcts_init().  A non-zero value means the search was
 * degraded (it kept evaluating but stopped growing the tree) and the caller
 * should raise max_nodes.  Declared here rather than in mcts.h, which may not
 * be modified. */
uint64_t mcts_pool_exhausted(const Mcts *m)
{
    const PoolHdr *hdr = pool_hdr(m);
    return hdr ? hdr->exhausted : 0;
}

void mcts_defaults(Mcts *m)
{
    if (!m) return;
    m->c_puct          = 1.4f;
    /* First-play urgency: the value assumed for an edge that has never been
     * visited.  0 is the neutral "unknown, assume a draw" choice and, unlike a
     * parent-derived FPU, introduces nothing that has to be tuned per game. */
    m->fpu             = 0.0f;
    m->dirichlet_alpha = 0.3f;
    m->dirichlet_eps   = 0.25f;
}

void mcts_init(Mcts *m, int max_nodes)
{
    if (!m) return;
    memset(m, 0, sizeof *m);
    mcts_defaults(m);

    if (max_nodes < MCTS_MIN_NODES) max_nodes = MCTS_MIN_NODES;

    PoolHdr *hdr = (PoolHdr *)malloc(sizeof(PoolHdr) +
                                     (size_t)max_nodes * sizeof(MctsNode));
    if (!hdr) return;                 /* pool == NULL: mcts_search returns 0 */

    hdr->magic     = MCTS_POOL_MAGIC;
    hdr->exhausted = 0;
    hdr->reserved  = 0;

    m->pool = (MctsNode *)(void *)(hdr + 1);
    m->cap  = max_nodes;
    m->used = 0;
    /* m->evals and m->max_depth_seen accumulate across searches (a running
     * total and a running maximum).  They are plain fields, so a caller that
     * wants per-move telemetry just zeroes them before each search. */
}

void mcts_free(Mcts *m)
{
    if (!m) return;
    if (m->pool) {
        PoolHdr *hdr = pool_hdr(m);
        if (hdr) free(hdr);
        else     free(m->pool);
    }
    m->pool = NULL;
    m->cap  = 0;
    m->used = 0;
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
    const float fpu = m->fpu;

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
        nd->terminal = (int8_t)TR_MAX_PLIES;    nd->tval = 0.0f; return 0.0f;
    }

    /* ---- the network: the only evaluation in this file ----------------- */
    uint16_t fidx[NF_MAXACTIVE];
    const int nf = nn_features(pos, fidx);

    Fwd fw;
    nn_eval(t, h, fidx, nf, &fw);
    m->evals++;

    /* Out of nodes: degrade gracefully.  Keep evaluating, stop growing.  The
     * node stays unexpanded, so it is re-evaluated on every later visit and the
     * search behaves like a 1-ply rollout from here down.  Never crashes, never
     * corrupts the tree. */
    if (m->used + n > m->cap) {
        PoolHdr *hdr = pool_hdr(m);
        if (hdr) hdr->exhausted++;
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

int mcts_search(Mcts *m, const Trunk *t, const Head *h, const Game *g,
                int sims, int root_noise, uint64_t *rng,
                int32_t *visits, float *root_value)
{
    if (root_value) *root_value = 0.0f;
    if (!m || !m->pool || m->cap <= 0 || !t || !h || !g) return 0;
    if (g->result != GR_ONGOING) return 0;       /* the game is already over */

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

    /* Fresh tree every move: no subtree reuse, no stale statistics. */
    m->used = 1;
    MctsNode *root = &m->pool[0];
    root->P        = 1.0f;
    root->W        = 0.0f;
    root->N        = 0;
    root->first    = -1;
    root->nchild   = 0;
    root->terminal = 0;
    root->tval     = 0.0f;
    root->move     = MV_NONE;

    const float v0 = mcts_expand(m, t, h, 0, &pos, keys, nkeys, 0);
    if (root->terminal) return 0;                /* checkmate/stalemate/draw */

    const int nroot = root->nchild;
    if (nroot <= 0 || root->first < 0) {         /* pool too small for a root */
        return 0;
    }
    root->N = 1;
    root->W = v0;

    if (root_noise && rng && m->dirichlet_eps > 0.0f && m->dirichlet_alpha > 0.0f)
        mcts_root_noise(m, root, rng);

    int  path[MCTS_MAX_DEPTH + 1];
    Undo undo[MCTS_MAX_DEPTH + 1];
    const int base_keys = nkeys;

    for (int s = 0; s < sims; s++) {
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
