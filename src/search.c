/* search.c -- MEASUREMENT-ONLY alpha-beta baseline.  NOT the shipped agent.
 *
 * ===========================================================================
 * WHAT THIS FILE IS FOR, AND WHY IT IS NOT THE AGENT
 * ===========================================================================
 * The shipped agent picks its moves with PUCT MCTS over the learned priors
 * (src/mcts.c, driven from api.c and uci.c).  This file is retained for ONE
 * purpose: measurement.  When you want to answer "how much of the strength is
 * the network and how much is the tree search?", you need a second, structurally
 * different searcher over the same network to compare against.  That is all
 * this is.  Nothing selects it by default:
 *
 *   - api.c   -- api_engine_move() uses MCTS.  Alpha-beta is reachable only via
 *                api_engine_set_mode(eid, "alphabeta").
 *   - uci.c   -- the `Engine` option defaults to `mcts`.  Alpha-beta is
 *                reachable only via `setoption name Engine value alphabeta`.
 *
 * ===========================================================================
 * WHAT WAS REMOVED, AND WHY  (docs/FROM_SCRATCH.md is the contract)
 * ===========================================================================
 * The previous version of this file was where essentially all of the old
 * engine's playing strength actually lived, and none of it was learned:
 *
 *   PIECE_CP[] / MAT_CP[] {100,320,330,500,900}  hand-written piece values
 *   material_cp()                                 hand-written material count
 *   blend_cp()  = 0.55*net + 0.45*material        the hand-written term, not
 *                                                 the network, chose the moves
 *   s_see() / s_attackers()                       static exchange evaluation
 *   MVV-LVA capture ordering                      hand-authored "good capture"
 *   order_captures()                              ditto, for quiescence
 *   killer moves, history-with-gravity            hand-authored move preference
 *   quiescence search over captures/promotions    "captures are the noisy moves"
 *   delta pruning (stand_pat + PIECE_CP[QUEEN])   piece values again
 *   reverse futility / quiet futility margins     margins calibrated in the
 *                                                 material centipawn scale
 *   null-move pruning guarded by has_big_piece()  piece-type zugzwang knowledge
 *
 * ALL of it is gone.  What survives is a general-purpose game-tree search whose
 * only two inputs are the network's two heads and the rules of chess:
 *
 *   LEAF EVALUATION  -- the value head, alone.  leaf_cp() is a fixed monotone
 *                       map of v into a "centipawn-ish" integer so the UCI
 *                       `score cp` field and search.h's score_cp stay the same
 *                       shape.  It is presentation, not evaluation: the ordering
 *                       of any two positions is exactly the ordering of their v.
 *   MOVE ORDERING    -- the policy head, alone (plus the transposition table's
 *                       own best move, which is a search result, not knowledge).
 *   TERMINAL VALUES  -- checkmate, stalemate, repetition, fifty-move and dead
 *                       positions, from chess.c.  These are the rules of the
 *                       game; a random mover uses the same facts.
 *
 * Everything between those (iterative deepening, aspiration windows, principal
 * variation search, the transposition table, late move reductions and late move
 * pruning by ordering rank) is domain-independent bandit/alpha-beta machinery.
 * Swap chess.c and net.c for another game's and this file is unchanged.
 *
 * tools/audit_knowledge.sh greps this file along with the rest of the play and
 * learning path.  It is deliberately NOT exempted: a measurement baseline that
 * quietly re-grew a material term would poison every measurement made with it.
 *
 * ===========================================================================
 * CONCURRENCY
 * ===========================================================================
 * Everything mutable lives either in the caller's Search object or in a Ctx
 * allocated per search_best() call, so distinct Search objects are completely
 * independent and may be driven from different threads.  There are no globals
 * and no mutable statics.
 */

#include "search.h"
#include "net.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* ------------------------------------------------------------- constants */

#define S_MAX_PLY     64                    /* hard recursion bound          */
#define S_PV_MAX      64                    /* == sizeof(Search.pv)/2        */
#define S_MATE        30000                 /* mate at ply 0                 */
#define S_MATE_BOUND  (S_MATE - S_MAX_PLY)  /* anything above this is a mate */
#define S_INF         31000

/* Ordering bands.  Strictly decreasing, no overlap.  There are exactly two:
 * the transposition table's move, and then every other move by its policy
 * prior.  There is no third band, because a third band would have to encode
 * an opinion about chess.                                                   */
#define ORD_TT        16777216
#define ORD_POLICY     1000000.0f           /* prior in [0,1] -> 0 .. 1e6    */

/* Leaf evaluation scale.  See leaf_cp(). */
#define EVAL_CLAMP       0.995f
#define EVAL_CP_SCALE  300.0f

/* Late move pruning by ORDERING RANK: at a shallow non-PV node that is not in
 * check and has not yet seen a losing-by-force score, stop after this many
 * moves.  The rank comes from the policy head, so this prunes what the NETWORK
 * thinks is unpromising -- it is not a hand-authored opinion about which moves
 * matter.  Index 0 is unused (depth <= 0 goes straight to a leaf).           */
#define LMP_MAX_DEPTH   3
static const int LMP_COUNT[LMP_MAX_DEPTH + 1] = { 0, 12, 18, 26 };

/* ------------------------------------------------------------- per-search */

typedef struct {
    Search  *s;
    Position pos;

    /* Zobrist keys of every position from the start of the real game up to and
     * including the node currently being searched.  reps[nreps-1] is "now". */
    uint64_t reps[MAX_GAME_PLIES + S_MAX_PLY + 16];
    int      nreps;

    /* Per-ply move lists. */
    Move     mlist[S_MAX_PLY][MAX_MOVES];
    int32_t  mscore[S_MAX_PLY][MAX_MOVES];
    Fwd      fwbuf[S_MAX_PLY];          /* network forward pass, per ply      */

    /* Scratch used only while scoring the moves of a single node (the children
     * are searched afterwards, so a single shared buffer is safe). */
    MoveKey  kbuf[MAX_MOVES];
    float    lbuf[MAX_MOVES];
    float    pbuf[MAX_MOVES];

    Move     pvtab[S_MAX_PLY][S_PV_MAX];
    uint8_t  pvlen[S_MAX_PLY];

    uint8_t  lmr[64][64];               /* [depth][move number] reduction     */

    Move     rootm[MAX_MOVES];
    int      rootsc[MAX_MOVES];
    int      nroot;

    int64_t  t0_ns;
    int64_t  budget_ns;                 /* <= 0 : no time limit               */
} Ctx;

/* ------------------------------------------------------------------ misc */

static int64_t now_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000000000LL + (int64_t)ts.tv_nsec;
}

static inline uint64_t rotl64(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }

/* xoshiro256** on the Search's own state -- no shared RNG anywhere. */
static uint64_t s_rand(uint64_t *st)
{
    const uint64_t r = rotl64(st[1] * 5, 7) * 9;
    const uint64_t t = st[1] << 17;
    st[2] ^= st[0];
    st[3] ^= st[1];
    st[1] ^= st[2];
    st[0] ^= st[3];
    st[2] ^= t;
    st[3] = rotl64(st[3], 45);
    return r;
}

static float s_randf(uint64_t *st)   /* [0,1) */
{
    return (float)((s_rand(st) >> 40) * (1.0 / 16777216.0));
}

static void seed_rng(uint64_t *st, uint64_t seed)
{
    uint64_t z = seed + 0x9E3779B97F4A7C15ull;
    for (int i = 0; i < 4; i++) {
        z += 0x9E3779B97F4A7C15ull;
        uint64_t x = z;
        x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
        x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
        st[i] = x ^ (x >> 31);
    }
    if (!(st[0] | st[1] | st[2] | st[3])) st[0] = 0x1234567890ABCDEFull;
}

static inline int imin(int a, int b) { return a < b ? a : b; }

/* -------------------------------------------------------------- evaluation */

/* Runs the network once.  Returns 1 when the forward pass is valid. */
static int s_forward(const Search *s, const Position *p, Fwd *fw)
{
    uint16_t fidx[NF_MAXACTIVE];
    int nf;
    if (!s->trunk || !s->head) return 0;
    nf = nn_features(p, fidx);
    nn_eval(s->trunk, s->head, fidx, nf, fw);
    return 1;
}

/* THE ONLY EVALUATION IN THIS FILE: the value head, and nothing else.
 *
 * The value head is a tanh, i.e. it saturates: near +-1 it carries no usable
 * gradient of "how much better".  atanh is its exact inverse, so
 *
 *      cp = 300 * atanh(clamp(v, -0.995, +0.995))
 *
 * is strictly monotone in v, maps v = 0 to 0, and bounds the saturated region
 * at about +-900 instead of infinity.  Because it is strictly monotone, the
 * search's preference between any two positions is exactly the network's
 * preference between them: the map changes the units, never the ordering.
 * 300 is an arbitrary presentation constant chosen so the integers land in the
 * range a UCI GUI expects from a `score cp`.  No part of it is derived from
 * piece values -- there are none in this file to derive it from.
 *
 * A position with no network (trunk/head NULL) evaluates to 0: unknown, and
 * therefore a draw, which is the only honest answer.                        */
static int leaf_cp(const Fwd *fw, int have_fw)
{
    float v, cp;
    int r;

    if (!have_fw) return 0;

    v = fw->v;
    if (!isfinite(v)) return 0;
    if (v >  EVAL_CLAMP) v =  EVAL_CLAMP;
    if (v < -EVAL_CLAMP) v = -EVAL_CLAMP;
    cp = EVAL_CP_SCALE * atanhf(v);

    r = (int)lrintf(cp);
    if (r >  20000) r =  20000;      /* stay far below the mate band */
    if (r < -20000) r = -20000;
    return r;
}

int search_eval_cp(Search *s, const Position *p)
{
    Fwd fw;
    int have;
    if (!s || !p) return 0;
    /* A dead position is a draw by the rules, whatever the network thinks. */
    if (insufficient_material(p)) return 0;
    have = s_forward(s, p, &fw);
    return leaf_cp(&fw, have);
}

/* ------------------------------------------------------ repetition / draw */

static inline void push_key(Ctx *c, uint64_t key)
{
    if (c->nreps < (int)(sizeof(c->reps) / sizeof(c->reps[0])))
        c->reps[c->nreps] = key;
    c->nreps++;
}

static inline void pop_key(Ctx *c) { c->nreps--; }

/* A position that has already occurred (anywhere in the real game history or
 * earlier on this search path) is scored as a draw immediately: inside a search
 * one repetition is enough, waiting for the third is pointless.  Zobrist keys
 * include the side to move, so a raw scan over the window is exact. */
static int is_repetition(const Ctx *c)
{
    const uint64_t key = c->pos.key;
    int lo = c->nreps - 1 - (int)c->pos.halfmove;
    int i;
    if (lo < 0) lo = 0;
    for (i = c->nreps - 2; i >= lo; i--)
        if (c->reps[i] == key) return 1;
    return 0;
}

/* ----------------------------------------------------------------- TT --- */

static inline int tt_to_search(int sc, int ply)
{
    if (sc >=  S_MATE_BOUND) return sc - ply;
    if (sc <= -S_MATE_BOUND) return sc + ply;
    return sc;
}

static inline int tt_from_search(int sc, int ply)
{
    if (sc >=  S_MATE_BOUND) return sc + ply;
    if (sc <= -S_MATE_BOUND) return sc - ply;
    return sc;
}

static void tt_store(Search *s, uint64_t key, int score, int depth, int flag,
                     Move best, int ply)
{
    TTEntry *e;
    int sc;
    if (!s->tt || !s->tt_size) return;
    e = &s->tt[key & (s->tt_size - 1)];

    /* Replace-by-depth: a shallower probe never evicts a deeper one for the
     * same position, but a different position always wins the slot (otherwise
     * deep entries from previous moves would wedge the table shut). */
    if (e->key == key && (int)e->depth > depth && flag != 0) {
        if (best != MV_NONE) e->best = best;
        return;
    }
    if (best == MV_NONE && e->key == key) best = e->best;

    sc = tt_from_search(score, ply);
    if (sc >  32000) sc =  32000;
    if (sc < -32000) sc = -32000;

    e->key   = key;
    e->score = (int16_t)sc;
    e->depth = (uint8_t)((depth < 0) ? 0 : (depth > 255 ? 255 : depth));
    e->flag  = (uint8_t)flag;
    e->best  = best;
}

/* ------------------------------------------------------------- ordering -- */

static inline void pick_best(Move *ml, int32_t *sc, int n, int i)
{
    int b = i, j;
    for (j = i + 1; j < n; j++) if (sc[j] > sc[b]) b = j;
    if (b != i) {
        Move    tm = ml[i]; ml[i] = ml[b]; ml[b] = tm;
        int32_t ts = sc[i]; sc[i] = sc[b]; sc[b] = ts;
    }
}

/* Score every move of `p` for ordering.  Exactly two rules:
 *
 *   1. the transposition-table move, if any.  That is a previous SEARCH RESULT
 *      for this very position, not a heuristic about chess.
 *   2. everything else, by the POLICY HEAD's probability.  Nothing is added to
 *      it, nothing is subtracted from it, and no move type is special-cased:
 *      a capture, a promotion and a quiet rook lift are ordered purely by what
 *      the network learned to expect.
 *
 * With no network (`use_policy` 0) every move scores the same and the ordering
 * degenerates to move-generation order, which is the honest fallback.        */
static void order_moves(Ctx *c, const Position *p, Move *ml, int32_t *sc, int n,
                        Move ttm, const Fwd *fw, int use_policy)
{
    int i;

    if (use_policy && n > 1) {
        for (i = 0; i < n; i++) nn_move_key(p, ml[i], &c->kbuf[i]);
        nn_logits(c->s->head, fw, c->kbuf, n, c->lbuf);
        softmax_t(c->lbuf, n, 1.0f, c->pbuf);
    }

    for (i = 0; i < n; i++) {
        if (ml[i] == ttm) { sc[i] = ORD_TT; continue; }
        sc[i] = (use_policy && n > 1 && isfinite(c->pbuf[i]))
              ? (int32_t)(ORD_POLICY * c->pbuf[i])
              : 0;
    }
}

/* ------------------------------------------------------------------ PV --- */

static void pv_store(Ctx *c, int ply, Move m)
{
    int cl = 0;
    if (ply + 1 < S_MAX_PLY) {
        cl = c->pvlen[ply + 1];
        if (cl > S_PV_MAX - 1) cl = S_PV_MAX - 1;
        if (cl > 0) memcpy(&c->pvtab[ply][1], c->pvtab[ply + 1], (size_t)cl * sizeof(Move));
    }
    c->pvtab[ply][0] = m;
    c->pvlen[ply] = (uint8_t)(cl + 1);
}

/* --------------------------------------------------------------- limits -- */

/* The limits apply from the very first node; they deliberately do NOT wait for
 * depth 1 to finish.  search_best() keeps a legal move in hand from before the
 * first node, so an iteration that is cut short costs quality, never
 * correctness.  Every node -- interior and leaf -- passes through the counter,
 * so the poll interval bounds the overrun in real work, not just in branches. */
static void check_limits(Ctx *c)
{
    Search *s = c->s;
    if (s->stop) return;
    if (s->max_nodes && s->nodes >= s->max_nodes) { s->stop = 1; return; }
    if (c->budget_ns > 0 && now_ns() - c->t0_ns >= c->budget_ns) s->stop = 1;
}

/* ------------------------------------------------------------- negamax -- */

static int negamax(Ctx *c, int depth, int alpha, int beta, int ply)
{
    Search *s = c->s;
    Position *p = &c->pos;
    Fwd *fw = &c->fwbuf[ply];
    const int is_pv = (beta - alpha) > 1;
    int in_chk, have_fw = 0;
    int best = -S_INF, orig_alpha, n, i, lmp = 0;
    Move ttm = MV_NONE, bestm = MV_NONE;
    Move *ml;
    int32_t *sa;

    c->pvlen[ply] = 0;
    if (s->stop) return 0;

    s->nodes++;
    if ((s->nodes & 511) == 0) { check_limits(c); if (s->stop) return 0; }

    /* ---- draws and mate distance.  Rules of chess, never the network. --- */
    if (ply > 0) {
        if (is_repetition(c) || insufficient_material(p)) return 0;
        /* mate-distance pruning: a mate found closer to the root always wins */
        if (alpha < -S_MATE + ply)     alpha = -S_MATE + ply;
        if (beta  >  S_MATE - ply - 1) beta  =  S_MATE - ply - 1;
        if (alpha >= beta) return alpha;
    }
    orig_alpha = alpha;      /* after mate-distance pruning, so the TT flag is right */

    if (depth <= 0 || ply >= S_MAX_PLY - 2) {
        if (insufficient_material(p)) return 0;
        have_fw = s_forward(s, p, fw);
        return leaf_cp(fw, have_fw);
    }

    /* --- transposition table ------------------------------------------- */
    if (s->tt && s->tt_size) {
        const TTEntry e = s->tt[p->key & (s->tt_size - 1)];
        if (e.key == p->key) {
            ttm = e.best;
            if (!is_pv && ply > 0 && (int)e.depth >= depth) {
                const int sc = tt_to_search((int)e.score, ply);
                if (e.flag == 0) return sc;
                if (e.flag == 1 && sc >= beta)  return sc;
                if (e.flag == 2 && sc <= alpha) return sc;
            }
        }
    }

    /* --- moves ---------------------------------------------------------- */
    ml = c->mlist[ply];
    sa = c->mscore[ply];
    n  = gen_legal(p, ml);
    in_chk = in_check(p, p->side);
    if (n == 0) return in_chk ? (-S_MATE + ply) : 0;   /* checkmate / stalemate */
    if (p->halfmove >= 100) return 0;                  /* fifty-move, not mate  */

    /* One network evaluation per node, used for the priors. */
    have_fw = s_forward(s, p, fw);
    order_moves(c, p, ml, sa, n, ttm, fw, have_fw);

    /* Late move pruning by ordering rank.  Never in the PV, never in check,
     * and never once a forced loss is on the board (which is what keeps a mate
     * proof from being pruned away: while every move searched so far loses by
     * force, `best` stays inside the mate band and nothing is skipped). */
    lmp = n;
    if (!is_pv && !in_chk && depth <= LMP_MAX_DEPTH)
        lmp = LMP_COUNT[depth];

    for (i = 0; i < n; i++) {
        Undo u;
        Move m;
        int gives_check, new_depth, r = 0, sc;

        if (i >= lmp && best > -S_MATE_BOUND) break;

        pick_best(ml, sa, n, i);
        m = ml[i];

        make_move(p, m, &u);
        push_key(c, p->key);
        gives_check = in_check(p, p->side);
        new_depth = depth - 1;

        /* --- late move reductions ---------------------------------------
         * Purely a function of (depth, ordering rank).  It says "the network
         * ranked this move 14th, look at it less hard", which is a statement
         * about the priors, not about chess.  A move that gives check or is
         * played out of check is not reduced, because in both cases the reply
         * set is tiny and the reduction buys nothing. */
        if (depth >= 3 && i >= 3 && !in_chk && !gives_check) {
            r = c->lmr[imin(depth, 63)][imin(i, 63)];
            if (is_pv) r--;
            if (r < 0) r = 0;
            if (r > new_depth - 1) r = new_depth - 1;
        }

        if (i == 0) {
            sc = -negamax(c, new_depth, -beta, -alpha, ply + 1);
        } else {
            sc = -negamax(c, new_depth - r, -alpha - 1, -alpha, ply + 1);
            if (sc > alpha && r > 0)
                sc = -negamax(c, new_depth, -alpha - 1, -alpha, ply + 1);
            if (sc > alpha && sc < beta)
                sc = -negamax(c, new_depth, -beta, -alpha, ply + 1);
        }

        pop_key(c);
        unmake_move(p, m, &u);
        if (s->stop) return best > -S_INF ? best : alpha;

        if (sc > best) {
            best  = sc;
            bestm = m;
            if (sc > alpha) {
                alpha = sc;
                pv_store(c, ply, m);
                if (alpha >= beta) break;
            }
        }
    }

    tt_store(s, p->key, best, depth,
             (best >= beta) ? 1 : (best > orig_alpha ? 0 : 2), bestm, ply);
    return best;
}

/* ---------------------------------------------------------------- root --- */

static int root_search(Ctx *c, int depth, int alpha, int beta, Move *bestm)
{
    Search *s = c->s;
    Position *p = &c->pos;
    const int n = c->nroot;
    int best = -S_INF, i;

    *bestm = c->rootm[0];
    for (i = 0; i < n; i++) c->rootsc[i] = -S_INF;

    /* Every root move is searched, always: no pruning and no reduction may
     * remove a candidate from the answer itself. */
    for (i = 0; i < n; i++) {
        const Move m = c->rootm[i];
        Undo u;
        int gives_check, sc, r = 0;

        make_move(p, m, &u);
        push_key(c, p->key);
        gives_check = in_check(p, p->side);

        if (i == 0) {
            sc = -negamax(c, depth - 1, -beta, -alpha, 1);
        } else {
            if (depth >= 3 && i >= 4 && !gives_check)
                r = 1 + (i >= 12 && depth >= 5);
            if (r > depth - 2) r = depth - 2;
            if (r < 0) r = 0;
            sc = -negamax(c, depth - 1 - r, -alpha - 1, -alpha, 1);
            if (sc > alpha && r > 0)
                sc = -negamax(c, depth - 1, -alpha - 1, -alpha, 1);
            if (sc > alpha && sc < beta)
                sc = -negamax(c, depth - 1, -beta, -alpha, 1);
        }

        pop_key(c);
        unmake_move(p, m, &u);
        if (s->stop) return best;

        c->rootsc[i] = sc;
        if (sc > best) {
            best = sc;
            *bestm = m;
            if (sc > alpha) {
                alpha = sc;
                pv_store(c, 0, m);
                if (sc >= beta) return best;   /* aspiration fail-high        */
            }
        }
    }
    return best;
}

/* Move `m` to the front of the root list, keeping every other move in its
 * existing (policy) order.
 *
 * This deliberately does NOT sort the root by the iteration's scores.  In a
 * PVS root every move after the first is searched with a null window, so its
 * score is an UPPER BOUND that has failed low -- a "not better than alpha"
 * token, not a measurement.  Sorting on those bounds makes the root order, and
 * therefore the tie-break between equally-valued moves, depend on search noise:
 * with the network value head as the only evaluation, whole blocks of moves
 * legitimately score the same, so that noise decides the answer.  Promoting the
 * principal move and leaving the rest in the policy's order keeps the ordering
 * reproducible while still putting the best candidate first. */
static void promote_root(Move *rm, int n, Move m)
{
    int i, j;
    if (m == MV_NONE || n <= 1) return;
    for (i = 0; i < n; i++) if (rm[i] == m) break;
    if (i <= 0 || i >= n) return;
    for (j = i; j > 0; j--) rm[j] = rm[j - 1];
    rm[0] = m;
}

/* ------------------------------------------------------------------ API -- */

void search_init(Search *s, const Trunk *t, const Head *h, size_t tt_mb)
{
    size_t entries, pow2 = 1024;

    if (!s) return;
    memset(s, 0, sizeof(*s));
    s->trunk       = t;
    s->head        = h;
    s->max_depth   = 64;
    s->movetime_ms = 1000;
    s->max_nodes   = 0;
    s->blunder_rate = 0.0f;
    seed_rng(s->rng, (uint64_t)(uintptr_t)s * 0x9E3779B97F4A7C15ull ^ (uint64_t)now_ns());

    if (tt_mb > 4096) tt_mb = 4096;
    entries = (tt_mb * (size_t)1024 * 1024) / sizeof(TTEntry);
    while (pow2 * 2 <= entries) pow2 *= 2;

    s->tt = (TTEntry *)calloc(pow2, sizeof(TTEntry));
    s->tt_size = s->tt ? pow2 : 0;
}

void search_free(Search *s)
{
    if (!s) return;
    free(s->tt);
    s->tt = NULL;
    s->tt_size = 0;
}

Move search_best(Search *s, const Game *g)
{
    Ctx *c;
    Move best_move = MV_NONE, chosen;
    int maxd, d, prev = 0, completed = 0, i, j;

    if (!s || !g) return MV_NONE;

    s->nodes = 0;
    s->depth_reached = 0;
    s->score_cp = 0;
    s->pv_len = 0;
    s->pv[0] = MV_NONE;
    s->stop = 0;
    if (!(s->rng[0] | s->rng[1] | s->rng[2] | s->rng[3]))
        seed_rng(s->rng, (uint64_t)now_ns());

    if (g->result != GR_ONGOING) return MV_NONE;

    c = (Ctx *)calloc(1, sizeof(Ctx));
    if (!c) {                                   /* degrade gracefully */
        Move ml[MAX_MOVES];
        int n = gen_legal(&g->pos, ml);
        return n > 0 ? ml[0] : MV_NONE;
    }
    c->s = s;
    c->pos = g->pos;

    c->nroot = gen_legal(&c->pos, c->rootm);
    if (c->nroot == 0) { free(c); return MV_NONE; }
    best_move = c->rootm[0];

    /* game history -> repetition stack (hist[hist_len-1] is the current pos) */
    {
        int hl = g->hist_len;
        if (hl < 1) hl = 0;
        if (hl > MAX_GAME_PLIES + 1) hl = MAX_GAME_PLIES + 1;
        for (i = 0; i < hl; i++) c->reps[i] = g->hist[i];
        c->nreps = hl;
        if (c->nreps == 0) { c->reps[0] = c->pos.key; c->nreps = 1; }
    }

    /* LMR table, built per search so nothing mutable is shared */
    for (i = 0; i < 64; i++)
        for (j = 0; j < 64; j++)
            c->lmr[i][j] = (i < 3 || j < 3)
                ? 0
                : (uint8_t)(0.75 + log((double)i) * log((double)j) / 2.25);

    c->t0_ns     = now_ns();
    c->budget_ns = (s->movetime_ms > 0) ? (int64_t)s->movetime_ms * 1000000LL : 0;

    maxd = s->max_depth;
    if (maxd <= 0) maxd = (s->movetime_ms > 0 || s->max_nodes) ? S_MAX_PLY - 4 : 8;
    if (maxd > S_MAX_PLY - 4) maxd = S_MAX_PLY - 4;

    /* Order the root once up front, by the policy head alone. */
    {
        Fwd fw;
        int have = s_forward(s, &c->pos, &fw);
        order_moves(c, &c->pos, c->rootm, c->rootsc, c->nroot, MV_NONE, &fw, have);
        /* Stable descending sort by prior: the one and only root ordering. */
        for (i = 1; i < c->nroot; i++) {
            const Move  m  = c->rootm[i];
            const int32_t v = c->rootsc[i];
            int k = i - 1;
            while (k >= 0 && c->rootsc[k] < v) {
                c->rootm[k + 1]  = c->rootm[k];
                c->rootsc[k + 1] = c->rootsc[k];
                k--;
            }
            c->rootm[k + 1]  = m;
            c->rootsc[k + 1] = v;
        }
        for (i = 0; i < c->nroot; i++) c->rootsc[i] = -S_INF;
        best_move = c->rootm[0];       /* best guess before a single node runs */
    }

    /* -------------------------------------------------- iterative deepening */
    for (d = 1; d <= maxd; d++) {
        int alpha = -S_INF, beta = S_INF, delta = 24, score;
        Move itm = MV_NONE;

        if (d >= 4 && prev > -S_MATE_BOUND && prev < S_MATE_BOUND) {
            alpha = prev - delta;
            beta  = prev + delta;
        }

        for (;;) {
            score = root_search(c, d, alpha, beta, &itm);
            if (s->stop) break;
            if (score <= alpha) {
                beta  = (alpha + beta) / 2;
                alpha = (score - delta < -S_INF) ? -S_INF : score - delta;
                delta += delta / 2 + 12;
            } else if (score >= beta) {
                beta = (score + delta > S_INF) ? S_INF : score + delta;
                delta += delta / 2 + 12;
            } else {
                break;
            }
            if (alpha <= -S_INF && beta >= S_INF) break;
        }

        /* Iteration incomplete: discard its score, but a partial root pass has
         * still measured some moves, so keep its best-so-far over the raw
         * move-ordering guess. */
        if (s->stop) {
            if (completed == 0 && itm != MV_NONE) best_move = itm;
            break;
        }

        prev = score;
        best_move = itm;
        completed = d;
        s->depth_reached = d;
        s->score_cp = score;
        s->pv_len = imin(c->pvlen[0], S_PV_MAX);
        for (i = 0; i < s->pv_len; i++) s->pv[i] = c->pvtab[0][i];
        if (s->pv_len == 0) { s->pv[0] = best_move; s->pv_len = 1; }

        promote_root(c->rootm, c->nroot, best_move);

        if (score >= S_MATE_BOUND || score <= -S_MATE_BOUND) break;  /* mate    */
        if (c->budget_ns > 0 && (now_ns() - c->t0_ns) * 2 >= c->budget_ns) break;
        if (s->max_nodes && s->nodes >= s->max_nodes) break;
    }

    if (completed == 0) {                    /* cut off early; stay legal      */
        s->depth_reached = 1;
        s->pv[0] = best_move;
        s->pv_len = 1;
    }

    /* ------------------------------------------------------------ blunder --
     * The UI's difficulty slider.  With probability blunder_rate we play a
     * uniformly random move out of the better-scoring half of the root moves
     * instead of the best one -- always a legal move.  It is a strength dial
     * over the search's OWN ranking and contains no chess knowledge. */
    chosen = best_move;
    if (s->blunder_rate > 0.0f && c->nroot > 1 && s_randf(s->rng) < s->blunder_rate) {
        int half = c->nroot / 2;
        int pick;
        if (half < 1) half = 1;
        pick = (int)(s_rand(s->rng) % (uint64_t)half);
        chosen = c->rootm[pick];
        if (c->rootsc[pick] > -S_INF) s->score_cp = c->rootsc[pick];
        s->pv[0] = chosen;
        s->pv_len = 1;
    }

    free(c);
    return chosen;
}
