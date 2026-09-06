/* search.c -- alpha-beta engine used when a human plays the champion.
 *
 * This file is NOT on the training hot path (training samples straight from the
 * policy head), so it optimises for playing strength and for answering inside a
 * fixed movetime rather than for raw node throughput.
 *
 * Structure
 *   iterative deepening
 *     + aspiration windows around the previous iteration's score
 *     + principal variation search (null-window scouts, full-window re-search)
 *     + transposition table (power-of-two entries, replace-by-depth)
 *     + null-move pruning, reverse futility, quiet futility, SEE pruning
 *     + late move reductions, check extensions
 *     + killers, history-with-gravity, and -- the big win here -- the network
 *       POLICY head as the primary ordering key for quiet moves
 *     + quiescence over captures / queen promotions with stand-pat + delta pruning
 *
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

#define SBIT(x)       (1ULL << (x))

/* Ordering bands.  Strictly decreasing, no overlap:
 *   TT  >  good captures/promos  >  killers  >  quiets  >  losing captures    */
#define ORD_TT        16777216
#define ORD_GOODCAP    8388608
#define ORD_KILLER1    7000000
#define ORD_KILLER2    6900000
#define ORD_QUIET      1000000     /* + policy (0..600k) + history (0..262k)  */
#define ORD_BADCAP      100000     /* + see (negative)                        */

#define POLICY_SCALE   600000.0f
#define HIST_MAX         16384
#define HIST_ORD_SCALE      16     /* (h + HIST_MAX) * 16 -> 0 .. 524288      */

/* Static piece values, centipawns.  KING is huge so that SEE never treats a
 * king capture as a real gain (the recapture then dominates the swap list). */
static const int PIECE_CP[NPIECES] = { 100, 320, 330, 500, 900, 20000 };
/* Material-count values: the king must not contribute to the balance. */
static const int MAT_CP[NPIECES]   = { 100, 320, 330, 500, 900, 0 };

/* Weighting of the two halves of the leaf evaluation (see search_eval_cp). */
#define EVAL_W_NET  0.55f
#define EVAL_W_MAT  0.45f

/* ------------------------------------------------------------- per-search */

typedef struct {
    Search  *s;
    Position pos;

    /* Zobrist keys of every position from the start of the real game up to and
     * including the node currently being searched.  reps[nreps-1] is "now". */
    uint64_t reps[MAX_GAME_PLIES + S_MAX_PLY + 16];
    int      nreps;

    /* Per-ply move lists.  negamax(ply) and qsearch(ply) never coexist, so one
     * slot per ply is enough for both. */
    Move     mlist[S_MAX_PLY][MAX_MOVES];
    int32_t  mscore[S_MAX_PLY][MAX_MOVES];
    Fwd      fwbuf[S_MAX_PLY];          /* network forward pass, per ply      */

    /* Scratch used only while scoring the moves of a single node (the children
     * are searched afterwards, so a single shared buffer is safe). */
    MoveKey  kbuf[MAX_MOVES];
    float    lbuf[MAX_MOVES];
    float    pbuf[MAX_MOVES];

    Move     killer[S_MAX_PLY][2];
    int32_t  hist[NCOLORS][64][64];

    Move     pvtab[S_MAX_PLY][S_PV_MAX];
    uint8_t  pvlen[S_MAX_PLY];

    uint8_t  lmr[64][64];               /* [depth][move number] reduction     */

    Move     rootm[MAX_MOVES];
    int      rootsc[MAX_MOVES];
    int      nroot;

    int64_t  t0_ns;
    int64_t  budget_ns;                 /* <= 0 : no time limit               */
    int      allow_stop;                /* 0 until depth 1 has completed      */
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
static inline int imax(int a, int b) { return a > b ? a : b; }

/* -------------------------------------------------------------- evaluation */

static int material_cp(const Position *p)
{
    int sc = 0;
    for (int pt = PAWN; pt <= QUEEN; pt++)
        sc += MAT_CP[pt] * (bb_count(p->piece[WHITE][pt]) - bb_count(p->piece[BLACK][pt]));
    return (p->side == WHITE) ? sc : -sc;
}

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

/* Value head -> centipawns.
 *
 * The value head is a tanh, i.e. it saturates: near +-1 it carries no usable
 * gradient of "how much better".  atanh is its exact inverse, so
 *
 *      cp_net = 300 * atanh(clamp(v, -0.995, +0.995))
 *
 * is monotone in v, maps v = 0 to 0, gives ~+-900cp (about a queen) at the
 * clamp, and turns the network's saturated region into a bounded score instead
 * of an infinite one.  300 is chosen so that v = 0.55 ("clearly better")
 * lands around +185cp, i.e. roughly two pawns -- a scale that mixes sensibly
 * with real material.
 *
 * The blend with plain material exists because the value head of a partially
 * trained agent is often wrong by a piece.  Material alone would play like a
 * beginner; the network alone would occasionally leave a queen en prise
 * because it "feels" fine about the position.  0.55/0.45 keeps the network in
 * charge of positional judgement while making a hung queen cost ~400cp, which
 * no amount of network optimism can hide.
 */
static int blend_cp(const Position *p, const Fwd *fw, int have_fw)
{
    const int mat = material_cp(p);
    float v, cp;
    int r;

    if (insufficient_material(p)) return 0;
    if (!have_fw) return mat;

    v = fw->v;
    if (v >  0.995f) v =  0.995f;
    if (v < -0.995f) v = -0.995f;
    cp = EVAL_W_NET * (300.0f * atanhf(v)) + EVAL_W_MAT * (float)mat;

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
    have = s_forward(s, p, &fw);
    return blend_cp(p, &fw, have);
}

/* --------------------------------------------------------------- SEE ---- */

static uint64_t s_attackers(const Position *p, int sq, uint64_t occ)
{
    uint64_t a;
    const uint64_t bq = p->piece[WHITE][BISHOP] | p->piece[BLACK][BISHOP] |
                        p->piece[WHITE][QUEEN]  | p->piece[BLACK][QUEEN];
    const uint64_t rq = p->piece[WHITE][ROOK]   | p->piece[BLACK][ROOK] |
                        p->piece[WHITE][QUEEN]  | p->piece[BLACK][QUEEN];

    /* attacks_pawn(sq, C) is what a pawn OF colour C standing on sq hits, so
     * the white pawns hitting sq are exactly those on attacks_pawn(sq, BLACK). */
    a  = attacks_pawn(sq, BLACK) & p->piece[WHITE][PAWN];
    a |= attacks_pawn(sq, WHITE) & p->piece[BLACK][PAWN];
    a |= attacks_knight(sq) & (p->piece[WHITE][KNIGHT] | p->piece[BLACK][KNIGHT]);
    a |= attacks_king(sq)   & (p->piece[WHITE][KING]   | p->piece[BLACK][KING]);
    a |= attacks_bishop(sq, occ) & bq;
    a |= attacks_rook(sq, occ)   & rq;
    return a & occ;
}

/* Static exchange evaluation of a capture / promotion, in centipawns, from the
 * moving side's point of view.  Classic swap-list with x-ray updates. */
static int s_see(const Position *p, Move m)
{
    const int from = MV_FROM(m), to = MV_TO(m), fl = MV_FLAG(m);
    const uint64_t bq = p->piece[WHITE][BISHOP] | p->piece[BLACK][BISHOP] |
                        p->piece[WHITE][QUEEN]  | p->piece[BLACK][QUEEN];
    const uint64_t rq = p->piece[WHITE][ROOK]   | p->piece[BLACK][ROOK] |
                        p->piece[WHITE][QUEEN]  | p->piece[BLACK][QUEEN];
    int gain[34];
    int d = 0, stm = p->side, moving = p->board[from];
    uint64_t occ = p->all, attackers;

    if (moving == NO_PIECE) return 0;

    gain[0] = 0;
    if (fl == MF_EP) {
        gain[0] = PIECE_CP[PAWN];
        occ ^= SBIT(to - ((stm == WHITE) ? 8 : -8));
    } else if (p->board[to] != NO_PIECE) {
        gain[0] = PIECE_CP[p->board[to]];
    }
    if (MV_IS_PROMO(m)) {
        const int pp = MV_PROMO_PIECE(m);
        gain[0] += PIECE_CP[pp] - PIECE_CP[PAWN];
        moving = pp;
    }

    occ ^= SBIT(from);
    occ |= SBIT(to);                       /* the mover now stands on `to`   */
    attackers = s_attackers(p, to, occ);
    stm ^= 1;

    for (;;) {
        uint64_t mine = attackers & p->occ[stm] & occ;
        uint64_t b = 0;
        int pt = PAWN, mx;
        if (!mine) break;
        for (; pt <= KING; pt++) { b = mine & p->piece[stm][pt]; if (b) break; }
        if (!b) break;

        d++;
        gain[d] = PIECE_CP[moving] - gain[d - 1];
        if (d >= 32) break;
        mx = imax(-gain[d - 1], gain[d]);
        if (mx < 0) break;                 /* neither side wants to continue */

        occ ^= SBIT(bb_lsb(b));
        moving = pt;
        /* only sliders can be discovered behind the piece we just removed */
        attackers |= (attacks_bishop(to, occ) & bq) | (attacks_rook(to, occ) & rq);
        attackers &= occ;
        stm ^= 1;
    }
    while (d > 0) {                        /* minimax back down the swap list */
        if (-gain[d] < gain[d - 1]) gain[d - 1] = -gain[d];
        d--;
    }
    return gain[0];
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

static inline void hist_add(Ctx *c, int side, int from, int to, int bonus)
{
    int32_t *h = &c->hist[side][from][to];
    const int32_t a = (bonus < 0) ? -bonus : bonus;
    *h += bonus - (int32_t)(((int64_t)(*h) * a) / HIST_MAX);
    if (*h >  HIST_MAX) *h =  HIST_MAX;
    if (*h < -HIST_MAX) *h = -HIST_MAX;
}

static inline void pick_best(Move *ml, int32_t *sc, int n, int i)
{
    int b = i, j;
    for (j = i + 1; j < n; j++) if (sc[j] > sc[b]) b = j;
    if (b != i) {
        Move    tm = ml[i]; ml[i] = ml[b]; ml[b] = tm;
        int32_t ts = sc[i]; sc[i] = sc[b]; sc[b] = ts;
    }
}

/* Score every move of `p` for ordering.
 *
 *   1. transposition-table move
 *   2. captures & promotions, MVV-LVA, but demoted below every quiet move when
 *      SEE says the exchange loses material
 *   3. killers
 *   4. quiet moves ordered by the POLICY head's probability -- the network has
 *      already learned which quiet moves are worth looking at, and this is by
 *      far the largest single ordering win available here
 *   5. history as a secondary key within the policy ordering
 */
static void order_moves(Ctx *c, const Position *p, Move *ml, int32_t *sc, int n,
                        Move ttm, int ply, const Fwd *fw, int use_policy)
{
    const int side = p->side;
    int i;

    if (use_policy && n > 1) {
        for (i = 0; i < n; i++) nn_move_key(p, ml[i], &c->kbuf[i]);
        nn_logits(c->s->head, fw, c->kbuf, n, c->lbuf);
        softmax_t(c->lbuf, n, 1.0f, c->pbuf);
    }

    for (i = 0; i < n; i++) {
        const Move m  = ml[i];
        const int from = MV_FROM(m), to = MV_TO(m);
        const int fl   = MV_FLAG(m);
        int32_t v;

        if (m == ttm) { sc[i] = ORD_TT; continue; }

        if ((fl & 4) || MV_IS_PROMO(m)) {           /* capture and/or promotion */
            int victim   = (fl == MF_EP) ? PAWN
                         : (p->board[to] == NO_PIECE ? -1 : p->board[to]);
            int attacker = p->board[from];
            int mvv      = (victim >= 0 ? PIECE_CP[victim] * 16 : 0)
                         - (attacker >= 0 && attacker < NPIECES ? PIECE_CP[attacker] : 0);
            int see      = s_see(p, m);
            if (MV_IS_PROMO(m)) mvv += (MV_PROMO_PIECE(m) == QUEEN) ? 20000 : 2000;
            sc[i] = (see >= 0) ? (ORD_GOODCAP + mvv) : (ORD_BADCAP + see);
            continue;
        }

        if (ply < S_MAX_PLY) {
            if (m == c->killer[ply][0]) { sc[i] = ORD_KILLER1; continue; }
            if (m == c->killer[ply][1]) { sc[i] = ORD_KILLER2; continue; }
        }

        v = ORD_QUIET;
        if (use_policy && n > 1) v += (int32_t)(POLICY_SCALE * c->pbuf[i]);
        v += (c->hist[side][from][to] + HIST_MAX) * HIST_ORD_SCALE / 32;
        sc[i] = v;
    }
}

/* Cheap MVV-LVA-only ordering for quiescence (SEE is computed once, later, in
 * the move loop where it also drives pruning). */
static void order_captures(const Position *p, Move *ml, int32_t *sc, int n)
{
    int i;
    for (i = 0; i < n; i++) {
        const Move m = ml[i];
        const int fl = MV_FLAG(m);
        int victim   = (fl == MF_EP) ? PAWN
                     : (p->board[MV_TO(m)] == NO_PIECE ? -1 : p->board[MV_TO(m)]);
        int attacker = p->board[MV_FROM(m)];
        int32_t v = (victim >= 0 ? PIECE_CP[victim] * 16 : 0)
                  - (attacker >= 0 && attacker < NPIECES ? PIECE_CP[attacker] : 0);
        if (MV_IS_PROMO(m)) v += (MV_PROMO_PIECE(m) == QUEEN) ? 20000 : 1000;
        sc[i] = v;
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

static void check_limits(Ctx *c)
{
    Search *s = c->s;
    if (!c->allow_stop || s->stop) return;
    if (s->max_nodes && s->nodes >= s->max_nodes) { s->stop = 1; return; }
    if (c->budget_ns > 0 && now_ns() - c->t0_ns >= c->budget_ns) s->stop = 1;
}

static inline int has_big_piece(const Position *p, int col)
{
    return (p->piece[col][KNIGHT] | p->piece[col][BISHOP] |
            p->piece[col][ROOK]   | p->piece[col][QUEEN]) != 0;
}

/* ---------------------------------------------------------- quiescence -- */

static int qsearch(Ctx *c, int alpha, int beta, int ply)
{
    Search *s = c->s;
    Position *p = &c->pos;
    Fwd *fw = &c->fwbuf[ply];
    int in_chk, have_fw = 0, stand, best, n, i;
    Move *ml;
    int32_t *sa;

    c->pvlen[ply] = 0;
    if (s->stop) return 0;
    s->nodes++;
    if ((s->nodes & 2047) == 0) { check_limits(c); if (s->stop) return 0; }

    if (insufficient_material(p)) return 0;
    if (ply >= S_MAX_PLY - 1) {
        have_fw = s_forward(s, p, fw);
        return blend_cp(p, fw, have_fw);
    }

    in_chk = in_check(p, p->side);
    stand  = -S_INF;
    if (!in_chk) {
        have_fw = s_forward(s, p, fw);
        stand   = blend_cp(p, fw, have_fw);
        if (stand >= beta) return stand;
        if (stand > alpha) alpha = stand;
        /* delta pruning: not even winning a queen outright would reach alpha */
        if (stand + PIECE_CP[QUEEN] + 200 < alpha) return alpha;
    }

    ml = c->mlist[ply];
    sa = c->mscore[ply];
    n  = in_chk ? gen_legal(p, ml) : gen_legal_captures(p, ml);
    if (n == 0) return in_chk ? (-S_MATE + ply) : stand;
    order_captures(p, ml, sa, n);

    best = stand;
    for (i = 0; i < n; i++) {
        Move m;
        Undo u;
        int sc;
        pick_best(ml, sa, n, i);
        m = ml[i];

        if (!in_chk) {
            int victim = (MV_FLAG(m) == MF_EP) ? PAWN
                       : (p->board[MV_TO(m)] == NO_PIECE ? -1 : p->board[MV_TO(m)]);
            int gain = (victim >= 0) ? PIECE_CP[victim] : 0;
            if (MV_IS_PROMO(m)) gain += PIECE_CP[MV_PROMO_PIECE(m)] - PIECE_CP[PAWN];
            if (stand + gain + 150 <= alpha) continue;   /* delta pruning      */
            if (s_see(p, m) < 0) continue;               /* losing exchange    */
        }

        make_move(p, m, &u);
        push_key(c, p->key);
        sc = -qsearch(c, -beta, -alpha, ply + 1);
        pop_key(c);
        unmake_move(p, m, &u);
        if (s->stop) return best > -S_INF ? best : stand;

        if (sc > best) {
            best = sc;
            if (sc > alpha) {
                alpha = sc;
                pv_store(c, ply, m);
                if (alpha >= beta) break;
            }
        }
    }
    return best;
}

/* ------------------------------------------------------------- negamax -- */

static int negamax(Ctx *c, int depth, int alpha, int beta, int ply, int can_null)
{
    Search *s = c->s;
    Position *p = &c->pos;
    Fwd *fw = &c->fwbuf[ply];
    const int is_pv = (beta - alpha) > 1;
    int in_chk, have_fw = 0, eval = 0;
    int best = -S_INF, orig_alpha = alpha, n, i, moved_quiets = 0;
    Move ttm = MV_NONE, bestm = MV_NONE;
    Move quiets[64];
    Move *ml;
    int32_t *sa;

    c->pvlen[ply] = 0;
    if (s->stop) return 0;

    if (ply > 0) {
        if (is_repetition(c) || insufficient_material(p)) return 0;
        /* mate-distance pruning: a mate found closer to the root always wins */
        if (alpha < -S_MATE + ply)     alpha = -S_MATE + ply;
        if (beta  >  S_MATE - ply - 1) beta  =  S_MATE - ply - 1;
        if (alpha >= beta) return alpha;
    }
    if (ply >= S_MAX_PLY - 2) {
        have_fw = s_forward(s, p, fw);
        return blend_cp(p, fw, have_fw);
    }
    if (depth <= 0) return qsearch(c, alpha, beta, ply);

    s->nodes++;
    if ((s->nodes & 2047) == 0) { check_limits(c); if (s->stop) return 0; }

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

    /* --- check extension ------------------------------------------------ */
    in_chk = in_check(p, p->side);
    if (in_chk && depth < S_MAX_PLY - ply - 2) depth++;

    if (!in_chk) {
        have_fw = s_forward(s, p, fw);
        eval    = blend_cp(p, fw, have_fw);

        /* reverse futility: so far ahead that giving away `margin` still wins */
        if (!is_pv && depth <= 6 && beta > -S_MATE_BOUND && beta < S_MATE_BOUND &&
            eval - 85 * depth >= beta)
            return eval;

        /* --- null move ---------------------------------------------------
         * Skipped in check (illegal), in PV nodes, and without a non-pawn
         * piece (zugzwang positions are exactly the pawn endings). */
        if (!is_pv && can_null && depth >= 3 && eval >= beta &&
            beta > -S_MATE_BOUND && has_big_piece(p, p->side)) {
            Undo u;
            int R = 2 + depth / 4 + ((eval - beta) > 200 ? 1 : 0);
            int sc;
            if (R > depth - 1) R = depth - 1;
            make_null(p, &u);
            push_key(c, 0);                 /* 0 never matches a real key     */
            sc = -negamax(c, depth - 1 - R, -beta, -beta + 1, ply + 1, 0);
            pop_key(c);
            unmake_null(p, &u);
            if (s->stop) return 0;
            if (sc >= beta) return (sc >= S_MATE_BOUND) ? beta : sc;
        }
    }

    /* --- internal iterative deepening ----------------------------------- */
    if (is_pv && depth >= 5 && ttm == MV_NONE && !in_chk) {
        negamax(c, depth - 2, alpha, beta, ply, 0);
        if (s->stop) return 0;
        if (s->tt && s->tt_size) {
            const TTEntry *e = &s->tt[p->key & (s->tt_size - 1)];
            if (e->key == p->key) ttm = e->best;
        }
        c->pvlen[ply] = 0;
    }

    /* --- moves ---------------------------------------------------------- */
    ml = c->mlist[ply];
    sa = c->mscore[ply];
    n  = gen_legal(p, ml);
    if (n == 0) return in_chk ? (-S_MATE + ply) : 0;
    if (p->halfmove >= 100) return 0;              /* fifty-move, not mate    */

    /* The policy pass costs a handful of microseconds; it pays for itself at
     * interior nodes and would not at depth 1, where ordering barely matters. */
    order_moves(c, p, ml, sa, n, ttm, ply, fw, have_fw && depth >= 2);

    for (i = 0; i < n; i++) {
        Move m;
        Undo u;
        int is_tactical, gives_check, new_depth, r = 0, sc;

        pick_best(ml, sa, n, i);
        m = ml[i];
        is_tactical = MV_IS_CAPTURE(m) || MV_IS_PROMO(m);

        /* shallow quiet pruning, never in the PV and never when a mate is at
         * stake or nothing has been searched yet */
        if (!is_pv && !in_chk && i > 0 && !is_tactical &&
            best > -S_MATE_BOUND && depth <= 5) {
            if (eval + 110 + 130 * depth <= alpha) continue;
            if (sa[i] < ORD_QUIET && depth <= 3) continue;      /* bad capture */
        }
        if (!is_pv && !in_chk && i > 0 && is_tactical && depth <= 4 &&
            best > -S_MATE_BOUND && sa[i] < ORD_QUIET) {
            if (s_see(p, m) < -100 * depth) continue;           /* SEE pruning */
        }

        make_move(p, m, &u);
        push_key(c, p->key);
        gives_check = in_check(p, p->side);
        new_depth = depth - 1;

        /* --- late move reductions --------------------------------------- */
        if (depth >= 3 && i >= 3 && !is_tactical && !in_chk && !gives_check &&
            sa[i] < ORD_KILLER2) {
            r = c->lmr[imin(depth, 63)][imin(i, 63)];
            if (is_pv) r--;
            if (c->hist[p->side ^ 1][MV_FROM(m)][MV_TO(m)] > HIST_MAX / 2) r--;
            if (r < 0) r = 0;
            if (r > new_depth - 1) r = new_depth - 1;
        }

        if (i == 0) {
            sc = -negamax(c, new_depth, -beta, -alpha, ply + 1, 1);
        } else {
            sc = -negamax(c, new_depth - r, -alpha - 1, -alpha, ply + 1, 1);
            if (sc > alpha && r > 0)
                sc = -negamax(c, new_depth, -alpha - 1, -alpha, ply + 1, 1);
            if (sc > alpha && sc < beta)
                sc = -negamax(c, new_depth, -beta, -alpha, ply + 1, 1);
        }

        pop_key(c);
        unmake_move(p, m, &u);
        if (s->stop) return best > -S_INF ? best : alpha;

        if (!is_tactical && moved_quiets < 64) quiets[moved_quiets++] = m;

        if (sc > best) {
            best  = sc;
            bestm = m;
            if (sc > alpha) {
                alpha = sc;
                pv_store(c, ply, m);
                if (alpha >= beta) {
                    if (!is_tactical) {
                        int j, bonus = imin(depth * depth, 1200);
                        if (ply < S_MAX_PLY && c->killer[ply][0] != m) {
                            c->killer[ply][1] = c->killer[ply][0];
                            c->killer[ply][0] = m;
                        }
                        hist_add(c, p->side, MV_FROM(m), MV_TO(m), bonus);
                        for (j = 0; j < moved_quiets - 1; j++)
                            hist_add(c, p->side, MV_FROM(quiets[j]), MV_TO(quiets[j]), -bonus);
                    }
                    break;
                }
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

    for (i = 0; i < n; i++) {
        const Move m = c->rootm[i];
        Undo u;
        int gives_check, sc, r = 0;

        make_move(p, m, &u);
        push_key(c, p->key);
        gives_check = in_check(p, p->side);

        if (i == 0) {
            sc = -negamax(c, depth - 1, -beta, -alpha, 1, 1);
        } else {
            if (depth >= 3 && i >= 4 && !MV_IS_CAPTURE(m) && !MV_IS_PROMO(m) && !gives_check)
                r = 1 + (i >= 12 && depth >= 5);
            if (r > depth - 2) r = depth - 2;
            if (r < 0) r = 0;
            sc = -negamax(c, depth - 1 - r, -alpha - 1, -alpha, 1, 1);
            if (sc > alpha && r > 0)
                sc = -negamax(c, depth - 1, -alpha - 1, -alpha, 1, 1);
            if (sc > alpha && sc < beta)
                sc = -negamax(c, depth - 1, -beta, -alpha, 1, 1);
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

/* Insertion sort, scores descending: best move first for the next iteration. */
static void sort_root(Move *rm, int *rsc, int n)
{
    int i;
    for (i = 1; i < n; i++) {
        const Move m = rm[i];
        const int sc = rsc[i];
        int j = i - 1;
        while (j >= 0 && rsc[j] < sc) { rm[j + 1] = rm[j]; rsc[j + 1] = rsc[j]; j--; }
        rm[j + 1] = m;
        rsc[j + 1] = sc;
    }
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
    c->allow_stop = 0;                 /* depth 1 always runs to completion   */

    maxd = s->max_depth;
    if (maxd <= 0) maxd = (s->movetime_ms > 0 || s->max_nodes) ? S_MAX_PLY - 4 : 8;
    if (maxd > S_MAX_PLY - 4) maxd = S_MAX_PLY - 4;

    /* Order the root once up front: policy + MVV-LVA. */
    {
        Fwd fw;
        int have = s_forward(s, &c->pos, &fw);
        order_moves(c, &c->pos, c->rootm, c->rootsc, c->nroot, MV_NONE, 0, &fw, have);
        sort_root(c->rootm, c->rootsc, c->nroot);
        for (i = 0; i < c->nroot; i++) c->rootsc[i] = -S_INF;
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

        if (s->stop) break;                 /* iteration incomplete: discard   */

        prev = score;
        best_move = itm;
        completed = d;
        s->depth_reached = d;
        s->score_cp = score;
        s->pv_len = imin(c->pvlen[0], S_PV_MAX);
        for (i = 0; i < s->pv_len; i++) s->pv[i] = c->pvtab[0][i];
        if (s->pv_len == 0) { s->pv[0] = best_move; s->pv_len = 1; }

        sort_root(c->rootm, c->rootsc, c->nroot);
        c->allow_stop = 1;                  /* from here on, the clock rules   */

        if (score >= S_MATE_BOUND || score <= -S_MATE_BOUND) break;  /* mate    */
        if (c->budget_ns > 0 && (now_ns() - c->t0_ns) * 2 >= c->budget_ns) break;
        if (s->max_nodes && s->nodes >= s->max_nodes) break;
    }

    if (completed == 0) {                    /* should not happen; stay legal  */
        best_move = c->rootm[0];
        s->depth_reached = 1;
        s->pv[0] = best_move;
        s->pv_len = 1;
    }

    /* ------------------------------------------------------------ blunder --
     * The UI's difficulty slider.  With probability blunder_rate we play a
     * uniformly random move out of the better-scoring half of the root moves
     * instead of the best one -- always a legal move, and never something
     * completely absurd, so easier levels still feel like chess. */
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
