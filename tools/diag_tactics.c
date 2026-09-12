/* tools/diag_tactics.c -- WHY does the agent hang pieces?
 *
 * A measuring instrument, not part of the agent.  docs/FROM_SCRATCH.md permits
 * chess knowledge in an instrument and forbids it in the learning or play path;
 * this file is never linked into build/chessrl, build/libchessrl.dylib or the
 * app.  It contains piece values and a quiescence search because that is the
 * ruler being held against the network, exactly as py/baselines.py does for the
 * external benchmark.
 *
 * BUILD (nothing in the Makefile references this; it is run by hand):
 *
 *   cc -O3 -std=c11 -D_DARWIN_C_SOURCE -Wall -Wextra -Wno-unused-parameter \
 *      -funroll-loops -fno-math-errno -ffp-contract=fast -mcpu=native -Isrc \
 *      tools/diag_tactics.c build/chess.o build/net.o build/mcts.o \
 *      -o build/diag_tactics -lm -lpthread
 *
 * THE ORACLE.  Every "blunder" in this file is defined by one instrument and
 * one only: depth-1 negamax over material with a 6-ply quiescence search and
 * delta pruning -- i.e. `material-1` from py/baselines.py, the opponent the
 * agent scores 5% against, reimplemented in C so it can be called tens of
 * millions of times.  Piece values are py/baselines.py's: P=1, N=3, B=3.25,
 * R=5, Q=9.
 *
 *     oracle_value(m) = -quiesce(position after m)      [mover's view, pawns]
 *     loss(m)         = max_m' oracle_value(m') - oracle_value(m)
 *     BLUNDER         = loss(m) >= 2.0 pawns
 *
 * so a "blunder" here always means "a move that a capture-resolving material
 * search says drops at least two pawns of material", never an opinion.
 *
 * SUBCOMMANDS
 *   game      E0:  grade EVERY move of complete games against material-1 or
 *                  against itself.  Conditions on nothing, so the per-move rate
 *                  multiplies out into the match result the benchmark saw.
 *   corpus    build and describe the blunder-position corpus
 *   value     E1a: what the VALUE head says about a blunder vs a safe move
 *   ablate    E3b: delete one of the mover's own pieces and re-evaluate --
 *                  the value head's exchange rate between material and winning
 *   prior     E1b: what PRIOR the policy head puts on the blunder
 *   sims      E1c: blunder rate against the simulation count   (KEY EXPERIMENT)
 *   tree      E2:  how deep the tree goes and where the visits land
 *   calib     E3:  value-head calibration, at play settings, at the TRAINING
 *                  settings (--train-mode) and against the oracle
 *   phase     E4:  by game phase, and classical vs Chess960 starts
 *   compare   E5:  best.crl vs an earlier checkpoint vs random weights
 *   all       every one of the above except `game` and `calib`
 *
 * The findings are written up in docs/TACTICS.md, which names the command
 * behind every number.
 */
#define _DARWIN_C_SOURCE 1

#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "chess.h"
#include "net.h"
#include "mcts.h"

/* ========================================================================= */
/*                                  utility                                  */
/* ========================================================================= */

static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static uint64_t sm64(uint64_t *s)
{
    uint64_t z = (*s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

/* xoshiro256** -- the same state shape mcts_search() wants. */
static uint64_t rotl64(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }

static uint64_t rng_next(uint64_t *s)
{
    const uint64_t r = rotl64(s[1] * 5, 7) * 9;
    const uint64_t t = s[1] << 17;
    s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3]; s[2] ^= t;
    s[3] = rotl64(s[3], 45);
    return r;
}

static void rng_seed(uint64_t *s, uint64_t seed)
{
    uint64_t z = seed ? seed : 0x9E3779B97F4A7C15ULL;
    for (int i = 0; i < 4; i++) s[i] = sm64(&z);
    for (int i = 0; i < 8; i++) (void)rng_next(s);
}

static int cpu_count_local(void)
{
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    if (n < 1) n = 4;
    if (n > 64) n = 64;
    return (int)n;
}

/* ========================================================================= */
/*                     the oracle: material + quiescence                     */
/* ========================================================================= */

#define OR_MATE    30000.0f
#define OR_INF     1e9f
#define OR_QDEPTH  6

/* py/baselines.py PIECE_VALUE, in pawns.  P N B R Q K. */
static const float OR_PV[NPIECES] = { 1.0f, 3.0f, 3.25f, 5.0f, 9.0f, 0.0f };

/* Material from the side to move's view, in pawns. */
static float or_mat(const Position *p)
{
    float w = 0.0f, b = 0.0f;
    for (int pc = PAWN; pc <= QUEEN; pc++) {
        w += OR_PV[pc] * (float)bb_count(p->piece[WHITE][pc]);
        b += OR_PV[pc] * (float)bb_count(p->piece[BLACK][pc]);
    }
    return (p->side == WHITE) ? (w - b) : (b - w);
}

/* Value of the piece a move captures, 0 when it captures nothing. */
static float or_victim(const Position *p, Move m)
{
    if (!MV_IS_CAPTURE(m)) return 0.0f;
    if (MV_FLAG(m) == MF_EP) return OR_PV[PAWN];
    return OR_PV[(int)p->board[MV_TO(m)]];
}

/* Quiescence over captures and promotions only, negamax, delta-pruned.
 * Identical in structure to SearchBaseline._quiesce in py/baselines.py. */
static float or_quiesce(Position *p, float alpha, float beta, int ply, int qleft,
                        uint64_t *nodes)
{
    Move  legal[MAX_MOVES];
    Move  tm[MAX_MOVES];
    float tv[MAX_MOVES], tk[MAX_MOVES];
    int   n, nt = 0, i, j;
    float stand, best;

    n = gen_legal(p, legal);
    if (n == 0) return in_check(p, p->side) ? -(OR_MATE - (float)ply) : 0.0f;
    (*nodes)++;

    stand = or_mat(p);
    if (qleft <= 0 || stand >= beta) return stand;
    if (stand > alpha) alpha = stand;
    best = stand;

    for (i = 0; i < n; i++) {
        const Move  m  = legal[i];
        const float v  = or_victim(p, m);
        const float pb = MV_IS_PROMO(m) ? (OR_PV[MV_PROMO_PIECE(m)] - 1.0f) : 0.0f;
        if (v == 0.0f && pb == 0.0f) continue;
        tm[nt] = m;
        tv[nt] = v;
        /* MVV-LVA: pure node-count ordering, it cannot change the value. */
        tk[nt] = v * 16.0f + pb * 8.0f - OR_PV[(int)p->board[MV_FROM(m)]];
        nt++;
    }
    for (i = 1; i < nt; i++) {                       /* insertion sort, desc */
        const Move  km = tm[i]; const float kv = tv[i], kk = tk[i];
        j = i - 1;
        while (j >= 0 && tk[j] < kk) { tm[j+1]=tm[j]; tv[j+1]=tv[j]; tk[j+1]=tk[j]; j--; }
        tm[j+1] = km; tv[j+1] = kv; tk[j+1] = kk;
    }

    for (i = 0; i < nt; i++) {
        const float delta = MV_IS_PROMO(tm[i]) ? 8.0f : 2.0f;
        Undo  u;
        float v;
        if (stand + tv[i] + delta <= alpha) continue;   /* delta pruning */
        make_move(p, tm[i], &u);
        v = -or_quiesce(p, -beta, -alpha, ply + 1, qleft - 1, nodes);
        unmake_move(p, tm[i], &u);
        if (v > best) {
            best = v;
            if (v > alpha) { alpha = v; if (alpha >= beta) break; }
        }
    }
    return best;
}

/* Exact oracle value of EVERY legal move (full window each, so no move comes
 * back as a bound).  Returns the number of legal moves. */
static int or_root(Position *p, Move *moves, float *vals, uint64_t *nodes)
{
    const int n = gen_legal(p, moves);
    for (int i = 0; i < n; i++) {
        Undo u;
        make_move(p, moves[i], &u);
        vals[i] = -or_quiesce(p, -OR_INF, OR_INF, 1, OR_QDEPTH, nodes);
        unmake_move(p, moves[i], &u);
    }
    return n;
}

/* The oracle's own move, position-only: the first of the best-scoring moves.
 * Used where no game history exists (inside the value probes). */
static Move or_best_move(Position *p)
{
    Move  mv[MAX_MOVES];
    float vl[MAX_MOVES];
    uint64_t nodes = 0;
    int n = or_root(p, mv, vl, &nodes), b = 0;
    if (n <= 0) return MV_NONE;
    for (int i = 1; i < n; i++) if (vl[i] > vl[b]) b = i;
    return mv[b];
}

/* THE PLAYER material-1 ACTUALLY IS, when it has a game and not just a
 * position.  Two details separate it from the function above and both change
 * the games it wins: a drawn result (repetition, fifty-move, insufficient
 * material) scores 0.0, so it will not shuffle while it is ahead; and ties are
 * broken UNIFORMLY AT RANDOM, so it does not lock into one line.  Both come
 * straight from SearchBaseline.root_values / .pick in py/baselines.py -- this
 * is the same opponent the external benchmark ran, so any statement about "the
 * bot the agent scores 5% against" is a statement about this code. */
#define OR_TIE_EPS 1e-4f

static Move or_pick_game(Game *g, uint64_t *rng)
{
    Move  mv[MAX_MOVES];
    int   ties[MAX_MOVES];
    uint64_t nodes = 0;
    float best = -OR_INF;
    int   n = gen_legal(&g->pos, mv), nt = 0;
    const int mover = (int)g->pos.side;

    if (n <= 0) return MV_NONE;
    for (int i = 0; i < n; i++) {
        float v;
        int   res;
        game_push(g, mv[i]);
        res = game_update_result(g, 0);              /* 0 = no ply cap here */
        if (res == GR_DRAW)                          v = 0.0f;
        else if (res == (mover == WHITE ? GR_WHITE_WIN : GR_BLACK_WIN))
                                                     v = OR_MATE - 1.0f;
        else if (res != GR_ONGOING)                  v = -(OR_MATE - 1.0f);
        else v = -or_quiesce(&g->pos, -OR_INF, OR_INF, 1, OR_QDEPTH, &nodes);
        game_pop(g);
        if (v > best + OR_TIE_EPS)      { best = v; ties[0] = i; nt = 1; }
        else if (v > best - OR_TIE_EPS) { if (v > best) best = v; ties[nt++] = i; }
    }
    if (nt <= 0) return mv[0];
    return mv[ties[nt == 1 ? 0 : (int)(rng_next(rng) % (uint64_t)nt)]];
}

/* ========================================================================= */
/*                                the model                                  */
/* ========================================================================= */

typedef struct {
    Trunk *tr;
    Head  *heads;
    Hyper *hy;
    float *elo;
    int    n, gen, best;
    char   name[256];
} Model;

static int model_open(Model *M, const char *path)
{
    int probe_n = 0, probe_gen = 0, cap;
    memset(M, 0, sizeof *M);
    if (!model_load(path, NULL, NULL, NULL, NULL, &probe_n, &probe_gen) || probe_n <= 0) {
        fprintf(stderr, "error: cannot read model '%s'\n", path ? path : "(null)");
        return 0;
    }
    M->tr    = calloc(1, sizeof(Trunk));
    M->heads = calloc((size_t)probe_n, sizeof(Head));
    M->hy    = calloc((size_t)probe_n, sizeof(Hyper));
    M->elo   = calloc((size_t)probe_n, sizeof(float));
    if (!M->tr || !M->heads || !M->hy || !M->elo) { fprintf(stderr, "oom\n"); return 0; }
    cap = probe_n;
    if (!model_load(path, M->tr, M->heads, M->hy, M->elo, &cap, &probe_gen) || cap <= 0) {
        fprintf(stderr, "error: model '%s' is corrupt\n", path);
        return 0;
    }
    M->n = cap; M->gen = probe_gen; M->best = 0;
    for (int i = 1; i < cap; i++) if (M->elo[i] > M->elo[M->best]) M->best = i;
    snprintf(M->name, sizeof M->name, "%s", path);
    return 1;
}

/* A model with randomly initialised weights: the from-scratch floor. */
static int model_random(Model *M, uint64_t seed)
{
    memset(M, 0, sizeof *M);
    M->tr    = calloc(1, sizeof(Trunk));
    M->heads = calloc(1, sizeof(Head));
    M->hy    = calloc(1, sizeof(Hyper));
    M->elo   = calloc(1, sizeof(float));
    if (!M->tr || !M->heads || !M->hy || !M->elo) { fprintf(stderr, "oom\n"); return 0; }
    nn_init(M->tr, M->heads, seed);
    hyper_default(M->hy);
    M->n = 1; M->gen = 0; M->best = 0;
    snprintf(M->name, sizeof M->name, "random-weights(seed=%llu)",
             (unsigned long long)seed);
    return 1;
}

static void model_close(Model *M)
{
    free(M->tr); free(M->heads); free(M->hy); free(M->elo);
    memset(M, 0, sizeof *M);
}

static const Head *model_head(const Model *M) { return &M->heads[M->best]; }

/* Value of `p` from the side to move's view. */
static float net_value(const Model *M, const Position *p)
{
    uint16_t f[NF_MAXACTIVE];
    Fwd      fw;
    const int nf = nn_features(p, f);
    nn_eval(M->tr, model_head(M), f, nf, &fw);
    return fw.v;
}

/* The priors the SEARCH would use: softmax at temperature 1 over the legal
 * moves, exactly as mcts.c does at an expansion. */
static void net_priors(const Model *M, const Position *p, const Move *list, int n,
                       float *P, float *v)
{
    uint16_t f[NF_MAXACTIVE];
    MoveKey  keys[MAX_MOVES];
    float    logits[MAX_MOVES];
    Fwd      fw;
    const int nf = nn_features(p, f);
    nn_eval(M->tr, model_head(M), f, nf, &fw);
    if (v) *v = fw.v;
    for (int i = 0; i < n; i++) nn_move_key(p, list[i], &keys[i]);
    nn_logits(model_head(M), &fw, keys, n, logits);
    softmax_t(logits, n, 1.0f, P);
}

/* ========================================================================= */
/*                                the corpus                                 */
/* ========================================================================= */

/* TWO DIFFERENT MISTAKES, kept apart on purpose.
 *
 *   drop(m) = material_before - oracle_value(m)    -- what the move GIVES AWAY
 *   loss(m) = best_value      - oracle_value(m)    -- what the move gives away
 *                                                     PLUS what it fails to win
 *
 * "The agent hangs pieces" is a claim about drop, so drop is the primary
 * metric everywhere below: a HANG is drop >= 2 pawns, i.e. the mover ends the
 * capture sequence at least two pawns poorer than it started.  loss is
 * reported alongside because it is the quantity that actually costs games
 * against material-1, which will also take a free piece when one is offered. */
#define BLUNDER_PAWNS   2.0f    /* drop at which a move is called a hang      */
#define SAFE_PAWNS      0.25f   /* drop under which a move is called safe     */
#define MAX_ORACLE_ABS  60.0f   /* reject positions the oracle calls decided  */

/* k-th move of position d hangs material / is worse than best by >= 2 pawns */
#define IS_HANG(d, k)  ((d)->mat  - (d)->val[k] >= BLUNDER_PAWNS)
#define IS_MISS(d, k)  ((d)->best - (d)->val[k] >= BLUNDER_PAWNS)
#define IS_HOLD(d, k)  ((d)->mat  - (d)->val[k] <= SAFE_PAWNS)

enum { PH_OPEN = 0, PH_MID = 1, PH_END = 2, PH_N = 3 };
static const char *PHASE_NAME[PH_N] = { "opening", "middlegame", "endgame" };

typedef struct {
    char   fen[104];
    Move   moves[MAX_MOVES];
    float  val[MAX_MOVES];     /* oracle value, mover's view, pawns           */
    int    nmoves;
    int    bi;                 /* index of the oracle-best move               */
    int    si;                 /* a material-NEUTRAL reference move           */
    int    wi;                 /* index of the worst hang (largest drop)      */
    int    nblunder;           /* legal moves that hang >= BLUNDER_PAWNS      */
    int    nmiss;              /* legal moves >= BLUNDER_PAWNS below the best */
    int    nhold;              /* legal moves that hold their material        */
    float  best;               /* oracle value of the best move               */
    float  mat;                /* material balance before the move, mover view */
    float  worst_loss;         /* drop of moves[wi]                           */
    int    ply;
    int    phase;
    int    npieces;
    int    incheck;
    int    c960;               /* the game started from a 960 array != 518    */
} DPos;

enum { SRC_RAND = 0, SRC_SELF = 1, SRC_VS = 2 };
static const char *SRC_NAME[3] = { "random-play", "agent-self-play", "agent-vs-oracle" };

enum { ST_CLASSICAL = 0, ST_960 = 1, ST_MIXED = 2 };

typedef struct {
    int      source;
    int      start;            /* ST_*                                        */
    int      want;             /* positions to collect                        */
    int      max_plies;
    int      per_game;         /* cap on positions taken from one game        */
    int      gen_sims;         /* sims used when the agent generates the games*/
    uint64_t seed;
    const Model *M;            /* needed for SRC_SELF / SRC_VS                */
} CorpusCfg;

/* Classify by how much wood is left.  An instrument's convention, nothing the
 * agent ever sees. */
static int phase_of(const Position *p)
{
    const int n = bb_count(p->all);
    if (n >= 26) return PH_OPEN;
    if (n >= 13) return PH_MID;
    return PH_END;
}

/* Does this position qualify?  Fills `d` and returns 1. */
static int scan_position(const Position *p, DPos *d, uint64_t *onodes)
{
    float lo, bestneu = 0.0f;
    int   i, b = 0, w = -1, s = -1, nb = 0, nm = 0, nh = 0;

    d->nmoves = or_root((Position *)p, d->moves, d->val, onodes);
    if (d->nmoves < 4) return 0;
    if (in_check(p, p->side)) return 0;      /* a check is a forced situation */
    for (i = 1; i < d->nmoves; i++) if (d->val[i] > d->val[b]) b = i;
    if (fabsf(d->val[b]) >= MAX_ORACLE_ABS) return 0;      /* mate in sight */
    lo = d->val[b];
    for (i = 0; i < d->nmoves; i++) if (d->val[i] < lo) lo = d->val[i];
    if (fabsf(lo) >= MAX_ORACLE_ABS) return 0;             /* mate in sight */

    d->mat  = or_mat(p);
    d->best = d->val[b];
    /* The side to move must be able to HOLD what it owns, otherwise "it gave a
     * piece away" is not the event being measured -- the piece was already
     * gone before it moved. */
    if (d->best < d->mat - 0.5f) return 0;

    for (i = 0; i < d->nmoves; i++) {
        if (IS_MISS(d, i)) nm++;
        if (IS_HANG(d, i)) {
            nb++;
            if (w < 0 || d->val[i] < d->val[w]) w = i;
        }
        if (IS_HOLD(d, i)) {
            const float dev = fabsf(d->val[i] - d->mat);
            nh++;
            /* the safe REFERENCE move: holds its material and does not win any
             * either, so V(safe) - V(hang) is exactly the hung material */
            if (s < 0 || dev < bestneu) { s = i; bestneu = dev; }
        }
    }
    if (nb == 0 || w < 0) return 0;          /* nothing is hanging on offer   */
    if (nh < 2 || s < 0)  return 0;          /* no safe alternative to pick   */

    d->bi = b; d->si = s; d->wi = w;
    d->nblunder = nb; d->nmiss = nm; d->nhold = nh;
    d->worst_loss = d->mat - d->val[w];
    pos_to_fen(p, d->fen, sizeof d->fen);
    d->phase   = phase_of(p);
    d->npieces = bb_count(p->all);
    d->incheck = in_check(p, p->side);
    return 1;
}

typedef struct {
    const CorpusCfg *cfg;
    DPos     *out;
    int       cap;
    int       got;
    uint64_t  seed;
    uint64_t  onodes;
    uint64_t  scanned;
    Mcts      mcts;
} CorpusWorker;

/* Picks the mover for corpus generation and returns the move to play. */
static Move corpus_move(CorpusWorker *w, Game *g, int agent_side, uint64_t *rng)
{
    const CorpusCfg *c = w->cfg;
    Move  list[MAX_MOVES];
    int   n = gen_legal(&g->pos, list);
    if (n <= 0) return MV_NONE;

    if (c->source == SRC_RAND) return list[rng_next(rng) % (uint64_t)n];

    if (c->source == SRC_VS && (int)g->pos.side != agent_side)
        return or_pick_game(g, rng);

    {   /* the agent, through the real search */
        int32_t visits[MAX_MOVES];
        float   rv;
        int     nr = mcts_search(&w->mcts, c->M->tr, model_head(c->M), g,
                                 c->gen_sims, 0, rng, visits, &rv);
        if (nr != n) return list[rng_next(rng) % (uint64_t)n];
        /* temperature 1 for the first 12 plies so the games differ, argmax
         * after -- the same shape az.c's self-play uses. */
        return list[mcts_pick(visits, n, g->ply < 12 ? 1.0f : 0.0f, rng)];
    }
}

/* Plays one complete game, keeping every position at which the side we care
 * about was to move, THEN samples plies uniformly from the whole game.  Taking
 * positions as they come instead would put almost every sample in the first ten
 * plies, because the per-game cap is reached before the game gets going. */
static void *corpus_thread(void *arg)
{
    CorpusWorker *w = (CorpusWorker *)arg;
    const CorpusCfg *c = w->cfg;
    uint64_t rng[4];
    uint64_t gi = 0;
    Position *snap = malloc((size_t)(c->max_plies + 2) * sizeof(Position));
    int      *sply = malloc((size_t)(c->max_plies + 2) * sizeof(int));

    if (!snap || !sply) { fprintf(stderr, "oom in corpus worker\n"); exit(1); }
    rng_seed(rng, w->seed);
    if (c->source != SRC_RAND) mcts_init(&w->mcts, 64 * (c->gen_sims + 8) + 1);

    while (w->got < w->cap) {
        Game g;
        int  taken = 0, agent_side = (int)(gi & 1), c960 = 0, ns = 0;

        if (c->start == ST_CLASSICAL) {
            game_start(&g);
        } else if (c->start == ST_960) {
            int id = (int)(rng_next(rng) % 960u);
            game_start960(&g, id);
            c960 = (id != CHESS960_CLASSICAL_ID);
        } else {
            if (rng_next(rng) & 1) { game_start(&g); }
            else { int id = (int)(rng_next(rng) % 960u); game_start960(&g, id);
                   c960 = (id != CHESS960_CLASSICAL_ID); }
        }
        gi++;

        while (g.result == GR_ONGOING && g.ply < c->max_plies) {
            Move m;
            if (c->source != SRC_VS || (int)g.pos.side == agent_side) {
                snap[ns] = g.pos;
                sply[ns] = g.ply;
                ns++;
            }
            m = corpus_move(w, &g, agent_side, rng);
            if (m == MV_NONE) break;
            game_push(&g, m);
            game_update_result(&g, c->max_plies);
        }

        /* Fisher-Yates over the ply indices, then scan until the cap is met. */
        for (int i = ns - 1; i > 0; i--) {
            int j = (int)(rng_next(rng) % (uint64_t)(i + 1));
            Position tp = snap[i]; snap[i] = snap[j]; snap[j] = tp;
            int ti = sply[i]; sply[i] = sply[j]; sply[j] = ti;
        }
        for (int i = 0; i < ns && taken < c->per_game && w->got < w->cap; i++) {
            DPos *d = &w->out[w->got];
            w->scanned++;
            if (scan_position(&snap[i], d, &w->onodes)) {
                d->ply  = sply[i];
                d->c960 = c960;
                w->got++;
                taken++;
            }
        }
    }
    if (c->source != SRC_RAND) mcts_free(&w->mcts);
    free(snap); free(sply);
    return NULL;
}

/* Collects `cfg->want` positions.  Deterministic given the seed and the thread
 * count: each thread owns a fixed, contiguous slice of the output. */
static DPos *corpus_build(const CorpusCfg *cfg, int threads, int *nout,
                          uint64_t *scanned, double *secs)
{
    CorpusWorker *w = calloc((size_t)threads, sizeof *w);
    pthread_t    *t = calloc((size_t)threads, sizeof *t);
    DPos *out = calloc((size_t)cfg->want + (size_t)threads, sizeof *out);
    const int per = (cfg->want + threads - 1) / threads;
    double t0 = now_sec();
    int i, n = 0;
    uint64_t sc = 0;

    if (!w || !t || !out) { fprintf(stderr, "oom building corpus\n"); exit(1); }
    for (i = 0; i < threads; i++) {
        w[i].cfg  = cfg;
        w[i].out  = out + (size_t)i * (size_t)per;
        w[i].cap  = (i == threads - 1) ? (cfg->want - per * (threads - 1)) : per;
        if (w[i].cap < 0) w[i].cap = 0;
        w[i].seed = cfg->seed + 0x1000ULL * (uint64_t)(i + 1);
        pthread_create(&t[i], NULL, corpus_thread, &w[i]);
    }
    for (i = 0; i < threads; i++) pthread_join(t[i], NULL);

    /* compact the per-thread slices into one dense array, in thread order */
    for (i = 0; i < threads; i++) {
        if (w[i].got > 0 && out + n != w[i].out)
            memmove(out + n, w[i].out, (size_t)w[i].got * sizeof(DPos));
        n += w[i].got;
        sc += w[i].scanned;
    }
    if (secs)    *secs = now_sec() - t0;
    if (scanned) *scanned = sc;
    *nout = n;
    free(w); free(t);
    return out;
}

/* ========================================================================= */
/*                        E1a: what the value head says                      */
/* ========================================================================= */

/* The opponent's oracle-best reply to `m`, applied; leaves the position with
 * the ROOT mover to move again.  Returns 0 if the game ended. */
static int apply_pair(Position *p, Move m, Undo *u1, Move *reply, Undo *u2)
{
    make_move(p, m, u1);
    *reply = or_best_move(p);
    if (*reply == MV_NONE) return 0;
    make_move(p, *reply, u2);
    return 1;
}

typedef struct {
    double d1_sum, d1_sq, d2_sum;
    long   n, sep1, sep2, n2;
    double v_blund1, v_safe1, v_blund2, v_safe2;
    double vabs_sum, v_sum, v_sq;
    double corr_xy, corr_x, corr_y, corr_xx, corr_yy;   /* v vs oracle value  */
    long   corr_n;
} ValueStat;

static void value_run(const Model *M, const DPos *C, int n, ValueStat *S)
{
    memset(S, 0, sizeof *S);
    for (int i = 0; i < n; i++) {
        const DPos *d = &C[i];
        Position p;
        Undo u1, u2;
        Move reply;
        float vb1, vs1, vb2 = 0.0f, vs2 = 0.0f;
        int ok2 = 1;

        if (!pos_from_fen(&p, d->fen)) continue;

        /* ---- one ply: the material has not been lost yet ---------------- */
        make_move(&p, d->moves[d->wi], &u1);
        vb1 = -net_value(M, &p);               /* flip: opponent to move      */
        unmake_move(&p, d->moves[d->wi], &u1);
        make_move(&p, d->moves[d->si], &u1);
        vs1 = -net_value(M, &p);
        unmake_move(&p, d->moves[d->si], &u1);

        /* ---- two plies: the recapture has happened ---------------------- */
        if (apply_pair(&p, d->moves[d->wi], &u1, &reply, &u2)) {
            vb2 = net_value(M, &p);            /* root mover to move again    */
            unmake_move(&p, reply, &u2);
            unmake_move(&p, d->moves[d->wi], &u1);
        } else { unmake_move(&p, d->moves[d->wi], &u1); ok2 = 0; }
        if (ok2 && apply_pair(&p, d->moves[d->si], &u1, &reply, &u2)) {
            vs2 = net_value(M, &p);
            unmake_move(&p, reply, &u2);
            unmake_move(&p, d->moves[d->si], &u1);
        } else if (ok2) { unmake_move(&p, d->moves[d->si], &u1); ok2 = 0; }

        S->d1_sum += (double)(vs1 - vb1);
        S->d1_sq  += (double)(vs1 - vb1) * (double)(vs1 - vb1);
        S->v_blund1 += vb1; S->v_safe1 += vs1;
        S->vabs_sum += fabs((double)vs1);
        S->v_sum += vs1; S->v_sq += (double)vs1 * (double)vs1;
        if (vs1 > vb1) S->sep1++;
        S->n++;
        if (ok2) {
            S->d2_sum += (double)(vs2 - vb2);
            S->v_blund2 += vb2; S->v_safe2 += vs2;
            if (vs2 > vb2) S->sep2++;
            S->n2++;
        }

        /* correlation of the 1-ply value with the oracle value of the move */
        for (int k = 0; k < d->nmoves; k++) {
            float vv;
            make_move(&p, d->moves[k], &u1);
            vv = -net_value(M, &p);
            unmake_move(&p, d->moves[k], &u1);
            S->corr_x  += d->val[k];  S->corr_y  += vv;
            S->corr_xx += (double)d->val[k] * d->val[k];
            S->corr_yy += (double)vv * vv;
            S->corr_xy += (double)d->val[k] * vv;
            S->corr_n++;
        }
    }
}

static double pearson(double sxy, double sx, double sy, double sxx, double syy, long n)
{
    double num, den;
    if (n < 2) return 0.0;
    num = sxy - sx * sy / (double)n;
    den = sqrt((sxx - sx * sx / (double)n) * (syy - sy * sy / (double)n));
    return den > 0.0 ? num / den : 0.0;
}

static void value_report(const char *tag, const Model *M, const DPos *C, int n)
{
    ValueStat S;
    value_run(M, C, n, &S);
    printf("\n--- E1a  VALUE HEAD: can it tell a blunder from a safe move? [%s] ---\n", tag);
    printf("model                     %s (gen %d, agent %d/%d)\n",
           M->name, M->gen, M->best, M->n);
    printf("positions                 %ld\n", S.n);
    printf("mean V after SAFE  move   %+.4f   (mover's view)\n", S.v_safe1  / (double)S.n);
    printf("mean V after BLUNDER move %+.4f\n", S.v_blund1 / (double)S.n);
    printf("mean separation V(safe)-V(blunder)        %+.4f  (sd %.4f)\n",
           S.d1_sum / (double)S.n,
           sqrt(fabs(S.d1_sq / (double)S.n - (S.d1_sum / (double)S.n) * (S.d1_sum / (double)S.n))));
    printf("P[V(safe) > V(blunder)]                   %.3f   (0.500 = blind)\n",
           (double)S.sep1 / (double)S.n);
    if (S.n2 > 0) {
        printf("AFTER the forced recapture (material actually gone):\n");
        printf("  mean V safe line %+.4f   blunder line %+.4f   separation %+.4f\n",
               S.v_safe2 / (double)S.n2, S.v_blund2 / (double)S.n2,
               S.d2_sum / (double)S.n2);
        printf("  P[V(safe line) > V(blunder line)]       %.3f\n",
               (double)S.sep2 / (double)S.n2);
    }
    printf("value-head output scale: mean |v| %.4f, sd %.4f\n",
           S.vabs_sum / (double)S.n,
           sqrt(fabs(S.v_sq / (double)S.n - (S.v_sum / (double)S.n) * (S.v_sum / (double)S.n))));
    printf("corr(V after move, oracle value of move)  r = %+.4f over %ld moves\n",
           pearson(S.corr_xy, S.corr_x, S.corr_y, S.corr_xx, S.corr_yy, S.corr_n),
           S.corr_n);
}

/* ========================================================================= */
/*                          E1b: what the priors say                         */
/* ========================================================================= */

typedef struct {
    double p_blund_sum, p_worst_sum, p_best_sum, p_uniform_sum;
    long   n, top1_blunder, top1_best;
    double rank_worst_sum;
    double mass_ratio_sum;
} PriorStat;

static void prior_run(const Model *M, const DPos *C, int n, PriorStat *S)
{
    memset(S, 0, sizeof *S);
    for (int i = 0; i < n; i++) {
        const DPos *d = &C[i];
        Position p;
        float P[MAX_MOVES];
        double pb = 0.0, uni;
        int top = 0, rank = 1;

        if (!pos_from_fen(&p, d->fen)) continue;
        net_priors(M, &p, d->moves, d->nmoves, P, NULL);
        for (int k = 0; k < d->nmoves; k++) {
            if (IS_HANG(d, k)) pb += P[k];
            if (P[k] > P[top]) top = k;
            if (P[k] > P[d->wi]) rank++;
        }
        uni = (double)d->nblunder / (double)d->nmoves;
        S->p_blund_sum   += pb;
        S->p_uniform_sum += uni;
        S->mass_ratio_sum += (uni > 0.0) ? pb / uni : 1.0;
        S->p_worst_sum   += P[d->wi];
        S->p_best_sum    += P[d->bi];
        S->rank_worst_sum += rank;
        if (IS_HANG(d, top)) S->top1_blunder++;
        if (IS_HOLD(d, top)) S->top1_best++;
        S->n++;
    }
}

static void prior_report(const char *tag, const Model *M, const DPos *C, int n)
{
    PriorStat S;
    prior_run(M, C, n, &S);
    printf("\n--- E1b  POLICY PRIORS: how much search budget goes to blunders? [%s] ---\n", tag);
    printf("positions                                 %ld\n", S.n);
    printf("prior mass on blunder moves               %.4f\n", S.p_blund_sum / (double)S.n);
    printf("  same mass under a UNIFORM prior         %.4f\n", S.p_uniform_sum / (double)S.n);
    printf("  ratio (1.0 = policy is blind to it)     %.3f\n",
           (S.p_blund_sum / (double)S.n) / (S.p_uniform_sum / (double)S.n));
    printf("prior on the WORST blunder                %.4f\n", S.p_worst_sum / (double)S.n);
    printf("prior on the oracle-best move             %.4f\n", S.p_best_sum / (double)S.n);
    printf("mean rank of the worst blunder            %.1f\n", S.rank_worst_sum / (double)S.n);
    printf("policy argmax IS a blunder                %.3f\n",
           (double)S.top1_blunder / (double)S.n);
    printf("policy argmax is a safe move              %.3f\n",
           (double)S.top1_best / (double)S.n);
}

/* ========================================================================= */
/*             E1c: the blunder rate against the simulation count            */
/* ========================================================================= */

#define MAX_SIMPTS 12

typedef struct {
    const Model *M;
    const DPos  *C;
    int          lo, hi;
    const int   *simpts;
    int          nsim;
    /* outputs, one per simulation point */
    long    blunders[MAX_SIMPTS];
    long    misses [MAX_SIMPTS];
    long    exact  [MAX_SIMPTS];
    double  loss   [MAX_SIMPTS];
    double  visfrac[MAX_SIMPTS];   /* visit share the blunder moves received  */
    long    n;
    uint64_t pool_exh;
    double  secs;
} SimWorker;

static void *sims_thread(void *arg)
{
    SimWorker *w = (SimWorker *)arg;
    const int maxsims = w->simpts[w->nsim - 1];
    Mcts m;
    double t0 = now_sec();

    mcts_init(&m, 50 * (maxsims + 8) + 1);
    m.reuse = 0;                       /* every probe gets its own fresh tree */

    for (int i = w->lo; i < w->hi; i++) {
        const DPos *d = &w->C[i];
        Game g;
        if (!game_start_fen(&g, d->fen)) continue;
        for (int s = 0; s < w->nsim; s++) {
            uint64_t rng[4];
            int32_t  visits[MAX_MOVES];
            float    rv;
            int      nr, top = 0;
            double   vb = 0.0, vt = 0.0;

            rng_seed(rng, 0xC0FFEEULL + (uint64_t)i * 1009u + (uint64_t)s);
            mcts_cache_clear(&m);
            nr = mcts_search(&m, w->M->tr, model_head(w->M), &g,
                             w->simpts[s], 0, rng, visits, &rv);
            if (nr != d->nmoves) continue;
            for (int k = 0; k < nr; k++) {
                if (visits[k] > visits[top]) top = k;
                vt += (double)visits[k];
                if (IS_HANG(d, k)) vb += (double)visits[k];
            }
            if (IS_HANG(d, top)) w->blunders[s]++;
            if (IS_MISS(d, top)) w->misses[s]++;
            if (IS_HOLD(d, top)) w->exact[s]++;
            w->loss[s]    += (double)(d->mat - d->val[top]);
            w->visfrac[s] += (vt > 0.0) ? vb / vt : 0.0;
        }
        w->n++;
    }
    w->pool_exh = mcts_pool_exhausted(&m);
    w->secs = now_sec() - t0;
    mcts_free(&m);
    return NULL;
}

typedef struct {
    long   n;
    long   blunders[MAX_SIMPTS];
    long   misses[MAX_SIMPTS];
    long   exact[MAX_SIMPTS];
    double loss[MAX_SIMPTS];
    double visfrac[MAX_SIMPTS];
    uint64_t pool_exh;
    double secs;
} SimResult;

static void sims_run(const Model *M, const DPos *C, int n, const int *simpts,
                     int nsim, int threads, SimResult *R)
{
    SimWorker *w = calloc((size_t)threads, sizeof *w);
    pthread_t *t = calloc((size_t)threads, sizeof *t);
    const int per = (n + threads - 1) / threads;
    double t0 = now_sec();

    memset(R, 0, sizeof *R);
    for (int i = 0; i < threads; i++) {
        w[i].M = M; w[i].C = C; w[i].simpts = simpts; w[i].nsim = nsim;
        w[i].lo = i * per;
        w[i].hi = (i + 1) * per; if (w[i].hi > n) w[i].hi = n;
        if (w[i].lo > n) w[i].lo = n;
        pthread_create(&t[i], NULL, sims_thread, &w[i]);
    }
    for (int i = 0; i < threads; i++) {
        pthread_join(t[i], NULL);
        R->n += w[i].n;
        R->pool_exh += w[i].pool_exh;
        for (int s = 0; s < nsim; s++) {
            R->blunders[s] += w[i].blunders[s];
            R->misses[s]   += w[i].misses[s];
            R->exact[s]    += w[i].exact[s];
            R->loss[s]     += w[i].loss[s];
            R->visfrac[s]  += w[i].visfrac[s];
        }
    }
    R->secs = now_sec() - t0;
    free(w); free(t);
}

/* The rate a uniformly random legal mover would post on this corpus. */
static double corpus_random_rate(const DPos *C, int n)
{
    double s = 0.0;
    for (int i = 0; i < n; i++) s += (double)C[i].nblunder / (double)C[i].nmoves;
    return n ? s / (double)n : 0.0;
}
static double corpus_random_miss(const DPos *C, int n)
{
    double s = 0.0;
    for (int i = 0; i < n; i++) s += (double)C[i].nmiss / (double)C[i].nmoves;
    return n ? s / (double)n : 0.0;
}

static void sims_report(const char *tag, const Model *M, const DPos *C, int n,
                        const int *simpts, int nsim, int threads)
{
    SimResult R;
    PriorStat P;
    sims_run(M, C, n, simpts, nsim, threads, &R);
    prior_run(M, C, n, &P);
    printf("\n--- E1c  BLUNDER RATE vs SIMULATION COUNT [%s] ---\n", tag);
    printf("model %s (gen %d)   positions %ld   %.1fs\n",
           M->name, M->gen, R.n, R.secs);
    printf("  HANG   = the chosen move drops >= %.1f pawns of its OWN material\n",
           (double)BLUNDER_PAWNS);
    printf("  MISS   = the chosen move is >= %.1f pawns below the oracle's best\n",
           (double)BLUNDER_PAWNS);
    printf("  HOLD   = the chosen move keeps its material (drop <= %.2f)\n",
           (double)SAFE_PAWNS);
    printf("\n   sims | HANG rate | MISS rate | HOLD rate | mean drop | visit share on hangs\n");
    printf("  ------+-----------+-----------+-----------+-----------+---------------------\n");
    printf("  POLICY|   %6.3f  |         - |   %6.3f  |       -   |            -\n",
           (double)P.top1_blunder / (double)P.n, (double)P.top1_best / (double)P.n);
    for (int s = 0; s < nsim; s++)
        printf("  %5d |   %6.3f  |   %6.3f  |   %6.3f  |   %+5.2f   |        %6.3f\n",
               simpts[s],
               (double)R.blunders[s] / (double)R.n,
               (double)R.misses[s]   / (double)R.n,
               (double)R.exact[s]    / (double)R.n,
               R.loss[s] / (double)R.n,
               R.visfrac[s] / (double)R.n);
    printf("  ------+-----------+-----------+-----------+-----------+---------------------\n");
    printf("  uniform random legal mover: HANG %.3f  MISS %.3f\n",
           corpus_random_rate(C, n), corpus_random_miss(C, n));
    printf("  POLICY = the policy head's argmax with no search at all.  The sims=1\n"
           "  row is NOT that: mcts_select() uses sqrt(N_parent - 1), which is 0 on a\n"
           "  node's first descent, so the first descent takes the FIRST LEGAL MOVE\n"
           "  and the priors only take over from the second (src/mcts.c:576-592).\n");
    if (R.pool_exh) printf("  WARNING: node pool exhausted %llu times\n",
                           (unsigned long long)R.pool_exh);
}

/* ========================================================================= */
/*                     E2: the shape of the tree it builds                   */
/* ========================================================================= */

#define MAXD 40

typedef struct {
    long   n;
    double legal, gt1, gt5, gt20, gt0;
    double top_share, second_share;
    double leafdepth[MAXD];
    double mean_leaf, max_leaf;
    double blunder_visits;      /* visits the worst blunder edge received     */
    long   blunder_expanded;    /* its subtree got a child expanded           */
    long   punish_visited;      /* the refuting recapture was actually tried  */
    double punish_visits;
    long   blunder_seen;
    /* what the SEARCH concluded about the two edges, root mover's view */
    double q_hang, q_safe;
    long   q_n, q_sep;
    /* the refutation as the CHILD's policy head ranks it */
    double ref_prior, ref_rank;
    long   ref_n, ref_top1, ref_top3;
    /* conditioned on the search actually choosing the hanging move */
    long   chose_hang;
    double chose_visits, chose_q;
    long   chose_ref_seen;
    double chose_ref_visits, chose_ref_prior, chose_ref_rank;
    double chose_ref_q;
    long   chose_ref_qn;
    /* the sqrt(N-1) first-descent rule: how often a node's only look went to
     * move-generation order instead of to the policy's own choice */
    double nodes_expanded, first_is_genorder, first_is_maxp, onelook;
} TreeStat;

/* Walks the finished tree accumulating the leaf-depth histogram.  A simulation
 * that ended at node v is counted at v's depth; that number is
 * N(v) - sum of N over v's children, which is exact under mcts.c's convention. */
static void tree_walk(const Mcts *m, int idx, int depth, TreeStat *S, double *tot)
{
    const MctsNode *nd = &m->pool[idx];
    long child_sum = 0;
    int d = depth < MAXD - 1 ? depth : MAXD - 1;
    if (nd->first >= 0 && nd->nchild > 0)
        for (int i = 0; i < nd->nchild; i++) child_sum += m->pool[nd->first + i].N;
    if (nd->N > child_sum) {
        S->leafdepth[d] += (double)(nd->N - child_sum);
        *tot            += (double)(nd->N - child_sum);
        S->mean_leaf    += (double)(nd->N - child_sum) * (double)depth;
        if ((double)depth > S->max_leaf) S->max_leaf = (double)depth;
    }
    if (nd->first >= 0 && nd->nchild > 0) {
        const MctsNode *ch = m->pool + nd->first;
        int mp = 0, nvis = 0;
        for (int i = 0; i < nd->nchild; i++) {
            if (ch[i].P > ch[mp].P) mp = i;
            if (ch[i].N > 0) nvis++;
        }
        if (nvis > 0) {
            S->nodes_expanded++;
            if (nvis == 1) S->onelook++;
            if (ch[0].N > 0 && ch[mp].N == 0) S->first_is_genorder++;
            if (ch[mp].N > 0) S->first_is_maxp++;
        }
        for (int i = 0; i < nd->nchild; i++)
            tree_walk(m, nd->first + i, depth + 1, S, tot);
    }
}

static void tree_report(const char *tag, const Model *M, const DPos *C, int n,
                        int sims)
{
    TreeStat S;
    Mcts m;
    double totleaf = 0.0;
    long used = 0;

    memset(&S, 0, sizeof S);
    mcts_init(&m, 50 * (sims + 8) + 1);
    m.reuse = 0;

    for (int i = 0; i < n; i++) {
        const DPos *d = &C[i];
        Game g;
        uint64_t rng[4];
        int32_t  visits[MAX_MOVES];
        float    rv;
        int      nr, top = 0, second = -1;
        double   leaf0 = totleaf;

        if (!game_start_fen(&g, d->fen)) continue;
        rng_seed(rng, 0xBEEFULL + (uint64_t)i);
        mcts_cache_clear(&m);
        nr = mcts_search(&m, M->tr, model_head(M), &g, sims, 0, rng, visits, &rv);
        if (nr != d->nmoves) continue;

        for (int k = 0; k < nr; k++) {
            if (visits[k] > 0)  S.gt0++;
            if (visits[k] > 1)  S.gt1++;
            if (visits[k] > 5)  S.gt5++;
            if (visits[k] > 20) S.gt20++;
            if (visits[k] > visits[top]) top = k;
        }
        for (int k = 0; k < nr; k++)
            if (k != top && (second < 0 || visits[k] > visits[second])) second = k;
        S.legal += nr;
        S.top_share    += (double)visits[top] / (double)sims;
        if (second >= 0) S.second_share += (double)visits[second] / (double)sims;

        tree_walk(&m, 0, 0, &S, &totleaf);
        (void)leaf0;
        used += m.used;

        /* Did the search ever LOOK at the punishment for the worst blunder? */
        {
            const MctsNode *root = &m.pool[0];
            const int wi = d->wi, si = d->si;
            if (root->first >= 0 && wi < root->nchild && si < root->nchild) {
                const MctsNode *bn = &m.pool[root->first + wi];
                const MctsNode *sn = &m.pool[root->first + si];
                int ref_seen = 0;
                double ref_vis = 0.0;

                S.blunder_visits += (double)bn->N;
                S.blunder_seen++;

                /* The search's OWN verdict on the two edges.  W is stored from
                 * the child's mover view, so the root mover's Q is -W/N. */
                if (bn->N > 0 && sn->N > 0) {
                    const double qh = -(double)bn->W / (double)bn->N;
                    const double qs = -(double)sn->W / (double)sn->N;
                    S.q_hang += qh; S.q_safe += qs; S.q_n++;
                    if (qs > qh) S.q_sep++;
                }

                {   /* Where does the CHILD's policy head rank the refutation?
                     * With only a couple of visits the child will expand only
                     * its top one or two priors, so this decides whether the
                     * punishment is ever looked at. */
                    Position p;
                    if (pos_from_fen(&p, d->fen)) {
                        Undo u;
                        Move  cl[MAX_MOVES];
                        float P[MAX_MOVES];
                        Move  refute;
                        int   cn, ri = -1, rank = 1;
                        make_move(&p, d->moves[wi], &u);
                        refute = or_best_move(&p);
                        cn = gen_legal(&p, cl);
                        if (cn > 0 && refute != MV_NONE) {
                            net_priors(M, &p, cl, cn, P, NULL);
                            for (int k = 0; k < cn; k++) if (cl[k] == refute) ri = k;
                            if (ri >= 0) {
                                for (int k = 0; k < cn; k++) if (P[k] > P[ri]) rank++;
                                S.ref_prior += P[ri];
                                S.ref_rank  += rank;
                                S.ref_n++;
                                if (rank == 1) S.ref_top1++;
                                if (rank <= 3) S.ref_top3++;
                            }
                        }
                        unmake_move(&p, d->moves[wi], &u);

                        if (bn->first >= 0 && bn->nchild > 0) {
                            S.blunder_expanded++;
                            for (int k = 0; k < bn->nchild; k++) {
                                const MctsNode *cnp = &m.pool[bn->first + k];
                                if (cnp->move == refute) {
                                    if (cnp->N > 0) { S.punish_visited++; ref_seen = 1; }
                                    S.punish_visits += (double)cnp->N;
                                    ref_vis = (double)cnp->N;
                                    break;
                                }
                            }
                        }
                    }
                }

                (void)ref_seen; (void)ref_vis;

                /* And when the search actually PLAYS a hanging move?  Redo the
                 * whole analysis for THAT move, not for the worst one. */
                if (IS_HANG(d, top)) {
                    const MctsNode *tn = &m.pool[root->first + top];
                    Position p;
                    S.chose_hang++;
                    S.chose_visits += (double)tn->N;
                    if (tn->N > 0) S.chose_q += -(double)tn->W / (double)tn->N;
                    if (pos_from_fen(&p, d->fen)) {
                        Undo u;
                        Move cl[MAX_MOVES], refute;
                        float P[MAX_MOVES];
                        int cn, ri = -1, rank = 1;
                        make_move(&p, d->moves[top], &u);
                        refute = or_best_move(&p);
                        cn = gen_legal(&p, cl);
                        if (cn > 0 && refute != MV_NONE) {
                            net_priors(M, &p, cl, cn, P, NULL);
                            for (int k = 0; k < cn; k++) if (cl[k] == refute) ri = k;
                            if (ri >= 0) {
                                for (int k = 0; k < cn; k++) if (P[k] > P[ri]) rank++;
                                S.chose_ref_prior += P[ri];
                                S.chose_ref_rank  += rank;
                            }
                        }
                        unmake_move(&p, d->moves[top], &u);
                        if (tn->first >= 0 && tn->nchild > 0 && refute != MV_NONE)
                            for (int k = 0; k < tn->nchild; k++) {
                                const MctsNode *cnp = &m.pool[tn->first + k];
                                if (cnp->move == refute) {
                                    if (cnp->N > 0) S.chose_ref_seen++;
                                    S.chose_ref_visits += (double)cnp->N;
                                    /* what did the search decide about the
                                     * refutation once it had looked at it? */
                                    if (cnp->N > 0) {
                                        S.chose_ref_q += -(double)cnp->W / (double)cnp->N;
                                        S.chose_ref_qn++;
                                    }
                                    break;
                                }
                            }
                    }
                }
            }
        }
        S.n++;
    }

    printf("\n--- E2  TREE SHAPE at %d simulations [%s] ---\n", sims, tag);
    printf("positions %ld   mean legal moves %.1f   mean nodes allocated %.0f\n",
           S.n, S.legal / (double)S.n, (double)used / (double)S.n);
    printf("root moves receiving  >0 visits  %.1f of %.1f  (%.0f%%)\n",
           S.gt0 / (double)S.n, S.legal / (double)S.n, 100.0 * S.gt0 / S.legal);
    printf("root moves receiving  >1 visits  %.1f\n",  S.gt1  / (double)S.n);
    printf("root moves receiving  >5 visits  %.1f\n",  S.gt5  / (double)S.n);
    printf("root moves receiving >20 visits  %.1f\n",  S.gt20 / (double)S.n);
    printf("visit share of the top move %.3f, of the runner-up %.3f\n",
           S.top_share / (double)S.n, S.second_share / (double)S.n);
    printf("\nsimulation-path (leaf) depth, plies below the root:\n");
    {
        double cum = 0.0;
        for (int d = 0; d < MAXD; d++) {
            if (S.leafdepth[d] <= 0.0) continue;
            cum += S.leafdepth[d];
            printf("   depth %2d : %6.2f%%   (cumulative %6.2f%%)\n",
                   d, 100.0 * S.leafdepth[d] / totleaf, 100.0 * cum / totleaf);
        }
    }
    printf("mean leaf depth %.2f plies, deepest %.0f\n",
           S.mean_leaf / totleaf, S.max_leaf);
    printf("\nexpanded nodes with at least one child visited: %.0f per position\n",
           S.nodes_expanded / (double)S.n);
    printf("  of those, only ONE child was ever looked at        %.3f\n",
           S.first_is_maxp ? S.onelook / S.nodes_expanded : 0.0);
    printf("  the policy's own top move was never visited        %.3f\n",
           1.0 - S.first_is_maxp / S.nodes_expanded);
    printf("  visited move-generation-order child 0 but NOT the top prior %.3f\n",
           S.first_is_genorder / S.nodes_expanded);
    printf("\nTHE BLUNDER EDGE (worst hanging move at the root):\n");
    printf("  mean visits it receives            %.2f of %d\n",
           S.blunder_visits / (double)S.blunder_seen, sims);
    printf("  fraction whose subtree was expanded %.3f\n",
           (double)S.blunder_expanded / (double)S.blunder_seen);
    printf("  fraction where the refuting capture was searched at all %.3f\n",
           (double)S.punish_visited / (double)S.blunder_seen);
    printf("  mean visits given to that refutation %.2f\n",
           S.punish_visits / (double)S.blunder_seen);
    if (S.ref_n)
        printf("  the CHILD policy head ranks that refutation: prior %.4f, "
               "mean rank %.1f, top-1 %.3f, top-3 %.3f\n",
               S.ref_prior / (double)S.ref_n, S.ref_rank / (double)S.ref_n,
               (double)S.ref_top1 / (double)S.ref_n,
               (double)S.ref_top3 / (double)S.ref_n);
    if (S.q_n)
        printf("\nTHE SEARCH'S OWN VERDICT after %d simulations:\n"
               "  mean Q(hanging edge) %+.4f   mean Q(safe edge) %+.4f   "
               "P[Q(safe) > Q(hang)] %.3f\n",
               sims, S.q_hang / (double)S.q_n, S.q_safe / (double)S.q_n,
               (double)S.q_sep / (double)S.q_n);
    if (S.chose_hang) {
        const double ch = (double)S.chose_hang;
        printf("\nWHEN THE SEARCH ACTUALLY PLAYS A HANGING MOVE (%ld of %ld positions):\n",
               S.chose_hang, S.n);
        printf("  visits it gave that move          %.1f of %d\n", S.chose_visits / ch, sims);
        printf("  Q the search settled on           %+.4f  (root mover's view)\n",
               S.chose_q / ch);
        printf("  refutation's prior at the child   %.4f, mean rank %.1f\n",
               S.chose_ref_prior / ch, S.chose_ref_rank / ch);
        printf("  refutation was searched           %.3f of the time, %.1f visits\n",
               (double)S.chose_ref_seen / ch, S.chose_ref_visits / ch);
        if (S.chose_ref_qn)
            printf("  Q the search gave the refutation  %+.4f  (root mover's view:"
                   " negative = it understood)\n", S.chose_ref_q / (double)S.chose_ref_qn);
    }
    mcts_free(&m);
}

/* ========================================================================= */
/*                     E3: is the value head calibrated?                     */
/* ========================================================================= */

#define CAL_MAXPOS 400000

typedef struct {
    float v;        /* value head, mover's view                              */
    float mat;      /* oracle material balance, mover's view                 */
    float z;        /* game result from that mover's view: +1 / 0 / -1       */
    int   ply;
    int   phase;
} CalRec;

typedef struct {
    const Model *M;
    int      games;
    int      sims;
    int      max_plies;
    int      vs_oracle;     /* 1 = the agent plays the material-1 oracle      */
    int      start;
    int      train_mode;    /* 1 = root noise + the training temperatures     */
    uint64_t seed;
    CalRec  *out;
    int      cap, got;
    long     wins, losses, draws;
} CalWorker;

static void *cal_thread(void *arg)
{
    CalWorker *w = (CalWorker *)arg;
    Mcts m;
    uint64_t rng[4];

    rng_seed(rng, w->seed);
    mcts_init(&m, 50 * (w->sims + 8) + 1);

    for (int gi = 0; gi < w->games && w->got < w->cap; gi++) {
        Game g;
        int  first = w->got, agent_side = gi & 1, res;

        if (w->start == ST_CLASSICAL) game_start(&g);
        else if (w->start == ST_960)  game_start960(&g, (int)(rng_next(rng) % 960u));
        else { if (rng_next(rng) & 1) game_start(&g);
               else game_start960(&g, (int)(rng_next(rng) % 960u)); }

        while (g.result == GR_ONGOING && g.ply < w->max_plies && w->got < w->cap) {
            Move list[MAX_MOVES];
            int  n = gen_legal(&g.pos, list);
            Move mv;
            if (n <= 0) break;

            if (w->vs_oracle && (int)g.pos.side != agent_side) {
                mv = or_pick_game(&g, rng);
            } else {
                int32_t visits[MAX_MOVES];
                float   rv;
                int     nr = mcts_search(&m, w->M->tr, model_head(w->M), &g,
                                         w->sims, w->train_mode, rng, visits, &rv);
                if (nr != n) break;
                {   /* record what the raw value head said about this position */
                    CalRec *r = &w->out[w->got++];
                    r->v     = net_value(w->M, &g.pos);
                    r->mat   = or_mat(&g.pos);
                    r->ply   = g.ply;
                    r->phase = phase_of(&g.pos);
                    r->z     = (float)(int)g.pos.side;   /* side, fixed up below */
                }
                mv = list[mcts_pick(visits, n,
                        w->train_mode ? (g.ply < 20 ? 1.0f : 0.25f)
                                      : (g.ply < 12 ? 1.0f : 0.0f), rng)];
            }
            game_push(&g, mv);
            game_update_result(&g, w->max_plies);
        }
        res = g.result;
        if (res == GR_WHITE_WIN) w->wins++;
        else if (res == GR_BLACK_WIN) w->losses++;
        else w->draws++;
        for (int i = first; i < w->got; i++) {
            const int side = (int)w->out[i].z;            /* stashed above */
            float z = 0.0f;
            if (res == GR_WHITE_WIN) z = (side == WHITE) ? 1.0f : -1.0f;
            else if (res == GR_BLACK_WIN) z = (side == BLACK) ? 1.0f : -1.0f;
            w->out[i].z = z;
        }
    }
    mcts_free(&m);
    return NULL;
}

static void calib_report(const char *tag, const Model *M, int games, int sims,
                         int vs_oracle, int start, int threads, uint64_t seed,
                         int train_mode)
{
    CalWorker *w = calloc((size_t)threads, sizeof *w);
    pthread_t *t = calloc((size_t)threads, sizeof *t);
    const int cap = CAL_MAXPOS / threads;
    long nb[12]; double vb[12], zb[12];
    long mn[13]; double mv[13], mz[13];
    long n = 0, wins = 0, losses = 0, draws = 0;
    double mse = 0.0, zmean = 0.0, zvar = 0.0;
    double cxy = 0, cx = 0, cy = 0, cxx = 0, cyy = 0;
    double t0 = now_sec();

    memset(nb, 0, sizeof nb); memset(vb, 0, sizeof vb); memset(zb, 0, sizeof zb);
    memset(mn, 0, sizeof mn); memset(mv, 0, sizeof mv); memset(mz, 0, sizeof mz);
    for (int i = 0; i < threads; i++) {
        w[i].M = M; w[i].games = (games + threads - 1) / threads;
        w[i].sims = sims; w[i].max_plies = 300; w[i].vs_oracle = vs_oracle;
        w[i].start = start; w[i].seed = seed + 77ULL * (uint64_t)(i + 1);
        w[i].train_mode = train_mode;
        w[i].out = calloc((size_t)cap, sizeof(CalRec));
        w[i].cap = cap;
        if (!w[i].out) { fprintf(stderr, "oom\n"); exit(1); }
        pthread_create(&t[i], NULL, cal_thread, &w[i]);
    }
    for (int i = 0; i < threads; i++) pthread_join(t[i], NULL);

    for (int i = 0; i < threads; i++) {
        wins += w[i].wins; losses += w[i].losses; draws += w[i].draws;
        for (int k = 0; k < w[i].got; k++) {
            const CalRec *r = &w[i].out[k];
            int b = (int)((r->v + 1.0f) * 5.0f);
            if (b < 0) b = 0; if (b > 9) b = 9;
            nb[b]++; vb[b] += r->v; zb[b] += r->z;
            {   /* material bucket: -6 .. +6 pawns, clamped */
                int mbi = (int)floor((double)r->mat + 0.5) + 6;
                if (mbi < 0) mbi = 0; if (mbi > 12) mbi = 12;
                mn[mbi]++; mv[mbi] += r->v; mz[mbi] += r->z;
            }
            mse += (double)(r->v - r->z) * (r->v - r->z);
            zmean += r->z; zvar += (double)r->z * r->z;
            cx += r->mat; cy += r->v;
            cxx += (double)r->mat * r->mat; cyy += (double)r->v * r->v;
            cxy += (double)r->mat * r->v;
            n++;
        }
    }
    zmean /= (double)n;
    zvar   = zvar / (double)n - zmean * zmean;
    mse   /= (double)n;

    printf("\n--- E3  VALUE-HEAD CALIBRATION [%s] ---\n", tag);
    printf("model %s   %d games at %d sims vs %s   %ld positions   %.1fs\n",
           M->name, games, sims, vs_oracle ? "the material-1 oracle" : "itself",
           n, now_sec() - t0);
    printf("move selection: %s\n", train_mode
           ? "TRAINING settings -- Dirichlet root noise on, temp 1.0 for 20 plies then 0.25"
           : "PLAY settings -- no root noise, temp 1.0 for 12 plies then argmax");
    printf("game results (white/black/draw): %ld / %ld / %ld\n", wins, losses, draws);
    printf("MSE(v, z) %.4f   Var(z) %.4f   MSE/Var %.4f   R^2 %.4f\n",
           mse, zvar, mse / zvar, 1.0 - mse / zvar);
    printf("\n   v bucket    |     n    | mean v  | actual score | gap\n");
    printf("  -------------+----------+---------+--------------+-------\n");
    for (int b = 0; b < 10; b++) {
        if (!nb[b]) continue;
        printf("  [%+.1f,%+.1f) | %8ld | %+.3f  |    %+.3f     | %+.3f\n",
               -1.0 + 0.2 * b, -1.0 + 0.2 * (b + 1), nb[b],
               vb[b] / (double)nb[b], zb[b] / (double)nb[b],
               vb[b] / (double)nb[b] - zb[b] / (double)nb[b]);
    }
    printf("\n  material (pawns) |     n    | mean v  | actual score\n");
    printf("  -----------------+----------+---------+-------------\n");
    for (int b = 0; b < 13; b++) {
        if (mn[b] < 20) continue;
        printf("   %s%+3d            | %8ld | %+.3f  |    %+.3f\n",
               b == 0 ? "<=" : b == 12 ? ">=" : "  ", b - 6, mn[b],
               mv[b] / (double)mn[b], mz[b] / (double)mn[b]);
    }
    printf("\ncorr(v, oracle material balance) r = %+.4f   (r^2 = %.4f)\n",
           pearson(cxy, cx, cy, cxx, cyy, n),
           pearson(cxy, cx, cy, cxx, cyy, n) * pearson(cxy, cx, cy, cxx, cyy, n));

    for (int i = 0; i < threads; i++) free(w[i].out);
    free(w); free(t);
}

/* ========================================================================= */
/*        E0: the end-to-end audit -- every move of real, complete games      */
/* ========================================================================= */
/* The corpus experiments condition on a position that offers a hang.  This one
 * conditions on nothing: it plays whole games and grades EVERY move the agent
 * makes with the same oracle, so the per-move rate can be multiplied out into
 * the match result that was actually observed externally. */

typedef struct {
    const Model *M;
    int      games, sims, max_plies, vs_oracle, start, temp0;
    uint64_t seed;
    long     moves, hangs, misses, holds, unholdable, forced;
    double   drop_sum, loss_sum;
    long     games_done, games_with_hang, plies;
    long     res_agent_win, res_agent_loss, res_draw;
    long     opp_moves, opp_hangs;
    double   mat_at[8];        /* mean material balance at ply 0,10,...,70    */
    long     mat_n[8];
    long     hang_per_game[16];
} GameWorker;

static void *game_thread(void *arg)
{
    GameWorker *w = (GameWorker *)arg;
    Mcts m;
    uint64_t rng[4];

    rng_seed(rng, w->seed);
    mcts_init(&m, 50 * (w->sims + 8) + 1);
    m.reuse = 0;

    for (int gi = 0; gi < w->games; gi++) {
        Game g;
        const int agent_side = gi & 1;
        int  hangs_here = 0;

        if (w->start == ST_CLASSICAL) game_start(&g);
        else game_start960(&g, (int)(rng_next(rng) % 960u));

        while (g.result == GR_ONGOING && g.ply < w->max_plies) {
            Move  mv[MAX_MOVES];
            float vl[MAX_MOVES];
            uint64_t onodes = 0;
            int   n = or_root(&g.pos, mv, vl, &onodes);
            int   b = 0, chosen = -1;
            float mat = or_mat(&g.pos);

            if (n <= 0) break;
            for (int i = 1; i < n; i++) if (vl[i] > vl[b]) b = i;

            if ((int)g.pos.side == agent_side) {
                {
                    int32_t visits[MAX_MOVES];
                    float   rv;
                    int nr = mcts_search(&m, w->M->tr, model_head(w->M), &g,
                                         w->sims, 0, rng, visits, &rv);
                    if (nr != n) break;
                    chosen = mcts_pick(visits, n,
                                       (!w->temp0 && g.ply < 12) ? 1.0f : 0.0f, rng);
                }
                if (fabsf(vl[b]) < MAX_ORACLE_ABS && fabsf(vl[chosen]) < MAX_ORACLE_ABS) {
                    const float drop = mat - vl[chosen];
                    const float loss = vl[b] - vl[chosen];
                    w->moves++;
                    w->drop_sum += drop;
                    w->loss_sum += loss;
                    if (vl[b] < mat - 0.5f) w->unholdable++;
                    if (n < 4)              w->forced++;
                    if (drop >= BLUNDER_PAWNS) { w->hangs++; hangs_here++; }
                    if (loss >= BLUNDER_PAWNS) w->misses++;
                    if (drop <= SAFE_PAWNS)    w->holds++;
                }
                {   /* material trajectory, always from the AGENT's view */
                    int slot = g.ply / 10;
                    if (slot < 8) { w->mat_at[slot] += mat; w->mat_n[slot]++; }
                }
            } else {
                if (w->vs_oracle) {
                    Move om = or_pick_game(&g, rng);
                    chosen = -1;
                    for (int i = 0; i < n; i++) if (mv[i] == om) { chosen = i; break; }
                    if (chosen < 0) break;
                    w->opp_moves++;
                    if (mat - vl[chosen] >= BLUNDER_PAWNS) w->opp_hangs++;
                } else {
                    int32_t visits[MAX_MOVES];
                    float   rv;
                    int nr = mcts_search(&m, w->M->tr, model_head(w->M), &g,
                                         w->sims, 0, rng, visits, &rv);
                    if (nr != n) break;
                    chosen = mcts_pick(visits, n,
                                       (!w->temp0 && g.ply < 12) ? 1.0f : 0.0f, rng);
                    w->opp_moves++;
                    if (mat - vl[chosen] >= BLUNDER_PAWNS) w->opp_hangs++;
                }
            }
            if (chosen < 0 || chosen >= n) break;
            game_push(&g, mv[chosen]);
            game_update_result(&g, w->max_plies);
        }
        w->games_done++;
        w->plies += g.ply;
        if (hangs_here) w->games_with_hang++;
        w->hang_per_game[hangs_here < 15 ? hangs_here : 15]++;
        if (g.result == GR_DRAW || g.result == GR_ONGOING) w->res_draw++;
        else if ((g.result == GR_WHITE_WIN) == (agent_side == WHITE)) w->res_agent_win++;
        else w->res_agent_loss++;
    }
    mcts_free(&m);
    return NULL;
}

static void game_report(const Model *M, int games, int sims, int vs_oracle,
                        int start, int threads, uint64_t seed, int temp0)
{
    GameWorker *w = calloc((size_t)threads, sizeof *w);
    pthread_t  *t = calloc((size_t)threads, sizeof *t);
    GameWorker  A;
    double t0 = now_sec();

    memset(&A, 0, sizeof A);
    for (int i = 0; i < threads; i++) {
        w[i].M = M; w[i].games = (games + threads - 1) / threads;
        w[i].sims = sims; w[i].max_plies = 300; w[i].vs_oracle = vs_oracle;
        w[i].start = start; w[i].seed = seed + 991ULL * (uint64_t)(i + 1);
        w[i].temp0 = temp0;
        pthread_create(&t[i], NULL, game_thread, &w[i]);
    }
    for (int i = 0; i < threads; i++) {
        pthread_join(t[i], NULL);
        A.moves += w[i].moves; A.hangs += w[i].hangs; A.misses += w[i].misses;
        A.holds += w[i].holds; A.unholdable += w[i].unholdable;
        A.forced += w[i].forced;
        A.drop_sum += w[i].drop_sum; A.loss_sum += w[i].loss_sum;
        A.games_done += w[i].games_done; A.games_with_hang += w[i].games_with_hang;
        A.plies += w[i].plies;
        A.res_agent_win += w[i].res_agent_win;
        A.res_agent_loss += w[i].res_agent_loss;
        A.res_draw += w[i].res_draw;
        A.opp_moves += w[i].opp_moves; A.opp_hangs += w[i].opp_hangs;
        for (int k = 0; k < 8;  k++) { A.mat_at[k] += w[i].mat_at[k]; A.mat_n[k] += w[i].mat_n[k]; }
        for (int k = 0; k < 16; k++) A.hang_per_game[k] += w[i].hang_per_game[k];
    }

    printf("\n--- E0  EVERY MOVE OF %ld COMPLETE GAMES (%d sims, vs %s, %s) ---\n",
           A.games_done, sims, vs_oracle ? "material-1" : "itself",
           temp0 ? "argmax from move 1" : "temperature 1 for 12 plies, then argmax");
    printf("model %s   %.1fs\n", M->name, now_sec() - t0);
    printf("agent result  W %ld  L %ld  D %ld   score %.3f   mean length %.0f plies\n",
           A.res_agent_win, A.res_agent_loss, A.res_draw,
           ((double)A.res_agent_win + 0.5 * (double)A.res_draw) / (double)A.games_done,
           (double)A.plies / (double)A.games_done);
    printf("agent moves graded            %ld\n", A.moves);
    printf("  HANG (drops >= 2 pawns)     %.4f   (%ld)\n",
           (double)A.hangs / (double)A.moves, A.hangs);
    printf("  MISS (>= 2 below best)      %.4f\n", (double)A.misses / (double)A.moves);
    printf("  HOLD (keeps its material)   %.4f\n", (double)A.holds / (double)A.moves);
    printf("  mean drop  %+.3f pawns/move   mean loss %+.3f pawns/move\n",
           A.drop_sum / (double)A.moves, A.loss_sum / (double)A.moves);
    printf("  positions where nothing could be held  %.4f\n",
           (double)A.unholdable / (double)A.moves);
    if (A.opp_moves)
        printf("opponent HANG rate for comparison   %.4f (%ld of %ld)\n",
               (double)A.opp_hangs / (double)A.opp_moves, A.opp_hangs, A.opp_moves);
    printf("games containing at least one hang  %.3f\n",
           (double)A.games_with_hang / (double)A.games_done);
    printf("hangs per game: ");
    for (int k = 0; k < 10; k++)
        if (A.hang_per_game[k]) printf("%d:%ld  ", k, A.hang_per_game[k]);
    printf("\n");
    printf("material balance (agent's view) by ply:  ");
    for (int k = 0; k < 8; k++)
        if (A.mat_n[k]) printf("p%d %+.2f  ", k * 10, A.mat_at[k] / (double)A.mat_n[k]);
    printf("\n");
    free(w); free(t);
}

/* ========================================================================= */
/*        E3b: how many value units is a piece worth to the value head?       */
/* ========================================================================= */
/* The most direct question there is, and it needs no search at all: delete one
 * of the side-to-move's pieces from the board and re-evaluate.  The change in v
 * IS the value head's exchange rate between material and winning chances. */

static int pos_delete_piece(Position *p, int sq)
{
    const int c  = p->color_at[sq];
    const int pc = p->board[sq];
    if (c < 0 || pc == NO_PIECE || pc == KING) return 0;
    p->piece[c][pc] &= ~(1ULL << sq);
    p->occ[c]       &= ~(1ULL << sq);
    p->all          &= ~(1ULL << sq);
    p->board[sq]     = NO_PIECE;
    p->color_at[sq]  = -1;
    p->key = pos_compute_key(p);
    /* the position must still be legal: the side NOT to move may not be in
     * check while the other side is to move */
    return !in_check(p, p->side ^ 1);
}

static void ablate_report(const char *tag, const Model *M, const DPos *C, int n)
{
    static const char *PN[5] = { "pawn", "knight", "bishop", "rook", "queen" };
    double dsum[5] = {0,0,0,0,0}, vbase = 0.0;
    long   dn[5] = {0,0,0,0,0}, nb = 0;

    for (int i = 0; i < n; i++) {
        Position p0;
        float v0;
        if (!pos_from_fen(&p0, C[i].fen)) continue;
        v0 = net_value(M, &p0);
        vbase += v0; nb++;
        for (int pc = PAWN; pc <= QUEEN; pc++) {
            uint64_t bb = p0.piece[p0.side][pc];
            while (bb) {
                Position p = p0;
                const int sq = bb_pop(&bb);
                if (!pos_delete_piece(&p, sq)) continue;
                dsum[pc] += (double)(net_value(M, &p) - v0);
                dn[pc]++;
                break;                 /* one piece of each type per position */
            }
        }
    }
    printf("\n--- E3b  DELETE ONE OF MY OWN PIECES AND RE-EVALUATE [%s] ---\n", tag);
    printf("model %s   %ld positions, mean v before %+.4f\n",
           M->name, nb, vbase / (double)nb);
    printf("  piece  |   n    | mean change in v | pawns | value units per pawn\n");
    printf("  -------+--------+------------------+-------+---------------------\n");
    for (int pc = PAWN; pc <= QUEEN; pc++) {
        if (!dn[pc]) continue;
        printf("  %-6s | %6ld |      %+.4f       | %5.2f |       %+.4f\n",
               PN[pc], dn[pc], dsum[pc] / (double)dn[pc], (double)OR_PV[pc],
               (dsum[pc] / (double)dn[pc]) / (double)OR_PV[pc]);
    }
    printf("  a perfectly scaled head would move v by most of its remaining\n"
           "  range when a queen leaves the board.\n");
}

/* ========================================================================= */
/*                       E4: phases, and 960 vs classical                    */
/* ========================================================================= */

static void phase_report(const char *tag, const Model *M, const DPos *C, int n,
                         int sims, int threads)
{
    for (int ph = 0; ph < PH_N; ph++) {
        DPos *sub = malloc((size_t)n * sizeof(DPos));
        int   k = 0;
        SimResult R;
        PriorStat P;
        ValueStat V;
        if (!sub) { fprintf(stderr, "oom\n"); return; }
        for (int i = 0; i < n; i++) if (C[i].phase == ph) sub[k++] = C[i];
        if (k < 20) { free(sub); printf("  %-11s : only %d positions, skipped\n",
                                        PHASE_NAME[ph], k); continue; }
        sims_run(M, sub, k, &sims, 1, threads, &R);
        prior_run(M, sub, k, &P);
        value_run(M, sub, k, &V);
        printf("  %-11s | n=%5d | blunder %.3f | random %.3f | prior-on-blunder %.3f"
               " | V sep %+.4f | P[V ok] %.3f\n",
               PHASE_NAME[ph], k,
               (double)R.blunders[0] / (double)R.n,
               corpus_random_rate(sub, k),
               P.p_blund_sum / (double)P.n,
               V.d1_sum / (double)V.n,
               (double)V.sep1 / (double)V.n);
        free(sub);
    }
    (void)tag;
}

/* ========================================================================= */
/*                                  main                                     */
/* ========================================================================= */

static void corpus_summary(const char *tag, const CorpusCfg *cfg, const DPos *C,
                           int n, uint64_t scanned, double secs)
{
    long ph[PH_N] = {0,0,0}, ck = 0, c9 = 0;
    double nm = 0, nb = 0, ns = 0, wl = 0, plysum = 0, mat = 0;
    for (int i = 0; i < n; i++) {
        ph[C[i].phase]++;
        ck += C[i].incheck;
        c9 += C[i].c960;
        nm += C[i].nmoves; nb += C[i].nblunder; ns += C[i].nmiss;
        wl += C[i].worst_loss; plysum += C[i].ply; mat += C[i].mat;
    }
    printf("\n--- CORPUS [%s] ---\n", tag);
    printf("source %s, starts %s, %d positions from %llu scanned (%.1f%%), %.1fs\n",
           SRC_NAME[cfg->source],
           cfg->start == ST_CLASSICAL ? "classical" :
           cfg->start == ST_960 ? "chess960" : "mixed",
           n, (unsigned long long)scanned,
           scanned ? 100.0 * (double)n / (double)scanned : 0.0, secs);
    printf("mean legal moves %.1f: %.1f hang material (%.1f%%), %.1f are >= 2 below best (%.1f%%)\n",
           nm / n, nb / n, 100.0 * nb / nm, ns / n, 100.0 * ns / nm);
    printf("mean worst hang %.2f pawns, mean ply %.0f, mean material balance %+.2f\n",
           wl / n, plysum / n, mat / n);
    printf("phases: opening %ld, middlegame %ld, endgame %ld; in check %ld; "
           "from a 960 array %ld\n", ph[0], ph[1], ph[2], ck, c9);
}

static const char *opt_str(int argc, char **argv, const char *flag, const char *def)
{
    for (int i = 0; i < argc - 1; i++) if (!strcmp(argv[i], flag)) return argv[i + 1];
    return def;
}
static long opt_int(int argc, char **argv, const char *flag, long def)
{
    const char *s = opt_str(argc, argv, flag, NULL);
    return s ? strtol(s, NULL, 10) : def;
}
static int opt_flag(int argc, char **argv, const char *flag)
{
    for (int i = 0; i < argc; i++) if (!strcmp(argv[i], flag)) return 1;
    return 0;
}

static void usage(void)
{
    printf(
    "diag_tactics <command> [options]\n"
    "  commands: game corpus value ablate prior sims tree calib phase compare all\n"
    "  --model P      checkpoint (default runs/az_hour/best.crl)\n"
    "  --old P        earlier checkpoint for `compare` (default runs/az_v3/best.crl)\n"
    "  --n N          corpus size (default 600)\n"
    "  --source S     random | self | vs        (default random)\n"
    "  --start S      classical | 960 | mixed   (default classical)\n"
    "  --gen-sims N   sims used to GENERATE agent games (default 160)\n"
    "  --sims N       sims for the single-budget experiments (default 160)\n"
    "  --games N      games for calib (default 64)\n"
    "  --vs-oracle    calib plays the material-1 oracle instead of itself\n"
    "  --self         `game` plays itself instead of the oracle\n"
    "  --train-mode   calib self-play uses root noise + the training temperatures\n"
    "  --temp0        `game` plays argmax from move 1 (no opening temperature)\n"
    "  --threads N    default = cores\n"
    "  --high         push the sims sweep to 25600\n"
    "  --seed N\n");
}

int main(int argc, char **argv)
{
    const char *cmd    = (argc > 1 && argv[1][0] != '-') ? argv[1] : "all";
    const char *mpath  = opt_str(argc, argv, "--model", "runs/az_hour/best.crl");
    const char *opath  = opt_str(argc, argv, "--old",   "runs/az_v3/best.crl");
    const int   want   = (int)opt_int(argc, argv, "--n", 600);
    const int   sims   = (int)opt_int(argc, argv, "--sims", 160);
    const int   gsims  = (int)opt_int(argc, argv, "--gen-sims", 160);
    const int   games  = (int)opt_int(argc, argv, "--games", 64);
    const uint64_t seed= (uint64_t)opt_int(argc, argv, "--seed", 20260912);
    int threads        = (int)opt_int(argc, argv, "--threads", cpu_count_local());
    const char *ssrc   = opt_str(argc, argv, "--source", "random");
    const char *sst    = opt_str(argc, argv, "--start",  "classical");
    const int   vsor   = opt_flag(argc, argv, "--vs-oracle");
    int simpts[MAX_SIMPTS] = { 1, 20, 80, 160, 400, 1600, 6400 };
    int nsim = 7;
    int high[4] = { 160, 1600, 6400, 25600 };
    CorpusCfg cfg;
    Model M;
    DPos *C;
    int n;
    uint64_t scanned;
    double secs;

    if (opt_flag(argc, argv, "-h") || opt_flag(argc, argv, "--help")) { usage(); return 0; }
    if (opt_flag(argc, argv, "--high")) { memcpy(simpts, high, sizeof high); nsim = 4; }
    if (threads < 1) threads = 1;
    chess_init();

    memset(&cfg, 0, sizeof cfg);
    cfg.source   = !strcmp(ssrc, "self") ? SRC_SELF : !strcmp(ssrc, "vs") ? SRC_VS : SRC_RAND;
    cfg.start    = !strcmp(sst, "960") ? ST_960 : !strcmp(sst, "mixed") ? ST_MIXED : ST_CLASSICAL;
    cfg.want     = want;
    cfg.max_plies= 300;
    cfg.per_game = 3;
    cfg.gen_sims = gsims;
    cfg.seed     = seed;

    if (!model_open(&M, mpath)) return 1;
    cfg.M = &M;
    printf("model      %s   generation %d   %d agents, best = %d (elo %.0f)\n",
           mpath, M.gen, M.n, M.best, (double)M.elo[M.best]);
    printf("backend    %s   threads %d   seed %llu\n",
           nn_backend(), threads, (unsigned long long)seed);
    printf("oracle     material-1 (depth 1 + %d-ply quiescence), P=1 N=3 B=3.25 R=5 Q=9\n",
           OR_QDEPTH);

    if (!strcmp(cmd, "game")) {
        game_report(&M, games, sims, !opt_flag(argc, argv, "--self"),
                    cfg.start, threads, seed, opt_flag(argc, argv, "--temp0"));
        model_close(&M);
        return 0;
    }

    /* `calib` needs no corpus. */
    if (!strcmp(cmd, "calib")) {
        calib_report(vsor ? "vs oracle" : "self-play", &M, games, sims, vsor,
                     cfg.start, threads, seed,
                     opt_flag(argc, argv, "--train-mode"));
        model_close(&M);
        return 0;
    }

    C = corpus_build(&cfg, threads, &n, &scanned, &secs);
    corpus_summary("main", &cfg, C, n, scanned, secs);
    if (n < 20) { fprintf(stderr, "corpus too small\n"); return 1; }

    if (!strcmp(cmd, "corpus")) {
        for (int i = 0; i < (n < 12 ? n : 12); i++) {
            char a[8], b[8], c[8];
            move_to_uci(C[i].moves[C[i].bi], a);
            move_to_uci(C[i].moves[C[i].si], b);
            move_to_uci(C[i].moves[C[i].wi], c);
            printf("  %s\n      mat %+.2f | best %s %+.2f | safe %s %+.2f | worst %s %+.2f"
                   " | %d/%d hang\n",
                   C[i].fen, (double)C[i].mat, a, (double)C[i].best,
                   b, (double)C[i].val[C[i].si], c, (double)C[i].val[C[i].wi],
                   C[i].nblunder, C[i].nmoves);
        }
    }
    if (!strcmp(cmd, "value") || !strcmp(cmd, "all")) value_report("main", &M, C, n);
    if (!strcmp(cmd, "ablate") || !strcmp(cmd, "all")) {
        Model O, Rw;
        ablate_report("best.crl", &M, C, n);
        if (model_open(&O, opath)) { ablate_report("earlier checkpoint", &O, C, n);
                                     model_close(&O); }
        if (model_random(&Rw, 12345)) { ablate_report("random weights", &Rw, C, n);
                                        model_close(&Rw); }
    }
    if (!strcmp(cmd, "prior") || !strcmp(cmd, "all")) prior_report("main", &M, C, n);
    if (!strcmp(cmd, "sims")  || !strcmp(cmd, "all"))
        sims_report("main", &M, C, n, simpts, nsim, threads);
    if (!strcmp(cmd, "tree")  || !strcmp(cmd, "all")) tree_report("main", &M, C, n, sims);

    if (!strcmp(cmd, "phase") || !strcmp(cmd, "all")) {
        printf("\n--- E4  BY PHASE (at %d sims) ---\n", sims);
        phase_report("main", &M, C, n, sims, threads);

        printf("\n--- E4  CLASSICAL vs CHESS960 STARTS (at %d sims) ---\n", sims);
        for (int s = 0; s < 2; s++) {
            CorpusCfg c2 = cfg;
            DPos *C2; int n2; uint64_t sc2; double se2;
            SimResult R; PriorStat P; ValueStat V;
            c2.start = s ? ST_960 : ST_CLASSICAL;
            c2.want  = want;
            c2.seed  = seed + 555u + (uint64_t)s;
            C2 = corpus_build(&c2, threads, &n2, &sc2, &se2);
            if (n2 < 20) { free(C2); continue; }
            sims_run(&M, C2, n2, &sims, 1, threads, &R);
            prior_run(&M, C2, n2, &P);
            value_run(&M, C2, n2, &V);
            printf("  %-9s | n=%5d | blunder %.3f | random %.3f | prior-on-blunder %.3f"
                   " | V sep %+.4f | P[V ok] %.3f\n",
                   s ? "chess960" : "classical", n2,
                   (double)R.blunders[0] / (double)R.n,
                   corpus_random_rate(C2, n2),
                   P.p_blund_sum / (double)P.n,
                   V.d1_sum / (double)V.n,
                   (double)V.sep1 / (double)V.n);
            free(C2);
        }
    }

    if (!strcmp(cmd, "compare") || !strcmp(cmd, "all")) {
        Model O, Rw;
        printf("\n--- E5  CHECKPOINT COMPARISON on the SAME corpus (%d positions) ---\n", n);
        printf("  %-34s | %-7s | %-7s | %-7s | %-8s | %-7s\n",
               "model", "blund@1", "blnd@160", "blnd@1600", "prior-bl", "V-sep");
        {
            Model *ms[3]; int nm = 0;
            ms[nm++] = &M;
            if (model_open(&O, opath)) ms[nm++] = &O; else O.tr = NULL;
            if (model_random(&Rw, 12345)) ms[nm++] = &Rw; else Rw.tr = NULL;
            for (int i = 0; i < nm; i++) {
                int pts[3] = { 1, 160, 1600 };
                SimResult R; PriorStat P; ValueStat V;
                char label[40];
                sims_run(ms[i], C, n, pts, 3, threads, &R);
                prior_run(ms[i], C, n, &P);
                value_run(ms[i], C, n, &V);
                snprintf(label, sizeof label, "%.25s g%d", ms[i]->name, ms[i]->gen);
                printf("  %-34s |  %.3f  |  %.3f  |  %.3f  |  %.4f  | %+.4f\n",
                       label,
                       (double)R.blunders[0] / (double)R.n,
                       (double)R.blunders[1] / (double)R.n,
                       (double)R.blunders[2] / (double)R.n,
                       P.p_blund_sum / (double)P.n,
                       V.d1_sum / (double)V.n);
            }
            printf("  %-34s |  %.3f  |  %.3f  |  %.3f  |  %.4f  |       -\n",
                   "uniform random legal mover", corpus_random_rate(C, n),
                   corpus_random_rate(C, n), corpus_random_rate(C, n),
                   corpus_random_rate(C, n));
            if (O.tr)  model_close(&O);
            if (Rw.tr) model_close(&Rw);
        }
    }

    free(C);
    model_close(&M);
    return 0;
}
