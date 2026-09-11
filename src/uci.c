/* uci.c -- Universal Chess Interface front end for the chessrl engine.
 *
 * Why this file exists
 * --------------------
 * The trainer's Elo is self-referential: it is measured only against frozen
 * snapshots of the population's own past selves, so it says "stronger than it
 * used to be" and nothing at all about absolute strength.  UCI is the bridge to
 * an external ruler -- with it the same binary is driven by cutechess-cli, by
 * any chess GUI, or by a bespoke match harness, and can therefore be matched
 * against Stockfish (or against a deliberately crippled copy of itself) using
 * standard tooling.
 *
 * WHICH ENGINE PLAYS  (docs/FROM_SCRATCH.md is the contract)
 * ---------------------------------------------------------
 * `Engine` selects the mover, and it DEFAULTS TO mcts:
 *
 *   mcts        PUCT Monte-Carlo tree search (src/mcts.c) over the learned
 *               priors, root noise OFF, temperature 0 (argmax of visit counts).
 *               THE SHIPPED AGENT.  The only evaluations are the network's
 *               policy and value heads; everything else is the rules of chess.
 *   policy      the raw policy argmax, no search whatsoever.  MEASUREMENT ONLY:
 *               it separates what the network learned from what the search
 *               contributes -- without it the two are confounded in one number.
 *   alphabeta   the src/search.c baseline -- a second, structurally different
 *               searcher over the SAME network.  MEASUREMENT ONLY.
 *
 * `Sims` is the MCTS strength dial: simulations per move.  `BlunderRate`
 * weakens play on purpose in every mode -- an agent that loses 100% of its
 * games tells you almost nothing, whereas a ladder of deliberately weakened
 * opponents brackets it from both sides.
 *
 * `UseSearch` is kept as a deprecated alias so existing harness scripts and
 * `--no-search` keep working: false means Engine=policy, true means the default
 * Engine=mcts.  It is no longer advertised in the option list.
 *
 * Structure
 * ---------
 * The reader loop owns stdin and never blocks on a search: searches run on a
 * worker thread, so `isready` and `stop` are answered while the engine thinks.
 *
 * Neither searcher exposes a per-iteration callback, and neither mcts.c nor
 * search.c is ours to change, so the `info` stream is produced by calling the
 * searcher once per budget step: search_best() once per target depth (1, 2, 3,
 * ...) in alphabeta mode, and mcts_search() once per simulation budget (32, 64,
 * 128, ...) in mcts mode.  Alpha-beta's transposition table survives across
 * those calls so revisiting shallow depths is nearly free; MCTS builds a fresh
 * tree each time, so the ramp costs at most 2x the final pass, which buys a
 * `go` that genuinely respects `movetime` and `wtime`.  Each step is given the
 * *remaining* clock, so the outer loop cannot overrun the budget.
 *
 * Nothing is ever written to stdout that is not valid UCI.  Diagnostics go out
 * as `info string`, which is part of the protocol; genuine errors go to stderr.
 */

#include <ctype.h>
#include <math.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <time.h>

#include "chess.h"
#include "mcts.h"
#include "net.h"
#include "search.h"

/* ------------------------------------------------------------- constants */

#define UCI_LINE_MAX      65536   /* 1024 plies of "e2e4 " still fits easily  */
#define UCI_MAX_TOKENS     4096
#define UCI_PV_MAX           64   /* == sizeof(Search.pv) / sizeof(Move)      */
#define UCI_MAX_DEPTH        64   /* search.c clamps to 60 of its own accord   */
#define UCI_PATH_MAX       1024

#define UCI_DEF_MODEL     "runs/pilot/best.crl"
#define UCI_DEF_HASH         64
#define UCI_MAX_HASH       1024
#define UCI_MAX_AGENT      1023
#define UCI_DEF_OVERHEAD     30   /* ms held back for GUI/transport latency   */
#define UCI_MAX_OVERHEAD   5000
#define UCI_BARE_GO_MS     1000   /* a "go" with no limits at all             */
#define UCI_MAX_SLICE_MS 3600000  /* clamp so movetime_ms cannot overflow int */
#define UCI_SETTLE_MS     60000  /* grace given to a self-terminating search  */

/* Mirrors of search.c's private S_MATE / S_MAX_PLY: anything at or beyond the
 * bound is a forced mate and is reported as `score mate N` rather than `cp`.
 * Duplicated rather than exported because search.h is not ours to change. */
#define UCI_MATE          30000
#define UCI_MATE_BOUND    (UCI_MATE - 64)

/* ------------------------------------------------------------ engine modes */
#define UCI_ENGINE_MCTS       0   /* the shipped agent, and the default       */
#define UCI_ENGINE_POLICY     1   /* measurement only                         */
#define UCI_ENGINE_ALPHABETA  2   /* measurement only                         */

/* ------------------------------------------------------------------- MCTS */
/* Simulations per move.  800 is the AlphaZero-paper figure for evaluation play
 * and costs about 4 ms here, which suits a 40/5' time control.  `go nodes N`
 * overrides it exactly (nodes ARE simulations); `go depth D` caps it at
 * D * UCI_SIMS_PER_DEPTH, the same map api.c uses, so a GUI's depth slider
 * still means "think harder". */
#define UCI_DEF_SIMS         800
#define UCI_MAX_SIMS      262144
#define UCI_SIMS_PER_DEPTH    64
#define UCI_SIMS_FIRST        32   /* first pass of the time-capped ramp      */

/* One simulation expands at most one node, which adds one child per legal move.
 * 40 is a comfortable bound on the average branch factor.  The pool is capped
 * so that an absurd `Sims` cannot ask for gigabytes: past the cap mcts.c
 * degrades gracefully -- it keeps evaluating and stops growing the tree. */
#define UCI_POOL_PER_SIM      40
#define UCI_POOL_MAX_NODES 2000000

/* `go infinite` / `go ponder` has no budget to ramp against, so the ramp would
 * otherwise run all the way to UCI_MAX_SIMS and a `stop` arriving early in that
 * pass would wait seconds for it.  mcts_search() cannot be interrupted (mcts.h
 * offers no stop hook and is not ours to change), so the pass size is what
 * bounds `stop` latency: 32768 simulations is about 300 ms, and the tree is
 * already far larger than any real time control would build. */
#define UCI_INF_SIMS       32768

/* ---------------------------------------------------------------- output */

static pthread_mutex_t g_out_mu = PTHREAD_MUTEX_INITIALIZER;

#if defined(__GNUC__) || defined(__clang__)
static void say(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
#endif

/* One complete UCI line, emitted atomically with respect to the other thread. */
static void say(const char *fmt, ...)
{
    va_list ap;
    pthread_mutex_lock(&g_out_mu);
    va_start(ap, fmt);
    vprintf(fmt, ap);
    va_end(ap);
    putchar('\n');
    fflush(stdout);
    pthread_mutex_unlock(&g_out_mu);
}

/* ----------------------------------------------------------------- clock */

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1.0e6;
}

static void nap_us(long us)
{
    struct timespec ts;
    ts.tv_sec  = us / 1000000L;
    ts.tv_nsec = (us % 1000000L) * 1000L;
    nanosleep(&ts, NULL);
}

/* ------------------------------------------------------------------- rng */
/* Only used by BlunderRate in raw-policy mode; xorshift64 is plenty. */

static uint64_t g_rng = 88172645463325252ULL;

static uint64_t rnd64(void)
{
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 7;
    g_rng ^= g_rng << 17;
    return g_rng;
}

static float rndf(void)
{
    return (float)((double)(rnd64() >> 11) * (1.0 / 9007199254740992.0));
}

/* --------------------------------------------------------- engine state */

static Trunk *g_trunk;
static Head  *g_heads;
static Hyper *g_hy;
static float *g_elo;
static int    g_nagents;
static int    g_generation;
static int    g_agent;          /* resolved index actually in use            */
static int    g_have_model;

static Search g_search;         /* measurement-only alpha-beta baseline      */
static Mcts   g_mcts;           /* the shipped agent                         */
static int    g_pool_sims;      /* sims the node pool is sized for           */
static uint64_t g_mcts_rng[4] = { 0x9E3779B97F4A7C15ull, 0xBF58476D1CE4E5B9ull,
                                  0x94D049BB133111EBull, 0x2545F4914F6CDD1Dull };
static Game   g_game;           /* position set by the last `position` cmd   */

/* options */
static char g_model_path[UCI_PATH_MAX] = UCI_DEF_MODEL;
static int  g_opt_agent    = -1;        /* -1 = champion by Elo              */
static int  g_hash_mb      = UCI_DEF_HASH;
static int  g_blunder_pct  = 0;
static int  g_engine       = UCI_ENGINE_MCTS;   /* THE DEFAULT IS MCTS       */
static int  g_sims         = UCI_DEF_SIMS;
static int  g_overhead_ms  = UCI_DEF_OVERHEAD;

/* ------------------------------------------------------------- threading */

typedef struct {
    int64_t  wtime, btime, winc, binc;
    int64_t  movetime;
    int      movestogo;
    int      depth;
    uint64_t nodes;
    int      infinite;
} GoLimits;

static pthread_mutex_t g_mu      = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t  g_cv_go   = PTHREAD_COND_INITIALIZER;
static pthread_cond_t  g_cv_done = PTHREAD_COND_INITIALIZER;

static int      g_job_pending;  /* guarded by g_mu                           */
static int      g_searching;    /* guarded by g_mu                           */
static int      g_worker_quit;  /* guarded by g_mu                           */
static _Atomic int g_stop_req;  /* read lock-free by the searching thread     */

static GoLimits g_limits;
static Game     g_job_game;     /* snapshot the worker searches              */
static int      g_job_infinite; /* the in-flight search cannot end by itself */

static int stopping(void)
{
    return atomic_load_explicit(&g_stop_req, memory_order_relaxed) != 0;
}

/* Search.stop is a plain `int` in search.h, and search.h is not ours to change,
 * so the running search polls it non-atomically.  Write it as a relaxed atomic
 * from this side: a single writer storing a single value that the reader only
 * ever tests against zero, on a naturally aligned word.  ThreadSanitizer still
 * flags the search's own plain load, which is expected -- interrupting a search
 * through that field is exactly what search.h offers. */
static void poke_search_stop(void)
{
    atomic_store_explicit((_Atomic int *)&g_search.stop, 1, memory_order_relaxed);
}

/* Ask the running search to give up.
 *
 * search_best() clears Search.stop on entry, so a single write from this thread
 * could be swallowed by a search that is just starting.  Re-assert it for a few
 * milliseconds -- the loop exits the instant the worker goes idle, so `stop` on
 * an idle engine costs nothing and `stop` during `go infinite` costs at most
 * 10ms of the reader thread. */
static void request_stop(void)
{
    int i;

    pthread_mutex_lock(&g_mu);
    atomic_store_explicit(&g_stop_req, 1, memory_order_relaxed);
    if (g_searching) poke_search_stop();
    pthread_mutex_unlock(&g_mu);

    for (i = 0; i < 40; i++) {
        int busy;
        pthread_mutex_lock(&g_mu);
        busy = g_searching || g_job_pending;
        if (busy) poke_search_stop();
        pthread_mutex_unlock(&g_mu);
        if (!busy) break;
        nap_us(250);
    }
}

/* Block until no search is in flight.  Callers that may be interrupting a
 * search must request_stop() first or this can wait forever on `go infinite`. */
static void wait_idle(void)
{
    pthread_mutex_lock(&g_mu);
    while (g_job_pending || g_searching)
        pthread_cond_wait(&g_cv_done, &g_mu);
    pthread_mutex_unlock(&g_mu);
}

/* Same, but gives up after `ms`.  Returns 1 if the engine is idle. */
static int wait_idle_ms(int ms)
{
    struct timespec deadline;
    int idle;

    clock_gettime(CLOCK_REALTIME, &deadline);
    deadline.tv_sec  += ms / 1000;
    deadline.tv_nsec += (long)(ms % 1000) * 1000000L;
    if (deadline.tv_nsec >= 1000000000L) { deadline.tv_sec++; deadline.tv_nsec -= 1000000000L; }

    pthread_mutex_lock(&g_mu);
    while (g_job_pending || g_searching)
        if (pthread_cond_timedwait(&g_cv_done, &g_mu, &deadline) != 0) break;
    idle = !(g_job_pending || g_searching);
    pthread_mutex_unlock(&g_mu);
    return idle;
}

/* Bring the engine to rest before a command that needs a stable position.
 *
 * A search that would end by itself is allowed to finish, so a piped script
 * ("go depth 4" immediately followed by "quit") still gets its complete reply;
 * only a search that cannot end on its own -- `go infinite`, `go ponder` -- is
 * cut short.  The grace period keeps even a pathological `go depth 60` from
 * wedging the reader for ever. */
static void settle(void)
{
    int inf;

    pthread_mutex_lock(&g_mu);
    inf = g_job_infinite && (g_job_pending || g_searching);
    pthread_mutex_unlock(&g_mu);

    if (inf) request_stop();
    if (!wait_idle_ms(UCI_SETTLE_MS)) { request_stop(); wait_idle(); }
}

/* ---------------------------------------------------------------- model */

static int best_by_elo(const float *elo, int n)
{
    int best = 0, i;
    for (i = 1; i < n; i++)
        if (elo[i] > elo[best]) best = i;
    return best;
}

static void resolve_agent(void)
{
    int a;
    if (!g_have_model) {
        g_agent        = 0;
        g_search.trunk = NULL;
        g_search.head  = NULL;
        return;
    }
    a = g_opt_agent;
    if (a < 0 || a >= g_nagents) a = best_by_elo(g_elo, g_nagents);
    g_agent        = a;
    g_search.trunk = g_trunk;
    g_search.head  = &g_heads[a];
}

static void free_model(void)
{
    free(g_trunk);  g_trunk  = NULL;
    free(g_heads);  g_heads  = NULL;
    free(g_hy);     g_hy     = NULL;
    free(g_elo);    g_elo    = NULL;
    g_nagents = 0;
    g_generation = 0;
    g_have_model = 0;
}

/* Loads a model and swaps it in.  On failure the previous model is untouched. */
static int load_model(const char *path)
{
    Trunk *tr = NULL;
    Head  *hd = NULL;
    Hyper *hy = NULL;
    float *el = NULL;
    int    n = 0, gen = 0, cap;

    if (!path || !*path) return 0;
    if (!model_load(path, NULL, NULL, NULL, NULL, &n, &gen) || n <= 0) return 0;

    tr = calloc(1, sizeof(Trunk));
    hd = calloc((size_t)n, sizeof(Head));
    hy = calloc((size_t)n, sizeof(Hyper));
    el = calloc((size_t)n, sizeof(float));
    if (!tr || !hd || !hy || !el) { free(tr); free(hd); free(hy); free(el); return 0; }

    cap = n;
    if (!model_load(path, tr, hd, hy, el, &cap, &gen) || cap <= 0) {
        free(tr); free(hd); free(hy); free(el);
        return 0;
    }

    free_model();
    g_trunk = tr; g_heads = hd; g_hy = hy; g_elo = el;
    g_nagents = cap; g_generation = gen; g_have_model = 1;
    resolve_agent();
    return 1;
}

/* Resize / clear the transposition table.  search_init() memsets the whole
 * Search, so every setting has to be reapplied afterwards. */
static void reset_search(size_t hash_mb)
{
    search_free(&g_search);
    search_init(&g_search, NULL, NULL, hash_mb);
    g_search.blunder_rate = (float)g_blunder_pct / 100.0f;
    resolve_agent();
}

static void tt_clear(void)
{
    if (g_search.tt && g_search.tt_size)
        memset(g_search.tt, 0, g_search.tt_size * sizeof(TTEntry));
}

/* ---------------------------------------------------------- policy head */

/* Ranks the legal moves of `p` by raw policy logit, best first.  `order`
 * receives indices into `list`; ties keep move-generation order.  Returns 0
 * when there is no model (the caller then falls back to the first legal move). */
static int policy_rank(const Position *p, const Move *list, int n, int *order,
                       float *value_out)
{
    uint16_t fidx[NF_MAXACTIVE];
    MoveKey  keys[MAX_MOVES];
    float    logits[MAX_MOVES];
    Fwd      fw;
    int      nf, i, j;

    if (n <= 0) return 0;
    if (!g_search.trunk || !g_search.head) return 0;

    nf = nn_features(p, fidx);
    nn_eval(g_search.trunk, g_search.head, fidx, nf, &fw);
    if (value_out) *value_out = fw.v;

    for (i = 0; i < n; i++) nn_move_key(p, list[i], &keys[i]);
    nn_logits(g_search.head, &fw, keys, n, logits);

    /* insertion sort, stable: equal logits keep the generator's order */
    for (i = 0; i < n; i++) order[i] = i;
    for (i = 1; i < n; i++) {
        const int   idx = order[i];
        const float v   = logits[idx];
        j = i - 1;
        while (j >= 0 && logits[order[j]] < v) { order[j + 1] = order[j]; j--; }
        order[j + 1] = idx;
    }
    return n;
}

/* The move the network alone would play.  With BlunderRate > 0 it picks
 * uniformly from the better-scoring half of the move list instead -- exactly
 * the rule search.c uses, so the option means the same thing in both modes. */
static Move policy_move(const Position *p, const Move *list, int n)
{
    int order[MAX_MOVES];
    int pick = 0;

    if (n <= 0) return MV_NONE;
    if (policy_rank(p, list, n, order, NULL) != n) return list[0];

    if (g_blunder_pct > 0 && n > 1 &&
        rndf() < (float)g_blunder_pct / 100.0f) {
        int half = n / 2;
        if (half < 1) half = 1;
        pick = (int)(rnd64() % (uint64_t)half);
    }
    return list[order[pick]];
}

/* ==========================================================================
 * the MCTS agent
 * ========================================================================== */

/* A value in [-1,1] rendered as the integer a GUI expects in `score cp`.
 *
 *      cp = 300 * atanh(clamp(v, -0.995, +0.995))
 *
 * atanh undoes the value head's tanh, so the map is strictly monotone: it
 * changes the units and reorders nothing.  300 is arbitrary presentation and is
 * NOT calibrated against a pawn -- this engine has never been told what a pawn
 * is worth, and `score cp` is the only place the word "pawn" survives at all.
 * MCTS has no mate distance to report, so mcts mode never emits `score mate`;
 * a forced mate shows up as the value saturating near the +-899 clamp. */
static int value_to_cp(double v)
{
    double cp;
    if (!(v == v)) return 0;                      /* NaN */
    if (v >  0.995) v =  0.995;
    if (v < -0.995) v = -0.995;
    cp = 300.0 * atanh(v);
    if (cp >  20000.0) cp =  20000.0;
    if (cp < -20000.0) cp = -20000.0;
    return (int)(cp < 0 ? cp - 0.5 : cp + 0.5);
}

/* Size the node pool for `sims` simulations.  Never shrinks; never called from
 * inside a search, so the hot path stays allocation-free. */
static void mcts_pool_for(int sims)
{
    long want, nodes;

    if (sims <= g_pool_sims && g_mcts.pool) return;
    want  = (long)sims + (sims >> 3) + 8;
    nodes = want * UCI_POOL_PER_SIM + 1;
    if (nodes > UCI_POOL_MAX_NODES) nodes = UCI_POOL_MAX_NODES;

    mcts_free(&g_mcts);
    mcts_init(&g_mcts, (int)nodes);
    g_pool_sims = g_mcts.pool ? (int)want : 0;
}

/* The principal variation of a finished MCTS tree: the most-visited child at
 * every step.  That is what "the search believes" means for a visit-count
 * agent, exactly as the root move is the most-visited root child.  pv_str()
 * re-validates every move against the position before printing it. */
static int mcts_pv(Move *pv, int max)
{
    int n = 0, idx = 0;

    if (!g_mcts.pool || g_mcts.used <= 0) return 0;
    while (n < max) {
        const MctsNode *nd = &g_mcts.pool[idx];
        int32_t bn = -1;
        int     b  = -1, i;

        if (nd->first < 0 || nd->nchild <= 0) break;
        for (i = 0; i < nd->nchild; i++) {
            const MctsNode *c = &g_mcts.pool[nd->first + i];
            if (c->N > bn) { bn = c->N; b = i; }
        }
        if (b < 0 || bn <= 0) break;
        idx = nd->first + b;
        pv[n++] = g_mcts.pool[idx].move;
    }
    return n;
}

/* Rank the root's legal moves by visit count, best first (stable on ties).
 * Used for the chosen move and for BlunderRate's "better-scoring half" rule,
 * which is then the same rule in all three engine modes. */
static void visit_rank(const int32_t *visits, int n, int *order)
{
    int i, j;
    for (i = 0; i < n; i++) order[i] = i;
    for (i = 1; i < n; i++) {
        const int     idx = order[i];
        const int32_t v   = visits[idx];
        j = i - 1;
        while (j >= 0 && visits[order[j]] < v) { order[j + 1] = order[j]; j--; }
        order[j + 1] = idx;
    }
}

/* ------------------------------------------------------------- formatting */

static void score_str(int cp, char *buf, size_t buflen)
{
    if (cp >= UCI_MATE_BOUND)
        snprintf(buf, buflen, "mate %d", (UCI_MATE - cp + 1) / 2);
    else if (cp <= -UCI_MATE_BOUND)
        snprintf(buf, buflen, "mate %d", -((UCI_MATE + cp + 1) / 2));
    else
        snprintf(buf, buflen, "cp %d", cp);
}

/* Renders the principal variation, stopping at the first move that is not legal
 * in the position it would be played from.  A PV table can go stale across
 * iterations; a GUI must never be handed an unplayable line. */
static int pv_str(const Position *root, const Move *pv, int n, char *buf, size_t buflen)
{
    Position p = *root;
    size_t   off = 0;
    int      i, k = 0;

    if (buflen == 0) return 0;
    buf[0] = '\0';

    for (i = 0; i < n; i++) {
        Move ml[MAX_MOVES];
        Undo u;
        char one[8];
        int  nm, j, ok = 0, w;

        nm = gen_legal(&p, ml);
        for (j = 0; j < nm; j++) if (ml[j] == pv[i]) { ok = 1; break; }
        if (!ok) break;

        move_to_uci(pv[i], one);
        w = snprintf(buf + off, buflen - off, "%s%s", k ? " " : "", one);
        if (w < 0 || (size_t)w >= buflen - off) break;
        off += (size_t)w;
        k++;
        make_move(&p, pv[i], &u);
    }
    return k;
}

static void emit_info(int depth, int score_cp, uint64_t nodes, double elapsed_ms,
                      const char *pv)
{
    char     sc[32];
    int64_t  ms  = (int64_t)(elapsed_ms + 0.5);
    uint64_t nps = nodes * 1000ULL / (uint64_t)(ms > 0 ? ms : 1);

    score_str(score_cp, sc, sizeof sc);
    if (pv && *pv)
        say("info depth %d seldepth %d score %s nodes %llu nps %llu time %lld pv %s",
            depth, depth, sc, (unsigned long long)nodes,
            (unsigned long long)nps, (long long)ms, pv);
    else
        say("info depth %d seldepth %d score %s nodes %llu nps %llu time %lld",
            depth, depth, sc, (unsigned long long)nodes,
            (unsigned long long)nps, (long long)ms);
}

/* ------------------------------------------------------------ time manager */

/* Returns the millisecond budget for this move, or 0 for "no time limit".
 *
 * remaining/30 + 80% of the increment is the usual sane default; it is then
 * capped at 40% of what is left and at (remaining - MoveOverhead), so the
 * engine can never flag itself no matter how absurd the clock is. */
static int64_t compute_budget_ms(const GoLimits *L, int side)
{
    int64_t remain, inc, budget, cap;

    if (L->movetime > 0) {
        budget = L->movetime - g_overhead_ms;
        if (budget < 1) budget = (L->movetime > 1) ? L->movetime / 2 : 1;
        return budget;
    }
    if (L->infinite) return 0;

    remain = (side == WHITE) ? L->wtime : L->btime;
    inc    = (side == WHITE) ? L->winc  : L->binc;
    if (inc < 0) inc = 0;

    if (remain <= 0)                                  /* no clock was given */
        return (L->depth > 0 || L->nodes > 0) ? 0 : UCI_BARE_GO_MS;

    budget = (L->movestogo > 0)
           ? remain / (L->movestogo + 2) + (inc * 4) / 5
           : remain / 30 + (inc * 4) / 5;

    if (budget > remain * 2 / 5) budget = remain * 2 / 5;
    cap = remain - g_overhead_ms;
    if (cap < 1) cap = 1;
    if (budget > cap) budget = cap;
    if (budget < 1) budget = 1;
    return budget;
}

/* ------------------------------------------------------------------ search */

/* ==========================================================================
 * running one `go`
 * ==========================================================================
 * Each of the three movers below emits its own `info` lines and returns the
 * move to play.  run_go() owns the single `bestmove` that a `go` must always
 * produce, so none of them can forget it or emit two.
 */

/* ---- mcts: THE SHIPPED AGENT --------------------------------------------
 *
 * Root noise OFF and temperature 0 (argmax of visit counts): this is real play,
 * not self-play exploration.  `nodes` is the simulation count and `depth` is
 * the deepest point the tree reached, both of which are what those words mean
 * for a visit-count agent.
 *
 * The simulation budget ramps 32, 64, 128, ... so that `movetime`, `wtime` and
 * `stop` are all honoured: mcts_search() runs to completion once called, so the
 * only place to check the clock is between passes.  A fresh tree per pass makes
 * the ramp cost at most 2x the final pass, and the final pass is the one whose
 * tree answers.  `stop` latency is therefore one pass -- 4 ms at the default
 * 800 simulations.                                                          */
static Move run_mcts(Game *g, const GoLimits *L, const Move *list, int nlegal,
                     Move fallback, double t0, uint64_t *nodes_out)
{
    int32_t visits[MAX_MOVES];
    int     order[MAX_MOVES];
    Move    pv[UCI_PV_MAX];
    char    pvbuf[UCI_PV_MAX * 6 + 8];
    int64_t budget = compute_budget_ms(L, g->pos.side);
    long    target;
    int     run, done = 0, nroot = 0, emitted = 0;
    float   rootv = 0.0f;
    Move    best = fallback;

    *nodes_out = 0;
    pvbuf[0] = '\0';

    if (!g_search.trunk || !g_search.head) {          /* no model loaded */
        say("info string no model: playing the first legal move");
        return fallback;
    }

    /* `Sims`, overridden exactly by `go nodes`, capped by `go depth`. */
    target = g_sims;
    if (L->nodes > 0)  target = (long)L->nodes;
    if (L->depth > 0) {
        const long cap = (long)L->depth * UCI_SIMS_PER_DEPTH;
        if (cap < target) target = cap;
    }
    if (L->infinite)   target = UCI_INF_SIMS;   /* bounds `stop` latency */
    if (target < 1)          target = 1;
    if (target > UCI_MAX_SIMS) target = UCI_MAX_SIMS;

    mcts_pool_for((int)target);
    if (!g_mcts.pool) {
        say("info string cannot allocate an MCTS node pool");
        return fallback;
    }

    run = (target < UCI_SIMS_FIRST) ? (int)target : UCI_SIMS_FIRST;
    for (;;) {
        int pvlen;

        g_mcts.evals          = 0;
        g_mcts.max_depth_seen = 0;

        nroot = mcts_search(&g_mcts, g_search.trunk, g_search.head, g,
                            run, 0 /* no root noise */, g_mcts_rng, visits, &rootv);
        if (nroot <= 0) break;                        /* already over */
        done = run;

        visit_rank(visits, nroot, order);
        if (order[0] >= 0 && order[0] < nlegal) best = list[order[0]];

        pvlen = mcts_pv(pv, UCI_PV_MAX);
        pv_str(&g->pos, pv, pvlen, pvbuf, sizeof pvbuf);
        emit_info(g_mcts.max_depth_seen > 0 ? g_mcts.max_depth_seen : 1,
                  value_to_cp((double)rootv), (uint64_t)done, now_ms() - t0, pvbuf);
        emitted = 1;

        if (stopping()) break;
        if (run >= target) {
            if (!L->infinite) break;
            /* `go infinite` must not answer until `stop`.  The tree is as big
             * as the pool allows, so idle rather than burn a core re-searching. */
            while (!stopping()) nap_us(2000);
            break;
        }
        /* The next pass is about twice this one; do not start one that cannot
         * finish inside the budget. */
        if (budget > 0 && (now_ms() - t0) * 3.0 >= (double)budget) break;
        run = (run * 2 > (int)target) ? (int)target : run * 2;
    }

    /* BlunderRate: play a uniformly random move from the better-visited half of
     * the root moves.  Identical in shape to the rule search.c and policy_move()
     * use, so the option means the same thing in all three modes. */
    if (nroot > 1 && g_blunder_pct > 0 &&
        rndf() < (float)g_blunder_pct / 100.0f) {
        int half = nroot / 2;
        if (half < 1) half = 1;
        best = list[order[(int)(rnd64() % (uint64_t)half)]];
    }

    if (!emitted) {
        char one[8];
        move_to_uci(best, one);
        emit_info(1, value_to_cp((double)rootv), (uint64_t)done, now_ms() - t0, one);
    }
    *nodes_out = (uint64_t)done;
    return best;
}

/* ---- policy: MEASUREMENT ONLY -------------------------------------------
 * The raw policy argmax.  One network evaluation, no search, no nodes.  This is
 * the number that says how much the network itself knows. */
static Move run_policy(Game *g, const Move *list, int nlegal, Move fallback,
                       double t0, uint64_t *nodes_out)
{
    int   order[MAX_MOVES];
    float v = 0.0f;
    char  one[8];

    *nodes_out = 1;
    if (policy_rank(&g->pos, list, nlegal, order, &v) != nlegal)
        return fallback;

    move_to_uci(fallback, one);
    emit_info(1, value_to_cp((double)v), 1, now_ms() - t0, one);
    return fallback;        /* policy_move() already applied BlunderRate */
}

/* ---- alphabeta: MEASUREMENT ONLY ----------------------------------------
 * The src/search.c baseline: a second, structurally different searcher over the
 * same network, kept so the MCTS agent has something to be measured against.
 * search_best() exposes no per-iteration callback, so the `info depth` stream
 * comes from calling it once per target depth; the transposition table survives
 * across the calls, so the shallow re-searches are nearly free. */
static Move run_alphabeta(Game *g, const GoLimits *L, const Move *list, int nlegal,
                          Move fallback, double t0, uint64_t *nodes_out)
{
    Move     pv[UCI_PV_MAX];
    char     pvbuf[UCI_PV_MAX * 6 + 8];
    Move     best = fallback;
    uint64_t total_nodes = 0;
    int64_t  budget = compute_budget_ms(L, g->pos.side);
    int      maxd = (L->depth > 0) ? L->depth : UCI_MAX_DEPTH;
    int      d, best_score = 0, emitted = 0;

    (void)nlegal;
    pvbuf[0] = '\0';

    for (d = 1; d <= maxd; d++) {
        Move   m;
        double elapsed = now_ms() - t0;

        if (stopping()) break;

        if (budget > 0) {
            int64_t remain = budget - (int64_t)elapsed;
            if (remain <= 0) break;
            /* Past the halfway mark another full iteration will not finish. */
            if (d > 1 && (int64_t)elapsed * 2 >= budget) break;
            if (remain > UCI_MAX_SLICE_MS) remain = UCI_MAX_SLICE_MS;
            g_search.movetime_ms = (int)remain;
        } else {
            g_search.movetime_ms = 0;
        }

        /* `go nodes` counts every node this command searches, re-searched
         * shallow depths included, so the limit is a true ceiling on work
         * done rather than on the deepest pass alone. */
        if (L->nodes) {
            if (total_nodes >= L->nodes) break;
            g_search.max_nodes = L->nodes - total_nodes;
        } else {
            g_search.max_nodes = 0;
        }
        g_search.max_depth = d;

        m = search_best(&g_search, g);
        total_nodes += g_search.nodes;
        if (m != MV_NONE) best = m;

        /* Search.stop set => the last iteration was abandoned, and every depth
         * it did finish was already reported by the previous outer pass. */
        if (!g_search.stop && g_search.depth_reached > emitted) {
            int pvlen = g_search.pv_len, i;
            if (pvlen > UCI_PV_MAX) pvlen = UCI_PV_MAX;
            if (pvlen < 0) pvlen = 0;
            for (i = 0; i < pvlen; i++) pv[i] = g_search.pv[i];
            best_score = g_search.score_cp;
            emitted    = g_search.depth_reached;
            pv_str(&g->pos, pv, pvlen, pvbuf, sizeof pvbuf);
            emit_info(emitted, best_score, total_nodes, now_ms() - t0, pvbuf);
        }

        if (g_search.stop) break;                       /* time / nodes / stop */
        if (g_search.depth_reached < d) break;          /* search cut itself off */
        if (best_score >= UCI_MATE_BOUND || best_score <= -UCI_MATE_BOUND) break;
    }

    if (!emitted) {                                     /* stopped before depth 1 */
        char one[8];
        move_to_uci(best, one);
        emit_info(1, search_eval_cp(&g_search, &g->pos), total_nodes,
                  now_ms() - t0, one);
    }
    *nodes_out = total_nodes;
    return best;
}

/* Runs one `go` to completion and emits exactly one bestmove.  Always. */
static void run_go(void)
{
    Game     *g = &g_job_game;
    GoLimits  L = g_limits;
    Move      list[MAX_MOVES];
    Move      best = MV_NONE;
    char      bestuci[8];
    double    t0 = now_ms();
    uint64_t  nodes = 0;
    int       nlegal;

    nlegal = gen_legal(&g->pos, list);
    if (nlegal <= 0) {                      /* checkmate, stalemate, no move */
        say("bestmove 0000");
        return;
    }

    /* search_best() and mcts_search() both refuse to run once Game.result is
     * set, which includes draws by repetition / fifty-move.  A UCI engine must
     * still answer while legal moves exist: adjudication is the GUI's job. */
    g->result = GR_ONGOING;
    g->reason = TR_NONE;

    /* A legal move in hand before a single node is searched, so an immediate
     * `stop` still produces a real answer. */
    best = policy_move(&g->pos, list, nlegal);
    if (best == MV_NONE) best = list[0];

    switch (g_engine) {
        case UCI_ENGINE_POLICY:
            best = run_policy(g, list, nlegal, best, t0, &nodes);
            break;
        case UCI_ENGINE_ALPHABETA:
            best = run_alphabeta(g, &L, list, nlegal, best, t0, &nodes);
            break;
        default:
            best = run_mcts(g, &L, list, nlegal, best, t0, &nodes);
            break;
    }

    move_to_uci(best, bestuci);
    say("bestmove %s", bestuci);
}

static void *worker_main(void *arg)
{
    (void)arg;
    for (;;) {
        pthread_mutex_lock(&g_mu);
        while (!g_job_pending && !g_worker_quit)
            pthread_cond_wait(&g_cv_go, &g_mu);
        if (!g_job_pending && g_worker_quit) { pthread_mutex_unlock(&g_mu); break; }
        g_job_pending = 0;
        g_searching   = 1;
        pthread_mutex_unlock(&g_mu);

        run_go();

        pthread_mutex_lock(&g_mu);
        g_searching = 0;
        pthread_cond_broadcast(&g_cv_done);
        pthread_mutex_unlock(&g_mu);
    }
    return NULL;
}

/* ------------------------------------------------------------- tokenising */

static int split(char *line, char **tok, int maxtok)
{
    int   n = 0;
    char *save = NULL;
    char *t = strtok_r(line, " \t\r\n", &save);
    while (t && n < maxtok) { tok[n++] = t; t = strtok_r(NULL, " \t\r\n", &save); }
    return n;
}

static int64_t tok_i64(char **tok, int n, int i)
{
    return (i < n) ? (int64_t)strtoll(tok[i], NULL, 10) : 0;
}

/* ---------------------------------------------------------------- position */

static void cmd_position(char **tok, int n)
{
    char fen[256];
    int  i = 1, first_move = -1;

    if (n < 2) return;

    if (!strcasecmp(tok[1], "startpos")) {
        game_start(&g_game);
        i = 2;
    } else if (!strcasecmp(tok[1], "fen")) {
        size_t off = 0;
        fen[0] = '\0';
        for (i = 2; i < n && strcasecmp(tok[i], "moves") != 0; i++) {
            int w = snprintf(fen + off, sizeof(fen) - off, "%s%s", off ? " " : "", tok[i]);
            if (w < 0 || (size_t)w >= sizeof(fen) - off) break;
            off += (size_t)w;
        }
        if (!game_start_fen(&g_game, fen)) {
            say("info string bad fen, position unchanged");
            return;
        }
    } else {
        return;
    }

    for (; i < n; i++) if (!strcasecmp(tok[i], "moves")) { first_move = i + 1; break; }
    if (first_move < 0) return;

    for (i = first_move; i < n; i++) {
        Move m;
        if (!move_from_uci(&g_game.pos, tok[i], &m)) {
            say("info string illegal move '%s', rest of line ignored", tok[i]);
            return;
        }
        game_push(&g_game, m);
    }
}

/* ---------------------------------------------------------------- setoption */

static int truthy(const char *s)
{
    return !strcasecmp(s, "true") || !strcasecmp(s, "on") ||
           !strcasecmp(s, "yes")  || !strcmp(s, "1");
}

static int clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

static const char *engine_name(int e)
{
    switch (e) {
        case UCI_ENGINE_POLICY:    return "policy";
        case UCI_ENGINE_ALPHABETA: return "alphabeta";
        default:                   return "mcts";
    }
}

static void cmd_setoption(char **tok, int n)
{
    char name[128], value[UCI_PATH_MAX];
    size_t off;
    int i, vstart = -1, nend = n;

    name[0] = value[0] = '\0';
    if (n < 3 || strcasecmp(tok[1], "name") != 0) return;

    for (i = 2; i < n; i++)
        if (!strcasecmp(tok[i], "value")) { nend = i; vstart = i + 1; break; }

    off = 0;
    for (i = 2; i < nend; i++) {
        int w = snprintf(name + off, sizeof(name) - off, "%s%s", off ? " " : "", tok[i]);
        if (w < 0 || (size_t)w >= sizeof(name) - off) break;
        off += (size_t)w;
    }
    off = 0;
    if (vstart >= 0)
        for (i = vstart; i < n; i++) {
            int w = snprintf(value + off, sizeof(value) - off, "%s%s", off ? " " : "", tok[i]);
            if (w < 0 || (size_t)w >= sizeof(value) - off) break;
            off += (size_t)w;
        }

    if (!strcasecmp(name, "Model")) {
        if (!value[0]) return;
        if (load_model(value)) {
            snprintf(g_model_path, sizeof g_model_path, "%s", value);
            say("info string model %s: %d agents, generation %d, playing agent %d",
                g_model_path, g_nagents, g_generation, g_agent);
        } else {
            say("info string cannot load model '%s'", value);
        }
    } else if (!strcasecmp(name, "Agent")) {
        g_opt_agent = clampi((int)strtol(value, NULL, 10), -1, UCI_MAX_AGENT);
        resolve_agent();
        say("info string playing agent %d of %d", g_agent, g_nagents);
    } else if (!strcasecmp(name, "Hash")) {
        g_hash_mb = clampi((int)strtol(value, NULL, 10), 1, UCI_MAX_HASH);
        reset_search((size_t)g_hash_mb);
    } else if (!strcasecmp(name, "BlunderRate")) {
        g_blunder_pct = clampi((int)strtol(value, NULL, 10), 0, 100);
        g_search.blunder_rate = (float)g_blunder_pct / 100.0f;
    } else if (!strcasecmp(name, "Engine")) {
        if      (!strcasecmp(value, "mcts"))      g_engine = UCI_ENGINE_MCTS;
        else if (!strcasecmp(value, "policy"))    g_engine = UCI_ENGINE_POLICY;
        else if (!strcasecmp(value, "alphabeta")) g_engine = UCI_ENGINE_ALPHABETA;
        else {
            say("info string unknown Engine '%s' (mcts|policy|alphabeta), keeping %s",
                value, engine_name(g_engine));
            return;
        }
        say("info string engine %s%s", engine_name(g_engine),
            g_engine == UCI_ENGINE_MCTS ? "" : "  (MEASUREMENT ONLY -- not the shipped agent)");
    } else if (!strcasecmp(name, "Sims")) {
        g_sims = clampi((int)strtol(value, NULL, 10), 1, UCI_MAX_SIMS);
    } else if (!strcasecmp(name, "UseSearch")) {
        /* Deprecated alias, kept so existing harness scripts keep working:
         * false selects the raw policy, true restores the default agent. */
        g_engine = (value[0] && !truthy(value)) ? UCI_ENGINE_POLICY : UCI_ENGINE_MCTS;
    } else if (!strcasecmp(name, "MoveOverhead")) {
        g_overhead_ms = clampi((int)strtol(value, NULL, 10), 0, UCI_MAX_OVERHEAD);
    }
    /* unknown options are ignored, as the protocol requires */
}

/* ---------------------------------------------------------------------- go */

static void cmd_go(char **tok, int n, int have_worker)
{
    GoLimits L;
    int i;

    memset(&L, 0, sizeof L);

    for (i = 1; i < n; i++) {
        if      (!strcasecmp(tok[i], "wtime"))     L.wtime     = tok_i64(tok, n, ++i);
        else if (!strcasecmp(tok[i], "btime"))     L.btime     = tok_i64(tok, n, ++i);
        else if (!strcasecmp(tok[i], "winc"))      L.winc      = tok_i64(tok, n, ++i);
        else if (!strcasecmp(tok[i], "binc"))      L.binc      = tok_i64(tok, n, ++i);
        else if (!strcasecmp(tok[i], "movestogo")) L.movestogo = (int)tok_i64(tok, n, ++i);
        else if (!strcasecmp(tok[i], "movetime"))  L.movetime  = tok_i64(tok, n, ++i);
        else if (!strcasecmp(tok[i], "depth"))     L.depth     = (int)tok_i64(tok, n, ++i);
        else if (!strcasecmp(tok[i], "nodes")) {
            int64_t v = tok_i64(tok, n, ++i);
            L.nodes = (v > 0) ? (uint64_t)v : 0;
        }
        else if (!strcasecmp(tok[i], "infinite"))  L.infinite = 1;
        else if (!strcasecmp(tok[i], "ponder"))    L.infinite = 1;
        else if (!strcasecmp(tok[i], "mate"))      i++;         /* not supported */
        else if (!strcasecmp(tok[i], "searchmoves")) break;     /* not supported */
        /* anything else is ignored */
    }
    if (L.depth < 0) L.depth = 0;

    /* A `go` while already searching is a protocol violation; honour it by
     * letting the previous search settle first, so every `go` still gets
     * exactly one bestmove. */
    settle();

    pthread_mutex_lock(&g_mu);
    g_limits       = L;
    g_job_game     = g_game;
    g_job_infinite = L.infinite;
    atomic_store_explicit(&g_stop_req, 0, memory_order_relaxed);
    g_search.stop  = 0;
    g_job_pending  = 1;
    pthread_cond_signal(&g_cv_go);
    pthread_mutex_unlock(&g_mu);

    if (!have_worker) {                 /* no threads: search inline */
        pthread_mutex_lock(&g_mu);
        g_job_pending = 0;
        g_searching   = 1;
        pthread_mutex_unlock(&g_mu);
        run_go();
        pthread_mutex_lock(&g_mu);
        g_searching = 0;
        pthread_cond_broadcast(&g_cv_done);
        pthread_mutex_unlock(&g_mu);
    }
}

/* --------------------------------------------------------------------- uci */

static void cmd_uci(void)
{
    if (g_have_model)
        say("id name ChessRL gen%d agent%d (Elo %.0f, %d agents)",
            g_generation, g_agent, (double)g_elo[g_agent], g_nagents);
    else
        say("id name ChessRL (no model)");
    say("id author ChessRL");

    say("option name Model type string default %s",
        g_model_path[0] ? g_model_path : "<empty>");
    say("option name Agent type spin default %d min -1 max %d", g_opt_agent, UCI_MAX_AGENT);
    /* The shipped agent is `mcts`; the other two are measurement baselines. */
    say("option name Engine type combo default %s var mcts var policy var alphabeta",
        engine_name(g_engine));
    say("option name Sims type spin default %d min 1 max %d", g_sims, UCI_MAX_SIMS);
    say("option name Hash type spin default %d min 1 max %d", g_hash_mb, UCI_MAX_HASH);
    say("option name BlunderRate type spin default %d min 0 max 100", g_blunder_pct);
    say("option name MoveOverhead type spin default %d min 0 max %d",
        g_overhead_ms, UCI_MAX_OVERHEAD);
    /* UseSearch is still accepted (see cmd_setoption) but no longer advertised:
     * Engine supersedes it and a GUI should not be offered both. */
    say("uciok");
}

/* -------------------------------------------------------------------- main */

int uci_main(int argc, char **argv)
{
    static char line[UCI_LINE_MAX];
    char  *tok[UCI_MAX_TOKENS];
    pthread_t worker;
    int    have_worker = 0, said_startup = 0, i;

    setvbuf(stdout, NULL, _IOLBF, 0);

    g_rng ^= (uint64_t)time(NULL) * 6364136223846793005ULL;
    g_rng ^= (uint64_t)(uintptr_t)&worker;
    if (!g_rng) g_rng = 88172645463325252ULL;

    /* Command-line overrides, so a cutechess-cli engine entry can be a plain
     * `chessrl uci --model M --agent I --no-search` with no init strings. */
    for (i = 0; i < argc; i++) {
        const char *v = (i + 1 < argc) ? argv[i + 1] : NULL;
        if      (!strcmp(argv[i], "--model")   && v) snprintf(g_model_path, sizeof g_model_path, "%s", v);
        else if (!strcmp(argv[i], "--agent")   && v) g_opt_agent   = clampi((int)strtol(v, NULL, 10), -1, UCI_MAX_AGENT);
        else if (!strcmp(argv[i], "--hash")    && v) g_hash_mb     = clampi((int)strtol(v, NULL, 10), 1, UCI_MAX_HASH);
        else if (!strcmp(argv[i], "--blunder") && v) g_blunder_pct = clampi((int)strtol(v, NULL, 10), 0, 100);
        else if (!strcmp(argv[i], "--overhead")&& v) g_overhead_ms = clampi((int)strtol(v, NULL, 10), 0, UCI_MAX_OVERHEAD);
        else if (!strcmp(argv[i], "--sims")    && v) g_sims        = clampi((int)strtol(v, NULL, 10), 1, UCI_MAX_SIMS);
        else if (!strcmp(argv[i], "--engine")  && v) {
            if      (!strcmp(v, "policy"))    g_engine = UCI_ENGINE_POLICY;
            else if (!strcmp(v, "alphabeta")) g_engine = UCI_ENGINE_ALPHABETA;
            else                              g_engine = UCI_ENGINE_MCTS;
        }
        else if (!strcmp(argv[i], "--no-search"))    g_engine      = UCI_ENGINE_POLICY;
    }

    reset_search((size_t)g_hash_mb);
    load_model(g_model_path);           /* silent: nothing may precede `uci` */
    resolve_agent();
    game_start(&g_game);

    /* mcts_search() wants a valid xoshiro state even with root noise off, and
     * a non-deterministic one so that two engines in the same match are not
     * locked in step by any tie-break that reaches for it. */
    g_mcts_rng[0] ^= g_rng;
    g_mcts_rng[3] ^= (uint64_t)(uintptr_t)&worker;
    mcts_pool_for(g_sims);

    have_worker = (pthread_create(&worker, NULL, worker_main, NULL) == 0);

    while (fgets(line, sizeof line, stdin)) {
        int n;
        char *cmd;

        n = split(line, tok, UCI_MAX_TOKENS);
        if (n == 0) continue;
        cmd = tok[0];

        if (!strcmp(cmd, "uci")) {
            cmd_uci();
            if (!said_startup) {
                said_startup = 1;
                say("info string engine %s, %d simulations/move", engine_name(g_engine), g_sims);
                if (g_have_model)
                    say("info string model %s: %d agents, generation %d, playing agent %d",
                        g_model_path, g_nagents, g_generation, g_agent);
                else
                    say("info string no model loaded ('%s'); "
                        "set it with 'setoption name Model value PATH'", g_model_path);
            }
        } else if (!strcmp(cmd, "isready")) {
            say("readyok");                     /* never blocks, even mid-search */
        } else if (!strcmp(cmd, "stop") || !strcmp(cmd, "ponderhit")) {
            /* Pondering is not advertised, so `go ponder` should never arrive;
             * if it does it is treated as `go infinite`, and ponderhit ends it
             * with the best move so far rather than hanging the GUI. */
            request_stop();
        } else if (!strcmp(cmd, "quit")) {
            break;
        } else if (!strcmp(cmd, "ucinewgame")) {
            settle();
            tt_clear();
            game_start(&g_game);
        } else if (!strcmp(cmd, "position")) {
            settle();
            cmd_position(tok, n);
        } else if (!strcmp(cmd, "setoption")) {
            settle();
            cmd_setoption(tok, n);
        } else if (!strcmp(cmd, "go")) {
            cmd_go(tok, n, have_worker);
        }
        /* debug, register, and anything unrecognised: ignored */
    }

    settle();
    pthread_mutex_lock(&g_mu);
    g_worker_quit = 1;
    pthread_cond_broadcast(&g_cv_go);
    pthread_mutex_unlock(&g_mu);
    if (have_worker) pthread_join(worker, NULL);

    search_free(&g_search);
    mcts_free(&g_mcts);
    free_model();
    return 0;
}
