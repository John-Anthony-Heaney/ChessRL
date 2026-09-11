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
 * Two options exist specifically for that calibration:
 *
 *   UseSearch false   plays the RAW POLICY ARGMAX with no search whatsoever.
 *                     This separates what the network learned from what the
 *                     hand-written alpha-beta contributes -- without it the two
 *                     contributions are hopelessly confounded in one number.
 *   BlunderRate       weakens play on purpose.  An agent that loses 100% of its
 *                     games tells you almost nothing; a ladder of deliberately
 *                     weakened opponents brackets it from both sides.
 *
 * Structure
 * ---------
 * The reader loop owns stdin and never blocks on a search: searches run on a
 * worker thread, so `isready` and `stop` are answered while the engine thinks.
 *
 * search_best() runs its own iterative deepening and exposes no per-iteration
 * callback, and search.c is out of bounds for this work, so the `info depth`
 * stream is produced by calling search_best() once per target depth (1, 2, 3,
 * ...).  The transposition table lives in the Search object and survives across
 * those calls, so revisiting the shallow depths is nearly free; each call still
 * respects the *remaining* clock, so the outer loop cannot overrun the budget.
 *
 * Nothing is ever written to stdout that is not valid UCI.  Diagnostics go out
 * as `info string`, which is part of the protocol; genuine errors go to stderr.
 */

#include <ctype.h>
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

static Search g_search;
static Game   g_game;           /* position set by the last `position` cmd   */

/* options */
static char g_model_path[UCI_PATH_MAX] = UCI_DEF_MODEL;
static int  g_opt_agent    = -1;        /* -1 = champion by Elo              */
static int  g_hash_mb      = UCI_DEF_HASH;
static int  g_blunder_pct  = 0;
static int  g_use_search   = 1;
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

/* Runs one `go` to completion and emits exactly one bestmove.  Always. */
static void run_go(void)
{
    Game     *g = &g_job_game;
    GoLimits  L = g_limits;
    Move      list[MAX_MOVES];
    Move      best = MV_NONE;
    Move      pv[UCI_PV_MAX];
    char      pvbuf[UCI_PV_MAX * 6 + 8];
    char      bestuci[8];
    double    t0 = now_ms();
    uint64_t  total_nodes = 0;
    int64_t   budget;
    int       nlegal, maxd, d;
    int       best_score = 0, emitted = 0;

    pvbuf[0] = '\0';

    nlegal = gen_legal(&g->pos, list);
    if (nlegal <= 0) {                      /* checkmate, stalemate, no move */
        say("bestmove 0000");
        return;
    }

    /* search_best() refuses to run once Game.result is set, which includes
     * draws by repetition / fifty-move.  A UCI engine must still answer while
     * legal moves exist: adjudication is the GUI's job, not the engine's. */
    g->result = GR_ONGOING;
    g->reason = TR_NONE;

    /* A legal move in hand before a single node is searched, so an immediate
     * `stop` still produces a real answer. */
    best = policy_move(&g->pos, list, nlegal);
    if (best == MV_NONE) best = list[0];

    if (!g_use_search) {
        /* Raw policy argmax: no search, one network evaluation, no nodes. */
        move_to_uci(best, bestuci);
        emit_info(1, search_eval_cp(&g_search, &g->pos), 1, now_ms() - t0, bestuci);
        say("bestmove %s", bestuci);
        return;
    }

    budget = compute_budget_ms(&L, g->pos.side);
    maxd   = (L.depth > 0) ? L.depth : UCI_MAX_DEPTH;

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
        if (L.nodes) {
            if (total_nodes >= L.nodes) break;
            g_search.max_nodes = L.nodes - total_nodes;
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

    move_to_uci(best, bestuci);
    if (!emitted)                                       /* stopped before depth 1 */
        emit_info(1, search_eval_cp(&g_search, &g->pos), total_nodes,
                  now_ms() - t0, bestuci);
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
    } else if (!strcasecmp(name, "UseSearch")) {
        g_use_search = value[0] ? truthy(value) : 1;
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
    say("option name Hash type spin default %d min 1 max %d", g_hash_mb, UCI_MAX_HASH);
    say("option name BlunderRate type spin default %d min 0 max 100", g_blunder_pct);
    say("option name UseSearch type check default %s", g_use_search ? "true" : "false");
    say("option name MoveOverhead type spin default %d min 0 max %d",
        g_overhead_ms, UCI_MAX_OVERHEAD);
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
        else if (!strcmp(argv[i], "--no-search"))    g_use_search  = 0;
    }

    reset_search((size_t)g_hash_mb);
    load_model(g_model_path);           /* silent: nothing may precede `uci` */
    resolve_agent();
    game_start(&g_game);

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
    free_model();
    return 0;
}
