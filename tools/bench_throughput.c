/* bench_throughput.c -- where does ChessRL's self-play throughput actually go?
 *
 * WHAT THIS MEASURES
 * ------------------
 * A faithful standalone replica of the AlphaZero self-play and learning loop
 * in src/az.c -- the same mcts_search() calls with the same budget and the
 * same playout-cap randomisation, the same recording of (features, visit
 * target, outcome) into a replay buffer, and the same minibatch step over
 * that buffer -- driven from tools/ so that src/ is never touched.
 *
 * It is a replica rather than a wrapper because az.c's worker pool, replay
 * buffer and learning job are all `static`: there is no seam to hook.  Every
 * primitive it calls (mcts_search, gen_legal, nn_eval, nn_backward,
 * adam_step) is the real one, so the per-call costs ARE the trainer's costs;
 * what is re-implemented here is only the loop around them.  `--mode gen`
 * reproduces one generation of `chessrl az` end to end, and its games/sec
 * should be read as the self-play rate that loop sustains.
 *
 *     make bench-throughput                     # build + default run
 *     build/bench_throughput --help             # every knob
 *     build/bench_throughput_probe --mode gen   # + the phase split
 *
 * MODES
 *   selfplay  self-play only: games/sec, moves/sec, evals/sec, evals/move
 *   learn     the replay-buffer minibatch step only, at T threads
 *   gen       one whole generation: self-play round, then the learning phase,
 *             reported as a split.  This is the number Amdahl's law needs.
 *   micro     isolated cost of each primitive: nn_eval, nn_logits, gen_legal,
 *             make/unmake, nn_backward, and a trunk-streaming bandwidth probe
 *   scale     the same self-play workload at 1,2,3,4,6,8 threads, one table
 *
 * The phase split is only printed by build/bench_throughput_probe, which is
 * the same source compiled with -DBENCH_PROBE against a second, instrumented
 * copy of src/mcts.c (see tools/bench_probe.h).  The normal build has no
 * timers in it at all.
 */

#include <errno.h>
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(__APPLE__)
#include <pthread/qos.h>
#include <sys/sysctl.h>
#endif

#include "chess.h"
#include "mcts.h"
#include "net.h"

#ifdef BENCH_PROBE
#include "bench_probe.h"
_Thread_local uint64_t bp_ns[BP_N];
_Thread_local uint64_t bp_cnt[BP_N];
static const char *const BP_NAMES[BP_N] = {
    "nn_eval", "nn_logits", "nn_features", "gen_legal", "make_move",
    "unmake_move", "in_check", "insufficient_material", "nn_move_key",
    "softmax_t", "mcts_search", "driver", "learn", "learn fwd", "nn_backward"
};
const char *bp_name(int b) { return (b >= 0 && b < BP_N) ? BP_NAMES[b] : "?"; }
#define BP_N_OR_1 BP_N
/* The interposition macros in bench_probe.h exist for src/mcts.c.  This file
 * must NOT have them applied to its own calls: if the driver's gen_legal() and
 * the learning step's nn_eval() landed in the same buckets as the search's,
 * then "bookkeeping = mcts_search - network - features - gen_legal" would
 * subtract work that never happened inside the search.  So they are undone
 * here, and this file times its own phases with explicit scopes instead. */
#undef nn_eval
#undef nn_logits
#undef nn_features
#undef gen_legal
#undef make_move
#undef unmake_move
#undef in_check
#undef insufficient_material
#undef nn_move_key
#undef softmax_t
#undef nn_backward
#else
#define BP_N_OR_1 1
/* The normal build: not a branch, not a counter, not a clock read. */
#define BP_T0(tag)     ((void)0)
#define BP_T1(tag, b)  ((void)0)
#endif

/* ========================================================================= */
/*                                  BASICS                                   */
/* ========================================================================= */

#define NODES_PER_SIM   72          /* same pool sizing as src/az.c          */
#define RESIGN_CONSEC   4
#define MAX_THREADS     64
#define REC_MAX_PLIES   512

static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* xoshiro256**, identical to the generator src/az.c drives self-play with. */
static inline uint64_t rotl64(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }

static uint64_t rng_next(uint64_t *s)
{
    const uint64_t r = rotl64(s[1] * 5, 7) * 9;
    const uint64_t t = s[1] << 17;
    s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3];
    s[2] ^= t;    s[3] = rotl64(s[3], 45);
    return r;
}

static void rng_seed(uint64_t *s, uint64_t seed)
{
    uint64_t z = seed ? seed : 0x9E3779B97F4A7C15ull;
    for (int i = 0; i < 4; i++) {
        z += 0x9E3779B97F4A7C15ull;
        uint64_t x = z;
        x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
        x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
        s[i] = x ^ (x >> 31);
    }
    for (int i = 0; i < 16; i++) (void)rng_next(s);
}

static float rng_u01(uint64_t *s) { return (float)((rng_next(s) >> 11) * 0x1.0p-53); }

/* --------------------------------------------------------------- barrier */
/* macOS has no pthread_barrier_t, so the trainer rolls its own; so does this. */

typedef struct {
    pthread_mutex_t mu;
    pthread_cond_t  cv;
    int total, count, gen;
} Bar;

static void bar_init(Bar *b, int total)
{
    pthread_mutex_init(&b->mu, NULL);
    pthread_cond_init(&b->cv, NULL);
    b->total = total; b->count = 0; b->gen = 0;
}
static void bar_destroy(Bar *b)
{
    pthread_mutex_destroy(&b->mu);
    pthread_cond_destroy(&b->cv);
}
static void bar_wait(Bar *b)
{
    pthread_mutex_lock(&b->mu);
    const int g = b->gen;
    if (++b->count == b->total) {
        b->count = 0; b->gen++;
        pthread_cond_broadcast(&b->cv);
    } else {
        while (g == b->gen) pthread_cond_wait(&b->cv, &b->mu);
    }
    pthread_mutex_unlock(&b->mu);
}

/* ------------------------------------------------------------------- QoS */

enum { QOS_OFF = 0, QOS_UI, QOS_UTILITY, QOS_BACKGROUND, QOS_DEFAULT_CLASS };

static const char *qos_name(int q)
{
    switch (q) {
        case QOS_UI:            return "USER_INTERACTIVE (prefers P-cores)";
        case QOS_UTILITY:       return "UTILITY (prefers E-cores)";
        case QOS_BACKGROUND:    return "BACKGROUND (E-cores only)";
        case QOS_DEFAULT_CLASS: return "DEFAULT";
        default:                return "inherited (no QoS set)";
    }
}

static void qos_apply(int q)
{
#if defined(__APPLE__)
    qos_class_t c;
    switch (q) {
        case QOS_UI:            c = QOS_CLASS_USER_INTERACTIVE; break;
        case QOS_UTILITY:       c = QOS_CLASS_UTILITY;          break;
        case QOS_BACKGROUND:    c = QOS_CLASS_BACKGROUND;       break;
        case QOS_DEFAULT_CLASS: c = QOS_CLASS_DEFAULT;          break;
        default: return;
    }
    if (pthread_set_qos_class_self_np(c, 0) != 0)
        fprintf(stderr, "bench: pthread_set_qos_class_self_np failed: %s\n",
                strerror(errno));
#else
    (void)q;   /* portable fallback: QoS is a Darwin concept; nothing to do. */
#endif
}

/* ========================================================================= */
/*                             REPLAY BUFFER                                 */
/* ========================================================================= */
/* Same shape and same commit protocol as src/az.c's AZBuf: a lock-free-ish
 * ring whose only serialisation is a short mutex around two cursor bumps. */

#define POOL_PER_POS  40

typedef struct {
    uint16_t fidx[NF_MAXACTIVE];
    uint64_t koff;
    float    z, q;
    uint16_t nmoves;
    int16_t  agent;
    uint8_t  nf, mover;
} BPos;

typedef struct {
    BPos     *pos;
    MoveKey  *keys;
    float    *pol;
    uint64_t  cap, kcap, head, khead;
    pthread_mutex_t mu;
    int ready;
} BBuf;

static int bbuf_init(BBuf *b, int positions)
{
    memset(b, 0, sizeof *b);
    if (positions < 1024) positions = 1024;
    b->cap  = (uint64_t)positions;
    b->kcap = b->cap * POOL_PER_POS;
    b->pos  = (BPos *)   calloc((size_t)b->cap,  sizeof(BPos));
    b->keys = (MoveKey *)calloc((size_t)b->kcap, sizeof(MoveKey));
    b->pol  = (float *)  calloc((size_t)b->kcap, sizeof(float));
    if (!b->pos || !b->keys || !b->pol) {
        free(b->pos); free(b->keys); free(b->pol);
        memset(b, 0, sizeof *b);
        return 0;
    }
    pthread_mutex_init(&b->mu, NULL);
    b->ready = 1;
    return 1;
}

static void bbuf_free(BBuf *b)
{
    if (!b) return;
    if (b->ready) pthread_mutex_destroy(&b->mu);
    free(b->pos); free(b->keys); free(b->pol);
    memset(b, 0, sizeof *b);
}

static uint64_t bbuf_count(const BBuf *b)
{
    return b->head < b->cap ? b->head : b->cap;
}

/* One game's recorded positions, staged per thread then committed in one go. */
typedef struct {
    BPos     *pos;
    MoveKey  *keys;
    float    *pol;
    int       n, npos_cap;
    uint64_t  nk, nk_cap;
} BRec;

static int brec_init(BRec *r, int max_plies)
{
    memset(r, 0, sizeof *r);
    if (max_plies < 8) max_plies = 8;
    if (max_plies > REC_MAX_PLIES) max_plies = REC_MAX_PLIES;
    r->npos_cap = max_plies;
    r->nk_cap   = (uint64_t)max_plies * MAX_MOVES;
    r->pos  = (BPos *)   calloc((size_t)r->npos_cap, sizeof(BPos));
    r->keys = (MoveKey *)calloc((size_t)r->nk_cap,   sizeof(MoveKey));
    r->pol  = (float *)  calloc((size_t)r->nk_cap,   sizeof(float));
    if (!r->pos || !r->keys || !r->pol) {
        free(r->pos); free(r->keys); free(r->pol);
        memset(r, 0, sizeof *r);
        return 0;
    }
    return 1;
}

static void brec_free(BRec *r)
{
    if (!r) return;
    free(r->pos); free(r->keys); free(r->pol);
    memset(r, 0, sizeof *r);
}

static void bbuf_commit(BBuf *b, const BRec *r)
{
    if (r->n <= 0 || r->nk == 0 || r->nk > b->kcap) return;

    uint64_t k0, p0;
    pthread_mutex_lock(&b->mu);
    const uint64_t off = b->khead % b->kcap;
    if (off + r->nk > b->kcap) b->khead += (b->kcap - off);
    k0 = b->khead;  b->khead += r->nk;
    p0 = b->head;   b->head  += (uint64_t)r->n;
    pthread_mutex_unlock(&b->mu);

    const uint64_t kbase = k0 % b->kcap;
    memcpy(b->keys + kbase, r->keys, (size_t)r->nk * sizeof(MoveKey));
    memcpy(b->pol  + kbase, r->pol,  (size_t)r->nk * sizeof(float));
    for (int i = 0; i < r->n; i++) {
        BPos p = r->pos[i];
        p.koff = k0 + p.koff;
        b->pos[(p0 + (uint64_t)i) % b->cap] = p;
    }
}

/* ========================================================================= */
/*                            CONFIG AND SHARED STATE                        */
/* ========================================================================= */

enum { MODE_SELFPLAY = 0, MODE_LEARN, MODE_GEN, MODE_MICRO, MODE_SCALE };
enum { JOB_SELFPLAY = 0, JOB_LEARN, JOB_EXIT };

typedef struct {
    int   mode;
    int   threads;
    int   n_agents;
    int   games;             /* games per self-play round                    */
    int   sims, cap_sims;
    float cap_frac;
    int   max_plies, opening_plies;
    float temp_start, temp_end;
    float c_puct, dirichlet_alpha, dirichlet_eps;
    float resign_threshold, resign_check_frac, draw_penalty;
    int   start_mode;        /* 0 classical, 1 960, 2 mixed                  */
    int   buffer_positions, batch_size, steps;
    float lr, weight_decay, grad_clip, value_coef, value_mix;
    int   qos;
    int   cache, reuse;
    uint64_t seed;
    int   csv, quiet;
} BCfg;

typedef struct BShared BShared;

typedef struct {
    BShared  *sh;
    int       tid;
    pthread_t th;
    uint64_t  rng[4];
    Mcts      m;
    Game      g;
    BRec      rec;
    Move      list[MAX_MOVES];
    int32_t   visits[MAX_MOVES];
    float     target[MAX_MOVES];
    TrunkGrad *tg;
    /* results */
    uint64_t  games, plies, moves, full_moves, evals;
    uint64_t  cache_hits, cache_misses, reuse_hits, reuse_misses;
    uint64_t  positions;
    double    busy_sec;
    uint64_t  probe_ns[BP_N_OR_1];
    uint64_t  probe_cnt[BP_N_OR_1];
} BWorker;

struct BShared {
    const BCfg *cfg;
    Trunk      *trunk;
    Head       *heads;
    HeadGrad   *hgrad;
    Adam       *head_adam;
    Adam        trunk_adam;
    TrunkGrad  *tgsum;
    BBuf        buf;
    BWorker    *workers;
    int         nthreads;
    _Atomic int next;
    int         ngames;
    int         cap_sims;
    int         job;
    Bar         bar;
    /* learning batch */
    int32_t    *batch, *batch_tmp;
    int        *acount, *aoff;
    int         tstart[MAX_THREADS + 1];
    float       lr_now;
    uint64_t    lrng[4];
};

/* ========================================================================= */
/*                                 SELF-PLAY                                 */
/* ========================================================================= */
/* A line-for-line replica of az_play_one(): same start-position policy, same
 * playout-cap randomisation, same resignation rule, same temperature
 * schedule, same recording. */

static void play_one(BWorker *w)
{
    BShared *sh = w->sh;
    const BCfg *c = sh->cfg;

    Game *g = &w->g;
    BRec *r = &w->rec;

    if (c->start_mode == 1) {
        game_start960(g, pos_960_random(w->rng));
    } else if (c->start_mode == 2) {
        if (rng_u01(w->rng) < 0.10f) game_start(g);
        else                         game_start960(g, pos_960_random(w->rng));
    } else {
        game_start(g);
    }
    r->n = 0;
    r->nk = 0;

    /* The two agents of this game.  Which agents play which game is a
     * population-dynamics question and irrelevant to throughput; what matters
     * for cost is that the two sides use DIFFERENT heads, because that is what
     * invalidates the evaluation cache and the standing tree between moves. */
    const int a_w = (int)(rng_next(w->rng) % (uint64_t)c->n_agents);
    int a_b = (int)(rng_next(w->rng) % (uint64_t)c->n_agents);
    if (c->n_agents > 1 && a_b == a_w) a_b = (a_w + 1) % c->n_agents;
    const Head *hw = &sh->heads[a_w], *hb = &sh->heads[a_b];

    const int resign_on   = (c->resign_threshold > -1.0f);
    const int check_game  = resign_on && (rng_u01(w->rng) < c->resign_check_frac);
    const int resign_live = resign_on && !check_game;
    int consec[2] = { 0, 0 };
    int would_resign = -1, resigned = -1;
    int result;

    for (;;) {
        if (g->result != GR_ONGOING) break;
        if (g->ply >= c->max_plies) { g->result = GR_DRAW; g->reason = TR_MAX_PLIES; break; }

        const int side = (int)g->pos.side;
        const int aid  = (side == WHITE) ? a_w : a_b;
        const Head *h  = (side == WHITE) ? hw : hb;

        const int full  = !(c->cap_frac < 1.0f) || (rng_u01(w->rng) < c->cap_frac);
        const int nsims = full ? c->sims : sh->cap_sims;

        float rootv = 0.0f;
        int n;
        {
            BP_T0(s);
            n = mcts_search(&w->m, sh->trunk, h, g, nsims, full,
                            w->rng, w->visits, &rootv);
            BP_T1(s, BP_SEARCH);
        }
        if (n <= 0) break;
        w->moves++;
        if (full) w->full_moves++;

        BP_T0(d);

        const int nl = gen_legal(&g->pos, w->list);
        if (nl != n) { BP_T1(d, BP_DRIVER); break; }

        mcts_target(w->visits, n, w->target);

        /* ---- record: features, the visit-count target, the mover --------- */
        if (full && r->n < r->npos_cap &&
            r->nk + (uint64_t)n <= r->nk_cap && n <= MAX_MOVES) {
            BPos *p = &r->pos[r->n];
            p->nf     = (uint8_t)nn_features(&g->pos, p->fidx);
            p->koff   = r->nk;
            p->nmoves = (uint16_t)n;
            p->mover  = (uint8_t)side;
            p->agent  = (int16_t)aid;
            p->z      = 0.0f;
            p->q      = rootv;
            for (int i = 0; i < n; i++) {
                nn_move_key(&g->pos, w->list[i], &r->keys[r->nk + (uint64_t)i]);
                r->pol[r->nk + (uint64_t)i] = w->target[i];
            }
            r->nk += (uint64_t)n;
            r->n++;
            w->positions++;
        }

        if (resign_on) {
            if (rootv < c->resign_threshold) {
                if (++consec[side] >= RESIGN_CONSEC && would_resign < 0)
                    would_resign = side;
            } else {
                consec[side] = 0;
            }
            if (resign_live && would_resign == side) { resigned = side; BP_T1(d, BP_DRIVER); break; }
        }

        const float temp = (g->ply < c->opening_plies) ? c->temp_start : c->temp_end;
        const int pick = mcts_pick(w->visits, n, temp, w->rng);
        game_push(g, w->list[pick < 0 ? 0 : pick]);

        BP_T1(d, BP_DRIVER);
    }
    (void)check_game;

    if (resigned >= 0)              result = (resigned == WHITE) ? GR_BLACK_WIN : GR_WHITE_WIN;
    else if (g->result == GR_ONGOING) result = GR_DRAW;
    else                            result = g->result;

    for (int i = 0; i < r->n; i++) {
        const int mover = (int)r->pos[i].mover;
        float z;
        if (result == GR_DRAW)           z = c->draw_penalty;
        else if (result == GR_WHITE_WIN) z = (mover == WHITE) ? 1.0f : -1.0f;
        else                             z = (mover == BLACK) ? 1.0f : -1.0f;
        r->pos[i].z = z;
    }
    bbuf_commit(&sh->buf, r);

    w->games++;
    w->plies += (uint64_t)g->ply;
}

static void selfplay_job(BWorker *w)
{
    BShared *sh = w->sh;
    const double t0 = now_sec();
    w->m.evals = 0;
    w->m.cache_hits = w->m.cache_misses = 0;
    w->m.reuse_hits = w->m.reuse_misses = 0;
    for (;;) {
        const int i = atomic_fetch_add_explicit(&sh->next, 1, memory_order_relaxed);
        if (i >= sh->ngames) break;
        play_one(w);
    }
    w->evals        += w->m.evals;
    w->cache_hits   += w->m.cache_hits;
    w->cache_misses += w->m.cache_misses;
    w->reuse_hits   += w->m.reuse_hits;
    w->reuse_misses += w->m.reuse_misses;
    w->busy_sec     += now_sec() - t0;
}

/* ========================================================================= */
/*                                 LEARNING                                  */
/* ========================================================================= */

static void make_batch(BShared *sh)
{
    const BCfg *c = sh->cfg;
    BBuf *b = &sh->buf;
    const int B = c->batch_size, n = c->n_agents;
    const uint64_t count = bbuf_count(b);

    int got = 0;
    if (count > 0) {
        const int tries = B * 8 + 64;
        for (int t = 0; t < tries && got < B; t++) {
            const uint64_t u = rng_next(sh->lrng) % count;
            const uint64_t slot = (b->head - 1u - u) % b->cap;
            const BPos *p = &b->pos[slot];
            if (p->nmoves == 0 || p->agent < 0 || p->agent >= n) continue;
            if (b->khead - p->koff > b->kcap) continue;
            sh->batch_tmp[got++] = (int32_t)slot;
        }
    }

    memset(sh->acount, 0, (size_t)n * sizeof(int));
    for (int i = 0; i < got; i++) sh->acount[b->pos[sh->batch_tmp[i]].agent]++;
    int acc = 0;
    for (int a = 0; a < n; a++) { sh->aoff[a] = acc; acc += sh->acount[a]; }
    for (int i = 0; i < got; i++) {
        const int a = b->pos[sh->batch_tmp[i]].agent;
        sh->batch[sh->aoff[a]++] = sh->batch_tmp[i];
    }

    const int T = sh->nthreads;
    sh->tstart[0] = 0;
    for (int t = 1; t < T; t++) {
        int j = (int)((long)t * (long)got / (long)T);
        while (j > 0 && j < got &&
               b->pos[sh->batch[j]].agent == b->pos[sh->batch[j - 1]].agent) j++;
        if (j < sh->tstart[t - 1]) j = sh->tstart[t - 1];
        sh->tstart[t] = j;
    }
    sh->tstart[T] = got;
    for (int t = T; t > 0; t--)
        if (sh->tstart[t - 1] > sh->tstart[t]) sh->tstart[t - 1] = sh->tstart[t];
}

/* Exactly az_position_grad()'s arithmetic, minus the telemetry it also
 * accumulates (which is not on the throughput path). */
static void position_grad(BWorker *w, const BPos *p, const MoveKey *keys,
                          const float *pi, HeadGrad *hg, const Head *h)
{
    BShared *sh = w->sh;
    const int n = (int)p->nmoves;
    float logits[MAX_MOVES], prob[MAX_MOVES], dl[MAX_MOVES];
    Fwd fw;

    {
        BP_T0(f);
        nn_eval(sh->trunk, h, p->fidx, (int)p->nf, &fw);
        nn_logits(h, &fw, keys, n, logits);
        softmax_t(logits, n, 1.0f, prob);
        BP_T1(f, BP_LEARN_FWD);
    }

    for (int i = 0; i < n; i++) dl[i] = prob[i] - pi[i];

    const float mix = sh->cfg->value_mix;
    float tgt = (1.0f - mix) * p->z + mix * p->q;
    if (tgt >  1.0f) tgt =  1.0f;
    if (tgt < -1.0f) tgt = -1.0f;
    const float dv = 2.0f * sh->cfg->value_coef * (fw.v - tgt);

    {
        BP_T0(b);
        nn_backward(sh->trunk, h, &fw, p->fidx, (int)p->nf, keys, n, dl, dv, w->tg, hg);
        BP_T1(b, BP_BACKWARD);
    }
}

static void learn_job(BWorker *w)
{
    BShared *sh = w->sh;
    const BCfg *c = sh->cfg;
    BBuf *b = &sh->buf;
    const int tid = w->tid;
    const double t0 = now_sec();
    BP_T0(L);

    for (int step = 0; step < c->steps; step++) {
        if (tid == 0) make_batch(sh);
        bar_wait(&sh->bar);

        const int lo = sh->tstart[tid], hi = sh->tstart[tid + 1];
        grad_zero(w->tg, (int)TRUNK_NPARAM);

        for (int i = lo; i < hi; i++) {
            const BPos *p = &b->pos[sh->batch[i]];
            const uint64_t kb = p->koff % b->kcap;
            position_grad(w, p, b->keys + kb, b->pol + kb,
                          &sh->hgrad[p->agent], &sh->heads[p->agent]);
        }

        for (int i = lo; i < hi; ) {
            const int a = b->pos[sh->batch[i]].agent;
            int j = i;
            while (j < hi && b->pos[sh->batch[j]].agent == a) j++;
            const float s = 1.0f / (float)(j - i);
            float *hgp = (float *)&sh->hgrad[a];
            for (size_t k = 0; k < HEAD_NPARAM; k++) hgp[k] *= s;
            adam_step(&sh->head_adam[a], (float *)&sh->heads[a], hgp,
                      sh->lr_now, c->weight_decay, c->grad_clip);
            i = j;
        }

        bar_wait(&sh->bar);

        if (tid == 0) {
            const int nb = sh->tstart[sh->nthreads];
            if (nb > 0) {
                float *gp = (float *)sh->tgsum;
                for (int t = 0; t < sh->nthreads; t++)
                    grad_add(gp, (const float *)sh->workers[t].tg, (int)TRUNK_NPARAM);
                const float s = 1.0f / (float)nb;
                for (size_t k = 0; k < TRUNK_NPARAM; k++) gp[k] *= s;
                adam_step(&sh->trunk_adam, (float *)sh->trunk, gp,
                          sh->lr_now, c->weight_decay, c->grad_clip);
            }
        }
        bar_wait(&sh->bar);
    }
    BP_T1(L, BP_LEARN);
    w->busy_sec += now_sec() - t0;
}

/* ========================================================================= */
/*                               THREAD POOL                                 */
/* ========================================================================= */

static void *worker_main(void *arg)
{
    BWorker *w = (BWorker *)arg;
    BShared *sh = w->sh;
    qos_apply(sh->cfg->qos);
    for (;;) {
        bar_wait(&sh->bar);
        if (sh->job == JOB_EXIT) break;
        if (sh->job == JOB_SELFPLAY) selfplay_job(w);
        else if (sh->job == JOB_LEARN) learn_job(w);
        bar_wait(&sh->bar);
    }
#ifdef BENCH_PROBE
    memcpy(w->probe_ns,  bp_ns,  sizeof bp_ns);
    memcpy(w->probe_cnt, bp_cnt, sizeof bp_cnt);
#endif
    return NULL;
}

static void dispatch(BShared *sh, int job)
{
    sh->job = job;
    bar_wait(&sh->bar);
    if (job == JOB_SELFPLAY)   selfplay_job(&sh->workers[0]);
    else if (job == JOB_LEARN) learn_job(&sh->workers[0]);
    bar_wait(&sh->bar);
}

/* ========================================================================= */
/*                              MICRO-BENCHMARKS                             */
/* ========================================================================= */

/* The cost of the clock itself, so the instrumented split can be corrected. */
static double clock_cost_ns(void)
{
    const int N = 200000;
    struct timespec ts;
    const double a = now_sec();
    for (int i = 0; i < N; i++) { clock_gettime(CLOCK_MONOTONIC, &ts); }
    const double b = now_sec();
    return (b - a) * 1e9 / (double)N;
}

typedef struct {
    const Trunk *trunk;
    const Head  *head;
    int          seconds_x100;
    int          which;
    uint64_t     iters;
    double       sec;
    double       checksum;
} MicroArg;

enum { MB_EVAL = 0, MB_LOGITS, MB_FEATURES, MB_GENLEGAL, MB_MAKEUNMAKE,
       MB_BACKWARD, MB_STREAM, MB_N };

static const char *mb_name(int w)
{
    static const char *n[MB_N] = {
        "nn_eval", "nn_logits (35 moves)", "nn_features", "gen_legal",
        "make_move+unmake_move", "nn_backward (35 moves)", "trunk stream (read)"
    };
    return n[w];
}

/* A mid-game position reached by a short random walk, so the micro-benchmarks
 * see a realistic branching factor and feature count rather than the opening. */
static void micro_position(Position *p, uint16_t *fidx, int *nf,
                           Move *list, int *nmoves, uint64_t *rng)
{
    Game g;
    game_start(&g);
    for (int i = 0; i < 24 && g.result == GR_ONGOING; i++) {
        Move l[MAX_MOVES];
        const int n = gen_legal(&g.pos, l);
        if (n <= 0) break;
        game_push(&g, l[rng_next(rng) % (uint64_t)n]);
    }
    *p = g.pos;
    *nf = nn_features(p, fidx);
    *nmoves = gen_legal(p, list);
}

static void *micro_main(void *arg)
{
    MicroArg *a = (MicroArg *)arg;
    uint64_t rng[4];
    rng_seed(rng, 0xB0BAB0BAull + (uint64_t)a->which);

    Position pos;
    uint16_t fidx[NF_MAXACTIVE];
    int nf = 0, nmoves = 0;
    Move list[MAX_MOVES];
    micro_position(&pos, fidx, &nf, list, &nmoves, rng);

    MoveKey keys[MAX_MOVES];
    for (int i = 0; i < nmoves; i++) nn_move_key(&pos, list[i], &keys[i]);

    Fwd fw;
    float logits[MAX_MOVES], dl[MAX_MOVES];
    nn_eval(a->trunk, a->head, fidx, nf, &fw);
    for (int i = 0; i < nmoves; i++) dl[i] = 0.001f * (float)i;

    TrunkGrad *tg = (TrunkGrad *)calloc(1, sizeof(TrunkGrad));
    HeadGrad  *hg = (HeadGrad  *)calloc(1, sizeof(HeadGrad));

    const double limit = (double)a->seconds_x100 / 100.0;
    const double t0 = now_sec();
    uint64_t it = 0;
    double sum = 0.0;

    for (;;) {
        for (int k = 0; k < 64; k++) {
            switch (a->which) {
                case MB_EVAL:
                    nn_eval(a->trunk, a->head, fidx, nf, &fw);
                    sum += (double)fw.v;
                    break;
                case MB_LOGITS:
                    nn_logits(a->head, &fw, keys, nmoves, logits);
                    sum += (double)logits[0];
                    break;
                case MB_FEATURES: {
                    uint16_t f2[NF_MAXACTIVE];
                    sum += (double)nn_features(&pos, f2);
                    break;
                }
                case MB_GENLEGAL: {
                    Move l2[MAX_MOVES];
                    sum += (double)gen_legal(&pos, l2);
                    break;
                }
                case MB_MAKEUNMAKE: {
                    Undo u;
                    Position q = pos;
                    make_move(&q, list[(int)(it & 7) % nmoves], &u);
                    unmake_move(&q, list[(int)(it & 7) % nmoves], &u);
                    sum += (double)q.side;
                    break;
                }
                case MB_BACKWARD:
                    if (tg && hg)
                        nn_backward(a->trunk, a->head, &fw, fidx, nf, keys, nmoves,
                                    dl, 0.5f, tg, hg);
                    sum += 1.0;
                    break;
                case MB_STREAM: {
                    /* Read the whole trunk once: the memory-bandwidth floor a
                     * single evaluation would hit if it touched every weight. */
                    const float *p = (const float *)a->trunk;
                    float s = 0.0f;
                    for (size_t i = 0; i < TRUNK_NPARAM; i += 16) s += p[i];
                    sum += (double)s;
                    break;
                }
                default: break;
            }
            it++;
        }
        if (now_sec() - t0 >= limit) break;
    }

    a->sec = now_sec() - t0;
    a->iters = it;
    a->checksum = sum;
    free(tg); free(hg);
    return NULL;
}

/* ========================================================================= */
/*                                  OUTPUT                                   */
/* ========================================================================= */

typedef struct {
    double wall, selfplay_sec, learn_sec;
    uint64_t games, plies, moves, full_moves, evals, positions;
    uint64_t cache_hits, cache_misses, reuse_hits, reuse_misses;
    uint64_t per_thread_games[MAX_THREADS];
    double   per_thread_busy[MAX_THREADS];
    int      threads;
} BResult;

static void collect(BShared *sh, BResult *R)
{
    memset(R, 0, sizeof *R);
    R->threads = sh->nthreads;
    for (int i = 0; i < sh->nthreads; i++) {
        BWorker *w = &sh->workers[i];
        R->games += w->games; R->plies += w->plies; R->moves += w->moves;
        R->full_moves += w->full_moves; R->evals += w->evals;
        R->positions += w->positions;
        R->cache_hits += w->cache_hits; R->cache_misses += w->cache_misses;
        R->reuse_hits += w->reuse_hits; R->reuse_misses += w->reuse_misses;
        if (i < MAX_THREADS) {
            R->per_thread_games[i] = w->games;
            R->per_thread_busy[i]  = w->busy_sec;
        }
    }
}

static void print_selfplay(const BResult *R, double sec, const BCfg *c)
{
    printf("  wall clock            %12.3f s\n", sec);
    printf("  games                 %12llu\n", (unsigned long long)R->games);
    printf("  plies                 %12llu\n", (unsigned long long)R->plies);
    printf("  moves searched        %12llu   (%llu full-budget, %llu capped)\n",
           (unsigned long long)R->moves, (unsigned long long)R->full_moves,
           (unsigned long long)(R->moves - R->full_moves));
    printf("  network evaluations   %12llu\n", (unsigned long long)R->evals);
    printf("  positions recorded    %12llu\n", (unsigned long long)R->positions);
    printf("  ------------------------------------------------------------\n");
    printf("  GAMES / SECOND        %12.2f\n", (double)R->games / sec);
    printf("  moves / second        %12.1f\n", (double)R->moves / sec);
    printf("  EVALUATIONS / SECOND  %12.0f\n", (double)R->evals / sec);
    printf("  evals / move          %12.2f\n",
           R->moves ? (double)R->evals / (double)R->moves : 0.0);
    printf("  evals / game          %12.1f\n",
           R->games ? (double)R->evals / (double)R->games : 0.0);
    printf("  plies / game          %12.1f\n",
           R->games ? (double)R->plies / (double)R->games : 0.0);
    printf("  ------------------------------------------------------------\n");
    {
        const uint64_t tot = R->cache_hits + R->cache_misses;
        printf("  eval-cache hit rate   %12.1f %%  (%llu hit / %llu lookup)\n",
               tot ? 100.0 * (double)R->cache_hits / (double)tot : 0.0,
               (unsigned long long)R->cache_hits, (unsigned long long)tot);
    }
    {
        const uint64_t tot = R->reuse_hits + R->reuse_misses;
        printf("  subtree reuse rate    %12.1f %%  (%llu / %llu searches)\n",
               tot ? 100.0 * (double)R->reuse_hits / (double)tot : 0.0,
               (unsigned long long)R->reuse_hits, (unsigned long long)tot);
    }
    /* GFLOP/s, at 2 FLOP per MAC.  The MAC count is the network's shape, not
     * an estimate: it is printed by --mode micro. */
    (void)c;
}

static void print_threads(const BResult *R, double sec)
{
    printf("  per-thread game counts (a shared atomic index: this is the\n"
           "  self-balancing check -- equal counts mean no thread starved)\n");
    uint64_t mn = (uint64_t)-1, mx = 0;
    for (int i = 0; i < R->threads; i++) {
        const uint64_t g = R->per_thread_games[i];
        if (g < mn) mn = g;
        if (g > mx) mx = g;
        printf("    thread %-2d  %6llu games   busy %6.3f s (%5.1f%% of wall)\n",
               i, (unsigned long long)g, R->per_thread_busy[i],
               sec > 0 ? 100.0 * R->per_thread_busy[i] / sec : 0.0);
    }
    if (mx > 0)
        printf("    spread: min %llu  max %llu  ratio %.2fx\n",
               (unsigned long long)mn, (unsigned long long)mx,
               mn ? (double)mx / (double)mn : 0.0);
}

#ifdef BENCH_PROBE
static void print_probe(BShared *sh, double wall, double clk_ns)
{
    uint64_t ns[BP_N], cnt[BP_N];
    memset(ns, 0, sizeof ns);
    memset(cnt, 0, sizeof cnt);
    for (int i = 0; i < sh->nthreads; i++) {
        for (int b = 0; b < BP_N; b++) {
            ns[b]  += sh->workers[i].probe_ns[b];
            cnt[b] += sh->workers[i].probe_cnt[b];
        }
    }

    /* Every timed wrapper spends 2 clock reads; the enclosing scope timer
     * therefore contains (2 * clk_ns * inner calls) that is not real work.
     * Correct each bucket for its own 2 reads and report both numbers. */
    const int timed[] = { BP_EVAL, BP_LOGITS, BP_FEATURES, BP_GENLEGAL,
                          BP_BACKWARD, BP_LEARN_FWD, BP_SEARCH, BP_DRIVER,
                          BP_LEARN };
    double corr[BP_N];
    for (int b = 0; b < BP_N; b++) {
        corr[b] = (double)ns[b];
        for (unsigned k = 0; k < sizeof timed / sizeof timed[0]; k++)
            if (timed[k] == b) { corr[b] -= 2.0 * clk_ns * (double)cnt[b]; break; }
        if (corr[b] < 0.0) corr[b] = 0.0;
    }

    const double thread_sec = wall * (double)sh->nthreads;

    printf("\n  PHASE SPLIT (instrumented build; thread-seconds, %d threads,\n"
           "  %.3f s wall = %.3f thread-seconds available)\n",
           sh->nthreads, wall, thread_sec);
    printf("  %-26s %12s %10s %12s %8s\n",
           "phase", "calls", "sec", "ns/call", "%% wall");
    printf("  ---------------------------------------------------------------------\n");
    for (int b = 0; b < BP_N; b++) {
        if (!cnt[b]) continue;
        printf("  %-26s %12llu %10.3f %12.1f %8.2f\n",
               bp_name(b), (unsigned long long)cnt[b], corr[b] * 1e-9,
               corr[b] / (double)cnt[b],
               100.0 * corr[b] * 1e-9 / thread_sec);
    }
    printf("  ---------------------------------------------------------------------\n");

    const double net   = corr[BP_EVAL] + corr[BP_LOGITS];
    const double enc   = corr[BP_FEATURES];
    const double rules = corr[BP_GENLEGAL];
    const double srch  = corr[BP_SEARCH];
    const double book  = srch - net - enc - rules;
    printf("  inside mcts_search: network %.3f s, features %.3f s,\n"
           "                      gen_legal %.3f s, bookkeeping (by\n"
           "                      subtraction) %.3f s\n",
           net * 1e-9, enc * 1e-9, rules * 1e-9, book * 1e-9);
    printf("  UNTIMED CHEAP CALLS (counted only; multiply by --mode micro's\n"
           "  ns/call to attribute them -- they are inside `bookkeeping` above)\n");
    printf("    make_move %llu   unmake_move %llu   in_check %llu\n"
           "    insufficient_material %llu   nn_move_key %llu   softmax_t %llu\n",
           (unsigned long long)cnt[BP_MAKE], (unsigned long long)cnt[BP_UNMAKE],
           (unsigned long long)cnt[BP_INCHECK], (unsigned long long)cnt[BP_INSUF],
           (unsigned long long)cnt[BP_MOVEKEY], (unsigned long long)cnt[BP_SOFTMAX]);
}
#endif

/* ========================================================================= */
/*                                   CLI                                     */
/* ========================================================================= */

static const char *opt_str(int argc, char **argv, const char *k, const char *d)
{
    for (int i = 1; i < argc - 1; i++) if (!strcmp(argv[i], k)) return argv[i + 1];
    return d;
}
static double opt_num(int argc, char **argv, const char *k, double d)
{
    const char *s = opt_str(argc, argv, k, NULL);
    return s ? atof(s) : d;
}
static long opt_int(int argc, char **argv, const char *k, long d)
{
    const char *s = opt_str(argc, argv, k, NULL);
    return s ? strtol(s, NULL, 10) : d;
}
static int opt_flag(int argc, char **argv, const char *k)
{
    for (int i = 1; i < argc; i++) if (!strcmp(argv[i], k)) return 1;
    return 0;
}

static const char USAGE[] =
"bench_throughput -- ChessRL self-play throughput and where it goes\n"
"\n"
"  --mode M          selfplay | learn | gen | micro | scale   (default gen)\n"
"  --threads N       worker threads                            (default 8)\n"
"  --games N         games in a self-play round                (default 256)\n"
"  --sims N          MCTS simulations for a full-budget move    (default 160)\n"
"  --cap-frac X      fraction of moves given the full budget    (default 0.25)\n"
"  --cap-sims N      budget for the rest; 0 = max(2, sims/5)    (default 0)\n"
"  --agents N        population size                            (default 32)\n"
"  --max-plies N     draw adjudication                          (default 200)\n"
"  --batch N         learning minibatch                         (default 256)\n"
"  --steps N         optimiser steps in the learning phase      (default 200)\n"
"  --buffer N        replay-buffer capacity in positions        (default 50000)\n"
"  --qos Q           off | ui | utility | background | default  (default off)\n"
"                    ui      = QOS_CLASS_USER_INTERACTIVE (P-cores)\n"
"                    utility = QOS_CLASS_UTILITY          (E-cores)\n"
"  --no-cache        disable the MCTS evaluation cache\n"
"  --no-reuse        disable subtree reuse\n"
"  --seed N          RNG seed                                   (default 20260911)\n"
"  --secs X          micro-benchmark seconds per primitive      (default 0.5)\n"
"  --csv             one machine-readable line per measurement\n"
"  --quiet           totals only\n"
"  --help\n";

/* ========================================================================= */

static int run_selfplay_round(BShared *sh, double *out_sec)
{
    atomic_store(&sh->next, 0);
    const double t0 = now_sec();
    dispatch(sh, JOB_SELFPLAY);
    *out_sec = now_sec() - t0;
    return 0;
}

static void zero_workers(BShared *sh)
{
    for (int i = 0; i < sh->nthreads; i++) {
        BWorker *w = &sh->workers[i];
        w->games = w->plies = w->moves = w->full_moves = w->evals = 0;
        w->cache_hits = w->cache_misses = w->reuse_hits = w->reuse_misses = 0;
        w->positions = 0;
        w->busy_sec = 0.0;
    }
}

int main(int argc, char **argv)
{
    if (opt_flag(argc, argv, "--help") || opt_flag(argc, argv, "-h")) {
        fputs(USAGE, stdout);
        return 0;
    }

    chess_init();

    BCfg c;
    memset(&c, 0, sizeof c);
    const char *mode = opt_str(argc, argv, "--mode", "gen");
    c.mode = !strcmp(mode, "selfplay") ? MODE_SELFPLAY
           : !strcmp(mode, "learn")    ? MODE_LEARN
           : !strcmp(mode, "micro")    ? MODE_MICRO
           : !strcmp(mode, "scale")    ? MODE_SCALE
           : MODE_GEN;

    c.threads          = (int)opt_int(argc, argv, "--threads", 8);
    c.n_agents         = (int)opt_int(argc, argv, "--agents", 32);
    c.games            = (int)opt_int(argc, argv, "--games", 256);
    c.sims             = (int)opt_int(argc, argv, "--sims", 160);
    c.cap_sims         = (int)opt_int(argc, argv, "--cap-sims", 0);
    c.cap_frac         = (float)opt_num(argc, argv, "--cap-frac", 0.25);
    c.max_plies        = (int)opt_int(argc, argv, "--max-plies", 200);
    c.opening_plies    = (int)opt_int(argc, argv, "--opening-plies", 20);
    c.temp_start       = 1.0f;
    c.temp_end         = 0.25f;
    c.c_puct           = (float)opt_num(argc, argv, "--c-puct", 1.4);
    c.dirichlet_alpha  = 0.3f;
    c.dirichlet_eps    = 0.25f;
    c.resign_threshold = (float)opt_num(argc, argv, "--resign", -0.90);
    c.resign_check_frac = 0.10f;
    c.draw_penalty     = -0.10f;
    c.start_mode       = (int)opt_int(argc, argv, "--start-mode", 2);
    c.buffer_positions = (int)opt_int(argc, argv, "--buffer", 50000);
    c.batch_size       = (int)opt_int(argc, argv, "--batch", 256);
    c.steps            = (int)opt_int(argc, argv, "--steps", 200);
    c.lr               = (float)opt_num(argc, argv, "--lr", 2e-3);
    c.weight_decay     = 1e-4f;
    c.grad_clip        = 4.0f;
    c.value_coef       = 4.0f;
    c.value_mix        = 0.0f;
    c.cache            = !opt_flag(argc, argv, "--no-cache");
    c.reuse            = !opt_flag(argc, argv, "--no-reuse");
    c.seed             = (uint64_t)opt_int(argc, argv, "--seed", 20260911);
    c.csv              = opt_flag(argc, argv, "--csv");
    c.quiet            = opt_flag(argc, argv, "--quiet");

    {
        const char *q = opt_str(argc, argv, "--qos", "off");
        c.qos = !strcmp(q, "ui")         ? QOS_UI
              : !strcmp(q, "utility")    ? QOS_UTILITY
              : !strcmp(q, "background") ? QOS_BACKGROUND
              : !strcmp(q, "default")    ? QOS_DEFAULT_CLASS
              : QOS_OFF;
    }

    if (c.threads < 1) c.threads = 1;
    if (c.threads > MAX_THREADS) c.threads = MAX_THREADS;
    if (c.n_agents < 1) c.n_agents = 1;
    if (c.sims < 2) c.sims = 2;
    if (c.cap_sims <= 0) { c.cap_sims = c.sims / 5; if (c.cap_sims < 2) c.cap_sims = 2; }

    const double micro_secs = opt_num(argc, argv, "--secs", 0.5);
    const double clk_ns = clock_cost_ns();

    /* ------------------------------------------------------------ model */
    Trunk *trunk = (Trunk *)calloc(1, sizeof(Trunk));
    Head  *heads = (Head  *)calloc((size_t)c.n_agents, sizeof(Head));
    if (!trunk || !heads) { fprintf(stderr, "bench: out of memory\n"); return 1; }
    for (int a = 0; a < c.n_agents; a++)
        nn_init(trunk, &heads[a], c.seed + 0x9E3779B9ull * (uint64_t)(a + 1));

    printf("bench_throughput  mode=%s threads=%d sims=%d cap_frac=%.2f cap_sims=%d\n",
           mode, c.threads, c.sims, (double)c.cap_frac, c.cap_sims);
    printf("  agents %d  max_plies %d  start %s  cache %s  reuse %s  qos %s\n",
           c.n_agents, c.max_plies,
           c.start_mode == 0 ? "classical" : c.start_mode == 1 ? "960" : "mixed",
           c.cache ? "on" : "OFF", c.reuse ? "on" : "OFF", qos_name(c.qos));
    printf("  trunk %.1f KB (%zu params)  head %.1f KB (%zu params) x %d agents\n",
           sizeof(Trunk) / 1024.0, (size_t)TRUNK_NPARAM,
           sizeof(Head) / 1024.0, (size_t)HEAD_NPARAM, c.n_agents);
    printf("  clock_gettime(CLOCK_MONOTONIC) costs %.1f ns/call\n", clk_ns);
#ifdef BENCH_PROBE
    printf("  build: INSTRUMENTED (-DBENCH_PROBE); phase timers are ON\n");
#else
    printf("  build: clean (no timers compiled in)\n");
#endif
    printf("\n");
    fflush(stdout);

    /* ------------------------------------------------------------ micro */
    if (c.mode == MODE_MICRO) {
        /* The network's MAC count, from its shape rather than from a guess. */
        const long macs_trunk = (long)35 * NF_ACC + (long)NF_ACC * NF_HID
                              + (long)35 * NF_VACC + (long)NF_VACC * NF_VHID
                              + NF_VHID + (long)NF_HID * NF_PDIM;
        printf("  network shape: W0 gather 35x%d, W1 %dx%d, W0v gather 35x%d,\n"
               "                 Wvh %dx%d, Wv %d, Wp %dx%d\n",
               NF_ACC, NF_ACC, NF_HID, NF_VACC, NF_VACC, NF_VHID, NF_VHID,
               NF_HID, NF_PDIM);
        printf("  MACs per nn_eval: %ld  (%.1f kFLOP at 2 FLOP/MAC)\n\n",
               macs_trunk, 2.0 * (double)macs_trunk / 1000.0);

        printf("  %-24s %10s %12s %12s\n", "primitive", "calls", "ns/call", "M calls/s");
        printf("  ------------------------------------------------------------\n");
        double eval_ns = 0.0;
        for (int wmode = 0; wmode < MB_N; wmode++) {
            MicroArg a;
            memset(&a, 0, sizeof a);
            a.trunk = trunk; a.head = &heads[0];
            a.seconds_x100 = (int)(micro_secs * 100.0);
            a.which = wmode;
            micro_main(&a);
            const double ns = a.sec * 1e9 / (double)a.iters;
            if (wmode == MB_EVAL) eval_ns = ns;
            printf("  %-24s %10llu %12.1f %12.3f\n", mb_name(wmode),
                   (unsigned long long)a.iters, ns, 1e3 / ns);
        }
        printf("  ------------------------------------------------------------\n");
        if (eval_ns > 0.0)
            printf("  single-thread nn_eval: %.0f evals/s = %.2f GFLOP/s\n",
                   1e9 / eval_ns, 2.0 * (double)macs_trunk / eval_ns);

        /* Scaling of the pure forward pass, 1 -> threads.  If evals/s/thread
         * falls as threads are added while nothing is shared but a read-only
         * 573 KB trunk, the limit is memory or the E-cores, not arithmetic. */
        printf("\n  nn_eval scaling (pure forward pass, no MCTS, no memory\n"
               "  written but the Fwd struct -- the arithmetic ceiling)\n");
        printf("  %-10s %14s %14s %10s\n", "threads", "evals/s", "per thread", "GFLOP/s");
        const int tset[] = { 1, 2, 3, 4, 6, 8 };
        for (unsigned ti = 0; ti < sizeof tset / sizeof tset[0]; ti++) {
            const int T = tset[ti];
            if (T > c.threads && T > 8) continue;
            MicroArg *as = (MicroArg *)calloc((size_t)T, sizeof(MicroArg));
            pthread_t *th = (pthread_t *)calloc((size_t)T, sizeof(pthread_t));
            for (int i = 0; i < T; i++) {
                as[i].trunk = trunk; as[i].head = &heads[i % c.n_agents];
                as[i].seconds_x100 = (int)(micro_secs * 100.0);
                as[i].which = MB_EVAL;
            }
            for (int i = 0; i < T; i++) pthread_create(&th[i], NULL, micro_main, &as[i]);
            double tot = 0.0, sec = 0.0;
            for (int i = 0; i < T; i++) {
                pthread_join(th[i], NULL);
                tot += (double)as[i].iters;
                if (as[i].sec > sec) sec = as[i].sec;
            }
            printf("  %-10d %14.0f %14.0f %10.2f\n", T, tot / sec, tot / sec / T,
                   2.0 * (double)macs_trunk * tot / sec / 1e9);
            free(as); free(th);
        }
        free(trunk); free(heads);
        return 0;
    }

    /* ------------------------------------------------------ shared state */
    BShared sh;
    memset(&sh, 0, sizeof sh);
    sh.cfg = &c;
    sh.trunk = trunk;
    sh.heads = heads;
    sh.nthreads = c.threads;
    sh.cap_sims = c.cap_sims;
    sh.lr_now = c.lr;
    rng_seed(sh.lrng, c.seed ^ 0xA5A5A5A5ull);

    if (!bbuf_init(&sh.buf, c.buffer_positions)) {
        fprintf(stderr, "bench: replay buffer allocation failed\n");
        return 1;
    }

    sh.hgrad     = (HeadGrad *)calloc((size_t)c.n_agents, sizeof(HeadGrad));
    sh.head_adam = (Adam *)    calloc((size_t)c.n_agents, sizeof(Adam));
    sh.tgsum     = (TrunkGrad *)calloc(1, sizeof(TrunkGrad));
    sh.batch     = (int32_t *)calloc((size_t)c.batch_size, sizeof(int32_t));
    sh.batch_tmp = (int32_t *)calloc((size_t)c.batch_size, sizeof(int32_t));
    sh.acount    = (int *)calloc((size_t)c.n_agents, sizeof(int));
    sh.aoff      = (int *)calloc((size_t)c.n_agents, sizeof(int));
    if (!sh.hgrad || !sh.head_adam || !sh.tgsum || !sh.batch || !sh.batch_tmp ||
        !sh.acount || !sh.aoff) {
        fprintf(stderr, "bench: out of memory\n");
        return 1;
    }
    adam_init(&sh.trunk_adam, (int)TRUNK_NPARAM);
    for (int a = 0; a < c.n_agents; a++) adam_init(&sh.head_adam[a], (int)HEAD_NPARAM);

    sh.workers = (BWorker *)calloc((size_t)c.threads, sizeof(BWorker));
    if (!sh.workers) { fprintf(stderr, "bench: out of memory\n"); return 1; }

    for (int i = 0; i < c.threads; i++) {
        BWorker *w = &sh.workers[i];
        w->sh = &sh;
        w->tid = i;
        rng_seed(w->rng, c.seed + 0xC0FFEEull * (uint64_t)(i + 1));
        mcts_init(&w->m, MAX_MOVES + 1 + c.sims * NODES_PER_SIM);
        w->m.c_puct          = c.c_puct;
        w->m.dirichlet_alpha = c.dirichlet_alpha;
        w->m.dirichlet_eps   = c.dirichlet_eps;
        w->m.cache           = c.cache;
        w->m.reuse           = c.reuse;
        w->tg = (TrunkGrad *)calloc(1, sizeof(TrunkGrad));
        if (!w->m.pool || !w->tg || !brec_init(&w->rec, c.max_plies)) {
            fprintf(stderr, "bench: out of memory (worker %d)\n", i);
            return 1;
        }
    }
    printf("  per-thread memory: node pool %.1f KB, eval cache 2^%d entries,\n"
           "                     trunk gradient %.1f KB (WRITTEN every step)\n",
           (double)(MAX_MOVES + 1 + c.sims * NODES_PER_SIM) * sizeof(MctsNode) / 1024.0,
           sh.workers[0].m.cache_bits, sizeof(TrunkGrad) / 1024.0);
    printf("  replay buffer: %.1f MB positions + %.1f MB move keys + %.1f MB policy\n\n",
           (double)sh.buf.cap * sizeof(BPos) / 1048576.0,
           (double)sh.buf.kcap * sizeof(MoveKey) / 1048576.0,
           (double)sh.buf.kcap * sizeof(float) / 1048576.0);
    fflush(stdout);

    bar_init(&sh.bar, c.threads);
    qos_apply(c.qos);                     /* thread 0 is a worker too */
    for (int i = 1; i < c.threads; i++)
        pthread_create(&sh.workers[i].th, NULL, worker_main, &sh.workers[i]);

    /* ------------------------------------------------------------ scale */
    if (c.mode == MODE_SCALE) {
        fprintf(stderr, "bench: --mode scale must be driven from the shell "
                        "(one process per thread count) so that each run gets "
                        "a fresh thread pool.  See docs/PERFORMANCE.md.\n");
    }

    /* ----------------------------------------------------------- warmup */
    /* One short round so that the replay buffer has data for the learning
     * phase and the page faults are already taken when the clock starts. */
    if (c.mode != MODE_LEARN) {
        sh.ngames = c.threads * 2;
        double warm = 0.0;
        run_selfplay_round(&sh, &warm);
        zero_workers(&sh);
        printf("  warmup: %d games in %.3f s\n\n", c.threads * 2, warm);
    } else {
        sh.ngames = c.threads * 8;
        double warm = 0.0;
        run_selfplay_round(&sh, &warm);
        zero_workers(&sh);
        printf("  warmup: filled the buffer with %llu positions in %.3f s\n\n",
               (unsigned long long)bbuf_count(&sh.buf), warm);
    }

    double sp_sec = 0.0, ln_sec = 0.0;
    BResult R;

    if (c.mode == MODE_SELFPLAY || c.mode == MODE_GEN || c.mode == MODE_SCALE) {
        sh.ngames = c.games;
        run_selfplay_round(&sh, &sp_sec);
        collect(&sh, &R);
        printf("SELF-PLAY\n");
        print_selfplay(&R, sp_sec, &c);
        if (!c.quiet) print_threads(&R, sp_sec);
    } else {
        memset(&R, 0, sizeof R);
        R.threads = c.threads;
    }

    if (c.mode == MODE_LEARN || c.mode == MODE_GEN) {
        zero_workers(&sh);
        const double t0 = now_sec();
        dispatch(&sh, JOB_LEARN);
        ln_sec = now_sec() - t0;
        printf("\nLEARNING (%d steps of batch %d over %llu buffered positions)\n",
               c.steps, c.batch_size, (unsigned long long)bbuf_count(&sh.buf));
        printf("  wall clock            %12.3f s\n", ln_sec);
        printf("  steps / second        %12.1f\n", (double)c.steps / ln_sec);
        printf("  ms / step             %12.3f\n", 1000.0 * ln_sec / (double)c.steps);
        printf("  positions / second    %12.0f\n",
               (double)c.steps * (double)c.batch_size / ln_sec);
        if (!c.quiet) {
            printf("  per-thread busy time (barrier-synchronised: the gap to wall\n"
                   "  clock is barrier and serial-reduction time)\n");
            for (int i = 0; i < c.threads; i++)
                printf("    thread %-2d  busy %6.3f s (%5.1f%% of wall)\n",
                       i, sh.workers[i].busy_sec,
                       100.0 * sh.workers[i].busy_sec / ln_sec);
        }
    }

    if (c.mode == MODE_GEN) {
        const double tot = sp_sec + ln_sec;
        printf("\nGENERATION SPLIT\n");
        printf("  self-play             %12.3f s   %6.2f %%\n", sp_sec, 100.0 * sp_sec / tot);
        printf("  learning              %12.3f s   %6.2f %%\n", ln_sec, 100.0 * ln_sec / tot);
        printf("  total                 %12.3f s\n", tot);
        printf("  games/sec over the WHOLE generation  %8.2f\n", (double)R.games / tot);
    }

    if (c.csv) {
        printf("\nCSV,mode,threads,sims,cap_frac,qos,games,moves,evals,sp_sec,ln_sec,"
               "games_per_s,evals_per_s\n");
        printf("CSV,%s,%d,%d,%.2f,%s,%llu,%llu,%llu,%.4f,%.4f,%.3f,%.0f\n",
               mode, c.threads, c.sims, (double)c.cap_frac,
               c.qos == QOS_UI ? "ui" : c.qos == QOS_UTILITY ? "utility" : "off",
               (unsigned long long)R.games, (unsigned long long)R.moves,
               (unsigned long long)R.evals, sp_sec, ln_sec,
               sp_sec > 0 ? (double)R.games / sp_sec : 0.0,
               sp_sec > 0 ? (double)R.evals / sp_sec : 0.0);
    }

    /* --------------------------------------------------------- teardown */
    sh.job = JOB_EXIT;
    bar_wait(&sh.bar);
#ifdef BENCH_PROBE
    memcpy(sh.workers[0].probe_ns,  bp_ns,  sizeof bp_ns);
    memcpy(sh.workers[0].probe_cnt, bp_cnt, sizeof bp_cnt);
#endif
    for (int i = 1; i < c.threads; i++) pthread_join(sh.workers[i].th, NULL);

#ifdef BENCH_PROBE
    print_probe(&sh, (c.mode == MODE_LEARN) ? ln_sec : sp_sec + ln_sec, clk_ns);
#endif

    for (int i = 0; i < c.threads; i++) {
        mcts_free(&sh.workers[i].m);
        brec_free(&sh.workers[i].rec);
        free(sh.workers[i].tg);
    }
    for (int a = 0; a < c.n_agents; a++) adam_free(&sh.head_adam[a]);
    adam_free(&sh.trunk_adam);
    bar_destroy(&sh.bar);
    bbuf_free(&sh.buf);
    free(sh.workers); free(sh.hgrad); free(sh.head_adam); free(sh.tgsum);
    free(sh.batch); free(sh.batch_tmp); free(sh.acount); free(sh.aoff);
    free(trunk); free(heads);
    return 0;
}
