/* az.c -- AlphaZero-style training: MCTS self-play, a replay buffer, and many
 * minibatch gradient steps per generation.
 *
 * ZERO CHESS KNOWLEDGE.  Grep this file for a piece value, a piece-square
 * table, a mobility term, MVV-LVA, SEE, a killer move, an opening book or a
 * shaping potential and you will find none.  The three inputs to learning are:
 *
 *   1. the rules of chess (chess.c)  -- which moves are legal, and whether a
 *      position is mate, stalemate, a repetition, a fifty-move draw or dead.
 *   2. the search (mcts.c)           -- a domain-independent PUCT bandit whose
 *      only evaluations are the network's policy and value heads.
 *   3. the game result                -- +1 / -1 / draw_penalty.  Nothing else.
 *
 * In particular Hyper.shaping is forced to 0 for every agent in this trainer
 * (see az_hyper_init): the A2C potential shaping it used to scale injected
 * piece values into the reward and is explicitly removed by
 * docs/FROM_SCRATCH.md.  material_balance() is never called from this file.
 *
 * ----------------------------------------------------------------- the loop
 *
 *   for g in 1..G:
 *       PAIR     Elo-adjacent pairings, hof_frac_pct routed to the hall of fame
 *       PLAY     threads play them with MCTS + root noise, recording
 *                (features, visit-count policy target, mover) per ply
 *       LABEL    every recorded ply gets z from THAT ply's mover's view
 *       LEARN    steps_per_gen minibatches of batch_size sampled uniformly
 *                from the replay buffer:
 *                  L = -sum_a pi_mcts(a) log p_net(a) + value_coef (v - z)^2
 *       RATE     Elo, K=24, hall-of-fame ratings frozen
 *       EVOLVE   cull the bottom, clone+mutate the elite
 *       LOG      one JSON line, checkpoint
 *
 * ------------------------------------------------------------------ threads
 *
 * One persistent pool of `threads` workers (the caller's thread is worker 0)
 * driven by an epoch barrier, so the per-generation cost of thread management
 * is two condvar broadcasts rather than `threads` pthread_create calls.
 *
 * SELF-PLAY is work-stolen from one atomic counter over the pairing list.  The
 * hot path allocates nothing: the MCTS node pool, the Game, the per-game record
 * scratch and every move/visit/target array are preallocated per worker.  One
 * mutex is taken per GAME (not per position) to splice the game's positions
 * into the shared replay ring.
 *
 * LEARNING is data-parallel over the minibatch.  The batch is counting-sorted
 * by agent and split on agent boundaries, so each agent's head gradient is
 * touched by exactly one thread -- no locks, no per-thread head-gradient
 * reduction.  Only the shared trunk gradient is reduced, once per step.
 */
#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <math.h>
#include <pthread.h>
#include <signal.h>
#include <stdatomic.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#include "az.h"
#include "chess.h"
#include "mcts.h"
#include "net.h"

/* ------------------------------------------------------------- constants */

#define ELO_K            24.0f
#define ELO_SEED         1500.0f
#define HOF_CAP          64
#define SHUFFLE_WINDOW   8      /* pair within +/- this many Elo ranks       */
#define SAVE_EVERY       5      /* checkpoint cadence, in generations        */
#define RESIGN_CONSEC    4      /* own moves below threshold before resigning */
/* Node-pool budget per simulation.  Branching is ~31 in practice, so this is
 * roughly 2x headroom; mcts.c degrades gracefully if it is ever exceeded. */
#define NODES_PER_SIM    72
/* Moves stored per recorded position, worst case, in the per-game scratch. */
#define REC_MAX_MOVES    MAX_MOVES
/* Calibration buckets for the value head: predictions are split evenly over
 * [-1,1] and the ACTUAL outcome rate is reported per bucket.  A head that has
 * collapsed to the mean occupies one bucket; a calibrated head has a
 * per-bucket actual rate that tracks the per-bucket prediction. */
#define VAL_BUCKETS      8
/* Which minibatch of a generation carries the extra instrumentation: the
 * policy/value gradient decomposition and the EMA-vs-raw comparison.  One step
 * in steps_per_gen, so the cost is under 2% of the learning time. */
#define PROBE_STEP       0

_Static_assert(sizeof(Hyper) == 8 * sizeof(float), "Hyper must be 8 packed floats");

static volatile sig_atomic_t g_az_interrupt = 0;

static void az_on_sigint(int sig) { (void)sig; g_az_interrupt = 1; }

/* ------------------------------------------------------------------ misc */

static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static int cpu_count(void)
{
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    if (n < 1) n = 1;
    if (n > 256) n = 256;
    return (int)n;
}

static int mkdir_p(const char *path)
{
    char tmp[1024];
    size_t len;

    snprintf(tmp, sizeof tmp, "%s", path);
    len = strlen(tmp);
    while (len > 1 && tmp[len - 1] == '/') tmp[--len] = '\0';
    for (char *p = tmp + 1; *p; p++) {
        if (*p != '/') continue;
        *p = '\0';
        if (mkdir(tmp, 0777) != 0 && errno != EEXIST) return -1;
        *p = '/';
    }
    if (mkdir(tmp, 0777) != 0 && errno != EEXIST) return -1;
    return 0;
}

static void fmt_dur(double s, char *buf, size_t n)
{
    long t;
    if (!(s > 0.0)) s = 0.0;
    if (s > 3.0e7) s = 3.0e7;
    t = (long)(s + 0.5);
    if (t >= 3600)    snprintf(buf, n, "%ldh%02ldm", t / 3600, (t % 3600) / 60);
    else if (t >= 60) snprintf(buf, n, "%ldm%02lds", t / 60, t % 60);
    else              snprintf(buf, n, "%lds", t);
}

/* ------------------------------------------------------------------- rng */
/* xoshiro256**, private to this file so az.c depends only on chess.c, net.c
 * and mcts.c.  Bit-identical to the generators in arena.c and mcts.c. */

static inline uint64_t az_rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }

static uint64_t az_next(uint64_t *s)
{
    const uint64_t r = az_rotl(s[1] * 5, 7) * 9;
    const uint64_t t = s[1] << 17;
    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];
    s[2] ^= t;
    s[3] = az_rotl(s[3], 45);
    return r;
}

static void az_seed(uint64_t *s, uint64_t seed)
{
    uint64_t z = seed + 0x9E3779B97F4A7C15ull;
    for (int i = 0; i < 4; i++) {
        z += 0x9E3779B97F4A7C15ull;
        uint64_t x = z;
        x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
        x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
        s[i] = x ^ (x >> 31);
    }
    for (int i = 0; i < 16; i++) az_next(s);
}

static float az_u01(uint64_t *s)          /* [0,1) */
{
    return (float)(az_next(s) >> 40) * (1.0f / 16777216.0f);
}

static float az_normal(uint64_t *s)
{
    float u, v, r;
    do {
        u = 2.0f * az_u01(s) - 1.0f;
        v = 2.0f * az_u01(s) - 1.0f;
        r = u * u + v * v;
    } while (r >= 1.0f || r == 0.0f);
    return u * sqrtf(-2.0f * logf(r) / r);
}

/* --------------------------------------------------------------- barrier */
/* macOS has no pthread_barrier_t, so here is the three-line version.  Every
 * thread traverses exactly the same sequence of waits, which is what makes one
 * barrier object safe to reuse for both the job dispatch and the inner
 * per-minibatch synchronisation. */

typedef struct {
    pthread_mutex_t mu;
    pthread_cond_t  cv;
    int             count, total;
    uint64_t        epoch;
} AZBar;

static void bar_init(AZBar *b, int total)
{
    pthread_mutex_init(&b->mu, NULL);
    pthread_cond_init(&b->cv, NULL);
    b->count = 0;
    b->total = total;
    b->epoch = 0;
}

static void bar_destroy(AZBar *b)
{
    pthread_mutex_destroy(&b->mu);
    pthread_cond_destroy(&b->cv);
}

static void bar_wait(AZBar *b)
{
    if (b->total <= 1) return;
    pthread_mutex_lock(&b->mu);
    const uint64_t e = b->epoch;
    if (++b->count == b->total) {
        b->count = 0;
        b->epoch++;
        pthread_cond_broadcast(&b->cv);
    } else {
        while (e == b->epoch) pthread_cond_wait(&b->cv, &b->mu);
    }
    pthread_mutex_unlock(&b->mu);
}

/* --------------------------------------------------------------- statistics */
/* Deliberately NOT arena.c's PlayStats: that struct carries
 * sum_final_material, and no material quantity may appear anywhere near this
 * learning path, not even as telemetry. */

typedef struct {
    uint64_t games, plies;
    uint64_t white_wins, black_wins, draws;
    uint64_t checkmates, stalemates, fifty, repetition, insufficient, maxplies, resigns;
    uint64_t captures, checks, castles, promotions, ep_captures;
    uint64_t first_move[64 * 64];
    double   sum_len;
} AZStats;

static void st_zero(AZStats *s) { memset(s, 0, sizeof *s); }

static void st_merge(AZStats *d, const AZStats *s)
{
    d->games        += s->games;        d->plies        += s->plies;
    d->white_wins   += s->white_wins;   d->black_wins   += s->black_wins;
    d->draws        += s->draws;
    d->checkmates   += s->checkmates;   d->stalemates   += s->stalemates;
    d->fifty        += s->fifty;        d->repetition   += s->repetition;
    d->insufficient += s->insufficient; d->maxplies     += s->maxplies;
    d->resigns      += s->resigns;
    d->captures     += s->captures;     d->checks       += s->checks;
    d->castles      += s->castles;      d->promotions   += s->promotions;
    d->ep_captures  += s->ep_captures;
    d->sum_len      += s->sum_len;
    for (int i = 0; i < 64 * 64; i++) d->first_move[i] += s->first_move[i];
}

/* ---------------------------------------------------------- replay buffer */
/* A fixed-capacity ring of positions.  The variable-length parts -- the move
 * keys and the MCTS policy target over those moves -- live in two parallel
 * pools which are themselves rings.
 *
 * A pool region is addressed by its MONOTONIC allocation offset `koff`, never
 * by a wrapped index, so a slot can tell whether its own moves have since been
 * overwritten:  valid  <=>  khead - koff <= kcap.  Allocation pads to the top
 * of the pool rather than splitting a game across the wrap, so every region is
 * contiguous.  Sampling retries on the rare invalid slot. */

typedef struct {
    uint16_t fidx[NF_MAXACTIVE];   /* active input features                  */
    uint64_t koff;                 /* monotonic offset into the move pools   */
    uint16_t nmoves;
    uint8_t  nf;
    uint8_t  mover;                /* WHITE / BLACK -- telemetry only        */
    int16_t  agent;                /* live agent that generated the position */
    float    z;                    /* game result from the mover's view      */
    float    q;                    /* MCTS root value AT this position, same
                                    * view.  The search's own estimate, kept
                                    * so the value target can be mixed
                                    * (see AZCfg.value_mix) without replaying
                                    * self-play.                             */
} AZPos;

typedef struct {
    AZPos    *pos;
    MoveKey  *keys;
    float    *pol;
    uint64_t  cap, kcap;
    uint64_t  head, khead;         /* monotonic write cursors                */
    pthread_mutex_t mu;
    int       ready;
} AZBuf;

/* Average legal-move count is ~31, so 36 slots per position keeps the pool
 * wrapping slightly slower than the position ring and makes an invalidated
 * sample rare. */
#define POOL_PER_POS  36u

static int azbuf_init(AZBuf *b, int positions)
{
    memset(b, 0, sizeof *b);
    if (positions < 1024) positions = 1024;
    b->cap  = (uint64_t)positions;
    b->kcap = b->cap * POOL_PER_POS;
    if (b->kcap < (uint64_t)MAX_MOVES * 4) b->kcap = (uint64_t)MAX_MOVES * 4;

    b->pos  = (AZPos *)  calloc((size_t)b->cap,  sizeof(AZPos));
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

static void azbuf_free(AZBuf *b)
{
    if (!b) return;
    if (b->ready) pthread_mutex_destroy(&b->mu);
    free(b->pos); free(b->keys); free(b->pol);
    memset(b, 0, sizeof *b);
}

static uint64_t azbuf_count(const AZBuf *b)
{
    return b->head < b->cap ? b->head : b->cap;
}

/* ------------------------------------------------------ per-game recorder */

typedef struct {
    AZPos   *pos;
    MoveKey *keys;
    float   *pol;
    int      n, npos_cap;
    uint64_t nk, nk_cap;
} AZRec;

static int rec_init(AZRec *r, int max_plies)
{
    memset(r, 0, sizeof *r);
    r->npos_cap = max_plies;
    r->nk_cap   = (uint64_t)max_plies * REC_MAX_MOVES;
    r->pos  = (AZPos *)  calloc((size_t)r->npos_cap, sizeof(AZPos));
    r->keys = (MoveKey *)calloc((size_t)r->nk_cap,   sizeof(MoveKey));
    r->pol  = (float *)  calloc((size_t)r->nk_cap,   sizeof(float));
    return r->pos && r->keys && r->pol;
}

static void rec_free(AZRec *r)
{
    free(r->pos); free(r->keys); free(r->pol);
    memset(r, 0, sizeof *r);
}

/* Splice one finished game into the ring.  One mutex acquisition per game;
 * the copies happen outside the lock because the reserved regions are
 * disjoint and nothing reads the buffer until every worker has joined. */
static void azbuf_commit(AZBuf *b, const AZRec *r)
{
    if (r->n <= 0 || r->nk == 0 || r->nk > b->kcap) return;

    uint64_t k0, p0;

    pthread_mutex_lock(&b->mu);
    const uint64_t off = b->khead % b->kcap;
    if (off + r->nk > b->kcap) b->khead += (b->kcap - off);   /* pad to the wrap */
    k0 = b->khead;  b->khead += r->nk;
    p0 = b->head;   b->head  += (uint64_t)r->n;
    pthread_mutex_unlock(&b->mu);

    const uint64_t kbase = k0 % b->kcap;
    memcpy(b->keys + kbase, r->keys, (size_t)r->nk * sizeof(MoveKey));
    memcpy(b->pol  + kbase, r->pol,  (size_t)r->nk * sizeof(float));

    for (int i = 0; i < r->n; i++) {
        AZPos p = r->pos[i];
        p.koff = k0 + p.koff;                    /* local offset -> monotonic */
        b->pos[(p0 + (uint64_t)i) % b->cap] = p;
    }
}

/* ------------------------------------------------------ hall of fame, pairs */

typedef struct {
    Head  head;
    Hyper hy;
    float elo;
    int   gen;
} HofEntry;

/* id >= 0: live agent index.  id < 0: hall-of-fame entry -(id + 1). */
typedef struct { int32_t white, black; } AZPair;
typedef struct { int16_t result, reason; } AZRes;

#define IS_HOF(id)   ((id) < 0)
#define HOF_IDX(id)  (-(id) - 1)

/* ------------------------------------------------------------ head tensors */
/* The layout table lives in net.h (NN_HEAD_TENSORS).  It used to be duplicated
 * here, which is exactly how a clone silently stops perturbing a tensor that
 * someone added to Head. */

/* Per-tensor N(0, sigma * rms(tensor)) perturbation. */
static void head_mutate(Head *h, float sigma, uint64_t *rng)
{
    float *base = (float *)h;
    for (int t = 0; t < NN_HEAD_NTENSORS; t++) {
        float *v = base + NN_HEAD_TENSORS[t].off;
        const size_t n = NN_HEAD_TENSORS[t].len;
        double ss = 0.0;
        for (size_t i = 0; i < n; i++) ss += (double)v[i] * (double)v[i];
        const float rms = (float)sqrt(ss / (double)n);
        if (!(rms > 0.0f)) continue;
        const float s = sigma * rms;
        for (size_t i = 0; i < n; i++) v[i] += s * az_normal(rng);
    }
}

/* Only two of Hyper's eight fields mean anything to an AlphaZero trainer:
 * lr_scale (per-agent head learning rate) and mutate_sigma (clone noise).  The
 * A2C fields are pinned at neutral values and never read, and `shaping` is
 * pinned at ZERO because potential shaping from material is exactly what
 * docs/FROM_SCRATCH.md forbids. */
static float g_mutate_sigma0 = 0.005f;

static void az_hyper_init(Hyper *h)
{
    hyper_default(h);
    /* AUDIT-OK: this PINS the A2C material-shaping weight to zero, which is the
     * opposite of using it.  hyper_default() ships 0.5; az.c never reads the
     * field, and zeroing it means a model saved here cannot resurrect shaping
     * if some other trainer loads it. */
    h->shaping      = 0.0f;   /* AUDIT-OK: pinned off, never read */
    h->entropy_coef = 0.0f;
    h->temperature  = 1.0f;
    h->value_coef   = 1.0f;
    h->gamma        = 1.0f;
    h->lambda       = 1.0f;
    h->lr_scale     = 1.0f;
    h->mutate_sigma = g_mutate_sigma0;
}

/* Declared in az.c rather than az.h, which may not be modified.  main.c
 * forward-declares it for the --mutate-sigma flag. */
void az_set_mutate_sigma(float s)
{
    if (s >= 0.0f && s <= 0.5f) g_mutate_sigma0 = s;
}

static void az_hyper_mutate(Hyper *h, uint64_t *rng)
{
    h->lr_scale     *= expf(0.2f * az_normal(rng));
    h->mutate_sigma *= expf(0.2f * az_normal(rng));
    if (!(h->lr_scale > 0.10f))     h->lr_scale = 0.10f;
    if (h->lr_scale > 4.00f)        h->lr_scale = 4.00f;
    if (!(h->mutate_sigma > 2e-4f)) h->mutate_sigma = 2e-4f;
    if (h->mutate_sigma > 0.20f)    h->mutate_sigma = 0.20f;
    /* AUDIT-OK: mutation must never be able to breed the shaping weight back
     * above zero.  Pinning it, not using it. */
    h->shaping = 0.0f;        /* AUDIT-OK: pinned off, never read */
}

/* ------------------------------------------------------------ run state */

enum { JOB_IDLE = 0, JOB_SELFPLAY, JOB_LEARN, JOB_EXIT };

typedef struct AZShared AZShared;

typedef struct {
    AZShared *sh;
    int       tid;
    uint64_t  rng[4];

    /* self-play */
    Mcts      m;
    Game      g;
    AZRec     rec;
    int32_t   visits[MAX_MOVES];
    float     target[MAX_MOVES];
    Move      list[MAX_MOVES];
    AZStats   st;
    uint64_t  evals;
    double    rootv_sum;
    uint64_t  rootv_n;
    double    tgt_ent_sum;
    uint64_t  tgt_ent_n;
    uint64_t  resign_checked, resign_would, resign_wrong;

    uint64_t  full_moves, all_moves; /* playout-cap randomisation accounting  */

    /* learning */
    TrunkGrad *tg;
    double     l_pol, l_val, l_ent;
    uint64_t   l_n, l_sign_ok;
    uint64_t   l_dec_n, l_dec_ok;   /* decisive positions only (|z| == 1)     */
    double     l_tent;              /* entropy of the MCTS target, same batch  */
    uint64_t   l_top1;              /* argmax(p_net) == argmax(pi_mcts)        */
    double     l_zsum, l_zsq;       /* for the constant-predictor baseline     */
    double     l_osum, l_osq;       /* same, for the PURE outcome target       */
    double     l_vout;              /* (v - z)^2 against the pure outcome      */
    double     l_qout;              /* (q_search - z)^2, the same way          */
    double     v_sum, v_sq;         /* mean and sd of the PREDICTION           */
    double     cal_pred[VAL_BUCKETS], cal_out[VAL_BUCKETS];
    uint64_t   cal_n[VAL_BUCKETS];
    double     hgnorm_sum;
    uint64_t   hgnorm_n, hclip_n;

    /* instrumentation, PROBE_STEP only */
    int        probe;
    TrunkGrad *tg_pol, *tg_val;     /* trunk gradient from ONE loss term each  */
    HeadGrad  *hg_probe;            /* sink; the head half is not reported     */
    uint64_t   ema_n, ema_top1;
    double     ema_vdiff;
} AZWorker;

struct AZShared {
    const AZCfg *cfg;
    Trunk    *trunk;
    Head     *heads;
    Hyper    *hypers;
    float    *elo;
    HofEntry *hof;
    int       hof_n, hof_next;

    AZPair   *pairs;
    AZRes    *results;
    int       npairs;
    atomic_int next;

    AZBuf     buf;
    int       generation;
    int       nthreads;
    int       job;
    AZBar     bar;
    AZWorker *workers;

    /* learning */
    HeadGrad  *hgrad;
    TrunkGrad *tgsum;
    Adam       trunk_adam;
    Adam      *head_adam;
    int32_t   *batch, *batch_tmp;
    int       *acount, *aoff;
    int       *tstart;
    float      lr_now;
    double     gnorm_sum;
    uint64_t   gnorm_n, gclip_n;
    int        probe_nb;            /* positions in the PROBE_STEP minibatch   */

    /* Polyak-averaged copy of every weight.  NULL when --ema-decay <= 0. */
    Trunk     *ema_trunk;
    Head      *ema_heads;
    float      head_clip;           /* effective per-head clip                 */
    int        cap_sims;            /* effective fast-search budget            */

    /* Set by the learning-rate range test only.  A JOB_LEARN dispatch then
     * takes EXACTLY ONE optimiser step instead of cfg->steps_per_gen, because
     * the sweep has to change lr_now between steps, and it skips the
     * PROBE_STEP instrumentation, which would otherwise fire on every step
     * (PROBE_STEP is 0) and triple the backward cost for a diagnostic the
     * sweep does not read.  Zero for az_run, which is therefore unchanged. */
    int        one_step;
};

/* e += (1 - d) * (p - e).  Polyak averaging, in place. */
static void ema_blend(float *e, const float *p, float d, size_t n)
{
    const float k = 1.0f - d;
    for (size_t i = 0; i < n; i++) e[i] += k * (p[i] - e[i]);
}

static void resolve_side(const AZShared *sh, int32_t id, const Head **h)
{
    *h = IS_HOF(id) ? &sh->hof[HOF_IDX(id)].head : &sh->heads[id];
}

/* ==========================================================================
 *                                SELF-PLAY
 * ========================================================================== */

static double target_entropy(const float *p, int n)
{
    double h = 0.0;
    for (int i = 0; i < n; i++)
        if (p[i] > 0.0f) h -= (double)p[i] * log((double)p[i]);
    return h;
}

/* Plays one pairing to completion, records every ply made by a LIVE agent, and
 * splices the labelled positions into the replay buffer. */
static void az_play_one(AZWorker *w, int idx)
{
    AZShared *sh = w->sh;
    const AZCfg *c = sh->cfg;
    const AZPair pr = sh->pairs[idx];
    const Head *hw, *hb;

    resolve_side(sh, pr.white, &hw);
    resolve_side(sh, pr.black, &hb);

    Game *g = &w->g;
    AZRec *r = &w->rec;
    /* Pick the starting array.  AZ_START_MIXED plays classical one game in ten
     * so the classical opening stays represented in the replay buffer while the
     * other 959 arrays supply the generalisation pressure. */
    if (c->start_mode == AZ_START_960) {
        game_start960(g, pos_960_random(w->rng));
    } else if (c->start_mode == AZ_START_MIXED) {
        if (az_u01(w->rng) < 0.10f) game_start(g);
        else                       game_start960(g, pos_960_random(w->rng));
    } else {
        game_start(g);
    }
    r->n = 0;
    r->nk = 0;

    /* Resignation bookkeeping.  A resign_check game turns resignation OFF and
     * plays on, so the threshold can be scored against the true result. */
    const int resign_on = (c->resign_threshold > -1.0f);
    const int check_game = resign_on && (az_u01(w->rng) < c->resign_check_frac);
    const int resign_live = resign_on && !check_game;
    int consec[2] = { 0, 0 };
    int would_resign = -1;        /* first colour that hit the threshold      */
    int resigned = -1;

    int result, reason;

    for (;;) {
        if (g->result != GR_ONGOING) break;
        if (g->ply >= c->max_plies) {
            g->result = GR_DRAW;
            g->reason = TR_MAX_PLIES;
            break;
        }

        const int side = (int)g->pos.side;
        const int32_t id = (side == WHITE) ? pr.white : pr.black;
        const Head *h = (side == WHITE) ? hw : hb;

        /* PLAYOUT CAP RANDOMISATION (KataGo).  A "full" move gets the whole
         * simulation budget, root noise and a slot in the replay buffer; a
         * "fast" move gets cap_sims, no root noise, and is played but never
         * learned from.  Nothing here is chess-specific: it is a statement
         * about where simulation budget buys training signal, and it applies
         * unchanged to any game. */
        const int full  = !(c->cap_frac < 1.0f) || (az_u01(w->rng) < c->cap_frac);
        const int nsims = full ? c->sims : sh->cap_sims;

        float rootv = 0.0f;
        const int n = mcts_search(&w->m, sh->trunk, h, g, nsims, full,
                                  w->rng, w->visits, &rootv);
        if (n <= 0) break;                        /* the search saw game over */
        w->all_moves++;
        if (full) w->full_moves++;

        const int nl = gen_legal(&g->pos, w->list);
        if (nl != n) break;                       /* cannot happen            */

        mcts_target(w->visits, n, w->target);

        w->rootv_sum += (double)rootv;
        w->rootv_n++;
        w->tgt_ent_sum += target_entropy(w->target, n);
        w->tgt_ent_n++;

        /* ---- record: features, the visit-count policy target, the mover --- */
        if (full && !IS_HOF(id) && r->n < r->npos_cap &&
            r->nk + (uint64_t)n <= r->nk_cap && n <= REC_MAX_MOVES) {
            AZPos *p = &r->pos[r->n];
            p->nf     = (uint8_t)nn_features(&g->pos, p->fidx);
            p->koff   = r->nk;                    /* local; made monotonic on commit */
            p->nmoves = (uint16_t)n;
            p->mover  = (uint8_t)side;
            p->agent  = (int16_t)id;
            p->z      = 0.0f;
            p->q      = rootv;      /* the search's own estimate HERE */
            for (int i = 0; i < n; i++) {
                nn_move_key(&g->pos, w->list[i], &r->keys[r->nk + (uint64_t)i]);
                r->pol[r->nk + (uint64_t)i] = w->target[i];
            }
            r->nk += (uint64_t)n;
            r->n++;
        }

        /* ---- resignation -------------------------------------------------- */
        if (resign_on) {
            if (rootv < c->resign_threshold) {
                if (++consec[side] >= RESIGN_CONSEC && would_resign < 0)
                    would_resign = side;
            } else {
                consec[side] = 0;
            }
            if (resign_live && would_resign == side) { resigned = side; break; }
        }

        /* ---- pick and play ------------------------------------------------ */
        const float temp = (g->ply < c->opening_plies) ? c->temp_start : c->temp_end;
        const int pick = mcts_pick(w->visits, n, temp, w->rng);
        const Move mv = w->list[pick < 0 ? 0 : pick];

        {   /* move statistics: pure counting, no evaluation */
            const int fl = MV_FLAG(mv);
            if (MV_IS_CAPTURE(mv))   w->st.captures++;
            if (fl == MF_EP)         w->st.ep_captures++;
            if (MV_IS_PROMO(mv))     w->st.promotions++;
            if (fl == MF_KCASTLE || fl == MF_QCASTLE) w->st.castles++;
            if (g->ply == 0 && side == WHITE)
                w->st.first_move[MV_FROM(mv) * 64 + MV_TO(mv)]++;
        }

        game_push(g, mv);
        if (in_check(&g->pos, g->pos.side)) w->st.checks++;
    }

    /* ---- the result.  The ONLY reward. --------------------------------- */
    if (resigned >= 0) {
        result = (resigned == WHITE) ? GR_BLACK_WIN : GR_WHITE_WIN;
        reason = TR_RESIGN;
    } else {
        result = g->result;
        reason = g->reason;
        if (result == GR_ONGOING) { result = GR_DRAW; reason = TR_MAX_PLIES; }
    }

    /* Resign-threshold validation: would the latched resignation have thrown
     * away a game the resigner did not actually lose? */
    if (check_game) {
        w->resign_checked++;
        if (would_resign >= 0) {
            const int lost = (would_resign == WHITE) ? (result == GR_BLACK_WIN)
                                                     : (result == GR_WHITE_WIN);
            w->resign_would++;
            if (!lost) w->resign_wrong++;
        }
    }

    /* ---- label every recorded ply from THAT ply's mover's view ---------- */
    for (int i = 0; i < r->n; i++) {
        const int mover = (int)r->pos[i].mover;
        float z;
        if (result == GR_DRAW)            z = sh->cfg->draw_penalty;
        else if (result == GR_WHITE_WIN)  z = (mover == WHITE) ? 1.0f : -1.0f;
        else                              z = (mover == BLACK) ? 1.0f : -1.0f;
        r->pos[i].z = z;
    }
    azbuf_commit(&sh->buf, r);

    /* ---- stats ---------------------------------------------------------- */
    w->st.games++;
    w->st.plies += (uint64_t)g->ply;
    w->st.sum_len += (double)g->ply;
    if (result == GR_WHITE_WIN)      w->st.white_wins++;
    else if (result == GR_BLACK_WIN) w->st.black_wins++;
    else                             w->st.draws++;
    switch (reason) {
        case TR_CHECKMATE:    w->st.checkmates++;   break;
        case TR_STALEMATE:    w->st.stalemates++;   break;
        case TR_FIFTY:        w->st.fifty++;        break;
        case TR_REPETITION:   w->st.repetition++;   break;
        case TR_INSUFFICIENT: w->st.insufficient++; break;
        case TR_RESIGN:       w->st.resigns++;      break;
        default:              w->st.maxplies++;     break;
    }

    sh->results[idx].result = (int16_t)result;
    sh->results[idx].reason = (int16_t)reason;
}

static void az_selfplay_job(AZWorker *w)
{
    AZShared *sh = w->sh;
    w->m.evals = 0;
    for (;;) {
        const int i = atomic_fetch_add_explicit(&sh->next, 1, memory_order_relaxed);
        if (i >= sh->npairs) break;
        az_play_one(w, i);
    }
    w->evals += w->m.evals;
}

/* ==========================================================================
 *                                 LEARNING
 * ========================================================================== */

/* Samples `batch_size` positions uniformly from the buffer and counting-sorts
 * them by agent, so that each agent's head gradient belongs to exactly one
 * thread.  Called by worker 0 only, between barriers. */
static void az_make_batch(AZShared *sh, uint64_t *rng)
{
    const AZCfg *c = sh->cfg;
    AZBuf *b = &sh->buf;
    const int B = c->batch_size;
    const int n = c->n_agents;
    const uint64_t count = azbuf_count(b);

    int got = 0;
    if (count > 0) {
        const int tries = B * 8 + 64;
        for (int t = 0; t < tries && got < B; t++) {
            const uint64_t u = az_next(rng) % count;
            const uint64_t L = b->head - 1u - u;              /* logical index */
            const uint64_t slot = L % b->cap;
            const AZPos *p = &b->pos[slot];
            if (p->nmoves == 0 || p->agent < 0 || p->agent >= n) continue;
            if (b->khead - p->koff > b->kcap) continue;       /* moves overwritten */
            sh->batch_tmp[got++] = (int32_t)slot;
        }
    }

    memset(sh->acount, 0, (size_t)n * sizeof(int));
    for (int i = 0; i < got; i++)
        sh->acount[b->pos[sh->batch_tmp[i]].agent]++;

    /* aoff doubles as the running placement cursor: after this loop it holds
     * one-past-the-end of each agent's run, which is all the caller needs. */
    int acc = 0;
    for (int a = 0; a < n; a++) { sh->aoff[a] = acc; acc += sh->acount[a]; }
    for (int i = 0; i < got; i++) {
        const int a = b->pos[sh->batch_tmp[i]].agent;
        sh->batch[sh->aoff[a]++] = sh->batch_tmp[i];
    }

    /* thread ranges, split on agent boundaries so no agent spans two threads */
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

/* One position's exact gradient.
 *
 *   target      = (1 - value_mix) * z  +  value_mix * q_search
 *   L           = -sum_a pi(a) log p(a)  +  value_coef * (v - target)^2
 *   dL/dlogit_a = p(a) - pi(a)
 *   dL/dv       = 2 * value_coef * (v - target) [post-tanh; nn_backward applies
 *                                                the tanh derivative itself]
 *
 * z is the game result and q_search is the MCTS root value recorded at THIS
 * position.  Both come from self-play; neither looks at the position.
 */
static void az_position_grad(AZWorker *w, const AZPos *p, const MoveKey *keys,
                             const float *pi, HeadGrad *hg, const Head *h)
{
    AZShared *sh = w->sh;
    const int n = (int)p->nmoves;

    float logits[MAX_MOVES], prob[MAX_MOVES], dl[MAX_MOVES];
    Fwd fw;

    nn_eval(sh->trunk, h, p->fidx, (int)p->nf, &fw);
    nn_logits(h, &fw, keys, n, logits);
    softmax_t(logits, n, 1.0f, prob);

    double lpol = 0.0, lent = 0.0, ltent = 0.0;
    int best_net = 0, best_tgt = 0;
    for (int i = 0; i < n; i++) {
        const float q = prob[i];
        const double lq = log((double)(q > 1e-30f ? q : 1e-30f));
        if (pi[i] > 0.0f) {
            lpol  -= (double)pi[i] * lq;
            ltent -= (double)pi[i] * log((double)pi[i]);
        }
        if (q > 0.0f) lent -= (double)q * lq;
        if (q      > prob[best_net]) best_net = i;
        if (pi[i]  > pi[best_tgt])   best_tgt = i;
        dl[i] = q - pi[i];
    }

    const float mix = sh->cfg->value_mix;
    float tgt = (1.0f - mix) * p->z + mix * p->q;
    if (tgt >  1.0f) tgt =  1.0f;
    if (tgt < -1.0f) tgt = -1.0f;

    const float diff = fw.v - tgt;
    const float dout = fw.v - p->z;
    const float dv = 2.0f * sh->cfg->value_coef * diff;

    nn_backward(sh->trunk, h, &fw, p->fidx, (int)p->nf, keys, n, dl, dv,
                w->tg, hg);

    if (w->probe) {
        /* The gradient DECOMPOSITION: run the backward pass twice more, once
         * with the value term silenced and once with the policy term silenced,
         * so the trunk gradient each head actually contributes can be reported
         * rather than guessed at. */
        float dzero[MAX_MOVES];
        for (int i = 0; i < n; i++) dzero[i] = 0.0f;
        nn_backward(sh->trunk, h, &fw, p->fidx, (int)p->nf, keys, n,
                    dl, 0.0f, w->tg_pol, w->hg_probe);
        nn_backward(sh->trunk, h, &fw, p->fidx, (int)p->nf, keys, n,
                    dzero, dv, w->tg_val, w->hg_probe);

        if (sh->ema_trunk) {         /* how far the EMA has drifted from raw  */
            Fwd fe;
            float le[MAX_MOVES];
            int be = 0;
            nn_eval(sh->ema_trunk, &sh->ema_heads[p->agent], p->fidx,
                    (int)p->nf, &fe);
            nn_logits(&sh->ema_heads[p->agent], &fe, keys, n, le);
            for (int i = 1; i < n; i++) if (le[i] > le[be]) be = i;
            if (be == best_net) w->ema_top1++;
            w->ema_vdiff += fabs((double)fe.v - (double)fw.v);
            w->ema_n++;
        }
    }

    w->l_pol  += lpol;
    w->l_val  += (double)diff * (double)diff;
    w->l_vout += (double)dout * (double)dout;
    {   /* How good is q_search as a target in the first place?  Mixing it in
         * can only help if the SEARCH predicts the outcome better than the
         * bare value head does -- MSE(q,z) < MSE(v,z).  When it does not, the
         * mix is regressing the head towards its own output. */
        const double dq = (double)p->q - (double)p->z;
        w->l_qout += dq * dq;
    }
    w->l_ent  += lent;
    w->l_tent += ltent;
    w->l_zsum += (double)tgt;
    w->l_zsq  += (double)tgt * (double)tgt;
    w->l_osum += (double)p->z;
    w->l_osq  += (double)p->z * (double)p->z;
    w->v_sum  += (double)fw.v;
    w->v_sq   += (double)fw.v * (double)fw.v;
    {   /* calibration: bucket the PREDICTION, average the ACTUAL outcome */
        int b = (int)((double)(fw.v + 1.0f) * 0.5 * (double)VAL_BUCKETS);
        if (b < 0) b = 0;
        if (b >= VAL_BUCKETS) b = VAL_BUCKETS - 1;
        w->cal_pred[b] += (double)fw.v;
        w->cal_out[b]  += (double)p->z;
        w->cal_n[b]++;
    }
    if (best_net == best_tgt) w->l_top1++;
    w->l_n++;
    {   /* value_accuracy: does the sign of the prediction match the outcome?
         * Reported twice on purpose.  With a negative draw_penalty EVERY draw
         * has sign(z) = -1, so a value head stuck at ~0- scores near 100% on
         * the headline number; the decisive-only figure is the honest one. */
        const int sv = (fw.v > 0.0f) - (fw.v < 0.0f);
        const int sz = (p->z > 0.0f) - (p->z < 0.0f);
        if (sv == sz) w->l_sign_ok++;
        if (p->z >= 1.0f || p->z <= -1.0f) {
            w->l_dec_n++;
            if (sv == sz) w->l_dec_ok++;
        }
    }
}

static void az_learn_job(AZWorker *w)
{
    AZShared *sh = w->sh;
    const AZCfg *c = sh->cfg;
    AZBuf *b = &sh->buf;
    const int tid = w->tid;

    const float ema_d = c->ema_decay;
    const int   ema_on = (ema_d > 0.0f) && (ema_d < 1.0f) && sh->ema_trunk;

    const int nsteps = sh->one_step ? 1 : c->steps_per_gen;

    for (int step = 0; step < nsteps; step++) {
        w->probe = !sh->one_step && (step == PROBE_STEP);
        if (w->probe) {
            grad_zero(w->tg_pol,   (int)TRUNK_NPARAM);
            grad_zero(w->tg_val,   (int)TRUNK_NPARAM);
            grad_zero(w->hg_probe, (int)HEAD_NPARAM);
        }
        if (tid == 0) az_make_batch(sh, w->rng);
        bar_wait(&sh->bar);
        if (tid == 0 && w->probe) sh->probe_nb = sh->tstart[sh->nthreads];

        const int lo = sh->tstart[tid], hi = sh->tstart[tid + 1];
        grad_zero(w->tg, (int)TRUNK_NPARAM);

        for (int i = lo; i < hi; i++) {
            const AZPos *p = &b->pos[sh->batch[i]];
            const uint64_t kb = p->koff % b->kcap;
            az_position_grad(w, p, b->keys + kb, b->pol + kb,
                             &sh->hgrad[p->agent], &sh->heads[p->agent]);
        }

        /* Each agent in [lo,hi) is owned by this thread alone, so its head can
         * be stepped right here -- the Adam traffic parallelises too. */
        for (int i = lo; i < hi; ) {
            const int a = b->pos[sh->batch[i]].agent;
            int j = i;
            while (j < hi && b->pos[sh->batch[j]].agent == a) j++;
            const float s = 1.0f / (float)(j - i);
            float *hgp = (float *)&sh->hgrad[a];
            double ss = 0.0;
            for (size_t k = 0; k < HEAD_NPARAM; k++) {
                hgp[k] *= s;
                ss += (double)hgp[k] * (double)hgp[k];
            }
            const double hn = sqrt(ss);
            w->hgnorm_sum += hn;
            w->hgnorm_n++;
            if (sh->head_clip > 0.0f && hn > (double)sh->head_clip) w->hclip_n++;
            adam_step(&sh->head_adam[a], (float *)&sh->heads[a], hgp,
                      sh->lr_now * sh->hypers[a].lr_scale,
                      c->weight_decay, sh->head_clip);
            /* This agent's head belongs to this thread alone for the whole
             * step, so its EMA can be advanced here without a lock. */
            if (ema_on)
                ema_blend((float *)&sh->ema_heads[a], (const float *)&sh->heads[a],
                          ema_d, HEAD_NPARAM);
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
                double ss = 0.0;
                for (size_t k = 0; k < TRUNK_NPARAM; k++) {
                    gp[k] *= s;
                    ss += (double)gp[k] * (double)gp[k];
                }
                const double gn = sqrt(ss);
                sh->gnorm_sum += gn;          /* PRE-clip, always */
                sh->gnorm_n++;
                if (c->grad_clip > 0.0f && gn > (double)c->grad_clip) sh->gclip_n++;
                /* adam_step zeroes tgsum on the way out. */
                adam_step(&sh->trunk_adam, (float *)sh->trunk, gp,
                          sh->lr_now, c->weight_decay, c->grad_clip);
                if (ema_on)
                    ema_blend((float *)sh->ema_trunk, (const float *)sh->trunk,
                              ema_d, TRUNK_NPARAM);
            }
        }
    }
}

/* ==========================================================================
 *                              THREAD POOL
 * ========================================================================== */

static void *az_worker_main(void *arg)
{
    AZWorker *w = (AZWorker *)arg;
    AZShared *sh = w->sh;
    for (;;) {
        bar_wait(&sh->bar);
        if (sh->job == JOB_EXIT) break;
        if (sh->job == JOB_SELFPLAY) az_selfplay_job(w);
        else if (sh->job == JOB_LEARN) az_learn_job(w);
        bar_wait(&sh->bar);
    }
    return NULL;
}

static void az_dispatch(AZShared *sh, int job)
{
    sh->job = job;
    bar_wait(&sh->bar);
    if (job == JOB_SELFPLAY)   az_selfplay_job(&sh->workers[0]);
    else if (job == JOB_LEARN) az_learn_job(&sh->workers[0]);
    bar_wait(&sh->bar);
}

/* ==========================================================================
 *                            RAW vs EMA HEAD-TO-HEAD
 * ========================================================================== */
/* Polyak-averaged weights are usually, but not always, stronger than the
 * weights they average.  Rather than assume it, play the two against each
 * other before deciding which set best.crl gets.  Called between generations,
 * with every worker parked on the barrier, so worker 0's search scratch is
 * free to borrow.  Returns the EMA's score in [0,1], or -1 if not measured. */
static double az_ema_h2h(AZShared *sh, AZWorker *w, int best_i, int games,
                         uint64_t *rng)
{
    const AZCfg *c = sh->cfg;
    double score = 0.0;
    int played = 0;

    if (games <= 0 || !sh->ema_trunk) return -1.0;

    for (int gi = 0; gi < games; gi++) {
        const int ema_white = (gi & 1);
        Game *g = &w->g;

        if (c->start_mode == AZ_START_960) {
            game_start960(g, pos_960_random(rng));
        } else if (c->start_mode == AZ_START_MIXED) {
            if (az_u01(rng) < 0.10f) game_start(g);
            else                     game_start960(g, pos_960_random(rng));
        } else {
            game_start(g);
        }

        for (;;) {
            if (g->result != GR_ONGOING) break;
            if (g->ply >= c->max_plies) {
                g->result = GR_DRAW;
                g->reason = TR_MAX_PLIES;
                break;
            }
            {
                const int side = (int)g->pos.side;
                const int use_ema = ((side == WHITE) == (ema_white != 0));
                const Trunk *t = use_ema ? sh->ema_trunk : sh->trunk;
                const Head  *h = use_ema ? &sh->ema_heads[best_i]
                                         : &sh->heads[best_i];
                float rv = 0.0f;
                const int n = mcts_search(&w->m, t, h, g, c->sims, 0,
                                          rng, w->visits, &rv);
                int nl, pick;
                float temp;
                if (n <= 0) break;
                nl = gen_legal(&g->pos, w->list);
                if (nl != n) break;
                /* Sampled through the opening so the games differ, greedy
                 * afterwards so the comparison measures strength. */
                temp = (g->ply < c->opening_plies) ? c->temp_start : 0.0f;
                pick = mcts_pick(w->visits, n, temp, rng);
                game_push(g, w->list[pick < 0 ? 0 : pick]);
            }
        }
        {
            int res = g->result;
            if (res == GR_ONGOING) res = GR_DRAW;
            if (res == GR_DRAW) score += 0.5;
            else if ((res == GR_WHITE_WIN) == (ema_white != 0)) score += 1.0;
            played++;
        }
    }
    return played ? score / (double)played : -1.0;
}

/* ==========================================================================
 *                        PAIRING, ELO, EVOLUTION
 * ========================================================================== */

typedef struct { float elo; int idx; } EloRank;

static int cmp_rank_desc(const void *a, const void *b)
{
    const EloRank *x = (const EloRank *)a, *y = (const EloRank *)b;
    if (x->elo < y->elo) return 1;
    if (x->elo > y->elo) return -1;
    return (x->idx > y->idx) - (x->idx < y->idx);
}

static int cmp_float_asc(const void *a, const void *b)
{
    const float x = *(const float *)a, y = *(const float *)b;
    return (x > y) - (x < y);
}

static void window_shuffle(int32_t *v, int n, int win, uint64_t *rng)
{
    if (win < 1) win = 1;
    for (int t = 0; t < n; t++) {
        int lo = t - win, hi = t + win;
        if (lo < 0) lo = 0;
        if (hi > n - 1) hi = n - 1;
        const int u = lo + (int)(az_next(rng) % (uint64_t)(hi - lo + 1));
        const int32_t tmp = v[t]; v[t] = v[u]; v[u] = tmp;
    }
}

static int build_pairings(AZShared *sh, EloRank *rank, int32_t *wslot,
                          int32_t *bslot, uint64_t *rng)
{
    const AZCfg *c = sh->cfg;
    const int n = c->n_agents;
    int g = c->games_per_agent / 2;
    if (g < 1) g = 1;
    const int np = n * g;

    for (int i = 0; i < n; i++) { rank[i].elo = sh->elo[i]; rank[i].idx = i; }
    qsort(rank, (size_t)n, sizeof(EloRank), cmp_rank_desc);

    for (int r = 0; r < n; r++)
        for (int k = 0; k < g; k++) {
            wslot[r * g + k] = rank[r].idx;
            bslot[r * g + k] = rank[r].idx;
        }

    const int win = SHUFFLE_WINDOW * g;
    window_shuffle(wslot, np, win, rng);
    window_shuffle(bslot, np, win, rng);

    for (int t = 0; t < np; t++) {
        if (wslot[t] != bslot[t]) continue;
        for (int d = 1; d <= win; d++) {
            const int u = t + d < np ? t + d : t - d;
            if (u < 0 || u >= np) continue;
            if (wslot[u] != bslot[t] && wslot[t] != bslot[u]) {
                const int32_t tmp = bslot[t]; bslot[t] = bslot[u]; bslot[u] = tmp;
                break;
            }
        }
    }

    for (int t = 0; t < np; t++) {
        sh->pairs[t].white = wslot[t];
        sh->pairs[t].black = bslot[t];
    }

    if (sh->hof_n > 0 && c->hof_frac_pct > 0) {
        for (int t = 0; t < np; t++) {
            if ((int)(az_next(rng) % 100u) >= c->hof_frac_pct) continue;
            const int32_t hid = -(int32_t)(az_next(rng) % (uint64_t)sh->hof_n) - 1;
            if (az_next(rng) & 1u) sh->pairs[t].white = hid;
            else                   sh->pairs[t].black = hid;
        }
    }
    return np;
}

static void apply_elo(AZShared *sh)
{
    for (int i = 0; i < sh->npairs; i++) {
        const AZPair p = sh->pairs[i];
        const int res = sh->results[i].result;
        if (p.white == p.black) continue;
        if (res == GR_ONGOING) continue;

        const float sw = (res == GR_WHITE_WIN) ? 1.0f : (res == GR_BLACK_WIN) ? 0.0f : 0.5f;
        const float rw = IS_HOF(p.white) ? sh->hof[HOF_IDX(p.white)].elo : sh->elo[p.white];
        const float rb = IS_HOF(p.black) ? sh->hof[HOF_IDX(p.black)].elo : sh->elo[p.black];
        const float ew = 1.0f / (1.0f + powf(10.0f, (rb - rw) / 400.0f));

        if (!IS_HOF(p.white)) sh->elo[p.white] += ELO_K * (sw - ew);
        if (!IS_HOF(p.black)) sh->elo[p.black] += ELO_K * ((1.0f - sw) - (1.0f - ew));
    }
}

/* ==========================================================================
 *                               TELEMETRY
 * ========================================================================== */

typedef struct {
    double sec, selfplay_sec, learn_sec;
    double elo_best, elo_mean, elo_p10;
    int    best_i;
    uint64_t evals;
    double loss_pol, loss_val, loss_tot, policy_entropy, target_entropy;
    double value_acc, value_acc_decisive, head_grad_norm, grad_norm, lr;
    double policy_kl, batch_target_entropy;
    double policy_top1, value_mse_baseline;
    uint64_t steps, train_positions;
    double buffer_fill;
    double resign_fp;
    uint64_t resign_would, resign_checked;
    double rootv_mean;
    double evals_per_move;

    /* value head, properly instrumented */
    double value_mse, value_mse_ratio;
    double value_mse_outcome, value_mse_outcome_baseline, value_mse_outcome_ratio;
    double search_mse_outcome, search_mse_outcome_ratio;
    double value_pred_mean, value_pred_std;
    double value_target_mean, value_target_std;
    double cal_pred[VAL_BUCKETS], cal_out[VAL_BUCKETS];
    uint64_t cal_n[VAL_BUCKETS];

    /* optimisation */
    double grad_clip_frac, head_clip_frac;
    double grad_trunk_policy, grad_trunk_value, grad_value_share;
    int    in_warmup;

    /* EMA */
    double ema_agree_top1, ema_value_l1, ema_h2h;
    int    best_is_ema, ema_h2h_games;

    /* playout cap randomisation */
    double full_search_frac, recorded_per_game;
} AZGen;

static void jnum(FILE *f, double x)
{
    if (!isfinite(x)) { fputs("0", f); return; }
    fprintf(f, "%.6g", x);
}

static void write_telemetry(FILE *f, const AZShared *sh, const AZStats *st,
                            const AZGen *gs)
{
    const AZCfg *c = sh->cfg;
    const double games = (double)st->games;
    const double gd = games > 0.0 ? games : 1.0;

    fprintf(f, "{\"gen\":%d", sh->generation);
    fprintf(f, ",\"games\":%llu", (unsigned long long)st->games);
    fprintf(f, ",\"plies\":%llu", (unsigned long long)st->plies);
    fprintf(f, ",\"sec\":");           jnum(f, gs->sec);
    fprintf(f, ",\"selfplay_sec\":");  jnum(f, gs->selfplay_sec);
    fprintf(f, ",\"learn_sec\":");     jnum(f, gs->learn_sec);
    fprintf(f, ",\"gps\":");           jnum(f, games / (gs->sec > 0.0 ? gs->sec : 1.0));
    fprintf(f, ",\"elo_best\":");      jnum(f, gs->elo_best);
    fprintf(f, ",\"elo_mean\":");      jnum(f, gs->elo_mean);
    fprintf(f, ",\"elo_p10\":");       jnum(f, gs->elo_p10);
    fprintf(f, ",\"white_win\":");     jnum(f, (double)st->white_wins / gd);
    fprintf(f, ",\"black_win\":");     jnum(f, (double)st->black_wins / gd);
    fprintf(f, ",\"draw\":");          jnum(f, (double)st->draws / gd);
    fprintf(f, ",\"avg_len\":");       jnum(f, st->sum_len / gd);
    fprintf(f, ",\"captures_per_game\":"); jnum(f, (double)st->captures / gd);
    fprintf(f, ",\"checks_per_game\":");   jnum(f, (double)st->checks / gd);
    fprintf(f, ",\"castle_rate\":");       jnum(f, (double)st->castles / gd);
    fprintf(f, ",\"promo_rate\":");        jnum(f, (double)st->promotions / gd);
    fprintf(f, ",\"ep_rate\":");           jnum(f, (double)st->ep_captures / gd);

    fprintf(f, ",\"term\":{\"checkmate\":"); jnum(f, (double)st->checkmates / gd);
    fprintf(f, ",\"stalemate\":");           jnum(f, (double)st->stalemates / gd);
    fprintf(f, ",\"fifty\":");               jnum(f, (double)st->fifty / gd);
    fprintf(f, ",\"repetition\":");          jnum(f, (double)st->repetition / gd);
    fprintf(f, ",\"insufficient\":");        jnum(f, (double)st->insufficient / gd);
    fprintf(f, ",\"maxplies\":");            jnum(f, (double)st->maxplies / gd);
    fprintf(f, ",\"resign\":");              jnum(f, (double)st->resigns / gd);
    fputc('}', f);

    {   /* top-8 White first moves */
        int top[8], ntop = 0;
        for (int i = 0; i < 64 * 64; i++) {
            const uint64_t cnt = st->first_move[i];
            if (cnt == 0) continue;
            int pos = ntop;
            while (pos > 0 && st->first_move[top[pos - 1]] < cnt) pos--;
            if (pos >= 8) continue;
            if (ntop < 8) ntop++;
            for (int q = ntop - 1; q > pos; q--) top[q] = top[q - 1];
            top[pos] = i;
        }
        fprintf(f, ",\"opening_top\":[");
        for (int i = 0; i < ntop; i++) {
            const int from = top[i] >> 6, to = top[i] & 63;
            fprintf(f, "%s{\"move\":\"%s%s\",\"n\":%llu}", i ? "," : "",
                    SQ_NAMES[from], SQ_NAMES[to],
                    (unsigned long long)st->first_move[top[i]]);
        }
        fputc(']', f);
    }

    /* ---- the AlphaZero-specific half ----------------------------------- */
    fprintf(f, ",\"sims_per_move\":%d", c->sims);
    fprintf(f, ",\"evals_per_move\":");  jnum(f, gs->evals_per_move);
    fprintf(f, ",\"evals\":%llu", (unsigned long long)gs->evals);
    fprintf(f, ",\"evals_per_sec\":");
    jnum(f, (double)gs->evals / (gs->selfplay_sec > 0.0 ? gs->selfplay_sec : 1.0));
    fprintf(f, ",\"buffer_fill\":");     jnum(f, gs->buffer_fill);
    fprintf(f, ",\"buffer_positions\":%llu",
            (unsigned long long)azbuf_count(&sh->buf));
    fprintf(f, ",\"steps\":%llu", (unsigned long long)gs->steps);
    fprintf(f, ",\"train_positions\":%llu", (unsigned long long)gs->train_positions);
    fprintf(f, ",\"policy_kl\":");          jnum(f, gs->policy_kl);
    fprintf(f, ",\"policy_top1\":");        jnum(f, gs->policy_top1);
    fprintf(f, ",\"value_mse_baseline\":"); jnum(f, gs->value_mse_baseline);
    fprintf(f, ",\"batch_target_entropy\":"); jnum(f, gs->batch_target_entropy);
    fprintf(f, ",\"loss\":{\"policy\":"); jnum(f, gs->loss_pol);
    fprintf(f, ",\"value\":");            jnum(f, gs->loss_val);
    fprintf(f, ",\"total\":");            jnum(f, gs->loss_tot);
    fputc('}', f);
    fprintf(f, ",\"grad_norm\":");       jnum(f, gs->grad_norm);
    fprintf(f, ",\"lr\":");              jnum(f, gs->lr);
    fprintf(f, ",\"resign_false_positive_rate\":"); jnum(f, gs->resign_fp);
    fprintf(f, ",\"resign_checks\":%llu", (unsigned long long)gs->resign_checked);
    fprintf(f, ",\"resign_would\":%llu",  (unsigned long long)gs->resign_would);
    fprintf(f, ",\"mcts_root_value_mean\":"); jnum(f, gs->rootv_mean);
    fprintf(f, ",\"policy_entropy\":");       jnum(f, gs->policy_entropy);
    fprintf(f, ",\"target_entropy\":");       jnum(f, gs->target_entropy);
    fprintf(f, ",\"value_accuracy\":");       jnum(f, gs->value_acc);
    fprintf(f, ",\"value_accuracy_decisive\":"); jnum(f, gs->value_acc_decisive);
    fprintf(f, ",\"head_grad_norm\":");      jnum(f, gs->head_grad_norm);

    /* ---- value head: what actually went wrong last time ----------------
     * value_accuracy_decisive above is a sign test, and a head that has
     * collapsed onto the mean still scores ~50% on it.  These are the
     * numbers that show the collapse: the ratio of the value MSE to the
     * MSE of simply predicting the mean went 0.345 -> 0.838 over the
     * 250-generation run while the sign test barely moved.               */
    fprintf(f, ",\"value_mse\":");            jnum(f, gs->value_mse);
    fprintf(f, ",\"value_mse_ratio\":");      jnum(f, gs->value_mse_ratio);
    fprintf(f, ",\"value_mse_outcome\":");    jnum(f, gs->value_mse_outcome);
    fprintf(f, ",\"value_mse_outcome_baseline\":");
    jnum(f, gs->value_mse_outcome_baseline);
    fprintf(f, ",\"value_mse_outcome_ratio\":");
    jnum(f, gs->value_mse_outcome_ratio);
    /* The ceiling on value_mix: q_search is only worth mixing in while the
     * SEARCH beats the bare head at predicting the result. */
    fprintf(f, ",\"search_mse_outcome\":");   jnum(f, gs->search_mse_outcome);
    fprintf(f, ",\"search_mse_outcome_ratio\":");
    jnum(f, gs->search_mse_outcome_ratio);
    fprintf(f, ",\"value_pred_mean\":");      jnum(f, gs->value_pred_mean);
    fprintf(f, ",\"value_pred_std\":");       jnum(f, gs->value_pred_std);
    fprintf(f, ",\"value_target_mean\":");    jnum(f, gs->value_target_mean);
    fprintf(f, ",\"value_target_std\":");     jnum(f, gs->value_target_std);
    fprintf(f, ",\"value_mix\":");            jnum(f, c->value_mix);
    fprintf(f, ",\"value_coef\":");           jnum(f, c->value_coef);
    fprintf(f, ",\"value_calibration\":[");
    for (int b = 0; b < VAL_BUCKETS; b++) {
        const double lo = -1.0 + 2.0 * (double)b / (double)VAL_BUCKETS;
        const double hi = lo + 2.0 / (double)VAL_BUCKETS;
        const double nb = gs->cal_n[b] ? (double)gs->cal_n[b] : 1.0;
        fprintf(f, "%s{\"lo\":", b ? "," : "");    jnum(f, lo);
        fprintf(f, ",\"hi\":");                   jnum(f, hi);
        fprintf(f, ",\"n\":%llu", (unsigned long long)gs->cal_n[b]);
        fprintf(f, ",\"pred\":");   jnum(f, gs->cal_n[b] ? gs->cal_pred[b] / nb : 0.0);
        fprintf(f, ",\"actual\":"); jnum(f, gs->cal_n[b] ? gs->cal_out[b]  / nb : 0.0);
        fputc('}', f);
    }
    fputc(']', f);

    /* ---- optimisation: is the update being throttled? ------------------ */
    fprintf(f, ",\"grad_norm_preclip\":");    jnum(f, gs->grad_norm);
    fprintf(f, ",\"grad_clip\":");            jnum(f, c->grad_clip);
    fprintf(f, ",\"grad_clip_frac\":");       jnum(f, gs->grad_clip_frac);
    fprintf(f, ",\"head_grad_norm_preclip\":"); jnum(f, gs->head_grad_norm);
    fprintf(f, ",\"head_grad_clip\":");       jnum(f, sh->head_clip);
    fprintf(f, ",\"head_clip_frac\":");       jnum(f, gs->head_clip_frac);
    fprintf(f, ",\"grad_trunk_policy\":");    jnum(f, gs->grad_trunk_policy);
    fprintf(f, ",\"grad_trunk_value\":");     jnum(f, gs->grad_trunk_value);
    fprintf(f, ",\"grad_value_share\":");     jnum(f, gs->grad_value_share);
    fprintf(f, ",\"warmup_gens\":%d", c->warmup_gens);
    fprintf(f, ",\"in_warmup\":%d", gs->in_warmup);

    /* ---- weight EMA ----------------------------------------------------- */
    fprintf(f, ",\"ema_decay\":");            jnum(f, c->ema_decay);
    fprintf(f, ",\"ema_agree_top1\":");       jnum(f, gs->ema_agree_top1);
    fprintf(f, ",\"ema_value_l1\":");         jnum(f, gs->ema_value_l1);
    fprintf(f, ",\"ema_h2h_games\":%d", gs->ema_h2h_games);
    fprintf(f, ",\"ema_h2h_score\":");        jnum(f, gs->ema_h2h);
    fprintf(f, ",\"best_is_ema\":%d", gs->best_is_ema);

    /* ---- playout cap randomisation -------------------------------------- */
    fprintf(f, ",\"cap_frac\":");             jnum(f, c->cap_frac);
    fprintf(f, ",\"cap_sims\":%d", sh->cap_sims);
    fprintf(f, ",\"full_search_frac\":");     jnum(f, gs->full_search_frac);
    fprintf(f, ",\"recorded_per_game\":");    jnum(f, gs->recorded_per_game);

    fprintf(f, ",\"draw_penalty\":");         jnum(f, c->draw_penalty);
    fprintf(f, ",\"best_agent\":{\"i\":%d,\"elo\":", gs->best_i);
    jnum(f, gs->elo_best);
    fprintf(f, ",\"lr_scale\":"); jnum(f, sh->hypers[gs->best_i].lr_scale);
    /* AUDIT-OK: published so a run's own telemetry proves material shaping was
     * off for every generation.  Reporting a zero, not computing one. */
    fprintf(f, ",\"material_shaping_weight\":");
    jnum(f, sh->hypers[gs->best_i].shaping);  /* AUDIT-OK: reports the zero */
    fputc('}', f);
    fputs("}\n", f);
    fflush(f);
}

/* ==========================================================================
 *                                DEFAULTS
 * ========================================================================== */

void az_default_cfg(AZCfg *c)
{
    if (!c) return;
    memset(c, 0, sizeof *c);

    c->start_mode       = AZ_START_MIXED;

    c->n_agents         = 32;
    c->generations      = 200;
    c->games_per_agent  = 4;
    c->threads          = cpu_count();

    c->sims             = 64;
    c->max_plies        = 200;
    c->opening_plies    = 20;
    c->temp_start       = 1.0f;
    c->temp_end         = 0.25f;
    c->c_puct           = 1.4f;
    c->dirichlet_alpha  = 0.3f;
    c->dirichlet_eps    = 0.25f;
    c->resign_threshold = -0.90f;
    c->resign_check_frac = 0.10f;

    c->draw_penalty     = -0.10f;

    /* MEASURED, not guessed.  The buffer's job is decorrelation, not volume:
     * expected samples per position over its lifetime is
     * steps_per_gen*batch_size/positions_per_gen, which does NOT depend on the
     * capacity -- capacity only sets how STALE the targets are.  At ~10k
     * positions/generation, 50k retains ~5 generations.  Raising it to 400k
     * (~40 generations) cost 18 points of policy top-1 agreement with the
     * search over a 30-generation run.  See the report. */
    c->buffer_positions = 50000;
    c->batch_size       = 256;
    c->steps_per_gen    = 200;
    c->lr               = 2e-3f;
    c->lr_final         = 2e-4f;
    c->weight_decay     = 1e-4f;
    c->grad_clip        = 4.0f;
    /* MEASURED.  At 1.0 the value head is out-gunned: it is 97 parameters
     * (Wv + bv) competing with ~11.8k policy parameters for the same shared
     * trunk.  Raising it to 4 improved BOTH heads over 40 generations -- value
     * R^2 0.54 -> 0.56 and, because a better value makes the search better and
     * therefore the policy targets better, policy top-1 agreement 55% -> 60%. */
    c->value_coef       = 4.0f;
    c->grad_clip_head   = 0.0f;   /* 0 = reuse grad_clip, as before */

    /* All three of the new techniques ship OFF, because "measure, do not
     * assume" cuts both ways: none of them earned a default on the evidence a
     * 90-second run can produce.  --value-mix, --warmup and --cap-frac turn
     * them on; the telemetry says what they did. */
    c->value_mix        = 0.0f;
    c->warmup_gens      = 0;
    c->cap_frac         = 1.0f;
    c->cap_sims         = 0;      /* 0 = derive as max(2, sims / 5) */

    /* The EMA is different: it costs one extra copy of the weights and cannot
     * make the live network worse, because it is never trained from.  It is on
     * by default, but it only takes over best.crl if it WINS a head-to-head. */
    c->ema_decay        = 0.999f;
    c->ema_h2h_games    = 8;

    c->elite_frac       = 0.25f;
    /* Cloning a culled agent's head discards what that head had learned, and
     * the buffer keeps feeding it targets its predecessor generated.  Measured
     * at 32 agents: cull 0.20 cost ~8 points of top-1 against cull 0.10, and
     * cull 0 was better still -- but the population mechanism is part of the
     * contract, so this is the gentlest setting that still evolves. */
    c->cull_frac        = 0.10f;
    c->hof_every        = 10;
    c->hof_frac_pct     = 15;

    c->seed             = 20260911u;
    c->run_dir          = "runs/az";
    c->quiet            = 0;
}

/* ==========================================================================
 *                                  RUN
 * ========================================================================== */

static void az_sanitise(AZCfg *c)
{
    if (c->n_agents < 2)          c->n_agents = 2;
    if (c->n_agents > 4096)       c->n_agents = 4096;
    if (c->generations < 1)       c->generations = 1;
    if (c->games_per_agent < 2)   c->games_per_agent = 2;
    c->games_per_agent &= ~1;                         /* half as White exactly */
    if (c->threads < 1)           c->threads = cpu_count();
    if (c->sims < 2)              c->sims = 2;
    if (c->max_plies < 4)         c->max_plies = 200;
    if (c->max_plies > MAX_GAME_PLIES) c->max_plies = MAX_GAME_PLIES;
    if (c->opening_plies < 0)     c->opening_plies = 0;
    if (!(c->temp_start >= 0.0f)) c->temp_start = 1.0f;
    if (!(c->temp_end >= 0.0f))   c->temp_end = 0.25f;
    if (!(c->c_puct > 0.0f))      c->c_puct = 1.4f;
    if (!(c->dirichlet_alpha > 0.0f)) c->dirichlet_alpha = 0.3f;
    if (!(c->dirichlet_eps >= 0.0f) || c->dirichlet_eps > 1.0f) c->dirichlet_eps = 0.25f;
    if (!(c->resign_check_frac >= 0.0f) || c->resign_check_frac > 1.0f)
        c->resign_check_frac = 0.10f;
    if (!(c->draw_penalty >= -1.0f) || c->draw_penalty > 1.0f) c->draw_penalty = 0.0f;
    if (c->buffer_positions < 1024)  c->buffer_positions = 1024;
    if (c->batch_size < 1)        c->batch_size = 256;
    if (c->steps_per_gen < 0)     c->steps_per_gen = 0;
    if (!(c->lr > 0.0f))          c->lr = 2e-3f;
    if (!(c->lr_final > 0.0f))    c->lr_final = c->lr * 0.1f;
    if (c->lr_final > c->lr)      c->lr_final = c->lr;
    if (!(c->weight_decay >= 0.0f)) c->weight_decay = 0.0f;
    if (!(c->value_coef >= 0.0f)) c->value_coef = 1.0f;
    if (!(c->grad_clip >= 0.0f))  c->grad_clip = 0.0f;
    if (!(c->grad_clip_head >= 0.0f)) c->grad_clip_head = 0.0f;
    if (!(c->value_mix >= 0.0f))  c->value_mix = 0.0f;
    if (c->value_mix > 1.0f)      c->value_mix = 1.0f;
    if (c->warmup_gens < 0)       c->warmup_gens = 0;
    if (c->warmup_gens > c->generations - 1) c->warmup_gens = c->generations - 1;
    if (c->warmup_gens < 0)       c->warmup_gens = 0;
    if (!(c->ema_decay >= 0.0f) || c->ema_decay >= 1.0f) c->ema_decay = 0.0f;
    if (c->ema_h2h_games < 0)     c->ema_h2h_games = 0;
    if (c->ema_h2h_games > 64)    c->ema_h2h_games = 64;
    if (!(c->cap_frac > 0.0f))    c->cap_frac = 1.0f;
    if (c->cap_frac > 1.0f)       c->cap_frac = 1.0f;
    if (c->cap_sims < 0)          c->cap_sims = 0;
    if (c->cap_sims > c->sims)    c->cap_sims = c->sims;
    if (!(c->elite_frac > 0.0f) || c->elite_frac > 1.0f) c->elite_frac = 0.25f;
    if (!(c->cull_frac >= 0.0f) || c->cull_frac > 0.9f)  c->cull_frac = 0.20f;
    if (c->hof_every < 1)         c->hof_every = 10;
    if (c->hof_frac_pct < 0)      c->hof_frac_pct = 0;
    if (c->hof_frac_pct > 100)    c->hof_frac_pct = 100;
    if (!c->run_dir || !*c->run_dir) c->run_dir = "runs/az";
}

int az_run(AZCfg *c)
{
    AZShared  sh;
    Trunk    *trunk = NULL;
    Head     *heads = NULL;
    Trunk    *ema_trunk = NULL;
    Head     *ema_heads = NULL;
    Hyper    *hypers = NULL;
    float    *elo = NULL, *elo_sorted = NULL;
    HofEntry *hof = NULL;
    AZPair   *pairs = NULL;
    AZRes    *results = NULL;
    EloRank  *rank = NULL;
    int32_t  *wslot = NULL, *bslot = NULL;
    HeadGrad *hgrad = NULL;
    TrunkGrad *tgsum = NULL;
    Adam     *head_adam = NULL;
    AZWorker *workers = NULL;
    pthread_t *tids = NULL;
    FILE     *tel = NULL;
    uint64_t  master[4], h2h_rng[4];
    char      path[1200];
    int       n, gpa, nthreads, npairs, rc = 1;
    int       adam_ready = 0, nadam = 0, nspawned = 0, bar_ready = 0;
    int       nworkers_init = 0;
    int       h2h_next = 1, ema_wins = 0;
    double    t_run0, ema_gen = 0.0, best_saved = -1e30;
    uint64_t  total_games = 0, total_plies = 0, total_evals = 0;
    void    (*old_sigint)(int) = SIG_DFL;

    if (!c) return 1;
    chess_init();
    az_sanitise(c);

    memset(&sh, 0, sizeof sh);

    n   = c->n_agents;
    gpa = c->games_per_agent;
    npairs = n * (gpa / 2);
    nthreads = c->threads;
    if (nthreads > npairs) nthreads = npairs;
    if (nthreads < 1) nthreads = 1;

    if (mkdir_p(c->run_dir) != 0) {
        fprintf(stderr, "az: cannot create run directory '%s': %s\n",
                c->run_dir, strerror(errno));
        return 1;
    }

    trunk      = (Trunk *)     calloc(1, sizeof(Trunk));
    heads      = (Head *)      calloc((size_t)n, sizeof(Head));
    if (c->ema_decay > 0.0f) {
        ema_trunk = (Trunk *)calloc(1, sizeof(Trunk));
        ema_heads = (Head *) calloc((size_t)n, sizeof(Head));
        if (!ema_trunk || !ema_heads) {
            fprintf(stderr, "az: out of memory for the EMA weights\n");
            free(ema_trunk); free(ema_heads);
            ema_trunk = NULL; ema_heads = NULL;
        }
    }
    hypers     = (Hyper *)     calloc((size_t)n, sizeof(Hyper));
    elo        = (float *)     calloc((size_t)n, sizeof(float));
    elo_sorted = (float *)     calloc((size_t)n, sizeof(float));
    hof        = (HofEntry *)  calloc((size_t)HOF_CAP, sizeof(HofEntry));
    pairs      = (AZPair *)    calloc((size_t)npairs, sizeof(AZPair));
    results    = (AZRes *)     calloc((size_t)npairs, sizeof(AZRes));
    rank       = (EloRank *)   calloc((size_t)n, sizeof(EloRank));
    wslot      = (int32_t *)   calloc((size_t)npairs, sizeof(int32_t));
    bslot      = (int32_t *)   calloc((size_t)npairs, sizeof(int32_t));
    hgrad      = (HeadGrad *)  calloc((size_t)n, sizeof(HeadGrad));
    tgsum      = (TrunkGrad *) calloc(1, sizeof(TrunkGrad));
    head_adam  = (Adam *)      calloc((size_t)n, sizeof(Adam));
    workers    = (AZWorker *)  calloc((size_t)nthreads, sizeof(AZWorker));
    tids       = (pthread_t *) calloc((size_t)nthreads, sizeof(pthread_t));

    sh.batch     = (int32_t *) calloc((size_t)c->batch_size, sizeof(int32_t));
    sh.batch_tmp = (int32_t *) calloc((size_t)c->batch_size, sizeof(int32_t));
    sh.acount    = (int *)     calloc((size_t)n, sizeof(int));
    sh.aoff      = (int *)     calloc((size_t)n, sizeof(int));
    sh.tstart    = (int *)     calloc((size_t)nthreads + 1, sizeof(int));

    if (!trunk || !heads || !hypers || !elo || !elo_sorted || !hof || !pairs ||
        !results || !rank || !wslot || !bslot || !hgrad || !tgsum ||
        !head_adam || !workers || !tids || !sh.batch || !sh.batch_tmp ||
        !sh.acount || !sh.aoff || !sh.tstart) {
        fprintf(stderr, "az: out of memory\n");
        goto done;
    }

    if (!azbuf_init(&sh.buf, c->buffer_positions)) {
        fprintf(stderr, "az: out of memory for a %d-position replay buffer\n",
                c->buffer_positions);
        goto done;
    }

    /* ---- population ----------------------------------------------------- */
    nn_init(trunk, &heads[0], c->seed);
    for (int i = 1; i < n; i++) {
        Trunk scratch;                             /* per-agent head init only */
        nn_init(&scratch, &heads[i], c->seed + 0x9E3779B97F4A7C15ull * (uint64_t)(i + 1));
    }
    az_seed(master, c->seed ^ 0xD1B54A32D192ED03ull);
    for (int i = 0; i < n; i++) {
        az_hyper_init(&hypers[i]);
        az_hyper_mutate(&hypers[i], master);
        elo[i] = ELO_SEED;
    }
    if (ema_trunk) {                       /* the average starts AT the start */
        *ema_trunk = *trunk;
        memcpy(ema_heads, heads, (size_t)n * sizeof(Head));
    }

    adam_init(&sh.trunk_adam, (int)TRUNK_NPARAM);
    for (nadam = 0; nadam < n; nadam++) adam_init(&head_adam[nadam], (int)HEAD_NPARAM);
    adam_ready = 1;

    sh.cfg       = c;
    sh.trunk     = trunk;
    sh.heads     = heads;
    sh.hypers    = hypers;
    sh.elo       = elo;
    sh.hof       = hof;
    sh.pairs     = pairs;
    sh.results   = results;
    sh.npairs    = npairs;
    sh.hgrad     = hgrad;
    sh.tgsum     = tgsum;
    sh.head_adam = head_adam;
    sh.workers   = workers;
    sh.nthreads  = nthreads;
    sh.ema_trunk = ema_trunk;
    sh.ema_heads = ema_heads;
    sh.head_clip = (c->grad_clip_head > 0.0f) ? c->grad_clip_head : c->grad_clip;
    sh.cap_sims  = c->cap_sims > 0 ? c->cap_sims : (c->sims / 5 > 2 ? c->sims / 5 : 2);
    if (sh.cap_sims > c->sims) sh.cap_sims = c->sims;
    sh.job       = JOB_IDLE;
    atomic_init(&sh.next, 0);

    bar_init(&sh.bar, nthreads);
    bar_ready = 1;

    /* ---- workers -------------------------------------------------------- */
    for (nworkers_init = 0; nworkers_init < nthreads; nworkers_init++) {
        AZWorker *w = &workers[nworkers_init];
        w->sh  = &sh;
        w->tid = nworkers_init;
        mcts_init(&w->m, MAX_MOVES + 1 + c->sims * NODES_PER_SIM);
        w->m.c_puct          = c->c_puct;
        w->m.dirichlet_alpha = c->dirichlet_alpha;
        w->m.dirichlet_eps   = c->dirichlet_eps;
        w->tg       = (TrunkGrad *)calloc(1, sizeof(TrunkGrad));
        w->tg_pol   = (TrunkGrad *)calloc(1, sizeof(TrunkGrad));
        w->tg_val   = (TrunkGrad *)calloc(1, sizeof(TrunkGrad));
        w->hg_probe = (HeadGrad *) calloc(1, sizeof(HeadGrad));
        if (!w->m.pool || !w->tg || !w->tg_pol || !w->tg_val || !w->hg_probe ||
            !rec_init(&w->rec, c->max_plies)) {
            fprintf(stderr, "az: out of memory (worker %d)\n", nworkers_init);
            nworkers_init++;
            goto done;
        }
    }

    snprintf(path, sizeof path, "%s/telemetry.jsonl", c->run_dir);
    tel = fopen(path, "a");
    if (!tel) {
        fprintf(stderr, "az: cannot open '%s': %s\n", path, strerror(errno));
        goto done;
    }

    for (nspawned = 1; nspawned < nthreads; nspawned++) {
        if (pthread_create(&tids[nspawned], NULL, az_worker_main, &workers[nspawned]) != 0) {
            fprintf(stderr, "az: cannot create thread %d\n", nspawned);
            goto done;
        }
    }

    old_sigint = signal(SIGINT, az_on_sigint);

    if (!c->quiet) {
        printf("chessrl az: %d agents, %d generations, %d games/agent "
               "(%d games/gen), %d threads\n",
               n, c->generations, gpa, npairs, nthreads);
        printf("            mcts %d sims, c_puct %.2f, dirichlet %.2f/%.2f, "
               "temp %.2f->%.2f after %d plies, max %d plies\n",
               c->sims, (double)c->c_puct, (double)c->dirichlet_alpha,
               (double)c->dirichlet_eps, (double)c->temp_start,
               (double)c->temp_end, c->opening_plies, c->max_plies);
        printf("            buffer %d, batch %d, %d steps/gen, lr %.3g->%.3g, "
               "wd %.3g, clip %.3g, value_coef %.2f, draw %.2f\n",
               c->buffer_positions, c->batch_size, c->steps_per_gen,
               (double)c->lr, (double)c->lr_final, (double)c->weight_decay,
               (double)c->grad_clip, (double)c->value_coef, (double)c->draw_penalty);
        printf("            value_mix %.2f, warmup %d gens, ema %.4f (h2h %d games), "
               "cap %.2f x %d sims, head clip %.3g\n",
               (double)c->value_mix, c->warmup_gens, (double)c->ema_decay,
               c->ema_h2h_games, (double)c->cap_frac, sh.cap_sims,
               (double)sh.head_clip);
        printf("            resign %.2f (%.0f%% checked), elite %.0f%%, cull %.0f%%, "
               "hof every %d (%d%% of games), seed %llu\n",
               (double)c->resign_threshold, (double)c->resign_check_frac * 100.0,
               (double)c->elite_frac * 100.0, (double)c->cull_frac * 100.0,
               c->hof_every, c->hof_frac_pct, (unsigned long long)c->seed);
        printf("            run dir %s\n", c->run_dir);
        fflush(stdout);
    }

    az_seed(h2h_rng, c->seed ^ 0x94D049BB133111EBull);
    /* Do not compare against the EMA until its averaging window has actually
     * filled.  Before ~1/(1-decay) optimiser steps the "average" is still
     * mostly the random initialisation, and MEASURED at 40 games it can beat
     * a network that has taken a hundred steps on a nearly-empty buffer -- a
     * true result about that moment, and a terrible reason to ship it as
     * best.crl. */
    if (c->ema_decay > 0.0f && c->steps_per_gen > 0) {
        const double window = 1.0 / (1.0 - (double)c->ema_decay);
        double g = window / (double)c->steps_per_gen;
        h2h_next = (int)(g + 0.999);
        if (h2h_next < 1) h2h_next = 1;
        if (h2h_next > c->generations) h2h_next = c->generations;
    }
    t_run0 = now_sec();

    /* ===================================================================== */
    for (int gen = 1; gen <= c->generations; gen++) {
        const double t0 = now_sec();
        AZStats st;
        AZGen   gs;

        memset(&gs, 0, sizeof gs);
        gs.ema_h2h = -1.0;          /* -1 = not measured this generation */
        st_zero(&st);
        sh.generation = gen;

        /* Linear warmup, then the cosine decay over what is left.  Adam's
         * second-moment estimate is worthless for its first few dozen steps,
         * so a full-size update there is the largest of the run applied with
         * the least information -- which is exactly when a shared trunk picks
         * up the scale it then has to be clipped away from.  With
         * warmup_gens = 0 this is bit-identical to the pure cosine. */
        {
            const int W = c->warmup_gens;
            if (W > 0 && gen <= W) {
                sh.lr_now = (float)((double)c->lr * (double)gen / (double)(W + 1));
                gs.in_warmup = 1;
            } else {
                const int span = c->generations - W;
                const double prog = (span > 1)
                                  ? (double)(gen - W - 1) / (double)(span - 1) : 1.0;
                sh.lr_now = (float)((double)c->lr_final +
                                    0.5 * ((double)c->lr - (double)c->lr_final) *
                                    (1.0 + cos(M_PI * prog)));
            }
        }

        /* -- 1. PAIR ----------------------------------------------------- */
        az_seed(master, c->seed + 0x2545F4914F6CDD1Dull * (uint64_t)gen);
        sh.npairs = build_pairings(&sh, rank, wslot, bslot, master);
        memset(results, 0, (size_t)sh.npairs * sizeof(AZRes));
        atomic_store(&sh.next, 0);

        /* -- 2. PLAY ----------------------------------------------------- */
        for (int t = 0; t < nthreads; t++) {
            AZWorker *w = &workers[t];
            az_seed(w->rng, c->seed
                    + 0x9E3779B97F4A7C15ull * (uint64_t)(t + 1)
                    + 0xBF58476D1CE4E5B9ull * (uint64_t)gen);
            st_zero(&w->st);
            w->evals = 0;
            w->rootv_sum = 0.0; w->rootv_n = 0;
            w->tgt_ent_sum = 0.0; w->tgt_ent_n = 0;
            w->resign_checked = w->resign_would = w->resign_wrong = 0;
            w->full_moves = w->all_moves = 0;
        }
        az_dispatch(&sh, JOB_SELFPLAY);
        gs.selfplay_sec = now_sec() - t0;

        {
            double rv = 0.0, te = 0.0;
            uint64_t rvn = 0, ten = 0, rchk = 0, rwld = 0, rwrong = 0;
            uint64_t fm = 0, am = 0;
            for (int t = 0; t < nthreads; t++) {
                st_merge(&st, &workers[t].st);
                gs.evals += workers[t].evals;
                rv  += workers[t].rootv_sum;  rvn += workers[t].rootv_n;
                te  += workers[t].tgt_ent_sum; ten += workers[t].tgt_ent_n;
                rchk   += workers[t].resign_checked;
                rwld   += workers[t].resign_would;
                rwrong += workers[t].resign_wrong;
                fm += workers[t].full_moves;
                am += workers[t].all_moves;
            }
            gs.full_search_frac  = am ? (double)fm / (double)am : 0.0;
            gs.recorded_per_game = st.games ? (double)fm / (double)st.games : 0.0;
            gs.rootv_mean     = rvn ? rv / (double)rvn : 0.0;
            gs.target_entropy = ten ? te / (double)ten : 0.0;
            gs.evals_per_move = rvn ? (double)gs.evals / (double)rvn : 0.0;
            gs.resign_checked = rchk;
            gs.resign_would   = rwld;
            gs.resign_fp      = rwld ? (double)rwrong / (double)rwld : 0.0;
        }

        /* -- 3. LEARN ---------------------------------------------------- */
        {
            const double tl0 = now_sec();
            for (int t = 0; t < nthreads; t++) {
                AZWorker *w = &workers[t];
                w->l_pol = w->l_val = w->l_ent = w->l_tent = 0.0;
                w->l_zsum = w->l_zsq = 0.0;
                w->l_osum = w->l_osq = 0.0;
                w->l_vout = w->l_qout = 0.0;
                w->v_sum = w->v_sq = 0.0;
                w->l_top1 = 0;
                w->l_n = w->l_sign_ok = 0;
                w->l_dec_n = w->l_dec_ok = 0;
                w->hgnorm_sum = 0.0;
                w->hgnorm_n = w->hclip_n = 0;
                w->ema_n = w->ema_top1 = 0;
                w->ema_vdiff = 0.0;
                w->probe = 0;
                memset(w->cal_pred, 0, sizeof w->cal_pred);
                memset(w->cal_out,  0, sizeof w->cal_out);
                memset(w->cal_n,    0, sizeof w->cal_n);
            }
            sh.gnorm_sum = 0.0;
            sh.gnorm_n = sh.gclip_n = 0;
            sh.probe_nb = 0;
            if (c->steps_per_gen > 0 && azbuf_count(&sh.buf) > 0)
                az_dispatch(&sh, JOB_LEARN);
            gs.learn_sec = now_sec() - tl0;
        }

        {
            double lp = 0.0, lv = 0.0, le = 0.0, lte = 0.0, hgn = 0.0;
            double zs = 0.0, zq = 0.0, os = 0.0, oq = 0.0, vo = 0.0, qo = 0.0;
            double vs = 0.0, vq = 0.0, emad = 0.0;
            uint64_t ln = 0, ok = 0, dn = 0, dok = 0, hgnn = 0, top1 = 0;
            uint64_t hclip = 0, eman = 0, ematop = 0;
            for (int t = 0; t < nthreads; t++) {
                lp += workers[t].l_pol;
                lv += workers[t].l_val;
                le += workers[t].l_ent;
                lte += workers[t].l_tent;
                ln += workers[t].l_n;
                ok += workers[t].l_sign_ok;
                dn += workers[t].l_dec_n;
                dok += workers[t].l_dec_ok;
                hgn += workers[t].hgnorm_sum;
                hgnn += workers[t].hgnorm_n;
                hclip += workers[t].hclip_n;
                zs += workers[t].l_zsum;
                zq += workers[t].l_zsq;
                os += workers[t].l_osum;
                oq += workers[t].l_osq;
                vo += workers[t].l_vout;
                qo += workers[t].l_qout;
                vs += workers[t].v_sum;
                vq += workers[t].v_sq;
                top1 += workers[t].l_top1;
                eman += workers[t].ema_n;
                ematop += workers[t].ema_top1;
                emad += workers[t].ema_vdiff;
                for (int b = 0; b < VAL_BUCKETS; b++) {
                    gs.cal_pred[b] += workers[t].cal_pred[b];
                    gs.cal_out[b]  += workers[t].cal_out[b];
                    gs.cal_n[b]    += workers[t].cal_n[b];
                }
            }
            gs.value_acc_decisive = dn ? (double)dok / (double)dn : 0.0;
            gs.head_grad_norm     = hgnn ? hgn / (double)hgnn : 0.0;
            gs.head_clip_frac     = hgnn ? (double)hclip / (double)hgnn : 0.0;
            gs.ema_agree_top1     = eman ? (double)ematop / (double)eman : 0.0;
            gs.ema_value_l1       = eman ? emad / (double)eman : 0.0;
            if (ln) {
                gs.loss_pol             = lp / (double)ln;
                gs.loss_val             = lv / (double)ln;
                gs.policy_entropy       = le / (double)ln;
                gs.batch_target_entropy = lte / (double)ln;
                gs.value_acc            = (double)ok / (double)ln;
                /* The cross-entropy against a MOVING target is not a learning
                 * curve: CE = H(pi) + KL(pi || p), and H(pi) drifts as the
                 * search sharpens or broadens.  The KL is the half the network
                 * actually controls, so it is the one to read. */
                gs.policy_kl = gs.loss_pol - gs.batch_target_entropy;
                /* Two controls, because a raw loss against a moving target and
                 * a growing buffer proves nothing on its own:
                 *  - policy_top1: how often the policy head alone picks the
                 *    move the SEARCH preferred.  Chance is ~1/35.
                 *  - value_mse_baseline: the MSE of simply predicting the mean
                 *    outcome.  loss.value must beat it or the value head has
                 *    learned nothing. */
                const double zm = zs / (double)ln;
                const double om = os / (double)ln;
                const double vm = vs / (double)ln;
                double ov;
                gs.policy_top1        = (double)top1 / (double)ln;
                gs.value_mse_baseline = zq / (double)ln - zm * zm;

                /* The headline pair.  value_mse is the head's error against
                 * the target it was trained on; value_mse_baseline is the
                 * error of a constant predictor sitting at the target mean.
                 * Their ratio is 0 for a perfect head and 1 for a head that
                 * has given up and is emitting the mean. */
                gs.value_mse       = gs.loss_val;
                gs.value_mse_ratio = (gs.value_mse_baseline > 1e-12)
                                   ? gs.value_mse / gs.value_mse_baseline : 0.0;
                /* Repeated against the PURE game outcome so runs with
                 * different --value-mix stay comparable. */
                ov = oq / (double)ln - om * om;
                gs.value_mse_outcome          = vo / (double)ln;
                gs.search_mse_outcome         = qo / (double)ln;
                gs.search_mse_outcome_ratio   = (ov > 1e-12) ? gs.search_mse_outcome / ov : 0.0;
                gs.value_mse_outcome_baseline = ov;
                gs.value_mse_outcome_ratio    = (ov > 1e-12)
                                              ? gs.value_mse_outcome / ov : 0.0;
                gs.value_target_mean = zm;
                gs.value_target_std  = sqrt(gs.value_mse_baseline > 0.0
                                            ? gs.value_mse_baseline : 0.0);
                gs.value_pred_mean   = vm;
                {   /* a collapsed head is exactly one whose sd goes to zero */
                    const double pv = vq / (double)ln - vm * vm;
                    gs.value_pred_std = sqrt(pv > 0.0 ? pv : 0.0);
                }
            }
            gs.loss_tot        = gs.loss_pol + (double)c->value_coef * gs.loss_val;
            gs.train_positions = ln;
            gs.steps           = (uint64_t)sh.gnorm_n;
            gs.grad_norm       = sh.gnorm_n ? sh.gnorm_sum / (double)sh.gnorm_n : 0.0;
            gs.grad_clip_frac  = sh.gnorm_n ? (double)sh.gclip_n / (double)sh.gnorm_n : 0.0;
            gs.lr              = (double)sh.lr_now;
            gs.buffer_fill     = (double)azbuf_count(&sh.buf) / (double)sh.buf.cap;
        }

        /* -- 3b. GRADIENT DECOMPOSITION ---------------------------------- */
        /* Reduce the probe step's two single-term trunk gradients.  This is
         * the number behind "the shared trunk is shaped by the policy": the
         * two loss VALUES say nothing about it, because they are on different
         * scales and pass through different heads.  These are the gradients
         * that actually reach the trunk, value_coef already applied. */
        if (sh.probe_nb > 0) {
            float *gp = (float *)tgsum;
            const double s2 = 1.0 / (double)sh.probe_nb;
            for (int pass = 0; pass < 2; pass++) {
                double ss = 0.0;
                memset(gp, 0, sizeof(TrunkGrad));
                for (int t = 0; t < nthreads; t++)
                    grad_add(gp, (const float *)(pass ? workers[t].tg_val
                                                      : workers[t].tg_pol),
                             (int)TRUNK_NPARAM);
                for (size_t k = 0; k < TRUNK_NPARAM; k++) {
                    const double v = (double)gp[k] * s2;
                    ss += v * v;
                }
                if (pass) gs.grad_trunk_value  = sqrt(ss);
                else      gs.grad_trunk_policy = sqrt(ss);
            }
            memset(gp, 0, sizeof(TrunkGrad));   /* leave it as adam_step does */
            {
                const double tot = gs.grad_trunk_policy + gs.grad_trunk_value;
                gs.grad_value_share = (tot > 0.0) ? gs.grad_trunk_value / tot : 0.0;
            }
        }

        /* -- 4. RATE ----------------------------------------------------- */
        apply_elo(&sh);
        {
            double mean = 0.0;
            int best = 0;
            for (int i = 0; i < n; i++) {
                mean += (double)elo[i];
                if (elo[i] > elo[best]) best = i;
            }
            memcpy(elo_sorted, elo, (size_t)n * sizeof(float));
            qsort(elo_sorted, (size_t)n, sizeof(float), cmp_float_asc);
            gs.elo_mean = mean / (double)n;
            gs.elo_best = (double)elo[best];
            gs.elo_p10  = (double)elo_sorted[(int)(0.10 * (double)(n - 1))];
            gs.best_i   = best;
        }

        /* -- 4b. EMA vs RAW ---------------------------------------------
         * Polyak averaging is USUALLY stronger, which is not the same as
         * always.  Before best.crl becomes the EMA copy, the two play each
         * other; the score is logged either way.  The cost is inside gs.sec,
         * so it shows up in the generation time rather than hiding. */
        if (gs.elo_best >= best_saved) {
            best_saved = gs.elo_best;
            /* The head-to-head is ema_h2h_games full-strength games played on
             * ONE thread while the pool is parked, so it is metered: at most
             * once every SAVE_EVERY generations, plus the last.  Between
             * measurements the previous verdict stands.  Unmetered it costs
             * ~15% of a generation at the shipped sim count, which would be
             * paying more for the decision than the decision is worth. */
            if (ema_trunk && c->ema_h2h_games > 0 &&
                (gen >= h2h_next || gen == c->generations)) {
                h2h_next = gen + SAVE_EVERY;
                gs.ema_h2h_games = c->ema_h2h_games;
                gs.ema_h2h = az_ema_h2h(&sh, &workers[0], gs.best_i,
                                        c->ema_h2h_games, h2h_rng);
                ema_wins = (gs.ema_h2h > 0.5);
            }
            gs.best_is_ema = ema_wins;
            if (ema_trunk) {
                snprintf(path, sizeof path, "%s/best_ema.crl", c->run_dir);
                if (!model_save(path, ema_trunk, ema_heads, hypers, elo, n, gen))
                    fprintf(stderr, "az: warning: could not write %s\n", path);
            }
            snprintf(path, sizeof path, "%s/best_raw.crl", c->run_dir);
            if (!model_save(path, trunk, heads, hypers, elo, n, gen))
                fprintf(stderr, "az: warning: could not write %s\n", path);
            snprintf(path, sizeof path, "%s/best.crl", c->run_dir);
            if (!model_save(path, gs.best_is_ema ? ema_trunk : trunk,
                            gs.best_is_ema ? ema_heads : heads,
                            hypers, elo, n, gen))
                fprintf(stderr, "az: warning: could not write %s\n", path);
        }

        gs.sec = now_sec() - t0;
        total_games += st.games;
        total_plies += st.plies;
        total_evals += gs.evals;

        /* -- 5. LOG ------------------------------------------------------ */
        write_telemetry(tel, &sh, &st, &gs);

        if (!c->quiet) {
            const double gd = st.games ? (double)st.games : 1.0;
            char eta[32];
            ema_gen = (gen == 1) ? gs.sec : (0.7 * ema_gen + 0.3 * gs.sec);
            fmt_dur(ema_gen * (double)(c->generations - gen), eta, sizeof eta);
            printf("gen %4d/%d  %5.1f g/s  %6.0f ev/s  elo %7.1f/%7.1f  "
                   "W%3.0f%% D%3.0f%% L%3.0f%%  len %5.1f  "
                   "kl %5.3f  top1 %4.0f%%  v %5.3f/%5.3f %4.2fx  "
                   "|g| %5.2f %3.0f%%clip  buf %4.1f%%  %5.2fs  eta %s\n",
                   gen, c->generations, (double)st.games / (gs.sec > 0.0 ? gs.sec : 1.0),
                   (double)gs.evals / (gs.selfplay_sec > 0.0 ? gs.selfplay_sec : 1.0),
                   gs.elo_best, gs.elo_mean,
                   100.0 * (double)st.white_wins / gd,
                   100.0 * (double)st.draws / gd,
                   100.0 * (double)st.black_wins / gd,
                   st.sum_len / gd, gs.policy_kl, 100.0 * gs.policy_top1,
                   gs.loss_val, gs.value_mse_baseline, gs.value_mse_ratio,
                   gs.grad_norm, 100.0 * gs.grad_clip_frac,
                   100.0 * gs.buffer_fill, gs.sec, eta);
            fflush(stdout);
        }

        /* -- 6. EVOLVE --------------------------------------------------- */
        {
            int n_elite = (int)(c->elite_frac * (double)n);
            int n_cull  = (int)(c->cull_frac  * (double)n);
            if (n_elite < 1) n_elite = 1;
            if (n_elite > n) n_elite = n;
            if (n_cull > n - n_elite) n_cull = n - n_elite;

            for (int i = 0; i < n; i++) { rank[i].elo = elo[i]; rank[i].idx = i; }
            qsort(rank, (size_t)n, sizeof(EloRank), cmp_rank_desc);

            for (int k = 0; k < n_cull; k++) {
                const int victim = rank[n - 1 - k].idx;
                const int elite  = rank[(int)(az_next(master) % (uint64_t)n_elite)].idx;
                heads[victim]  = heads[elite];
                hypers[victim] = hypers[elite];
                if (head_adam[victim].m && head_adam[elite].m) {
                    memcpy(head_adam[victim].m, head_adam[elite].m,
                           (size_t)HEAD_NPARAM * sizeof(float));
                    memcpy(head_adam[victim].v, head_adam[elite].v,
                           (size_t)HEAD_NPARAM * sizeof(float));
                    head_adam[victim].t = head_adam[elite].t;
                }
                head_mutate(&heads[victim], hypers[victim].mutate_sigma, master);
                az_hyper_mutate(&hypers[victim], master);
                /* The victim's average was tracking a head that no longer
                 * exists, so restart it from the clone's own weights. */
                if (ema_heads) ema_heads[victim] = heads[victim];
            }
        }

        /* -- 7. HALL OF FAME --------------------------------------------- */
        if (sh.hof_n == 0 || gen % c->hof_every == 0) {
            HofEntry *e = &hof[sh.hof_next];
            e->head = heads[gs.best_i];
            e->hy   = hypers[gs.best_i];
            e->elo  = elo[gs.best_i];
            e->gen  = gen;
            sh.hof_next = (sh.hof_next + 1) % HOF_CAP;
            if (sh.hof_n < HOF_CAP) sh.hof_n++;
        }

        /* -- 8. CHECKPOINT ----------------------------------------------- */
        if (gen % SAVE_EVERY == 0 || gen == c->generations || g_az_interrupt) {
            snprintf(path, sizeof path, "%s/checkpoint.crl", c->run_dir);
            if (!model_save(path, trunk, heads, hypers, elo, n, gen))
                fprintf(stderr, "az: warning: could not write %s\n", path);
            if (ema_trunk) {
                snprintf(path, sizeof path, "%s/checkpoint_ema.crl", c->run_dir);
                if (!model_save(path, ema_trunk, ema_heads, hypers, elo, n, gen))
                    fprintf(stderr, "az: warning: could not write %s\n", path);
            }
        }

        if (g_az_interrupt) {
            snprintf(path, sizeof path, "%s/checkpoint.crl", c->run_dir);
            model_save(path, trunk, heads, hypers, elo, n, gen);
            if (!c->quiet)
                printf("\ninterrupted: generation %d finished and saved.\n", gen);
            break;
        }
    }
    /* ===================================================================== */

    if (!c->quiet) {
        const double dt = now_sec() - t_run0;
        char buf[32];
        fmt_dur(dt, buf, sizeof buf);
        printf("done: %llu games, %llu plies, %llu network evaluations in %s\n",
               (unsigned long long)total_games, (unsigned long long)total_plies,
               (unsigned long long)total_evals, buf);
        printf("      %.1f games/s, %.0f positions/s, %.0f evals/s\n",
               (double)total_games / (dt > 0.0 ? dt : 1.0),
               (double)total_plies / (dt > 0.0 ? dt : 1.0),
               (double)total_evals / (dt > 0.0 ? dt : 1.0));
        printf("model: %s/best.crl  (raw %s/best_raw.crl%s)\n", c->run_dir,
               c->run_dir, ema_trunk ? ", ema best_ema.crl" : "");
        fflush(stdout);
    }
    rc = 0;

done:
    if (nspawned > 0) {
        sh.job = JOB_EXIT;
        bar_wait(&sh.bar);
        for (int t = 1; t < nspawned; t++) pthread_join(tids[t], NULL);
    }
    if (old_sigint != SIG_ERR) signal(SIGINT, old_sigint);
    if (tel) fclose(tel);
    if (workers) {
        for (int t = 0; t < nworkers_init; t++) {
            mcts_free(&workers[t].m);
            rec_free(&workers[t].rec);
            free(workers[t].tg);
            free(workers[t].tg_pol);
            free(workers[t].tg_val);
            free(workers[t].hg_probe);
        }
    }
    if (bar_ready) bar_destroy(&sh.bar);
    if (adam_ready) {
        adam_free(&sh.trunk_adam);
        for (int i = 0; i < nadam; i++) adam_free(&head_adam[i]);
    }
    azbuf_free(&sh.buf);
    free(sh.tstart); free(sh.aoff); free(sh.acount);
    free(sh.batch_tmp); free(sh.batch);
    free(tids); free(workers); free(head_adam); free(tgsum); free(hgrad);
    free(bslot); free(wslot); free(rank); free(results); free(pairs);
    free(hof); free(elo_sorted); free(elo); free(hypers); free(heads); free(trunk);
    free(ema_heads); free(ema_trunk);
    return rc;
}

/* ==========================================================================
 *                    LEARNING-RATE RANGE TEST (Smith 2015)
 * ==========================================================================
 * https://arxiv.org/abs/1506.01186, section 3.3.
 *
 * The procedure, and the three things that make it honest here:
 *
 *   1. THE DATA IS REAL.  The buffer is filled by az_dispatch(JOB_SELFPLAY),
 *      the identical call training makes, with the identical AZCfg.  A range
 *      test on synthetic or stale positions measures the wrong curvature.
 *
 *   2. THE OPTIMISER STEP IS REAL.  Each point on the curve is one
 *      az_dispatch(JOB_LEARN) with AZShared.one_step set, i.e. exactly the
 *      minibatch, the counting-sort by agent, the per-head Adam with that
 *      agent's lr_scale, the trunk reduction and the same clip that training
 *      uses.  Only lr_now differs from step to step.
 *
 *   3. THE MODEL IS NOT TOUCHED.  The weights are snapshotted before the sweep
 *      and restored after it, and the restoration is CHECKED with memcmp over
 *      every float.  The command never writes a checkpoint.
 *
 * WHAT IS DELIBERATELY NOT REAL: Adam's moment estimates start at zero.  A
 * checkpoint stores weights, not optimiser state, so there is nothing else to
 * start them from.  The sweep therefore measures the rate at which a FRESH
 * Adam is stable on this weight configuration and this data distribution --
 * which is the question a range test is meant to answer, and is also exactly
 * the situation at the start of a training run.
 *
 * The loss recorded at step i is the loss of the minibatch BEFORE that step's
 * update is applied, which is the usual convention: it reflects the damage
 * done by steps 1..i-1, so the curve turns up once the rate has become too
 * large to recover from.
 */

typedef struct {
    double lr;
    double loss, pol, val;      /* total = pol + value_coef * val            */
    double gnorm, hgnorm;       /* PRE-clip trunk / mean per-head grad norm  */
    double s_tot, s_pol, s_val; /* debiased EMA of each                      */
    int    nb;                  /* positions actually in the minibatch       */
    int    finite;
} LrfRow;

void az_lrfind_default_cfg(AZLrFindCfg *c)
{
    if (!c) return;
    memset(c, 0, sizeof *c);
    az_default_cfg(&c->az);
    /* Two games per agent per round is the minimum the pairing code allows,
     * and makes --warm-games quantise in units of one game per agent. */
    c->az.games_per_agent = 2;
    c->az.quiet           = 1;
    c->model       = NULL;
    c->lo          = 1e-6;
    c->hi          = 1.0;
    c->steps       = 300;
    c->warm_games  = 256;
    /* beta = 0.9 is a ~10-step window.  Over a 300-step sweep of six decades
     * that is 0.2 decades of lag -- enough to suppress minibatch noise, small
     * enough not to displace the minimum.  Printed with the result so the
     * number can never be read without knowing what smoothed it. */
    c->smooth      = 0.9;
    c->stop_factor = 0.0;        /* run the whole sweep by default           */
    c->csv         = NULL;
    c->plot_rows   = 18;
    c->plot_cols   = 62;
}

/* Debiased exponential moving average, in place over n rows. */
static void lrf_smooth(LrfRow *r, int n, double beta)
{
    double a = 0.0, b = 0.0, d = 0.0, w = 1.0;
    for (int i = 0; i < n; i++) {
        w *= beta;
        const double corr = 1.0 - w;
        if (r[i].finite) {
            a = beta * a + (1.0 - beta) * r[i].loss;
            b = beta * b + (1.0 - beta) * r[i].pol;
            d = beta * d + (1.0 - beta) * r[i].val;
            r[i].s_tot = a / corr;
            r[i].s_pol = b / corr;
            r[i].s_val = d / corr;
        } else {
            /* A non-finite loss cannot enter an average without destroying
             * every later value, so the EMA is frozen and the row is marked. */
            r[i].s_tot = r[i].s_pol = r[i].s_val = (double)INFINITY;
        }
    }
}

/* argmin of a smoothed column, over finite rows only.  Returns -1 if none. */
static int lrf_argmin(const LrfRow *r, int n, int which)
{
    int best = -1;
    for (int i = 0; i < n; i++) {
        if (!r[i].finite) continue;
        const double v = (which == 0) ? r[i].s_tot
                       : (which == 1) ? r[i].s_pol : r[i].s_val;
        if (!isfinite(v)) continue;
        const double bv = (best < 0) ? 0.0
                        : ((which == 0) ? r[best].s_tot
                        :  (which == 1) ? r[best].s_pol : r[best].s_val);
        if (best < 0 || v < bv) best = i;
    }
    return best;
}

/* d(smoothed total)/d(log10 lr), differenced over +/- win rows so that the
 * slope is read off the curve rather than off the minibatch noise. */
static double lrf_slope(const LrfRow *r, int n, int i, int win)
{
    int a = i - win, b = i + win;
    if (a < 0) a = 0;
    if (b > n - 1) b = n - 1;
    while (a < b && !(r[a].finite && isfinite(r[a].s_tot))) a++;
    while (b > a && !(r[b].finite && isfinite(r[b].s_tot))) b--;
    if (b <= a) return 0.0;
    const double dx = log10(r[b].lr) - log10(r[a].lr);
    if (!(dx > 0.0)) return 0.0;
    return (r[b].s_tot - r[a].s_tot) / dx;
}

/* -------------------------------------------------------------- the plot */
/* Smoothed total loss against log10(lr).  Columns are log-lr bins; the value
 * plotted in a bin is the mean of the smoothed losses that fall in it.  Rows
 * above the clip ceiling are drawn as '!' on the top row and counted, so a
 * divergence to 1e9 cannot flatten the interesting part of the curve. */
static void lrf_plot(const LrfRow *r, int n, int rows, int cols,
                     int i_min, int i_rec, int i_div)
{
    if (n < 2 || rows < 6 || cols < 20) return;

    double *acc = (double *)calloc((size_t)cols, sizeof(double));
    int    *cnt = (int *)   calloc((size_t)cols, sizeof(int));
    int    *bad = (int *)   calloc((size_t)cols, sizeof(int));
    char   *grid = (char *) malloc((size_t)rows * (size_t)cols);
    char   *mark = (char *) malloc((size_t)cols + 1);
    if (!acc || !cnt || !bad || !grid || !mark) {
        free(acc); free(cnt); free(bad); free(grid); free(mark);
        return;
    }
    memset(grid, ' ', (size_t)rows * (size_t)cols);
    memset(mark, ' ', (size_t)cols);
    mark[cols] = 0;

    const double x0 = log10(r[0].lr), x1 = log10(r[n - 1].lr);
    const double xs = (x1 > x0) ? (double)(cols - 1) / (x1 - x0) : 0.0;

    /* y range: floor at the minimum, ceiling at 1.5x the starting loss, so
     * the descent and the turn-up both occupy most of the plot. */
    double ymin = 1e300, ystart = 0.0, ymaxf = -1e300;
    int have = 0;
    for (int i = 0; i < n; i++) {
        if (!r[i].finite || !isfinite(r[i].s_tot)) continue;
        if (!have) { ystart = r[i].s_tot; have = 1; }
        if (r[i].s_tot < ymin)  ymin  = r[i].s_tot;
        if (r[i].s_tot > ymaxf) ymaxf = r[i].s_tot;
    }
    if (!have) { free(acc); free(cnt); free(bad); free(grid); free(mark); return; }
    double ymax = ystart + 0.5 * (ystart - ymin);
    if (ymax > ymaxf) ymax = ymaxf;
    if (!(ymax > ymin)) ymax = ymin + (fabs(ymin) > 0.0 ? fabs(ymin) * 0.1 : 1.0);

    int clipped = 0;
    for (int i = 0; i < n; i++) {
        int col = (int)((log10(r[i].lr) - x0) * xs + 0.5);
        if (col < 0) col = 0;
        if (col > cols - 1) col = cols - 1;
        if (!r[i].finite || !isfinite(r[i].s_tot)) { bad[col]++; continue; }
        if (r[i].s_tot > ymax) { bad[col]++; clipped++; continue; }
        acc[col] += r[i].s_tot;
        cnt[col]++;
    }
    for (int col = 0; col < cols; col++) {
        if (cnt[col] > 0) {
            const double v = acc[col] / (double)cnt[col];
            int row = (int)((ymax - v) / (ymax - ymin) * (double)(rows - 1) + 0.5);
            if (row < 0) row = 0;
            if (row > rows - 1) row = rows - 1;
            grid[(size_t)row * (size_t)cols + (size_t)col] = '*';
        } else if (bad[col] > 0) {
            grid[(size_t)col] = '!';
        }
    }
    {   /* markers under the axis */
        const int idx[3] = { i_min, i_rec, i_div };
        const char ch[3] = { 'm', 'R', 'd' };
        for (int k = 0; k < 3; k++) {
            if (idx[k] < 0 || idx[k] >= n) continue;
            int col = (int)((log10(r[idx[k]].lr) - x0) * xs + 0.5);
            if (col < 0) col = 0;
            if (col > cols - 1) col = cols - 1;
            mark[col] = ch[k];
        }
    }

    printf("\n  smoothed total loss against log10(learning rate)\n\n");
    for (int row = 0; row < rows; row++) {
        const double v = ymax - (ymax - ymin) * (double)row / (double)(rows - 1);
        if (row == 0 || row == rows - 1 || row == rows / 2)
            printf("  %8.4f |", v);
        else
            printf("           |");
        fwrite(grid + (size_t)row * (size_t)cols, 1, (size_t)cols, stdout);
        printf("|\n");
    }
    printf("           +");
    for (int col = 0; col < cols; col++) putchar('-');
    printf("+\n            %s\n", mark);
    printf("           ");
    {   /* decade ticks along the x axis */
        char axis[256];
        const int w = (cols < 250) ? cols : 250;
        memset(axis, ' ', sizeof axis);
        for (int e = (int)ceil(x0); e <= (int)floor(x1); e++) {
            int col = (int)(((double)e - x0) * xs + 0.5);
            char t[16];
            int tw;
            snprintf(t, sizeof t, "1e%d", e);
            tw = (int)strlen(t);
            if (col < 0) continue;
            if (col + tw > w) col = w - tw;          /* keep the last tick on */
            if (col < 0) continue;
            if (axis[col] != ' ') continue;          /* do not overwrite one  */
            memcpy(axis + col, t, (size_t)tw);
        }
        axis[w] = 0;
        printf(" %s\n", axis);
    }
    printf("            m = min smoothed loss   R = recommendation   "
           "d = divergence\n");
    if (clipped > 0)
        printf("            %d step(s) above the top of the plot, drawn as '!'\n",
               clipped);

    free(acc); free(cnt); free(bad); free(grid); free(mark);
}

/* ------------------------------------------------------------ the command */

int az_lrfind(AZLrFindCfg *lc)
{
    AZShared  sh;
    AZCfg    *c;
    Trunk    *trunk = NULL, *snap_trunk = NULL;
    Head     *heads = NULL, *snap_heads = NULL;
    Hyper    *hypers = NULL;
    float    *elo = NULL;
    HofEntry *hof = NULL;
    AZPair   *pairs = NULL;
    AZRes    *results = NULL;
    EloRank  *rank = NULL;
    int32_t  *wslot = NULL, *bslot = NULL;
    HeadGrad *hgrad = NULL;
    TrunkGrad *tgsum = NULL;
    Adam     *head_adam = NULL;
    AZWorker *workers = NULL;
    pthread_t *tids = NULL;
    LrfRow   *row = NULL;
    uint64_t  master[4];
    int  n, gpa, nthreads, npairs_max, rc = 1;
    int  adam_ready = 0, nadam = 0, nspawned = 0, bar_ready = 0, nworkers_init = 0;
    int  gen_loaded = 0, nsteps = 0, n_rows = 0;
    int  restored_ok = 0;
    uint64_t games_played = 0;
    double t0;

    if (!lc) return 1;
    c = &lc->az;
    chess_init();

    /* ---- sanitise the sweep --------------------------------------------- */
    if (!(lc->lo > 0.0))        lc->lo = 1e-6;
    if (!(lc->hi > lc->lo))     lc->hi = lc->lo * 1e6;
    if (lc->steps < 8)          lc->steps = 8;
    if (lc->steps > 100000)     lc->steps = 100000;
    if (lc->warm_games < 1)     lc->warm_games = 1;
    if (!(lc->smooth >= 0.0) || lc->smooth >= 1.0) lc->smooth = 0.9;
    if (!(lc->stop_factor >= 0.0)) lc->stop_factor = 0.0;
    if (lc->plot_rows < 6)      lc->plot_rows = 18;
    if (lc->plot_cols < 20)     lc->plot_cols = 62;
    if (lc->plot_cols > 200)    lc->plot_cols = 200;
    /* generations is irrelevant here but az_sanitise clamps warmup against it */
    c->generations = 1;
    c->warmup_gens = 0;
    c->ema_decay   = 0.0f;       /* nothing is saved, so nothing to average   */
    c->ema_h2h_games = 0;
    az_sanitise(c);

    /* ---- how many agents? the checkpoint decides ------------------------ */
    if (lc->model) {
        int probe_n = 0, probe_gen = 0;
        if (!model_load(lc->model, NULL, NULL, NULL, NULL, &probe_n, &probe_gen)) {
            fprintf(stderr,
                "lrfind: cannot read model '%s'.\n"
                "        model_load accepts MODEL_VERSION %u with widths "
                "%d/%d/%d/%d only; an older checkpoint has different tensors in a\n"
                "        different order and is refused rather than read as noise "
                "(see net.h).\n",
                lc->model, (unsigned)MODEL_VERSION,
                NF_INPUT, NF_ACC, NF_HID, NF_PDIM);
            return 1;
        }
        if (probe_n < 2) {
            fprintf(stderr, "lrfind: model '%s' holds %d agents; need at least 2\n",
                    lc->model, probe_n);
            return 1;
        }
        c->n_agents = probe_n;
        gen_loaded  = probe_gen;
    }

    n   = c->n_agents;
    gpa = c->games_per_agent;
    npairs_max = n * (gpa / 2);
    nthreads = c->threads;
    if (nthreads > npairs_max) nthreads = npairs_max;
    if (nthreads < 1) nthreads = 1;

    memset(&sh, 0, sizeof sh);

    trunk      = (Trunk *)     calloc(1, sizeof(Trunk));
    snap_trunk = (Trunk *)     calloc(1, sizeof(Trunk));
    heads      = (Head *)      calloc((size_t)n, sizeof(Head));
    snap_heads = (Head *)      calloc((size_t)n, sizeof(Head));
    hypers     = (Hyper *)     calloc((size_t)n, sizeof(Hyper));
    elo        = (float *)     calloc((size_t)n, sizeof(float));
    hof        = (HofEntry *)  calloc((size_t)HOF_CAP, sizeof(HofEntry));
    pairs      = (AZPair *)    calloc((size_t)npairs_max, sizeof(AZPair));
    results    = (AZRes *)     calloc((size_t)npairs_max, sizeof(AZRes));
    rank       = (EloRank *)   calloc((size_t)n, sizeof(EloRank));
    wslot      = (int32_t *)   calloc((size_t)npairs_max, sizeof(int32_t));
    bslot      = (int32_t *)   calloc((size_t)npairs_max, sizeof(int32_t));
    hgrad      = (HeadGrad *)  calloc((size_t)n, sizeof(HeadGrad));
    tgsum      = (TrunkGrad *) calloc(1, sizeof(TrunkGrad));
    head_adam  = (Adam *)      calloc((size_t)n, sizeof(Adam));
    workers    = (AZWorker *)  calloc((size_t)nthreads, sizeof(AZWorker));
    tids       = (pthread_t *) calloc((size_t)nthreads, sizeof(pthread_t));
    row        = (LrfRow *)    calloc((size_t)lc->steps, sizeof(LrfRow));

    sh.batch     = (int32_t *) calloc((size_t)c->batch_size, sizeof(int32_t));
    sh.batch_tmp = (int32_t *) calloc((size_t)c->batch_size, sizeof(int32_t));
    sh.acount    = (int *)     calloc((size_t)n, sizeof(int));
    sh.aoff      = (int *)     calloc((size_t)n, sizeof(int));
    sh.tstart    = (int *)     calloc((size_t)nthreads + 1, sizeof(int));

    if (!trunk || !snap_trunk || !heads || !snap_heads || !hypers || !elo ||
        !hof || !pairs || !results || !rank || !wslot || !bslot || !hgrad ||
        !tgsum || !head_adam || !workers || !tids || !row || !sh.batch ||
        !sh.batch_tmp || !sh.acount || !sh.aoff || !sh.tstart) {
        fprintf(stderr, "lrfind: out of memory\n");
        goto done;
    }
    if (!azbuf_init(&sh.buf, c->buffer_positions)) {
        fprintf(stderr, "lrfind: out of memory for a %d-position replay buffer\n",
                c->buffer_positions);
        goto done;
    }

    /* ---- weights: load, or initialise exactly as generation 1 does ------- */
    if (lc->model) {
        int cap = n, g2 = 0;
        if (!model_load(lc->model, trunk, heads, hypers, elo, &cap, &g2)) {
            fprintf(stderr, "lrfind: cannot read model '%s'\n", lc->model);
            goto done;
        }
    } else {
        nn_init(trunk, &heads[0], c->seed);
        for (int i = 1; i < n; i++) {
            Trunk scratch;
            nn_init(&scratch, &heads[i],
                    c->seed + 0x9E3779B97F4A7C15ull * (uint64_t)(i + 1));
        }
        az_seed(master, c->seed ^ 0xD1B54A32D192ED03ull);
        for (int i = 0; i < n; i++) {
            az_hyper_init(&hypers[i]);
            az_hyper_mutate(&hypers[i], master);
            elo[i] = ELO_SEED;
        }
    }

    /* ---- THE SNAPSHOT --------------------------------------------------- */
    *snap_trunk = *trunk;
    memcpy(snap_heads, heads, (size_t)n * sizeof(Head));

    adam_init(&sh.trunk_adam, (int)TRUNK_NPARAM);
    for (nadam = 0; nadam < n; nadam++) adam_init(&head_adam[nadam], (int)HEAD_NPARAM);
    adam_ready = 1;

    sh.cfg       = c;
    sh.trunk     = trunk;
    sh.heads     = heads;
    sh.hypers    = hypers;
    sh.elo       = elo;
    sh.hof       = hof;
    sh.pairs     = pairs;
    sh.results   = results;
    sh.npairs    = npairs_max;
    sh.hgrad     = hgrad;
    sh.tgsum     = tgsum;
    sh.head_adam = head_adam;
    sh.workers   = workers;
    sh.nthreads  = nthreads;
    sh.head_clip = (c->grad_clip_head > 0.0f) ? c->grad_clip_head : c->grad_clip;
    sh.cap_sims  = c->cap_sims > 0 ? c->cap_sims : (c->sims / 5 > 2 ? c->sims / 5 : 2);
    if (sh.cap_sims > c->sims) sh.cap_sims = c->sims;
    sh.job       = JOB_IDLE;
    sh.one_step  = 1;
    atomic_init(&sh.next, 0);

    bar_init(&sh.bar, nthreads);
    bar_ready = 1;

    for (nworkers_init = 0; nworkers_init < nthreads; nworkers_init++) {
        AZWorker *w = &workers[nworkers_init];
        w->sh  = &sh;
        w->tid = nworkers_init;
        mcts_init(&w->m, MAX_MOVES + 1 + c->sims * NODES_PER_SIM);
        w->m.c_puct          = c->c_puct;
        w->m.dirichlet_alpha = c->dirichlet_alpha;
        w->m.dirichlet_eps   = c->dirichlet_eps;
        w->tg       = (TrunkGrad *)calloc(1, sizeof(TrunkGrad));
        w->tg_pol   = (TrunkGrad *)calloc(1, sizeof(TrunkGrad));
        w->tg_val   = (TrunkGrad *)calloc(1, sizeof(TrunkGrad));
        w->hg_probe = (HeadGrad *) calloc(1, sizeof(HeadGrad));
        if (!w->m.pool || !w->tg || !w->tg_pol || !w->tg_val || !w->hg_probe ||
            !rec_init(&w->rec, c->max_plies)) {
            fprintf(stderr, "lrfind: out of memory (worker %d)\n", nworkers_init);
            nworkers_init++;
            goto done;
        }
    }
    for (nspawned = 1; nspawned < nthreads; nspawned++) {
        if (pthread_create(&tids[nspawned], NULL, az_worker_main, &workers[nspawned]) != 0) {
            fprintf(stderr, "lrfind: cannot create thread %d\n", nspawned);
            goto done;
        }
    }

    printf("chessrl lrfind -- learning-rate range test "
           "(Smith 2015, arxiv 1506.01186)\n");
    printf("  model        %s", lc->model ? lc->model : "(fresh nn_init)");
    if (lc->model) printf("   generation %d", gen_loaded);
    else           printf("   generation 0, seed %llu",
                          (unsigned long long)c->seed);
    printf("\n");
    printf("  population   %d agents, %zu trunk + %zu head parameters each\n",
           n, (size_t)TRUNK_NPARAM, (size_t)HEAD_NPARAM);
    printf("  self-play    %d sims, cap %.2f x %d sims, start %s, "
           "draw %.2f, max %d plies\n",
           c->sims, (double)c->cap_frac, sh.cap_sims,
           c->start_mode == AZ_START_CLASSICAL ? "classical" :
           c->start_mode == AZ_START_960 ? "960" : "mixed",
           (double)c->draw_penalty, c->max_plies);
    printf("  objective    L = policy_CE + %.2f * value_MSE, wd %.3g, "
           "clip %.3g trunk / %.3g head\n",
           (double)c->value_coef, (double)c->weight_decay,
           (double)c->grad_clip, (double)sh.head_clip);
    printf("  sweep        %d steps, batch %d, lr %.3g -> %.3g geometric "
           "(%.4f decades/step)\n",
           lc->steps, c->batch_size, lc->lo, lc->hi,
           (log10(lc->hi) - log10(lc->lo)) / (double)(lc->steps - 1));
    printf("  threads      %d\n", nthreads);
    fflush(stdout);

    /* ---- 1. WARM THE BUFFER, WITH THE REAL SELF-PLAY PATH ---------------- */
    t0 = now_sec();
    {
        const int per_round = npairs_max;
        int rounds = (lc->warm_games + per_round - 1) / per_round;
        if (rounds < 1) rounds = 1;
        printf("\n  warming the replay buffer: %d round(s) x %d games ...",
               rounds, per_round);
        fflush(stdout);
        for (int r = 0; r < rounds; r++) {
            az_seed(master, c->seed + 0x2545F4914F6CDD1Dull * (uint64_t)(r + 1));
            sh.npairs = build_pairings(&sh, rank, wslot, bslot, master);
            memset(results, 0, (size_t)sh.npairs * sizeof(AZRes));
            atomic_store(&sh.next, 0);
            for (int t = 0; t < nthreads; t++) {
                AZWorker *w = &workers[t];
                az_seed(w->rng, c->seed
                        + 0x9E3779B97F4A7C15ull * (uint64_t)(t + 1)
                        + 0xBF58476D1CE4E5B9ull * (uint64_t)(r + 1));
                st_zero(&w->st);
                w->evals = 0;
                w->rootv_sum = 0.0; w->rootv_n = 0;
                w->tgt_ent_sum = 0.0; w->tgt_ent_n = 0;
                w->resign_checked = w->resign_would = w->resign_wrong = 0;
                w->full_moves = w->all_moves = 0;
            }
            az_dispatch(&sh, JOB_SELFPLAY);
            apply_elo(&sh);
            for (int t = 0; t < nthreads; t++) games_played += workers[t].st.games;
        }
    }
    {
        const uint64_t have = azbuf_count(&sh.buf);
        printf(" %llu games, %llu positions in %.1fs\n",
               (unsigned long long)games_played, (unsigned long long)have,
               now_sec() - t0);
        if (have == 0) {
            fprintf(stderr,
                "lrfind: the replay buffer is EMPTY after %llu games. Nothing can "
                "be measured.\n", (unsigned long long)games_played);
            goto done;
        }
        {   /* Degenerate case: too little data to sample a batch from. */
            const double epochs = (double)lc->steps * (double)c->batch_size /
                                  (double)have;
            printf("  buffer       %llu positions, batch %d -> %.1f passes over "
                   "the buffer across the sweep\n",
                   (unsigned long long)have, c->batch_size, epochs);
            if ((uint64_t)c->batch_size > have) {
                printf("  WARNING      the buffer holds FEWER positions (%llu) than "
                       "one minibatch (%d).\n"
                       "               Every batch is a resample of the same data; "
                       "the curve measures\n"
                       "               memorisation of %llu positions, not learning. "
                       "Raise --warm-games.\n",
                       (unsigned long long)have, c->batch_size,
                       (unsigned long long)have);
            } else if (epochs > 30.0) {
                printf("  WARNING      %.0f passes over the buffer: the descent may "
                       "be memorisation rather\n"
                       "               than a learning rate that suits the task. "
                       "Raise --warm-games or\n"
                       "               lower --steps to check.\n", epochs);
            }
        }
    }

    /* ---- 2. THE SWEEP ---------------------------------------------------- */
    t0 = now_sec();
    {
        const double lx0 = log10(lc->lo), lx1 = log10(lc->hi);
        const double dx  = (lx1 - lx0) / (double)(lc->steps - 1);
        double best_s = 1e300;
        int stop_reason = 0;       /* 1 = non-finite, 2 = stop_factor         */

        for (nsteps = 0; nsteps < lc->steps; nsteps++) {
            const double lr = pow(10.0, lx0 + dx * (double)nsteps);
            double lp = 0.0, lv = 0.0, hg = 0.0;
            uint64_t ln = 0, hgn = 0;

            sh.lr_now = (float)lr;
            for (int t = 0; t < nthreads; t++) {
                AZWorker *w = &workers[t];
                w->l_pol = w->l_val = 0.0;
                w->l_n = 0;
                w->hgnorm_sum = 0.0;
                w->hgnorm_n = w->hclip_n = 0;
                w->probe = 0;
            }
            sh.gnorm_sum = 0.0;
            sh.gnorm_n = sh.gclip_n = 0;

            az_dispatch(&sh, JOB_LEARN);

            for (int t = 0; t < nthreads; t++) {
                lp  += workers[t].l_pol;
                lv  += workers[t].l_val;
                ln  += workers[t].l_n;
                hg  += workers[t].hgnorm_sum;
                hgn += workers[t].hgnorm_n;
            }
            row[nsteps].lr     = lr;
            row[nsteps].nb     = (int)ln;
            row[nsteps].pol    = ln ? lp / (double)ln : 0.0;
            row[nsteps].val    = ln ? lv / (double)ln : 0.0;
            row[nsteps].loss   = row[nsteps].pol + (double)c->value_coef * row[nsteps].val;
            row[nsteps].gnorm  = sh.gnorm_n ? sh.gnorm_sum / (double)sh.gnorm_n : 0.0;
            row[nsteps].hgnorm = hgn ? hg / (double)hgn : 0.0;
            /* A finite loss is not enough.  relu(NaN) is 0 on every C
             * implementation (NaN > 0 is false), so a network whose value
             * trunk has gone non-finite still reports a perfectly ordinary
             * loss while every gradient reaching the trunk is NaN and every
             * optimiser step is garbage.  The gradient norm is the honest
             * detector, so a step counts as usable only if BOTH are finite. */
            row[nsteps].finite = ln > 0 && isfinite(row[nsteps].loss) &&
                                 isfinite(row[nsteps].gnorm);

            if (!row[nsteps].finite) { nsteps++; stop_reason = 1; break; }
            if (row[nsteps].loss < best_s) best_s = row[nsteps].loss;
            if (lc->stop_factor > 0.0 && row[nsteps].loss > lc->stop_factor * best_s) {
                nsteps++; stop_reason = 2; break;
            }
        }
        n_rows = nsteps;
        printf("  sweep        %d step(s) in %.1fs%s\n", n_rows, now_sec() - t0,
               stop_reason == 1 ? "   (stopped: the loss went non-finite)"
             : stop_reason == 2 ? "   (stopped: --stop-factor exceeded)" : "");
    }

    /* ---- 3. RESTORE AND PROVE IT ---------------------------------------- */
    *trunk = *snap_trunk;
    memcpy(heads, snap_heads, (size_t)n * sizeof(Head));
    restored_ok = (memcmp(trunk, snap_trunk, sizeof(Trunk)) == 0) &&
                  (memcmp(heads, snap_heads, (size_t)n * sizeof(Head)) == 0);
    printf("  weights      restored from the pre-sweep snapshot: %s"
           "  (%zu trunk + %zu x %zu head floats compared)\n",
           restored_ok ? "BIT-IDENTICAL" : "*** MISMATCH ***",
           (size_t)TRUNK_NPARAM, (size_t)n, (size_t)HEAD_NPARAM);
    if (!restored_ok) {
        fprintf(stderr, "lrfind: the weight snapshot did not restore. Refusing to "
                        "report a result.\n");
        goto done;
    }
    printf("  model file   never written; lrfind opens no checkpoint for "
           "writing\n");

    /* ---- 4. ANALYSE ------------------------------------------------------ */
    if (n_rows < 1) {
        fprintf(stderr, "lrfind: no steps completed\n");
        goto done;
    }
    lrf_smooth(row, n_rows, lc->smooth);
    {
        const int win = (n_rows / 40 > 1) ? n_rows / 40 : 1;
        int i_min = lrf_argmin(row, n_rows, 0);
        int i_minp = lrf_argmin(row, n_rows, 1);
        int i_minv = lrf_argmin(row, n_rows, 2);
        int i_steep = -1, i_div = -1, i_div4 = -1, i_nan = -1;
        int n_finite = 0;
        double s0 = 0.0, best_slope = 0.0;
        double resid_sd = 0.0, sd_s = 0.0, depth = 0.0;
        int have_s0 = 0, significant = 0;

        for (int i = 0; i < n_rows; i++) {
            if (row[i].finite && isfinite(row[i].s_tot)) {
                n_finite++;
                if (!have_s0) { s0 = row[i].s_tot; have_s0 = 1; }
            } else if (i_nan < 0) i_nan = i;
        }
        for (int i = 0; i < n_rows; i++) {
            if (!row[i].finite || !isfinite(row[i].s_tot)) continue;
            const double sl = lrf_slope(row, n_rows, i, win);
            if (i_steep < 0 || sl < best_slope) { best_slope = sl; i_steep = i; }
        }
        if (i_min >= 0) {
            for (int i = i_min + 1; i < n_rows; i++) {
                if (!row[i].finite || !isfinite(row[i].s_tot)) { i_div = i; break; }
                if (have_s0 && row[i].s_tot > s0) { i_div = i; break; }
            }
            for (int i = i_min + 1; i < n_rows; i++) {
                if (!row[i].finite || !isfinite(row[i].s_tot)) { i_div4 = i; break; }
                if (row[i].s_tot > 4.0 * row[i_min].s_tot) { i_div4 = i; break; }
            }
        }

        /* ---- the table ------------------------------------------------- */
        {
            int stride = n_rows / 24;
            if (stride < 1) stride = 1;
            printf("\n  step      lr      loss    policy    value   smoothed"
                   "   |g|trunk   |g|head   n\n");
            for (int i = 0; i < n_rows; i += stride) {
                printf("  %4d  %9.3g  %8.4f  %7.4f  %7.4f  %9.4f  %8.3f  %8.3f  %4d\n",
                       i, row[i].lr, row[i].loss, row[i].pol, row[i].val,
                       row[i].s_tot, row[i].gnorm, row[i].hgnorm, row[i].nb);
            }
            if ((n_rows - 1) % stride != 0) {
                const int i = n_rows - 1;
                printf("  %4d  %9.3g  %8.4f  %7.4f  %7.4f  %9.4f  %8.3f  %8.3f  %4d\n",
                       i, row[i].lr, row[i].loss, row[i].pol, row[i].val,
                       row[i].s_tot, row[i].gnorm, row[i].hgnorm, row[i].nb);
            }
        }

        /* ---- degenerate cases, said out loud --------------------------- */
        printf("\n  READING THE CURVE   (smoothing: debiased EMA, beta = %.3f, "
               "~%.0f-step window)\n", lc->smooth,
               lc->smooth > 0.0 ? 1.0 / (1.0 - lc->smooth) : 1.0);

        if (i_nan == 0) {
            printf("\n  DEGENERATE: the FIRST step was already unusable at "
                   "lr = %.3g:\n"
                   "    loss               %g\n"
                   "    trunk grad norm    %g\n"
                   "  Nothing in this sweep is interpretable. Either the model is "
                   "already broken --\n"
                   "  note that a finite loss with a non-finite gradient means "
                   "non-finite weights that\n"
                   "  relu() is silently flattening to zero -- or --lo is far too "
                   "high. No recommendation.\n",
                   row[0].lr, row[0].loss, row[0].gnorm);
            rc = 0;
            goto report_done;
        }
        if (n_finite < 8) {
            printf("\n  DEGENERATE: only %d finite step(s). Too few to read a "
                   "curve from. No recommendation.\n", n_finite);
            rc = 0;
            goto report_done;
        }

        /* IS THERE A CURVE AT ALL?  A range test over a window that contains no
         * usable rate produces a flat, noisy line, and every landmark read off
         * it -- the minimum, the steepest point, the first crossing back above
         * the start -- is then a reading of minibatch noise.  So the descent is
         * tested against the noise before anything is reported: resid_sd is the
         * spread of the RAW loss about its own smoothed curve, and an EMA with
         * coefficient b reduces variance by (1-b)/(1+b), so 3 * sd_smoothed is
         * the bar the descent has to clear. */
        {
            double rss = 0.0;
            int rn = 0;
            for (int i = 0; i < n_rows; i++) {
                if (!row[i].finite || !isfinite(row[i].s_tot)) continue;
                const double d = row[i].loss - row[i].s_tot;
                rss += d * d;
                rn++;
            }
            resid_sd = (rn > 1) ? sqrt(rss / (double)(rn - 1)) : 0.0;
            sd_s  = resid_sd * sqrt((1.0 - lc->smooth) / (1.0 + lc->smooth));
            depth = (i_min >= 0) ? s0 - row[i_min].s_tot : 0.0;
            significant = (depth > 3.0 * sd_s);
            printf("  noise floor          minibatch sd %.4f about the smoothed "
                   "curve -> sd(smoothed) %.4f\n", resid_sd, sd_s);
            printf("  descent depth        %.4f  (%.1f x sd(smoothed))%s\n",
                   depth, sd_s > 0.0 ? depth / sd_s : 0.0,
                   significant ? "" : "   NOT SIGNIFICANT");
        }
        if (!significant) {
            printf("\n  DEGENERATE: the smoothed loss is FLAT to within the "
                   "minibatch noise across the\n"
                   "  whole of [%.3g, %.3g]. The minimum, the steepest point and "
                   "the first crossing back\n"
                   "  above the starting loss are all readings of noise here, not "
                   "of a curve. Widen the\n"
                   "  sweep, raise --warm-games, or raise --steps. No "
                   "recommendation.\n", lc->lo, lc->hi);
            rc = 0;
            goto report_done;
        }

        printf("  min smoothed loss    lr = %-10.3g  loss %.4f   (step %d)\n",
               row[i_min].lr, row[i_min].s_tot, i_min);
        printf("  steepest descent     lr = %-10.3g  %.3f loss per decade "
               "(step %d)\n", row[i_steep].lr, best_slope, i_steep);
        if (i_div >= 0) {
            printf("  divergence           lr = %-10.3g  %s (step %d)%s\n",
                   row[i_div].lr,
                   row[i_div].finite && isfinite(row[i_div].s_tot)
                       ? "smoothed loss back above its starting value"
                       : "smoothed loss went non-finite", i_div,
                   i_div == n_rows - 1 ? "  [LAST STEP: this is a lower bound]" : "");
        } else {
            printf("  divergence           NOT REACHED anywhere in [%.3g, %.3g]. The "
                   "smoothed loss never\n"
                   "                       climbed back above its starting value, so "
                   "this sweep gives NO\n"
                   "                       upper bound on a usable rate. Re-run with "
                   "a larger --hi.\n", lc->lo, lc->hi);
        }
        if (i_div4 >= 0)
            printf("  4x-the-minimum       lr = %-10.3g (step %d)\n",
                   row[i_div4].lr, i_div4);
        else
            printf("  4x-the-minimum       NOT REACHED in [%.3g, %.3g]\n",
                   lc->lo, lc->hi);
        /* The two terms of the objective do not have to want the same rate,
         * and value_coef multiplies one of them, so the total-loss curve can
         * be almost entirely the value head's curve.  Both are reported. */
        printf("  policy term alone    lr(min) = %-10.3g -> min/10 = %-10.3g "
               "(range %.4f nats)\n",
               i_minp >= 0 ? row[i_minp].lr : 0.0,
               i_minp >= 0 ? row[i_minp].lr / 10.0 : 0.0,
               i_minp >= 0 ? row[0].s_pol - row[i_minp].s_pol : 0.0);
        printf("  value term alone     lr(min) = %-10.3g -> min/10 = %-10.3g "
               "(range %.4f, x%.2f coef)\n",
               i_minv >= 0 ? row[i_minv].lr : 0.0,
               i_minv >= 0 ? row[i_minv].lr / 10.0 : 0.0,
               i_minv >= 0 ? row[0].s_val - row[i_minv].s_val : 0.0,
               (double)c->value_coef);
        if (i_min >= 0 && i_minp >= 0 && i_minv >= 0) {
            const double dp = row[0].s_pol - row[i_minp].s_pol;
            const double dv = (double)c->value_coef * (row[0].s_val - row[i_minv].s_val);
            const double tot = dp + dv;
            if (tot > 1e-9)
                printf("  the descent is       %.0f%% value term, %.0f%% policy term\n",
                       100.0 * dv / tot, 100.0 * dp / tot);
        }

        /* ---- the recommendation ---------------------------------------- */
        printf("\n  RECOMMENDATION\n");
        if (i_min == 0) {
            printf("  NONE. The smoothed loss was already at its minimum at the "
                   "BOTTOM of the sweep\n"
                   "  (lr = %.3g): this sweep contains no descending region. "
                   "Either --lo is already\n"
                   "  too large, or the data carries no gradient signal. "
                   "Nothing to recommend.\n", row[0].lr);
        } else if (i_min >= n_rows - 1 - (n_rows / 50)) {
            printf("  LOWER BOUND ONLY. The minimum sits at the TOP of the sweep "
                   "(lr = %.3g), so\n"
                   "  [%.3g, %.3g] does not contain the point where the rate stops "
                   "helping. Re-run\n"
                   "  with a larger --hi. On the evidence here a peak of %.3g is "
                   "safe but may be low.\n",
                   row[i_min].lr, lc->lo, lc->hi, row[i_min].lr / 10.0);
        } else {
            const double rec_min10 = row[i_min].lr / 10.0;
            const double rec_steep = row[i_steep].lr;
            const double rec = rec_min10;
            printf("  RULE APPLIED: peak LR = lr(minimum smoothed loss) / 10.\n"
                   "\n"
                   "    lr(min)/10           %.3g   <- the recommendation\n"
                   "    lr(steepest)         %.3g   (cross-check, does not bind)\n",
                   rec_min10, rec_steep);
            printf("\n  WHY THAT RULE. The minimum of a range-test curve is the "
                   "rate at which the step\n"
                   "  has grown large enough that the update stops reducing the "
                   "loss -- it is an upper\n"
                   "  bound, not a setting. The conventional margin is one decade "
                   "below it (fastai's\n"
                   "  lr_find default). Our schedule quotes the PEAK of a cosine "
                   "decay, and a peak has\n"
                   "  to keep working for the whole run on data that keeps moving, "
                   "so the peak is the\n"
                   "  right thing to compare against a range-test recommendation.\n"
                   "  The steepest-descent point is reported but NOT used: it is a "
                   "finite difference of\n"
                   "  a noisy series, so it is a strictly noisier estimator than the "
                   "argmin of the same\n"
                   "  smoothed curve, and on a shallow descent it wanders by a "
                   "decade between seeds.\n");
            if (rec_steep < rec_min10 * 0.5)
                printf("  NOTE: the steepest-descent point (%.3g) is well BELOW the "
                       "recommendation. The\n"
                       "  descent is shallow here; a conservative reading would take "
                       "the lower number.\n", rec_steep);
            if (i_div >= 0)
                printf("  Smith's own upper bound for a CYCLICAL schedule is the "
                       "divergence point, %.3g;\n"
                       "  that is the max of a triangular range, not a fixed rate.\n",
                       row[i_div].lr);
            {   /* what is actually configured */
                const double cur = (double)c->lr;
                const double ratio = (cur > 0.0 && rec > 0.0) ? cur / rec : 0.0;
                printf("\n  AGAINST THE CONFIGURED SCHEDULE (--lr %.3g cosine-"
                       "decayed to %.3g)\n", cur, (double)c->lr_final);
                printf("    configured peak / recommended peak = %.2fx "
                       "(%.2f decades)\n", ratio,
                       ratio > 0.0 ? log10(ratio) : 0.0);
                if (i_div >= 0 && cur >= row[i_div].lr)
                    printf("    the configured peak is AT OR ABOVE the measured "
                           "divergence point. Too high.\n");
                else if (i_min >= 0 && cur > row[i_min].lr)
                    printf("    the configured peak is ABOVE the loss minimum: "
                           "past the point where larger\n"
                           "    steps stop helping. Too high.\n");
                else if (ratio > 3.16)
                    printf("    the configured peak is more than half a decade "
                           "ABOVE the recommendation.\n");
                else if (ratio > 0.0 && ratio < 0.316)
                    printf("    the configured peak is more than half a decade "
                           "BELOW the recommendation.\n");
                else
                    printf("    within half a decade of the recommendation: the "
                           "configured peak is in range.\n");
            }
        }

        lrf_plot(row, n_rows, lc->plot_rows, lc->plot_cols,
                 i_min, i_steep, i_div);
        rc = 0;

    report_done:
        /* ---- CSV --------------------------------------------------------- */
        if (lc->csv) {
            FILE *f = fopen(lc->csv, "w");
            if (!f) {
                fprintf(stderr, "lrfind: cannot write '%s': %s\n",
                        lc->csv, strerror(errno));
            } else {
                fprintf(f, "step,lr,log10_lr,loss_total,loss_policy,loss_value,"
                           "loss_smoothed,policy_smoothed,value_smoothed,"
                           "grad_norm_trunk_preclip,grad_norm_head_preclip,"
                           "batch_positions,finite\n");
                for (int i = 0; i < n_rows; i++) {
                    fprintf(f, "%d,%.10g,%.6f,", i, row[i].lr, log10(row[i].lr));
                    if (row[i].finite)
                        fprintf(f, "%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,",
                                row[i].loss, row[i].pol, row[i].val,
                                row[i].s_tot, row[i].s_pol, row[i].s_val);
                    else
                        fprintf(f, "nan,nan,nan,nan,nan,nan,");
                    fprintf(f, "%.8g,%.8g,%d,%d\n",
                            row[i].gnorm, row[i].hgnorm, row[i].nb, row[i].finite);
                }
                fclose(f);
                printf("\n  csv          %s (%d rows)\n", lc->csv, n_rows);
            }
        }
    }

done:
    if (nspawned > 0) {
        sh.job = JOB_EXIT;
        bar_wait(&sh.bar);
        for (int t = 1; t < nspawned; t++) pthread_join(tids[t], NULL);
    }
    if (workers) {
        for (int t = 0; t < nworkers_init; t++) {
            mcts_free(&workers[t].m);
            rec_free(&workers[t].rec);
            free(workers[t].tg);
            free(workers[t].tg_pol);
            free(workers[t].tg_val);
            free(workers[t].hg_probe);
        }
    }
    if (bar_ready) bar_destroy(&sh.bar);
    if (adam_ready) {
        adam_free(&sh.trunk_adam);
        for (int i = 0; i < nadam; i++) adam_free(&head_adam[i]);
    }
    azbuf_free(&sh.buf);
    free(sh.tstart); free(sh.aoff); free(sh.acount);
    free(sh.batch_tmp); free(sh.batch);
    free(row); free(tids); free(workers); free(head_adam); free(tgsum);
    free(hgrad); free(bslot); free(wslot); free(rank); free(results);
    free(pairs); free(hof); free(elo); free(hypers);
    free(snap_heads); free(heads); free(snap_trunk); free(trunk);
    return rc;
}
