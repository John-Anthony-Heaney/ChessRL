/* train.c -- population-based reinforcement learning driver.
 *
 * Implements docs/ALGORITHM.md:
 *
 *   for g in 1..G:
 *       PAIR    Swiss-like pairing by Elo + a slice of hall-of-fame opponents
 *       PLAY    threads play the pairings, recording trajectories
 *       LEARN   A2C update: per-agent heads + the shared trunk
 *       RATE    Elo from results (hall-of-fame ratings frozen)
 *       EVOLVE  cull the bottom, clone+mutate from the elite, perturb hypers
 *       LOG     one JSON telemetry line; checkpoint periodically
 *
 * Threading model
 * ---------------
 * One shared Trunk, n_agents Heads.  Each worker thread owns
 *   - its own xoshiro RNG, seeded from (cfg->seed, thread_id, generation),
 *   - its own Traj pair, Game, PlayStats,
 *   - a *private* TrunkGrad accumulated over every game it plays and reduced
 *     into one master TrunkGrad after the join,
 *   - a private HeadGrad scratch that is folded into the shared per-agent
 *     HeadGrad under that agent's spin lock once per game side (folding once
 *     per game instead of once per decision keeps lock traffic negligible).
 * Pairings are handed out with an atomic fetch-add, so a thread that draws a
 * batch of short games immediately steals more work instead of idling.
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

#include "train.h"
#include "arena.h"

/* ------------------------------------------------------------------ misc */

#define ELO_K            24.0f
#define ELO_SEED         1500.0f
#define HOF_CAP          64
#define SHUFFLE_WINDOW   8       /* pair within +/- this many Elo ranks      */
#define ANTI_SHUFFLE     (-0.05f)
#define WINNING_MARGIN   3.0f    /* pawns: "clearly winning" for the nudge   */
#define SAVE_EVERY       10
#define MAX_SIDE_PLIES   (MAX_GAME_PLIES / 2 + 8)

_Static_assert(sizeof(Hyper) == 8 * sizeof(float), "Hyper must be 8 packed floats");

/* Learning self-checks.  Off by default so they cost exactly nothing at -O3;
 * build with  make CC='cc -DCHESSRL_DEBUG_LEARN'  to turn them on.  They verify
 * that every advantage/return is finite, that the normalised advantages have
 * ~zero mean and ~unit variance over the generation, and that the per-game
 * means still spread out -- which is what separates generation-wide from
 * per-game normalisation, since both give a pooled mean of zero. */
#ifdef CHESSRL_DEBUG_LEARN
#define DBG_CHECK(cond, ...)                                                   \
    do {                                                                       \
        if (!(cond)) {                                                         \
            fprintf(stderr, "train: self-check failed (%s:%d): ",              \
                    __FILE__, __LINE__);                                       \
            fprintf(stderr, __VA_ARGS__);                                      \
            fputc('\n', stderr);                                               \
            abort();                                                           \
        }                                                                      \
    } while (0)
#else
#define DBG_CHECK(cond, ...) ((void)0)
#endif

static volatile sig_atomic_t g_interrupt = 0;

static void on_sigint(int sig) { (void)sig; g_interrupt = 1; }

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

static inline void cpu_relax(void)
{
#if defined(__aarch64__)
    __asm__ __volatile__("yield" ::: "memory");
#elif defined(__x86_64__) || defined(__i386__)
    __asm__ __volatile__("pause" ::: "memory");
#else
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
#endif
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
    if (t >= 3600)      snprintf(buf, n, "%ldh%02ldm", t / 3600, (t % 3600) / 60);
    else if (t >= 60)   snprintf(buf, n, "%ldm%02lds", t / 60, t % 60);
    else                snprintf(buf, n, "%lds", t);
}

/* ------------------------------------------------------------- spin lock */

typedef struct {
    _Alignas(64) atomic_int v;
    char pad[64 - sizeof(atomic_int)];
} Spin;

static inline void spin_lock(Spin *s)
{
    int expect = 0;
    while (!atomic_compare_exchange_weak_explicit(&s->v, &expect, 1,
                                                  memory_order_acquire,
                                                  memory_order_relaxed)) {
        expect = 0;
        while (atomic_load_explicit(&s->v, memory_order_relaxed) != 0) cpu_relax();
    }
}

static inline void spin_unlock(Spin *s)
{
    atomic_store_explicit(&s->v, 0, memory_order_release);
}

/* ------------------------------------------------------- hyper-parameters */

typedef struct { float lo, hi; } Range;

/* Same order as the fields of Hyper. */
static const Range HYP_RANGE[8] = {
    { 0.20f,  3.00f  },   /* temperature  */
    { 1e-4f,  0.20f  },   /* entropy_coef */
    { 0.10f,  4.00f  },   /* lr_scale     */
    { 0.02f,  1.50f  },   /* shaping      */
    { 0.05f,  2.00f  },   /* value_coef   */
    { 0.90f,  0.999f },   /* gamma        */
    { 0.70f,  0.99f  },   /* lambda       */
    { 0.002f, 0.20f  },   /* mutate_sigma */
};

static void hyper_clamp(Hyper *h)
{
    float *p = (float *)h;
    for (int i = 0; i < 8; i++) {
        if (!isfinite(p[i]) || p[i] < HYP_RANGE[i].lo) p[i] = HYP_RANGE[i].lo;
        else if (p[i] > HYP_RANGE[i].hi)               p[i] = HYP_RANGE[i].hi;
    }
}

/* Every hyper-parameter multiplied by exp(N(0, 0.2)) and clamped. */
static void hyper_mutate(Hyper *h, uint64_t *rng)
{
    float *p = (float *)h;
    for (int i = 0; i < 8; i++) p[i] *= expf(0.2f * rng_normal(rng));
    hyper_clamp(h);
}

/* ------------------------------------------------------------ head tensors */

typedef struct { size_t off, len; } TensorSpan;

#define TSPAN(field, n) { offsetof(Head, field) / sizeof(float), (size_t)(n) }

static const TensorSpan HEAD_TENSORS[] = {
    TSPAN(z,      NF_ACC),
    TSPAN(Wv,     NF_HID),
    TSPAN(bv,     1),
    TSPAN(Wp,     NF_HID * NF_PDIM),
    TSPAN(Efrom,  64 * NF_PDIM),
    TSPAN(Eto,    64 * NF_PDIM),
    TSPAN(Epc,    6 * NF_PDIM),
    TSPAN(Epromo, 5 * NF_PDIM),
    TSPAN(Ecap,   7 * NF_PDIM),
    TSPAN(Bft,    64 * 64),
};
#define N_HEAD_TENSORS ((int)(sizeof(HEAD_TENSORS) / sizeof(HEAD_TENSORS[0])))

/* Per-tensor N(0, sigma * rms(tensor)) perturbation. */
static void head_mutate(Head *h, float sigma, uint64_t *rng)
{
    float *base = (float *)h;
    for (int t = 0; t < N_HEAD_TENSORS; t++) {
        float *v = base + HEAD_TENSORS[t].off;
        size_t n = HEAD_TENSORS[t].len;
        double ss = 0.0;
        for (size_t i = 0; i < n; i++) ss += (double)v[i] * (double)v[i];
        float rms = (float)sqrt(ss / (double)n);
        if (!(rms > 0.0f)) continue;            /* dead tensor: nothing to scale */
        float s = sigma * rms;
        for (size_t i = 0; i < n; i++) v[i] += s * rng_normal(rng);
    }
}

/* ------------------------------------------------------------ hall of fame */

typedef struct {
    Head  head;
    Hyper hy;
    float elo;
    int   gen;
    int   src;
} HofEntry;

/* ------------------------------------------------------------- pairings */

/* id >= 0: live agent index.  id < 0: hall-of-fame entry -(id + 1). */
typedef struct { int32_t white, black; } Pairing;
typedef struct { int16_t result, reason; } GameRes;

#define IS_HOF(id)   ((id) < 0)
#define HOF_IDX(id)  (-(id) - 1)

/* ------------------------------------------------------------- run state */

typedef struct Shared Shared;

typedef struct {
    Shared  *sh;
    int      tid;
    uint64_t rng[4];
    Traj     tw, tb;
    Game     game;
    PlayStats stats;
    TrunkGrad *tg;          /* private trunk gradient                        */
    HeadGrad  *hscratch;    /* private per-game head gradient                */
    Fwd      *fw;           /* cached forward passes for one player's subseq */
    int      *ix;           /* decision indices                              */
    float    *adv, *ret;
    /* running advantage statistics for this generation (Welford)            */
    double    adv_n, adv_mean, adv_m2;
#ifdef CHESSRL_DEBUG_LEARN
    /* post-normalisation moments, checked once per generation                */
    double    dbg_n, dbg_sum, dbg_sumsq;
    /* moments of the PER-SUBSEQUENCE means of the normalised advantages:
     * these must spread out, or the normalisation is happening per game       */
    double    dbg_gn, dbg_gsum, dbg_gsumsq;
#endif
    /* telemetry                                                             */
    double    l_pol, l_val, l_ent;
    uint64_t  l_n, decisions;
} Worker;

struct Shared {
    const TrainCfg *cfg;
    Trunk    *trunk;
    Head     *heads;
    Hyper    *hypers;
    float    *elo;
    HofEntry *hof;
    int       hof_n, hof_next;
    Pairing  *pairs;
    GameRes  *results;
    int       npairs;
    HeadGrad *hgrad;
    Spin     *hlock;
    uint64_t *hcount;       /* decision points per agent, under hlock        */
    atomic_int next;
    int       generation;
};

/* --------------------------------------------------------------- helpers */

static void resolve_side(const Shared *sh, int32_t id, const Head **h, const Hyper **y)
{
    if (IS_HOF(id)) {
        const HofEntry *e = &sh->hof[HOF_IDX(id)];
        *h = &e->head;
        *y = &e->hy;
    } else {
        *h = &sh->heads[id];
        *y = &sh->hypers[id];
    }
}

/* Terminal reward from `color`'s view, including the anti-shuffle nudge. */
static float terminal_reward(int result, int reason, float mat_white, int color)
{
    if (result == GR_WHITE_WIN) return (color == WHITE) ? 1.0f : -1.0f;
    if (result == GR_BLACK_WIN) return (color == BLACK) ? 1.0f : -1.0f;
    /* draw (or, defensively, an unterminated game) */
    if (reason == TR_MAX_PLIES || reason == TR_REPETITION) {
        float mc = (color == WHITE) ? mat_white : -mat_white;
        if (mc >= WINNING_MARGIN) return ANTI_SHUFFLE;
    }
    return 0.0f;
}

/* ------------------------------------------------------------------ learn */
/* A2C update for one player's subsequence of one game.
 *
 *   Phi(s)   = tanh(material_balance_white(s) / 5)          (recorded by arena)
 *   Phi_c    = Phi for White, -Phi for Black, Phi(terminal) = 0
 *   r_t      = shaping * (gamma * Phi_c(s_t') - Phi_c(s_t))  [+ z at the last]
 *   A_t      = GAE(lambda) over the player's own decision points
 *   L        = -A * log pi(a|s) - beta * H(pi) + c_v * (v - R)^2
 */
static void learn_side(Worker *w, int agent, const Traj *tr, int color,
                       float term_r, const Hyper *hy)
{
    Shared *sh = w->sh;
    const Head *h = &sh->heads[agent];
    int n = 0;

    for (int k = 0; k < tr->ply && n < MAX_SIDE_PLIES; k++)
        if ((int)tr->mover[k] == color) w->ix[n++] = k;
    if (n == 0) return;

    /* ---- forward passes (cached: reused by the backward sweep) ---------- */
    for (int j = 0; j < n; j++) {
        int k = w->ix[j];
        nn_eval(sh->trunk, h, tr->fidx[k], (int)tr->nf[k], &w->fw[j]);
    }

    /* ---- shaped rewards + GAE(lambda), backwards ------------------------ */
    {
        float g = hy->gamma, lam = hy->lambda, shp = hy->shaping;
        float lastgae = 0.0f;
        for (int j = n - 1; j >= 0; j--) {
            float phi_j    = tr->phi[w->ix[j]];
            float phi_next = 0.0f;                  /* Phi(terminal) == 0    */
            float vnext    = 0.0f;                  /* V(terminal)   == 0    */
            float r, delta;
            if (color == BLACK) phi_j = -phi_j;
            if (j + 1 < n) {
                phi_next = tr->phi[w->ix[j + 1]];
                if (color == BLACK) phi_next = -phi_next;
                vnext = w->fw[j + 1].v;
            }
            r = shp * (g * phi_next - phi_j);
            if (j == n - 1) r += term_r;
            delta   = r + g * vnext - w->fw[j].v;
            lastgae = delta + g * lam * lastgae;
            w->adv[j] = lastgae;
            w->ret[j] = lastgae + w->fw[j].v;
        }
    }

    /* ---- advantage normalisation (running, per generation, per thread) --- */
    float amean, ainv;
    {
        double var;
        for (int j = 0; j < n; j++) {
            double x = (double)w->adv[j], d;
            w->adv_n += 1.0;
            d = x - w->adv_mean;
            w->adv_mean += d / w->adv_n;
            w->adv_m2 += d * (x - w->adv_mean);
        }
        var = (w->adv_n > 1.0) ? w->adv_m2 / (w->adv_n - 1.0) : 0.0;
        amean = (float)w->adv_mean;
        ainv = 1.0f / ((float)sqrt(var) + 1e-6f);
        if (!(ainv < 100.0f)) ainv = 100.0f;        /* also catches NaN      */
    }

#ifdef CHESSRL_DEBUG_LEARN
    {
        double gm = 0.0;
        for (int j = 0; j < n; j++) {
            double a = ((double)w->adv[j] - (double)amean) * (double)ainv;
            DBG_CHECK(isfinite(w->adv[j]), "advantage %d/%d = %g", j, n, (double)w->adv[j]);
            DBG_CHECK(isfinite(w->ret[j]), "return %d/%d = %g", j, n, (double)w->ret[j]);
            DBG_CHECK(isfinite(a), "normalised advantage %d/%d = %g", j, n, a);
            w->dbg_n += 1.0;
            w->dbg_sum += a;
            w->dbg_sumsq += a * a;
            gm += a;
        }
        if (n >= 4) {
            gm /= (double)n;
            w->dbg_gn += 1.0;
            w->dbg_gsum += gm;
            w->dbg_gsumsq += gm * gm;
        }
    }
#endif

    /* ---- per-decision-point gradients ---------------------------------- */
    grad_zero(w->hscratch, (int)HEAD_NPARAM);
    {
        float T = hy->temperature;
        float invT, beta = hy->entropy_coef, cv = hy->value_coef;
        if (!(T > 0.05f)) T = 0.05f;
        invT = 1.0f / T;

        for (int j = 0; j < n; j++) {
            int k = w->ix[j];
            int nm = (int)tr->nmoves[k];
            const MoveKey *keys = tr->keys + tr->koff[k];
            const Fwd *fw = &w->fw[j];
            float logits[MAX_MOVES], pr[MAX_MOVES], dl[MAX_MOVES];
            float H = 0.0f, A, v, dv, lpa;
            int a = (int)tr->chosen[k];

            if (nm <= 0 || nm > MAX_MOVES) continue;
            if (a < 0 || a >= nm) a = 0;

            nn_logits(h, fw, keys, nm, logits);
            softmax_t(logits, nm, T, pr);

            for (int m = 0; m < nm; m++)
                if (pr[m] > 1e-12f) H -= pr[m] * logf(pr[m]);

            A = (w->adv[j] - amean) * ainv;

            for (int m = 0; m < nm; m++) {
                float pm = pr[m];
                float lp = (pm > 1e-12f) ? logf(pm) : -27.63102f;
                /* d/dlogit of  -A log pi_a  is  (A/T)(pi_m - [m == a]) */
                float d = A * invT * (pm - (m == a ? 1.0f : 0.0f));
                /* d/dlogit of  -beta H      is  (beta/T) pi_m (log pi_m + H) */
                d += beta * invT * pm * (lp + H);
                dl[m] = d;
            }

            v = fw->v;
            dv = 2.0f * cv * (v - w->ret[j]);

            nn_backward(sh->trunk, h, fw, tr->fidx[k], (int)tr->nf[k],
                        keys, nm, dl, dv, w->tg, w->hscratch);

            lpa = (pr[a] > 1e-12f) ? logf(pr[a]) : -27.63102f;
            w->l_pol += -(double)A * (double)lpa;
            w->l_val += (double)(v - w->ret[j]) * (double)(v - w->ret[j]);
            w->l_ent += (double)H;
            w->l_n++;
        }
    }

    /* ---- fold into the shared per-agent gradient ------------------------ */
    spin_lock(&sh->hlock[agent]);
    grad_add((float *)&sh->hgrad[agent], (const float *)w->hscratch, (int)HEAD_NPARAM);
    sh->hcount[agent] += (uint64_t)n;
    spin_unlock(&sh->hlock[agent]);
    w->decisions += (uint64_t)n;
}

/* ------------------------------------------------------------ play thread */

static void *worker_main(void *arg)
{
    Worker *w = (Worker *)arg;
    Shared *sh = w->sh;
    const TrainCfg *c = sh->cfg;

    for (;;) {
        int i = atomic_fetch_add_explicit(&sh->next, 1, memory_order_relaxed);
        Pairing p;
        const Head *hw, *hb;
        const Hyper *yw, *yb;
        PlayCfg pc;
        int reason = TR_NONE, res;
        float mb, matw;

        if (i >= sh->npairs) break;
        p = sh->pairs[i];
        resolve_side(sh, p.white, &hw, &yw);
        resolve_side(sh, p.black, &hb, &yb);

        pc.max_plies     = c->max_plies;
        pc.greedy        = 0;
        pc.temp_white    = yw->temperature;
        pc.temp_black    = yb->temperature;
        pc.opening_temp  = yw->temperature;   /* unused: opening_plies == 0   */
        pc.opening_plies = 0;                 /* keep training strictly on-policy */
        pc.record        = 1;
        pc.rng           = w->rng;

        traj_reset(&w->tw);
        traj_reset(&w->tb);

        res = play_game(sh->trunk, hw, hb, yw, yb, &pc, &w->tw, &w->tb,
                        &w->stats, &w->game, &reason);

        sh->results[i].result = (int16_t)res;
        sh->results[i].reason = (int16_t)reason;

        mb = material_balance(&w->game.pos);
        matw = (w->game.pos.side == WHITE) ? mb : -mb;

        if (!IS_HOF(p.white))
            learn_side(w, (int)p.white, &w->tw, WHITE,
                       terminal_reward(res, reason, matw, WHITE), yw);
        if (!IS_HOF(p.black))
            learn_side(w, (int)p.black, &w->tb, BLACK,
                       terminal_reward(res, reason, matw, BLACK), yb);
    }
    return NULL;
}

/* ------------------------------------------------------------- pairings */

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
    float x = *(const float *)a, y = *(const float *)b;
    return (x > y) - (x < y);
}

/* Local shuffle: every slot swaps with a uniformly chosen slot within +/- win,
 * which keeps opponents Elo-adjacent while adding real jitter. */
static void window_shuffle(int32_t *v, int n, int win, uint64_t *rng)
{
    if (win < 1) win = 1;
    for (int t = 0; t < n; t++) {
        int lo = t - win, hi = t + win, u;
        int32_t tmp;
        if (lo < 0) lo = 0;
        if (hi > n - 1) hi = n - 1;
        u = lo + (int)(rng_next(rng) % (uint64_t)(hi - lo + 1));
        tmp = v[t]; v[t] = v[u]; v[u] = tmp;
    }
}

static int build_pairings(Shared *sh, EloRank *rank, int32_t *wslot, int32_t *bslot,
                          uint64_t *rng)
{
    const TrainCfg *c = sh->cfg;
    int n = c->n_agents;
    int g = c->games_per_agent / 2;
    int np, win;

    if (g < 1) g = 1;
    np = n * g;

    for (int i = 0; i < n; i++) { rank[i].elo = sh->elo[i]; rank[i].idx = i; }
    qsort(rank, (size_t)n, sizeof(EloRank), cmp_rank_desc);

    /* rank-major slot lists: every agent appears g times as White and g as Black */
    for (int r = 0; r < n; r++)
        for (int k = 0; k < g; k++) {
            wslot[r * g + k] = rank[r].idx;
            bslot[r * g + k] = rank[r].idx;
        }

    win = SHUFFLE_WINDOW * g;
    window_shuffle(wslot, np, win, rng);
    window_shuffle(bslot, np, win, rng);

    /* Avoid self-pairings by swapping the Black slot with a nearby one. */
    for (int t = 0; t < np; t++) {
        if (wslot[t] != bslot[t]) continue;
        for (int d = 1; d <= win; d++) {
            int u = t + d < np ? t + d : t - d;
            if (u < 0 || u >= np) continue;
            if (wslot[u] != bslot[t] && wslot[t] != bslot[u]) {
                int32_t tmp = bslot[t]; bslot[t] = bslot[u]; bslot[u] = tmp;
                break;
            }
        }
    }

    for (int t = 0; t < np; t++) {
        sh->pairs[t].white = wslot[t];
        sh->pairs[t].black = bslot[t];
    }

    /* Route hof_frac_pct of the games against a random hall-of-fame opponent;
     * the replaced colour is chosen at random so colour balance is preserved. */
    if (sh->hof_n > 0 && c->hof_frac_pct > 0) {
        for (int t = 0; t < np; t++) {
            if ((int)(rng_next(rng) % 100u) >= c->hof_frac_pct) continue;
            int32_t hid = -(int32_t)(rng_next(rng) % (uint64_t)sh->hof_n) - 1;
            if (rng_next(rng) & 1u) sh->pairs[t].white = hid;
            else                    sh->pairs[t].black = hid;
        }
    }
    return np;
}

/* ------------------------------------------------------------------- Elo */

static void apply_elo(Shared *sh)
{
    for (int i = 0; i < sh->npairs; i++) {
        Pairing p = sh->pairs[i];
        int res = sh->results[i].result;
        float sw, rw, rb, ew;

        if (p.white == p.black) continue;                   /* self-play      */
        sw = (res == GR_WHITE_WIN) ? 1.0f : (res == GR_BLACK_WIN) ? 0.0f : 0.5f;
        rw = IS_HOF(p.white) ? sh->hof[HOF_IDX(p.white)].elo : sh->elo[p.white];
        rb = IS_HOF(p.black) ? sh->hof[HOF_IDX(p.black)].elo : sh->elo[p.black];
        ew = 1.0f / (1.0f + powf(10.0f, (rb - rw) / 400.0f));

        if (!IS_HOF(p.white)) sh->elo[p.white] += ELO_K * (sw - ew);
        if (!IS_HOF(p.black)) sh->elo[p.black] += ELO_K * ((1.0f - sw) - (1.0f - ew));
    }
}

/* ------------------------------------------------------------- telemetry */

static void jnum(FILE *f, double x)
{
    if (!isfinite(x)) { fputs("0", f); return; }
    fprintf(f, "%.6g", x);
}

static double entropy_bits(const uint64_t *counts, int n)
{
    double tot = 0.0, h = 0.0;
    for (int i = 0; i < n; i++) tot += (double)counts[i];
    if (tot <= 0.0) return -1.0;
    for (int i = 0; i < n; i++) {
        double p = (double)counts[i] / tot;
        if (p > 0.0) h -= p * (log(p) / log(2.0));
    }
    return h;
}

static void write_telemetry(FILE *f, const Shared *sh, const PlayStats *st,
                            double sec, double elo_best, double elo_mean,
                            double elo_p10, int best_i,
                            double lp, double lv, double le, double gnorm)
{
    double games = (double)st->games;
    double gd = games > 0.0 ? games : 1.0;
    const Hyper *bh = &sh->hypers[best_i];

    fprintf(f, "{\"gen\":%d", sh->generation);
    fprintf(f, ",\"games\":%llu", (unsigned long long)st->games);
    fprintf(f, ",\"plies\":%llu", (unsigned long long)st->plies);
    fprintf(f, ",\"sec\":");  jnum(f, sec);
    fprintf(f, ",\"gps\":");  jnum(f, games / (sec > 0.0 ? sec : 1.0));
    fprintf(f, ",\"elo_best\":"); jnum(f, elo_best);
    fprintf(f, ",\"elo_mean\":"); jnum(f, elo_mean);
    fprintf(f, ",\"elo_p10\":");  jnum(f, elo_p10);
    fprintf(f, ",\"white_win\":"); jnum(f, (double)st->white_wins / gd);
    fprintf(f, ",\"black_win\":"); jnum(f, (double)st->black_wins / gd);
    fprintf(f, ",\"draw\":");      jnum(f, (double)st->draws / gd);
    fprintf(f, ",\"avg_len\":");   jnum(f, st->sum_len / gd);
    fprintf(f, ",\"captures_per_game\":"); jnum(f, (double)st->captures / gd);
    fprintf(f, ",\"checks_per_game\":");   jnum(f, (double)st->checks / gd);
    fprintf(f, ",\"castle_rate\":");
    jnum(f, (double)(st->castles_k + st->castles_q) / gd);
    fprintf(f, ",\"promo_rate\":"); jnum(f, (double)st->promotions / gd);
    fprintf(f, ",\"ep_rate\":");    jnum(f, (double)st->ep_captures / gd);

    fprintf(f, ",\"term\":{\"checkmate\":");   jnum(f, (double)st->checkmates / gd);
    fprintf(f, ",\"stalemate\":");             jnum(f, (double)st->stalemates / gd);
    fprintf(f, ",\"fifty\":");                 jnum(f, (double)st->fifty / gd);
    fprintf(f, ",\"repetition\":");            jnum(f, (double)st->repetition / gd);
    fprintf(f, ",\"insufficient\":");          jnum(f, (double)st->insufficient / gd);
    fprintf(f, ",\"maxplies\":");              jnum(f, (double)st->maxplies / gd);
    fputc('}', f);

    /* top-8 White first moves */
    {
        int top[8];
        int ntop = 0;
        for (int i = 0; i < 64 * 64; i++) {
            uint64_t c = st->first_move[i];
            int pos;
            if (c == 0) continue;
            /* Scan only -- the shift below must happen exactly once, or an
             * entry moved more than one slot gets copied twice and appears
             * twice in the output. */
            pos = ntop;
            while (pos > 0 && st->first_move[top[pos - 1]] < c) pos--;
            if (pos >= 8) continue;              /* below the whole current top-8 */
            if (ntop < 8) ntop++;
            for (int q = ntop - 1; q > pos; q--) top[q] = top[q - 1];
            top[pos] = i;
        }
        fprintf(f, ",\"opening_top\":[");
        for (int i = 0; i < ntop; i++) {
            int from = top[i] >> 6, to = top[i] & 63;
            fprintf(f, "%s{\"move\":\"%s%s\",\"n\":%llu}", i ? "," : "",
                    SQ_NAMES[from], SQ_NAMES[to],
                    (unsigned long long)st->first_move[top[i]]);
        }
        fputc(']', f);
    }

    /* mean destination-square entropy over the piece types that moved */
    {
        double sum = 0.0;
        int cnt = 0;
        for (int p = 0; p < 6; p++) {
            double h = entropy_bits(st->piece_dest[p], 64);
            if (h >= 0.0) { sum += h; cnt++; }
        }
        fprintf(f, ",\"piece_dest_entropy\":");
        jnum(f, cnt ? sum / (double)cnt : 0.0);
    }

    fprintf(f, ",\"avg_final_material\":");
    jnum(f, st->sum_final_material / gd);

    fprintf(f, ",\"best_agent\":{\"i\":%d,\"elo\":", best_i);
    jnum(f, elo_best);
    fprintf(f, ",\"temperature\":");  jnum(f, bh->temperature);
    fprintf(f, ",\"entropy_coef\":"); jnum(f, bh->entropy_coef);
    fprintf(f, ",\"lr_scale\":");     jnum(f, bh->lr_scale);
    fprintf(f, ",\"shaping\":");      jnum(f, bh->shaping);
    fputc('}', f);

    fprintf(f, ",\"loss\":{\"policy\":"); jnum(f, lp);
    fprintf(f, ",\"value\":");            jnum(f, lv);
    fprintf(f, ",\"entropy\":");          jnum(f, le);
    fputc('}', f);

    fprintf(f, ",\"grad_norm\":"); jnum(f, gnorm);
    fputs("}\n", f);
    fflush(f);
}

/* --------------------------------------------------------------- defaults */

void train_default_cfg(TrainCfg *c)
{
    if (!c) return;
    memset(c, 0, sizeof *c);
    c->n_agents        = 256;
    c->generations     = 300;
    c->games_per_agent = 16;
    c->threads         = cpu_count();
    c->max_plies       = 240;
    /* One optimiser step per generation, so the learning rates are large by
     * deep-learning standards: 300 generations == 300 Adam steps. */
    c->base_lr         = 0.005f;
    c->trunk_lr        = 0.002f;
    c->weight_decay    = 1e-5f;
    c->grad_clip       = 1.0f;
    c->elite_frac      = 0.25f;
    c->cull_frac       = 0.20f;
    c->hof_every       = 10;
    c->hof_frac_pct    = 15;
    c->seed            = 20260906u;
    c->run_dir         = "runs/default";
    c->eval_games      = 200;
    c->quiet           = 0;
}

/* --------------------------------------------------------------- training */

static void scale_grad(float *g, size_t n, float s)
{
    for (size_t i = 0; i < n; i++) g[i] *= s;
}

int train_run(TrainCfg *c)
{
    Trunk    *trunk = NULL;
    Head     *heads = NULL;
    Hyper    *hypers = NULL;
    float    *elo = NULL, *elo_sorted = NULL;
    HofEntry *hof = NULL;
    HeadGrad *hgrad = NULL;
    TrunkGrad *tgsum = NULL;
    Spin     *hlock = NULL;
    uint64_t *hcount = NULL;
    Pairing  *pairs = NULL;
    GameRes  *results = NULL;
    EloRank  *rank = NULL;
    int32_t  *wslot = NULL, *bslot = NULL;
    Worker   *workers = NULL;
    pthread_t *tids = NULL;
    Adam      trunk_adam;
    Adam     *head_adam = NULL;
    FILE     *tel = NULL;
    char      path[1200];
    uint64_t  master_rng[4];
    double    t_run0, ema_gen = 0.0;
    double    best_saved = -1e30;
    uint64_t  total_games = 0, total_plies = 0;
    int       n, gpa, nthreads, npairs, rc = 1;
    int       adam_ready = 0, nadam = 0;
    void    (*old_sigint)(int) = SIG_DFL;
    Shared    sh;

    if (!c) return 1;
    chess_init();

    /* ---- sanitise the configuration ------------------------------------ */
    if (c->n_agents < 2)        c->n_agents = 2;
    if (c->generations < 1)     c->generations = 1;
    if (c->games_per_agent < 2) c->games_per_agent = 2;
    c->games_per_agent &= ~1;                     /* half as White exactly    */
    if (c->threads < 1)         c->threads = cpu_count();
    if (c->max_plies < 2)       c->max_plies = 240;
    if (c->max_plies > MAX_GAME_PLIES) c->max_plies = MAX_GAME_PLIES;
    if (!(c->base_lr > 0.0f))   c->base_lr = 0.005f;
    if (!(c->trunk_lr > 0.0f))  c->trunk_lr = 0.002f;
    if (!(c->weight_decay >= 0.0f)) c->weight_decay = 0.0f;
    if (!(c->elite_frac > 0.0f) || c->elite_frac > 1.0f) c->elite_frac = 0.25f;
    if (!(c->cull_frac >= 0.0f) || c->cull_frac > 0.9f)  c->cull_frac = 0.20f;
    if (c->hof_every < 1)       c->hof_every = 10;
    if (c->hof_frac_pct < 0)    c->hof_frac_pct = 0;
    if (c->hof_frac_pct > 100)  c->hof_frac_pct = 100;
    if (!c->run_dir || !*c->run_dir) c->run_dir = "runs/default";

    n        = c->n_agents;
    gpa      = c->games_per_agent;
    npairs   = n * (gpa / 2);
    nthreads = c->threads;
    if (nthreads > npairs) nthreads = npairs;
    if (nthreads < 1) nthreads = 1;

    if (mkdir_p(c->run_dir) != 0) {
        fprintf(stderr, "train: cannot create run directory '%s': %s\n",
                c->run_dir, strerror(errno));
        return 1;
    }

    /* ---- allocate ------------------------------------------------------- */
    trunk      = (Trunk *)     calloc(1, sizeof(Trunk));
    heads      = (Head *)      calloc((size_t)n, sizeof(Head));
    hypers     = (Hyper *)     calloc((size_t)n, sizeof(Hyper));
    elo        = (float *)     calloc((size_t)n, sizeof(float));
    elo_sorted = (float *)     calloc((size_t)n, sizeof(float));
    hof        = (HofEntry *)  calloc((size_t)HOF_CAP, sizeof(HofEntry));
    hgrad      = (HeadGrad *)  calloc((size_t)n, sizeof(HeadGrad));
    tgsum      = (TrunkGrad *) calloc(1, sizeof(TrunkGrad));
    hlock      = (Spin *)      calloc((size_t)n, sizeof(Spin));
    hcount     = (uint64_t *)  calloc((size_t)n, sizeof(uint64_t));
    pairs      = (Pairing *)   calloc((size_t)npairs, sizeof(Pairing));
    results    = (GameRes *)   calloc((size_t)npairs, sizeof(GameRes));
    rank       = (EloRank *)   calloc((size_t)n, sizeof(EloRank));
    wslot      = (int32_t *)   calloc((size_t)npairs, sizeof(int32_t));
    bslot      = (int32_t *)   calloc((size_t)npairs, sizeof(int32_t));
    workers    = (Worker *)    calloc((size_t)nthreads, sizeof(Worker));
    tids       = (pthread_t *) calloc((size_t)nthreads, sizeof(pthread_t));
    head_adam  = (Adam *)      calloc((size_t)n, sizeof(Adam));

    if (!trunk || !heads || !hypers || !elo || !elo_sorted || !hof || !hgrad ||
        !tgsum || !hlock || !hcount || !pairs || !results || !rank || !wslot ||
        !bslot || !workers || !tids || !head_adam) {
        fprintf(stderr, "train: out of memory\n");
        goto done;
    }

    for (int i = 0; i < n; i++) atomic_init(&hlock[i].v, 0);

    /* ---- initialise the population ------------------------------------- */
    nn_init(trunk, &heads[0], c->seed);
    for (int i = 1; i < n; i++) {
        Trunk scratch;                            /* per-agent head init only */
        nn_init(&scratch, &heads[i], c->seed + 0x9E3779B97F4A7C15ull * (uint64_t)(i + 1));
    }
    for (int i = 0; i < n; i++) {
        hyper_default(&hypers[i]);
        hyper_clamp(&hypers[i]);
        elo[i] = ELO_SEED;
    }
    /* Spread the starting hypers so PBT has something to select on. */
    rng_seed(master_rng, c->seed ^ 0xD1B54A32D192ED03ull);
    for (int i = 0; i < n; i++) hyper_mutate(&hypers[i], master_rng);

    adam_init(&trunk_adam, (int)TRUNK_NPARAM);
    for (nadam = 0; nadam < n; nadam++) adam_init(&head_adam[nadam], (int)HEAD_NPARAM);
    adam_ready = 1;

    for (int t = 0; t < nthreads; t++) {
        Worker *w = &workers[t];
        w->sh  = &sh;
        w->tid = t;
        traj_init(&w->tw);
        traj_init(&w->tb);
        w->tg       = (TrunkGrad *)calloc(1, sizeof(TrunkGrad));
        w->hscratch = (HeadGrad *) calloc(1, sizeof(HeadGrad));
        w->fw       = (Fwd *)      calloc((size_t)MAX_SIDE_PLIES, sizeof(Fwd));
        w->ix       = (int *)      calloc((size_t)MAX_SIDE_PLIES, sizeof(int));
        w->adv      = (float *)    calloc((size_t)MAX_SIDE_PLIES, sizeof(float));
        w->ret      = (float *)    calloc((size_t)MAX_SIDE_PLIES, sizeof(float));
        if (!w->tg || !w->hscratch || !w->fw || !w->ix || !w->adv || !w->ret) {
            fprintf(stderr, "train: out of memory (worker %d)\n", t);
            goto done;
        }
    }

    sh.cfg     = c;
    sh.trunk   = trunk;
    sh.heads   = heads;
    sh.hypers  = hypers;
    sh.elo     = elo;
    sh.hof     = hof;
    sh.hof_n   = 0;
    sh.hof_next = 0;
    sh.pairs   = pairs;
    sh.results = results;
    sh.npairs  = npairs;
    sh.hgrad   = hgrad;
    sh.hlock   = hlock;
    sh.hcount  = hcount;
    sh.generation = 0;
    atomic_init(&sh.next, 0);

    snprintf(path, sizeof path, "%s/telemetry.jsonl", c->run_dir);
    tel = fopen(path, "a");
    if (!tel) {
        fprintf(stderr, "train: cannot open '%s': %s\n", path, strerror(errno));
        goto done;
    }

    old_sigint = signal(SIGINT, on_sigint);

    if (!c->quiet) {
        printf("chessrl train: %d agents, %d generations, %d games/agent "
               "(%d games/gen), %d threads, max %d plies\n",
               n, c->generations, gpa, npairs, nthreads, c->max_plies);
        printf("               lr head %.4g  trunk %.4g  wd %.3g  clip %.3g  "
               "elite %.0f%%  cull %.0f%%  hof every %d (%d%% of games)  seed %llu\n",
               (double)c->base_lr, (double)c->trunk_lr, (double)c->weight_decay,
               (double)c->grad_clip, (double)c->elite_frac * 100.0,
               (double)c->cull_frac * 100.0, c->hof_every, c->hof_frac_pct,
               (unsigned long long)c->seed);
        printf("               run dir %s\n", c->run_dir);
        fflush(stdout);
    }

    t_run0 = now_sec();

    /* =================================================================== */
    for (int gen = 1; gen <= c->generations; gen++) {
        double t0 = now_sec(), sec;
        PlayStats st;
        double lp = 0.0, lv = 0.0, le = 0.0, gnorm = 0.0;
        uint64_t ln = 0, total_dec = 0;
        double elo_best, elo_mean, elo_p10;
        int best_i = 0;

        sh.generation = gen;

        /* -- 1. PAIR ---------------------------------------------------- */
        rng_seed(master_rng, c->seed + 0x2545F4914F6CDD1Dull * (uint64_t)gen);
        sh.npairs = build_pairings(&sh, rank, wslot, bslot, master_rng);
        memset(results, 0, (size_t)sh.npairs * sizeof(GameRes));
        memset(hcount, 0, (size_t)n * sizeof(uint64_t));
        atomic_store(&sh.next, 0);

        /* -- 2. PLAY + 3. LEARN ----------------------------------------- */
        for (int t = 0; t < nthreads; t++) {
            Worker *w = &workers[t];
            rng_seed(w->rng, c->seed
                     + 0x9E3779B97F4A7C15ull * (uint64_t)(t + 1)
                     + 0xBF58476D1CE4E5B9ull * (uint64_t)gen);
            stats_zero(&w->stats);
            grad_zero(w->tg, (int)TRUNK_NPARAM);
            w->adv_n = w->adv_mean = w->adv_m2 = 0.0;
#ifdef CHESSRL_DEBUG_LEARN
            w->dbg_n = w->dbg_sum = w->dbg_sumsq = 0.0;
            w->dbg_gn = w->dbg_gsum = w->dbg_gsumsq = 0.0;
#endif
            w->l_pol = w->l_val = w->l_ent = 0.0;
            w->l_n = 0;
            w->decisions = 0;
        }
        for (int t = 1; t < nthreads; t++)
            pthread_create(&tids[t], NULL, worker_main, &workers[t]);
        worker_main(&workers[0]);
        for (int t = 1; t < nthreads; t++) pthread_join(tids[t], NULL);

        stats_zero(&st);
        for (int t = 0; t < nthreads; t++) {
            stats_merge(&st, &workers[t].stats);
            lp += workers[t].l_pol;
            lv += workers[t].l_val;
            le += workers[t].l_ent;
            ln += workers[t].l_n;
            total_dec += workers[t].decisions;
        }
        if (ln) { lp /= (double)ln; lv /= (double)ln; le /= (double)ln; }

#ifdef CHESSRL_DEBUG_LEARN
        /* Advantages must be finite, centred and unit-scaled over the whole
         * generation.  The two moment checks catch a sign flip, a missing mean
         * subtraction or a NaN; the third catches per-game normalisation. */
        for (int t = 0; t < nthreads; t++) {
            const Worker *w = &workers[t];
            double m, var;
            if (w->dbg_n < 256.0) continue;         /* too few points to judge */
            m   = w->dbg_sum / w->dbg_n;
            var = w->dbg_sumsq / w->dbg_n - m * m;
            /* Measured over many runs: |mean| stays under 0.18 and the variance
             * inside [0.74, 1.35]; the estimator is streaming, so the tolerance
             * has to absorb its drift.  A missing mean subtraction or a flipped
             * sign puts the mean at order 1, far outside this. */
            DBG_CHECK(isfinite(m) && fabs(m) < 0.40,
                      "gen %d thread %d: normalised advantage mean %g over %.0f points",
                      gen, t, m, w->dbg_n);
            DBG_CHECK(isfinite(var) && var > 0.25 && var < 4.0,
                      "gen %d thread %d: normalised advantage variance %g over %.0f points",
                      gen, t, var, w->dbg_n);
            /* Between-subsequence spread.  Pooled mean/variance cannot tell
             * per-game from per-generation normalisation (per-game gives a
             * pooled mean of 0 too); this can: per-game normalisation forces
             * every subsequence mean to 0, so their variance collapses. */
            if (w->dbg_gn >= 32.0) {
                double gm  = w->dbg_gsum / w->dbg_gn;
                double gvar = w->dbg_gsumsq / w->dbg_gn - gm * gm;
                DBG_CHECK(isfinite(gvar) && gvar > 1e-3,
                          "gen %d thread %d: per-subsequence advantage means have "
                          "variance %g over %.0f subsequences -- advantages look "
                          "normalised per game, not per generation",
                          gen, t, gvar, w->dbg_gn);
            }
        }
#endif

        /* -- 4. OPTIMISE ------------------------------------------------ */
        grad_zero(tgsum, (int)TRUNK_NPARAM);
        for (int t = 0; t < nthreads; t++)
            grad_add((float *)tgsum, (const float *)workers[t].tg, (int)TRUNK_NPARAM);

        if (total_dec > 0) {
            const float *gp = (const float *)tgsum;
            double s2 = 0.0;
            scale_grad((float *)tgsum, TRUNK_NPARAM, 1.0f / (float)total_dec);
            for (size_t i = 0; i < TRUNK_NPARAM; i++) s2 += (double)gp[i] * (double)gp[i];
            gnorm = sqrt(s2);
            adam_step(&trunk_adam, (float *)trunk, (float *)tgsum,
                      c->trunk_lr, c->weight_decay, c->grad_clip);
        }

        for (int i = 0; i < n; i++) {
            if (hcount[i] == 0) continue;
            scale_grad((float *)&hgrad[i], HEAD_NPARAM, 1.0f / (float)hcount[i]);
            adam_step(&head_adam[i], (float *)&heads[i], (float *)&hgrad[i],
                      c->base_lr * hypers[i].lr_scale, c->weight_decay, c->grad_clip);
        }

        /* -- 5. RATE ---------------------------------------------------- */
        apply_elo(&sh);

        elo_mean = 0.0;
        for (int i = 0; i < n; i++) {
            elo_mean += (double)elo[i];
            if (elo[i] > elo[best_i]) best_i = i;
        }
        elo_mean /= (double)n;
        elo_best = (double)elo[best_i];
        memcpy(elo_sorted, elo, (size_t)n * sizeof(float));
        qsort(elo_sorted, (size_t)n, sizeof(float), cmp_float_asc);
        elo_p10 = (double)elo_sorted[(int)(0.10 * (double)(n - 1))];

        sec = now_sec() - t0;
        total_games += st.games;
        total_plies += st.plies;

        /* -- 6. LOG ----------------------------------------------------- */
        write_telemetry(tel, &sh, &st, sec, elo_best, elo_mean, elo_p10, best_i,
                        lp, lv, le, gnorm);

        if (!c->quiet) {
            double gd = st.games ? (double)st.games : 1.0;
            double gps = (double)st.games / (sec > 0.0 ? sec : 1.0);
            char eta[32];
            ema_gen = (gen == 1) ? sec : (0.7 * ema_gen + 0.3 * sec);
            fmt_dur(ema_gen * (double)(c->generations - gen), eta, sizeof eta);
            printf("gen %4d/%d  %6.0f g/s  elo %7.1f/%7.1f  W%3.0f%% D%3.0f%% L%3.0f%%"
                   "  len %5.1f  %5.2fs  eta %s\n",
                   gen, c->generations, gps, elo_best, elo_mean,
                   100.0 * (double)st.white_wins / gd,
                   100.0 * (double)st.draws / gd,
                   100.0 * (double)st.black_wins / gd,
                   st.sum_len / gd, sec, eta);
            fflush(stdout);
        }

        /* -- 7. EVOLVE -------------------------------------------------- */
        {
            int n_elite = (int)(c->elite_frac * (double)n);
            int n_cull  = (int)(c->cull_frac  * (double)n);
            if (n_elite < 1) n_elite = 1;
            if (n_elite > n) n_elite = n;
            if (n_cull > n - n_elite) n_cull = n - n_elite;

            for (int i = 0; i < n; i++) { rank[i].elo = elo[i]; rank[i].idx = i; }
            qsort(rank, (size_t)n, sizeof(EloRank), cmp_rank_desc);

            for (int k = 0; k < n_cull; k++) {
                int victim = rank[n - 1 - k].idx;
                int elite  = rank[(int)(rng_next(master_rng) % (uint64_t)n_elite)].idx;
                heads[victim]  = heads[elite];
                hypers[victim] = hypers[elite];
                /* clone the optimiser state too: the copy continues the elite's
                 * per-parameter scaling rather than restarting cold */
                memcpy(head_adam[victim].m, head_adam[elite].m,
                       (size_t)HEAD_NPARAM * sizeof(float));
                memcpy(head_adam[victim].v, head_adam[elite].v,
                       (size_t)HEAD_NPARAM * sizeof(float));
                head_adam[victim].t = head_adam[elite].t;
                head_mutate(&heads[victim], hypers[victim].mutate_sigma, master_rng);
                hyper_mutate(&hypers[victim], master_rng);
                /* Elo stays with the slot: the hall-of-fame anchor only means
                 * something if rating mass is conserved among live agents. */
            }
        }

        /* -- 8. HALL OF FAME -------------------------------------------- */
        if (sh.hof_n == 0 || gen % c->hof_every == 0) {
            HofEntry *e = &hof[sh.hof_next];
            e->head = heads[best_i];
            e->hy   = hypers[best_i];
            e->elo  = elo[best_i];
            e->gen  = gen;
            e->src  = best_i;
            sh.hof_next = (sh.hof_next + 1) % HOF_CAP;
            if (sh.hof_n < HOF_CAP) sh.hof_n++;
        }

        /* -- 9. CHECKPOINT ---------------------------------------------- */
        if (gen % SAVE_EVERY == 0 || gen == c->generations || g_interrupt) {
            snprintf(path, sizeof path, "%s/checkpoint.crl", c->run_dir);
            if (!model_save(path, trunk, heads, hypers, elo, n, gen))
                fprintf(stderr, "train: warning: could not write %s\n", path);
            if (elo_best >= best_saved) {
                best_saved = elo_best;
                snprintf(path, sizeof path, "%s/best.crl", c->run_dir);
                if (!model_save(path, trunk, heads, hypers, elo, n, gen))
                    fprintf(stderr, "train: warning: could not write %s\n", path);
            }
        }

        if (g_interrupt) {
            if (!c->quiet)
                printf("\ninterrupted: generation %d finished and saved.\n", gen);
            break;
        }
    }
    /* =================================================================== */

    if (!c->quiet) {
        double dt = now_sec() - t_run0;
        char buf[32];
        fmt_dur(dt, buf, sizeof buf);
        printf("done: %llu games, %llu plies in %s (%.0f games/s, %.0f positions/s)\n",
               (unsigned long long)total_games, (unsigned long long)total_plies, buf,
               (double)total_games / (dt > 0.0 ? dt : 1.0),
               (double)total_plies / (dt > 0.0 ? dt : 1.0));
        printf("model: %s/best.crl\n", c->run_dir);
        fflush(stdout);
    }
    rc = 0;

done:
    if (old_sigint != SIG_ERR) signal(SIGINT, old_sigint);
    if (tel) fclose(tel);
    if (workers) {
        for (int t = 0; t < nthreads; t++) {
            traj_free(&workers[t].tw);
            traj_free(&workers[t].tb);
            free(workers[t].tg);
            free(workers[t].hscratch);
            free(workers[t].fw);
            free(workers[t].ix);
            free(workers[t].adv);
            free(workers[t].ret);
        }
    }
    if (adam_ready) {
        adam_free(&trunk_adam);
        for (int i = 0; i < nadam; i++) adam_free(&head_adam[i]);
    }
    free(head_adam);
    free(tids);
    free(workers);
    free(bslot);
    free(wslot);
    free(rank);
    free(results);
    free(pairs);
    free(hcount);
    free(hlock);
    free(tgsum);
    free(hgrad);
    free(hof);
    free(elo_sorted);
    free(elo);
    free(hypers);
    free(heads);
    free(trunk);
    return rc;
}
