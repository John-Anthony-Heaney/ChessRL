/* tools/profile.c -- where the forward pass actually spends its time.
 *
 * Compiled as ONE translation unit with src/net.c so the per-stage timers call
 * the very same always_inline kernels nn_eval() calls.  A breakdown you can
 * trust needs the timed code to BE the shipped code, not a copy of it that can
 * drift.  Nothing links this; `make profile` builds it on request.
 *
 *   profile fingerprint   bitwise digest of Fwd + logits over the corpus
 *   profile dump          per-position digest and every float of Fwd
 *   profile stages        per-stage nanoseconds and percent of nn_eval
 *   profile threads       evals/sec at 1..8 threads, plus the false-sharing test
 *   profile batch         evals/sec against batch size, 1 thread and 8
 *   profile all           stages + threads + batch
 *
 * The corpus is 512 positions from deterministic random self-play (fixed seed),
 * so every number below is over the same positions from run to run.
 *
 * LOCALITY.  A self-play worker plays a whole game with ONE agent, so the head
 * (55 KB) stays hot while the positions -- and therefore the gathered W0 rows --
 * change every evaluation.  The benchmarks reproduce that: the position advances
 * every evaluation, the head every HEAD_PERIOD evaluations.  Benchmarking one
 * position against one head instead (as tests/test_net.c does) reports a rate
 * roughly 2x higher because the ~35 gathered rows never leave L1.
 */
#define _DARWIN_C_SOURCE 1

#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef NN_PROFILE_SRC
#define NN_PROFILE_SRC "../src/net.c"
#endif
#include NN_PROFILE_SRC

/* ------------------------------------------------------------------ timing */

static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

/* ------------------------------------------------------------------ corpus */

#define CORPUS_N     512
#define CORPUS_MOVES 128
#define NHEADS       16
#define HEAD_PERIOD  256     /* evaluations per head, ~4 moves of one game */

typedef struct {
    uint16_t fidx[NF_MAXACTIVE];
    int      nf;
    int      nmoves;
    MoveKey  keys[CORPUS_MOVES];
} CPos;

static CPos g_corpus[CORPUS_N];
static int  g_ncorpus = 0;

static Trunk *g_trunk;
static Head  *g_heads;

static uint64_t g_rs = 0x243F6A8885A308D3ULL;
static uint64_t rnd_u64(void)
{
    uint64_t z = (g_rs += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static void corpus_build(void)
{
    Position p;
    Move mv[MAX_MOVES];
    int plies = 0;

    pos_startpos(&p);
    while (g_ncorpus < CORPUS_N) {
        const int n = gen_legal(&p, mv);
        if (n == 0 || plies >= 160) { pos_startpos(&p); plies = 0; continue; }

        CPos *c = &g_corpus[g_ncorpus++];
        c->nf     = nn_features(&p, c->fidx);
        c->nmoves = n < CORPUS_MOVES ? n : CORPUS_MOVES;
        for (int i = 0; i < c->nmoves; i++) nn_move_key(&p, mv[i], &c->keys[i]);

        Undo u;
        make_move(&p, mv[rnd_u64() % (uint64_t)n], &u);
        plies++;
    }
}

static void weights_build(void)
{
    g_trunk = (Trunk *)malloc(sizeof(Trunk));
    g_heads = (Head *)malloc(sizeof(Head) * NHEADS);
    if (!g_trunk || !g_heads) { fprintf(stderr, "OOM\n"); exit(1); }
    nn_init(g_trunk, &g_heads[0], 12345ULL);
    for (int i = 1; i < NHEADS; i++)
        nn_init(NULL, &g_heads[i], 12345ULL + (uint64_t)i * 977ULL);
}

/* ------------------------------------------------------------- fingerprint */
/* FNV-1a over the raw bits: any change to any float anywhere changes this. */

static uint64_t fnv_bits(uint64_t h, const void *p, size_t nbytes)
{
    const uint8_t *b = (const uint8_t *)p;
    for (size_t i = 0; i < nbytes; i++) { h ^= b[i]; h *= 0x100000001B3ULL; }
    return h;
}

static uint64_t fingerprint(int verbose)
{
    uint64_t h = 0xCBF29CE484222325ULL;
    double vsum = 0.0, lsum = 0.0;
    float logits[CORPUS_MOVES];
    Fwd fw;

    for (int i = 0; i < g_ncorpus; i++) {
        const CPos *c = &g_corpus[i];
        const Head *hd = &g_heads[i % NHEADS];
        nn_eval(g_trunk, hd, c->fidx, c->nf, &fw);
        nn_logits(hd, &fw, c->keys, c->nmoves, logits);
        h = fnv_bits(h, &fw, sizeof fw);
        h = fnv_bits(h, logits, sizeof(float) * (size_t)c->nmoves);
        vsum += (double)fw.v;
        for (int m = 0; m < c->nmoves; m++) lsum += (double)logits[m];
        if (verbose == 2) {
            printf("pos %4d v %08x", i, *(const uint32_t *)&fw.v);
            for (int m = 0; m < c->nmoves; m++) printf(" %08x", *(const uint32_t *)&logits[m]);
            printf("\n");
        }
    }
    if (verbose)
        printf("fingerprint %016llx   sum(v)=%.9f  sum(logits)=%.9f  positions=%d\n",
               (unsigned long long)h, vsum, lsum, g_ncorpus);
    return h;
}

/* -------------------------------------------------------------- the stages */
/*
 * Each stage is timed on its own over the corpus, with the live values in one
 * Fwd and one set of scratch buffers exactly as nn_eval() keeps them, so the
 * per-eval state is in L1 and the weights are where they really are.  Every
 * stage is idempotent in its inputs (none overwrites what it reads), so running
 * one repeatedly measures what running it once inside nn_eval() costs.
 *
 * The honest check on all of it is at the bottom: the stages have to add up to
 * the measured cost of the whole nn_eval + nn_logits.
 */

#ifndef NN_NO_STAGES
#define STAGE_ITERS 400000

typedef struct { const char *name; double ns; } Stage;

static volatile float g_sink_f;

static void report_stages(void)
{
    Fwd fw;
    float acc[NF_ACC], z2[NF_HID], av[NF_VACC], zv[NF_VHID];
    float logits[CORPUS_MOVES];
    Stage st[11];
    int ns = 0;

    /* prime every buffer with a real evaluation */
    nn_eval(g_trunk, &g_heads[0], g_corpus[0].fidx, g_corpus[0].nf, &fw);
    nn_gather(g_trunk->W0, g_trunk->b0, g_heads[0].z, g_corpus[0].fidx, g_corpus[0].nf,
              NF_ACC, acc);
    nn_matvec(g_trunk->W1, g_trunk->b1, fw.h1, NF_HID, NF_ACC, z2);
    nn_gather(g_trunk->W0v, g_trunk->b0v, NULL, g_corpus[0].fidx, g_corpus[0].nf,
              NF_VACC, av);
    nn_matvec(g_heads[0].Wvh, g_heads[0].bvh, fw.hv0, NF_VHID, NF_VACC, zv);

#define TIME_STAGE(NAME, BODY)                                             \
    do {                                                                   \
        const double t0 = now_sec();                                       \
        for (int it = 0; it < STAGE_ITERS; it++) {                         \
            const CPos *c = &g_corpus[it % g_ncorpus];                     \
            const Head *hd = &g_heads[(it / HEAD_PERIOD) % NHEADS];        \
            (void)c; (void)hd;                                             \
            BODY;                                                          \
        }                                                                  \
        const double dt = now_sec() - t0;                                  \
        st[ns].name = (NAME);                                              \
        st[ns].ns = 1e9 * dt / (double)STAGE_ITERS;                        \
        ns++;                                                              \
        g_sink_f += fw.v + fw.h1[0] + fw.h2[0] + fw.hv0[1] + fw.hv[0] +   \
                    fw.x1[0] + fw.x2[0] + fw.xv0[1] + fw.xv[0] + fw.q[0] +\
                    fw.r1 + fw.r2 + fw.rv0 + fw.rv +                      \
                    acc[0] + z2[0] + av[0] + zv[0] + logits[0];            \
    } while (0)

    TIME_STAGE("W0 gather (sparse, 128 wide)",
               nn_gather(g_trunk->W0, g_trunk->b0, hd->z, c->fidx, c->nf, NF_ACC, acc));
    TIME_STAGE("LayerNorm 1 + relu (128)",
               fw.r1 = nn_norm_relu(acc, g_trunk->g0, g_trunk->c0, NF_ACC, fw.x1, fw.h1));
    TIME_STAGE("W1 residual block (128x128)",
               nn_matvec(g_trunk->W1, g_trunk->b1, fw.h1, NF_HID, NF_ACC, z2));
    TIME_STAGE("LayerNorm 2 + relu + skip (128)",
               fw.r2 = nn_norm_relu_add(z2, g_trunk->g1, g_trunk->c1, fw.h1, NF_HID,
                                        fw.x2, fw.h2));
    TIME_STAGE("Wp policy projection (32x128)",
               nn_matvec(hd->Wp, NULL, fw.h2, NF_PDIM, NF_HID, fw.q));
    TIME_STAGE("W0v gather (sparse, 32 wide)",
               nn_gather(g_trunk->W0v, g_trunk->b0v, NULL, c->fidx, c->nf, NF_VACC, av));
    TIME_STAGE("LayerNorm 3 + relu (32)",
               fw.rv0 = nn_norm_relu(av, g_trunk->g0v, g_trunk->c0v, NF_VACC, fw.xv0, fw.hv0));
    TIME_STAGE("Wvh value hidden (32x32)",
               nn_matvec(hd->Wvh, hd->bvh, fw.hv0, NF_VHID, NF_VACC, zv));
    TIME_STAGE("LayerNorm 4 + relu (32)",
               fw.rv = nn_norm_relu(zv, hd->gv, hd->cv, NF_VHID, fw.xv, fw.hv));
    TIME_STAGE("Wv + tanh (scalar out)",
               do { nn_matvec(hd->Wv, hd->bv, fw.hv, 1, NF_VHID, &fw.raw_v);
                    fw.v = tanhf(fw.raw_v); } while (0));
    TIME_STAGE("nn_logits (per legal move)",
               nn_logits(hd, &fw, c->keys, c->nmoves, logits));

#undef TIME_STAGE

    /* the whole thing, same access pattern, for the cross-check */
    double whole = 1e30, wholeeval = 1e30;
    for (int r = 0; r < 3; r++) {
        double t0 = now_sec();
        for (int it = 0; it < STAGE_ITERS; it++) {
            const CPos *c = &g_corpus[it % g_ncorpus];
            const Head *hd = &g_heads[(it / HEAD_PERIOD) % NHEADS];
            nn_eval(g_trunk, hd, c->fidx, c->nf, &fw);
            nn_logits(hd, &fw, c->keys, c->nmoves, logits);
            g_sink_f += fw.v + logits[0];
        }
        double dt = 1e9 * (now_sec() - t0) / (double)STAGE_ITERS;
        if (dt < whole) whole = dt;

        t0 = now_sec();
        for (int it = 0; it < STAGE_ITERS; it++) {
            const CPos *c = &g_corpus[it % g_ncorpus];
            const Head *hd = &g_heads[(it / HEAD_PERIOD) % NHEADS];
            nn_eval(g_trunk, hd, c->fidx, c->nf, &fw);
            g_sink_f += fw.v;
        }
        dt = 1e9 * (now_sec() - t0) / (double)STAGE_ITERS;
        if (dt < wholeeval) wholeeval = dt;
    }

    double sum = 0.0;
    for (int i = 0; i < ns; i++) sum += st[i].ns;

    printf("\n--- where one evaluation goes (single thread, %d positions x %d heads)\n",
           g_ncorpus, NHEADS);
    printf("    stage                              ns/eval   %% of eval+logits\n");
    for (int i = 0; i < ns; i++)
        printf("    %-34s %7.1f   %5.1f%%\n", st[i].name, st[i].ns, 100.0 * st[i].ns / sum);
    printf("    %-34s %7.1f   %5.1f%%\n", "sum of the stages", sum, 100.0);
    printf("    %-34s %7.1f   (cross-check: stages are %+.1f%% of it)\n",
           "measured nn_eval + nn_logits", whole, 100.0 * (sum / whole - 1.0));
    printf("    %-34s %7.1f   %.0f evals/sec on one thread\n",
           "measured nn_eval alone", wholeeval, 1e9 / wholeeval);
}

#endif /* NN_NO_STAGES */

/* ------------------------------------------------------------ single thread */

static double bench_eval_once(int iters, uint64_t *sink)
{
    Fwd fw;
    float logits[CORPUS_MOVES];
    uint64_t s = 0;
    const double t0 = now_sec();
    for (int i = 0; i < iters; i++) {
        const CPos *c = &g_corpus[i % g_ncorpus];
        const Head *hd = &g_heads[(i / HEAD_PERIOD) % NHEADS];
        nn_eval(g_trunk, hd, c->fidx, c->nf, &fw);
        nn_logits(hd, &fw, c->keys, c->nmoves, logits);
        s += (uint64_t)(int64_t)(fw.v * 1e6f) + (uint64_t)(int64_t)(logits[0] * 1e6f);
    }
    const double dt = now_sec() - t0;
    *sink += s;
    return (double)iters / dt;
}

/* ------------------------------------------------------------------ threads */

typedef struct {
    int      iters;
    int      tid;
    uint64_t sink;
    double   rate;
    char     pad[64];
} Task;

static Fwd *g_shared_fwd;
static int  g_pack_fwd = 0;

/* macOS has no pthread_barrier; a condvar gate is enough to start together. */
typedef struct {
    pthread_mutex_t m;
    pthread_cond_t  c;
    int             need, seen, gen;
} Gate;
static Gate g_gate;

static void gate_init(Gate *g, int n)
{
    pthread_mutex_init(&g->m, NULL);
    pthread_cond_init(&g->c, NULL);
    g->need = n; g->seen = 0; g->gen = 0;
}
static void gate_wait(Gate *g)
{
    pthread_mutex_lock(&g->m);
    const int gen = g->gen;
    if (++g->seen == g->need) { g->seen = 0; g->gen++; pthread_cond_broadcast(&g->c); }
    else while (gen == g->gen) pthread_cond_wait(&g->c, &g->m);
    pthread_mutex_unlock(&g->m);
}

static void *thread_body(void *arg)
{
    Task *t = (Task *)arg;
    Fwd local;
    Fwd *fw = g_pack_fwd ? &g_shared_fwd[t->tid] : &local;
    float logits[CORPUS_MOVES];
    uint64_t s = 0;

    gate_wait(&g_gate);
    const double t0 = now_sec();
    for (int i = 0; i < t->iters; i++) {
        const int k = (i + t->tid * 37) % g_ncorpus;
        const CPos *c = &g_corpus[k];
        const Head *hd = &g_heads[(i / HEAD_PERIOD + t->tid) % NHEADS];
        nn_eval(g_trunk, hd, c->fidx, c->nf, fw);
        nn_logits(hd, fw, c->keys, c->nmoves, logits);
        s += (uint64_t)(int64_t)(fw->v * 1e6f) + (uint64_t)(int64_t)(logits[0] * 1e6f);
    }
    t->rate = (double)t->iters / (now_sec() - t0);
    t->sink = s;
    return NULL;
}

static double bench_threads(int nthread, int iters, uint64_t *sink)
{
    pthread_t th[32];
    Task task[32];
    gate_init(&g_gate, nthread);
    for (int i = 0; i < nthread; i++) {
        memset(&task[i], 0, sizeof task[i]);
        task[i].iters = iters;
        task[i].tid   = i;
    }
    const double t0 = now_sec();
    for (int i = 0; i < nthread; i++) pthread_create(&th[i], NULL, thread_body, &task[i]);
    for (int i = 0; i < nthread; i++) pthread_join(th[i], NULL);
    const double dt = now_sec() - t0;
    for (int i = 0; i < nthread; i++) *sink += task[i].sink;
    return (double)(iters * nthread) / dt;
}

static void report_threads(uint64_t *sink)
{
    printf("\n--- thread scaling (nn_eval + nn_logits) ---\n");
    double one = 0.0;
    for (int r = 0; r < 3; r++) {
        const double v = bench_eval_once(150000, sink);
        if (v > one) one = v;
    }
    printf("  1 thread, direct call   %10.0f evals/sec\n", one);

    const int THREADS[] = {1, 2, 4, 6, 8};
    for (size_t i = 0; i < sizeof THREADS / sizeof THREADS[0]; i++) {
        const int n = THREADS[i];
        double best = 0.0;
        for (int r = 0; r < 3; r++) {
            const double v = bench_threads(n, 100000, sink);
            if (v > best) best = v;
        }
        printf("  %d thread%s              %10.0f evals/sec   %5.2fx   %3.0f%% efficiency\n",
               n, n == 1 ? " " : "s", best, best / one, 100.0 * best / (one * n));
    }

    /* false sharing: every thread's Fwd adjacent in one array instead of on its
     * own stack.  sizeof(Fwd) is 2712 bytes, so adjacent Fwds share the cache
     * lines at their boundaries. */
    g_shared_fwd = (Fwd *)malloc(sizeof(Fwd) * 32);
    double packed = 0.0, priv = 0.0;
    g_pack_fwd = 1;
    for (int r = 0; r < 3; r++) {
        const double v = bench_threads(8, 100000, sink);
        if (v > packed) packed = v;
    }
    g_pack_fwd = 0;
    for (int r = 0; r < 3; r++) {
        const double v = bench_threads(8, 100000, sink);
        if (v > priv) priv = v;
    }
    printf("  8 threads, Fwd packed adjacent in one array: %.0f vs %.0f evals/sec (%.1f%%)\n",
           packed, priv, 100.0 * packed / priv);
    free(g_shared_fwd);
}

int main(int argc, char **argv)
{
    const char *mode = argc > 1 ? argv[1] : "all";
    uint64_t sink = 0;

    chess_init();
    corpus_build();
    weights_build();

    printf("corpus %d positions, %d heads, trunk %.0f KB, head %.0f KB, Fwd %zu B\n",
           g_ncorpus, NHEADS, sizeof(Trunk) / 1024.0, sizeof(Head) / 1024.0, sizeof(Fwd));

    if (!strcmp(mode, "dump")) { fingerprint(2); return 0; }
    if (!strcmp(mode, "fingerprint")) { fingerprint(1); return 0; }
#ifndef NN_NO_STAGES
    if (!strcmp(mode, "stages") || !strcmp(mode, "all")) { fingerprint(1); report_stages(); }
#endif
    if (!strcmp(mode, "threads") || !strcmp(mode, "all")) report_threads(&sink);

    if (sink == 0x123456789ULL) printf("(impossible)\n");
    return 0;
}
