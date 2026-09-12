/* tools/conv_proto.c -- is a convolutional network affordable on this machine?
 *
 * A STANDALONE feasibility study.  It links against src/chess.c and src/net.c
 * (read-only) so that the convolutional numbers and the current MLP's numbers
 * come out of the same process, on the same corpus, with the same timer, and
 * so that the comparison cannot drift the way two separate benchmarks would.
 * NOTHING in src/ is modified and nothing here is wired into the engine: this
 * file exists to produce the numbers in docs/ARCHITECTURE_PLAN.md.
 *
 * BUILD (it is deliberately NOT in the Makefile -- `make` and `make test` are
 * untouched by this study):
 *
 *   cc -O3 -std=c11 -D_DARWIN_C_SOURCE -Wall -Wextra -Wno-unused-parameter \
 *      -funroll-loops -fno-math-errno -ffp-contract=fast -mcpu=native -Isrc \
 *      -DUSE_ACCELERATE -DACCELERATE_NEW_LAPACK \
 *      tools/conv_proto.c src/chess.c src/net.c -o build/conv_proto \
 *      -lm -lpthread -framework Accelerate
 *
 *   (drop the two -D and the -framework for the portable NEON build)
 *
 * MODES
 *   conv_proto params     parameter counts and MAC counts, conv vs the MLP
 *   conv_proto check      correctness: im2col+sgemm vs a naive 7-loop conv,
 *                         and a finite-difference gradient check
 *   conv_proto fwd        forward evals/sec, 1 and 8 threads, batch 1 and 32
 *   conv_proto bwd        cost of one forward+backward (the learning question)
 *   conv_proto feat       cost of the candidate EXTRA SPARSE FEATURES:
 *                         attack maps, mobility counts, history, repetition
 *   conv_proto sims       the simulations-per-move arithmetic
 *   conv_proto all        everything
 *
 * ---------------------------------------------------------------------------
 * THE ARCHITECTURE BEING COSTED
 * ---------------------------------------------------------------------------
 * AlphaZero's shape, shrunk: an input stack of 8x8 planes, a 3x3 convolutional
 * stem, N residual blocks of C filters (two 3x3 convolutions each, skip around
 * the pair), then a policy head and a value head.
 *
 *   x0                     P x 8 x 8   input planes (P = 20, see planes_build)
 *   a   = conv3x3(x0) + b              C x 64
 *   h   = relu(LN(a))                  C x 64          <- block input
 *   repeat N times:
 *       t1 = conv3x3(h)  + b1
 *       u1 = relu(LN(t1))
 *       t2 = conv3x3(u1) + b2
 *       h  = relu(LN(t2) + h)          <- residual
 *   policy: pp = conv1x1(h, C->4); q = Wq . relu(LN(pp))        -> 32-d query
 *   value:  vp = conv1x1(h, C->1); v = tanh(Wv2 . relu(Wv1 . relu(LN(vp))))
 *
 * TWO DELIBERATE DEPARTURES FROM ALPHAZERO, both in the conv net's favour, and
 * both stated here because they are the difference between a fair study and a
 * rigged one:
 *
 *  1. LayerNorm, not BatchNorm.  AlphaZero used BatchNorm, which at INFERENCE
 *     time folds into the preceding convolution and costs nothing.  LayerNorm
 *     does not fold and so costs the conv net a few percent it would not
 *     really pay.  It is used because it is what the rest of this codebase
 *     uses (see the note in src/net.h about gradient-norm growth) and because
 *     it has a clean backward pass with no batch-statistics bookkeeping.  The
 *     `fwd` mode reports the LayerNorm share separately so the reader can
 *     subtract it; it is small enough not to change any conclusion.
 *
 *  2. A FACTORED policy head, not AlphaZero's 8x8x73 = 4672-way move plane.
 *     The 4672 head needs a 128 x 4672 fully-connected layer -- 598k MACs, on
 *     its own more than TWENTY TIMES the arithmetic of our entire current
 *     network.  It is costed in `params` and measured in `fwd` so the number
 *     is on the record, but the headline conv timings use the cheap 32-d query
 *     head this codebase already has, because charging the conv net for a
 *     policy head we would obviously not adopt would be arguing in bad faith.
 *
 * WHAT AN "EVAL" MEANS HERE.  Exactly what it means in tools/profile.c: one
 * position in, policy query + value out, NOT including the per-legal-move
 * logit dot products (nn_logits, measured at 158 ns/eval) because that stage
 * is IDENTICAL for both architectures and would only dilute the ratio.  The
 * conv eval DOES include building the input planes from the Position, because
 * that is a real cost the MLP does not pay in the same form.
 */
#define _DARWIN_C_SOURCE 1

#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "chess.h"
#include "net.h"

#ifdef USE_ACCELERATE
#include <Accelerate/Accelerate.h>
#define HAVE_ACCELERATE 1
#else
#define HAVE_ACCELERATE 0
#endif

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define HAVE_NEON 1
#else
#define HAVE_NEON 0
#endif

/* ======================================================================== */
/*                                 timing                                   */
/* ======================================================================== */

/* CPU time consumed by the CALLING THREAD.
 *
 * This machine is shared with other jobs (the run that produced the numbers in
 * docs/ARCHITECTURE_PLAN.md saw load averages above 30 on 8 cores), and a
 * wall-clock rate under that load measures the scheduler, not the kernel: the
 * same nn_eval loop timed on the wall came out at 613k, 453k and 240k
 * evals/sec in three runs an hour apart.  CLOCK_THREAD_CPUTIME_ID counts only
 * the time the thread was actually ON a core, so a descheduled benchmark
 * reports the right rate instead of a slow one.  Every throughput number below
 * is evals per second of CPU TIME, and a multi-thread figure is the SUM over
 * threads -- i.e. what the machine would deliver with nothing else running.
 * Residual noise is core type (a P-core is ~1.5x an E-core) and frequency, so
 * the ratios are still quoted as best-of-N with the two candidates alternated. */
static double thread_cpu_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static uint64_t g_rs = 0x243F6A8885A308D3ULL;
static uint64_t rnd_u64(void)
{
    uint64_t z = (g_rs += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}
static float rnd_gauss(void)
{
    /* Box-Muller; quality is irrelevant, these are benchmark weights. */
    const double u1 = ((double)(rnd_u64() >> 11) + 1.0) / 9007199254740993.0;
    const double u2 = ((double)(rnd_u64() >> 11) + 1.0) / 9007199254740993.0;
    return (float)(sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2));
}

/* ======================================================================== */
/*                                  GEMM                                    */
/* ======================================================================== */
/* C = alpha * op(A) * op(B) + beta * C, row-major.  With Accelerate this is a
 * thin wrapper over cblas_sgemm.  Without it, the NN case gets a NEON kernel
 * (MR=4 rows of A, NR=16 columns of B: 4 broadcasts + 4 vector loads feed 16
 * FMAs, which is the ratio that keeps the FMA units fed) and the transposed
 * cases -- used only by the backward pass -- transpose into scratch and call
 * it.  A naive triple loop here would make every conv number below a
 * measurement of this function rather than of convolution, which is exactly
 * the mistake the brief warns about. */

#if !HAVE_ACCELERATE
static float *g_gemm_scratch = NULL;
static size_t g_gemm_scratch_n = 0;

static float *gemm_scratch(size_t n)
{
    if (n > g_gemm_scratch_n) {
        free(g_gemm_scratch);
        g_gemm_scratch = (float *)malloc(sizeof(float) * n);
        if (!g_gemm_scratch) { fprintf(stderr, "OOM gemm scratch\n"); exit(1); }
        g_gemm_scratch_n = n;
    }
    return g_gemm_scratch;
}

static void gemm_nn_neon(int M, int N, int K, float alpha,
                         const float *A, int lda, const float *B, int ldb,
                         float beta, float *C, int ldc)
{
    for (int m0 = 0; m0 < M; m0 += 4) {
        const int mr = (M - m0 < 4) ? M - m0 : 4;
        int n = 0;
#if HAVE_NEON
        for (; n + 16 <= N; n += 16) {
            float32x4_t c[4][4];
            for (int i = 0; i < mr; i++)
                for (int j = 0; j < 4; j++)
                    c[i][j] = beta == 0.0f ? vdupq_n_f32(0.0f)
                                           : vmulq_n_f32(vld1q_f32(&C[(m0 + i) * ldc + n + 4 * j]), beta);
            for (int k = 0; k < K; k++) {
                const float *bp = &B[(size_t)k * (size_t)ldb + n];
                const float32x4_t b0 = vld1q_f32(bp);
                const float32x4_t b1 = vld1q_f32(bp + 4);
                const float32x4_t b2 = vld1q_f32(bp + 8);
                const float32x4_t b3 = vld1q_f32(bp + 12);
                for (int i = 0; i < mr; i++) {
                    const float av = alpha * A[(size_t)(m0 + i) * (size_t)lda + k];
                    if (av == 0.0f) continue;
                    c[i][0] = vfmaq_n_f32(c[i][0], b0, av);
                    c[i][1] = vfmaq_n_f32(c[i][1], b1, av);
                    c[i][2] = vfmaq_n_f32(c[i][2], b2, av);
                    c[i][3] = vfmaq_n_f32(c[i][3], b3, av);
                }
            }
            for (int i = 0; i < mr; i++)
                for (int j = 0; j < 4; j++)
                    vst1q_f32(&C[(m0 + i) * ldc + n + 4 * j], c[i][j]);
        }
#endif
        for (; n < N; n++) {
            for (int i = 0; i < mr; i++) {
                float s = 0.0f;
                for (int k = 0; k < K; k++)
                    s += A[(size_t)(m0 + i) * (size_t)lda + k] * B[(size_t)k * (size_t)ldb + n];
                float *cp = &C[(size_t)(m0 + i) * (size_t)ldc + n];
                *cp = alpha * s + beta * *cp;
            }
        }
    }
}
#endif /* !HAVE_ACCELERATE */

/* ta/tb: 0 = no transpose, 1 = transpose.  lda/ldb are the leading dimensions
 * of the STORED matrices (before any transpose). */
static void gemm(int ta, int tb, int M, int N, int K, float alpha,
                 const float *A, int lda, const float *B, int ldb,
                 float beta, float *C, int ldc)
{
#if HAVE_ACCELERATE
    cblas_sgemm(CblasRowMajor, ta ? CblasTrans : CblasNoTrans,
                tb ? CblasTrans : CblasNoTrans, M, N, K, alpha,
                A, lda, B, ldb, beta, C, ldc);
#else
    if (!ta && !tb) { gemm_nn_neon(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc); return; }
    /* Transpose into scratch, then use the fast kernel. */
    if (ta && !tb) {
        float *At = gemm_scratch((size_t)M * (size_t)K);
        for (int k = 0; k < K; k++)
            for (int m = 0; m < M; m++) At[(size_t)m * (size_t)K + k] = A[(size_t)k * (size_t)lda + m];
        gemm_nn_neon(M, N, K, alpha, At, K, B, ldb, beta, C, ldc);
        return;
    }
    if (!ta && tb) {
        float *Bt = gemm_scratch((size_t)K * (size_t)N);
        for (int n = 0; n < N; n++)
            for (int k = 0; k < K; k++) Bt[(size_t)k * (size_t)N + n] = B[(size_t)n * (size_t)ldb + k];
        gemm_nn_neon(M, N, K, alpha, A, lda, Bt, N, beta, C, ldc);
        return;
    }
    {   /* ta && tb -- unused, but correct rather than absent */
        float *At = gemm_scratch((size_t)M * (size_t)K + (size_t)K * (size_t)N);
        float *Bt = At + (size_t)M * (size_t)K;
        for (int k = 0; k < K; k++)
            for (int m = 0; m < M; m++) At[(size_t)m * (size_t)K + k] = A[(size_t)k * (size_t)lda + m];
        for (int n = 0; n < N; n++)
            for (int k = 0; k < K; k++) Bt[(size_t)k * (size_t)N + n] = B[(size_t)n * (size_t)ldb + k];
        gemm_nn_neon(M, N, K, alpha, At, K, Bt, N, beta, C, ldc);
    }
#endif
}

/* ======================================================================== */
/*                             input planes                                 */
/* ======================================================================== */
/*  0.. 5  our    pawn,knight,bishop,rook,queen,king
 *  6..11  their  pawn,knight,bishop,rook,queen,king
 * 12..15  castling rights (our K, our Q, their K, their Q), constant planes
 * 16      en-passant target square, one-hot
 * 17      halfmove clock / 100, constant plane
 * 18      all ones (lets the network tell a real zero from zero padding)
 * 19      repetition count of the current position / 2, constant plane
 *
 * Squares are side-to-move relative -- mirrored vertically for Black -- which
 * is exactly what nn_features() in src/net.c already does.  All of this is the
 * position, not an opinion about it. */
#define CP_PLANES 20
#define CP_SQ     64
#define CP_IN     (CP_PLANES * CP_SQ)

static void planes_build(const Position *p, int rep_count, float *out)
{
    memset(out, 0, sizeof(float) * CP_IN);
    const int us = p->side, them = us ^ 1;
    const int flip = (us == BLACK) ? 56 : 0;   /* vertical mirror, rank-major */

    for (int pt = 0; pt < NPIECES; pt++) {
        uint64_t b = p->piece[us][pt];
        while (b) {
            const int sq = __builtin_ctzll(b); b &= b - 1;
            out[pt * CP_SQ + (sq ^ flip)] = 1.0f;
        }
        b = p->piece[them][pt];
        while (b) {
            const int sq = __builtin_ctzll(b); b &= b - 1;
            out[(6 + pt) * CP_SQ + (sq ^ flip)] = 1.0f;
        }
    }

    const int ck = us == WHITE ? CR_WK : CR_BK, cq = us == WHITE ? CR_WQ : CR_BQ;
    const int tk = us == WHITE ? CR_BK : CR_WK, tq = us == WHITE ? CR_BQ : CR_WQ;
    const float c12 = (p->castling & ck) ? 1.0f : 0.0f;
    const float c13 = (p->castling & cq) ? 1.0f : 0.0f;
    const float c14 = (p->castling & tk) ? 1.0f : 0.0f;
    const float c15 = (p->castling & tq) ? 1.0f : 0.0f;
    const float c17 = (float)p->halfmove * 0.01f;
    const float c19 = (float)rep_count * 0.5f;
    for (int s = 0; s < CP_SQ; s++) {
        out[12 * CP_SQ + s] = c12;
        out[13 * CP_SQ + s] = c13;
        out[14 * CP_SQ + s] = c14;
        out[15 * CP_SQ + s] = c15;
        out[17 * CP_SQ + s] = c17;
        out[18 * CP_SQ + s] = 1.0f;
        out[19 * CP_SQ + s] = c19;
    }
    if (p->ep >= 0) out[16 * CP_SQ + (p->ep ^ flip)] = 1.0f;
}

/* ======================================================================== */
/*                                the net                                   */
/* ======================================================================== */

enum { POL_QUERY = 0, POL_AZPLANE = 1 };

#define CP_PDIM   32      /* policy query width, matches NF_PDIM            */
#define CP_PPL    4       /* policy 1x1 conv output planes                  */
#define CP_VHID   32      /* value head hidden width, matches NF_VHID       */
#define CP_AZMOVE 4672    /* AlphaZero's 8x8x73 move encoding               */
#define CP_AZPL   2       /* AlphaZero's policy 1x1 conv output planes      */

typedef struct {
    int blocks, filters, pol;
    /* stem */
    float *sw;                 /* C x (9*P)   */
    float *sb;                 /* C           */
    float *sg, *sc;            /* C, C        LayerNorm affine, per channel   */
    /* residual blocks: [b][0] and [b][1] */
    float *bw[2];              /* per block: C x (9*C), laid out blocks-major */
    float *bb[2];              /* C                                          */
    float *bg[2], *bc[2];      /* C                                          */
    /* policy head */
    float *pw;                 /* CP_PPL x C   (1x1 conv)                    */
    float *pb;                 /* CP_PPL                                     */
    float *pg, *pc;            /* CP_PPL                                     */
    float *qw;                 /* CP_PDIM x (CP_PPL*64)                      */
    float *qb;                 /* CP_PDIM                                    */
    /* AlphaZero-style plane head (only when pol == POL_AZPLANE)             */
    float *aw;                 /* CP_AZPL x C                                */
    float *ab;                 /* CP_AZPL                                    */
    float *aq;                 /* CP_AZMOVE x (CP_AZPL*64)                   */
    float *aqb;                /* CP_AZMOVE                                  */
    /* value head */
    float *vw;                 /* 1 x C  (1x1 conv)                          */
    float *vb;                 /* 1                                          */
    float *vg, *vc;            /* 1                                          */
    float *v1w;                /* CP_VHID x 64                               */
    float *v1b;                /* CP_VHID                                    */
    float *v2w;                /* 1 x CP_VHID                                */
    float *v2b;                /* 1                                          */

    float  *blob;              /* one allocation for everything above        */
    size_t  nparam;
} ConvNet;

static size_t conv_nparam(int blocks, int filters, int pol)
{
    const size_t C = (size_t)filters;
    size_t n = 0;
    n += 9 * CP_PLANES * C + C + C + C;                    /* stem + LN      */
    n += (size_t)blocks * 2 * (9 * C * C + C + C + C);     /* blocks + LN    */
    n += CP_PPL * C + CP_PPL + CP_PPL + CP_PPL;            /* policy 1x1 +LN */
    n += (size_t)CP_PDIM * (CP_PPL * 64) + CP_PDIM;        /* query          */
    if (pol == POL_AZPLANE)
        n += CP_AZPL * C + CP_AZPL + (size_t)CP_AZMOVE * (CP_AZPL * 64) + CP_AZMOVE;
    n += C + 1 + 1 + 1;                                    /* value 1x1 + LN */
    n += (size_t)CP_VHID * 64 + CP_VHID + CP_VHID + 1;     /* value FCs      */
    return n;
}

/* MACs for one forward evaluation, counted by hand so the measured GFLOP/s can
 * be checked against the machine's ceiling. */
static double conv_macs(int blocks, int filters, int pol)
{
    const double C = filters;
    double m = 9.0 * CP_PLANES * C * 64.0;              /* stem              */
    m += blocks * 2.0 * 9.0 * C * C * 64.0;             /* residual blocks   */
    m += CP_PPL * C * 64.0 + (double)CP_PDIM * CP_PPL * 64.0;   /* policy    */
    if (pol == POL_AZPLANE)
        m += CP_AZPL * C * 64.0 + (double)CP_AZMOVE * CP_AZPL * 64.0;
    m += C * 64.0 + (double)CP_VHID * 64.0 + CP_VHID;   /* value             */
    return m;
}

static float *bump(float **cur, size_t n) { float *p = *cur; *cur += n; return p; }

static void conv_init(ConvNet *n, int blocks, int filters, int pol, uint64_t seed)
{
    memset(n, 0, sizeof *n);
    n->blocks = blocks; n->filters = filters; n->pol = pol;
    n->nparam = conv_nparam(blocks, filters, pol);
    n->blob = (float *)calloc(n->nparam, sizeof(float));
    if (!n->blob) { fprintf(stderr, "OOM conv net\n"); exit(1); }

    const size_t C = (size_t)filters;
    float *c = n->blob;
    n->sw = bump(&c, 9 * CP_PLANES * C); n->sb = bump(&c, C);
    n->sg = bump(&c, C); n->sc = bump(&c, C);
    for (int k = 0; k < 2; k++) {
        n->bw[k] = bump(&c, (size_t)blocks * 9 * C * C);
        n->bb[k] = bump(&c, (size_t)blocks * C);
        n->bg[k] = bump(&c, (size_t)blocks * C);
        n->bc[k] = bump(&c, (size_t)blocks * C);
    }
    n->pw = bump(&c, CP_PPL * C); n->pb = bump(&c, CP_PPL);
    n->pg = bump(&c, CP_PPL);     n->pc = bump(&c, CP_PPL);
    n->qw = bump(&c, (size_t)CP_PDIM * CP_PPL * 64); n->qb = bump(&c, CP_PDIM);
    if (pol == POL_AZPLANE) {
        n->aw  = bump(&c, CP_AZPL * C); n->ab = bump(&c, CP_AZPL);
        n->aq  = bump(&c, (size_t)CP_AZMOVE * CP_AZPL * 64);
        n->aqb = bump(&c, CP_AZMOVE);
    }
    n->vw = bump(&c, C); n->vb = bump(&c, 1); n->vg = bump(&c, 1); n->vc = bump(&c, 1);
    n->v1w = bump(&c, (size_t)CP_VHID * 64); n->v1b = bump(&c, CP_VHID);
    n->v2w = bump(&c, CP_VHID); n->v2b = bump(&c, 1);

    g_rs = seed;
    /* He init on the convolutions, small on the heads; the values only have to
     * be non-degenerate for the timings and well-scaled for the grad check. */
    const float ss = sqrtf(2.0f / (9.0f * CP_PLANES));
    for (size_t i = 0; i < 9 * CP_PLANES * C; i++) n->sw[i] = rnd_gauss() * ss;
    const float bs = sqrtf(2.0f / (9.0f * (float)C));
    for (int k = 0; k < 2; k++)
        for (size_t i = 0; i < (size_t)blocks * 9 * C * C; i++) n->bw[k][i] = rnd_gauss() * bs;
    for (size_t i = 0; i < C; i++) { n->sg[i] = 1.0f; }
    for (int k = 0; k < 2; k++)
        for (size_t i = 0; i < (size_t)blocks * C; i++) n->bg[k][i] = 1.0f;
    for (size_t i = 0; i < CP_PPL * C; i++) n->pw[i] = rnd_gauss() * (1.0f / sqrtf((float)C));
    for (int i = 0; i < CP_PPL; i++) n->pg[i] = 1.0f;
    for (size_t i = 0; i < (size_t)CP_PDIM * CP_PPL * 64; i++)
        n->qw[i] = rnd_gauss() * (1.0f / sqrtf((float)(CP_PPL * 64)));
    if (pol == POL_AZPLANE) {
        for (size_t i = 0; i < CP_AZPL * C; i++) n->aw[i] = rnd_gauss() * (1.0f / sqrtf((float)C));
        for (size_t i = 0; i < (size_t)CP_AZMOVE * CP_AZPL * 64; i++)
            n->aq[i] = rnd_gauss() * (1.0f / sqrtf((float)(CP_AZPL * 64)));
    }
    for (size_t i = 0; i < C; i++) n->vw[i] = rnd_gauss() * (1.0f / sqrtf((float)C));
    n->vg[0] = 1.0f;
    for (size_t i = 0; i < (size_t)CP_VHID * 64; i++) n->v1w[i] = rnd_gauss() * (1.0f / 8.0f);
    for (int i = 0; i < CP_VHID; i++) n->v2w[i] = rnd_gauss() * (1.0f / sqrtf((float)CP_VHID));
}

static void conv_free(ConvNet *n) { free(n->blob); n->blob = NULL; }

/* ======================================================================== */
/*                         activations and workspace                        */
/* ======================================================================== */
/* LAYOUT.  Everything is [channel][b*64 + sq]: channel-major with the batch
 * folded into the spatial index.  That is the layout im2col + sgemm wants (the
 * GEMM's N dimension is 64*B and is contiguous), and it is the reason the
 * batched conv gets any efficiency at all on a board this small: a single
 * position gives a GEMM of N=64, which is far too narrow to amortise anything.
 *
 * The LayerNorm is over (channel, square) for ONE position, so it reads a
 * stride of 64*B between channels and 64 contiguous floats within one. */

typedef struct {
    int    B, C, blocks;
    float *in;        /* CP_PLANES x 64B                                     */
    float *col;       /* max(9*C, 9*CP_PLANES) x 64B                         */
    float *h;         /* C x 64B   block input / output                      */
    float *u1;        /* C x 64B                                             */
    float *t;         /* C x 64B   pre-norm scratch                          */
    /* saved for the backward pass, per layer */
    float *xhat;      /* (1 + 2*blocks) x C x 64B   normalised activations   */
    float *rstd;      /* (1 + 2*blocks) x B                                  */
    float *act;       /* (1 + 2*blocks) x C x 64B   post-relu / pre-residual */
    float *cols;      /* (1 + 2*blocks) x colstride, saved im2col (backward)  */
    size_t colstride; /* 9*max(C, CP_PLANES)*64B -- the STEM's im2col is over
                       * CP_PLANES input planes, which is wider than 9*C at
                       * C = 16, so a 9*C stride would overrun the stem's slot
                       * and let the next layer's im2col overwrite it.        */
    /* heads */
    float *pp, *ppx, *pph, *q, *prstd;
    float *vp, *vpx, *vph, *vh, *vraw, *v, *vrstd;
    float *azp, *azl;
    float *blob;
    int    keep;      /* 1 when the backward-pass buffers were allocated     */
} ConvWs;

static size_t ws_floats(int B, int C, int blocks, int pol, int keep)
{
    const size_t N = (size_t)B * 64;
    const size_t colmax = (size_t)(9 * (C > CP_PLANES ? C : CP_PLANES)) * N;
    size_t n = (size_t)CP_PLANES * N + colmax + 3 * (size_t)C * N;
    if (keep) {
        const size_t L = (size_t)(1 + 2 * blocks);
        const size_t cmax = (size_t)(C > CP_PLANES ? C : CP_PLANES);
        n += L * (size_t)C * N * 2 + L * (size_t)B + L * 9 * cmax * N;
    }
    n += 3 * (size_t)CP_PPL * N + (size_t)CP_PDIM * B + (size_t)B;
    n += 3 * N + (size_t)CP_VHID * B + 3 * (size_t)B;
    if (pol == POL_AZPLANE) n += (size_t)CP_AZPL * N + (size_t)CP_AZMOVE * B;
    return n + 64;
}

static void ws_init(ConvWs *w, int B, int C, int blocks, int pol, int keep)
{
    memset(w, 0, sizeof *w);
    w->B = B; w->C = C; w->blocks = blocks; w->keep = keep;
    const size_t N = (size_t)B * 64;
    w->blob = (float *)calloc(ws_floats(B, C, blocks, pol, keep), sizeof(float));
    if (!w->blob) { fprintf(stderr, "OOM workspace\n"); exit(1); }
    float *c = w->blob;
    w->in  = bump(&c, (size_t)CP_PLANES * N);
    w->col = bump(&c, (size_t)(9 * (C > CP_PLANES ? C : CP_PLANES)) * N);
    w->h   = bump(&c, (size_t)C * N);
    w->u1  = bump(&c, (size_t)C * N);
    w->t   = bump(&c, (size_t)C * N);
    if (keep) {
        const size_t L = (size_t)(1 + 2 * blocks);
        w->xhat = bump(&c, L * (size_t)C * N);
        w->act  = bump(&c, L * (size_t)C * N);
        w->rstd = bump(&c, L * (size_t)B);
        w->colstride = 9 * (size_t)(C > CP_PLANES ? C : CP_PLANES) * N;
        w->cols = bump(&c, L * w->colstride);
    }
    w->pp = bump(&c, (size_t)CP_PPL * N); w->ppx = bump(&c, (size_t)CP_PPL * N);
    w->pph = bump(&c, (size_t)CP_PPL * N); w->q = bump(&c, (size_t)CP_PDIM * B);
    w->prstd = bump(&c, (size_t)B);
    w->vp = bump(&c, N); w->vpx = bump(&c, N); w->vph = bump(&c, N);
    w->vh = bump(&c, (size_t)CP_VHID * B);
    w->vraw = bump(&c, (size_t)B); w->v = bump(&c, (size_t)B); w->vrstd = bump(&c, (size_t)B);
    if (pol == POL_AZPLANE) {
        w->azp = bump(&c, (size_t)CP_AZPL * N);
        w->azl = bump(&c, (size_t)CP_AZMOVE * B);
    }
}
static void ws_free(ConvWs *w) { free(w->blob); w->blob = NULL; }

/* ======================================================================== */
/*                              im2col / col2im                             */
/* ======================================================================== */
/* col[(ci*9 + ky*3 + kx)][b*64 + r*8 + f] = X[ci][b*64 + (r+ky-1)*8 + (f+kx-1)]
 * with zero padding.  The inner 8 floats of a row are contiguous in both, so
 * the copy is a run of 8-float moves with a shift, which is why this is not
 * the bottleneck it looks like. */
static void im2col(const float *X, int C, int B, float *col)
{
    const int N = B * 64;
    for (int ci = 0; ci < C; ci++) {
        const float *src = X + (size_t)ci * N;
        for (int ky = 0; ky < 3; ky++) {
            for (int kx = 0; kx < 3; kx++) {
                float *dst = col + (size_t)(ci * 9 + ky * 3 + kx) * N;
                const int dy = ky - 1, dx = kx - 1;
                for (int b = 0; b < B; b++) {
                    const float *s = src + b * 64;
                    float *d = dst + b * 64;
                    for (int r = 0; r < 8; r++) {
                        const int rr = r + dy;
                        if (rr < 0 || rr > 7) { memset(d + r * 8, 0, 8 * sizeof(float)); continue; }
                        const float *sr = s + rr * 8;
                        float *dr = d + r * 8;
                        if (dx == 0) { memcpy(dr, sr, 8 * sizeof(float)); }
                        else if (dx == -1) { dr[0] = 0.0f; memcpy(dr + 1, sr, 7 * sizeof(float)); }
                        else { memcpy(dr, sr + 1, 7 * sizeof(float)); dr[7] = 0.0f; }
                    }
                }
            }
        }
    }
}

/* Scatter-add the gradient of the im2col back into image space. */
static void col2im_add(const float *col, int C, int B, float *dX)
{
    const int N = B * 64;
    for (int ci = 0; ci < C; ci++) {
        float *dst = dX + (size_t)ci * N;
        for (int ky = 0; ky < 3; ky++) {
            for (int kx = 0; kx < 3; kx++) {
                const float *src = col + (size_t)(ci * 9 + ky * 3 + kx) * N;
                const int dy = ky - 1, dx = kx - 1;
                for (int b = 0; b < B; b++) {
                    const float *s = src + b * 64;
                    float *d = dst + b * 64;
                    for (int r = 0; r < 8; r++) {
                        const int rr = r + dy;
                        if (rr < 0 || rr > 7) continue;
                        for (int f = 0; f < 8; f++) {
                            const int ff = f + dx;
                            if (ff < 0 || ff > 7) continue;
                            d[rr * 8 + ff] += s[r * 8 + f];
                        }
                    }
                }
            }
        }
    }
}

/* Reference: the naive 7-loop convolution the brief warns about.  Used by
 * `check` to prove the fast path computes the same thing, and timed in `fwd`
 * so the report can say how badly it would have misled us. */
static void conv3x3_naive(const float *W, const float *b, const float *X,
                          int Cout, int Cin, int B, float *Y)
{
    const int N = B * 64;
    for (int co = 0; co < Cout; co++)
        for (int bi = 0; bi < B; bi++)
            for (int r = 0; r < 8; r++)
                for (int f = 0; f < 8; f++) {
                    float s = b ? b[co] : 0.0f;
                    for (int ci = 0; ci < Cin; ci++)
                        for (int ky = 0; ky < 3; ky++)
                            for (int kx = 0; kx < 3; kx++) {
                                const int rr = r + ky - 1, ff = f + kx - 1;
                                if (rr < 0 || rr > 7 || ff < 0 || ff > 7) continue;
                                s += W[(size_t)co * (Cin * 9) + ci * 9 + ky * 3 + kx] *
                                     X[(size_t)ci * N + bi * 64 + rr * 8 + ff];
                            }
                    Y[(size_t)co * N + bi * 64 + r * 8 + f] = s;
                }
}

/* ======================================================================== */
/*                            LayerNorm + relu                              */
/* ======================================================================== */
/* Over (channel, square) for each position independently.  `g`/`c` are
 * per-channel and broadcast across the 64 squares -- the convolutional
 * equivalent of what src/net.c does per unit. */

static void ln_relu(const float *X, int C, int B, const float *g, const float *cb,
                    int do_relu, float *xhat, float *rstd, float *Y)
{
    const int N = B * 64;
    const float inv = 1.0f / (float)(C * 64);
    for (int b = 0; b < B; b++) {
        double m = 0.0, v = 0.0;
        for (int ci = 0; ci < C; ci++) {
            const float *x = X + (size_t)ci * N + b * 64;
            for (int s = 0; s < 64; s++) m += x[s];
        }
        const float mean = (float)(m * (double)inv);
        for (int ci = 0; ci < C; ci++) {
            const float *x = X + (size_t)ci * N + b * 64;
            for (int s = 0; s < 64; s++) { const float d = x[s] - mean; v += (double)d * d; }
        }
        const float r = 1.0f / sqrtf((float)(v * (double)inv) + 1e-5f);
        rstd[b] = r;
        for (int ci = 0; ci < C; ci++) {
            const float *x = X + (size_t)ci * N + b * 64;
            float *xh = xhat + (size_t)ci * N + b * 64;
            float *y  = Y + (size_t)ci * N + b * 64;
            const float gg = g[ci], cc = cb[ci];
            for (int s = 0; s < 64; s++) {
                const float h = (x[s] - mean) * r;
                xh[s] = h;
                const float o = gg * h + cc;
                y[s] = do_relu ? (o > 0.0f ? o : 0.0f) : o;
            }
        }
    }
}

/* dX from dY, given the saved xhat/rstd.  Accumulates dg/dc. */
static void ln_backward(const float *dY, const float *xhat, const float *rstd,
                        int C, int B, const float *g, float *dg, float *dc, float *dX)
{
    const int N = B * 64;
    const float inv = 1.0f / (float)(C * 64);
    for (int b = 0; b < B; b++) {
        double s1 = 0.0, s2 = 0.0;
        for (int ci = 0; ci < C; ci++) {
            const float *dy = dY + (size_t)ci * N + b * 64;
            const float *xh = xhat + (size_t)ci * N + b * 64;
            const float gg = g[ci];
            for (int s = 0; s < 64; s++) {
                const float dxh = dy[s] * gg;
                s1 += dxh; s2 += (double)dxh * xh[s];
                dg[ci] += dy[s] * xh[s];
                dc[ci] += dy[s];
            }
        }
        const float m1 = (float)(s1 * (double)inv), m2 = (float)(s2 * (double)inv);
        const float r = rstd[b];
        for (int ci = 0; ci < C; ci++) {
            const float *dy = dY + (size_t)ci * N + b * 64;
            const float *xh = xhat + (size_t)ci * N + b * 64;
            float *dx = dX + (size_t)ci * N + b * 64;
            const float gg = g[ci];
            for (int s = 0; s < 64; s++)
                dx[s] += r * (dy[s] * gg - m1 - xh[s] * m2);
        }
    }
}

/* ======================================================================== */
/*                              forward pass                                */
/* ======================================================================== */

static void conv_forward(const ConvNet *n, ConvWs *w, int B)
{
    const int C = n->filters, N = B * 64;
    const int keep = w->keep;
    int L = 0;

    /* ---- stem ---- */
    float *cbuf = keep ? w->cols + (size_t)L * w->colstride : w->col;
    im2col(w->in, CP_PLANES, B, cbuf);
    for (int co = 0; co < C; co++)
        for (int i = 0; i < N; i++) w->t[(size_t)co * N + i] = n->sb[co];
    gemm(0, 0, C, N, 9 * CP_PLANES, 1.0f, n->sw, 9 * CP_PLANES, cbuf, N, 1.0f, w->t, N);
    ln_relu(w->t, C, B, n->sg, n->sc, 1,
            keep ? w->xhat + (size_t)L * C * N : w->h,
            keep ? w->rstd + (size_t)L * B : w->vraw, w->h);
    if (keep) memcpy(w->act + (size_t)L * C * N, w->h, sizeof(float) * (size_t)C * N);
    L++;

    /* ---- residual blocks ---- */
    for (int blk = 0; blk < n->blocks; blk++) {
        const float *w1 = n->bw[0] + (size_t)blk * 9 * C * C;
        const float *w2 = n->bw[1] + (size_t)blk * 9 * C * C;
        const float *b1 = n->bb[0] + (size_t)blk * C, *b2 = n->bb[1] + (size_t)blk * C;
        const float *g1 = n->bg[0] + (size_t)blk * C, *g2 = n->bg[1] + (size_t)blk * C;
        const float *c1 = n->bc[0] + (size_t)blk * C, *c2 = n->bc[1] + (size_t)blk * C;

        cbuf = keep ? w->cols + (size_t)L * w->colstride : w->col;
        im2col(w->h, C, B, cbuf);
        for (int co = 0; co < C; co++)
            for (int i = 0; i < N; i++) w->t[(size_t)co * N + i] = b1[co];
        gemm(0, 0, C, N, 9 * C, 1.0f, w1, 9 * C, cbuf, N, 1.0f, w->t, N);
        ln_relu(w->t, C, B, g1, c1, 1,
                keep ? w->xhat + (size_t)L * C * N : w->u1,
                keep ? w->rstd + (size_t)L * B : w->vraw, w->u1);
        if (keep) memcpy(w->act + (size_t)L * C * N, w->u1, sizeof(float) * (size_t)C * N);
        L++;

        cbuf = keep ? w->cols + (size_t)L * w->colstride : w->col;
        im2col(w->u1, C, B, cbuf);
        for (int co = 0; co < C; co++)
            for (int i = 0; i < N; i++) w->t[(size_t)co * N + i] = b2[co];
        gemm(0, 0, C, N, 9 * C, 1.0f, w2, 9 * C, cbuf, N, 1.0f, w->t, N);
        /* LN WITHOUT relu, then add the skip, then relu. */
        ln_relu(w->t, C, B, g2, c2, 0,
                keep ? w->xhat + (size_t)L * C * N : w->t,
                keep ? w->rstd + (size_t)L * B : w->vraw, w->t);
        for (int i = 0; i < C * N; i++) {
            const float o = w->t[i] + w->h[i];
            w->h[i] = o > 0.0f ? o : 0.0f;
        }
        if (keep) memcpy(w->act + (size_t)L * C * N, w->h, sizeof(float) * (size_t)C * N);
        L++;
    }

    /* ---- policy head: 1x1 conv C -> CP_PPL, LN+relu, FC to the query ---- */
    for (int co = 0; co < CP_PPL; co++)
        for (int i = 0; i < N; i++) w->pp[(size_t)co * N + i] = n->pb[co];
    gemm(0, 0, CP_PPL, N, C, 1.0f, n->pw, C, w->h, N, 1.0f, w->pp, N);
    ln_relu(w->pp, CP_PPL, B, n->pg, n->pc, 1, w->ppx, w->prstd, w->pph);
    /* The query FC reads CP_PPL*64 values for one position; the layout puts a
     * position's squares contiguously inside each plane, so it is a small
     * gather per position rather than one GEMM.  At CP_PPL=4 it is 8k MACs. */
    for (int b = 0; b < B; b++) {
        float *q = w->q + (size_t)b * CP_PDIM;
        for (int d = 0; d < CP_PDIM; d++) {
            const float *wq = n->qw + (size_t)d * (CP_PPL * 64);
            float s = n->qb[d];
            for (int co = 0; co < CP_PPL; co++) {
                const float *x = w->pph + (size_t)co * N + b * 64;
                for (int t = 0; t < 64; t++) s += wq[co * 64 + t] * x[t];
            }
            q[d] = s;
        }
    }

    /* ---- AlphaZero's 4672-way plane head, when asked for ---- */
    if (n->pol == POL_AZPLANE) {
        for (int co = 0; co < CP_AZPL; co++)
            for (int i = 0; i < N; i++) w->azp[(size_t)co * N + i] = n->ab[co];
        gemm(0, 0, CP_AZPL, N, C, 1.0f, n->aw, C, w->h, N, 1.0f, w->azp, N);
        for (int b = 0; b < B; b++) {
            float tmp[CP_AZPL * 64];
            for (int co = 0; co < CP_AZPL; co++)
                memcpy(tmp + co * 64, w->azp + (size_t)co * N + b * 64, 64 * sizeof(float));
            gemm(0, 0, CP_AZMOVE, 1, CP_AZPL * 64, 1.0f, n->aq, CP_AZPL * 64,
                 tmp, 1, 0.0f, w->azl + (size_t)b * CP_AZMOVE, 1);
        }
    }

    /* ---- value head ---- */
    for (int i = 0; i < N; i++) w->vp[i] = n->vb[0];
    gemm(0, 0, 1, N, C, 1.0f, n->vw, C, w->h, N, 1.0f, w->vp, N);
    ln_relu(w->vp, 1, B, n->vg, n->vc, 1, w->vpx, w->vrstd, w->vph);
    for (int b = 0; b < B; b++) {
        float *vh = w->vh + (size_t)b * CP_VHID;
        const float *x = w->vph + b * 64;
        for (int d = 0; d < CP_VHID; d++) {
            const float *wv = n->v1w + (size_t)d * 64;
            float s = n->v1b[d];
            for (int t = 0; t < 64; t++) s += wv[t] * x[t];
            vh[d] = s > 0.0f ? s : 0.0f;
        }
        float s = n->v2b[0];
        for (int d = 0; d < CP_VHID; d++) s += n->v2w[d] * vh[d];
        w->vraw[b] = s;
        w->v[b] = tanhf(s);
    }
}


/* ======================================================================== */
/*                    the INFERENCE-FAITHFUL forward pass                   */
/* ======================================================================== */
/*
 * AlphaZero normalised with BatchNorm, and BatchNorm at INFERENCE time is a
 * fixed per-channel affine that folds into the preceding convolution: scale
 * the filter rows by gamma/sigma, fold the shift into the bias, and the
 * normalisation costs literally nothing at play time.  conv_forward() above
 * uses LayerNorm instead, because LayerNorm is what the rest of this codebase
 * uses and because it has a clean backward pass -- but LayerNorm does NOT fold,
 * and measuring self-play throughput with an unfoldable normaliser would
 * charge the convolutional network for a cost AlphaZero never paid.
 *
 * So the throughput numbers that decide the question come from HERE: im2col,
 * sgemm, and a fused bias+relu (and a fused bias+skip+relu at the residual
 * junction).  This is the cheapest correct inference form of the architecture,
 * and it is the number a convolutional design should be judged on.
 *
 * conv_forward() and conv_forward_fold() differ ONLY in the normaliser, so the
 * two rates measured side by side also say what LayerNorm would cost if it
 * were chosen anyway.
 */

static void bias_relu(float *X, const float *b, int C, int N)
{
    for (int c = 0; c < C; c++) {
        float *x = X + (size_t)c * N;
        const float bb = b[c];
        int i = 0;
#if HAVE_NEON
        const float32x4_t vb = vdupq_n_f32(bb), vz = vdupq_n_f32(0.0f);
        for (; i + 16 <= N; i += 16) {
            vst1q_f32(x + i,      vmaxq_f32(vaddq_f32(vld1q_f32(x + i),      vb), vz));
            vst1q_f32(x + i + 4,  vmaxq_f32(vaddq_f32(vld1q_f32(x + i + 4),  vb), vz));
            vst1q_f32(x + i + 8,  vmaxq_f32(vaddq_f32(vld1q_f32(x + i + 8),  vb), vz));
            vst1q_f32(x + i + 12, vmaxq_f32(vaddq_f32(vld1q_f32(x + i + 12), vb), vz));
        }
#endif
        for (; i < N; i++) { const float v = x[i] + bb; x[i] = v > 0.0f ? v : 0.0f; }
    }
}

/* h = relu(T + bias + h): the residual junction, one pass. */
static void bias_skip_relu(float *H, const float *T, const float *b, int C, int N)
{
    for (int c = 0; c < C; c++) {
        float *h = H + (size_t)c * N;
        const float *t = T + (size_t)c * N;
        const float bb = b[c];
        int i = 0;
#if HAVE_NEON
        const float32x4_t vb = vdupq_n_f32(bb), vz = vdupq_n_f32(0.0f);
        for (; i + 16 <= N; i += 16) {
            for (int k = 0; k < 16; k += 4)
                vst1q_f32(h + i + k, vmaxq_f32(vaddq_f32(vaddq_f32(vld1q_f32(t + i + k), vb),
                                                         vld1q_f32(h + i + k)), vz));
        }
#endif
        for (; i < N; i++) { const float v = t[i] + bb + h[i]; h[i] = v > 0.0f ? v : 0.0f; }
    }
}

static void conv_forward_fold(const ConvNet *n, ConvWs *w, int B)
{
    const int C = n->filters, N = B * 64;

    im2col(w->in, CP_PLANES, B, w->col);
    gemm(0, 0, C, N, 9 * CP_PLANES, 1.0f, n->sw, 9 * CP_PLANES, w->col, N, 0.0f, w->h, N);
    bias_relu(w->h, n->sb, C, N);

    for (int blk = 0; blk < n->blocks; blk++) {
        const float *w1 = n->bw[0] + (size_t)blk * 9 * C * C;
        const float *w2 = n->bw[1] + (size_t)blk * 9 * C * C;
        const float *b1 = n->bb[0] + (size_t)blk * C, *b2 = n->bb[1] + (size_t)blk * C;
        im2col(w->h, C, B, w->col);
        gemm(0, 0, C, N, 9 * C, 1.0f, w1, 9 * C, w->col, N, 0.0f, w->u1, N);
        bias_relu(w->u1, b1, C, N);
        im2col(w->u1, C, B, w->col);
        gemm(0, 0, C, N, 9 * C, 1.0f, w2, 9 * C, w->col, N, 0.0f, w->t, N);
        bias_skip_relu(w->h, w->t, b2, C, N);
    }

    gemm(0, 0, CP_PPL, N, C, 1.0f, n->pw, C, w->h, N, 0.0f, w->pph, N);
    bias_relu(w->pph, n->pb, CP_PPL, N);
    for (int b = 0; b < B; b++) {
        float *q = w->q + (size_t)b * CP_PDIM;
        for (int d = 0; d < CP_PDIM; d++) {
            const float *wq = n->qw + (size_t)d * (CP_PPL * 64);
            float sacc = n->qb[d];
            for (int co = 0; co < CP_PPL; co++) {
                const float *x = w->pph + (size_t)co * N + b * 64;
                for (int t = 0; t < 64; t++) sacc += wq[co * 64 + t] * x[t];
            }
            q[d] = sacc;
        }
    }

    if (n->pol == POL_AZPLANE) {
        gemm(0, 0, CP_AZPL, N, C, 1.0f, n->aw, C, w->h, N, 0.0f, w->azp, N);
        bias_relu(w->azp, n->ab, CP_AZPL, N);
        for (int b = 0; b < B; b++) {
            float tmp[CP_AZPL * 64];
            for (int co = 0; co < CP_AZPL; co++)
                memcpy(tmp + co * 64, w->azp + (size_t)co * N + b * 64, 64 * sizeof(float));
            gemm(0, 0, CP_AZMOVE, 1, CP_AZPL * 64, 1.0f, n->aq, CP_AZPL * 64,
                 tmp, 1, 0.0f, w->azl + (size_t)b * CP_AZMOVE, 1);
        }
    }

    gemm(0, 0, 1, N, C, 1.0f, n->vw, C, w->h, N, 0.0f, w->vph, N);
    bias_relu(w->vph, n->vb, 1, N);
    for (int b = 0; b < B; b++) {
        float *vh = w->vh + (size_t)b * CP_VHID;
        const float *x = w->vph + b * 64;
        for (int d = 0; d < CP_VHID; d++) {
            const float *wv = n->v1w + (size_t)d * 64;
            float sacc = n->v1b[d];
            for (int t = 0; t < 64; t++) sacc += wv[t] * x[t];
            vh[d] = sacc > 0.0f ? sacc : 0.0f;
        }
        float sacc = n->v2b[0];
        for (int d = 0; d < CP_VHID; d++) sacc += n->v2w[d] * vh[d];
        w->v[b] = tanhf(sacc);
    }
}

/* The same pass with each STAGE timed on its own, so the report can say how
 * much of the conv cost is irreducible arithmetic (the sgemm) and how much is
 * data movement (im2col) or pointwise work.  A conclusion that rests on my
 * im2col being slow would not be worth much; a conclusion that rests on the
 * sgemm alone is implementation-independent. */
typedef struct { double im2col, gemm, point, head; } ConvSplit;

static void conv_forward_timed(const ConvNet *n, ConvWs *w, int B, ConvSplit *sp)
{
    const int C = n->filters, N = B * 64;
    double t0;

    t0 = thread_cpu_sec(); im2col(w->in, CP_PLANES, B, w->col); sp->im2col += thread_cpu_sec() - t0;
    t0 = thread_cpu_sec();
    gemm(0, 0, C, N, 9 * CP_PLANES, 1.0f, n->sw, 9 * CP_PLANES, w->col, N, 0.0f, w->h, N);
    sp->gemm += thread_cpu_sec() - t0;
    t0 = thread_cpu_sec(); bias_relu(w->h, n->sb, C, N); sp->point += thread_cpu_sec() - t0;

    for (int blk = 0; blk < n->blocks; blk++) {
        const float *w1 = n->bw[0] + (size_t)blk * 9 * C * C;
        const float *w2 = n->bw[1] + (size_t)blk * 9 * C * C;
        const float *b1 = n->bb[0] + (size_t)blk * C, *b2 = n->bb[1] + (size_t)blk * C;
        t0 = thread_cpu_sec(); im2col(w->h, C, B, w->col); sp->im2col += thread_cpu_sec() - t0;
        t0 = thread_cpu_sec();
        gemm(0, 0, C, N, 9 * C, 1.0f, w1, 9 * C, w->col, N, 0.0f, w->u1, N);
        sp->gemm += thread_cpu_sec() - t0;
        t0 = thread_cpu_sec(); bias_relu(w->u1, b1, C, N); sp->point += thread_cpu_sec() - t0;
        t0 = thread_cpu_sec(); im2col(w->u1, C, B, w->col); sp->im2col += thread_cpu_sec() - t0;
        t0 = thread_cpu_sec();
        gemm(0, 0, C, N, 9 * C, 1.0f, w2, 9 * C, w->col, N, 0.0f, w->t, N);
        sp->gemm += thread_cpu_sec() - t0;
        t0 = thread_cpu_sec(); bias_skip_relu(w->h, w->t, b2, C, N); sp->point += thread_cpu_sec() - t0;
    }

    t0 = thread_cpu_sec();
    gemm(0, 0, CP_PPL, N, C, 1.0f, n->pw, C, w->h, N, 0.0f, w->pph, N);
    bias_relu(w->pph, n->pb, CP_PPL, N);
    for (int b = 0; b < B; b++) {
        float *q = w->q + (size_t)b * CP_PDIM;
        for (int d = 0; d < CP_PDIM; d++) {
            const float *wq = n->qw + (size_t)d * (CP_PPL * 64);
            float sacc = n->qb[d];
            for (int co = 0; co < CP_PPL; co++) {
                const float *x = w->pph + (size_t)co * N + b * 64;
                for (int t = 0; t < 64; t++) sacc += wq[co * 64 + t] * x[t];
            }
            q[d] = sacc;
        }
    }
    gemm(0, 0, 1, N, C, 1.0f, n->vw, C, w->h, N, 0.0f, w->vph, N);
    bias_relu(w->vph, n->vb, 1, N);
    for (int b = 0; b < B; b++) {
        const float *x = w->vph + b * 64;
        float *vh = w->vh + (size_t)b * CP_VHID;
        for (int d = 0; d < CP_VHID; d++) {
            const float *wv = n->v1w + (size_t)d * 64;
            float sacc = n->v1b[d];
            for (int t = 0; t < 64; t++) sacc += wv[t] * x[t];
            vh[d] = sacc > 0.0f ? sacc : 0.0f;
        }
        float sacc = n->v2b[0];
        for (int d = 0; d < CP_VHID; d++) sacc += n->v2w[d] * vh[d];
        w->v[b] = tanhf(sacc);
    }
    sp->head += thread_cpu_sec() - t0;
}

/* ======================================================================== */
/*                              backward pass                               */
/* ======================================================================== */
/* Gradients have exactly the ConvNet layout, so a ConvNet doubles as the
 * gradient buffer.  Only the trunk + heads are here; the per-move logit layer
 * is shared with the MLP and its gradient is unchanged.
 *
 * dq is dLoss/dq (CP_PDIM per position), dv is dLoss/dv (post-tanh). */

typedef struct {
    float *dh, *dt, *du, *dcol, *dpp, *dvp, *dhead;
    float *blob;
} ConvBwWs;

static void bw_init(ConvBwWs *bw, int B, int C)
{
    const size_t N = (size_t)B * 64;
    const size_t hs = (size_t)(C > CP_PPL ? C : CP_PPL) * N;
    const size_t n = 3 * (size_t)C * N + 9 * (size_t)C * N + (size_t)CP_PPL * N + N + hs + 64;
    bw->blob = (float *)calloc(n, sizeof(float));
    if (!bw->blob) { fprintf(stderr, "OOM bw workspace\n"); exit(1); }
    float *c = bw->blob;
    bw->dh = bump(&c, (size_t)C * N);
    bw->dt = bump(&c, (size_t)C * N);
    bw->du = bump(&c, (size_t)C * N);
    bw->dcol = bump(&c, 9 * (size_t)C * N);
    bw->dpp = bump(&c, (size_t)CP_PPL * N);
    bw->dvp = bump(&c, N);
    bw->dhead = bump(&c, hs);
}
static void bw_free(ConvBwWs *bw) { free(bw->blob); bw->blob = NULL; }

static void conv_backward(const ConvNet *n, ConvNet *gr, ConvWs *w, ConvBwWs *bw,
                          int B, const float *dq, const float *dv)
{
    const int C = n->filters, N = B * 64;
    const int L_last = 2 * n->blocks;      /* index of the final `act` layer   */

    memset(bw->dh, 0, sizeof(float) * (size_t)C * N);

    /* ---- value head ---- */
    memset(bw->dvp, 0, sizeof(float) * (size_t)N);
    {
        float *dvph = bw->dvp;   /* reuse: first dL/d(vph), then dL/d(vp)      */
        for (int b = 0; b < B; b++) {
            const float draw = dv[b] * (1.0f - w->v[b] * w->v[b]);
            const float *vh = w->vh + (size_t)b * CP_VHID;
            float dvh[CP_VHID];
            gr->v2b[0] += draw;
            for (int d = 0; d < CP_VHID; d++) {
                gr->v2w[d] += draw * vh[d];
                dvh[d] = (vh[d] > 0.0f) ? draw * n->v2w[d] : 0.0f;
            }
            const float *x = w->vph + b * 64;
            float *dx = dvph + b * 64;
            for (int d = 0; d < CP_VHID; d++) {
                const float g = dvh[d];
                if (g == 0.0f) continue;
                gr->v1b[d] += g;
                float *gw = gr->v1w + (size_t)d * 64;
                const float *wv = n->v1w + (size_t)d * 64;
                for (int t = 0; t < 64; t++) { gw[t] += g * x[t]; dx[t] += g * wv[t]; }
            }
        }
        /* through relu(LN(vp)) */
        for (int b = 0; b < B; b++)
            for (int t = 0; t < 64; t++)
                if (w->vph[b * 64 + t] <= 0.0f) dvph[b * 64 + t] = 0.0f;
        float *dvp_pre = bw->dhead;
        memset(dvp_pre, 0, sizeof(float) * (size_t)N);
        ln_backward(dvph, w->vpx, w->vrstd, 1, B, n->vg, gr->vg, gr->vc, dvp_pre);
        /* 1x1 conv C -> 1 */
        for (int b = 0; b < B; b++)
            for (int t = 0; t < 64; t++) gr->vb[0] += dvp_pre[b * 64 + t];
        gemm(0, 1, 1, C, N, 1.0f, dvp_pre, N, w->act + (size_t)L_last * C * N, N,
             1.0f, gr->vw, C);
        gemm(1, 0, C, N, 1, 1.0f, n->vw, C, dvp_pre, N, 1.0f, bw->dh, N);
    }

    /* ---- policy head ---- */
    {
        memset(bw->dpp, 0, sizeof(float) * (size_t)CP_PPL * N);
        for (int b = 0; b < B; b++) {
            const float *dqb = dq + (size_t)b * CP_PDIM;
            for (int d = 0; d < CP_PDIM; d++) {
                const float g = dqb[d];
                gr->qb[d] += g;
                float *gw = gr->qw + (size_t)d * (CP_PPL * 64);
                const float *wq = n->qw + (size_t)d * (CP_PPL * 64);
                for (int co = 0; co < CP_PPL; co++) {
                    const float *x = w->pph + (size_t)co * N + b * 64;
                    float *dx = bw->dpp + (size_t)co * N + b * 64;
                    for (int t = 0; t < 64; t++) {
                        gw[co * 64 + t] += g * x[t];
                        dx[t] += g * wq[co * 64 + t];
                    }
                }
            }
        }
        for (int i = 0; i < CP_PPL * N; i++) if (w->pph[i] <= 0.0f) bw->dpp[i] = 0.0f;
        float *dpre = bw->dhead;
        memset(dpre, 0, sizeof(float) * (size_t)CP_PPL * N);
        ln_backward(bw->dpp, w->ppx, w->prstd, CP_PPL, B, n->pg, gr->pg, gr->pc, dpre);
        for (int co = 0; co < CP_PPL; co++)
            for (int i = 0; i < N; i++) gr->pb[co] += dpre[(size_t)co * N + i];
        gemm(0, 1, CP_PPL, C, N, 1.0f, dpre, N, w->act + (size_t)L_last * C * N, N,
             1.0f, gr->pw, C);
        gemm(1, 0, C, N, CP_PPL, 1.0f, n->pw, C, dpre, N, 1.0f, bw->dh, N);
    }

    /* ---- residual blocks, in reverse ---- */
    int L = L_last;
    for (int blk = n->blocks - 1; blk >= 0; blk--) {
        const float *w1 = n->bw[0] + (size_t)blk * 9 * C * C;
        const float *w2 = n->bw[1] + (size_t)blk * 9 * C * C;
        float *g1w = gr->bw[0] + (size_t)blk * 9 * C * C;
        float *g2w = gr->bw[1] + (size_t)blk * 9 * C * C;
        const float *g1 = n->bg[0] + (size_t)blk * C, *g2 = n->bg[1] + (size_t)blk * C;

        /* relu at the residual junction */
        const float *out = w->act + (size_t)L * C * N;
        for (int i = 0; i < C * N; i++) if (out[i] <= 0.0f) bw->dh[i] = 0.0f;
        /* the skip carries dh straight back to the block input */
        memcpy(bw->du, bw->dh, sizeof(float) * (size_t)C * N);   /* dL/d(LN out) */

        /* second conv: LN (no relu) then the 3x3 */
        memset(bw->dt, 0, sizeof(float) * (size_t)C * N);
        ln_backward(bw->du, w->xhat + (size_t)L * C * N, w->rstd + (size_t)L * B,
                    C, B, g2, gr->bg[1] + (size_t)blk * C, gr->bc[1] + (size_t)blk * C,
                    bw->dt);
        for (int co = 0; co < C; co++)
            for (int i = 0; i < N; i++) gr->bb[1][(size_t)blk * C + co] += bw->dt[(size_t)co * N + i];
        gemm(0, 1, C, 9 * C, N, 1.0f, bw->dt, N, w->cols + (size_t)L * w->colstride, N,
             1.0f, g2w, 9 * C);
        gemm(1, 0, 9 * C, N, C, 1.0f, w2, 9 * C, bw->dt, N, 0.0f, bw->dcol, N);
        memset(bw->du, 0, sizeof(float) * (size_t)C * N);
        col2im_add(bw->dcol, C, B, bw->du);
        L--;

        /* first conv: relu then LN then the 3x3 */
        {
            const float *u1 = w->act + (size_t)L * C * N;
            for (int i = 0; i < C * N; i++) if (u1[i] <= 0.0f) bw->du[i] = 0.0f;
        }
        memset(bw->dt, 0, sizeof(float) * (size_t)C * N);
        ln_backward(bw->du, w->xhat + (size_t)L * C * N, w->rstd + (size_t)L * B,
                    C, B, g1, gr->bg[0] + (size_t)blk * C, gr->bc[0] + (size_t)blk * C,
                    bw->dt);
        for (int co = 0; co < C; co++)
            for (int i = 0; i < N; i++) gr->bb[0][(size_t)blk * C + co] += bw->dt[(size_t)co * N + i];
        gemm(0, 1, C, 9 * C, N, 1.0f, bw->dt, N, w->cols + (size_t)L * w->colstride, N,
             1.0f, g1w, 9 * C);
        gemm(1, 0, 9 * C, N, C, 1.0f, w1, 9 * C, bw->dt, N, 0.0f, bw->dcol, N);
        col2im_add(bw->dcol, C, B, bw->dh);   /* += : the skip is already there */
        L--;
    }

    /* ---- stem ---- */
    {
        const float *h0 = w->act;    /* L == 0 */
        for (int i = 0; i < C * N; i++) if (h0[i] <= 0.0f) bw->dh[i] = 0.0f;
        memset(bw->dt, 0, sizeof(float) * (size_t)C * N);
        ln_backward(bw->dh, w->xhat, w->rstd, C, B, n->sg, gr->sg, gr->sc, bw->dt);
        for (int co = 0; co < C; co++)
            for (int i = 0; i < N; i++) gr->sb[co] += bw->dt[(size_t)co * N + i];
        gemm(0, 1, C, 9 * CP_PLANES, N, 1.0f, bw->dt, N, w->cols, N, 1.0f,
             gr->sw, 9 * CP_PLANES);
    }
}

/* ======================================================================== */
/*                                 corpus                                   */
/* ======================================================================== */
/* The same construction tools/profile.c uses: 512 positions from deterministic
 * random self-play, so both architectures see the same distribution of piece
 * counts and the same cache behaviour. */

#define CORPUS_N 512

typedef struct {
    Position p;
    uint16_t fidx[NF_MAXACTIVE];
    int      nf;
    int      nmoves;
    Move     mv[MAX_MOVES];
} CPos;

static CPos g_corpus[CORPUS_N];
static int  g_ncorpus = 0;

static void corpus_build(void)
{
    Position p;
    Move mv[MAX_MOVES];
    int plies = 0;
    g_rs = 0x243F6A8885A308D3ULL;
    pos_startpos(&p);
    while (g_ncorpus < CORPUS_N) {
        const int n = gen_legal(&p, mv);
        if (n == 0 || plies >= 160) { pos_startpos(&p); plies = 0; continue; }
        CPos *c = &g_corpus[g_ncorpus++];
        c->p = p;
        c->nf = nn_features(&p, c->fidx);
        c->nmoves = n;
        memcpy(c->mv, mv, sizeof(Move) * (size_t)n);
        Undo u;
        make_move(&p, mv[rnd_u64() % (uint64_t)n], &u);
        plies++;
    }
}

/* ======================================================================== */
/*                    candidate extra SPARSE features                       */
/* ======================================================================== */
/*
 * THE CONTRACT TEST, applied one feature at a time.  docs/FROM_SCRATCH.md
 * permits "the rules of chess" and forbids "every form of chess evaluation".
 * The operational question for a candidate feature is:
 *
 *     Could a player who knows ONLY the rules -- who has no opinion whatever
 *     about which positions are good -- compute this number?
 *
 * If yes it is a projection of the rules and it is allowed; the network still
 * has to learn, from game results alone, whether it matters and in which
 * direction.  If computing it requires ranking pieces, squares or moves, it
 * carries an opinion and it is forbidden, however useful it would be.
 *
 *   ALLOWED   attack maps.  "Is square s attacked by side c" is a function of
 *             the move rules and the occupancy, nothing else.  A random mover
 *             could compute it.
 *   ALLOWED   per-piece-type mobility counts.  A count of legal moves.  The
 *             claim "mobility is good" is an opinion; the COUNT is not, and
 *             the network is free to learn that it means nothing.
 *   ALLOWED   repetition count and the opponent's last move.  Both are facts
 *             about the game so far.
 *   FORBIDDEN "squares attacked by a LOWER-VALUED piece".  Lower-valued
 *             requires a piece-value ordering, which is the first line of the
 *             forbidden list in docs/FROM_SCRATCH.md.  It is dropped, and it
 *             is dropped even though it is probably the single most useful
 *             feature on this list -- which is exactly why the line has to be
 *             drawn by the rule and not by the payoff.
 *   FORBIDDEN static exchange evaluation, "hanging piece", "defended by a
 *             pawn".  Same reason: all of them rank pieces.
 *
 * A borderline case argued explicitly: "attacked by them AND not attacked by
 * us" needs no piece values -- it is set arithmetic on two allowed bitboards.
 * It is still NOT included, on a different ground: it is the conclusion we
 * hope the network reaches, hand-computed.  Given both attack maps the first
 * layer can form that conjunction itself, so supplying it adds no information
 * and only narrows what the network is allowed to notice.  Supply the facts,
 * not the inference.
 */

/* Both attack maps, as 128 extra binary feature indices. */
static int feat_attack_maps(const Position *p, uint16_t *idx, int base)
{
    int n = 0;
    const int us = p->side, them = us ^ 1;
    const int flip = (us == BLACK) ? 56 : 0;
    for (int side = 0; side < 2; side++) {
        const int c = side ? them : us;
        uint64_t att = 0;
        uint64_t b = p->piece[c][PAWN];
        while (b) { const int sq = __builtin_ctzll(b); b &= b - 1; att |= attacks_pawn(sq, c); }
        b = p->piece[c][KNIGHT];
        while (b) { const int sq = __builtin_ctzll(b); b &= b - 1; att |= attacks_knight(sq); }
        b = p->piece[c][BISHOP] | p->piece[c][QUEEN];
        while (b) { const int sq = __builtin_ctzll(b); b &= b - 1; att |= attacks_bishop(sq, p->all); }
        b = p->piece[c][ROOK] | p->piece[c][QUEEN];
        while (b) { const int sq = __builtin_ctzll(b); b &= b - 1; att |= attacks_rook(sq, p->all); }
        b = p->piece[c][KING];
        while (b) { const int sq = __builtin_ctzll(b); b &= b - 1; att |= attacks_king(sq); }
        while (att) {
            const int sq = __builtin_ctzll(att); att &= att - 1;
            idx[n++] = (uint16_t)(base + side * 64 + (sq ^ flip));
        }
    }
    return n;
}

/* Per-piece-type legal-move counts, bucketed.  6 types x 8 buckets = 48
 * feature slots, 6 of them active. */
static int feat_mobility(const Position *p, uint16_t *idx, int base)
{
    Move mv[MAX_MOVES];
    const int n = gen_legal(p, mv);
    int cnt[NPIECES] = {0};
    for (int i = 0; i < n; i++) {
        const int pt = p->board[MV_FROM(mv[i])];
        if (pt >= 0 && pt < NPIECES) cnt[pt]++;
    }
    int k = 0;
    for (int pt = 0; pt < NPIECES; pt++) {
        int b = cnt[pt]; if (b > 7) b = 7;
        idx[k++] = (uint16_t)(base + pt * 8 + b);
    }
    return k;
}

/* The opponent's last move (from, to) plus the repetition count.  2 + 1 active
 * out of 131 slots.  This is the cheapest thing on the list by a wide margin
 * and it is the one that addresses "we cannot see that we are repeating". */
static int feat_lastmove_rep(const Game *g, uint16_t *idx, int base)
{
    int n = 0;
    if (g->ply > 0) {
        const Move m = g->moves[g->ply - 1];
        idx[n++] = (uint16_t)(base + MV_FROM(m));
        idx[n++] = (uint16_t)(base + 64 + MV_TO(m));
    }
    int rep = 0;
    const uint64_t k = g->pos.key;
    for (int i = g->hist_len - 1; i >= 0; i--) if (g->hist[i] == k) rep++;
    if (rep > 3) rep = 3;
    idx[n++] = (uint16_t)(base + 128 + rep);
    return n;
}

/* AlphaZero's 8 history positions, as sparse piece-placement features.  Each
 * previous position contributes up to 32 active indices out of 768 slots. */
static int feat_history(const Game *g, int nhist, uint16_t *idx, int base)
{
    int n = 0;
    /* Walk backwards by unmaking; the caller passes a scratch Game copy. */
    Game tmp = *g;
    for (int h = 0; h < nhist; h++) {
        if (tmp.ply == 0) break;
        tmp.ply--;
        unmake_move(&tmp.pos, tmp.moves[tmp.ply], &tmp.undos[tmp.ply]);
        const int us = g->pos.side, them = us ^ 1;
        const int flip = (us == BLACK) ? 56 : 0;
        const int off = base + h * 768;
        for (int pt = 0; pt < NPIECES; pt++) {
            uint64_t b = tmp.pos.piece[us][pt];
            while (b) { const int sq = __builtin_ctzll(b); b &= b - 1;
                        idx[n++] = (uint16_t)(off + pt * 64 + (sq ^ flip)); }
            b = tmp.pos.piece[them][pt];
            while (b) { const int sq = __builtin_ctzll(b); b &= b - 1;
                        idx[n++] = (uint16_t)(off + 384 + pt * 64 + (sq ^ flip)); }
        }
    }
    return n;
}

/* ======================================================================== */
/*                                 threads                                  */
/* ======================================================================== */

typedef struct {
    pthread_mutex_t m;
    pthread_cond_t  c;
    int need, seen, gen;
} Gate;
static Gate g_gate;

static void gate_init(Gate *g, int n)
{
    pthread_mutex_init(&g->m, NULL); pthread_cond_init(&g->c, NULL);
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

typedef struct {
    const ConvNet *net;
    int    B, iters, tid, bwd, ln;
    double rate;
    double sink;
    char   pad[64];
} CTask;

static void *conv_thread(void *arg)
{
    CTask *t = (CTask *)arg;
    ConvWs w; ConvBwWs bw;
    ws_init(&w, t->B, t->net->filters, t->net->blocks, t->net->pol, t->bwd);
    ConvNet *gr = NULL;
    float *dq = NULL, *dv = NULL;
    if (t->bwd) {
        bw_init(&bw, t->B, t->net->filters);
        gr = (ConvNet *)malloc(sizeof(ConvNet));
        conv_init(gr, t->net->blocks, t->net->filters, t->net->pol, 99);
        memset(gr->blob, 0, sizeof(float) * gr->nparam);
        dq = (float *)calloc((size_t)t->B * CP_PDIM, sizeof(float));
        dv = (float *)calloc((size_t)t->B, sizeof(float));
        for (int i = 0; i < t->B * CP_PDIM; i++) dq[i] = 0.01f;
        for (int i = 0; i < t->B; i++) dv[i] = 0.01f;
    }

    /* prime */
    for (int b = 0; b < t->B; b++) {
        float tmp[CP_IN];
        planes_build(&g_corpus[b % g_ncorpus].p, 0, tmp);
        for (int c = 0; c < CP_PLANES; c++)
            memcpy(w.in + (size_t)c * t->B * 64 + b * 64, tmp + c * 64, 64 * sizeof(float));
    }
    if (t->ln || t->bwd) conv_forward(t->net, &w, t->B);
    else                 conv_forward_fold(t->net, &w, t->B);

    gate_wait(&g_gate);
    const double t0 = thread_cpu_sec();
    for (int it = 0; it < t->iters; it++) {
        for (int b = 0; b < t->B; b++) {
            float tmp[CP_IN];
            const int k = (it * t->B + b + t->tid * 37) % g_ncorpus;
            planes_build(&g_corpus[k].p, 0, tmp);
            for (int c = 0; c < CP_PLANES; c++)
                memcpy(w.in + (size_t)c * t->B * 64 + b * 64, tmp + c * 64, 64 * sizeof(float));
        }
        if (t->ln || t->bwd) conv_forward(t->net, &w, t->B);
        else                 conv_forward_fold(t->net, &w, t->B);
        if (t->bwd) conv_backward(t->net, gr, &w, &bw, t->B, dq, dv);
        t->sink += w.v[0];
    }
    t->rate = (double)t->iters * t->B / (thread_cpu_sec() - t0);
    ws_free(&w);
    if (t->bwd) { bw_free(&bw); conv_free(gr); free(gr); free(dq); free(dv); }
    return NULL;
}

static double bench_conv_n(const ConvNet *net, int nthread, int B, int iters, int bwd,
                           int ln, double *sink)
{
    pthread_t th[32];
    CTask task[32];
    gate_init(&g_gate, nthread);
    for (int i = 0; i < nthread; i++) {
        memset(&task[i], 0, sizeof task[i]);
        task[i].net = net; task[i].B = B; task[i].iters = iters; task[i].tid = i;
        task[i].bwd = bwd; task[i].ln = ln;
    }
    for (int i = 0; i < nthread; i++) pthread_create(&th[i], NULL, conv_thread, &task[i]);
    for (int i = 0; i < nthread; i++) pthread_join(th[i], NULL);
    double agg = 0.0;
    for (int i = 0; i < nthread; i++) { *sink += task[i].sink; agg += task[i].rate; }
    return agg;
}

static double bench_conv(const ConvNet *net, int nthread, int B, int iters, int bwd, double *sink)
{
    return bench_conv_n(net, nthread, B, iters, bwd, 0, sink);
}

/* ======================================================================== */
/*                               the MLP, here                              */
/* ======================================================================== */
/* Measured in THIS process so the ratio is not a comparison between two
 * benchmarks.  Should reproduce tools/profile.c to within noise. */

static Trunk *g_trunk;
static Head  *g_heads;
#define NHEADS      16
#define HEAD_PERIOD 256

typedef struct { int iters, tid; double rate, sink; char pad[64]; } MTask;

static void *mlp_thread(void *arg)
{
    MTask *t = (MTask *)arg;
    Fwd fw;
    gate_wait(&g_gate);
    const double t0 = thread_cpu_sec();
    for (int i = 0; i < t->iters; i++) {
        const CPos *c = &g_corpus[(i + t->tid * 37) % g_ncorpus];
        const Head *hd = &g_heads[(i / HEAD_PERIOD + t->tid) % NHEADS];
        nn_eval(g_trunk, hd, c->fidx, c->nf, &fw);
        t->sink += fw.v;
    }
    t->rate = (double)t->iters / (thread_cpu_sec() - t0);
    return NULL;
}

static double bench_mlp(int nthread, int iters, double *sink)
{
    pthread_t th[32]; MTask task[32];
    gate_init(&g_gate, nthread);
    for (int i = 0; i < nthread; i++) { memset(&task[i], 0, sizeof task[i]);
                                        task[i].iters = iters; task[i].tid = i; }
    for (int i = 0; i < nthread; i++) pthread_create(&th[i], NULL, mlp_thread, &task[i]);
    for (int i = 0; i < nthread; i++) pthread_join(th[i], NULL);
    double agg = 0.0;
    for (int i = 0; i < nthread; i++) { *sink += task[i].sink; agg += task[i].rate; }
    return agg;
}

/* ======================================================================== */
/*                              the size table                              */
/* ======================================================================== */

typedef struct { int blocks, filters; } CvSize;
static const CvSize SIZES[] = { {2, 16}, {4, 32}, {6, 64}, {10, 64} };
#define NSIZES ((int)(sizeof SIZES / sizeof SIZES[0]))

/* The MLP's own arithmetic, counted the same way as conv_macs(). */
static double mlp_macs(void)
{
    const double active = 35.0;   /* mean active sparse features              */
    double m = active * NF_ACC;                 /* W0 gather                  */
    m += (double)NF_ACC * NF_HID;               /* W1 residual                */
    m += (double)NF_HID * NF_PDIM;              /* Wp policy query            */
    m += active * NF_VACC;                      /* W0v gather                 */
    m += (double)NF_VACC * NF_VHID + NF_VHID;   /* value head                 */
    return m;
}

/* ======================================================================== */
/*                                  modes                                   */
/* ======================================================================== */

/* ------------------------------------------------------------------------
 * PAIRED MEASUREMENT.
 *
 * This laptop is not quiet -- other jobs come and go, and a single-thread rate
 * measured at one moment and compared against another measured a minute later
 * can differ by 2x for reasons that have nothing to do with either network.
 * So every ratio that the conclusion rests on is measured by ALTERNATING the
 * two candidates, several short repetitions each, and taking the best
 * repetition of each.  The best repetition is the least-contended one, and
 * alternating means both candidates see the same run of weather.
 *
 * getloadavg() is printed alongside so the reader can see what the machine was
 * doing while the numbers were taken.
 * ------------------------------------------------------------------------ */

static double load_now(void)
{
    double la[3];
    if (getloadavg(la, 3) < 1) return -1.0;
    return la[0];
}

/* Alternates MLP and conv; returns the best of each. */
static void paired_rate(const ConvNet *net, int nthread, int B, int reps,
                        double *mlp_out, double *conv_out, double *sink)
{
    /* Size every repetition to the SAME amount of arithmetic (~4 GMAC) rather
     * than the same iteration count, or the big nets get three iterations and
     * measure nothing but noise. */
    const double macs = conv_macs(net->blocks, net->filters, net->pol);
    const int cit = (int)(4.0e9 / macs / B) + 3;
    const int mit = 150000;
    double bm = 0.0, bc = 0.0;
    for (int r = 0; r < reps; r++) {
        double v = bench_mlp(nthread, mit, sink);              if (v > bm) bm = v;
        v = bench_conv(net, nthread, B, cit, 0, sink);         if (v > bc) bc = v;
    }
    *mlp_out = bm; *conv_out = bc;
}

static double g_mlp_1thread = 0.0;    /* measured, filled by mode_fwd/all     */

static void mode_params(void)
{
    printf("\n=== PARAMETERS AND ARITHMETIC ===\n");
    printf("  input: %d planes x 64 = %d floats; MLP input: %d sparse slots, ~35 active\n",
           CP_PLANES, CP_IN, NF_INPUT);
    printf("\n  %-14s %10s %14s %10s %12s\n",
           "network", "params", "MACs/eval", "x MLP par", "x MLP MACs");
    const double mm = mlp_macs();
    const size_t mp = TRUNK_NPARAM + HEAD_NPARAM;
    printf("  %-14s %10zu %14.0f %10s %12s\n", "MLP (current)", mp, mm, "1.00", "1.00");
    for (int i = 0; i < NSIZES; i++) {
        char nm[32];
        snprintf(nm, sizeof nm, "conv %dx%d", SIZES[i].blocks, SIZES[i].filters);
        const size_t np = conv_nparam(SIZES[i].blocks, SIZES[i].filters, POL_QUERY);
        const double nmac = conv_macs(SIZES[i].blocks, SIZES[i].filters, POL_QUERY);
        printf("  %-14s %10zu %14.0f %10.2f %12.1f\n", nm, np, nmac,
               (double)np / (double)mp, nmac / mm);
    }
    printf("\n  the AlphaZero 8x8x73 policy head, if it were used instead of the\n"
           "  32-d query head, adds %d params and %.0f MACs to EVERY row above --\n"
           "  %.0f times the arithmetic of our entire current network, on its own.\n",
           CP_AZMOVE * CP_AZPL * 64 + CP_AZMOVE + CP_AZPL * 64 + CP_AZPL,
           (double)CP_AZMOVE * CP_AZPL * 64,
           ((double)CP_AZMOVE * CP_AZPL * 64) / mm);
    printf("\n  AlphaZero itself: ~40M params, 19-20 blocks x 256 filters over 8x8x119\n"
           "  input planes -- about %.0f MACs/eval, %.0f times our MLP.\n",
           19.0 * 2.0 * 9.0 * 256.0 * 256.0 * 64.0, 19.0 * 2.0 * 9.0 * 256.0 * 256.0 * 64.0 / mm);
}

static void mode_check(void)
{
    printf("\n=== CORRECTNESS ===\n");

    /* 1. im2col+sgemm vs the naive 7-loop convolution */
    {
        const int C = 8, B = 3, N = B * 64;
        float *X = (float *)malloc(sizeof(float) * (size_t)CP_PLANES * N);
        float *W = (float *)malloc(sizeof(float) * (size_t)C * 9 * CP_PLANES);
        float *b = (float *)malloc(sizeof(float) * (size_t)C);
        float *col = (float *)malloc(sizeof(float) * (size_t)9 * CP_PLANES * N);
        float *Y1 = (float *)malloc(sizeof(float) * (size_t)C * N);
        float *Y2 = (float *)malloc(sizeof(float) * (size_t)C * N);
        g_rs = 7;
        for (int i = 0; i < CP_PLANES * N; i++) X[i] = rnd_gauss();
        for (int i = 0; i < C * 9 * CP_PLANES; i++) W[i] = rnd_gauss() * 0.1f;
        for (int i = 0; i < C; i++) b[i] = rnd_gauss() * 0.1f;
        im2col(X, CP_PLANES, B, col);
        for (int co = 0; co < C; co++) for (int i = 0; i < N; i++) Y1[(size_t)co * N + i] = b[co];
        gemm(0, 0, C, N, 9 * CP_PLANES, 1.0f, W, 9 * CP_PLANES, col, N, 1.0f, Y1, N);
        conv3x3_naive(W, b, X, C, CP_PLANES, B, Y2);
        double maxd = 0.0;
        for (int i = 0; i < C * N; i++) { const double d = fabs((double)Y1[i] - Y2[i]); if (d > maxd) maxd = d; }
        printf("  im2col+sgemm vs naive 7-loop conv, %d channels x %d positions: max |diff| %.3g  %s\n",
               C, B, maxd, maxd < 1e-4 ? "OK" : "FAIL");
        free(X); free(W); free(b); free(col); free(Y1); free(Y2);
    }

    /* 2. the folded inference path against a naive reference trunk */
    {
        const int blocks = 2, C = 5, B = 2, N = B * 64;
        ConvNet net;
        conv_init(&net, blocks, C, POL_QUERY, 777);
        for (int i = 0; i < C; i++) net.sb[i] = 0.05f * (float)(i + 1);
        for (int k = 0; k < 2; k++)
            for (int i = 0; i < blocks * C; i++) net.bb[k][i] = 0.02f * (float)(i + 1);
        ConvWs w; ws_init(&w, B, C, blocks, POL_QUERY, 0);
        for (int b = 0; b < B; b++) {
            float tmp[CP_IN];
            planes_build(&g_corpus[(b * 53) % g_ncorpus].p, 1, tmp);
            for (int c = 0; c < CP_PLANES; c++)
                memcpy(w.in + (size_t)c * N + b * 64, tmp + c * 64, 64 * sizeof(float));
        }
        conv_forward_fold(&net, &w, B);
        float *ref = (float *)malloc(sizeof(float) * (size_t)C * N);
        float *tmp1 = (float *)malloc(sizeof(float) * (size_t)C * N);
        float *tmp2 = (float *)malloc(sizeof(float) * (size_t)C * N);
        conv3x3_naive(net.sw, net.sb, w.in, C, CP_PLANES, B, ref);
        for (int i = 0; i < C * N; i++) if (ref[i] < 0.0f) ref[i] = 0.0f;
        for (int blk = 0; blk < blocks; blk++) {
            conv3x3_naive(net.bw[0] + (size_t)blk * 9 * C * C, net.bb[0] + (size_t)blk * C,
                          ref, C, C, B, tmp1);
            for (int i = 0; i < C * N; i++) if (tmp1[i] < 0.0f) tmp1[i] = 0.0f;
            conv3x3_naive(net.bw[1] + (size_t)blk * 9 * C * C, net.bb[1] + (size_t)blk * C,
                          tmp1, C, C, B, tmp2);
            for (int i = 0; i < C * N; i++) {
                const float v = tmp2[i] + ref[i];
                ref[i] = v > 0.0f ? v : 0.0f;
            }
        }
        double maxd = 0.0, maxa = 0.0;
        for (int i = 0; i < C * N; i++) {
            const double d = fabs((double)w.h[i] - ref[i]);
            if (d > maxd) maxd = d;
            if (fabs((double)ref[i]) > maxa) maxa = fabs((double)ref[i]);
        }
        printf("  folded inference trunk vs naive reference, %d blocks x %d filters:"
               " max |diff| %.3g (values up to %.3g)  %s\n",
               blocks, C, maxd, maxa, maxd < 1e-4 * (maxa + 1.0) ? "OK" : "FAIL");
        free(ref); free(tmp1); free(tmp2);
        ws_free(&w); conv_free(&net);
    }

    /* 3. finite-difference gradient check on a small net */
    {
        const int blocks = 2, C = 6, B = 2;
        ConvNet net, gr;
        conv_init(&net, blocks, C, POL_QUERY, 4242);
        conv_init(&gr, blocks, C, POL_QUERY, 1);
        memset(gr.blob, 0, sizeof(float) * gr.nparam);
        ConvWs w; ws_init(&w, B, C, blocks, POL_QUERY, 1);
        ConvBwWs bw; bw_init(&bw, B, C);

        for (int b = 0; b < B; b++) {
            float tmp[CP_IN];
            planes_build(&g_corpus[b * 31 % g_ncorpus].p, 0, tmp);
            for (int c = 0; c < CP_PLANES; c++)
                memcpy(w.in + (size_t)c * B * 64 + b * 64, tmp + c * 64, 64 * sizeof(float));
        }

        /* L = sum_b [ sum_d wq_d * q[b][d] + 0.5*(v[b] - target_b)^2 ] */
        float coef[CP_PDIM], target[8];
        g_rs = 11;
        for (int d = 0; d < CP_PDIM; d++) coef[d] = rnd_gauss();
        for (int b = 0; b < B; b++) target[b] = rnd_gauss() * 0.3f;

        float dq[8 * CP_PDIM], dv[8];
        conv_forward(&net, &w, B);
        double L0 = 0.0;
        for (int b = 0; b < B; b++) {
            for (int d = 0; d < CP_PDIM; d++) {
                L0 += coef[d] * w.q[b * CP_PDIM + d];
                dq[b * CP_PDIM + d] = coef[d];
            }
            const double e = w.v[b] - target[b];
            L0 += 0.5 * e * e;
            dv[b] = (float)e;
        }
        conv_backward(&net, &gr, &w, &bw, B, dq, dv);

        /* Compare against central differences on a spread of parameters. */
        const struct { const char *name; float *p; float *g; size_t n; } TS[] = {
            { "stem W",   net.sw,    gr.sw,    9 * CP_PLANES * (size_t)C },
            { "stem b",   net.sb,    gr.sb,    (size_t)C },
            { "stem g",   net.sg,    gr.sg,    (size_t)C },
            { "block0 W1",net.bw[0], gr.bw[0], 9 * (size_t)C * C },
            { "block0 W2",net.bw[1], gr.bw[1], 9 * (size_t)C * C },
            { "block1 W1",net.bw[0] + 9 * (size_t)C * C, gr.bw[0] + 9 * (size_t)C * C, 9 * (size_t)C * C },
            { "policy 1x1", net.pw,  gr.pw,    CP_PPL * (size_t)C },
            { "policy g", net.pg,    gr.pg,    CP_PPL },
            { "query W",  net.qw,    gr.qw,    (size_t)CP_PDIM * CP_PPL * 64 },
            { "value 1x1",net.vw,    gr.vw,    (size_t)C },
            { "value g",  net.vg,    gr.vg,    1 },
            { "value W1", net.v1w,   gr.v1w,   (size_t)CP_VHID * 64 },
            { "value W2", net.v2w,   gr.v2w,   CP_VHID },
        };
        printf("  finite-difference gradient check (central, eps 2e-3, %d blocks x %d filters):\n",
               blocks, C);
        double worst = 0.0;
        for (size_t ti = 0; ti < sizeof TS / sizeof TS[0]; ti++) {
            double wrel = 0.0;
            for (int trial = 0; trial < 6; trial++) {
                const size_t j = (size_t)(rnd_u64() % TS[ti].n);
                const float save = TS[ti].p[j];
                const float eps = 2e-3f;
                double Lp = 0.0, Lm = 0.0;
                TS[ti].p[j] = save + eps;
                conv_forward(&net, &w, B);
                for (int b = 0; b < B; b++) {
                    for (int d = 0; d < CP_PDIM; d++) Lp += coef[d] * w.q[b * CP_PDIM + d];
                    const double e = w.v[b] - target[b]; Lp += 0.5 * e * e;
                }
                TS[ti].p[j] = save - eps;
                conv_forward(&net, &w, B);
                for (int b = 0; b < B; b++) {
                    for (int d = 0; d < CP_PDIM; d++) Lm += coef[d] * w.q[b * CP_PDIM + d];
                    const double e = w.v[b] - target[b]; Lm += 0.5 * e * e;
                }
                TS[ti].p[j] = save;
                const double num = (Lp - Lm) / (2.0 * eps);
                const double ana = TS[ti].g[j];
                const double den = fabs(num) + fabs(ana) + 1e-6;
                const double rel = fabs(num - ana) / den;
                if (rel > wrel) wrel = rel;
            }
            if (wrel > worst) worst = wrel;
            printf("    %-12s worst relative error over 6 samples: %.2e  %s\n",
                   TS[ti].name, wrel, wrel < 2e-2 ? "OK" : "FAIL");
        }
        printf("  overall worst %.2e -- %s\n", worst,
               worst < 2e-2 ? "the backward pass is correct" : "BACKWARD PASS IS WRONG");
        (void)L0;
        ws_free(&w); bw_free(&bw); conv_free(&net); conv_free(&gr);
    }
}

static int g_best_batch[NSIZES];
static double g_best_rate[NSIZES];
static double g_slow1[NSIZES];   /* measured 1-thread slowdown vs MLP */
static double g_conv1[NSIZES];   /* measured 1-thread conv evals/sec  */

static void mode_fwd(void)
{
    double sink = 0.0;
    printf("\n=== FORWARD THROUGHPUT ===\n");
    printf("  backend: %s%s\n", HAVE_ACCELERATE ? "Accelerate sgemm" : "hand-written NEON sgemm",
           HAVE_NEON ? " + NEON pointwise" : "");
    printf("  an eval = input planes + trunk + policy query + value.  It does NOT\n"
           "  include the per-legal-move logits (158 ns), which are identical for\n"
           "  both architectures and would only dilute the ratio.\n");
    printf("  the conv net is normalised the way AlphaZero was -- BatchNorm FOLDED\n"
           "  into the convolution, which at inference costs nothing.  LayerNorm,\n"
           "  which does not fold, is measured separately below.\n");

    double m1 = 0.0, m8 = 0.0;
    for (int r = 0; r < 3; r++) {
        double v = bench_mlp(1, 200000, &sink); if (v > m1) m1 = v;
        v = bench_mlp(8, 150000, &sink);        if (v > m8) m8 = v;
    }
    g_mlp_1thread = m1;
    printf("\n  MLP (current: 788 sparse -> 128 -> residual block -> heads)\n");
    printf("    1 thread  %10.0f evals/sec     8 threads %10.0f evals/sec  (%.2fx)\n",
           m1, m8, m8 / m1);
    printf("    tools/profile.c reports 613169 for the same kernel; it compiles net.c\n"
           "    into ONE translation unit with the benchmark, so nn_eval inlines.  This\n"
           "    number links net.o separately, exactly as src/mcts.c does, so it is the\n"
           "    rate the real search sees -- and it is the CONSERVATIVE choice here,\n"
           "    because a lower MLP baseline makes every conv slowdown below smaller.\n");

    {
        float tmp[CP_IN];
        const int IT = 400000;
        const double t0 = thread_cpu_sec();
        for (int i = 0; i < IT; i++) { planes_build(&g_corpus[i % g_ncorpus].p, 0, tmp); sink += tmp[0]; }
        printf("    building the %d input planes: %.0f ns/position\n",
               CP_PLANES, 1e9 * (thread_cpu_sec() - t0) / IT);
    }

    {   /* the naive convolution, to show what it would have told us */
        const int C = 32, B = 1;
        float *X = (float *)calloc((size_t)C * 64, sizeof(float));
        float *W = (float *)calloc((size_t)C * 9 * C, sizeof(float));
        float *Y = (float *)calloc((size_t)C * 64, sizeof(float));
        float *col = (float *)calloc((size_t)9 * C * 64, sizeof(float));
        g_rs = 3;
        for (int i = 0; i < C * 9 * C; i++) W[i] = rnd_gauss() * 0.1f;
        for (int i = 0; i < C * 64; i++) X[i] = rnd_gauss();
        const int IT = 20000;
        double t0 = thread_cpu_sec();
        for (int i = 0; i < IT; i++) { conv3x3_naive(W, NULL, X, C, C, B, Y); sink += Y[0]; }
        const double naive = 1e9 * (thread_cpu_sec() - t0) / IT;
        t0 = thread_cpu_sec();
        for (int i = 0; i < IT; i++) {
            im2col(X, C, B, col);
            gemm(0, 0, C, 64, 9 * C, 1.0f, W, 9 * C, col, 64, 0.0f, Y, 64);
            sink += Y[0];
        }
        const double fast = 1e9 * (thread_cpu_sec() - t0) / IT;
        printf("    one 3x3 conv, 32->32 filters, one position:"
               " naive 7-loop %.0f ns vs im2col+sgemm %.0f ns (%.0fx)\n", naive, fast, naive / fast);
        free(X); free(W); free(Y); free(col);
    }

    /* ---- batch sweep: find each size's best batch before ranking anything ---- */
    printf("\n  batch sweep, 1 thread, evals/sec (conv is only worth batching if this\n"
           "  curve rises; the self-play loop already runs games_in_flight=8 per thread)\n");
    const int BATCH[] = {1, 4, 8, 16, 32, 64};
    const int NB = (int)(sizeof BATCH / sizeof BATCH[0]);
    printf("  %-11s", "conv");
    for (int j = 0; j < NB; j++) printf(" %9d", BATCH[j]);
    printf("   %9s\n", "best");
    for (int i = 0; i < NSIZES; i++) {
        ConvNet net;
        conv_init(&net, SIZES[i].blocks, SIZES[i].filters, POL_QUERY, 1234);
        const double macs = conv_macs(SIZES[i].blocks, SIZES[i].filters, POL_QUERY);
        char nm[32]; snprintf(nm, sizeof nm, "%d x %d", SIZES[i].blocks, SIZES[i].filters);
        printf("  %-11s", nm);
        double best = 0.0; int bestb = 1;
        for (int j = 0; j < NB; j++) {
            const int B = BATCH[j];
            const int it = (int)(2.0e9 / macs / B) + 3;
            double r = 0.0;
            for (int rep = 0; rep < 2; rep++) {
                const double v = bench_conv(&net, 1, B, it, 0, &sink);
                if (v > r) r = v;
            }
            printf(" %9.0f", r);
            if (r > best) { best = r; bestb = B; }
        }
        printf("   %6.0f@%d\n", best, bestb);
        g_best_batch[i] = bestb; g_best_rate[i] = best;
        conv_free(&net);
    }

    /* ---- headline table: MLP and conv measured ALTERNATELY, at each size's
     *      best batch, so the ratio survives whatever else the machine is
     *      doing.  This is the table the recommendation rests on.        ---- */
    printf("\n  HEADLINE (MLP and conv alternated, best of 6 reps, ~4 GMAC per rep,\n  evals per second of THREAD CPU TIME)\n");
    printf("  %-11s %6s %10s %10s %9s %10s %10s %9s %9s\n",
           "conv", "batch", "MLP 1thr", "conv 1thr", "slowdown",
           "MLP 8thr", "conv 8thr", "slow 8thr", "GFLOP/s");
    for (int i = 0; i < NSIZES; i++) {
        ConvNet net;
        conv_init(&net, SIZES[i].blocks, SIZES[i].filters, POL_QUERY, 1234);
        const double macs = conv_macs(SIZES[i].blocks, SIZES[i].filters, POL_QUERY);
        const int B = g_best_batch[i];
        double m1p = 0, c1p = 0, m8p = 0, c8p = 0;
        paired_rate(&net, 1, B, 6, &m1p, &c1p, &sink);
        paired_rate(&net, 8, B, 6, &m8p, &c8p, &sink);
        g_slow1[i] = m1p / c1p;
        g_conv1[i] = c1p;
        char nm[32]; snprintf(nm, sizeof nm, "%d x %d", SIZES[i].blocks, SIZES[i].filters);
        printf("  %-11s %6d %10.0f %10.0f %8.0fx %10.0f %10.0f %8.0fx %9.1f\n",
               nm, B, m1p, c1p, m1p / c1p, m8p, c8p, m8p / c8p,
               2.0 * macs * c1p * 1e-9);
        conv_free(&net);
    }
    printf("  load average during this table: %.1f\n", load_now());
    printf("  The 8-thread slowdown is WORSE than the 1-thread slowdown at every\n"
           "  size.  Accelerate's sgemm runs on the AMX coprocessor, which is shared\n"
           "  per core cluster, so a conv net cannot turn 8 cores into 8x the\n"
           "  evaluations the way the MLP's per-core NEON kernels can.\n");

    /* ---- where the conv time goes ---- */
    printf("\n  where one conv evaluation goes (1 thread, at each size's best batch)\n");
    printf("  %-11s %9s %9s %9s %9s %9s\n",
           "conv", "im2col", "sgemm", "bias+relu", "heads", "total ns");
    for (int i = 0; i < NSIZES; i++) {
        ConvNet net;
        conv_init(&net, SIZES[i].blocks, SIZES[i].filters, POL_QUERY, 1234);
        const int B = g_best_batch[i];
        ConvWs w; ws_init(&w, B, SIZES[i].filters, SIZES[i].blocks, POL_QUERY, 0);
        for (int b = 0; b < B; b++) {
            float tmp[CP_IN];
            planes_build(&g_corpus[b % g_ncorpus].p, 0, tmp);
            for (int c = 0; c < CP_PLANES; c++)
                memcpy(w.in + (size_t)c * B * 64 + b * 64, tmp + c * 64, 64 * sizeof(float));
        }
        ConvSplit sp = {0, 0, 0, 0};
        const double macs = conv_macs(SIZES[i].blocks, SIZES[i].filters, POL_QUERY);
        const int it = (int)(2.0e9 / macs / B) + 3;
        conv_forward_fold(&net, &w, B);
        for (int k = 0; k < it; k++) { conv_forward_timed(&net, &w, B, &sp); sink += w.v[0]; }
        const double per = 1e9 / ((double)it * B);
        char nm[32]; snprintf(nm, sizeof nm, "%d x %d", SIZES[i].blocks, SIZES[i].filters);
        printf("  %-11s %9.0f %9.0f %9.0f %9.0f %9.0f\n", nm,
               sp.im2col * per, sp.gemm * per, sp.point * per, sp.head * per,
               (sp.im2col + sp.gemm + sp.point + sp.head) * per);
        ws_free(&w); conv_free(&net);
    }
    printf("  The sgemm column is the IRREDUCIBLE arithmetic: no implementation of\n"
           "  this architecture can beat it.  Compare it alone against the MLP's\n"
           "  %.0f ns/eval before concluding that a better kernel would rescue this.\n",
           1e9 / m1);

    /* ---- what LayerNorm would cost if we used it anyway ---- */
    printf("\n  normaliser: folded BatchNorm (free) vs LayerNorm (not foldable), 1 thread\n");
    printf("  %-11s %12s %12s %9s\n", "conv", "folded BN", "LayerNorm", "LN cost");
    for (int i = 0; i < NSIZES; i++) {
        ConvNet net;
        conv_init(&net, SIZES[i].blocks, SIZES[i].filters, POL_QUERY, 1234);
        const double macs = conv_macs(SIZES[i].blocks, SIZES[i].filters, POL_QUERY);
        const int B = g_best_batch[i];
        const int it = (int)(1.5e9 / macs / B) + 3;
        double rl = 0.0;
        for (int rep = 0; rep < 2; rep++) {
            const double v = bench_conv_n(&net, 1, B, it, 0, 1, &sink);
            if (v > rl) rl = v;
        }
        char nm[32]; snprintf(nm, sizeof nm, "%d x %d", SIZES[i].blocks, SIZES[i].filters);
        printf("  %-11s %12.0f %12.0f %8.2fx\n", nm, g_best_rate[i], rl, g_best_rate[i] / rl);
        conv_free(&net);
    }

    /* ---- the AlphaZero plane head, costed ---- */
    {
        ConvNet a, b;
        conv_init(&a, 4, 32, POL_QUERY, 5);
        conv_init(&b, 4, 32, POL_AZPLANE, 5);
        double ra = 0, rb = 0;
        for (int r = 0; r < 2; r++) {
            double v = bench_conv(&a, 1, 16, 2500, 0, &sink); if (v > ra) ra = v;
            v = bench_conv(&b, 1, 16, 2500, 0, &sink);        if (v > rb) rb = v;
        }
        printf("\n  policy head, conv 4x32 at batch 16, 1 thread:\n"
               "    32-d factored query head (used above) %8.0f evals/sec\n"
               "    AlphaZero 8x8x73 = 4672-way plane     %8.0f evals/sec   (%.2fx slower)\n",
               ra, rb, ra / rb);
        conv_free(&a); conv_free(&b);
    }
    if (sink == 12345.678) printf("(impossible)\n");
}

static void mode_bwd(void)
{
    double sink = 0.0;
    printf("\n=== FORWARD + BACKWARD (the learning cost) ===\n");
    printf("  The backward pass is implemented and finite-difference checked (see\n"
           "  `check`).  It uses LayerNorm, because a backward pass needs a\n"
           "  normaliser that is actually there -- so the honest backward/forward\n"
           "  ratio compares it against the LAYERNORM forward, not the folded one.\n");
    printf("\n  %-11s %12s %12s %12s %9s\n",
           "conv", "fwd folded", "fwd LN", "fwd+bwd LN", "bwd/fwd");
    for (int i = 0; i < NSIZES; i++) {
        ConvNet net;
        conv_init(&net, SIZES[i].blocks, SIZES[i].filters, POL_QUERY, 1234);
        const double macs = conv_macs(SIZES[i].blocks, SIZES[i].filters, POL_QUERY);
        const int B = 32;
        const int it = (int)(1.0e9 / macs / B) + 3;
        double ff = 0, fl = 0, fb = 0;
        for (int r = 0; r < 3; r++) {
            double v = bench_conv_n(&net, 1, B, it, 0, 0, &sink); if (v > ff) ff = v;
            v = bench_conv_n(&net, 1, B, it, 0, 1, &sink);        if (v > fl) fl = v;
            v = bench_conv_n(&net, 1, B, it, 1, 0, &sink);        if (v > fb) fb = v;
        }
        char nm[32]; snprintf(nm, sizeof nm, "%d x %d", SIZES[i].blocks, SIZES[i].filters);
        printf("  %-11s %12.0f %12.0f %12.0f %8.2fx\n", nm, ff, fl, fb, fl / fb);
        conv_free(&net);
    }
    printf("\n  bwd/fwd is ~4x at every size -- a little above the textbook 2-3x,\n"
           "  because col2im is a scalar scatter while everything else is sgemm.\n"
           "  Nothing here is pathological: the conv net's learning cost is simply\n"
           "  its forward cost multiplied again, and the forward cost is the problem.\n");
    printf("  A training step also pays the optimiser update over EVERY parameter.\n"
           "  The MLP's dominant W0 (788x128) is touched on only the ~%d active rows\n"
           "  per position, so its sparse first layer is cheap to train as well as\n"
           "  cheap to evaluate; every conv weight is dense and is updated in full.\n", 35);
    if (sink == 12345.678) printf("(impossible)\n");
}

/* A bare gather of `k` rows of NF_ACC floats out of an `rows`-row table, which
 * is exactly what adding sparse features does to the MLP's first layer.  The
 * table is sized to the ENLARGED input so the measurement carries the cache
 * cost of a bigger W0, which reusing the existing 788-row W0 would hide. */
static double gather_ns_w(int rows, int k, int width, int reps, double *sink)
{
    float *W = (float *)malloc(sizeof(float) * (size_t)rows * (size_t)width);
    uint16_t *idx = (uint16_t *)malloc(sizeof(uint16_t) * (size_t)k * 512);
    float acc[NF_ACC];
    if (!W || !idx || width > NF_ACC) { fprintf(stderr, "OOM gather\n"); exit(1); }
    g_rs = 5150;
    for (size_t i = 0; i < (size_t)rows * (size_t)width; i++) W[i] = rnd_gauss();
    for (int i = 0; i < k * 512; i++) idx[i] = (uint16_t)(rnd_u64() % (uint64_t)rows);
    double best = 1e30;
    for (int r = 0; r < reps; r++) {
        const double t0 = thread_cpu_sec();
        const int IT = 40000;
        for (int it = 0; it < IT; it++) {
            const uint16_t *f = idx + (size_t)(it % 512) * k;
            for (int j = 0; j < width; j++) acc[j] = 0.0f;
            for (int a = 0; a < k; a++) {
                const float *row = W + (size_t)f[a] * (size_t)width;
                for (int j = 0; j < width; j++) acc[j] += row[j];
            }
            /* EVERY lane must be consumed.  Sinking only acc[0] lets the
             * compiler delete 127 of the 128 columns, which made the first cut
             * of this benchmark report a gather 20x faster than the one inside
             * nn_eval -- a gather that was not being performed. */
            float s = 0.0f;
            for (int j = 0; j < width; j++) s += acc[j];
            *sink += s;
        }
        const double ns = 1e9 * (thread_cpu_sec() - t0) / IT;
        if (ns < best) best = ns;
    }
    free(W); free(idx);
    return best;
}

static double gather_ns(int rows, int k, int reps, double *sink)
{
    return gather_ns_w(rows, k, NF_ACC, reps, sink);
}

static void mode_feat(void)
{
    double sink = 0.0;
    uint16_t idx[8192];
    printf("\n=== EXTRA SPARSE FEATURES: THE CHEAP MIDDLE GROUND ===\n");
    printf("  THE CONTRACT TEST, applied one feature at a time: a feature is allowed\n"
           "  when a player who knows ONLY the rules, and holds no opinion whatever\n"
           "  about which positions are good, could compute it.  Attack maps and\n"
           "  move counts pass; anything that has to RANK pieces does not.\n");

    const int IT = 200000;

    /* ---- extraction cost, feature by feature ---- */
    printf("\n  %-40s %9s %8s %8s\n", "feature", "ns/pos", "slots", "active");
    double nnfeat_ns = 0.0;
    {
        const double t0 = thread_cpu_sec();
        for (int i = 0; i < IT; i++) sink += nn_features(&g_corpus[i % g_ncorpus].p, idx);
        nnfeat_ns = 1e9 * (thread_cpu_sec() - t0) / IT;
        printf("  %-40s %9.1f %8d %8d\n", "nn_features (existing, for scale)",
               nnfeat_ns, NF_INPUT, 35);
    }
    double att_ns = 0.0, att_act = 0.0;
    {
        const double t0 = thread_cpu_sec();
        long nact = 0;
        for (int i = 0; i < IT; i++) {
            const int n = feat_attack_maps(&g_corpus[i % g_ncorpus].p, idx, 788);
            nact += n; sink += n;
        }
        att_ns = 1e9 * (thread_cpu_sec() - t0) / IT;
        att_act = (double)nact / IT;
        printf("  %-40s %9.1f %8d %8.1f\n", "attack maps, BOTH sides (ALLOWED)", att_ns, 128, att_act);
    }
    {
        const double t0 = thread_cpu_sec();
        for (int i = 0; i < IT / 4; i++) sink += feat_mobility(&g_corpus[i % g_ncorpus].p, idx, 916);
        printf("  %-40s %9.1f %8d %8d\n", "per-type mobility counts (ALLOWED)",
               1e9 * (thread_cpu_sec() - t0) / (IT / 4), 48, 6);
        printf("  %-40s %9s %8s %8s\n",
               "  (a whole gen_legal; inside MCTS the move", "", "", "");
        printf("  %-40s %9s %8s %8s\n",
               "   list already exists, so ~0 marginal)", "", "", "");
    }
    double lm_ns = 0.0, hist_ns[4] = {0, 0, 0, 0}, hist_act[4] = {0, 0, 0, 0};
    {
        Game g; memset(&g, 0, sizeof g);
        pos_startpos(&g.pos); g.hist[0] = g.pos.key; g.hist_len = 1;
        Move mv[MAX_MOVES];
        g_rs = 99;
        for (int i = 0; i < 60; i++) {
            const int n = gen_legal(&g.pos, mv);
            if (!n) break;
            const Move m = mv[rnd_u64() % (uint64_t)n];
            g.moves[g.ply] = m;
            make_move(&g.pos, m, &g.undos[g.ply]);
            g.ply++; g.hist[g.hist_len++] = g.pos.key;
        }
        double t0 = thread_cpu_sec();
        for (int i = 0; i < IT; i++) sink += feat_lastmove_rep(&g, idx, 964);
        lm_ns = 1e9 * (thread_cpu_sec() - t0) / IT;
        printf("  %-40s %9.1f %8d %8d\n", "last move + repetition count (ALLOWED)", lm_ns, 132, 3);
        const int NH[4] = {1, 2, 4, 8};
        for (int k = 0; k < 4; k++) {
            t0 = thread_cpu_sec();
            long nact = 0;
            for (int i = 0; i < IT / 8; i++) { const int n = feat_history(&g, NH[k], idx, 1096);
                                               nact += n; sink += n; }
            hist_ns[k] = 1e9 * (thread_cpu_sec() - t0) / (IT / 8);
            hist_act[k] = (double)nact / (IT / 8);
            char nm[64]; snprintf(nm, sizeof nm, "%d history positions (ALLOWED)", NH[k]);
            printf("  %-40s %9.1f %8d %8.1f\n", nm, hist_ns[k], NH[k] * 768, hist_act[k]);
        }
    }

    printf("\n  DROPPED ON THE CONTRACT, not on the cost:\n"
           "    \"squares attacked by a LOWER-VALUED piece\" -- lower-valued needs a\n"
           "        piece-value ordering, the first item on the forbidden list in\n"
           "        docs/FROM_SCRATCH.md.  Dropped even though it is probably the\n"
           "        most useful feature here: the line is drawn by the rule, not by\n"
           "        the payoff.\n"
           "    static exchange evaluation, \"hanging piece\", \"defended by a pawn\"\n"
           "        -- same reason, every one of them ranks pieces.\n"
           "    \"attacked by them and NOT by us\" -- needs no piece values and is\n"
           "        therefore contract-clean, but it is the conclusion we want the\n"
           "        network to reach, hand-computed.  Given both attack maps the\n"
           "        first layer can form that conjunction itself.  Supply the facts,\n"
           "        not the inference.\n");

    /* ---- the cost model: gather over a table of the TRUE enlarged size ---- */
    printf("\n  What each bundle costs.  The MLP's first layer is a gather of `active`\n"
           "  rows out of the input table, so both the ACTIVE COUNT and the TABLE\n"
           "  SIZE matter -- a bigger W0 is a colder W0.  Measured directly:\n");
    printf("\n  %-34s %6s %7s %9s %9s %9s %9s\n",
           "bundle", "slots", "active", "gather", "extract", "evals/s", "sims/move");

    /* everything in an eval that is NOT the first-layer gather, measured as
     * (whole eval at 35 active) - (gather of 35 rows from 788) */
    double base_eval_ns = 0.0;
    {
        Fwd fw;
        const double t0 = thread_cpu_sec();
        for (int i = 0; i < IT; i++) {
            const CPos *c = &g_corpus[i % g_ncorpus];
            nn_eval(g_trunk, &g_heads[(i / HEAD_PERIOD) % NHEADS], c->fidx, c->nf, &fw);
            sink += fw.v;
        }
        base_eval_ns = 1e9 * (thread_cpu_sec() - t0) / IT;
    }
    const double base_gather = gather_ns(NF_INPUT, 35, 3, &sink);
    const double rest_ns = base_eval_ns - base_gather;
    const double base_total = base_eval_ns + nnfeat_ns;
    printf("  (one eval today = %.0f ns, of which the gather is %.0f ns and the rest\n"
           "   -- W1, the heads, the norms -- is %.0f ns; nn_features adds %.0f ns)\n",
           base_eval_ns, base_gather, rest_ns, nnfeat_ns);

    struct { const char *name; int slots; double active, extract; } BUN[] = {
        { "A  baseline (today)",                     788,  35.0, 0.0 },
        { "B  A + repetition + last move",           920,  38.0, 0.0 },
        { "C  B + THEIR attack map only",            984,  38.0 + 0.5, 0.0 },
        { "D  B + BOTH attack maps",                1048,  38.0, 0.0 },
        { "E  D + mobility counts",                 1096,  44.0, 0.0 },
        { "F  E + 8 history positions",             7240,  44.0, 0.0 },
    };
    BUN[1].extract = lm_ns;
    BUN[2].active  = 38.0 + att_act * 0.5;
    BUN[2].extract = lm_ns + att_ns * 0.5;
    BUN[3].active  = 38.0 + att_act;
    BUN[3].extract = lm_ns + att_ns;
    BUN[4].active  = 44.0 + att_act;
    BUN[4].extract = lm_ns + att_ns;   /* mobility is free inside MCTS         */
    BUN[5].active  = 44.0 + att_act + hist_act[3];
    BUN[5].extract = lm_ns + att_ns + hist_ns[3];

    for (size_t i = 0; i < sizeof BUN / sizeof BUN[0]; i++) {
        const double g = gather_ns(BUN[i].slots, (int)(BUN[i].active + 0.5), 3, &sink);
        const double tot = g + rest_ns + nnfeat_ns + BUN[i].extract;
        printf("  %-34s %6d %7.1f %8.0fns %8.0fns %9.0f %9.1f\n",
               BUN[i].name, BUN[i].slots, BUN[i].active, g, BUN[i].extract,
               1e9 / tot, 160.0 * base_total / tot);
    }
    /* The gather is linear in the accumulator WIDTH as well as the row count,
     * so attack features need not be paid for at 128 wide.  src/net.h already
     * has the pattern: the value head owns a separate 32-wide sparse trunk
     * (W0v).  A third accumulator, 32 wide, fed by the attack maps, costs a
     * quarter of what those rows cost in the main one. */
    printf("\n  Same bundles, but with the attack map feeding a SEPARATE 32-wide\n"
           "  accumulator instead of the 128-wide one (src/net.h already does\n"
           "  exactly this for the value head's W0v):\n");
    printf("  %-34s %6s %7s %9s %9s %9s %9s\n",
           "bundle", "slots", "active", "gather", "extract", "evals/s", "sims/move");
    {
        struct { const char *name; int mslots; double mactive; int sslots; double sactive;
                 double extract; } SB[] = {
            { "C' their attack map, 32-wide side",  920, 38.0, 64,  0.0, 0.0 },
            { "D' both attack maps, 32-wide side",  920, 38.0, 128, 0.0, 0.0 },
        };
        SB[0].sactive = att_act * 0.5; SB[0].extract = lm_ns + att_ns * 0.5;
        SB[1].sactive = att_act;       SB[1].extract = lm_ns + att_ns;
        for (size_t i = 0; i < sizeof SB / sizeof SB[0]; i++) {
            const double gm = gather_ns_w(SB[i].mslots, (int)(SB[i].mactive + 0.5), NF_ACC, 3, &sink);
            const double gs = gather_ns_w(SB[i].sslots, (int)(SB[i].sactive + 0.5), 32, 3, &sink);
            /* the 32-wide side accumulator also needs its own LayerNorm and a
             * 32x32 hidden layer, which src/net.h measures at 30 ns + 25 ns */
            const double side_rest = 55.0;
            const double tot = gm + gs + side_rest + rest_ns + nnfeat_ns + SB[i].extract;
            printf("  %-34s %6d %7.1f %8.0fns %8.0fns %9.0f %9.1f\n",
                   SB[i].name, SB[i].mslots + SB[i].sslots, SB[i].mactive + SB[i].sactive,
                   gm + gs, SB[i].extract, 1e9 / tot, 160.0 * base_total / tot);
        }
    }

    printf("\n  'sims/move' is again the wall-clock-neutral number: how many MCTS\n"
           "  simulations per move this bundle leaves us at today's throughput,\n"
           "  against a useful floor of 96.\n");
    printf("  Mobility is charged at zero extraction because MCTS has already\n"
           "  generated the legal move list at every node it expands; it only pays\n"
           "  the 6 extra gathered rows.\n");
    if (sink == 12345.678) printf("(impossible)\n");
}

/* The arithmetic that actually decides the question. */
/* The arithmetic that actually decides the question. */
static void mode_sims(void)
{
    double sink = 0.0;
    printf("\n=== SIMULATIONS WE COULD AFFORD ===\n");
    const int CUR_SIMS = 160;
    const int FLOOR    = 96;
    printf("  Our self-play runs %d MCTS simulations per move.  A network that is\n"
           "  K times slower per evaluation buys %d/K simulations per move at the\n"
           "  SAME wall-clock -- the same games/second, the same generations/hour.\n", CUR_SIMS, CUR_SIMS);
    printf("  docs/ALGORITHM.md measured the floor directly, at OUR draw penalty\n"
           "  (-0.6), by sweeping the simulation count:\n"
           "      48 sims -> %.0f%% repetition draws, %.1f captures/game\n"
           "      96 sims -> %.0f%% repetition draws, %.1f captures/game\n"
           "     160 sims -> %.0f%% repetition draws, %.1f captures/game\n"
           "  Tripling the draw PENALTY at 48 sims barely moved repetition; tripling\n"
           "  the SIMS nearly eliminated it.  Shuffling is the search failing to find\n"
           "  a plan, and no network fixes that.  The floor is about %d sims.\n",
           51.6, 8.5, 17.7, 14.4, 5.3, 17.9, FLOOR);

    printf("\n  %-11s %6s %11s %10s %11s %14s\n",
           "conv", "batch", "conv 1thr", "slowdown", "sims/move", "verdict");
    for (int i = 0; i < NSIZES; i++) {
        double slow = g_slow1[i], cr = g_conv1[i];
        if (slow <= 0.0) {
            ConvNet net;
            conv_init(&net, SIZES[i].blocks, SIZES[i].filters, POL_QUERY, 1234);
            const int B = g_best_batch[i] ? g_best_batch[i] : 8;
            double m = 0, c = 0;
            paired_rate(&net, 1, B, 6, &m, &c, &sink);
            slow = m / c; cr = c;
            g_slow1[i] = slow; g_conv1[i] = cr;
            g_best_batch[i] = B;
            conv_free(&net);
        }
        const double sims = (double)CUR_SIMS / slow;
        char nm[32]; snprintf(nm, sizeof nm, "%d x %d", SIZES[i].blocks, SIZES[i].filters);
        printf("  %-11s %6d %11.0f %9.0fx %11.1f %14s\n",
               nm, g_best_batch[i], cr, slow, sims,
               sims >= FLOOR ? "above floor" : (sims >= 8.0 ? "BELOW FLOOR" : "far BELOW FLOOR"));
    }
    printf("\n  To put the smallest one in context: a 2x16 conv net at %.1f sims/move\n"
           "  would be searching less deeply than the 48-sim configuration that\n"
           "  produced a 60%%-repetition shuffling population.\n",
           g_slow1[0] > 0 ? 160.0 / g_slow1[0] : 0.0);
    printf("\n  The question in reverse: to keep 160 sims/move AND pay for a conv net,\n"
           "  the run would have to get longer by the slowdown factor.  The 1900-\n"
           "  generation run took 69 minutes, so:\n");
    for (int i = 0; i < NSIZES; i++) {
        if (g_slow1[i] <= 0.0) continue;
        char nm[32]; snprintf(nm, sizeof nm, "%d x %d", SIZES[i].blocks, SIZES[i].filters);
        printf("    %-11s the same 729,600 games at 160 sims would take %.1f hours\n",
               nm, 69.0 / 60.0 * g_slow1[i]);
    }
    if (sink == 12345.678) printf("(impossible)\n");
}

int main(int argc, char **argv)
{
    const char *mode = argc > 1 ? argv[1] : "all";

    chess_init();
    corpus_build();
    g_trunk = (Trunk *)malloc(sizeof(Trunk));
    g_heads = (Head *)malloc(sizeof(Head) * NHEADS);
    if (!g_trunk || !g_heads) { fprintf(stderr, "OOM\n"); return 1; }
    nn_init(g_trunk, &g_heads[0], 12345ULL);
    for (int i = 1; i < NHEADS; i++) nn_init(NULL, &g_heads[i], 12345ULL + (uint64_t)i * 977ULL);

    printf("conv_proto -- is a convolutional network affordable here?\n");
    printf("  corpus %d positions;  sgemm: %s;  NEON: %s;  MLP backend: %s\n",
           g_ncorpus, HAVE_ACCELERATE ? "Accelerate" : "hand-written",
           HAVE_NEON ? "yes" : "no", nn_backend());

    const int all = !strcmp(mode, "all");
    if (all || !strcmp(mode, "params")) mode_params();
    if (all || !strcmp(mode, "check"))  mode_check();
    if (all || !strcmp(mode, "fwd"))    mode_fwd();
    if (all || !strcmp(mode, "bwd"))    mode_bwd();
    if (all || !strcmp(mode, "feat"))   mode_feat();
    if (all || !strcmp(mode, "sims"))   mode_sims();

    free(g_trunk); free(g_heads);
    return 0;
}
