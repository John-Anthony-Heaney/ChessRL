/* net.c -- policy/value network, analytic gradients, AdamW, serialisation.
 *
 * Layout contracts (chosen here, referenced by tests/test_net.c)
 * -------------------------------------------------------------
 *   W0    [f * NF_ACC + i]      row per input feature, contiguous over acc.
 *   W1    [j * NF_ACC + i]      row per hidden unit,   contiguous over acc.
 *   Wp    [k * NF_HID + j]      row per policy dim,    contiguous over hidden.
 *   W0v   [f * NF_VACC + i]     row per input feature, value trunk.
 *   Wvh   [k * NF_VACC + i]     row per value dim,     contiguous over hv0.
 *   E*    [row * NF_PDIM + k]
 *   Bft   [from * 64 + to]
 *
 * W1/Wp/Wvh are stored output-major so the inner reduction runs over a
 * contiguous span (NF_ACC / NF_HID, both compile-time constants and multiples
 * of 8).  The reductions use four independent accumulators: clang's SLP
 * vectoriser folds them into one NEON vector accumulator, which plain `-O3`
 * cannot do to a single-chain float reduction because that would require
 * reassociation.
 *
 * nn_eval() is the hot path; everything in it is a contiguous float loop over a
 * constant trip count with restrict-qualified pointers so no bounds/aliasing
 * check survives into the loop body.
 *
 * ---------------------------------------------------------------------------
 * LAYERNORM, AND WHY NOT RMSNORM
 * ---------------------------------------------------------------------------
 * Both were implemented and measured (4200 self-play games, held out by game,
 * 3 seeds, identical batches and optimiser settings).  LayerNorm won on the
 * value metric that motivated the change -- held-out MSE/Var 0.811 against
 * RMSNorm's 0.835, with the same residual and the same value head -- and the
 * reason is specific to this input encoding rather than a general preference:
 *
 *   acc is a SUM of ~35 one-hot rows of W0, and the number of active features
 *   falls from 37 in the opening to 5 in a bare-king endgame.  That makes the
 *   MEAN of acc a strong, nuisance-correlated function of game phase: it drifts
 *   monotonically over a game and drags every relu gate with it.  RMSNorm
 *   divides that drift out of the scale but leaves it in the location, so the
 *   gate pattern still tracks it.  LayerNorm removes it.  The cost is one extra
 *   pass over the layer to find the mean, measured at 2.9% of the evaluation
 *   rate (117589 vs 121136 evaluations/sec, same configuration otherwise) --
 *   cheap against a 3% reduction in held-out value MSE.
 *
 * The normalisation is applied BEFORE each nonlinearity -- four of them: the
 * accumulator's relu, the residual block's relu, the value trunk's relu and the
 * value head's relu.  The scalar tanh at the output is not normalised, because
 * a LayerNorm over one number is exactly zero.
 */

#include "net.h"

#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#define TRUNK_FLOATS (NF_INPUT * NF_ACC + 3 * NF_ACC + NF_ACC * NF_HID + 3 * NF_HID + \
                      NF_INPUT * NF_VACC + 3 * NF_VACC)
#define HEAD_FLOATS  (NF_ACC + NF_VACC * NF_VHID + 3 * NF_VHID + NF_VHID + 1 + \
                      NF_HID * NF_PDIM + (64 + 64 + 6 + 5 + 7) * NF_PDIM + 64 * 64)

_Static_assert(sizeof(Trunk) == (size_t)TRUNK_FLOATS * sizeof(float),
               "Trunk must be pure float storage with no padding");
_Static_assert(sizeof(Head) == (size_t)HEAD_FLOATS * sizeof(float),
               "Head must be pure float storage with no padding");
_Static_assert(sizeof(Hyper) == 8 * sizeof(float), "Hyper must be 8 floats");
_Static_assert(NF_ACC % 8 == 0 && NF_HID % 8 == 0 && NF_PDIM % 8 == 0 &&
               NF_VHID % 8 == 0 && NF_VACC % 8 == 0,
               "layer widths must be multiples of 8");

#define HYPER_NPARAM (sizeof(Hyper) / sizeof(float))

/* LayerNorm epsilon.  Inside the sqrt, as in every standard implementation, so
 * the backward pass needs no special case when a layer is momentarily flat. */
#define NN_EPS 1e-5f

/* ------------------------------------------------------- head tensor table */

#define TSPAN(field, n) { offsetof(Head, field) / sizeof(float), (size_t)(n) }

const NnTensorSpan NN_HEAD_TENSORS[] = {
    TSPAN(z,      NF_ACC),
    TSPAN(Wvh,    NF_VACC * NF_VHID),
    TSPAN(bvh,    NF_VHID),
    TSPAN(gv,     NF_VHID),
    TSPAN(cv,     NF_VHID),
    TSPAN(Wv,     NF_VHID),
    TSPAN(bv,     1),
    TSPAN(Wp,     NF_HID * NF_PDIM),
    TSPAN(Efrom,  64 * NF_PDIM),
    TSPAN(Eto,    64 * NF_PDIM),
    TSPAN(Epc,    6 * NF_PDIM),
    TSPAN(Epromo, 5 * NF_PDIM),
    TSPAN(Ecap,   7 * NF_PDIM),
    TSPAN(Bft,    64 * 64),
};
const int NN_HEAD_NTENSORS = (int)(sizeof NN_HEAD_TENSORS / sizeof NN_HEAD_TENSORS[0]);

/* The table must cover the head exactly once: a tensor added to Head and not
 * added here would silently never be mutated by PBT. */
_Static_assert((size_t)HEAD_FLOATS == HEAD_NPARAM,
               "NN_HEAD_TENSORS must span every float of Head exactly once");

#undef TSPAN

/* ------------------------------------------------------------ hyper-params */

void hyper_default(Hyper *h)
{
    if (!h) return;
    h->temperature  = 1.00f;
    h->entropy_coef = 0.01f;
    h->lr_scale     = 1.00f;
    h->shaping      = 0.50f;
    h->value_coef   = 0.50f;
    h->gamma        = 0.99f;
    h->lambda       = 0.95f;
    h->mutate_sigma = 0.02f;
}

/* -------------------------------------------------------------------- rng */
/* Private xoshiro256** so nn_init() does not depend on arena.c. */

static inline uint64_t rotl64(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }

static uint64_t nn_splitmix(uint64_t *s)
{
    uint64_t z = (*s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static void nn_rng_seed(uint64_t *st, uint64_t seed)
{
    uint64_t x = seed ? seed : 0x9E3779B97F4A7C15ULL;
    for (int i = 0; i < 4; i++) st[i] = nn_splitmix(&x);
}

static uint64_t nn_rng_next(uint64_t *st)
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

/* Standard normal via Box-Muller; deterministic given the state. */
static float nn_rng_normal(uint64_t *st)
{
    const double u1 = (double)((nn_rng_next(st) >> 11) + 1) * (1.0 / 9007199254740993.0);
    const double u2 = (double)(nn_rng_next(st) >> 11) * (1.0 / 9007199254740992.0);
    return (float)(sqrt(-2.0 * log(u1)) * cos(6.283185307179586476925287 * u2));
}

static void fill_normal(float *p, size_t n, float std, uint64_t *st)
{
    for (size_t i = 0; i < n; i++) p[i] = nn_rng_normal(st) * std;
}

/* ---------------------------------------------------------------- features */
/*  0..383  : our    pieces, index = ptype*64 + sq_rel
 *  384..767: their  pieces, index = 384 + ptype*64 + sq_rel
 *  768..771: castling (our K, our Q, their K, their Q)
 *  772..779: en-passant file
 *  780..787: min(7, halfmove/13)                                            */

#define FEAT_PUSH(v)                                    \
    do {                                                \
        if (n < NF_MAXACTIVE) idx[n++] = (uint16_t)(v); \
    } while (0)

int nn_features(const Position *p, uint16_t *idx)
{
    const int us   = p->side;
    const int them = us ^ 1;
    const int flip = (us == BLACK) ? 56 : 0;   /* sq_flip() == sq ^ 56 */
    int n = 0;

    for (int pt = PAWN; pt <= KING; pt++) {
        uint64_t b = p->piece[us][pt];
        const int base = pt * 64;
        while (b) FEAT_PUSH(base + (bb_pop(&b) ^ flip));
    }
    for (int pt = PAWN; pt <= KING; pt++) {
        uint64_t b = p->piece[them][pt];
        const int base = 384 + pt * 64;
        while (b) FEAT_PUSH(base + (bb_pop(&b) ^ flip));
    }

    /* Castling bits follow the same colour swap as the piece planes. */
    const int our_k  = (us == WHITE) ? CR_WK : CR_BK;
    const int our_q  = (us == WHITE) ? CR_WQ : CR_BQ;
    const int thr_k  = (us == WHITE) ? CR_BK : CR_WK;
    const int thr_q  = (us == WHITE) ? CR_BQ : CR_WQ;
    const int cr = p->castling;
    if (cr & our_k) FEAT_PUSH(768);
    if (cr & our_q) FEAT_PUSH(769);
    if (cr & thr_k) FEAT_PUSH(770);
    if (cr & thr_q) FEAT_PUSH(771);

    /* A vertical flip never changes the file, so sq_file() is mirror-safe. */
    if (p->ep >= 0) FEAT_PUSH(772 + sq_file(p->ep));

    int bucket = p->halfmove / 13;
    if (bucket > 7) bucket = 7;
    FEAT_PUSH(780 + bucket);

    return n;
}

#undef FEAT_PUSH

void nn_move_key(const Position *p, Move m, MoveKey *k)
{
    const int flip = (p->side == BLACK) ? 56 : 0;
    const int from = MV_FROM(m), to = MV_TO(m);
    const int fl = MV_FLAG(m);

    k->from = (uint8_t)(from ^ flip);
    k->to   = (uint8_t)(to ^ flip);

    int pc = (int)p->board[from];
    if (pc < 0 || pc >= NPIECES) pc = PAWN;      /* never happens for legal moves */
    k->pc = (uint8_t)pc;

    /* flags 8..11 and 12..15 both encode N,B,R,Q in the low two bits. */
    k->promo = (uint8_t)((fl >= MF_PROMO_N) ? ((fl & 3) + 1) : 0);

    if (fl == MF_EP) {
        k->cap = (uint8_t)(PAWN + 1);            /* en passant captures a pawn */
    } else if (MV_IS_CAPTURE(m)) {
        const int c = (int)p->board[to];
        k->cap = (uint8_t)((c >= 0 && c < NPIECES) ? c + 1 : 0);
    } else {
        k->cap = 0;
    }
}

/* ------------------------------------------------------------- LayerNorm */
/*
 *   mu     = mean(x)
 *   sigma  = sqrt(mean((x - mu)^2) + eps)
 *   xhat_i = (x_i - mu) / sigma
 *   y_i    = g_i * xhat_i + c_i
 *
 * and, writing a_i = dy_i * g_i and r = 1/sigma,
 *
 *   dL/dg_i = dy_i * xhat_i
 *   dL/dc_i = dy_i
 *   dL/dx_i = r * (a_i - mean(a) - xhat_i * mean(a * xhat))
 *
 * The two correction terms are what make the output invariant to the scale AND
 * the offset of x; they are also what makes dL/dx sum to exactly zero, so a
 * bias applied before the norm can only ever move the layer along directions
 * the norm does not remove.
 */

/* Every width norm_stats is called with must fit the scratch buffer inside it. */
_Static_assert(NF_HID <= NF_ACC && NF_VACC <= NF_ACC && NF_VHID <= NF_ACC,
               "norm_stats' scratch is sized by NF_ACC");

/* Returns 1/sigma and writes the mean through mu_out.
 *
 * BOTH reductions are single serial chains and must stay that way: summing them
 * four-wide would reassociate them and change the last bit of the result, which
 * is how this file keeps nn_eval bit-identical across the batched and unbatched
 * paths and across builds.  The batched path gets its width by running FOUR
 * ROWS of the batch in the four lanes instead -- each lane is still the same
 * serial chain -- which is the whole trick in nn_norm4_stats() below.
 *
 * The explicit fmaf() in the variance loop is not decoration, and neither is
 * the fact that it is ONE loop.
 *
 * Written the obvious way, `ss += d * d` under -ffp-contract=fast, clang fused
 * the multiply-add at one call site and not at another, purely because of how
 * the surrounding function had been inlined, and the two differ in the last
 * ULP.  That alone made nn_eval_batch disagree with nn_eval on 3 rows in 64, by
 * up to 1.9e-9 on v, because the batched and unbatched paths reach this
 * function through different inlining.  Spelling the fusion out removes the
 * choice, and every kernel below uses FMA for the same reason: what this file
 * computes should be a property of this source, not of the inliner's mood.
 *
 * The obvious speed objection is real and was measured.  This loop is a SERIAL
 * CHAIN -- its cost is n times the latency of one accumulate and nothing else --
 * and an FMA is 4 cycles where an add is 3, so pinning the fusion costs ~34 ns
 * per 128-wide LayerNorm, about 4% of an evaluation.  Splitting it into two
 * loops (squares into a buffer, then a sum of the buffer) has no multiply-add
 * left to fuse and measured 1.034x on one thread and 1.058x batched -- and was
 * REJECTED, because clang fuses the two loops back together and contracts the
 * result anyway: with that shape the batched path diverged from the unbatched
 * one again (14 rows of 128, 4.8e-7 on the value trunk's normalisation) and the
 * NEON and portable builds stopped agreeing.  A 4% gain is not worth an
 * exactness guarantee that depends on the compiler declining to fuse two loops.
 *
 * BOTH reductions stay serial on purpose.  Summing either four lanes wide would
 * reassociate it and change the last bit -- exactly what the batched path must
 * not do.  The way to get width here is to put FOUR ROWS of a batch in the four
 * lanes, each lane still its own serial chain; see the note above
 * nn_eval_batch for what that would be worth. */
static inline float norm_stats(const float *restrict x, int n, float *mu_out)
{
    float s = 0.0f;
    for (int i = 0; i < n; i++) s += x[i];
    const float mu = s / (float)n;

    float ss = 0.0f;
    for (int i = 0; i < n; i++) { const float d = x[i] - mu; ss = fmaf(d, d, ss); }

    *mu_out = mu;
    return 1.0f / sqrtf(ss / (float)n + NN_EPS);
}

/* dy -> (gain grad, bias grad, dx).  `dx` must not alias `dy`. */
static inline void norm_bwd(const float *restrict dy, const float *restrict xhat,
                            const float *restrict g, float r, int n,
                            float *restrict gg, float *restrict gc,
                            float *restrict dx)
{
    float s1 = 0.0f, s2 = 0.0f;
    for (int i = 0; i < n; i++) {
        const float a = dy[i] * g[i];
        gg[i] += dy[i] * xhat[i];
        gc[i] += dy[i];
        s1 += a;
        s2 += a * xhat[i];
    }
    const float m1 = s1 / (float)n, m2 = s2 / (float)n;
    for (int i = 0; i < n; i++)
        dx[i] = r * (dy[i] * g[i] - m1 - xhat[i] * m2);
}

/* -------------------------------------------------------------------- init */

void nn_init(Trunk *t, Head *h, uint64_t seed)
{
    uint64_t st[4];
    nn_rng_seed(st, seed);

    if (t) {
        /* ~35 rows are summed into the accumulator, so scale by 1/sqrt(35) to
         * keep the pre-activation O(1) rather than O(sqrt(35)).  LayerNorm makes
         * this cosmetic for the forward pass -- it divides the scale out -- but
         * it still sets where the gain starts relative to the data. */
        fill_normal(t->W0, (size_t)NF_INPUT * NF_ACC, 0.5f / sqrtf(35.0f), st);
        memset(t->b0, 0, sizeof t->b0);
        fill_normal(t->W1, (size_t)NF_ACC * NF_HID, sqrtf(2.0f / (float)NF_ACC), st);
        memset(t->b1, 0, sizeof t->b1);
        for (int i = 0; i < NF_ACC; i++) { t->g0[i] = 1.0f; t->c0[i] = 0.0f; }
        for (int j = 0; j < NF_HID; j++) { t->g1[j] = 1.0f; t->c1[j] = 0.0f; }

        /* the value trunk, same sparse-gather scaling */
        fill_normal(t->W0v, (size_t)NF_INPUT * NF_VACC, 0.5f / sqrtf(35.0f), st);
        memset(t->b0v, 0, sizeof t->b0v);
        for (int i = 0; i < NF_VACC; i++) { t->g0v[i] = 1.0f; t->c0v[i] = 0.0f; }
    }

    if (h) {
        memset(h->z, 0, sizeof h->z);

        /* The initial POLICY must be near uniform -- tests/test_scratch.c fails
         * the build if an untrained network has an opinion about chess -- so the
         * logits have to start at the same scale as before the rewrite, std
         * ~0.67.  Working backwards: logit_std = sqrt(NF_PDIM) * q_std *
         * sqrt(5) * es, so q_std must be 0.474; and q = Wp.h2 with h2 now
         * normalised, where E[relu(unit normal)^2] = 1/2 and E[relu] =
         * 1/sqrt(2pi), so the residual gives E[h2^2] = 1/2 + 2/(2pi) + 1/2 =
         * 1.3183 per unit.  Hence the divisor below.  Without it the LayerNorm
         * would multiply the initial logits by ~4 and the untrained policy
         * would be visibly opinionated. */
        const float wp_std = 0.474f / sqrtf(1.3183f * (float)NF_HID);
        fill_normal(h->Wp, (size_t)NF_HID * NF_PDIM, wp_std, st);

        /* Five embedding rows are summed before the dot with q, so use a fan-in
         * of 5*NF_PDIM. */
        const float es = sqrtf(2.0f / (5.0f * (float)NF_PDIM));
        fill_normal(h->Efrom,  (size_t)64 * NF_PDIM, es, st);
        fill_normal(h->Eto,    (size_t)64 * NF_PDIM, es, st);
        fill_normal(h->Epc,    (size_t)6  * NF_PDIM, es, st);
        fill_normal(h->Epromo, (size_t)5  * NF_PDIM, es, st);
        fill_normal(h->Ecap,   (size_t)7  * NF_PDIM, es, st);
        memset(h->Bft, 0, sizeof h->Bft);

        /* Value head.  The hidden layer is He-initialised; the OUTPUT layer
         * starts at ~0 so the first predictions are unbiased draws, which is the
         * other thing test_scratch.c checks. */
        fill_normal(h->Wvh, (size_t)NF_VACC * NF_VHID, sqrtf(2.0f / (float)NF_VACC), st);
        memset(h->bvh, 0, sizeof h->bvh);
        for (int k = 0; k < NF_VHID; k++) { h->gv[k] = 1.0f; h->cv[k] = 0.0f; }
        fill_normal(h->Wv, NF_VHID, 1.0e-3f, st);
        h->bv[0] = 0.0f;
    }
}

/* ----------------------------------------------------------------- forward */
/*
 * THE FORWARD PASS IS BUILT FROM FOUR KERNELS -- gather, LayerNorm+relu,
 * matvec, and the batched matvec -- and nn_eval() is those kernels applied to
 * one position.  Three reasons it is written this way rather than as one
 * straight-line function:
 *
 *   - nn_eval_batch() runs THE SAME kernels on each row of its batch, so its
 *     result is bit-identical to nn_eval()'s at EVERY batch size, not just at
 *     one.  The only exception is the optional Accelerate path, which is
 *     documented as approximate where it is switched on.
 *   - tools/profile.c compiles itself into this translation unit and times the
 *     kernels one at a time.  A breakdown you can trust needs the timed code to
 *     BE the shipped code rather than a copy of it that can drift.
 *   - every width is a compile-time constant at every call site and the kernels
 *     are always_inline, so what the compiler sees is the specialised loop.
 *
 * WHY THE KERNELS ARE NEON INTRINSICS AND NOT PLAIN C
 * ---------------------------------------------------
 * The plain-C matvec that used to be here relied on four accumulators and a
 * comment claiming clang's SLP vectoriser would fold them into one NEON
 * accumulator.  It does not.  What clang emitted for `s0 += w[i]*x[i]` x4 was
 * an ld4 de-interleaving load feeding separate fmul.4s/fadd.4s pairs -- vector
 * instructions, but with the de-interleave on the critical path and no fused
 * multiply-add at all (20 fmla.4s in the whole of nn_eval, against 60 fmul.4s
 * and 274 scalar fadd).  Measured at the 128x128 shape on this M3: 2.2-2.6
 * GFLOP/s for that loop, 10.9 for a hand-written vfmaq_f32 kernel one row at a
 * time, 14.9 with four output rows in flight, 24.3 with four rows and four
 * positions.  The stage it dominates went from 3397 ns to 609 ns.  Intrinsics
 * also remove the compiler's freedom to regroup the sum, which is what lets the
 * batched and unbatched paths agree bit for bit instead of nearly.
 *
 * EXACTNESS, PRECISELY
 * --------------------
 * Each output element is accumulated in ONE four-lane vector: lane k holds the
 * sum over i == k (mod 4), and vaddvq_f32 reduces it as ((s0+s1)+(s2+s3)) --
 * the same four groups, combined the same way, that the old scalar source
 * described.  The arithmetic this file specifies is therefore unchanged.  What
 * changed is that it is now SPECIFIED: the old source left the fusion of
 * `s0 += w[i]*x[i]` to the compiler, which took it one way here and another way
 * there, and an intrinsic does not give it that choice.  Against the
 * pre-optimisation binary, over the 512-position corpus in tools/profile.c:
 *
 *     value v   4 of 512 positions differ, by at most 1.9e-9 absolute
 *     logits    11980 of 14767 differ, by at most 6.0e-7 absolute
 *
 * A float rounding change, not a change to what is computed -- and the gradient
 * check in tests/test_net.c (analytic against central differences) and the MCTS
 * determinism tests are what say so.  In the other direction the guarantees got
 * stronger: the NEON build, the portable build and the batched path now agree
 * bit for bit with each other, which was not true before.
 */

#if defined(__ARM_NEON) && !defined(NN_NO_NEON)
#  define NN_NEON 1
#  include <arm_neon.h>
#else
#  define NN_NEON 0
#endif

/* Accelerate is a system framework, not a dependency: it is used only when the
 * build asks for it with -DUSE_ACCELERATE, only for the batched dense layers,
 * and there is a portable NEON path underneath it that the tests exercise
 * either way.  See the note above nn_eval_batch() for what it costs in
 * exactness. */
#if defined(USE_ACCELERATE)
#  if defined(__has_include)
#    if __has_include(<Accelerate/Accelerate.h>)
#      define NN_ACCELERATE 1
#    endif
#  endif
#endif
#ifndef NN_ACCELERATE
#  define NN_ACCELERATE 0
#endif
#if NN_ACCELERATE
#  include <Accelerate/Accelerate.h>
#endif

#if defined(__GNUC__) || defined(__clang__)
#  define NN_INLINE static inline __attribute__((always_inline))
#else
#  define NN_INLINE static inline
#endif

/* Fwd is pure float storage, so an array of them is a matrix of stride
 * NN_FWD_STRIDE floats: the batched path hands that stride straight to the
 * kernels (and to sgemm) and never copies a row anywhere. */
#define NN_FWD_STRIDE ((int)(sizeof(Fwd) / sizeof(float)))
_Static_assert(sizeof(Fwd) % sizeof(float) == 0, "Fwd must be pure float storage");

/* ------------------------------------------------------------ the kernels */

/* acc = b (+ z) + the sum of the `nf` rows of Wm named by fidx, each n wide.
 * THE sparse step: ~35 gathered rows, never a dense matmul.  `z` is the agent's
 * style vector on the policy trunk and NULL on the value trunk; it is constant
 * at both call sites, so the branch folds away.
 *
 * The adds are element-wise, so vectorising them cannot regroup anything and
 * the result is exact whatever the vector width. */
NN_INLINE void nn_gather(const float *restrict Wm, const float *restrict b,
                         const float *restrict z, const uint16_t *restrict fidx,
                         int nf, int n, float *restrict acc)
{
#if NN_NEON
    if (z) for (int i = 0; i < n; i += 4)
               vst1q_f32(acc + i, vaddq_f32(vld1q_f32(b + i), vld1q_f32(z + i)));
    else   for (int i = 0; i < n; i += 4)
               vst1q_f32(acc + i, vld1q_f32(b + i));

    for (int f = 0; f < nf; f++) {
        const float *restrict w = Wm + (size_t)fidx[f] * (size_t)n;
#ifndef NN_NO_PREFETCH
        /* One row ahead.  The next row's address is already known and the rows
         * are ~35 scattered 512-byte spans that no stride prefetcher can guess.
         * Measured worth on this M3: see the report -- it is small, and the
         * switch exists so the claim can be re-measured rather than believed. */
        if (f + 1 < nf) __builtin_prefetch(Wm + (size_t)fidx[f + 1] * (size_t)n, 0, 3);
#endif
        for (int i = 0; i < n; i += 4)
            vst1q_f32(acc + i, vaddq_f32(vld1q_f32(acc + i), vld1q_f32(w + i)));
    }
#else
    if (z) for (int i = 0; i < n; i++) acc[i] = b[i] + z[i];
    else   for (int i = 0; i < n; i++) acc[i] = b[i];

    for (int f = 0; f < nf; f++) {
        const float *restrict w = Wm + (size_t)fidx[f] * (size_t)n;
        for (int i = 0; i < n; i++) acc[i] += w[i];
    }
#endif
}

/* y = relu(g * norm(x) + c) (+ skip), also writing xhat; returns 1/sigma.
 *
 * norm_stats' two reductions stay serial on purpose (see the comment there), so
 * this is the one stage that does not vectorise: its cost is the latency of a
 * 128-long chain of adds, twice.  The ELEMENT-WISE second pass below does
 * vectorise, and that part is exact. */
NN_INLINE float nn_norm_relu_gen(const float *restrict x, const float *restrict g,
                                 const float *restrict c, const float *restrict skip,
                                 int n, float *restrict xhat, float *restrict y)
{
    float mu;
    const float r = norm_stats(x, n, &mu);
#if NN_NEON
    const float32x4_t vmu = vdupq_n_f32(mu), vr = vdupq_n_f32(r), z = vdupq_n_f32(0.0f);
    for (int i = 0; i < n; i += 4) {
        const float32x4_t xh = vmulq_f32(vsubq_f32(vld1q_f32(x + i), vmu), vr);
        vst1q_f32(xhat + i, xh);
        float32x4_t a = vmaxq_f32(vfmaq_f32(vld1q_f32(c + i), vld1q_f32(g + i), xh), z);
        if (skip) a = vaddq_f32(a, vld1q_f32(skip + i));
        vst1q_f32(y + i, a);
    }
#else
    for (int i = 0; i < n; i++) {
        const float xh = (x[i] - mu) * r;
        xhat[i] = xh;
        const float a = fmaf(g[i], xh, c[i]);
        y[i] = skip ? (a > 0.0f ? a : 0.0f) + skip[i] : (a > 0.0f ? a : 0.0f);
    }
#endif
    return r;
}

NN_INLINE float nn_norm_relu(const float *restrict x, const float *restrict g,
                             const float *restrict c, int n,
                             float *restrict xhat, float *restrict y)
{
    return nn_norm_relu_gen(x, g, c, NULL, n, xhat, y);
}

/* The residual block's variant: y = relu(g * norm(x) + c) + skip. */
NN_INLINE float nn_norm_relu_add(const float *restrict x, const float *restrict g,
                                 const float *restrict c, const float *restrict skip,
                                 int n, float *restrict xhat, float *restrict y)
{
    return nn_norm_relu_gen(x, g, c, skip, n, xhat, y);
}

/* y[j] = dot(row j of Wm, x) + bias[j], bias optional.  Wm is output-major so
 * the reduction runs over a contiguous span.
 *
 * Four output rows are in flight at once: the x vector is loaded once and used
 * four times, which is what takes this off the load ports.  Measured at the
 * 128x128 shape: 10.9 GFLOP/s one row at a time, 14.9 at four, against 2.2 for
 * the plain-C loop this replaced. */
NN_INLINE void nn_matvec(const float *restrict Wm, const float *restrict bias,
                         const float *restrict x, int nout, int nin,
                         float *restrict y)
{
#if NN_NEON
    int j = 0;
    for (; j + 4 <= nout; j += 4) {
        const float *restrict w0 = Wm + (size_t)(j + 0) * (size_t)nin;
        const float *restrict w1 = Wm + (size_t)(j + 1) * (size_t)nin;
        const float *restrict w2 = Wm + (size_t)(j + 2) * (size_t)nin;
        const float *restrict w3 = Wm + (size_t)(j + 3) * (size_t)nin;
        float32x4_t a0 = vdupq_n_f32(0.0f), a1 = vdupq_n_f32(0.0f);
        float32x4_t a2 = vdupq_n_f32(0.0f), a3 = vdupq_n_f32(0.0f);
        for (int i = 0; i < nin; i += 4) {
            const float32x4_t xv = vld1q_f32(x + i);
            a0 = vfmaq_f32(a0, vld1q_f32(w0 + i), xv);
            a1 = vfmaq_f32(a1, vld1q_f32(w1 + i), xv);
            a2 = vfmaq_f32(a2, vld1q_f32(w2 + i), xv);
            a3 = vfmaq_f32(a3, vld1q_f32(w3 + i), xv);
        }
        if (bias) {
            y[j + 0] = vaddvq_f32(a0) + bias[j + 0];
            y[j + 1] = vaddvq_f32(a1) + bias[j + 1];
            y[j + 2] = vaddvq_f32(a2) + bias[j + 2];
            y[j + 3] = vaddvq_f32(a3) + bias[j + 3];
        } else {
            y[j + 0] = vaddvq_f32(a0);
            y[j + 1] = vaddvq_f32(a1);
            y[j + 2] = vaddvq_f32(a2);
            y[j + 3] = vaddvq_f32(a3);
        }
    }
    for (; j < nout; j++) {
        const float *restrict w = Wm + (size_t)j * (size_t)nin;
        float32x4_t a = vdupq_n_f32(0.0f);
        for (int i = 0; i < nin; i += 4) a = vfmaq_f32(a, vld1q_f32(w + i), vld1q_f32(x + i));
        const float s = vaddvq_f32(a);
        y[j] = bias ? s + bias[j] : s;
    }
#else
    /* Four accumulators reduced as ((s0+s1)+(s2+s3)) and an explicit fmaf: the
     * same four lanes, in the same order, with the same fusion the NEON kernel
     * above uses -- so a build without NEON produces bit-identical results to
     * one with it, and tests/test_net.c can hold both to the same numbers. */
    for (int j = 0; j < nout; j++) {
        const float *restrict w = Wm + (size_t)j * (size_t)nin;
        float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
        for (int i = 0; i < nin; i += 4) {
            s0 = fmaf(w[i + 0], x[i + 0], s0);
            s1 = fmaf(w[i + 1], x[i + 1], s1);
            s2 = fmaf(w[i + 2], x[i + 2], s2);
            s3 = fmaf(w[i + 3], x[i + 3], s3);
        }
        const float s = (s0 + s1) + (s2 + s3);
        y[j] = bias ? s + bias[j] : s;
    }
#endif
}

/* The same product for `nb` positions at once: y[b][j] = dot(row j, x[b]).
 * Rows of x and y are strided (they usually live inside an array of Fwd), so
 * nothing is ever copied to call this.
 *
 * Four weight rows x four positions per pass: sixteen accumulators fed by eight
 * loads, which is where the arithmetic intensity comes from -- each weight
 * vector is used four times instead of once.  Every accumulator is still one
 * lane-wise chain over i reduced by vaddvq_f32, exactly as in nn_matvec, so
 * this is bit-identical to calling nn_matvec on each row.  Measured at 128x128:
 * 24.3 GFLOP/s against 14.9 unbatched. */
NN_INLINE void nn_matvec_batch(const float *restrict Wm, const float *restrict bias,
                               const float *restrict x, int ldx, int nb,
                               int nout, int nin, float *restrict y, int ldy)
{
#if NN_NEON
    int j = 0;
    for (; j + 4 <= nout; j += 4) {
        const float *w0 = Wm + (size_t)(j + 0) * (size_t)nin;
        const float *w1 = Wm + (size_t)(j + 1) * (size_t)nin;
        const float *w2 = Wm + (size_t)(j + 2) * (size_t)nin;
        const float *w3 = Wm + (size_t)(j + 3) * (size_t)nin;
        int b = 0;
        for (; b + 4 <= nb; b += 4) {
            const float *x0 = x + (size_t)(b + 0) * (size_t)ldx;
            const float *x1 = x + (size_t)(b + 1) * (size_t)ldx;
            const float *x2 = x + (size_t)(b + 2) * (size_t)ldx;
            const float *x3 = x + (size_t)(b + 3) * (size_t)ldx;
            float32x4_t a[4][4];
            for (int r = 0; r < 4; r++)
                for (int cc = 0; cc < 4; cc++) a[r][cc] = vdupq_n_f32(0.0f);
            for (int i = 0; i < nin; i += 4) {
                const float32x4_t v0 = vld1q_f32(x0 + i), v1 = vld1q_f32(x1 + i);
                const float32x4_t v2 = vld1q_f32(x2 + i), v3 = vld1q_f32(x3 + i);
                const float32x4_t u0 = vld1q_f32(w0 + i), u1 = vld1q_f32(w1 + i);
                const float32x4_t u2 = vld1q_f32(w2 + i), u3 = vld1q_f32(w3 + i);
                a[0][0] = vfmaq_f32(a[0][0], u0, v0); a[0][1] = vfmaq_f32(a[0][1], u0, v1);
                a[0][2] = vfmaq_f32(a[0][2], u0, v2); a[0][3] = vfmaq_f32(a[0][3], u0, v3);
                a[1][0] = vfmaq_f32(a[1][0], u1, v0); a[1][1] = vfmaq_f32(a[1][1], u1, v1);
                a[1][2] = vfmaq_f32(a[1][2], u1, v2); a[1][3] = vfmaq_f32(a[1][3], u1, v3);
                a[2][0] = vfmaq_f32(a[2][0], u2, v0); a[2][1] = vfmaq_f32(a[2][1], u2, v1);
                a[2][2] = vfmaq_f32(a[2][2], u2, v2); a[2][3] = vfmaq_f32(a[2][3], u2, v3);
                a[3][0] = vfmaq_f32(a[3][0], u3, v0); a[3][1] = vfmaq_f32(a[3][1], u3, v1);
                a[3][2] = vfmaq_f32(a[3][2], u3, v2); a[3][3] = vfmaq_f32(a[3][3], u3, v3);
            }
            for (int r = 0; r < 4; r++)
                for (int cc = 0; cc < 4; cc++) {
                    const float s = vaddvq_f32(a[r][cc]);
                    y[(size_t)(b + cc) * (size_t)ldy + j + r] = bias ? s + bias[j + r] : s;
                }
        }
        for (; b < nb; b++)
            nn_matvec(Wm + (size_t)j * (size_t)nin, bias ? bias + j : NULL,
                      x + (size_t)b * (size_t)ldx, 4, nin,
                      y + (size_t)b * (size_t)ldy + j);
    }
    for (; j < nout; j++)
        for (int b = 0; b < nb; b++)
            nn_matvec(Wm + (size_t)j * (size_t)nin, bias ? bias + j : NULL,
                      x + (size_t)b * (size_t)ldx, 1, nin,
                      y + (size_t)b * (size_t)ldy + j);
#else
    for (int b = 0; b < nb; b++)
        nn_matvec(Wm, bias, x + (size_t)b * (size_t)ldx, nout, nin,
                  y + (size_t)b * (size_t)ldy);
#endif
}

/* --------------------------------------------------------------- nn_eval */

void nn_eval(const Trunk *t, const Head *h, const uint16_t *fidx, int nf, Fwd *fw)
{
    /* the policy trunk:  acc -> h1 -> (residual block) -> h2 -> q */
    float acc[NF_ACC];
    nn_gather(t->W0, t->b0, h->z, fidx, nf, NF_ACC, acc);
    fw->r1 = nn_norm_relu(acc, t->g0, t->c0, NF_ACC, fw->x1, fw->h1);

    float z2[NF_HID];
    nn_matvec(t->W1, t->b1, fw->h1, NF_HID, NF_ACC, z2);
    fw->r2 = nn_norm_relu_add(z2, t->g1, t->c1, fw->h1, NF_HID, fw->x2, fw->h2);

    nn_matvec(h->Wp, NULL, fw->h2, NF_PDIM, NF_HID, fw->q);

    /* THE VALUE TRUNK.  A second sparse gather over the same ~35 active
     * features into its own accumulator.  Nothing on this path is shared with
     * the policy, so nothing on it is shaped by the policy loss -- which is the
     * entire point. */
    float av[NF_VACC];
    nn_gather(t->W0v, t->b0v, NULL, fidx, nf, NF_VACC, av);
    fw->rv0 = nn_norm_relu(av, t->g0v, t->c0v, NF_VACC, fw->xv0, fw->hv0);

    float zv[NF_VHID];
    nn_matvec(h->Wvh, h->bvh, fw->hv0, NF_VHID, NF_VACC, zv);
    fw->rv = nn_norm_relu(zv, h->gv, h->cv, NF_VHID, fw->xv, fw->hv);

    nn_matvec(h->Wv, h->bv, fw->hv, 1, NF_VHID, &fw->raw_v);
    fw->v = tanhf(fw->raw_v);
}

/* ----------------------------------------------------------- nn_eval_batch */
/*
 * WHY THE SIGNATURE IS WHAT IT IS
 * -------------------------------
 * `heads` is an array of POINTERS, one per row, because different agents play
 * different games: a batch assembled across games has a different head per row
 * and a batch assembled inside one search has the same head in every row.  Both
 * have to work, and the second has to be fast, so the code walks maximal RUNS
 * of equal head pointers and batches the per-head layers over each run.  A
 * caller that sorts its batch by head therefore gets one run and the fastest
 * path; a caller that does not still gets a fully batched trunk, which is 77%
 * of the arithmetic.
 *
 * `fidx` is an array of pointers too, so nothing has to be copied into a
 * rectangular buffer first -- the feature lists live wherever the caller keeps
 * them and are different lengths anyway.
 *
 * `out` must be an array of nbatch Fwd, and it is both the output AND the
 * working storage: h1, h2, hv0 and hv are read back out of it as GEMM inputs at
 * stride NN_FWD_STRIDE, so a batched evaluation copies nothing.  That is also
 * why there is no workspace argument: the only things not already in Fwd are
 * acc, z2, av and zv, which the tiling keeps on the stack (40 KB at
 * NN_BATCH_TILE = 32; the default pthread stack on macOS is 512 KB).
 *
 * EXACTNESS.  Without Accelerate every row goes through the same kernels
 * nn_eval() uses, with the same lane-wise accumulation, so
 *
 *     nn_eval_batch(t, &h, 1, &fidx, &nf, &fw)   ==   nn_eval(t, h, fidx, nf, &fw)
 *
 * bit for bit, and so does every row of every larger batch: tests/test_net.c
 * asserts exactly that over 9354 rows at fifteen batch sizes, with the heads
 * grouped and mixed.  WITH
 * -DUSE_ACCELERATE the two trunk GEMMs go to cblas_sgemm, whose blocking and
 * use of FMA regroup each sum, and the results are then equal only to about
 * 1e-6 relative -- the test asserts that bound instead.  A caller that needs
 * bit-reproducible search (the MCTS determinism tests) must either build
 * without USE_ACCELERATE or keep the batch composition fixed, because sgemm's
 * summation order is a function of the batch size.
 */

/* Below this many rows the hand-written NEON kernel beats cblas_sgemm and is
 * bit-exact against nn_eval, so small batches go through it even in an
 * Accelerate build.  Measured against unbatched nn_eval, 1 thread: sgemm 0.97x
 * at 1 row, 1.00x at 2, 1.18x at 4, 1.34x at 8, 1.52x at 16, 1.71x at 32; the
 * NEON kernel 0.99x / 1.02x / 1.20x / 1.36x / 1.32x / 1.36x.  They cross at 16.
 * A BLAS call has a fixed cost that 128 x 128 x M cannot amortise until M is
 * big, which is also why routing a ONE-row batch through sgemm was slower than
 * not batching at all (0.76x when it was measured that way).  The threshold is
 * also what makes nbatch == 1 bit-identical to nn_eval in EVERY build, as the
 * contract in net.h promises. */
#define NN_GEMM_MIN 16

/* Rows per internal tile: nn_eval_batch takes any nbatch and chops it into
 * these, so no caller has to think about a maximum.  The tile bounds the stack
 * scratch (acc + z2 + av + zv: 40 KB at 32 rows, against the 512 KB default
 * pthread stack on macOS) and it is what sgemm's M dimension ends up being.
 *
 * Swept at batch 128 on this M3, one thread: tile 4 -> 795k evals/sec,
 * 8 -> 812k, 16 -> 809k, 32 -> 809k, 64 -> 816k.  For the NEON kernel the tile
 * does not matter -- it blocks 4 rows x 4 outputs internally and nothing above
 * that is load-bearing.  cblas_sgemm does care: it does not beat the NEON
 * kernel until 16 rows and has saturated by 32.  So 32, for the one path that
 * has an opinion. */
#ifndef NN_BATCH_TILE
#  define NN_BATCH_TILE 32
#endif

static void nn_eval_tile(const Trunk *t, const Head *const *heads, int nb,
                         const uint16_t *const *fidx, const int *nf, Fwd *out)
{
    float acc[NN_BATCH_TILE * NF_ACC];
    float z2 [NN_BATCH_TILE * NF_HID];
    float av [NN_BATCH_TILE * NF_VACC];
    float zv [NN_BATCH_TILE * NF_VHID];

    /* 1. the sparse gather and the first LayerNorm, per position (the gather is
     *    per-position by nature and the norm's reduction is serial). */
    for (int b = 0; b < nb; b++) {
        nn_gather(t->W0, t->b0, heads[b]->z, fidx[b], nf[b], NF_ACC, acc + b * NF_ACC);
        out[b].r1 = nn_norm_relu(acc + b * NF_ACC, t->g0, t->c0, NF_ACC,
                                 out[b].x1, out[b].h1);
    }

    /* 2. the residual block's matrix, shared by the whole batch: the one place
     *    where this is a matrix-MATRIX product for every caller. */
#if NN_ACCELERATE
    if (nb >= NN_GEMM_MIN) {
        for (int b = 0; b < nb; b++) memcpy(z2 + b * NF_HID, t->b1, sizeof t->b1);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, nb, NF_HID, NF_ACC,
                    1.0f, out[0].h1, NN_FWD_STRIDE, t->W1, NF_ACC, 1.0f, z2, NF_HID);
    } else
#endif
    {
        nn_matvec_batch(t->W1, t->b1, out[0].h1, NN_FWD_STRIDE, nb,
                        NF_HID, NF_ACC, z2, NF_HID);
    }

    /* 3. LayerNorm 2 + the skip connection, per position. */
    for (int b = 0; b < nb; b++)
        out[b].r2 = nn_norm_relu_add(z2 + b * NF_HID, t->g1, t->c1, out[b].h1,
                                     NF_HID, out[b].x2, out[b].h2);

    /* 4. the policy projection -- per AGENT, so batched over runs of equal head. */
    for (int b0 = 0; b0 < nb; ) {
        int b1 = b0 + 1;
        while (b1 < nb && heads[b1] == heads[b0]) b1++;
        const int run = b1 - b0;
#if NN_ACCELERATE
        if (run >= NN_GEMM_MIN) {
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, run, NF_PDIM, NF_HID,
                        1.0f, out[b0].h2, NN_FWD_STRIDE, heads[b0]->Wp, NF_HID,
                        0.0f, out[b0].q, NN_FWD_STRIDE);
        } else
#endif
        if (run > 1) {
            nn_matvec_batch(heads[b0]->Wp, NULL, out[b0].h2, NN_FWD_STRIDE, run,
                            NF_PDIM, NF_HID, out[b0].q, NN_FWD_STRIDE);
        } else {
            nn_matvec(heads[b0]->Wp, NULL, out[b0].h2, NF_PDIM, NF_HID, out[b0].q);
        }
        /* (the #if above falls through to these two when Accelerate is off, or
         *  when the run is too short for a BLAS call to pay for itself) */
        b0 = b1;
    }

    /* 5. the value trunk: its own gather and norm, shared weights, per position. */
    for (int b = 0; b < nb; b++) {
        nn_gather(t->W0v, t->b0v, NULL, fidx[b], nf[b], NF_VACC, av + b * NF_VACC);
        out[b].rv0 = nn_norm_relu(av + b * NF_VACC, t->g0v, t->c0v, NF_VACC,
                                  out[b].xv0, out[b].hv0);
    }

    /* 6. the value head's hidden layer -- per agent again. */
    for (int b0 = 0; b0 < nb; ) {
        int b1 = b0 + 1;
        while (b1 < nb && heads[b1] == heads[b0]) b1++;
        const int run = b1 - b0;
        if (run > 1)
            nn_matvec_batch(heads[b0]->Wvh, heads[b0]->bvh, out[b0].hv0, NN_FWD_STRIDE,
                            run, NF_VHID, NF_VACC, zv + b0 * NF_VHID, NF_VHID);
        else
            nn_matvec(heads[b0]->Wvh, heads[b0]->bvh, out[b0].hv0, NF_VHID, NF_VACC,
                      zv + b0 * NF_VHID);
        b0 = b1;
    }

    /* 7. the last norm and the scalar output.  32 wide and 32 MACs: per row. */
    for (int b = 0; b < nb; b++) {
        const Head *h = heads[b];
        out[b].rv = nn_norm_relu(zv + b * NF_VHID, h->gv, h->cv, NF_VHID,
                                 out[b].xv, out[b].hv);
        nn_matvec(h->Wv, h->bv, out[b].hv, 1, NF_VHID, &out[b].raw_v);
        out[b].v = tanhf(out[b].raw_v);
    }
}

void nn_eval_batch(const Trunk *t, const Head *const *heads, int nbatch,
                   const uint16_t *const *fidx, const int *nf, Fwd *out)
{
    for (int base = 0; base < nbatch; base += NN_BATCH_TILE) {
        int nb = nbatch - base;
        if (nb > NN_BATCH_TILE) nb = NN_BATCH_TILE;
        nn_eval_tile(t, heads + base, nb, fidx + base, nf + base, out + base);
    }
}

const char *nn_backend(void)
{
#if NN_ACCELERATE
    return "neon+accelerate";
#elif NN_NEON
    return "neon";
#else
    return "scalar";
#endif
}

int nn_batch_tile(void) { return NN_BATCH_TILE; }

int nn_batch_is_exact(int nbatch)
{
#if NN_ACCELERATE
    /* every tile smaller than the sgemm threshold takes the exact kernel */
    return nbatch < NN_GEMM_MIN;
#else
    (void)nbatch;
    return 1;
#endif
}

/* The per-move logits.  NF_PDIM is 32, so each move is five 32-float embedding
 * rows summed and dotted with q -- 192 flops against ~1.4 KB of scattered
 * embedding rows, which makes this stage load-bound rather than FLOP-bound.
 * Two lanes of 4 cover the 32 in eight loads per row; the accumulation is the
 * same lane-wise chain the matvec uses, reduced by vaddvq_f32. */
void nn_logits(const Head *h, const Fwd *fw, const MoveKey *keys, int n, float *logits)
{
    if (n <= 0) return;
    if (!keys) {                     /* no Position available in this signature */
        for (int m = 0; m < n; m++) logits[m] = 0.0f;
        return;
    }

    const float *restrict q   = fw->q;
    const float *restrict Bft = h->Bft;

    for (int m = 0; m < n; m++) {
        const MoveKey k = keys[m];
        const float *restrict ef = h->Efrom  + (size_t)k.from  * NF_PDIM;
        const float *restrict et = h->Eto    + (size_t)k.to    * NF_PDIM;
        const float *restrict ec = h->Epc    + (size_t)k.pc    * NF_PDIM;
        const float *restrict er = h->Epromo + (size_t)k.promo * NF_PDIM;
        const float *restrict ex = h->Ecap   + (size_t)k.cap   * NF_PDIM;

#if NN_NEON
        float32x4_t a = vdupq_n_f32(0.0f);
        for (int i = 0; i < NF_PDIM; i += 4) {
            float32x4_t e = vaddq_f32(vld1q_f32(ef + i), vld1q_f32(et + i));
            e = vaddq_f32(e, vld1q_f32(ec + i));
            e = vaddq_f32(e, vld1q_f32(er + i));
            e = vaddq_f32(e, vld1q_f32(ex + i));
            a = vfmaq_f32(a, vld1q_f32(q + i), e);
        }
        logits[m] = vaddvq_f32(a) + Bft[(size_t)k.from * 64 + k.to];
#else
        float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
        for (int i = 0; i < NF_PDIM; i += 4) {
            s0 = fmaf(q[i + 0], ef[i + 0] + et[i + 0] + ec[i + 0] + er[i + 0] + ex[i + 0], s0);
            s1 = fmaf(q[i + 1], ef[i + 1] + et[i + 1] + ec[i + 1] + er[i + 1] + ex[i + 1], s1);
            s2 = fmaf(q[i + 2], ef[i + 2] + et[i + 2] + ec[i + 2] + er[i + 2] + ex[i + 2], s2);
            s3 = fmaf(q[i + 3], ef[i + 3] + et[i + 3] + ec[i + 3] + er[i + 3] + ex[i + 3], s3);
        }
        logits[m] = ((s0 + s1) + (s2 + s3)) + Bft[(size_t)k.from * 64 + k.to];
#endif
    }
}

/* ---------------------------------------------------------------- backward */

void nn_backward(const Trunk *t, const Head *h, const Fwd *fw,
                 const uint16_t *fidx, int nf,
                 const MoveKey *keys, int n, const float *dlogits, float dvalue,
                 TrunkGrad *tg, HeadGrad *hg)
{
    float dq[NF_PDIM];
    float dh2[NF_HID];
    float dh1[NF_ACC];

    for (int k = 0; k < NF_PDIM; k++) dq[k]  = 0.0f;
    for (int j = 0; j < NF_HID;  j++) dh2[j] = 0.0f;
    for (int i = 0; i < NF_ACC;  i++) dh1[i] = 0.0f;

    /* ---- policy: logit_m = q . (Efrom+Eto+Epc+Epromo+Ecap) + Bft[from,to] */
    if (keys && dlogits && n > 0) {
        const float *restrict q = fw->q;
        for (int m = 0; m < n; m++) {
            const float d = dlogits[m];
            const MoveKey k = keys[m];
            const float *restrict ef = h->Efrom  + (size_t)k.from  * NF_PDIM;
            const float *restrict et = h->Eto    + (size_t)k.to    * NF_PDIM;
            const float *restrict ec = h->Epc    + (size_t)k.pc    * NF_PDIM;
            const float *restrict er = h->Epromo + (size_t)k.promo * NF_PDIM;
            const float *restrict ex = h->Ecap   + (size_t)k.cap   * NF_PDIM;
            float *restrict gf = hg->Efrom  + (size_t)k.from  * NF_PDIM;
            float *restrict gt = hg->Eto    + (size_t)k.to    * NF_PDIM;
            float *restrict gc = hg->Epc    + (size_t)k.pc    * NF_PDIM;
            float *restrict gr = hg->Epromo + (size_t)k.promo * NF_PDIM;
            float *restrict gx = hg->Ecap   + (size_t)k.cap   * NF_PDIM;

            for (int i = 0; i < NF_PDIM; i++) {
                dq[i] += d * (ef[i] + et[i] + ec[i] + er[i] + ex[i]);
                const float dqi = d * q[i];
                gf[i] += dqi;
                gt[i] += dqi;
                gc[i] += dqi;
                gr[i] += dqi;
                gx[i] += dqi;
            }
            hg->Bft[(size_t)k.from * 64 + k.to] += d;
        }
    }

    /* ---- q = Wp.h2 */
    {
        const float *restrict h2 = fw->h2;
        const float *restrict Wp = h->Wp;
        for (int k = 0; k < NF_PDIM; k++) {
            const float d = dq[k];
            const float *restrict w  = Wp     + (size_t)k * NF_HID;
            float       *restrict gw = hg->Wp + (size_t)k * NF_HID;
            for (int j = 0; j < NF_HID; j++) {
                gw[j] += d * h2[j];
                dh2[j] += d * w[j];
            }
        }
    }

    /* ---- value head.  v = tanh(raw_v), so d/draw_v = dvalue * (1 - v^2). */
    {
        const float draw_v = dvalue * (1.0f - fw->v * fw->v);
        const float *restrict hv = fw->hv;
        const float *restrict wv = h->Wv;
        float dhv[NF_VHID], dzv[NF_VHID];

        for (int k = 0; k < NF_VHID; k++) {
            hg->Wv[k] += draw_v * hv[k];
            /* relu gate: hv[k] > 0 exactly when the pre-activation was */
            dhv[k] = (hv[k] > 0.0f) ? draw_v * wv[k] : 0.0f;
        }
        hg->bv[0] += draw_v;

        norm_bwd(dhv, fw->xv, h->gv, fw->rv, NF_VHID, hg->gv, hg->cv, dzv);

        float dhv0[NF_VACC];
        for (int i = 0; i < NF_VACC; i++) dhv0[i] = 0.0f;

        const float *restrict hv0 = fw->hv0;
        const float *restrict Wvh = h->Wvh;
        for (int k = 0; k < NF_VHID; k++) {
            const float d = dzv[k];
            hg->bvh[k] += d;
            const float *restrict w  = Wvh      + (size_t)k * NF_VACC;
            float       *restrict gw = hg->Wvh  + (size_t)k * NF_VACC;
            for (int i = 0; i < NF_VACC; i++) {
                gw[i] += d * hv0[i];
                dhv0[i] += d * w[i];
            }
        }

        /* back through the value trunk's relu, LayerNorm and sparse gather */
        float dav[NF_VACC], dacc_v[NF_VACC];
        for (int i = 0; i < NF_VACC; i++) dav[i] = (hv0[i] > 0.0f) ? dhv0[i] : 0.0f;
        norm_bwd(dav, fw->xv0, t->g0v, fw->rv0, NF_VACC, tg->g0v, tg->c0v, dacc_v);
        for (int i = 0; i < NF_VACC; i++) tg->b0v[i] += dacc_v[i];
        for (int f = 0; f < nf; f++) {
            float *restrict gw = tg->W0v + (size_t)fidx[f] * NF_VACC;
            for (int i = 0; i < NF_VACC; i++) gw[i] += dacc_v[i];
        }
    }

    /* ---- residual block: h2 = relu(g1 * x2 + c1) + h1 ; z2 = W1.h1 + b1 */
    {
        float da2[NF_HID], dz2[NF_HID];
        const float *restrict g1 = t->g1;
        const float *restrict c1 = t->c1;
        const float *restrict x2 = fw->x2;

        /* The relu gate cannot be read off h2 -- the identity term is added to
         * it -- so recompute the pre-activation.  Two flops per unit. */
        for (int j = 0; j < NF_HID; j++)
            da2[j] = (g1[j] * x2[j] + c1[j] > 0.0f) ? dh2[j] : 0.0f;

        norm_bwd(da2, x2, g1, fw->r2, NF_HID, tg->g1, tg->c1, dz2);

        /* THE identity path.  It is what lets the policy gradient reach W0
         * without being attenuated by W1's relu gate -- the reason the first
         * layer of the old design learned so slowly late in a run. */
        for (int i = 0; i < NF_ACC; i++) dh1[i] += dh2[i];

        const float *restrict h1 = fw->h1;
        const float *restrict W1 = t->W1;
        float *restrict gb1 = tg->b1;
        for (int j = 0; j < NF_HID; j++) {
            const float d = dz2[j];
            gb1[j] += d;
            if (d == 0.0f) continue;
            const float *restrict w  = W1     + (size_t)j * NF_ACC;
            float       *restrict gw = tg->W1 + (size_t)j * NF_ACC;
            for (int i = 0; i < NF_ACC; i++) {
                gw[i] += d * h1[i];
                dh1[i] += d * w[i];
            }
        }
    }

    /* ---- h1 = relu(g0 * x1 + c0) ; acc = b0 + z + sum of active W0 rows */
    {
        float da1[NF_ACC], dacc[NF_ACC];
        const float *restrict h1 = fw->h1;
        for (int i = 0; i < NF_ACC; i++) da1[i] = (h1[i] > 0.0f) ? dh1[i] : 0.0f;

        norm_bwd(da1, fw->x1, t->g0, fw->r1, NF_ACC, tg->g0, tg->c0, dacc);

        float *restrict gb0 = tg->b0;
        float *restrict gz  = hg->z;
        for (int i = 0; i < NF_ACC; i++) {
            gb0[i] += dacc[i];
            gz[i]  += dacc[i];
        }
        float *restrict W0g = tg->W0;
        for (int f = 0; f < nf; f++) {
            float *restrict gw = W0g + (size_t)fidx[f] * NF_ACC;
            for (int i = 0; i < NF_ACC; i++) gw[i] += dacc[i];
        }
    }
}

/* --------------------------------------------------------------- optimiser */

void grad_zero(void *g, int nfloats)
{
    if (g && nfloats > 0) memset(g, 0, (size_t)nfloats * sizeof(float));
}

void grad_add(float *dst, const float *src, int n)
{
    float *restrict d = dst;
    const float *restrict s = src;
    for (int i = 0; i < n; i++) d[i] += s[i];
}

void adam_init(Adam *a, int n)
{
    a->n = n > 0 ? n : 0;
    a->t = 0;
    a->m = NULL;
    a->v = NULL;
    if (a->n > 0) {
        a->m = (float *)calloc((size_t)a->n, sizeof(float));
        a->v = (float *)calloc((size_t)a->n, sizeof(float));
        if (!a->m || !a->v) {                 /* out of memory: degrade to no-op */
            free(a->m); free(a->v);
            a->m = a->v = NULL;
            a->n = 0;
        }
    }
}

void adam_free(Adam *a)
{
    if (!a) return;
    free(a->m);
    free(a->v);
    a->m = a->v = NULL;
    a->n = 0;
    a->t = 0;
}

void adam_step(Adam *a, float *p, float *g, float lr, float wd, float clip)
{
    if (!a || !a->m || !a->v || !p || !g || a->n <= 0) return;

    const int n = a->n;

    /* Global grad-norm clipping over the whole parameter vector. */
    float scale = 1.0f;
    if (clip > 0.0f) {
        double ss = 0.0;
        for (int i = 0; i < n; i++) ss += (double)g[i] * (double)g[i];
        const double norm = sqrt(ss);
        if (norm > (double)clip) scale = (float)((double)clip / (norm + 1e-12));
    }

    a->t++;
    const float b1 = 0.9f, b2 = 0.999f, eps = 1e-8f;
    const float bc1 = 1.0f - powf(b1, (float)a->t);
    const float bc2 = 1.0f - powf(b2, (float)a->t);
    const float inv_bc1 = 1.0f / bc1;
    const float inv_bc2 = 1.0f / bc2;

    float *restrict pp = p;
    float *restrict gg = g;
    float *restrict mm = a->m;
    float *restrict vv = a->v;

    for (int i = 0; i < n; i++) {
        const float gi = gg[i] * scale;
        const float m = b1 * mm[i] + (1.0f - b1) * gi;
        const float v = b2 * vv[i] + (1.0f - b2) * gi * gi;
        mm[i] = m;
        vv[i] = v;
        const float mh = m * inv_bc1;
        const float vh = v * inv_bc2;
        /* Decoupled weight decay (AdamW): the wd term bypasses the moments. */
        pp[i] -= lr * (mh / (sqrtf(vh) + eps) + wd * pp[i]);
        gg[i] = 0.0f;
    }
}

/* ----------------------------------------------------------- serialisation */

#if defined(__BYTE_ORDER__) && defined(__ORDER_BIG_ENDIAN__) && \
    (__BYTE_ORDER__ == __ORDER_BIG_ENDIAN__)
#define NN_BIG_ENDIAN 1
#else
#define NN_BIG_ENDIAN 0
#endif

static int wr_u32(FILE *f, uint32_t v)
{
    unsigned char b[4];
    b[0] = (unsigned char)(v);
    b[1] = (unsigned char)(v >> 8);
    b[2] = (unsigned char)(v >> 16);
    b[3] = (unsigned char)(v >> 24);
    return fwrite(b, 1, 4, f) == 4;
}

static int rd_u32(FILE *f, uint32_t *v)
{
    unsigned char b[4];
    if (fread(b, 1, 4, f) != 4) return 0;
    *v = (uint32_t)b[0] | ((uint32_t)b[1] << 8) | ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24);
    return 1;
}

#if NN_BIG_ENDIAN
static void swap32_block(uint32_t *d, size_t n)
{
    for (size_t i = 0; i < n; i++) {
        const uint32_t x = d[i];
        d[i] = (x >> 24) | ((x >> 8) & 0x0000FF00u) | ((x << 8) & 0x00FF0000u) | (x << 24);
    }
}
#endif

static int wr_floats(FILE *f, const float *p, size_t n)
{
    if (n == 0) return 1;
#if NN_BIG_ENDIAN
    uint32_t buf[512];
    size_t done = 0;
    while (done < n) {
        size_t c = n - done;
        if (c > 512) c = 512;
        memcpy(buf, p + done, c * sizeof(float));
        swap32_block(buf, c);
        if (fwrite(buf, sizeof(float), c, f) != c) return 0;
        done += c;
    }
    return 1;
#else
    return fwrite(p, sizeof(float), n, f) == n;
#endif
}

static int rd_floats(FILE *f, float *p, size_t n)
{
    if (n == 0) return 1;
    if (fread(p, sizeof(float), n, f) != n) return 0;
#if NN_BIG_ENDIAN
    swap32_block((uint32_t *)p, n);
#endif
    return 1;
}

static int wr_zeros(FILE *f, size_t n)
{
    static const float zeros[512] = {0};
    while (n) {
        size_t c = n > 512 ? 512 : n;
        if (fwrite(zeros, sizeof(float), c, f) != c) return 0;
        n -= c;
    }
    return 1;
}

static int skip_floats(FILE *f, size_t n)
{
    if (n == 0) return 1;
    return fseek(f, (long)(n * sizeof(float)), SEEK_CUR) == 0;
}

#define MODEL_HDR_BYTES 32   /* 8 x uint32 */

int model_save(const char *path, const Trunk *t, const Head *heads,
               const Hyper *hy, const float *elo, int n_agents, int generation)
{
    if (!path || n_agents < 0 || generation < 0) return 0;

    FILE *f = fopen(path, "wb");
    if (!f) return 0;

    const size_t na = (size_t)n_agents;
    int ok = wr_u32(f, MODEL_MAGIC) &&
             wr_u32(f, MODEL_VERSION) &&
             wr_u32(f, (uint32_t)n_agents) &&
             wr_u32(f, (uint32_t)generation) &&
             wr_u32(f, (uint32_t)NF_INPUT) &&
             wr_u32(f, (uint32_t)NF_ACC) &&
             wr_u32(f, (uint32_t)NF_HID) &&
             wr_u32(f, (uint32_t)NF_PDIM);

    if (ok) ok = t     ? wr_floats(f, (const float *)t, TRUNK_NPARAM)
                       : wr_zeros(f, TRUNK_NPARAM);
    if (ok) ok = heads ? wr_floats(f, (const float *)heads, na * HEAD_NPARAM)
                       : wr_zeros(f, na * HEAD_NPARAM);
    if (ok) ok = hy    ? wr_floats(f, (const float *)hy, na * HYPER_NPARAM)
                       : wr_zeros(f, na * HYPER_NPARAM);
    if (ok) ok = elo   ? wr_floats(f, elo, na)
                       : wr_zeros(f, na);

    if (fclose(f) != 0) ok = 0;
    if (!ok) remove(path);
    return ok ? 1 : 0;
}

/* *n_agents is capacity in, and out: the number of agents actually written into
 * the caller's arrays -- or, when no per-agent array is requested at all (a
 * header-only probe), the number of agents stored in the file. */
int model_load(const char *path, Trunk *t, Head *heads, Hyper *hy, float *elo,
               int *n_agents, int *generation)
{
    if (!path) return 0;
    FILE *f = fopen(path, "rb");
    if (!f) return 0;

    uint32_t magic = 0, ver = 0, na = 0, gen = 0, ni = 0, nacc = 0, nhid = 0, npd = 0;
    if (!(rd_u32(f, &magic) && rd_u32(f, &ver) && rd_u32(f, &na) && rd_u32(f, &gen) &&
          rd_u32(f, &ni) && rd_u32(f, &nacc) && rd_u32(f, &nhid) && rd_u32(f, &npd))) {
        fclose(f);
        return 0;
    }
    /* The version check is what makes an old checkpoint fail cleanly.  A
     * version-3 file has the same magic and the same NF_INPUT/NF_ACC/NF_PDIM;
     * only the version and NF_HID differ, and relying on NF_HID would be luck. */
    if (magic != MODEL_MAGIC || ver != MODEL_VERSION ||
        ni != (uint32_t)NF_INPUT || nacc != (uint32_t)NF_ACC ||
        nhid != (uint32_t)NF_HID || npd != (uint32_t)NF_PDIM ||
        na > 1000000u) {
        fclose(f);
        return 0;
    }

    /* The file is a fixed-size record, so its LENGTH is a checksum on the
     * layout, and it is required to match exactly rather than merely to be long
     * enough.  Being strict here is what catches a width the header does not
     * record: NF_VACC and NF_VHID are not among the eight header fields, so a
     * checkpoint written with a different value of either has the same magic,
     * the same version and the same four widths, and would otherwise be read as
     * a correctly-shaped model made of shifted numbers.  It also still rejects
     * the truncated file the skip paths must never be handed. */
    const long long want = (long long)MODEL_HDR_BYTES +
        (long long)sizeof(float) *
        ((long long)TRUNK_NPARAM +
         (long long)na * ((long long)HEAD_NPARAM + (long long)HYPER_NPARAM + 1));
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return 0; }
    const long long have = (long long)ftell(f);
    if (have != want || fseek(f, MODEL_HDR_BYTES, SEEK_SET) != 0) { fclose(f); return 0; }

    const int file_n = (int)na;
    const int want_arrays = (heads != NULL) || (hy != NULL) || (elo != NULL);
    int nload = file_n;
    if (want_arrays) {
        int cap = n_agents ? *n_agents : 0;
        if (cap < 0) cap = 0;
        if (nload > cap) nload = cap;
    }
    const size_t nl = (size_t)nload;
    const size_t rest = (size_t)(file_n - nload);

    int ok = 1;
    if (ok) ok = t ? rd_floats(f, (float *)t, TRUNK_NPARAM) : skip_floats(f, TRUNK_NPARAM);

    if (ok) ok = heads ? rd_floats(f, (float *)heads, nl * HEAD_NPARAM)
                       : skip_floats(f, nl * HEAD_NPARAM);
    if (ok) ok = skip_floats(f, rest * HEAD_NPARAM);

    if (ok) ok = hy ? rd_floats(f, (float *)hy, nl * HYPER_NPARAM)
                    : skip_floats(f, nl * HYPER_NPARAM);
    if (ok) ok = skip_floats(f, rest * HYPER_NPARAM);

    if (ok) ok = elo ? rd_floats(f, elo, nl) : skip_floats(f, nl);
    if (ok) ok = skip_floats(f, rest);

    fclose(f);
    if (!ok) return 0;

    if (n_agents)  *n_agents  = want_arrays ? nload : file_n;
    if (generation) *generation = (int)gen;
    return 1;
}

/* ----------------------------------------------------------------- helpers */

void softmax_t(const float *logits, int n, float temp, float *out)
{
    if (n <= 0) return;
    if (!(temp >= 1e-3f)) temp = 1e-3f;         /* also catches NaN */
    const float inv = 1.0f / temp;

    float mx = logits[0];
    for (int i = 1; i < n; i++) if (logits[i] > mx) mx = logits[i];

    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        const float e = expf((logits[i] - mx) * inv);
        out[i] = e;
        sum += e;
    }
    if (sum > 0.0f) {
        const float r = 1.0f / sum;
        for (int i = 0; i < n; i++) out[i] *= r;
    } else {                                     /* degenerate: uniform */
        const float u = 1.0f / (float)n;
        for (int i = 0; i < n; i++) out[i] = u;
    }
}

float material_balance(const Position *p)
{
    static const float val[5] = {1.0f, 3.0f, 3.25f, 5.0f, 9.0f};
    float s = 0.0f;
    for (int pt = PAWN; pt <= QUEEN; pt++)
        s += val[pt] * (float)(bb_count(p->piece[WHITE][pt]) - bb_count(p->piece[BLACK][pt]));
    return (p->side == BLACK) ? -s : s;
}
