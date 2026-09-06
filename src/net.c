/* net.c -- policy/value network, analytic gradients, AdamW, serialisation.
 *
 * Layout contracts (chosen here, referenced by tests/test_net.c)
 * -------------------------------------------------------------
 *   W0    [f * NF_ACC + i]      row per input feature, contiguous over acc.
 *   W1    [j * NF_ACC + i]      row per hidden unit,   contiguous over acc.
 *   Wp    [k * NF_HID + j]      row per policy dim,    contiguous over hidden.
 *   E*    [row * NF_PDIM + k]
 *   Bft   [from * 64 + to]
 *
 * W1/Wp are stored output-major so the inner reduction runs over a contiguous
 * span (NF_ACC / NF_HID, both compile-time constants and multiples of 8).  The
 * reductions use four independent accumulators: clang's SLP vectoriser folds
 * them into one NEON vector accumulator, which plain `-O3` cannot do to a
 * single-chain float reduction because that would require reassociation.
 *
 * nn_eval() is the hot path (~120k calls/second); everything in it is a
 * contiguous float loop over a constant trip count with restrict-qualified
 * pointers so no bounds/aliasing check survives into the loop body.
 */

#include "net.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

_Static_assert(sizeof(Trunk) ==
               (size_t)(NF_INPUT * NF_ACC + NF_ACC + NF_ACC * NF_HID + NF_HID) * sizeof(float),
               "Trunk must be pure float storage with no padding");
_Static_assert(sizeof(Head) ==
               (size_t)(NF_ACC + NF_HID + 1 + NF_HID * NF_PDIM + 64 * NF_PDIM + 64 * NF_PDIM +
                        6 * NF_PDIM + 5 * NF_PDIM + 7 * NF_PDIM + 64 * 64) * sizeof(float),
               "Head must be pure float storage with no padding");
_Static_assert(sizeof(Hyper) == 8 * sizeof(float), "Hyper must be 8 floats");
_Static_assert(NF_ACC % 8 == 0 && NF_HID % 8 == 0 && NF_PDIM % 8 == 0,
               "layer widths must be multiples of 8");

#define HYPER_NPARAM (sizeof(Hyper) / sizeof(float))

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

/* -------------------------------------------------------------------- init */

void nn_init(Trunk *t, Head *h, uint64_t seed)
{
    uint64_t st[4];
    nn_rng_seed(st, seed);

    if (t) {
        /* ~35 rows are summed into the accumulator, so scale by 1/sqrt(35) to
         * keep the pre-activation O(1) rather than O(sqrt(35)). */
        fill_normal(t->W0, (size_t)NF_INPUT * NF_ACC, 0.5f / sqrtf(35.0f), st);
        memset(t->b0, 0, sizeof t->b0);
        fill_normal(t->W1, (size_t)NF_ACC * NF_HID, sqrtf(2.0f / (float)NF_ACC), st);
        memset(t->b1, 0, sizeof t->b1);
    }

    if (h) {
        memset(h->z, 0, sizeof h->z);                     /* style vector starts flat */
        fill_normal(h->Wp, (size_t)NF_HID * NF_PDIM, sqrtf(2.0f / (float)NF_HID), st);

        /* Five embedding rows are summed before the dot with q, so use a fan-in
         * of 5*NF_PDIM; that keeps the initial logits at std ~0.7 (policy is
         * close to uniform but not degenerate). */
        const float es = sqrtf(2.0f / (5.0f * (float)NF_PDIM));
        fill_normal(h->Efrom,  (size_t)64 * NF_PDIM, es, st);
        fill_normal(h->Eto,    (size_t)64 * NF_PDIM, es, st);
        fill_normal(h->Epc,    (size_t)6  * NF_PDIM, es, st);
        fill_normal(h->Epromo, (size_t)5  * NF_PDIM, es, st);
        fill_normal(h->Ecap,   (size_t)7  * NF_PDIM, es, st);
        memset(h->Bft, 0, sizeof h->Bft);

        /* Value head starts ~0 so the first predictions are unbiased draws. */
        fill_normal(h->Wv, NF_HID, 1.0e-3f, st);
        h->bv[0] = 0.0f;
    }
}

/* ----------------------------------------------------------------- forward */

void nn_eval(const Trunk *t, const Head *h, const uint16_t *fidx, int nf, Fwd *fw)
{
    float *restrict acc = fw->acc;
    float *restrict h1  = fw->h1;
    float *restrict z2  = fw->z2;
    float *restrict h2  = fw->h2;
    float *restrict q   = fw->q;

    {   /* acc = b0 + z */
        const float *restrict b0 = t->b0;
        const float *restrict zz = h->z;
        for (int i = 0; i < NF_ACC; i++) acc[i] = b0[i] + zz[i];
    }

    {   /* acc += sum of the W0 rows named by the active features */
        const float *restrict W0 = t->W0;
        for (int f = 0; f < nf; f++) {
            const float *restrict w = W0 + (size_t)fidx[f] * NF_ACC;
            for (int i = 0; i < NF_ACC; i++) acc[i] += w[i];
        }
    }

    for (int i = 0; i < NF_ACC; i++) {
        const float a = acc[i];
        h1[i] = a > 0.0f ? a : 0.0f;
    }

    {   /* z2 = W1.h1 + b1 ; h2 = relu(z2) */
        const float *restrict W1 = t->W1;
        const float *restrict b1 = t->b1;
        for (int j = 0; j < NF_HID; j++) {
            const float *restrict w = W1 + (size_t)j * NF_ACC;
            float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
            for (int i = 0; i < NF_ACC; i += 4) {
                s0 += w[i + 0] * h1[i + 0];
                s1 += w[i + 1] * h1[i + 1];
                s2 += w[i + 2] * h1[i + 2];
                s3 += w[i + 3] * h1[i + 3];
            }
            const float s = ((s0 + s1) + (s2 + s3)) + b1[j];
            z2[j] = s;
            h2[j] = s > 0.0f ? s : 0.0f;
        }
    }

    {   /* q = Wp.h2 */
        const float *restrict Wp = h->Wp;
        for (int k = 0; k < NF_PDIM; k++) {
            const float *restrict w = Wp + (size_t)k * NF_HID;
            float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
            for (int j = 0; j < NF_HID; j += 4) {
                s0 += w[j + 0] * h2[j + 0];
                s1 += w[j + 1] * h2[j + 1];
                s2 += w[j + 2] * h2[j + 2];
                s3 += w[j + 3] * h2[j + 3];
            }
            q[k] = (s0 + s1) + (s2 + s3);
        }
    }

    {   /* raw_v = Wv.h2 + bv ; v = tanh(raw_v) */
        const float *restrict wv = h->Wv;
        float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
        for (int j = 0; j < NF_HID; j += 4) {
            s0 += wv[j + 0] * h2[j + 0];
            s1 += wv[j + 1] * h2[j + 1];
            s2 += wv[j + 2] * h2[j + 2];
            s3 += wv[j + 3] * h2[j + 3];
        }
        fw->raw_v = ((s0 + s1) + (s2 + s3)) + h->bv[0];
        fw->v = tanhf(fw->raw_v);
    }
}

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

        float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
        for (int i = 0; i < NF_PDIM; i += 4) {
            s0 += q[i + 0] * (ef[i + 0] + et[i + 0] + ec[i + 0] + er[i + 0] + ex[i + 0]);
            s1 += q[i + 1] * (ef[i + 1] + et[i + 1] + ec[i + 1] + er[i + 1] + ex[i + 1]);
            s2 += q[i + 2] * (ef[i + 2] + et[i + 2] + ec[i + 2] + er[i + 2] + ex[i + 2]);
            s3 += q[i + 3] * (ef[i + 3] + et[i + 3] + ec[i + 3] + er[i + 3] + ex[i + 3]);
        }
        logits[m] = ((s0 + s1) + (s2 + s3)) + Bft[(size_t)k.from * 64 + k.to];
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

    for (int k = 0; k < NF_PDIM; k++) dq[k] = 0.0f;

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

    /* ---- value: v = tanh(raw_v), so d/draw_v = dvalue * (1 - v^2) */
    const float draw_v = dvalue * (1.0f - fw->v * fw->v);
    {
        const float *restrict h2 = fw->h2;
        const float *restrict wv = h->Wv;
        float *restrict gv = hg->Wv;
        for (int j = 0; j < NF_HID; j++) {
            gv[j] += draw_v * h2[j];
            dh2[j] = draw_v * wv[j];
        }
        hg->bv[0] += draw_v;
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

    /* ---- h2 = relu(z2) ; z2 = W1.h1 + b1 */
    for (int i = 0; i < NF_ACC; i++) dh1[i] = 0.0f;
    {
        const float *restrict z2 = fw->z2;
        const float *restrict h1 = fw->h1;
        const float *restrict W1 = t->W1;
        float *restrict gb1 = tg->b1;
        for (int j = 0; j < NF_HID; j++) {
            if (z2[j] <= 0.0f) continue;              /* relu gate */
            const float d = dh2[j];
            gb1[j] += d;
            const float *restrict w  = W1     + (size_t)j * NF_ACC;
            float       *restrict gw = tg->W1 + (size_t)j * NF_ACC;
            for (int i = 0; i < NF_ACC; i++) {
                gw[i] += d * h1[i];
                dh1[i] += d * w[i];
            }
        }
    }

    /* ---- h1 = relu(acc) ; acc = b0 + z + sum of active W0 rows */
    {
        const float *restrict acc = fw->acc;
        float *restrict gb0 = tg->b0;
        float *restrict gz  = hg->z;
        for (int i = 0; i < NF_ACC; i++) {
            const float d = (acc[i] > 0.0f) ? dh1[i] : 0.0f;
            dh1[i] = d;
            gb0[i] += d;
            gz[i]  += d;
        }
        float *restrict W0g = tg->W0;
        for (int f = 0; f < nf; f++) {
            float *restrict gw = W0g + (size_t)fidx[f] * NF_ACC;
            for (int i = 0; i < NF_ACC; i++) gw[i] += dh1[i];
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
    if (magic != MODEL_MAGIC || ver != MODEL_VERSION ||
        ni != (uint32_t)NF_INPUT || nacc != (uint32_t)NF_ACC ||
        nhid != (uint32_t)NF_HID || npd != (uint32_t)NF_PDIM ||
        na > 1000000u) {
        fclose(f);
        return 0;
    }

    /* Reject truncated files up front so the skip paths cannot read past EOF. */
    const long long want = (long long)MODEL_HDR_BYTES +
        (long long)sizeof(float) *
        ((long long)TRUNK_NPARAM +
         (long long)na * ((long long)HEAD_NPARAM + (long long)HYPER_NPARAM + 1));
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return 0; }
    const long long have = (long long)ftell(f);
    if (have < want || fseek(f, MODEL_HDR_BYTES, SEEK_SET) != 0) { fclose(f); return 0; }

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
