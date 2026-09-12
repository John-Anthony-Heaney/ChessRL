/* net.h -- the policy/value network, its gradients, and the optimiser.
 *
 * Design notes (why it looks like this)
 * ------------------------------------
 * The whole system must sustain a large number of complete games/second on an
 * 8-core laptop, so the network is deliberately small and, crucially, its first
 * (and by far largest) layer consumes a SPARSE binary feature vector: only ~35
 * of 788 inputs are ever 1, so "W0 @ x" is a gather-and-sum of ~35 rows rather
 * than a 788x160 matmul.  This is the NNUE trick, minus the incremental
 * accumulator (unnecessary here -- a full sparse refresh is already cheap).
 *
 * The population shares one trunk and gives every agent its own head plus a
 * "style vector" z that is added into the accumulator.  Under an extremely
 * limited compute budget this is what makes the run work: the trunk sees every
 * position played by all agents, while the per-agent heads still give evolution
 * something real to select on and keep agents genuinely distinct opponents.
 *
 * Everything is float32 and framework-free so it runs with zero dependencies.
 *
 * ---------------------------------------------------------------------------
 * THE SHAPE OF THE NETWORK, AND WHY IT CHANGED
 * ---------------------------------------------------------------------------
 * The previous shape was
 *
 *     acc = W0.x + b0 + z ; h1 = relu(acc)
 *     h2  = relu(W1.h1 + b1)                     (160 -> 96, no normalisation)
 *     q   = Wp.h2                                (the policy)
 *     v   = tanh(Wv.h2 + bv)                     (the value: ONE linear layer)
 *
 * and over a 250-generation run it collapsed: the value MSE went from 0.345x
 * the variance-of-the-target baseline to 0.838x, and the trunk's gradient norm
 * grew 16x against a clip of 4, so the trunk spent the whole second half of the
 * run permanently clipped.  Three things were wrong with the shape, and this
 * file fixes all three:
 *
 *   1. NO NORMALISATION.  Nothing bounded the scale of a pre-activation, so as
 *      the weights grew the activations and hence the gradients grew with them.
 *      Every nonlinearity is now preceded by LayerNorm (gain g, bias c).  Under
 *      LayerNorm the layer's output is invariant to the scale of the weight
 *      matrix that feeds it, which is exactly the property that stops gradient
 *      norm tracking weight norm.
 *
 *   2. NO SKIP CONNECTION.  Every gradient reaching W0 had to pass through two
 *      relu gates, and a closed gate transmits nothing.  The hidden width now
 *      MATCHES the accumulator width so the second layer is a real residual
 *      block, h2 = relu(norm(W1.h1 + b1)) + h1, and the identity path reaches
 *      the first layer whatever the gates are doing.
 *
 *   3. A ONE-LAYER VALUE HEAD.  Wv was 96 numbers reading a representation that
 *      the policy loss alone was shaping; the only thing it could do was pick a
 *      direction in someone else's feature space.  The value head now has its
 *      OWN SPARSE TRUNK and its own hidden layer:
 *
 *          av  = W0v.x + b0v            (the same ~35-row gather, 32 wide)
 *          hv0 = relu(norm(av))
 *          hv  = relu(norm(Wvh.hv0 + bvh))
 *          v   = tanh(Wv.hv + bv)
 *
 *      so no parameter on the value path is touched by the policy loss at all.
 *      Feeding the value head from the shared h2 instead -- the smaller change,
 *      also implemented and measured -- recovers about three quarters of the
 *      benefit at the best point of the run (0.7184 -> 0.7130 against 0.7111)
 *      and essentially none of it by the end (0.8019 -> 0.8002 against 0.7457):
 *      a value head reading the shared representation keeps drifting with it.
 *
 * ---------------------------------------------------------------------------
 * WHAT THE WIDTHS ARE, AND WHY THEY SHRANK
 * ---------------------------------------------------------------------------
 * The residual forces NF_HID == NF_ACC, which at the old NF_ACC = 160 turns the
 * second layer from 160x96 into 160x160 and costs 41% of the evaluation rate --
 * a straight loss of MCTS simulations per second.  Measured at 128 / 160 / 192
 * / 224, the accumulator is on the flat part of the quality curve and the steep
 * part of the cost curve, so it is now 128: that keeps the mandated residual
 * shape, uses FEWER parameters in the shared trunk than the old design, and
 * gives back most of the throughput.
 *
 * NF_VACC is 32 for the same reason.  The value trunk's width is a pure
 * regularisation knob here -- measured at 16 / 24 / 32 / 64, the held-out value
 * error rises monotonically with it (0.707 / 0.717 / 0.719 / 0.746), because the
 * wider it is the faster it memorises which GAME a position came from, and every
 * position of a game carries that game's single result.  32 is the widest
 * setting statistically tied with the best on the early-stopped number -- the one
 * that measures learning rather than resistance to over-training.  16 was better
 * still at the end of the run and is a one-constant change if a full run
 * confirms it at 20x the data.
 *
 * MEASURED, on 4200 self-play games from the gen-243 checkpoint, held out BY
 * GAME, single pass, 3 seeds (see the report for the full table):
 *
 *     held-out value MSE / Var(z)   0.802 -> 0.719   (R^2 0.198 -> 0.281)
 *     single-core evaluations/sec   234171 -> 186359 (-20%)
 *     parameters (trunk + 1 head)   153793 -> 157473
 *
 * and the gradient-norm growth that motivated all of this is gone at the root.
 * Multiply the trained W1 by 16 and re-measure: the old shape moved the loss
 * from 3.37 to 16.04, multiplied the trunk gradient norm by 33.3 and drove
 * mean |v| from 0.29 to 0.92 (a saturated value head); this one moves the loss
 * by 0.001%, the trunk gradient norm by -0.04% and mean |v| not at all, while
 * |dL/dW1| falls by 15.9x -- a big weight can no longer produce a big gradient.
 */
#ifndef NET_H
#define NET_H

#include <stdint.h>
#include <stdio.h>
#include "chess.h"

/* ------------------------------------------------------- feature encoding */
/*  0..383  : our    pieces, index = ptype*64 + sq_rel
 *  384..767: their  pieces, index = 384 + ptype*64 + sq_rel
 *  768..771: castling rights  (our K, our Q, their K, their Q)
 *  772..779: en-passant file (only when an ep square exists)
 *  780..787: halfmove-clock bucket, min(7, halfmove/13)
 * All squares are "side-to-move relative": mirrored vertically when black moves,
 * so the network always reasons as if it were White.                        */
#define NF_INPUT      788
#define NF_MAXACTIVE  40      /* 32 pieces + 4 castle + 1 ep + 1 clock bucket */

#define NF_ACC   128          /* accumulator width                            */
#define NF_HID   128          /* hidden width -- EQUAL to NF_ACC, see below    */
#define NF_PDIM   32          /* policy embedding width                       */
#define NF_VACC   32          /* the value trunk's own accumulator width      */
#define NF_VHID   32          /* the value head's hidden width                */

/* The residual is h2 = relu(norm(W1.h1 + b1)) + h1, so the two widths must
 * agree.  This is not an accident of tuning: it is what makes the block a skip
 * connection rather than a projection. */
_Static_assert(NF_HID == NF_ACC, "the residual block requires NF_HID == NF_ACC");

/* --------------------------------------------------------------- weights */

/* Shared across the whole population.
 *
 *     acc = b0 + z + sum of the W0 rows named by the active features
 *     h1  = relu(g0 * norm(acc) + c0)
 *     z2  = W1.h1 + b1
 *     h2  = relu(g1 * norm(z2) + c1) + h1                      <- residual
 */
typedef struct {
    float W0[NF_INPUT * NF_ACC];
    float b0[NF_ACC];
    float g0[NF_ACC], c0[NF_ACC];     /* LayerNorm 1: gain and bias           */
    float W1[NF_ACC * NF_HID];
    float b1[NF_HID];
    float g1[NF_HID], c1[NF_HID];     /* LayerNorm 2                          */
    /* The VALUE TRUNK: its own sparse first layer over the same features.
     *     av  = b0v + sum of the W0v rows named by the active features
     *     hv0 = relu(g0v * norm(av) + c0v)
     * Shared across the population like the rest of the trunk, and reached by
     * the value loss and by nothing else. */
    float W0v[NF_INPUT * NF_VACC];
    float b0v[NF_VACC];
    float g0v[NF_VACC], c0v[NF_VACC]; /* LayerNorm 3                          */
} Trunk;

/* One per agent.
 *
 *     q   = Wp.h2                                     (policy query)
 *     hv  = relu(gv * norm(Wvh.hv0 + bvh) + cv)       (value hidden layer)
 *     v   = tanh(Wv.hv + bv)
 */
typedef struct {
    float z[NF_ACC];                  /* style vector, added into accumulator */
    float Wvh[NF_VACC * NF_VHID];     /* value head's hidden layer            */
    float bvh[NF_VHID];
    float gv[NF_VHID], cv[NF_VHID];   /* LayerNorm 3                          */
    float Wv[NF_VHID];                /* value head's output layer            */
    float bv[1];
    float Wp[NF_HID * NF_PDIM];       /* h2 -> policy query q                 */
    float Efrom[64 * NF_PDIM];
    float Eto[64 * NF_PDIM];
    float Epc[6 * NF_PDIM];           /* moving piece type                    */
    float Epromo[5 * NF_PDIM];        /* 0 = none, 1..4 = N,B,R,Q             */
    float Ecap[7 * NF_PDIM];          /* 0 = none, 1..6 = captured type + 1   */
    float Bft[64 * 64];               /* scalar from->to bias                 */
} Head;

#define TRUNK_NPARAM (sizeof(Trunk) / sizeof(float))
#define HEAD_NPARAM  (sizeof(Head)  / sizeof(float))

/* The head's tensor layout, exported because more than one caller needs to walk
 * the head tensor by tensor (PBT's per-tensor RMS-scaled mutation in az.c, the
 * per-tensor gradient telemetry in train.c).  Keeping it here means adding a
 * tensor to Head cannot silently leave a caller's private copy stale.
 *
 * NOTE TO az.c AND train.c: both of you currently keep a PRIVATE `HEAD_TENSORS`
 * table built with a local TSPAN() macro.  Those tables are now wrong in two
 * ways -- they are missing Wvh/bvh/gv/cv entirely, and their `Wv` entry still
 * says NF_HID when Wv is NF_VHID long, so its span runs off the end of Wv and
 * over bv and the front of Wp.  Delete the private tables and walk
 * NN_HEAD_TENSORS instead; the shapes are identical (offset in floats, length
 * in floats) so `HEAD_TENSORS[t].off/.len` becomes `NN_HEAD_TENSORS[t].off/.len`
 * and `N_HEAD_TENSORS` becomes `NN_HEAD_NTENSORS`. */
typedef struct { size_t off, len; } NnTensorSpan;
extern const NnTensorSpan NN_HEAD_TENSORS[];
extern const int NN_HEAD_NTENSORS;

/* Per-agent PBT hyper-parameters (mutated by evolution). */
typedef struct {
    float temperature;   /* softmax temperature used when sampling moves      */
    float entropy_coef;  /* entropy bonus weight                              */
    float lr_scale;      /* multiplier on the base learning rate              */
    float shaping;       /* weight of potential-based material shaping        */
    float value_coef;    /* value-loss weight                                 */
    float gamma;         /* discount                                          */
    float lambda;        /* GAE lambda                                        */
    float mutate_sigma;  /* std-dev used when this agent is cloned            */
} Hyper;

void hyper_default(Hyper *h);

/* --------------------------------------------------------------- forward */

/* Everything nn_backward needs and nothing it does not.  The normalised values
 * (xhat) and the reciprocal standard deviations are kept rather than the raw
 * pre-activations because that is what the LayerNorm backward pass consumes;
 * the relu gates are recovered from h1, from h2 - h1, and from hv. */
typedef struct {
    float x1[NF_ACC];      /* (acc - mean) / sigma                            */
    float h1[NF_ACC];      /* relu(g0 * x1 + c0)                              */
    float x2[NF_HID];      /* (z2 - mean) / sigma                             */
    float h2[NF_HID];      /* relu(g1 * x2 + c1) + h1                         */
    float xv0[NF_VACC];    /* (av - mean) / sigma,  the value trunk           */
    float hv0[NF_VACC];    /* relu(g0v * xv0 + c0v)                           */
    float xv[NF_VHID];     /* (zv - mean) / sigma                             */
    float hv[NF_VHID];     /* relu(gv * xv + cv)                              */
    float q[NF_PDIM];      /* Wp.h2                                           */
    float r1, r2, rv0, rv; /* 1/sigma of each LayerNorm                       */
    float raw_v;           /* Wv.hv + bv                                      */
    float v;               /* tanh(raw_v)  in [-1,1], from side-to-move's view */
} Fwd;

/* Move descriptor in network space (side-to-move relative). */
typedef struct {
    uint8_t from, to, pc, promo, cap;
} MoveKey;

/* Extract active feature indices.  Returns the count (<= NF_MAXACTIVE). */
int  nn_features(const Position *p, uint16_t *idx);
/* Build the network-space key for a legal move of `p`. */
void nn_move_key(const Position *p, Move m, MoveKey *k);

void nn_init(Trunk *t, Head *h, uint64_t seed);
void nn_eval(const Trunk *t, const Head *h, const uint16_t *fidx, int nf, Fwd *fw);
/* Logits for n moves.  `keys` may be NULL, in which case it is derived from p. */
void nn_logits(const Head *h, const Fwd *fw, const MoveKey *keys, int n, float *logits);

/* ------------------------------------------------------- batched forward */
/* `nbatch` independent positions at once, each against ITS OWN head -- different
 * agents play different games, so nothing here assumes one head.
 *
 *   heads[b]  the agent evaluating row b.  Rows that share a head are batched
 *             together when they are ADJACENT, so a caller that groups its batch
 *             by agent gets the fastest path (worth ~25% at batch 32); one that
 *             does not still gets a batched W1, which is 60% of the arithmetic.
 *   fidx[b]   row b's active features, nf[b] of them.  An array of pointers so
 *             the caller never has to copy ragged feature lists into a
 *             rectangular buffer.
 *   out       an array of `nbatch` Fwd.  It is the output AND the working
 *             storage: h1/h2/hv0 are read back out of it as GEMM inputs at
 *             their natural stride, so a batched evaluation copies nothing and
 *             needs no workspace argument.
 *
 * Each row of the result is bit-identical to what nn_eval() produces for that
 * row at every batch size, INCLUDING 1 -- the batched kernels accumulate each
 * output in the same lanes in the same order.  Built with -DUSE_ACCELERATE the
 * two trunk matrices go through cblas_sgemm instead, which regroups the sums:
 * rows then agree with nn_eval to about 1e-6 relative rather than exactly, and
 * the summation order becomes a function of the batch size.  tests/test_net.c
 * checks whichever of the two guarantees the build is making.
 *
 * Internally the batch is processed in tiles of 32; any nbatch is allowed. */
void nn_eval_batch(const Trunk *t, const Head *const *heads, int nbatch,
                   const uint16_t *const *fidx, const int *nf, Fwd *out);

/* Which kernels this build actually compiled -- "scalar", "neon",
 * "neon+accelerate" -- and the batch size the tiling uses internally.  Worth
 * logging next to any throughput number, since the three differ by 3x. */
const char *nn_backend(void);
int  nn_batch_tile(void);
/* 1 when nn_eval_batch at this batch size is bit-identical to nn_eval, 0 when
 * the batched dense layers go through cblas_sgemm and are therefore only equal
 * to ~1e-6.  Always 1 without -DUSE_ACCELERATE, and always 1 for small batches
 * even with it, because sgemm loses to the NEON kernel there anyway.  A caller
 * that needs a bit-reproducible search should ask this. */
int  nn_batch_is_exact(int nbatch);

/* --------------------------------------------------------------- backward */
/* Gradients use exactly the same layout as the weights. */
typedef Trunk TrunkGrad;
typedef Head  HeadGrad;

/* Accumulates dLoss/dTheta into tg/hg.  `dlogits` is dLoss/dlogit for each of
 * the n moves; `dvalue` is dLoss/dv (w.r.t. the post-tanh value).            */
void nn_backward(const Trunk *t, const Head *h, const Fwd *fw,
                 const uint16_t *fidx, int nf,
                 const MoveKey *keys, int n, const float *dlogits, float dvalue,
                 TrunkGrad *tg, HeadGrad *hg);

/* ------------------------------------------------------------- optimiser */

typedef struct { float *m, *v; int n; int t; } Adam;

void adam_init(Adam *a, int n);
void adam_free(Adam *a);
/* Adam with decoupled weight decay; clips the global grad norm to `clip`
 * (clip <= 0 disables).  Zeroes `g` afterwards. */
void adam_step(Adam *a, float *p, float *g, float lr, float wd, float clip);

void grad_zero(void *g, int nfloats);
void grad_add(float *dst, const float *src, int n);   /* dst += src          */

/* ---------------------------------------------------------- serialisation */

typedef struct {
    uint32_t magic, version;
    uint32_t n_agents, generation;
    uint32_t nf_input, nf_acc, nf_hid, nf_pdim;
} ModelHeader;

#define MODEL_MAGIC   0x43524C31u   /* "CRL1" */
/* 4: LayerNorm, residual trunk, two-layer value head.  A version-3 checkpoint
 * has different tensors in a different order and would be read as noise, so
 * model_load rejects it on the version field rather than on the widths. */
#define MODEL_VERSION 4u

int  model_save(const char *path, const Trunk *t, const Head *heads,
                const Hyper *hy, const float *elo, int n_agents, int generation);
/* Loads into caller-provided buffers.  *n_agents is in/out (capacity/actual). */
int  model_load(const char *path, Trunk *t, Head *heads, Hyper *hy, float *elo,
                int *n_agents, int *generation);

/* --------------------------------------------------------------- helpers */

/* Softmax over `n` logits with temperature; writes probabilities into out. */
void softmax_t(const float *logits, int n, float temp, float *out);
/* Material balance from the side-to-move's perspective, in pawns. */
float material_balance(const Position *p);

#endif /* NET_H */
