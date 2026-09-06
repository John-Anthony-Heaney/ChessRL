/* net.h -- the policy/value network, its gradients, and the optimiser.
 *
 * Design notes (why it looks like this)
 * ------------------------------------
 * The whole system must sustain >1000 complete games/second on an 8-core laptop,
 * which is ~120k position evaluations/second.  That gives a budget of roughly
 * 50 kFLOP per evaluation, so the network is deliberately small and, crucially,
 * its first (and by far largest) layer consumes a SPARSE binary feature vector:
 * only ~35 of 788 inputs are ever 1, so "W0 @ x" is a gather-and-sum of ~35 rows
 * rather than a 788x160 matmul.  This is the NNUE trick, minus the incremental
 * accumulator (unnecessary here -- a full sparse refresh is already cheap).
 *
 * The population shares one trunk (W0/W1) and gives every agent its own head
 * plus a "style vector" z that is added into the accumulator.  Under an extremely
 * limited compute budget this is what makes the run work: the trunk sees every
 * position played by all 256 agents (tens of millions of positions), while the
 * per-agent heads (12k parameters each) still give evolution something real to
 * select on and keep agents genuinely distinct opponents.
 *
 * Everything is float32 and framework-free so it runs with zero dependencies.
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

#define NF_ACC   160          /* accumulator width                            */
#define NF_HID    96          /* hidden width                                 */
#define NF_PDIM   32          /* policy embedding width                       */

/* --------------------------------------------------------------- weights */

typedef struct {              /* shared across the whole population          */
    float W0[NF_INPUT * NF_ACC];
    float b0[NF_ACC];
    float W1[NF_ACC * NF_HID];
    float b1[NF_HID];
} Trunk;

typedef struct {              /* one per agent                               */
    float z[NF_ACC];                  /* style vector, added into accumulator */
    float Wv[NF_HID];                 /* value head                           */
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

typedef struct {
    float acc[NF_ACC];    /* W0.x + b0 + z  (pre-activation)                  */
    float h1[NF_ACC];     /* relu(acc)                                        */
    float z2[NF_HID];     /* W1.h1 + b1                                       */
    float h2[NF_HID];     /* relu(z2)                                         */
    float q[NF_PDIM];     /* Wp.h2                                            */
    float raw_v;          /* Wv.h2 + bv                                       */
    float v;              /* tanh(raw_v)  in [-1,1], from side-to-move's view  */
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
#define MODEL_VERSION 3u

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
