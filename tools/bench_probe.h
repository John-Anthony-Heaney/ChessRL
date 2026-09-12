/* bench_probe.h -- phase timers for tools/bench_throughput.c.
 *
 * WHY THIS FILE EXISTS AT ALL
 * ---------------------------
 * The question "where does self-play time actually go?" cannot be answered
 * from outside mcts_search(): one call does move generation, network
 * evaluation and tree bookkeeping, and only the sum is visible to a caller.
 * The obvious fix -- put timers inside src/mcts.c -- is not available here:
 * src/ belongs to another agent and must not be edited.
 *
 * So the timers are injected at COMPILE TIME instead, without changing a byte
 * of src/.  The `bench-throughput` Makefile target compiles a SECOND copy of
 * src/mcts.c into build/probe/mcts.o with
 *
 *     -DBENCH_PROBE -include tools/bench_probe.h
 *
 * and this header defines function-like macros that rename the primitives at
 * their CALL SITES inside that translation unit.  `nn_eval(...)` inside
 * mcts.c becomes `bp_nn_eval(...)`, a static inline that reads the clock,
 * calls the real nn_eval, and accumulates the delta into a thread-local
 * counter.  The normal build never sees this header and pays nothing --
 * not a branch, not a counter increment.
 *
 * chess.h and net.h are included FIRST and on purpose: their include guards
 * mean mcts.c's own `#include "net.h"` is a no-op afterwards, so the real
 * declarations (`void nn_eval(const Trunk *t, ...)`) are parsed BEFORE the
 * macros exist and are not themselves mangled by them.  Each wrapper body is
 * likewise written before the macro that renames the call, so the wrapper
 * calls the real function rather than itself.
 *
 * WHAT IS TIMED AND WHAT IS ONLY COUNTED
 * --------------------------------------
 * clock_gettime(CLOCK_MONOTONIC) costs ~20-25 ns a call on this machine (the
 * harness measures it and prints it).  That is a rounding error against
 * nn_eval (~5-11 us) and gen_legal (~0.5-1 us), and a catastrophe against
 * make_move (~40 ns), which would be inflated by more than 100%.
 *
 * So the split is measured two ways, and both are reported:
 *
 *   TIMED    nn_eval, nn_logits, nn_features, gen_legal -- the coarse calls.
 *            Two clock reads each; overhead is subtracted using the measured
 *            per-read cost and the exact call count.
 *   COUNTED  make_move, unmake_move, in_check, insufficient_material,
 *            nn_move_key, softmax_t -- one thread-local increment each, which
 *            is free.  Their cost is attributed as (calls x ns/call) with
 *            ns/call taken from `--mode micro`, which times each of them in a
 *            tight loop.
 *
 * -DBENCH_PROBE_FINE times the COUNTED group directly as well, as a
 * cross-check on the multiplication.  It distorts the total; the harness says
 * by how much.
 */
#ifndef BENCH_PROBE_H
#define BENCH_PROBE_H

#include <stdint.h>
#include <time.h>

#include "chess.h"
#include "net.h"

/* Phase buckets.  Everything inside mcts_search() that is not attributed to
 * NET, ENCODE or RULES is MCTS bookkeeping by subtraction: selection (the
 * PUCT argmax), backup, node allocation, the evaluation cache, the subtree
 * reuse checks and the weight stamp. */
enum {
    BP_EVAL = 0,    /* nn_eval            : the forward pass                 */
    BP_LOGITS,      /* nn_logits          : priors for a node's move list    */
    BP_FEATURES,    /* nn_features        : position -> sparse index list    */
    BP_GENLEGAL,    /* gen_legal          : legal move generation            */
    BP_MAKE,        /* make_move                                             */
    BP_UNMAKE,      /* unmake_move                                           */
    BP_INCHECK,     /* in_check                                              */
    BP_INSUF,       /* insufficient_material                                 */
    BP_MOVEKEY,     /* nn_move_key        : move -> network-space key        */
    BP_SOFTMAX,     /* softmax_t                                             */
    BP_SEARCH,      /* the whole mcts_search() call, timed by the driver     */
    BP_DRIVER,      /* the self-play game loop outside mcts_search()         */
    BP_LEARN,       /* the replay-buffer learning step                       */
    BP_LEARN_FWD,   /* nn_eval+nn_logits+softmax in learning, inside BP_LEARN */
    BP_BACKWARD,    /* nn_backward, inside BP_LEARN                          */
    BP_N
};

extern _Thread_local uint64_t bp_ns[BP_N];
extern _Thread_local uint64_t bp_cnt[BP_N];

extern const char *bp_name(int b);

static inline uint64_t bp_now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

/* Scope timers for regions the harness itself owns (search, driver, learn). */
#define BP_T0(tag)      const uint64_t bp_t0_##tag = bp_now()
#define BP_T1(tag, b)   do { bp_ns[b] += bp_now() - bp_t0_##tag; bp_cnt[b]++; } while (0)

/* ---------------------------------------------------------- timed wrappers */

static inline void bp_nn_eval(const Trunk *t, const Head *h,
                              const uint16_t *fidx, int nf, Fwd *fw)
{
    const uint64_t a = bp_now();
    nn_eval(t, h, fidx, nf, fw);
    bp_ns[BP_EVAL] += bp_now() - a;
    bp_cnt[BP_EVAL]++;
}

static inline void bp_nn_logits(const Head *h, const Fwd *fw,
                                const MoveKey *keys, int n, float *logits)
{
    const uint64_t a = bp_now();
    nn_logits(h, fw, keys, n, logits);
    bp_ns[BP_LOGITS] += bp_now() - a;
    bp_cnt[BP_LOGITS]++;
}

static inline int bp_nn_features(const Position *p, uint16_t *idx)
{
    const uint64_t a = bp_now();
    const int r = nn_features(p, idx);
    bp_ns[BP_FEATURES] += bp_now() - a;
    bp_cnt[BP_FEATURES]++;
    return r;
}

static inline int bp_gen_legal(const Position *p, Move *out)
{
    const uint64_t a = bp_now();
    const int r = gen_legal(p, out);
    bp_ns[BP_GENLEGAL] += bp_now() - a;
    bp_cnt[BP_GENLEGAL]++;
    return r;
}

static inline void bp_nn_backward(const Trunk *t, const Head *h, const Fwd *fw,
                                  const uint16_t *fidx, int nf,
                                  const MoveKey *keys, int n,
                                  const float *dlogits, float dvalue,
                                  TrunkGrad *tg, HeadGrad *hg)
{
    const uint64_t a = bp_now();
    nn_backward(t, h, fw, fidx, nf, keys, n, dlogits, dvalue, tg, hg);
    bp_ns[BP_BACKWARD] += bp_now() - a;
    bp_cnt[BP_BACKWARD]++;
}

/* -------------------------------------------- counted (optionally timed) */

#ifdef BENCH_PROBE_FINE
#define BP_CHEAP(b, call)  do { const uint64_t a_ = bp_now(); call;            \
                                bp_ns[b] += bp_now() - a_; bp_cnt[b]++; } while (0)
#define BP_CHEAP_R(b, ty, call)                                               \
    ty r_; do { const uint64_t a_ = bp_now(); r_ = (call);                     \
                bp_ns[b] += bp_now() - a_; bp_cnt[b]++; } while (0); return r_
#else
#define BP_CHEAP(b, call)  do { bp_cnt[b]++; call; } while (0)
#define BP_CHEAP_R(b, ty, call)  do { bp_cnt[b]++; } while (0); return (call)
#endif

static inline void bp_make_move(Position *p, Move m, Undo *u)
{ BP_CHEAP(BP_MAKE, make_move(p, m, u)); }

static inline void bp_unmake_move(Position *p, Move m, const Undo *u)
{ BP_CHEAP(BP_UNMAKE, unmake_move(p, m, u)); }

static inline int bp_in_check(const Position *p, int c)
{ BP_CHEAP_R(BP_INCHECK, int, in_check(p, c)); }

static inline int bp_insufficient_material(const Position *p)
{ BP_CHEAP_R(BP_INSUF, int, insufficient_material(p)); }

static inline void bp_nn_move_key(const Position *p, Move m, MoveKey *k)
{ BP_CHEAP(BP_MOVEKEY, nn_move_key(p, m, k)); }

static inline void bp_softmax_t(const float *l, int n, float temp, float *out)
{ BP_CHEAP(BP_SOFTMAX, softmax_t(l, n, temp, out)); }

/* ------------------------------------------------------- the interposition */
/* From here on, every CALL to one of these names in this translation unit is
 * a call to the wrapper.  Definitions and declarations are already parsed. */

#define nn_eval(t, h, f, nf, fw)          bp_nn_eval((t), (h), (f), (nf), (fw))
#define nn_logits(h, fw, k, n, l)         bp_nn_logits((h), (fw), (k), (n), (l))
#define nn_features(p, i)                 bp_nn_features((p), (i))
#define gen_legal(p, o)                   bp_gen_legal((p), (o))
#define make_move(p, m, u)                bp_make_move((p), (m), (u))
#define unmake_move(p, m, u)              bp_unmake_move((p), (m), (u))
#define in_check(p, c)                    bp_in_check((p), (c))
#define insufficient_material(p)          bp_insufficient_material((p))
#define nn_move_key(p, m, k)              bp_nn_move_key((p), (m), (k))
#define softmax_t(l, n, t, o)             bp_softmax_t((l), (n), (t), (o))
#define nn_backward(t, h, fw, f, nf, k, n, dl, dv, tg, hg)                     \
    bp_nn_backward((t), (h), (fw), (f), (nf), (k), (n), (dl), (dv), (tg), (hg))

#endif /* BENCH_PROBE_H */
