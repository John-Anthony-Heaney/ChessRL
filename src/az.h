/* az.h -- AlphaZero-style training: self-play with MCTS, a replay buffer, and
 * minibatch gradient descent on (policy target, game outcome).
 *
 * Differences from the previous A2C trainer, and why they matter:
 *  - The reward is ONLY the game result. No material shaping anywhere.
 *  - The policy target is the MCTS visit distribution, which is a far stronger
 *    learning signal than a REINFORCE gradient: search improves on the policy,
 *    and the network is trained to imitate the improvement.
 *  - Positions go into a replay buffer and the network takes MANY minibatch
 *    steps per generation. The old loop took exactly one optimiser step per
 *    generation, which is why 300 generations produced only 300 updates.
 */
#ifndef AZ_H
#define AZ_H

#include <stdint.h>
#include "net.h"

enum { AZ_START_CLASSICAL = 0, AZ_START_960 = 1, AZ_START_MIXED = 2 };

typedef struct {
    /* population */
    int   n_agents;
    int   generations;
    int   games_per_agent;
    int   threads;
    /* CONCURRENT GAMES PER THREAD.  MCTS is sequential, so one game in flight
     * can only ever ask the network for one position at a time and every
     * evaluation is a batch of one.  G games per thread are stepped together
     * and their pending leaves evaluated in a single batched call.  The games
     * are independent -- own node pool, own tree, own repetition history, own
     * rng stream -- so this is pure throughput: no virtual loss, and every
     * game plays exactly the game it would have played alone.  1 reproduces
     * the old one-game-at-a-time loop bit for bit.                           */
    int   games_in_flight;

    /* self-play */
    /* Where games start.  Training across the 960 Chess960 arrays stops the
     * agent memorising patterns tied to one opening array: whatever it learns
     * has to transfer, because it never sees the same start twice in a row. */
    int   start_mode;        /* AZ_START_*                                    */
    int   sims;              /* MCTS simulations per move                     */
    int   max_plies;         /* adjudicate a draw at this many plies          */
    int   opening_plies;     /* plies played at temp_start before temp_end    */
    float temp_start;        /* sampling temperature early in the game        */
    float temp_end;          /* sampling temperature after opening_plies      */
    float c_puct;
    float dirichlet_alpha;
    float dirichlet_eps;
    float resign_threshold;  /* value below which a game may be resigned; use
                              * -1.0 to disable. Speeds up self-play a lot.   */
    float resign_check_frac; /* fraction of games played out anyway, so the
                              * resign threshold can be validated             */

    /* draws. A draw penalty is symmetric: it makes a draw worth less than an
     * even chance at a win for BOTH sides. It encodes no chess knowledge, it
     * only stops the population settling into the shuffling equilibrium. */
    float draw_penalty;      /* value assigned to a draw, e.g. -0.1           */

    /* learning */
    int   buffer_positions;  /* replay buffer capacity                        */
    int   batch_size;
    int   steps_per_gen;     /* minibatch updates per generation              */
    float lr;
    float lr_final;          /* cosine-decayed to this by the last generation */
    float weight_decay;
    float grad_clip;         /* global grad-norm clip on the shared trunk     */
    float grad_clip_head;    /* same for a per-agent head; <=0 means "reuse
                              * grad_clip", which is the historical behaviour */
    float value_coef;        /* weight of the value loss vs the policy loss   */
    int   warmup_gens;       /* linear LR warmup before the cosine decay      */

    /* VALUE TARGET MIXING (KataGo / Leela).  The Monte-Carlo outcome z gives
     * every position in a game the same label, which is an unbiased but very
     * high-variance target: a 140-ply game contributes 140 copies of one coin
     * flip.  q_search is the MCTS root value already computed at that exact
     * position, so
     *
     *     target = (1 - value_mix) * z  +  value_mix * q_search
     *
     * trades a little bias for a lot of variance.  It is NOT injected
     * knowledge: q_search is this network's own search over its own value
     * head, i.e. bootstrapping from self-play, which docs/FROM_SCRATCH.md
     * permits in the same way it permits the MCTS visit distribution as the
     * policy target.  Default 0 (pure outcome) so it can be measured.       */
    float value_mix;

    /* WEIGHT EMA (Polyak averaging).  An exponential moving average of the
     * weights is kept alongside the live ones and saved as a second model.
     * decay <= 0 disables it.                                               */
    float ema_decay;
    int   ema_h2h_games;     /* raw-vs-EMA games played before best.crl is
                              * written; 0 = never promote the EMA           */

    /* PLAYOUT CAP RANDOMISATION (KataGo).  Only `cap_frac` of moves run the
     * full `sims` budget with root noise and get recorded as training data;
     * the rest run `cap_sims` and are played but not learned from.  Most of
     * self-play's cost buys policy targets that are thrown away by the
     * temperature anyway, so this is throughput for free.
     * cap_frac >= 1 disables it (every move is a full search).              */
    float cap_frac;
    int   cap_sims;          /* 0 = derive as max(2, sims / 5)               */

    /* population dynamics */
    float elite_frac;
    float cull_frac;
    int   hof_every;
    int   hof_frac_pct;

    /* ======================================================================
     *                        THE ANCHORED RATING
     * ======================================================================
     * The incremental Elo this trainer has always reported is not a strength
     * measurement: it drifts upward at a constant rate whether or not the
     * population is learning anything.  docs/RATING.md has the evidence, the
     * mechanism and the acid test.  The fields below configure the
     * replacement: two players of permanently fixed strength whose rating is
     * PINNED, and a Bradley-Terry maximum-likelihood fit over a sliding
     * window of results that re-estimates every rating from scratch each
     * generation instead of accumulating increments.
     *
     * Neither anchor holds any chess knowledge -- one is a uniform random
     * legal mover, the other is the untrained generation-0 network -- so
     * docs/FROM_SCRATCH.md is untouched.                                   */
    int   anchor_elo;        /* 1 = run the anchors and the fit (default 1)  */
    int   anchor_games;      /* pairing slots per generation given to an
                              * anchor.  They REPLACE population games, so
                              * they cost no extra wall-clock time.          */
    int   anchor_cal_games;  /* one-off gen0-vs-random games that pin the
                              * gen-0 anchor's rating                        */
    int   anchor_pin_gen;    /* generation after which gen0's pin is fixed   */
    int   elo_window;        /* generations of results in the fit window     */
    int   elo_iters;         /* maximum minorisation-maximisation sweeps     */
    int   hof_pin_lag;       /* generations a hall-of-fame entry stays a FREE
                              * parameter before its rating is frozen.  Long
                              * enough that the frozen value comes from games
                              * played after the snapshot was selected, so
                              * the winner's curse is not frozen with it.
                              * 0 = use elo_window.                          */

    uint64_t    seed;
    const char *run_dir;
    int         quiet;
} AZCfg;

void az_default_cfg(AZCfg *c);
int  az_run(AZCfg *c);

/* ==========================================================================
 *                    LEARNING-RATE RANGE TEST (Smith 2015)
 * ==========================================================================
 * https://arxiv.org/abs/1506.01186 section 3.3: fill a replay buffer with the
 * real self-play distribution, snapshot the weights, then take `steps`
 * optimiser steps on minibatches drawn from that buffer while the learning
 * rate rises GEOMETRICALLY from `lo` to `hi`, and read the usable range off
 * the loss curve.  The weights are restored afterwards and the restoration is
 * asserted, so the test is read-only with respect to the model.
 *
 * `az` carries the ENTIRE self-play and learning configuration, unchanged, so
 * the data the test learns from is the data training would have produced.
 * az.lr / az.lr_final / az.steps_per_gen / az.generations are ignored: the
 * sweep supplies the learning rate and the step count itself.                */
typedef struct {
    AZCfg       az;          /* self-play + optimiser config, used verbatim   */
    const char *model;       /* checkpoint to test; NULL = fresh nn_init      */
    double      lo, hi;      /* learning-rate sweep bounds, lo < hi           */
    int         steps;       /* optimiser steps across the sweep              */
    int         warm_games;  /* self-play games played to fill the buffer     */
    double      smooth;      /* EMA coefficient for the smoothed loss, [0,1)  */
    double      stop_factor; /* abandon the sweep once the smoothed loss is
                              * this multiple of its best; <=0 disables       */
    const char *csv;         /* per-step CSV output path, NULL = none         */
    int         plot_rows, plot_cols;
} AZLrFindCfg;

void az_lrfind_default_cfg(AZLrFindCfg *c);
int  az_lrfind(AZLrFindCfg *c);

#endif /* AZ_H */
