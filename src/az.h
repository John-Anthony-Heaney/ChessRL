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
    float grad_clip;
    float value_coef;        /* weight of the value loss vs the policy loss   */

    /* population dynamics */
    float elite_frac;
    float cull_frac;
    int   hof_every;
    int   hof_frac_pct;

    uint64_t    seed;
    const char *run_dir;
    int         quiet;
} AZCfg;

void az_default_cfg(AZCfg *c);
int  az_run(AZCfg *c);

#endif /* AZ_H */
