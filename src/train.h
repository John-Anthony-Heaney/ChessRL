/* train.h -- population-based reinforcement learning driver. */
#ifndef TRAIN_H
#define TRAIN_H

#include "net.h"
#include "arena.h"

typedef struct {
    int    n_agents;          /* agents per generation                        */
    int    generations;
    int    games_per_agent;   /* games each agent plays per generation        */
    int    threads;
    int    max_plies;
    float  base_lr;
    float  trunk_lr;
    float  weight_decay;
    float  grad_clip;
    float  elite_frac;        /* top fraction that may reproduce              */
    float  cull_frac;         /* bottom fraction replaced each generation     */
    int    hof_every;         /* snapshot champion into hall of fame every N   */
    int    hof_frac_pct;      /* % of games played against hall-of-fame agents */
    uint64_t seed;
    const char *run_dir;
    int    eval_games;        /* champion-vs-champion evaluation games        */
    int    quiet;
} TrainCfg;

void train_default_cfg(TrainCfg *c);
int  train_run(TrainCfg *c);

#endif /* TRAIN_H */
