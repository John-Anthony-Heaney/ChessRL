/* arena.h -- self-play game generation and trajectory recording. */
#ifndef ARENA_H
#define ARENA_H

#include "chess.h"
#include "net.h"

/* Per-ply record kept while a game is played, consumed by the learner. */
typedef struct {
    int      ply;                                  /* number of recorded plies */
    uint16_t fidx[MAX_GAME_PLIES][NF_MAXACTIVE];   /* active input features    */
    uint8_t  nf[MAX_GAME_PLIES];
    uint8_t  mover[MAX_GAME_PLIES];                /* WHITE / BLACK            */
    int32_t  koff[MAX_GAME_PLIES];                 /* offset into keys[]       */
    uint16_t nmoves[MAX_GAME_PLIES];
    uint16_t chosen[MAX_GAME_PLIES];               /* index into that move set */
    float    v[MAX_GAME_PLIES];                    /* value prediction         */
    float    phi[MAX_GAME_PLIES];                  /* shaping potential, WHITE view */
    MoveKey *keys;
    int      nkeys, keycap;
} Traj;

void traj_init(Traj *t);
void traj_reset(Traj *t);
void traj_free(Traj *t);

/* Statistics accumulated over a batch of games (used for the strategy report). */
typedef struct {
    uint64_t games, plies;
    uint64_t white_wins, black_wins, draws;
    uint64_t checkmates, stalemates, fifty, repetition, insufficient, maxplies;
    uint64_t captures, checks, castles_k, castles_q, promotions, ep_captures;
    uint64_t first_move[64 * 64];      /* histogram of White's opening move    */
    uint64_t piece_dest[6][64];        /* where each piece type ends up moving */
    double   sum_len, sum_final_material;
} PlayStats;

void stats_zero(PlayStats *s);
void stats_merge(PlayStats *dst, const PlayStats *src);

typedef struct {
    int      max_plies;       /* hard cap; game adjudicated as draw when hit   */
    int      greedy;          /* 1 = argmax instead of sampling                */
    float    temp_white, temp_black;
    float    opening_temp;    /* temperature for the first `opening_plies`     */
    int      opening_plies;
    int      record;          /* 1 = fill trajectories                         */
    uint64_t *rng;            /* xoshiro state (4 x uint64)                    */
} PlayCfg;

/* Play one complete game.  Trajectories may be NULL when `record` is 0.
 * Returns GR_*; `reason` receives TR_*. */
int play_game(const Trunk *trunk, const Head *hw, const Head *hb,
              const Hyper *yw, const Hyper *yb,
              const PlayCfg *cfg, Traj *tw, Traj *tb,
              PlayStats *stats, Game *game_out, int *reason);

/* xoshiro256** */
void     rng_seed(uint64_t *s, uint64_t seed);
uint64_t rng_next(uint64_t *s);
float    rng_float(uint64_t *s);        /* [0,1) */
float    rng_normal(uint64_t *s);

#endif /* ARENA_H */
