/* search.h -- alpha-beta search used when a human plays the champion.
 * Training never calls this; it exists purely to make the final opponent strong. */
#ifndef SEARCH_H
#define SEARCH_H

#include "chess.h"
#include "net.h"

typedef struct TTEntry {
    uint64_t key;
    int16_t  score;
    uint8_t  depth, flag;   /* 0 exact, 1 lower, 2 upper */
    Move     best;
} TTEntry;

typedef struct {
    const Trunk *trunk;
    const Head  *head;
    int          max_depth;
    int          movetime_ms;
    uint64_t     max_nodes;
    float        blunder_rate;   /* 0 = perfect play, >0 picks a worse move sometimes */
    uint64_t     rng[4];
    /* outputs */
    uint64_t     nodes;
    int          depth_reached;
    int          score_cp;       /* from side-to-move's view, "centipawn-ish"  */
    Move         pv[64];
    int          pv_len;
    /* internal */
    TTEntry     *tt;
    size_t       tt_size;
    int          stop;
} Search;

void search_init(Search *s, const Trunk *t, const Head *h, size_t tt_mb);
void search_free(Search *s);
/* Returns the chosen move for g->pos (MV_NONE when the game is over). */
Move search_best(Search *s, const Game *g);
/* Static network evaluation, side-to-move view, in "centipawns". */
int  search_eval_cp(Search *s, const Position *p);

#endif /* SEARCH_H */
