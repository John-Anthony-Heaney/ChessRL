/* arena.c -- self-play game generation, trajectory recording and RNG.
 *
 * play_game() is the throughput bottleneck of the whole system (>1000 complete
 * games/second on 8 cores == ~120k position evaluations/second), so it:
 *   - allocates nothing on the heap (the only exception is growing a
 *     trajectory's key array, which after the first couple of games never
 *     happens again because traj_reset() keeps the capacity),
 *   - keeps the legal-move list, the move keys, the logits and the game itself
 *     on the stack,
 *   - calls gen_legal() exactly once per ply.  That last point is why the game
 *     loop does not use game_push(): game_push() re-derives the result after
 *     every move, which costs a second full legal-move generation.  Instead the
 *     terminal tests are inlined here in exactly the order (and with exactly
 *     the semantics) that game_update_result() uses, and game_update_result()
 *     itself is only called for the mate/stalemate case, once per game.
 */
#include "arena.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* =========================================================== xoshiro256** */

static inline uint64_t rotl64(uint64_t x, int k)
{
    return (x << k) | (x >> (64 - k));
}

static inline uint64_t splitmix64(uint64_t *x)
{
    uint64_t z = (*x += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

void rng_seed(uint64_t *s, uint64_t seed)
{
    uint64_t x = seed;
    s[0] = splitmix64(&x);
    s[1] = splitmix64(&x);
    s[2] = splitmix64(&x);
    s[3] = splitmix64(&x);
    if ((s[0] | s[1] | s[2] | s[3]) == 0) {          /* the one forbidden state */
        s[0] = 0x9E3779B97F4A7C15ULL;
        s[1] = 0xBF58476D1CE4E5B9ULL;
        s[2] = 0x94D049BB133111EBULL;
        s[3] = 0x2545F4914F6CDD1DULL;
    }
}

uint64_t rng_next(uint64_t *s)
{
    const uint64_t result = rotl64(s[1] * 5, 7) * 9;
    const uint64_t t = s[1] << 17;

    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];
    s[2] ^= t;
    s[3] = rotl64(s[3], 45);

    return result;
}

/* 24 random bits scaled by 2^-24: exactly representable, never reaches 1. */
float rng_float(uint64_t *s)
{
    return (float)(rng_next(s) >> 40) * (1.0f / 16777216.0f);
}

/* Marsaglia polar method.  Fully reentrant: no cached second deviate, all
 * state lives in `s`, so it is safe to call from any number of threads with
 * distinct states. */
float rng_normal(uint64_t *s)
{
    float u, v, r;
    do {
        u = 2.0f * rng_float(s) - 1.0f;
        v = 2.0f * rng_float(s) - 1.0f;
        r = u * u + v * v;
    } while (r >= 1.0f || r <= 0.0f);
    return u * sqrtf(-2.0f * logf(r) / r);
}

/* ============================================================ trajectory */

#define TRAJ_KEYS_MIN 4096     /* ~80 recorded plies x ~35 moves, so a typical
                                * game grows the array once, ever.           */

void traj_init(Traj *t)
{
    memset(t, 0, sizeof(*t));
}

/* Keeps the keys[] allocation: this runs once per game per agent. */
void traj_reset(Traj *t)
{
    t->ply   = 0;
    t->nkeys = 0;
}

void traj_free(Traj *t)
{
    free(t->keys);
    t->keys   = NULL;
    t->nkeys  = 0;
    t->keycap = 0;
    t->ply    = 0;
}

/* Returns 0 only when the allocation fails, in which case the caller silently
 * drops the ply rather than corrupting the trajectory. */
static int traj_reserve(Traj *t, int extra)
{
    const int need = t->nkeys + extra;
    if (need <= t->keycap) return 1;

    int cap = t->keycap > 0 ? t->keycap : TRAJ_KEYS_MIN;
    while (cap < need) {
        if (cap > (1 << 28)) { cap = need; break; }
        cap <<= 1;
    }
    MoveKey *nk = (MoveKey *)realloc(t->keys, (size_t)cap * sizeof(MoveKey));
    if (!nk) return 0;
    t->keys   = nk;
    t->keycap = cap;
    return 1;
}

/* ================================================================= stats */

void stats_zero(PlayStats *s)
{
    memset(s, 0, sizeof(*s));
}

void stats_merge(PlayStats *dst, const PlayStats *src)
{
    dst->games        += src->games;
    dst->plies        += src->plies;
    dst->white_wins   += src->white_wins;
    dst->black_wins   += src->black_wins;
    dst->draws        += src->draws;

    dst->checkmates   += src->checkmates;
    dst->stalemates   += src->stalemates;
    dst->fifty        += src->fifty;
    dst->repetition   += src->repetition;
    dst->insufficient += src->insufficient;
    dst->maxplies     += src->maxplies;

    dst->captures     += src->captures;
    dst->checks       += src->checks;
    dst->castles_k    += src->castles_k;
    dst->castles_q    += src->castles_q;
    dst->promotions   += src->promotions;
    dst->ep_captures  += src->ep_captures;

    for (int i = 0; i < 64 * 64; i++) dst->first_move[i] += src->first_move[i];
    for (int p = 0; p < 6; p++)
        for (int sq = 0; sq < 64; sq++) dst->piece_dest[p][sq] += src->piece_dest[p][sq];

    dst->sum_len            += src->sum_len;
    dst->sum_final_material += src->sum_final_material;
}

/* ============================================================== self-play */

/* game_push() minus the redundant result refresh (see the file header). */
static inline void game_push_raw(Game *g, Move m)
{
    g->moves[g->ply] = m;
    make_move(&g->pos, m, &g->undos[g->ply]);
    g->ply++;
    g->hist[g->ply] = g->pos.key;
    g->hist_len     = g->ply + 1;
}

/* Material balance of `p` from WHITE's point of view, in pawns.
 * material_balance() reports the side-to-move's view, hence the flip. */
static inline float white_material(const Position *p)
{
    const float mb = material_balance(p);
    return (p->side == WHITE) ? mb : -mb;
}

int play_game(const Trunk *trunk, const Head *hw, const Head *hb,
              const Hyper *yw, const Hyper *yb,
              const PlayCfg *cfg, Traj *tw, Traj *tb,
              PlayStats *stats, Game *game_out, int *reason)
{
    Game     local;                    /* used only when the caller wants no game */
    Move     list[MAX_MOVES];
    MoveKey  keys[MAX_MOVES];
    float    logits[MAX_MOVES];
    float    probs[MAX_MOVES];
    uint16_t fidx[NF_MAXACTIVE];
    Fwd      fw;
    uint64_t fallback_rng[4];

    Game *g = game_out ? game_out : &local;
    game_start(g);

    uint64_t *rs = cfg->rng;
    if (!rs) {                          /* degenerate config: stay deterministic */
        rng_seed(fallback_rng, 0x5DEECE66DULL);
        rs = fallback_rng;
    }

    int cap = cfg->max_plies;
    if (cap <= 0 || cap > MAX_GAME_PLIES) cap = MAX_GAME_PLIES;

    const int record       = cfg->record;
    const int greedy       = cfg->greedy;
    const int opening_ply  = cfg->opening_plies;
    const float opening_t  = cfg->opening_temp;

    for (;;) {
        Position *p = &g->pos;

        /* ---- one, and only one, legal-move generation per ply ---- */
        const int n = gen_legal(p, list);

        /* Terminal tests, in game_update_result()'s own order of precedence. */
        if (n == 0) {                       /* checkmate or stalemate         */
            game_update_result(g, 0);
            break;
        }
        if (insufficient_material(p)) {
            g->result = GR_DRAW; g->reason = TR_INSUFFICIENT; break;
        }
        if (p->halfmove >= 100) {
            g->result = GR_DRAW; g->reason = TR_FIFTY; break;
        }
        if (game_repetitions(g) >= 3) {
            g->result = GR_DRAW; g->reason = TR_REPETITION; break;
        }
        if (g->ply >= cap) {                /* adjudicated at the ply cap     */
            g->result = GR_DRAW; g->reason = TR_MAX_PLIES; break;
        }

        const int   stm = p->side;
        const Head *h   = (stm == WHITE) ? hw : hb;

        /* ---- network ---- */
        int nf = nn_features(p, fidx);
        if (nf > NF_MAXACTIVE) nf = NF_MAXACTIVE;
        nn_eval(trunk, h, fidx, nf, &fw);

        for (int i = 0; i < n; i++) nn_move_key(p, list[i], &keys[i]);
        nn_logits(h, &fw, keys, n, logits);

        /* ---- temperature ---- */
        float temp = (opening_ply > 0 && g->ply < opening_ply)
                     ? opening_t
                     : ((stm == WHITE) ? cfg->temp_white : cfg->temp_black);
        if (!(temp > 0.0f)) {               /* unset / NaN: fall back sanely  */
            const Hyper *y = (stm == WHITE) ? yw : yb;
            temp = (y && y->temperature > 0.0f) ? y->temperature : 1.0f;
        }

        /* ---- pick a move ---- */
        int ci = 0;
        if (greedy) {
            float best = logits[0];
            for (int i = 1; i < n; i++)
                if (logits[i] > best) { best = logits[i]; ci = i; }
        } else if (n > 1) {
            softmax_t(logits, n, temp, probs);

            float tot = 0.0f;
            for (int i = 0; i < n; i++) {
                float pi = probs[i];
                if (!(pi > 0.0f)) pi = 0.0f;    /* NaN/negative safe          */
                probs[i] = pi;
                tot += pi;
            }
            if (tot > 0.0f) {
                const float r = rng_float(rs) * tot;
                float c = 0.0f;
                ci = n - 1;                     /* guards fp round-off        */
                for (int i = 0; i < n; i++) {
                    c += probs[i];
                    if (r < c) { ci = i; break; }
                }
            } else {                            /* degenerate distribution    */
                ci = (int)(rng_next(rs) % (uint64_t)n);
            }
        }
        const Move mv = list[ci];

        /* ---- record the decision for the learner ---- */
        if (record) {
            Traj *tr = (stm == WHITE) ? tw : tb;
            if (tr && tr->ply < MAX_GAME_PLIES && traj_reserve(tr, n)) {
                const int t = tr->ply;
                memcpy(tr->fidx[t], fidx, (size_t)nf * sizeof(uint16_t));
                tr->nf[t]     = (uint8_t)nf;
                tr->mover[t]  = (uint8_t)stm;
                tr->koff[t]   = tr->nkeys;
                memcpy(tr->keys + tr->nkeys, keys, (size_t)n * sizeof(MoveKey));
                tr->nkeys    += n;
                tr->nmoves[t] = (uint16_t)n;
                tr->chosen[t] = (uint16_t)ci;
                tr->v[t]      = fw.v;
                /* potential of the position BEFORE the move, WHITE's view */
                tr->phi[t]    = tanhf(white_material(p) / 5.0f);
                tr->ply       = t + 1;
            }
        }

        /* ---- statistics about the move being played ---- */
        if (stats) {
            const int from = MV_FROM(mv), to = MV_TO(mv), fl = MV_FLAG(mv);

            if (fl & 4)                      stats->captures++;
            if (fl == MF_EP)                 stats->ep_captures++;
            else if (fl == MF_KCASTLE)       stats->castles_k++;
            else if (fl == MF_QCASTLE)       stats->castles_q++;
            if (fl >= MF_PROMO_N)            stats->promotions++;

            const int pc = p->board[from];
            if ((unsigned)pc < (unsigned)NPIECES) stats->piece_dest[pc][to]++;
            if (g->ply == 0 && stm == WHITE)      stats->first_move[from * 64 + to]++;
        }

        game_push_raw(g, mv);

        if (stats && in_check(&g->pos, g->pos.side)) stats->checks++;
    }

    /* ---- per-game statistics ---- */
    if (stats) {
        stats->games++;
        stats->plies   += (uint64_t)g->ply;
        stats->sum_len += (double)g->ply;

        switch (g->result) {
            case GR_WHITE_WIN: stats->white_wins++; break;
            case GR_BLACK_WIN: stats->black_wins++; break;
            default:           stats->draws++;      break;
        }
        switch (g->reason) {
            case TR_CHECKMATE:    stats->checkmates++;   break;
            case TR_STALEMATE:    stats->stalemates++;   break;
            case TR_FIFTY:        stats->fifty++;        break;
            case TR_REPETITION:   stats->repetition++;   break;
            case TR_INSUFFICIENT: stats->insufficient++; break;
            case TR_MAX_PLIES:    stats->maxplies++;     break;
            default: break;
        }
        stats->sum_final_material += (double)white_material(&g->pos);
    }

    if (reason) *reason = g->reason;
    return g->result;
}
