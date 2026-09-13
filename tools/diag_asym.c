/* tools/diag_asym.c -- does an ASYMMETRIC SEARCH BUDGET make material predict
 * the result in self-play?
 *
 * ===========================================================================
 * THE QUESTION
 * ===========================================================================
 * docs/TACTICS.md diagnosed the value head's failure as a DATA problem: in
 * symmetric self-play, being two pawns down is worth an actual score of
 * -0.036, because the opponent is the same network and hands the material
 * straight back.  Material therefore does not predict the outcome, the value
 * head correctly learns that it does not, and the agent goes on hanging
 * pieces.
 *
 * If one side searched a bigger budget it should win material AND KEEP it, so
 * within those games material would start predicting the result.  This tool
 * measures exactly that, on ONE model, with everything else held fixed: it
 * plays self-play games under training settings, makes `--asym-frac` of them
 * unequal, and reports the material-to-result relationship SEPARATELY for the
 * symmetric and the asymmetric games.  Because both arms come out of the same
 * invocation, the same model and the same opening distribution, the comparison
 * is paired and the only difference is the handicap.
 *
 * ===========================================================================
 * WHY THIS FILE MAY COUNT MATERIAL
 * ===========================================================================
 * It contains piece values (py/baselines.py's, so the numbers are comparable
 * with tools/diag_tactics.c and docs/TACTICS.md).  docs/FROM_SCRATCH.md
 * forbids that in the learning or play path and permits it in a measuring
 * instrument.  This file is linked into nothing -- not build/chessrl, not
 * libchessrl.dylib -- and tools/audit_knowledge.sh does not scan tools/.
 * It is built by hand:
 *
 *   cc -O3 -std=c11 -D_DARWIN_C_SOURCE -Wall -Wextra -Wno-unused-parameter \
 *      -funroll-loops -fno-math-errno -ffp-contract=fast -mcpu=native -Isrc \
 *      tools/diag_asym.c build/chess.o build/net.o build/mcts.o \
 *      -o build/diag_asym -lm -lpthread
 *
 * ===========================================================================
 * WHAT IT DOES NOT DO
 * ===========================================================================
 * Hang rate, the piece-deletion probe and the score against the material-1
 * oracle are NOT reimplemented here: tools/diag_tactics.c already measures all
 * three and docs/TACTICS.md's numbers come from it.  Re-running that tool
 * against a model trained with asymmetry is the comparable measurement; a
 * second implementation would not be.
 */

#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
#include <unistd.h>

#include "chess.h"
#include "net.h"
#include "mcts.h"

/* ========================================================================= */
/*                                  utility                                  */
/* ========================================================================= */

static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static uint64_t sm64(uint64_t *s)
{
    uint64_t z = (*s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static uint64_t rotl64(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }

static uint64_t rng_next(uint64_t *s)
{
    const uint64_t r = rotl64(s[1] * 5, 7) * 9;
    const uint64_t t = s[1] << 17;
    s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3]; s[2] ^= t;
    s[3] = rotl64(s[3], 45);
    return r;
}

static void rng_seed(uint64_t *s, uint64_t seed)
{
    uint64_t z = seed ? seed : 0x9E3779B97F4A7C15ULL;
    for (int i = 0; i < 4; i++) s[i] = sm64(&z);
    for (int i = 0; i < 8; i++) (void)rng_next(s);
}

static int cpu_count_local(void)
{
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    if (n < 1) n = 4;
    if (n > 64) n = 64;
    return (int)n;
}

static const char *opt_str(int argc, char **argv, const char *flag, const char *dflt)
{
    for (int i = 1; i + 1 < argc; i++) if (!strcmp(argv[i], flag)) return argv[i + 1];
    return dflt;
}

static double opt_num(int argc, char **argv, const char *flag, double dflt)
{
    const char *s = opt_str(argc, argv, flag, NULL);
    return s ? atof(s) : dflt;
}

static long opt_int(int argc, char **argv, const char *flag, long dflt)
{
    const char *s = opt_str(argc, argv, flag, NULL);
    return s ? strtol(s, NULL, 10) : dflt;
}

static int opt_flag(int argc, char **argv, const char *flag)
{
    for (int i = 0; i < argc; i++) if (!strcmp(argv[i], flag)) return 1;
    return 0;
}

/* ========================================================================= */
/*                            the material ruler                             */
/* ========================================================================= */
/* py/baselines.py PIECE_VALUE, in pawns -- identical to tools/diag_tactics.c
 * so that every number here can be read next to docs/TACTICS.md.           */
static const float PV[NPIECES] = { 1.0f, 3.0f, 3.25f, 5.0f, 9.0f, 0.0f };

/* Material balance from the side to move's view, in pawns. */
static float mat_bal(const Position *p)
{
    float w = 0.0f, b = 0.0f;
    for (int pc = PAWN; pc <= QUEEN; pc++) {
        w += PV[pc] * (float)bb_count(p->piece[WHITE][pc]);
        b += PV[pc] * (float)bb_count(p->piece[BLACK][pc]);
    }
    return (p->side == WHITE) ? (w - b) : (b - w);
}

/* ========================================================================= */
/*                                 the model                                 */
/* ========================================================================= */

typedef struct {
    Trunk *tr;
    Head  *heads;
    Hyper *hy;
    float *elo;
    int    n, gen, best;
    char   name[256];
} Model;

static int model_open(Model *M, const char *path)
{
    int probe_n = 0, probe_gen = 0, cap;
    memset(M, 0, sizeof *M);
    if (!model_load(path, NULL, NULL, NULL, NULL, &probe_n, &probe_gen) || probe_n <= 0) {
        fprintf(stderr, "error: cannot read model '%s'\n", path ? path : "(null)");
        return 0;
    }
    M->tr    = calloc(1, sizeof(Trunk));
    M->heads = calloc((size_t)probe_n, sizeof(Head));
    M->hy    = calloc((size_t)probe_n, sizeof(Hyper));
    M->elo   = calloc((size_t)probe_n, sizeof(float));
    if (!M->tr || !M->heads || !M->hy || !M->elo) { fprintf(stderr, "oom\n"); return 0; }
    cap = probe_n;
    if (!model_load(path, M->tr, M->heads, M->hy, M->elo, &cap, &probe_gen) || cap <= 0) {
        fprintf(stderr, "error: model '%s' is corrupt\n", path);
        return 0;
    }
    M->n = cap; M->gen = probe_gen; M->best = 0;
    for (int i = 1; i < cap; i++) if (M->elo[i] > M->elo[M->best]) M->best = i;
    snprintf(M->name, sizeof M->name, "%s", path);
    return 1;
}

static void model_close(Model *M)
{
    free(M->tr); free(M->heads); free(M->hy); free(M->elo);
    memset(M, 0, sizeof *M);
}

static const Head *model_head(const Model *M) { return &M->heads[M->best]; }

/* The raw value head on `p`, from the side to move's view.  No search. */
static float net_value(const Model *M, const Position *p)
{
    uint16_t f[NF_MAXACTIVE];
    Fwd      fw;
    const int nf = nn_features(p, f);
    nn_eval(M->tr, model_head(M), f, nf, &fw);
    return fw.v;
}

/* ========================================================================= */
/*                          self-play, with a handicap                       */
/* ========================================================================= */

enum { ST_CLASSICAL = 0, ST_960 = 1, ST_MIXED = 2 };

/* One recorded position: everything needed to relate material to the result. */
typedef struct {
    float v;        /* the raw value head, mover's view                      */
    float mat;      /* material balance, mover's view, in pawns              */
    float z;        /* the game result from that mover's view: +1 / 0 / -1   */
    int32_t ply;
    int8_t  asym;   /* 1 = this game had unequal budgets                     */
    int8_t  weak;   /* 1 = the mover was the side on the reduced budget      */
} Rec;

typedef struct {
    const Model *M;
    int      games, sims, max_plies, start, train_mode;
    double   asym_frac, asym_ratio;
    uint64_t seed;
    Rec     *out;
    int      cap, got;

    /* per-game accounting, split symmetric / asymmetric */
    long     ngame[2];              /* [0] symmetric, [1] asymmetric         */
    long     ndraw[2], nplies[2];
    long     ncap_mat[2];           /* games decided by something other than  */
    double   strong_pts;            /* points taken by the FULL-budget side   */
    long     strong_n, white_strong;
} Worker;

static void *play_thread(void *arg)
{
    Worker *w = (Worker *)arg;
    Mcts m;
    uint64_t rng[4];

    rng_seed(rng, w->seed);
    /* mcts_init()'s own rule, sized for the LARGEST budget this thread runs. */
    mcts_init(&m, 50 * (w->sims + 8) + 1);

    for (int gi = 0; gi < w->games && w->got < w->cap; gi++) {
        Game g;
        const int first = w->got;
        int  res, strong = -1;

        /* Bresenham over the game index: exactly round(frac * games) games are
         * asymmetric, and the full budget alternates between White and Black,
         * so the colours are balanced by construction rather than on average.
         * This mirrors src/az.c's build_pairings() assignment. */
        if (w->asym_frac > 0.0) {
            const long want = (long)(w->asym_frac * (double)w->games + 0.5);
            const long a0 = ((long)gi       * want) / (long)w->games;
            const long a1 = ((long)(gi + 1) * want) / (long)w->games;
            if (a0 != a1) strong = (a0 & 1L) ? BLACK : WHITE;
        }

        if (w->start == ST_CLASSICAL) game_start(&g);
        else if (w->start == ST_960)  game_start960(&g, (int)(rng_next(rng) % 960u));
        else { if (rng_next(rng) & 1) game_start(&g);
               else game_start960(&g, (int)(rng_next(rng) % 960u)); }

        while (g.result == GR_ONGOING && g.ply < w->max_plies && w->got < w->cap) {
            Move    list[MAX_MOVES];
            int32_t visits[MAX_MOVES];
            float   rv;
            int     n = gen_legal(&g.pos, list), nr, nsims = w->sims;
            const int weak = (strong >= 0 && (int)g.pos.side != strong);
            if (n <= 0) break;

            if (weak) {
                nsims = (int)((double)w->sims / w->asym_ratio + 0.5);
                if (nsims < 2) nsims = 2;
            }
            nr = mcts_search(&m, w->M->tr, model_head(w->M), &g,
                             nsims, w->train_mode, rng, visits, &rv);
            if (nr != n) break;

            {   /* record the position BEFORE the move is played */
                Rec *r = &w->out[w->got++];
                r->v    = net_value(w->M, &g.pos);
                r->mat  = mat_bal(&g.pos);
                r->ply  = g.ply;
                r->asym = (int8_t)(strong >= 0);
                r->weak = (int8_t)weak;
                r->z    = (float)(int)g.pos.side;   /* the side; fixed up below */
            }
            game_push(&g, list[mcts_pick(visits, n,
                        w->train_mode ? (g.ply < 20 ? 1.0f : 0.25f)
                                      : (g.ply < 12 ? 1.0f : 0.0f), rng)]);
            game_update_result(&g, w->max_plies);
        }

        res = g.result;
        if (res == GR_ONGOING) res = GR_DRAW;
        {
            const int k = (strong >= 0);
            w->ngame[k]++;
            w->nplies[k] += g.ply;
            if (res == GR_DRAW) w->ndraw[k]++;
            if (g.ply >= w->max_plies) w->ncap_mat[k]++;
        }
        if (strong >= 0) {
            const double sw = (res == GR_WHITE_WIN) ? 1.0
                            : (res == GR_BLACK_WIN) ? 0.0 : 0.5;
            w->strong_pts += (strong == WHITE) ? sw : 1.0 - sw;
            w->strong_n++;
            if (strong == WHITE) w->white_strong++;
        }
        for (int i = first; i < w->got; i++) {
            const int side = (int)w->out[i].z;
            float z = 0.0f;
            if (res == GR_WHITE_WIN)      z = (side == WHITE) ? 1.0f : -1.0f;
            else if (res == GR_BLACK_WIN) z = (side == BLACK) ? 1.0f : -1.0f;
            w->out[i].z = z;
        }
    }
    mcts_free(&m);
    return NULL;
}

/* ========================================================================= */
/*                                 statistics                                */
/* ========================================================================= */

#define MB   13                    /* material buckets, -6 .. +6 pawns       */
#define VB   10                    /* value buckets, -1 .. +1                */
#define PB    6                    /* ply buckets                            */

static const int PLY_EDGE[PB] = { 10, 20, 40, 60, 100, 1 << 30 };

typedef struct {
    long   n;
    /* material -> outcome */
    long   mn[MB];  double mv[MB], mz[MB];
    /* value -> outcome */
    long   vn[VB];  double vv[VB], vz[VB];
    /* correlations: x = material, y = z (outcome), u = v (prediction)       */
    double sx, sy, su, sxx, syy, suu, sxy, suy;
    /* per-ply-bucket corr(material, z)                                      */
    long   pn[PB];  double px[PB], py[PB], pxx[PB], pyy[PB], pxy[PB];
    /* value-head error                                                      */
    double mse, zsum, zsq, vsum, vsq;
} Acc;

static void acc_add(Acc *a, const Rec *r)
{
    int mb = (int)floor((double)r->mat + 0.5) + 6;
    int vb = (int)((r->v + 1.0f) * (float)(VB / 2));
    int pb = 0;
    if (mb < 0) mb = 0; if (mb > MB - 1) mb = MB - 1;
    if (vb < 0) vb = 0; if (vb > VB - 1) vb = VB - 1;
    while (pb < PB - 1 && r->ply >= PLY_EDGE[pb]) pb++;

    a->n++;
    a->mn[mb]++; a->mv[mb] += r->v; a->mz[mb] += r->z;
    a->vn[vb]++; a->vv[vb] += r->v; a->vz[vb] += r->z;
    a->sx += r->mat;  a->sy += r->z;   a->su += r->v;
    a->sxx += (double)r->mat * r->mat;
    a->syy += (double)r->z   * r->z;
    a->suu += (double)r->v   * r->v;
    a->sxy += (double)r->mat * r->z;
    a->suy += (double)r->v   * r->z;
    a->pn[pb]++;
    a->px[pb] += r->mat;  a->py[pb] += r->z;
    a->pxx[pb] += (double)r->mat * r->mat;
    a->pyy[pb] += (double)r->z   * r->z;
    a->pxy[pb] += (double)r->mat * r->z;
    a->mse  += (double)(r->v - r->z) * (r->v - r->z);
    a->zsum += r->z; a->zsq += (double)r->z * r->z;
    a->vsum += r->v; a->vsq += (double)r->v * r->v;
}

static double corr(double sxy, double sx, double sy, double sxx, double syy, long n)
{
    double num, den;
    if (n < 2) return 0.0;
    num = sxy - sx * sy / (double)n;
    den = sqrt((sxx - sx * sx / (double)n) * (syy - sy * sy / (double)n));
    return den > 0.0 ? num / den : 0.0;
}

/* OLS slope of y on x: score per pawn of material.  This is the single number
 * the whole experiment is about. */
static double slope(double sxy, double sx, double sy, double sxx, long n)
{
    const double den = sxx - sx * sx / (double)n;
    if (n < 2 || den <= 0.0) return 0.0;
    return (sxy - sx * sy / (double)n) / den;
}

/* The standard error of that slope, so a difference between the two arms can
 * be read against something. */
static double slope_se(const Acc *a)
{
    const double n = (double)a->n;
    double sxx, b, ssr, sst;
    if (a->n < 3) return 0.0;
    sxx = a->sxx - a->sx * a->sx / n;
    if (sxx <= 0.0) return 0.0;
    b   = (a->sxy - a->sx * a->sy / n) / sxx;
    sst = a->syy - a->sy * a->sy / n;
    ssr = sst - b * b * sxx;                     /* residual sum of squares  */
    if (ssr < 0.0) ssr = 0.0;
    return sqrt(ssr / (n - 2.0) / sxx);
}

static void report(const char *tag, const Acc *a, long games, long draws,
                   long plies)
{
    const double n = (double)a->n;
    double zmean, zvar, mse, b, se;

    if (a->n < 2) { printf("\n--- %s: no positions ---\n", tag); return; }
    zmean = a->zsum / n;
    zvar  = a->zsq / n - zmean * zmean;
    mse   = a->mse / n;
    b     = slope(a->sxy, a->sx, a->sy, a->sxx, a->n);
    se    = slope_se(a);

    printf("\n=== %s ===\n", tag);
    printf("games %ld   draw rate %.3f   mean plies %.1f   positions %ld\n",
           games, games ? (double)draws / (double)games : 0.0,
           games ? (double)plies / (double)games : 0.0, a->n);
    printf("corr(material, result) r = %+.4f      <-- THE NUMBER\n",
           corr(a->sxy, a->sx, a->sy, a->sxx, a->syy, a->n));
    printf("OLS slope  d(result)/d(pawn) = %+.4f +/- %.4f   "
           "=> two pawns down is worth %+.3f\n", b, se, -2.0 * b);
    printf("corr(v, result)        r = %+.4f\n",
           corr(a->suy, a->su, a->sy, a->suu, a->syy, a->n));
    printf("value head:  MSE %.4f   Var(z) %.4f   MSE/Var %.4f   R^2 %+.4f\n",
           mse, zvar, zvar > 0 ? mse / zvar : 0.0,
           zvar > 0 ? 1.0 - mse / zvar : 0.0);
    printf("mean predicted v %+.4f (sd %.4f)   mean outcome z %+.4f (sd %.4f)"
           "   BIAS %+.4f\n",
           a->vsum / n, sqrt(fabs(a->vsq / n - (a->vsum / n) * (a->vsum / n))),
           zmean, sqrt(fabs(zvar)), a->vsum / n - zmean);

    printf("\n  material (pawns) |      n   | mean v  | ACTUAL score\n");
    printf("  -----------------+----------+---------+-------------\n");
    for (int i = 0; i < MB; i++) {
        if (a->mn[i] < 20) continue;
        printf("   %s%+3d            | %8ld | %+.3f  |    %+.3f\n",
               i == 0 ? "<=" : i == MB - 1 ? ">=" : "  ", i - 6, a->mn[i],
               a->mv[i] / (double)a->mn[i], a->mz[i] / (double)a->mn[i]);
    }

    printf("\n  ply range |      n   | corr(material, result)\n");
    printf("  ----------+----------+-----------------------\n");
    for (int i = 0; i < PB; i++) {
        char lab[32];
        if (a->pn[i] < 50) continue;
        if (i == PB - 1) snprintf(lab, sizeof lab, "%4d+     ", i ? PLY_EDGE[i-1] : 0);
        else snprintf(lab, sizeof lab, "%4d-%-4d ", i ? PLY_EDGE[i-1] : 0,
                      PLY_EDGE[i] - 1);
        printf("  %s| %8ld |        %+.4f\n", lab, a->pn[i],
               corr(a->pxy[i], a->px[i], a->py[i], a->pxx[i], a->pyy[i], a->pn[i]));
    }

    printf("\n   v bucket    |      n   | mean v  | actual score | gap\n");
    printf("  -------------+----------+---------+--------------+-------\n");
    for (int i = 0; i < VB; i++) {
        if (a->vn[i] < 20) continue;
        printf("  [%+.1f,%+.1f) | %8ld | %+.3f  |    %+.3f     | %+.3f\n",
               -1.0 + 2.0 * i / VB, -1.0 + 2.0 * (i + 1) / VB, a->vn[i],
               a->vv[i] / (double)a->vn[i], a->vz[i] / (double)a->vn[i],
               a->vv[i] / (double)a->vn[i] - a->vz[i] / (double)a->vn[i]);
    }
}

static void usage(void)
{
    printf(
    "diag_asym [options]\n"
    "  Plays self-play games under TRAINING settings and reports the\n"
    "  material-to-result relationship separately for the symmetric and the\n"
    "  asymmetric games, on the same model and the same openings.\n"
    "  --model P        checkpoint            (default runs/az_hour/best.crl)\n"
    "  --games N        games                                  (default 400)\n"
    "  --sims N         the FULL simulation budget              (default 160)\n"
    "  --asym-frac F    fraction of games given unequal budgets (default 0.5)\n"
    "  --asym-ratio R   the weak side searches sims/R             (default 4)\n"
    "  --start S        classical | 960 | mixed           (default classical)\n"
    "  --max-plies N    adjudicate a draw here                  (default 300)\n"
    "  --play-mode      no root noise, temp 1.0 for 12 plies then argmax\n"
    "  --threads N      default = cores\n"
    "  --seed N         default 20260912\n");
}

int main(int argc, char **argv)
{
    const char *mpath = opt_str(argc, argv, "--model", "runs/az_hour/best.crl");
    const char *sst   = opt_str(argc, argv, "--start", "classical");
    const int   games = (int)opt_int(argc, argv, "--games", 400);
    const int   sims  = (int)opt_int(argc, argv, "--sims", 160);
    const int   maxp  = (int)opt_int(argc, argv, "--max-plies", 300);
    const double af   = opt_num(argc, argv, "--asym-frac", 0.5);
    const double ar   = opt_num(argc, argv, "--asym-ratio", 4.0);
    const uint64_t sd = (uint64_t)opt_int(argc, argv, "--seed", 20260912);
    const int train   = !opt_flag(argc, argv, "--play-mode");
    int threads = (int)opt_int(argc, argv, "--threads", cpu_count_local());
    int start = ST_CLASSICAL;
    Model M;
    Worker *w;
    pthread_t *t;
    Acc sym, asy, wk, st;
    long gsym = 0, gasy = 0, dsym = 0, dasy = 0, psym = 0, pasy = 0;
    long strong_n = 0, white_strong = 0;
    double strong_pts = 0.0, t0;
    int cap, weak_sims;

    if (opt_flag(argc, argv, "-h") || opt_flag(argc, argv, "--help")) { usage(); return 0; }
    chess_init();                    /* magics, zobrist -- nothing works without it */
    if (!strcmp(sst, "960")) start = ST_960;
    else if (!strcmp(sst, "mixed")) start = ST_MIXED;
    if (threads < 1) threads = 1;
    if (threads > games) threads = games;

    if (!model_open(&M, mpath)) return 1;
    weak_sims = (int)((double)sims / (ar > 1.0 ? ar : 1.0) + 0.5);
    if (weak_sims < 2) weak_sims = 2;

    printf("diag_asym: %s (gen %d, agent %d of %d)\n", M.name, M.gen, M.best, M.n);
    printf("%d games, %d threads, %s starts, %s, max %d plies, seed %llu\n",
           games, threads, sst,
           train ? "TRAINING settings (root noise, temp 1.0/20 then 0.25)"
                 : "PLAY settings (no noise, temp 1.0/12 then argmax)",
           maxp, (unsigned long long)sd);
    printf("asymmetry: %.0f%% of games at %d vs %d sims (ratio %.2f)\n",
           af * 100.0, sims, weak_sims, ar);

    cap = 400 * maxp;                        /* positions per thread          */
    w = calloc((size_t)threads, sizeof *w);
    t = calloc((size_t)threads, sizeof *t);
    if (!w || !t) { fprintf(stderr, "oom\n"); return 1; }
    t0 = now_sec();
    for (int i = 0; i < threads; i++) {
        w[i].M = &M;
        w[i].games = (games + threads - 1) / threads;
        w[i].sims = sims; w[i].max_plies = maxp; w[i].start = start;
        w[i].train_mode = train;
        w[i].asym_frac = af; w[i].asym_ratio = ar > 1.0 ? ar : 1.0;
        w[i].seed = sd + 77ULL * (uint64_t)(i + 1);
        w[i].out = calloc((size_t)cap, sizeof(Rec));
        w[i].cap = cap;
        if (!w[i].out) { fprintf(stderr, "oom\n"); return 1; }
        pthread_create(&t[i], NULL, play_thread, &w[i]);
    }
    for (int i = 0; i < threads; i++) pthread_join(t[i], NULL);

    memset(&sym, 0, sizeof sym); memset(&asy, 0, sizeof asy);
    memset(&wk,  0, sizeof wk);  memset(&st,  0, sizeof st);
    for (int i = 0; i < threads; i++) {
        gsym += w[i].ngame[0]; gasy += w[i].ngame[1];
        dsym += w[i].ndraw[0]; dasy += w[i].ndraw[1];
        psym += w[i].nplies[0]; pasy += w[i].nplies[1];
        strong_pts += w[i].strong_pts;
        strong_n   += w[i].strong_n;
        white_strong += w[i].white_strong;
        for (int k = 0; k < w[i].got; k++) {
            const Rec *r = &w[i].out[k];
            if (!r->asym) { acc_add(&sym, r); continue; }
            acc_add(&asy, r);
            acc_add(r->weak ? &wk : &st, r);
        }
    }
    printf("elapsed %.1fs\n", now_sec() - t0);

    if (strong_n) {
        printf("\nasymmetric games: %ld   full budget held by White in %ld "
               "(%.3f)   STRONG SIDE SCORED %.4f\n",
               strong_n, white_strong, (double)white_strong / (double)strong_n,
               strong_pts / (double)strong_n);
        printf("(0.5 would mean the handicap did nothing; the mechanism needs "
               "this well above 0.5)\n");
    }

    report("SYMMETRIC GAMES -- both sides at the full budget", &sym,
           gsym, dsym, psym);
    report("ASYMMETRIC GAMES -- one side handicapped", &asy, gasy, dasy, pasy);
    if (st.n) report("ASYMMETRIC, positions where the STRONG side is to move",
                     &st, 0, 0, 0);
    if (wk.n) report("ASYMMETRIC, positions where the WEAK side is to move",
                     &wk, 0, 0, 0);

    for (int i = 0; i < threads; i++) free(w[i].out);
    free(w); free(t);
    model_close(&M);
    return 0;
}
