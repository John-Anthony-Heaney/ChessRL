/* main.c -- command dispatch for the chessrl binary.
 *
 *   chessrl train     population RL training run
 *   chessrl bench     headline throughput number (full games/sec, net in the loop)
 *   chessrl perft     move-generator node counts
 *   chessrl selfplay  watch two trained agents play
 *   chessrl eval      round-robin cross table among the strongest agents
 *   chessrl play      text loop to play the champion from the terminal
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <pthread.h>

#include "chess.h"
#include "net.h"
#include "arena.h"
#include "search.h"
#include "train.h"

/* --------------------------------------------------------------- utilities */

static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static int default_threads(void)
{
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    if (n < 1) n = 4;
    if (n > 64) n = 64;
    return (int)n;
}

/* Returns the value following `flag`, or NULL. Consumes nothing. */
static const char *opt_str(int argc, char **argv, const char *flag, const char *def)
{
    for (int i = 0; i < argc - 1; i++)
        if (strcmp(argv[i], flag) == 0) return argv[i + 1];
    return def;
}

static long opt_int(int argc, char **argv, const char *flag, long def)
{
    const char *s = opt_str(argc, argv, flag, NULL);
    return s ? strtol(s, NULL, 10) : def;
}

static double opt_num(int argc, char **argv, const char *flag, double def)
{
    const char *s = opt_str(argc, argv, flag, NULL);
    return s ? strtod(s, NULL) : def;
}

static int opt_flag(int argc, char **argv, const char *flag)
{
    for (int i = 0; i < argc; i++)
        if (strcmp(argv[i], flag) == 0) return 1;
    return 0;
}

/* Loads a model, allocating the head/hyper/elo arrays. Caller frees them. */
static int load_model(const char *path, Trunk *tr, Head **heads, Hyper **hy,
                      float **elo, int *n, int *gen)
{
    int probe_n = 0, probe_gen = 0;
    if (!model_load(path, NULL, NULL, NULL, NULL, &probe_n, &probe_gen)) {
        fprintf(stderr, "error: cannot read model '%s'\n", path ? path : "(null)");
        return 0;
    }
    *heads = calloc((size_t)probe_n, sizeof(Head));
    *hy    = calloc((size_t)probe_n, sizeof(Hyper));
    *elo   = calloc((size_t)probe_n, sizeof(float));
    if (!*heads || !*hy || !*elo) {
        fprintf(stderr, "error: out of memory for %d agents\n", probe_n);
        free(*heads); free(*hy); free(*elo);
        return 0;
    }
    int cap = probe_n;
    if (!model_load(path, tr, *heads, *hy, *elo, &cap, &probe_gen)) {
        fprintf(stderr, "error: model '%s' is corrupt\n", path);
        free(*heads); free(*hy); free(*elo);
        return 0;
    }
    *n = cap;
    *gen = probe_gen;
    return 1;
}

static int best_agent(const float *elo, int n)
{
    int best = 0;
    for (int i = 1; i < n; i++)
        if (elo[i] > elo[best]) best = i;
    return best;
}

/* ------------------------------------------------------------------- train */

static int cmd_train(int argc, char **argv)
{
    TrainCfg c;
    train_default_cfg(&c);

    c.n_agents        = (int)opt_int(argc, argv, "--agents",          c.n_agents);
    c.generations     = (int)opt_int(argc, argv, "--gens",            c.generations);
    c.games_per_agent = (int)opt_int(argc, argv, "--games-per-agent", c.games_per_agent);
    c.threads         = (int)opt_int(argc, argv, "--threads",         c.threads);
    c.max_plies       = (int)opt_int(argc, argv, "--max-plies",       c.max_plies);
    c.base_lr         = (float)opt_num(argc, argv, "--lr",            c.base_lr);
    c.trunk_lr        = (float)opt_num(argc, argv, "--trunk-lr",      c.trunk_lr);
    c.weight_decay    = (float)opt_num(argc, argv, "--wd",            c.weight_decay);
    c.grad_clip       = (float)opt_num(argc, argv, "--clip",          c.grad_clip);
    c.elite_frac      = (float)opt_num(argc, argv, "--elite",         c.elite_frac);
    c.cull_frac       = (float)opt_num(argc, argv, "--cull",          c.cull_frac);
    c.hof_every       = (int)opt_int(argc, argv, "--hof-every",       c.hof_every);
    c.hof_frac_pct    = (int)opt_int(argc, argv, "--hof-pct",         c.hof_frac_pct);
    c.seed            = (uint64_t)opt_int(argc, argv, "--seed",       (long)c.seed);
    c.quiet           = opt_flag(argc, argv, "--quiet");

    static char run_dir[512];
    const char *run = opt_str(argc, argv, "--run", NULL);
    if (run) {
        if (strchr(run, '/')) snprintf(run_dir, sizeof run_dir, "%s", run);
        else                  snprintf(run_dir, sizeof run_dir, "runs/%s", run);
        c.run_dir = run_dir;
    }

    if (c.n_agents < 2 || c.generations < 1 || c.games_per_agent < 1 || c.threads < 1) {
        fprintf(stderr, "error: --agents >= 2, --gens >= 1, --games-per-agent >= 1, "
                        "--threads >= 1\n");
        return 2;
    }
    return train_run(&c) ? 0 : 1;
}

/* ------------------------------------------------------------------- bench */
/* The headline number. Full games, network in the loop, exactly the same code
 * path play_game() takes during training (minus the gradient work). */

typedef struct {
    const Trunk *trunk;
    const Head  *ha, *hb;
    const Hyper *ya, *yb;
    int          max_plies, record;
    double       deadline;
    uint64_t     seed;
    /* results */
    uint64_t     games, plies;
    Traj         ta, tb;
    PlayStats    stats;
} BenchWorker;

static void *bench_main(void *arg)
{
    BenchWorker *w = arg;
    uint64_t rng[4];
    rng_seed(rng, w->seed);

    PlayCfg cfg;
    memset(&cfg, 0, sizeof cfg);
    cfg.max_plies     = w->max_plies;
    cfg.greedy        = 0;
    cfg.temp_white    = 1.0f;
    cfg.temp_black    = 1.0f;
    cfg.opening_temp  = 1.2f;
    cfg.opening_plies = 12;
    cfg.record        = w->record;
    cfg.rng           = rng;

    traj_init(&w->ta);
    traj_init(&w->tb);
    stats_zero(&w->stats);

    /* Check the clock every 32 games so timing overhead stays negligible. */
    while (now_sec() < w->deadline) {
        for (int k = 0; k < 32; k++) {
            int reason = 0;
            traj_reset(&w->ta);
            traj_reset(&w->tb);
            play_game(w->trunk, w->ha, w->hb, w->ya, w->yb, &cfg,
                      w->record ? &w->ta : NULL, w->record ? &w->tb : NULL,
                      &w->stats, NULL, &reason);
            w->games++;
            w->plies += (uint64_t)(w->ta.ply + w->tb.ply);
        }
    }
    traj_free(&w->ta);
    traj_free(&w->tb);
    return NULL;
}

static int cmd_bench(int argc, char **argv)
{
    double seconds  = opt_num(argc, argv, "--seconds", 10.0);
    int    threads  = (int)opt_int(argc, argv, "--threads", default_threads());
    int    maxplies = (int)opt_int(argc, argv, "--max-plies", 240);
    int    record   = !opt_flag(argc, argv, "--no-record");
    const char *model = opt_str(argc, argv, "--model", NULL);

    if (threads < 1) threads = 1;

    Trunk *trunk = calloc(1, sizeof(Trunk));
    Head  *heads = NULL;
    Hyper *hy    = NULL;
    float *elo   = NULL;
    int    nag = 0, gen = 0;

    if (model) {
        if (!load_model(model, trunk, &heads, &hy, &elo, &nag, &gen)) { free(trunk); return 1; }
        printf("model %s  generation %d  agents %d\n", model, gen, nag);
    } else {
        nag   = 2;
        heads = calloc(2, sizeof(Head));
        hy    = calloc(2, sizeof(Hyper));
        elo   = calloc(2, sizeof(float));
        nn_init(trunk, &heads[0], 0x9E3779B97F4A7C15ull);
        nn_init(trunk, &heads[1], 0xD1B54A32D192ED03ull);
        hyper_default(&hy[0]);
        hyper_default(&hy[1]);
    }

    int a = 0, b = nag > 1 ? 1 : 0;
    printf("bench: %d threads, %.1fs, max_plies %d, trajectory recording %s\n",
           threads, seconds, maxplies, record ? "ON (as in training)" : "off");
    fflush(stdout);

    BenchWorker *w = calloc((size_t)threads, sizeof(BenchWorker));
    pthread_t   *th = calloc((size_t)threads, sizeof(pthread_t));
    double t0 = now_sec(), deadline = t0 + seconds;

    for (int i = 0; i < threads; i++) {
        w[i].trunk = trunk;
        w[i].ha = &heads[a]; w[i].hb = &heads[b];
        w[i].ya = &hy[a];    w[i].yb = &hy[b];
        w[i].max_plies = maxplies;
        w[i].record    = record;
        w[i].deadline  = deadline;
        w[i].seed      = 0xC0FFEEull * (uint64_t)(i + 1) + 12345u;
        pthread_create(&th[i], NULL, bench_main, &w[i]);
    }

    uint64_t games = 0, plies = 0;
    PlayStats total; stats_zero(&total);
    for (int i = 0; i < threads; i++) {
        pthread_join(th[i], NULL);
        games += w[i].games;
        plies += w[i].plies;
        stats_merge(&total, &w[i].stats);
    }
    double dt = now_sec() - t0;

    /* play_game only counts plies into the trajectories when recording, so fall
     * back to the stats counter otherwise. */
    if (!record || plies == 0) plies = total.plies;

    printf("\n");
    printf("  games            %10llu\n", (unsigned long long)games);
    printf("  plies            %10llu\n", (unsigned long long)plies);
    printf("  wall clock       %10.3f s\n", dt);
    printf("  ----------------------------------------\n");
    printf("  GAMES / SECOND   %10.1f\n", (double)games / dt);
    printf("  plies / second   %10.0f\n", (double)plies / dt);
    printf("  positions / sec  %10.0f   (one network eval each)\n", (double)plies / dt);
    printf("  avg game length  %10.1f plies\n", games ? (double)plies / (double)games : 0.0);
    printf("  ----------------------------------------\n");
    /* A serious over-the-board game is ~40 minutes. */
    printf("  vs. real time    %10.0fx faster than humans playing\n",
           (double)games / dt * 40.0 * 60.0);
    if (total.games) {
        printf("  W/D/L            %.1f%% / %.1f%% / %.1f%%\n",
               100.0 * (double)total.white_wins / (double)total.games,
               100.0 * (double)total.draws      / (double)total.games,
               100.0 * (double)total.black_wins / (double)total.games);
        printf("  terminations     mate %llu  stale %llu  50mv %llu  rep %llu  "
               "insuf %llu  cap %llu\n",
               (unsigned long long)total.checkmates, (unsigned long long)total.stalemates,
               (unsigned long long)total.fifty,      (unsigned long long)total.repetition,
               (unsigned long long)total.insufficient, (unsigned long long)total.maxplies);
    }

    free(w); free(th); free(trunk); free(heads); free(hy); free(elo);
    return 0;
}

/* ------------------------------------------------------------------- perft */

static int cmd_perft(int argc, char **argv)
{
    const char *fen = opt_str(argc, argv, "--fen", NULL);
    int depth = (int)opt_int(argc, argv, "--depth", 5);
    int divide = opt_flag(argc, argv, "--divide");

    Position p;
    if (fen) {
        if (!pos_from_fen(&p, fen)) { fprintf(stderr, "error: bad FEN\n"); return 2; }
    } else {
        pos_startpos(&p);
    }

    if (divide) {
        Move list[MAX_MOVES];
        int n = gen_legal(&p, list);
        uint64_t total = 0;
        for (int i = 0; i < n; i++) {
            Undo u; char uci[6];
            make_move(&p, list[i], &u);
            uint64_t sub = depth > 1 ? perft(&p, depth - 1) : 1;
            unmake_move(&p, list[i], &u);
            move_to_uci(list[i], uci);
            printf("%s: %llu\n", uci, (unsigned long long)sub);
            total += sub;
        }
        printf("\nnodes %llu\n", (unsigned long long)total);
        return 0;
    }

    for (int d = 1; d <= depth; d++) {
        double t0 = now_sec();
        uint64_t n = perft(&p, d);
        double dt = now_sec() - t0;
        printf("depth %2d  %14llu nodes  %8.3f s  %8.2f Mnps\n",
               d, (unsigned long long)n, dt, dt > 0 ? (double)n / dt / 1e6 : 0.0);
    }
    return 0;
}

/* ---------------------------------------------------------------- selfplay */

static const char *result_word(int r)
{
    switch (r) {
        case GR_WHITE_WIN: return "1-0";
        case GR_BLACK_WIN: return "0-1";
        case GR_DRAW:      return "1/2-1/2";
        default:           return "*";
    }
}

static const char *reason_word(int r)
{
    switch (r) {
        case TR_CHECKMATE:    return "checkmate";
        case TR_STALEMATE:    return "stalemate";
        case TR_FIFTY:        return "fifty-move rule";
        case TR_REPETITION:   return "threefold repetition";
        case TR_INSUFFICIENT: return "insufficient material";
        case TR_MAX_PLIES:    return "move limit";
        case TR_ADJUDICATED:  return "adjudicated";
        default:              return "unknown";
    }
}

static void print_pgn(const Game *g, int ai, int bi, int gameno)
{
    Position p;
    pos_startpos(&p);
    printf("[Event \"chessrl self-play\"]\n[Round \"%d\"]\n", gameno);
    printf("[White \"agent %d\"]\n[Black \"agent %d\"]\n", ai, bi);
    printf("[Result \"%s\"]\n[Termination \"%s\"]\n\n", result_word(g->result),
           reason_word(g->reason));
    int col = 0;
    for (int i = 0; i < g->ply; i++) {
        char san[16]; Undo u;
        move_to_san(&p, g->moves[i], san, sizeof san);
        if (i % 2 == 0) col += printf("%d. ", i / 2 + 1);
        col += printf("%s ", san);
        if (col > 72) { printf("\n"); col = 0; }
        make_move(&p, g->moves[i], &u);
    }
    printf("%s\n\n", result_word(g->result));
}

static int cmd_selfplay(int argc, char **argv)
{
    const char *model = opt_str(argc, argv, "--model", NULL);
    if (!model) { fprintf(stderr, "error: --model PATH is required\n"); return 2; }

    int games   = (int)opt_int(argc, argv, "--games", 1);
    int maxply  = (int)opt_int(argc, argv, "--max-plies", 300);
    int greedy  = opt_flag(argc, argv, "--greedy");
    double temp = opt_num(argc, argv, "--temp", 0.35);

    Trunk *trunk = calloc(1, sizeof(Trunk));
    Head *heads = NULL; Hyper *hy = NULL; float *elo = NULL;
    int n = 0, gen = 0;
    if (!load_model(model, trunk, &heads, &hy, &elo, &n, &gen)) { free(trunk); return 1; }

    int a = (int)opt_int(argc, argv, "--a", best_agent(elo, n));
    int b = (int)opt_int(argc, argv, "--b", -1);
    if (b < 0) { b = (a + 1) % n; }
    if (a < 0 || a >= n || b < 0 || b >= n) {
        fprintf(stderr, "error: agent index out of range (model has %d agents)\n", n);
        free(trunk); free(heads); free(hy); free(elo); return 2;
    }

    uint64_t rng[4]; rng_seed(rng, 0x5EEDu ^ (uint64_t)gen);
    PlayCfg cfg; memset(&cfg, 0, sizeof cfg);
    cfg.max_plies = maxply; cfg.greedy = greedy;
    cfg.temp_white = cfg.temp_black = (float)temp;
    cfg.opening_temp = (float)temp; cfg.opening_plies = 0;
    cfg.record = 0; cfg.rng = rng;

    Game *g = calloc(1, sizeof(Game));
    int w = 0, d = 0, l = 0;
    for (int i = 0; i < games; i++) {
        int reason = 0;
        int r = play_game(trunk, &heads[a], &heads[b], &hy[a], &hy[b], &cfg,
                          NULL, NULL, NULL, g, &reason);
        print_pgn(g, a, b, i + 1);
        if (r == GR_WHITE_WIN) w++; else if (r == GR_BLACK_WIN) l++; else d++;
    }
    printf("agent %d (Elo %.0f) as White vs agent %d (Elo %.0f): +%d =%d -%d\n",
           a, (double)elo[a], b, (double)elo[b], w, d, l);

    free(g); free(trunk); free(heads); free(hy); free(elo);
    return 0;
}

/* -------------------------------------------------------------------- eval */

typedef struct {
    const Trunk *trunk;
    const Head  *heads;
    const Hyper *hy;
    const int   *idx;
    int          k, games, maxply;
    uint64_t     seed;
    int         *score;    /* k*k, doubled points: win 2, draw 1 */
} EvalJob;

static void *eval_main(void *arg)
{
    EvalJob *j = arg;
    uint64_t rng[4]; rng_seed(rng, j->seed);
    PlayCfg cfg; memset(&cfg, 0, sizeof cfg);
    cfg.max_plies = j->maxply; cfg.greedy = 0;
    cfg.temp_white = cfg.temp_black = 0.30f;
    cfg.opening_temp = 0.90f; cfg.opening_plies = 8;
    cfg.record = 0; cfg.rng = rng;

    for (int a = 0; a < j->k; a++)
        for (int b = 0; b < j->k; b++) {
            if (a == b) continue;
            for (int m = 0; m < j->games; m++) {
                int reason = 0;
                int r = play_game(j->trunk, &j->heads[j->idx[a]], &j->heads[j->idx[b]],
                                  &j->hy[j->idx[a]], &j->hy[j->idx[b]], &cfg,
                                  NULL, NULL, NULL, NULL, &reason);
                if (r == GR_WHITE_WIN)      j->score[a * j->k + b] += 2;
                else if (r == GR_BLACK_WIN) j->score[b * j->k + a] += 2;
                else { j->score[a * j->k + b] += 1; j->score[b * j->k + a] += 1; }
            }
        }
    return NULL;
}

static int cmd_eval(int argc, char **argv)
{
    const char *model = opt_str(argc, argv, "--model", NULL);
    if (!model) { fprintf(stderr, "error: --model PATH is required\n"); return 2; }
    int games  = (int)opt_int(argc, argv, "--games", 200);
    int top    = (int)opt_int(argc, argv, "--top", 8);
    int maxply = (int)opt_int(argc, argv, "--max-plies", 300);

    Trunk *trunk = calloc(1, sizeof(Trunk));
    Head *heads = NULL; Hyper *hy = NULL; float *elo = NULL;
    int n = 0, gen = 0;
    if (!load_model(model, trunk, &heads, &hy, &elo, &n, &gen)) { free(trunk); return 1; }
    if (top > n) top = n;

    /* Pick the `top` highest-Elo agents. */
    int *idx = malloc((size_t)n * sizeof(int));
    for (int i = 0; i < n; i++) idx[i] = i;
    for (int i = 0; i < top; i++)
        for (int j2 = i + 1; j2 < n; j2++)
            if (elo[idx[j2]] > elo[idx[i]]) { int t = idx[i]; idx[i] = idx[j2]; idx[j2] = t; }

    int per_pair = games / (top * (top - 1));
    if (per_pair < 1) per_pair = 1;

    EvalJob job;
    job.trunk = trunk; job.heads = heads; job.hy = hy; job.idx = idx;
    job.k = top; job.games = per_pair; job.maxply = maxply; job.seed = 0xA11CEu;
    job.score = calloc((size_t)(top * top), sizeof(int));

    printf("round robin: top %d agents from %s (generation %d), %d games per ordered pair\n\n",
           top, model, gen, per_pair);
    eval_main(&job);

    printf("       ");
    for (int b = 0; b < top; b++) printf("  a%-3d", idx[b]);
    printf("   total   Elo\n");
    for (int a = 0; a < top; a++) {
        printf("a%-4d ", idx[a]);
        int tot = 0;
        for (int b = 0; b < top; b++) {
            if (a == b) { printf("    - "); continue; }
            printf(" %4.1f ", job.score[a * top + b] / 2.0);
            tot += job.score[a * top + b];
        }
        printf("  %5.1f  %5.0f\n", tot / 2.0, (double)elo[idx[a]]);
    }
    printf("\n(scores are points out of %d per opponent; 1 = win, 0.5 = draw)\n",
           per_pair);

    free(job.score); free(idx); free(trunk); free(heads); free(hy); free(elo);
    return 0;
}

/* -------------------------------------------------------------------- play */

static int cmd_play(int argc, char **argv)
{
    const char *model = opt_str(argc, argv, "--model", NULL);
    if (!model) { fprintf(stderr, "error: --model PATH is required\n"); return 2; }
    int depth    = (int)opt_int(argc, argv, "--depth", 6);
    int movetime = (int)opt_int(argc, argv, "--movetime", 1000);

    Trunk *trunk = calloc(1, sizeof(Trunk));
    Head *heads = NULL; Hyper *hy = NULL; float *elo = NULL;
    int n = 0, gen = 0;
    if (!load_model(model, trunk, &heads, &hy, &elo, &n, &gen)) { free(trunk); return 1; }
    int ai = (int)opt_int(argc, argv, "--agent", best_agent(elo, n));
    if (ai < 0 || ai >= n) ai = best_agent(elo, n);

    Search s;
    search_init(&s, trunk, &heads[ai], 64);
    s.max_depth = depth; s.movetime_ms = movetime; s.max_nodes = 0;
    s.blunder_rate = (float)opt_num(argc, argv, "--blunder", 0.0);

    Game *g = calloc(1, sizeof(Game));
    game_start(g);

    printf("chessrl play -- agent %d (Elo %.0f) from generation %d\n", ai, (double)elo[ai], gen);
    printf("enter moves in UCI (e2e4, e7e8q). commands: quit, board, fen, go, undo\n\n");

    char line[256];
    for (;;) {
        char fen[128];
        pos_to_fen(&g->pos, fen, sizeof fen);
        if (g->result != GR_ONGOING) {
            printf("game over: %s (%s)\n", result_word(g->result), reason_word(g->reason));
            break;
        }
        printf("%s\n%s to move> ", fen, g->pos.side == WHITE ? "white" : "black");
        fflush(stdout);
        if (!fgets(line, sizeof line, stdin)) break;
        line[strcspn(line, "\r\n")] = 0;

        if (!strcmp(line, "quit") || !strcmp(line, "q")) break;
        if (!strcmp(line, "fen"))  { continue; }
        if (!strcmp(line, "undo")) { game_pop(g); continue; }
        if (!strcmp(line, "board")) {
            for (int r = 7; r >= 0; r--) {
                printf("%d ", r + 1);
                for (int f = 0; f < 8; f++) {
                    int sq = sq_make(f, r), pt = g->pos.board[sq], c = g->pos.color_at[sq];
                    char ch = '.';
                    if (pt != NO_PIECE) {
                        ch = "pnbrqk"[pt];
                        if (c == WHITE) ch = (char)(ch - 32);
                    }
                    printf("%c ", ch);
                }
                printf("\n");
            }
            printf("  a b c d e f g h\n");
            continue;
        }
        if (strcmp(line, "go") != 0) {
            Move m;
            if (!move_from_uci(&g->pos, line, &m)) { printf("illegal move\n"); continue; }
            game_push(g, m);
            game_update_result(g, 0);
            if (g->result != GR_ONGOING) continue;
        }

        double t0 = now_sec();
        Move best = search_best(&s, g);
        double dt = now_sec() - t0;
        if (best == MV_NONE) { printf("engine has no move\n"); break; }
        char uci[6], san[16];
        move_to_uci(best, uci);
        move_to_san(&g->pos, best, san, sizeof san);
        printf("engine: %s (%s)  score %+d cp  depth %d  %llu nodes  %.0f ms\n\n",
               san, uci, s.score_cp, s.depth_reached, (unsigned long long)s.nodes, dt * 1e3);
        game_push(g, best);
        game_update_result(g, 0);
    }

    search_free(&s);
    free(g); free(trunk); free(heads); free(hy); free(elo);
    return 0;
}

/* -------------------------------------------------------------------- help */

static int usage(void)
{
    printf(
"chessrl -- population reinforcement learning for chess\n"
"\n"
"USAGE\n"
"  chessrl <command> [options]\n"
"\n"
"COMMANDS\n"
"  train      run a population RL training session\n"
"    --agents N            agents per generation           (default 256)\n"
"    --gens N              number of generations           (default 300)\n"
"    --games-per-agent N   games each agent plays per gen  (default 16)\n"
"    --threads N           worker threads                  (default %d here)\n"
"    --max-plies N         adjudicate a draw after N plies (default 240)\n"
"    --lr X --trunk-lr X --wd X --clip X\n"
"    --elite F --cull F --hof-every N --hof-pct N\n"
"    --seed N --run NAME --quiet\n"
"\n"
"  bench      measure throughput: full games/sec with the network in the loop\n"
"    --seconds S --threads N --max-plies N --no-record --model PATH\n"
"\n"
"  perft      move generator verification / speed\n"
"    --fen FEN --depth D --divide\n"
"\n"
"  selfplay   play trained agents against each other and print PGN\n"
"    --model PATH --a I --b J --games N --temp T --greedy\n"
"\n"
"  eval       round-robin cross table among the strongest agents\n"
"    --model PATH --games N --top K\n"
"\n"
"  play       play the champion from the terminal (UCI move strings)\n"
"    --model PATH --agent I --depth D --movetime MS --blunder R\n"
"\n"
"NETWORK   %d sparse inputs -> %d accumulator -> %d hidden -> policy(%d) + value\n"
"          %zu trunk parameters shared, %zu per agent\n",
    default_threads(), NF_INPUT, NF_ACC, NF_HID, NF_PDIM,
    (size_t)TRUNK_NPARAM, (size_t)HEAD_NPARAM);
    return 0;
}

/* -------------------------------------------------------------------- main */

int main(int argc, char **argv)
{
    chess_init();

    if (argc < 2) return usage();
    const char *cmd = argv[1];
    if (!strcmp(cmd, "-h") || !strcmp(cmd, "--help") || !strcmp(cmd, "help")) return usage();

    int    rargc = argc - 2;
    char **rargv = argv + 2;

    if (!strcmp(cmd, "train"))    return cmd_train(rargc, rargv);
    if (!strcmp(cmd, "bench"))    return cmd_bench(rargc, rargv);
    if (!strcmp(cmd, "perft"))    return cmd_perft(rargc, rargv);
    if (!strcmp(cmd, "selfplay")) return cmd_selfplay(rargc, rargv);
    if (!strcmp(cmd, "eval"))     return cmd_eval(rargc, rargv);
    if (!strcmp(cmd, "play"))     return cmd_play(rargc, rargv);

    fprintf(stderr, "unknown command '%s'\n\n", cmd);
    usage();
    return 2;
}
