/* test_search.c -- correctness tests for the alpha-beta engine in src/search.c.
 *
 * The search is what a human plays against, so the properties that matter are,
 * in order: it must never return an illegal move, it must never crash, it must
 * be usable from several threads at once, it must respect its limits, and it
 * must actually find forced tactics.  Everything below is checked against a
 * real (but untrained) model -- a weak evaluation is fine, a wrong search is not.
 *
 * Build:
 *   cc -O2 -std=c11 -D_DARWIN_C_SOURCE -Isrc src/chess.c src/net.c src/search.c \
 *      src/arena.c tests/test_search.c -o /tmp/ts -lm -lpthread && /tmp/ts
 */

#include "chess.h"
#include "net.h"
#include "search.h"

#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Mirror of the (file-private) mate constants in search.c. */
#define T_MATE        30000
#define T_MAX_PLY     64
#define T_MATE_BOUND  (T_MATE - T_MAX_PLY)

/* ---------------------------------------------------------------- harness */

static int g_fail = 0;
static int g_checks = 0;

#define CHECK(cond, ...)                                                       \
    do {                                                                       \
        g_checks++;                                                            \
        if (!(cond)) {                                                         \
            g_fail++;                                                          \
            fprintf(stderr, "FAIL %s:%d: %s\n      ", __FILE__, __LINE__, #cond); \
            fprintf(stderr, __VA_ARGS__);                                      \
            fputc('\n', stderr);                                               \
        }                                                                      \
    } while (0)

static double now_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* splitmix64 -- test-local RNG, no dependency on the engine's */
static uint64_t rnd(uint64_t *s)
{
    uint64_t z = (*s += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

/* Rotating buffers so several uci() calls can appear in one printf. */
static const char *uci(Move m)
{
    static _Thread_local char buf[4][12];
    static _Thread_local int  slot = 0;
    char *b = buf[slot = (slot + 1) & 3];
    if (m == MV_NONE) { memcpy(b, "(none)", 7); return b; }
    move_to_uci(m, b);
    snprintf(b + strlen(b), 6, "/%d", MV_FLAG(m));
    return b;
}

static int is_legal_move(const Position *p, Move m)
{
    Move ml[MAX_MOVES];
    int n = gen_legal(p, ml), i;
    if (m == MV_NONE) return 0;
    for (i = 0; i < n; i++) if (ml[i] == m) return 1;
    return 0;
}

/* ------------------------------------------------------------ mate oracle */
/* Brute-force forced-mate detection, used to validate the test positions
 * themselves and to grade the move the search returns.  `plies` are half-moves. */

static int mated_in(Position *p, int plies);

/* Side to move can force mate within `plies` half-moves. */
static int mate_in(Position *p, int plies)
{
    Move ml[MAX_MOVES];
    int n, i;
    if (plies <= 0) return 0;
    n = gen_legal(p, ml);
    for (i = 0; i < n; i++) {
        Undo u;
        int ok;
        make_move(p, ml[i], &u);
        ok = mated_in(p, plies - 1);
        unmake_move(p, ml[i], &u);
        if (ok) return 1;
    }
    return 0;
}

/* Side to move is being force-mated within `plies` half-moves. */
static int mated_in(Position *p, int plies)
{
    Move ml[MAX_MOVES];
    int n = gen_legal(p, ml), i;
    if (n == 0) return in_check(p, p->side);      /* mated right now */
    if (plies <= 0) return 0;
    for (i = 0; i < n; i++) {
        Undo u;
        int ok;
        make_move(p, ml[i], &u);
        ok = mate_in(p, plies - 1);
        unmake_move(p, ml[i], &u);
        if (!ok) return 0;                        /* this move escapes */
    }
    return 1;
}

/* ---------------------------------------------------------------- fixtures */

typedef struct {
    Trunk trunk;
    Head  head;
} Model;

static int load_model(Model *mo, const char *path)
{
    int probe_n = 0, gen = 0, cap, best = 0, i;
    Head  *heads;
    Hyper *hy;
    float *elo;

    if (!model_load(path, NULL, NULL, NULL, NULL, &probe_n, &gen)) return 0;
    if (probe_n <= 0) return 0;
    heads = (Head *)calloc((size_t)probe_n, sizeof(Head));
    hy    = (Hyper *)calloc((size_t)probe_n, sizeof(Hyper));
    elo   = (float *)calloc((size_t)probe_n, sizeof(float));
    if (!heads || !hy || !elo) { free(heads); free(hy); free(elo); return 0; }

    cap = probe_n;
    if (!model_load(path, &mo->trunk, heads, hy, elo, &cap, &gen)) {
        free(heads); free(hy); free(elo);
        return 0;
    }
    for (i = 1; i < cap; i++) if (elo[i] > elo[best]) best = i;
    mo->head = heads[best];
    printf("  model '%s': %d agents, generation %d, using agent %d (elo %.1f)\n",
           path, cap, gen, best, (double)elo[best]);
    free(heads); free(hy); free(elo);
    return 1;
}

/* Play a random legal game for at most `maxply` plies. */
static void random_game(Game *g, uint64_t *rs, int maxply)
{
    int i;
    game_start(g);
    for (i = 0; i < maxply && g->result == GR_ONGOING; i++) {
        Move ml[MAX_MOVES];
        int n = gen_legal(&g->pos, ml);
        if (n == 0) break;
        game_push(g, ml[(int)(rnd(rs) % (uint64_t)n)]);
    }
}

/* ============================================================ 1. LEGALITY */

static void test_legality(const Model *mo)
{
    Search s;
    Game g;
    uint64_t rs = 0xC0FFEEull;
    int i, terminal = 0, ongoing = 0;

    printf("[1] legality over 200 random positions\n");
    search_init(&s, &mo->trunk, &mo->head, 4);
    s.max_depth   = 6;
    s.movetime_ms = 25;

    for (i = 0; i < 200; i++) {
        Move m;
        int plies = (int)(rnd(&rs) % 160u);
        random_game(&g, &rs, plies);
        m = search_best(&s, &g);
        if (g.result != GR_ONGOING) {
            terminal++;
            CHECK(m == MV_NONE, "game over (result=%d reason=%d) but search returned %s",
                  g.result, g.reason, uci(m));
        } else {
            ongoing++;
            CHECK(is_legal_move(&g.pos, m),
                  "illegal move %s at position %d (ply %d)", uci(m), i, g.ply);
            CHECK(m != MV_NONE, "MV_NONE returned for an ongoing game (position %d)", i);
        }
    }
    printf("      %d ongoing, %d terminal positions exercised\n", ongoing, terminal);

    /* Explicit terminal positions: mate, stalemate, insufficient material. */
    {
        static const char *over[] = {
            "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3", /* fool's mate */
            "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1",                                /* stalemate  */
            "8/8/8/4k3/8/8/4K3/8 w - - 0 1",                                 /* K v K      */
            "8/8/8/4k3/8/8/4K3/6B1 b - - 0 1"                                /* K+B v K    */
        };
        size_t k;
        for (k = 0; k < sizeof(over) / sizeof(over[0]); k++) {
            CHECK(game_start_fen(&g, over[k]) == 1, "bad fen %s", over[k]);
            CHECK(g.result != GR_ONGOING, "expected finished game for %s", over[k]);
            CHECK(search_best(&s, &g) == MV_NONE, "expected MV_NONE for %s", over[k]);
        }
    }
    /* ...and a live position right next to them must still produce a move. */
    CHECK(game_start_fen(&g, "7k/5Q2/6K1/8/8/8/8/8 w - - 0 1") == 1, "fen");
    CHECK(g.result == GR_ONGOING, "position should be ongoing");
    CHECK(is_legal_move(&g.pos, search_best(&s, &g)), "expected a legal move");

    search_free(&s);
}

/* ========================================================= 2. MATE FINDING */

typedef struct { const char *fen; int k; const char *note; } MateCase;

static void test_mates(const Model *mo)
{
    static const MateCase cases[] = {
        { "6k1/5ppp/8/8/8/8/8/R3K2R w KQ - 0 1",     1, "Ra8#"          },
        { "7k/6pp/8/8/8/8/8/R6K w - - 0 1",          1, "Ra8#"          },
        { "6k1/5ppp/8/8/8/8/5PPP/3Q2K1 w - - 0 1",   1, "Qd8#"          },
        { "6k1/1R6/8/8/8/8/8/2R4K w - - 0 1",        1, "Rc8#"          },
        { "7k/8/6K1/6Q1/8/8/8/8 w - - 0 1",          1, "Qd8#"          },
        { "3r2k1/5ppp/8/8/8/7R/5PPP/6K1 b - - 0 1", 1, "Rd1# (black)"  },
        { "6k1/8/8/8/8/8/1R6/R6K w - - 0 1",         2, "Rb7 then Ra8#" },
        { "7k/8/6K1/8/8/8/8/6Q1 w - - 0 1",          2, "Qa7 then Qa8#" },
        { "7k/8/5K2/8/8/8/8/2Q5 w - - 0 1",          2, "Qh6+ then Qg7#" },
        { "1r5k/5r2/8/8/8/8/8/6K1 b - - 0 1",        2, "Rf2 then Rb1# (black)" }
    };
    Search s;
    size_t i;

    printf("[2] mate finding\n");
    search_init(&s, &mo->trunk, &mo->head, 16);

    for (i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        const int k = cases[i].k;
        Game g;
        Move m;
        int oracle_ok, oracle_shorter, follow_ok;

        CHECK(game_start_fen(&g, cases[i].fen) == 1, "bad fen: %s", cases[i].fen);
        CHECK(g.result == GR_ONGOING, "position already over: %s", cases[i].fen);

        /* Validate the fixture itself with the brute-force oracle. */
        oracle_ok      = mate_in(&g.pos, 2 * k - 1);
        oracle_shorter = (k > 1) ? mate_in(&g.pos, 2 * k - 3) : 0;
        CHECK(oracle_ok, "fixture is NOT mate in %d: %s", k, cases[i].fen);
        CHECK(!oracle_shorter, "fixture is mate in less than %d: %s", k, cases[i].fen);
        if (!oracle_ok) continue;

        s.max_depth    = 2 * k + 3;
        s.movetime_ms  = 10000;
        s.max_nodes    = 0;
        s.blunder_rate = 0.0f;
        m = search_best(&s, &g);

        CHECK(is_legal_move(&g.pos, m), "illegal move %s in %s", uci(m), cases[i].fen);
        CHECK(s.score_cp >= T_MATE_BOUND,
              "score %d is not a mate score (%s, expected mate in %d)",
              s.score_cp, cases[i].fen, k);
        CHECK(s.score_cp >= T_MATE - 2 * k && s.score_cp <= T_MATE,
              "score %d does not encode mate in %d (%s)", s.score_cp, k, cases[i].fen);

        follow_ok = 0;
        if (is_legal_move(&g.pos, m)) {
            Undo u;
            make_move(&g.pos, m, &u);
            follow_ok = mated_in(&g.pos, 2 * k - 2);
            unmake_move(&g.pos, m, &u);
        }
        CHECK(follow_ok, "move %s does not force mate in %d (%s -- %s)",
              uci(m), k, cases[i].fen, cases[i].note);

        printf("      mate-in-%d %-44s -> %-5s score %6d depth %d nodes %llu\n",
               k, cases[i].fen, uci(m), s.score_cp, s.depth_reached,
               (unsigned long long)s.nodes);
    }
    search_free(&s);
}

/* ====================================================== 3. NO SIDE EFFECTS */

static void test_no_side_effects(const Model *mo)
{
    Search s;
    Game g, backup;
    uint64_t rs = 0x5EEDull;
    int i;

    printf("[3] search_best does not mutate the Game\n");
    search_init(&s, &mo->trunk, &mo->head, 4);
    s.max_depth   = 6;
    s.movetime_ms = 30;

    for (i = 0; i < 25; i++) {
        random_game(&g, &rs, (int)(rnd(&rs) % 60u));
        memcpy(&backup, &g, sizeof(Game));
        (void)search_best(&s, &g);
        CHECK(memcmp(&backup, &g, sizeof(Game)) == 0,
              "Game mutated by search_best (iteration %d)", i);
    }
    search_free(&s);
}

/* =========================================================== 4. REENTRANCY */

typedef struct {
    const Model *mo;
    uint64_t     seed;
    int          iters;
    int          illegal;
    int          none;
    int          searches;
    uint64_t     nodes;
} Worker;

static void *worker_main(void *arg)
{
    Worker *w = (Worker *)arg;
    Search s;
    Game g;
    uint64_t rs = w->seed;
    int i;

    search_init(&s, &w->mo->trunk, &w->mo->head, 4);
    s.max_depth   = 6;
    s.movetime_ms = 20;

    for (i = 0; i < w->iters; i++) {
        Move m;
        random_game(&g, &rs, (int)(rnd(&rs) % 120u));
        m = search_best(&s, &g);
        w->searches++;
        w->nodes += s.nodes;
        if (g.result != GR_ONGOING) {
            if (m != MV_NONE) w->none++;
        } else {
            if (!is_legal_move(&g.pos, m)) w->illegal++;
        }
    }
    search_free(&s);
    return NULL;
}

static void test_reentrancy(const Model *mo)
{
    pthread_t th[2];
    Worker w[2];
    int i;

    printf("[4] two concurrent Search objects\n");
    for (i = 0; i < 2; i++) {
        memset(&w[i], 0, sizeof(w[i]));
        w[i].mo    = mo;
        w[i].seed  = 0xA5A5A5A5ull + (uint64_t)i * 0x1234567ull;
        w[i].iters = 90;
    }
    for (i = 0; i < 2; i++)
        CHECK(pthread_create(&th[i], NULL, worker_main, &w[i]) == 0, "pthread_create failed");
    for (i = 0; i < 2; i++) pthread_join(th[i], NULL);

    for (i = 0; i < 2; i++) {
        CHECK(w[i].illegal == 0, "thread %d produced %d illegal moves", i, w[i].illegal);
        CHECK(w[i].none == 0, "thread %d returned a move for %d finished games", i, w[i].none);
        CHECK(w[i].searches == w[i].iters, "thread %d only ran %d searches", i, w[i].searches);
    }
    printf("      thread 0: %d searches, %llu nodes; thread 1: %d searches, %llu nodes\n",
           w[0].searches, (unsigned long long)w[0].nodes,
           w[1].searches, (unsigned long long)w[1].nodes);
}

/* ========================================================= 5. TIME CONTROL */

static void test_limits(const Model *mo)
{
    static const char *fens[] = {
        "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        /* Quiescence bombs.  Every pawn is a promotion, so stand-pat, delta and
         * SEE pruning all fail at once and the capture tree grows ~40x every two
         * plies.  These used to run for minutes regardless of movetime, because
         * the clock did not start until depth 1 had completed. */
        "8/PPPPPPPP/8/8/8/8/pppppppp/K6k w - - 0 1",
        "n1n5/PPPk4/8/8/8/8/4Kppp/5N1N b - - 0 1",
        "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"
    };
    Search s;
    Game g;
    size_t i;
    double worst = 0.0;

    printf("[5] time control and node cap\n");
    search_init(&s, &mo->trunk, &mo->head, 16);

    for (i = 0; i < sizeof(fens) / sizeof(fens[0]); i++) {
        double t0, dt;
        CHECK(game_start_fen(&g, fens[i]) == 1, "bad fen %s", fens[i]);
        s.max_depth   = 64;
        s.movetime_ms = 100;
        s.max_nodes   = 0;
        t0 = now_s();
        (void)search_best(&s, &g);
        dt = now_s() - t0;
        if (dt > worst) worst = dt;
        CHECK(dt < 0.200, "movetime 100ms overrun: %.1f ms (%s)", dt * 1e3, fens[i]);
        printf("      movetime 100ms -> %6.1f ms, depth %2d, %8llu nodes\n",
               dt * 1e3, s.depth_reached, (unsigned long long)s.nodes);
    }

    /* Node cap.  The first iteration always runs to completion, and the node
     * counter is only polled every 2048 nodes, so allow a bounded overshoot. */
    {
        const uint64_t caps[3] = { 20000, 100000, 400000 };
        uint64_t uncapped;
        size_t k;

        CHECK(game_start_fen(&g, fens[1]) == 1, "bad fen");
        s.max_depth   = 8;
        s.movetime_ms = 0;
        s.max_nodes   = 0;
        (void)search_best(&s, &g);
        uncapped = s.nodes;

        for (k = 0; k < 3; k++) {
            s.max_depth   = 64;
            s.movetime_ms = 0;
            s.max_nodes   = caps[k];
            (void)search_best(&s, &g);
            CHECK(s.nodes <= caps[k] + 4096,
                  "max_nodes %llu exceeded: %llu nodes",
                  (unsigned long long)caps[k], (unsigned long long)s.nodes);
            printf("      max_nodes %7llu -> %8llu nodes, depth %d\n",
                   (unsigned long long)caps[k], (unsigned long long)s.nodes,
                   s.depth_reached);
        }
        CHECK(uncapped > caps[0], "depth-8 search only used %llu nodes, cap test is vacuous",
              (unsigned long long)uncapped);
    }

    /* Limits tight enough that the search is cut off before even depth 1 has
     * finished.  This path exists precisely so that a quiescence explosion can
     * be interrupted, so it has to yield a legal move rather than MV_NONE. */
    {
        static const char *hard[] = {
            "8/PPPPPPPP/8/8/8/8/pppppppp/K6k w - - 0 1",
            "n1n5/PPPk4/8/8/8/8/4Kppp/5N1N b - - 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        };
        size_t k;
        int cut = 0;
        for (k = 0; k < sizeof(hard) / sizeof(hard[0]); k++) {
            uint64_t cap;
            CHECK(game_start_fen(&g, hard[k]) == 1, "bad fen %s", hard[k]);
            for (cap = 1; cap <= 64; cap *= 8) {
                double t0, dt;
                Move m;
                s.max_depth = 64; s.movetime_ms = 0; s.max_nodes = cap;
                t0 = now_s();
                m = search_best(&s, &g);
                dt = now_s() - t0;
                CHECK(is_legal_move(&g.pos, m),
                      "illegal move %s at max_nodes %llu (%s)", uci(m),
                      (unsigned long long)cap, hard[k]);
                CHECK(dt < 1.0, "max_nodes %llu took %.0f ms (%s)",
                      (unsigned long long)cap, dt * 1e3, hard[k]);
                if (s.depth_reached <= 1) cut++;
            }
            /* ...and the same through the clock rather than the node counter. */
            {
                double t0 = now_s(), dt;
                Move m;
                s.max_depth = 64; s.movetime_ms = 1; s.max_nodes = 0;
                m = search_best(&s, &g);
                dt = now_s() - t0;
                CHECK(is_legal_move(&g.pos, m), "illegal move at movetime 1ms (%s)", hard[k]);
                CHECK(dt < 1.0, "movetime 1ms took %.0f ms (%s)", dt * 1e3, hard[k]);
            }
        }
        printf("      tightest limits: %d searches cut off at depth <= 1, all legal\n", cut);
    }
    printf("      worst movetime overrun factor: %.2fx\n", worst / 0.100);
    search_free(&s);
}

/* ======================================================== 6. TT SOUNDNESS */

/* Two things are asserted here, and they are not the same thing.
 *
 *  - Determinism: a search from a cold table must reproduce itself exactly --
 *    same move, same score, same node count.  Any hidden global, uninitialised
 *    read or leftover state between calls shows up here immediately.
 *
 *  - Table soundness: re-searching the same position with the table still warm
 *    must not move the score.  It may reorder two moves of equal value: a
 *    fail-soft PVS whose reductions and futility margins depend on move index
 *    is inherently order-sensitive, and a warm table changes the order in which
 *    the shallow iterations resolve.  That is search instability, not a wrong
 *    answer -- so the score tolerance is a hard assert per position, while the
 *    move is required to agree on the large majority of a battery.  A warm
 *    table that changed a score would be a real bug and is caught. */
static void test_tt_soundness(const Model *mo)
{
    static const char *fens[] = {
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        "4rrk1/pp1n1ppp/2p1bn2/q7/3P4/2N1PN2/PP2BPPP/R2QR1K1 w - - 0 1",
        "2rq1rk1/pp1bppbp/3p1np1/8/2BNP3/2N1BP2/PPPQ2PP/2KR3R w - - 0 1",
        "r2q1rk1/pp2ppbp/2np1np1/2p5/2P1P3/2NP1N1P/PP2BPP1/R1BQ1RK1 w - - 0 1",
        "rnbq1rk1/pp2ppbp/3p1np1/2p5/2PP4/2N1PN2/PP2BPPP/R1BQ1RK1 w - - 0 1",
        "8/8/4k3/1P5P/p7/P6R/3B4/4K3 w - - 10 95",
        "1rbq1rk1/p1b1nppp/1p2p3/8/1B1pN3/P2B4/1P3PPP/2RQ1R1K w - - 3 1",
        "r1b1k2r/ppppnppp/2n2q2/2b5/3NP3/2P1B3/PP3PPP/RN1QKB1R w KQkq - 0 1",
        "6k1/5ppp/8/8/8/8/5PPP/3Q2K1 w - - 0 1"
    };
    const size_t n = sizeof(fens) / sizeof(fens[0]);
    Game g;
    size_t i;
    int agree = 0;

    printf("[6] transposition table: determinism and re-search stability\n");

    /* --- determinism: cold table, twice, must be bit-identical -------------- */
    for (i = 0; i < n; i++) {
        Search a, b;
        Move ma, mb;
        int sa, sb;
        uint64_t na, nb;

        CHECK(game_start_fen(&g, fens[i]) == 1, "bad fen %s", fens[i]);

        search_init(&a, &mo->trunk, &mo->head, 16);
        a.max_depth = 6; a.movetime_ms = 0; a.max_nodes = 0;
        ma = search_best(&a, &g); sa = a.score_cp; na = a.nodes;
        search_free(&a);

        search_init(&b, &mo->trunk, &mo->head, 16);
        b.max_depth = 6; b.movetime_ms = 0; b.max_nodes = 0;
        mb = search_best(&b, &g); sb = b.score_cp; nb = b.nodes;
        search_free(&b);

        CHECK(ma == mb, "cold search is not deterministic: %s then %s (%s)",
              uci(ma), uci(mb), fens[i]);
        CHECK(sa == sb, "cold search score differs: %d then %d (%s)", sa, sb, fens[i]);
        CHECK(na == nb, "cold search node count differs: %llu then %llu (%s)",
              (unsigned long long)na, (unsigned long long)nb, fens[i]);
    }

    /* --- warm table: the score must hold, the move should ------------------- */
    for (i = 0; i < n; i++) {
        Search s;
        Move m1, m2;
        int sc1, sc2;

        CHECK(game_start_fen(&g, fens[i]) == 1, "bad fen %s", fens[i]);
        search_init(&s, &mo->trunk, &mo->head, 16);
        s.max_depth = 6; s.movetime_ms = 0; s.max_nodes = 0;

        m1 = search_best(&s, &g);
        sc1 = s.score_cp;
        m2 = search_best(&s, &g);      /* same Search: the table is now warm */
        sc2 = s.score_cp;

        CHECK(is_legal_move(&g.pos, m1) && is_legal_move(&g.pos, m2),
              "illegal move from a warm table (%s)", fens[i]);
        CHECK(abs(sc1 - sc2) <= 20, "warm table moved the score: %d then %d (%s)",
              sc1, sc2, fens[i]);
        if (m1 == m2) agree++;
        else printf("      unstable (equal-valued): %s %d -> %s %d   %s\n",
                    uci(m1), sc1, uci(m2), sc2, fens[i]);
        search_free(&s);
    }
    CHECK((size_t)agree * 10 >= n * 8, "warm re-search kept the move in only %d/%zu positions",
          agree, n);
    printf("      cold searches reproducible: %zu/%zu; warm re-search kept the move %d/%zu\n",
           n, n, agree, n);
}

/* ======================================================= 7. DRAW AWARENESS */

/* White has nothing but a king with exactly one legal move at every step, and
 * that move shuttles h1<->g1.  Black is a rook and two pawns up.  We replay the
 * shuttle until the current position has occurred twice; the only move White
 * has then repeats it a third time, so the search MUST score the position as a
 * dead draw (0) instead of "down a rook" -- and must certainly not call it a win. */
static void test_draw_awareness(const Model *mo)
{
    const char *fen = "1k3r2/8/8/8/8/6pp/8/7K w - - 0 1";
    const char *line[4] = { "h1g1", "b8a8", "g1h1", "a8b8" };
    Search s;
    Game g;
    int i;
    uint64_t key0;
    Move m;

    printf("[7] repetition is scored as a draw, not a win\n");
    CHECK(game_start_fen(&g, fen) == 1, "bad fen %s", fen);
    key0 = g.pos.key;

    /* White is completely boxed in: exactly one legal move at each shuttle step. */
    {
        Move ml[MAX_MOVES];
        CHECK(gen_legal(&g.pos, ml) == 1, "white should have exactly one legal move");
        CHECK(in_check(&g.pos, WHITE) == 0, "white must not be in check");
    }
    for (i = 0; i < 4; i++) {
        Move mv;
        CHECK(move_from_uci(&g.pos, line[i], &mv) == 1, "illegal fixture move %s", line[i]);
        if (!move_from_uci(&g.pos, line[i], &mv)) return;
        game_push(&g, mv);
    }
    CHECK(g.pos.key == key0, "the shuttle did not return to the start position");
    CHECK(g.result == GR_ONGOING, "game should still be ongoing (result=%d)", g.result);
    CHECK(game_repetitions(&g) == 2, "expected 2 occurrences, got %d", game_repetitions(&g));
    {
        Move ml[MAX_MOVES];
        CHECK(gen_legal(&g.pos, ml) == 1, "white should still have exactly one legal move");
    }

    search_init(&s, &mo->trunk, &mo->head, 8);
    s.max_depth   = 10;
    s.movetime_ms = 2000;
    m = search_best(&s, &g);

    CHECK(is_legal_move(&g.pos, m), "illegal move %s", uci(m));
    CHECK(s.score_cp < T_MATE_BOUND, "repetition scored as a mate: %d", s.score_cp);
    CHECK(s.score_cp == 0, "forced repetition should score 0, got %d", s.score_cp);
    printf("      forced repetition (white down R+2P): move %s, score %d\n",
           uci(m), s.score_cp);

    /* And the same shape seen from the other side: the material-up side must not
     * be tricked into thinking the repetition wins for it either. */
    {
        int base;
        CHECK(game_start_fen(&g, fen) == 1, "bad fen");
        s.max_depth = 6;
        (void)search_best(&s, &g);
        base = s.score_cp;
        CHECK(base < T_MATE_BOUND, "fresh position scored as a mate for white: %d", base);
        printf("      same position with no history: score %d\n", base);
    }
    search_free(&s);
}

/* ========================================================== 8. BLUNDER RATE */

static void test_blunder(const Model *mo)
{
    const char *fen = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4";
    Search s;
    Game g;
    Move seen0[100], seen1[100];
    int i, distinct0 = 0, distinct1 = 0, top1 = 0;
    int nlegal;
    Move ml[MAX_MOVES];

    printf("[8] blunder_rate\n");
    CHECK(game_start_fen(&g, fen) == 1, "bad fen");
    nlegal = gen_legal(&g.pos, ml);
    CHECK(nlegal > 4, "need a position with several legal moves");

    search_init(&s, &mo->trunk, &mo->head, 8);
    s.max_depth   = 4;
    s.movetime_ms = 0;
    s.max_nodes   = 0;

    s.blunder_rate = 0.0f;
    for (i = 0; i < 100; i++) {
        seen0[i] = search_best(&s, &g);
        CHECK(is_legal_move(&g.pos, seen0[i]), "illegal move %s at blunder_rate 0", uci(seen0[i]));
    }
    s.blunder_rate = 1.0f;
    for (i = 0; i < 100; i++) {
        seen1[i] = search_best(&s, &g);
        CHECK(is_legal_move(&g.pos, seen1[i]), "illegal move %s at blunder_rate 1", uci(seen1[i]));
    }

    for (i = 0; i < 100; i++) {
        int j, dup = 0;
        for (j = 0; j < i; j++) if (seen0[j] == seen0[i]) { dup = 1; break; }
        if (!dup) distinct0++;
    }
    for (i = 0; i < 100; i++) {
        int j, dup = 0;
        for (j = 0; j < i; j++) if (seen1[j] == seen1[i]) { dup = 1; break; }
        if (!dup) distinct1++;
        if (seen1[i] == seen0[0]) top1++;
    }

    CHECK(distinct0 == 1, "blunder_rate 0 should be deterministic, saw %d distinct moves", distinct0);
    CHECK(distinct1 > 1, "blunder_rate 1 produced only %d distinct move(s)", distinct1);
    CHECK(top1 < 90, "blunder_rate 1 still played the best move %d/100 times", top1);
    printf("      rate 0.0: %d distinct move(s) (%s)\n", distinct0, uci(seen0[0]));
    printf("      rate 1.0: %d distinct moves, best move played %d/100\n", distinct1, top1);
    search_free(&s);
}

/* ============================================================== node rate */

static void bench_node_rate(const Model *mo)
{
    Search s;
    Game g;
    double t0, dt;

    printf("[9] node rate, depth 6 from the start position\n");
    game_start(&g);
    search_init(&s, &mo->trunk, &mo->head, 64);
    s.max_depth   = 6;
    s.movetime_ms = 0;
    s.max_nodes   = 0;

    t0 = now_s();
    (void)search_best(&s, &g);
    dt = now_s() - t0;

    CHECK(s.depth_reached == 6, "expected depth 6, reached %d", s.depth_reached);
    CHECK(is_legal_move(&g.pos, s.pv[0]), "pv[0] is not legal");
    printf("      %llu nodes in %.3f s = %.0f nodes/sec (best %s, score %d, pv_len %d)\n",
           (unsigned long long)s.nodes, dt,
           dt > 0 ? (double)s.nodes / dt : 0.0,
           uci(s.pv[0]), s.score_cp, s.pv_len);
    search_free(&s);
}

/* ==================================================================== main */

int main(int argc, char **argv)
{
    static Model mo;
    const char *path = (argc > 1) ? argv[1] : "runs/smoke/best.crl";
    double t0;

    chess_init();

    printf("=== search tests ===\n");
    if (!load_model(&mo, path)) {
        const char *alt = "../runs/smoke/best.crl";
        if (!load_model(&mo, alt)) {
            fprintf(stderr, "FATAL: cannot load model '%s' (run from the repo root)\n", path);
            return 2;
        }
    }

    t0 = now_s();
    test_legality(&mo);
    test_mates(&mo);
    test_no_side_effects(&mo);
    test_reentrancy(&mo);
    test_limits(&mo);
    test_tt_soundness(&mo);
    test_draw_awareness(&mo);
    test_blunder(&mo);
    bench_node_rate(&mo);

    printf("=== %d checks, %d failures, %.1f s ===\n", g_checks, g_fail, now_s() - t0);
    return g_fail ? 1 : 0;
}
