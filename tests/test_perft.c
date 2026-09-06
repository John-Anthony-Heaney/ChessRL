/* test_perft.c -- exact perft verification for the chess core.
 *
 *   cc -O2 -std=c11 -Isrc src/chess.c tests/test_perft.c -o tp -lm
 *   ./tp          # fast set only
 *   ./tp full     # includes the deep (slow) nodes
 *
 * Prints PASS/FAIL per line and exits non-zero if anything failed.
 */

#include "chess.h"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>

typedef struct {
    const char *name;
    const char *fen;
    int         depth;
    uint64_t    nodes;
    int         slow;
} Case;

static const Case CASES[] = {
    /* ---- startpos ---- */
    { "startpos", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 1,        20u, 0 },
    { "startpos", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 2,       400u, 0 },
    { "startpos", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 3,      8902u, 0 },
    { "startpos", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 4,    197281u, 0 },
    { "startpos", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 5,   4865609u, 0 },
    { "startpos", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 6, 119060324u, 1 },

    /* ---- kiwipete ---- */
    { "kiwipete", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 1,        48u, 0 },
    { "kiwipete", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 2,      2039u, 0 },
    { "kiwipete", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 3,     97862u, 0 },
    { "kiwipete", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 4,   4085603u, 0 },
    { "kiwipete", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 5, 193690690u, 1 },

    /* ---- pos3 ---- */
    { "pos3", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 1,        14u, 0 },
    { "pos3", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 2,       191u, 0 },
    { "pos3", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 3,      2812u, 0 },
    { "pos3", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 4,     43238u, 0 },
    { "pos3", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 5,    674624u, 0 },
    { "pos3", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 6,  11030083u, 0 },

    /* ---- pos4 ---- */
    { "pos4", "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", 1,        6u, 0 },
    { "pos4", "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", 2,      264u, 0 },
    { "pos4", "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", 3,     9467u, 0 },
    { "pos4", "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", 4,   422333u, 0 },
    { "pos4", "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", 5, 15833292u, 0 },

    /* ---- pos4 mirrored ---- */
    { "pos4-mirror", "r2q1rk1/pP1p2pp/Q4n2/bbp1p3/Np6/1B3NBn/pPPP1PPP/R3K2R b KQ - 0 1", 1,        6u, 0 },
    { "pos4-mirror", "r2q1rk1/pP1p2pp/Q4n2/bbp1p3/Np6/1B3NBn/pPPP1PPP/R3K2R b KQ - 0 1", 2,      264u, 0 },
    { "pos4-mirror", "r2q1rk1/pP1p2pp/Q4n2/bbp1p3/Np6/1B3NBn/pPPP1PPP/R3K2R b KQ - 0 1", 3,     9467u, 0 },
    { "pos4-mirror", "r2q1rk1/pP1p2pp/Q4n2/bbp1p3/Np6/1B3NBn/pPPP1PPP/R3K2R b KQ - 0 1", 4,   422333u, 0 },
    { "pos4-mirror", "r2q1rk1/pP1p2pp/Q4n2/bbp1p3/Np6/1B3NBn/pPPP1PPP/R3K2R b KQ - 0 1", 5, 15833292u, 0 },

    /* ---- pos5 ---- */
    { "pos5", "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", 1,       44u, 0 },
    { "pos5", "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", 2,     1486u, 0 },
    { "pos5", "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", 3,    62379u, 0 },
    { "pos5", "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", 4,  2103487u, 0 },
    { "pos5", "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", 5, 89941194u, 1 },

    /* ---- pos6 ---- */
    { "pos6", "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10", 1,        46u, 0 },
    { "pos6", "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10", 2,      2079u, 0 },
    { "pos6", "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10", 3,     89890u, 0 },
    { "pos6", "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10", 4,   3894594u, 0 },
    { "pos6", "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10", 5, 164075551u, 1 },
};

#define NCASES ((int)(sizeof(CASES) / sizeof(CASES[0])))

static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

/* Same walk as perft() but without bulk counting at depth 1, i.e. one
 * gen_legal + make_move + unmake_move for every single leaf.  This is the
 * honest "how fast can the core actually step through positions" number that
 * the self-play throughput budget depends on. */
static uint64_t perft_nobulk(Position *p, int depth)
{
    if (depth == 0) return 1;
    Move list[MAX_MOVES];
    int n = gen_legal(p, list);
    uint64_t nodes = 0;
    Undo u;
    for (int i = 0; i < n; i++) {
        make_move(p, list[i], &u);
        nodes += perft_nobulk(p, depth - 1);
        unmake_move(p, list[i], &u);
    }
    return nodes;
}

int main(int argc, char **argv)
{
    const int full = (argc > 1 && strcmp(argv[1], "full") == 0);

    double t_init0 = now_sec();
    chess_init();
    double init_ms = (now_sec() - t_init0) * 1e3;

    int failed = 0, skipped = 0, ran = 0;
    double start_d5_sec = 0.0;
    uint64_t start_d5_nodes = 0;

    printf("perft %s set\n", full ? "FULL" : "fast");
    printf("---------------------------------------------------------------\n");

    for (int i = 0; i < NCASES; i++) {
        const Case *c = &CASES[i];
        if (c->slow && !full) {
            printf("SKIP  %-12s d%d  (slow -- run with \"full\")\n", c->name, c->depth);
            skipped++;
            continue;
        }

        Position p;
        if (!pos_from_fen(&p, c->fen)) {
            printf("FAIL  %-12s d%d  bad FEN: %s\n", c->name, c->depth, c->fen);
            failed++;
            continue;
        }

        /* Verify the FEN survives a round trip before we count anything. */
        char rt[128];
        pos_to_fen(&p, rt, sizeof(rt));
        if (strcmp(rt, c->fen) != 0) {
            printf("FAIL  %-12s d%d  FEN round-trip: got \"%s\"\n", c->name, c->depth, rt);
            failed++;
            continue;
        }

        double t0 = now_sec();
        uint64_t got = perft(&p, c->depth);
        double dt = now_sec() - t0;
        ran++;

        /* The position must be untouched after a full perft walk. */
        char after[128];
        pos_to_fen(&p, after, sizeof(after));
        int intact = (strcmp(after, c->fen) == 0) && (p.key == pos_compute_key(&p));

        if (got == c->nodes && intact) {
            printf("PASS  %-12s d%d  %12llu nodes  %7.3fs  %8.2f Mnps\n",
                   c->name, c->depth, (unsigned long long)got, dt,
                   dt > 0.0 ? (double)got / dt / 1e6 : 0.0);
        } else {
            if (got != c->nodes)
                printf("FAIL  %-12s d%d  expected %llu, got %llu\n", c->name, c->depth,
                       (unsigned long long)c->nodes, (unsigned long long)got);
            else
                printf("FAIL  %-12s d%d  position corrupted by perft walk\n", c->name, c->depth);
            failed++;
        }

        if (strcmp(c->name, "startpos") == 0 && c->depth == 5) {
            start_d5_sec = dt;
            start_d5_nodes = got;
        }
    }

    printf("---------------------------------------------------------------\n");
    if (start_d5_sec > 0.0) {
        printf("startpos d5: %llu nodes in %.4f s  =  %.0f nodes/second (%.2f Mnps)\n",
               (unsigned long long)start_d5_nodes, start_d5_sec,
               (double)start_d5_nodes / start_d5_sec,
               (double)start_d5_nodes / start_d5_sec / 1e6);
    }

    /* Honest, non-bulk-counted throughput: gen_legal + make + unmake per node. */
    {
        Position p;
        pos_from_fen(&p, CASES[0].fen);
        double t0 = now_sec();
        uint64_t nb = perft_nobulk(&p, 5);
        double dt = now_sec() - t0;
        if (nb != 4865609u) {
            printf("FAIL  perft_nobulk startpos d5 = %llu\n", (unsigned long long)nb);
            failed++;
        }
        printf("startpos d5 (no bulk counting, make/unmake every leaf): "
               "%.0f nodes/second (%.2f Mnps)\n",
               (double)nb / dt, (double)nb / dt / 1e6);
    }
    printf("chess_init(): %.2f ms (magic search + tables)\n", init_ms);
    printf("%d run, %d skipped, %d FAILED\n", ran, skipped, failed);
    return failed ? 1 : 0;
}
