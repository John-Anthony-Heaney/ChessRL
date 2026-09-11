/* test_rules.c -- correctness tests for the chess core that perft cannot see.
 *
 *   cc -O2 -std=c11 -Isrc src/chess.c tests/test_rules.c -o tr -lm && ./tr
 *
 * Covers: FEN round-trip, incremental-vs-recomputed zobrist across long random
 * playouts (and across unmake), the en-passant horizontal-pin case, castling
 * legality and rights bookkeeping, underpromotion mate, stalemate, fifty-move
 * and threefold draws, insufficient material, and SAN.
 * Exits non-zero if anything fails.
 */

#include "chess.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_fail = 0;
static int g_checks = 0;

#define CHECK(cond, ...)                                                       \
    do {                                                                       \
        g_checks++;                                                            \
        if (!(cond)) {                                                         \
            g_fail++;                                                          \
            printf("  FAIL [%s:%d] ", __FILE__, __LINE__);                     \
            printf(__VA_ARGS__);                                               \
            printf("\n");                                                      \
        }                                                                      \
    } while (0)

static void section(const char *name) { printf("-- %s\n", name); }
static void done(const char *name, int before)
{
    printf("%s  %s (%d checks)\n", (g_fail == before) ? "PASS" : "FAIL",
           name, g_checks);
}

/* ------------------------------------------------------------- helpers */

static int uci_legal(const Position *p, const char *u)
{
    Move m;
    return move_from_uci(p, u, &m);
}

static Move uci_move(const Position *p, const char *u)
{
    Move m = MV_NONE;
    move_from_uci(p, u, &m);
    return m;
}

/* SAN for a UCI move; returns "<illegal>" when the move is not legal. */
static const char *san_of(const Position *p, const char *u, char *buf, size_t n)
{
    Move m;
    if (!move_from_uci(p, u, &m)) { snprintf(buf, n, "<illegal>"); return buf; }
    move_to_san(p, m, buf, n);
    return buf;
}

static int has_flag(const Position *p, int flag)
{
    Move list[MAX_MOVES];
    int n = gen_legal(p, list);
    for (int i = 0; i < n; i++)
        if (MV_FLAG(list[i]) == flag) return 1;
    return 0;
}

/* xorshift128+ -- test-local, deterministic. */
static uint64_t rs[2] = { 0x853C49E6748FEA9BULL, 0xDA3E39CB94B95BDBULL };
static uint64_t rnd(void)
{
    uint64_t x = rs[0], y = rs[1];
    rs[0] = y;
    x ^= x << 23;
    rs[1] = x ^ y ^ (x >> 17) ^ (y >> 26);
    return rs[1] + y;
}

/* --------------------------------------------------------- 1. FEN ----- */

static const char *FENS[] = {
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
    "r2q1rk1/pP1p2pp/Q4n2/bbp1p3/Np6/1B3NBn/pPPP1PPP/R3K2R b KQ - 0 1",
    "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
    "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
    "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
    "8/8/8/8/k2Pp2Q/8/8/3K4 b - d3 0 1",
    "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1",
    "2rqn3/2pkpP2/2pp4/8/8/8/8/4K3 w - - 0 1",
    "8/8/8/4k3/8/8/8/4K3 w - - 12 34",
};

static void test_fen(void)
{
    int before = g_fail;
    section("FEN round-trip");

    for (size_t i = 0; i < sizeof(FENS) / sizeof(FENS[0]); i++) {
        Position p;
        char out[128];
        CHECK(pos_from_fen(&p, FENS[i]), "pos_from_fen rejected \"%s\"", FENS[i]);
        pos_to_fen(&p, out, sizeof(out));
        CHECK(strcmp(out, FENS[i]) == 0, "round-trip:\n    in  \"%s\"\n    out \"%s\"",
              FENS[i], out);
        CHECK(p.key == pos_compute_key(&p), "key mismatch for \"%s\"", FENS[i]);
    }

    /* 4-field and 5-field FENs must be accepted, with sane clock defaults. */
    {
        Position p4, p5, p6;
        CHECK(pos_from_fen(&p4, "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -"),
              "4-field FEN rejected");
        CHECK(p4.halfmove == 0 && p4.fullmove == 1, "4-field clocks: %u %u",
              p4.halfmove, p4.fullmove);

        CHECK(pos_from_fen(&p5, "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 7"),
              "5-field FEN rejected");
        CHECK(p5.halfmove == 7 && p5.fullmove == 1, "5-field clocks: %u %u",
              p5.halfmove, p5.fullmove);

        CHECK(pos_from_fen(&p6, "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 3 9"),
              "6-field FEN rejected");
        CHECK(p6.halfmove == 3 && p6.fullmove == 9, "6-field clocks: %u %u",
              p6.halfmove, p6.fullmove);

        char b[128];
        pos_to_fen(&p4, b, sizeof(b));
        CHECK(strcmp(b, "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1") == 0,
              "4-field FEN emitted as \"%s\"", b);
    }

    /* Malformed FENs must be rejected. */
    {
        Position p;
        CHECK(!pos_from_fen(&p, ""),                          "empty FEN accepted");
        CHECK(!pos_from_fen(&p, "8/8/8/8/8/8/8 w - - 0 1"),    "7-rank FEN accepted");
        CHECK(!pos_from_fen(&p, "9/8/8/8/8/8/8/8 w - - 0 1"),  "rank overflow accepted");
        CHECK(!pos_from_fen(&p, "8/8/8/8/8/8/8/8 x - - 0 1"),  "bad side accepted");
        CHECK(!pos_from_fen(&p, "8/8/8/8/8/8/8/8 w KZ - 0 1"), "bad castling accepted");
        CHECK(!pos_from_fen(&p, "8/8/8/8/8/8/8/8 w - z9 0 1"), "bad ep accepted");
        CHECK(!pos_from_fen(&p, "8/8/8/8/8/8/8/8X w - - 0 1"), "bad piece char accepted");
    }

    /* pos_to_fen must respect a short buffer. */
    {
        Position p;
        char small[8];
        pos_from_fen(&p, FENS[0]);
        pos_to_fen(&p, small, sizeof(small));
        CHECK(strlen(small) == 7, "short-buffer FEN length %zu", strlen(small));
    }

    done("FEN", before);
}

/* ------------------------------------------------- 2. zobrist / undo -- */

/* Plays `plies` random moves from `fen`, checking after every make that the
 * incrementally maintained key equals pos_compute_key(), and that make+unmake
 * of every legal move restores the Position bit-for-bit. */
static void playout(const char *fen, int plies)
{
    Position p;
    if (!pos_from_fen(&p, fen)) { CHECK(0, "playout FEN rejected: %s", fen); return; }

    for (int ply = 0; ply < plies; ply++) {
        Move list[MAX_MOVES];
        int n = gen_legal(&p, list);
        if (n == 0) break;

        /* every legal move must be perfectly reversible */
        for (int i = 0; i < n; i++) {
            Position snap;
            Undo u;
            memcpy(&snap, &p, sizeof(Position));
            make_move(&p, list[i], &u);
            if (p.key != pos_compute_key(&p)) {
                char b[128], uc[6];
                pos_to_fen(&snap, b, sizeof(b));
                move_to_uci(list[i], uc);
                CHECK(0, "incremental key != computed after %s from \"%s\"", uc, b);
                return;
            }
            unmake_move(&p, list[i], &u);
            if (memcmp(&snap, &p, sizeof(Position)) != 0) {
                char b[128], uc[6];
                pos_to_fen(&snap, b, sizeof(b));
                move_to_uci(list[i], uc);
                CHECK(0, "unmake did not restore position after %s from \"%s\"", uc, b);
                return;
            }
        }

        /* null moves must be reversible too */
        {
            Position snap;
            Undo u;
            memcpy(&snap, &p, sizeof(Position));
            make_null(&p, &u);
            if (p.key != pos_compute_key(&p)) { CHECK(0, "null-move key mismatch"); return; }
            unmake_null(&p, &u);
            if (memcmp(&snap, &p, sizeof(Position)) != 0) {
                CHECK(0, "unmake_null did not restore position");
                return;
            }
        }

        Undo u;
        make_move(&p, list[rnd() % (uint64_t)n], &u);
        if (p.halfmove >= 100) break;
    }
}

static void test_zobrist(void)
{
    int before = g_fail;
    section("zobrist + make/unmake reversibility (random playouts)");

    static const char *seeds[] = {
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
        "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
    };
    const int nseeds = (int)(sizeof(seeds) / sizeof(seeds[0]));

    for (int rep = 0; rep < 24 && g_fail == before; rep++)
        for (int i = 0; i < nseeds && g_fail == before; i++)
            playout(seeds[i], 160);

    CHECK(g_fail == before, "random playouts found a zobrist/undo inconsistency");

    /* Game-level: play a full random game, then pop every ply back. */
    {
        Game g;
        game_start(&g);
        Position start;
        memcpy(&start, &g.pos, sizeof(Position));
        while (g.result == GR_ONGOING && g.ply < 400) {
            Move list[MAX_MOVES];
            int n = gen_legal(&g.pos, list);
            if (n == 0) break;
            game_push(&g, list[rnd() % (uint64_t)n]);
            CHECK(g.pos.key == pos_compute_key(&g.pos), "game_push key mismatch at ply %d", g.ply);
            CHECK(g.hist[g.ply] == g.pos.key, "history key mismatch at ply %d", g.ply);
            CHECK(g.hist_len == g.ply + 1, "hist_len %d != ply+1 %d", g.hist_len, g.ply + 1);
            if (g_fail != before) break;
        }
        while (g.ply > 0) game_pop(&g);
        CHECK(memcmp(&start, &g.pos, sizeof(Position)) == 0,
              "popping a whole game did not restore the start position");
    }

    done("zobrist", before);
}

/* -------------------------------------------------- 3. en passant ----- */

static void test_en_passant(void)
{
    int before = g_fail;
    section("en passant");

    /* The horizontal discovered-check case: exd3 e.p. would clear both pawns
     * off the 4th rank and expose the black king on a4 to the queen on h4. */
    {
        Position p;
        CHECK(pos_from_fen(&p, "8/8/8/8/k2Pp2Q/8/8/3K4 b - d3 0 1"), "ep-pin FEN");
        CHECK(p.ep == 19 /* d3 */, "ep square parsed as %d", (int)p.ep);
        CHECK(!has_flag(&p, MF_EP), "illegal exd3 e.p. was generated");
        CHECK(!uci_legal(&p, "e4d3"), "exd3 e.p. accepted by move_from_uci");
        CHECK(uci_legal(&p, "e4e3"), "the plain push e3 should still be legal");
    }
    /* Same idea with colours swapped (white capturing, black rook on the rank). */
    {
        Position p;
        CHECK(pos_from_fen(&p, "8/8/8/K2pP2r/8/8/8/3k4 w - d6 0 1"), "ep-pin FEN 2");
        CHECK(!has_flag(&p, MF_EP), "illegal exd6 e.p. was generated (mirror)");
        CHECK(uci_legal(&p, "e5e6"), "the plain push e6 should still be legal");
    }
    /* A perfectly ordinary e.p. capture must still be produced. */
    {
        Position p;
        CHECK(pos_from_fen(&p, "8/8/8/3pP3/8/8/8/4K2k w - d6 0 1"), "ep-ok FEN");
        CHECK(has_flag(&p, MF_EP), "legal e.p. capture was not generated");
        CHECK(uci_legal(&p, "e5d6"), "exd6 e.p. missing");

        Move m = uci_move(&p, "e5d6");
        Undo u;
        make_move(&p, m, &u);
        CHECK(p.board[35] == NO_PIECE, "e.p. did not remove the captured pawn on d5");
        CHECK(p.board[43] == PAWN && p.color_at[43] == WHITE, "e.p. pawn not on d6");
        CHECK(p.key == pos_compute_key(&p), "e.p. key mismatch");
        unmake_move(&p, m, &u);
        CHECK(p.board[35] == PAWN && p.color_at[35] == BLACK, "e.p. unmake lost the black pawn");
    }
    /* A pin along the file must forbid an e.p. capture that leaves it. */
    {
        Position p;
        CHECK(pos_from_fen(&p, "8/8/8/2KpP2r/8/8/8/7k w - d6 0 1"), "ep-pin FEN 3");
        CHECK(!has_flag(&p, MF_EP), "e.p. that exposes the king on the rank was generated");
    }

    done("en passant", before);
}

/* ---------------------------------------------------- 4. castling ----- */

static void test_castling(void)
{
    int before = g_fail;
    section("castling");

    /* Both castles available. */
    {
        Position p;
        pos_from_fen(&p, "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1");
        CHECK(uci_legal(&p, "e1g1"), "O-O should be legal");
        CHECK(uci_legal(&p, "e1c1"), "O-O-O should be legal");
    }
    /* Passing through an attacked square (f1) is refused; O-O-O is unaffected. */
    {
        Position p;
        pos_from_fen(&p, "4kr2/8/8/8/8/8/8/R3K2R w KQ - 0 1");
        CHECK(!uci_legal(&p, "e1g1"), "O-O through the attacked f1 must be refused");
        CHECK(uci_legal(&p, "e1c1"), "O-O-O should still be legal");
    }
    /* Landing on an attacked square (g1) is refused. */
    {
        Position p;
        pos_from_fen(&p, "4k1r1/8/8/8/8/8/8/R3K2R w KQ - 0 1");
        CHECK(!uci_legal(&p, "e1g1"), "O-O onto the attacked g1 must be refused");
        CHECK(uci_legal(&p, "e1c1"), "O-O-O should still be legal");
    }
    /* Castling out of check is refused (both sides). */
    {
        Position p;
        pos_from_fen(&p, "4k3/8/8/8/4r3/8/8/R3K2R w KQ - 0 1");
        CHECK(in_check(&p, WHITE), "white should be in check here");
        CHECK(!uci_legal(&p, "e1g1") && !uci_legal(&p, "e1c1"),
              "castling out of check must be refused");
    }
    /* b1 occupied blocks O-O-O even though b1 is never attacked-tested. */
    {
        Position p;
        pos_from_fen(&p, "4k3/8/8/8/8/8/8/RN2K2R w KQ - 0 1");
        CHECK(!uci_legal(&p, "e1c1"), "O-O-O with a piece on b1 must be refused");
        CHECK(uci_legal(&p, "e1g1"), "O-O should be legal");
    }
    /* d1 attacked blocks O-O-O; b1 attacked does NOT. */
    {
        Position p;
        pos_from_fen(&p, "3rk3/8/8/8/8/8/8/R3K2R w KQ - 0 1");
        CHECK(!uci_legal(&p, "e1c1"), "O-O-O through the attacked d1 must be refused");
        pos_from_fen(&p, "1r2k3/8/8/8/8/8/8/R3K2R w KQ - 0 1");
        CHECK(uci_legal(&p, "e1c1"), "O-O-O is legal when only b1 is attacked");
    }
    /* Black castles. */
    {
        Position p;
        pos_from_fen(&p, "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1");
        CHECK(uci_legal(&p, "e8g8"), "black O-O should be legal");
        CHECK(uci_legal(&p, "e8c8"), "black O-O-O should be legal");

        Move m = uci_move(&p, "e8c8");
        Undo u;
        make_move(&p, m, &u);
        CHECK(p.board[58] == KING && p.color_at[58] == BLACK, "black king not on c8");
        CHECK(p.board[59] == ROOK && p.color_at[59] == BLACK, "black rook not on d8");
        CHECK(p.board[56] == NO_PIECE && p.board[60] == NO_PIECE, "a8/e8 not vacated");
        CHECK((p.castling & (CR_BK | CR_BQ)) == 0, "black rights not cleared after castling");
        CHECK(p.key == pos_compute_key(&p), "castling key mismatch");
        unmake_move(&p, m, &u);
        CHECK(p.castling == CR_ALL, "castling rights not restored by unmake");
    }
    /* Rights are cleared when the king moves. */
    {
        Position p;
        pos_from_fen(&p, "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1");
        Undo u;
        make_move(&p, uci_move(&p, "e1e2"), &u);
        CHECK(p.castling == (CR_BK | CR_BQ), "king move left white rights: %u", p.castling);
    }
    /* Rights are cleared when a rook moves. */
    {
        Position p;
        pos_from_fen(&p, "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1");
        Undo u;
        make_move(&p, uci_move(&p, "a1b1"), &u);
        CHECK(p.castling == (CR_WK | CR_BK | CR_BQ), "a1 rook move: rights %u", p.castling);
    }
    /* Rights are cleared when a rook is CAPTURED on its home square. */
    {
        Position p;
        pos_from_fen(&p, "4k3/8/8/8/8/8/6b1/R3K2R b KQ - 0 1");
        CHECK(uci_legal(&p, "g2h1"), "Bxh1 should be legal");
        Move m = uci_move(&p, "g2h1");
        Undo u;
        make_move(&p, m, &u);
        CHECK(p.castling == CR_WQ, "capture on h1: rights %u (want %u)", p.castling, CR_WQ);
        CHECK(p.key == pos_compute_key(&p), "rook-capture key mismatch");
        unmake_move(&p, m, &u);
        CHECK(p.castling == (CR_WK | CR_WQ), "rights not restored after unmake: %u", p.castling);
    }
    /* Same on the queen side, for black's a8 rook. */
    {
        Position p;
        pos_from_fen(&p, "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1");
        Undo u;
        make_move(&p, uci_move(&p, "a1a8"), &u);
        CHECK(p.castling == (CR_WK | CR_BK), "Rxa8: rights %u", p.castling);
    }
    /* Castling rights claimed by a FEN but with no rook present must not
     * produce a castling move. */
    {
        Position p;
        pos_from_fen(&p, "4k3/8/8/8/8/8/8/4K3 w KQkq - 0 1");
        CHECK(!uci_legal(&p, "e1g1") && !uci_legal(&p, "e1c1"),
              "castling generated with no rooks on the board");
    }

    done("castling", before);
}

/* ------------------------------------------------ 4b. chess 960 ------- */

static void test_castling_960(void)
{
    int before = g_fail;
    section("castling (Chess960)");

    /* Shredder-FEN and X-FEN both have to land on the same rook files. */
    {
        Position p;
        CHECK(pos_from_fen(&p, "1rqbkrbn/pppppppp/1n6/8/8/1N6/PPPPPPPP/1RQBKRBN w FBfb - 0 1"),
              "Shredder-FEN rejected");
        CHECK(p.chess960 == 1, "Shredder-FEN must set chess960");
        CHECK(p.crook[WHITE][0] == 5 && p.crook[WHITE][1] == 1, "rook files %u/%u",
              p.crook[WHITE][0], p.crook[WHITE][1]);
        Position x;
        CHECK(pos_from_fen(&x, "1rqbkrbn/pppppppp/1n6/8/8/1N6/PPPPPPPP/1RQBKRBN w KQkq - 0 1"),
              "X-FEN rejected");
        CHECK(x.chess960 == 1 && x.castling == p.castling
              && x.crook[WHITE][0] == 5 && x.crook[WHITE][1] == 1,
              "X-FEN did not resolve K/Q to the outermost rooks");
        CHECK(x.key == p.key, "the two spellings must be the same position");
        char fen[128];
        pos_to_fen(&p, fen, sizeof(fen));
        CHECK(strcmp(fen, "1rqbkrbn/pppppppp/1n6/8/8/1N6/PPPPPPPP/1RQBKRBN w FBfb - 0 1") == 0,
              "Chess960 FEN must round-trip in Shredder form: %s", fen);
    }

    /* The king does not move at all: O-O with the king on g1, rook on h1. */
    {
        Position p;
        Move m;
        Undo u;
        CHECK(pos_from_fen(&p, "1rk5/8/8/8/8/8/8/1R4KR w H - 0 1"), "960 FEN");
        CHECK(move_from_uci(&p, "g1h1", &m), "king-takes-rook O-O rejected");
        CHECK(MV_FLAG(m) == MF_KCASTLE, "g1h1 should be castling, flag %d", MV_FLAG(m));
        char uc[6];
        move_to_uci_pos(&p, m, uc);
        CHECK(strcmp(uc, "g1h1") == 0, "move_to_uci_pos emitted %s", uc);
        Position snap = p;
        make_move(&p, m, &u);
        CHECK(p.board[6] == KING && p.color_at[6] == WHITE, "king not on g1");
        CHECK(p.board[5] == ROOK && p.color_at[5] == WHITE, "rook not on f1");
        CHECK(p.board[7] == NO_PIECE, "h1 not vacated");
        CHECK(p.key == pos_compute_key(&p), "960 castling key mismatch");
        unmake_move(&p, m, &u);
        CHECK(memcmp(&snap, &p, sizeof(Position)) == 0, "unmake of a null-king castle");
    }

    /* A piece that only the ROOK has to cross still blocks. */
    {
        Position p;
        CHECK(pos_from_fen(&p, "4k3/8/8/8/8/8/8/1R2K3 w B - 0 1"), "960 FEN");
        CHECK(uci_legal(&p, "e1b1"), "O-O-O with the rook on b1 should be legal");
        CHECK(pos_from_fen(&p, "4k3/8/8/8/8/8/8/1RN1K3 w B - 0 1"), "960 FEN");
        CHECK(!uci_legal(&p, "e1b1"), "a knight on c1 must block the rook's path");
    }

    /* Vacating the castling rook must not expose the king. */
    {
        Position p;
        CHECK(pos_from_fen(&p, "4k3/8/8/8/8/8/8/rRK5 w B - 0 1"), "960 FEN");
        CHECK(!in_check(&p, WHITE), "the b1 rook shields the king on c1");
        CHECK(!uci_legal(&p, "c1b1"), "castling may not open a line onto our own king");
    }

    /* Rights follow the 960 rook files, not a1/h1. */
    {
        Position p;
        Undo u;
        Move m;
        CHECK(pos_from_fen(&p, "1rk5/8/8/8/8/8/8/1R2K1R1 w GB - 0 1"), "960 FEN");
        CHECK(move_from_uci(&p, "g1g2", &m), "g1g2");
        make_move(&p, m, &u);
        CHECK(p.castling == CR_WQ, "moving the g1 rook: rights %u, want %u", p.castling, CR_WQ);
        CHECK(p.key == pos_compute_key(&p), "key after the rook move");
        unmake_move(&p, m, &u);
        CHECK(p.castling == (CR_WK | CR_WQ), "rights restored");

        CHECK(pos_from_fen(&p, "1rk3r1/8/8/8/8/8/8/1R2K1R1 b GBgb - 0 1"), "960 FEN");
        CHECK(move_from_uci(&p, "g8g1", &m), "g8g1");
        make_move(&p, m, &u);
        CHECK(p.castling == (CR_WQ | CR_BQ),
              "capturing the g1 rook at home: rights %u, want %u", p.castling, CR_WQ | CR_BQ);
        CHECK(p.key == pos_compute_key(&p), "key after the rook capture");
        unmake_move(&p, m, &u);
        CHECK(p.castling == CR_ALL, "rights restored after unmake");
    }

    /* Random 960 playouts must keep every invariant the classical ones do. */
    {
        static const char *seeds[] = {
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w HAha - 0 1",
            "bqnb1rkr/pp3ppp/3ppn2/2p5/5P2/P2P4/NPP1P1PP/BQ1BNRKR w HFhf - 2 9",
            "1rqbkrbn/pppppppp/1n6/8/8/1N6/PPPPPPPP/1RQBKRBN w FBfb - 0 1",
            "qnr1bkrb/pppp2pp/3np3/5p2/8/P2P2P1/NPP1PP1P/QN1RBKRB w GDg - 3 9",
            "2k5/8/8/8/8/8/8/1RK4R w HB - 0 1",
        };
        const int nseeds = (int)(sizeof(seeds) / sizeof(seeds[0]));
        for (int rep = 0; rep < 8 && g_fail == before; rep++)
            for (int i = 0; i < nseeds && g_fail == before; i++)
                playout(seeds[i], 120);
        CHECK(g_fail == before, "960 playouts found a zobrist/undo inconsistency");
    }

    /* Chess960 perft, from the published tables. */
    {
        Position p;
        CHECK(pos_from_fen(&p, "bqnb1rkr/pp3ppp/3ppn2/2p5/5P2/P2P4/NPP1P1PP/BQ1BNRKR w HFhf - 2 9"),
              "960 perft FEN");
        CHECK(perft(&p, 4) == 326672u, "960 perft(4) = %llu, want 326672",
              (unsigned long long)perft(&p, 4));
        CHECK(pos_from_fen(&p, "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w HAha - 0 1"), "960 518");
        CHECK(perft(&p, 4) == 197281u, "id 518 must reproduce the classical perft(4)");
    }

    done("chess 960 castling", before);
}

/* ------------------------------------------------- 5. promotions ------ */

static void test_promotions(void)
{
    int before = g_fail;
    section("promotions");

    /* All four promotion pieces, quiet and capturing. */
    {
        Position p;
        pos_from_fen(&p, "3q1k2/4P3/8/8/8/8/8/4K3 w - - 0 1");
        const char *quiet[] = { "e7e8q", "e7e8r", "e7e8b", "e7e8n" };
        const char *caps[]  = { "e7d8q", "e7d8r", "e7d8b", "e7d8n" };
        for (int i = 0; i < 4; i++) {
            CHECK(uci_legal(&p, quiet[i]), "quiet promotion %s missing", quiet[i]);
            CHECK(uci_legal(&p, caps[i]),  "capture promotion %s missing", caps[i]);
        }
        Move m = uci_move(&p, "e7d8n");
        CHECK(MV_IS_PROMO(m) && MV_IS_CAPTURE(m), "e7d8n flags wrong: %d", MV_FLAG(m));
        CHECK(MV_PROMO_PIECE(m) == KNIGHT, "e7d8n promo piece wrong");
        Undo u;
        make_move(&p, m, &u);
        CHECK(p.board[59] == KNIGHT && p.color_at[59] == WHITE, "d8 is not a white knight");
        CHECK(p.piece[BLACK][QUEEN] == 0, "black queen survived the capture");
        CHECK(p.key == pos_compute_key(&p), "promotion-capture key mismatch");
        unmake_move(&p, m, &u);
        CHECK(p.board[52] == PAWN && p.color_at[52] == WHITE, "unmake lost the e7 pawn");
        CHECK(p.piece[BLACK][QUEEN] != 0, "unmake lost the black queen");
    }

    /* Underpromotion to a knight delivering mate.  Black king d7 is walled in
     * by its own pieces; Nf8 checks d7 and covers e6, and nothing attacks f8. */
    {
        Position p;
        CHECK(pos_from_fen(&p, "2rqn3/2pkpP2/2pp4/8/8/8/8/4K3 w - - 0 1"), "underpromo FEN");
        CHECK(uci_legal(&p, "f7f8n"), "f8=N was not generated");

        char san[16];
        CHECK(strcmp(san_of(&p, "f7f8n", san, sizeof(san)), "f8=N#") == 0,
              "SAN for f7f8n was \"%s\", want \"f8=N#\"", san);
        CHECK(strcmp(san_of(&p, "f7f8q", san, sizeof(san)), "f8=Q") == 0,
              "SAN for f7f8q was \"%s\", want \"f8=Q\"", san);

        Game g;
        game_start_fen(&g, "2rqn3/2pkpP2/2pp4/8/8/8/8/4K3 w - - 0 1");
        CHECK(g.result == GR_ONGOING, "underpromo position should start ongoing");
        game_push(&g, uci_move(&g.pos, "f7f8n"));
        CHECK(g.result == GR_WHITE_WIN && g.reason == TR_CHECKMATE,
              "f8=N should be checkmate (result %d reason %d)", g.result, g.reason);
    }

    /* gen_legal_captures: captures + queen promotions only. */
    {
        Position p;
        pos_from_fen(&p, "3q1k2/4P3/8/8/8/8/8/4K3 w - - 0 1");
        Move caps[MAX_MOVES];
        int n = gen_legal_captures(&p, caps);
        int seen_qq = 0, seen_qc = 0;
        for (int i = 0; i < n; i++) {
            int fl = MV_FLAG(caps[i]);
            CHECK(MV_IS_CAPTURE(caps[i]) || fl == MF_PROMO_Q,
                  "gen_legal_captures emitted a quiet non-promotion (flag %d)", fl);
            if (fl == MF_PROMO_Q)  seen_qq = 1;
            if (fl == MF_PROMO_QC) seen_qc = 1;
            if (MV_IS_PROMO(caps[i]))
                CHECK(MV_PROMO_PIECE(caps[i]) == QUEEN,
                      "gen_legal_captures emitted an underpromotion");
        }
        CHECK(seen_qq, "gen_legal_captures missed the quiet queen promotion");
        CHECK(seen_qc, "gen_legal_captures missed the capturing queen promotion");
    }

    done("promotions", before);
}

/* ------------------------------------------- 6. terminal conditions --- */

static void test_terminal(void)
{
    int before = g_fail;
    section("checkmate / stalemate / draws");

    /* Stalemate. */
    {
        Game g;
        CHECK(game_start_fen(&g, "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1"), "stalemate FEN");
        Move list[MAX_MOVES];
        CHECK(gen_legal(&g.pos, list) == 0, "stalemate position has legal moves");
        CHECK(!in_check(&g.pos, BLACK), "stalemate position must not be check");
        CHECK(g.result == GR_DRAW && g.reason == TR_STALEMATE,
              "stalemate not detected (result %d reason %d)", g.result, g.reason);
    }
    /* Checkmate, both colours. */
    {
        Game g;
        game_start_fen(&g, "7k/6Q1/6K1/8/8/8/8/8 b - - 1 1");   /* Qg7 mate */
        CHECK(g.result == GR_WHITE_WIN && g.reason == TR_CHECKMATE,
              "black should be mated (result %d reason %d)", g.result, g.reason);

        /* ... and the same mate reached by playing Qg7 from the parent node. */
        Game g2;
        game_start_fen(&g2, "7k/5Q2/6K1/8/8/8/8/8 w - - 0 1");
        CHECK(g2.result == GR_ONGOING, "parent of the mate should be ongoing");
        game_push(&g2, uci_move(&g2.pos, "f7g7"));
        CHECK(g2.result == GR_WHITE_WIN && g2.reason == TR_CHECKMATE,
              "Qg7# not detected (result %d reason %d)", g2.result, g2.reason);

        /* Fool's mate. */
        Game h;
        game_start(&h);
        const char *fools[] = { "f2f3", "e7e5", "g2g4", "d8h4" };
        for (int i = 0; i < 4; i++) game_push(&h, uci_move(&h.pos, fools[i]));
        CHECK(h.result == GR_BLACK_WIN && h.reason == TR_CHECKMATE,
              "fool's mate not detected (result %d reason %d)", h.result, h.reason);
    }
    /* Fifty-move rule: halfmove reaches 100. */
    {
        Game g;
        CHECK(game_start_fen(&g, "4k3/8/8/8/8/8/8/R3K2R w KQ - 99 60"), "fifty FEN");
        CHECK(g.result == GR_ONGOING, "should be ongoing at halfmove 99");
        game_push(&g, uci_move(&g.pos, "h1g1"));
        CHECK(g.pos.halfmove == 100, "halfmove is %u", g.pos.halfmove);
        CHECK(g.result == GR_DRAW && g.reason == TR_FIFTY,
              "fifty-move draw missed (result %d reason %d)", g.result, g.reason);

        /* A capture or pawn move resets the counter. */
        Game h;
        game_start_fen(&h, "4k3/7p/8/8/8/8/8/R3K2R w KQ - 99 60");
        game_push(&h, uci_move(&h.pos, "h1g1"));
        CHECK(h.result == GR_DRAW && h.reason == TR_FIFTY, "fifty with pawns present");
        Game i2;
        game_start_fen(&i2, "4k3/7p/8/8/8/8/8/R3K2R b KQ - 99 60");
        game_push(&i2, uci_move(&i2.pos, "h7h6"));
        CHECK(i2.pos.halfmove == 0, "pawn move did not reset halfmove: %u", i2.pos.halfmove);
        CHECK(i2.result == GR_ONGOING, "pawn move should reset the fifty-move count");
    }
    /* Threefold repetition (automatic draw). */
    {
        Game g;
        game_start(&g);
        const char *shuffle[] = { "g1f3", "g8f6", "f3g1", "f6g8" };
        for (int i = 0; i < 4; i++) game_push(&g, uci_move(&g.pos, shuffle[i]));
        CHECK(game_repetitions(&g) == 2, "expected 2 repetitions, got %d", game_repetitions(&g));
        CHECK(g.result == GR_ONGOING, "two occurrences must not be a draw yet");
        for (int i = 0; i < 4; i++) game_push(&g, uci_move(&g.pos, shuffle[i]));
        CHECK(game_repetitions(&g) == 3, "expected 3 repetitions, got %d", game_repetitions(&g));
        CHECK(g.result == GR_DRAW && g.reason == TR_REPETITION,
              "threefold draw missed (result %d reason %d)", g.result, g.reason);
    }
    /* The max_plies cap. */
    {
        Game g;
        game_start(&g);
        game_push(&g, uci_move(&g.pos, "e2e4"));
        game_push(&g, uci_move(&g.pos, "e7e5"));
        CHECK(game_update_result(&g, 0) == GR_ONGOING, "max_plies <= 0 must disable the cap");
        CHECK(game_update_result(&g, 2) == GR_DRAW && g.reason == TR_MAX_PLIES,
              "ply cap did not adjudicate a draw");
        CHECK(game_update_result(&g, 10) == GR_ONGOING, "cap fired too early");
    }

    done("terminal conditions", before);
}

/* ------------------------------------------ 7. insufficient material -- */

static void test_material(void)
{
    int before = g_fail;
    section("insufficient material");

    struct { const char *fen; int want; const char *what; } t[] = {
        { "8/8/8/4k3/8/8/8/4K3 w - - 0 1",     1, "K v K" },
        { "8/8/8/4k3/8/8/8/2B1K3 w - - 0 1",   1, "K+B v K" },
        { "8/8/8/4k3/8/8/8/2N1K3 w - - 0 1",   1, "K+N v K" },
        { "8/8/8/4k3/8/8/8/1NN1K3 w - - 0 1",  1, "K+N+N v K" },
        { "4kn2/8/8/8/8/8/8/4K3 w - - 0 1",    1, "K v K+N" },
        { "4kb2/8/8/8/8/8/8/4K3 w - - 0 1",    1, "K v K+B" },
        { "1nn1k3/8/8/8/8/8/8/4K3 w - - 0 1",  1, "K v K+N+N" },
        /* Bc1 (dark) v Bf8 (dark): same complex -> dead. */
        { "5bk1/8/8/8/8/8/8/2B1K3 w - - 0 1",  1, "K+B v K+B same colour" },
        /* Bc1 (dark) v Be8 (light): opposite complexes -> NOT a draw. */
        { "4bk2/8/8/8/8/8/8/2B1K3 w - - 0 1",  0, "K+B v K+B opposite colours" },
        { "8/8/8/4k3/8/8/8/2R1K3 w - - 0 1",   0, "K+R v K" },
        { "8/8/8/4k3/8/8/8/2Q1K3 w - - 0 1",   0, "K+Q v K" },
        { "8/8/7p/4k3/8/8/8/4K3 w - - 0 1",    0, "K+P v K" },
        { "8/8/8/4k3/8/8/8/1BB1K3 w - - 0 1",  0, "K+B+B v K" },
        { "8/8/8/3nk3/8/8/8/2N1K3 w - - 0 1",  0, "K+N v K+N" },
        { "8/8/8/3nk3/8/8/8/2B1K3 w - - 0 1",  0, "K+B v K+N" },
        { "8/8/8/4k3/8/8/8/1NB1K3 w - - 0 1",  0, "K+B+N v K" },
    };

    for (size_t i = 0; i < sizeof(t) / sizeof(t[0]); i++) {
        Position p;
        CHECK(pos_from_fen(&p, t[i].fen), "material FEN: %s", t[i].fen);
        int got = insufficient_material(&p);
        CHECK(got == t[i].want, "%s: insufficient_material -> %d, want %d (%s)",
              t[i].what, got, t[i].want, t[i].fen);
    }

    /* And it must feed through to game_update_result. */
    {
        Game g;
        game_start_fen(&g, "5bk1/8/8/8/8/8/8/2B1K3 w - - 0 1");
        CHECK(g.result == GR_DRAW && g.reason == TR_INSUFFICIENT,
              "same-colour bishops should be an immediate draw");
        Game h;
        game_start_fen(&h, "4bk2/8/8/8/8/8/8/2B1K3 w - - 0 1");
        CHECK(h.result == GR_ONGOING,
              "opposite-colour bishops must NOT be a draw (result %d reason %d)",
              h.result, h.reason);
    }

    done("insufficient material", before);
}

/* -------------------------------------------------------- 8. SAN ------ */

static void test_san(void)
{
    int before = g_fail;
    section("SAN");

    struct { const char *fen; const char *uci; const char *san; } t[] = {
        /* plain moves */
        { "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "e2e4", "e4"     },
        { "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "b1c3", "Nc3"    },
        { "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "g1f3", "Nf3"    },
        /* file disambiguation: Nb1 and Nf3 both reach d2 */
        { "4k3/8/8/8/8/5N2/8/1N2K3 w - - 0 1", "b1d2", "Nbd2" },
        { "4k3/8/8/8/8/5N2/8/1N2K3 w - - 0 1", "f3d2", "Nfd2" },
        /* rank disambiguation: Nb1 and Nb5 share a file, both reach c3 */
        { "4k3/8/8/1N6/8/8/8/1N2K3 w - - 0 1", "b1c3", "N1c3" },
        { "4k3/8/8/1N6/8/8/8/1N2K3 w - - 0 1", "b5c3", "N5c3" },
        /* three queens reaching f3: Qd1 needs file+rank, Qf1 file, Qd3 rank */
        { "4k3/8/8/8/8/3Q4/8/K2Q1Q2 w - - 0 1", "d1f3", "Qd1f3" },
        { "4k3/8/8/8/8/3Q4/8/K2Q1Q2 w - - 0 1", "f1f3", "Qff3"  },
        { "4k3/8/8/8/8/3Q4/8/K2Q1Q2 w - - 0 1", "d3f3", "Q3f3"  },
        /* pawn capture and en passant */
        { "4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1",  "e4d5", "exd5" },
        { "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",  "e5d6", "exd6" },
        /* castling */
        { "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1g1", "O-O"   },
        { "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1c1", "O-O-O" },
        { "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1", "e8g8", "O-O"   },
        { "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1", "e8c8", "O-O-O" },
        /* check and mate suffixes */
        { "4k3/8/8/8/8/8/8/4K2R w K - 0 1",  "h1h8", "Rh8+"  },
        { "7k/5Q2/6K1/8/8/8/8/8 w - - 0 1",  "f7g7", "Qg7#"  },
        /* promotions */
        { "3q1k2/4P3/8/8/8/8/8/4K3 w - - 0 1", "e7e8q", "e8=Q+"  },
        { "3q1k2/4P3/8/8/8/8/8/4K3 w - - 0 1", "e7d8r", "exd8=R+" },
        { "3q1k2/4P3/8/8/8/8/8/4K3 w - - 0 1", "e7d8n", "exd8=N"  },
        { "2rqn3/2pkpP2/2pp4/8/8/8/8/4K3 w - - 0 1", "f7f8n", "f8=N#" },
        /* capture with a piece */
        { "4k3/8/8/3p4/8/8/8/3RK3 w - - 0 1", "d1d5", "Rxd5" },
    };

    for (size_t i = 0; i < sizeof(t) / sizeof(t[0]); i++) {
        Position p;
        char san[16];
        CHECK(pos_from_fen(&p, t[i].fen), "SAN FEN: %s", t[i].fen);
        san_of(&p, t[i].uci, san, sizeof(san));
        CHECK(strcmp(san, t[i].san) == 0, "SAN(%s in \"%s\") = \"%s\", want \"%s\"",
              t[i].uci, t[i].fen, san, t[i].san);
    }

    /* SAN must never overflow a short buffer. */
    {
        Position p;
        char small[4];
        pos_from_fen(&p, "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1");
        move_to_san(&p, uci_move(&p, "e1c1"), small, sizeof(small));
        CHECK(strlen(small) == 3, "short SAN buffer produced %zu chars", strlen(small));
    }

    done("SAN", before);
}

/* -------------------------------------------------------- 9. UCI ------ */

static void test_uci(void)
{
    int before = g_fail;
    section("UCI round-trip");

    static const char *fens[] = {
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "3q1k2/4P3/8/8/8/8/8/4K3 w - - 0 1",
        "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
    };
    for (size_t i = 0; i < sizeof(fens) / sizeof(fens[0]); i++) {
        Position p;
        pos_from_fen(&p, fens[i]);
        Move list[MAX_MOVES];
        int n = gen_legal(&p, list);
        CHECK(n > 0, "no legal moves in %s", fens[i]);
        for (int j = 0; j < n; j++) {
            char u[6];
            Move back;
            move_to_uci(list[j], u);
            CHECK(move_from_uci(&p, u, &back), "move_from_uci rejected its own output \"%s\"", u);
            CHECK(back == list[j], "UCI round-trip changed move \"%s\" (%d -> %d)",
                  u, (int)list[j], (int)back);
        }
    }
    /* Rejections. */
    {
        Position p;
        pos_from_fen(&p, "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        Move m;
        CHECK(!move_from_uci(&p, "e2e5", &m), "illegal e2e5 accepted");
        CHECK(!move_from_uci(&p, "zzzz", &m), "garbage accepted");
        CHECK(!move_from_uci(&p, "e2", &m),   "truncated UCI accepted");
        CHECK(!move_from_uci(&p, "e7e8q", &m), "promotion for the wrong side accepted");
    }

    done("UCI", before);
}

/* -------------------------------------------------- 10. sanity: gen --- */

static void test_gen_sanity(void)
{
    int before = g_fail;
    section("generator sanity (legal-only, captures subset)");

    /* In check: only evasions may be generated, and gen_legal_captures must
     * narrow to captures of the checker (no quiet blocks, no king shuffles). */
    {
        Position p;
        CHECK(pos_from_fen(&p, "4k3/8/8/8/4q3/6N1/2B5/4K3 w - - 0 1"), "check FEN");
        CHECK(in_check(&p, WHITE), "white should be in check from Qe4");

        Move all[MAX_MOVES], caps[MAX_MOVES];
        int na = gen_legal(&p, all);
        int nc = gen_legal_captures(&p, caps);
        CHECK(na == 7, "evasions: got %d, want 7 (Kd1 Kd2 Kf1 Kf2 Bxe4 Nxe4 Nge2)", na);
        CHECK(nc == 2, "capture evasions: got %d, want 2 (Bxe4, Nxe4)", nc);
        CHECK(uci_legal(&p, "c2e4") && uci_legal(&p, "g3e4"), "Bxe4 / Nxe4 missing");
        CHECK(uci_legal(&p, "g3e2"), "the blocking Nge2 is missing");
        CHECK(!uci_legal(&p, "e1e2"), "the king may not step along the checking ray");
    }
    /* Double check: only king moves. */
    {
        Position p;
        CHECK(pos_from_fen(&p, "4k3/8/8/8/4q3/5n2/2B5/4K3 w - - 0 1"), "double-check FEN");
        Move all[MAX_MOVES];
        int na = gen_legal(&p, all);
        for (int i = 0; i < na; i++)
            CHECK(p.board[MV_FROM(all[i])] == KING,
                  "non-king move generated while in double check");
        CHECK(na > 0, "double check should still leave the king a square");
    }

    static const char *fens[] = {
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
        "4k3/8/8/8/4q3/6N1/2B5/4K3 w - - 0 1",
        "n1n5/PPPk4/8/8/8/8/4Kppp/5N1N b - - 0 1",
    };

    for (size_t i = 0; i < sizeof(fens) / sizeof(fens[0]); i++) {
        Position p;
        pos_from_fen(&p, fens[i]);
        for (int ply = 0; ply < 400; ply++) {
            Move list[MAX_MOVES];
            int n = gen_legal(&p, list);
            if (n == 0) break;
            CHECK(n <= MAX_MOVES, "move count %d exceeds MAX_MOVES", n);

            /* Every generated move must leave our own king safe. */
            for (int j = 0; j < n; j++) {
                Undo u;
                int us = p.side;
                make_move(&p, list[j], &u);
                if (in_check(&p, us)) {
                    char b[128], uc[6];
                    unmake_move(&p, list[j], &u);
                    pos_to_fen(&p, b, sizeof(b));
                    move_to_uci(list[j], uc);
                    CHECK(0, "illegal move %s generated in \"%s\"", uc, b);
                    return;
                }
                unmake_move(&p, list[j], &u);
            }
            /* gen_legal_captures must be EXACTLY {captures} + {queen promotions}
             * drawn from gen_legal -- no more, no less. */
            {
                Move caps[MAX_MOVES];
                int nc = gen_legal_captures(&p, caps);
                int want = 0;
                for (int k = 0; k < n; k++) {
                    Move m = list[k];
                    int keep = MV_IS_PROMO(m) ? (MV_PROMO_PIECE(m) == QUEEN)
                                              : (MV_IS_CAPTURE(m) != 0);
                    if (!keep) continue;
                    want++;
                    int found = 0;
                    for (int j = 0; j < nc; j++) if (caps[j] == m) { found = 1; break; }
                    if (!found) {
                        char b[128], uc[6];
                        pos_to_fen(&p, b, sizeof(b));
                        move_to_uci(m, uc);
                        CHECK(0, "gen_legal_captures missed %s in \"%s\"", uc, b);
                        return;
                    }
                }
                if (nc != want) {
                    char b[128];
                    pos_to_fen(&p, b, sizeof(b));
                    CHECK(0, "gen_legal_captures returned %d, want %d in \"%s\"", nc, want, b);
                    return;
                }
            }
            Undo u;
            make_move(&p, list[rnd() % (uint64_t)n], &u);
            if (p.halfmove >= 100) break;
        }
    }

    done("generator sanity", before);
}

/* ---------------------------------------------------------------- main */

int main(void)
{
    chess_init();

    printf("chess core rule tests\n");
    printf("===============================================================\n");

    test_fen();
    test_zobrist();
    test_en_passant();
    test_castling();
    test_castling_960();
    test_promotions();
    test_terminal();
    test_material();
    test_san();
    test_uci();
    test_gen_sanity();

    printf("===============================================================\n");
    printf("%d checks, %d failures\n", g_checks, g_fail);
    if (g_fail == 0) printf("ALL PASS\n");
    return g_fail ? 1 : 0;
}
