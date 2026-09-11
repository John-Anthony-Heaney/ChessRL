/* tests/test_net.c -- verification of src/net.c.
 *
 * The point of this file is the gradient check (section 3).  nn_backward() is
 * the one piece of the system whose output nothing else can sanity-check: a
 * wrong sign or a transposed index there does not crash, it just silently makes
 * every training run meaningless.  So every parameter tensor is compared
 * against central finite differences of the actual forward pass.
 *
 * Every parameter tensor means EVERY one: the LayerNorm gains and biases
 * (g0/c0, g1/c1, gv/cv) and the value head's hidden layer (Wvh/bvh) are checked
 * exactly like the weight matrices, because a normalisation whose backward pass
 * is subtly wrong is the easiest way in the world to train a network that
 * almost works.
 *
 * Sections
 *   1. feature encoding + side-to-move mirroring
 *   2. move keys
 *   3. gradient check vs central differences (the centrepiece)
 *   4. gradient accumulation (nn_backward must add, not overwrite)
 *   5. AdamW: convergence, grad buffer zeroing, global-norm clipping
 *   6. model_save / model_load round-trip, header probe, capacity truncation
 *   7. micro-benchmark
 *
 * Build:
 *   cc -O1 -std=c11 -D_DARWIN_C_SOURCE -Isrc src/chess.c src/net.c \
 *      tests/test_net.c -o /tmp/tn -lm && /tmp/tn
 */

#include "chess.h"
#include "net.h"

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ------------------------------------------------------------- harness */

static int g_fail = 0;
static int g_checks = 0;

#define CHECK(cond, ...)                                   \
    do {                                                   \
        g_checks++;                                        \
        if (!(cond)) {                                     \
            g_fail++;                                      \
            printf("  FAIL (line %d): ", __LINE__);        \
            printf(__VA_ARGS__);                           \
            printf("\n");                                  \
        }                                                  \
    } while (0)

static void section(const char *s) { printf("\n=== %s ===\n", s); }

/* ----------------------------------------------------------------- rng */

/* Deterministic by default; TEST_NET_SEED lets the sweep be re-run with other
 * draws (weights, perturbed entries, dlogits) to show the tolerances are not
 * tuned to one lucky seed. */
static uint64_t g_rs = 0x0123456789ABCDEFULL;

static void seed_from_env(void)
{
    const char *s = getenv("TEST_NET_SEED");
    if (s && *s) {
        g_rs = strtoull(s, NULL, 0);
        printf("          TEST_NET_SEED=%llu\n", (unsigned long long)g_rs);
    }
}

static uint64_t rnd_u64(void)
{
    uint64_t z = (g_rs += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}
static double rnd_uni(void) { return (double)(rnd_u64() >> 11) * (1.0 / 9007199254740992.0); }
static float  rnd_sym(void) { return (float)(rnd_uni() * 2.0 - 1.0); }
static float  rnd_norm(void)
{
    const double u1 = (double)((rnd_u64() >> 11) + 1) * (1.0 / 9007199254740993.0);
    const double u2 = rnd_uni();
    return (float)(sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2));
}
static int rnd_int(int n) { return n > 0 ? (int)(rnd_u64() % (uint64_t)n) : 0; }

static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

/* ------------------------------------------------------------ utilities */

static int cmp_u16(const void *a, const void *b)
{
    const uint16_t x = *(const uint16_t *)a, y = *(const uint16_t *)b;
    return (x > y) - (x < y);
}
static void sort_u16(uint16_t *v, int n) { qsort(v, (size_t)n, sizeof(uint16_t), cmp_u16); }

/* Build the colour-flipped, vertically mirrored FEN of `fen`. */
static int mirror_fen(const char *fen, char *out, size_t outlen)
{
    char board[128], side[8], cast[16], ep[16];
    int hm = 0, fm = 1;
    const int got = sscanf(fen, "%127s %7s %15s %15s %d %d", board, side, cast, ep, &hm, &fm);
    if (got < 4) return 0;

    char ranks[8][32];
    int nr = 0, pos = 0;
    memset(ranks, 0, sizeof ranks);
    for (int i = 0;; i++) {
        const char c = board[i];
        if (c == '/' || c == '\0') {
            if (nr >= 8 || pos >= 31) return 0;
            ranks[nr][pos] = '\0';
            nr++; pos = 0;
            if (c == '\0') break;
        } else {
            if (nr >= 8 || pos >= 31) return 0;
            ranks[nr][pos++] = c;
        }
    }
    if (nr != 8) return 0;

    char nb[160];
    int o = 0;
    for (int r = 7; r >= 0; r--) {                 /* reverse the rank order */
        for (int i = 0; ranks[r][i]; i++) {
            char c = ranks[r][i];
            if      (c >= 'a' && c <= 'z') c = (char)(c - 'a' + 'A');  /* swap colour */
            else if (c >= 'A' && c <= 'Z') c = (char)(c - 'A' + 'a');
            nb[o++] = c;
        }
        if (r) nb[o++] = '/';
    }
    nb[o] = '\0';

    const char nside = (side[0] == 'w') ? 'b' : 'w';

    char nc[8];
    int co = 0;
    if (strchr(cast, 'k')) nc[co++] = 'K';
    if (strchr(cast, 'q')) nc[co++] = 'Q';
    if (strchr(cast, 'K')) nc[co++] = 'k';
    if (strchr(cast, 'Q')) nc[co++] = 'q';
    if (!co) nc[co++] = '-';
    nc[co] = '\0';

    char ne[8];
    if (ep[0] == '-') { ne[0] = '-'; ne[1] = '\0'; }
    else { ne[0] = ep[0]; ne[1] = (char)('0' + (9 - (ep[1] - '0'))); ne[2] = '\0'; }

    snprintf(out, outlen, "%s %c %s %s %d %d", nb, nside, nc, ne, hm, fm);
    return 1;
}

/* =========================================================== 1. FEATURES */

static void feats_of(const char *fen, uint16_t *idx, int *n)
{
    Position p;
    if (!pos_from_fen(&p, fen)) { printf("  FAIL: bad FEN '%s'\n", fen); g_fail++; *n = 0; return; }
    *n = nn_features(&p, idx);
}

static int has_feat(const uint16_t *v, int n, int f)
{
    for (int i = 0; i < n; i++) if (v[i] == (uint16_t)f) return 1;
    return 0;
}

static void check_mirror(const char *fen)
{
    char mfen[192];
    if (!mirror_fen(fen, mfen, sizeof mfen)) { CHECK(0, "mirror_fen failed for %s", fen); return; }

    uint16_t a[NF_MAXACTIVE], b[NF_MAXACTIVE];
    int na = 0, nb = 0;
    feats_of(fen, a, &na);
    feats_of(mfen, b, &nb);
    sort_u16(a, na);
    sort_u16(b, nb);

    int same = (na == nb);
    if (same) for (int i = 0; i < na; i++) if (a[i] != b[i]) { same = 0; break; }
    CHECK(same, "mirror mismatch\n    orig   %s (%d feats)\n    mirror %s (%d feats)",
          fen, na, mfen, nb);
    if (!same) {
        printf("    orig  :"); for (int i = 0; i < na; i++) printf(" %d", a[i]); printf("\n");
        printf("    mirror:"); for (int i = 0; i < nb; i++) printf(" %d", b[i]); printf("\n");
    }
}

static void test_features(void)
{
    section("1. feature encoding");

    /* ---- hand-computed start position ---- */
    Position p;
    pos_startpos(&p);
    uint16_t idx[NF_MAXACTIVE];
    const int nf = nn_features(&p, idx);

    int exp[64];
    int ne = 0;
    for (int s = 8;  s <= 15; s++) exp[ne++] = 0 * 64 + s;      /* our pawns a2..h2 */
    exp[ne++] = 64 + 1;  exp[ne++] = 64 + 6;                    /* knights b1,g1    */
    exp[ne++] = 128 + 2; exp[ne++] = 128 + 5;                   /* bishops c1,f1    */
    exp[ne++] = 192 + 0; exp[ne++] = 192 + 7;                   /* rooks   a1,h1    */
    exp[ne++] = 256 + 3;                                        /* queen   d1       */
    exp[ne++] = 320 + 4;                                        /* king    e1       */
    for (int s = 48; s <= 55; s++) exp[ne++] = 384 + 0 * 64 + s;  /* their pawns    */
    exp[ne++] = 384 + 64 + 57;  exp[ne++] = 384 + 64 + 62;
    exp[ne++] = 384 + 128 + 58; exp[ne++] = 384 + 128 + 61;
    exp[ne++] = 384 + 192 + 56; exp[ne++] = 384 + 192 + 63;
    exp[ne++] = 384 + 256 + 59;
    exp[ne++] = 384 + 320 + 60;
    exp[ne++] = 768; exp[ne++] = 769; exp[ne++] = 770; exp[ne++] = 771;
    exp[ne++] = 780;                                            /* halfmove 0       */

    CHECK(nf == ne, "start position: %d features, expected %d", nf, ne);
    int missing = 0;
    for (int i = 0; i < ne; i++)
        if (!has_feat(idx, nf, exp[i])) { missing++; printf("    missing feature %d\n", exp[i]); }
    CHECK(missing == 0, "start position: %d expected features missing", missing);
    for (int i = 0; i < nf; i++) CHECK(idx[i] < NF_INPUT, "feature %d out of range", idx[i]);

    /* ---- castling bits (our K, our Q, their K, their Q) ---- */
    uint16_t f[NF_MAXACTIVE];
    int n = 0;
    feats_of("4k3/8/8/8/8/8/8/R3K2R w KQ - 0 1", f, &n);
    CHECK(has_feat(f,n,768) && has_feat(f,n,769) && !has_feat(f,n,770) && !has_feat(f,n,771),
          "white to move, KQ -> 768,769 only");
    feats_of("r3k2r/8/8/8/8/8/8/4K3 w kq - 0 1", f, &n);
    CHECK(!has_feat(f,n,768) && !has_feat(f,n,769) && has_feat(f,n,770) && has_feat(f,n,771),
          "white to move, kq -> 770,771 only (their rights)");
    feats_of("r3k2r/8/8/8/8/8/8/4K3 b kq - 0 1", f, &n);
    CHECK(has_feat(f,n,768) && has_feat(f,n,769) && !has_feat(f,n,770) && !has_feat(f,n,771),
          "black to move, kq -> 768,769 (our rights)");
    feats_of("r3k2r/8/8/8/8/8/8/R3K2R w Kq - 0 1", f, &n);
    CHECK(has_feat(f,n,768) && !has_feat(f,n,769) && !has_feat(f,n,770) && has_feat(f,n,771),
          "white to move, Kq -> 768,771");
    feats_of("4k3/8/8/8/8/8/8/4K3 w - - 0 1", f, &n);
    CHECK(!has_feat(f,n,768) && !has_feat(f,n,769) && !has_feat(f,n,770) && !has_feat(f,n,771),
          "no castling rights -> no castling features");

    /* ---- en-passant file ---- */
    feats_of("rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3", f, &n);
    CHECK(has_feat(f, n, 772 + 3), "ep on d6 -> feature 775");
    for (int i = 0; i < 8; i++)
        if (i != 3) CHECK(!has_feat(f, n, 772 + i), "ep: spurious file feature %d", 772 + i);
    feats_of("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", f, &n);
    for (int i = 0; i < 8; i++)
        CHECK(!has_feat(f, n, 772 + i), "no ep: feature %d must be off", 772 + i);

    /* ---- halfmove buckets: min(7, halfmove/13) ---- */
    const int hm_cases[][2] = { {0,0}, {1,0}, {12,0}, {13,1}, {25,1}, {26,2}, {77,5},
                                {90,6}, {91,7}, {99,7}, {120,7} };
    for (size_t c = 0; c < sizeof hm_cases / sizeof hm_cases[0]; c++) {
        char fen[96];
        snprintf(fen, sizeof fen, "4k3/8/8/8/8/8/8/4K3 w - - %d 1", hm_cases[c][0]);
        feats_of(fen, f, &n);
        CHECK(has_feat(f, n, 780 + hm_cases[c][1]), "halfmove %d -> feature %d",
              hm_cases[c][0], 780 + hm_cases[c][1]);
        int nb = 0;
        for (int i = 0; i < 8; i++) if (has_feat(f, n, 780 + i)) nb++;
        CHECK(nb == 1, "halfmove %d: exactly one bucket feature, got %d", hm_cases[c][0], nb);
    }

    /* ---- side-to-move mirroring: the two feature multisets must be identical ---- */
    const char *mirror_cases[] = {
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 3 2",
        "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 27 1",
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        "n3k3/1P6/8/8/8/8/8/4K3 w - - 0 1",
        "4k3/8/8/8/8/8/8/R3K2R w KQ - 40 60",
    };
    for (size_t i = 0; i < sizeof mirror_cases / sizeof mirror_cases[0]; i++)
        check_mirror(mirror_cases[i]);

    printf("  start position: %d active features; %zu mirror pairs identical\n",
           nf, sizeof mirror_cases / sizeof mirror_cases[0]);
}

/* ========================================================== 2. MOVE KEYS */

static MoveKey key_of(const char *fen, const char *uci, int *ok)
{
    Position p;
    MoveKey k;
    Move m = MV_NONE;
    memset(&k, 0, sizeof k);
    *ok = 0;
    if (!pos_from_fen(&p, fen)) { printf("  FAIL: bad FEN '%s'\n", fen); g_fail++; return k; }
    if (!move_from_uci(&p, uci, &m)) {
        printf("  FAIL: illegal move %s in %s\n", uci, fen); g_fail++; return k;
    }
    nn_move_key(&p, m, &k);
    *ok = 1;
    return k;
}

static void expect_key(const char *fen, const char *uci, int from, int to,
                       int pc, int promo, int cap, const char *what)
{
    int ok = 0;
    const MoveKey k = key_of(fen, uci, &ok);
    if (!ok) { g_checks++; g_fail++; return; }
    CHECK(k.from == from && k.to == to && k.pc == pc && k.promo == promo && k.cap == cap,
          "%s (%s): got from=%d to=%d pc=%d promo=%d cap=%d, expected %d/%d/%d/%d/%d",
          what, uci, k.from, k.to, k.pc, k.promo, k.cap, from, to, pc, promo, cap);
}

/* The same move played from the mirrored position must give an identical key. */
static void expect_key_mirrors(const char *fen, const char *uci, const char *muci, const char *what)
{
    char mfen[192];
    if (!mirror_fen(fen, mfen, sizeof mfen)) { CHECK(0, "mirror_fen failed"); return; }
    int ok1 = 0, ok2 = 0;
    const MoveKey a = key_of(fen, uci, &ok1);
    const MoveKey b = key_of(mfen, muci, &ok2);
    if (!ok1 || !ok2) { g_checks++; g_fail++; return; }
    CHECK(a.from == b.from && a.to == b.to && a.pc == b.pc && a.promo == b.promo && a.cap == b.cap,
          "%s: key not mirror-invariant. %s -> %d/%d/%d/%d/%d ; %s -> %d/%d/%d/%d/%d",
          what, uci, a.from, a.to, a.pc, a.promo, a.cap,
          muci, b.from, b.to, b.pc, b.promo, b.cap);
}

static void test_move_keys(void)
{
    section("2. move keys");

    /* white to move: no flip */
    expect_key("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
               "e2e4", 12, 28, PAWN, 0, 0, "quiet double push");
    expect_key("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
               "g1f3", 6, 21, KNIGHT, 0, 0, "quiet knight move");

    /* black to move: from/to flipped (sq ^ 56) */
    expect_key("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1",
               "e7e5", 52 ^ 56, 36 ^ 56, PAWN, 0, 0, "black double push (mirrored)");
    expect_key_mirrors("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                       "e2e4", "e7e5", "double push");
    expect_key_mirrors("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                       "g1f3", "g8f6", "knight development");

    /* captures: cap == captured type + 1, pc == moving piece */
    expect_key("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2",
               "e4d5", 28, 35, PAWN, 0, PAWN + 1, "pawn takes pawn");
    expect_key("4k3/8/8/3q4/4B3/8/8/4K3 w - - 0 1",
               "e4d5", 28, 35, BISHOP, 0, QUEEN + 1, "bishop takes queen");
    expect_key("4k3/8/3r4/8/4N3/8/8/4K3 w - - 0 1",
               "e4d6", 28, 43, KNIGHT, 0, ROOK + 1, "knight takes rook");
    expect_key("4k3/8/8/4b3/4R3/8/8/4K3 w - - 0 1",
               "e4e5", 28, 36, ROOK, 0, BISHOP + 1, "rook takes bishop");
    expect_key("4k3/8/8/3n4/4Q3/8/8/4K3 w - - 0 1",
               "e4d5", 28, 35, QUEEN, 0, KNIGHT + 1, "queen takes knight");
    expect_key("4k3/8/8/8/8/8/4p3/4K3 w - - 0 1",
               "e1e2", 4, 12, KING, 0, PAWN + 1, "king takes pawn");

    /* en passant: cap must be PAWN + 1 even though board[to] is empty */
    expect_key("rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",
               "e5f6", 36, 45, PAWN, 0, PAWN + 1, "en passant");
    expect_key_mirrors("rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",
                       "e5f6", "e4f3", "en passant");

    /* promotions: promo 1..4 == N,B,R,Q */
    const char *pfen = "n3k3/1P6/8/8/8/8/8/4K3 w - - 0 1";
    expect_key(pfen, "b7b8n", 49, 57, PAWN, 1, 0, "promotion to knight");
    expect_key(pfen, "b7b8b", 49, 57, PAWN, 2, 0, "promotion to bishop");
    expect_key(pfen, "b7b8r", 49, 57, PAWN, 3, 0, "promotion to rook");
    expect_key(pfen, "b7b8q", 49, 57, PAWN, 4, 0, "promotion to queen");

    /* capture-promotions: promo and cap both set */
    expect_key(pfen, "b7a8n", 49, 56, PAWN, 1, KNIGHT + 1, "capture-promotion to knight");
    expect_key(pfen, "b7a8b", 49, 56, PAWN, 2, KNIGHT + 1, "capture-promotion to bishop");
    expect_key(pfen, "b7a8r", 49, 56, PAWN, 3, KNIGHT + 1, "capture-promotion to rook");
    expect_key(pfen, "b7a8q", 49, 56, PAWN, 4, KNIGHT + 1, "capture-promotion to queen");
    expect_key_mirrors(pfen, "b7b8q", "b2b1q", "promotion");
    expect_key_mirrors(pfen, "b7a8n", "b2a1n", "capture-promotion");

    /* castling is a king move with no capture */
    expect_key("4k3/8/8/8/8/8/8/R3K2R w KQ - 0 1", "e1g1", 4, 6, KING, 0, 0, "king-side castle");
    expect_key("4k3/8/8/8/8/8/8/R3K2R w KQ - 0 1", "e1c1", 4, 2, KING, 0, 0, "queen-side castle");

    /* every legal move of several busy positions must have consistent fields */
    const char *busy[] = {
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", /* kiwipete   */
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R b KQkq - 0 1", /* black side */
        "n3k3/1P6/8/8/8/8/8/4K3 w - - 0 1",                                     /* promotions */
        "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",        /* ep         */
    };
    int total = 0;
    for (size_t bi = 0; bi < sizeof busy / sizeof busy[0]; bi++) {
        Position p;
        Move mv[MAX_MOVES];
        CHECK(pos_from_fen(&p, busy[bi]) == 1, "busy FEN %zu parses", bi);
        const int nm = gen_legal(&p, mv);
        CHECK(nm > 0, "busy position %zu has legal moves", bi);
        const int flip = (p.side == BLACK) ? 56 : 0;
        int bad = 0;
        for (int i = 0; i < nm; i++) {
            MoveKey k;
            nn_move_key(&p, mv[i], &k);
            if (k.from > 63 || k.to > 63 || k.pc > 5 || k.promo > 4 || k.cap > 6) bad++;
            if (k.from != (MV_FROM(mv[i]) ^ flip)) bad++;
            if (k.to   != (MV_TO(mv[i])   ^ flip)) bad++;
            if (k.pc   != (int)p.board[MV_FROM(mv[i])]) bad++;
            if (MV_IS_PROMO(mv[i])  && k.promo != MV_PROMO_PIECE(mv[i]) - KNIGHT + 1) bad++;
            if (!MV_IS_PROMO(mv[i]) && k.promo != 0) bad++;
            if (MV_IS_CAPTURE(mv[i])  && k.cap == 0) bad++;
            if (!MV_IS_CAPTURE(mv[i]) && k.cap != 0) bad++;
            if (MV_FLAG(mv[i]) == MF_EP && k.cap != PAWN + 1) bad++;
            if (MV_IS_CAPTURE(mv[i]) && MV_FLAG(mv[i]) != MF_EP &&
                k.cap != (int)p.board[MV_TO(mv[i])] + 1) bad++;
        }
        CHECK(bad == 0, "%d inconsistent move keys in busy position %zu (%d moves)", bad, bi, nm);
        total += nm;
    }
    printf("  checked every field of %d legal move keys across %zu positions\n",
           total, sizeof busy / sizeof busy[0]);
}

/* ==================================================== 3. GRADIENT CHECK */

#define NTENSOR 26

typedef struct {
    const char *name;
    float       *w;      /* base of the tensor inside the working weights */
    const float *g;      /* base of the tensor inside the gradients       */
    int  n;              /* number of entries                             */
    int  stride;         /* row stride when sampling by row, else 0       */
    const int *rows;     /* reached rows                                  */
    int  nrows;
    const int *idxs;     /* reached flat indices (Bft)                    */
    int  nidx;
} Tensor;

typedef struct {
    const char *name;
    double worst_rel, worst_abs, gmax;
    int tested, kinks, over;
} Stat;

static const char *TENSOR_NAMES[NTENSOR] = {
    "W0", "b0", "g0", "c0", "W1", "b1", "g1", "c1",
    "W0v", "b0v", "g0v", "c0v",
    "z", "Wvh", "bvh", "gv", "cv", "Wv", "bv", "Wp",
    "Efrom", "Eto", "Epc", "Epromo", "Ecap", "Bft"
};

static const char *GRAD_FENS[] = {
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",              /* start       */
    "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",   /* tactical    */
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R b KQkq - 3 2",  /* black moves */
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 27 1",                            /* endgame     */
    "n3k3/1P6/8/8/8/8/8/4K3 w - - 0 1",                                      /* promotions  */
    "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",         /* en passant  */
};
#define NGRAD_FEN ((int)(sizeof GRAD_FENS / sizeof GRAD_FENS[0]))

/* Total number of relu gates in one forward pass. */
#define NGATE (NF_ACC + NF_HID + NF_VACC + NF_VHID)

/* Forward pass + scalar loss, plus the relu gate pattern so kinks can be seen. */
static double fwd_loss(const Trunk *t, const Head *h, const uint16_t *fidx, int nf,
                       const MoveKey *keys, int n, const float *dl, float dv,
                       Fwd *fwout, unsigned char *gate)
{
    Fwd fw;
    float logits[MAX_MOVES];
    nn_eval(t, h, fidx, nf, &fw);
    nn_logits(h, &fw, keys, n, logits);
    double L = 0.0;
    for (int i = 0; i < n; i++) L += (double)dl[i] * (double)logits[i];
    L += (double)dv * (double)fw.v;
    if (fwout) *fwout = fw;
    if (gate) {
        /* The three relu gate patterns, read off fields that survive the
         * residual: h1 is a plain relu, the block's relu is h2 minus the
         * identity term, and hv is a plain relu.  A central difference is only
         * valid while all three are unchanged. */
        int n = 0;
        for (int i = 0; i < NF_ACC; i++)  gate[n++] = fw.h1[i] > 0.0f;
        for (int j = 0; j < NF_HID; j++)  gate[n++] = (fw.h2[j] - fw.h1[j]) > 0.0f;
        for (int i = 0; i < NF_VACC; i++) gate[n++] = fw.hv0[i] > 0.0f;
        for (int k = 0; k < NF_VHID; k++) gate[n++] = fw.hv[k] > 0.0f;
    }
    return L;
}

static void check_zero_rows(const char *name, const float *g, int nrows_total, int stride,
                            const int *rows, int nrows)
{
    unsigned char *mark = (unsigned char *)calloc((size_t)nrows_total, 1);
    for (int i = 0; i < nrows; i++) mark[rows[i]] = 1;
    int bad = 0, badrow = -1;
    for (int r = 0; r < nrows_total; r++) {
        if (mark[r]) continue;
        for (int c = 0; c < stride; c++)
            if (g[(size_t)r * stride + c] != 0.0f) { bad++; if (badrow < 0) badrow = r; }
    }
    CHECK(bad == 0, "%s: %d unreached entries non-zero (first bad row %d)", name, bad, badrow);
    free(mark);
}

static void check_zero_idxs(const char *name, const float *g, int n, const int *idxs, int nidx)
{
    unsigned char *mark = (unsigned char *)calloc((size_t)n, 1);
    for (int i = 0; i < nidx; i++) mark[idxs[i]] = 1;
    int bad = 0, badi = -1;
    for (int i = 0; i < n; i++)
        if (!mark[i] && g[i] != 0.0f) { bad++; if (badi < 0) badi = i; }
    CHECK(bad == 0, "%s: %d unreached entries non-zero (first bad index %d)", name, bad, badi);
    free(mark);
}

/* Randomise every tensor so nothing is left at its zero initialisation
 * (z, Bft and the value head all start at zero and would hide bugs). */
static void perturb_weights(Trunk *t, Head *h)
{
    for (size_t i = 0; i < (size_t)NF_INPUT * NF_ACC; i++) t->W0[i] += 0.02f * rnd_norm();
    for (int i = 0; i < NF_ACC; i++) t->b0[i] += 0.30f * rnd_norm();
    for (size_t i = 0; i < (size_t)NF_ACC * NF_HID; i++) t->W1[i] += 0.02f * rnd_norm();
    for (int j = 0; j < NF_HID; j++) t->b1[j] += 0.30f * rnd_norm();

    /* The LayerNorm gains start at exactly 1 and the biases at exactly 0.  A
     * gain left at 1 would hide a missing factor of g in the backward pass and
     * a bias left at 0 would hide a sign error, so both are moved well away
     * from their initial values -- the gains stay positive so the relu gate
     * pattern is not shredded. */
    for (int i = 0; i < NF_ACC; i++) { t->g0[i] += 0.40f * rnd_sym(); t->c0[i] += 0.30f * rnd_norm(); }
    for (int j = 0; j < NF_HID; j++) { t->g1[j] += 0.40f * rnd_sym(); t->c1[j] += 0.30f * rnd_norm(); }

    for (size_t i = 0; i < (size_t)NF_INPUT * NF_VACC; i++) t->W0v[i] += 0.02f * rnd_norm();
    for (int i = 0; i < NF_VACC; i++) {
        t->b0v[i] += 0.30f * rnd_norm();
        t->g0v[i] += 0.40f * rnd_sym();
        t->c0v[i] += 0.30f * rnd_norm();
    }

    for (int i = 0; i < NF_ACC; i++) h->z[i]  += 0.30f * rnd_norm();

    for (size_t i = 0; i < (size_t)NF_VACC * NF_VHID; i++) h->Wvh[i] += 0.02f * rnd_norm();
    for (int k = 0; k < NF_VHID; k++) {
        h->bvh[k] += 0.30f * rnd_norm();
        h->gv[k]  += 0.40f * rnd_sym();
        h->cv[k]  += 0.30f * rnd_norm();
        h->Wv[k]  += 0.20f * rnd_norm();
    }
    h->bv[0] += 0.20f * rnd_norm();
    for (size_t i = 0; i < (size_t)NF_HID * NF_PDIM; i++) h->Wp[i]    += 0.02f * rnd_norm();
    for (size_t i = 0; i < (size_t)64 * NF_PDIM; i++)     h->Efrom[i] += 0.05f * rnd_norm();
    for (size_t i = 0; i < (size_t)64 * NF_PDIM; i++)     h->Eto[i]   += 0.05f * rnd_norm();
    for (size_t i = 0; i < (size_t)6  * NF_PDIM; i++)     h->Epc[i]   += 0.05f * rnd_norm();
    for (size_t i = 0; i < (size_t)5  * NF_PDIM; i++)     h->Epromo[i]+= 0.05f * rnd_norm();
    for (size_t i = 0; i < (size_t)7  * NF_PDIM; i++)     h->Ecap[i]  += 0.05f * rnd_norm();
    for (size_t i = 0; i < (size_t)64 * 64; i++)          h->Bft[i]   += 0.10f * rnd_norm();
}

/* Absolute-error floor for the comparison.
 *
 * The forward pass is float32, so L is computed with an absolute error of about
 * eps * (sum of the magnitudes of the terms that make it up).  A central
 * difference divides by 2h, so the numeric gradient carries an unavoidable
 * absolute noise of ~KAPPA * eps * Lscale / h.  Entries whose true gradient is
 * comparable to that simply cannot be resolved by finite differences, and
 * comparing them relatively would produce meaningless numbers -- so that noise
 * level is used as the floor of the relative-error denominator.
 *
 * Note this floor is proportional to 1/h: the h = 1e-2 sweep therefore runs
 * against a floor ten times tighter than the h = 1e-3 sweep, entirely from the
 * same formula.  A real gradient bug does not shrink when h grows; float noise
 * does.  KAPPA = 96 was chosen so the worst observed relative error stays at
 * roughly half the 2e-2 tolerance across a 30-seed sweep. */
#define FD_KAPPA 96.0

static double loss_scale(const Trunk *t, const Head *h, const uint16_t *fidx, int nf,
                         const MoveKey *keys, int n, const float *dl, float dv)
{
    Fwd fw;
    float logits[MAX_MOVES];
    nn_eval(t, h, fidx, nf, &fw);
    nn_logits(h, &fw, keys, n, logits);
    double s = 0.0;
    for (int i = 0; i < n; i++) s += fabs((double)dl[i] * (double)logits[i]);
    return s + fabs((double)dv * (double)fw.v);
}

/* One full sweep: every tensor, every position, `trials` central differences. */
static void grad_sweep(Trunk *t, Head *h, float H, double tol,
                       Stat *agg, int trials, int verbose)
{
    Trunk *tg = (Trunk *)malloc(sizeof(Trunk));
    Head  *hg = (Head  *)malloc(sizeof(Head));
    if (!tg || !hg) { printf("  FAIL: OOM\n"); g_fail++; free(tg); free(hg); return; }

    for (int i = 0; i < NTENSOR; i++) {
        memset(&agg[i], 0, sizeof agg[i]);
        agg[i].name = TENSOR_NAMES[i];
    }

    for (int fi = 0; fi < NGRAD_FEN; fi++) {
        Position p;
        if (!pos_from_fen(&p, GRAD_FENS[fi])) { CHECK(0, "bad FEN %s", GRAD_FENS[fi]); continue; }

        uint16_t fidx[NF_MAXACTIVE];
        const int nf = nn_features(&p, fidx);
        Move mv[MAX_MOVES];
        const int nm = gen_legal(&p, mv);
        CHECK(nm > 0, "%s has legal moves", GRAD_FENS[fi]);
        if (nm <= 0) continue;

        MoveKey keys[MAX_MOVES];
        for (int i = 0; i < nm; i++) nn_move_key(&p, mv[i], &keys[i]);

        float dl[MAX_MOVES];
        for (int i = 0; i < nm; i++) dl[i] = 0.5f * rnd_sym();
        const float dv = rnd_sym();

        const double Lscale = loss_scale(t, h, fidx, nf, keys, nm, dl, dv);
        const double floor_abs = FD_KAPPA * (double)FLT_EPSILON * Lscale / (double)H;

        grad_zero(tg, (int)TRUNK_NPARAM);
        grad_zero(hg, (int)HEAD_NPARAM);
        Fwd fw;
        unsigned char gate0[NGATE];
        fwd_loss(t, h, fidx, nf, keys, nm, dl, dv, &fw, gate0);
        nn_backward(t, h, &fw, fidx, nf, keys, nm, dl, dv, tg, hg);

        /* which rows / indices the forward pass actually touched */
        int rows_w0[NF_MAXACTIVE];
        for (int i = 0; i < nf; i++) rows_w0[i] = fidx[i];
        int r_from[64], r_to[64], r_pc[6], r_promo[5], r_cap[7], r_bft[MAX_MOVES];
        int n_from = 0, n_to = 0, n_pc = 0, n_promo = 0, n_cap = 0, n_bft = 0;
        unsigned char s_from[64] = {0}, s_to[64] = {0}, s_pc[6] = {0};
        unsigned char s_promo[5] = {0}, s_cap[7] = {0};
        static unsigned char s_bft[64 * 64];
        memset(s_bft, 0, sizeof s_bft);
        for (int i = 0; i < nm; i++) {
            const MoveKey k = keys[i];
            if (!s_from[k.from])   { s_from[k.from] = 1;   r_from[n_from++]   = k.from; }
            if (!s_to[k.to])       { s_to[k.to] = 1;       r_to[n_to++]       = k.to; }
            if (!s_pc[k.pc])       { s_pc[k.pc] = 1;       r_pc[n_pc++]       = k.pc; }
            if (!s_promo[k.promo]) { s_promo[k.promo] = 1; r_promo[n_promo++] = k.promo; }
            if (!s_cap[k.cap])     { s_cap[k.cap] = 1;     r_cap[n_cap++]     = k.cap; }
            const int b = k.from * 64 + k.to;
            if (!s_bft[b])         { s_bft[b] = 1;         r_bft[n_bft++]     = b; }
        }

        if (verbose) {                    /* unreached entries must be exactly zero */
            check_zero_rows("W0",     tg->W0,    NF_INPUT, NF_ACC,  rows_w0, nf);
            check_zero_rows("W0v",    tg->W0v,   NF_INPUT, NF_VACC, rows_w0, nf);
            check_zero_rows("Efrom",  hg->Efrom, 64, NF_PDIM, r_from,  n_from);
            check_zero_rows("Eto",    hg->Eto,   64, NF_PDIM, r_to,    n_to);
            check_zero_rows("Epc",    hg->Epc,    6, NF_PDIM, r_pc,    n_pc);
            check_zero_rows("Epromo", hg->Epromo, 5, NF_PDIM, r_promo, n_promo);
            check_zero_rows("Ecap",   hg->Ecap,   7, NF_PDIM, r_cap,   n_cap);
            check_zero_idxs("Bft",    hg->Bft, 64 * 64, r_bft, n_bft);
        }

        const Tensor ts[NTENSOR] = {
            { "W0",     t->W0,     tg->W0,     NF_INPUT*NF_ACC, NF_ACC,  rows_w0, nf,      NULL,  0 },
            { "b0",     t->b0,     tg->b0,     NF_ACC,          0,       NULL,    0,       NULL,  0 },
            { "g0",     t->g0,     tg->g0,     NF_ACC,          0,       NULL,    0,       NULL,  0 },
            { "c0",     t->c0,     tg->c0,     NF_ACC,          0,       NULL,    0,       NULL,  0 },
            { "W1",     t->W1,     tg->W1,     NF_ACC*NF_HID,   0,       NULL,    0,       NULL,  0 },
            { "b1",     t->b1,     tg->b1,     NF_HID,          0,       NULL,    0,       NULL,  0 },
            { "g1",     t->g1,     tg->g1,     NF_HID,          0,       NULL,    0,       NULL,  0 },
            { "c1",     t->c1,     tg->c1,     NF_HID,          0,       NULL,    0,       NULL,  0 },
            { "W0v",    t->W0v,    tg->W0v,    NF_INPUT*NF_VACC, NF_VACC, rows_w0, nf,     NULL,  0 },
            { "b0v",    t->b0v,    tg->b0v,    NF_VACC,         0,       NULL,    0,       NULL,  0 },
            { "g0v",    t->g0v,    tg->g0v,    NF_VACC,         0,       NULL,    0,       NULL,  0 },
            { "c0v",    t->c0v,    tg->c0v,    NF_VACC,         0,       NULL,    0,       NULL,  0 },
            { "z",      h->z,      hg->z,      NF_ACC,          0,       NULL,    0,       NULL,  0 },
            { "Wvh",    h->Wvh,    hg->Wvh,    NF_VACC*NF_VHID, 0,       NULL,    0,       NULL,  0 },
            { "bvh",    h->bvh,    hg->bvh,    NF_VHID,         0,       NULL,    0,       NULL,  0 },
            { "gv",     h->gv,     hg->gv,     NF_VHID,         0,       NULL,    0,       NULL,  0 },
            { "cv",     h->cv,     hg->cv,     NF_VHID,         0,       NULL,    0,       NULL,  0 },
            { "Wv",     h->Wv,     hg->Wv,     NF_VHID,         0,       NULL,    0,       NULL,  0 },
            { "bv",     h->bv,     hg->bv,     1,               0,       NULL,    0,       NULL,  0 },
            { "Wp",     h->Wp,     hg->Wp,     NF_HID*NF_PDIM,  0,       NULL,    0,       NULL,  0 },
            { "Efrom",  h->Efrom,  hg->Efrom,  64*NF_PDIM,      NF_PDIM, r_from,  n_from,  NULL,  0 },
            { "Eto",    h->Eto,    hg->Eto,    64*NF_PDIM,      NF_PDIM, r_to,    n_to,    NULL,  0 },
            { "Epc",    h->Epc,    hg->Epc,     6*NF_PDIM,      NF_PDIM, r_pc,    n_pc,    NULL,  0 },
            { "Epromo", h->Epromo, hg->Epromo,  5*NF_PDIM,      NF_PDIM, r_promo, n_promo, NULL,  0 },
            { "Ecap",   h->Ecap,   hg->Ecap,    7*NF_PDIM,      NF_PDIM, r_cap,   n_cap,   NULL,  0 },
            { "Bft",    h->Bft,    hg->Bft,    64*64,           0,       NULL,    0,       r_bft, n_bft },
        };

        for (int ti = 0; ti < NTENSOR; ti++) {
            const Tensor *T = &ts[ti];

            double gmax = 0.0;
            for (int i = 0; i < T->n; i++) {
                const double a = fabs((double)T->g[i]);
                if (a > gmax) gmax = a;
            }
            if (gmax > agg[ti].gmax) agg[ti].gmax = gmax;

            int done = 0, draws = 0;
            while (done < trials && draws < trials * 50) {
                draws++;
                int idx;
                if (T->idxs)        idx = T->idxs[rnd_int(T->nidx)];
                else if (T->stride) idx = T->rows[rnd_int(T->nrows)] * T->stride + rnd_int(T->stride);
                else                idx = rnd_int(T->n);

                float *w = &T->w[idx];
                const float orig = *w;
                unsigned char gp[NGATE], gm[NGATE];

                *w = orig + H;
                const double Lp = fwd_loss(t, h, fidx, nf, keys, nm, dl, dv, NULL, gp);
                *w = orig - H;
                const double Lm = fwd_loss(t, h, fidx, nf, keys, nm, dl, dv, NULL, gm);
                *w = orig;

                /* a relu gate flip makes the central difference invalid here */
                if (memcmp(gp, gate0, sizeof gate0) != 0 || memcmp(gm, gate0, sizeof gate0) != 0) {
                    agg[ti].kinks++;
                    continue;
                }

                const double gnum = (Lp - Lm) / (2.0 * (double)H);
                const double gana = (double)T->g[idx];
                const double aerr = fabs(gnum - gana);
                double denom = fabs(gana) > fabs(gnum) ? fabs(gana) : fabs(gnum);
                if (denom < floor_abs) denom = floor_abs;
                const double rel = aerr / denom;

                if (rel > agg[ti].worst_rel) agg[ti].worst_rel = rel;
                if (aerr > agg[ti].worst_abs) agg[ti].worst_abs = aerr;
                agg[ti].tested++;
                done++;

                if (rel > tol) {
                    agg[ti].over++;
                    printf("    %-7s h=%.0e pos %d idx %-8d analytic %+.6e numeric %+.6e rel %.3e\n",
                           T->name, (double)H, fi, idx, gana, gnum, rel);
                }
            }
            CHECK(done == trials, "%s: only %d/%d usable perturbations for position %d",
                  T->name, done, trials, fi);
        }
    }
    free(tg);
    free(hg);
}

static void tensor_report(const char *title, const Stat *s, double tol)
{
    printf("\n  %s\n", title);
    printf("  %-8s %8s %6s %14s %14s %14s\n",
           "tensor", "tested", "kinks", "worst rel err", "worst abs err", "max |grad|");
    printf("  %-8s %8s %6s %14s %14s %14s\n",
           "--------", "--------", "------", "--------------", "--------------", "--------------");
    for (int i = 0; i < NTENSOR; i++)
        printf("  %-8s %8d %6d %14.3e %14.3e %14.3e%s\n",
               s[i].name, s[i].tested, s[i].kinks, s[i].worst_rel, s[i].worst_abs, s[i].gmax,
               s[i].worst_rel > tol ? "   <-- FAIL" : "");
}

static void test_gradients(void)
{
    section("3. gradient check (central differences)");

    const double TOL = 2e-2;

    Trunk *t = (Trunk *)malloc(sizeof(Trunk));
    Head  *h = (Head  *)malloc(sizeof(Head));
    if (!t || !h) { printf("  FAIL: OOM\n"); g_fail++; free(t); free(h); return; }

    nn_init(t, h, 0xC0FFEEULL);
    perturb_weights(t, h);

    {   /* The check would be vacuous if a tensor were still at its initial
         * value.  For the LayerNorm gains that means 1, not 0: a gradient bug
         * that drops the gain entirely is invisible while g == 1. */
        double s_z = 0, s_bft = 0, s_wv = 0, s_b0 = 0, s_b1 = 0;
        double s_wvh = 0, s_bvh = 0, s_cv = 0, s_c0 = 0, s_c1 = 0, s_b0v = 0, s_c0v = 0;
        int g_at_one = 0;
        for (int i = 0; i < NF_ACC; i++) {
            s_z += fabs(h->z[i]); s_b0 += fabs(t->b0[i]); s_c0 += fabs(t->c0[i]);
            if (t->g0[i] == 1.0f) g_at_one++;
        }
        for (int i = 0; i < NF_HID; i++) {
            s_b1 += fabs(t->b1[i]); s_c1 += fabs(t->c1[i]);
            if (t->g1[i] == 1.0f) g_at_one++;
        }
        for (int k = 0; k < NF_VHID; k++) {
            s_wv += fabs(h->Wv[k]); s_bvh += fabs(h->bvh[k]); s_cv += fabs(h->cv[k]);
            if (h->gv[k] == 1.0f) g_at_one++;
        }
        for (size_t i = 0; i < (size_t)NF_VACC * NF_VHID; i++) s_wvh += fabs(h->Wvh[i]);
        for (int i = 0; i < NF_VACC; i++) {
            s_b0v += fabs(t->b0v[i]); s_c0v += fabs(t->c0v[i]);
            if (t->g0v[i] == 1.0f) g_at_one++;
        }
        for (int i = 0; i < 64 * 64; i++) s_bft += fabs(h->Bft[i]);
        CHECK(s_z > 0 && s_bft > 0 && s_wv > 0 && s_b0 > 0 && s_b1 > 0 && h->bv[0] != 0.0f &&
              s_wvh > 0 && s_bvh > 0 && s_cv > 0 && s_c0 > 0 && s_c1 > 0 &&
              s_b0v > 0 && s_c0v > 0,
              "every tensor must be non-zero before the gradient check");
        CHECK(g_at_one == 0, "%d LayerNorm gains are still exactly 1: a missing "
              "factor of g in the backward pass would not be visible", g_at_one);
    }

    /* ---- the two heads must be gradient-isolated --------------------------
     * The whole point of giving the value head its own hidden layer is that
     * those parameters are reached by the value loss and by nothing else.  That
     * is a checkable claim: with dvalue = 0 every value-head tensor must come
     * back EXACTLY zero, and with dlogits = 0 every policy tensor must.  (The
     * trunk is shared on purpose and is excluded from both directions.) */
    {
        Trunk *tg = (Trunk *)malloc(sizeof(Trunk));
        Head  *hg = (Head  *)malloc(sizeof(Head));
        Position p;
        uint16_t fidx[NF_MAXACTIVE];
        Move mv[MAX_MOVES];
        MoveKey keys[MAX_MOVES];
        float dl[MAX_MOVES];
        Fwd fw;

        if (!tg || !hg || !pos_from_fen(&p, GRAD_FENS[1])) {
            CHECK(0, "isolation setup");
        } else {
            const int nf = nn_features(&p, fidx);
            const int nm = gen_legal(&p, mv);
            for (int i = 0; i < nm; i++) nn_move_key(&p, mv[i], &keys[i]);
            nn_eval(t, h, fidx, nf, &fw);

            /* value loss only */
            for (int i = 0; i < nm; i++) dl[i] = 0.0f;
            grad_zero(tg, (int)TRUNK_NPARAM);
            grad_zero(hg, (int)HEAD_NPARAM);
            nn_backward(t, h, &fw, fidx, nf, keys, nm, dl, 0.7f, tg, hg);
            int pol_nz = 0, val_nz = 0;
            for (size_t i = 0; i < (size_t)NF_HID * NF_PDIM; i++) if (hg->Wp[i] != 0.0f) pol_nz++;
            for (size_t i = 0; i < (size_t)64 * NF_PDIM; i++) {
                if (hg->Efrom[i] != 0.0f) pol_nz++;
                if (hg->Eto[i]   != 0.0f) pol_nz++;
            }
            for (int i = 0; i < 64 * 64; i++) if (hg->Bft[i] != 0.0f) pol_nz++;
            for (size_t i = 0; i < (size_t)NF_VACC * NF_VHID; i++) if (hg->Wvh[i] != 0.0f) val_nz++;
            CHECK(pol_nz == 0, "dvalue-only backward touched %d policy parameters", pol_nz);
            CHECK(val_nz > 0, "dvalue-only backward left the value hidden layer at zero");

            /* policy loss only */
            for (int i = 0; i < nm; i++) dl[i] = 0.5f * rnd_sym();
            grad_zero(tg, (int)TRUNK_NPARAM);
            grad_zero(hg, (int)HEAD_NPARAM);
            nn_backward(t, h, &fw, fidx, nf, keys, nm, dl, 0.0f, tg, hg);
            int vnz = 0, pnz = 0;
            for (size_t i = 0; i < (size_t)NF_VACC * NF_VHID; i++) if (hg->Wvh[i] != 0.0f) vnz++;
            for (int k = 0; k < NF_VHID; k++) {
                if (hg->bvh[k] != 0.0f) vnz++;
                if (hg->gv[k]  != 0.0f) vnz++;
                if (hg->cv[k]  != 0.0f) vnz++;
                if (hg->Wv[k]  != 0.0f) vnz++;
            }
            if (hg->bv[0] != 0.0f) vnz++;
            for (size_t i = 0; i < (size_t)NF_HID * NF_PDIM; i++) if (hg->Wp[i] != 0.0f) pnz++;
            CHECK(vnz == 0, "dlogits-only backward touched %d value-head parameters", vnz);
            CHECK(pnz > 0, "dlogits-only backward left Wp at zero");
            printf("  head isolation: value-only backward touches 0 policy parameters, "
                   "policy-only backward touches 0 of the %zu value-head parameters\n",
                   (size_t)NF_VACC * NF_VHID + 3 * NF_VHID + NF_VHID + 1);
        }
        free(tg);
        free(hg);
    }

    Stat a3[NTENSOR], a2[NTENSOR];
    grad_sweep(t, h, 1e-3f, TOL, a3, 40, 1);
    grad_sweep(t, h, 1e-2f, TOL, a2, 40, 0);

    tensor_report("h = 1e-3   (float32 fd noise floor ~ 96 * eps * |L| / h)", a3, TOL);
    tensor_report("h = 1e-2   (same formula, so the floor is 10x tighter)", a2, TOL);

    int worst3 = 0, worst2 = 0;
    double sum3 = 0.0, sum2 = 0.0;
    for (int i = 0; i < NTENSOR; i++) {
        CHECK(a3[i].tested >= 40, "%s: only %d perturbations at h=1e-3", a3[i].name, a3[i].tested);
        CHECK(a2[i].tested >= 40, "%s: only %d perturbations at h=1e-2", a2[i].name, a2[i].tested);
        CHECK(a3[i].worst_rel <= TOL, "%s: worst relative error %.3e exceeds %.1e at h=1e-3",
              a3[i].name, a3[i].worst_rel, TOL);
        CHECK(a2[i].worst_rel <= TOL, "%s: worst relative error %.3e exceeds %.1e at h=1e-2",
              a2[i].name, a2[i].worst_rel, TOL);
        if (a3[i].worst_rel > a3[worst3].worst_rel) worst3 = i;
        if (a2[i].worst_rel > a2[worst2].worst_rel) worst2 = i;
        sum3 += a3[i].worst_abs;
        sum2 += a2[i].worst_abs;
    }

    /* A real gradient error is independent of h.  The residual here is not: it
     * is float32 ROUNDING, which falls as 1/h, plus the central difference's own
     * TRUNCATION error (h^2/6) f''', which grows as h^2.  The normalisation
     * layers make the loss genuinely nonlinear in every individual weight -- one
     * weight moves the whole layer's mean and variance, and so every unit in it
     * -- which makes the truncation term matter at h = 1e-2, so the fall is well
     * short of the 10x that pure rounding would give.  The check is therefore
     * that it falls; the positive control below is what gives the tolerance its
     * teeth. */
    const double ratio = sum2 > 0.0 ? sum3 / sum2 : 0.0;
    printf("\n  worst tensor: %s %.3e at h=1e-3, %s %.3e at h=1e-2 (tolerance %.1e)\n",
           a3[worst3].name, a3[worst3].worst_rel, a2[worst2].name, a2[worst2].worst_rel, TOL);
    printf("  residual falls as h grows (sum of worst abs errors %.2e vs %.2e, ratio %.1fx)\n"
           "  => rounding noise, not an h-independent gradient error\n",
           sum3, sum2, ratio);
    CHECK(ratio > 1.2, "residual did not shrink at all with larger h (ratio %.2f); "
          "that would indicate a real gradient error, not float noise", ratio);

    /* ---- positive control ------------------------------------------------
     * The h-scaling argument says the residual is noise.  This says the
     * tolerance still has teeth: inject a 3%% systematic error into ONE tensor's
     * analytic gradient and require the very same comparison to reject it.  A
     * check that can never fail proves nothing. */
    {
        Trunk *tg = (Trunk *)malloc(sizeof(Trunk));
        Head  *hg = (Head  *)malloc(sizeof(Head));
        Position p;
        uint16_t fidx[NF_MAXACTIVE];
        Move mv[MAX_MOVES];
        MoveKey keys[MAX_MOVES];
        float dl[MAX_MOVES];
        Fwd fw;
        unsigned char g0[NGATE], gp[NGATE], gm[NGATE];

        if (!tg || !hg || !pos_from_fen(&p, GRAD_FENS[1])) {
            CHECK(0, "positive-control setup");
        } else {
            const int nf = nn_features(&p, fidx);
            const int nm = gen_legal(&p, mv);
            for (int i = 0; i < nm; i++) nn_move_key(&p, mv[i], &keys[i]);
            for (int i = 0; i < nm; i++) dl[i] = 0.5f * rnd_sym();
            const float dv = rnd_sym();
            /* h = 1e-2 here, not 1e-3: the finite-difference noise floor is
             * proportional to 1/h, and at 1e-3 it swamps every entry of dW1, so
             * a 3% error in any of them is genuinely unresolvable and testing
             * for it would test nothing. */
            const float H = 1e-2f;

            grad_zero(tg, (int)TRUNK_NPARAM);
            grad_zero(hg, (int)HEAD_NPARAM);
            fwd_loss(t, h, fidx, nf, keys, nm, dl, dv, &fw, g0);
            nn_backward(t, h, &fw, fidx, nf, keys, nm, dl, dv, tg, hg);

            const double Lscale = loss_scale(t, h, fidx, nf, keys, nm, dl, dv);
            const double floor_abs = FD_KAPPA * (double)FLT_EPSILON * Lscale / (double)H;

            /* Use the entries with the LARGEST analytic gradient: those are the
             * ones a finite difference can resolve, so a 3% error in them is a
             * fair test of the tolerance rather than a test of float noise. */
            enum { NCTRL = 40 };
            int top[NCTRL];
            {
                int n = 0;
                for (int i = 0; i < NF_ACC * NF_HID; i++) {
                    const float a = fabsf(tg->W1[i]);
                    if (n < NCTRL) {
                        top[n++] = i;
                    } else {
                        int worst = 0;
                        for (int k = 1; k < NCTRL; k++)
                            if (fabsf(tg->W1[top[k]]) < fabsf(tg->W1[top[worst]])) worst = k;
                        if (a > fabsf(tg->W1[top[worst]])) top[worst] = i;
                    }
                }
            }

            int caught = 0, tried = 0;
            for (int trial = 0; trial < NCTRL; trial++) {
                const int idx = top[trial];
                const double gana = (double)tg->W1[idx] * 1.03;   /* the injected 3%% */
                if (fabs(gana) < 5.0 * floor_abs) continue;       /* unresolvable */

                float *w = &t->W1[idx];
                const float orig = *w;
                *w = orig + H;
                const double Lp = fwd_loss(t, h, fidx, nf, keys, nm, dl, dv, NULL, gp);
                *w = orig - H;
                const double Lm = fwd_loss(t, h, fidx, nf, keys, nm, dl, dv, NULL, gm);
                *w = orig;
                if (memcmp(gp, g0, sizeof g0) != 0 || memcmp(gm, g0, sizeof g0) != 0) continue;

                const double gnum = (Lp - Lm) / (2.0 * (double)H);
                double denom = fabs(gana) > fabs(gnum) ? fabs(gana) : fabs(gnum);
                if (denom < floor_abs) denom = floor_abs;
                tried++;
                if (fabs(gnum - gana) / denom > TOL) caught++;
            }
            CHECK(tried >= 20, "positive control: only %d resolvable W1 entries", tried);
            CHECK(caught * 4 >= tried * 3,
                  "positive control: a 3%% error in dW1 was caught in only %d/%d entries -- "
                  "the %.1e tolerance is too loose to detect a real gradient bug",
                  caught, tried, TOL);
            printf("  positive control: a 3%% error injected into dL/dW1 is rejected in "
                   "%d of %d resolvable entries\n", caught, tried);
        }
        free(tg);
        free(hg);
    }

    free(t);
    free(h);
}

/* ===================================================== 4. ACCUMULATION */

/* Entries that receive exactly one `+=` per nn_backward call must double
 * bit-exactly (x + x == 2x in IEEE754).  Rows of the embedding tables that are
 * hit by several moves are summed in a different order on the second call, so
 * they are only required to double to within a few ulps. */
static void dbl_exact(const char *name, const float *a, const float *b, size_t n, int *bad)
{
    int local = 0;
    double worst = 0.0;
    for (size_t i = 0; i < n; i++)
        if (b[i] != 2.0f * a[i]) {
            local++;
            const double d = fabs((double)b[i] - 2.0 * (double)a[i]);
            if (d > worst) worst = d;
        }
    CHECK(local == 0, "%s: %d/%zu entries did not double bit-exactly (worst %.3e)",
          name, local, n, worst);
    *bad += local;
}

/* Rows hit by several moves are re-summed in the same order but from a non-zero
 * start, so the two runs round differently.  The bound on that is the float
 * rounding of the partial sums: ~eps * (sum of the term magnitudes), which the
 * caller supplies per entry in `absS`.  A `=` instead of `+=` in nn_backward
 * would leave |b - 2a| == |a| ~ absS -- five orders of magnitude above this
 * bound -- so the check keeps all of its teeth.  `margin` reports how much:
 * it is |a| / bound for the largest entry. */
static void dbl_rows(const char *name, const float *a, const float *b,
                     int nrows, int stride, const int *cnt, const double *absS)
{
    int bad_exact = 0, bad_tol = 0;
    double worst = 0.0, margin = 1e300;
    for (int r = 0; r < nrows; r++) {
        for (int c = 0; c < stride; c++) {
            const size_t i = (size_t)r * stride + c;
            if (cnt[r] <= 1) { if (b[i] != 2.0f * a[i]) bad_exact++; continue; }
            const double err = fabs((double)b[i] - 2.0 * (double)a[i]);
            const double bound = 64.0 * (double)FLT_EPSILON * absS[i];
            if (err > bound) bad_tol++;
            if (bound > 0.0) {
                if (err / bound > worst) worst = err / bound;
                const double m = fabs((double)a[i]) / bound;
                if (fabs((double)a[i]) > 0.0 && m < margin) margin = m;
            }
        }
    }
    CHECK(bad_exact == 0, "%s: %d single-accumulation entries did not double bit-exactly",
          name, bad_exact);
    CHECK(bad_tol == 0, "%s: %d multi-accumulation entries exceed the float rounding bound "
          "(worst %.2fx of it)", name, bad_tol, worst);
    if (margin < 1e299)
        printf("    %-12s doubling error <= %.2f of the rounding bound; an overwrite bug "
               "would be >= %.0fx it\n", name, worst, margin);
}

/* absS[row*stride + c] = sum over the moves hitting `row` of |contribution| */
static void build_absS(double *absS, int nrows, int stride, const uint8_t *rowof,
                       int nm, const float *dl, const float *q)
{
    for (int i = 0; i < nrows * stride; i++) absS[i] = 0.0;
    for (int m = 0; m < nm; m++) {
        const int r = rowof[m];
        for (int c = 0; c < stride; c++)
            absS[(size_t)r * stride + c] += fabs((double)dl[m]) *
                                            (q ? fabs((double)q[c]) : 1.0);
    }
}

static void test_accumulation(void)
{
    section("4. gradient accumulation (nn_backward must add, not overwrite)");

    Trunk *t  = (Trunk *)malloc(sizeof(Trunk));
    Head  *h  = (Head  *)malloc(sizeof(Head));
    Trunk *tg = (Trunk *)malloc(sizeof(Trunk));
    Head  *hg = (Head  *)malloc(sizeof(Head));
    Trunk *t1 = (Trunk *)malloc(sizeof(Trunk));
    Head  *h1 = (Head  *)malloc(sizeof(Head));
    if (!t || !h || !tg || !hg || !t1 || !h1) { printf("  FAIL: OOM\n"); g_fail++; return; }

    nn_init(t, h, 0xBEEF1234ULL);
    perturb_weights(t, h);

    Position p;
    CHECK(pos_from_fen(&p, "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4") == 1,
          "accumulation FEN parses");

    uint16_t fidx[NF_MAXACTIVE];
    const int nf = nn_features(&p, fidx);
    Move mv[MAX_MOVES];
    const int nm = gen_legal(&p, mv);
    MoveKey keys[MAX_MOVES];
    for (int i = 0; i < nm; i++) nn_move_key(&p, mv[i], &keys[i]);
    float dl[MAX_MOVES];
    for (int i = 0; i < nm; i++) dl[i] = 0.5f * rnd_sym();
    const float dv = rnd_sym();

    int c_from[64] = {0}, c_to[64] = {0}, c_pc[6] = {0}, c_promo[5] = {0}, c_cap[7] = {0};
    static int c_bft[64 * 64];
    memset(c_bft, 0, sizeof c_bft);
    for (int i = 0; i < nm; i++) {
        c_from[keys[i].from]++;  c_to[keys[i].to]++;    c_pc[keys[i].pc]++;
        c_promo[keys[i].promo]++; c_cap[keys[i].cap]++; c_bft[keys[i].from * 64 + keys[i].to]++;
    }

    Fwd fw;
    nn_eval(t, h, fidx, nf, &fw);

    grad_zero(tg, (int)TRUNK_NPARAM);
    grad_zero(hg, (int)HEAD_NPARAM);
    nn_backward(t, h, &fw, fidx, nf, keys, nm, dl, dv, tg, hg);
    memcpy(t1, tg, sizeof(Trunk));
    memcpy(h1, hg, sizeof(Head));
    nn_backward(t, h, &fw, fidx, nf, keys, nm, dl, dv, tg, hg);

    int nz = 0, nzh = 0, bad = 0;
    {
        const float *a = (const float *)t1, *b = (const float *)tg;
        for (size_t i = 0; i < TRUNK_NPARAM; i++) if (a[i] != 0.0f) nz++;
        dbl_exact("trunk (W0,b0,W1,b1)", a, b, TRUNK_NPARAM, &bad);
    }
    {
        const float *a = (const float *)h1;
        for (size_t i = 0; i < HEAD_NPARAM; i++) if (a[i] != 0.0f) nzh++;
    }
    dbl_exact("head z",   h1->z,   hg->z,   NF_ACC, &bad);
    dbl_exact("head Wvh", h1->Wvh, hg->Wvh, (size_t)NF_VACC * NF_VHID, &bad);
    dbl_exact("head bvh", h1->bvh, hg->bvh, NF_VHID, &bad);
    dbl_exact("head gv",  h1->gv,  hg->gv,  NF_VHID, &bad);
    dbl_exact("head cv",  h1->cv,  hg->cv,  NF_VHID, &bad);
    dbl_exact("head Wv",  h1->Wv,  hg->Wv,  NF_VHID, &bad);
    dbl_exact("head bv",  h1->bv,  hg->bv,  1,       &bad);
    dbl_exact("head Wp",  h1->Wp,  hg->Wp,  (size_t)NF_HID * NF_PDIM, &bad);
    {
        uint8_t rf[MAX_MOVES], rt[MAX_MOVES], rp[MAX_MOVES], rr[MAX_MOVES], rc[MAX_MOVES];
        static double absS[64 * 64];
        static int bft_row[MAX_MOVES];
        for (int i = 0; i < nm; i++) {
            rf[i] = keys[i].from; rt[i] = keys[i].to; rp[i] = keys[i].pc;
            rr[i] = keys[i].promo; rc[i] = keys[i].cap;
            bft_row[i] = keys[i].from * 64 + keys[i].to;
        }
        build_absS(absS, 64, NF_PDIM, rf, nm, dl, fw.q);
        dbl_rows("head Efrom", h1->Efrom, hg->Efrom, 64, NF_PDIM, c_from, absS);
        build_absS(absS, 64, NF_PDIM, rt, nm, dl, fw.q);
        dbl_rows("head Eto", h1->Eto, hg->Eto, 64, NF_PDIM, c_to, absS);
        build_absS(absS, 6, NF_PDIM, rp, nm, dl, fw.q);
        dbl_rows("head Epc", h1->Epc, hg->Epc, 6, NF_PDIM, c_pc, absS);
        build_absS(absS, 5, NF_PDIM, rr, nm, dl, fw.q);
        dbl_rows("head Epromo", h1->Epromo, hg->Epromo, 5, NF_PDIM, c_promo, absS);
        build_absS(absS, 7, NF_PDIM, rc, nm, dl, fw.q);
        dbl_rows("head Ecap", h1->Ecap, hg->Ecap, 7, NF_PDIM, c_cap, absS);

        for (int i = 0; i < 64 * 64; i++) absS[i] = 0.0;
        for (int m = 0; m < nm; m++) absS[bft_row[m]] += fabs((double)dl[m]);
        dbl_rows("head Bft", h1->Bft, hg->Bft, 64 * 64, 1, c_bft, absS);
    }

    CHECK(nz  > 1000, "trunk grad should be substantially non-zero, got %d", nz);
    CHECK(nzh > 500,  "head grad should be substantially non-zero, got %d", nzh);

    /* A third call must give 3x.  (Not bit-exact: 3*d needs one mantissa bit
     * more than d, so fl(2d + d) can round.  2*d, by contrast, is always exact,
     * which is why the doubling check above can demand bit equality.) */
    nn_backward(t, h, &fw, fidx, nf, keys, nm, dl, dv, tg, hg);
    {
        const float *a = (const float *)t1, *b = (const float *)tg;
        int bad3 = 0;
        double worst3 = 0.0;
        for (size_t i = 0; i < TRUNK_NPARAM; i++) {
            const double want = 3.0 * (double)a[i];
            const double e = fabs((double)b[i] - want) / (fabs(want) + 1e-30);
            if (a[i] != 0.0f && e > worst3) worst3 = e;
            if (a[i] != 0.0f && e > 1e-6) bad3++;
            if (a[i] == 0.0f && b[i] != 0.0f) bad3++;
        }
        CHECK(bad3 == 0, "trunk grads not tripled after a third call (%d entries, worst rel %.2e)",
              bad3, worst3);
    }

    printf("  trunk: %d/%zu non-zero, exactly doubled, tripled to <1e-6\n", nz, TRUNK_NPARAM);
    printf("  head : %d/%zu non-zero, doubled (bit-exact where a row is hit once)\n",
           nzh, HEAD_NPARAM);

    /* grad_add / grad_zero */
    {
        float a[4] = {1.0f, -2.0f, 0.5f, 0.0f};
        const float b[4] = {0.25f, 1.0f, -0.5f, 3.0f};
        grad_add(a, b, 4);
        CHECK(a[0] == 1.25f && a[1] == -1.0f && a[2] == 0.0f && a[3] == 3.0f,
              "grad_add is dst += src");
        grad_zero(a, 4);
        CHECK(a[0] == 0 && a[1] == 0 && a[2] == 0 && a[3] == 0, "grad_zero clears");
    }

    free(t); free(h); free(tg); free(hg); free(t1); free(h1);
}

/* ============================================================== 5. ADAM */

static void test_adam(void)
{
    section("5. AdamW");

    /* Minimise f(x) = sum_i 0.5 * c_i * (x_i - t_i)^2, grad = c_i * (x_i - t_i).
     *
     * Phase A (descent) uses a constant lr with every coordinate far from its
     * optimum.  Adam's per-element step is bounded by lr, so every coordinate
     * moves strictly towards its target and the loss MUST fall on every single
     * step -- that is the monotonicity assertion.
     *
     * Phase B (convergence) decays the lr geometrically and drives the
     * parameters onto the optimum.  Strict monotonicity is deliberately not
     * asserted there: once |x_i - t_i| < lr, Adam's sign-like step overshoots
     * and dithers around the optimum with amplitude ~lr.  That is a property of
     * Adam, not of this implementation. */
    enum { N = 64 };
    float x[N], g[N], c[N], target[N];
    for (int i = 0; i < N; i++) {
        c[i] = 0.5f + (float)rnd_uni() * 1.5f;
        target[i] = 2.0f * rnd_sym();
        /* start 2.0 .. 3.0 away, so 150 steps of <= 0.011 cannot overshoot */
        x[i] = target[i] + (rnd_uni() < 0.5 ? -1.0f : 1.0f) * (2.0f + (float)rnd_uni());
        g[i] = 0.0f;
    }

    Adam a;
    adam_init(&a, N);
    CHECK(a.m && a.v && a.n == N && a.t == 0, "adam_init allocates state");

    double prev = 0.0;
    for (int i = 0; i < N; i++) { const double d = x[i] - target[i]; prev += 0.5 * c[i] * d * d; }
    const double loss0 = prev;

    int nonmono = 0, notzeroed = 0;
    const int DESCENT = 150;
    for (int s = 0; s < DESCENT; s++) {
        for (int i = 0; i < N; i++) g[i] = c[i] * (x[i] - target[i]);
        adam_step(&a, x, g, 0.01f, 0.0f, 0.0f);
        for (int i = 0; i < N; i++) if (g[i] != 0.0f) notzeroed++;
        double L = 0.0;
        for (int i = 0; i < N; i++) { const double d = x[i] - target[i]; L += 0.5 * c[i] * d * d; }
        if (!(L < prev)) nonmono++;
        prev = L;
    }
    CHECK(notzeroed == 0, "adam_step must zero the grad buffer (%d entries left set)", notzeroed);
    CHECK(nonmono == 0, "loss not monotonically decreasing during descent (%d/%d steps increased)",
          nonmono, DESCENT);
    const double loss_a = prev;

    const int TAIL = 9000;
    for (int s = 0; s < TAIL; s++) {
        for (int i = 0; i < N; i++) g[i] = c[i] * (x[i] - target[i]);
        adam_step(&a, x, g, 0.02f * powf(0.999f, (float)s), 0.0f, 0.0f);
    }
    double L = 0.0, maxdev = 0.0;
    for (int i = 0; i < N; i++) {
        const double d = x[i] - target[i];
        L += 0.5 * c[i] * d * d;
        if (fabs(d) > maxdev) maxdev = fabs(d);
    }
    CHECK(L < loss0 * 1e-10, "final loss %.3e not << initial %.3e", L, loss0);
    CHECK(maxdev < 1e-4, "parameters did not converge: max |x - x*| = %.3e", maxdev);
    CHECK(a.t == DESCENT + TAIL, "adam step counter should be %d, got %d", DESCENT + TAIL, a.t);
    printf("  quadratic (%d dims): loss %.3e -> %.3e over %d strictly monotone descent steps,\n"
           "                       then -> %.3e after %d decayed steps, max |x-x*| = %.2e\n",
           N, loss0, loss_a, DESCENT, L, TAIL, maxdev);
    adam_free(&a);
    CHECK(a.m == NULL && a.v == NULL && a.n == 0, "adam_free clears the state");

    /* ---- global grad-norm clipping ---- */
    {
        enum { M = 16 };
        float p1[M], p2[M], gg[M], hh[M], gsave[M];
        double ss = 0.0;
        for (int i = 0; i < M; i++) { gg[i] = 10.0f * rnd_sym(); ss += (double)gg[i] * gg[i]; }
        const double norm = sqrt(ss);
        const float clip = 1.0f;
        const double scale = (double)clip / (norm + 1e-12);
        CHECK(norm > clip, "clipping test needs |g| > clip (got %.3f)", norm);

        Adam a1, a2;
        adam_init(&a1, M);
        adam_init(&a2, M);
        for (int i = 0; i < M; i++) { p1[i] = p2[i] = 0.0f; gsave[i] = gg[i]; }
        adam_step(&a1, p1, gg, 0.01f, 0.0f, clip);

        int bad = 0;
        double worst = 0.0;
        for (int i = 0; i < M; i++) {
            const double want = 0.1 * (double)gsave[i] * scale;      /* (1-b1) * clipped g */
            const double e = fabs((double)a1.m[i] - want) / (fabs(want) + 1e-12);
            if (e > worst) worst = e;
            if (e > 1e-5) bad++;
        }
        CHECK(bad == 0, "clipping not applied to the Adam moment (%d entries, worst rel %.3e)",
              bad, worst);

        /* a 1000x larger gradient must give an identical clipped step */
        for (int i = 0; i < M; i++) hh[i] = gsave[i] * 1000.0f;
        adam_step(&a2, p2, hh, 0.01f, 0.0f, clip);
        int diff = 0;
        for (int i = 0; i < M; i++)
            if (fabs((double)p1[i] - (double)p2[i]) / (fabs((double)p1[i]) + 1e-12) > 1e-4) diff++;
        CHECK(diff == 0, "clipped step depends on the pre-clip gradient scale (%d entries)", diff);

        int over = 0;
        for (int i = 0; i < M; i++) if (fabs((double)p1[i]) > 0.01 * 1.000001) over++;
        CHECK(over == 0, "%d updates exceeded the lr bound", over);

        /* with clipping disabled the moment is the raw gradient */
        Adam a4;
        float p4[M], g4[M];
        adam_init(&a4, M);
        for (int i = 0; i < M; i++) { p4[i] = 0.0f; g4[i] = gsave[i]; }
        adam_step(&a4, p4, g4, 0.01f, 0.0f, 0.0f);
        int badu = 0;
        for (int i = 0; i < M; i++) {
            const double want = 0.1 * (double)gsave[i];
            if (fabs((double)a4.m[i] - want) / (fabs(want) + 1e-12) > 1e-5) badu++;
        }
        CHECK(badu == 0, "clip <= 0 must disable clipping (%d entries scaled)", badu);
        adam_free(&a4);

        printf("  clipping: |g| = %.2f -> scale %.4f, moment matches (worst rel %.1e), "
               "1000x gradient gives an identical step\n", norm, scale, worst);
        adam_free(&a1);
        adam_free(&a2);
    }

    /* ---- decoupled weight decay ---- */
    {
        enum { M = 8 };
        Adam a3;
        float p[M], gz[M];
        adam_init(&a3, M);
        for (int i = 0; i < M; i++) { p[i] = 1.0f; gz[i] = 0.0f; }
        adam_step(&a3, p, gz, 0.1f, 0.5f, 0.0f);
        int bad = 0;
        for (int i = 0; i < M; i++) if (fabsf(p[i] - 0.95f) > 1e-6f) bad++;
        CHECK(bad == 0, "decoupled weight decay: zero grad, p should go 1 -> 0.95");
        adam_free(&a3);
    }

    /* ---- degenerate inputs must not crash ---- */
    {
        Adam a5;
        adam_init(&a5, 0);
        adam_step(&a5, NULL, NULL, 0.1f, 0.0f, 1.0f);
        adam_free(&a5);
        CHECK(1, "adam_step tolerates a zero-size / NULL optimiser");
    }
}

/* ==================================================== 6. SERIALISATION */

static void tmp_path(char *buf, size_t n, const char *name)
{
    const char *d = getenv("TMPDIR");
    if (!d || !*d) d = "/tmp";
    snprintf(buf, n, "%schessrl_%s_%d.crl", d, name, (int)(rnd_u64() & 0xFFFFFF));
    /* TMPDIR may or may not end in '/' -- normalise */
    const size_t l = strlen(d);
    if (l && d[l - 1] != '/') snprintf(buf, n, "%s/chessrl_%s_%d.crl", d, name,
                                       (int)(rnd_u64() & 0xFFFFFF));
}

static void test_serialisation(void)
{
    section("6. serialisation");

    enum { NA = 3 };
    Trunk *t  = (Trunk *)malloc(sizeof(Trunk));
    Trunk *t2 = (Trunk *)malloc(sizeof(Trunk));
    Head  *heads  = (Head *)malloc(sizeof(Head) * NA);
    Head  *heads2 = (Head *)malloc(sizeof(Head) * NA);
    if (!t || !t2 || !heads || !heads2) { printf("  FAIL: OOM\n"); g_fail++; return; }

    float *tf = (float *)t;
    for (size_t i = 0; i < TRUNK_NPARAM; i++) tf[i] = rnd_sym();
    float *hf = (float *)heads;
    for (size_t i = 0; i < HEAD_NPARAM * NA; i++) hf[i] = rnd_sym();

    Hyper hy[NA], hy2[NA];
    float elo[NA], elo2[NA];
    for (int i = 0; i < NA; i++) {
        hyper_default(&hy[i]);
        hy[i].temperature  = 0.5f + (float)rnd_uni();
        hy[i].entropy_coef = (float)rnd_uni() * 0.1f;
        hy[i].lr_scale     = 0.3f + (float)rnd_uni();
        hy[i].shaping      = (float)rnd_uni();
        hy[i].value_coef   = (float)rnd_uni();
        hy[i].gamma        = 0.9f + 0.09f * (float)rnd_uni();
        hy[i].lambda       = 0.8f + 0.19f * (float)rnd_uni();
        hy[i].mutate_sigma = 0.01f + 0.05f * (float)rnd_uni();
        elo[i] = 1200.0f + 700.0f * (float)rnd_uni();
    }
    const int GEN = 41;

    char path[512];
    tmp_path(path, sizeof path, "model");
    CHECK(model_save(path, t, heads, hy, elo, NA, GEN) == 1, "model_save succeeds (%s)", path);

    memset(t2, 0xAB, sizeof(Trunk));
    memset(heads2, 0xAB, sizeof(Head) * NA);
    memset(hy2, 0xAB, sizeof hy2);
    memset(elo2, 0xAB, sizeof elo2);

    int na = NA, gen = -1;
    CHECK(model_load(path, t2, heads2, hy2, elo2, &na, &gen) == 1, "model_load succeeds");
    CHECK(na == NA, "n_agents round-trips: got %d expected %d", na, NA);
    CHECK(gen == GEN, "generation round-trips: got %d expected %d", gen, GEN);
    CHECK(memcmp(t, t2, sizeof(Trunk)) == 0, "trunk is bit-exact after round-trip");
    CHECK(memcmp(heads, heads2, sizeof(Head) * NA) == 0, "heads are bit-exact after round-trip");
    CHECK(memcmp(hy, hy2, sizeof hy) == 0, "hypers are bit-exact after round-trip");
    CHECK(memcmp(elo, elo2, sizeof elo) == 0, "elo is bit-exact after round-trip");

    /* header-only probe: every array pointer NULL */
    {
        int pna = -1, pgen = -1;
        CHECK(model_load(path, NULL, NULL, NULL, NULL, &pna, &pgen) == 1, "header probe succeeds");
        CHECK(pna == NA, "header probe n_agents: got %d expected %d", pna, NA);
        CHECK(pgen == GEN, "header probe generation: got %d expected %d", pgen, GEN);

        /* a probe reports the file's agent count even if *n_agents says less */
        pna = 1; pgen = -1;
        CHECK(model_load(path, NULL, NULL, NULL, NULL, &pna, &pgen) == 1, "header probe (cap 1)");
        CHECK(pna == NA, "header probe ignores capacity: got %d expected %d", pna, NA);

        /* trunk-only load is still a header probe as far as n_agents goes */
        memset(t2, 0xCD, sizeof(Trunk));
        pna = 1; pgen = -1;
        CHECK(model_load(path, t2, NULL, NULL, NULL, &pna, &pgen) == 1, "trunk-only load succeeds");
        CHECK(pna == NA, "trunk-only load n_agents: got %d expected %d", pna, NA);
        CHECK(memcmp(t, t2, sizeof(Trunk)) == 0, "trunk-only load is bit-exact");
    }

    /* smaller capacity must truncate, not overflow */
    {
        enum { CAP = 2 };
        const size_t guard = 64;
        unsigned char *hbuf = (unsigned char *)malloc(sizeof(Head) * CAP + guard);
        unsigned char *ybuf = (unsigned char *)malloc(sizeof(Hyper) * CAP + guard);
        unsigned char *ebuf = (unsigned char *)malloc(sizeof(float) * CAP + guard);
        if (!hbuf || !ybuf || !ebuf) { printf("  FAIL: OOM\n"); g_fail++; return; }
        memset(hbuf, 0x5A, sizeof(Head) * CAP + guard);
        memset(ybuf, 0x5A, sizeof(Hyper) * CAP + guard);
        memset(ebuf, 0x5A, sizeof(float) * CAP + guard);

        int cna = CAP, cgen = -1;
        CHECK(model_load(path, NULL, (Head *)hbuf, (Hyper *)ybuf, (float *)ebuf, &cna, &cgen) == 1,
              "truncating load succeeds");
        CHECK(cna == CAP, "truncating load reports %d agents, expected %d", cna, CAP);
        CHECK(cgen == GEN, "truncating load generation: got %d expected %d", cgen, GEN);
        CHECK(memcmp(heads, hbuf, sizeof(Head) * CAP) == 0, "first %d heads load correctly", CAP);
        CHECK(memcmp(hy, ybuf, sizeof(Hyper) * CAP) == 0, "first %d hypers load correctly", CAP);
        CHECK(memcmp(elo, ebuf, sizeof(float) * CAP) == 0, "first %d elos load correctly", CAP);

        int over = 0;
        for (size_t i = 0; i < guard; i++) {
            if (hbuf[sizeof(Head)  * CAP + i] != 0x5A) over++;
            if (ybuf[sizeof(Hyper) * CAP + i] != 0x5A) over++;
            if (ebuf[sizeof(float) * CAP + i] != 0x5A) over++;
        }
        CHECK(over == 0, "truncating load overflowed the caller's buffers (%d guard bytes)", over);
        free(hbuf); free(ybuf); free(ebuf);
    }

    /* a truncated file must be rejected, not read past EOF */
    {
        char bad[512];
        tmp_path(bad, sizeof bad, "trunc");
        FILE *src = fopen(path, "rb");
        FILE *dst = fopen(bad, "wb");
        CHECK(src != NULL && dst != NULL, "temp files open");
        if (src && dst) {
            unsigned char buf[4096];
            size_t left = 40000, r;
            while (left && (r = fread(buf, 1, left < sizeof buf ? left : sizeof buf, src)) > 0) {
                fwrite(buf, 1, r, dst);
                left -= r;
            }
        }
        if (src) fclose(src);
        if (dst) fclose(dst);
        int bna = NA, bgen = -1;
        CHECK(model_load(bad, t2, heads2, hy2, elo2, &bna, &bgen) == 0,
              "truncated file must be rejected");
        remove(bad);
    }

    /* A file that is LONGER than the layout demands must be rejected too: that
     * is what a checkpoint from a build with a different NF_VACC / NF_VHID looks
     * like, and neither width appears in the header. */
    {
        char big[512];
        tmp_path(big, sizeof big, "long");
        FILE *src = fopen(path, "rb");
        FILE *dst = fopen(big, "wb");
        CHECK(src != NULL && dst != NULL, "temp files open");
        if (src && dst) {
            unsigned char buf[8192];
            size_t r;
            while ((r = fread(buf, 1, sizeof buf, src)) > 0) fwrite(buf, 1, r, dst);
            memset(buf, 0, sizeof buf);
            fwrite(buf, 1, 4096, dst);          /* one extra tensor's worth */
        }
        if (src) fclose(src);
        if (dst) fclose(dst);
        int lna = NA, lgen = -1;
        CHECK(model_load(big, t2, heads2, hy2, elo2, &lna, &lgen) == 0,
              "an over-long file must be rejected (a different NF_VACC would look "
              "exactly like this)");
        remove(big);
    }

    /* a bad magic must be rejected */
    {
        char bad[512];
        tmp_path(bad, sizeof bad, "magic");
        FILE *src = fopen(path, "rb");
        FILE *dst = fopen(bad, "wb");
        if (src && dst) {
            unsigned char buf[8192];
            size_t r;
            int first = 1;
            while ((r = fread(buf, 1, sizeof buf, src)) > 0) {
                if (first && r >= 4) { buf[0] ^= 0xFF; first = 0; }
                fwrite(buf, 1, r, dst);
            }
        }
        if (src) fclose(src);
        if (dst) fclose(dst);
        int bna = NA, bgen = -1;
        CHECK(model_load(bad, t2, heads2, hy2, elo2, &bna, &bgen) == 0,
              "bad magic must be rejected");
        remove(bad);
    }

    /* a missing file must fail cleanly */
    {
        int mna = NA, mgen = -1;
        CHECK(model_load("/nonexistent/dir/nope.crl", t2, heads2, hy2, elo2, &mna, &mgen) == 0,
              "missing file must be rejected");
    }

    printf("  round-trip bit-exact: trunk %zu floats, %d heads x %zu floats, hypers, elo\n",
           TRUNK_NPARAM, NA, HEAD_NPARAM);

    remove(path);
    free(t); free(t2); free(heads); free(heads2);
}

/* ======================================================== 7. BENCHMARK */

static void test_bench(void)
{
    section("7. micro-benchmark (single core)");

    Trunk *t = (Trunk *)malloc(sizeof(Trunk));
    Head  *h = (Head  *)malloc(sizeof(Head));
    if (!t || !h) { printf("  FAIL: OOM\n"); g_fail++; return; }
    nn_init(t, h, 7ULL);

    Position p;
    CHECK(pos_from_fen(&p, "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4") == 1,
          "bench FEN parses");
    uint16_t fidx[NF_MAXACTIVE];
    const int nf = nn_features(&p, fidx);
    Move mv[MAX_MOVES];
    const int nm = gen_legal(&p, mv);
    MoveKey keys[MAX_MOVES];
    for (int i = 0; i < nm; i++) nn_move_key(&p, mv[i], &keys[i]);

    Fwd fw;
    float logits[MAX_MOVES];
    double sink = 0.0;

    for (int i = 0; i < 5000; i++) {                       /* warm-up */
        nn_eval(t, h, fidx, nf, &fw);
        nn_logits(h, &fw, keys, nm, logits);
        sink += fw.v;
    }

    const int ITER = 300000;
    const double t0 = now_sec();
    for (int i = 0; i < ITER; i++) {
        nn_eval(t, h, fidx, nf, &fw);
        nn_logits(h, &fw, keys, nm, logits);
        sink += (double)fw.v + (double)logits[i % nm];
    }
    const double dt = now_sec() - t0;
    CHECK(dt > 0.0, "benchmark timer advanced");
    printf("  %d features, %d moves: %.0f (nn_eval + nn_logits)/sec  (%.2f us each)\n",
           nf, nm, (double)ITER / dt, 1e6 * dt / (double)ITER);
    printf("  [checksum %.6f]\n", sink * 1e-9);

    /* nn_logits with a NULL key array must produce zeros, not garbage */
    for (int i = 0; i < nm; i++) logits[i] = 1234.0f;
    nn_logits(h, &fw, NULL, nm, logits);
    int nzl = 0;
    for (int i = 0; i < nm; i++) if (logits[i] != 0.0f) nzl++;
    CHECK(nzl == 0, "nn_logits(keys=NULL) must zero the logits");

    /* softmax_t sanity */
    {
        float lg[4] = {1.0f, 2.0f, 3.0f, 4.0f}, out[4];
        softmax_t(lg, 4, 1.0f, out);
        double s = 0.0;
        for (int i = 0; i < 4; i++) s += out[i];
        CHECK(fabs(s - 1.0) < 1e-5, "softmax sums to 1 (got %.6f)", s);
        CHECK(out[3] > out[2] && out[2] > out[1] && out[1] > out[0], "softmax is monotone");
        softmax_t(lg, 4, 1e-6f, out);          /* temperature clamped, must not NaN */
        s = 0.0;
        for (int i = 0; i < 4; i++) s += out[i];
        CHECK(fabs(s - 1.0) < 1e-5, "softmax with tiny temperature still normalised (%.6f)", s);
        CHECK(out[3] > 0.99f, "near-zero temperature is nearly argmax (%.4f)", out[3]);
    }

    /* material_balance is from the side to move's view */
    {
        Position q;
        CHECK(pos_from_fen(&q, "4k3/8/8/8/8/8/8/3QK3 w - - 0 1") == 1, "material FEN parses");
        CHECK(fabsf(material_balance(&q) - 9.0f) < 1e-5f, "white to move, +Q -> +9");
        CHECK(pos_from_fen(&q, "4k3/8/8/8/8/8/8/3QK3 b - - 0 1") == 1, "material FEN parses");
        CHECK(fabsf(material_balance(&q) + 9.0f) < 1e-5f, "black to move, +Q for white -> -9");
    }

    free(t); free(h);
}

/* =============================================================== main */

int main(void)
{
    chess_init();

    printf("test_net: NF_INPUT=%d NF_ACC=%d NF_HID=%d NF_PDIM=%d NF_VACC=%d NF_VHID=%d\n",
           NF_INPUT, NF_ACC, NF_HID, NF_PDIM, NF_VACC, NF_VHID);
    printf("          TRUNK_NPARAM=%zu HEAD_NPARAM=%zu\n", TRUNK_NPARAM, HEAD_NPARAM);
    seed_from_env();

    test_features();
    test_move_keys();
    test_gradients();
    test_accumulation();
    test_adam();
    test_serialisation();
    test_bench();

    printf("\n%s: %d checks, %d failures\n", g_fail ? "FAILED" : "PASSED", g_checks, g_fail);
    return g_fail ? 1 : 0;
}
