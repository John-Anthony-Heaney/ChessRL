/* test_960.c -- Chess960 verification for the chess core.
 *
 *   cc -O2 -std=c11 -D_DARWIN_C_SOURCE -Isrc src/chess.c tests/test_960.c -o t960 -lm
 *   ./t960           # the full gate (all 960 ids, differential to depth 4)
 *   ./t960 quick     # a subset, for edit-compile-run
 *   ./t960 full      # adds depth 5 differentials and deeper perft
 *
 * The centre of this file is a SECOND, deliberately naive move generator:
 * brute-force loops over all 64 squares, rays walked one square at a time, no
 * magic bitboards, no pin masks, no check-evasion masks.  Legality is decided
 * the slow obvious way -- play the move on a plain 64-square array and ask
 * whether the mover's king is attacked.  Castling is transcribed straight from
 * the rules.  Nothing in it shares code with src/chess.c's generator, so when
 * the two agree on every move of every node of hundreds of millions of
 * positions, that is real evidence rather than a tautology.
 */

#include "chess.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ------------------------------------------------------------- harness -- */

static int g_fail = 0;
static long g_checks = 0;

#define CHECK(cond, ...)                                                       \
    do {                                                                       \
        g_checks++;                                                            \
        if (!(cond)) {                                                         \
            printf("FAIL  %s:%d  ", __FILE__, __LINE__);                       \
            printf(__VA_ARGS__);                                               \
            printf("\n");                                                      \
            g_fail++;                                                          \
            if (g_fail > 30) { printf("too many failures, stopping\n"); exit(1); } \
        }                                                                      \
    } while (0)

static void section(const char *name) { printf("-- %s\n", name); }

static void done(const char *name, int before)
{
    if (g_fail == before) printf("PASS  %s (%ld checks)\n", name, g_checks);
    else                  printf("FAIL  %s (%d new failures)\n", name, g_fail - before);
}

static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

/* xorshift128+, local to the test so the engine's RNG cannot influence it */
static uint64_t rs[2] = { 0x9E3779B97F4A7C15ULL, 0xD1B54A32D192ED03ULL };
static uint64_t rnd(void)
{
    uint64_t x = rs[0], y = rs[1];
    rs[0] = y;
    x ^= x << 23;
    rs[1] = x ^ y ^ (x >> 17) ^ (y >> 26);
    return rs[1] + y;
}
static int rnd_below(int n) { return (int)(rnd() % (uint64_t)n); }

/* ================================================================= */
/* ==================  the naive reference engine  ================= */
/* ================================================================= */

typedef struct {
    int8_t pc[64];    /* piece type, NO_PIECE when empty */
    int8_t col[64];   /* colour, -1 when empty           */
} RefBoard;

static void ref_load(RefBoard *b, const Position *p)
{
    for (int s = 0; s < 64; s++) { b->pc[s] = p->board[s]; b->col[s] = p->color_at[s]; }
}

static int rf(int sq) { return sq % 8; }
static int rr(int sq) { return sq / 8; }
static int on_board(int f, int r) { return f >= 0 && f < 8 && r >= 0 && r < 8; }

static const int KNIGHT_D[8][2] = { {1,2},{2,1},{2,-1},{1,-2},{-1,-2},{-2,-1},{-2,1},{-1,2} };
static const int KING_D[8][2]   = { {1,0},{1,1},{0,1},{-1,1},{-1,0},{-1,-1},{0,-1},{1,-1} };
/* first four orthogonal, last four diagonal */
static const int RAY_D[8][2]    = { {1,0},{-1,0},{0,1},{0,-1},{1,1},{1,-1},{-1,1},{-1,-1} };

/* Is `sq` attacked by side `by`?  Walks outward from the square itself. */
static int ref_attacked(const RefBoard *b, int sq, int by)
{
    const int f = rf(sq), r = rr(sq);

    /* pawns: an enemy pawn stands one rank "behind" the square it attacks */
    const int pr = (by == WHITE) ? r - 1 : r + 1;
    for (int df = -1; df <= 1; df += 2) {
        const int nf = f + df;
        if (!on_board(nf, pr)) continue;
        const int s = pr * 8 + nf;
        if (b->pc[s] == PAWN && b->col[s] == by) return 1;
    }
    for (int i = 0; i < 8; i++) {
        const int nf = f + KNIGHT_D[i][0], nr = r + KNIGHT_D[i][1];
        if (!on_board(nf, nr)) continue;
        const int s = nr * 8 + nf;
        if (b->pc[s] == KNIGHT && b->col[s] == by) return 1;
    }
    for (int i = 0; i < 8; i++) {
        const int nf = f + KING_D[i][0], nr = r + KING_D[i][1];
        if (!on_board(nf, nr)) continue;
        const int s = nr * 8 + nf;
        if (b->pc[s] == KING && b->col[s] == by) return 1;
    }
    for (int d = 0; d < 8; d++) {
        int nf = f + RAY_D[d][0], nr = r + RAY_D[d][1];
        while (on_board(nf, nr)) {
            const int s = nr * 8 + nf;
            if (b->pc[s] != NO_PIECE) {
                if (b->col[s] == by) {
                    const int t = b->pc[s];
                    if (t == QUEEN || (d < 4 ? t == ROOK : t == BISHOP)) return 1;
                }
                break;
            }
            nf += RAY_D[d][0];
            nr += RAY_D[d][1];
        }
    }
    return 0;
}

static int ref_king_sq(const RefBoard *b, int c)
{
    for (int s = 0; s < 64; s++) if (b->pc[s] == KING && b->col[s] == c) return s;
    return -1;
}

/* The four squares a castling move touches, in the engine's encoding: `to` is
 * the rook's origin in Chess960 and the king's destination in classical. */
static void ref_castle_squares(const Position *p, int fl, int us, int to,
                               int *rfrom, int *kto, int *rto)
{
    const int side = (fl == MF_KCASTLE) ? 0 : 1;
    const int base = (us == WHITE) ? 0 : 56;
    *rfrom = p->chess960 ? to : base + (side == 0 ? 7 : 0);
    *kto   = base + (side == 0 ? 6 : 2);
    *rto   = base + (side == 0 ? 5 : 3);
}

/* Plays `m` on a plain board.  Used both to decide legality and, elsewhere, to
 * check what the engine's make_move() actually produced. */
static void ref_apply(RefBoard *b, const Position *p, Move m)
{
    const int from = MV_FROM(m), to = MV_TO(m), fl = MV_FLAG(m);
    const int us = p->side;

    if (fl == MF_KCASTLE || fl == MF_QCASTLE) {
        int rfrom, kto, rto;
        ref_castle_squares(p, fl, us, to, &rfrom, &kto, &rto);
        b->pc[from]  = NO_PIECE; b->col[from]  = -1;
        b->pc[rfrom] = NO_PIECE; b->col[rfrom] = -1;
        b->pc[kto]   = KING;     b->col[kto]   = (int8_t)us;
        b->pc[rto]   = ROOK;     b->col[rto]   = (int8_t)us;
        return;
    }
    if (fl == MF_EP) {
        const int capsq = to - ((us == WHITE) ? 8 : -8);
        b->pc[capsq] = NO_PIECE;
        b->col[capsq] = -1;
    }
    const int moved = MV_IS_PROMO(m) ? MV_PROMO_PIECE(m) : b->pc[from];
    b->pc[to]   = (int8_t)moved;
    b->col[to]  = (int8_t)us;
    b->pc[from] = NO_PIECE;
    b->col[from] = -1;
}

static int ref_is_legal(const RefBoard *b0, const Position *p, Move m)
{
    RefBoard b = *b0;
    ref_apply(&b, p, m);
    const int ksq = ref_king_sq(&b, p->side);
    if (ksq < 0) return 1;                 /* kingless test positions */
    return !ref_attacked(&b, ksq, p->side ^ 1);
}

static int ref_add_promos(Move *out, int n, int from, int to, int cap)
{
    const int base = cap ? MF_PROMO_NC : MF_PROMO_N;
    for (int i = 0; i < 4; i++) out[n++] = MV_MAKE(from, to, base + i);
    return n;
}

/* Castling, transcribed from the rules:
 *   - the right is held and its rook is really on its origin square;
 *   - the king is not in check;
 *   - every square of the king's path and of the rook's path, destinations
 *     included, is empty apart from the castling king and the castling rook;
 *   - no square the king passes over, origin and destination included, is
 *     attacked (judged in the position before the move);
 *   - the position after the move does not leave the king in check.
 */
static int ref_gen_castling(const Position *p, const RefBoard *b, Move *out, int n)
{
    const int us = p->side, them = us ^ 1;
    const int base = (us == WHITE) ? 0 : 56;
    const int ksq = ref_king_sq(b, us);
    if (ksq < 0 || rr(ksq) != rr(base)) return n;
    if (ref_attacked(b, ksq, them)) return n;

    for (int side = 0; side < 2; side++) {
        const int bit = (us == WHITE) ? (side == 0 ? CR_WK : CR_WQ)
                                      : (side == 0 ? CR_BK : CR_BQ);
        if (!(p->castling & bit)) continue;

        const int rfrom = base + p->crook[us][side];
        if (b->pc[rfrom] != ROOK || b->col[rfrom] != us) continue;
        const int kto = base + (side == 0 ? 6 : 2);
        const int rto = base + (side == 0 ? 5 : 3);

        int ok = 1;
        /* emptiness over both closed intervals */
        int lo = ksq < kto ? ksq : kto, hi = ksq < kto ? kto : ksq;
        for (int s = lo; s <= hi && ok; s++)
            if (s != ksq && s != rfrom && b->pc[s] != NO_PIECE) ok = 0;
        lo = rfrom < rto ? rfrom : rto;
        hi = rfrom < rto ? rto : rfrom;
        for (int s = lo; s <= hi && ok; s++)
            if (s != ksq && s != rfrom && b->pc[s] != NO_PIECE) ok = 0;
        if (!ok) continue;

        /* the king's walk must be attack-free in the position as it stands */
        lo = ksq < kto ? ksq : kto;
        hi = ksq < kto ? kto : ksq;
        for (int s = lo; s <= hi && ok; s++)
            if (ref_attacked(b, s, them)) ok = 0;
        if (!ok) continue;

        /* and the move must not leave the king in check */
        {
            RefBoard after = *b;
            after.pc[ksq]   = NO_PIECE; after.col[ksq]   = -1;
            after.pc[rfrom] = NO_PIECE; after.col[rfrom] = -1;
            after.pc[kto]   = KING;     after.col[kto]   = (int8_t)us;
            after.pc[rto]   = ROOK;     after.col[rto]   = (int8_t)us;
            if (ref_attacked(&after, kto, them)) continue;
        }

        out[n++] = MV_MAKE(ksq, p->chess960 ? rfrom : kto,
                           side == 0 ? MF_KCASTLE : MF_QCASTLE);
    }
    return n;
}

/* Every legal move of `p`, generated the naive way. */
static int ref_gen(const Position *p, Move *out)
{
    RefBoard b;
    ref_load(&b, p);

    const int us = p->side, them = us ^ 1;
    Move pseudo[512];
    int np = 0;

    for (int from = 0; from < 64; from++) {
        if (b.col[from] != us) continue;
        const int f = rf(from), r = rr(from);

        switch (b.pc[from]) {
        case PAWN: {
            const int dr   = (us == WHITE) ? 1 : -1;
            const int home = (us == WHITE) ? 1 : 6;
            const int last = (us == WHITE) ? 7 : 0;
            if (on_board(f, r + dr)) {
                const int t = (r + dr) * 8 + f;
                if (b.pc[t] == NO_PIECE) {
                    if (r + dr == last) np = ref_add_promos(pseudo, np, from, t, 0);
                    else                pseudo[np++] = MV_MAKE(from, t, MF_QUIET);
                    if (r == home) {
                        const int t2 = (r + 2 * dr) * 8 + f;
                        if (b.pc[t2] == NO_PIECE) pseudo[np++] = MV_MAKE(from, t2, MF_DOUBLE);
                    }
                }
            }
            for (int df = -1; df <= 1; df += 2) {
                if (!on_board(f + df, r + dr)) continue;
                const int t = (r + dr) * 8 + f + df;
                if (b.pc[t] != NO_PIECE && b.col[t] == them) {
                    if (r + dr == last) np = ref_add_promos(pseudo, np, from, t, 1);
                    else                pseudo[np++] = MV_MAKE(from, t, MF_CAPTURE);
                } else if (b.pc[t] == NO_PIECE && p->ep >= 0 && t == p->ep) {
                    pseudo[np++] = MV_MAKE(from, t, MF_EP);
                }
            }
            break;
        }
        case KNIGHT:
        case KING: {
            const int (*d)[2] = (b.pc[from] == KNIGHT) ? KNIGHT_D : KING_D;
            for (int i = 0; i < 8; i++) {
                const int nf = f + d[i][0], nr = r + d[i][1];
                if (!on_board(nf, nr)) continue;
                const int t = nr * 8 + nf;
                if (b.col[t] == us) continue;
                pseudo[np++] = MV_MAKE(from, t, b.pc[t] == NO_PIECE ? MF_QUIET : MF_CAPTURE);
            }
            break;
        }
        case BISHOP:
        case ROOK:
        case QUEEN: {
            const int d0 = (b.pc[from] == BISHOP) ? 4 : 0;
            const int d1 = (b.pc[from] == ROOK)   ? 4 : 8;
            for (int d = d0; d < d1; d++) {
                int nf = f + RAY_D[d][0], nr = r + RAY_D[d][1];
                while (on_board(nf, nr)) {
                    const int t = nr * 8 + nf;
                    if (b.col[t] == us) break;
                    pseudo[np++] = MV_MAKE(from, t, b.pc[t] == NO_PIECE ? MF_QUIET : MF_CAPTURE);
                    if (b.pc[t] != NO_PIECE) break;
                    nf += RAY_D[d][0];
                    nr += RAY_D[d][1];
                }
            }
            break;
        }
        default: break;
        }
    }

    int n = 0;
    for (int i = 0; i < np; i++)
        if (ref_is_legal(&b, p, pseudo[i])) out[n++] = pseudo[i];
    return ref_gen_castling(p, &b, out, n);
}

/* ================================================================= */
/* ========================  comparison  =========================== */
/* ================================================================= */

/* Castling in the king-destination spelling, so that a Chess960-flagged
 * position and its classical twin can be compared move for move. */
static Move canon_move(const Position *p, Move m)
{
    const int fl = MV_FLAG(m);
    if (fl != MF_KCASTLE && fl != MF_QCASTLE) return m;
    const int base = (p->side == WHITE) ? 0 : 56;
    return MV_MAKE(MV_FROM(m), base + (fl == MF_KCASTLE ? 6 : 2), fl);
}

static void sort_moves(Move *a, int n)
{
    for (int i = 1; i < n; i++) {
        const Move v = a[i];
        int j = i - 1;
        while (j >= 0 && a[j] > v) { a[j + 1] = a[j]; j--; }
        a[j + 1] = v;
    }
}

static void dump_mismatch(const Position *p, const Move *a, int na, const Move *b, int nb)
{
    char fen[128];
    pos_to_fen(p, fen, sizeof(fen));
    printf("      position: %s\n", fen);
    printf("      engine (%d):", na);
    for (int i = 0; i < na; i++) {
        char u[8];
        move_to_uci_pos(p, a[i], u);
        printf(" %s/%d", u, MV_FLAG(a[i]));
    }
    printf("\n      naive  (%d):", nb);
    for (int i = 0; i < nb; i++) {
        char u[8];
        move_to_uci_pos(p, b[i], u);
        printf(" %s/%d", u, MV_FLAG(b[i]));
    }
    printf("\n");
}

/* Compares the two generators' move SETS (not just their sizes). */
static int same_moves(const Position *p, int verbose)
{
    Move a[MAX_MOVES], b[MAX_MOVES];
    int na = gen_legal(p, a);
    int nb = ref_gen(p, b);
    sort_moves(a, na);
    sort_moves(b, nb);
    if (na != nb || memcmp(a, b, (size_t)na * sizeof(Move)) != 0) {
        if (verbose) dump_mismatch(p, a, na, b, nb);
        return 0;
    }
    /* duplicates would make two wrong sets compare equal */
    for (int i = 1; i < na; i++)
        if (a[i] == a[i - 1]) { if (verbose) printf("      duplicate move generated\n"); return 0; }
    return 1;
}

/* perft that compares the full move set at every interior node. */
static uint64_t diff_perft(Position *p, int depth)
{
    Move list[MAX_MOVES];
    const int n = gen_legal(p, list);

    if (!same_moves(p, 1)) { g_fail++; return 0; }
    if (depth <= 1) return (uint64_t)n;

    uint64_t nodes = 0;
    Undo u;
    for (int i = 0; i < n; i++) {
        make_move(p, list[i], &u);
        nodes += diff_perft(p, depth - 1);
        unmake_move(p, list[i], &u);
        if (g_fail > 30) break;
    }
    return nodes;
}

/* perft driven entirely by the naive generator (engine make/unmake only). */
static uint64_t ref_perft(Position *p, int depth)
{
    Move list[MAX_MOVES];
    const int n = ref_gen(p, list);
    if (depth <= 1) return (uint64_t)n;
    uint64_t nodes = 0;
    Undo u;
    for (int i = 0; i < n; i++) {
        make_move(p, list[i], &u);
        nodes += ref_perft(p, depth - 1);
        unmake_move(p, list[i], &u);
    }
    return nodes;
}

/* ================================================================= */
/* =====================  invariant checks  ======================== */
/* ================================================================= */

/* Bitboards, mailbox, occupancies and the incremental key must all agree. */
static int pos_consistent(const Position *p, const char *what)
{
    int ok = 1;
    uint64_t occ[NCOLORS] = { 0, 0 };
    for (int c = 0; c < NCOLORS; c++)
        for (int pt = 0; pt < NPIECES; pt++) occ[c] |= p->piece[c][pt];

    for (int s = 0; s < 64; s++) {
        const int pt = p->board[s], c = p->color_at[s];
        if (pt == NO_PIECE) {
            if (c != -1 || ((occ[WHITE] | occ[BLACK]) >> s) & 1) { ok = 0; break; }
            continue;
        }
        if (c != WHITE && c != BLACK) { ok = 0; break; }
        if (!((p->piece[c][pt] >> s) & 1)) { ok = 0; break; }
        if (((p->piece[c ^ 1][pt]) >> s) & 1) { ok = 0; break; }
    }
    if (ok && (occ[WHITE] != p->occ[WHITE] || occ[BLACK] != p->occ[BLACK])) ok = 0;
    if (ok && (occ[WHITE] & occ[BLACK]) != 0) ok = 0;
    if (ok && p->all != (p->occ[WHITE] | p->occ[BLACK])) ok = 0;
    if (ok && p->key != pos_compute_key(p)) ok = 0;
    if (!ok) {
        char fen[128];
        pos_to_fen(p, fen, sizeof(fen));
        CHECK(0, "position invariants broken (%s): %s", what, fen);
    }
    return ok;
}

/* Two positions are the same modulo the fields a FEN cannot carry: a rook file
 * for a right that no longer exists, and the 960 flag of a position that has
 * no castling rights left at all (with no rights the two rule sets coincide). */
static int pos_same_modulo_fen(const Position *a, const Position *b)
{
    Position x = *a, y = *b;
    for (int c = 0; c < NCOLORS; c++) {
        for (int side = 0; side < 2; side++) {
            const int bit = (c == WHITE) ? (side == 0 ? CR_WK : CR_WQ)
                                         : (side == 0 ? CR_BK : CR_BQ);
            if (!(x.castling & bit)) x.crook[c][side] = y.crook[c][side] = (side == 0) ? 7 : 0;
        }
    }
    if (!x.castling) x.chess960 = y.chess960 = 0;
    return memcmp(&x, &y, sizeof(Position)) == 0;
}

/* The X-FEN / KQkq spelling of a position's castling field. */
static void xfen_castling(const Position *p, char *out)
{
    int i = 0;
    if (!p->castling) { out[i++] = '-'; out[i] = '\0'; return; }
    if (p->castling & CR_WK) out[i++] = 'K';
    if (p->castling & CR_WQ) out[i++] = 'Q';
    if (p->castling & CR_BK) out[i++] = 'k';
    if (p->castling & CR_BQ) out[i++] = 'q';
    out[i] = '\0';
}

/* Is KQkq an unambiguous spelling here?  Only when every claimed rook really
 * is the outermost one on its side of its king. */
static int xfen_is_exact(const Position *p)
{
    for (int c = 0; c < NCOLORS; c++) {
        const int base = (c == WHITE) ? 0 : 56;
        if (!p->piece[c][KING]) return 0;
        const int kf = sq_file(bb_lsb(p->piece[c][KING]));
        if (sq_rank(bb_lsb(p->piece[c][KING])) != sq_rank(base)) return 0;
        for (int side = 0; side < 2; side++) {
            const int bit = (c == WHITE) ? (side == 0 ? CR_WK : CR_WQ)
                                         : (side == 0 ? CR_BK : CR_BQ);
            if (!(p->castling & bit)) continue;
            int outer = -1;
            if (side == 0) {
                for (int f = 7; f > kf; f--)
                    if (p->board[base + f] == ROOK && p->color_at[base + f] == c) { outer = f; break; }
            } else {
                for (int f = 0; f < kf; f++)
                    if (p->board[base + f] == ROOK && p->color_at[base + f] == c) { outer = f; break; }
            }
            if (outer != p->crook[c][side]) return 0;
        }
    }
    return 1;
}

/* Rebuilds a FEN with a replacement castling field. */
static void fen_with_castling(const Position *p, const char *castling, char *out, size_t n)
{
    char fen[128];
    pos_to_fen(p, fen, sizeof(fen));
    char board[80], side[4], ep[8];
    unsigned hm, fm;
    if (sscanf(fen, "%79s %3s %*s %7s %u %u", board, side, ep, &hm, &fm) != 5) {
        snprintf(out, n, "%s", fen);
        return;
    }
    snprintf(out, n, "%s %s %s %s %u %u", board, side, castling, ep, hm, fm);
}

/* ================================================================= */
/* ==========================  the tests  ========================== */
/* ================================================================= */

/* ---- 1. Scharnagl numbering ------------------------------------- */

static void test_numbering(void)
{
    const int before = g_fail;
    section("Scharnagl numbering (all 960 ids)");

    /* The self-check that matters: 518 is classical chess. */
    {
        Position p, c;
        pos_startpos960(&p, CHESS960_CLASSICAL_ID);
        pos_startpos(&c);
        for (int s = 0; s < 64; s++) {
            CHECK(p.board[s] == c.board[s] && p.color_at[s] == c.color_at[s],
                  "id 518 differs from the classical array at %s", SQ_NAMES[s]);
            if (g_fail != before) break;
        }
        CHECK(p.side == WHITE && p.castling == CR_ALL && p.ep == -1,
              "id 518 header wrong");
        CHECK(p.crook[WHITE][0] == 7 && p.crook[WHITE][1] == 0
              && p.crook[BLACK][0] == 7 && p.crook[BLACK][1] == 0,
              "id 518 rook files are not h/a");
        CHECK(p.chess960 == 1, "pos_startpos960 must mark the position as 960");
        CHECK(pos_960_id(&c) == CHESS960_CLASSICAL_ID,
              "pos_960_id(classical startpos) = %d, want 518", pos_960_id(&c));

        char fen[128];
        pos_to_fen(&p, fen, sizeof(fen));
        CHECK(strcmp(fen, "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w HAha - 0 1") == 0,
              "id 518 FEN: %s", fen);
    }

    /* Every id: legal array, mirrored, and an exact inverse. */
    int seen[960];
    memset(seen, 0, sizeof(seen));
    char arrays[960][9];
    for (int id = 0; id < 960; id++) {
        Position p;
        pos_startpos960(&p, id);

        int nb = 0, nr = 0, nn = 0, nq = 0, nk = 0;
        int bishops[2] = { -1, -1 }, rooks[2] = { -1, -1 }, king = -1;
        for (int f = 0; f < 8; f++) {
            CHECK(p.color_at[f] == WHITE && p.color_at[f + 56] == BLACK,
                  "id %d: back rank colours", id);
            CHECK(p.board[f] == p.board[f + 56], "id %d: black does not mirror white", id);
            CHECK(p.board[f + 8] == PAWN && p.color_at[f + 8] == WHITE, "id %d: white pawns", id);
            CHECK(p.board[f + 48] == PAWN && p.color_at[f + 48] == BLACK, "id %d: black pawns", id);
            for (int r = 2; r <= 5; r++)
                CHECK(p.board[r * 8 + f] == NO_PIECE, "id %d: middle board not empty", id);
            switch (p.board[f]) {
                case BISHOP: if (nb < 2) bishops[nb] = f; nb++; break;
                case ROOK:   if (nr < 2) rooks[nr] = f;   nr++; break;
                case KNIGHT: nn++; break;
                case QUEEN:  nq++; break;
                case KING:   king = f; nk++; break;
                default:     CHECK(0, "id %d: stray piece on the back rank", id); break;
            }
            arrays[id][f] = "PNBRQK"[p.board[f]];
        }
        arrays[id][8] = '\0';
        CHECK(nb == 2 && nr == 2 && nn == 2 && nq == 1 && nk == 1,
              "id %d: piece counts %d/%d/%d/%d/%d", id, nb, nr, nn, nq, nk);
        CHECK(((bishops[0] ^ bishops[1]) & 1) == 1,
              "id %d: bishops on the same colour (%d,%d)", id, bishops[0], bishops[1]);
        CHECK(rooks[0] < king && king < rooks[1],
              "id %d: king not strictly between the rooks (%d,%d,%d)",
              id, rooks[0], king, rooks[1]);
        CHECK(p.crook[WHITE][0] == rooks[1] && p.crook[WHITE][1] == rooks[0],
              "id %d: crook white = {%u,%u}, want {%d,%d}",
              id, p.crook[WHITE][0], p.crook[WHITE][1], rooks[1], rooks[0]);
        CHECK(p.crook[BLACK][0] == rooks[1] && p.crook[BLACK][1] == rooks[0],
              "id %d: crook black wrong", id);
        CHECK(p.castling == CR_ALL, "id %d: castling rights %u", id, p.castling);
        CHECK(p.key == pos_compute_key(&p), "id %d: key mismatch", id);
        CHECK(pos_consistent(&p, "startpos960"), "id %d: inconsistent", id);

        const int back = pos_960_id(&p);
        CHECK(back == id, "pos_960_id(pos_startpos960(%d)) = %d", id, back);
        seen[id] = 1;
        if (g_fail != before) break;
    }

    if (g_fail == before) {
        int distinct = 1;
        for (int i = 0; i < 960 && distinct; i++)
            for (int j = i + 1; j < 960; j++)
                if (!strcmp(arrays[i], arrays[j])) {
                    CHECK(0, "ids %d and %d produce the same array %s", i, j, arrays[i]);
                    distinct = 0;
                    break;
                }
        int all = 1;
        for (int i = 0; i < 960; i++) if (!seen[i]) all = 0;
        CHECK(all, "not every id was generated");
    }

    /* ids are taken modulo 960 */
    {
        Position a, b;
        pos_startpos960(&a, 518);
        pos_startpos960(&b, 518 + 960 * 3);
        CHECK(memcmp(&a, &b, sizeof(Position)) == 0, "id wrap-around");
        pos_startpos960(&b, 518 - 960);
        CHECK(memcmp(&a, &b, sizeof(Position)) == 0, "negative id wrap-around");
    }

    /* pos_960_id rejects what is not a legal array */
    {
        Position p;
        CHECK(pos_from_fen(&p, "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"), "fen");
        CHECK(pos_960_id(&p) == 518, "classical array must still be 518");
        CHECK(pos_from_fen(&p, "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBRN w - - 0 1"), "fen");
        CHECK(pos_960_id(&p) == -1, "king not between the rooks must be rejected");
        CHECK(pos_from_fen(&p, "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBBKQNR w - - 0 1"), "fen");
        CHECK(pos_960_id(&p) == -1, "bishops on the same colour must be rejected");
        CHECK(pos_from_fen(&p, "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/1NBQKBNR w - - 0 1"), "fen");
        CHECK(pos_960_id(&p) == -1, "hole in the back rank must be rejected");
        CHECK(pos_from_fen(&p, "nrbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1"), "fen");
        CHECK(pos_960_id(&p) == -1, "unmirrored back ranks must be rejected");
        CHECK(pos_from_fen(&p, "8/8/8/4k3/8/8/8/4K3 w - - 0 1"), "fen");
        CHECK(pos_960_id(&p) == -1, "an endgame is not a 960 array");
    }

    /* pos_960_random: in range, and it covers the whole space */
    {
        uint64_t st[4] = { 1, 2, 3, 4 };
        int hist[960];
        memset(hist, 0, sizeof(hist));
        const int draws = 200000;
        for (int i = 0; i < draws; i++) {
            const int id = pos_960_random(st);
            if (id < 0 || id >= 960) { CHECK(0, "pos_960_random out of range: %d", id); break; }
            hist[id]++;
        }
        int missing = 0, lo = draws, hi = 0;
        for (int i = 0; i < 960; i++) {
            if (!hist[i]) missing++;
            if (hist[i] < lo) lo = hist[i];
            if (hist[i] > hi) hi = hist[i];
        }
        CHECK(missing == 0, "pos_960_random never produced %d of the ids", missing);
        /* 200k draws over 960 buckets: mean 208, sd ~14.4; +-6 sd is a very
         * loose but non-vacuous uniformity check. */
        CHECK(lo > 120 && hi < 300, "pos_960_random looks skewed (min %d, max %d)", lo, hi);

        uint64_t zero[4] = { 0, 0, 0, 0 };
        const int id = pos_960_random(zero);
        CHECK(id >= 0 && id < 960, "all-zero rng state must not break the sampler");
        CHECK(pos_960_random(NULL) == CHESS960_CLASSICAL_ID, "NULL rng");
    }

    done("numbering", before);
}

/* ---- 2. FEN -------------------------------------------------------- */

static void check_fen_roundtrip(const Position *p, const char *what)
{
    char fen[128], fen2[128];
    Position q;

    pos_to_fen(p, fen, sizeof(fen));
    if (!pos_from_fen(&q, fen)) { CHECK(0, "%s: pos_from_fen rejected \"%s\"", what, fen); return; }
    if (!pos_same_modulo_fen(p, &q)) { CHECK(0, "%s: FEN round-trip changed the position: %s", what, fen); return; }
    pos_to_fen(&q, fen2, sizeof(fen2));
    CHECK(strcmp(fen, fen2) == 0, "%s: FEN not stable: \"%s\" -> \"%s\"", what, fen, fen2);
    CHECK(q.key == pos_compute_key(&q), "%s: key after parse", what);

    /* the same position spelled KQkq (X-FEN) must parse to the same rules */
    if (xfen_is_exact(p)) {
        char cast[8], xf[128];
        Position x;
        xfen_castling(p, cast);
        fen_with_castling(p, cast, xf, sizeof(xf));
        if (!pos_from_fen(&x, xf)) { CHECK(0, "%s: X-FEN rejected \"%s\"", what, xf); return; }
        CHECK(x.castling == p->castling, "%s: X-FEN \"%s\" lost rights", what, xf);
        for (int c = 0; c < NCOLORS; c++)
            for (int side = 0; side < 2; side++) {
                const int bit = (c == WHITE) ? (side == 0 ? CR_WK : CR_WQ)
                                             : (side == 0 ? CR_BK : CR_BQ);
                if (p->castling & bit)
                    CHECK(x.crook[c][side] == p->crook[c][side],
                          "%s: X-FEN \"%s\" resolved %c%d to file %u, want %u",
                          what, xf, c == WHITE ? 'W' : 'B', side,
                          x.crook[c][side], p->crook[c][side]);
            }
        Move a[MAX_MOVES], b[MAX_MOVES];
        int na = gen_legal(p, a), nb = gen_legal(&x, b);
        /* the two spellings encode castling differently on purpose */
        for (int i = 0; i < na; i++) a[i] = canon_move(p, a[i]);
        for (int i = 0; i < nb; i++) b[i] = canon_move(&x, b[i]);
        sort_moves(a, na);
        sort_moves(b, nb);
        CHECK(na == nb && memcmp(a, b, (size_t)na * sizeof(Move)) == 0,
              "%s: X-FEN \"%s\" changed the legal moves (%d vs %d)", what, xf, na, nb);
    }
}

static void test_fen(void)
{
    const int before = g_fail;
    section("FEN: Shredder + X-FEN, all 960 starts");

    for (int id = 0; id < 960; id++) {
        Position p;
        pos_startpos960(&p, id);
        char what[32];
        snprintf(what, sizeof(what), "id %d", id);
        check_fen_roundtrip(&p, what);
        if (g_fail != before) break;
    }

    /* explicit notation acceptance */
    {
        Position p;
        CHECK(pos_from_fen(&p, "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w HAha - 0 1"),
              "Shredder-FEN rejected");
        CHECK(p.chess960 == 1 && p.castling == CR_ALL, "Shredder-FEN flags");
        CHECK(p.crook[WHITE][0] == 7 && p.crook[WHITE][1] == 0, "Shredder-FEN white files");
        CHECK(p.crook[BLACK][0] == 7 && p.crook[BLACK][1] == 0, "Shredder-FEN black files");

        /* the classical spelling of the same board is NOT flagged as 960 */
        CHECK(pos_from_fen(&p, "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"), "fen");
        CHECK(p.chess960 == 0, "classical KQkq must stay classical");

        /* X-FEN on a real 960 board resolves to the outermost rook */
        CHECK(pos_from_fen(&p, "rbbqknnr/pppppppp/8/8/8/8/PPPPPPPP/RBBQKNNR w KQkq - 0 1"),
              "X-FEN on a 960 board rejected");
        CHECK(p.chess960 == 0, "that board is a 960 array but its rooks are on a/h");
        CHECK(p.crook[WHITE][0] == 7 && p.crook[WHITE][1] == 0, "X-FEN a/h resolution");

        CHECK(pos_from_fen(&p, "1rqbkrbn/pppppppp/1n6/8/8/1N6/PPPPPPPP/1RQBKRBN w KQkq - 0 1"),
              "X-FEN with rooks on b/f rejected");
        CHECK(p.chess960 == 1, "rooks on b1/f1 must be recognised as 960");
        CHECK(p.crook[WHITE][0] == 5 && p.crook[WHITE][1] == 1,
              "X-FEN resolved to files %u/%u, want 5/1", p.crook[WHITE][0], p.crook[WHITE][1]);

        /* hybrid: file letters for one side, KQ for the other */
        CHECK(pos_from_fen(&p, "1rqbkrbn/pppppppp/1n6/8/8/1N6/PPPPPPPP/1RQBKRBN w FBkq - 0 1"),
              "hybrid X-FEN rejected");
        CHECK(p.castling == CR_ALL, "hybrid rights %u", p.castling);
        CHECK(p.crook[WHITE][0] == 5 && p.crook[WHITE][1] == 1, "hybrid white files");
        CHECK(p.crook[BLACK][0] == 5 && p.crook[BLACK][1] == 1, "hybrid black files");

        /* outermost, not innermost: three rooks on the back rank */
        CHECK(pos_from_fen(&p, "4k3/8/8/8/8/8/8/R2RK2R w KQ - 0 1"), "three-rook FEN");
        CHECK(p.crook[WHITE][1] == 0, "X-FEN Q must pick the a1 rook, got %u", p.crook[WHITE][1]);
        CHECK(pos_from_fen(&p, "4k3/8/8/8/8/8/8/R2RK2R w KD - 0 1"), "Shredder three-rook FEN");
        CHECK(p.crook[WHITE][1] == 3, "Shredder D must pick the d1 rook, got %u", p.crook[WHITE][1]);
        CHECK(p.chess960 == 1, "a d1 castling rook is 960");

        /* malformed */
        CHECK(!pos_from_fen(&p, "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w HAhz - 0 1"),
              "bad file letter accepted");
        CHECK(!pos_from_fen(&p, "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w E - 0 1"),
              "castling file on the king's own square accepted");
        CHECK(!pos_from_fen(&p, "8/8/8/8/8/8/8/8 w A - 0 1"),
              "castling file with no king accepted");
    }

    done("FEN", before);
}

/* ---- 3. id 518 is classical chess ---------------------------------- */

static void test_518_is_classical(int full)
{
    const int before = g_fail;
    section("id 518 perft == classical perft");

    static const uint64_t want[] = { 1, 20, 400, 8902, 197281, 4865609, 119060324 };
    const int maxd = full ? 6 : 5;

    Position p;
    pos_startpos960(&p, CHESS960_CLASSICAL_ID);
    CHECK(p.chess960 == 1, "the 960 castling path must be the one under test here");
    for (int d = 1; d <= maxd; d++) {
        const uint64_t got = perft(&p, d);
        CHECK(got == want[d], "id 518 perft(%d) = %llu, want %llu",
              d, (unsigned long long)got, (unsigned long long)want[d]);
    }

    /* and the same board under the classical rules must agree move for move */
    {
        Position c;
        pos_startpos(&c);
        for (int d = 1; d <= 4; d++)
            CHECK(perft(&c, d) == perft(&p, d), "classical vs 960-flagged perft(%d)", d);
    }
    done("id 518", before);
}

/* ---- 3b. published Chess960 perft values ---------------------------- */

/* From the Chess Programming Wiki's Chess960 perft table (Shredder-FEN, the
 * numbers Reinhard Scharnagl's and several engines' runs agree on).  These are
 * an INDEPENDENT check: they were produced by other people's move generators,
 * so they cannot share a bug with either generator in this repository. */
static void test_published(int full)
{
    const int before = g_fail;
    section("published Chess960 perft values");

    static const struct { const char *fen; uint64_t n[7]; } cases[] = {
        { "bqnb1rkr/pp3ppp/3ppn2/2p5/5P2/P2P4/NPP1P1PP/BQ1BNRKR w HFhf - 2 9",
          { 0, 21, 528, 12189, 326672, 8146062, 227689589 } },
        { "2nnrbkr/p1qppppp/8/1ppb4/6PP/3PP3/PPP2P2/BQNNRBKR w HEhe - 1 9",
          { 0, 21, 807, 18002, 667366, 16253601, 590751109 } },
        { "b1q1rrkb/pppppppp/3nn3/8/P7/1PPP4/4PPPP/BQNNRKRB w GE - 1 9",
          { 0, 20, 479, 10471, 273318, 6417013, 177654692 } },
        { "qbbnnrkr/2pp2pp/p7/1p2pp2/8/P3PP2/1PPP1KPP/QBBNNR1R w hf - 0 9",
          { 0, 22, 593, 13440, 382958, 9183776, 274103539 } },
        { "1nbbnrkr/p1p1ppp1/3p4/1p3P1p/3Pq2P/8/PPP1P1P1/QNBBNRKR w HFhf - 0 9",
          { 0, 28, 1120, 31058, 1171749, 34030312, 1250970898 } },
        { "qnbnr1kr/ppp1b1pp/4p3/3p1p2/8/2NPP3/PPP1BPPP/QNB1R1KR w HEhe - 1 9",
          { 0, 29, 899, 26578, 824055, 24851983, 775718317 } },
        { "qbn1brkr/ppp1p1p1/2n4p/3p1p2/P7/6PP/QPPPPP2/1BNNBRKR w HFhf - 0 9",
          { 0, 25, 635, 17054, 465806, 13203304, 377184252 } },
        { "qnnbbrkr/1p2ppp1/2pp3p/p7/1P5P/2NP4/P1P1PPP1/Q1NBBRKR w HFhf - 0 9",
          { 0, 24, 572, 15243, 384260, 11110203, 293989890 } },
        { "qnr1bkrb/pppp2pp/3np3/5p2/8/P2P2P1/NPP1PP1P/QN1RBKRB w GDg - 3 9",
          { 0, 33, 823, 26895, 713420, 23114629, 646390782 } },
        { "qb1nrkbr/1pppp1p1/1n3p2/p1B4p/8/3P1P1P/PPP1P1P1/QBNNRK1R w HEhe - 0 9",
          { 0, 31, 855, 25620, 735703, 21796206, 651054626 } },
    };

    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        Position p;
        if (!pos_from_fen(&p, cases[i].fen)) {
            CHECK(0, "published FEN rejected: %s", cases[i].fen);
            continue;
        }
        char rt[128];
        pos_to_fen(&p, rt, sizeof(rt));
        CHECK(strcmp(rt, cases[i].fen) == 0,
              "published FEN does not round-trip:\n    in  %s\n    out %s", cases[i].fen, rt);
        CHECK(same_moves(&p, 1), "published position disagrees with the reference: %s", cases[i].fen);

        const int maxd = full ? 6 : 5;
        for (int d = 1; d <= maxd; d++) {
            const uint64_t got = perft(&p, d);
            CHECK(got == cases[i].n[d], "perft(%d) = %llu, published %llu for %s",
                  d, (unsigned long long)got, (unsigned long long)cases[i].n[d], cases[i].fen);
            if (g_fail != before) return;
        }
    }
    done("published perft", before);
}

/* ---- 4. the differential gate -------------------------------------- */

static void test_differential(int quick, int full)
{
    const int before = g_fail;
    section("differential: fast generator vs naive reference");

    const int depth = 4;
    const int step = quick ? 32 : 1;
    uint64_t total = 0;
    double t0 = now_sec();
    int tested = 0;

    for (int id = 0; id < 960; id += step) {
        Position p;
        pos_startpos960(&p, id);
        Position snap = p;
        const uint64_t n = diff_perft(&p, depth);
        total += n;
        tested++;
        CHECK(memcmp(&snap, &p, sizeof(Position)) == 0, "id %d: perft walk corrupted the position", id);
        if (g_fail != before) { printf("      (first failure at id %d)\n", id); break; }
    }
    printf("      %d start positions, move sets compared at every node to depth %d,"
           " %llu leaves, %.1fs\n", tested, depth, (unsigned long long)total, now_sec() - t0);

    /* An independent second opinion at depth 5: two complete recursions, one
     * driven by each generator, compared only on the final counts. */
    if (g_fail == before) {
        const int nsample = full ? 60 : (quick ? 8 : 34);
        t0 = now_sec();
        uint64_t sum = 0;
        for (int i = 0; i < nsample; i++) {
            const int id = (i * 97 + 11) % 960;
            Position p;
            pos_startpos960(&p, id);
            const uint64_t a = perft(&p, 5);
            const uint64_t b = ref_perft(&p, 5);
            CHECK(a == b, "id %d: perft(5) engine %llu, naive %llu",
                  id, (unsigned long long)a, (unsigned long long)b);
            sum += a;
            if (g_fail != before) break;
        }
        printf("      %d start positions cross-checked at depth 5, %llu leaves, %.1fs\n",
               nsample, (unsigned long long)sum, now_sec() - t0);
    }

    done("differential", before);
}

/* ---- 5. random games: sets, undo, zobrist, UCI, SAN ---------------- */

static void playout_checks(int ngames, int maxply, int deep)
{
    for (int g = 0; g < ngames && g_fail == 0; g++) {
        Position p;
        pos_startpos960(&p, rnd_below(960));

        for (int ply = 0; ply < maxply; ply++) {
            if (!pos_consistent(&p, "playout")) return;
            if (!same_moves(&p, 1)) { CHECK(0, "generator mismatch during playout"); return; }

            Move list[MAX_MOVES];
            const int n = gen_legal(&p, list);
            if (n == 0) break;

            /* captures must be exactly the capturing subset */
            {
                Move caps[MAX_MOVES], want[MAX_MOVES];
                const int nc = gen_legal_captures(&p, caps);
                int nw = 0;
                for (int i = 0; i < n; i++) {
                    const int fl = MV_FLAG(list[i]);
                    /* captures, plus queen promotions (under-promotions are
                     * quiescence noise and are deliberately left out) */
                    const int keep = MV_IS_PROMO(list[i])
                                   ? (fl == MF_PROMO_Q || fl == MF_PROMO_QC)
                                   : ((fl & 4) != 0);
                    if (keep) want[nw++] = list[i];
                }
                sort_moves(caps, nc);
                sort_moves(want, nw);
                CHECK(nc == nw && memcmp(caps, want, (size_t)nc * sizeof(Move)) == 0,
                      "gen_legal_captures is not the capture subset (%d vs %d)", nc, nw);
            }

            RefBoard b0;
            ref_load(&b0, &p);

            for (int i = 0; i < n && g_fail == 0; i++) {
                const Move m = list[i];

                /* UCI round-trip, castling included */
                char u[8];
                Move back = MV_NONE;
                move_to_uci_pos(&p, m, u);
                CHECK(move_from_uci(&p, u, &back), "move_from_uci rejected \"%s\"", u);
                CHECK(back == m, "UCI round-trip changed \"%s\" (%u -> %u)", u, m, back);

                /* SAN must be produced and, for castling, be O-O / O-O-O */
                char san[16];
                move_to_san(&p, m, san, sizeof(san));
                CHECK(san[0] != '\0', "empty SAN for \"%s\"", u);
                if (MV_FLAG(m) == MF_KCASTLE)
                    CHECK(strncmp(san, "O-O", 3) == 0 && san[3] != '-', "SAN for O-O: %s", san);
                if (MV_FLAG(m) == MF_QCASTLE)
                    CHECK(strncmp(san, "O-O-O", 5) == 0, "SAN for O-O-O: %s", san);

                /* make: the board must be what the naive applier says, the key
                 * must match a recomputation, and unmake must restore bit for bit */
                Position snap = p;
                Undo un;
                make_move(&p, m, &un);

                RefBoard expect = b0;
                ref_apply(&expect, &snap, m);
                for (int s = 0; s < 64; s++) {
                    if (p.board[s] != expect.pc[s] || p.color_at[s] != expect.col[s]) {
                        char fen[128];
                        pos_to_fen(&snap, fen, sizeof(fen));
                        CHECK(0, "make_move(%s) wrong at %s in \"%s\"", u, SQ_NAMES[s], fen);
                        break;
                    }
                }
                CHECK(p.key == pos_compute_key(&p), "incremental key after \"%s\"", u);
                CHECK(p.side == (snap.side ^ 1), "side not flipped by \"%s\"", u);
                if (MV_FLAG(m) == MF_KCASTLE || MV_FLAG(m) == MF_QCASTLE) {
                    const int base = (snap.side == WHITE) ? 0 : 56;
                    const int kto = base + (MV_FLAG(m) == MF_KCASTLE ? 6 : 2);
                    const int rto = base + (MV_FLAG(m) == MF_KCASTLE ? 5 : 3);
                    CHECK(p.board[kto] == KING && p.color_at[kto] == snap.side,
                          "king not on %s after \"%s\"", SQ_NAMES[kto], u);
                    CHECK(p.board[rto] == ROOK && p.color_at[rto] == snap.side,
                          "rook not on %s after \"%s\"", SQ_NAMES[rto], u);
                    const int mine = (snap.side == WHITE) ? (CR_WK | CR_WQ) : (CR_BK | CR_BQ);
                    CHECK((p.castling & mine) == 0, "castling did not burn the rights");
                }
                if (deep) pos_consistent(&p, "after make");

                unmake_move(&p, m, &un);
                CHECK(memcmp(&snap, &p, sizeof(Position)) == 0,
                      "unmake did not restore the position after \"%s\"", u);
            }
            if (g_fail) return;

            Undo un;
            make_move(&p, list[rnd_below(n)], &un);
            if (p.halfmove >= 100) break;
        }
    }
}

static void test_random_games(int quick, int full)
{
    const int before = g_fail;
    section("random 960 games: sets, make/unmake, zobrist, UCI, SAN");
    const int games = quick ? 12 : (full ? 400 : 120);
    playout_checks(games, 120, 1);
    done("random games", before);
}

/* ---- 6. FEN round-trip over random positions ----------------------- */

static void test_fen_random(int quick)
{
    const int before = g_fail;
    section("FEN round-trip over random 960 positions");

    const int want = quick ? 800 : 10000;
    int seen = 0;
    while (seen < want && g_fail == before) {
        Position p;
        pos_startpos960(&p, rnd_below(960));
        for (int ply = 0; ply < 160 && seen < want; ply++) {
            char what[64];
            snprintf(what, sizeof(what), "random position %d", seen);
            check_fen_roundtrip(&p, what);
            seen++;
            if (g_fail != before) return;
            Move list[MAX_MOVES];
            const int n = gen_legal(&p, list);
            if (!n) break;
            Undo u;
            make_move(&p, list[rnd_below(n)], &u);
            if (p.halfmove >= 100) break;
        }
    }
    printf("      %d positions round-tripped in both notations\n", seen);
    done("FEN random", before);
}

/* ---- 7. castling edge cases ---------------------------------------- */

static int has_castle(const Position *p, int flag)
{
    Move list[MAX_MOVES];
    const int n = gen_legal(p, list);
    for (int i = 0; i < n; i++) if (MV_FLAG(list[i]) == flag) return 1;
    return 0;
}

static Move get_castle(const Position *p, int flag)
{
    Move list[MAX_MOVES];
    const int n = gen_legal(p, list);
    for (int i = 0; i < n; i++) if (MV_FLAG(list[i]) == flag) return list[i];
    return MV_NONE;
}

/* Plays the castling move of `flag` and checks the two pieces land right. */
static void check_castle_lands(const char *fen, int flag, const char *name)
{
    Position p;
    if (!pos_from_fen(&p, fen)) { CHECK(0, "%s: bad FEN %s", name, fen); return; }
    const Move m = get_castle(&p, flag);
    if (m == MV_NONE) { CHECK(0, "%s: castling not generated in %s", name, fen); return; }

    const int us = p.side, base = (us == WHITE) ? 0 : 56;
    const int kto = base + (flag == MF_KCASTLE ? 6 : 2);
    const int rto = base + (flag == MF_KCASTLE ? 5 : 3);
    Position snap = p;
    Undo u;
    make_move(&p, m, &u);
    CHECK(p.board[kto] == KING && p.color_at[kto] == us,
          "%s: king not on %s", name, SQ_NAMES[kto]);
    CHECK(p.board[rto] == ROOK && p.color_at[rto] == us,
          "%s: rook not on %s", name, SQ_NAMES[rto]);
    CHECK(bb_count(p.piece[us][KING]) == 1 && bb_count(p.piece[us][ROOK]) == bb_count(snap.piece[us][ROOK]),
          "%s: pieces lost or duplicated", name);
    CHECK(p.key == pos_compute_key(&p), "%s: key after castling", name);
    pos_consistent(&p, name);
    unmake_move(&p, m, &u);
    CHECK(memcmp(&snap, &p, sizeof(Position)) == 0, "%s: unmake", name);
}

static void test_edge_cases(void)
{
    const int before = g_fail;
    section("Chess960 castling edge cases");

    /* King already on g1 with its rook on h1: O-O is a NULL KING MOVE, only the
     * rook travels (h1 -> f1).  Note that the king can never move LEFT for
     * king-side castling: the king-side rook is by definition on a higher file
     * than the king, so the king starts at f1 or below... or exactly on g1. */
    check_castle_lands("1rk5/8/8/8/8/8/8/1R4KR w H - 0 1", MF_KCASTLE, "king on g1, rook on h1");
    {
        Position p;
        pos_from_fen(&p, "1rk5/8/8/8/8/8/8/1R4KR w H - 0 1");
        const Move m = get_castle(&p, MF_KCASTLE);
        CHECK(m != MV_NONE, "null-king O-O not generated");
        CHECK(MV_FROM(m) == 6 && MV_TO(m) == 7, "null-king castle encoded %d->%d",
              MV_FROM(m), MV_TO(m));
        char u[8];
        move_to_uci_pos(&p, m, u);
        CHECK(strcmp(u, "g1h1") == 0, "king-takes-rook UCI: %s", u);
        Move back;
        CHECK(move_from_uci(&p, "g1h1", &back) && back == m, "g1h1 must be the castle");
        CHECK(move_from_uci(&p, "g1g1", &back) && back == m, "g1g1 (king-destination form)");
    }

    /* A king on g1 whose castling rook is on f1 is a QUEEN-side right (the rook
     * is on the queen's side of the king), and the king walks g1 -> c1. */
    check_castle_lands("1rk5/8/8/8/8/8/8/1R3RK1 w F - 0 1", MF_QCASTLE, "king g1 -> c1, rook f1 -> d1");
    /* The longest walk there is: king h1 -> c1 with its rook on g1. */
    check_castle_lands("1rk5/8/8/8/8/8/8/1R4RK w G - 0 1", MF_QCASTLE, "king h1 -> c1");
    /* The king moves RIGHT for queen-side castling: king b1, rook a1. */
    check_castle_lands("2k5/8/8/8/8/8/8/RK6 w A - 0 1", MF_QCASTLE, "king b1 -> c1 (rightwards O-O-O)");

    /* the rook stands on the king's destination square, and vice versa */
    check_castle_lands("1rk5/8/8/8/8/8/8/1R2KR2 w F - 0 1", MF_KCASTLE, "rook already on f1");
    check_castle_lands("2k5/8/8/8/8/8/8/4K1R1 w G - 0 1", MF_KCASTLE, "rook on the king's destination g1");
    check_castle_lands("2k5/8/8/8/8/8/8/1R1K3R w HB - 0 1", MF_QCASTLE, "king on the rook's destination d1");
    check_castle_lands("2k5/8/8/8/8/8/8/1RK4R w HB - 0 1", MF_QCASTLE, "king already on c1");
    check_castle_lands("2k5/8/8/8/8/8/8/1RK4R w HB - 0 1", MF_KCASTLE, "king c1 -> g1 across the board");

    /* queen-side rook on b1: b1 is on the rook's path but not the king's */
    check_castle_lands("4k3/8/8/8/8/8/8/1R2K3 w B - 0 1", MF_QCASTLE, "queen-side rook on b1");
    {
        Position p;
        /* a knight on c1 sits only on the ROOK's path and must still block */
        CHECK(pos_from_fen(&p, "4k3/8/8/8/8/8/8/1RN1K3 w B - 0 1"), "fen");
        CHECK(!has_castle(&p, MF_QCASTLE), "a piece on the rook's path must block castling");
        /* while a1, which neither piece crosses, must not block */
        CHECK(pos_from_fen(&p, "4k3/8/8/8/8/8/8/NR2K3 w B - 0 1"), "fen");
        CHECK(has_castle(&p, MF_QCASTLE), "a piece on a1 must not block a b1 rook");
        /* b1 is not on the king's path, so an attack on it does not matter */
        CHECK(pos_from_fen(&p, "1r2k3/8/8/8/8/8/8/1R2K3 w B - 0 1"), "fen");
        CHECK(has_castle(&p, MF_QCASTLE), "an attack on b1 must not stop castling");
        /* d1 is on the king's path: an attack there does stop it */
        CHECK(pos_from_fen(&p, "3rk3/8/8/8/8/8/8/1R2K3 w B - 0 1"), "fen");
        CHECK(!has_castle(&p, MF_QCASTLE), "an attack on d1 must stop castling");
    }

    /* A FEN may claim a right whose king is not on its back rank at all: no
     * castling may be generated from it (and the naive reference agrees). */
    {
        Position p;
        CHECK(pos_from_fen(&p, "4k3/8/8/8/4K3/8/8/7R w H - 0 1"), "off-rank king FEN");
        CHECK(p.castling == CR_WK && p.chess960 == 1, "off-rank rights %u", p.castling);
        CHECK(!has_castle(&p, MF_KCASTLE) && !has_castle(&p, MF_QCASTLE),
              "castling generated for a king that is not on its back rank");
        CHECK(same_moves(&p, 1), "off-rank king disagrees with the reference");
        CHECK(pos_from_fen(&p, "4k3/8/8/8/4K3/8/8/R6R w HA - 0 1"), "off-rank king FEN 2");
        CHECK(!has_castle(&p, MF_KCASTLE) && !has_castle(&p, MF_QCASTLE), "off-rank castling");
        CHECK(same_moves(&p, 1), "off-rank king disagrees with the reference (2)");
    }

    /* in check: never */
    {
        Position p;
        CHECK(pos_from_fen(&p, "4k3/8/8/8/8/8/8/1R2K1r1 w B - 0 1"), "fen");
        CHECK(in_check(&p, WHITE), "white should be in check");
        CHECK(!has_castle(&p, MF_QCASTLE), "castling out of check");
    }

    /* the rook was the only thing shielding the king: vacating it is illegal.
     * Black rook a1, white rook b1, white king c1 -- the king does not move at
     * all, but the rook leaving b1 would open the a1 rook onto c1. */
    {
        Position p;
        CHECK(pos_from_fen(&p, "4k3/8/8/8/8/8/8/rRK5 w B - 0 1"), "fen");
        CHECK(!in_check(&p, WHITE), "the b1 rook shields c1");
        CHECK(!has_castle(&p, MF_QCASTLE),
              "castling that opens a line onto the king's own square must be refused");
        /* one file further along: the king does move, same reasoning */
        CHECK(pos_from_fen(&p, "4k3/8/8/8/8/8/8/rR1K4 w B - 0 1"), "fen");
        CHECK(!has_castle(&p, MF_QCASTLE), "castling into the vacated rook's line");
        /* but with the a-file rook gone it is fine again */
        CHECK(pos_from_fen(&p, "4k3/8/8/8/8/8/8/1R1K4 w B - 0 1"), "fen");
        CHECK(has_castle(&p, MF_QCASTLE), "castling should be legal without the enemy rook");
    }

    /* the king's transit square is judged in the position before the move */
    {
        Position p;
        /* king b1, rook a1, enemy rook h1: c1 is attacked right now, so no */
        CHECK(pos_from_fen(&p, "4k3/8/8/8/8/8/8/RK5r w A - 0 1"), "fen");
        CHECK(!has_castle(&p, MF_QCASTLE), "the king may not cross an attacked c1");
    }

    /* rights die with the king, the rook, or the rook's capture at home */
    {
        Position p;
        Undo u;
        CHECK(pos_from_fen(&p, "1rk5/8/8/8/8/8/8/1R2K1R1 w GB - 0 1"), "fen");
        CHECK(p.castling == (CR_WK | CR_WQ), "rights %u", p.castling);
        Move m;
        CHECK(move_from_uci(&p, "e1e2", &m), "e1e2");
        make_move(&p, m, &u);
        CHECK(p.castling == 0, "a king move must burn both rights, left %u", p.castling);
        CHECK(p.key == pos_compute_key(&p), "key");
        unmake_move(&p, m, &u);
        CHECK(p.castling == (CR_WK | CR_WQ), "rights restored");

        CHECK(move_from_uci(&p, "g1g2", &m), "g1g2");
        make_move(&p, m, &u);
        CHECK(p.castling == CR_WQ, "a rook move must burn only its own right, left %u", p.castling);
        unmake_move(&p, m, &u);

        CHECK(move_from_uci(&p, "b1b2", &m), "b1b2");
        make_move(&p, m, &u);
        CHECK(p.castling == CR_WK, "queen-side right survived its rook moving: %u", p.castling);
        unmake_move(&p, m, &u);
    }
    {
        /* black rook captures the white rook on its 960 home square g1 */
        Position p;
        Undo u;
        CHECK(pos_from_fen(&p, "1rk3r1/8/8/8/8/8/8/1R2K1R1 b GBgb - 0 1"), "fen");
        Move m;
        CHECK(move_from_uci(&p, "g8g1", &m), "g8g1");
        make_move(&p, m, &u);
        CHECK((p.castling & CR_WK) == 0, "capturing the g1 rook must kill white's K right");
        CHECK((p.castling & CR_WQ) != 0, "it must not kill the queen-side right");
        CHECK((p.castling & CR_BK) == 0, "the capturing rook lost its own right");
        CHECK(p.key == pos_compute_key(&p), "key after the rook capture");
        unmake_move(&p, m, &u);
        CHECK(p.castling == CR_ALL, "rights restored after unmake, got %u", p.castling);
    }

    /* both notations of the same castling move, and the ambiguous "e1g1" */
    {
        Position p;
        Move m, back;
        /* king e1, rooks a1/h1 but flagged 960: e1h1 and e1g1 both mean O-O */
        CHECK(pos_from_fen(&p, "4k3/8/8/8/8/8/8/R3K2R w HA - 0 1"), "fen");
        m = get_castle(&p, MF_KCASTLE);
        CHECK(m != MV_NONE && MV_TO(m) == 7, "960 castling must be king-takes-rook");
        char u[8];
        move_to_uci_pos(&p, m, u);
        CHECK(strcmp(u, "e1h1") == 0, "move_to_uci_pos emitted %s", u);
        CHECK(move_from_uci(&p, "e1h1", &back) && back == m, "e1h1");
        CHECK(move_from_uci(&p, "e1g1", &back) && back == m, "e1g1 must fall back to castling");

        /* the classical spelling of the same board emits the classical form */
        CHECK(pos_from_fen(&p, "4k3/8/8/8/8/8/8/R3K2R w KQ - 0 1"), "fen");
        m = get_castle(&p, MF_KCASTLE);
        CHECK(MV_TO(m) == 6, "classical castling must be king-to-g1");
        move_to_uci_pos(&p, m, u);
        CHECK(strcmp(u, "e1g1") == 0, "classical move_to_uci_pos emitted %s", u);
        move_to_uci(m, u);
        CHECK(strcmp(u, "e1g1") == 0, "the old move_to_uci must be unchanged: %s", u);
        CHECK(move_from_uci(&p, "e1g1", &back) && back == m, "classical e1g1");
        CHECK(move_from_uci(&p, "e1h1", &back) && back == m, "classical e1h1 tolerated");

        /* THE ambiguity: king f1, rook h1, g1 empty.  "f1g1" is both a legal
         * plain king move and the classical spelling of O-O; the plain move
         * wins, and "f1h1" is the unambiguous king-takes-rook spelling. */
        CHECK(pos_from_fen(&p, "4k3/8/8/8/8/8/8/1R3K1R w HB - 0 1"), "fen");
        CHECK(move_from_uci(&p, "f1g1", &back), "f1g1 must be legal");
        CHECK(MV_FLAG(back) == MF_QUIET && MV_FROM(back) == 5 && MV_TO(back) == 6,
              "f1g1 must be read as the plain king move, got flag %d", MV_FLAG(back));
        CHECK(move_from_uci(&p, "f1h1", &back) && MV_FLAG(back) == MF_KCASTLE,
              "f1h1 must be king-side castling");
        CHECK(move_from_uci(&p, "f1b1", &back) && MV_FLAG(back) == MF_QCASTLE,
              "f1b1 must be queen-side castling");
        /* and the plain king move and the castle are different moves */
        {
            Move quiet, castle;
            move_from_uci(&p, "f1g1", &quiet);
            move_from_uci(&p, "f1h1", &castle);
            CHECK(quiet != castle, "the two readings collapsed onto one move");
        }
    }

    /* a castling move that is a null king move must not confuse make/unmake */
    {
        Position p;
        CHECK(pos_from_fen(&p, "1rk5/8/8/8/8/8/8/1R4KR w H - 0 1"), "fen");
        const Move m = get_castle(&p, MF_KCASTLE);
        Position snap = p;
        Undo u;
        make_move(&p, m, &u);
        CHECK(p.board[6] == KING && p.board[5] == ROOK, "g1/f1 after the null-king castle");
        CHECK(bb_count(p.piece[WHITE][KING]) == 1, "king vanished");
        CHECK(bb_count(p.piece[WHITE][ROOK]) == 2, "rook vanished");
        CHECK(p.key == pos_compute_key(&p), "key");
        unmake_move(&p, m, &u);
        CHECK(memcmp(&snap, &p, sizeof(Position)) == 0, "unmake of a null-king castle");
    }

    /* every hand-built position above must also satisfy the naive reference */
    {
        static const char *fens[] = {
            "1rk5/8/8/8/8/8/8/1R3RK1 w F - 0 1",
            "1rk5/8/8/8/8/8/8/1R4RK w G - 0 1",
            "1rk5/8/8/8/8/8/8/1R4KR w H - 0 1",
            "2k5/8/8/8/8/8/8/RK6 w A - 0 1",
            "2k5/8/8/8/8/8/8/4K1R1 w G - 0 1",
            "2k5/8/8/8/8/8/8/1R1K3R w HB - 0 1",
            "2k5/8/8/8/8/8/8/1RK4R w HB - 0 1",
            "4k3/8/8/8/8/8/8/1R3K1R w HB - 0 1",
            "4k3/8/8/8/4K3/8/8/7R w H - 0 1",
            "4k3/8/8/8/4K3/8/8/R6R w HA - 0 1",
            "1rk5/8/8/8/8/8/8/1R2KR2 w F - 0 1",
            "2kr4/8/8/8/8/8/8/2KR3R w HD - 0 1",
            "4k3/8/8/8/8/8/8/1R2K3 w B - 0 1",
            "4k3/8/8/8/8/8/8/1RN1K3 w B - 0 1",
            "4k3/8/8/8/8/8/8/NR2K3 w B - 0 1",
            "1r2k3/8/8/8/8/8/8/1R2K3 w B - 0 1",
            "3rk3/8/8/8/8/8/8/1R2K3 w B - 0 1",
            "4k3/8/8/8/8/8/8/1R2K1r1 w B - 0 1",
            "4k3/8/8/8/8/8/8/rRK5 w B - 0 1",
            "4k3/8/8/8/8/8/8/rR1K4 w B - 0 1",
            "4k3/8/8/8/8/8/8/1R1K4 w B - 0 1",
            "4k3/8/8/8/8/8/8/RK5r w A - 0 1",
            "1rk3r1/8/8/8/8/8/8/1R2K1R1 b GBgb - 0 1",
            "4k3/8/8/8/8/8/8/R3K2R w HA - 0 1",
            "4k3/8/8/8/8/8/8/1R3KR1 w GB - 0 1",
            "2r1k2r/8/8/8/8/8/8/2R1K2R w HChc - 0 1",
            "rk2r3/pppppppp/8/8/8/8/PPPPPPPP/RK2R3 w AEae - 0 1",
        };
        for (size_t i = 0; i < sizeof(fens) / sizeof(fens[0]); i++) {
            Position p;
            CHECK(pos_from_fen(&p, fens[i]), "edge FEN rejected: %s", fens[i]);
            CHECK(same_moves(&p, 1), "edge case disagrees with the reference: %s", fens[i]);
            const uint64_t a = perft(&p, 3), b = ref_perft(&p, 3);
            CHECK(a == b, "edge perft(3) %llu vs %llu: %s",
                  (unsigned long long)a, (unsigned long long)b, fens[i]);
            check_fen_roundtrip(&p, fens[i]);
        }
    }

    done("edge cases", before);
}

/* ---- 7b. differential where castling actually happens --------------- */

/* Depth 4 from the 960 start positions barely reaches a castling move: the
 * squares between the king and its rook have to be cleared first.  This walk
 * hunts for positions where castling IS legal and runs the differential from
 * there, so the 960 castling code is exercised in real games rather than only
 * in hand-built FENs. */
static void test_castling_differential(int quick)
{
    const int before = g_fail;
    section("differential from positions where castling is legal");

    const int want = quick ? 150 : 2500;
    int found = 0, guard = 0;
    uint64_t leaves = 0;
    const double t0 = now_sec();

    while (found < want && g_fail == before && guard++ < 4000) {
        Position p;
        pos_startpos960(&p, rnd_below(960));
        for (int ply = 0; ply < 140 && found < want; ply++) {
            Move list[MAX_MOVES];
            const int n = gen_legal(&p, list);
            if (!n) break;
            int castles = 0;
            for (int i = 0; i < n; i++) {
                const int fl = MV_FLAG(list[i]);
                if (fl == MF_KCASTLE || fl == MF_QCASTLE) castles++;
            }
            if (castles) {
                Position snap = p;
                leaves += diff_perft(&p, 3);
                CHECK(memcmp(&snap, &p, sizeof(Position)) == 0, "castling perft corrupted the position");
                found++;
                if (g_fail != before) return;
            }
            Undo u;
            /* bias towards castling so that the follow-up positions differ */
            int pick = rnd_below(n);
            if (castles && (rnd() & 3) == 0)
                for (int i = 0; i < n; i++) {
                    const int fl = MV_FLAG(list[i]);
                    if (fl == MF_KCASTLE || fl == MF_QCASTLE) { pick = i; break; }
                }
            make_move(&p, list[pick], &u);
            if (p.halfmove >= 100) break;
        }
    }
    printf("      %d positions with a legal castling move, depth 3 each,"
           " %llu leaves, %.1fs\n", found, (unsigned long long)leaves, now_sec() - t0);
    CHECK(found >= want / 2, "only found %d castling positions", found);
    done("castling differential", before);
}

/* ---- 7c. the Game layer -------------------------------------------- */

static void test_game_layer(void)
{
    const int before = g_fail;
    section("game_start960");

    for (int rep = 0; rep < 40 && g_fail == before; rep++) {
        const int id = rnd_below(960);
        Game g;
        game_start960(&g, id);
        Position start = g.pos;

        CHECK(g.pos.chess960 == 1, "game_start960 did not flag the position");
        CHECK(pos_960_id(&g.pos) == id, "game_start960(%d) produced a different array", id);
        CHECK(g.ply == 0 && g.hist_len == 1 && g.hist[0] == g.pos.key, "game_start960 bookkeeping");
        CHECK(g.result == GR_ONGOING && g.reason == TR_NONE, "a 960 start is not ongoing");

        while (g.result == GR_ONGOING && g.ply < 300) {
            Move list[MAX_MOVES];
            const int n = gen_legal(&g.pos, list);
            if (!n) break;
            game_push(&g, list[rnd_below(n)]);
            if (g.pos.key != pos_compute_key(&g.pos)) { CHECK(0, "game_push key"); break; }
            if (g.hist[g.ply] != g.pos.key) { CHECK(0, "game history key"); break; }
        }
        while (g.ply > 0) game_pop(&g);
        CHECK(memcmp(&start, &g.pos, sizeof(Position)) == 0,
              "popping a whole 960 game did not restore the start position");
    }
    done("game layer", before);
}

/* ---- 8. SAN disambiguation from 960 starts -------------------------- */

static void test_san(void)
{
    const int before = g_fail;
    section("SAN uniqueness from 960 starts");

    /* Correct disambiguation means every legal move of a position has its own
     * SAN.  Two knights or two rooks reaching the same square from a 960 array
     * is exactly the case that breaks a naive implementation. */
    int positions = 0;
    for (int g = 0; g < 40 && g_fail == before; g++) {
        Position p;
        pos_startpos960(&p, rnd_below(960));
        for (int ply = 0; ply < 60; ply++) {
            Move list[MAX_MOVES];
            const int n = gen_legal(&p, list);
            if (!n) break;
            char sans[MAX_MOVES][16];
            for (int i = 0; i < n; i++) move_to_san(&p, list[i], sans[i], sizeof(sans[i]));
            for (int i = 0; i < n && g_fail == before; i++)
                for (int j = i + 1; j < n; j++)
                    if (!strcmp(sans[i], sans[j])) {
                        char fen[128];
                        pos_to_fen(&p, fen, sizeof(fen));
                        CHECK(0, "two moves share the SAN \"%s\" in %s", sans[i], fen);
                        break;
                    }
            positions++;
            Undo u;
            make_move(&p, list[rnd_below(n)], &u);
            if (p.halfmove >= 100) break;
        }
    }
    printf("      %d positions checked\n", positions);

    /* explicit: two rooks that can both reach d1 from a 960 array */
    {
        Position p;
        CHECK(pos_from_fen(&p, "4k3/8/8/8/8/8/4K3/R6R w - - 0 1"), "fen");
        Move m;
        char san[16];
        CHECK(move_from_uci(&p, "a1d1", &m), "a1d1");
        move_to_san(&p, m, san, sizeof(san));
        CHECK(strcmp(san, "Rad1") == 0, "SAN disambiguation by file: %s", san);
        CHECK(move_from_uci(&p, "h1d1", &m), "h1d1");
        move_to_san(&p, m, san, sizeof(san));
        CHECK(strcmp(san, "Rhd1") == 0, "SAN disambiguation by file: %s", san);
        /* a rook that cannot really reach the square must not force a file */
        CHECK(pos_from_fen(&p, "4k3/8/8/8/8/8/8/R2K3R w - - 0 1"), "fen");
        CHECK(move_from_uci(&p, "a1c1", &m), "a1c1");
        move_to_san(&p, m, san, sizeof(san));
        CHECK(strcmp(san, "Rc1") == 0, "the h1 rook is blocked by the king: %s", san);
    }
    done("SAN", before);
}

/* ---- 9. throughput -------------------------------------------------- */

static uint64_t perft_nobulk(Position *p, int depth)
{
    if (depth == 0) return 1;
    Move list[MAX_MOVES];
    const int n = gen_legal(p, list);
    uint64_t nodes = 0;
    Undo u;
    for (int i = 0; i < n; i++) {
        make_move(p, list[i], &u);
        nodes += perft_nobulk(p, depth - 1);
        unmake_move(p, list[i], &u);
    }
    return nodes;
}

static void test_speed(void)
{
    section("throughput");
    static const int ids[] = { CHESS960_CLASSICAL_ID, 0, 137, 646, 959 };
    for (size_t i = 0; i < sizeof(ids) / sizeof(ids[0]); i++) {
        Position p;
        pos_startpos960(&p, ids[i]);
        double best = 0.0;
        uint64_t nodes = 0;
        for (int r = 0; r < 5; r++) {
            const double t0 = now_sec();
            nodes = perft_nobulk(&p, 5);
            const double dt = now_sec() - t0;
            const double rate = (double)nodes / dt / 1e6;
            if (rate > best) best = rate;
        }
        printf("      960 id %3d  d5 %10llu nodes   %6.2f Mnps (make/unmake every leaf)\n",
               ids[i], (unsigned long long)nodes, best);
    }
    {
        Position p;
        pos_startpos(&p);
        double best = 0.0;
        uint64_t nodes = 0;
        for (int r = 0; r < 5; r++) {
            const double t0 = now_sec();
            nodes = perft_nobulk(&p, 5);
            const double dt = now_sec() - t0;
            const double rate = (double)nodes / dt / 1e6;
            if (rate > best) best = rate;
        }
        printf("      classical   d5 %10llu nodes   %6.2f Mnps (make/unmake every leaf)\n",
               (unsigned long long)nodes, best);
    }
}

/* ---------------------------------------------------------------- main */

int main(int argc, char **argv)
{
    const int quick = (argc > 1 && strcmp(argv[1], "quick") == 0);
    const int full  = (argc > 1 && strcmp(argv[1], "full") == 0);

    chess_init();
    printf("chess960 test suite (%s)\n", quick ? "quick" : (full ? "FULL" : "standard"));
    printf("===============================================================\n");

    test_numbering();
    test_fen();
    test_518_is_classical(full);
    test_published(full);
    test_edge_cases();
    test_san();
    test_game_layer();
    test_random_games(quick, full);
    test_fen_random(quick);
    test_castling_differential(quick);
    test_differential(quick, full);
    test_speed();

    printf("===============================================================\n");
    printf("%ld checks, %d failures\n", g_checks, g_fail);
    printf("%s\n", g_fail ? "FAILED" : "ALL PASS");
    return g_fail ? 1 : 0;
}
