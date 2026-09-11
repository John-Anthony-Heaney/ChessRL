/* chess.c -- complete FIDE-rules bitboard chess core.
 *
 * Implementation notes
 * --------------------
 *  * Sliding attacks use "fancy" magic bitboards.  The magic multipliers are
 *    searched for at chess_init() time with a fixed-seed xorshift64* PRNG, so
 *    the tables are bit-for-bit reproducible across runs and machines and no
 *    hand-typed constant table can rot.  (PEXT is not available on arm64.)
 *  * gen_legal() is a *true* legal generator: it computes the checker set, the
 *    check-evasion target mask and the absolute-pin rays up front, so every
 *    move it emits is legal without a make/unmake filter.  The single exception
 *    is en passant, whose horizontal discovered-check case is verified with an
 *    explicit occupancy test (cheap, and it happens at most twice per node).
 *  * Zobrist: the key is
 *        XOR over pieces of Z_PIECE[colour][type][square]
 *      ^ Z_CASTLE[castling rights bitmask]          (always, index 0..15)
 *      ^ Z_EP[file(ep)]   when p->ep >= 0           (file only, not the square)
 *      ^ Z_SIDE           when side to move == BLACK
 *    The en-passant square is recorded after *every* double pawn push (the
 *    classical FEN convention), whether or not a capture is actually available;
 *    pos_compute_key() and the incremental update in make_move() use exactly
 *    the same rule, so they agree at all times.
 *  * Draw policy: threefold repetition and the fifty-move rule are AUTOMATIC
 *    draws here (no claim required).  That is what self-play engines do, and it
 *    is what game_update_result() implements.
 *  * Chess960: Position.chess960 selects the castling rules and Position.crook
 *    holds the castling rooks' origin FILES.  Castling MOVES are encoded as
 *        classical : from = king square, to = king DESTINATION (g1/c1)
 *        chess960  : from = king square, to = castling ROOK's origin square
 *    ("king takes rook").  The 960 form is needed because the king's
 *    destination can coincide with its origin (king already on g1) or with a
 *    normal king move, so (from,to) would not identify the move.  Everything
 *    outside the castling code is identical for the two variants, and the
 *    classical path never touches the 960 branches.
 */

#include "chess.h"

#include <string.h>
#include <stdio.h>
#include <stdlib.h>

/* ------------------------------------------------------------------ misc */

#define BIT(sq) (1ULL << (sq))

/* The Chess960 castling helpers are cold: they must not be inlined into
 * gen_moves() / make_move(), whose register allocation and I-cache footprint
 * are what the self-play throughput is made of. */
#if defined(__GNUC__) || defined(__clang__)
#define COLD_HELPER  __attribute__((noinline))
#define HOT_INLINE   __attribute__((always_inline)) inline
#else
#define COLD_HELPER
#define HOT_INLINE inline
#endif

#define FILE_A 0x0101010101010101ULL
#define FILE_H 0x8080808080808080ULL
#define RANK_1 0x00000000000000FFULL
#define RANK_2 0x000000000000FF00ULL
#define RANK_3 0x0000000000FF0000ULL
#define RANK_6 0x0000FF0000000000ULL
#define RANK_7 0x00FF000000000000ULL
#define RANK_8 0xFF00000000000000ULL

#define LIGHT_SQUARES 0x55AA55AA55AA55AAULL

const char *SQ_NAMES[64] = {
    "a1","b1","c1","d1","e1","f1","g1","h1",
    "a2","b2","c2","d2","e2","f2","g2","h2",
    "a3","b3","c3","d3","e3","f3","g3","h3",
    "a4","b4","c4","d4","e4","f4","g4","h4",
    "a5","b5","c5","d5","e5","f5","g5","h5",
    "a6","b6","c6","d6","e6","f6","g6","h6",
    "a7","b7","c7","d7","e7","f7","g7","h7",
    "a8","b8","c8","d8","e8","f8","g8","h8"
};

#if !defined(__GNUC__) && !defined(__clang__)
int bb_lsb(uint64_t b)   { int i = 0; while (!((b >> i) & 1)) i++; return i; }
int bb_count(uint64_t b) { int c = 0; while (b) { b &= b - 1; c++; } return c; }
#endif

/* -------------------------------------------------------------- tables */

static uint64_t KNIGHT_ATT[64];
static uint64_t KING_ATT[64];
static uint64_t PAWN_ATT[NCOLORS][64];
static uint64_t BETWEEN[64][64];   /* squares strictly between a and b (0 if not aligned) */
static uint64_t LINE[64][64];      /* whole line through a and b       (0 if not aligned) */

typedef struct {
    const uint64_t *attacks;
    uint64_t mask;
    uint64_t magic;
    unsigned shift;
} Magic;

static Magic  BMAGIC[64], RMAGIC[64];
static uint64_t BATTACK[5248];
static uint64_t RATTACK[102400];

static uint64_t Z_PIECE[NCOLORS][NPIECES][64];
static uint64_t Z_CASTLE[16];
static uint64_t Z_EP[8];
static uint64_t Z_SIDE;

static uint8_t CASTLE_MASK[64];

/* Castling-right bit by colour and side (0 = king side, 1 = queen side). */
static const uint8_t CR_BIT[NCOLORS][2] = { { CR_WK, CR_WQ }, { CR_BK, CR_BQ } };

/* Back rank of colour c (a1 or a8). */
static inline int back_rank(int c) { return c == WHITE ? 0 : 56; }
/* Origin square of the castling rook, and the two castling destinations. */
static inline int crook_sq(const Position *p, int c, int side)
{
    return back_rank(c) + (int)p->crook[c][side];
}
static inline int castle_kto(int c, int side) { return back_rank(c) + (side == 0 ? 6 : 2); }
static inline int castle_rto(int c, int side) { return back_rank(c) + (side == 0 ? 5 : 3); }

/* Internal (inlinable) sliding-attack lookups. */
static inline uint64_t bishop_att(int sq, uint64_t occ)
{
    const Magic *m = &BMAGIC[sq];
    return m->attacks[((occ & m->mask) * m->magic) >> m->shift];
}
static inline uint64_t rook_att(int sq, uint64_t occ)
{
    const Magic *m = &RMAGIC[sq];
    return m->attacks[((occ & m->mask) * m->magic) >> m->shift];
}

uint64_t attacks_knight(int sq)               { return KNIGHT_ATT[sq]; }
uint64_t attacks_king(int sq)                 { return KING_ATT[sq]; }
uint64_t attacks_pawn(int sq, int color)      { return PAWN_ATT[color][sq]; }
uint64_t attacks_bishop(int sq, uint64_t occ) { return bishop_att(sq, occ); }
uint64_t attacks_rook(int sq, uint64_t occ)   { return rook_att(sq, occ); }
uint64_t attacks_queen(int sq, uint64_t occ)  { return bishop_att(sq, occ) | rook_att(sq, occ); }

/* ------------------------------------------------------------- init ---- */

static const int BDELTA[4][2] = { { 1, 1 }, { 1, -1 }, { -1, 1 }, { -1, -1 } };
static const int RDELTA[4][2] = { { 1, 0 }, { -1, 0 }, { 0, 1 }, { 0, -1 } };

static uint64_t slide_attacks(int sq, uint64_t occ, const int d[4][2])
{
    uint64_t a = 0;
    int f = sq_file(sq), r = sq_rank(sq);
    for (int i = 0; i < 4; i++) {
        int nf = f + d[i][0], nr = r + d[i][1];
        while (nf >= 0 && nf < 8 && nr >= 0 && nr < 8) {
            int s = nr * 8 + nf;
            a |= BIT(s);
            if (occ & BIT(s)) break;
            nf += d[i][0];
            nr += d[i][1];
        }
    }
    return a;
}

static uint64_t slide_mask(int sq, const int d[4][2])
{
    uint64_t m = 0;
    int f = sq_file(sq), r = sq_rank(sq);
    for (int i = 0; i < 4; i++) {
        int nf = f + d[i][0], nr = r + d[i][1];
        while (nf >= 0 && nf < 8 && nr >= 0 && nr < 8) {
            int ff = nf + d[i][0], rr = nr + d[i][1];
            if (ff < 0 || ff > 7 || rr < 0 || rr > 7) break;   /* edge square: not relevant */
            m |= BIT(nr * 8 + nf);
            nf = ff;
            nr = rr;
        }
    }
    return m;
}

/* xorshift64* -- deterministic, fixed seed. */
static uint64_t g_rs;
static uint64_t mrand(void)
{
    uint64_t x = g_rs;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    g_rs = x;
    return x * 0x2545F4914F6CDD1DULL;
}
static uint64_t mrand_sparse(void) { return mrand() & mrand() & mrand(); }

/* Magic multipliers produced by the search below (fixed seed 0x0F1E2D3C4B5A6978).
 * They are embedded only so that chess_init() costs ~1 ms instead of ~130 ms;
 * every one of them is *verified* at init and, if any candidate ever failed to
 * give a collision-free table, the deterministic search runs for that square. */
static const uint64_t BMAGIC_SEED[64] = {
    0x00120A08520C0041ULL, 0x3804480824408000ULL, 0x0011014202010900ULL, 0x2008048110200402ULL,
    0x000404210000010CULL, 0x0007014840000800ULL, 0x1080825002200000ULL, 0x0002240404013828ULL,
    0x2041854410064A00ULL, 0x12140490008A2082ULL, 0x00000808004C8001ULL, 0x8800144102200100ULL,
    0x4000811040056004ULL, 0x8200008884400000ULL, 0x0014040088284803ULL, 0x0C0006022A0A0204ULL,
    0x0210080821010400ULL, 0x5804880204280A00ULL, 0x6004000200220200ULL, 0x800C200202020000ULL,
    0xC104010180E00401ULL, 0x0000200200842004ULL, 0x70061000480C0400ULL, 0x0002811220841000ULL,
    0x0008408008020882ULL, 0x9022228008084800ULL, 0x0800500028082140ULL, 0x5000802008020220ULL,
    0x2001010084104000ULL, 0x0408142002008411ULL, 0x020081000E051044ULL, 0x0000720000808400ULL,
    0x0C10109060090201ULL, 0x800A511000041000ULL, 0x0000180400480042ULL, 0x0404020080080080ULL,
    0x0A010A0201040104ULL, 0x4070100020314402ULL, 0x0A21220400E20100ULL, 0x02220840400A0214ULL,
    0x0008010822400840ULL, 0x060404018406284AULL, 0x0494140208020102ULL, 0x0042820202006420ULL,
    0x0008082008200500ULL, 0x00081008A0208A00ULL, 0x2120014111004202ULL, 0x0028054502006024ULL,
    0x010400A804908800ULL, 0x400022020260108AULL, 0x0084020201043012ULL, 0x6000114020880010ULL,
    0x2000521202020028ULL, 0x00A0A18C09820004ULL, 0x8010901001004002ULL, 0x0010100100408400ULL,
    0x0042022404022900ULL, 0x5010208208010408ULL, 0x0010540104033400ULL, 0x0483040801084800ULL,
    0x00000004402A8600ULL, 0x4502002204100080ULL, 0x4248040408024400ULL, 0x0020E00C00408320ULL,
};

static const uint64_t RMAGIC_SEED[64] = {
    0x60800010A2400980ULL, 0x0840100020084002ULL, 0x1500084020001100ULL, 0x1200100600402008ULL,
    0x8480040082080080ULL, 0x0200100802000401ULL, 0x0480020000800100ULL, 0x0200020226441081ULL,
    0x8808800880400820ULL, 0x4005401000402009ULL, 0x0011002000104902ULL, 0x0008801000080084ULL,
    0x0CC6000422001008ULL, 0x0002800200040080ULL, 0x1084800100802200ULL, 0x0A20802050800100ULL,
    0x8080004000200040ULL, 0x8080808040002000ULL, 0x0201010010402000ULL, 0x114C808008001000ULL,
    0x0038808004000800ULL, 0x0000080120044010ULL, 0x2800040008029041ULL, 0xA000020020804401ULL,
    0x0440004180008020ULL, 0x0110400080201080ULL, 0x4060001010020400ULL, 0x0418100100090020ULL,
    0x0248110100080004ULL, 0x0002000200041008ULL, 0xA000040101000200ULL, 0x1800800380004900ULL,
    0x4020400424800080ULL, 0x0110004004402000ULL, 0x0102008042001020ULL, 0x200C801000800800ULL,
    0x0230080080800400ULL, 0x2000040080800200ULL, 0x0400081084000201ULL, 0x00010000E1000292ULL,
    0x8090902040008000ULL, 0x0000402010004004ULL, 0x0002001020820040ULL, 0x0100090010010022ULL,
    0x2040040008008080ULL, 0x4005020004008080ULL, 0x010A018810040002ULL, 0x048A404400820001ULL,
    0x820170800D410100ULL, 0x0800400110802100ULL, 0x0242120420824200ULL, 0x84120014C0210A00ULL,
    0x1008080004008080ULL, 0x0222000410888200ULL, 0x0004420850010400ULL, 0x000000884C090200ULL,
    0x0088820100102042ULL, 0x4400120106402182ULL, 0x081B044028200131ULL, 0x08C501602810000DULL,
    0x080200204C100816ULL, 0x1001000400020801ULL, 0x0201209002610804ULL, 0x0000008100440022ULL,
};

static uint64_t g_occs[4096], g_refs[4096];
static int      g_epoch[4096];
static int      g_gen;

/* Fills att[] for one square; returns 0 as soon as two occupancies that need
 * different attack sets collide on the same index. */
static int try_magic(uint64_t *att, int size, int bits, uint64_t magic)
{
    g_gen++;
    for (int i = 0; i < size; i++) {
        unsigned idx = (unsigned)((g_occs[i] * magic) >> (64 - bits));
        if (g_epoch[idx] < g_gen) {
            g_epoch[idx] = g_gen;
            att[idx] = g_refs[i];
        } else if (att[idx] != g_refs[i]) {
            return 0;
        }
    }
    return 1;
}

static void init_magics(Magic *tab, uint64_t *table, const int d[4][2],
                        const uint64_t *seed, size_t total)
{
    memset(g_epoch, 0, sizeof(g_epoch));
    g_gen = 0;

    size_t offset = 0;
    for (int sq = 0; sq < 64; sq++) {
        uint64_t mask = slide_mask(sq, d);
        int bits = bb_count(mask);
        int size = 1 << bits;
        uint64_t *att = table + offset;

        uint64_t b = 0;
        int i = 0;
        do {
            g_occs[i] = b;
            g_refs[i] = slide_attacks(sq, b, d);
            i++;
            b = (b - mask) & mask;
        } while (b);

        uint64_t magic = seed[sq];
        if (!try_magic(att, size, bits, magic)) {
            for (;;) {
                do {
                    magic = mrand_sparse();
                } while (bb_count((mask * magic) >> 56) < 6);
                if (try_magic(att, size, bits, magic)) break;
            }
        }

        tab[sq].attacks = att;
        tab[sq].mask    = mask;
        tab[sq].magic   = magic;
        tab[sq].shift   = (unsigned)(64 - bits);
        offset += (size_t)size;
    }
    (void)total;   /* offset == total by construction */
}

static uint64_t splitmix64(uint64_t *s)
{
    uint64_t z = (*s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static int g_inited = 0;

void chess_init(void)
{
    if (g_inited) return;

    /* leaper attacks */
    static const int NJ[8][2] = { {1,2},{2,1},{2,-1},{1,-2},{-1,-2},{-2,-1},{-2,1},{-1,2} };
    static const int KJ[8][2] = { {1,0},{1,1},{0,1},{-1,1},{-1,0},{-1,-1},{0,-1},{1,-1} };
    for (int sq = 0; sq < 64; sq++) {
        int f = sq_file(sq), r = sq_rank(sq);
        uint64_t n = 0, k = 0;
        for (int i = 0; i < 8; i++) {
            int nf = f + NJ[i][0], nr = r + NJ[i][1];
            if (nf >= 0 && nf < 8 && nr >= 0 && nr < 8) n |= BIT(nr * 8 + nf);
            nf = f + KJ[i][0]; nr = r + KJ[i][1];
            if (nf >= 0 && nf < 8 && nr >= 0 && nr < 8) k |= BIT(nr * 8 + nf);
        }
        KNIGHT_ATT[sq] = n;
        KING_ATT[sq]   = k;

        uint64_t wp = 0, bp = 0;
        if (f > 0 && r < 7) wp |= BIT(sq + 7);
        if (f < 7 && r < 7) wp |= BIT(sq + 9);
        if (f > 0 && r > 0) bp |= BIT(sq - 9);
        if (f < 7 && r > 0) bp |= BIT(sq - 7);
        PAWN_ATT[WHITE][sq] = wp;
        PAWN_ATT[BLACK][sq] = bp;
    }

    /* magics (deterministic: fixed seed) */
    g_rs = 0x0F1E2D3C4B5A6978ULL;
    init_magics(BMAGIC, BATTACK, BDELTA, BMAGIC_SEED, sizeof(BATTACK) / sizeof(uint64_t));
    init_magics(RMAGIC, RATTACK, RDELTA, RMAGIC_SEED, sizeof(RATTACK) / sizeof(uint64_t));

    /* between / line */
    for (int a = 0; a < 64; a++) {
        for (int b = 0; b < 64; b++) {
            BETWEEN[a][b] = 0;
            LINE[a][b]    = 0;
            if (a == b) continue;
            if (rook_att(a, 0) & BIT(b)) {
                BETWEEN[a][b] = rook_att(a, BIT(b)) & rook_att(b, BIT(a));
                LINE[a][b]    = (rook_att(a, 0) & rook_att(b, 0)) | BIT(a) | BIT(b);
            } else if (bishop_att(a, 0) & BIT(b)) {
                BETWEEN[a][b] = bishop_att(a, BIT(b)) & bishop_att(b, BIT(a));
                LINE[a][b]    = (bishop_att(a, 0) & bishop_att(b, 0)) | BIT(a) | BIT(b);
            }
        }
    }

    /* zobrist */
    uint64_t s = 0x1571E5B4A7C39D0FULL;
    for (int c = 0; c < NCOLORS; c++)
        for (int pt = 0; pt < NPIECES; pt++)
            for (int sq = 0; sq < 64; sq++)
                Z_PIECE[c][pt][sq] = splitmix64(&s);
    for (int i = 0; i < 16; i++) Z_CASTLE[i] = splitmix64(&s);
    for (int i = 0; i < 8; i++)  Z_EP[i]     = splitmix64(&s);
    Z_SIDE = splitmix64(&s);

    /* castling-right invalidation masks */
    for (int i = 0; i < 64; i++) CASTLE_MASK[i] = CR_ALL;
    CASTLE_MASK[0]  = CR_ALL & (uint8_t)~CR_WQ;              /* a1 rook */
    CASTLE_MASK[7]  = CR_ALL & (uint8_t)~CR_WK;              /* h1 rook */
    CASTLE_MASK[4]  = CR_ALL & (uint8_t)~(CR_WK | CR_WQ);    /* e1 king */
    CASTLE_MASK[56] = CR_ALL & (uint8_t)~CR_BQ;              /* a8 rook */
    CASTLE_MASK[63] = CR_ALL & (uint8_t)~CR_BK;              /* h8 rook */
    CASTLE_MASK[60] = CR_ALL & (uint8_t)~(CR_BK | CR_BQ);    /* e8 king */

    g_inited = 1;
}

/* ----------------------------------------------------------- attacks --- */

/* All pieces (both colours) attacking `sq` given occupancy `occ`. */
static inline uint64_t attackers_to(const Position *p, int sq, uint64_t occ)
{
    uint64_t bq = p->piece[WHITE][BISHOP] | p->piece[BLACK][BISHOP]
                | p->piece[WHITE][QUEEN]  | p->piece[BLACK][QUEEN];
    uint64_t rq = p->piece[WHITE][ROOK]   | p->piece[BLACK][ROOK]
                | p->piece[WHITE][QUEEN]  | p->piece[BLACK][QUEEN];
    return (PAWN_ATT[WHITE][sq] & p->piece[BLACK][PAWN])
         | (PAWN_ATT[BLACK][sq] & p->piece[WHITE][PAWN])
         | (KNIGHT_ATT[sq] & (p->piece[WHITE][KNIGHT] | p->piece[BLACK][KNIGHT]))
         | (KING_ATT[sq]   & (p->piece[WHITE][KING]   | p->piece[BLACK][KING]))
         | (bishop_att(sq, occ) & bq)
         | (rook_att(sq, occ)   & rq);
}

static inline int attacked_by(const Position *p, int sq, int by, uint64_t occ)
{
    const uint64_t *e = p->piece[by];
    if (PAWN_ATT[by ^ 1][sq] & e[PAWN])   return 1;
    if (KNIGHT_ATT[sq]       & e[KNIGHT]) return 1;
    if (KING_ATT[sq]         & e[KING])   return 1;
    if (bishop_att(sq, occ) & (e[BISHOP] | e[QUEEN])) return 1;
    if (rook_att(sq, occ)   & (e[ROOK]   | e[QUEEN])) return 1;
    return 0;
}

int square_attacked(const Position *p, int sq, int by)
{
    return attacked_by(p, sq, by, p->all);
}

int in_check(const Position *p, int c)
{
    uint64_t k = p->piece[c][KING];
    if (!k) return 0;
    return attacked_by(p, bb_lsb(k), c ^ 1, p->all);
}

/* -------------------------------------------------- move generation ---- */

static inline int add_promos(Move *out, int n, int from, int to, int cap, int caps_only)
{
    const int base = cap ? MF_PROMO_NC : MF_PROMO_N;
    out[n++] = MV_MAKE(from, to, base + 3);       /* queen */
    if (!caps_only) {
        out[n++] = MV_MAKE(from, to, base + 0);   /* knight */
        out[n++] = MV_MAKE(from, to, base + 2);   /* rook   */
        out[n++] = MV_MAKE(from, to, base + 1);   /* bishop */
    }
    return n;
}

static inline int add_promos_or_quiet(Move *out, int n, int from, int to,
                                      int cap, int promoting, int caps_only)
{
    if (promoting) return add_promos(out, n, from, to, cap, caps_only);
    out[n++] = MV_MAKE(from, to, cap ? MF_CAPTURE : MF_QUIET);
    return n;
}

static inline uint64_t sh_fwd(uint64_t b, int us)
{
    return us == WHITE ? (b << 8) : (b >> 8);
}
static inline uint64_t sh_capl(uint64_t b, int us)   /* toward file-1 */
{
    return us == WHITE ? ((b & ~FILE_A) << 7) : ((b & ~FILE_A) >> 9);
}
static inline uint64_t sh_capr(uint64_t b, int us)   /* toward file+1 */
{
    return us == WHITE ? ((b & ~FILE_H) << 9) : ((b & ~FILE_H) >> 7);
}

/* ---- Chess960 castling ----------------------------------------------------
 *
 * The destinations are the classical ones (king g1/c1, rook f1/d1) but the two
 * pieces may start anywhere on the back rank, so the king can move left, right
 * or not at all, and the rook may jump over the king's square.  Conditions:
 *
 *   1. the right is still held and its rook really stands on its origin file;
 *   2. every square of the king's path and of the rook's path (destinations
 *      included) is empty apart from the castling king and the castling rook
 *      themselves -- a piece on a square only the ROOK crosses blocks too;
 *   3. no square the king passes over, including its origin and destination,
 *      is attacked -- testing the origin here is what makes "you may not castle
 *      out of check" hold, so this function is self-contained and the caller
 *      does not have to run its own in-check test.  As everywhere else,
 *      "attacked" is judged in the position BEFORE the move, which is the
 *      standard reading of the rule and what published 960 perft numbers
 *      assume;
 *   4. the move must not leave our own king in check.  This is NOT implied by
 *      (3) in Chess960: vacating the rook's origin can open a rank-1 line onto
 *      the king's destination (enemy rook a1, our castling rook b1, king c1),
 *      so the king's final square is re-tested against the final occupancy.
 *
 * `occ` is p->all.
 */
static COLD_HELPER int gen_castle_960(const Position *p, Move *out, int n, int us, int ksq, uint64_t occ)
{
    const int them = us ^ 1;
    /* A FEN may claim rights whose king has long since left home; the
     * classical generator rejects that by testing board[e1], and this is the
     * same guard.  Without it BETWEEN[ksq][kto] would be a ray that has
     * nothing to do with the back rank. */
    if (sq_rank(ksq) != sq_rank(back_rank(us))) return n;

    for (int side = 0; side < 2; side++) {
        if (!(p->castling & CR_BIT[us][side])) continue;

        const int rfrom = crook_sq(p, us, side);
        if (p->board[rfrom] != ROOK || p->color_at[rfrom] != us) continue;
        const int kto = castle_kto(us, side);
        const int rto = castle_rto(us, side);

        /* (2) both paths clear of everything except the king and that rook */
        const uint64_t path = BETWEEN[ksq][kto] | BIT(kto) | BETWEEN[rfrom][rto] | BIT(rto);
        if ((occ ^ BIT(ksq) ^ BIT(rfrom)) & path) continue;

        /* (3) the king's walk, origin and destination included, must be
         * attack-free -- BIT(ksq) is the "not out of check" half of the rule */
        uint64_t walk = BETWEEN[ksq][kto] | BIT(kto) | BIT(ksq);
        int blocked = 0;
        while (walk) {
            const int sq = bb_pop(&walk);
            if (attacked_by(p, sq, them, occ)) { blocked = 1; break; }
        }
        if (blocked) continue;

        /* (4) the resulting position must be legal */
        const uint64_t occ_after = occ ^ BIT(ksq) ^ BIT(kto) ^ BIT(rfrom) ^ BIT(rto);
        if (attacked_by(p, kto, them, occ_after)) continue;

        out[n++] = MV_MAKE(ksq, rfrom, side == 0 ? MF_KCASTLE : MF_QCASTLE);
    }
    return n;
}

static inline int gen_moves(const Position *p, Move *out, const int caps_only)
{
    int n = 0;
    const int us = p->side, them = us ^ 1;
    const uint64_t ourP = p->occ[us], theirP = p->occ[them];
    const uint64_t occ = p->all, empty = ~occ;

    if (!p->piece[us][KING]) return 0;
    const int ksq = bb_lsb(p->piece[us][KING]);
    const uint64_t *e = p->piece[them];

    /* ---- checkers ---- */
    const uint64_t checkers =
          (PAWN_ATT[us][ksq]  & e[PAWN])
        | (KNIGHT_ATT[ksq]    & e[KNIGHT])
        | (bishop_att(ksq, occ) & (e[BISHOP] | e[QUEEN]))
        | (rook_att(ksq, occ)   & (e[ROOK]   | e[QUEEN]));
    const int nchk = bb_count(checkers);

    /* ---- king moves (always legal-tested with the king removed) ---- */
    {
        const uint64_t occ_nk = occ ^ BIT(ksq);
        uint64_t t = KING_ATT[ksq] & ~ourP;
        if (caps_only) t &= theirP;
        while (t) {
            int to = bb_pop(&t);
            if (!attacked_by(p, to, them, occ_nk))
                out[n++] = MV_MAKE(ksq, to, (BIT(to) & theirP) ? MF_CAPTURE : MF_QUIET);
        }
    }
    if (nchk > 1) return n;              /* double check: only the king may move */

    /* ---- destination mask for every non-king piece ---- */
    uint64_t target;
    if (nchk == 0) {
        target = ~ourP;
    } else {
        int cs = bb_lsb(checkers);
        target = BETWEEN[ksq][cs] | BIT(cs);
    }
    const uint64_t movemask = caps_only ? (target & theirP) : target;

    /* ---- absolute pins ---- */
    uint64_t pinned = 0;
    {
        uint64_t snipers = (rook_att(ksq, 0)   & (e[ROOK]   | e[QUEEN]))
                         | (bishop_att(ksq, 0) & (e[BISHOP] | e[QUEEN]));
        while (snipers) {
            int s = bb_pop(&snipers);
            uint64_t btw = BETWEEN[ksq][s] & occ;
            if (btw && !(btw & (btw - 1))) pinned |= btw & ourP;
        }
    }

    /* ---- pawns ---- */
    const uint64_t pawns = p->piece[us][PAWN];
    const int up = (us == WHITE) ? 8 : -8;
    const uint64_t r_home = (us == WHITE) ? RANK_2 : RANK_7;
    const uint64_t r_dbl  = (us == WHITE) ? RANK_3 : RANK_6;
    const uint64_t r_pro  = (us == WHITE) ? RANK_7 : RANK_2;

    {
        const uint64_t np      = pawns & ~pinned;
        const uint64_t prom    = np & r_pro;
        const uint64_t nonprom = np & ~r_pro;

        if (!caps_only) {
            uint64_t s1 = sh_fwd(nonprom, us) & empty;
            uint64_t s2 = sh_fwd(s1 & r_dbl, us) & empty & target;
            s1 &= target;
            while (s1) { int t = bb_pop(&s1); out[n++] = MV_MAKE(t - up, t, MF_QUIET); }
            while (s2) { int t = bb_pop(&s2); out[n++] = MV_MAKE(t - 2 * up, t, MF_DOUBLE); }
        }
        {
            uint64_t cl = sh_capl(nonprom, us) & theirP & target;
            uint64_t cr = sh_capr(nonprom, us) & theirP & target;
            while (cl) { int t = bb_pop(&cl); out[n++] = MV_MAKE(t - up + 1, t, MF_CAPTURE); }
            while (cr) { int t = bb_pop(&cr); out[n++] = MV_MAKE(t - up - 1, t, MF_CAPTURE); }
        }
        if (prom) {
            uint64_t s1 = sh_fwd(prom, us)   & empty  & target;
            uint64_t cl = sh_capl(prom, us)  & theirP & target;
            uint64_t cr = sh_capr(prom, us)  & theirP & target;
            while (s1) { int t = bb_pop(&s1); n = add_promos(out, n, t - up,     t, 0, caps_only); }
            while (cl) { int t = bb_pop(&cl); n = add_promos(out, n, t - up + 1, t, 1, caps_only); }
            while (cr) { int t = bb_pop(&cr); n = add_promos(out, n, t - up - 1, t, 1, caps_only); }
        }

        /* pinned pawns: rare, handled one at a time along the pin ray */
        uint64_t pp = pawns & pinned;
        while (pp) {
            int from = bb_pop(&pp);
            const uint64_t line = LINE[ksq][from];
            const int f = sq_file(from);
            const int t1 = from + up;
            const int promoting = (BIT(from) & r_pro) != 0;

            if (!(occ & BIT(t1))) {
                if ((BIT(t1) & line & target) && (!caps_only || promoting))
                    n = add_promos_or_quiet(out, n, from, t1, 0, promoting, caps_only);
                if (!caps_only && (BIT(from) & r_home)) {
                    int t2 = from + 2 * up;
                    if (!(occ & BIT(t2)) && (BIT(t2) & line & target))
                        out[n++] = MV_MAKE(from, t2, MF_DOUBLE);
                }
            }
            for (int d = -1; d <= 1; d += 2) {
                if (f + d < 0 || f + d > 7) continue;
                int t = from + up + d;
                if (BIT(t) & theirP & line & target)
                    n = add_promos_or_quiet(out, n, from, t, 1, promoting, caps_only);
            }
        }
    }

    /* ---- en passant (full legality test: handles the horizontal pin) ---- */
    if (p->ep >= 0) {
        const int to    = p->ep;
        const int capsq = to - up;
        uint64_t cands = PAWN_ATT[them][to] & pawns;
        while (cands) {
            int from = bb_pop(&cands);
            uint64_t occ2  = (occ ^ BIT(from) ^ BIT(capsq)) | BIT(to);
            uint64_t enemy = theirP ^ BIT(capsq);
            if (!(attackers_to(p, ksq, occ2) & enemy))
                out[n++] = MV_MAKE(from, to, MF_EP);
        }
    }

    /* ---- knights (a pinned knight can never move) ---- */
    {
        uint64_t b = p->piece[us][KNIGHT] & ~pinned;
        while (b) {
            int from = bb_pop(&b);
            uint64_t t = KNIGHT_ATT[from] & movemask;
            while (t) {
                int to = bb_pop(&t);
                out[n++] = MV_MAKE(from, to, (BIT(to) & theirP) ? MF_CAPTURE : MF_QUIET);
            }
        }
    }

    /* ---- bishops + queens ---- */
    {
        uint64_t b = p->piece[us][BISHOP] | p->piece[us][QUEEN];
        while (b) {
            int from = bb_pop(&b);
            uint64_t t = bishop_att(from, occ) & movemask;
            if (BIT(from) & pinned) t &= LINE[ksq][from];
            while (t) {
                int to = bb_pop(&t);
                out[n++] = MV_MAKE(from, to, (BIT(to) & theirP) ? MF_CAPTURE : MF_QUIET);
            }
        }
    }

    /* ---- rooks + queens ---- */
    {
        uint64_t b = p->piece[us][ROOK] | p->piece[us][QUEEN];
        while (b) {
            int from = bb_pop(&b);
            uint64_t t = rook_att(from, occ) & movemask;
            if (BIT(from) & pinned) t &= LINE[ksq][from];
            while (t) {
                int to = bb_pop(&t);
                out[n++] = MV_MAKE(from, to, (BIT(to) & theirP) ? MF_CAPTURE : MF_QUIET);
            }
        }
    }

    /* ---- castling ---- */
    /* p->chess960 is read here rather than passed in, so that the shared body
     * keeps nothing extra alive across it; the 960 castling moves are appended
     * by gen_legal_960() instead. */
    if (!caps_only && nchk == 0 && !p->chess960) {
        if (us == WHITE) {
            if ((p->castling & CR_WK)
                && p->board[4] == KING && p->color_at[4] == WHITE
                && p->board[7] == ROOK && p->color_at[7] == WHITE
                && !(occ & (BIT(5) | BIT(6)))
                && !attacked_by(p, 5, BLACK, occ) && !attacked_by(p, 6, BLACK, occ))
                out[n++] = MV_MAKE(4, 6, MF_KCASTLE);
            if ((p->castling & CR_WQ)
                && p->board[4] == KING && p->color_at[4] == WHITE
                && p->board[0] == ROOK && p->color_at[0] == WHITE
                && !(occ & (BIT(1) | BIT(2) | BIT(3)))
                && !attacked_by(p, 3, BLACK, occ) && !attacked_by(p, 2, BLACK, occ))
                out[n++] = MV_MAKE(4, 2, MF_QCASTLE);
        } else {
            if ((p->castling & CR_BK)
                && p->board[60] == KING && p->color_at[60] == BLACK
                && p->board[63] == ROOK && p->color_at[63] == BLACK
                && !(occ & (BIT(61) | BIT(62)))
                && !attacked_by(p, 61, WHITE, occ) && !attacked_by(p, 62, WHITE, occ))
                out[n++] = MV_MAKE(60, 62, MF_KCASTLE);
            if ((p->castling & CR_BQ)
                && p->board[60] == KING && p->color_at[60] == BLACK
                && p->board[56] == ROOK && p->color_at[56] == BLACK
                && !(occ & (BIT(57) | BIT(58) | BIT(59)))
                && !attacked_by(p, 59, WHITE, occ) && !attacked_by(p, 58, WHITE, occ))
                out[n++] = MV_MAKE(60, 58, MF_QCASTLE);
        }
    }

    return n;
}

/* Chess960 castling is appended here rather than from inside gen_moves(),
 * whose body is shared with the classical generator: a call in the middle of
 * it would cost every classical node the spills around that call. */
static COLD_HELPER int gen_legal_960(const Position *p, Move *out)
{
    int n = gen_moves(p, out, 0);
    const int us = p->side;
    if ((p->castling & (us == WHITE ? (CR_WK | CR_WQ) : (CR_BK | CR_BQ)))
        && p->piece[us][KING])
        n = gen_castle_960(p, out, n, us, bb_lsb(p->piece[us][KING]), p->all);
    return n;
}

int gen_legal(const Position *p, Move *out)
{
    if (p->chess960) return gen_legal_960(p, out);   /* tail call, classical pays a test */
    return gen_moves(p, out, 0);
}
/* Castling is never a capture, so the capture generator has no 960 variant. */
int gen_legal_captures(const Position *p, Move *out) { return gen_moves(p, out, 1); }

/* ------------------------------------------------------- make / unmake - */

/* Castling-right invalidation for Chess960, where the rooks are not on a1/h1.
 * A right dies when the king moves (whatever the king's origin file is), when
 * its rook leaves its origin square, or when anything lands on that square --
 * the latter covers the rook being captured at home.  Rights are only ever
 * cleared, never set, so the conservative "something moved off / onto the rook
 * square" test can never resurrect a dead right. */
static COLD_HELPER uint8_t castle_mask_960(const Position *p, int pc, int us, int from, int to)
{
    uint8_t mask = CR_ALL;
    if (pc == KING) mask &= (uint8_t)~(CR_BIT[us][0] | CR_BIT[us][1]);
    for (int c = 0; c < NCOLORS; c++) {
        for (int side = 0; side < 2; side++) {
            const uint8_t bit = CR_BIT[c][side];
            if (!(p->castling & bit)) continue;
            const int rsq = crook_sq(p, c, side);
            if (from == rsq || to == rsq) mask &= (uint8_t)~bit;
        }
    }
    return mask;
}

/* The four squares a castling move touches.  The rook's origin comes from
 * p->crook, which is {h,a} for every classical position, so this is the same
 * code for both variants: playing a castling move never has to look at
 * p->chess960, and the move's `to` field (the rook's square in Chess960, the
 * king's destination in classical chess) is pure move IDENTITY -- it never
 * feeds the board update. */
static inline void castle_squares(const Position *p, int fl, int us,
                                  int *rfrom, int *kto, int *rto)
{
    const int side = (fl == MF_KCASTLE) ? 0 : 1;
    *rfrom = crook_sq(p, us, side);
    *kto   = castle_kto(us, side);
    *rto   = castle_rto(us, side);
}

/* Castling relocates two pieces, and in Chess960 either of them may end where
 * it already stands (king on g1, rook on f1) or on the other's origin square.
 * XOR-updating the bitboards and clearing both origins before writing both
 * destinations handles every overlap, including from == kto (a null king
 * move).  `k` is the running zobrist with the old ep/castling terms already
 * removed; this finishes the whole move. */
static inline void make_castle(Position *p, int fl, int us, int from, uint64_t k)
{
    int rfrom, kto, rto;
    castle_squares(p, fl, us, &rfrom, &kto, &rto);

    p->piece[us][KING] ^= BIT(from) ^ BIT(kto);
    p->piece[us][ROOK] ^= BIT(rfrom) ^ BIT(rto);
    p->occ[us]         ^= BIT(from) ^ BIT(kto) ^ BIT(rfrom) ^ BIT(rto);
    k ^= Z_PIECE[us][KING][from]  ^ Z_PIECE[us][KING][kto]
       ^ Z_PIECE[us][ROOK][rfrom] ^ Z_PIECE[us][ROOK][rto];

    p->board[from]     = NO_PIECE;
    p->color_at[from]  = -1;
    p->board[rfrom]    = NO_PIECE;
    p->color_at[rfrom] = -1;
    p->board[kto]      = KING;
    p->color_at[kto]   = (int8_t)us;
    p->board[rto]      = ROOK;
    p->color_at[rto]   = (int8_t)us;

    /* Castling always burns both of the mover's rights and can never touch the
     * opponent's (every square involved is on our own back rank). */
    p->castling &= (uint8_t)~(CR_BIT[us][0] | CR_BIT[us][1]);
    k ^= Z_CASTLE[p->castling];
    p->ep = -1;
    p->halfmove++;
    if (us == BLACK) p->fullmove++;
    p->side = (uint8_t)(us ^ 1);
    k ^= Z_SIDE;

    p->all = p->occ[WHITE] | p->occ[BLACK];
    p->key = k;
}

static inline void unmake_castle(Position *p, int fl, int us, int from)
{
    int rfrom, kto, rto;
    castle_squares(p, fl, us, &rfrom, &kto, &rto);

    p->piece[us][KING] ^= BIT(from) ^ BIT(kto);
    p->piece[us][ROOK] ^= BIT(rfrom) ^ BIT(rto);
    p->occ[us]         ^= BIT(from) ^ BIT(kto) ^ BIT(rfrom) ^ BIT(rto);

    p->board[kto]      = NO_PIECE;
    p->color_at[kto]   = -1;
    p->board[rto]      = NO_PIECE;
    p->color_at[rto]   = -1;
    p->board[from]     = KING;
    p->color_at[from]  = (int8_t)us;
    p->board[rfrom]    = ROOK;
    p->color_at[rfrom] = (int8_t)us;

    p->all = p->occ[WHITE] | p->occ[BLACK];
}

/* One body, two instantiations.  `c960` is a compile-time constant, so the
 * classical instantiation is the pre-960 make_move(): no extra test, no call
 * and no register pressure from code it never executes. */
static HOT_INLINE void make_move_impl(Position *p, Move m, Undo *u, const int c960)
{
    const int from = MV_FROM(m), to = MV_TO(m), fl = MV_FLAG(m);
    const int us = p->side, them = us ^ 1;
    const uint64_t bfrom = BIT(from), bto = BIT(to);

    u->key      = p->key;
    u->ep       = p->ep;
    u->castling = p->castling;
    u->halfmove = p->halfmove;
    u->fullmove = p->fullmove;
    u->captured = NO_PIECE;

    uint64_t k = p->key;
    if (p->ep >= 0) k ^= Z_EP[sq_file(p->ep)];
    k ^= Z_CASTLE[p->castling];

    const int pc = p->board[from];

    if (fl == MF_KCASTLE || fl == MF_QCASTLE) { make_castle(p, fl, us, from, k); return; }

    /* --- remove the captured piece --- */
    if (fl == MF_EP) {
        const int capsq = to - ((us == WHITE) ? 8 : -8);
        u->captured = PAWN;
        p->piece[them][PAWN] ^= BIT(capsq);
        p->occ[them]         ^= BIT(capsq);
        p->board[capsq]       = NO_PIECE;
        p->color_at[capsq]    = -1;
        k ^= Z_PIECE[them][PAWN][capsq];
    } else if (fl & 4) {
        const int cap = p->board[to];
        u->captured = (int8_t)cap;
        p->piece[them][cap] ^= bto;
        p->occ[them]        ^= bto;
        k ^= Z_PIECE[them][cap][to];
    }

    /* --- move the piece (with promotion) --- */
    const int newpc = (fl >= MF_PROMO_N) ? MV_PROMO_PIECE(m) : pc;
    p->piece[us][pc]    ^= bfrom;
    p->piece[us][newpc] ^= bto;
    p->occ[us]          ^= bfrom | bto;
    k ^= Z_PIECE[us][pc][from] ^ Z_PIECE[us][newpc][to];
    p->board[from]    = NO_PIECE;
    p->color_at[from] = -1;
    p->board[to]      = (int8_t)newpc;
    p->color_at[to]   = (int8_t)us;

    /* --- rights, ep, clocks, side --- */
    if (c960) {
        /* A right can only die to a king move or to something touching a back
         * rank, and most moves are neither -- worth a test to skip the call. */
        if (p->castling && (pc == KING || ((bfrom | bto) & (RANK_1 | RANK_8))))
            p->castling &= castle_mask_960(p, pc, us, from, to);
    } else {
        p->castling &= CASTLE_MASK[from] & CASTLE_MASK[to];
    }
    k ^= Z_CASTLE[p->castling];

    if (fl == MF_DOUBLE) {
        p->ep = (int8_t)((from + to) / 2);
        k ^= Z_EP[sq_file(p->ep)];
    } else {
        p->ep = -1;
    }

    if (pc == PAWN || (fl & 4)) p->halfmove = 0;
    else                        p->halfmove++;
    if (us == BLACK) p->fullmove++;

    p->side = (uint8_t)them;
    k ^= Z_SIDE;

    p->all = p->occ[WHITE] | p->occ[BLACK];
    p->key = k;
}

void unmake_move(Position *p, Move m, const Undo *u)
{
    const int from = MV_FROM(m), to = MV_TO(m), fl = MV_FLAG(m);
    const int us = p->side ^ 1, them = p->side;
    const uint64_t bfrom = BIT(from), bto = BIT(to);

    p->side     = (uint8_t)us;
    p->key      = u->key;
    p->ep       = u->ep;
    p->castling = u->castling;
    p->halfmove = u->halfmove;
    p->fullmove = u->fullmove;

    if (fl == MF_KCASTLE || fl == MF_QCASTLE) { unmake_castle(p, fl, us, from); return; }

    const int newpc = p->board[to];
    const int pc    = (fl >= MF_PROMO_N) ? PAWN : newpc;

    p->piece[us][newpc] ^= bto;
    p->piece[us][pc]    ^= bfrom;
    p->occ[us]          ^= bfrom | bto;
    p->board[from]    = (int8_t)pc;
    p->color_at[from] = (int8_t)us;
    p->board[to]      = NO_PIECE;
    p->color_at[to]   = -1;

    if (fl == MF_EP) {
        const int capsq = to - ((us == WHITE) ? 8 : -8);
        p->piece[them][PAWN] ^= BIT(capsq);
        p->occ[them]         ^= BIT(capsq);
        p->board[capsq]       = PAWN;
        p->color_at[capsq]    = (int8_t)them;
    } else if (fl & 4) {
        const int cap = u->captured;
        p->piece[them][cap] ^= bto;
        p->occ[them]        ^= bto;
        p->board[to]        = (int8_t)cap;
        p->color_at[to]     = (int8_t)them;
    }

    p->all = p->occ[WHITE] | p->occ[BLACK];
}

/* unmake_move() needs no variant at all (castle_squares() reads p->crook), and
 * make_move()'s only Chess960-specific step is the castling-right mask, so the
 * classical body is inlined right here and only the 960 body is called.  These
 * two run once per node: routing the classical path through a call of its own
 * measures ~9% slower, and a shared body with a runtime flag ~3%. */
static COLD_HELPER void make_move_960(Position *p, Move m, Undo *u)
{
    make_move_impl(p, m, u, 1);
}

void make_move(Position *p, Move m, Undo *u)
{
    if (p->chess960) { make_move_960(p, m, u); return; }
    make_move_impl(p, m, u, 0);
}

void make_null(Position *p, Undo *u)
{
    u->key      = p->key;
    u->ep       = p->ep;
    u->castling = p->castling;
    u->halfmove = p->halfmove;
    u->fullmove = p->fullmove;
    u->captured = NO_PIECE;

    uint64_t k = p->key;
    if (p->ep >= 0) k ^= Z_EP[sq_file(p->ep)];
    p->ep = -1;
    if (p->side == BLACK) p->fullmove++;
    p->halfmove++;
    p->side ^= 1;
    k ^= Z_SIDE;
    p->key = k;
}

void unmake_null(Position *p, const Undo *u)
{
    p->side    ^= 1;
    p->key      = u->key;
    p->ep       = u->ep;
    p->castling = u->castling;
    p->halfmove = u->halfmove;
    p->fullmove = u->fullmove;
}

/* --------------------------------------------------------- zobrist ----- */

uint64_t pos_compute_key(const Position *p)
{
    uint64_t k = 0;
    for (int c = 0; c < NCOLORS; c++) {
        for (int pt = 0; pt < NPIECES; pt++) {
            uint64_t b = p->piece[c][pt];
            while (b) {
                int sq = bb_pop(&b);
                k ^= Z_PIECE[c][pt][sq];
            }
        }
    }
    k ^= Z_CASTLE[p->castling];
    if (p->ep >= 0)        k ^= Z_EP[sq_file(p->ep)];
    if (p->side == BLACK)  k ^= Z_SIDE;
    return k;
}

/* -------------------------------------------------------------- FEN ---- */

static void pos_clear(Position *p)
{
    memset(p, 0, sizeof(*p));
    for (int i = 0; i < 64; i++) { p->board[i] = NO_PIECE; p->color_at[i] = -1; }
    p->ep = -1;
    p->fullmove = 1;
    /* Classical rook files, so that a Position never carries a nonsense crook
     * even when it holds no castling rights at all. */
    p->crook[WHITE][0] = p->crook[BLACK][0] = 7;
    p->crook[WHITE][1] = p->crook[BLACK][1] = 0;
}

static void pos_finish(Position *p)
{
    p->occ[WHITE] = p->occ[BLACK] = 0;
    for (int pt = 0; pt < NPIECES; pt++) {
        p->occ[WHITE] |= p->piece[WHITE][pt];
        p->occ[BLACK] |= p->piece[BLACK][pt];
    }
    p->all = p->occ[WHITE] | p->occ[BLACK];
    p->key = pos_compute_key(p);
}

int pos_from_fen(Position *p, const char *fen)
{
    if (!fen) return 0;
    pos_clear(p);

    const char *s = fen;
    while (*s == ' ') s++;

    int rank = 7, file = 0;
    for (;;) {
        char c = *s;
        if (c == '\0' || c == ' ') break;
        if (c == '/') {
            if (file != 8) return 0;
            rank--; file = 0;
            if (rank < 0) return 0;
            s++;
            continue;
        }
        if (c >= '1' && c <= '8') {
            file += c - '0';
            if (file > 8) return 0;
            s++;
            continue;
        }
        int color, pt;
        switch (c) {
            case 'P': color = WHITE; pt = PAWN;   break;
            case 'N': color = WHITE; pt = KNIGHT; break;
            case 'B': color = WHITE; pt = BISHOP; break;
            case 'R': color = WHITE; pt = ROOK;   break;
            case 'Q': color = WHITE; pt = QUEEN;  break;
            case 'K': color = WHITE; pt = KING;   break;
            case 'p': color = BLACK; pt = PAWN;   break;
            case 'n': color = BLACK; pt = KNIGHT; break;
            case 'b': color = BLACK; pt = BISHOP; break;
            case 'r': color = BLACK; pt = ROOK;   break;
            case 'q': color = BLACK; pt = QUEEN;  break;
            case 'k': color = BLACK; pt = KING;   break;
            default: return 0;
        }
        if (file > 7) return 0;
        int sq = rank * 8 + file;
        p->piece[color][pt] |= BIT(sq);
        p->board[sq]    = (int8_t)pt;
        p->color_at[sq] = (int8_t)color;
        file++;
        s++;
    }
    if (rank != 0 || file != 8) return 0;
    /* Pawns can never stand on the back ranks; rejecting them here keeps the
     * move generator's "pawn + forward step is on the board" assumption true. */
    if ((p->piece[WHITE][PAWN] | p->piece[BLACK][PAWN]) & (RANK_1 | RANK_8)) return 0;

    while (*s == ' ') s++;
    if (*s == 'w')      p->side = WHITE;
    else if (*s == 'b') p->side = BLACK;
    else return 0;
    s++;

    /* ---- castling rights -------------------------------------------------
     * Three notations are accepted:
     *   KQkq   classical, and the X-FEN reading of it for a Chess960 board:
     *          K/Q mean "the OUTERMOST own rook on that side of the king";
     *   HAha   Shredder-FEN, the rook's origin file (upper case = White);
     *   any mixture of the two (the X-FEN hybrid).
     * The position counts as Chess960 when a file letter was used, or when the
     * X-FEN resolution does not land on the classical e1/a1/h1 geometry.  A
     * right whose rook is missing entirely is kept with the classical default
     * file, which is what the pre-960 parser did and what the generator (which
     * re-checks that the rook is really there) safely ignores. */
    while (*s == ' ') s++;
    p->castling = 0;
    if (*s == '-') {
        s++;
    } else {
        int kfile[NCOLORS];
        for (int c = 0; c < NCOLORS; c++)
            kfile[c] = p->piece[c][KING] ? sq_file(bb_lsb(p->piece[c][KING])) : -1;

        int given[NCOLORS][2] = { { -1, -1 }, { -1, -1 } };   /* explicit files */
        int seen = 0, is960 = 0;

        while (*s && *s != ' ') {
            const char ch = *s;
            int c, side, f = -1;
            if      (ch == 'K') { c = WHITE; side = 0; }
            else if (ch == 'Q') { c = WHITE; side = 1; }
            else if (ch == 'k') { c = BLACK; side = 0; }
            else if (ch == 'q') { c = BLACK; side = 1; }
            else if (ch >= 'A' && ch <= 'H') { c = WHITE; f = ch - 'A'; side = 0; }
            else if (ch >= 'a' && ch <= 'h') { c = BLACK; f = ch - 'a'; side = 0; }
            else return 0;
            if (f >= 0) {
                /* Which side of the king the rook stands on decides the right. */
                if (kfile[c] < 0 || f == kfile[c]) return 0;
                side = (f > kfile[c]) ? 0 : 1;
                given[c][side] = f;
                is960 = 1;
            }
            p->castling |= CR_BIT[c][side];
            seen++;
            if (seen > 4) return 0;
            s++;
        }
        if (!seen) return 0;

        for (int c = 0; c < NCOLORS; c++) {
            const int base = (c == WHITE) ? 0 : 56;
            for (int side = 0; side < 2; side++) {
                if (!(p->castling & CR_BIT[c][side])) continue;
                if (given[c][side] >= 0) {
                    p->crook[c][side] = (uint8_t)given[c][side];
                    continue;
                }
                int rf = -1;                        /* X-FEN: outermost rook */
                if (kfile[c] >= 0 && side == 0) {
                    for (int f = 7; f > kfile[c]; f--)
                        if (p->board[base + f] == ROOK && p->color_at[base + f] == c) { rf = f; break; }
                } else if (kfile[c] >= 0) {
                    for (int f = 0; f < kfile[c]; f++)
                        if (p->board[base + f] == ROOK && p->color_at[base + f] == c) { rf = f; break; }
                }
                if (rf < 0) continue;               /* bogus right: keep the default */
                p->crook[c][side] = (uint8_t)rf;
                if (rf != (side == 0 ? 7 : 0) || kfile[c] != 4) is960 = 1;
            }
        }
        p->chess960 = (uint8_t)(is960 != 0);
    }

    while (*s == ' ') s++;
    if (*s == '-') {
        p->ep = -1;
        s++;
    } else if (*s >= 'a' && *s <= 'h' && s[1] >= '1' && s[1] <= '8') {
        p->ep = (int8_t)((s[1] - '1') * 8 + (s[0] - 'a'));
        s += 2;
    } else {
        return 0;
    }

    /* optional halfmove / fullmove */
    while (*s == ' ') s++;
    if (*s >= '0' && *s <= '9') {
        long v = 0;
        while (*s >= '0' && *s <= '9') { v = v * 10 + (*s - '0'); s++; if (v > 65535) v = 65535; }
        p->halfmove = (uint16_t)v;
        while (*s == ' ') s++;
        if (*s >= '0' && *s <= '9') {
            long f = 0;
            while (*s >= '0' && *s <= '9') { f = f * 10 + (*s - '0'); s++; if (f > 65535) f = 65535; }
            p->fullmove = (uint16_t)(f ? f : 1);
        } else {
            p->fullmove = 1;
        }
    } else {
        p->halfmove = 0;
        p->fullmove = 1;
    }

    pos_finish(p);
    return 1;
}

void pos_startpos(Position *p)
{
    pos_from_fen(p, "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
}

/* ---------------------------------------------------------- chess 960 --- */

/* Scharnagl's numbering, the scheme the FIDE/Shredder world uses:
 *
 *   n       -> (n % 4)  picks the light-square bishop out of b1,d1,f1,h1
 *   n /= 4  -> (n % 4)  picks the dark-square  bishop out of a1,c1,e1,g1
 *   n /= 4  -> (n % 6)  puts the queen on that many-th still-empty square
 *   n /= 6  -> 0..9     indexes the table below, whose N/R/K pattern is laid
 *                       out over the five squares that are still empty
 *
 * 518 comes out as RNBQKBNR, which is asserted in tests/test_960.c. */
static const char KRN_TABLE[10][6] = {
    "NNRKR", "NRNKR", "NRKNR", "NRKRN", "RNNKR",
    "RNKNR", "RNKRN", "RKNNR", "RKNRN", "RKRNN"
};

/* Fills back[0..7] with the piece letters (upper case) of White's back rank. */
static void scharnagl_back_rank(int id, char *back)
{
    int n = id % 960;
    if (n < 0) n += 960;

    for (int f = 0; f < 8; f++) back[f] = 0;
    back[8] = '\0';

    back[2 * (n % 4) + 1] = 'B';        /* light squares: files 1,3,5,7 */
    n /= 4;
    back[2 * (n % 4)]     = 'B';        /* dark squares:  files 0,2,4,6 */
    n /= 4;

    int q = n % 6;                      /* queen on the q-th free square */
    n /= 6;
    for (int f = 0; f < 8; f++) {
        if (back[f]) continue;
        if (q-- == 0) { back[f] = 'Q'; break; }
    }

    const char *pat = KRN_TABLE[n % 10];
    for (int f = 0, k = 0; f < 8; f++)
        if (!back[f]) back[f] = pat[k++];
}

void pos_startpos960(Position *p, int id)
{
    char back[9];
    scharnagl_back_rank(id, back);

    int kf = 0, rk = 0, rq = 0;
    for (int f = 0; f < 8; f++) if (back[f] == 'K') kf = f;
    for (int f = 0; f < 8; f++)
        if (back[f] == 'R') { if (f < kf) rq = f; else { rk = f; break; } }

    /* Built through the FEN parser, in Shredder notation, so that a 960 game
     * and a 960 FEN can never drift apart. */
    char fen[96];
    int i = 0;
    for (int f = 0; f < 8; f++) fen[i++] = (char)(back[f] - 'A' + 'a');
    i += snprintf(fen + i, sizeof(fen) - (size_t)i, "/pppppppp/8/8/8/8/PPPPPPPP/");
    for (int f = 0; f < 8; f++) fen[i++] = back[f];
    snprintf(fen + i, sizeof(fen) - (size_t)i, " w %c%c%c%c - 0 1",
             'A' + rk, 'A' + rq, 'a' + rk, 'a' + rq);

    pos_from_fen(p, fen);
}

int pos_960_id(const Position *p)
{
    /* White's back rank must be a legal 960 array and Black must mirror it. */
    char back[9];
    int count[NPIECES] = { 0 };
    for (int f = 0; f < 8; f++) {
        const int pt = p->board[f];
        if (pt == NO_PIECE || p->color_at[f] != WHITE) return -1;
        if (p->board[f + 56] != pt || p->color_at[f + 56] != BLACK) return -1;
        count[pt]++;
        back[f] = "PNBRQK"[pt];
    }
    back[8] = '\0';
    if (count[PAWN] || count[KNIGHT] != 2 || count[BISHOP] != 2
        || count[ROOK] != 2 || count[QUEEN] != 1 || count[KING] != 1) return -1;

    int lb = -1, db = -1;
    for (int f = 0; f < 8; f++)
        if (back[f] == 'B') { if (f & 1) lb = f; else db = f; }
    if (lb < 0 || db < 0) return -1;            /* both bishops same colour */

    int q = -1, free_idx = 0;
    for (int f = 0; f < 8; f++) {
        if (back[f] == 'B') continue;
        if (back[f] == 'Q') { q = free_idx; break; }
        free_idx++;
    }
    if (q < 0) return -1;

    char pat[6];
    int k = 0;
    for (int f = 0; f < 8 && k < 5; f++)
        if (back[f] != 'B' && back[f] != 'Q') pat[k++] = back[f];
    pat[5] = '\0';

    int krn = -1;
    for (int t = 0; t < 10; t++) if (!strcmp(pat, KRN_TABLE[t])) { krn = t; break; }
    if (krn < 0) return -1;                     /* king not between the rooks */

    return ((krn * 6 + q) * 4 + db / 2) * 4 + (lb - 1) / 2;
}

/* xoshiro256** -- the same generator the trainer uses, kept local so that
 * chess.c stays free-standing. */
static inline uint64_t x960_rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }

static uint64_t x960_next(uint64_t *s)
{
    const uint64_t result = x960_rotl(s[1] * 5, 7) * 9;
    const uint64_t t = s[1] << 17;
    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];
    s[2] ^= t;
    s[3] = x960_rotl(s[3], 45);
    return result;
}

int pos_960_random(uint64_t *rng)
{
    if (!rng) return CHESS960_CLASSICAL_ID;
    if (!(rng[0] | rng[1] | rng[2] | rng[3])) {        /* the forbidden state */
        rng[0] = 0x9E3779B97F4A7C15ULL;
        rng[1] = 0xBF58476D1CE4E5B9ULL;
        rng[2] = 0x94D049BB133111EBULL;
        rng[3] = 0x2545F4914F6CDD1DULL;
    }
    /* Rejection sampling: plain % would favour the first 2^64 mod 960 ids. */
    const uint64_t limit = (UINT64_MAX / 960) * 960;
    uint64_t r;
    do { r = x960_next(rng); } while (r >= limit);
    return (int)(r % 960);
}

void pos_to_fen(const Position *p, char *buf, size_t buflen)
{
    static const char PCH[2][6] = { { 'P','N','B','R','Q','K' }, { 'p','n','b','r','q','k' } };
    char tmp[128];
    int i = 0;

    for (int r = 7; r >= 0; r--) {
        int run = 0;
        for (int f = 0; f < 8; f++) {
            int sq = r * 8 + f;
            if (p->board[sq] == NO_PIECE) {
                run++;
            } else {
                if (run) { tmp[i++] = (char)('0' + run); run = 0; }
                tmp[i++] = PCH[p->color_at[sq]][p->board[sq]];
            }
        }
        if (run) tmp[i++] = (char)('0' + run);
        if (r) tmp[i++] = '/';
    }
    tmp[i++] = ' ';
    tmp[i++] = p->side == WHITE ? 'w' : 'b';
    tmp[i++] = ' ';
    if (!p->castling) {
        tmp[i++] = '-';
    } else if (p->chess960) {
        /* Shredder-FEN.  A Chess960 position is emitted with file letters
         * unconditionally: KQkq is only unambiguous while the castling rook is
         * still the outermost one on its side, and a promoted rook landing
         * outside it would silently change the meaning of the FEN. */
        if (p->castling & CR_WK) tmp[i++] = (char)('A' + p->crook[WHITE][0]);
        if (p->castling & CR_WQ) tmp[i++] = (char)('A' + p->crook[WHITE][1]);
        if (p->castling & CR_BK) tmp[i++] = (char)('a' + p->crook[BLACK][0]);
        if (p->castling & CR_BQ) tmp[i++] = (char)('a' + p->crook[BLACK][1]);
    } else {
        if (p->castling & CR_WK) tmp[i++] = 'K';
        if (p->castling & CR_WQ) tmp[i++] = 'Q';
        if (p->castling & CR_BK) tmp[i++] = 'k';
        if (p->castling & CR_BQ) tmp[i++] = 'q';
    }
    tmp[i++] = ' ';
    if (p->ep < 0) {
        tmp[i++] = '-';
    } else {
        tmp[i++] = (char)('a' + sq_file(p->ep));
        tmp[i++] = (char)('1' + sq_rank(p->ep));
    }
    tmp[i] = '\0';
    i += snprintf(tmp + i, sizeof(tmp) - (size_t)i, " %u %u",
                  (unsigned)p->halfmove, (unsigned)p->fullmove);

    if (!buf || buflen == 0) return;
    size_t len = (size_t)i;
    if (len >= buflen) len = buflen - 1;
    memcpy(buf, tmp, len);
    buf[len] = '\0';
}

/* -------------------------------------------------------------- UCI ---- */

void move_to_uci(Move m, char *buf)
{
    static const char PROMO[6] = { ' ', 'n', 'b', 'r', 'q', ' ' };
    int from = MV_FROM(m), to = MV_TO(m);
    buf[0] = (char)('a' + sq_file(from));
    buf[1] = (char)('1' + sq_rank(from));
    buf[2] = (char)('a' + sq_file(to));
    buf[3] = (char)('1' + sq_rank(to));
    if (MV_IS_PROMO(m)) {
        buf[4] = PROMO[MV_PROMO_PIECE(m)];
        buf[5] = '\0';
    } else {
        buf[4] = '\0';
    }
}

/* Chess960 writes castling as king-takes-rook ("e1h1"), because the king's
 * destination may be its own origin or an ordinary king move.  The internal
 * encoding already stores exactly that, but this is derived from the position
 * rather than from the move so that it is right whichever encoding a caller
 * hands in.  The move is assumed to belong to `p`, i.e. to p->side. */
void move_to_uci_pos(const Position *p, Move m, char *buf)
{
    const int fl = MV_FLAG(m);
    if (p && (fl == MF_KCASTLE || fl == MF_QCASTLE)) {
        const int us = p->side, side = (fl == MF_KCASTLE) ? 0 : 1;
        const int to = p->chess960 ? crook_sq(p, us, side) : castle_kto(us, side);
        buf[0] = (char)('a' + sq_file(MV_FROM(m)));
        buf[1] = (char)('1' + sq_rank(MV_FROM(m)));
        buf[2] = (char)('a' + sq_file(to));
        buf[3] = (char)('1' + sq_rank(to));
        buf[4] = '\0';
        return;
    }
    move_to_uci(m, buf);
}

/* Accepts both castling notations.  DISAMBIGUATION RULE, in two passes:
 *
 *   1. an exact (from,to) match against the legal moves always wins.  Castling
 *      is stored king-takes-rook in Chess960, so "e1h1" resolves to castling,
 *      and "e1g1" resolves to the ordinary king move whenever that move is
 *      legal -- which is the only reading that lets a 960 game express both;
 *   2. only if nothing matched exactly is the string re-read as the other
 *      castling notation: from the king's square to its castling destination
 *      (the classical form of a 960 castle) or to the castling rook's square
 *      (the 960 form of a classical castle, accepted for tolerance).
 *
 * So the rook-square form is king-takes-rook, and the g1/c1 form is a normal
 * king move whenever one exists and castling otherwise. */
int move_from_uci(const Position *p, const char *s, Move *out)
{
    if (!s) return 0;
    if (s[0] < 'a' || s[0] > 'h' || s[1] < '1' || s[1] > '8') return 0;
    if (s[2] < 'a' || s[2] > 'h' || s[3] < '1' || s[3] > '8') return 0;
    int from = (s[1] - '1') * 8 + (s[0] - 'a');
    int to   = (s[3] - '1') * 8 + (s[2] - 'a');
    int promo = 0;   /* 0 = none, else piece type */
    switch (s[4]) {
        case '\0': case ' ': promo = 0; break;
        case 'n': case 'N': promo = KNIGHT; break;
        case 'b': case 'B': promo = BISHOP; break;
        case 'r': case 'R': promo = ROOK;   break;
        case 'q': case 'Q': promo = QUEEN;  break;
        default: return 0;
    }

    Move list[MAX_MOVES];
    int n = gen_legal(p, list);
    for (int i = 0; i < n; i++) {
        Move m = list[i];
        if (MV_FROM(m) != from || MV_TO(m) != to) continue;
        if (MV_IS_PROMO(m)) {
            if (promo && MV_PROMO_PIECE(m) != promo) continue;
            if (!promo && MV_PROMO_PIECE(m) != QUEEN) continue;  /* default to queen */
        } else if (promo) {
            continue;
        }
        if (out) *out = m;
        return 1;
    }

    /* Pass 2: the other castling notation. */
    if (!promo) {
        for (int i = 0; i < n; i++) {
            const Move m = list[i];
            const int fl = MV_FLAG(m);
            if (fl != MF_KCASTLE && fl != MF_QCASTLE) continue;
            if (MV_FROM(m) != from) continue;
            const int us = p->side, side = (fl == MF_KCASTLE) ? 0 : 1;
            if (to != castle_kto(us, side) && to != crook_sq(p, us, side)) continue;
            if (out) *out = m;
            return 1;
        }
    }
    return 0;
}

/* -------------------------------------------------------------- SAN ---- */

void move_to_san(const Position *p, Move m, char *buf, size_t buflen)
{
    static const char PCH[6] = { 'P', 'N', 'B', 'R', 'Q', 'K' };
    char tmp[16];
    int i = 0;

    const int from = MV_FROM(m), to = MV_TO(m), fl = MV_FLAG(m);
    const int pc = p->board[from];

    if (fl == MF_KCASTLE) {
        tmp[i++] = 'O'; tmp[i++] = '-'; tmp[i++] = 'O';
    } else if (fl == MF_QCASTLE) {
        tmp[i++] = 'O'; tmp[i++] = '-'; tmp[i++] = 'O';
        tmp[i++] = '-'; tmp[i++] = 'O';
    } else {
        const int is_cap = (fl & 4) != 0;
        if (pc == PAWN) {
            if (is_cap) tmp[i++] = (char)('a' + sq_file(from));
        } else {
            tmp[i++] = PCH[pc];
            Move list[MAX_MOVES];
            int n = gen_legal(p, list);
            int amb = 0, same_file = 0, same_rank = 0;
            for (int j = 0; j < n; j++) {
                int f2 = MV_FROM(list[j]);
                if (f2 == from) continue;
                if (MV_TO(list[j]) != to) continue;
                if (p->board[f2] != pc) continue;
                amb = 1;
                if (sq_file(f2) == sq_file(from)) same_file = 1;
                if (sq_rank(f2) == sq_rank(from)) same_rank = 1;
            }
            if (amb) {
                if (!same_file) {
                    tmp[i++] = (char)('a' + sq_file(from));
                } else if (!same_rank) {
                    tmp[i++] = (char)('1' + sq_rank(from));
                } else {
                    tmp[i++] = (char)('a' + sq_file(from));
                    tmp[i++] = (char)('1' + sq_rank(from));
                }
            }
        }
        if (is_cap) tmp[i++] = 'x';
        tmp[i++] = (char)('a' + sq_file(to));
        tmp[i++] = (char)('1' + sq_rank(to));
        if (fl >= MF_PROMO_N) {
            tmp[i++] = '=';
            tmp[i++] = PCH[MV_PROMO_PIECE(m)];
        }
    }

    /* check / mate suffix */
    {
        Position q;
        Undo u;
        memcpy(&q, p, sizeof(Position));
        make_move(&q, m, &u);
        if (in_check(&q, q.side)) {
            Move list[MAX_MOVES];
            tmp[i++] = gen_legal(&q, list) ? '+' : '#';
        }
    }
    tmp[i] = '\0';

    if (!buf || buflen == 0) return;
    size_t len = (size_t)i;
    if (len >= buflen) len = buflen - 1;
    memcpy(buf, tmp, len);
    buf[len] = '\0';
}

/* ------------------------------------------------------------- perft --- */

uint64_t perft(Position *p, int depth)
{
    if (depth <= 0) return 1;
    Move list[MAX_MOVES];
    int n = gen_legal(p, list);
    if (depth == 1) return (uint64_t)n;   /* bulk counting */

    uint64_t nodes = 0;
    Undo u;
    for (int i = 0; i < n; i++) {
        make_move(p, list[i], &u);
        nodes += perft(p, depth - 1);
        unmake_move(p, list[i], &u);
    }
    return nodes;
}

/* ------------------------------------------------- insufficient material */

int insufficient_material(const Position *p)
{
    if (p->piece[WHITE][PAWN] | p->piece[BLACK][PAWN]) return 0;
    if (p->piece[WHITE][ROOK] | p->piece[BLACK][ROOK]) return 0;
    if (p->piece[WHITE][QUEEN] | p->piece[BLACK][QUEEN]) return 0;

    const int wb = bb_count(p->piece[WHITE][BISHOP]);
    const int wn = bb_count(p->piece[WHITE][KNIGHT]);
    const int bb_ = bb_count(p->piece[BLACK][BISHOP]);
    const int bn = bb_count(p->piece[BLACK][KNIGHT]);
    const int w = wb + wn, b = bb_ + bn;

    if (w == 0 && b == 0) return 1;                      /* K v K            */

    if (w == 0 || b == 0) {
        const int m = w + b;
        if (m == 1) return 1;                            /* K+B v K, K+N v K */
        if (m == 2 && (wn == 2 || bn == 2)) return 1;    /* K+N+N v K        */
        return 0;
    }

    /* K+B v K+B with both bishops on the same colour complex */
    if (w == 1 && b == 1 && wb == 1 && bb_ == 1) {
        const int wl = (p->piece[WHITE][BISHOP] & LIGHT_SQUARES) != 0;
        const int bl = (p->piece[BLACK][BISHOP] & LIGHT_SQUARES) != 0;
        if (wl == bl) return 1;
    }
    return 0;
}

/* -------------------------------------------------------------- game --- */

void game_start(Game *g)
{
    memset(g, 0, sizeof(*g));
    pos_startpos(&g->pos);
    g->ply = 0;
    g->hist[0] = g->pos.key;
    g->hist_len = 1;
    g->result = GR_ONGOING;
    g->reason = TR_NONE;
    game_update_result(g, 0);
}

void game_start960(Game *g, int id)
{
    memset(g, 0, sizeof(*g));
    pos_startpos960(&g->pos, id);
    g->ply = 0;
    g->hist[0] = g->pos.key;
    g->hist_len = 1;
    g->result = GR_ONGOING;
    g->reason = TR_NONE;
    game_update_result(g, 0);
}

int game_start_fen(Game *g, const char *fen)
{
    memset(g, 0, sizeof(*g));
    if (!pos_from_fen(&g->pos, fen)) {
        pos_startpos(&g->pos);
        g->hist[0] = g->pos.key;
        g->hist_len = 1;
        g->result = GR_ONGOING;
        g->reason = TR_NONE;
        return 0;
    }
    g->ply = 0;
    g->hist[0] = g->pos.key;
    g->hist_len = 1;
    g->result = GR_ONGOING;
    g->reason = TR_NONE;
    game_update_result(g, 0);
    return 1;
}

int game_repetitions(const Game *g)
{
    const uint64_t key = g->pos.key;
    int count = 0;
    int lo = g->hist_len - 1 - (int)g->pos.halfmove;
    if (lo < 0) lo = 0;
    for (int i = g->hist_len - 1; i >= lo; i -= 2)
        if (g->hist[i] == key) count++;
    return count;
}

int game_update_result(Game *g, int max_plies)
{
    Position *p = &g->pos;
    Move list[MAX_MOVES];
    const int n = gen_legal(p, list);

    if (n == 0) {
        if (in_check(p, p->side)) {
            g->result = (p->side == WHITE) ? GR_BLACK_WIN : GR_WHITE_WIN;
            g->reason = TR_CHECKMATE;
        } else {
            g->result = GR_DRAW;
            g->reason = TR_STALEMATE;
        }
        return g->result;
    }
    if (insufficient_material(p)) {
        g->result = GR_DRAW;
        g->reason = TR_INSUFFICIENT;
        return g->result;
    }
    /* Fifty-move and threefold repetition are AUTOMATIC draws here: self-play
     * has nobody to make a claim, and letting games run on would only burn
     * compute on shuffling. */
    if (p->halfmove >= 100) {
        g->result = GR_DRAW;
        g->reason = TR_FIFTY;
        return g->result;
    }
    if (game_repetitions(g) >= 3) {
        g->result = GR_DRAW;
        g->reason = TR_REPETITION;
        return g->result;
    }
    if (max_plies > 0 && g->ply >= max_plies) {
        g->result = GR_DRAW;
        g->reason = TR_MAX_PLIES;
        return g->result;
    }
    g->result = GR_ONGOING;
    g->reason = TR_NONE;
    return g->result;
}

int game_push(Game *g, Move m)
{
    if (g->ply >= MAX_GAME_PLIES) {
        g->result = GR_DRAW;
        g->reason = TR_MAX_PLIES;
        return g->result;
    }
    g->moves[g->ply] = m;
    make_move(&g->pos, m, &g->undos[g->ply]);
    g->ply++;
    g->hist[g->ply] = g->pos.key;
    g->hist_len = g->ply + 1;
    if (g->ply >= MAX_GAME_PLIES) {
        /* hard structural cap -- no room to record another ply */
        int r = game_update_result(g, 0);
        if (r == GR_ONGOING) { g->result = GR_DRAW; g->reason = TR_MAX_PLIES; }
        return g->result;
    }
    return game_update_result(g, 0);
}

void game_pop(Game *g)
{
    if (g->ply <= 0) return;
    g->ply--;
    unmake_move(&g->pos, g->moves[g->ply], &g->undos[g->ply]);
    g->hist_len = g->ply + 1;
    game_update_result(g, 0);
}
