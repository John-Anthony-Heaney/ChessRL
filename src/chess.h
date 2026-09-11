/* chess.h -- complete FIDE-rules bitboard chess core.
 *
 * Conventions
 *   - Squares 0..63 == a1..h8 (rank-major: sq = rank*8 + file, a1 = 0, h8 = 63).
 *   - Bitboard bit i corresponds to square i.
 *   - Colours: WHITE = 0, BLACK = 1.
 *   - All generated moves from gen_legal() are FULLY legal (no pseudo-legal leakage).
 */
#ifndef CHESS_H
#define CHESS_H

#include <stdint.h>
#include <stddef.h>

/* ------------------------------------------------------------------ types */

enum { WHITE = 0, BLACK = 1, NCOLORS = 2 };
enum { PAWN = 0, KNIGHT = 1, BISHOP = 2, ROOK = 3, QUEEN = 4, KING = 5, NPIECES = 6, NO_PIECE = 6 };

/* Castling-right bits. */
enum { CR_WK = 1, CR_WQ = 2, CR_BK = 4, CR_BQ = 8, CR_ALL = 15 };

/* Move = 16 bits: from(0..5) | to(6..11) | flag(12..15). */
typedef uint16_t Move;

enum {
    MF_QUIET     = 0,
    MF_DOUBLE    = 1,   /* double pawn push (sets ep square)      */
    MF_KCASTLE   = 2,   /* king-side castle                       */
    MF_QCASTLE   = 3,   /* queen-side castle                      */
    MF_CAPTURE   = 4,
    MF_EP        = 5,   /* en-passant capture                     */
    MF_PROMO_N   = 8,  MF_PROMO_B = 9,  MF_PROMO_R = 10, MF_PROMO_Q = 11,
    MF_PROMO_NC  = 12, MF_PROMO_BC = 13, MF_PROMO_RC = 14, MF_PROMO_QC = 15
};

#define MV_FROM(m)        ((int)((m) & 63))
#define MV_TO(m)          ((int)(((m) >> 6) & 63))
#define MV_FLAG(m)        ((int)(((m) >> 12) & 15))
#define MV_MAKE(f, t, fl) ((Move)((Move)(f) | ((Move)(t) << 6) | ((Move)(fl) << 12)))
#define MV_IS_PROMO(m)    (MV_FLAG(m) >= MF_PROMO_N)
#define MV_IS_CAPTURE(m)  ((MV_FLAG(m) & 4) != 0)
/* Promotion target piece: flags 8/12 -> KNIGHT ... 11/15 -> QUEEN. */
#define MV_PROMO_PIECE(m) (((MV_FLAG(m)) & 3) + KNIGHT)
#define MV_NONE           ((Move)0)

#define MAX_MOVES         256

typedef struct {
    uint64_t piece[NCOLORS][NPIECES]; /* piece bitboards                     */
    uint64_t occ[NCOLORS];            /* per-colour occupancy                */
    uint64_t all;                     /* occ[WHITE] | occ[BLACK]             */
    uint64_t key;                     /* zobrist hash (incl. side/castle/ep) */
    int8_t   board[64];               /* piece type per square, NO_PIECE if empty */
    int8_t   color_at[64];            /* colour per square, -1 if empty      */
    uint8_t  side;                    /* side to move                        */
    uint8_t  castling;                /* CR_* bits                           */
    int8_t   ep;                      /* en-passant target square, -1 if none */
    uint8_t  chess960;                /* 0 = classical, 1 = Chess960 castling */
    /* Origin FILE of each castling rook, [colour][0 = king-side, 1 = queen-side].
     * Classical chess is simply {7,0}; Chess960 needs this because the rooks do
     * not start on a1/h1.  Only meaningful where the matching CR_* bit is set. */
    uint8_t  crook[NCOLORS][2];
    uint16_t halfmove;                /* plies since last capture/pawn move  */
    uint16_t fullmove;                /* starts at 1, increments after black */
} Position;

/* State needed to undo a move. */
typedef struct {
    uint64_t key;
    int8_t   captured;   /* piece type captured, NO_PIECE if none (EP: PAWN) */
    int8_t   ep;
    uint8_t  castling;
    uint16_t halfmove;
    uint16_t fullmove;
} Undo;

/* Game results. */
enum { GR_ONGOING = 0, GR_WHITE_WIN = 1, GR_BLACK_WIN = 2, GR_DRAW = 3 };
/* Termination reasons. */
enum {
    TR_NONE = 0, TR_CHECKMATE, TR_STALEMATE, TR_FIFTY, TR_REPETITION,
    TR_INSUFFICIENT, TR_MAX_PLIES, TR_ADJUDICATED, TR_RESIGN
};

#define MAX_GAME_PLIES 1024

typedef struct {
    Position pos;
    uint64_t hist[MAX_GAME_PLIES + 8];  /* zobrist keys, hist[0] = start position */
    Move     moves[MAX_GAME_PLIES];
    Undo     undos[MAX_GAME_PLIES];
    int      ply;                       /* number of moves played             */
    int      hist_len;                  /* == ply + 1                          */
    int      result;                    /* GR_*                                */
    int      reason;                    /* TR_*                                */
} Game;

/* ------------------------------------------------------------------ API */

/* Must be called once before anything else (attack tables + zobrist keys). */
void chess_init(void);

void     pos_startpos(Position *p);
/* Returns 1 on success, 0 on malformed FEN.  Accepts standard 4/5/6-field FEN. */
int      pos_from_fen(Position *p, const char *fen);
/* Writes a NUL-terminated FEN into buf (>= 96 bytes). */
void     pos_to_fen(const Position *p, char *buf, size_t buflen);
uint64_t pos_compute_key(const Position *p);

/* Legal move generation.  `out` must hold MAX_MOVES entries.  Returns count. */
int  gen_legal(const Position *p, Move *out);
/* Legal captures + queen promotions only (for quiescence search). */
int  gen_legal_captures(const Position *p, Move *out);

/* Is square `sq` attacked by side `by`? */
int  square_attacked(const Position *p, int sq, int by);
/* Is the side `c` king currently in check? */
int  in_check(const Position *p, int c);

void make_move(Position *p, Move m, Undo *u);
void unmake_move(Position *p, Move m, const Undo *u);
/* Make/unmake a null move (side flips, ep cleared) -- used by search. */
void make_null(Position *p, Undo *u);
void unmake_null(Position *p, const Undo *u);

/* True when neither side can possibly deliver mate (FIDE dead position subset:
 * K v K, K+minor v K, K+B v K+B with same-coloured bishops, K+N+N v K). */
int  insufficient_material(const Position *p);

/* Long-algebraic ("e2e4", "e7e8q") <-> Move, validated against legal moves. */
int  move_from_uci(const Position *p, const char *s, Move *out);
void move_to_uci(Move m, char *buf);      /* buf >= 6 bytes */
/* Standard algebraic notation (SAN) for a legal move, e.g. "Nxf7+". */
void move_to_san(const Position *p, Move m, char *buf, size_t buflen);

uint64_t perft(Position *p, int depth);

/* ------------------------------------------------------------- chess 960 */
/* Chess960 (Fischer Random).  Castling is defined by the DESTINATION squares,
 * which are the same as classical chess (king to g1/c1, rook to f1/d1), but the
 * king and rooks may start anywhere on the back rank, so castling legality has
 * to be expressed in terms of the rook origin files in Position.crook.
 *
 * Positions are numbered 0..959 using the standard Scharnagl scheme, in which
 * 518 is the classical starting array.  Every classical rule is unchanged. */
#define CHESS960_CLASSICAL_ID 518

void pos_startpos960(Position *p, int id);       /* id is taken modulo 960     */
/* Scharnagl id of a back-rank arrangement, or -1 if it is not a legal 960 array. */
int  pos_960_id(const Position *p);
/* Uniformly samples a 960 id from `rng`, an xoshiro256** state. */
int  pos_960_random(uint64_t *rng);

/* X-FEN / Shredder-FEN.  pos_from_fen accepts KQkq, the file-letter form
 * (e.g. "HAha"), and the X-FEN hybrid; pos_to_fen emits file letters whenever
 * the position is Chess960 and KQkq would be ambiguous. */

/* In Chess960 a castling move is conventionally written king-takes-rook
 * ("e1h1"), because the king's destination can coincide with a normal king
 * move.  move_from_uci accepts BOTH that form and the classical king-destination
 * form; move_to_uci emits king-takes-rook when p->chess960 is set, so the extra
 * argument is the position the move belongs to (may be NULL for classical). */
void move_to_uci_pos(const Position *p, Move m, char *buf);

/* ------------------------------------------------------------------ game */

void game_start(Game *g);
void game_start960(Game *g, int id);
int  game_start_fen(Game *g, const char *fen);
/* Applies a move, appends history, refreshes result/reason.  Returns result. */
int  game_push(Game *g, Move m);
void game_pop(Game *g);
/* Number of times the current position key appears in the history. */
int  game_repetitions(const Game *g);
/* Recomputes g->result / g->reason.  max_plies <= 0 disables the ply cap. */
int  game_update_result(Game *g, int max_plies);

/* ------------------------------------------------------------------ utils */

extern const char *SQ_NAMES[64];
static inline int sq_make(int file, int rank) { return rank * 8 + file; }
static inline int sq_file(int sq) { return sq & 7; }
static inline int sq_rank(int sq) { return sq >> 3; }
static inline int sq_flip(int sq) { return sq ^ 56; }   /* vertical mirror */

#if defined(__GNUC__) || defined(__clang__)
static inline int  bb_lsb(uint64_t b)  { return __builtin_ctzll(b); }
static inline int  bb_count(uint64_t b){ return __builtin_popcountll(b); }
#else
int bb_lsb(uint64_t b);
int bb_count(uint64_t b);
#endif
static inline int  bb_pop(uint64_t *b) { int s = bb_lsb(*b); *b &= *b - 1; return s; }

/* Attack helpers exposed for the evaluation / feature code. */
uint64_t attacks_knight(int sq);
uint64_t attacks_king(int sq);
uint64_t attacks_pawn(int sq, int color);
uint64_t attacks_bishop(int sq, uint64_t occ);
uint64_t attacks_rook(int sq, uint64_t occ);
uint64_t attacks_queen(int sq, uint64_t occ);

#endif /* CHESS_H */
