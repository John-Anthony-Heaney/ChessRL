/* api.c -- flat C ABI consumed by py/engine.py through ctypes.
 *
 * ============================================================================
 * TRUNCATION CONTRACT  (applies to EVERY function here that fills a char *buf)
 * ============================================================================
 * Each buffer-filling entry point builds its payload through the string builder
 * `SB` below, which keeps counting the bytes it *would* have produced even after
 * it stops copying into the caller's buffer.  That makes the required size exact
 * on the first try, so a caller never has to guess-and-double.
 *
 *   success  ->  the number of payload bytes written, NOT counting the trailing
 *                NUL.  The buffer is always NUL-terminated on success, so the
 *                caller may read it either by the returned length or as a C
 *                string; both give the same bytes.
 *
 *   too small ->  the NEGATIVE of the buffer size required, i.e.
 *                 -(payload_bytes + 1).  The "+1" is the NUL terminator, so
 *                 re-calling with a buffer of exactly `-ret` bytes is guaranteed
 *                 to succeed.  Whatever bytes happen to be in the buffer are
 *                 meaningless in this case (they are a truncated prefix); the
 *                 caller must retry, not parse.
 *
 *   error    ->  -1 (bad handle, bad argument, out of memory, ...).  This is
 *                unambiguous: a genuine "too small" answer is always
 *                -(payload+1) <= -2, and callers pass buffers far larger than
 *                one byte, so a -1 can never be mistaken for a size request.
 *
 * Non-buffer entry points return their documented value, or -1 on a bad handle.
 * No entry point may crash on a garbage handle: every one of them validates.
 *
 * ============================================================================
 * SIGN CONVENTIONS  (read this before touching the eval numbers)
 * ============================================================================
 * `score` in api_engine_move is Search::score_cp, which search.h defines as
 * being from the SIDE-TO-MOVE's point of view (positive = good for whoever is
 * about to move).
 *
 * `value` in api_engine_move is the network's tanh value head for the position
 * the engine was asked about, re-signed into WHITE's frame of reference:
 *
 *     value = (side_to_move == WHITE) ? fw.v : -fw.v
 *
 * so +1 always means "White is winning" and -1 always means "Black is winning",
 * no matter whose turn it is.  This is what keeps a UI eval bar from flipping
 * left and right on every ply.  NOTE that this deliberately differs from the
 * convention used by `score`; see the note in the project hand-off about the
 * two consumers that currently re-sign `value` as if it were side-to-move
 * relative.
 *
 * `prob` / `logit` in api_engine_move's top[] and in api_engine_policy are the
 * raw policy head at temperature 1.0 (api.h calls it "raw policy"), NOT at the
 * agent's PBT `temperature` hyper-parameter.  Both endpoints use the same
 * temperature so the numbers they report for one position always agree.
 *
 * ============================================================================
 * HANDLES AND LOCKING
 * ============================================================================
 * Games and engines live in fixed-capacity tables of pointers.  One pthread
 * mutex (`g_lock`) guards allocation, lookup and free of table entries; it is
 * held only for the pointer shuffling, never across a search or a file read.
 * Once an entry point has resolved its handle to a pointer it works on that
 * object with the global lock released, so two threads driving two different
 * handles never contend.  Handles are small non-negative ints and freed slots
 * are reused.
 *
 * As documented in api.h, a single handle must not be used by two threads at
 * once -- py/engine.py enforces that with a per-handle lock.
 */

#include <limits.h>
#include <math.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "api.h"
#include "chess.h"
#include "net.h"
#include "search.h"

/* ------------------------------------------------------------- capacities */

#define API_MAX_GAMES    1024   /* py/server.py caps live games at 200        */
#define API_MAX_ENGINES    64   /* py/server.py caps its agent cache at 12    */
#define API_MAX_AGENTS   1024   /* refuse absurd n_agents from a corrupt file */
#define API_TT_MB          64   /* transposition table per loaded engine      */
#define API_SAN_LEN        12   /* "Qa1xb2#" + NUL, with room to spare        */
#define API_MAX_PATH     4096

#define API_DEF_DEPTH       4
#define API_DEF_MOVETIME 1000
#define API_MAX_DEPTH      63   /* Search::pv holds 64 plies                  */
#define API_MAX_MOVETIME 300000

/* ==========================================================================
 * string builder
 * ========================================================================== */

typedef struct {
    char  *buf;   /* caller's buffer, may be NULL                             */
    size_t cap;   /* usable bytes in buf (0 when there is nowhere to write)   */
    size_t len;   /* payload bytes produced so far; may exceed cap - 1        */
} SB;

static void sb_init(SB *sb, char *buf, int buflen)
{
    sb->buf = buf;
    sb->cap = (buf && buflen > 0) ? (size_t)buflen : 0;
    sb->len = 0;
    if (sb->cap) sb->buf[0] = '\0';
}

/* Append n bytes.  Copies as much as fits, always keeps the buffer
 * NUL-terminated, and always advances the required-length counter. */
static void sb_putn(SB *sb, const char *s, size_t n)
{
    if (sb->cap) {
        size_t off  = (sb->len < sb->cap - 1) ? sb->len : sb->cap - 1;
        size_t room = sb->cap - 1 - off;
        size_t k    = (n < room) ? n : room;
        if (k) memcpy(sb->buf + off, s, k);
        sb->buf[off + k] = '\0';
    }
    sb->len += n;
}

static void sb_puts(SB *sb, const char *s) { if (s) sb_putn(sb, s, strlen(s)); }
static void sb_putc(SB *sb, char c)        { sb_putn(sb, &c, 1); }

#define SB_LIT(sb, s) sb_putn((sb), "" s, sizeof(s) - 1)

/* Final return value, per the TRUNCATION CONTRACT at the top of the file. */
static int sb_finish(const SB *sb)
{
    if (sb->len >= (size_t)INT_MAX - 1) return -1;          /* absurd, refuse */
    if (sb->cap > 0 && sb->len + 1 <= sb->cap) return (int)sb->len;
    return -(int)(sb->len + 1);
}

/* ------------------------------------------------------------ JSON pieces */

static void sb_json_str(SB *sb, const char *s)
{
    static const char HEX[] = "0123456789abcdef";
    sb_putc(sb, '"');
    if (s) {
        for (const unsigned char *p = (const unsigned char *)s; *p; p++) {
            unsigned char c = *p;
            switch (c) {
                case '"':  SB_LIT(sb, "\\\""); break;
                case '\\': SB_LIT(sb, "\\\\"); break;
                case '\n': SB_LIT(sb, "\\n");  break;
                case '\r': SB_LIT(sb, "\\r");  break;
                case '\t': SB_LIT(sb, "\\t");  break;
                case '\b': SB_LIT(sb, "\\b");  break;
                case '\f': SB_LIT(sb, "\\f");  break;
                default:
                    if (c < 0x20) {
                        char esc[6] = { '\\', 'u', '0', '0', HEX[c >> 4], HEX[c & 15] };
                        sb_putn(sb, esc, 6);
                    } else {
                        sb_putc(sb, (char)c);
                    }
            }
        }
    }
    sb_putc(sb, '"');
}

/* Emits a single character as a JSON string, e.g. 'p' -> "p". */
static void sb_json_str_char(SB *sb, char c)
{
    char tmp[2];
    tmp[0] = c;
    tmp[1] = '\0';
    sb_json_str(sb, tmp);
}

static void sb_i64(SB *sb, long long v)
{
    char tmp[32];
    int n = snprintf(tmp, sizeof(tmp), "%lld", v);
    if (n > 0) sb_putn(sb, tmp, (size_t)n);
}

static void sb_u64(SB *sb, unsigned long long v)
{
    char tmp[32];
    int n = snprintf(tmp, sizeof(tmp), "%llu", v);
    if (n > 0) sb_putn(sb, tmp, (size_t)n);
}

/* JSON has no NaN/Infinity: anything non-finite is emitted as 0 so the payload
 * always parses.  %.6g keeps six significant digits and never emits a bare
 * leading '.' or a trailing '.', both of which would be invalid JSON. */
static void sb_f(SB *sb, double v)
{
    char tmp[40];
    int n;
    if (!isfinite(v)) v = 0.0;
    n = snprintf(tmp, sizeof(tmp), "%.6g", v);
    if (n > 0) sb_putn(sb, tmp, (size_t)n);
    else SB_LIT(sb, "0");
}

static void sb_key(SB *sb, const char *k)
{
    sb_json_str(sb, k);
    sb_putc(sb, ':');
}

/* ==========================================================================
 * handle tables
 * ========================================================================== */

typedef struct {
    Game     g;
    Position start_pos;                       /* position the game started in */
    char     san[MAX_GAME_PLIES][API_SAN_LEN];/* SAN of each played move      */
} GameSlot;

typedef struct {
    Trunk  trunk;                /* Search keeps pointers to these two, so    */
    Head   head;                 /* the slot must never be moved or copied.   */
    Hyper  hyper;
    Search search;
    Game   scratch;              /* search runs on this copy, never on the
                                  * caller's Game                            */
    float  elo;
    int    agent_index;
    int    n_agents;
    int    generation;
    char   path[API_MAX_PATH];
} EngineSlot;

static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_once_t  g_once = PTHREAD_ONCE_INIT;

static GameSlot   *g_games[API_MAX_GAMES];
static EngineSlot *g_engines[API_MAX_ENGINES];
static int         g_game_hint;      /* round-robin start for the free scan  */
static int         g_engine_hint;

static void api_once_init(void) { chess_init(); }

int api_init(void)
{
    pthread_once(&g_once, api_once_init);
    return 1;
}

/* Resolve a handle to its object.  The table is only read under the lock so a
 * concurrent allocation of a *different* slot can never be observed torn. */
static GameSlot *game_get(int gid)
{
    GameSlot *s = NULL;
    if (gid < 0 || gid >= API_MAX_GAMES) return NULL;
    pthread_mutex_lock(&g_lock);
    s = g_games[gid];
    pthread_mutex_unlock(&g_lock);
    return s;
}

static EngineSlot *engine_get(int eid)
{
    EngineSlot *s = NULL;
    if (eid < 0 || eid >= API_MAX_ENGINES) return NULL;
    pthread_mutex_lock(&g_lock);
    s = g_engines[eid];
    pthread_mutex_unlock(&g_lock);
    return s;
}

/* Claim the lowest free slot at or after the rotating hint.  Returns -1 when
 * the table is full.  Caller must already hold g_lock. */
static int table_claim(void **tab, int cap, int *hint)
{
    for (int k = 0; k < cap; k++) {
        int i = (*hint + k) % cap;
        if (!tab[i]) {
            *hint = (i + 1) % cap;
            return i;
        }
    }
    return -1;
}

/* ==========================================================================
 * games
 * ========================================================================== */

/* Record the SAN of `m` (which must be legal in slot->g.pos) at the current
 * ply, then play it.  SAN needs the position *before* the move, so the order
 * here matters. */
static void game_slot_push(GameSlot *s, Move m)
{
    int at = s->g.ply;
    if (at >= 0 && at < MAX_GAME_PLIES)
        move_to_san(&s->g.pos, m, s->san[at], API_SAN_LEN);
    game_push(&s->g, m);
}

int api_game_new(const char *fen)
{
    GameSlot *s;
    int gid;

    api_init();

    s = (GameSlot *)calloc(1, sizeof(GameSlot));
    if (!s) return -1;

    if (fen && *fen) {
        if (!game_start_fen(&s->g, fen)) { free(s); return -1; }
    } else {
        game_start(&s->g);
    }
    s->start_pos = s->g.pos;

    pthread_mutex_lock(&g_lock);
    gid = table_claim((void **)g_games, API_MAX_GAMES, &g_game_hint);
    if (gid >= 0) g_games[gid] = s;
    pthread_mutex_unlock(&g_lock);

    if (gid < 0) { free(s); return -1; }
    return gid;
}

void api_game_free(int gid)
{
    GameSlot *s = NULL;
    if (gid < 0 || gid >= API_MAX_GAMES) return;
    pthread_mutex_lock(&g_lock);
    s = g_games[gid];
    g_games[gid] = NULL;
    pthread_mutex_unlock(&g_lock);
    free(s);
}

int api_game_legal(int gid, char *buf, int buflen)
{
    GameSlot *s;
    Move list[MAX_MOVES];
    int n;
    SB sb;

    api_init();
    s = game_get(gid);
    if (!s) return -1;

    sb_init(&sb, buf, buflen);
    n = gen_legal(&s->g.pos, list);
    for (int i = 0; i < n; i++) {
        char uci[8];
        if (i) sb_putc(&sb, ' ');
        move_to_uci(list[i], uci);
        sb_puts(&sb, uci);
    }
    return sb_finish(&sb);
}

int api_game_move(int gid, const char *uci)
{
    GameSlot *s;
    Move m = MV_NONE;

    api_init();
    s = game_get(gid);
    if (!s) return -1;
    if (!uci || !*uci) return 0;
    if (s->g.ply >= MAX_GAME_PLIES) return 0;          /* no room to record it */
    if (!move_from_uci(&s->g.pos, uci, &m)) return 0;  /* not legal here       */

    game_slot_push(s, m);
    return 1;
}

int api_game_undo(int gid)
{
    GameSlot *s;

    api_init();
    s = game_get(gid);
    if (!s) return -1;
    if (s->g.ply <= 0) return 0;

    game_pop(&s->g);
    return 1;
}

/* ------------------------------------------------------------ state JSON */

/* Human-readable termination reason, exactly the vocabulary documented in
 * docs/API.md (py/server.py passes these strings through unchanged). */
static const char *reason_name(int reason)
{
    switch (reason) {
        case TR_CHECKMATE:    return "checkmate";
        case TR_STALEMATE:    return "stalemate";
        case TR_FIFTY:        return "fifty-move";
        case TR_REPETITION:   return "threefold repetition";
        case TR_INSUFFICIENT: return "insufficient material";
        case TR_MAX_PLIES:    return "move limit";
        case TR_ADJUDICATED:  return "adjudicated";
        case TR_RESIGN:       return "resignation";
        default:              return "";
    }
}

/* Standard piece values, indexed by PAWN..KING. */
static const int PIECE_VALUE[NPIECES] = { 1, 3, 3, 5, 9, 0 };
/* Full army, indexed by PAWN..KING. */
static const int START_COUNT[NPIECES] = { 8, 2, 2, 2, 1, 1 };
/* Lowercase letters, indexed by PAWN..KING. */
static const char PIECE_CHAR[NPIECES] = { 'p', 'n', 'b', 'r', 'q', 'k' };
/* Order captured pieces are listed in: most valuable first (matches the
 * fallback in py/server.py so the UI looks the same either way). */
static const int CAPTURE_ORDER[5] = { QUEEN, ROOK, BISHOP, KNIGHT, PAWN };

static int material_of(const Position *p, int color)
{
    int total = 0;
    for (int pt = PAWN; pt < NPIECES; pt++)
        total += PIECE_VALUE[pt] * bb_count(p->piece[color][pt]);
    return total;
}

/* The pieces `color` has LOST, relative to a full army, as lowercase letters. */
static void sb_captured(SB *sb, const Position *p, int color)
{
    int first = 1;
    sb_putc(sb, '[');
    for (int k = 0; k < 5; k++) {
        int pt   = CAPTURE_ORDER[k];
        int gone = START_COUNT[pt] - bb_count(p->piece[color][pt]);
        for (int i = 0; i < gone; i++) {          /* negative => promoted, skip */
            char c = PIECE_CHAR[pt];
            if (!first) sb_putc(sb, ',');
            first = 0;
            sb_json_str_char(sb, c);
        }
    }
    sb_putc(sb, ']');
}

int api_game_state(int gid, char *buf, int buflen)
{
    GameSlot *s;
    const Position *p;
    Move list[MAX_MOVES];
    char fen[128];
    int n;
    SB sb;

    api_init();
    s = game_get(gid);
    if (!s) return -1;

    p = &s->g.pos;
    n = gen_legal(p, list);
    pos_to_fen(p, fen, sizeof(fen));

    sb_init(&sb, buf, buflen);
    sb_putc(&sb, '{');

    sb_key(&sb, "gid");    sb_i64(&sb, gid);
    SB_LIT(&sb, ",");
    sb_key(&sb, "fen");    sb_json_str(&sb, fen);
    SB_LIT(&sb, ",");
    sb_key(&sb, "turn");   sb_json_str(&sb, p->side == WHITE ? "white" : "black");
    SB_LIT(&sb, ",");
    sb_key(&sb, "ply");    sb_i64(&sb, s->g.ply);
    SB_LIT(&sb, ",");
    sb_key(&sb, "result"); sb_i64(&sb, s->g.result);
    SB_LIT(&sb, ",");
    sb_key(&sb, "reason"); sb_json_str(&sb, reason_name(s->g.reason));
    SB_LIT(&sb, ",");
    sb_key(&sb, "check");
    sb_puts(&sb, in_check(p, p->side) ? "true" : "false");

    /* legal[] and san[] are parallel: san[i] is the SAN of legal[i]. */
    SB_LIT(&sb, ",");
    sb_key(&sb, "legal");
    sb_putc(&sb, '[');
    for (int i = 0; i < n; i++) {
        char uci[8];
        if (i) sb_putc(&sb, ',');
        move_to_uci(list[i], uci);
        sb_json_str(&sb, uci);
    }
    sb_putc(&sb, ']');

    SB_LIT(&sb, ",");
    sb_key(&sb, "san");
    sb_putc(&sb, '[');
    for (int i = 0; i < n; i++) {
        char san[API_SAN_LEN];
        if (i) sb_putc(&sb, ',');
        move_to_san(p, list[i], san, sizeof(san));
        sb_json_str(&sb, san);
    }
    sb_putc(&sb, ']');

    /* moves[] / history_san[] are the full history, also parallel. */
    SB_LIT(&sb, ",");
    sb_key(&sb, "moves");
    sb_putc(&sb, '[');
    for (int i = 0; i < s->g.ply; i++) {
        char uci[8];
        if (i) sb_putc(&sb, ',');
        move_to_uci(s->g.moves[i], uci);
        sb_json_str(&sb, uci);
    }
    sb_putc(&sb, ']');

    SB_LIT(&sb, ",");
    sb_key(&sb, "history_san");
    sb_putc(&sb, '[');
    for (int i = 0; i < s->g.ply; i++) {
        if (i) sb_putc(&sb, ',');
        sb_json_str(&sb, s->san[i]);
    }
    sb_putc(&sb, ']');

    SB_LIT(&sb, ",");
    sb_key(&sb, "last");
    if (s->g.ply > 0) {
        Move m = s->g.moves[s->g.ply - 1];
        SB_LIT(&sb, "{");
        sb_key(&sb, "from"); sb_json_str(&sb, SQ_NAMES[MV_FROM(m)]);
        SB_LIT(&sb, ",");
        sb_key(&sb, "to");   sb_json_str(&sb, SQ_NAMES[MV_TO(m)]);
        SB_LIT(&sb, ",");
        sb_key(&sb, "san");  sb_json_str(&sb, s->san[s->g.ply - 1]);
        SB_LIT(&sb, "}");
    } else {
        SB_LIT(&sb, "null");
    }

    SB_LIT(&sb, ",");
    sb_key(&sb, "material");
    SB_LIT(&sb, "{");
    sb_key(&sb, "white"); sb_i64(&sb, material_of(p, WHITE));
    SB_LIT(&sb, ",");
    sb_key(&sb, "black"); sb_i64(&sb, material_of(p, BLACK));
    SB_LIT(&sb, "}");

    SB_LIT(&sb, ",");
    sb_key(&sb, "captured");
    SB_LIT(&sb, "{");
    sb_key(&sb, "white"); sb_captured(&sb, p, WHITE);
    SB_LIT(&sb, ",");
    sb_key(&sb, "black"); sb_captured(&sb, p, BLACK);
    SB_LIT(&sb, "}");

    sb_putc(&sb, '}');
    return sb_finish(&sb);
}

/* ==========================================================================
 * model loading
 * ========================================================================== */

typedef struct {
    Trunk *trunk;
    Head  *heads;
    Hyper *hy;
    float *elo;
    int    n_agents;
    int    generation;
} ModelBlob;

static void blob_free(ModelBlob *mb)
{
    free(mb->trunk); free(mb->heads); free(mb->hy); free(mb->elo);
    memset(mb, 0, sizeof(*mb));
}

/* Peek at the on-disk ModelHeader so the head/hyper arrays can be sized exactly
 * once instead of being guessed at.  Also rejects files that are not ours, or
 * that were written by a build with different network dimensions -- model_load
 * would refuse them anyway, but failing here gives a cheap, allocation-free no. */
static int model_peek(const char *path, ModelHeader *hdr)
{
    FILE *f;
    size_t got;

    if (!path || !*path) return 0;
    f = fopen(path, "rb");
    if (!f) return 0;
    got = fread(hdr, 1, sizeof(*hdr), f);
    fclose(f);

    if (got != sizeof(*hdr))          return 0;
    if (hdr->magic   != MODEL_MAGIC)  return 0;
    if (hdr->version != MODEL_VERSION) return 0;
    if (hdr->nf_input != NF_INPUT || hdr->nf_acc  != NF_ACC ||
        hdr->nf_hid   != NF_HID   || hdr->nf_pdim != NF_PDIM) return 0;
    if (hdr->n_agents == 0 || hdr->n_agents > API_MAX_AGENTS) return 0;
    return 1;
}

static int blob_load(const char *path, ModelBlob *mb)
{
    ModelHeader hdr;
    int cap, gen = 0;

    memset(mb, 0, sizeof(*mb));
    if (!model_peek(path, &hdr)) return 0;

    cap = (int)hdr.n_agents;
    mb->trunk = (Trunk *)calloc(1, sizeof(Trunk));
    mb->heads = (Head  *)calloc((size_t)cap, sizeof(Head));
    mb->hy    = (Hyper *)calloc((size_t)cap, sizeof(Hyper));
    mb->elo   = (float *)calloc((size_t)cap, sizeof(float));
    if (!mb->trunk || !mb->heads || !mb->hy || !mb->elo) { blob_free(mb); return 0; }

    if (!model_load(path, mb->trunk, mb->heads, mb->hy, mb->elo, &cap, &gen)) {
        blob_free(mb);
        return 0;
    }
    if (cap <= 0 || cap > (int)hdr.n_agents) { blob_free(mb); return 0; }

    mb->n_agents   = cap;
    mb->generation = gen;
    return 1;
}

/* Index of the highest-rated agent; ties go to the lower index. */
static int best_agent(const float *elo, int n)
{
    int best = 0;
    for (int i = 1; i < n; i++)
        if (elo[i] > elo[best]) best = i;
    return best;
}

/* ==========================================================================
 * engines
 * ========================================================================== */

int api_engine_load(const char *model_path, int agent_index)
{
    ModelBlob mb;
    EngineSlot *e;
    int eid, pick;

    api_init();
    if (!model_path || !*model_path) return -1;
    if (strlen(model_path) >= API_MAX_PATH) return -1;
    if (!blob_load(model_path, &mb)) return -1;

    /* agent_index < 0 means "whichever agent the model rates highest". */
    pick = (agent_index < 0) ? best_agent(mb.elo, mb.n_agents) : agent_index;
    if (pick < 0 || pick >= mb.n_agents) { blob_free(&mb); return -1; }

    e = (EngineSlot *)calloc(1, sizeof(EngineSlot));
    if (!e) { blob_free(&mb); return -1; }

    e->trunk       = *mb.trunk;
    e->head        = mb.heads[pick];
    e->hyper       = mb.hy[pick];
    e->elo         = mb.elo[pick];
    e->agent_index = pick;
    e->n_agents    = mb.n_agents;
    e->generation  = mb.generation;
    snprintf(e->path, sizeof(e->path), "%s", model_path);
    blob_free(&mb);

    /* Search holds borrowed pointers into this slot, which is heap-allocated
     * and never moved, so they stay valid until api_engine_free. */
    search_init(&e->search, &e->trunk, &e->head, API_TT_MB);

    pthread_mutex_lock(&g_lock);
    eid = table_claim((void **)g_engines, API_MAX_ENGINES, &g_engine_hint);
    if (eid >= 0) g_engines[eid] = e;
    pthread_mutex_unlock(&g_lock);

    if (eid < 0) { search_free(&e->search); free(e); return -1; }
    return eid;
}

void api_engine_free(int eid)
{
    EngineSlot *e = NULL;
    if (eid < 0 || eid >= API_MAX_ENGINES) return;
    pthread_mutex_lock(&g_lock);
    e = g_engines[eid];
    g_engines[eid] = NULL;
    pthread_mutex_unlock(&g_lock);
    if (e) {
        search_free(&e->search);
        free(e);
    }
}

/* ------------------------------------------------------------- policy math */

typedef struct {
    int   idx;      /* index into the legal move list */
    float prob;
    float logit;
} PolEnt;

static int pol_cmp(const void *a, const void *b)
{
    const PolEnt *x = (const PolEnt *)a, *y = (const PolEnt *)b;
    if (x->prob > y->prob) return -1;
    if (x->prob < y->prob) return  1;
    return (x->idx < y->idx) ? -1 : (x->idx > y->idx);   /* stable tie-break */
}

/* Evaluate `p` and fill in the policy over its legal moves.
 * Returns the number of legal moves (0 when the position is terminal).
 * `pol` (if non-NULL) comes back sorted best-first. */
static int policy_of(const EngineSlot *e, const Position *p,
                     Move *list, PolEnt *pol, float *value_stm)
{
    uint16_t fidx[NF_MAXACTIVE];
    MoveKey  keys[MAX_MOVES];
    float    logits[MAX_MOVES], probs[MAX_MOVES];
    Fwd      fw;
    int      nf, n;

    n  = gen_legal(p, list);
    nf = nn_features(p, fidx);
    nn_eval(&e->trunk, &e->head, fidx, nf, &fw);
    if (value_stm) *value_stm = fw.v;
    if (n <= 0) return 0;

    /* net.h says `keys` may be NULL "in which case it is derived from p", but
     * nn_logits has no Position parameter, so the keys are always built here. */
    for (int i = 0; i < n; i++)
        nn_move_key(p, list[i], &keys[i]);

    nn_logits(&e->head, &fw, keys, n, logits);
    softmax_t(logits, n, 1.0f, probs);          /* raw policy: temperature 1 */

    if (pol) {
        for (int i = 0; i < n; i++) {
            pol[i].idx   = i;
            pol[i].prob  = isfinite(probs[i])  ? probs[i]  : 0.0f;
            pol[i].logit = isfinite(logits[i]) ? logits[i] : 0.0f;
        }
        qsort(pol, (size_t)n, sizeof(PolEnt), pol_cmp);
    }
    return n;
}

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1.0e6;
}

static int clampi(int v, int lo, int hi)
{
    return v < lo ? lo : (v > hi ? hi : v);
}

int api_engine_move(int eid, int gid, int depth, int movetime_ms,
                    char *buf, int buflen)
{
    EngineSlot *e;
    GameSlot   *s;
    Move  list[MAX_MOVES];
    PolEnt pol[MAX_MOVES];
    const Position *p;
    Move  best = MV_NONE;
    float v_stm = 0.0f;
    double t0, t1;
    int n, ntop, legal_ok = 0;
    char uci[8], san[API_SAN_LEN];
    SB sb;

    api_init();
    e = engine_get(eid);
    s = game_get(gid);
    if (!e || !s) return -1;

    /* Never touch the caller's Game: search runs against a private copy that
     * lives in the engine slot.  (search_best takes a const Game *, but the
     * copy also means a search that internally const-casts still cannot
     * corrupt the game the UI is showing.) */
    e->scratch = s->g;
    p = &e->scratch.pos;

    n = policy_of(e, p, list, pol, &v_stm);

    e->search.max_depth   = clampi(depth > 0 ? depth : API_DEF_DEPTH, 1, API_MAX_DEPTH);
    e->search.movetime_ms = clampi(movetime_ms > 0 ? movetime_ms : API_DEF_MOVETIME,
                                   1, API_MAX_MOVETIME);
    e->search.blunder_rate = 0.0f;
    e->search.stop         = 0;
    e->search.nodes        = 0;
    e->search.depth_reached = 0;
    e->search.score_cp      = 0;
    e->search.pv_len        = 0;

    t0 = now_ms();
    if (n > 0) best = search_best(&e->search, &e->scratch);
    t1 = now_ms();

    /* Defensive: only report a move the position actually allows.  If the
     * search came back empty or with something unplayable, fall back to the
     * policy's favourite, which is legal by construction. */
    for (int i = 0; i < n; i++) if (list[i] == best) { legal_ok = 1; break; }
    if (!legal_ok) best = (n > 0) ? list[pol[0].idx] : MV_NONE;

    uci[0] = '\0';
    san[0] = '\0';
    if (n > 0) {
        move_to_uci(best, uci);
        move_to_san(p, best, san, sizeof(san));
    }

    sb_init(&sb, buf, buflen);
    sb_putc(&sb, '{');
    sb_key(&sb, "move");  sb_json_str(&sb, uci);
    SB_LIT(&sb, ",");
    sb_key(&sb, "san");   sb_json_str(&sb, san);
    SB_LIT(&sb, ",");
    sb_key(&sb, "score"); sb_i64(&sb, n > 0 ? e->search.score_cp : 0);
    SB_LIT(&sb, ",");
    sb_key(&sb, "depth"); sb_i64(&sb, n > 0 ? e->search.depth_reached : 0);
    SB_LIT(&sb, ",");
    sb_key(&sb, "nodes"); sb_u64(&sb, (unsigned long long)e->search.nodes);
    SB_LIT(&sb, ",");
    sb_key(&sb, "ms");    sb_i64(&sb, (long long)(t1 - t0 + 0.5));

    /* WHITE's frame of reference -- see SIGN CONVENTIONS at the top. */
    SB_LIT(&sb, ",");
    sb_key(&sb, "value");
    sb_f(&sb, (p->side == WHITE) ? (double)v_stm : -(double)v_stm);

    SB_LIT(&sb, ",");
    sb_key(&sb, "top");
    sb_putc(&sb, '[');
    ntop = n < 3 ? n : 3;
    for (int i = 0; i < ntop; i++) {
        char tu[8], ts[API_SAN_LEN];
        Move m = list[pol[i].idx];
        if (i) sb_putc(&sb, ',');
        move_to_uci(m, tu);
        move_to_san(p, m, ts, sizeof(ts));
        SB_LIT(&sb, "{");
        sb_key(&sb, "move");  sb_json_str(&sb, tu);
        SB_LIT(&sb, ",");
        sb_key(&sb, "san");   sb_json_str(&sb, ts);
        SB_LIT(&sb, ",");
        sb_key(&sb, "prob");  sb_f(&sb, pol[i].prob);
        SB_LIT(&sb, ",");
        sb_key(&sb, "logit"); sb_f(&sb, pol[i].logit);
        SB_LIT(&sb, "}");
    }
    sb_putc(&sb, ']');
    sb_putc(&sb, '}');
    return sb_finish(&sb);
}

int api_engine_policy(int eid, int gid, char *buf, int buflen)
{
    EngineSlot *e;
    GameSlot   *s;
    Move   list[MAX_MOVES];
    PolEnt pol[MAX_MOVES];
    const Position *p;
    int n;
    SB sb;

    api_init();
    e = engine_get(eid);
    s = game_get(gid);
    if (!e || !s) return -1;

    /* Read-only: policy never mutates the game, but the copy keeps the rule
     * "the caller's Game is never touched" uniform across both engine calls. */
    e->scratch = s->g;
    p = &e->scratch.pos;

    n = policy_of(e, p, list, pol, NULL);

    sb_init(&sb, buf, buflen);
    sb_putc(&sb, '[');
    for (int i = 0; i < n; i++) {
        char uci[8], san[API_SAN_LEN];
        Move m = list[pol[i].idx];
        if (i) sb_putc(&sb, ',');
        move_to_uci(m, uci);
        move_to_san(p, m, san, sizeof(san));
        SB_LIT(&sb, "{");
        sb_key(&sb, "move");  sb_json_str(&sb, uci);
        SB_LIT(&sb, ",");
        sb_key(&sb, "san");   sb_json_str(&sb, san);
        SB_LIT(&sb, ",");
        sb_key(&sb, "prob");  sb_f(&sb, pol[i].prob);
        SB_LIT(&sb, ",");
        sb_key(&sb, "logit"); sb_f(&sb, pol[i].logit);
        SB_LIT(&sb, "}");
    }
    sb_putc(&sb, ']');
    return sb_finish(&sb);
}

/* ==========================================================================
 * model metadata
 * ========================================================================== */

typedef struct {
    int   i;
    float elo;
} AgentRank;

static int agent_cmp(const void *a, const void *b)
{
    const AgentRank *x = (const AgentRank *)a, *y = (const AgentRank *)b;
    float xe = isfinite(x->elo) ? x->elo : -1e30f;
    float ye = isfinite(y->elo) ? y->elo : -1e30f;
    if (xe > ye) return -1;
    if (xe < ye) return  1;
    return (x->i < y->i) ? -1 : (x->i > y->i);
}

/* NOTE on the return value: api.h describes this as "Number of agents in a
 * model file, and JSON metadata about them".  It follows the same TRUNCATION
 * CONTRACT as every other buffer-filling call here -- bytes written, or the
 * negative required size -- because a caller cannot retry a short buffer
 * otherwise.  n_agents is reported inside the JSON, where it is unambiguous. */
int api_model_info(const char *model_path, char *buf, int buflen)
{
    ModelBlob mb;
    AgentRank *order;
    SB sb;

    api_init();
    if (!model_path || !*model_path) return -1;
    if (!blob_load(model_path, &mb)) return -1;

    order = (AgentRank *)calloc((size_t)mb.n_agents, sizeof(AgentRank));
    if (!order) { blob_free(&mb); return -1; }
    for (int i = 0; i < mb.n_agents; i++) {
        order[i].i   = i;
        order[i].elo = mb.elo[i];
    }
    qsort(order, (size_t)mb.n_agents, sizeof(AgentRank), agent_cmp);

    sb_init(&sb, buf, buflen);
    sb_putc(&sb, '{');
    sb_key(&sb, "generation"); sb_i64(&sb, mb.generation);
    SB_LIT(&sb, ",");
    sb_key(&sb, "n_agents");   sb_i64(&sb, mb.n_agents);
    SB_LIT(&sb, ",");
    sb_key(&sb, "agents");
    sb_putc(&sb, '[');
    for (int r = 0; r < mb.n_agents; r++) {
        int i = order[r].i;
        const Hyper *h = &mb.hy[i];
        if (r) sb_putc(&sb, ',');
        SB_LIT(&sb, "{");
        sb_key(&sb, "i");            sb_i64(&sb, i);
        SB_LIT(&sb, ",");
        sb_key(&sb, "elo");          sb_f(&sb, mb.elo[i]);
        SB_LIT(&sb, ",");
        sb_key(&sb, "rank");         sb_i64(&sb, r + 1);
        SB_LIT(&sb, ",");
        sb_key(&sb, "temperature");  sb_f(&sb, h->temperature);
        SB_LIT(&sb, ",");
        sb_key(&sb, "entropy_coef"); sb_f(&sb, h->entropy_coef);
        SB_LIT(&sb, ",");
        sb_key(&sb, "lr_scale");     sb_f(&sb, h->lr_scale);
        SB_LIT(&sb, ",");
        sb_key(&sb, "shaping");      sb_f(&sb, h->shaping);
        SB_LIT(&sb, "}");
    }
    sb_putc(&sb, ']');
    sb_putc(&sb, '}');

    free(order);
    blob_free(&mb);
    return sb_finish(&sb);
}
