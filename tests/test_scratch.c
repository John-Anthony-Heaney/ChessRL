/* test_scratch.c -- the empirical half of the from-scratch contract.
 *
 * ===========================================================================
 * WHAT THIS TEST IS FOR
 * ===========================================================================
 * tools/audit_knowledge.sh greps the source for hand-coded chess knowledge.
 * Grep is easy to fool: knowledge can arrive as an innocuous-looking table, or
 * as a constant folded into an unrelated expression, or through a header.  This
 * file asks the question the other way round, and the answer is much harder to
 * fake:
 *
 *      A NETWORK THAT HAS NEVER BEEN TRAINED MUST NOT BE ABLE TO PLAY CHESS.
 *
 * If nn_init() + mcts_search() produces a competent player, then the competence
 * did not come from training, because there has not been any.  It came from
 * somewhere in the code, and that is exactly the failure docs/FROM_SCRATCH.md
 * exists to prevent.
 *
 * These numbers are therefore a FLOOR, not a target.  They are what training
 * has to climb away from.  Every threshold below is a named constant with the
 * reasoning attached, and every measurement is printed whether it passes or
 * fails -- a bare PASS would tell you nothing about how close to the line the
 * system is sitting.
 *
 * ===========================================================================
 * WHAT COUNTS AS A LEGITIMATE EDGE
 * ===========================================================================
 * An untrained agent is NOT expected to score exactly 50% against a random
 * mover.  Search alone is worth something, for one reason that has nothing to
 * do with chess knowledge: the tree reaches terminal states, and the RULES say
 * what a terminal state is worth.  A search that looks two plies ahead will
 * take a mate in one and will decline to walk into a mate in one, and a random
 * opponent offers both constantly.  That edge is earned by the search plus the
 * rules, which docs/FROM_SCRATCH.md explicitly allows, so the test asserts a
 * CEILING rather than equality, and separately measures the no-search case,
 * where that explanation is unavailable.
 *
 * The shape of the measured result is the tell.  An untrained MCTS agent beats
 * a random mover mostly by NOT LOSING: it wins the mates a random opponent
 * hands it, never walks into one itself, and draws everything else by running
 * into the ply cap.  The printed W/D/L makes that visible.  An agent that had
 * been given an evaluation would instead win nearly every game, and quickly.
 *
 * The three checks:
 *
 *   [1] value/policy sanity -- an untrained value head must be near zero and an
 *       untrained policy must be near uniform over the legal moves.
 *   [2] strength vs random -- with and without search.  Without search there is
 *       no terminal-value argument, so the bar is much tighter.
 *   [3] material blindness -- offered a free queen, the untrained agent must
 *       take it at roughly the rate a coin-flipping player would.
 *
 * ===========================================================================
 * A NOTE ON THIS FILE COUNTING PIECES
 * ===========================================================================
 * find_free_queen() below counts queens and asks square_attacked().  That is
 * the TEST's oracle, not the agent's: the test has to know what a free queen is
 * in order to check that the agent does not.  tools/audit_knowledge.sh scans
 * only src/ and never tests/, which is exactly the right boundary -- an oracle
 * that shares the agent's blind spots cannot detect them.
 *
 * Build:
 *   cc -O2 -std=c11 -D_DARWIN_C_SOURCE -Isrc \
 *      src/chess.c src/net.c src/mcts.c tests/test_scratch.c -o /tmp/tsc -lm && /tmp/tsc
 */

#include "chess.h"
#include "mcts.h"
#include "net.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ==========================================================================
 * thresholds -- every one of them argued for
 * ========================================================================== */

/* ---- [1] an untrained network's own output ------------------------------ */

/* nn_init() draws small random weights, so the value head is tanh(something
 * near zero).  "Near zero" is asserted two ways, because they fail differently:
 *
 *   MEAN  -- a non-zero mean is a systematic bias: the network prefers one side
 *            of every position before it has seen a game.  This is the one that
 *            catches a hard-coded first-move-advantage or material term, so it
 *            is tight.
 *   MAGNITUDE -- mean |v| bounds how confident it is.  An untrained net should
 *            be roughly agnostic everywhere; anything approaching 1 means it is
 *            making strong claims about positions it has never been taught. */
#define VAL_MEAN_MAX   0.05   /* |mean v| over sampled positions            */
#define VAL_ABS_MAX    0.25   /* mean |v|                                   */

/* Policy uniformity is measured as KL(policy || uniform) in nats, averaged over
 * positions.  0 is exactly uniform.  A trained policy in a sharp position runs
 * well over 1.0 nat; log(30) = 3.4 nats is a policy that has collapsed onto a
 * single move out of thirty.  0.35 nats leaves ample room for the accidental
 * asymmetry of random weights while flagging anything that has learned -- or
 * been told -- which moves matter. */
#define POL_KL_MAX     0.35

/* ---- [2] strength against a uniformly random legal mover ---------------- */

/* Four independently seeded networks so the result is a property of "untrained
 * weights", not of one lucky draw.  60 games each, colours alternating, is 240
 * games: the standard error of a score in {0, 0.5, 1} is about 0.45/sqrt(240)
 * = 0.029, so a ceiling set a few points above the measurement is a real test
 * and not a coin flip. */
#define MATCH_SEEDS      4
#define MATCH_GAMES     60    /* per seed; half as White, half as Black      */
#define MATCH_MAX_PLIES 160   /* adjudicated a draw at the cap               */

/* Simulations per move for the searching agent.  100 is a realistic play
 * budget -- the same order as the 50-100 evaluations per move the self-play
 * loop uses -- so this measures the agent as it would actually be shipped. */
#define MATCH_SIMS     100

/* THE HEADLINE CEILING.  An untrained network searching 100 simulations per
 * move may beat a random mover, because the tree finds terminal states and the
 * rules score them (see "WHAT COUNTS AS A LEGITIMATE EDGE" above).  What it may
 * NOT do is play well.  The measured value on this machine is 0.64, and it is
 * reproducible: nn_init() is seeded, the random mover is seeded, and the search
 * is deterministic at temperature 0 with root noise off, so the whole test is
 * a fixed number that only moves when the CODE moves.  0.80 sits about five
 * standard errors (0.029 per 240 games) above that, and far below what a
 * network that actually understood the game would score -- a real evaluation
 * beats a random mover ~1.00, and does it in a fraction of the moves.  If this
 * assert fires, something in the play path is evaluating positions with
 * knowledge that was not learned. */
#define MCTS_SCORE_CEILING   0.80

/* The same match with NO SEARCH: one network evaluation, play the policy
 * argmax.  Here the terminal-value argument is unavailable -- a one-ply policy
 * lookup cannot see a mate -- so an untrained policy head is a random-but-fixed
 * preference over moves and must score like a coin flip.  The measured value is
 * 0.48; 0.60 is about four standard errors above 0.50, which tolerates the
 * accidental bias of random weights while still catching a leak.  For scale:
 * the PREVIOUS version of this project scored 0.65 here WITH a fully trained
 * policy, and that result is why the system was rebuilt. */
#define POLICY_SCORE_CEILING 0.60

/* ---- [3] material blindness -------------------------------------------- */

/* Positions in which exactly one legal move captures an undefended queen. */
#define FQ_POSITIONS   24
#define FQ_SEEDS       40    /* networks; 24 x 40 = 960 independent choices  */
#define FQ_SIMS         8    /* deliberately low: barely any search at all   */
#define FQ_MIN_LEGAL   14    /* so "chance" is a small number, not 1-in-3    */

/* The agent may exceed chance by this much before the test fails.  Chance here
 * is measured, not assumed: the same positions are played by a uniformly random
 * mover and its queen-grab rate is the baseline.  A network that knew what a
 * queen was worth would take it at 80-100%; chance is around 4%.  Allowing 3x
 * chance, or chance + 10 points, is generous to noise and nowhere near
 * "understands material". */
#define FQ_RATE_FACTOR  3.0
#define FQ_RATE_MARGIN  0.10

/* Positions sampled for the value/policy statistics. */
#define STAT_POSITIONS 400
#define STAT_SEEDS       4

/* ==========================================================================
 * harness
 * ========================================================================== */

static int g_fail   = 0;
static int g_checks = 0;

#define CHECK(cond, ...)                                                      \
    do {                                                                      \
        g_checks++;                                                           \
        if (!(cond)) {                                                        \
            g_fail++;                                                         \
            fprintf(stderr, "  FAIL %s:%d: %s\n        ",                     \
                    __FILE__, __LINE__, #cond);                               \
            fprintf(stderr, __VA_ARGS__);                                     \
            fputc('\n', stderr);                                              \
        }                                                                     \
    } while (0)

static void banner(const char *s)
{
    printf("\n--- %s\n", s);
}

static double now_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* splitmix64 -- the test's own RNG, deliberately not the engine's, so the
 * "random mover" cannot be correlated with anything inside the agent. */
static uint64_t rnd(uint64_t *s)
{
    uint64_t z = (*s += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

/* Far too large for the stack. */
static Trunk g_trunk;
static Head  g_head;

/* ==========================================================================
 * players
 * ========================================================================== */

/* The move the untrained network plays with NO search: argmax of the policy
 * head over the legal moves.  Ties keep move-generation order. */
static Move policy_argmax(const Trunk *t, const Head *h, const Position *p,
                          const Move *list, int n)
{
    uint16_t fidx[NF_MAXACTIVE];
    MoveKey  keys[MAX_MOVES];
    float    logits[MAX_MOVES];
    Fwd      fw;
    int      nf, i, best = 0;

    if (n <= 0) return MV_NONE;
    nf = nn_features(p, fidx);
    nn_eval(t, h, fidx, nf, &fw);
    for (i = 0; i < n; i++) nn_move_key(p, list[i], &keys[i]);
    nn_logits(h, &fw, keys, n, logits);
    for (i = 1; i < n; i++) if (logits[i] > logits[best]) best = i;
    return list[best];
}

/* The move the shipped agent plays: MCTS, root noise off, temperature 0. */
static Move mcts_argmax(Mcts *m, const Trunk *t, const Head *h, const Game *g,
                        int sims, uint64_t *mrng, const Move *list, int n)
{
    int32_t visits[MAX_MOVES];
    float   rootv = 0.0f;
    int     nroot, pick;

    nroot = mcts_search(m, t, h, g, sims, 0 /* no root noise */, mrng,
                        visits, &rootv);
    if (nroot <= 0) return (n > 0) ? list[0] : MV_NONE;
    pick = mcts_pick(visits, nroot, 0.0f /* temperature 0 */, NULL);
    if (pick < 0 || pick >= n) pick = 0;
    return list[pick];
}

/* One game: the agent plays `agent_color`, a uniformly random legal mover plays
 * the other side.  `sims` <= 0 selects the no-search policy agent.
 * Returns GR_*; a game still running at the ply cap is adjudicated a draw, which
 * is a statement about the test's patience and not about chess. */
static int play_one(Mcts *m, int agent_color, int sims, uint64_t *rng,
                    uint64_t *mrng, int *plies_out)
{
    Game g;
    Move list[MAX_MOVES];
    int  ply = 0;

    game_start(&g);
    while (g.result == GR_ONGOING && ply < MATCH_MAX_PLIES) {
        const int n = gen_legal(&g.pos, list);
        Move mv;
        if (n <= 0) break;                     /* game_update_result handles it */

        if ((int)g.pos.side == agent_color) {
            mv = (sims > 0)
               ? mcts_argmax(m, &g_trunk, &g_head, &g, sims, mrng, list, n)
               : policy_argmax(&g_trunk, &g_head, &g.pos, list, n);
            if (mv == MV_NONE) mv = list[0];
        } else {
            mv = list[(int)(rnd(rng) % (uint64_t)n)];
        }
        game_push(&g, mv);
        ply++;
    }
    *plies_out = ply;
    return (g.result == GR_ONGOING) ? GR_DRAW : g.result;
}

/* Score of `agent_color` in a finished game: 1 win, 0.5 draw, 0 loss. */
static double score_of(int result, int agent_color)
{
    if (result == GR_DRAW) return 0.5;
    if (agent_color == WHITE) return (result == GR_WHITE_WIN) ? 1.0 : 0.0;
    return (result == GR_BLACK_WIN) ? 1.0 : 0.0;
}

/* ==========================================================================
 * [1] the untrained network's own output
 * ========================================================================== */

/* Sample positions by playing random legal moves from the start, so the
 * statistics cover openings, middlegames and endgames rather than one corner
 * of the space. */
static int sample_positions(Position *out, int want, uint64_t *rng)
{
    int got = 0;

    while (got < want) {
        Game g;
        Move list[MAX_MOVES];
        int  steps = 2 + (int)(rnd(rng) % 70), i;

        game_start(&g);
        for (i = 0; i < steps && g.result == GR_ONGOING; i++) {
            const int n = gen_legal(&g.pos, list);
            if (n <= 0) break;
            game_push(&g, list[(int)(rnd(rng) % (uint64_t)n)]);
        }
        if (g.result != GR_ONGOING) continue;
        if (gen_legal(&g.pos, list) < 2) continue;
        out[got++] = g.pos;
    }
    return got;
}

static void test_untrained_output(void)
{
    static Position pos[STAT_POSITIONS];
    uint64_t rng = 0xC0FFEE123456789ull;
    double   worst_mean = 0.0, worst_abs = 0.0, worst_kl = 0.0;
    int      s;

    banner("[1] an untrained network: value near zero, policy near uniform");
    sample_positions(pos, STAT_POSITIONS, &rng);

    for (s = 0; s < STAT_SEEDS; s++) {
        const uint64_t seed = 0x5EED0000ull + (uint64_t)s * 0x9E3779B9ull;
        double sum_v = 0.0, sum_abs = 0.0, sum_kl = 0.0, max_abs = 0.0;
        int    i, counted = 0;

        nn_init(&g_trunk, &g_head, seed);

        for (i = 0; i < STAT_POSITIONS; i++) {
            uint16_t fidx[NF_MAXACTIVE];
            MoveKey  keys[MAX_MOVES];
            float    logits[MAX_MOVES], probs[MAX_MOVES];
            Move     list[MAX_MOVES];
            Fwd      fw;
            int      nf, n, j;
            double   kl = 0.0, u;

            n  = gen_legal(&pos[i], list);
            nf = nn_features(&pos[i], fidx);
            nn_eval(&g_trunk, &g_head, fidx, nf, &fw);

            sum_v   += (double)fw.v;
            sum_abs += fabs((double)fw.v);
            if (fabs((double)fw.v) > max_abs) max_abs = fabs((double)fw.v);

            if (n < 2) continue;
            for (j = 0; j < n; j++) nn_move_key(&pos[i], list[j], &keys[j]);
            nn_logits(&g_head, &fw, keys, n, logits);
            softmax_t(logits, n, 1.0f, probs);

            /* KL(policy || uniform) = sum p log(p*n).  0 iff exactly uniform. */
            u = 1.0 / (double)n;
            for (j = 0; j < n; j++)
                if (probs[j] > 0.0f) kl += (double)probs[j] * log((double)probs[j] / u);
            sum_kl += kl;
            counted++;
        }

        {
            const double mean_v   = sum_v   / STAT_POSITIONS;
            const double mean_abs = sum_abs / STAT_POSITIONS;
            const double mean_kl  = counted ? sum_kl / counted : 0.0;

            printf("      seed %-10llu  mean v %+8.5f   mean |v| %7.5f   "
                   "max |v| %7.5f   mean KL(policy||uniform) %7.5f nats\n",
                   (unsigned long long)seed, mean_v, mean_abs, max_abs, mean_kl);

            if (fabs(mean_v) > worst_mean) worst_mean = fabs(mean_v);
            if (mean_abs > worst_abs)      worst_abs  = mean_abs;
            if (mean_kl  > worst_kl)       worst_kl   = mean_kl;
        }
    }

    printf("      over %d seeds x %d positions: worst |mean v| %.5f (ceiling %.2f), "
           "worst mean |v| %.5f (ceiling %.2f), worst KL %.5f nats (ceiling %.2f)\n",
           STAT_SEEDS, STAT_POSITIONS, worst_mean, VAL_MEAN_MAX,
           worst_abs, VAL_ABS_MAX, worst_kl, POL_KL_MAX);

    CHECK(worst_mean <= VAL_MEAN_MAX,
          "untrained value head is biased: |mean v| = %.5f > %.2f -- it prefers "
          "one side of every position before seeing a single game",
          worst_mean, VAL_MEAN_MAX);
    CHECK(worst_abs <= VAL_ABS_MAX,
          "untrained value head is confident: mean |v| = %.5f > %.2f",
          worst_abs, VAL_ABS_MAX);
    CHECK(worst_kl <= POL_KL_MAX,
          "untrained policy is not near uniform: KL = %.5f nats > %.2f -- it has "
          "an opinion about which moves matter, and nothing taught it one",
          worst_kl, POL_KL_MAX);
}

/* ==========================================================================
 * [2] strength against a uniformly random legal mover
 * ========================================================================== */

/* Returns the agent's score in [0,1] and fills in the W/D/L breakdown. */
static double run_match(Mcts *m, int sims, int *w, int *d, int *l,
                        double *avg_plies, uint64_t seed_base)
{
    double   total = 0.0, plies = 0.0;
    int      s, i;
    int      games = 0;

    *w = *d = *l = 0;

    for (s = 0; s < MATCH_SEEDS; s++) {
        const uint64_t seed = seed_base + (uint64_t)s * 0x9E3779B97F4A7C15ull;
        uint64_t rng  = 0xA5A5A5A5A5A5A5A5ull ^ seed;
        uint64_t mrng[4];
        double   sub = 0.0;

        nn_init(&g_trunk, &g_head, seed);
        mrng[0] = seed ^ 0x243F6A8885A308D3ull;
        mrng[1] = 0x13198A2E03707344ull;
        mrng[2] = 0xA4093822299F31D0ull;
        mrng[3] = 0x082EFA98EC4E6C89ull;

        for (i = 0; i < MATCH_GAMES; i++) {
            const int colour = (i & 1) ? BLACK : WHITE;
            int       ply = 0;
            const int r = play_one(m, colour, sims, &rng, mrng, &ply);
            const double sc = score_of(r, colour);

            if      (sc > 0.75) (*w)++;
            else if (sc > 0.25) (*d)++;
            else                (*l)++;
            total += sc;
            sub   += sc;
            plies += ply;
            games++;
        }
        printf("      seed %-18llu score %5.1f%%  over %d games\n",
               (unsigned long long)seed, 100.0 * sub / MATCH_GAMES, MATCH_GAMES);
    }

    *avg_plies = plies / (double)games;
    return total / (double)games;
}

static void test_vs_random(Mcts *m)
{
    int    w, d, l;
    double plies, score;

    banner("[2a] UNTRAINED + NO SEARCH (raw policy argmax) vs a random mover");
    printf("      nothing here can see a mate, so the only honest answer is a coin flip\n");
    score = run_match(m, 0, &w, &d, &l, &plies, 0x1000000ull);
    printf("      SCORE %.4f   (%d W / %d D / %d L over %d games, %.0f plies avg)"
           "   ceiling %.2f\n",
           score, w, d, l, MATCH_SEEDS * MATCH_GAMES, plies, POLICY_SCORE_CEILING);
    CHECK(score <= POLICY_SCORE_CEILING,
          "an UNTRAINED policy head scores %.4f against random moves (ceiling %.2f). "
          "Nothing has trained it, so this strength was coded in somewhere.",
          score, POLICY_SCORE_CEILING);

    banner("[2b] UNTRAINED + MCTS vs a random mover");
    printf("      %d simulations/move, root noise off, temperature 0.\n"
           "      An edge is expected and legitimate: the tree reaches terminal\n"
           "      states and the RULES score them.  Competence is not.\n",
           MATCH_SIMS);
    score = run_match(m, MATCH_SIMS, &w, &d, &l, &plies, 0x2000000ull);
    printf("      SCORE %.4f   (%d W / %d D / %d L over %d games, %.0f plies avg)"
           "   ceiling %.2f\n",
           score, w, d, l, MATCH_SEEDS * MATCH_GAMES, plies, MCTS_SCORE_CEILING);
    CHECK(score <= MCTS_SCORE_CEILING,
          "an UNTRAINED network scores %.4f against random moves (ceiling %.2f). "
          "Search over terminal values cannot explain this much; the play path "
          "is evaluating positions with knowledge it was never taught.",
          score, MCTS_SCORE_CEILING);
}

/* ==========================================================================
 * [3] material blindness
 * ========================================================================== */

/* A "free queen" position, by the TEST's oracle:
 *   - the side to move is not in check and has at least FQ_MIN_LEGAL moves;
 *   - EXACTLY ONE legal move captures an enemy queen, so "chance" is 1/n;
 *   - after that capture the capturing square is not attacked, so the queen is
 *     genuinely free rather than a trade or a trap.
 * Positions come from random play, which hangs queens constantly. */
static int is_free_queen(const Position *p, const Move *list, int n, int *grab)
{
    int i, found = -1;

    *grab = -1;
    if (n < FQ_MIN_LEGAL) return 0;
    if (in_check(p, p->side)) return 0;

    for (i = 0; i < n; i++) {
        const Move m = list[i];
        if (!MV_IS_CAPTURE(m)) continue;
        if (MV_FLAG(m) == MF_EP) continue;
        if (p->board[MV_TO(m)] != QUEEN) continue;
        if (found >= 0) return 0;              /* more than one grabber */
        found = i;
    }
    if (found < 0) return 0;

    {
        Position q = *p;
        Undo u;
        int  safe;
        make_move(&q, list[found], &u);
        safe = !square_attacked(&q, MV_TO(list[found]), q.side);
        unmake_move(&q, list[found], &u);
        if (!safe) return 0;                   /* defended: not a free queen */
    }

    *grab = found;
    return 1;
}

static int find_free_queens(Position *out, int *grab_idx, int *nlegal,
                            int want, uint64_t *rng)
{
    int got = 0;
    long tries = 0;

    while (got < want && tries < 2000000L) {
        Game g;
        Move list[MAX_MOVES];
        int  steps = 8 + (int)(rnd(rng) % 60), i;

        tries++;
        game_start(&g);
        for (i = 0; i < steps && g.result == GR_ONGOING; i++) {
            int n = gen_legal(&g.pos, list), grab;
            if (n <= 0) break;
            if (is_free_queen(&g.pos, list, n, &grab)) {
                out[got]      = g.pos;
                grab_idx[got] = grab;
                nlegal[got]   = n;
                got++;
                break;
            }
            game_push(&g, list[(int)(rnd(rng) % (uint64_t)n)]);
        }
    }
    return got;
}

static void test_material_blindness(Mcts *m)
{
    static Position pos[FQ_POSITIONS];
    int      grab[FQ_POSITIONS], nleg[FQ_POSITIONS];
    uint64_t rng = 0xBADC0FFEE0DDF00Dull;
    uint64_t mrng[4] = { 0x243F6A8885A308D3ull, 0x13198A2E03707344ull,
                         0xA4093822299F31D0ull, 0x082EFA98EC4E6C89ull };
    int      npos, s, i, took = 0, trials = 0;
    double   chance = 0.0, ceiling;
    char     fen[128];

    banner("[3] material blindness: offered a free queen, does it take it?");

    npos = find_free_queens(pos, grab, nleg, FQ_POSITIONS, &rng);
    CHECK(npos == FQ_POSITIONS, "only built %d of %d free-queen positions",
          npos, FQ_POSITIONS);
    if (npos <= 0) return;

    {
        int lo = nleg[0], hi = nleg[0];
        for (i = 0; i < npos; i++) {
            chance += 1.0 / (double)nleg[i];
            if (nleg[i] < lo) lo = nleg[i];
            if (nleg[i] > hi) hi = nleg[i];
        }
        chance /= (double)npos;
        pos_to_fen(&pos[0], fen, sizeof fen);
        printf("      %d positions, %d..%d legal moves each; example: %s\n",
               npos, lo, hi, fen);
    }
    printf("      a uniformly random legal mover takes the queen %.2f%% of the "
           "time (mean of 1/n)\n", 100.0 * chance);
    printf("      the agent searches %d simulations -- barely any search at all\n",
           FQ_SIMS);

    for (s = 0; s < FQ_SEEDS; s++) {
        const uint64_t seed = 0x7A55EDull + (uint64_t)s * 0x9E3779B97F4A7C15ull;
        int seed_took = 0;

        nn_init(&g_trunk, &g_head, seed);
        for (i = 0; i < npos; i++) {
            Game g;
            Move list[MAX_MOVES];
            int  n;
            Move chosen;

            memset(&g, 0, sizeof g);
            g.pos      = pos[i];
            g.hist[0]  = pos[i].key;
            g.hist_len = 1;
            g.result   = GR_ONGOING;
            g.reason   = TR_NONE;

            n = gen_legal(&g.pos, list);
            chosen = mcts_argmax(m, &g_trunk, &g_head, &g, FQ_SIMS, mrng, list, n);
            if (chosen == list[grab[i]]) { took++; seed_took++; }
            trials++;
        }
        if (s < 4 || seed_took > npos / 4)
            printf("      seed %-20llu took the queen %2d/%d\n",
                   (unsigned long long)seed, seed_took, npos);
    }

    ceiling = FQ_RATE_FACTOR * chance;
    if (chance + FQ_RATE_MARGIN > ceiling) ceiling = chance + FQ_RATE_MARGIN;

    printf("      RATE %.4f  (%d of %d choices)   chance %.4f   ceiling %.4f\n",
           (double)took / (double)trials, took, trials, chance, ceiling);
    printf("      for scale: an agent that knows a queen is worth something\n"
           "      takes it at 0.80-1.00 here.  A rate at or BELOW chance, as\n"
           "      measured, is what blindness looks like.\n");

    CHECK((double)took / (double)trials <= ceiling,
          "an UNTRAINED network takes a free queen %.1f%% of the time against a "
          "chance rate of %.1f%%. Nothing in this system has been told what a "
          "queen is worth, so it should not be able to tell.",
          100.0 * (double)took / (double)trials, 100.0 * chance);
}

/* ========================================================================== */

int main(void)
{
    Mcts   m;
    double t0;

    chess_init();

    printf("=== from-scratch floor ===\n");
    printf("These are the numbers an UNTRAINED network produces.  Training has to\n"
           "climb away from them; if it starts above them, it did not start from\n"
           "scratch.  See docs/FROM_SCRATCH.md.\n");

    /* One simulation expands at most one node, adding one child per legal move;
     * 64 is a generous bound on the branch factor, so the pool never runs dry
     * and no measurement is taken on a degraded search. */
    mcts_init(&m, 1 + MATCH_SIMS * 64);
    if (!m.pool) {
        fprintf(stderr, "FATAL: cannot allocate the MCTS node pool\n");
        return 2;
    }

    t0 = now_s();
    test_untrained_output();
    test_vs_random(&m);
    test_material_blindness(&m);
    mcts_free(&m);

    printf("\n=== %d checks, %d failures, %.1f s ===\n",
           g_checks, g_fail, now_s() - t0);
    return g_fail ? 1 : 0;
}
