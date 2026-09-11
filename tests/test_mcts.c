/* test_mcts.c -- correctness tests for the PUCT search in src/mcts.c.
 *
 * The tests that matter most are the SIGN tests.  A search whose backup does
 * not negate at every ply plays for its opponent, and the failure is easy to
 * miss because a randomly initialised network produces plausible-looking output
 * either way.  So the sign is pinned down from both colours, with forced mates
 * where the correct answer is known from the rules alone.
 *
 * The other theme is the from-scratch contract (docs/FROM_SCRATCH.md): with a
 * RANDOMLY INITIALISED network the search must still find a mate in 1 (search
 * plus terminal values can do that unaided) and must NOT grab free material,
 * because nothing in the system knows what material is.
 *
 * Build:
 *   cc -O2 -std=c11 -D_DARWIN_C_SOURCE -Isrc \
 *      src/chess.c src/net.c src/mcts.c tests/test_mcts.c -o /tmp/tm -lm && /tmp/tm
 */

#include "chess.h"
#include "net.h"
#include "mcts.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Not in mcts.h (headers are fixed); defined in mcts.c.  See the report. */
uint64_t mcts_pool_exhausted(const Mcts *m);

/* ------------------------------------------------------------ harness */

static int g_fail = 0;
static int g_checks = 0;

#define CHECK(cond, ...) do {                                                 \
        g_checks++;                                                           \
        if (!(cond)) {                                                        \
            g_fail++;                                                         \
            fprintf(stderr, "  FAIL %s:%d: ", __FILE__, __LINE__);            \
            fprintf(stderr, __VA_ARGS__);                                     \
            fputc('\n', stderr);                                              \
        }                                                                     \
    } while (0)

static void banner(const char *s) { printf("\n--- %s\n", s); }

/* One network, reinitialised per test with an explicit seed.  Both structs are
 * far too large for the stack. */
static Trunk g_trunk;
static Head  g_head;

static void net_seed(uint64_t seed) { nn_init(&g_trunk, &g_head, seed); }

/* xoshiro256** seeding, identical to arena.c's rng_seed(). */
static void seed_rng(uint64_t *s, uint64_t seed)
{
    uint64_t x = seed;
    for (int i = 0; i < 4; i++) {
        uint64_t z = (x += 0x9E3779B97F4A7C15ULL);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        s[i] = z ^ (z >> 31);
    }
}

static void load(Game *g, const char *fen)
{
    if (!game_start_fen(g, fen)) { fprintf(stderr, "BAD FEN: %s\n", fen); exit(2); }
}

/* Index of `uci` in gen_legal()'s list -- the same order mcts_search uses. */
static int idx_of(const Position *p, const char *uci)
{
    Move l[MAX_MOVES];
    char buf[8];
    const int n = gen_legal(p, l);
    for (int i = 0; i < n; i++) {
        move_to_uci(l[i], buf);
        if (!strcmp(buf, uci)) return i;
    }
    return -1;
}

static int argmax_i(const int32_t *v, int n)
{
    int b = 0;
    for (int i = 1; i < n; i++) if (v[i] > v[b]) b = i;
    return b;
}

static int32_t sum_i(const int32_t *v, int n)
{
    int32_t s = 0;
    for (int i = 0; i < n; i++) s += v[i];
    return s;
}

static const MctsNode *root_child(const Mcts *m, int i)
{
    return &m->pool[m->pool[0].first + i];
}

static double now_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1.0e-9 * (double)ts.tv_nsec;
}

/* ------------------------------------------------------ 1. sign: mate in 1 */

static void t_mate_in_1(const char *name, const char *fen, const char *mate_uci)
{
    Game g; load(&g, fen);
    const int mi = idx_of(&g.pos, mate_uci);
    CHECK(mi >= 0, "%s: %s is not legal", name, mate_uci);

    Mcts m; mcts_init(&m, 200000);
    int32_t v[MAX_MOVES]; float rv = 0.0f;
    uint64_t rng[4]; seed_rng(rng, 7u);

    const int n = mcts_search(&m, &g_trunk, &g_head, &g, 400, 0, rng, v, &rv);
    CHECK(n > 0, "%s: search returned %d", name, n);

    const int best = argmax_i(v, n);
    printf("  %-22s n=%2d  mate idx=%d visits=%d/%d  best=%d  root_value=%+.4f\n",
           name, n, mi, v[mi], sum_i(v, n), best, (double)rv);

    CHECK(best == mi, "%s: MCTS chose index %d, not the mate at %d", name, best, mi);
    CHECK(v[mi] > (int32_t)(0.60 * 400), "%s: mate got only %d/400 visits", name, v[mi]);
    /* The mating child is terminal for the opponent: -1 from ITS mover's view. */
    CHECK(root_child(&m, mi)->terminal == TR_CHECKMATE, "%s: mate child not TR_CHECKMATE", name);
    CHECK(root_child(&m, mi)->tval == -1.0f, "%s: mate child tval %+.3f, want -1",
          name, (double)root_child(&m, mi)->tval);
    /* ... and therefore +1 from the root's view: a sign flip shows up here. */
    CHECK(rv > 0.80f, "%s: root value %+.4f, want ~+1", name, (double)rv);

    mcts_free(&m);
}

/* ------------------------------- 1b. sign: do not hand over a mate in 1 */

static void t_avoid_losing(const char *name, const char *fen,
                           const char **losers, int nlosers, int sims)
{
    Game g; load(&g, fen);

    int li[16];
    for (int i = 0; i < nlosers; i++) {
        li[i] = idx_of(&g.pos, losers[i]);
        CHECK(li[i] >= 0, "%s: %s is not legal", name, losers[i]);
    }

    Mcts m; mcts_init(&m, 1000000);
    int32_t v[MAX_MOVES]; float rv = 0.0f;
    uint64_t rng[4]; seed_rng(rng, 99u);

    const int n = mcts_search(&m, &g_trunk, &g_head, &g, sims, 0, rng, v, &rv);
    CHECK(n > 0, "%s: search returned %d", name, n);

    int32_t lost = 0;
    for (int i = 0; i < nlosers; i++) lost += v[li[i]];
    const int best = argmax_i(v, n);

    int best_is_loser = 0;
    for (int i = 0; i < nlosers; i++) if (best == li[i]) best_is_loser = 1;

    printf("  %-22s n=%2d  visits to the %d self-mating moves: %d/%d (%.1f%%)  "
           "best=%d%s  root_value=%+.4f\n",
           name, n, nlosers, lost, sims, 100.0 * (double)lost / (double)sims,
           best, best_is_loser ? " <-- LOSER" : "", (double)rv);

    CHECK(!best_is_loser, "%s: MCTS preferred a move that allows mate in 1", name);
    CHECK((double)lost < 0.35 * (double)sims,
          "%s: %.1f%% of visits went to moves that allow mate in 1",
          name, 100.0 * (double)lost / (double)sims);

    mcts_free(&m);
}

/* ------------------------------------------- 2. root values at +/-1 */

static void t_all_moves_lose(void)
{
    /* White to move, two legal moves, both allow ...Qg2#.  Seeing this needs
     * three plies and a correctly alternating backup. */
    Game g; load(&g, "8/8/1k6/8/8/4n1q1/P7/7K w - - 0 1");

    Mcts m; mcts_init(&m, 1000000);
    int32_t v[MAX_MOVES]; float rv = 0.0f;
    uint64_t rng[4]; seed_rng(rng, 3u);

    const int n = mcts_search(&m, &g_trunk, &g_head, &g, 4000, 0, rng, v, &rv);
    printf("  every move loses:      n=%d  root_value=%+.4f (want ~-1)\n", n, (double)rv);
    CHECK(n == 2, "expected 2 legal moves, got %d", n);
    CHECK(rv < -0.60f, "root value %+.4f, want ~-1", (double)rv);
    mcts_free(&m);
}

/* --------------------------------------- 3. every drawn termination is 0 */

static void t_draw_terminals(void)
{
    Mcts m; mcts_init(&m, 500000);
    int32_t v[MAX_MOVES]; float rv = 0.0f;
    uint64_t rng[4]; seed_rng(rng, 11u);

    /* (a) stalemate.  Rg2-g1 leaves Black with no move and no check.  White is
     *     winning here (K+R v K), which is the point: a search that scored
     *     stalemate as a win would throw the game away by playing it. */
    {
        Game g; load(&g, "7k/8/7K/8/8/8/6R1/8 w - - 0 1");
        const int si = idx_of(&g.pos, "g2g1");
        const int n = mcts_search(&m, &g_trunk, &g_head, &g, 4000, 0, rng, v, &rv);
        CHECK(si >= 0 && si < n, "stalemate move g2g1 missing");
        const MctsNode *c = root_child(&m, si);
        const int best = argmax_i(v, n);
        printf("  stalemate child:       terminal=%d tval=%+.1f W=%+.1f N=%d of %d"
               "  (best move is index %d)  root_value=%+.4f\n",
               c->terminal, (double)c->tval, (double)c->W, c->N, 4000, best, (double)rv);
        CHECK(c->N > 0, "stalemate child was never visited");
        CHECK(c->terminal == TR_STALEMATE, "stalemate child terminal=%d, want %d",
              c->terminal, TR_STALEMATE);
        CHECK(c->tval == 0.0f, "stalemate tval %+.3f, want 0 (a draw, not a win)",
              (double)c->tval);
        CHECK(c->W == 0.0f, "stalemate child W %+.3f, want exactly 0", (double)c->W);
        CHECK(best != si, "MCTS preferred the stalemate: it is scoring a draw as a win");
        CHECK(c->N < v[best], "stalemate got %d visits vs %d for the best move",
              c->N, v[best]);
    }

    /* (b) insufficient material.  Kxg2 is the only legal move and leaves K v K. */
    {
        Game g; load(&g, "7k/8/8/8/8/8/6r1/7K w - - 0 1");
        const int n = mcts_search(&m, &g_trunk, &g_head, &g, 512, 0, rng, v, &rv);
        CHECK(n == 1, "expected 1 legal move, got %d", n);
        const MctsNode *c = root_child(&m, 0);
        printf("  K v K child:           terminal=%d tval=%+.1f W=%+.1f N=%d  root_value=%+.4f\n",
               c->terminal, (double)c->tval, (double)c->W, c->N, (double)rv);
        CHECK(c->terminal == TR_INSUFFICIENT, "child terminal=%d, want %d",
              c->terminal, TR_INSUFFICIENT);
        CHECK(c->tval == 0.0f, "insufficient-material tval %+.3f, want 0", (double)c->tval);
        CHECK(c->N == 512, "forced move got %d/512 visits", c->N);
        CHECK(fabs((double)rv) < 0.02, "root value %+.4f, want ~0", (double)rv);
    }

    /* (c) fifty-move rule: halfmove 99, so every reply is the 100th ply. */
    {
        Game g; load(&g, "3k4/8/8/8/8/8/8/3K1R2 w - - 99 60");
        const int n = mcts_search(&m, &g_trunk, &g_head, &g, 512, 0, rng, v, &rv);
        CHECK(n > 0, "fifty-move root returned %d", n);
        int all_fifty = 1;
        for (int i = 0; i < n; i++) {
            const MctsNode *c = root_child(&m, i);
            if (c->N > 0 && (c->terminal != TR_FIFTY || c->tval != 0.0f)) all_fifty = 0;
        }
        printf("  fifty-move children:   n=%d all TR_FIFTY with tval 0: %s  root_value=%+.4f\n",
               n, all_fifty ? "yes" : "NO", (double)rv);
        CHECK(all_fifty, "a visited child was not a TR_FIFTY draw worth 0");
        CHECK(fabs((double)rv) < 0.02, "root value %+.4f, want ~0", (double)rv);
    }

    /* (d) threefold: the history already holds the position twice, so the
     *     search must count history + path, not the path alone. */
    {
        Game g; load(&g, "3k4/8/8/8/8/8/8/3K1R2 w - - 0 1");
        const char *seq[] = { "f1f2","d8d7","f2f1","d7d8","f1f2","d8d7","f2f1" };
        for (int i = 0; i < 7; i++) {
            Move mv;
            CHECK(move_from_uci(&g.pos, seq[i], &mv) != 0, "illegal setup move %s", seq[i]);
            game_push(&g, mv);
        }
        CHECK(g.result == GR_ONGOING, "setup already ended: result=%d", g.result);

        const int ri = idx_of(&g.pos, "d7d8");
        const int n = mcts_search(&m, &g_trunk, &g_head, &g, 1024, 0, rng, v, &rv);
        CHECK(ri >= 0 && ri < n, "repeating move d7d8 missing");
        const MctsNode *c = root_child(&m, ri);
        printf("  threefold child:       terminal=%d tval=%+.1f W=%+.1f N=%d\n",
               c->terminal, (double)c->tval, (double)c->W, c->N);
        CHECK(c->N > 0, "repeating child was never visited");
        CHECK(c->terminal == TR_REPETITION, "child terminal=%d, want %d (history+path)",
              c->terminal, TR_REPETITION);
        CHECK(c->tval == 0.0f, "repetition tval %+.3f, want 0", (double)c->tval);
    }

    mcts_free(&m);
}

/* ----------------------------------------- 4. a finished game returns 0 */

static void t_game_over(void)
{
    const char *fens[] = {
        "6k1/5ppp/8/8/8/8/8/R5K1 b - - 0 1",   /* not over: sanity, must be > 0 */
        "R5k1/5ppp/8/8/8/8/8/6K1 b - - 0 1",   /* checkmate                     */
        "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1",      /* stalemate                     */
        "7k/8/6K1/8/8/8/8/8 w - - 0 1",        /* K v K, insufficient           */
    };
    const int want_zero[] = { 0, 1, 1, 1 };

    Mcts m; mcts_init(&m, 20000);
    int32_t v[MAX_MOVES]; float rv;
    uint64_t rng[4]; seed_rng(rng, 5u);

    for (int i = 0; i < 4; i++) {
        Game g; load(&g, fens[i]);
        const int n = mcts_search(&m, &g_trunk, &g_head, &g, 64, 0, rng, v, &rv);
        printf("  %-40s -> n=%d\n", fens[i], n);
        if (want_zero[i]) CHECK(n == 0, "finished game returned %d moves", n);
        else              CHECK(n  > 0, "live position returned 0 moves");
    }
    mcts_free(&m);
}

/* ------------------------------------------------------ 5. determinism */

static void t_determinism(void)
{
    Game g; load(&g, "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4");

    int32_t a[MAX_MOVES], b[MAX_MOVES];
    float ra = 0.0f, rb = 0.0f;
    uint64_t r1[4], r2[4];
    seed_rng(r1, 424242u);
    seed_rng(r2, 424242u);

    Mcts m1; mcts_init(&m1, 100000);
    Mcts m2; mcts_init(&m2, 100000);
    const int n1 = mcts_search(&m1, &g_trunk, &g_head, &g, 600, 0, r1, a, &ra);
    const int n2 = mcts_search(&m2, &g_trunk, &g_head, &g, 600, 0, r2, b, &rb);

    int same = (n1 == n2);
    for (int i = 0; i < n1 && same; i++) if (a[i] != b[i]) same = 0;
    printf("  two fresh searches identical: %s  (n=%d, root_value %+.6f / %+.6f)\n",
           same ? "yes" : "NO", n1, (double)ra, (double)rb);
    CHECK(same, "same seed, same position, no noise -> different visit counts");
    CHECK(ra == rb, "root values differ: %+.8f vs %+.8f", (double)ra, (double)rb);

    /* Re-using one Mcts must give the same answer as a fresh one: the tree is
     * rebuilt from scratch every search, and it must not creep up the pool. */
    const int used1 = m1.used;
    const int n3 = mcts_search(&m1, &g_trunk, &g_head, &g, 600, 0, r1, b, &rb);
    same = (n3 == n1);
    for (int i = 0; i < n1 && same; i++) if (a[i] != b[i]) same = 0;
    printf("  re-used Mcts identical:       %s   (pool used %d -> %d, exhausted %llu)\n",
           same ? "yes" : "NO", used1, m1.used,
           (unsigned long long)mcts_pool_exhausted(&m1));
    CHECK(same, "re-using an Mcts changed the result (stale tree?)");
    CHECK(m1.used == used1, "re-using an Mcts leaked pool nodes: %d -> %d", used1, m1.used);
    CHECK(mcts_pool_exhausted(&m1) == 0, "a 100k-node pool was exhausted by 600 sims");

    mcts_free(&m1);
    mcts_free(&m2);
}

/* -------------------------------------------------- 6. Dirichlet noise */

static void t_dirichlet(void)
{
    Game g; load(&g, "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4");

    /* the reference: the plain softmax of the policy head over the legal moves */
    Move l[MAX_MOVES];
    const int n = gen_legal(&g.pos, l);
    uint16_t fidx[NF_MAXACTIVE];
    const int nf = nn_features(&g.pos, fidx);
    Fwd fw; nn_eval(&g_trunk, &g_head, fidx, nf, &fw);
    MoveKey mk[MAX_MOVES];
    for (int i = 0; i < n; i++) nn_move_key(&g.pos, l[i], &mk[i]);
    float logits[MAX_MOVES], ref[MAX_MOVES];
    nn_logits(&g_head, &fw, mk, n, logits);
    softmax_t(logits, n, 1.0f, ref);

    int32_t v[MAX_MOVES]; float rv;
    uint64_t rng[4];
    Mcts m; mcts_init(&m, 100000);

    /* eps = 0 must leave the priors bit-for-bit equal to the softmax. */
    m.dirichlet_eps = 0.0f;
    seed_rng(rng, 1234u);
    int n0 = mcts_search(&m, &g_trunk, &g_head, &g, 0, 1, rng, v, &rv);
    int exact = (n0 == n);
    for (int i = 0; i < n && exact; i++) if (root_child(&m, i)->P != ref[i]) exact = 0;
    printf("  eps=0 priors == softmax (exact): %s\n", exact ? "yes" : "NO");
    CHECK(exact, "eps=0 changed the priors");

    /* noise on: priors move, the legal move set does not. */
    mcts_defaults(&m);
    seed_rng(rng, 1234u);
    const int n1 = mcts_search(&m, &g_trunk, &g_head, &g, 0, 1, rng, v, &rv);
    CHECK(n1 == n, "noise changed the number of root moves: %d vs %d", n1, n);

    float sum = 0.0f, maxdiff = 0.0f;
    int moves_same = 1, all_nonneg = 1;
    for (int i = 0; i < n1; i++) {
        const MctsNode *c = root_child(&m, i);
        sum += c->P;
        const float d = fabsf(c->P - ref[i]);
        if (d > maxdiff) maxdiff = d;
        if (c->move != l[i]) moves_same = 0;
        if (!(c->P >= 0.0f)) all_nonneg = 0;
    }
    printf("  noise on: sum(P)=%.6f  max|P-softmax|=%.4f  same moves: %s\n",
           (double)sum, (double)maxdiff, moves_same ? "yes" : "NO");
    CHECK(maxdiff > 1.0e-4f, "Dirichlet noise did not change the priors");
    CHECK(moves_same, "Dirichlet noise altered the legal move list");
    CHECK(all_nonneg, "a prior went negative under noise");
    CHECK(fabs((double)sum - 1.0) < 1.0e-3, "priors sum to %.6f, want 1", (double)sum);

    /* different draws from different states */
    seed_rng(rng, 777u);
    mcts_search(&m, &g_trunk, &g_head, &g, 0, 1, rng, v, &rv);
    float p_a[MAX_MOVES];
    for (int i = 0; i < n1; i++) p_a[i] = root_child(&m, i)->P;
    seed_rng(rng, 778u);
    mcts_search(&m, &g_trunk, &g_head, &g, 0, 1, rng, v, &rv);
    int differ = 0;
    for (int i = 0; i < n1; i++) if (p_a[i] != root_child(&m, i)->P) differ = 1;
    CHECK(differ, "two different RNG states produced identical noise");

    /* and noise OFF must be the plain softmax again */
    seed_rng(rng, 777u);
    mcts_search(&m, &g_trunk, &g_head, &g, 0, 0, rng, v, &rv);
    exact = 1;
    for (int i = 0; i < n1; i++) if (root_child(&m, i)->P != ref[i]) exact = 0;
    printf("  root_noise=0 priors == softmax (exact): %s\n", exact ? "yes" : "NO");
    CHECK(exact, "root_noise=0 still perturbed the priors");

    mcts_free(&m);
}

/* ------------------------------------------------------- 7. visit sums */

static void t_visit_sums(void)
{
    Game g; load(&g, "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4");
    const int sims[] = { 0, 1, 2, 7, 64, 400, 1000 };

    Mcts m; mcts_init(&m, 200000);
    int32_t v[MAX_MOVES]; float rv;
    uint64_t rng[4]; seed_rng(rng, 8u);

    printf("  convention: the root is expanded once outside the loop (root.N = 1),\n"
           "              then each simulation passes through exactly one root child,\n"
           "              so sum(child visits) == sims and root.N == sims + 1.\n");

    for (size_t k = 0; k < sizeof sims / sizeof sims[0]; k++) {
        const int s = sims[k];
        const int n = mcts_search(&m, &g_trunk, &g_head, &g, s, 0, rng, v, &rv);
        const int32_t tot = sum_i(v, n);
        const int32_t rn  = m.pool[0].N;
        printf("    sims=%4d  sum(visits)=%4d  root.N=%4d\n", s, tot, rn);
        CHECK(tot == s, "sims=%d but visits sum to %d", s, tot);
        CHECK(rn == s + 1, "sims=%d but root.N=%d, want %d", s, rn, s + 1);
    }

    /* mcts_target normalises those counts. */
    const int n = mcts_search(&m, &g_trunk, &g_head, &g, 800, 0, rng, v, &rv);
    float tgt[MAX_MOVES];
    mcts_target(v, n, tgt);
    double tsum = 0.0, worst = 0.0;
    for (int i = 0; i < n; i++) {
        tsum += tgt[i];
        const double want = (double)v[i] / 800.0;
        const double d = fabs(want - (double)tgt[i]);
        if (d > worst) worst = d;
    }
    printf("  mcts_target: sum=%.7f  max error vs N/sum=%.3e\n", tsum, worst);
    CHECK(fabs(tsum - 1.0) < 1.0e-5, "policy target sums to %.7f", tsum);
    CHECK(worst < 1.0e-6, "policy target is not visits/sum (err %.3e)", worst);

    /* degenerate input: a uniform target rather than a divide by zero */
    const int32_t zeros[4] = { 0, 0, 0, 0 };
    mcts_target(zeros, 4, tgt);
    CHECK(fabsf(tgt[0] - 0.25f) < 1e-6f, "all-zero visits should give a uniform target");

    mcts_free(&m);
}

/* --------------------------------------------- 8. mcts_pick temperature */

static void t_pick(void)
{
    uint64_t rng[4]; seed_rng(rng, 20240816u);

    /* temp <= 0 is the argmax, always. */
    {
        const int32_t v[5] = { 5, 100, 3, 100 - 1, 0 };
        int ok = 1;
        for (int i = 0; i < 2000; i++) if (mcts_pick(v, 5, 0.0f, rng) != 1) ok = 0;
        for (int i = 0; i < 2000; i++) if (mcts_pick(v, 5, -1.0f, rng) != 1) ok = 0;
        printf("  temp=0 always argmax: %s\n", ok ? "yes" : "NO");
        CHECK(ok, "temp<=0 did not always return the argmax");
    }

    /* temp=1 samples proportionally to N.  chi-squared, 2 d.o.f. */
    {
        const int32_t v[3] = { 5, 100, 3 };
        const int N = 30000;
        int cnt[3] = { 0, 0, 0 };
        for (int i = 0; i < N; i++) {
            const int k = mcts_pick(v, 3, 1.0f, rng);
            CHECK(k >= 0 && k < 3, "mcts_pick returned %d", k);
            cnt[k]++;
        }
        const double tot = 108.0;
        double chi2 = 0.0;
        for (int i = 0; i < 3; i++) {
            const double e = (double)N * (double)v[i] / tot;
            const double d = (double)cnt[i] - e;
            chi2 += d * d / e;
        }
        printf("  temp=1  counts %d/%d/%d of %d  chi2=%.2f (df=2, 0.1%% crit 13.8)\n",
               cnt[0], cnt[1], cnt[2], N, chi2);
        CHECK(chi2 < 13.8, "temp=1 sampling is not proportional to N (chi2=%.2f)", chi2);
    }

    /* a large temperature approaches uniform, even for very unequal counts. */
    {
        const int32_t v[4] = { 1, 2, 3, 400 };
        const int N = 20000;
        int cnt[4] = { 0, 0, 0, 0 };
        for (int i = 0; i < N; i++) cnt[mcts_pick(v, 4, 50.0f, rng)]++;
        double lo = 1.0, hi = 0.0;
        for (int i = 0; i < 4; i++) {
            const double f = (double)cnt[i] / (double)N;
            if (f < lo) lo = f;
            if (f > hi) hi = f;
        }
        printf("  temp=50 on {1,2,3,400}: frequencies %.3f .. %.3f (uniform = 0.250)\n", lo, hi);
        CHECK(lo > 0.20 && hi < 0.30, "temp=50 is not close to uniform (%.3f..%.3f)", lo, hi);
    }

    /* uniform counts stay uniform: chi-squared, 3 d.o.f. */
    {
        const int32_t v[4] = { 10, 10, 10, 10 };
        const int N = 20000;
        int cnt[4] = { 0, 0, 0, 0 };
        for (int i = 0; i < N; i++) cnt[mcts_pick(v, 4, 1.0f, rng)]++;
        double chi2 = 0.0;
        const double e = (double)N / 4.0;
        for (int i = 0; i < 4; i++) { const double d = cnt[i] - e; chi2 += d * d / e; }
        printf("  temp=1 on equal counts: chi2=%.2f (df=3, 0.1%% crit 16.3)\n", chi2);
        CHECK(chi2 < 16.3, "equal visit counts did not sample uniformly (chi2=%.2f)", chi2);
    }

    /* never visited -> still a legal index; single move -> that move */
    {
        const int32_t z[3] = { 0, 0, 0 };
        int ok = 1;
        for (int i = 0; i < 500; i++) { const int k = mcts_pick(z, 3, 1.0f, rng); if (k < 0 || k > 2) ok = 0; }
        CHECK(ok, "mcts_pick on all-zero visits returned an out-of-range index");
        const int32_t one[1] = { 7 };
        CHECK(mcts_pick(one, 1, 1.0f, rng) == 0, "single move not returned");
    }
}

/* ------------------------------------------- 9. no chess knowledge leak */

static void t_no_knowledge_leak(void)
{
    /* (a) a random network still finds mate in 1: that comes from the terminal
     *     values, which are rules, not knowledge. */
    {
        Game g; load(&g, "6k1/5ppp/8/8/8/8/8/R6K w - - 0 1");
        const int mi = idx_of(&g.pos, "a1a8");
        Mcts m; mcts_init(&m, 500000);
        int32_t v[MAX_MOVES]; float rv;
        int found = 0;
        for (int s = 0; s < 6; s++) {
            net_seed(0xABCDEF00ull + (uint64_t)s * 7919u);
            uint64_t rng[4]; seed_rng(rng, 100u + (uint64_t)s);
            const int n = mcts_search(&m, &g_trunk, &g_head, &g, 2000, 0, rng, v, &rv);
            if (argmax_i(v, n) == mi) found++;
        }
        printf("  random-net mate in 1 found: %d/6 seeds\n", found);
        CHECK(found == 6, "a randomly initialised net failed to find mate in 1 (%d/6)", found);
        mcts_free(&m);
    }

    /* (b) but it must NOT hoover up free material.  Four quiet positions, each
     *     with exactly one capture, each capture free (nothing recaptures).
     *     A material-driven agent takes them every time; a search that knows
     *     nothing about material takes them at roughly the base rate. */
    {
        const char *fens[4] = {
            "4k3/8/8/3q4/8/8/8/3RK3 w - - 0 1",              /* Rxd5, 1/10 by chance  */
            "3rk3/8/8/8/3Q4/8/8/4K3 b - - 0 1",              /* ...Rxd4, black to move */
            "4k3/8/n7/8/8/8/8/4KB2 w - - 0 1",               /* Bxa6, 1/11            */
            "4k3/pp3ppp/8/8/8/2r5/PP3PPP/3RK3 w - - 0 1",    /* bxc3, 1/24            */
        };
        const char *caps[4] = { "d1d5", "d8d4", "f1a6", "b2c3" };

        Mcts m; mcts_init(&m, 500000);
        int32_t v[MAX_MOVES]; float rv;
        int grabs = 0, trials = 0;
        double chance = 0.0;

        for (int p = 0; p < 4; p++) {
            Game g; load(&g, fens[p]);
            const int ci = idx_of(&g.pos, caps[p]);
            CHECK(ci >= 0, "capture %s is not legal", caps[p]);
            int local = 0;
            for (int s = 0; s < 10; s++) {
                net_seed(0x5EED0000ull + (uint64_t)(p * 101 + s) * 2654435761u);
                uint64_t rng[4]; seed_rng(rng, 4000u + (uint64_t)s);
                const int n = mcts_search(&m, &g_trunk, &g_head, &g, 512, 0, rng, v, &rv);
                chance += 1.0 / (double)n;
                if (argmax_i(v, n) == ci) { local++; grabs++; }
                trials++;
            }
            printf("  %-46s free capture chosen %2d/10\n", fens[p], local);
        }
        chance /= (double)trials;
        printf("  free material grabbed in %d/%d trials (%.0f%%); chance = %.0f%%;"
               " a material grabber would score 100%%\n",
               grabs, trials, 100.0 * grabs / trials, 100.0 * chance);
        CHECK(grabs <= (int)(0.35 * trials),
              "MCTS grabbed free material in %d/%d trials -- material knowledge has leaked in",
              grabs, trials);
        mcts_free(&m);
    }

    net_seed(12345u);          /* restore the shared net for later tests */
}

/* ------------------------------------------------- 10. pool exhaustion */

static void t_pool_exhaustion(void)
{
    Game g; game_start(&g);

    Mcts m; mcts_init(&m, 8);          /* deliberately tiny */
    printf("  requested 8 nodes, pool floor gives cap=%d (root + one full move list)\n", m.cap);

    int32_t v[MAX_MOVES]; float rv = 0.0f;
    uint64_t rng[4]; seed_rng(rng, 31337u);
    const int n = mcts_search(&m, &g_trunk, &g_head, &g, 400, 1, rng, v, &rv);

    const uint64_t ex = mcts_pool_exhausted(&m);
    const int pick = mcts_pick(v, n, 1.0f, rng);
    printf("  n=%d sum(visits)=%d used=%d/%d exhausted=%llu pick=%d root_value=%+.4f\n",
           n, sum_i(v, n), m.used, m.cap, (unsigned long long)ex, pick, (double)rv);

    CHECK(n == 20, "start position has 20 legal moves, got %d", n);
    CHECK(sum_i(v, n) == 400, "visits sum to %d, want 400", sum_i(v, n));
    CHECK(ex > 0, "a tiny pool did not report any exhaustion");
    CHECK(m.used <= m.cap, "pool overflowed: used=%d cap=%d", m.used, m.cap);
    CHECK(pick >= 0 && pick < n, "pick %d is not a legal move index", pick);

    /* a merely small pool: still no crash, still a full visit count */
    Mcts m2; mcts_init(&m2, 1500);
    const int n2 = mcts_search(&m2, &g_trunk, &g_head, &g, 800, 0, rng, v, &rv);
    printf("  cap=1500: n=%d sum(visits)=%d used=%d exhausted=%llu\n",
           n2, sum_i(v, n2), m2.used, (unsigned long long)mcts_pool_exhausted(&m2));
    CHECK(sum_i(v, n2) == 800, "visits sum to %d, want 800", sum_i(v, n2));
    CHECK(m2.used <= m2.cap, "pool overflowed");

    mcts_free(&m);
    mcts_free(&m2);
}

/* ------------------------- 11b. deep lines, and the caller's Game is const */

static void t_deep_and_const(void)
{
    /* A locked pawn position: five legal moves, nothing to capture, every line
     * shuffling towards a fifty-move or repetition draw.  That drives the tree
     * far deeper than an opening position does, and repetition counting has to
     * keep working at depth.  The path arrays are fixed size, so this is where
     * an off-by-one would show up. */
    Game g; load(&g, "8/8/1p6/pPp5/PkP5/8/1K6/8 w - - 0 1");
    Game before = g;

    Mcts m; mcts_init(&m, 2000000);
    int32_t v[MAX_MOVES]; float rv = 0.0f;
    uint64_t rng[4]; seed_rng(rng, 606u);

    const int n = mcts_search(&m, &g_trunk, &g_head, &g, 20000, 0, rng, v, &rv);
    printf("  20000 sims in a locked pawn position: n=%d sum(visits)=%d max depth=%d "
           "root_value=%+.4f\n", n, sum_i(v, n), m.max_depth_seen, (double)rv);
    CHECK(sum_i(v, n) == 20000, "visits sum to %d, want 20000", sum_i(v, n));
    CHECK(m.max_depth_seen > 8, "search never went deep (max depth %d)", m.max_depth_seen);
    CHECK(m.max_depth_seen <= 128, "path depth %d exceeded the cap", m.max_depth_seen);
    CHECK(fabs((double)rv) < 0.10, "a dead-drawn position scored %+.4f", (double)rv);

    /* mcts_search takes a const Game *: the caller's game, position, history and
     * result must come back untouched, even though the search makes and unmakes
     * thousands of moves. */
    CHECK(memcmp(&before, &g, sizeof(Game)) == 0, "mcts_search mutated the caller's Game");
    printf("  caller's Game unchanged: %s\n",
           memcmp(&before, &g, sizeof(Game)) == 0 ? "yes" : "NO");

    mcts_free(&m);
}

/* ------------------------------------------------------ 11. benchmark */

static void t_bench(void)
{
    Game g; game_start(&g);
    Mcts m; mcts_init(&m, 400000);
    int32_t v[MAX_MOVES]; float rv;
    uint64_t rng[4]; seed_rng(rng, 1u);

    const int cfg[2]  = { 64, 400 };
    const int reps[2] = { 400, 100 };

    for (int c = 0; c < 2; c++) {
        /* warm up */
        mcts_search(&m, &g_trunk, &g_head, &g, cfg[c], 0, rng, v, &rv);

        const uint64_t e0 = m.evals;
        const double t0 = now_s();
        for (int i = 0; i < reps[c]; i++)
            mcts_search(&m, &g_trunk, &g_head, &g, cfg[c], 1, rng, v, &rv);
        const double dt = now_s() - t0;

        const double sims = (double)cfg[c] * (double)reps[c];
        printf("  %3d sims/move: %7.0f sims/s   %7.0f evals/s   %6.1f us/sim   "
               "%.2f ms/move   max depth %d\n",
               cfg[c], sims / dt, (double)(m.evals - e0) / dt,
               1.0e6 * dt / sims, 1000.0 * dt / (double)reps[c], m.max_depth_seen);
        CHECK(dt > 0.0, "benchmark timer did not advance");
    }
    mcts_free(&m);
}

/* ------------------------------------------------------------ main */

int main(void)
{
    chess_init();
    net_seed(12345u);

    printf("=== mcts ===\n");

    banner("1. sign: a forced mate in 1 is found, from both colours");
    t_mate_in_1("white mates (Ra8#)", "6k1/5ppp/8/8/8/8/8/R6K w - - 0 1", "a1a8");
    t_mate_in_1("black mates (Ra1#)", "r6k/8/8/8/8/8/5PPP/6K1 b - - 0 1", "a8a1");

    banner("2. sign: moves that hand the opponent a mate in 1 are avoided");
    {
        const char *w[8] = { "a1d1","a1a2","a1a3","a1a4","a1a5","a1a6","a1a7","a1a8" };
        const char *b[8] = { "a8a1","a8a2","a8a3","a8a4","a8a5","a8a6","a8a7","a8d8" };
        t_avoid_losing("white to move", "3r4/6pk/7p/8/8/8/5PPP/R5K1 w - - 0 1", w, 8, 4000);
        t_avoid_losing("black to move", "r5k1/5ppp/8/8/8/7P/6PK/3R4 b - - 0 1", b, 8, 4000);
    }

    banner("3. root value: ~-1 when every move loses to a mate in 1");
    t_all_moves_lose();

    banner("4. every drawn termination evaluates to 0, not to a win");
    t_draw_terminals();

    banner("5. a finished game returns 0 legal moves");
    t_game_over();

    banner("6. determinism");
    t_determinism();

    banner("7. Dirichlet root noise");
    t_dirichlet();

    banner("8. visit counts and the policy target");
    t_visit_sums();

    banner("9. mcts_pick temperature");
    t_pick();

    banner("10. no chess knowledge leak (randomly initialised network)");
    t_no_knowledge_leak();

    banner("11. pool exhaustion degrades, never crashes");
    t_pool_exhaustion();

    banner("11b. deep forced lines; the caller's Game is not touched");
    t_deep_and_const();

    banner("12. benchmark (single core)");
    t_bench();

    printf("\n%d checks, %d failures\n", g_checks, g_fail);
    if (g_fail) { printf("FAILED\n"); return 1; }
    printf("all mcts tests passed\n");
    return 0;
}
