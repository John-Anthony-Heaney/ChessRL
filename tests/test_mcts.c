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

/* mcts_pool_exhausted() is declared in mcts.h now, as a proper part of the
 * interface, and the counter is a field of Mcts rather than a hidden block in
 * front of the node pool. */

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

/* ============================== 13. subtree reuse changes nothing ========= */

static int same_visits(const int32_t *a, const int32_t *b, int n)
{
    for (int i = 0; i < n; i++) if (a[i] != b[i]) return 0;
    return 1;
}

/* Plays `plies` moves, searching every ply with two engines: one that inherits
 * the subtree under the move played (and caches evaluations) and one that
 * rebuilds from scratch with no cache.  Returns the number of plies at which
 * the two disagreed about anything.
 *
 * With the root noise off this must be ZERO, and not by luck.  A simulation
 * that descends from the old root into child C and carries on applies the same
 * PUCT rule to C's own statistics that a search rooted at C would, so after N
 * visits to C its subtree IS the tree a root-at-C search holds after N-1
 * simulations.  `sims` is a TOTAL budget, so topping up to sims + 1 finishes
 * exactly that sequence -- and no simulation reads the rng, so the two engines
 * agree bit for bit, every visit count and the root value.
 *
 * `stride` is how many plies pass between searches: 1 is self-play's cadence,
 * 2 is an engine playing one side of a game. */
static int reuse_disagreements(const char *fen, int sims, int plies, int stride,
                               uint64_t *hits, uint64_t *misses)
{
    Game g; if (fen) load(&g, fen); else game_start(&g);

    Mcts a, b;
    mcts_init(&a, 200000);                  /* inherits, and caches           */
    mcts_init(&b, 200000);
    b.reuse = 0; b.cache = 0;               /* rebuilds every move, no cache  */

    int32_t va[MAX_MOVES], vb[MAX_MOVES];
    float   rva = 0.0f, rvb = 0.0f;
    uint64_t ra[4], rb[4];
    Move list[MAX_MOVES];
    int bad = 0;

    seed_rng(ra, 4242u);
    seed_rng(rb, 4242u);

    for (int p = 0; p < plies && g.result == GR_ONGOING; p++) {
        int n = gen_legal(&g.pos, list);
        if (n <= 0) break;
        if (p % stride == 0) {
            const int na = mcts_search(&a, &g_trunk, &g_head, &g, sims, 0, ra, va, &rva);
            const int nb = mcts_search(&b, &g_trunk, &g_head, &g, sims, 0, rb, vb, &rvb);
            if (na != nb || na != n)               { bad++; break; }
            if (!same_visits(va, vb, na))            bad++;
            else if (rva != rvb)                     bad++;
            if (sum_i(va, na) != sims)               bad++;
            if (a.pool[0].N != sims + 1)             bad++;
            /* the caller's visits[] is only meaningful if the root children are
             * gen_legal()'s list, in order */
            for (int i = 0; i < na; i++)
                if (a.pool[a.pool[0].first + i].move != list[i]) { bad++; break; }
            game_push(&g, list[argmax_i(va, na)]);
        } else {
            game_push(&g, list[argmax_i(vb, n > 0 ? n : 1)]);
        }
    }
    if (hits)   *hits   = a.reuse_hits;
    if (misses) *misses = a.reuse_misses;

    mcts_free(&a);
    mcts_free(&b);
    return bad;
}

static void t_reuse_is_a_fresh_search(void)
{
    const struct { const char *fen; int sims; int plies; int stride; } cases[] = {
        { NULL,                                                    300, 20, 1 },
        { NULL,                                                    300, 20, 2 },
        { "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
                                                                   600, 14, 1 },
        /* a shuffling position: the repetition window moves under the tree     */
        { "8/8/1p6/pPp5/PkP5/8/1K6/8 w - - 0 1",                  800, 16, 1 },
        /* nothing but pawn moves and captures, so the halfmove clock resets    */
        { "4k3/pp3ppp/8/8/8/2r5/PP3PPP/3RK3 w - - 0 1",           400, 16, 1 },
    };

    for (size_t k = 0; k < sizeof cases / sizeof cases[0]; k++) {
        uint64_t hits = 0, misses = 0;
        const int bad = reuse_disagreements(cases[k].fen, cases[k].sims,
                                            cases[k].plies, cases[k].stride,
                                            &hits, &misses);
        printf("  %-46s sims=%3d stride=%d  reuse %llu/%llu  disagreements: %d\n",
               cases[k].fen ? cases[k].fen : "(start position)",
               cases[k].sims, cases[k].stride,
               (unsigned long long)hits, (unsigned long long)(hits + misses), bad);
        CHECK(bad == 0, "reuse/cache changed the search result in %s",
              cases[k].fen ? cases[k].fen : "the start position");
        CHECK(hits > 0, "no subtree was ever inherited in %s -- the test is vacuous",
              cases[k].fen ? cases[k].fen : "the start position");
    }
}

/* The 32, 64, 128, ... ramp uci.c and api.c use to honour a movetime cap.
 * Re-rooting at the SAME position makes each pass a top-up of the last, and the
 * answer must be the answer a single search of the final budget would give. */
static void t_reuse_ramp(void)
{
    Game g; load(&g, "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4");

    Mcts a, b;
    mcts_init(&a, 200000);
    mcts_init(&b, 200000);
    b.reuse = 0; b.cache = 0;

    int32_t va[MAX_MOVES], vb[MAX_MOVES];
    float   rva = 0.0f, rvb = 0.0f;
    uint64_t ra[4], rb[4]; seed_rng(ra, 9u); seed_rng(rb, 9u);

    const int target = 800;
    int na = 0;
    for (int run = 32; ; run = (run * 2 > target) ? target : run * 2) {
        na = mcts_search(&a, &g_trunk, &g_head, &g, run, 0, ra, va, &rva);
        if (run >= target) break;
    }
    const int nb = mcts_search(&b, &g_trunk, &g_head, &g, target, 0, rb, vb, &rvb);

    printf("  ramp 32..800 evals=%llu vs one 800-sim search evals=%llu "
           "(%.2fx)  identical: %s\n",
           (unsigned long long)a.evals, (unsigned long long)b.evals,
           (double)b.evals / (double)(a.evals ? a.evals : 1),
           (na == nb && same_visits(va, vb, na) && rva == rvb) ? "yes" : "NO");
    CHECK(na == nb, "ramp returned %d moves, single search %d", na, nb);
    CHECK(same_visits(va, vb, na), "the ramp did not land on the single search's tree");
    CHECK(rva == rvb, "ramp root value %+.8f vs %+.8f", (double)rva, (double)rvb);
    CHECK(sum_i(va, na) == target, "ramp visits sum to %d, want %d", sum_i(va, na), target);

    mcts_free(&a);
    mcts_free(&b);
}

/* ====================== 14. a stale tree is never used for a wrong position */

/* The most-visited two-ply path through the standing tree, which is the path a
 * real game is most likely to take and the one with statistics worth keeping. */
static int best_two_ply(const Mcts *m, Move *m0, Move *m1)
{
    const MctsNode *root = &m->pool[0];
    if (root->first < 0 || root->nchild <= 0) return 0;

    int b0 = 0;
    for (int i = 1; i < root->nchild; i++)
        if (m->pool[root->first + i].N > m->pool[root->first + b0].N) b0 = i;
    const MctsNode *c = &m->pool[root->first + b0];
    *m0 = c->move;
    if (c->first < 0 || c->nchild <= 0) return 1;

    int b1 = 0;
    for (int i = 1; i < c->nchild; i++)
        if (m->pool[c->first + i].N > m->pool[c->first + b1].N) b1 = i;
    const MctsNode *gc = &m->pool[c->first + b1];
    if (gc->N <= 0 || gc->first < 0) return 1;
    *m1 = gc->move;
    return 2;
}

static void t_reuse_refuses_stale(void)
{
    Move mv;
    int32_t v[MAX_MOVES], vfresh[MAX_MOVES];
    float rv = 0.0f, rvf = 0.0f;
    uint64_t rng[4];

    Mcts m;  mcts_init(&m, 400000);
    Mcts f;  mcts_init(&f, 400000); f.reuse = 0; f.cache = 0;

    /* Game A: a tree is built, then the two most-visited plies of that tree are
     * played, which is the case a real engine hits between its own moves. */
    Game a; game_start(&a);
    Move a0 = MV_NONE, a1 = MV_NONE;
    seed_rng(rng, 17u);
    mcts_search(&m, &g_trunk, &g_head, &a, 4000, 0, rng, v, &rv);
    CHECK(best_two_ply(&m, &a0, &a1) == 2, "the tree has no two-ply path to follow");
    game_push(&a, a0);
    game_push(&a, a1);
    mcts_search(&m, &g_trunk, &g_head, &a, 4000, 0, rng, v, &rv);
    const uint64_t hits_after_a = m.reuse_hits;
    CHECK(hits_after_a > 0, "the two-ply advance did not inherit anything");

    /* Game B: a DIFFERENT game, the same number of plies in.  The ply counter
     * agrees, so only the position check can catch this. */
    {
        Game b; game_start(&b);
        const char *bmoves[] = { "d2d4", "d7d5" };
        for (int i = 0; i < 2; i++) {
            CHECK(move_from_uci(&b.pos, bmoves[i], &mv) != 0, "illegal setup move");
            game_push(&b, mv);
        }
        uint64_t r1[4], r2[4]; seed_rng(r1, 5u); seed_rng(r2, 5u);
        const int n1 = mcts_search(&m, &g_trunk, &g_head, &b, 400, 0, r1, v,      &rv);
        const int n2 = mcts_search(&f, &g_trunk, &g_head, &b, 400, 0, r2, vfresh, &rvf);
        printf("  a different game at the same ply: inherited %s, answer matches a "
               "fresh tree: %s\n",
               (m.reuse_hits == hits_after_a) ? "nothing" : "SOMETHING",
               (n1 == n2 && same_visits(v, vfresh, n1)) ? "yes" : "NO");
        CHECK(m.reuse_hits == hits_after_a, "a tree from another game was inherited");
        CHECK(n1 == n2 && same_visits(v, vfresh, n1),
              "the answer for a foreign position was not a fresh search's answer");
    }

    /* Three plies at once: beyond what the tree can be re-rooted on. */
    {
        Game c = a;
        Move cl[MAX_MOVES];
        for (int i = 0; i < 3; i++) {
            const int n = gen_legal(&c.pos, cl);
            CHECK(n > 0, "no legal move to play");
            game_push(&c, cl[0]);
        }
        /* re-root the standing tree on game A again first */
        uint64_t r[4]; seed_rng(r, 3u);
        mcts_search(&m, &g_trunk, &g_head, &a, 400, 0, r, v, &rv);
        const uint64_t before = m.reuse_hits;
        mcts_search(&m, &g_trunk, &g_head, &c, 400, 0, r, v, &rv);
        printf("  a three-ply jump: inherited %s\n",
               (m.reuse_hits == before) ? "nothing" : "SOMETHING");
        CHECK(m.reuse_hits == before, "a three-ply jump inherited a subtree");
    }

    /* Same position, same ply, but the caller's repetition history is shorter
     * than the one the standing tree was built under.  A cached TR_REPETITION
     * inside that tree would be a fact about a history this caller does not
     * have, so the tree must be refused. */
    {
        /* A pawnless position with the fifty-move clock already running, so six
         * plies of argmax play cannot reset it.  The earlier fixture was pawn-locked
         * and a pawn move zeroed the clock, which collapsed the repetition window to
         * one ply and made truncating the history a no-op -- the assertion below then
         * tested nothing.  halfmove > 1 is what gives this case its teeth. */
        Game d; load(&d, "8/8/8/3k4/8/8/8/3K3R w - - 12 40");
        uint64_t r[4]; seed_rng(r, 21u);
        Move list[MAX_MOVES];
        for (int p = 0; p < 6 && d.result == GR_ONGOING; p++) {
            const int n = mcts_search(&m, &g_trunk, &g_head, &d, 300, 0, r, v, &rv);
            if (n <= 0) break;
            gen_legal(&d.pos, list);
            game_push(&d, list[argmax_i(v, n)]);
        }
        mcts_search(&m, &g_trunk, &g_head, &d, 300, 0, r, v, &rv);
        const uint64_t before = m.reuse_hits;

        Game e = d;                       /* same position, same ply, no history */
        e.hist_len = 1;
        e.hist[0]  = e.pos.key;
        CHECK(e.pos.halfmove > 1, "the history-truncation case needs a halfmove clock");

        uint64_t r1[4], r2[4]; seed_rng(r1, 8u); seed_rng(r2, 8u);
        const int n1 = mcts_search(&m, &g_trunk, &g_head, &e, 300, 0, r1, v,      &rv);
        const int n2 = mcts_search(&f, &g_trunk, &g_head, &e, 300, 0, r2, vfresh, &rvf);
        printf("  a truncated repetition history: inherited %s, answer matches a "
               "fresh tree: %s\n",
               (m.reuse_hits == before) ? "nothing" : "SOMETHING",
               (n1 == n2 && same_visits(v, vfresh, n1)) ? "yes" : "NO");
        CHECK(m.reuse_hits == before,
              "a subtree was inherited across a change of repetition history");
        CHECK(n1 == n2 && same_visits(v, vfresh, n1),
              "the answer after refusing a tree was not a fresh search's answer");
    }

    /* A changed network invalidates everything.  In self-play the two sides are
     * different agents, so this fires on every ply.
     *
     * The TREE is dropped, because tree statistics are about particular weights
     * and cannot be re-labelled.  The evaluation CACHE is not flushed any more
     * and does not need to be: the weight stamp is part of every cache tag, so
     * an entry made under the old weights simply cannot be hit under the new
     * ones.  That is the stronger guarantee, so it is what is tested -- not the
     * flush counter, which only ever said that an attempt had been made.
     *
     * The control is a THIRD search with no cache at all and no standing tree,
     * so nothing about the mechanism under test is assumed. */
    {
        uint64_t r[4]; seed_rng(r, 2u);
        mcts_search(&m, &g_trunk, &g_head, &a, 400, 0, r, v, &rv);
        const uint64_t before = m.reuse_hits, hits0 = m.cache_hits;

        Game a2 = a;
        Move al[MAX_MOVES];
        CHECK(gen_legal(&a2.pos, al) > 0, "no legal move to play");
        game_push(&a2, al[0]);

        net_seed(0xFEEDu);                       /* the weights move */
        uint64_t r2[4]; seed_rng(r2, 2u);
        const int n1 = mcts_search(&m, &g_trunk, &g_head, &a2, 400, 0, r, v, &rv);

        Mcts ctl;
        mcts_init(&ctl, 64000);
        mcts_defaults(&ctl);
        ctl.cache = 0;
        ctl.reuse = 0;
        int32_t vctl[MAX_MOVES]; float rvctl = 0.0f;
        const int n2 = mcts_search(&ctl, &g_trunk, &g_head, &a2, 400, 0, r2, vctl, &rvctl);
        mcts_free(&ctl);

        printf("  the weights changed: inherited %s, %llu lookups served from the "
               "old weights: %s\n",
               (m.reuse_hits == before) ? "nothing" : "SOMETHING",
               (unsigned long long)(m.cache_hits - hits0),
               (n1 == n2 && same_visits(v, vctl, n1)) ? "none" : "SOME");
        CHECK(m.reuse_hits == before, "a subtree survived a change of weights");
        CHECK(n1 == n2 && same_visits(v, vctl, n1),
              "an evaluation made under the old weights survived the change");
        net_seed(12345u);                        /* restore the shared net */
    }

    mcts_free(&m);
    mcts_free(&f);
}

/* ======================= 15. the cache keys on the whole network input ===== */

static float root_v(Mcts *m, const char *fen)
{
    Game g; load(&g, fen);
    int32_t v[MAX_MOVES];
    float rv = 0.0f;
    uint64_t rng[4]; seed_rng(rng, 1u);
    /* sims = 0 leaves root.N = 1 and root.W = V(s), so the root value IS the
     * value head's output for this position and nothing else. */
    mcts_search(m, &g_trunk, &g_head, &g, 0, 0, rng, v, &rv);
    return rv;
}

static void t_cache_keys_on_the_input(void)
{
    /* Two positions that the ZOBRIST KEY cannot tell apart: the halfmove clock
     * is not hashed.  The network sees them differently -- feature 780 + the
     * clock bucket -- so a cache keyed on the bare zobrist key would hand the
     * second one the first one's evaluation. */
    const char *h0  = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4";
    const char *h91 = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 91 50";

    Game a, b; load(&a, h0); load(&b, h91);
    printf("  clock 4 vs 91: same zobrist key: %s (buckets %d and %d)\n",
           a.pos.key == b.pos.key ? "yes" : "no",
           a.pos.halfmove / 13, b.pos.halfmove / 13 > 7 ? 7 : b.pos.halfmove / 13);
    CHECK(a.pos.key == b.pos.key,
          "the two clocks gave different zobrist keys -- this test proves nothing");

    Mcts warm;  mcts_init(&warm, 20000);              /* cache on  */
    Mcts cold;  mcts_init(&cold, 20000); cold.cache = 0;

    const float v0_cold = root_v(&cold, h0);
    const float v91_cold = root_v(&cold, h91);
    const float v0_warm = root_v(&warm, h0);          /* fills the cache */
    const float v91_warm = root_v(&warm, h91);        /* must NOT hit v0 */

    printf("  V(clock 4)=%+.6f  V(clock 91)=%+.6f   cached: %+.6f / %+.6f\n",
           (double)v0_cold, (double)v91_cold, (double)v0_warm, (double)v91_warm);
    CHECK(v0_cold != v91_cold,
          "the clock bucket did not change the value head -- test is vacuous");
    CHECK(v0_warm == v0_cold && v91_warm == v91_cold,
          "the cache returned the wrong position's evaluation");

    /* Castling rights and the en-passant square ARE in the zobrist key, so the
     * key alone separates them.  Check that, since the cache relies on it. */
    {
        Game c, d, e;
        load(&c, "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQ - 4 4");
        load(&d, "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3");
        load(&e, "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq - 0 3");
        printf("  castling rights change the key: %s   en passant: %s\n",
               a.pos.key != c.pos.key ? "yes" : "NO",
               d.pos.key != e.pos.key ? "yes" : "NO");
        CHECK(a.pos.key != c.pos.key, "castling rights are not in the zobrist key");
        CHECK(d.pos.key != e.pos.key, "the en-passant file is not in the zobrist key");
    }

    /* And the cache must never change an answer.  A whole game, cache on vs
     * cache off, tree reuse out of the picture on both sides. */
    {
        Game g; game_start(&g);
        Mcts on, off;
        mcts_init(&on, 200000);  on.reuse = 0;
        mcts_init(&off, 200000); off.reuse = 0; off.cache = 0;
        int32_t va[MAX_MOVES], vb[MAX_MOVES];
        float rva, rvb;
        uint64_t ra[4], rb[4]; seed_rng(ra, 6u); seed_rng(rb, 6u);
        Move list[MAX_MOVES];
        int bad = 0;

        for (int p = 0; p < 16 && g.result == GR_ONGOING; p++) {
            const int na = mcts_search(&on,  &g_trunk, &g_head, &g, 300, 0, ra, va, &rva);
            const int nb = mcts_search(&off, &g_trunk, &g_head, &g, 300, 0, rb, vb, &rvb);
            if (na != nb || !same_visits(va, vb, na) || rva != rvb) bad++;
            if (na <= 0) break;
            gen_legal(&g.pos, list);
            game_push(&g, list[argmax_i(va, na)]);
        }
        const double hr = 100.0 * (double)on.cache_hits
                        / (double)(on.cache_hits + on.cache_misses);
        printf("  16 plies, cache on vs off: disagreements %d   hit rate %.1f%%  "
               "(%llu hits, %llu network evaluations)\n",
               bad, hr, (unsigned long long)on.cache_hits,
               (unsigned long long)on.evals);
        CHECK(bad == 0, "the cache changed the search result");
        CHECK(on.cache_hits > 0, "the cache never hit -- the test is vacuous");
        CHECK(on.evals < off.evals, "the cache saved no network evaluations");
        mcts_free(&on);
        mcts_free(&off);
    }

    mcts_free(&warm);
    mcts_free(&cold);
}

/* ============================= 16. FPU reduction, which DOES change play === */

static void t_fpu_reduction(void)
{
    Game g; load(&g, "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4");

    Mcts flat, red;
    mcts_init(&flat, 200000);
    mcts_init(&red,  200000);
    red.fpu_reduction = 0.2f;

    int32_t vf[MAX_MOVES], vr[MAX_MOVES];
    float rvf, rvr;
    uint64_t r1[4], r2[4]; seed_rng(r1, 12u); seed_rng(r2, 12u);

    const int nf = mcts_search(&flat, &g_trunk, &g_head, &g, 800, 0, r1, vf, &rvf);
    const int nr = mcts_search(&red,  &g_trunk, &g_head, &g, 800, 0, r2, vr, &rvr);

    int touched_f = 0, touched_r = 0;
    for (int i = 0; i < nf; i++) if (vf[i] > 0) touched_f++;
    for (int i = 0; i < nr; i++) if (vr[i] > 0) touched_r++;

    printf("  fpu_reduction 0.0 vs 0.2: distributions differ: %s   "
           "moves given a visit: %d vs %d of %d\n",
           same_visits(vf, vr, nf) ? "NO" : "yes", touched_f, touched_r, nf);
    CHECK(nf == nr, "fpu_reduction changed the legal move count");
    CHECK(!same_visits(vf, vr, nf),
          "fpu_reduction 0.2 changed nothing -- it is not wired in");
    /* whatever it does to the shape, the conventions are not negotiable */
    CHECK(sum_i(vr, nr) == 800, "fpu_reduction broke sum(visits) == sims (%d)",
          sum_i(vr, nr));
    CHECK(red.pool[0].N == 801, "fpu_reduction broke root.N == sims + 1 (%d)",
          red.pool[0].N);

    /* 0 must reproduce the flat rule exactly, so the default is the old search. */
    {
        Mcts zero; mcts_init(&zero, 200000);
        zero.fpu_reduction = 0.0f;
        int32_t vz[MAX_MOVES]; float rvz;
        uint64_t r3[4]; seed_rng(r3, 12u);
        const int nz = mcts_search(&zero, &g_trunk, &g_head, &g, 800, 0, r3, vz, &rvz);
        printf("  fpu_reduction 0.0 == the flat fpu, exactly: %s\n",
               (nz == nf && same_visits(vz, vf, nz) && rvz == rvf) ? "yes" : "NO");
        CHECK(nz == nf && same_visits(vz, vf, nz) && rvz == rvf,
              "fpu_reduction = 0 is not the old behaviour");
        mcts_free(&zero);
    }

    /* and it must still find a forced mate, which comes from the rules */
    {
        Game mate; load(&mate, "6k1/5ppp/8/8/8/8/8/R6K w - - 0 1");
        const int mi = idx_of(&mate.pos, "a1a8");
        int32_t v[MAX_MOVES]; float rv;
        uint64_t r[4]; seed_rng(r, 7u);
        const int n = mcts_search(&red, &g_trunk, &g_head, &mate, 400, 0, r, v, &rv);
        printf("  with fpu_reduction on, mate in 1 is still found: %s (root value %+.3f)\n",
               argmax_i(v, n) == mi ? "yes" : "NO", (double)rv);
        CHECK(argmax_i(v, n) == mi, "fpu_reduction lost a mate in 1");
        CHECK(rv > 0.80f, "fpu_reduction root value %+.4f, want ~+1", (double)rv);
    }

    mcts_free(&flat);
    mcts_free(&red);
}

/* ================= 17. reuse under root noise, and under a starved pool === */

static void t_reuse_rough_conditions(void)
{
    /* Root noise on.  Reuse is NOT equivalent to a fresh search here -- a fresh
     * search noises the root priors before its first simulation and an
     * inherited subtree was built without them -- so the guarantee tested is
     * the one callers depend on: the conventions hold and the answer is a legal
     * move distribution over gen_legal()'s list. */
    {
        Game g; game_start(&g);
        Mcts m; mcts_init(&m, 200000);
        uint64_t rng[4]; seed_rng(rng, 88u);
        int32_t v[MAX_MOVES]; float rv;
        Move list[MAX_MOVES];
        int bad = 0;

        for (int p = 0; p < 20 && g.result == GR_ONGOING; p++) {
            const int n = mcts_search(&m, &g_trunk, &g_head, &g, 400, 1, rng, v, &rv);
            const int nl = gen_legal(&g.pos, list);
            if (n != nl || sum_i(v, n) != 400 || m.pool[0].N != 401) bad++;
            for (int i = 0; i < n; i++)
                if (m.pool[m.pool[0].first + i].move != list[i]) { bad++; break; }
            if (n <= 0) break;
            game_push(&g, list[mcts_pick(v, n, 1.0f, rng)]);
        }
        printf("  20 noisy plies with reuse on: violations %d   inherited %llu/%llu\n",
               bad, (unsigned long long)m.reuse_hits,
               (unsigned long long)(m.reuse_hits + m.reuse_misses));
        CHECK(bad == 0, "reuse under root noise broke a convention");
        CHECK(m.reuse_hits > 0, "reuse never fired under root noise");
        mcts_free(&m);
    }

    /* A pool far too small for the budget, played out over many moves.  The
     * inherited subtree competes with the new one for the same pool, so this is
     * where a compaction bug would show up as an overflow or a lost node. */
    {
        Game g; game_start(&g);
        Mcts m; mcts_init(&m, 1500);
        uint64_t rng[4]; seed_rng(rng, 606u);
        int32_t v[MAX_MOVES]; float rv;
        Move list[MAX_MOVES];
        int bad = 0;

        for (int p = 0; p < 24 && g.result == GR_ONGOING; p++) {
            const int n = mcts_search(&m, &g_trunk, &g_head, &g, 800, 0, rng, v, &rv);
            if (n <= 0) break;
            if (sum_i(v, n) != 800 || m.used > m.cap || m.pool[0].N != 801) bad++;
            gen_legal(&g.pos, list);
            game_push(&g, list[argmax_i(v, n)]);
        }
        printf("  24 plies with a 1500-node pool: violations %d  used %d/%d  "
               "exhausted %llu\n", bad, m.used, m.cap,
               (unsigned long long)mcts_pool_exhausted(&m));
        CHECK(bad == 0, "a starved pool plus reuse broke a convention");
        CHECK(mcts_pool_exhausted(&m) > 0, "a 1500-node pool was not exhausted");
        mcts_free(&m);
    }
}

/* ------------------------------------------------------ 11. benchmark */

/* One configuration, measured over a GAME rather than over one position
 * searched again and again: repeating a single search is the one case the cache
 * answers almost for free, and it would flatter these numbers badly.  `stride`
 * is the plies between searches -- 1 as in self-play, 2 as when playing one
 * side.  Returns wall-clock seconds; the caller keeps the minimum over repeats,
 * because this may well be sharing the machine. */
static double bench_run(int sims, int stride, int reuse, int cache,
                        uint64_t *evals, uint64_t *hits, uint64_t *misses,
                        int *moves)
{
    Game g; game_start(&g);
    Mcts m; mcts_init(&m, 1 + sims * 80);
    m.reuse = reuse;
    m.cache = cache;

    int32_t v[MAX_MOVES]; float rv;
    uint64_t rng[4]; seed_rng(rng, 1u);
    Move list[MAX_MOVES];
    double secs = 0.0;
    int nm = 0;

    for (int p = 0; p < 40 && g.result == GR_ONGOING; p++) {
        int n = gen_legal(&g.pos, list);
        if (n <= 0) break;
        if (p % stride == 0) {
            const double t0 = now_s();
            n = mcts_search(&m, &g_trunk, &g_head, &g, sims, 0, rng, v, &rv);
            secs += now_s() - t0;
            if (n <= 0) break;
            nm++;
        }
        game_push(&g, list[argmax_i(v, n)]);
    }
    *evals = m.evals; *hits = m.cache_hits; *misses = m.cache_misses; *moves = nm;
    mcts_free(&m);
    return secs;
}

static void t_bench(void)
{
    const int sims = 400;

    for (int stride = 1; stride <= 2; stride++) {
        printf("  %d sims/move, %s:\n", sims,
               stride == 1 ? "a search every ply (self-play's cadence)"
                           : "a search every other ply (playing one side)");
        for (int cfg = 0; cfg < 4; cfg++) {
            const int reuse = cfg & 1, cache = (cfg >> 1) & 1;
            double best = 1.0e9;
            uint64_t ev = 0, hi = 0, mi = 0;
            int moves = 1;
            for (int rep = 0; rep < 3; rep++) {
                uint64_t e, h, s; int mv;
                const double dt = bench_run(sims, stride, reuse, cache, &e, &h, &s, &mv);
                if (dt < best) { best = dt; ev = e; hi = h; mi = s; moves = mv; }
            }
            CHECK(best > 0.0, "benchmark timer did not advance");
            printf("    reuse %-3s cache %-3s  %7.0f sims/s  %7.0f evals/s  "
                   "%6.1f evals/move  %5.2f ms/move  hit rate %4.1f%%\n",
                   reuse ? "on" : "off", cache ? "on" : "off",
                   (double)(sims * moves) / best, (double)ev / best,
                   (double)ev / moves, 1000.0 * best / moves,
                   (hi + mi) ? 100.0 * (double)hi / (double)(hi + mi) : 0.0);
        }
    }
}

/* ------------------------------------------------------------ main */


/* ======================= 19. concurrent searches, one batched evaluation ==
 *
 * The claim being tested is the strong one: G INDEPENDENT searches driven
 * together, with their pending leaves evaluated in a single batched network
 * call, produce EXACTLY what each of them produces alone.  Not approximately --
 * every visit count and every root value bit-for-bit.  That is what makes this
 * different from leaf-parallelising a single search, which needs virtual loss
 * and changes the answer.
 *
 * The comparison is run at several G, over positions of different character,
 * across several plies of real play (so subtree reuse, the repetition window
 * and terminal nodes are all in the mix), with different agents on different
 * boards (so the batch is genuinely ragged: different move counts, different
 * heads, searches finishing at different moments). */

#define CC_NPOS 12
static const char *const CC_FENS[CC_NPOS] = {
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 4 4",
    "6k1/5ppp/8/8/8/8/8/R6K w - - 0 1",
    "r6k/8/8/8/8/8/5PPP/6K1 b - - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1",
    "8/8/8/8/8/6k1/6p1/6K1 w - - 0 1",              /* black is stalemated soon */
    "2kr3r/pp1q1ppp/5n2/1Nb5/2Pp1B2/7Q/P4PPP/1R3RK1 w - - 0 1",
    "8/k7/3p4/p2P1p2/P2P1P2/8/8/K7 w - - 0 1",       /* a locked position: draws */
    "5k2/8/8/8/8/8/6PP/6K1 b - - 0 1",
    "r2q1rk1/pP1p2pp/Q4n2/bbp1p3/Np6/1B3NBn/pPPP1PPP/R3K2R b KQ - 0 1",
};

/* One concurrent slot: everything a game in flight owns privately. */
typedef struct {
    Mcts     m;
    Game     g;
    uint64_t rng[4];
    int32_t  vis[MAX_MOVES];
    float    rv;
    int      n;
    int      sims;
    int      live;
} CcSlot;

static int cc_same(const int32_t *a, const int32_t *b, int n)
{
    for (int i = 0; i < n; i++) if (a[i] != b[i]) return 0;
    return 1;
}

static int cc_bitsame(float a, float b)
{
    uint32_t x, y;
    memcpy(&x, &a, 4); memcpy(&y, &b, 4);
    return x == y;
}

/* Runs `ng` games for `plies` plies, once with every search driven ALONE and
 * once with all `ng` driven TOGETHER, and compares.  `share` attaches one
 * evaluation cache to every concurrent slot instead of giving each its own.
 * Returns the number of disagreements; fills *batches / *rows for reporting. */
static int cc_compare(int ng, int plies, int share, const Head *const *heads,
                      long *rounds, long *rows, long *maxb)
{
    CcSlot *ref = (CcSlot *)calloc((size_t)ng, sizeof(CcSlot));
    CcSlot *cur = (CcSlot *)calloc((size_t)ng, sizeof(CcSlot));
    if (!ref || !cur) { free(ref); free(cur); return 1; }

    MctsBatch *b = mcts_batch_create(ng);
    struct MctsCache *shared = share ? mcts_cache_create(14) : NULL;

    Mcts       *msv[64];
    const Head *hsv[64];
    const Game *gsv[64];
    int         simv[64], noisev[64], nout[64];
    uint64_t   *rngv[64];
    int32_t    *visv[64];
    float       rvv[64];

    for (int i = 0; i < ng; i++) {
        CcSlot *R = &ref[i], *C = &cur[i];
        mcts_init(&R->m, MAX_MOVES + 1 + 200 * 72);  mcts_defaults(&R->m);
        mcts_init(&C->m, MAX_MOVES + 1 + 200 * 72);  mcts_defaults(&C->m);
        if (shared) mcts_cache_attach(&C->m, shared);
        load(&R->g, CC_FENS[i % CC_NPOS]);
        load(&C->g, CC_FENS[i % CC_NPOS]);
        seed_rng(R->rng, (uint64_t)(i + 1) * 7919u);
        seed_rng(C->rng, (uint64_t)(i + 1) * 7919u);
        /* Ragged on purpose: different budgets finish at different moments, so
         * the batch is not a neat rectangle. */
        R->sims = C->sims = 24 + 17 * (i % 5);
        R->live = C->live = 1;
    }

    int bad = 0;
    for (int ply = 0; ply < plies; ply++) {
        /* ---- the reference: each search alone, one leaf at a time -------- */
        for (int i = 0; i < ng; i++) {
            CcSlot *R = &ref[i];
            R->n = R->live ? mcts_search(&R->m, &g_trunk, heads[i % 4], &R->g,
                                         R->sims, 0, R->rng, R->vis, &R->rv) : 0;
        }
        /* ---- the same searches, driven together -------------------------- */
        int nlive = 0;
        for (int i = 0; i < ng; i++) {
            if (!cur[i].live) { cur[i].n = 0; continue; }
            msv[nlive]   = &cur[i].m;
            hsv[nlive]   = heads[i % 4];
            gsv[nlive]   = &cur[i].g;
            simv[nlive]  = cur[i].sims;
            noisev[nlive]= 0;
            rngv[nlive]  = cur[i].rng;
            visv[nlive]  = cur[i].vis;
            nlive++;
        }
        if (nlive > 0) {
            mcts_search_many(nlive, msv, &g_trunk, hsv, gsv, simv, noisev, rngv,
                             visv, rvv, nout, b);
            int k = 0;
            for (int i = 0; i < ng; i++) {
                if (!cur[i].live) continue;
                cur[i].n  = nout[k];
                cur[i].rv = rvv[k];
                k++;
            }
            if (rounds) (*rounds)++;
        }

        for (int i = 0; i < ng; i++) {
            if (ref[i].n != cur[i].n) { bad++; continue; }
            if (ref[i].n <= 0) { ref[i].live = cur[i].live = 0; continue; }
            if (!cc_same(ref[i].vis, cur[i].vis, ref[i].n)) bad++;
            if (!cc_bitsame(ref[i].rv, cur[i].rv))          bad++;

            Move l[MAX_MOVES];
            const int nl = gen_legal(&ref[i].g.pos, l);
            if (nl != ref[i].n) { bad++; continue; }
            const int pick = mcts_pick(ref[i].vis, ref[i].n, 0.8f, ref[i].rng);
            (void)mcts_pick(cur[i].vis, cur[i].n, 0.8f, cur[i].rng);
            game_push(&ref[i].g, l[pick < 0 ? 0 : pick]);
            game_push(&cur[i].g, l[pick < 0 ? 0 : pick]);
            if (ref[i].g.result != GR_ONGOING) { ref[i].live = cur[i].live = 0; }
        }
    }

    /* How wide the batches actually were.  A game sitting on a terminal node or
     * serving its leaf from the cache does not contribute a row, so this is the
     * real distribution, not the nominal G. */
    for (int i = 0; i < ng; i++) {
        if (rows) *rows += (long)cur[i].m.evals;
        mcts_free(&ref[i].m);
        mcts_free(&cur[i].m);
    }
    if (maxb) *maxb = ng;
    mcts_batch_free(b);
    mcts_cache_destroy(shared);
    free(ref); free(cur);
    return bad;
}

static void t_concurrent(void)
{
    Head *heads = (Head *)malloc(4 * sizeof(Head));
    if (!heads) { CHECK(0, "out of memory"); return; }
    for (int i = 0; i < 4; i++) nn_init(&g_trunk, &heads[i], 1000u + 37u * (uint64_t)i);
    net_seed(12345u);                         /* nn_init rewrote the trunk */
    for (int i = 0; i < 4; i++) nn_init(&g_trunk, &heads[i], 1000u + 37u * (uint64_t)i);
    const Head *hp[4] = { &heads[0], &heads[1], &heads[2], &heads[3] };

    static const int GS[] = { 1, 2, 3, 5, 8, 16, 32 };
    for (int k = 0; k < (int)(sizeof GS / sizeof GS[0]); k++) {
        long rounds = 0, rows = 0, mx = 0;
        const int bad = cc_compare(GS[k], 6, 0, hp, &rounds, &rows, &mx);
        printf("  G=%-2d  private caches   disagreements: %d\n", GS[k], bad);
        CHECK(bad == 0, "G=%d with a private cache did not match a lone search", GS[k]);
    }

    for (int k = 0; k < (int)(sizeof GS / sizeof GS[0]); k++) {
        long rounds = 0, rows = 0, mx = 0;
        const int bad = cc_compare(GS[k], 6, 1, hp, &rounds, &rows, &mx);
        printf("  G=%-2d  ONE shared cache disagreements: %d\n", GS[k], bad);
        CHECK(bad == 0, "G=%d sharing one cache did not match a lone search", GS[k]);
    }

    /* What the agreement above is WORTH depends on the build.  Without
     * Accelerate nn_eval_batch is bit-identical to nn_eval at every batch size,
     * so "0 disagreements" is exact by construction and would stay 0 over any
     * number of positions.  With Accelerate the batched trunk goes through
     * cblas_sgemm, which regroups the sums: rows then agree with nn_eval to
     * ~1e-6 and a search COULD in principle order two nearly-equal PUCT scores
     * differently.  It does not here -- but that is an observation, not a
     * guarantee, and saying so is the point. */
    printf("  backend %s, batch tile %d -- the agreement above is %s\n",
           nn_backend(), nn_batch_tile(),
           nn_batch_is_exact(32)
             ? "EXACT BY CONSTRUCTION (batched == nn_eval bit for bit)"
             : "EMPIRICAL: sgemm regroups the sums, rows differ by ~1e-6");

    /* Determinism: the same G, twice, must agree with itself as well. */
    {
        long r1 = 0, r2 = 0, w1 = 0, w2 = 0, m1 = 0, m2 = 0;
        const int a = cc_compare(8, 5, 0, hp, &r1, &w1, &m1);
        const int b = cc_compare(8, 5, 0, hp, &r2, &w2, &m2);
        printf("  repeatability at G=8: %ld vs %ld network rows\n", w1, w2);
        CHECK(a == 0 && b == 0 && w1 == w2,
              "a concurrent run was not reproducible");
    }

    /* The guarantees that matter most, driven through the batched path: a mate
     * in 1 is still found, from both colours, with other games in flight. */
    {
        const char *fens[4] = {
            "6k1/5ppp/8/8/8/8/8/R6K w - - 0 1",
            "r6k/8/8/8/8/8/5PPP/6K1 b - - 0 1",
            "6k1/5ppp/8/8/8/8/8/R6K w - - 0 1",
            "r6k/8/8/8/8/8/5PPP/6K1 b - - 0 1",
        };
        const char *best[4] = { "a1a8", "a8a1", "a1a8", "a8a1" };
        Mcts        ms[4];
        Game        gs[4];
        uint64_t    rng[4][4];
        int32_t     vis[4][MAX_MOVES];
        float       rv[4];
        int         nout[4], sims[4], noise[4];
        Mcts       *mp[4]; const Game *gp[4]; uint64_t *rp[4]; int32_t *vp[4];

        for (int i = 0; i < 4; i++) {
            mcts_init(&ms[i], 200000); mcts_defaults(&ms[i]);
            load(&gs[i], fens[i]);
            seed_rng(rng[i], 9u + (uint64_t)i);
            mp[i] = &ms[i]; gp[i] = &gs[i]; rp[i] = rng[i]; vp[i] = vis[i];
            sims[i] = 800; noise[i] = 0;
        }
        MctsBatch *b = mcts_batch_create(4);
        mcts_search_many(4, mp, &g_trunk, hp, gp, sims, noise, rp, vp, rv, nout, b);
        mcts_batch_free(b);

        for (int i = 0; i < 4; i++) {
            const int want = idx_of(&gs[i].pos, best[i]);
            int arg = 0;
            for (int j = 1; j < nout[i]; j++) if (vis[i][j] > vis[i][arg]) arg = j;
            printf("  mate in 1 in flight with 3 others (%s): %s  root value %+.3f\n",
                   best[i], (arg == want) ? "found" : "MISSED", (double)rv[i]);
            CHECK(arg == want, "the batched search missed a mate in 1 (%s)", best[i]);
            CHECK(rv[i] > 0.5f, "a forced mate in 1 did not read as winning");
            mcts_free(&ms[i]);
        }
    }

    free(heads);
    net_seed(12345u);
}

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

    banner("13. subtree reuse is exactly a fresh search of the same budget");
    t_reuse_is_a_fresh_search();
    t_reuse_ramp();

    banner("14. a stale tree is never used for the wrong position");
    t_reuse_refuses_stale();

    banner("15. the evaluation cache keys on the whole network input");
    t_cache_keys_on_the_input();

    banner("16. FPU reduction (it changes play by design, so it is tested alone)");
    t_fpu_reduction();

    banner("17. reuse under root noise and under a starved pool");
    t_reuse_rough_conditions();

    banner("19. G independent searches, batched, are exactly G lone searches");
    t_concurrent();

    banner("18. benchmark (single core)");
    t_bench();

    printf("\n%d checks, %d failures\n", g_checks, g_fail);
    if (g_fail) { printf("FAILED\n"); return 1; }
    printf("all mcts tests passed\n");
    return 0;
}
