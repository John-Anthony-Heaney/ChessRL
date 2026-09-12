#!/usr/bin/env python3
"""A graded ladder of dependency-free opponents -- measuring instruments, not agents.

===========================================================================
THESE ARE RULERS.  THEY ARE ALLOWED TO KNOW CHESS.
===========================================================================
Every player in this file contains hand-coded chess knowledge: piece values,
capture ordering, a mobility term.  That is deliberate and it is not a breach
of anything.

`docs/FROM_SCRATCH.md` constrains what the LEARNING AGENT may use -- the
network, the MCTS that ships with it, and the reward it is trained on.  It says
nothing about what we are allowed to MEASURE that agent against.  A ruler with
no markings measures nothing; an opponent with no chess knowledge is either
`random` (which our agent already beats 96% of the time) or Stockfish (which
beats it 60-0).  The whole point of this file is to fill the enormous gap
between those two with opponents whose strength we can dial.

`tools/audit_knowledge.sh` scans `src/*.c` and nothing else, so it does not and
should not flag this file.  If someone later extends the audit's scope to `py/`,
the fix is an exemption for this file, not the removal of its piece values --
and it would also have to exempt `py/benchmark.py`, which has contained a
material-counting opponent since the day it was written.

Nothing here is ever loaded by the agent, by training, or by the play path.
`py/benchmark.py` imports it to build opponents and nothing else does.

===========================================================================
THE LADDER
===========================================================================
  random            a uniformly random legal move.  The floor.
  randomplus-P      the material-1 move with probability P, else a random legal
                    move.  P in [0,1] is a CONTINUOUS dial between `random`
                    (P=0) and `material-1` (P=1).
  material-N        negamax to depth N over MATERIAL ONLY, alpha-beta, with a
                    quiescence search over captures and promotions, and uniform
                    random tie-breaking.  N = 1..4 (any N >= 1 works; 5+ is slow).
  mobility-N        the same search with a small mobility term added to the
                    evaluation.  A different, slightly stronger style at the
                    same depth.

`py/benchmark.py`'s legacy `material` opponent -- 1-ply greedy with NO
quiescence -- is kept under that name and is a rung in its own right.  It is
NOT the same player as `material-1`: quiescence is what stops a searcher from
hanging the piece it just grabbed, and it is worth a lot of Elo.  See
docs/BENCHMARK.md.

===========================================================================
HOW LEGALITY WORKS -- and what the shadow board is for
===========================================================================
Move generation and legality come from `py/engine.py`'s `GameHandle`, i.e. from
`src/chess.c`, exactly like every other consumer.  Nothing here decides what is
legal, what is check, or when a game is over.

The search does keep its own piece-placement array (`_Board`), because the only
way to ask the C library what is on a square is `api_game_state()`, and at
~14us a call that is three times the cost of generating the moves.  The shadow
board answers two questions the search asks at every node -- "what is the
material balance" and "is this move a capture" -- for free.

It is bookkeeping, not rules: it never generates or filters a move.  It is
initialised from the library's own FEN at the start of every search and
`selftest` checks it square-by-square against the library's FEN over thousands
of plies of random play.  Run it:

    python3 py/baselines.py selftest

===========================================================================
CALIBRATION
===========================================================================
A ladder of unknown-strength opponents measures nothing.  `calibrate` plays the
rungs against each other (and, if you point it at one, against Stockfish), fits
a Bradley-Terry model to all the pairwise results by iterative maximum
likelihood, and prints an internal Elo scale for the whole set:

    python3 py/baselines.py calibrate --games 20 --threads 2

Usage:
  python3 py/baselines.py list
  python3 py/baselines.py selftest [--games 40] [--plies 120]
  python3 py/baselines.py speed [--depths 1,2,3,4]
  python3 py/baselines.py calibrate [--players ...] [--games N] [--band K]
                                    [--model PATH] [--stockfish PATH] [--out JSON]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine as chessrl  # noqa: E402

__all__ = [
    "PIECE_VALUE", "BASELINE_HELP", "is_baseline", "make_baseline", "describe",
    "parse_name", "expand_group", "RandomBaseline", "SearchBaseline",
    "RandomPlusBaseline", "fit_bradley_terry", "bootstrap_ratings",
]

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ==========================================================================
# the evaluation these instruments are built from
# ==========================================================================
#
# Piece values in pawns.  The bishop is 3.25 so that a bishop pair is not
# silently equal to two knights; this is the same table src/net.c uses for the
# telemetry it prints, and it is the one the brief specifies.
PIECE_VALUE = {"p": 1.0, "n": 3.0, "b": 3.25, "r": 5.0, "q": 9.0, "k": 0.0}

MATE = 30000.0          # a mate is worth more than any material imbalance
INF = 1e9
TIE_EPS = 1e-4          # see _root_search: makes root tie detection exact
DEFAULT_QDEPTH = 6      # plies of quiescence past the nominal horizon
DEFAULT_MOBILITY_W = 0.05   # pawns per extra legal move (5 "centipawns")

_FILES = "abcdefgh"
SQ_INDEX = {f + r: (ri * 8 + fi)
            for ri, r in enumerate("12345678")
            for fi, f in enumerate(_FILES)}
SQ_NAME = [""] * 64
for _n, _i in SQ_INDEX.items():
    SQ_NAME[_i] = _n

#: uci -> (from square, to square, promotion piece or "").  A memo, because the
#: search decodes the same few thousand move strings millions of times and
#: string slicing dominates otherwise.  Pure caching: no chess in here.
_MV = {}


def _decode(uci):
    got = _MV.get(uci)
    if got is None:
        got = (SQ_INDEX[uci[0:2]], SQ_INDEX[uci[2:4]],
               uci[4] if len(uci) > 4 else "")
        _MV[uci] = got
    return got


# ==========================================================================
# the shadow board -- placement bookkeeping, never rules
# ==========================================================================


class _Board:
    """Piece placement + running material balance, driven by UCI strings.

    ``mat`` is WHITE minus BLACK in pawns.  ``push``/``pop`` mirror
    ``GameHandle.move``/``undo`` and must be called in lockstep with them.

    This class decides nothing.  It is told which move is being played, by a
    move list that came from the C library, and it updates its own array so the
    evaluation does not have to ask the library for a FEN at every leaf.
    """

    __slots__ = ("sq", "mat", "_undo")

    def __init__(self, fen):
        self.sq = [None] * 64
        rows = fen.split()[0].split("/")
        if len(rows) != 8:
            raise ValueError("bad FEN: %r" % fen)
        for i, row in enumerate(rows):
            rank = 7 - i
            f = 0
            for ch in row:
                if ch.isdigit():
                    f += int(ch)
                else:
                    self.sq[rank * 8 + f] = ch
                    f += 1
            if f != 8:
                raise ValueError("bad FEN rank %r in %r" % (row, fen))
        self.mat = 0.0
        for ch in self.sq:
            if ch is not None:
                v = PIECE_VALUE[ch.lower()]
                self.mat += v if ch.isupper() else -v
        self._undo = []

    # -- queries used by the search ---------------------------------------
    def victim(self, uci):
        """Value of the piece ``uci`` captures, 0.0 if it captures nothing.

        En passant is recognised the only way it can be from a UCI string plus
        a board: a pawn changing file onto an empty square.
        """
        frm, to, _ = _decode(uci)
        v = self.sq[to]
        if v is not None:
            return PIECE_VALUE[v.lower()]
        p = self.sq[frm]
        if (p == "P" or p == "p") and (frm & 7) != (to & 7):
            return PIECE_VALUE["p"]          # en passant
        return 0.0

    def attacker(self, uci):
        p = self.sq[_decode(uci)[0]]
        return PIECE_VALUE[p.lower()] if p else 0.0

    # -- make / unmake ----------------------------------------------------
    def push(self, uci):
        sq = self.sq
        frm, to, promo = _decode(uci)
        piece = sq[frm]
        if piece is None:                      # cannot happen; loud if it does
            raise AssertionError("shadow board empty on %s (move %s)"
                                 % (SQ_NAME[frm], uci))
        white = piece.isupper()
        cap_sq = to
        cap = sq[to]
        if cap is None and (piece == "P" or piece == "p") and (frm & 7) != (to & 7):
            cap_sq = (frm & ~7) | (to & 7)     # en passant: same rank as `frm`
            cap = sq[cap_sq]
        rook_frm = rook_to = -1
        if (piece == "K" or piece == "k") and abs((frm & 7) - (to & 7)) == 2:
            base = frm & ~7
            if to > frm:
                rook_frm, rook_to = base | 7, to - 1
            else:
                rook_frm, rook_to = base, to + 1

        self._undo.append((frm, to, piece, cap, cap_sq, rook_frm, rook_to,
                           self.mat))
        if cap is not None:
            v = PIECE_VALUE[cap.lower()]
            self.mat -= v if cap.isupper() else -v
            sq[cap_sq] = None
        sq[frm] = None
        if promo:
            np = promo.upper() if white else promo.lower()
            sq[to] = np
            d = PIECE_VALUE[promo.lower()] - PIECE_VALUE["p"]
            self.mat += d if white else -d
        else:
            sq[to] = piece
        if rook_frm >= 0:
            sq[rook_to] = sq[rook_frm]
            sq[rook_frm] = None

    def pop(self):
        frm, to, piece, cap, cap_sq, rook_frm, rook_to, mat = self._undo.pop()
        sq = self.sq
        if rook_frm >= 0:
            sq[rook_frm] = sq[rook_to]
            sq[rook_to] = None
        sq[to] = None
        sq[frm] = piece
        if cap is not None:
            sq[cap_sq] = cap
        self.mat = mat

    # -- verification -----------------------------------------------------
    def placement(self):
        """The board as a FEN placement field, for comparison with the library."""
        out = []
        for rank in range(7, -1, -1):
            run = 0
            row = []
            for f in range(8):
                ch = self.sq[rank * 8 + f]
                if ch is None:
                    run += 1
                else:
                    if run:
                        row.append(str(run))
                        run = 0
                    row.append(ch)
            if run:
                row.append(str(run))
            out.append("".join(row))
        return "/".join(out)


# ==========================================================================
# players
# ==========================================================================


class BaselinePlayer:
    """Duck-typed to py/benchmark.py's ``Player``; deliberately not a subclass.

    Importing benchmark.py from here would be circular (benchmark imports this
    module to build opponents).  The contract is four methods and two
    attributes, and it is checked by benchmark.py's own use of them.
    """

    kind = "baseline"
    reports_eval = False

    def __init__(self, name):
        self.name = name
        self.last_eval_cp = None
        self.nodes = 0
        self.move_ms = 0.0
        self.moves_made = 0

    def new_game(self, opening):
        self.last_eval_cp = None

    def pick(self, game, state, history, rng):
        raise NotImplementedError

    def close(self):
        pass

    def stats(self):
        d = {"moves": self.moves_made, "ms": self.move_ms}
        if self.nodes:
            d["nodes"] = self.nodes
        return d

    def describe(self):
        return self.name


class RandomBaseline(BaselinePlayer):
    """Uniformly random over the legal moves.  The floor of the ladder."""

    kind = "random"

    def __init__(self, name="random"):
        super().__init__(name)

    def pick(self, game, state, history, rng):
        self.moves_made += 1
        return rng.choice(state["legal"])

    def describe(self):
        return "uniformly random legal move"


class SearchBaseline(BaselinePlayer):
    """Negamax + alpha-beta + quiescence over a fixed, hand-written evaluation.

    depth       nominal plies of full-width search (>= 1)
    mobility_w  0.0 for `material-N`; > 0 adds the mobility term (`mobility-N`)

    Evaluation, from the side to move:

        material balance in pawns  (P=1 N=3 B=3.25 R=5 Q=9)
      + mobility_w * (my legal move count - my opponent's, one ply earlier)

    The mobility term compares two adjacent positions because a null move is
    the only way to count both sides' moves in one position and GameHandle
    has no null move.  That is the standard cheap approximation and it is
    what makes `mobility-N` a different player from `material-N`, which is
    the entire point of it.

    Terminal positions come from the library: when it reports no legal moves,
    `api_game_state()` says whether that is mate or stalemate.  Draws by
    repetition, the fifty-move rule and insufficient material are detected at
    the ROOT (where every move's resulting state is queried) but NOT inside the
    search -- a depth-4 material searcher that also understood repetition would
    be a different and slower instrument.  This is a documented limitation, not
    an accident; see docs/BENCHMARK.md.

    Determinism: the search itself is deterministic.  The only randomness is
    the uniform choice among moves whose searched value is exactly equal, drawn
    from the seeded RNG benchmark.py hands to `pick`.  Root tie detection is
    exact, not "whatever alpha-beta happened to return": each root move is
    searched with the window (best - 1e-4, +inf), and fail-soft alpha-beta
    returns an EXACT value for anything above alpha, so a move that truly ties
    the best returns the best value and joins the tie list.
    """

    kind = "search"

    def __init__(self, depth, mobility_w=0.0, qdepth=DEFAULT_QDEPTH, name=None):
        self.depth = max(1, int(depth))
        self.mobility_w = float(mobility_w)
        self.qdepth = max(0, int(qdepth))
        if name is None:
            name = ("mobility-%d" if self.mobility_w else "material-%d") % self.depth
        super().__init__(name)
        self.b = None

    # -- evaluation -------------------------------------------------------
    def _eval(self, white_to_move, mob, parent_mob):
        s = self.b.mat if white_to_move else -self.b.mat
        if self.mobility_w and parent_mob is not None:
            s += self.mobility_w * (mob - parent_mob)
        return s

    def _terminal(self, game, ply):
        """Score of a position with no legal moves, from the side to move.

        The library decides which it is; this only reads the verdict.  A result
        of 0 (game not over) with an empty move list is impossible and would
        mean the rules are broken, so it is raised rather than scored.
        """
        res = game.state()["result"]
        if res == 3:
            return 0.0                     # stalemate / any drawn terminal
        if res == 0:
            raise RuntimeError("no legal moves but the game is not over")
        return -(MATE - ply)               # the side to move is mated

    # -- ordering ---------------------------------------------------------
    def _rank(self, moves):
        """(sort key, victim value, move) for every move.

        The key is most-valuable-victim / least-valuable-attacker with a bonus
        for promotions: pure search efficiency, it changes node counts and
        never the value returned.  Quiet moves get key 0.0 and victim 0.0.
        """
        sq = self.b.sq
        out = []
        add = out.append
        for mv in moves:
            frm, to, promo = _decode(mv)
            t = sq[to]
            if t is not None:
                v = PIECE_VALUE[t.lower()]
            else:
                p = sq[frm]
                v = (PIECE_VALUE["p"]
                     if (p == "P" or p == "p") and (frm & 7) != (to & 7) else 0.0)
            pb = PIECE_VALUE[promo.lower()] - 1.0 if promo else 0.0
            if v or pb:
                a = sq[frm]
                add((v * 16.0 + pb * 8.0 - (PIECE_VALUE[a.lower()] if a else 0.0),
                     v, mv))
            else:
                add((0.0, 0.0, mv))
        return out

    def _order(self, moves):
        ranked = self._rank(moves)
        ranked.sort(key=lambda t: -t[0])
        return [t[2] for t in ranked]

    def _tactical(self, moves):
        out = [t for t in self._rank(moves) if t[0] or t[1]]
        out.sort(key=lambda t: -t[0])
        return out

    # -- search -----------------------------------------------------------
    def _quiesce(self, game, alpha, beta, ply, white, parent_mob, qleft,
                 legal=None):
        if legal is None:
            legal = game.legal()
        if not legal:
            return self._terminal(game, ply)
        self.nodes += 1
        stand = self._eval(white, len(legal), parent_mob)
        if qleft <= 0 or stand >= beta:
            return stand
        if stand > alpha:
            alpha = stand
        best = stand
        mob = len(legal)
        for _, victim, mv in self._tactical(legal):
            # delta pruning: if even winning this piece outright (plus a two
            # pawn cushion, eight if a promotion is involved) cannot reach
            # alpha, the line is not worth a node.
            if stand + victim + (8.0 if len(mv) > 4 else 2.0) <= alpha:
                continue
            self.b.push(mv)
            game.move(mv)
            v = -self._quiesce(game, -beta, -alpha, ply + 1, not white, mob,
                               qleft - 1)
            game.undo()
            self.b.pop()
            if v > best:
                best = v
                if v > alpha:
                    alpha = v
                    if alpha >= beta:
                        break
        return best

    def _search(self, game, depth, alpha, beta, ply, white, parent_mob):
        legal = game.legal()
        if not legal:
            return self._terminal(game, ply)
        if depth <= 0:
            return self._quiesce(game, alpha, beta, ply, white, parent_mob,
                                 self.qdepth, legal)
        self.nodes += 1
        mob = len(legal)
        best = -INF
        for mv in self._order(legal):
            self.b.push(mv)
            game.move(mv)
            v = -self._search(game, depth - 1, -beta, -alpha, ply + 1,
                              not white, mob)
            game.undo()
            self.b.pop()
            if v > best:
                best = v
                if v > alpha:
                    alpha = v
                    if alpha >= beta:
                        break
        return best

    def root_values(self, game, state):
        """Every root move's exact searched value, plus the tie list.

        Returns ``(best_value, [moves that achieve it], {move: value})``.  The
        per-move dict holds an exact value for every move that was ever the
        best so far and an upper bound for the rest -- alpha-beta cannot do
        better than that, and the tie list is exact either way.
        """
        white = state["turn"] == "white"
        self.b = _Board(state["fen"])
        legal = state["legal"]
        mob = len(legal)
        won = 1 if white else 2            # api_game_state()'s code for our win
        best = -INF
        ties = []
        values = {}
        for mv in self._order(legal):
            self.b.push(mv)
            game.move(mv)
            st2 = game.state()
            res = st2["result"]
            # The root is the only place a draw by repetition, the fifty-move
            # rule or insufficient material is seen, because it is the only
            # place the library's verdict is cheap enough to ask for.
            if res == 3:
                v = 0.0                                   # any drawn result
            elif res == won:
                v = MATE - 1.0                            # we just mated them
            elif res != 0:
                v = -(MATE - 1.0)                         # cannot happen
            else:
                v = -self._search(game, self.depth - 1, -INF,
                                  -(best - TIE_EPS), 1, not white, mob)
            game.undo()
            self.b.pop()
            values[mv] = v
            if v > best + TIE_EPS:
                best, ties = v, [mv]
            elif v > best - TIE_EPS:
                best = max(best, v)
                ties.append(mv)
        return best, ties, values

    def pick(self, game, state, history, rng):
        self.moves_made += 1
        t0 = time.monotonic()
        fen_before = state["fen"]
        best, ties, _ = self.root_values(game, state)
        self.move_ms += (time.monotonic() - t0) * 1000.0
        if game.state()["fen"] != fen_before:
            raise RuntimeError("%s corrupted the referee position" % self.name)
        if not ties:
            return rng.choice(state["legal"])
        self.last_eval_cp = int(max(-MATE, min(MATE, best)) * 100.0)
        return ties[0] if len(ties) == 1 else rng.choice(ties)

    def describe(self):
        what = "material" if not self.mobility_w else \
               "material + %.2f/move mobility" % self.mobility_w
        return ("negamax depth %d over %s, alpha-beta, quiescence %d ply, "
                "uniform random tie-break" % (self.depth, what, self.qdepth))

    def stats(self):
        d = super().stats()
        d["nodes"] = self.nodes
        return d


class RandomPlusBaseline(BaselinePlayer):
    """The `material-1` move with probability P, a uniform random move otherwise.

    A continuous dial between `random` (P=0) and `material-1` (P=1), which is
    the region an early-training checkpoint actually lives in.  The coin is
    flipped from the seeded RNG, so the player is reproducible.
    """

    kind = "randomplus"

    def __init__(self, p, depth=1, name=None):
        self.p = min(1.0, max(0.0, float(p)))
        if name is None:
            name = "randomplus-%g" % self.p
        super().__init__(name)
        self.inner = SearchBaseline(depth, 0.0, name=name + ":inner")
        self.random_moves = 0

    def pick(self, game, state, history, rng):
        self.moves_made += 1
        if rng.random() >= self.p:
            self.random_moves += 1
            return rng.choice(state["legal"])
        return self.inner.pick(game, state, history, rng)

    def stats(self):
        d = {"moves": self.moves_made, "ms": self.inner.move_ms,
             "nodes": self.inner.nodes, "random_moves": self.random_moves}
        return d

    def describe(self):
        return ("plays the material-%d move with probability %.2f, a uniformly "
                "random legal move otherwise" % (self.inner.depth, self.p))


# ==========================================================================
# names
# ==========================================================================

BASELINE_HELP = ("random | randomplus-P (P in [0,1]) | material-N | mobility-N "
                 "(N >= 1)")


def parse_name(name):
    """``name`` -> (factory, detail) or ``None`` if it is not one of ours."""
    s = (name or "").strip().lower()
    if s == "random":
        return (lambda: RandomBaseline()), "uniformly random legal move"
    for prefix, mob in (("material-", 0.0), ("mobility-", DEFAULT_MOBILITY_W)):
        if s.startswith(prefix):
            try:
                n = int(s[len(prefix):])
            except ValueError:
                return None
            if n < 1 or n > 8:
                return None
            return ((lambda n=n, mob=mob: SearchBaseline(n, mob, name=s)),
                    SearchBaseline(n, mob, name=s).describe())
    if s.startswith("randomplus-"):
        try:
            p = float(s[len("randomplus-"):])
        except ValueError:
            return None
        if not (0.0 <= p <= 1.0):
            return None
        return ((lambda p=p: RandomPlusBaseline(p, name=s)),
                RandomPlusBaseline(p, name=s).describe())
    return None


def is_baseline(name):
    return parse_name(name) is not None


def make_baseline(name):
    got = parse_name(name)
    if got is None:
        raise ValueError("not a baseline opponent: %r (expected %s)"
                         % (name, BASELINE_HELP))
    return got[0]()


def describe(name):
    got = parse_name(name)
    if got is None:
        raise ValueError("not a baseline opponent: %r" % (name,))
    return got[1]


#: The default ladder, weakest first.  `material` is benchmark.py's legacy
#: 1-ply greedy player (no quiescence) and is deliberately kept distinct from
#: `material-1`.
DEFAULT_LADDER = ["random", "randomplus-0.25", "randomplus-0.5",
                  "randomplus-0.75", "material", "material-1", "material-2",
                  "material-3", "mobility-2", "material-4", "mobility-3"]


def expand_group(spec):
    """``"baselines"`` or a comma list -> a list of opponent names."""
    s = (spec or "").strip()
    if not s or s == "baselines":
        return list(DEFAULT_LADDER)
    return [x.strip() for x in s.split(",") if x.strip()]


# ==========================================================================
# Bradley-Terry: turn a pile of pairwise results into one rating scale
# ==========================================================================
#
# P(i scores against j) = 1 / (1 + 10 ** (-(r_i - r_j) / 400))
#
# Fitted by the standard MM (minorise-maximise) iteration, which for the
# Bradley-Terry likelihood is
#
#     p_i  <-  W_i / sum_j ( N_ij / (p_i + p_j) )        p_i = 10 ** (r_i / 400)
#
# with W_i the points i scored (a draw is half a point) and N_ij the number of
# games between i and j.  It is monotone in the likelihood and needs nothing
# but arithmetic.  No scipy, no numpy, no external solver.


def fit_bradley_terry(records, prior_games=1.0, iters=2000, tol=1e-12):
    """Fit ratings to ``records`` = [(a, b, wins_a, draws, losses_a), ...].

    ``prior_games`` adds that many virtual DRAWN games for every player against
    a virtual opponent pinned at the centre of the scale.  Without it any
    player with a clean sweep (or a clean shutout) has an infinite rating and
    the iteration runs away.  With it every rating is finite, and the ones that
    are finite ONLY because of the prior are flagged by
    :func:`degenerate_players` so a report can say so out loud.

    Returns ``{name: elo}`` centred on the unweighted mean of the players.
    """
    names = []
    for a, b, _, _, _ in records:
        for n in (a, b):
            if n not in names:
                names.append(n)
    if not names:
        return {}
    idx = {n: i for i, n in enumerate(names)}
    k = len(names)

    points = [0.0] * k          # W_i
    n_ij = [dict() for _ in range(k)]
    for a, b, wa, d, la in records:
        i, j = idx[a], idx[b]
        ga, gb = wa + 0.5 * d, la + 0.5 * d
        n = wa + d + la
        if n <= 0:
            continue
        points[i] += ga
        points[j] += gb
        n_ij[i][j] = n_ij[i].get(j, 0.0) + n
        n_ij[j][i] = n_ij[j].get(i, 0.0) + n

    # the virtual anchor: half a point each way, `prior_games` games
    anchor = None
    if prior_games > 0:
        anchor = 1.0            # p of the virtual player, fixed at the centre
        for i in range(k):
            points[i] += 0.5 * prior_games

    p = [1.0] * k
    for _ in range(int(iters)):
        newp = list(p)
        for i in range(k):
            denom = 0.0
            for j, n in n_ij[i].items():
                denom += n / (newp[i] + newp[j])
            if anchor is not None:
                denom += prior_games / (newp[i] + anchor)
            if denom <= 0:
                continue
            newp[i] = points[i] / denom
        # renormalise (the likelihood is scale-invariant) and test convergence
        g = math.exp(sum(math.log(max(x, 1e-300)) for x in newp) / k)
        newp = [x / g for x in newp]
        delta = max(abs(math.log(max(a, 1e-300)) - math.log(max(b, 1e-300)))
                    for a, b in zip(newp, p))
        p = newp
        if delta < tol:
            break

    elo = {n: 400.0 * math.log10(max(p[idx[n]], 1e-300)) for n in names}
    mean = sum(elo.values()) / k
    return {n: v - mean for n, v in elo.items()}


def degenerate_players(records):
    """Players who never scored, or never conceded, in the whole data set.

    Their true rating is unbounded on one side; whatever number the fit prints
    for them is the prior talking, not the data.
    """
    got = {}
    for a, b, wa, d, la in records:
        for n in (a, b):
            got.setdefault(n, [0.0, 0.0])
        got[a][0] += wa + 0.5 * d
        got[a][1] += wa + d + la
        got[b][0] += la + 0.5 * d
        got[b][1] += wa + d + la
    out = {}
    for n, (s, g) in got.items():
        if g <= 0:
            out[n] = "no games"
        elif s <= 0:
            out[n] = "lost every game"
        elif s >= g:
            out[n] = "won every game"
    return out


def bootstrap_ratings(match_pairs, n_boot=200, prior_games=1.0, seed=12345,
                      anchor=None):
    """Percentile CIs by resampling GAME PAIRS, the way benchmark.py does.

    ``match_pairs`` = [(a, b, [pair_score_for_a, ...]), ...] where each pair
    score is 0, 0.5 or 1 -- the mean of the two colour-reversed games of one
    opening.  The two games of a pair share an opening and near-deterministic
    engines, so they are strongly correlated; resampling pairs rather than
    games is what keeps the interval honest.
    """
    rng = random.Random(seed)
    samples = {}
    for _ in range(int(n_boot)):
        recs = []
        for a, b, pairs in match_pairs:
            if not pairs:
                continue
            n = len(pairs)
            tot = 0.0
            for _ in range(n):
                tot += pairs[rng.randrange(n)]
            # back to a (w, d, l) triple over 2n games, preserving the score
            games = 2.0 * n
            sa = tot / n * games
            recs.append((a, b, sa, 0.0, games - sa))
        r = fit_bradley_terry(recs, prior_games=prior_games)
        if anchor and anchor[0] in r:
            shift = anchor[1] - r[anchor[0]]
            r = {k: v + shift for k, v in r.items()}
        for k, v in r.items():
            samples.setdefault(k, []).append(v)
    out = {}
    for k, vs in samples.items():
        vs.sort()
        n = len(vs)
        if n < 3:
            out[k] = (None, None)
            continue
        lo = vs[max(0, int(0.025 * n))]
        hi = vs[min(n - 1, int(math.ceil(0.975 * n)) - 1)]
        out[k] = (lo, hi)
    return out


# ==========================================================================
# self-test: legality and the shadow board
# ==========================================================================


def selftest(n_games=40, max_plies=120, seed=7, verbose=True):
    """Every baseline plays only legal moves, and the shadow board never drifts.

    Plays whole games with the ladder's players, checking at EVERY ply that the
    move returned is in the library's own legal list, and -- for the searchers
    -- that the shadow board's placement still matches the library's FEN after
    every make and every unmake in the real search tree.
    """
    players = ["random", "randomplus-0.5", "material-1", "material-2",
               "material-3", "mobility-2"]
    rng = random.Random(seed)
    plies = 0
    fails = []
    t0 = time.monotonic()

    for gi in range(int(n_games)):
        who = players[gi % len(players)]
        pw = make_baseline(who)
        pb = make_baseline(players[(gi + 3) % len(players)])
        with chessrl.GameHandle() as g:
            for ply in range(int(max_plies)):
                st = g.state()
                if st["result"] != 0 or not st["legal"]:
                    break
                p = pw if st["turn"] == "white" else pb
                mv = p.pick(g, st, [], rng)
                if mv not in st["legal"]:
                    fails.append("%s played illegal %r in %s" % (p.name, mv, st["fen"]))
                    break
                if g.state()["fen"] != st["fen"]:
                    fails.append("%s corrupted the position at %s" % (p.name, st["fen"]))
                    break
                # the shadow board, rebuilt and walked over this position's whole
                # move list, must agree with the library square for square
                if isinstance(p, SearchBaseline):
                    b = _Board(st["fen"])
                    if b.placement() != st["fen"].split()[0]:
                        fails.append("shadow board != library at %s" % st["fen"])
                        break
                    for cand in st["legal"]:
                        b.push(cand)
                        g.move(cand)
                        want = g.state()["fen"].split()[0]
                        if b.placement() != want:
                            fails.append("shadow board drift: %s after %s (got %s want %s)"
                                         % (st["fen"], cand, b.placement(), want))
                        g.undo()
                        b.pop()
                        if fails:
                            break
                    if b.placement() != st["fen"].split()[0]:
                        fails.append("shadow board not restored at %s" % st["fen"])
                if fails:
                    break
                g.move(mv)
                plies += 1
        if fails:
            break

    ok = not fails
    if verbose:
        print("selftest: %d games, %d plies, %.1fs" % (n_games, plies,
                                                       time.monotonic() - t0))
        if ok:
            print("  OK -- every move legal, shadow board exact at every ply")
        else:
            for f in fails[:10]:
                print("  FAIL " + f)
    return ok, fails


def speed(depths=(1, 2, 3, 4), positions=6, seed=11, mobility=False):
    """ms per move and nodes per move at each depth, from real game positions."""
    rng = random.Random(seed)
    rows = []
    for d in depths:
        p = SearchBaseline(d, DEFAULT_MOBILITY_W if mobility else 0.0)
        t0 = time.monotonic()
        with chessrl.GameHandle() as g:
            for i in range(int(positions) * 2):
                st = g.state()
                if st["result"] != 0 or not st["legal"]:
                    break
                mv = p.pick(g, st, [], rng)
                g.move(mv)
        dt = time.monotonic() - t0
        rows.append((p.name, p.moves_made, dt / max(1, p.moves_made) * 1000.0,
                     p.nodes / max(1, p.moves_made)))
    return rows


# ==========================================================================
# calibration
# ==========================================================================


def _bench():
    """Import benchmark.py lazily: it imports us, so a top-level import loops."""
    import benchmark as B
    return B


def _args_for(seed, threads, ply_cap, opening_plies):
    import argparse as _a
    ns = _a.Namespace()
    ns.seed = seed
    ns.threads = threads
    ns.ply_cap = ply_cap
    ns.opening_plies = opening_plies
    ns.resign_cp = 0
    ns.resign_plies = 8
    ns.opp_option_map = {}
    ns.opp_go = "nodes=1"
    ns.opp_timeout = 30.0
    ns.debug_uci = False
    ns.opponent = "random"
    return ns


def parse_sf(name):
    """``sf:<label>:<go spec>[:opt=val;opt=val]`` -> (label, go, options).

    Options are separated by ``;`` so the whole thing survives a comma-separated
    ``--players`` list.  ``None`` if this is not a Stockfish entrant.
    """
    if not name.startswith("sf:"):
        return None
    bits = name.split(":")
    if len(bits) < 3:
        raise ValueError("bad Stockfish entrant %r (want sf:<label>:<go>[:opts])"
                         % name)
    opts = {}
    if len(bits) > 3 and bits[3]:
        for kv in bits[3].split(";"):
            if not kv:
                continue
            k, v = kv.split("=", 1)
            opts[k.strip()] = v.strip()
    return bits[1], bits[2], opts


def sf_anchor(name):
    """The published rating an ``sf:`` entrant carries, or ``None``.

    Only a rung that actually sets ``UCI_Elo`` has one.  ``go nodes 1`` is a
    perfectly reproducible opponent and a completely unrated one.
    """
    got = parse_sf(name)
    if not got:
        return None
    opts = got[2]
    if str(opts.get("UCI_LimitStrength", "")).lower() != "true":
        return None
    try:
        return float(opts["UCI_Elo"])
    except (KeyError, ValueError):
        return None


def _spec_for(name, cfg):
    """A benchmark.PlayerSpec for one calibration entrant."""
    B = _bench()
    if name == "material":
        return B.PlayerSpec("material", lambda: B.MaterialPlayer(),
                            "1-ply greedy on material, NO quiescence (legacy)")
    if is_baseline(name):
        return B.PlayerSpec(name, (lambda n=name: make_baseline(n)), describe(name))
    if name.startswith("champion"):
        # "champion" uses --model; "champion@PATH" enters a second checkpoint,
        # which is how two generations get placed on one scale.
        model = name.split("@", 1)[1] if "@" in name else cfg["model"]
        depth = cfg["depth"]
        if not model:
            raise ValueError("entrant 'champion' needs --model")
        return B.PlayerSpec(name,
                            (lambda: B.ChampionPlayer(model, -1, "search", depth,
                                                      1000, 0.0, name=name)),
                            "ChessRL champion, MCTS depth %d (= %d simulations)"
                            % (depth, depth * 64))
    got = parse_sf(name)
    if got:
        _, go, opts = got
        if not cfg.get("stockfish"):
            raise ValueError("entrant %r needs --stockfish" % name)
        argv = [cfg["stockfish"]]
        return B.PlayerSpec(name,
                            (lambda: B.UciPlayer(argv, opts, go, 60.0, name=name)),
                            "Stockfish, go %s%s" % (go, (", " + ", ".join(
                                "%s=%s" % kv for kv in sorted(opts.items())))
                                if opts else ""))
    raise ValueError("unknown calibration entrant %r" % name)


def pairings(names, band):
    """Play each entrant against its `band` nearest neighbours in the ladder.

    A full round robin is mostly wasted games: a match between rungs six apart
    is a 40-0 shutout that carries almost no information about either rating.
    A band keeps the graph connected -- which is all Bradley-Terry needs -- and
    spends the games where the scores are informative.
    """
    out = []
    for i in range(len(names)):
        for k in range(1, band + 1):
            if i + k < len(names):
                out.append((names[i], names[i + k]))
    return out


def calibrate(names, cfg, quiet=False):
    """Play the pairings, fit the scale, return the whole record."""
    B = _bench()
    args = _args_for(cfg["seed"], cfg["threads"], cfg["ply_cap"],
                     cfg["opening_plies"])
    specs = {n: _spec_for(n, cfg) for n in names}
    for n in names:                       # any rung that sets UCI_Elo is an anchor
        a = sf_anchor(n)
        if a is not None:
            cfg["anchors"].setdefault(n, a)
    todo = pairings(names, cfg["band"]) + list(cfg.get("extra_pairs", []))
    seen = set()
    matches = []
    t0 = time.monotonic()
    for a, b in todo:
        if (a, b) in seen or (b, a) in seen:
            continue
        seen.add((a, b))
        # run_match's own progress line is suppressed: one summary line per
        # match is enough, and the two interleave badly on a shared terminal.
        s = B.run_match(specs[a], specs[b], cfg["games"], args,
                        label="%s vs %s" % (a, b), quiet=True)
        pairs = {}
        for gm in s["_games"]:
            pairs.setdefault(gm["pair"], []).append(gm["score_a"])
        pair_scores = [sum(v) / len(v) for _, v in sorted(pairs.items())]
        rec = {"a": a, "b": b, "wins_a": s["wins"], "draws": s["draws"],
               "losses_a": s["losses"], "score_a": s["score"],
               "pair_scores": pair_scores, "games": s["games"],
               "elapsed_s": s["elapsed_s"]}
        matches.append(rec)
        if not quiet:
            print("  %-22s vs %-22s  %3dW %3dD %3dL  %6.1f%%  (%.0fs)"
                  % (a, b, s["wins"], s["draws"], s["losses"],
                     100.0 * s["score"], s["elapsed_s"]))
            sys.stdout.flush()

    records = [(m["a"], m["b"], m["wins_a"], m["draws"], m["losses_a"])
               for m in matches]
    ratings = fit_bradley_terry(records, prior_games=cfg["prior"])
    degen = degenerate_players(records)

    # ---- anchoring -----------------------------------------------------
    anchor = None
    anchor_note = None
    best = None
    for m in matches:
        for side, other, score in ((m["a"], m["b"], m["score_a"]),
                                   (m["b"], m["a"], 1.0 - m["score_a"])):
            if not side.startswith("sf:"):
                continue
            ref = cfg["anchors"].get(side)
            if ref is None:
                continue
            opp_score = 1.0 - score
            if opp_score <= 0.0:
                continue                    # nothing scored against it: no anchor
            cand = (abs(opp_score - 0.5), side, ref)
            if best is None or cand < best:
                best = cand
    if best is not None:
        _, side, ref = best
        anchor = (side, float(ref))
        shift = anchor[1] - ratings[side]
        ratings = {k: v + shift for k, v in ratings.items()}
        anchor_note = ("pinned %s to %g; the baselines scored against it, so the "
                       "scale is anchored" % (side, ref))
    else:
        anchor_note = ("NOT ANCHORED -- no rated opponent conceded a single point "
                       "to any baseline, so nothing ties this scale to an "
                       "external one.  Ratings below are internal and centred on "
                       "the mean of the entrants; differences are meaningful, "
                       "the zero point is arbitrary.")

    match_pairs = [(m["a"], m["b"], m["pair_scores"]) for m in matches]
    ci = bootstrap_ratings(match_pairs, n_boot=cfg["boot"],
                           prior_games=cfg["prior"], seed=cfg["seed"],
                           anchor=anchor)

    return {"generated": time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime()),
            "entrants": names, "config": {k: v for k, v in cfg.items()
                                          if k != "extra_pairs"},
            "matches": matches, "ratings": ratings, "ci": ci,
            "degenerate": degen, "anchor": anchor, "anchor_note": anchor_note,
            "elapsed_s": time.monotonic() - t0}


def render_calibration(res):
    L = []
    L.append("")
    L.append("Bradley-Terry fit over %d matches (%d games)"
             % (len(res["matches"]), sum(m["games"] for m in res["matches"])))
    L.append(res["anchor_note"])
    L.append("")
    order = sorted(res["ratings"], key=lambda n: -res["ratings"][n])
    w = max([len(n) for n in order] + [8])
    L.append("%-*s %8s  %-15s %s" % (w, "opponent", "rating", "95% CI", "note"))
    L.append("-" * (w + 40))
    for n in order:
        lo, hi = res["ci"].get(n, (None, None))
        ci = "[%5.0f,%5.0f]" % (lo, hi) if lo is not None else "      --       "
        note = res["degenerate"].get(n, "")
        L.append("%-*s %8.0f  %-15s %s" % (w, n, res["ratings"][n], ci, note))
    return "\n".join(L)


# ==========================================================================
# CLI
# ==========================================================================


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="baselines.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="The graded baseline ladder, and its calibration.",
        epilog="""examples:
  python3 py/baselines.py list
  python3 py/baselines.py selftest
  python3 py/baselines.py speed --depths 1,2,3,4
  python3 py/baselines.py calibrate --games 20 --threads 2 \\
      --model runs/az_hour/best.crl --stockfish /opt/homebrew/bin/stockfish
""")
    sub = ap.add_subparsers(dest="cmd")

    sub.add_parser("list", help="the ladder and what each rung is")

    st = sub.add_parser("selftest", help="legality + shadow-board verification")
    st.add_argument("--games", type=int, default=40)
    st.add_argument("--plies", type=int, default=120)
    st.add_argument("--seed", type=int, default=7)

    sp = sub.add_parser("speed", help="ms/move and nodes/move by depth")
    sp.add_argument("--depths", default="1,2,3,4")
    sp.add_argument("--positions", type=int, default=6)
    sp.add_argument("--mobility", action="store_true")

    ca = sub.add_parser("calibrate", help="play the ladder against itself and fit a scale")
    ca.add_argument("--players", default="baselines",
                    help="comma list, or 'baselines' for the default ladder")
    ca.add_argument("--games", type=int, default=20, help="games per match (default 20)")
    ca.add_argument("--band", type=int, default=2,
                    help="play each rung against its K nearest neighbours (default 2)")
    ca.add_argument("--threads", type=int, default=2)
    ca.add_argument("--seed", type=int, default=20260912)
    ca.add_argument("--ply-cap", type=int, default=200)
    ca.add_argument("--opening-plies", type=int, default=4)
    ca.add_argument("--prior", type=float, default=1.0,
                    help="virtual drawn games per entrant, to keep sweeps finite")
    ca.add_argument("--boot", type=int, default=200, help="bootstrap resamples")
    ca.add_argument("--model", default=None,
                    help="also enter the champion at this model path")
    ca.add_argument("--depth", type=int, default=4, help="champion depth (default 4)")
    ca.add_argument("--stockfish", default=None,
                    help="also enter Stockfish rungs from this binary")
    ca.add_argument("--sf-rungs", default="nodes=1,nodes=10",
                    help="Stockfish go specs to enter (default nodes=1,nodes=10)")
    ca.add_argument("--sf-elo", default="1320",
                    help="UCI_Elo rungs to enter, comma separated ('' for none)")
    ca.add_argument("--out", default=None, help="write the full record as JSON")

    args = ap.parse_args(argv)
    cmd = args.cmd or "list"

    if cmd == "list":
        print("The baseline ladder (weakest first).  All are deterministic given\n"
              "a seed, need nothing installed, and get legality from src/chess.c.\n")
        for n in DEFAULT_LADDER:
            if n == "material":
                d = "1-ply greedy on material, NO quiescence (py/benchmark.py legacy)"
            else:
                d = describe(n)
            print("  %-18s %s" % (n, d))
        print("\nAlso accepted: randomplus-P for any P in [0,1], material-N and\n"
              "mobility-N for N in 1..8.")
        return 0

    if not chessrl.library_available():
        sys.stderr.write("libchessrl is not built.  Run:  make -j8 lib\n")
        return 2

    if cmd == "selftest":
        ok, _ = selftest(args.games, args.plies, args.seed)
        return 0 if ok else 1

    if cmd == "speed":
        depths = [int(x) for x in str(args.depths).replace(",", " ").split()]
        print("%-14s %6s %10s %12s" % ("player", "moves", "ms/move", "nodes/move"))
        for name, moves, ms, nodes in speed(depths, args.positions,
                                            mobility=args.mobility):
            print("%-14s %6d %10.1f %12.0f" % (name, moves, ms, nodes))
        return 0

    if cmd == "calibrate":
        names = expand_group(args.players)
        cfg = {"games": args.games, "band": args.band, "threads": args.threads,
               "seed": args.seed, "ply_cap": args.ply_cap,
               "opening_plies": args.opening_plies, "prior": args.prior,
               "boot": args.boot, "model": args.model, "depth": args.depth,
               "stockfish": args.stockfish, "anchors": {}, "extra_pairs": []}
        if args.model and not any(n.startswith("champion") for n in names):
            names.append("champion")
        if args.stockfish and not any(n.startswith("sf:") for n in names):
            for go in [x.strip() for x in args.sf_rungs.split(",") if x.strip()]:
                names.append("sf:%s:%s" % (go.replace("=", ""), go))
            for e in [x.strip() for x in args.sf_elo.split(",") if x.strip()]:
                names.append("sf:elo%s:nodes=20000:"
                             "UCI_LimitStrength=true;UCI_Elo=%s" % (e, e))
        print("calibrating %d entrants, %d games per match, band %d, seed %d"
              % (len(names), args.games, args.band, args.seed))
        print("  " + " -> ".join(names))
        print()
        res = calibrate(names, cfg)
        print(render_calibration(res))
        if args.out:
            path = args.out if os.path.isabs(args.out) else os.path.join(REPO_ROOT, args.out)
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "w") as f:
                json.dump(res, f, indent=1)
            print("\nwrote %s" % path)
        return 0

    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
