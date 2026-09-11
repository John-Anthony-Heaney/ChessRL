#!/usr/bin/env python3
"""External calibration for the ChessRL champion.

The Elo printed by the trainer is *self-referential*: it is an internal rating
computed inside one population, anchored to nothing.  This module measures the
champion against opponents that live outside that population, and reports the
result with an honest error bar.

    python3 py/benchmark.py --model runs/pilot/best.crl --games 200
    python3 py/benchmark.py --ladder --opponent 'uci:/opt/homebrew/bin/stockfish'
    python3 py/benchmark.py --self-ladder --self-games 60

Opponents
---------
``random``      uniformly random legal move.  Needs nothing installed, so the
                harness is useful on a bare machine.
``material``    1-ply greedy on material, random tie-break.  A slightly higher
                floor than random.
``uci:<cmd>``   any UCI engine driven over stdin/stdout pipes.  ``<cmd>`` is a
                full command line, e.g. ``uci:./build/chessrl uci`` or
                ``uci:/opt/homebrew/bin/stockfish``.  Options are passed through
                with ``--opp-option "Skill Level=0"`` and the search is
                controlled with ``--opp-go "nodes=100"``.

Measurement design (see docs/BENCHMARK.md for the long version)
---------------------------------------------------------------
* Colours alternate, and every opening position is played TWICE with colours
  reversed, so both engines get both sides of the same position.  Pair results
  are reported separately from game results, and the confidence interval is
  computed from the PAIR scores, because the two games of a pair are strongly
  correlated when the engines are near-deterministic.
* Openings come from a seeded RNG playing a few random legal plies (default 4).
  Both games of a pair start from the identical position.  Every game also gets
  its own RNG derived from ``(seed, game index)``, so a run is bit-for-bit
  reproducible regardless of ``--threads``.
* Elo uses ``elo = -400*log10(1/score - 1)``.  At 0% or 100% that is infinite;
  the harness says so and prints a one-sided bound instead of inventing a
  number.

Standard library only.  Python 3.9+.
"""

from __future__ import annotations

import argparse
import atexit
import json
import math
import os
import queue
import random
import shlex
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import engine as chessrl  # noqa: E402  (path juggling above is deliberate)

REPO_ROOT = _HERE.parent
DEFAULT_MODEL = REPO_ROOT / "runs" / "pilot" / "best.crl"
DEFAULT_MD = REPO_ROOT / "docs" / "BENCHMARK.md"

Z95 = 1.959963984540054
ALPHA = 0.05

STOP = threading.Event()


class BenchError(RuntimeError):
    """Anything the harness cannot honestly continue past."""


# ==========================================================================
# statistics
# ==========================================================================


def clamp01(x):
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def elo_from_score(score):
    """Standard logistic Elo difference.  ``None`` when score is 0 or 1."""
    if score is None or score <= 0.0 or score >= 1.0:
        return None
    return -400.0 * math.log10(1.0 / score - 1.0)


def score_from_elo(elo):
    return 1.0 / (1.0 + 10.0 ** (-elo / 400.0))


def los(wins, losses):
    """Likelihood of superiority: P(the tested side is genuinely stronger).

    The usual chess-match formula; draws carry no information about which side
    is better, so only decisive games enter it.
    """
    d = wins + losses
    if d <= 0:
        return 0.5
    return 0.5 * (1.0 + math.erf((wins - losses) / math.sqrt(2.0 * d)))


def interval_from_samples(samples, z=Z95):
    """95% interval for the mean of per-game (or per-pair) scores.

    Returns ``(mean, se, lo, hi, method)``.  The variance is the empirical
    variance of the observed outcomes, which automatically accounts for draws
    (a match that is all draws has less spread than one that is half wins and
    half losses).

    Degenerate case: if every sample is identical the empirical variance is
    zero and the interval would claim infinite precision.  That is a lie, so we
    fall back on the rule of three -- with ``n`` observations and none of them
    showing a different outcome, the 95% one-sided bound on the rate of that
    unseen outcome is ``1 - 0.05**(1/n)``, and the worst case is that every one
    of those unseen outcomes lands on the same side.
    """
    n = len(samples)
    if n == 0:
        return None, None, 0.0, 1.0, "no-samples"
    mean = sum(samples) / n
    if n < 2:
        return mean, None, 0.0, 1.0, "single-sample"
    var = sum((x - mean) ** 2 for x in samples) / (n - 1)
    se = math.sqrt(var / n)
    if se <= 0.0:
        eps = 1.0 - ALPHA ** (1.0 / n)
        half = eps * max(mean, 1.0 - mean)
        return mean, 0.0, clamp01(mean - half), clamp01(mean + half), "rule-of-three (all outcomes identical)"
    return mean, se, clamp01(mean - z * se), clamp01(mean + z * se), "normal approximation on the mean"


def one_sided_score_bound(n, direction):
    """Clopper-Pearson style 95% one-sided bound after a shutout.

    ``direction='upper'`` for 0 points in ``n`` games, ``'lower'`` for ``n``.
    """
    if n <= 0:
        return None
    if direction == "upper":
        return 1.0 - ALPHA ** (1.0 / n)
    return ALPHA ** (1.0 / n)


def summarize_match(games, label=""):
    """Turn a list of per-game records into the full statistical summary."""
    n = len(games)
    scores = [g["score_a"] for g in games]
    wins = sum(1 for s in scores if s == 1.0)
    draws = sum(1 for s in scores if s == 0.5)
    losses = sum(1 for s in scores if s == 0.0)

    # pair scores: both games of a pair share an opening, colours reversed
    pairs = {}
    for g in games:
        pairs.setdefault(g["pair"], []).append(g["score_a"])
    complete_pairs = [v for v in pairs.values() if len(v) == 2]
    pair_scores = [sum(v) / 2.0 for v in complete_pairs]
    pair_sweeps = sum(1 for v in complete_pairs if v[0] == 1.0 and v[1] == 1.0)
    pair_losses = sum(1 for v in complete_pairs if v[0] == 0.0 and v[1] == 0.0)
    pair_both_draw = sum(1 for v in complete_pairs if v[0] == 0.5 and v[1] == 0.5)
    pair_split = len(complete_pairs) - pair_sweeps - pair_losses - pair_both_draw

    g_mean, g_se, _, _, _ = interval_from_samples(scores)
    if pair_scores:
        p_mean, p_se, p_lo, p_hi, p_method = interval_from_samples(pair_scores)
    else:
        p_mean, p_se, p_lo, p_hi, p_method = g_mean, g_se, 0.0, 1.0, "no complete pairs"

    score = g_mean if g_mean is not None else 0.0
    elo = {"point": None, "lo": None, "hi": None, "bound": None,
           "ci_method": p_method, "ci_basis": "pairs" if pair_scores else "games"}

    if n == 0:
        elo["note"] = "no games played"
    elif score <= 0.0:
        p_u = one_sided_score_bound(n, "upper")
        elo["bound"] = "upper"
        elo["hi"] = elo_from_score(p_u)
        elo["note"] = ("scored 0/%d -- the Elo difference is negative infinity; "
                       "reporting a one-sided 95%% bound instead" % n)
    elif score >= 1.0:
        p_l = one_sided_score_bound(n, "lower")
        elo["bound"] = "lower"
        elo["lo"] = elo_from_score(p_l)
        elo["note"] = ("scored %d/%d -- the Elo difference is positive infinity; "
                       "reporting a one-sided 95%% bound instead" % (n, n))
    else:
        elo["point"] = elo_from_score(score)
        elo["lo"] = elo_from_score(p_lo)   # None when the interval touches 0
        elo["hi"] = elo_from_score(p_hi)   # None when the interval touches 1
        notes = []
        if elo["lo"] is None:
            notes.append("lower end of the score interval reaches 0%, so the lower Elo bound is -inf")
        if elo["hi"] is None:
            notes.append("upper end of the score interval reaches 100%, so the upper Elo bound is +inf")
        if notes:
            elo["note"] = "; ".join(notes)

    plies = [g["plies"] for g in games]
    reasons = {}
    for g in games:
        reasons[g["reason"]] = reasons.get(g["reason"], 0) + 1

    return {
        "label": label,
        "games": n,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "score": score,
        "score_pct": 100.0 * score,
        "draw_rate": (draws / n) if n else 0.0,
        "white_games": sum(1 for g in games if g["a_white"]),
        "black_games": sum(1 for g in games if not g["a_white"]),
        "score_as_white": (sum(g["score_a"] for g in games if g["a_white"]) /
                           max(1, sum(1 for g in games if g["a_white"]))),
        "score_as_black": (sum(g["score_a"] for g in games if not g["a_white"]) /
                           max(1, sum(1 for g in games if not g["a_white"]))),
        "se_per_game": g_se,
        "se_per_pair": p_se,
        "score_ci": [p_lo, p_hi],
        "ci_method": p_method,
        "pairs": {
            "complete": len(complete_pairs),
            "swept_by_a": pair_sweeps,
            "swept_by_opponent": pair_losses,
            "both_drawn": pair_both_draw,
            "split": pair_split,
        },
        "elo": elo,
        "los": los(wins, losses),
        "avg_plies": (sum(plies) / n) if n else 0.0,
        "median_plies": (sorted(plies)[n // 2] if n else 0),
        "max_plies_seen": max(plies) if plies else 0,
        "ply_cap_games": sum(1 for g in games if g["reason"] == "ply cap"),
        "terminations": dict(sorted(reasons.items(), key=lambda kv: -kv[1])),
    }


# ==========================================================================
# UCI client
# ==========================================================================

_LIVE_PROCS = set()
_LIVE_LOCK = threading.Lock()


def _register_proc(p):
    with _LIVE_LOCK:
        _LIVE_PROCS.add(p)


def _unregister_proc(p):
    with _LIVE_LOCK:
        _LIVE_PROCS.discard(p)


def reap_all_children(*_args):
    """Kill every UCI child we ever started.  Safe to call repeatedly."""
    with _LIVE_LOCK:
        procs = list(_LIVE_PROCS)
        _LIVE_PROCS.clear()
    for p in procs:
        try:
            if p.poll() is None:
                p.kill()
        except Exception:
            pass
        try:
            p.wait(timeout=2)
        except Exception:
            pass


atexit.register(reap_all_children)


def _install_signal_handlers():
    if threading.current_thread() is not threading.main_thread():
        return

    def handler(signum, frame):
        STOP.set()
        reap_all_children()
        raise KeyboardInterrupt("signal %d" % signum)

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, handler)
        except (ValueError, OSError):
            pass


class UciError(BenchError):
    pass


class UciEngine:
    """A minimal, strict UCI client.

    Strict on purpose: every wait has a deadline and a missed deadline raises
    instead of hanging, and the child process is always killed on the way out
    (normal return, exception, or Ctrl-C).
    """

    def __init__(self, argv, options=None, timeout=30.0, debug=False):
        self.argv = list(argv)
        self.label = engine_label(self.argv)
        self.timeout = float(timeout)
        self.debug = debug
        self.id_name = None
        self.id_author = None
        self.options = {}      # name -> {'type','default','min','max','vars'}
        self.proc = None
        self._q = queue.Queue()
        self._closed = False
        self._log = []

        try:
            self.proc = subprocess.Popen(
                self.argv,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
                close_fds=True,
                cwd=str(REPO_ROOT),
            )
        except OSError as exc:
            raise UciError("cannot start UCI engine %r: %s"
                           % (" ".join(self.argv), exc)) from exc
        _register_proc(self.proc)

        self._reader = threading.Thread(target=self._pump, daemon=True,
                                        name="uci-reader")
        self._reader.start()

        try:
            self._handshake()
            for name, value in (options or {}).items():
                self.setoption(name, value)
            self.isready()
        except BaseException:
            self.close()
            raise

    # -- plumbing ---------------------------------------------------------
    def _pump(self):
        try:
            for line in self.proc.stdout:
                self._q.put(line.rstrip("\r\n"))
        except Exception:
            pass
        finally:
            self._q.put(None)

    def _send(self, line):
        if self.debug:
            sys.stderr.write("  >> %s\n" % line)
        self._log.append(">> " + line)
        if self.proc.poll() is not None:
            raise UciError("UCI engine %r exited (rc=%s) before '%s'"
                           % (self.label, self.proc.returncode, line))
        try:
            self.proc.stdin.write(line + "\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, ValueError, OSError) as exc:
            raise UciError("UCI engine %r closed its input while sending '%s': %s"
                           % (self.label, line, exc)) from exc

    def _readline(self, deadline, what):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise UciError("timed out after %.1fs waiting for %s from %r"
                           % (self.timeout, what, self.label))
        try:
            line = self._q.get(timeout=remaining)
        except queue.Empty:
            raise UciError("timed out after %.1fs waiting for %s from %r"
                           % (self.timeout, what, self.label)) from None
        if line is None:
            try:
                self.proc.wait(timeout=1.0)
            except Exception:
                pass
            raise UciError("UCI engine %r exited (rc=%s) while we waited for %s"
                           % (self.label, self.proc.returncode, what))
        if self.debug:
            sys.stderr.write("  << %s\n" % line)
        self._log.append("<< " + line)
        return line

    # -- protocol ---------------------------------------------------------
    def _handshake(self):
        self._send("uci")
        deadline = time.monotonic() + self.timeout
        while True:
            line = self._readline(deadline, "'uciok'")
            if line == "uciok":
                return
            if line.startswith("id name "):
                self.id_name = line[8:].strip()
            elif line.startswith("id author "):
                self.id_author = line[10:].strip()
            elif line.startswith("option name "):
                self._parse_option(line)

    def _parse_option(self, line):
        rest = line[len("option name "):]
        if " type " not in rest:
            return
        name, tail = rest.split(" type ", 1)
        toks = tail.split()
        info = {"type": toks[0] if toks else "string", "vars": []}
        i = 1
        while i < len(toks):
            key = toks[i]
            if key in ("default", "min", "max"):
                # value runs until the next keyword
                j = i + 1
                vals = []
                while j < len(toks) and toks[j] not in ("default", "min", "max", "var"):
                    vals.append(toks[j])
                    j += 1
                info[key] = " ".join(vals)
                i = j
            elif key == "var":
                if i + 1 < len(toks):
                    info["vars"].append(toks[i + 1])
                i += 2
            else:
                i += 1
        self.options[name.strip()] = info

    def setoption(self, name, value):
        if value is None or value == "":
            self._send("setoption name %s" % name)
        else:
            self._send("setoption name %s value %s" % (name, value))

    def isready(self):
        self._send("isready")
        deadline = time.monotonic() + self.timeout
        while True:
            if self._readline(deadline, "'readyok'") == "readyok":
                return

    def newgame(self):
        self._send("ucinewgame")
        self.isready()

    def bestmove(self, moves, go_spec, extra_timeout=0.0):
        """``position startpos moves ...`` + ``go <spec>`` -> (bestmove, score_cp).

        ``score_cp`` is the last score the engine reported, from the point of
        view of the side to move, or ``None`` if it reported none.
        """
        if moves:
            self._send("position startpos moves " + " ".join(moves))
        else:
            self._send("position startpos")
        self._send("go " + go_spec)
        deadline = time.monotonic() + self.timeout + extra_timeout
        score = None
        while True:
            line = self._readline(deadline, "'bestmove'")
            if line.startswith("info "):
                s = _parse_info_score(line)
                if s is not None:
                    score = s
            elif line.startswith("bestmove"):
                parts = line.split()
                mv = parts[1] if len(parts) > 1 else ""
                if mv in ("", "(none)", "0000", "NULL"):
                    raise UciError("UCI engine %r answered '%s' in a position "
                                   "where we asked it to move" % (self.label, line))
                return mv, score

    # -- lifetime ---------------------------------------------------------
    def close(self):
        if self._closed:
            return
        self._closed = True
        p = self.proc
        if p is None:
            return
        try:
            if p.poll() is None:
                try:
                    p.stdin.write("quit\n")
                    p.stdin.flush()
                except Exception:
                    pass
                try:
                    p.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    p.terminate()
                    try:
                        p.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        p.kill()
                        try:
                            p.wait(timeout=2)
                        except Exception:
                            pass
        finally:
            for stream in (p.stdin, p.stdout):
                try:
                    if stream:
                        stream.close()
                except Exception:
                    pass
            _unregister_proc(p)
            try:
                self._reader.join(timeout=1.0)
            except Exception:
                pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def _parse_info_score(line):
    """cp score (side-to-move POV) from an ``info`` line; mates map to +-30000."""
    toks = line.split()
    try:
        i = toks.index("score")
    except ValueError:
        return None
    if i + 2 >= len(toks):
        return None
    kind, raw = toks[i + 1], toks[i + 2]
    try:
        val = int(raw)
    except ValueError:
        return None
    if kind == "cp":
        return val
    if kind == "mate":
        if val == 0:
            return -30000
        return (30000 - abs(val)) * (1 if val > 0 else -1)
    return None


# ==========================================================================
# players
# ==========================================================================


class Player:
    """One side of a game.  ``pick`` returns a UCI move, it does not play it."""

    kind = "player"
    reports_eval = False

    def __init__(self, name):
        self.name = name
        self.last_eval_cp = None     # side-to-move POV, cp
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
        return {"moves": self.moves_made, "ms": self.move_ms}

    def describe(self):
        return self.name


class RandomPlayer(Player):
    kind = "random"

    def __init__(self, name="random"):
        super().__init__(name)

    def pick(self, game, state, history, rng):
        self.moves_made += 1
        return rng.choice(state["legal"])


class MaterialPlayer(Player):
    """1-ply greedy on material, uniform random tie-break.

    Value of a move, from the mover's side: ``my_material - their_material``
    after the move, with a mate scored 10000 and any drawn result scored 0.
    """

    kind = "material"

    def __init__(self, name="material"):
        super().__init__(name)

    def pick(self, game, state, history, rng):
        self.moves_made += 1
        me = state["turn"]
        them = "black" if me == "white" else "white"
        want = 1 if me == "white" else 2
        fen_before = state["fen"]
        best_val = None
        best = []
        for uci in state["legal"]:
            if not game.move(uci):
                raise BenchError("material player: library rejected legal move %s in %s"
                                 % (uci, fen_before))
            st2 = game.state()
            res = st2["result"]
            if res == want:
                val = 10000.0
            elif res == 3:
                val = 0.0
            else:
                m = st2["material"]
                val = float(m[me] - m[them])
            if not game.undo():
                raise BenchError("material player: undo failed after %s in %s"
                                 % (uci, fen_before))
            if best_val is None or val > best_val:
                best_val, best = val, [uci]
            elif val == best_val:
                best.append(uci)
        if game.state()["fen"] != fen_before:
            raise BenchError("material player corrupted the referee position")
        if not best:
            raise BenchError("material player found no move in %s" % fen_before)
        return rng.choice(best)


class ChampionPlayer(Player):
    """Our trained agent, either raw policy or alpha-beta search."""

    kind = "champion"
    reports_eval = True

    def __init__(self, model, agent_index=-1, mode="search", depth=4,
                 movetime=1000, policy_temp=0.0, name=None):
        if name is None:
            name = "champion policy" if mode == "policy" else "champion d%d" % depth
        super().__init__(name)
        self.mode = mode
        self.depth = int(depth)
        self.movetime = int(movetime)
        self.policy_temp = float(policy_temp)
        self.model = model
        self.agent_index = agent_index
        self.reloads = 0
        self.truncated = 0          # searches that stopped before the target depth
        self.eng = chessrl.Engine(model, agent_index)

    def new_game(self, opening):
        super().new_game(opening)
        if self.mode != "policy":
            # A fresh engine handle for every game.  The search keeps a
            # transposition table (and killer/history tables) inside its slot,
            # so carrying one over from the previous game would make a game's
            # moves depend on which games happened to run before it on this
            # thread -- i.e. on --threads.  Reloading costs ~10ms and buys
            # bit-for-bit reproducibility at any thread count.
            try:
                self.eng.close()
            except Exception:
                pass
            self.eng = chessrl.Engine(self.model, self.agent_index)
            self.reloads += 1

    def pick(self, game, state, history, rng):
        self.moves_made += 1
        if self.mode == "policy":
            self.last_eval_cp = None
            pol = self.eng.policy(game)
            if not pol:
                return rng.choice(state["legal"])
            if self.policy_temp > 0.0:
                return _sample_policy(pol, self.policy_temp, rng)
            return max(pol, key=lambda d: d.get("prob", 0.0))["move"]
        t0 = time.monotonic()
        r = self.eng.move(game, depth=self.depth, movetime_ms=self.movetime)
        self.move_ms += (time.monotonic() - t0) * 1000.0
        self.nodes += int(r.get("nodes") or 0)
        if int(r.get("depth") or 0) < self.depth:
            self.truncated += 1
        self.last_eval_cp = r.get("score")
        mv = r.get("move") or ""
        if not mv:
            return rng.choice(state["legal"])
        return mv

    def stats(self):
        d = {"moves": self.moves_made, "ms": self.move_ms}
        if self.mode != "policy":
            d.update({"nodes": self.nodes, "truncated_searches": self.truncated,
                      "engine_reloads": self.reloads})
        return d

    def close(self):
        try:
            self.eng.close()
        except Exception:
            pass

    def describe(self):
        if self.mode == "policy":
            return "champion, raw policy (no search)%s" % (
                ", temp %.2f" % self.policy_temp if self.policy_temp > 0 else ", argmax")
        return "champion, alpha-beta depth %d (movetime cap %dms)" % (self.depth, self.movetime)


def _sample_policy(pol, temp, rng):
    logits = [d.get("logit", 0.0) for d in pol]
    mx = max(logits)
    ws = [math.exp((x - mx) / max(1e-6, temp)) for x in logits]
    tot = sum(ws)
    if tot <= 0:
        return pol[0]["move"]
    r = rng.random() * tot
    acc = 0.0
    for d, w in zip(pol, ws):
        acc += w
        if r <= acc:
            return d["move"]
    return pol[-1]["move"]


class UciPlayer(Player):
    kind = "uci"
    reports_eval = True

    def __init__(self, argv, options=None, go_spec="movetime=100", timeout=30.0,
                 name=None, debug=False):
        self.argv = list(argv)
        self.options = dict(options or {})
        self.go_spec, self._extra_timeout = _compile_go(go_spec)
        self.eng = UciEngine(self.argv, self.options, timeout=timeout, debug=debug)
        super().__init__(name or (self.eng.id_name or engine_label(self.argv)))

    def new_game(self, opening):
        super().new_game(opening)
        self.eng.newgame()

    def pick(self, game, state, history, rng):
        self.moves_made += 1
        t0 = time.monotonic()
        mv, score = self.eng.bestmove(history, self.go_spec,
                                      extra_timeout=self._extra_timeout)
        self.move_ms += (time.monotonic() - t0) * 1000.0
        self.last_eval_cp = score
        return mv

    def close(self):
        try:
            self.eng.close()
        except Exception:
            pass

    def describe(self):
        bits = ["%s" % " ".join(self.argv), "go %s" % self.go_spec]
        if self.options:
            bits.append("options " + ", ".join("%s=%s" % kv for kv in sorted(self.options.items())))
        return "; ".join(bits)


def _compile_go(spec):
    """``"nodes=100"`` / ``"movetime=10 depth=2"`` -> (``"nodes 100"``, extra_timeout_s)."""
    spec = (spec or "").strip()
    if not spec:
        return "depth 1", 0.0
    if " " in spec and "=" not in spec:
        # already a raw UCI go argument list, e.g. "nodes 100"
        toks = spec.split()
        extra = 0.0
        for i, t in enumerate(toks):
            if t == "movetime" and i + 1 < len(toks):
                try:
                    extra = int(toks[i + 1]) / 1000.0 * 2.0
                except ValueError:
                    pass
        return spec, extra
    out = []
    extra = 0.0
    for chunk in spec.replace(",", " ").split():
        if "=" in chunk:
            k, v = chunk.split("=", 1)
            k, v = k.strip(), v.strip()
            out.extend([k, v])
            if k == "movetime":
                try:
                    extra = int(v) / 1000.0 * 2.0
                except ValueError:
                    pass
        else:
            out.append(chunk)
    return " ".join(out), extra


# ==========================================================================
# player specs (a factory + a human-readable description)
# ==========================================================================


class PlayerSpec:
    def __init__(self, name, factory, detail=""):
        self.name = name
        self.factory = factory
        self.detail = detail

    def make(self):
        return self.factory()


_INTERPRETERS = {"python", "python3", "python3.9", "python3.10", "python3.11",
                 "python3.12", "python3.13", "python3.14", "sh", "bash", "zsh",
                 "env", "node", "ruby", "perl", "wine"}


def engine_label(argv):
    """A short, human-readable name for a UCI command line."""
    for tok in argv:
        base = Path(tok).name
        if base.startswith("-"):
            continue
        if base in _INTERPRETERS:
            continue
        return base
    return Path(argv[0]).name


def parse_uci_spec(spec):
    """``uci:./build/chessrl uci`` -> argv list."""
    body = spec[4:].strip()
    if not body:
        raise BenchError("empty UCI command in %r" % spec)
    return shlex.split(body)


def opponent_spec(args, go_spec=None, options=None, label=None):
    """Build a PlayerSpec from ``args.opponent`` (with per-rung overrides)."""
    s = args.opponent
    if s == "random":
        return PlayerSpec("random", lambda: RandomPlayer(), "uniformly random legal move")
    if s == "material":
        return PlayerSpec("material", lambda: MaterialPlayer(),
                          "1-ply greedy on material, random tie-break")
    if s.startswith("uci:"):
        argv = parse_uci_spec(s)
        opts = dict(args.opp_option_map)
        if options:
            opts.update(options)
        go = go_spec or args.opp_go
        nm = label or ("uci:" + engine_label(argv))
        detail_bits = ["%s" % " ".join(argv), "go %s" % _compile_go(go)[0]]
        if opts:
            detail_bits.append(", ".join("%s=%s" % kv for kv in sorted(opts.items())))
        return PlayerSpec(nm,
                          lambda: UciPlayer(argv, opts, go, args.opp_timeout,
                                            name=nm, debug=args.debug_uci),
                          "; ".join(detail_bits))
    raise BenchError("unknown opponent %r (expected 'random', 'material' or 'uci:<command>')" % s)


def champion_spec(args, mode=None, depth=None, name=None):
    mode = mode or ("policy" if args.policy_only else "search")
    depth = args.depth if depth is None else depth
    nm = name or ("champion policy" if mode == "policy" else "champion d%d" % depth)

    def make():
        return ChampionPlayer(args.model, args.agent, mode, depth,
                              args.movetime, args.policy_temp, name=nm)

    if mode == "policy":
        detail = "raw policy head, argmax" if args.policy_temp <= 0 else \
                 "raw policy head, sampled at temperature %.2f" % args.policy_temp
    else:
        detail = "alpha-beta depth %d, movetime cap %dms" % (depth, args.movetime)
    return PlayerSpec(nm, make, detail)


# ==========================================================================
# one game
# ==========================================================================


def play_game(pa, pb, a_white, opening, ply_cap, rng, resign_cfg, adjudicator):
    """Play one game.  ``pa`` is the side being measured; score is from its POV."""
    white, black = (pa, pb) if a_white else (pb, pa)
    pa.new_game(opening)
    pb.new_game(opening)

    resign_cp, resign_plies = resign_cfg
    streak_sign = 0
    streak_len = 0

    with chessrl.GameHandle() as g:
        for uci in opening:
            if not g.move(uci):
                raise BenchError("opening move %s was rejected" % uci)
        st = g.state()
        history = list(opening)
        result = reason = None

        while True:
            if st["result"] != 0:
                result = st["result"]
                reason = st["reason"] or "terminal"
                break
            if st["ply"] >= ply_cap:
                result, reason = 3, "ply cap"
                break
            if STOP.is_set():
                raise KeyboardInterrupt("stopped")
            if not st["legal"]:
                raise BenchError("no legal moves but result is 0: %s" % st["fen"])

            side = st["turn"]
            mover = white if side == "white" else black
            uci = mover.pick(g, st, history, rng)
            if uci not in st["legal"]:
                result = 2 if side == "white" else 1
                reason = "illegal move (%s played %r)" % (mover.name, uci)
                sys.stderr.write("!! %s produced the illegal move %r in %s -- "
                                 "scoring the game as a loss for it\n"
                                 % (mover.name, uci, st["fen"]))
                break
            if not g.move(uci):
                raise BenchError("library rejected the legal move %s in %s" % (uci, st["fen"]))
            history.append(uci)

            # optional adjudication from the designated reporter's own eval
            if resign_cp > 0 and adjudicator is not None and mover is adjudicator:
                cp = mover.last_eval_cp
                if cp is not None:
                    white_cp = cp if side == "white" else -cp
                    sign = 1 if white_cp > 0 else (-1 if white_cp < 0 else 0)
                    if abs(white_cp) >= resign_cp and sign != 0 and sign == streak_sign:
                        streak_len += 1
                    elif abs(white_cp) >= resign_cp and sign != 0:
                        streak_sign, streak_len = sign, 1
                    else:
                        streak_sign, streak_len = 0, 0
                else:
                    streak_sign, streak_len = 0, 0

            st = g.state()

            if resign_cp > 0 and streak_len >= resign_plies and st["result"] == 0:
                result = 1 if streak_sign > 0 else 2
                reason = "adjudicated (%s reported >=%dcp for %d plies)" % (
                    adjudicator.name, resign_cp, streak_len)
                break

        if result == 3:
            score_a = 0.5
        elif result == 1:
            score_a = 1.0 if a_white else 0.0
        else:
            score_a = 0.0 if a_white else 1.0

        return {
            "result": result,
            "reason": reason,
            "score_a": score_a,
            "a_white": a_white,
            "plies": st["ply"],
            "opening": list(opening),
            "moves": history,
        }


# ==========================================================================
# openings
# ==========================================================================


def make_openings(n, plies, seed):
    """``n`` distinct opening lines of ``plies`` random legal plies, seeded."""
    if n <= 0:
        return []
    if plies <= 0:
        return [[] for _ in range(n)]
    rng = random.Random("chessrl-openings|%s|%d" % (seed, plies))
    out, seen = [], set()

    def draw():
        moves = []
        with chessrl.GameHandle() as g:
            for _ in range(plies):
                st = g.state()
                if st["result"] != 0 or not st["legal"]:
                    return None
                mv = rng.choice(st["legal"])
                if not g.move(mv):
                    return None
                moves.append(mv)
            if g.state()["result"] != 0:
                return None
        return moves

    attempts = 0
    budget = max(500, n * 100)
    while len(out) < n and attempts < budget:
        attempts += 1
        moves = draw()
        if moves is None:
            continue
        key = tuple(moves)
        if key in seen:
            continue
        seen.add(key)
        out.append(moves)
    while len(out) < n:               # exhausted the distinct-line budget
        moves = draw()
        if moves is not None:
            out.append(moves)
    return out


# ==========================================================================
# match
# ==========================================================================


def run_match(spec_a, spec_b, n_games, args, label=None, quiet=False):
    """Play ``n_games`` (rounded up to an even number) colour-balanced games."""
    n_games = int(n_games)
    if n_games < 2:
        n_games = 2
    if n_games % 2:
        n_games += 1
    n_pairs = n_games // 2
    openings = make_openings(n_pairs, args.opening_plies, args.seed)
    label = label or ("%s vs %s" % (spec_a.name, spec_b.name))

    local = threading.local()
    created = []
    created_lock = threading.Lock()

    def players():
        pair = getattr(local, "pair", None)
        if pair is None:
            a = spec_a.make()
            try:
                b = spec_b.make()
            except BaseException:
                a.close()
                raise
            with created_lock:
                created.append(a)
                created.append(b)
            pair = (a, b)
            local.pair = pair
        return pair

    def one(i):
        a, b = players()
        rng = random.Random("chessrl-game|%s|%d" % (args.seed, i))
        rec = play_game(a, b, i % 2 == 0, openings[i // 2], args.ply_cap, rng,
                        (args.resign_cp, args.resign_plies), b)
        rec["index"] = i
        rec["pair"] = i // 2
        return rec

    t0 = time.monotonic()
    results = [None] * n_games
    threads = max(1, min(int(args.threads), n_games))
    done = 0
    try:
        if threads == 1:
            for i in range(n_games):
                results[i] = one(i)
                done += 1
                if not quiet:
                    _progress(label, done, n_games, results, t0)
        else:
            with ThreadPoolExecutor(max_workers=threads) as ex:
                futs = [ex.submit(one, i) for i in range(n_games)]
                try:
                    for fut in as_completed(futs):
                        rec = fut.result()
                        results[rec["index"]] = rec
                        done += 1
                        if not quiet:
                            _progress(label, done, n_games, results, t0)
                except BaseException:
                    STOP.set()
                    for f in futs:
                        f.cancel()
                    raise
    except KeyboardInterrupt:
        STOP.set()
        reap_all_children()
        raise
    finally:
        if not quiet:
            sys.stderr.write("\n")
        with created_lock:
            stats_a = _merge_stats(created[0::2])
            stats_b = _merge_stats(created[1::2])
            for p in created:
                try:
                    p.close()
                except Exception:
                    pass

    games = [r for r in results if r is not None]
    summary = summarize_match(games, label=label)
    summary["elapsed_s"] = time.monotonic() - t0
    summary["a"] = {"name": spec_a.name, "detail": spec_a.detail, "stats": stats_a}
    summary["b"] = {"name": spec_b.name, "detail": spec_b.detail, "stats": stats_b}
    summary["opening_plies"] = args.opening_plies
    summary["ply_cap"] = args.ply_cap
    summary["seed"] = args.seed
    summary["_games"] = games
    return summary


def _merge_stats(players):
    out = {}
    for p in players:
        try:
            st = p.stats()
        except Exception:
            continue
        for k, v in st.items():
            out[k] = out.get(k, 0) + v
    if out.get("moves"):
        out["ms_per_move"] = out.get("ms", 0.0) / out["moves"]
        if "nodes" in out:
            out["nodes_per_move"] = out["nodes"] / out["moves"]
    return out


def _progress(label, done, total, results, t0):
    part = [r for r in results if r is not None]
    s = sum(r["score_a"] for r in part) / max(1, len(part))
    el = time.monotonic() - t0
    eta = (el / max(1, done)) * (total - done)
    sys.stderr.write("\r  %-34s %3d/%-3d  score %5.1f%%  %5.1fs elapsed, ~%.0fs left   "
                     % (label[:34], done, total, 100.0 * s, el, eta))
    sys.stderr.flush()


# ==========================================================================
# ladder
# ==========================================================================


def probe_uci(args):
    """Start the opponent once to learn its name and its option list."""
    argv = parse_uci_spec(args.opponent)
    eng = UciEngine(argv, {}, timeout=args.opp_timeout, debug=args.debug_uci)
    try:
        return {"argv": argv, "id_name": eng.id_name, "options": dict(eng.options)}
    finally:
        eng.close()


def build_ladder(args, probe):
    """The graded series of opponents, weakest first."""
    rungs = [
        {"name": "random", "spec": PlayerSpec("random", lambda: RandomPlayer(),
                                              "uniformly random legal move"),
         "detail": "uniformly random legal move", "anchor_elo": None},
        {"name": "material", "spec": PlayerSpec("material", lambda: MaterialPlayer(),
                                                "1-ply greedy on material"),
         "detail": "1-ply greedy on material, random tie-break", "anchor_elo": None},
    ]
    if probe is None:
        return rungs

    ename = probe["id_name"] or engine_label(probe["argv"])
    opts = probe["options"]

    for nodes in (1, 10, 100, 1000):
        nm = "%s nodes=%d" % (ename, nodes)
        rungs.append({
            "name": nm,
            "spec": opponent_spec(args, go_spec="nodes=%d" % nodes, label=nm),
            "detail": "%s, go nodes %d" % (ename, nodes),
            "anchor_elo": None,
        })

    if "Skill Level" in opts:
        lo = opts["Skill Level"].get("min", "0")
        nm = "%s Skill=%s" % (ename, lo)
        rungs.append({
            "name": nm,
            "spec": opponent_spec(args, go_spec=args.opp_go or "movetime=10",
                                  options={"Skill Level": lo}, label=nm),
            "detail": "%s, Skill Level %s, go %s" % (ename, lo, _compile_go(args.opp_go or 'movetime=10')[0]),
            "anchor_elo": None,
        })

    if "UCI_LimitStrength" in opts and "UCI_Elo" in opts:
        floor = opts["UCI_Elo"].get("min", "1320")
        try:
            anchor = float(floor)
        except ValueError:
            anchor = None
        nm = "%s UCI_Elo=%s" % (ename, floor)
        rungs.append({
            "name": nm,
            "spec": opponent_spec(args, go_spec=args.opp_go or "movetime=100",
                                  options={"UCI_LimitStrength": "true", "UCI_Elo": floor},
                                  label=nm),
            "detail": "%s, UCI_LimitStrength true, UCI_Elo %s (its floor)" % (ename, floor),
            "anchor_elo": anchor,
        })

    return rungs


def run_ladder(args, out, quiet=False):
    probe = None
    probe_err = None
    if args.opponent.startswith("uci:"):
        try:
            probe = probe_uci(args)
        except BenchError as exc:
            probe_err = str(exc)
            sys.stderr.write("!! could not start the external engine: %s\n" % exc)
    rungs = build_ladder(args, probe)

    champ = champion_spec(args)
    entries = []
    stopped_at = None
    for i, rung in enumerate(rungs):
        anchored = rung.get("anchor_elo") is not None
        if stopped_at is not None and not (anchored and not args.no_anchor_rung):
            entries.append({"name": rung["name"], "detail": rung["detail"],
                            "played": False, "skipped_because":
                            "the champion scored below %.0f%% on '%s'"
                            % (100 * args.ladder_stop, stopped_at)})
            continue
        if stopped_at is not None:
            # This rung carries a real rating anchor, so play it even though the
            # ladder has already bottomed out: a shutout against a calibrated
            # opponent still converts into an absolute upper bound, which is the
            # whole point of the exercise.
            sys.stderr.write("   (playing the rating-anchored rung '%s' anyway, "
                             "for an absolute bound)\n" % rung["name"])
        try:
            s = run_match(champ, rung["spec"], args.ladder_games, args,
                          label="ladder: %s" % rung["name"], quiet=quiet)
        except BenchError as exc:
            entries.append({"name": rung["name"], "detail": rung["detail"],
                            "played": False, "error": str(exc)})
            sys.stderr.write("!! rung '%s' failed: %s\n" % (rung["name"], exc))
            continue
        e = {"name": rung["name"], "detail": rung["detail"], "played": True,
             "anchor_elo": rung.get("anchor_elo"), "summary": s,
             "played_after_stop": stopped_at is not None}
        if rung.get("anchor_elo") is not None:
            a = rung["anchor_elo"]
            el = s["elo"]
            if el["point"] is not None:
                e["absolute_elo"] = {
                    "anchor": a, "kind": "estimate",
                    "point": a + el["point"],
                    "lo": (a + el["lo"]) if el["lo"] is not None else None,
                    "hi": (a + el["hi"]) if el["hi"] is not None else None,
                }
            elif el.get("bound") == "upper" and el.get("hi") is not None:
                e["absolute_elo"] = {"anchor": a, "kind": "upper bound",
                                     "point": None, "lo": None, "hi": a + el["hi"]}
            elif el.get("bound") == "lower" and el.get("lo") is not None:
                e["absolute_elo"] = {"anchor": a, "kind": "lower bound",
                                     "point": None, "lo": a + el["lo"], "hi": None}
        entries.append(e)
        if s["score"] < args.ladder_stop and stopped_at is None:
            stopped_at = rung["name"]

    crossing = ladder_crossing(entries)
    out["ladder"] = {
        "rungs": entries,
        "games_per_rung": args.ladder_games,
        "stop_threshold": args.ladder_stop,
        "stopped_after": stopped_at,
        "crossing": crossing,
        "external_engine": probe["id_name"] if probe else None,
        "external_engine_error": probe_err,
    }
    return out["ladder"]


def ladder_crossing(entries):
    """Where the champion's score crosses 50%, interpolated between rungs.

    The rungs are ordered by the champion's MEASURED score, not by the order we
    happened to configure them in.  A configured ladder is only a guess at the
    strength ordering -- Stockfish at ``UCI_Elo 1320`` and Stockfish at ``go
    nodes 1000`` are not obviously ranked against each other, and the match
    result is the evidence, not the guess.  ``configured_order_held`` records
    whether the guess survived contact with the data.
    """
    played = [e for e in entries if e.get("played")]
    if not played:
        return {"status": "no rungs played"}
    configured = [e["summary"]["score"] for e in played]
    held = all(configured[i] >= configured[i + 1] - 1e-12
               for i in range(len(configured) - 1))
    played = sorted(played, key=lambda e: -e["summary"]["score"])
    scores = [e["summary"]["score"] for e in played]
    note = {"configured_order_held": held,
            "measured_order": [e["name"] for e in played]}
    if scores[0] < 0.5:
        return dict(note, status="below the bottom of the ladder",
                    detail="the champion scored %.1f%% against the weakest rung it "
                           "faced (%s), so the 50%% crossing is below the whole ladder"
                           % (100 * scores[0], played[0]["name"]))
    if scores[-1] >= 0.5:
        return dict(note, status="above the top of the ladder",
                    detail="the champion still scored %.1f%% against the strongest rung "
                           "it faced (%s), so the 50%% crossing is above the whole "
                           "ladder" % (100 * scores[-1], played[-1]["name"]))
    for i in range(1, len(played)):
        lo_s, hi_s = scores[i - 1], scores[i]
        if lo_s >= 0.5 > hi_s:
            denom = (lo_s - hi_s)
            t = (lo_s - 0.5) / denom if denom > 0 else 0.0
            return dict(
                note,
                status="bracketed",
                lower_rung=played[i - 1]["name"],
                upper_rung=played[i]["name"],
                lower_score=lo_s,
                upper_score=hi_s,
                fraction=t,
                detail="50%% falls between '%s' (%.1f%%) and '%s' (%.1f%%), "
                       "%.0f%% of the way up"
                       % (played[i - 1]["name"], 100 * lo_s,
                          played[i]["name"], 100 * hi_s, 100 * t),
            )
    return dict(note, status="not bracketed",
                detail="the score never crossed 50% cleanly")


# ==========================================================================
# self-ladder
# ==========================================================================


def run_self_ladder(args, out, quiet=False):
    base = champion_spec(args, mode="policy", name="champion policy")
    entries = []
    for d in args.self_depths:
        opp = champion_spec(args, mode="search", depth=d, name="champion d%d" % d)
        s = run_match(base, opp, args.self_games, args,
                      label="self: policy vs depth %d" % d, quiet=quiet)
        # Elo is reported from the raw policy's point of view; the gap the
        # search buys is the negation of that.
        gap = None
        gap_lo = gap_hi = None
        if s["elo"]["point"] is not None:
            gap = -s["elo"]["point"]
            gap_lo = -s["elo"]["hi"] if s["elo"]["hi"] is not None else None
            gap_hi = -s["elo"]["lo"] if s["elo"]["lo"] is not None else None
        entries.append({"depth": d, "summary": s, "search_gain_elo": gap,
                        "search_gain_lo": gap_lo, "search_gain_hi": gap_hi})
    out["self_ladder"] = {"baseline": "champion raw policy (argmax, no search)",
                          "games_per_rung": args.self_games,
                          "rungs": entries}
    return out["self_ladder"]


# ==========================================================================
# rating-scale conversion
#
# READ docs/RATING_SCALES.md BEFORE TRUSTING ANYTHING IN THIS SECTION.
#
# What the harness measures is a score against Stockfish running with
# UCI_LimitStrength, turned into an Elo difference and added to that opponent's
# UCI_Elo setting.  The result lives on Stockfish's UCI_Elo scale, which
# Stockfish's own source says is a fit "anchored to the Stash engine", covering
# "CCRL Blitz Elo from 1320 to 3190, approximately" (src/search.h).  CCRL Blitz
# is an ENGINE-vs-ENGINE rating list.  It is not FIDE, and it is emphatically
# not a chess.com Glicko rating.
#
# Everything below is therefore a POOL CONVERSION, not a measurement, and it
# carries far more uncertainty than the match does.
# ==========================================================================

SCALES = ("uci", "chesscom", "fide", "lichess")

SCALE_LABELS = {
    "uci": "Stockfish UCI_Elo",
    "chesscom": "chess.com Rapid",
    "fide": "FIDE standard",
    "lichess": "Lichess Rapid",
}


def convert_rating(uci_elo, scale):
    """Map a UCI_Elo figure onto another rating pool.

    Returns a dict.  ``in_range`` is False when the input falls outside the
    anchors, in which case ``point`` is the value AT the nearest anchor and
    ``bound_side`` says which side the input fell off.  The caller must not
    present an out-of-range result as an estimate.
    """
    # ----------------------------------------------------------------------
    # THE ANCHOR TABLE.  Change these numbers to change the assumptions.
    #
    # Built in two documented steps:
    #
    #   step 1 (ASSUMPTION, one number):  fide_equivalent = UCI_Elo + OFFSET,
    #       and we set OFFSET = 0.  There is no published study that fixes it.
    #       The two arguments that exist point in OPPOSITE directions and
    #       bracket it at roughly -400 .. +300:
    #         * CCRL's low-end engine ratings are widely held to UNDERSTATE
    #           human-equivalent strength (TalkChess: an engine at CCRL 40/4
    #           1712 plays "not that far from FIDE 2000").  That argues the
    #           FIDE equivalent is HIGHER than the UCI_Elo number, i.e. OFFSET
    #           around +300.
    #         * a deliberately-handicapped strong engine blunders in a way no
    #           human of that rating does, and play reports of Stockfish's
    #           lowest settings put them well below their label.  That argues
    #           the FIDE equivalent is LOWER, i.e. OFFSET around -400.
    #       0 is the midpoint of that bracket, not a finding.  The bracket's
    #       WIDTH is what CONV95 below is sized to cover.
    #
    #   step 2 (SOURCED):  FIDE <-> chess.com Rapid <-> Lichess Rapid, from the
    #       ChessGoals rating comparison (updated July 2026, ~20,000 profiles).
    #       Their FIDE column stops: they state there is "no accurate data for
    #       players under about 1550 FIDE".  Rows at or above UCI_Elo 1740 sit
    #       on their data; the two rows below it continue the lowest sourced
    #       segment's slope and are marked "extrapolated" for that reason.
    #
    # Columns: UCI_Elo, chess.com Rapid, FIDE, Lichess Rapid, provenance.
    # ----------------------------------------------------------------------
    ANCHORS = [
        (1320, 1020, 1320, 1440, "extrapolated"),  # Stockfish's own UCI_Elo floor
        (1500, 1290, 1500, 1640, "extrapolated"),
        (1740, 1655, 1740, 1905, "sourced"),       # ChessGoals: FIDE 1740 / cc blitz 1500
        (1965, 1995, 1965, 2165, "sourced"),       # ChessGoals: FIDE 1965 / cc blitz 2000
        (2245, 2260, 2245, 2400, "sourced"),       # ChessGoals: FIDE 2245 / cc blitz 2500
        (2485, 2430, 2485, 2615, "sourced"),       # ChessGoals: FIDE 2485 / cc blitz 3000
    ]

    # 95% half-width of the POOL-CONVERSION error alone, in target-pool points.
    # Sized from the +/-350ish bracket on step 1 carried through the local slope
    # of the map, NOT from any published interval -- no such interval exists.
    CONV95 = {"chesscom": 450, "fide": 350, "lichess": 400}
    # Added in quadrature on the two rows that sit below the sourced FIDE data.
    EXTRAP_EXTRA95 = 150

    col = {"chesscom": 1, "fide": 2, "lichess": 3}[scale]
    lo_anchor, hi_anchor = ANCHORS[0], ANCHORS[-1]

    def interp(x):
        """Piecewise-linear lookup; also returns the provenance of the segment."""
        if x <= ANCHORS[0][0]:
            return float(ANCHORS[0][col]), ANCHORS[0][4]
        if x >= ANCHORS[-1][0]:
            return float(ANCHORS[-1][col]), ANCHORS[-1][4]
        for a, b in zip(ANCHORS, ANCHORS[1:]):
            if a[0] <= x <= b[0]:
                t = (x - a[0]) / float(b[0] - a[0])
                prov = "extrapolated" if "extrapolated" in (a[4], b[4]) else "sourced"
                return a[col] + t * (b[col] - a[col]), prov
        return float(ANCHORS[-1][col]), ANCHORS[-1][4]

    res = {
        "scale": scale,
        "label": SCALE_LABELS[scale],
        "anchor_range": (lo_anchor[0], hi_anchor[0]),
        "in_range": True,
        "bound_side": None,
        "provenance": None,
        "conv95": None,
        "point": None,
    }

    if uci_elo is None:
        res["in_range"] = False
        res["bound_side"] = "none"
        return res

    if uci_elo < lo_anchor[0]:
        res["in_range"] = False
        res["bound_side"] = "below"
    elif uci_elo > hi_anchor[0]:
        res["in_range"] = False
        res["bound_side"] = "above"

    clamped = min(max(uci_elo, lo_anchor[0]), hi_anchor[0])
    value, prov = interp(clamped)
    conv = CONV95[scale]
    if prov == "extrapolated":
        conv = math.sqrt(conv ** 2 + EXTRAP_EXTRA95 ** 2)

    res["point"] = value
    res["provenance"] = prov
    res["conv95"] = conv
    res["clamped_from"] = uci_elo
    res["clamped_to"] = clamped
    return res


def convert_interval(point, lo, hi, scale):
    """Convert a measured UCI_Elo point + 95% CI into a target-pool band.

    The measurement CI is mapped through the (piecewise-linear, so locally
    non-uniform) conversion first, then combined with the pool-conversion
    uncertainty in quadrature.  Returns None when the point cannot be converted.
    """
    c = convert_rating(point, scale)
    if c["point"] is None:
        return None
    p = c["point"]
    m_lo = p - convert_rating(lo, scale)["point"] if lo is not None else None
    m_hi = convert_rating(hi, scale)["point"] - p if hi is not None else None
    # An open-ended measurement CI stays open-ended after conversion.
    conv = c["conv95"]
    out = dict(c)
    out["meas95_lo"] = m_lo
    out["meas95_hi"] = m_hi
    out["band_lo"] = None if m_lo is None else p - math.sqrt(m_lo ** 2 + conv ** 2)
    out["band_hi"] = None if m_hi is None else p + math.sqrt(m_hi ** 2 + conv ** 2)
    widest_meas = max([v for v in (m_lo, m_hi) if v is not None], default=0.0)
    out["conversion_dominates_by"] = (conv / widest_meas) if widest_meas > 0 else None
    return out


def round50(x):
    return None if x is None else int(round(x / 50.0) * 50)


# Lichess Rapid rating distribution, computed from the histogram Lichess itself
# publishes at https://lichess.org/stat/rating/distribution/rapid
# (477,455 active rapid players, read 2026-09-11).  This is real, primary data.
LICHESS_RAPID_PCTL = [
    (800, 1.9), (900, 3.9), (1000, 7.2), (1100, 11.9), (1200, 18.1),
    (1300, 25.7), (1400, 33.9), (1500, 42.7), (1600, 51.7), (1700, 60.8),
    (1800, 69.9), (1900, 78.0), (2000, 85.1), (2100, 90.7), (2200, 94.8),
    (2300, 97.3), (2400, 98.8),
]

# chess.com Rapid percentiles.  chess.com does not publish a distribution, so
# this is a COMMUNITY estimate (chess.com community blog, Aug 2026) and it
# drifts fast -- the same pool had 1000 at the 49th percentile five years ago
# and the low 80s today, because the beginner influx keeps moving the median.
CHESSCOM_RAPID_PCTL = [
    (400, 25.0), (600, 47.0), (800, 65.0), (1000, 80.0), (1200, 90.0),
    (1400, 95.0), (1600, 97.5), (1800, 99.0),
]

PCTL_TABLES = {
    "lichess": (LICHESS_RAPID_PCTL, "Lichess's own published rapid distribution"),
    "chesscom": (CHESSCOM_RAPID_PCTL, "a COMMUNITY estimate, not chess.com data"),
}


def percentile_for(rating, scale):
    tbl = PCTL_TABLES.get(scale)
    if tbl is None or rating is None:
        return None
    rows, _ = tbl
    if rating <= rows[0][0]:
        return rows[0][1]
    if rating >= rows[-1][0]:
        return rows[-1][1]
    for a, b in zip(rows, rows[1:]):
        if a[0] <= rating <= b[0]:
            t = (rating - a[0]) / float(b[0] - a[0])
            return a[1] + t * (b[1] - a[1])
    return None


def collect_anchored_ratings(out):
    """Every absolute UCI_Elo figure this run produced, with its 95% CI.

    Only opponents that carry a real UCI_Elo anchor qualify.  A match against
    ``random`` or ``material`` measures a difference against an unrated
    opponent and yields no absolute rating at all -- that is not a gap to be
    papered over, it is the correct answer.
    """
    found = []
    lad = out.get("ladder") or {}
    for e in lad.get("rungs", []):
        a = e.get("absolute_elo")
        if not a:
            continue
        found.append({
            "via": e["name"], "anchor": a["anchor"], "kind": a["kind"],
            "point": a.get("point"), "lo": a.get("lo"), "hi": a.get("hi"),
            "score_pct": (e.get("summary") or {}).get("score_pct"),
        })

    # The single-match path: anchored only when the opponent was explicitly run
    # with UCI_LimitStrength + UCI_Elo, which is the documented way to measure.
    m = out.get("match")
    opts = {k.lower(): v for k, v in (out.get("config", {}).get("opp_options") or {}).items()}
    if m and opts.get("uci_limitstrength", "").strip().lower() in ("true", "1", "yes"):
        try:
            anchor = float(opts.get("uci_elo"))
        except (TypeError, ValueError):
            anchor = None
        if anchor is not None:
            el = m["elo"]
            if el.get("point") is not None:
                found.append({
                    "via": "%s (match)" % m["b"]["name"], "anchor": anchor,
                    "kind": "estimate", "point": anchor + el["point"],
                    "lo": (anchor + el["lo"]) if el.get("lo") is not None else None,
                    "hi": (anchor + el["hi"]) if el.get("hi") is not None else None,
                    "score_pct": m.get("score_pct"),
                })
            elif el.get("bound") == "upper" and el.get("hi") is not None:
                found.append({"via": "%s (match)" % m["b"]["name"], "anchor": anchor,
                              "kind": "upper bound", "point": None, "lo": None,
                              "hi": anchor + el["hi"], "score_pct": m.get("score_pct")})
            elif el.get("bound") == "lower" and el.get("lo") is not None:
                found.append({"via": "%s (match)" % m["b"]["name"], "anchor": anchor,
                              "kind": "lower bound", "point": None,
                              "lo": anchor + el["lo"], "hi": None,
                              "score_pct": m.get("score_pct")})
    return found


def render_scale_section(out, scale):
    """The converted report.  Never emitted for --scale uci."""
    label = SCALE_LABELS[scale]
    L = []
    L.append("-" * 78)
    L.append("RATING SCALE CONVERSION  ->  %s" % label)
    L.append("-" * 78)
    L.append("This is a conversion BETWEEN RATING POOLS, not a measurement.  The match")
    L.append("measures strength on Stockfish's UCI_Elo scale, which Stockfish's own source")
    L.append("anchors to CCRL Blitz -- an ENGINE-vs-engine list.  The target pool below")
    L.append("(%s) has a different population.  docs/RATING_SCALES.md gives the" % label)
    L.append("anchors, the sources and the size of the error; read it before quoting a")
    L.append("number from here.")
    L.append("")

    found = collect_anchored_ratings(out)
    if not found:
        L.append("NO CONVERTED NUMBER: this run produced no absolute rating to convert.")
        L.append("An absolute rating needs an opponent with a published rating.  The")
        L.append("built-in 'random' and 'material' opponents have none, and a plain match")
        L.append("against them measures only a difference against an unrated opponent.")
        L.append("")
        L.append("To get one, play a rating-anchored opponent:")
        L.append("  python3 py/benchmark.py --ladder --opponent 'uci:/path/to/stockfish' \\")
        L.append("      --scale %s" % scale)
        L.append("  python3 py/benchmark.py --opponent 'uci:/path/to/stockfish' \\")
        L.append("      --opp-option UCI_LimitStrength=true --opp-option UCI_Elo=1320 \\")
        L.append("      --scale %s" % scale)
        L.append("")
        return "\n".join(L)

    for f in found:
        L.append("via %s" % f["via"])
        L.append("  (opponent set to UCI_Elo %.0f; the champion scored %s against it)"
                 % (f["anchor"],
                    "%.1f%%" % f["score_pct"] if f.get("score_pct") is not None else "?"))

        if f["kind"] != "estimate":
            side = "BELOW" if f["kind"] == "upper bound" else "ABOVE"
            b = f["hi"] if f["kind"] == "upper bound" else f["lo"]
            L.append("  measured (Stockfish UCI_Elo) : %s %.0f (one-sided 95%% bound, the"
                     % (side, b))
            L.append("                                 match was a shutout)")
            c = convert_rating(b, scale)
            if c["point"] is None:
                L.append("  converted (%s): NOT CONVERTED" % label)
            else:
                tag = "" if c["in_range"] else "  [OUTSIDE THE ANCHORS -- see the warning below]"
                L.append("  converted (%s): %s roughly %d%s"
                         % (label, side, round50(c["point"]), tag))
                L.append("    A one-sided bound converts to a one-sided bound.  The +/-%d-point"
                         % round(c["conv95"]))
                L.append("    pool-conversion uncertainty applies to it as well, so treat it as")
                L.append("    soft in both directions.")
            _append_range_warning(L, f.get("hi") if f["kind"] == "upper bound" else f.get("lo"),
                                  scale, label)
            L.append("")
            continue

        meas = f["point"]
        L.append("  measured (Stockfish UCI_Elo) : %.0f  [95%% CI %s - %s]"
                 % (meas,
                    "%.0f" % f["lo"] if f["lo"] is not None else "-inf",
                    "%.0f" % f["hi"] if f["hi"] is not None else "+inf"))

        conv = convert_interval(meas, f["lo"], f["hi"], scale)
        if conv is None or not conv["in_range"]:
            L.append("  converted (%s): NOT CONVERTED" % label)
            _append_range_warning(L, meas, scale, label)
            L.append("")
            continue

        lo50, hi50 = round50(conv["band_lo"]), round50(conv["band_hi"])
        if lo50 is None or hi50 is None:
            L.append("  converted (%s): NOT CONVERTED -- the measurement's own" % label)
            L.append("    confidence interval is open-ended, so the converted band would be")
            L.append("    open-ended too.  Play more games, or pick an anchor nearer 50%%.")
            L.append("")
            continue

        L.append("  converted (%s): %d - %d" % (label, max(lo50, 0), hi50))
        L.append("    (midpoint %d, rounded to the nearest 50 -- the band does not justify"
                 % round50(conv["point"]))
        L.append("     any more precision than that, and the midpoint is not the answer)")
        mw = max(v for v in (conv["meas95_lo"], conv["meas95_hi"]) if v is not None)
        L.append("    band = match error (+/-%d) and pool-conversion error (+/-%d), added in"
                 % (round(mw), round(conv["conv95"])))
        if conv["conversion_dominates_by"]:
            L.append("    quadrature.  The conversion term is %.1fx the match term: playing more"
                     % conv["conversion_dominates_by"])
            L.append("    games will NOT meaningfully narrow this band.")
        else:
            L.append("    quadrature.")
        if conv["provenance"] == "extrapolated":
            L.append("    NOTE: this sits below the lowest FIDE data point the sources cover")
            L.append("    (~1550 FIDE).  That segment of the anchor table is extrapolated, and")
            L.append("    its extra uncertainty is already folded into the band above.")

        p_lo, p_hi = percentile_for(max(lo50, 0), scale), percentile_for(hi50, scale)
        if p_lo is not None and p_hi is not None:
            _, note = PCTL_TABLES[scale]
            L.append("    percentile: that band is stronger than roughly %.0f%% - %.0f%% of rated"
                     % (p_lo, p_hi))
            L.append("    %s players (source: %s)." % (label, note))
        elif scale in PCTL_TABLES:
            L.append("    percentile: no distribution is published for this pool.")
        else:
            L.append("    percentile: FIDE publishes no rating distribution, so none is given.")
        if scale == "fide" and lo50 < 1400:
            L.append("    NOTE: FIDE has not published a rating below 1400 since March 2024, and")
            L.append("    every player then rated 1000-2000 was given a one-time increase.  Part")
            L.append("    of this band is off the bottom of the FIDE list entirely.")
        L.append("")

    return "\n".join(L)


def _append_range_warning(L, value, scale, label):
    """Say plainly that we refused to extrapolate, and give the nearest bound."""
    c = convert_rating(value, scale)
    if c["in_range"] or c["point"] is None:
        return
    lo_a, hi_a = c["anchor_range"]
    side = c["bound_side"]
    L.append("    REFUSING TO EXTRAPOLATE.  %.0f is %s the anchor table, which covers"
             % (value, "below" if side == "below" else "above"))
    L.append("    UCI_Elo %d - %d." % (lo_a, hi_a))
    if side == "below":
        L.append("    %d is also Stockfish's own floor for UCI_Elo, so nothing below it has" % lo_a)
        L.append("    even been calibrated by Stockfish, let alone mapped to a human pool.")
    near = convert_rating(lo_a if side == "below" else hi_a, scale)
    n_lo = round50(near["point"] - near["conv95"])
    n_hi = round50(near["point"] + near["conv95"])
    L.append("    Nearest bound: UCI_Elo %d converts to %s %d - %d."
             % (lo_a if side == "below" else hi_a, label, max(n_lo, 0), n_hi))
    L.append("    The measurement fell %s that anchor, so the only honest reading is"
             % ("below" if side == "below" else "above"))
    L.append("    \"somewhere %s that range\"." % ("below" if side == "below" else "above"))


# ==========================================================================
# rendering
# ==========================================================================


def _nz(v):
    """Kill negative zero so an exactly even match prints '+0', not '-0'."""
    return 0.0 if v == 0 else v


def fmt_elo_cell(elo):
    if elo.get("bound") == "upper":
        v = elo.get("hi")
        return "< %+.0f" % _nz(v) if v is not None else "-inf"
    if elo.get("bound") == "lower":
        v = elo.get("lo")
        return "> %+.0f" % _nz(v) if v is not None else "+inf"
    if elo.get("point") is None:
        return "n/a"
    lo = "-inf" if elo.get("lo") is None else "%+.0f" % _nz(elo["lo"])
    hi = "+inf" if elo.get("hi") is None else "%+.0f" % _nz(elo["hi"])
    return "%+.0f [%s, %s]" % (_nz(elo["point"]), lo, hi)


def render_table(headers, rows, aligns=None):
    cols = len(headers)
    aligns = aligns or ["l"] * cols
    w = [len(str(h)) for h in headers]
    for r in rows:
        for i in range(cols):
            w[i] = max(w[i], len(str(r[i])))

    def line(cells):
        out = []
        for i, c in enumerate(cells):
            c = str(c)
            out.append(c.ljust(w[i]) if aligns[i] == "l" else c.rjust(w[i]))
        return "  ".join(out).rstrip()

    sep = "  ".join("-" * x for x in w)
    return "\n".join([line(headers), sep] + [line(r) for r in rows])


def match_lines(s, indent=""):
    o = []
    a, b = s["a"]["name"], s["b"]["name"]
    o.append("%s%s  vs  %s" % (indent, a, b))
    o.append("%s  %s" % (indent, s["a"]["detail"]))
    o.append("%s  %s" % (indent, s["b"]["detail"]))
    o.append("%sgames %d   W %d  D %d  L %d   score %.1f%%  (95%% CI %.1f%%-%.1f%%)"
             % (indent, s["games"], s["wins"], s["draws"], s["losses"], s["score_pct"],
                100 * s["score_ci"][0], 100 * s["score_ci"][1]))
    o.append("%sElo diff (%s POV): %s" % (indent, a, fmt_elo_cell(s["elo"])))
    if s["elo"].get("note"):
        o.append("%s  note: %s" % (indent, s["elo"]["note"]))
    o.append("%sCI basis: %s (%d complete pairs), %s"
             % (indent, s["elo"]["ci_basis"], s["pairs"]["complete"], s["ci_method"]))
    o.append("%sLOS (P that %s is genuinely stronger): %.1f%%" % (indent, a, 100 * s["los"]))
    o.append("%spairs: swept %d, split %d, both drawn %d, swept against %d"
             % (indent, s["pairs"]["swept_by_a"], s["pairs"]["split"],
                s["pairs"]["both_drawn"], s["pairs"]["swept_by_opponent"]))
    o.append("%sas white %.1f%% (%d games), as black %.1f%% (%d games)"
             % (indent, 100 * s["score_as_white"], s["white_games"],
                100 * s["score_as_black"], s["black_games"]))
    o.append("%sdraw rate %.1f%%   avg length %.1f plies (median %d, max %d)   "
             "hit the %d-ply cap: %d"
             % (indent, 100 * s["draw_rate"], s["avg_plies"], s["median_plies"],
                s["max_plies_seen"], s["ply_cap"], s["ply_cap_games"]))
    o.append("%sterminations: %s" % (indent, ", ".join(
        "%s %d" % (k, v) for k, v in s["terminations"].items())))
    for who in ("a", "b"):
        st = s[who].get("stats") or {}
        if not st.get("moves"):
            continue
        bits = ["%d moves" % st["moves"], "%.1f ms/move" % st.get("ms_per_move", 0.0)]
        if "nodes_per_move" in st:
            bits.append("%.0f nodes/move" % st["nodes_per_move"])
        if st.get("truncated_searches"):
            bits.append("%d searches (%.0f%%) reported a depth below the target "
                        "(movetime cap, forced move, or mate found)"
                        % (st["truncated_searches"],
                           100.0 * st["truncated_searches"] / st["moves"]))
        o.append("%s%s: %s" % (indent, s[who]["name"], ", ".join(bits)))
    o.append("%s%.1fs elapsed" % (indent, s["elapsed_s"]))
    return o


def render_report(out):
    L = []
    L.append("=" * 78)
    L.append("ChessRL external benchmark")
    L.append("=" * 78)
    m = out["model"]
    L.append("model      : %s" % m["path"])
    L.append("generation : %s   agents: %s   champion: agent %s (internal Elo %s)"
             % (m.get("generation"), m.get("n_agents"), m.get("champion_index"),
                ("%.0f" % m["champion_elo"]) if m.get("champion_elo") is not None else "?"))
    L.append("seed       : %s   opening plies: %d   ply cap: %d   threads: %d"
             % (out["config"]["seed"], out["config"]["opening_plies"],
                out["config"]["ply_cap"], out["config"]["threads"]))
    L.append("")

    if out.get("match"):
        L.append("-" * 78)
        L.append("MATCH")
        L.append("-" * 78)
        L.extend(match_lines(out["match"]))
        L.append("")

    if out.get("ladder"):
        lad = out["ladder"]
        L.append("-" * 78)
        L.append("LADDER  (%d games per rung, stop below %.0f%%)"
                 % (lad["games_per_rung"], 100 * lad["stop_threshold"]))
        L.append("-" * 78)
        if lad.get("external_engine_error"):
            L.append("external engine unavailable: %s" % lad["external_engine_error"])
            L.append("(the built-in rungs below still ran)")
        elif not lad.get("external_engine"):
            L.append("no external UCI engine given (--opponent uci:<cmd>), so the ladder")
            L.append("stops at the built-in floors.  The number below is a LOWER BOUND on")
            L.append("where the champion sits; it is not calibrated to human ratings.")
        rows = []
        for e in lad["rungs"]:
            if not e.get("played"):
                why = e.get("error") or e.get("skipped_because") or "not played"
                rows.append([e["name"], "-", "-", "-", "-", "-", "skipped: %s" % why])
                continue
            s = e["summary"]
            rows.append([
                e["name"] + (" *" if e.get("played_after_stop") else ""),
                "%d" % s["games"],
                "%d-%d-%d" % (s["wins"], s["draws"], s["losses"]),
                "%.1f%%" % s["score_pct"],
                fmt_elo_cell(s["elo"]),
                "%.0f%%" % (100 * s["los"]),
                "%.0f%% draws, %.0f plies avg" % (100 * s["draw_rate"], s["avg_plies"]),
            ])
        L.append(render_table(
            ["rung", "n", "W-D-L", "score", "Elo diff [95% CI]", "LOS", "notes"],
            rows, ["l", "r", "r", "r", "r", "r", "l"]))
        if any(e.get("played_after_stop") for e in lad["rungs"]):
            L.append("  * played after the ladder had already stopped, because this rung "
                     "carries a real")
            L.append("    rating anchor and even a shutout against it gives an absolute bound.")
        L.append("")
        cr = lad["crossing"]
        L.append("50%% crossing: %s" % cr.get("detail", cr.get("status")))
        if cr.get("configured_order_held") is False:
            L.append("  (the configured rung order did NOT match the measured one; the")
            L.append("   crossing above uses the measured order: %s)"
                     % " < ".join(reversed(cr["measured_order"])))
        for e in lad["rungs"]:
            if e.get("absolute_elo"):
                a = e["absolute_elo"]
                if a["kind"] == "estimate":
                    lo = "-inf" if a["lo"] is None else "%.0f" % a["lo"]
                    hi = "+inf" if a["hi"] is None else "%.0f" % a["hi"]
                    L.append("ANCHORED ESTIMATE via %s (anchor %.0f Elo): champion ~ %.0f Elo [%s, %s]"
                             % (e["name"], a["anchor"], a["point"], lo, hi))
                elif a["kind"] == "upper bound":
                    L.append("ANCHORED BOUND via %s (anchor %.0f Elo): the champion is BELOW "
                             "%.0f Elo on that scale, with 95%% confidence"
                             % (e["name"], a["anchor"], a["hi"]))
                else:
                    L.append("ANCHORED BOUND via %s (anchor %.0f Elo): the champion is ABOVE "
                             "%.0f Elo on that scale, with 95%% confidence"
                             % (e["name"], a["anchor"], a["lo"]))
        L.append("")

    if out.get("self_ladder"):
        sl = out["self_ladder"]
        L.append("-" * 78)
        L.append("SELF-LADDER  (%s, %d games per rung)"
                 % (sl["baseline"], sl["games_per_rung"]))
        L.append("-" * 78)
        rows = []
        for e in sl["rungs"]:
            s = e["summary"]
            gain = "n/a"
            if e["search_gain_elo"] is not None:
                lo = "-inf" if e["search_gain_lo"] is None else "%+.0f" % _nz(e["search_gain_lo"])
                hi = "+inf" if e["search_gain_hi"] is None else "%+.0f" % _nz(e["search_gain_hi"])
                gain = "%+.0f [%s, %s]" % (_nz(e["search_gain_elo"]), lo, hi)
            elif s["elo"].get("bound") == "upper":
                gain = "> %+.0f" % (-s["elo"]["hi"]) if s["elo"]["hi"] is not None else "+inf"
            elif s["elo"].get("bound") == "lower":
                gain = "< %+.0f" % (-s["elo"]["lo"]) if s["elo"]["lo"] is not None else "-inf"
            rows.append([
                "depth %d" % e["depth"],
                "%d" % s["games"],
                "%d-%d-%d" % (s["wins"], s["draws"], s["losses"]),
                "%.1f%%" % s["score_pct"],
                gain,
                "%.0f%%" % (100 * s["draw_rate"]),
                "%.0f" % s["avg_plies"],
            ])
        L.append(render_table(
            ["search", "n", "policy W-D-L", "policy score", "Elo the search adds [95% CI]",
             "draws", "plies"],
            rows, ["l", "r", "r", "r", "r", "r", "r"]))
        L.append("")
        L.append("Read this as: how much of the champion's playing strength comes from the")
        L.append("network's policy head alone, and how much is added by the hand-written")
        L.append("alpha-beta search sitting on top of it.")
        L.append("")

    # --scale uci (the default) is the identity and adds nothing: the report
    # above is already on the UCI_Elo scale, and must stay byte-identical.
    scale = (out.get("config") or {}).get("scale", "uci")
    if scale != "uci":
        L.append(render_scale_section(out, scale))

    return "\n".join(L)


# ==========================================================================
# markdown
# ==========================================================================

MD_HEADER = """# BENCHMARK -- how strong is the champion, really?

`runs/pilot/best.crl` reports an internal Elo of about 1750.  That number is
**self-referential**: it is a rating computed inside one population, anchored
only to frozen snapshots of that population's own past.  It measures "stronger
than it used to be" and says nothing about absolute strength.

`py/benchmark.py` measures the champion against opponents from outside the
population and reports the result with an honest error bar.

## Running it

```sh
make -j8 lib
python3 py/benchmark.py --model runs/pilot/best.crl --games 200            # one match
python3 py/benchmark.py --ladder --ladder-games 40                         # the ladder
python3 py/benchmark.py --ladder --opponent 'uci:/opt/homebrew/bin/stockfish'
python3 py/benchmark.py --self-ladder --self-games 60                      # search vs policy
```

Every run appends a section to this file.  `--no-md` suppresses that,
`--json PATH` writes the complete result including every game's move list.

## Opponents

| name | what it is |
| --- | --- |
| `random` | a uniformly random legal move.  Needs nothing installed. |
| `material` | 1-ply greedy on material with uniform random tie-breaking; a mate scores 10000, any drawn result scores 0.  It grabs hanging pieces and takes mate in 1, and hangs everything it owns. |
| `uci:<cmd>` | any external UCI engine, driven over stdin/stdout pipes.  `--opp-option "Skill Level=0"` passes options through, `--opp-go "nodes=100"` controls its search. |

## Method

**Colour balance and pairing.**  Games are played in pairs.  Both games of a
pair start from the identical position; the champion has White in the first and
Black in the second.  Both engines therefore get both sides of every opening,
which removes the first-move advantage from the comparison.

**Opening diversity without a book.**  Each pair's start position is produced by
playing `--opening-plies` (default 4) uniformly random legal plies from the
initial position, drawn from a seeded RNG.  Lines are de-duplicated.  Without
this every game between two near-deterministic engines would be the same game.

**Reproducibility.**  `--seed` fixes the openings, and every game additionally
gets its own RNG seeded from `(seed, game index)`, so the random and material
opponents make identical choices whatever `--threads` is set to.  Two runs with
the same seed and the same flags produce the same games.

**Score and Elo.**  A win is 1, a draw 0.5, a loss 0.  Elo difference is the
standard logistic inverse

```
elo = -400 * log10(1/score - 1)
```

so 50% is exactly 0 Elo and 75% is +191.

**Confidence interval.**  The interval is computed on the *pair* scores, not the
game scores.  The two games of a pair share an opening and are played by
engines that are close to deterministic, so they are strongly correlated;
treating them as two independent observations would understate the error.  The
standard error is the empirical standard error of the mean of the pair scores,
which accounts for draws automatically (an all-draws match has less spread than
a match split evenly between wins and losses).  The 95% interval on the score is
transformed through the Elo formula to give the Elo interval.

Degenerate cases are reported, not papered over:

* If every game has the identical result the empirical variance is zero.  The
  harness falls back on the rule of three -- with `n` games and no game showing
  a different outcome, the 95% one-sided bound on the rate of that unseen
  outcome is `1 - 0.05**(1/n)` -- and reports which method it used.
* If the score is exactly 0% or 100% the Elo difference is infinite.  The
  harness says so and prints a **one-sided bound** instead of a number: with 0
  points in `n` games the true score is below `1 - 0.05**(1/n)` with 95%
  confidence, which transforms to an Elo ceiling.

**Likelihood of superiority** is the usual match formula
`0.5*(1 + erf((W-L)/sqrt(2(W+L))))`; draws carry no information about which side
is better and do not enter it.

**Adjudication.**  A game is a draw once it reaches `--ply-cap` plies (default
300); those games are counted separately in the report.  `--resign-cp N` (off by
default) additionally adjudicates a game when the *opponent* reports an
evaluation of at least N centipawns, in a consistent direction, for
`--resign-plies` consecutive plies of its own.

It is off by default for a measured reason.  Adjudicating on one side's own
evaluation hands that side the benefit of the doubt: in a 6-game probe against
Stockfish at `go nodes 100`, `--resign-cp 400 --resign-plies 6` turned a
1-2-3 (33.3%) result into 0-0-6 (0.0%), because Stockfish repeatedly reached
+400 in positions our engine went on to hold or win.  Use it to save wall time
on long shutouts, not to produce a headline number.

**The ladder.**  A single match against one strong opponent tells you nothing
when the score is 0%.  `--ladder` plays a short match against a graded series --
`random`, `material`, then, if an external engine is given, `go nodes 1`, `10`,
`100`, `1000`, then its lowest `Skill Level`, then `UCI_LimitStrength` at its
`UCI_Elo` floor.  Rungs whose options the engine does not advertise are skipped.
The ladder stops early once the champion drops below `--ladder-stop` (default
5%), because further rungs would only produce more 0%s.

Two details matter:

* A rung that carries a real **rating anchor** (the `UCI_Elo` rung does) is
  played *even if the ladder has already stopped* -- marked `*` in the table.
  A shutout against a calibrated opponent still converts into an absolute
  bound, and an absolute number is the entire point.  `--no-anchor-rung` turns
  that off.
* The 50% crossing is computed over the rungs sorted by **measured** score, not
  by the order they were configured in.  The configured order is only a guess at
  the strength ordering -- Stockfish at `nodes 1000` and Stockfish at
  `UCI_Elo 1320` are not obviously ranked against each other -- and the match
  results are the evidence.  When the guess and the measurement disagree, the
  report says so.

**Anchoring, and what the anchored number is worth.**  When a rung has an
`anchor_elo`, the measured Elo difference is added to it: `UCI_Elo 1320` plus a
measured `-58` gives `~1262`.  That number inherits every assumption in the
anchor:

1. Stockfish's own `UCI_Elo` calibration is approximate, and is calibrated at
   normal time controls, not at the `--opp-go movetime` used here.
2. The logistic Elo model is only trustworthy over modest differences.  An
   anchor the champion scores 3% against implies a ~600-point extrapolation from
   two wins, and should be given very little weight.  Prefer anchors where the
   score is near 50%.
3. It is a *relative* rating against one engine's scale, not a FIDE rating, and
   not a rating against humans.  Stockfish's own source anchors `UCI_Elo` to
   CCRL Blitz, an engine-versus-engine list.

The right way to use it is to run several anchors and check that the estimates
agree.  If they do not, the disagreement is the honest error bar.

**Putting it on a human scale.**  `--scale {uci,chesscom,fide,lichess}` adds a
section converting the anchored number onto another rating pool.  `uci` is the
default and changes nothing.  Converting between rating pools is an
approximation with an error bar roughly five times larger than the match's
own, so the converted figure is always printed as a rounded range next to the
measured `UCI_Elo` figure, never on its own.  Read `docs/RATING_SCALES.md`
before quoting any converted number: it gives the anchor table, every source
behind it, and an honest account of what could not be sourced.

**The self-ladder.**  `--self-ladder` plays the champion's raw policy head
(argmax over the policy, no search at all) against the same champion using
alpha-beta at depths 1, 2, 4 and 6.  Each side runs its own engine handle, and a
fresh one per game, so no transposition table is shared or carried over.  The
result separates what the *network* learned from what the hand-written material
evaluation inside the search adds.

**Effort accounting.**  Every match reports moves, ms/move and nodes/move for
both sides, plus how many searches returned a depth below the target.  If that
last number is large the movetime cap is binding and the "depth N" label is a
lie -- raise `--movetime`.  (At depths 1-6 on this model the default 1000ms cap
never binds; the worst observed depth-6 search was 570ms.)

## Runs
"""


def append_markdown(path, out, report_text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(MD_HEADER, encoding="utf-8")
    elif "## Runs" not in path.read_text(encoding="utf-8"):
        with path.open("a", encoding="utf-8") as fh:
            fh.write("\n## Runs\n")

    L = []
    L.append("")
    L.append("### %s" % out["run"]["started"])
    L.append("")
    L.append("```")
    L.append(out["run"]["command"])
    L.append("```")
    L.append("")
    L.append("* model `%s`, generation %s, champion agent %s (internal Elo %s)"
             % (out["model"]["path"], out["model"].get("generation"),
                out["model"].get("champion_index"),
                ("%.0f" % out["model"]["champion_elo"])
                if out["model"].get("champion_elo") is not None else "?"))
    L.append("* seed %s, %d random opening plies, %d-ply cap, %d thread(s), commit `%s`"
             % (out["config"]["seed"], out["config"]["opening_plies"],
                out["config"]["ply_cap"], out["config"]["threads"],
                out["run"].get("commit", "?")))
    L.append("")

    if out.get("match"):
        s = out["match"]
        L.append("**Match** -- %s vs %s" % (s["a"]["name"], s["b"]["name"]))
        L.append("")
        L.append("| games | W-D-L | score | Elo diff [95% CI] | LOS | draws | avg plies | ply-cap draws |")
        L.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        L.append("| %d | %d-%d-%d | %.1f%% | %s | %.0f%% | %.0f%% | %.0f | %d |"
                 % (s["games"], s["wins"], s["draws"], s["losses"], s["score_pct"],
                    fmt_elo_cell(s["elo"]).replace("|", "/"), 100 * s["los"],
                    100 * s["draw_rate"], s["avg_plies"], s["ply_cap_games"]))
        if s["elo"].get("note"):
            L.append("")
            L.append("> %s" % s["elo"]["note"])
        L.append("")

    if out.get("ladder"):
        lad = out["ladder"]
        L.append("**Ladder** -- %d games per rung" % lad["games_per_rung"])
        L.append("")
        L.append("| rung | n | W-D-L | score | Elo diff [95% CI] | LOS |")
        L.append("| --- | ---: | ---: | ---: | ---: | ---: |")
        for e in lad["rungs"]:
            if not e.get("played"):
                why = e.get("error") or e.get("skipped_because") or "not played"
                L.append("| %s | - | - | - | - | *%s* |" % (e["name"], why))
                continue
            s = e["summary"]
            L.append("| %s | %d | %d-%d-%d | %.1f%% | %s | %.0f%% |"
                     % (e["name"], s["games"], s["wins"], s["draws"], s["losses"],
                        s["score_pct"], fmt_elo_cell(s["elo"]).replace("|", "/"),
                        100 * s["los"]))
        L.append("")
        L.append("50%% crossing: %s" % lad["crossing"].get("detail", lad["crossing"].get("status")))
        for e in lad["rungs"]:
            if e.get("absolute_elo"):
                a = e["absolute_elo"]
                L.append("")
                if a["kind"] == "estimate":
                    L.append("Anchored estimate via %s (anchor %.0f): **%.0f Elo** [%s, %s]"
                             % (e["name"], a["anchor"], a["point"],
                                "-inf" if a["lo"] is None else "%.0f" % a["lo"],
                                "+inf" if a["hi"] is None else "%.0f" % a["hi"]))
                elif a["kind"] == "upper bound":
                    L.append("Anchored bound via %s (anchor %.0f): the champion is **below %.0f Elo**"
                             " on that scale (95%% one-sided)." % (e["name"], a["anchor"], a["hi"]))
                else:
                    L.append("Anchored bound via %s (anchor %.0f): the champion is **above %.0f Elo**"
                             " on that scale (95%% one-sided)." % (e["name"], a["anchor"], a["lo"]))
        L.append("")

    if out.get("self_ladder"):
        sl = out["self_ladder"]
        L.append("**Self-ladder** -- %s, %d games per rung"
                 % (sl["baseline"], sl["games_per_rung"]))
        L.append("")
        L.append("| search | n | policy W-D-L | policy score | Elo the search adds | draws |")
        L.append("| --- | ---: | ---: | ---: | ---: | ---: |")
        for e in sl["rungs"]:
            s = e["summary"]
            if e["search_gain_elo"] is None:
                gain = "*infinite (%s)*" % ("policy scored 0%" if s["score"] <= 0
                                            else "policy scored 100%")
            else:
                lo = "-inf" if e["search_gain_lo"] is None else "%+.0f" % _nz(e["search_gain_lo"])
                hi = "+inf" if e["search_gain_hi"] is None else "%+.0f" % _nz(e["search_gain_hi"])
                gain = "%+.0f [%s, %s]" % (_nz(e["search_gain_elo"]), lo, hi)
            L.append("| depth %d | %d | %d-%d-%d | %.1f%% | %s | %.0f%% |"
                     % (e["depth"], s["games"], s["wins"], s["draws"], s["losses"],
                        s["score_pct"], gain, 100 * s["draw_rate"]))
        L.append("")

    L.append("<details><summary>full terminal report</summary>")
    L.append("")
    L.append("```")
    L.append(report_text.rstrip())
    L.append("```")
    L.append("")
    L.append("</details>")
    L.append("")

    with path.open("a", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")


# ==========================================================================
# CLI
# ==========================================================================


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="benchmark.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Externally calibrate the ChessRL champion.",
        epilog="""examples:
  python3 py/benchmark.py --model runs/pilot/best.crl --games 200
  python3 py/benchmark.py --ladder --ladder-games 40
  python3 py/benchmark.py --ladder --opponent 'uci:./build/chessrl uci' --opp-go nodes=1000
  python3 py/benchmark.py --opponent 'uci:/opt/homebrew/bin/stockfish' \\
      --opp-option 'Skill Level=0' --opp-go movetime=10 --games 100
  python3 py/benchmark.py --self-ladder --self-games 60 --json out.json
""")
    p.add_argument("--model", default=str(DEFAULT_MODEL), help="model file (default runs/pilot/best.crl)")
    p.add_argument("--agent", type=int, default=-1,
                   help="agent index inside the model; <0 selects the highest-Elo agent")
    p.add_argument("--games", type=int, default=200, help="games in the single match (default 200)")
    p.add_argument("--opponent", default="random",
                   help="random | material | uci:<command line>  (default random)")

    p.add_argument("--depth", type=int, default=4, help="champion search depth (default 4)")
    p.add_argument("--movetime", type=int, default=1000,
                   help="champion movetime cap in ms (default 1000)")
    p.add_argument("--policy-only", action="store_true",
                   help="champion plays the raw policy head with no search")
    p.add_argument("--policy-temp", type=float, default=0.0,
                   help="sample the policy at this temperature instead of argmax (default 0 = argmax)")

    p.add_argument("--opening-plies", type=int, default=4,
                   help="random legal plies played to start each pair (default 4)")
    p.add_argument("--seed", type=int, default=20260909, help="RNG seed (default 20260909)")
    p.add_argument("--ply-cap", type=int, default=300,
                   help="adjudicate a draw at this many plies (default 300)")
    p.add_argument("--threads", type=int, default=1, help="games in parallel (default 1)")

    p.add_argument("--opp-option", action="append", default=[], metavar="NAME=VALUE",
                   help="UCI setoption passed to the opponent; repeatable")
    p.add_argument("--opp-go", default="movetime=100", metavar="SPEC",
                   help="opponent go arguments, e.g. nodes=100 or movetime=10 or depth=1")
    p.add_argument("--opp-timeout", type=float, default=30.0,
                   help="seconds to wait for any UCI reply before failing (default 30)")
    p.add_argument("--debug-uci", action="store_true", help="echo the whole UCI conversation")

    p.add_argument("--resign-cp", type=int, default=0,
                   help="adjudicate when the opponent reports >= this many cp (0 = off)")
    p.add_argument("--resign-plies", type=int, default=8,
                   help="consecutive opponent plies required by --resign-cp (default 8)")

    p.add_argument("--ladder", action="store_true", help="play the graded opponent ladder")
    p.add_argument("--ladder-games", type=int, default=40, help="games per ladder rung (default 40)")
    p.add_argument("--ladder-stop", type=float, default=0.05,
                   help="stop climbing once the score drops below this (default 0.05)")
    p.add_argument("--no-anchor-rung", action="store_true",
                   help="do not play the rating-anchored rung after the ladder stops early "
                        "(by default it is played anyway, because even a shutout there "
                        "converts into an absolute bound)")

    p.add_argument("--self-ladder", action="store_true",
                   help="champion policy-only vs champion at several search depths")
    p.add_argument("--self-games", type=int, default=40, help="games per self rung (default 40)")
    p.add_argument("--self-depths", default="1,2,4,6", help="search depths to test (default 1,2,4,6)")

    p.add_argument("--scale", default="uci", choices=list(SCALES),
                   help="rating pool to ALSO report the anchored rating on (default uci, "
                        "which is the identity and changes nothing).  A non-uci scale adds "
                        "a section showing the measured UCI_Elo figure AND a converted band "
                        "that is deliberately much wider, because converting between rating "
                        "pools is an approximation.  See docs/RATING_SCALES.md")
    p.add_argument("--json", default=None, metavar="PATH", help="write the full result as JSON")
    p.add_argument("--md", default=str(DEFAULT_MD), metavar="PATH",
                   help="append a summary here (default docs/BENCHMARK.md)")
    p.add_argument("--no-md", action="store_true", help="do not touch the markdown file")
    p.add_argument("--quiet", action="store_true", help="no progress line")

    args = p.parse_args(argv)

    opts = {}
    for item in args.opp_option:
        if "=" not in item:
            p.error("--opp-option needs NAME=VALUE, got %r" % item)
        k, v = item.split("=", 1)
        opts[k.strip()] = v.strip()
    args.opp_option_map = opts

    try:
        args.self_depths = [int(x) for x in str(args.self_depths).replace(",", " ").split()]
    except ValueError:
        p.error("--self-depths must be a comma-separated list of integers")
    if not args.self_depths:
        p.error("--self-depths is empty")

    if args.opening_plies < 0:
        p.error("--opening-plies must be >= 0")
    if args.ply_cap < 2:
        p.error("--ply-cap must be >= 2")
    return args


def git_commit():
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=str(REPO_ROOT),
                           capture_output=True, text=True, timeout=5)
        if r.returncode == 0:
            return r.stdout.strip()
    except Exception:
        pass
    return "?"


def model_summary(path, agent_index):
    info = chessrl.model_info(path)
    agents = info.get("agents") or []
    champ = None
    if agent_index is not None and agent_index >= 0:
        for a in agents:
            if a.get("i") == agent_index:
                champ = a
                break
    if champ is None and agents:
        champ = max(agents, key=lambda a: a.get("elo", float("-inf")))
    return {
        "path": str(path),
        "generation": info.get("generation"),
        "n_agents": info.get("n_agents"),
        "champion_index": champ.get("i") if champ else None,
        "champion_elo": champ.get("elo") if champ else None,
        "champion_elo_is": "internal, self-referential -- that is what this tool exists to replace",
    }


def strip_games_named(obj):
    """Drop the (large) per-game records once they have been written out."""
    if isinstance(obj, dict):
        obj.pop("game_records", None)
        for v in obj.values():
            strip_games_named(v)
    elif isinstance(obj, list):
        for v in obj:
            strip_games_named(v)
    return obj


def strip_games(obj, keep):
    """Move the per-game records out of the summaries unless we are keeping them."""
    if isinstance(obj, dict):
        if "_games" in obj:
            g = obj.pop("_games")
            if keep:
                obj["game_records"] = g
        for v in obj.values():
            strip_games(v, keep)
    elif isinstance(obj, list):
        for v in obj:
            strip_games(v, keep)
    return obj


def main(argv=None):
    _install_signal_handlers()
    args = parse_args(argv)

    if not chessrl.library_available():
        sys.stderr.write("libchessrl is not built.  Run:  make -j8 lib\n")
        return 2
    model_path = Path(args.model).expanduser()
    if not model_path.is_file():
        sys.stderr.write("model file not found: %s\n" % model_path)
        return 2
    args.model = str(model_path)

    out = {
        "run": {
            "started": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ"),
            "command": "python3 " + " ".join(
                [os.path.relpath(sys.argv[0], str(REPO_ROOT))] +
                [shlex.quote(a) for a in (argv if argv is not None else sys.argv[1:])]),
            "commit": git_commit(),
            "python": sys.version.split()[0],
        },
        "config": {
            "seed": args.seed,
            "opening_plies": args.opening_plies,
            "ply_cap": args.ply_cap,
            "threads": args.threads,
            "depth": args.depth,
            "movetime": args.movetime,
            "policy_only": bool(args.policy_only),
            "policy_temp": args.policy_temp,
            "opponent": args.opponent,
            "opp_go": args.opp_go,
            "opp_options": args.opp_option_map,
            "resign_cp": args.resign_cp,
            "resign_plies": args.resign_plies,
            "scale": args.scale,
        },
        "model": model_summary(args.model, args.agent),
    }

    rc = 0
    try:
        if args.ladder:
            run_ladder(args, out, quiet=args.quiet)
        if args.self_ladder:
            run_self_ladder(args, out, quiet=args.quiet)
        if not args.ladder and not args.self_ladder:
            champ = champion_spec(args)
            opp = opponent_spec(args)
            out["match"] = run_match(champ, opp, args.games, args, quiet=args.quiet)
    except KeyboardInterrupt:
        sys.stderr.write("\ninterrupted -- killing any child engines\n")
        reap_all_children()
        return 130
    except BenchError as exc:
        sys.stderr.write("\nbenchmark failed: %s\n" % exc)
        reap_all_children()
        return 1
    finally:
        reap_all_children()

    report = render_report(out)
    print(report)

    strip_games(out, bool(args.json))
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
        sys.stderr.write("wrote %s\n" % args.json)
        strip_games_named(out)
    if not args.no_md:
        append_markdown(args.md, out, report)
        sys.stderr.write("appended a summary to %s\n" % args.md)

    return rc


if __name__ == "__main__":
    sys.exit(main())
