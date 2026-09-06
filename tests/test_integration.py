#!/usr/bin/env python3
"""End-to-end integration test for the whole ChessRL stack.

    python3 tests/test_integration.py [-v]

Python standard library only (unittest + urllib + subprocess + threading), the
same constraint the rest of the project lives under.

What it exercises, bottom to top:

* the C shared library through ``py/engine.py``'s ctypes bindings, including
  the negative-return "your buffer was too small" retry protocol,
* every HTTP route in ``docs/API.md`` served by ``py/server.py``, launched as a
  real subprocess on an ephemeral port, spoken to over real sockets,
* the invariant the browser UI depends on above all others: **the server never
  accepts a move that was not in the ``legal`` list it just handed out**,
* a complete engine-vs-engine game driven one ``/api/move``-shaped request at a
  time, eight concurrent games from eight threads, and a pile of malformed
  requests that must all come back as clean 4xx JSON.

The first run builds ``build/libchessrl.dylib`` and, if there is no model at
all, trains a throwaway one into ``runs/smoke/``.  Both steps are skipped when
their outputs already exist, so a warm run is a few seconds.
"""

from __future__ import annotations

import json
import os
import random
import re
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.parse
import urllib.request

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY_DIR = os.path.join(REPO_ROOT, "py")
BUILD_DIR = os.path.join(REPO_ROOT, "build")
RUNS_DIR = os.path.join(REPO_ROOT, "runs")
SMOKE_MODEL = os.path.join(RUNS_DIR, "smoke", "best.crl")
SERVER_PY = os.path.join(PY_DIR, "server.py")

sys.path.insert(0, PY_DIR)
import engine  # noqa: E402  (local module, after the sys.path fix)

# --------------------------------------------------------------------------
# the contract, transcribed from docs/API.md
# --------------------------------------------------------------------------

STATE_KEYS = {
    "gid", "fen", "turn", "ply", "result", "reason", "check",
    "legal", "san", "moves", "history_san", "last", "material", "captured",
}

#: docs/API.md, `<State>.reason`
DOCUMENTED_REASONS = {
    "checkmate",
    "stalemate",
    "fifty-move",
    "threefold repetition",
    "insufficient material",
    "move limit",
}

#: py/server.py may additionally end a *server-side* game this way.
WATCH_REASONS = DOCUMENTED_REASONS | {"adjudicated", "resignation"}

UCI_RE = re.compile(r"^[a-h][1-8][a-h][1-8][qrbn]?$")

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

#: Scholar's Mate: 1.e4 e5 2.Bc4 Nc6 3.Qh5 Nf6?? 4.Qxf7#
SCHOLARS_MATE = ["e2e4", "e7e5", "f1c4", "b8c6", "d1h5", "g8f6", "h5f7"]

PIECE_VALUE = {"p": 1, "n": 3, "b": 3, "r": 5, "q": 9, "k": 0}
FULL_ARMY = {"p": 8, "n": 2, "b": 2, "r": 2, "q": 1, "k": 1}
CAPTURE_ORDER = "qrbnp"

# Engine settings used by the HTTP tests.  Shallow and short: this is a test of
# plumbing, not of playing strength.
TEST_DEPTH = 2
TEST_MOVETIME = 120
# A self-play game must terminate by itself (checkmate / repetition / fifty
# move / ...).  Observed: 80-410 plies with a smoke model.  1024 is the C-side
# structural cap in chess.h, so stop well short of it.
SELF_PLAY_PLY_CAP = 900


def board_counts(fen):
    """{'white': {piece: n}, 'black': {...}} from the board field of a FEN."""
    white, black = {}, {}
    for ch in (fen or "").split(" ", 1)[0]:
        if ch.isalpha():
            side = white if ch.isupper() else black
            side[ch.lower()] = side.get(ch.lower(), 0) + 1
    return {"white": white, "black": black}


def expected_material(fen):
    """Each side's own remaining material, recomputed from the FEN."""
    counts = board_counts(fen)
    return {
        side: sum(PIECE_VALUE[k] * n for k, n in have.items())
        for side, have in counts.items()
    }


def expected_captured(fen):
    """Pieces each side is missing relative to a full army, most valuable first."""
    counts = board_counts(fen)
    out = {}
    for side, have in counts.items():
        lost = []
        for piece in CAPTURE_ORDER:
            gone = FULL_ARMY[piece] - have.get(piece, 0)
            lost.extend([piece] * max(0, gone))
        out[side] = lost
    return out


# ==========================================================================
# one-time bootstrap: build the library, train a throwaway model
# ==========================================================================


def _run(cmd, what, timeout):
    sys.stderr.write("[integration] %s: %s\n" % (what, " ".join(cmd)))
    started = time.monotonic()
    proc = subprocess.run(
        cmd, cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        timeout=timeout,
    )
    took = time.monotonic() - started
    if proc.returncode != 0:
        raise RuntimeError(
            "%s failed (exit %d after %.1fs)\n%s"
            % (what, proc.returncode, took, proc.stdout.decode("utf-8", "replace")[-4000:])
        )
    sys.stderr.write("[integration] %s: ok (%.1fs)\n" % (what, took))


def ensure_library():
    if engine.library_available():
        return
    _run(["make", "-j8", "lib"], "build libchessrl", timeout=600)
    if not engine.library_available():
        raise RuntimeError("'make -j8 lib' succeeded but no library appeared in %s" % BUILD_DIR)


def ensure_model():
    """Path of the throwaway smoke model, training one if it is not there yet.

    The suite deliberately pins itself to ``runs/smoke`` rather than to whatever
    the newest run happens to be: a two-generation model is quick to make, and
    using a known one keeps the self-play game a couple of hundred plies instead
    of however long a strong model would grind.
    """
    if os.path.isfile(SMOKE_MODEL):
        return SMOKE_MODEL
    binary = os.path.join(BUILD_DIR, "chessrl")
    if not os.path.isfile(binary):
        _run(["make", "-j8"], "build chessrl", timeout=600)
    _run(
        [binary, "train", "--agents", "8", "--gens", "2", "--games-per-agent", "4",
         "--max-plies", "100", "--run", "smoke"],
        "train smoke model",
        timeout=600,
    )
    if not os.path.isfile(SMOKE_MODEL):
        raise RuntimeError("training finished but %s does not exist" % SMOKE_MODEL)
    return SMOKE_MODEL


MODEL_PATH = None


def setUpModule():
    global MODEL_PATH
    ensure_library()
    engine.load_library()
    MODEL_PATH = ensure_model()


# ==========================================================================
# the server under test
# ==========================================================================


class ServerProcess:
    """``py/server.py`` running for real on a free loopback port."""

    START_TIMEOUT = 40.0

    def __init__(self, model_path):
        self.model_path = model_path
        self.proc = None
        self.port = None
        self._tmpdir = None
        self._logfile = None
        self._log_fh = None

    # -- lifetime ---------------------------------------------------------
    @staticmethod
    def _free_port():
        sock = socket.socket()
        try:
            sock.bind(("127.0.0.1", 0))
            return sock.getsockname()[1]
        finally:
            sock.close()

    def start(self):
        self._tmpdir = tempfile.mkdtemp(prefix="chessrl-itest-")
        self._logfile = os.path.join(self._tmpdir, "server.log")
        last = None
        # A free port can be taken between the probe bind and the server's own
        # bind; that is rare, and retrying a handful of times makes it a non-issue.
        for _ in range(6):
            port = self._free_port()
            self._log_fh = open(self._logfile, "wb")
            cmd = [sys.executable, "-u", SERVER_PY, "--port", str(port),
                   "--host", "127.0.0.1"]
            if self.model_path:
                cmd += ["--model", self.model_path]
            self.proc = subprocess.Popen(
                cmd, cwd=REPO_ROOT, stdout=self._log_fh, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
            )
            try:
                self._await_port(port)
            except RuntimeError as exc:
                last = exc
                self.stop()
                continue
            self.port = port
            return self
        raise RuntimeError("could not start py/server.py: %s" % last)

    def _await_port(self, port):
        deadline = time.monotonic() + self.START_TIMEOUT
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(
                    "server exited with %s before listening:\n%s"
                    % (self.proc.returncode, self.log_text()[-2000:])
                )
            try:
                socket.create_connection(("127.0.0.1", port), 0.25).close()
                return
            except OSError:
                time.sleep(0.05)
        raise RuntimeError("server did not listen on %d within %.0fs" % (port, self.START_TIMEOUT))

    def stop(self):
        proc, self.proc = self.proc, None
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    pass
        if self._log_fh is not None:
            try:
                self._log_fh.close()
            except OSError:
                pass
            self._log_fh = None
        self.port = None

    def cleanup(self):
        """Drop the temporary log directory.  Safe to call twice."""
        self.stop()
        if self._tmpdir and os.path.isdir(self._tmpdir):
            for name in os.listdir(self._tmpdir):
                try:
                    os.unlink(os.path.join(self._tmpdir, name))
                except OSError:
                    pass
            try:
                os.rmdir(self._tmpdir)
            except OSError:
                pass
        self._tmpdir = None

    # -- observation ------------------------------------------------------
    def log_text(self):
        if not self._logfile or not os.path.isfile(self._logfile):
            return ""
        with open(self._logfile, "rb") as fh:
            return fh.read().decode("utf-8", "replace")

    @property
    def base(self):
        return "http://127.0.0.1:%d" % self.port


class Response:
    """A tiny HTTP response record: status, headers, raw body, parsed JSON."""

    __slots__ = ("status", "headers", "body", "json")

    def __init__(self, status, headers, body):
        self.status = status
        self.headers = headers
        self.body = body
        try:
            self.json = json.loads(body.decode("utf-8")) if body else None
        except (ValueError, UnicodeDecodeError):
            self.json = None

    def __repr__(self):
        return "<Response %d %r>" % (self.status, self.body[:160])


class ApiClient:
    """Blocking JSON client, one connection per call (thread-safe by construction)."""

    def __init__(self, base, timeout=120.0):
        self.base = base
        self.timeout = timeout

    def request(self, method, path, body=None, raw_body=None, headers=None):
        data = raw_body
        hdrs = dict(headers or {})
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            hdrs.setdefault("Content-Type", "application/json")
        req = urllib.request.Request(self.base + path, data=data, method=method, headers=hdrs)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as fh:
                return Response(fh.status, dict(fh.headers), fh.read())
        except urllib.error.HTTPError as exc:
            with exc:
                return Response(exc.code, dict(exc.headers or {}), exc.read())

    def get(self, path, **kw):
        return self.request("GET", path, **kw)

    def post(self, path, body=None, **kw):
        return self.request("POST", path, body=body, **kw)


class ServerTestCase(unittest.TestCase):
    """Base class: one ``py/server.py`` subprocess per test class.

    ``tearDownClass`` runs even when ``setUpClass`` or any test blows up, so no
    server process can outlive the run.
    """

    server = None
    api = None

    @classmethod
    def setUpClass(cls):
        cls.server = ServerProcess(MODEL_PATH)
        try:
            cls.server.start()
        except BaseException:
            cls.server.cleanup()
            cls.server = None
            raise
        cls.api = ApiClient(cls.server.base)

    @classmethod
    def tearDownClass(cls):
        server = cls.server
        cls.server = None
        cls.api = None
        if server is None:
            return
        log = server.log_text()
        server.cleanup()
        # An unhandled exception anywhere in a request handler prints a
        # traceback and answers 500.  Nothing this suite does should provoke one.
        if "Traceback (most recent call last)" in log:
            where = log.index("Traceback (most recent call last)")
            raise AssertionError(
                "py/server.py printed a traceback while serving %s:\n%s"
                % (cls.__name__, log[where:where + 3000])
            )

    # -- shared assertions ------------------------------------------------
    def ok(self, resp, path=""):
        """Assert 2xx and return the parsed JSON."""
        self.assertEqual(
            200, resp.status,
            "%s -> %d %r" % (path, resp.status, resp.body[:400]),
        )
        self.assertIsNotNone(resp.json, "%s returned non-JSON: %r" % (path, resp.body[:200]))
        return resp.json

    def assert_json_error(self, resp, lo=400, hi=499, path=""):
        """Assert a clean 4xx: right status class, JSON body, an ``error`` string."""
        self.assertTrue(
            lo <= resp.status <= hi,
            "%s expected %d..%d, got %d %r" % (path, lo, hi, resp.status, resp.body[:400]),
        )
        self.assertIsNotNone(
            resp.json, "%s error body was not JSON: %r" % (path, resp.body[:300])
        )
        self.assertIsInstance(resp.json, dict, "%s error body was not an object" % path)
        self.assertIsInstance(resp.json.get("error"), str, "%s has no error string" % path)
        self.assertNotIn(b"Traceback", resp.body, "%s leaked a traceback" % path)

    def assert_state(self, state, gid=None, where="<State>"):
        """Validate the FULL documented ``<State>`` shape."""
        self.assertIsInstance(state, dict, where)
        self.assertEqual(
            STATE_KEYS, set(state),
            "%s key set differs: missing %s, extra %s"
            % (where, sorted(STATE_KEYS - set(state)), sorted(set(state) - STATE_KEYS)),
        )

        # -- scalars
        self.assertIsInstance(state["gid"], int)
        self.assertNotIsInstance(state["gid"], bool)
        if gid is not None:
            self.assertEqual(gid, state["gid"], "%s gid" % where)

        fen = state["fen"]
        self.assertIsInstance(fen, str)
        fields = fen.split()
        self.assertEqual(6, len(fields), "%s fen has %d fields: %r" % (where, len(fields), fen))
        self.assertEqual(8, len(fields[0].split("/")), "%s fen board: %r" % (where, fen))
        self.assertIn(fields[1], ("w", "b"))

        self.assertIn(state["turn"], ("white", "black"), where)
        self.assertEqual({"w": "white", "b": "black"}[fields[1]], state["turn"],
                         "%s turn disagrees with the fen" % where)

        self.assertIsInstance(state["ply"], int)
        self.assertNotIsInstance(state["ply"], bool)
        self.assertGreaterEqual(state["ply"], 0)

        self.assertIn(state["result"], (0, 1, 2, 3), where)
        self.assertNotIsInstance(state["result"], bool)
        self.assertIsInstance(state["reason"], str)
        if state["result"] == 0:
            self.assertEqual("", state["reason"], "%s ongoing games have no reason" % where)
        else:
            self.assertIn(state["reason"], DOCUMENTED_REASONS, where)
        self.assertIsInstance(state["check"], bool)

        # -- legal / san
        legal, san = state["legal"], state["san"]
        self.assertIsInstance(legal, list)
        self.assertIsInstance(san, list)
        self.assertEqual(len(legal), len(san), "%s len(san) != len(legal)" % where)
        self.assertEqual(len(set(legal)), len(legal), "%s legal has duplicates" % where)
        for uci in legal:
            self.assertIsInstance(uci, str)
            self.assertRegex(uci, UCI_RE, "%s legal entry %r" % (where, uci))
        for text in san:
            self.assertIsInstance(text, str)
            self.assertTrue(text, "%s empty san entry" % where)
        if not legal:
            self.assertNotEqual(0, state["result"],
                                "%s has no legal moves but claims to be ongoing" % where)

        # -- history
        moves, hist = state["moves"], state["history_san"]
        self.assertIsInstance(moves, list)
        self.assertIsInstance(hist, list)
        self.assertEqual(len(moves), len(hist),
                         "%s len(history_san) != len(moves)" % where)
        self.assertEqual(len(moves), state["ply"], "%s ply != len(moves)" % where)
        for uci in moves:
            self.assertRegex(uci, UCI_RE, "%s history entry %r" % (where, uci))

        # -- last
        last = state["last"]
        if not moves:
            self.assertIsNone(last, "%s last must be null before any move" % where)
        else:
            self.assertIsInstance(last, dict, where)
            self.assertIn("from", last)
            self.assertIn("to", last)
            self.assertEqual(moves[-1][:2], last["from"], "%s last.from" % where)
            self.assertEqual(moves[-1][2:4], last["to"], "%s last.to" % where)

        # -- material / captured, recomputed independently from the fen
        material = state["material"]
        self.assertIsInstance(material, dict)
        self.assertEqual({"white", "black"}, set(material), "%s material keys" % where)
        for side in ("white", "black"):
            self.assertIsInstance(material[side], int)
            self.assertNotIsInstance(material[side], bool)
            self.assertTrue(0 <= material[side] <= 103,
                            "%s material[%s]=%r" % (where, side, material[side]))
        self.assertEqual(expected_material(fen), material,
                         "%s material disagrees with the fen %r" % (where, fen))

        captured = state["captured"]
        self.assertIsInstance(captured, dict)
        self.assertEqual({"white", "black"}, set(captured), "%s captured keys" % where)
        for side in ("white", "black"):
            self.assertIsInstance(captured[side], list)
            self.assertLessEqual(len(captured[side]), 15, "%s captured[%s]" % (where, side))
            for piece in captured[side]:
                self.assertIn(piece, CAPTURE_ORDER, "%s captured[%s]" % (where, side))
        self.assertEqual(expected_captured(fen), captured,
                         "%s captured disagrees with the fen %r" % (where, fen))
        return state

    # -- convenience ------------------------------------------------------
    def new_game(self, query=""):
        payload = self.ok(self.api.get("/api/new" + query), "/api/new")
        self.assertIsInstance(payload.get("gid"), int)
        state = self.assert_state(payload["state"], payload["gid"], "/api/new state")
        return payload["gid"], state


# ==========================================================================
# 3. the ctypes layer, straight through py/engine.py
# ==========================================================================


class TestCtypesLayer(unittest.TestCase):
    """No HTTP: py/engine.py talking to libchessrl over ctypes."""

    def test_new_game_start_position(self):
        with engine.GameHandle() as game:
            state = game.state()
            self.assertEqual(START_FEN, state["fen"])
            self.assertEqual("white", state["turn"])
            self.assertEqual(0, state["result"])
            self.assertEqual(20, len(state["legal"]))
            self.assertEqual(len(state["legal"]), len(state["san"]))
            self.assertEqual(sorted(state["legal"]), sorted(game.legal()),
                             "api_game_legal disagrees with api_game_state")

    def test_scholars_mate_ends_in_checkmate(self):
        with engine.GameHandle() as game:
            for uci in SCHOLARS_MATE:
                legal = game.legal()
                self.assertIn(uci, legal, "%s is not legal here" % uci)
                self.assertTrue(game.move(uci), "the library refused legal move %s" % uci)
            state = game.state()
            self.assertEqual(1, state["result"], "Scholar's Mate is a white win")
            self.assertEqual("checkmate", state["reason"])
            self.assertTrue(state["check"])
            self.assertEqual([], state["legal"], "a mated side has no legal moves")
            self.assertEqual(SCHOLARS_MATE, state["moves"])
            self.assertEqual(
                ["e4", "e5", "Bc4", "Nc6", "Qh5", "Nf6", "Qxf7#"], state["history_san"]
            )
            self.assertEqual({"white": 39, "black": 38}, state["material"])
            self.assertEqual({"white": [], "black": ["p"]}, state["captured"])

    def test_illegal_moves_are_rejected(self):
        with engine.GameHandle() as game:
            for uci in ("e2e5", "e7e5", "a1a8", "e1e3", "d1h5", "b1b3"):
                self.assertFalse(game.move(uci), "%r was accepted from the start position" % uci)
            self.assertEqual(0, game.state()["ply"], "a rejected move must not change the game")
            self.assertEqual(START_FEN, game.state()["fen"])

    def test_undo_restores_the_prior_state(self):
        with engine.GameHandle() as game:
            self.assertFalse(game.undo(), "there is nothing to undo at ply 0")
            before = game.state()
            for uci in SCHOLARS_MATE[:4]:
                self.assertTrue(game.move(uci))
            mid = game.state()
            for _ in range(4):
                self.assertTrue(game.undo())
            self.assertEqual(before, game.state(), "undo did not restore the start position")
            # ... and one single-ply round trip in the middle of a game.
            for uci in SCHOLARS_MATE[:4]:
                game.move(uci)
            self.assertEqual(mid, game.state())
            snapshot = game.state()
            self.assertTrue(game.move("d1h5"))
            self.assertNotEqual(snapshot, game.state())
            self.assertTrue(game.undo())
            self.assertEqual(snapshot, game.state(), "single-ply undo did not restore the state")

    def test_buffer_grow_and_retry(self):
        """Force the negative-return path in engine._call_json / _call_text."""

        class Counting:
            def __init__(self, fn):
                self.fn = fn
                self.calls = 0

            def __call__(self, *args):
                self.calls += 1
                return self.fn(*args)

        lib = engine.load_library()

        # A start-position <State> is ~700 bytes, comfortably past the 256-byte
        # floor _call_json clamps `initial` to, so the first call must come back
        # negative and the helper must retry with the size the library asked for.
        with engine.GameHandle() as game:
            reference = game.state()
            counted = Counting(lib.api_game_state)
            grown = engine._call_json(counted, (game.gid,), "api_game_state", initial=1)
            self.assertGreaterEqual(counted.calls, 2, "the buffer never had to grow")
            self.assertEqual(reference, grown, "the retry returned different bytes")

        # 218 legal moves is the most any chess position has: ~1.1 kB of UCI.
        many = "R6R/3Q4/1Q4Q1/4Q3/2Q4Q/Q4Q2/pp1Q4/kBNN1KB1 w - - 0 1"
        with engine.GameHandle(many) as game:
            counted = Counting(lib.api_game_legal)
            text = engine._call_text(counted, (game.gid,), "api_game_legal", initial=1)
            self.assertGreaterEqual(counted.calls, 2, "the buffer never had to grow")
            self.assertEqual(218, len(text.split()), "lost moves across the retry")
            self.assertEqual(sorted(game.legal()), sorted(text.split()))

        # A buffer that is already big enough must be filled in a single call.
        with engine.GameHandle() as game:
            counted = Counting(lib.api_game_state)
            engine._call_json(counted, (game.gid,), "api_game_state", initial=1 << 16)
            self.assertEqual(1, counted.calls, "a large enough buffer should not retry")

    def test_bad_fen_is_reported_not_crashed(self):
        with self.assertRaises(ValueError):
            engine.GameHandle("this is not a fen")

    def test_closed_handle_is_refused(self):
        game = engine.GameHandle()
        game.close()
        game.close()  # idempotent
        with self.assertRaises(engine.ChessRLError):
            game.state()

    def test_model_info(self):
        info = engine.model_info(MODEL_PATH)
        self.assertIsInstance(info, dict)
        self.assertIsInstance(info["n_agents"], int)
        self.assertGreater(info["n_agents"], 0)
        self.assertEqual(info["n_agents"], len(info["agents"]))
        for agent in info["agents"]:
            self.assertIsInstance(agent["i"], int)
            self.assertIsInstance(agent["elo"], (int, float))
            self.assertIsInstance(agent["rank"], int)

    def test_engine_only_proposes_legal_moves(self):
        with engine.Engine(MODEL_PATH, -1) as eng, engine.GameHandle() as game:
            for _ in range(6):
                state = game.state()
                if state["result"] or not state["legal"]:
                    break
                reply = eng.move(game, TEST_DEPTH, TEST_MOVETIME)
                self.assertIn(reply["move"], state["legal"],
                              "api_engine_move proposed an illegal move")
                policy = eng.policy(game)
                self.assertTrue(policy)
                for item in policy:
                    self.assertIn(item["move"], state["legal"],
                                  "api_engine_policy listed an illegal move")
                self.assertTrue(game.move(reply["move"]))


# ==========================================================================
# 4. every endpoint in docs/API.md
# ==========================================================================


class TestHttpEndpoints(ServerTestCase):

    def test_static_index_is_served(self):
        resp = self.api.get("/")
        self.assertEqual(200, resp.status)
        self.assertIn("text/html", resp.headers.get("Content-Type", ""))
        self.assertIn(b"<", resp.body)
        for name, mime in (("app.js", "javascript"), ("style.css", "css"), ("pieces.js", "javascript")):
            if os.path.isfile(os.path.join(REPO_ROOT, "web", name)):
                asset = self.api.get("/" + name)
                self.assertEqual(200, asset.status, "/" + name)
                self.assertGreater(len(asset.body), 0, "/" + name)
                self.assertIn(mime, asset.headers.get("Content-Type", ""), "/" + name)

    def test_api_new(self):
        gid, state = self.new_game()
        self.assertEqual(START_FEN, state["fen"])
        self.assertEqual(0, state["ply"])
        self.assertEqual(20, len(state["legal"]))
        self.assertIsNone(state["last"])

        gid2, _ = self.new_game("?human=black")
        self.assertNotEqual(gid, gid2, "every /api/new must mint a fresh gid")

        fen = "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1"
        gid3, state3 = self.new_game("?fen=" + urllib.parse.quote(fen))
        self.assertEqual(fen, state3["fen"])
        self.assertEqual({"white": 1, "black": 0}, state3["material"])

        # `captured` lists the pieces a side has LOST, most valuable first.
        # Black here is missing its queen and one pawn.
        stripped = "rnb1kbnr/ppp1pppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        _, state4 = self.new_game("?fen=" + urllib.parse.quote(stripped))
        self.assertEqual([], state4["captured"]["white"])
        self.assertEqual(["q", "p"], state4["captured"]["black"],
                         "captured must be ordered most valuable first")
        self.assertEqual({"white": 39, "black": 29}, state4["material"])

    def test_api_state(self):
        gid, first = self.new_game()
        again = self.assert_state(
            self.ok(self.api.get("/api/state?gid=%d" % gid), "/api/state"), gid
        )
        self.assertEqual(first, again, "/api/state changed a game it only read")

    def test_api_move(self):
        gid, state = self.new_game()
        payload = self.ok(self.api.post("/api/move", {"gid": gid, "move": "e2e4"}), "/api/move")
        self.assertEqual({"state"}, set(payload), "/api/move returns only a state")
        after = self.assert_state(payload["state"], gid, "/api/move state")
        self.assertEqual(["e2e4"], after["moves"])
        self.assertEqual(["e4"], after["history_san"])
        self.assertEqual("black", after["turn"])
        self.assertEqual({"from": "e2", "to": "e4"}, {k: after["last"][k] for k in ("from", "to")})

    def test_api_move_promotion(self):
        fen = "4k3/P7/8/8/8/8/8/4K3 w - - 0 1"
        gid, state = self.new_game("?fen=" + urllib.parse.quote(fen))
        self.assertIn("a7a8q", state["legal"])
        promoted = self.ok(
            self.api.post("/api/move", {"gid": gid, "move": "a7a8q"}), "/api/move promo"
        )["state"]
        self.assert_state(promoted, gid, "promotion state")
        self.assertIn("Q", promoted["fen"].split()[0])
        self.assertEqual(["a8=Q+"], promoted["history_san"])

    def test_api_undo(self):
        gid, start = self.new_game()
        for uci in SCHOLARS_MATE[:4]:
            self.ok(self.api.post("/api/move", {"gid": gid, "move": uci}), "/api/move")
        payload = self.ok(self.api.post("/api/undo", {"gid": gid, "plies": 2}), "/api/undo")
        self.assertEqual({"state"}, set(payload))
        back = self.assert_state(payload["state"], gid, "/api/undo state")
        self.assertEqual(SCHOLARS_MATE[:2], back["moves"])
        self.assertEqual(2, back["ply"])

        whole = self.ok(self.api.post("/api/undo", {"gid": gid, "plies": 64}), "/api/undo")["state"]
        self.assertEqual(start, whole, "undoing everything must restore the opening position")

    def test_api_ai(self):
        gid, state = self.new_game()
        payload = self.ok(
            self.api.post("/api/ai", {"gid": gid, "depth": TEST_DEPTH, "movetime": TEST_MOVETIME}),
            "/api/ai",
        )
        for key in ("move", "san", "score", "depth", "nodes", "ms", "value", "top", "state"):
            self.assertIn(key, payload, "/api/ai is missing %r" % key)
        self.assertIn(payload["move"], state["legal"], "/api/ai played an illegal move")
        self.assertIsInstance(payload["san"], str)
        for key in ("score", "depth", "nodes", "ms"):
            self.assertIsInstance(payload[key], int, "/api/ai %s" % key)
        self.assertIsInstance(payload["value"], (int, float))
        self.assertTrue(-1.001 <= float(payload["value"]) <= 1.001, payload["value"])
        self.assertIsInstance(payload["top"], list)
        for item in payload["top"]:
            self.assertIn(item["move"], state["legal"], "/api/ai top[] lists an illegal move")
            self.assertIsInstance(item["san"], str)
            self.assertTrue(0.0 <= float(item["prob"]) <= 1.0, item)
        after = self.assert_state(payload["state"], gid, "/api/ai state")
        self.assertEqual([payload["move"]], after["moves"], "/api/ai did not play its own move")

    def test_api_ai_accepts_the_payload_the_ui_sends(self):
        """web/app.js posts a `blunder` difficulty knob on every engine move."""
        gid, _ = self.new_game()
        resp = self.api.post(
            "/api/ai",
            {"gid": gid, "depth": 2, "movetime": 250, "blunder": 0.18},
        )
        payload = self.ok(resp, "/api/ai with blunder")
        self.assertTrue(payload["move"])

    def test_api_hint_does_not_play(self):
        gid, state = self.new_game()
        payload = self.ok(self.api.get("/api/hint?gid=%d" % gid), "/api/hint")
        for key in ("move", "san", "score", "depth", "nodes", "ms", "value", "top", "state"):
            self.assertIn(key, payload, "/api/hint is missing %r" % key)
        self.assertIn(payload["move"], state["legal"])
        self.assertEqual(state, self.assert_state(payload["state"], gid, "/api/hint state"),
                         "/api/hint must not advance the game")
        self.assertEqual(state, self.assert_state(
            self.ok(self.api.get("/api/state?gid=%d" % gid)), gid), "/api/hint mutated the game")

    def test_api_model(self):
        payload = self.ok(self.api.get("/api/model"), "/api/model")
        self.assertIsInstance(payload["path"], str)
        self.assertIsInstance(payload["generation"], int)
        self.assertIsInstance(payload["n_agents"], int)
        self.assertIsInstance(payload["agents"], list)
        self.assertEqual(payload["n_agents"], len(payload["agents"]))
        seen = set()
        for agent in payload["agents"]:
            self.assertIsInstance(agent["i"], int)
            self.assertIsInstance(agent["elo"], (int, float))
            self.assertIsInstance(agent["rank"], int)
            self.assertNotIn(agent["i"], seen, "duplicate agent index")
            seen.add(agent["i"])

    def test_api_report(self):
        payload = self.ok(self.api.get("/api/report"), "/api/report")
        self.assertIsInstance(payload, dict, "/api/report must be a JSON object")


# ==========================================================================
# 5. THE critical property: legality is decided by the server, not the client
# ==========================================================================


class TestMoveLegalityProperty(ServerTestCase):

    TARGET_MOVES = 200

    def _off_list_candidates(self, state, rng, want=3):
        """Well-formed UCI strings that are NOT in this position's legal list."""
        squares = [f + r for f in "abcdefgh" for r in "12345678"]
        legal = set(state["legal"])
        out = []
        guard = 0
        while len(out) < want and guard < 400:
            guard += 1
            src, dst = rng.choice(squares), rng.choice(squares)
            if src == dst:
                continue
            uci = src + dst
            if uci in legal or uci in out:
                continue
            # a promotion suffix would make a different move; keep it plain
            if any(x.startswith(uci) for x in legal):
                continue
            out.append(uci)
        return out

    def test_server_never_accepts_a_move_outside_its_own_legal_list(self):
        rng = random.Random(20240607)
        played = 0
        games = 0
        rejected = 0
        while played < self.TARGET_MOVES:
            gid, state = self.new_game()
            games += 1
            for _ in range(80):
                if played >= self.TARGET_MOVES:
                    break
                if state["result"] != 0 or not state["legal"]:
                    break

                offered = list(state["legal"])

                # (a) anything NOT on the list must be refused, and must leave
                #     the position untouched.
                for bogus in self._off_list_candidates(state, rng):
                    resp = self.api.post("/api/move", {"gid": gid, "move": bogus})
                    self.assert_json_error(resp, 400, 400, "/api/move %s" % bogus)
                    self.assertEqual("illegal move", resp.json["error"])
                    rejected += 1
                    unchanged = self.assert_state(
                        self.ok(self.api.get("/api/state?gid=%d" % gid)), gid, "after refusal"
                    )
                    self.assertEqual(state, unchanged,
                                     "a rejected move changed the game")

                # (b) a move from the list must be accepted, and the state that
                #     comes back must be the same state a fresh GET reports.
                uci = rng.choice(offered)
                resp = self.api.post("/api/move", {"gid": gid, "move": uci})
                moved = self.ok(resp, "/api/move %s" % uci)["state"]
                self.assert_state(moved, gid, "after %s" % uci)
                self.assertEqual(uci, moved["moves"][-1], "the server played a different move")
                fresh = self.assert_state(
                    self.ok(self.api.get("/api/state?gid=%d" % gid)), gid, "fresh state"
                )
                self.assertEqual(moved["legal"], fresh["legal"],
                                 "/api/move's legal list disagrees with /api/state")
                self.assertEqual(moved, fresh,
                                 "/api/move's state disagrees with a fresh /api/state")
                state = fresh
                played += 1

        self.assertEqual(self.TARGET_MOVES, played)
        self.assertGreaterEqual(games, 3, "the property should span several games")
        self.assertGreater(rejected, 0)
        sys.stderr.write(
            "[integration] %d random legal moves accepted across %d games, "
            "%d off-list moves refused\n" % (played, games, rejected)
        )

    def test_well_formed_but_illegal_move_is_400(self):
        gid, _ = self.new_game()
        for uci in ("e2e5", "e1e3", "a1a5", "d1d5", "b1b3", "e7e5", "a7a6"):
            resp = self.api.post("/api/move", {"gid": gid, "move": uci})
            self.assert_json_error(resp, 400, 400, "/api/move %s" % uci)
            self.assertEqual("illegal move", resp.json["error"])
        self.assertEqual(0, self.ok(self.api.get("/api/state?gid=%d" % gid))["ply"])

    def test_moves_into_check_are_refused(self):
        # White king on e1, black rook on e8: Kd1/Kf1 are fine, Ke2 is not.
        fen = "4r2k/8/8/8/8/8/8/4K3 w - - 0 1"
        gid, state = self.new_game("?fen=" + urllib.parse.quote(fen))
        self.assertNotIn("e1e2", state["legal"])
        self.assert_json_error(
            self.api.post("/api/move", {"gid": gid, "move": "e1e2"}), 400, 400, "e1e2"
        )
        self.assertIn("e1d1", state["legal"])
        self.ok(self.api.post("/api/move", {"gid": gid, "move": "e1d1"}), "e1d1")


# ==========================================================================
# 6 + 7. a whole engine game, and a whole /api/watch game
# ==========================================================================


class TestFullGames(ServerTestCase):

    def test_engine_plays_both_sides_until_the_game_ends(self):
        gid, state = self.new_game()
        plies = 0
        started = time.monotonic()
        while plies < SELF_PLAY_PLY_CAP:
            if state["result"] != 0 or not state["legal"]:
                break
            offered = list(state["legal"])
            payload = self.ok(
                self.api.post(
                    "/api/ai",
                    {"gid": gid, "depth": TEST_DEPTH, "movetime": TEST_MOVETIME},
                ),
                "/api/ai at ply %d" % plies,
            )
            self.assertIn(payload["move"], offered,
                          "engine move %r at ply %d was not in the legal list"
                          % (payload["move"], plies))
            state = self.assert_state(payload["state"], gid, "ply %d" % plies)
            self.assertEqual(payload["move"], state["moves"][-1])
            plies += 1

        took = time.monotonic() - started
        self.assertNotEqual(
            0, state["result"],
            "the engine did not finish a game in %d plies (%.1fs); last fen %r"
            % (plies, took, state["fen"]),
        )
        self.assertIn(state["reason"], DOCUMENTED_REASONS,
                      "undocumented termination reason %r" % state["reason"])
        sys.stderr.write(
            "[integration] engine self-play: %d plies, result %d (%s), %.1fs\n"
            % (plies, state["result"], state["reason"], took)
        )
        self.assertEqual(plies, state["ply"])
        self.assertEqual(plies, len(state["moves"]))
        self.assertEqual(plies, len(state["history_san"]))
        if state["reason"] == "checkmate":
            self.assertIn(state["result"], (1, 2))
            self.assertTrue(state["check"])
            self.assertEqual([], state["legal"])
        else:
            self.assertEqual(3, state["result"], "non-mate terminations are draws")

        # The engine must refuse to move in a finished game rather than 500.
        self.assert_json_error(
            self.api.post("/api/ai", {"gid": gid, "depth": 1, "movetime": 20}),
            400, 400, "/api/ai after the game ended",
        )

    def test_api_watch_returns_a_self_consistent_game(self):
        payload = self.ok(
            self.api.get("/api/watch?a=0&b=1&plies=120&depth=1&movetime=30"), "/api/watch"
        )
        for key in ("result", "reason", "moves", "sans", "fens", "evals", "white", "black"):
            self.assertIn(key, payload, "/api/watch is missing %r" % key)
        moves, fens, sans, evals = (
            payload["moves"], payload["fens"], payload["sans"], payload["evals"]
        )
        self.assertIn(payload["result"], (0, 1, 2, 3))
        self.assertIn(payload["reason"], WATCH_REASONS | {""}, payload["reason"])
        self.assertIsInstance(payload["white"], str)
        self.assertIsInstance(payload["black"], str)
        self.assertGreater(len(moves), 0, "/api/watch produced no moves")
        self.assertEqual(len(moves), len(sans), "sans is not parallel to moves")
        self.assertEqual(len(moves), len(evals), "evals is not parallel to moves")
        self.assertEqual(len(moves) + 1, len(fens),
                         "fens must hold the position before every move plus the last one")
        for value in evals:
            self.assertIsInstance(value, (int, float))
            self.assertTrue(-1.001 <= float(value) <= 1.001, value)

        # Replay the whole game through the library: applying moves[i] to fens[i]
        # must land exactly on fens[i+1].
        for i, uci in enumerate(moves):
            self.assertRegex(uci, UCI_RE)
            with engine.GameHandle(fens[i]) as game:
                self.assertIn(uci, game.legal(),
                              "watch move %d (%s) is not legal in fens[%d]=%r"
                              % (i, uci, i, fens[i]))
                self.assertTrue(game.move(uci))
                self.assertEqual(
                    fens[i + 1], game.state()["fen"],
                    "watch fens[%d] is not reachable from fens[%d] by %s" % (i + 1, i, uci),
                )

        # The declared result must match the final position.
        with engine.GameHandle(fens[-1]) as game:
            final = game.state()
        if payload["reason"] == "checkmate":
            self.assertEqual([], final["legal"])
            self.assertTrue(final["check"])

    def test_api_watch_defaults_to_the_champion(self):
        payload = self.ok(self.api.get("/api/watch?plies=20&depth=1&movetime=20"), "/api/watch")
        self.assertEqual("champion", payload["white"])
        self.assertEqual("champion", payload["black"])


# ==========================================================================
# 8. concurrency
# ==========================================================================


class TestConcurrency(ServerTestCase):

    THREADS = 8
    MOVES_EACH = 12
    # Deep enough that every request actually spends its movetime thinking, so
    # the eight threads genuinely overlap inside py/server.py's engine pool
    # (ENGINE_POOL is 4) instead of finishing one at a time.
    DEPTH = 8
    MOVETIME = 60

    def test_eight_simultaneous_games_do_not_interfere(self):
        results = [None] * self.THREADS
        errors = []
        barrier = threading.Barrier(self.THREADS)

        def worker(slot):
            api = ApiClient(self.server.base)
            try:
                created = api.get("/api/new")
                if created.status != 200:
                    raise AssertionError("/api/new -> %d %r" % (created.status, created.body[:200]))
                gid = created.json["gid"]
                mine = []
                barrier.wait(timeout=60)
                for _ in range(self.MOVES_EACH):
                    resp = api.post(
                        "/api/ai",
                        {"gid": gid, "depth": self.DEPTH, "movetime": self.MOVETIME},
                    )
                    if resp.status != 200:
                        raise AssertionError(
                            "/api/ai -> %d %r" % (resp.status, resp.body[:200])
                        )
                    state = resp.json["state"]
                    if state["gid"] != gid:
                        raise AssertionError(
                            "cross-talk: asked for gid %d, got gid %d" % (gid, state["gid"])
                        )
                    mine.append(resp.json["move"])
                    if state["moves"] != mine:
                        raise AssertionError(
                            "cross-talk in gid %d: history %r, my moves %r"
                            % (gid, state["moves"], mine)
                        )
                    if state["ply"] != len(mine):
                        raise AssertionError(
                            "gid %d jumped to ply %d after %d of my moves"
                            % (gid, state["ply"], len(mine))
                        )
                    if state["result"] != 0:
                        break
                results[slot] = (gid, mine)
            except BaseException as exc:  # noqa: BLE001 - reported, not swallowed
                errors.append("thread %d: %r" % (slot, exc))
                try:
                    barrier.abort()
                except Exception:
                    pass

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(self.THREADS)]
        started = time.monotonic()
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=180)
            self.assertFalse(thread.is_alive(), "a worker thread hung")
        took = time.monotonic() - started

        self.assertEqual([], errors, "concurrent games interfered:\n  " + "\n  ".join(errors))
        finished = [r for r in results if r is not None]
        self.assertEqual(self.THREADS, len(finished), "not every thread finished")
        gids = [gid for gid, _ in finished]
        self.assertEqual(len(set(gids)), len(gids), "the server reused a gid: %r" % (gids,))

        # Every game must still read back exactly what its own thread played.
        for gid, mine in finished:
            state = self.assert_state(
                self.ok(self.api.get("/api/state?gid=%d" % gid)), gid, "gid %d" % gid
            )
            self.assertEqual(mine, state["moves"], "gid %d was mutated by another game" % gid)
        sys.stderr.write("[integration] %d concurrent games in %.1fs\n" % (self.THREADS, took))

    def test_concurrent_state_reads_are_consistent(self):
        gid, _ = self.new_game()
        for uci in SCHOLARS_MATE[:4]:
            self.ok(self.api.post("/api/move", {"gid": gid, "move": uci}))
        seen = []
        lock = threading.Lock()

        def reader():
            api = ApiClient(self.server.base)
            for _ in range(12):
                resp = api.get("/api/state?gid=%d" % gid)
                with lock:
                    seen.append((resp.status, json.dumps(resp.json, sort_keys=True)))

        threads = [threading.Thread(target=reader) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=60)
        self.assertEqual(96, len(seen))
        self.assertEqual({200}, {status for status, _ in seen})
        self.assertEqual(1, len({body for _, body in seen}),
                         "concurrent reads of an idle game disagreed")


# ==========================================================================
# 9. robustness: everything malformed must come back as clean JSON
# ==========================================================================


class TestRobustness(ServerTestCase):

    def test_malformed_json_body(self):
        for raw in (b"{not json", b"{", b"[1,2,3]", b'{"gid":}', b"\xff\xfe\x00", b'"just a string"'):
            resp = self.api.request(
                "POST", "/api/move", raw_body=raw,
                headers={"Content-Type": "application/json"},
            )
            self.assert_json_error(resp, 400, 400, "/api/move body=%r" % raw)

    def test_missing_and_unknown_gid(self):
        cases = [
            ("GET", "/api/state", None, 400),
            ("GET", "/api/hint", None, 400),
            ("POST", "/api/move", {"move": "e2e4"}, 400),
            ("POST", "/api/ai", {"depth": 1}, 400),
            ("POST", "/api/undo", {"plies": 1}, 400),
            ("GET", "/api/state?gid=", None, 400),
            ("GET", "/api/state?gid=abc", None, 400),
            ("GET", "/api/state?gid=-1", None, 400),
            ("GET", "/api/state?gid=1.5", None, 400),
            ("GET", "/api/state?gid=999999", None, 404),
            ("POST", "/api/move", {"gid": 999999, "move": "e2e4"}, 404),
            ("POST", "/api/ai", {"gid": 999999}, 404),
            ("POST", "/api/undo", {"gid": 999999, "plies": 1}, 404),
        ]
        for method, path, body, want in cases:
            resp = self.api.request(method, path, body=body)
            self.assertEqual(
                want, resp.status,
                "%s %s %r -> %d %r" % (method, path, body, resp.status, resp.body[:200]),
            )
            self.assert_json_error(resp, want, want, "%s %s" % (method, path))

    def test_out_of_range_and_junk_parameters(self):
        gid, _ = self.new_game()
        # Junk must be a clean 400 ...
        for body in (
            {"gid": gid, "depth": "deep"},
            {"gid": gid, "depth": 1.5},
            {"gid": gid, "depth": True},
            {"gid": gid, "movetime": "soon"},
            {"gid": gid, "movetime": []},
            {"gid": gid, "depth": 1, "movetime": 20, "blunder": "lots"},
            {"gid": gid, "depth": 1, "movetime": 20, "nonsense": 1},
        ):
            self.assert_json_error(self.api.post("/api/ai", body), 400, 400, "/api/ai %r" % body)

        # ... while an out-of-range but well-typed depth/movetime is clamped into
        # the supported band (py/server.py: want_int(..., clamp=True)).  Either
        # way it must never be a 5xx and never move outside 1..8.
        for body in (
            {"gid": gid, "depth": 999, "movetime": 20},
            {"gid": gid, "depth": -5, "movetime": 20},
            {"gid": gid, "depth": 0, "movetime": 20},
            {"gid": gid, "depth": 1, "movetime": 10 ** 9},
            {"gid": gid, "depth": 1, "movetime": -1},
        ):
            resp = self.api.post("/api/ai", body)
            self.assertLess(resp.status, 500, "/api/ai %r -> %d" % (body, resp.status))
            if resp.status == 200:
                self.assertTrue(1 <= resp.json["depth"] <= 8,
                                "/api/ai %r reported depth %r" % (body, resp.json["depth"]))
                self.ok(self.api.post("/api/undo", {"gid": gid, "plies": 1}))
            else:
                self.assert_json_error(resp, 400, 400, "/api/ai %r" % body)

    def test_malformed_moves_and_fens(self):
        gid, _ = self.new_game()
        for move in ("", "e2", "e2e4e5x", "z9z9", "e2e9", "e2e4k", 42, None, [], {"a": 1}, "  "):
            self.assert_json_error(
                self.api.post("/api/move", {"gid": gid, "move": move}), 400, 400,
                "/api/move %r" % (move,),
            )
        for fen in ("zzz", "not/a/fen", "8/8/8/8 w - - 0 1", "x" * 400):
            self.assert_json_error(
                self.api.get("/api/new?fen=" + urllib.parse.quote(fen)), 400, 400,
                "/api/new fen=%r" % fen,
            )

    def test_unknown_agents_and_endpoints(self):
        self.assert_json_error(self.api.get("/api/watch?a=99999&b=0&plies=4"), 400, 400, "watch a")
        self.assert_json_error(self.api.get("/api/watch?a=abc"), 400, 400, "watch a=abc")
        for path in ("/api/nope", "/api/", "/api", "/api/state/extra"):
            self.assert_json_error(self.api.get(path), 400, 404, path)
        self.assert_json_error(self.api.post("/api/state", {"gid": 1}), 400, 404, "POST /api/state")
        # HEAD must not be able to create a game or start an agent-vs-agent game.
        # (A HEAD reply carries no body, so there is no JSON to inspect.)
        for path in ("/api/new", "/api/watch"):
            resp = self.api.request("HEAD", path)
            self.assertEqual(405, resp.status, "HEAD %s -> %d" % (path, resp.status))
            self.assertEqual(b"", resp.body)
        self.assertEqual(200, self.api.request("HEAD", "/").status)

    def test_path_traversal_is_refused(self):
        secrets = ("py/server.py", "Makefile", "src/api.c", "runs/smoke/best.crl")
        attempts = [
            "/../Makefile",
            "/../../etc/passwd",
            "/..%2f..%2fetc%2fpasswd",
            "/%2e%2e/%2e%2e/Makefile",
            "/....//Makefile",
            "/web/../py/server.py",
            "/./../../py/engine.py",
            "//etc/passwd",
            "/%00",
            "/..;/Makefile",
            "/" + "../" * 12 + "etc/passwd",
        ]
        for path in attempts:
            resp = self.api.get(path)
            self.assertTrue(
                400 <= resp.status <= 499,
                "%s -> %d (%r)" % (path, resp.status, resp.body[:200]),
            )
            for name in secrets:
                full = os.path.join(REPO_ROOT, name)
                if os.path.isfile(full):
                    with open(full, "rb") as fh:
                        head = fh.read(64)
                    self.assertNotIn(head, resp.body, "%s served %s" % (path, name))

    def test_a_body_nobody_reads_does_not_poison_keep_alive(self):
        """HTTP/1.1 pipelining regression.

        A POST to a route that answers without consuming its body used to leave
        those bytes in the socket, so the NEXT request on the same connection was
        parsed starting from them and came back as a 501 whose status line echoed
        the previous body.
        """
        body = b'{"gid":1,"move":"e2e4"}'
        first = (
            b"POST /index.html HTTP/1.1\r\nHost: 127.0.0.1\r\n"
            b"Content-Type: application/json\r\n"
            b"Content-Length: %d\r\n\r\n" % len(body)
        ) + body
        second = b"GET /api/model HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n"

        sock = socket.create_connection(("127.0.0.1", self.server.port), 30)
        sock.settimeout(30)
        try:
            sock.sendall(first + second)
            chunks = []
            while True:
                chunk = sock.recv(65536)
                if not chunk:
                    break
                chunks.append(chunk)
        finally:
            sock.close()
        data = b"".join(chunks)

        self.assertEqual(2, data.count(b"HTTP/1.1 "), "expected two responses: %r" % data[:400])
        self.assertIn(b"HTTP/1.1 405", data, "POST to a static path should be 405")
        self.assertIn(b"HTTP/1.1 200 OK", data,
                      "the second request was corrupted by the first body: %r" % data[:600])
        self.assertNotIn(b"HTTP/1.1 501", data, "the connection desynced: %r" % data[:600])
        self.assertNotIn(b"Unsupported method", data,
                         "the unread body was parsed as the next request line")

    def test_oversized_and_odd_bodies(self):
        huge = json.dumps({"gid": 1, "move": "e2e4", "pad": "x" * (80 * 1024)}).encode()
        resp = self.api.request(
            "POST", "/api/move", raw_body=huge, headers={"Content-Type": "application/json"}
        )
        self.assertIn(resp.status, (400, 413), "oversized body -> %d" % resp.status)
        self.assertLess(resp.status, 500)

        # An empty POST body is "no fields", i.e. a missing gid, not a crash.
        self.assert_json_error(
            self.api.request("POST", "/api/move", raw_body=b"",
                             headers={"Content-Type": "application/json"}),
            400, 400, "empty body",
        )

    def test_no_endpoint_ever_returns_5xx_for_client_mistakes(self):
        """A sweep: nothing a confused client can send should be a server error."""
        gid, _ = self.new_game()
        probes = [
            ("GET", "/api/new?human=purple", None),
            ("GET", "/api/new?fen=" + "/" * 40, None),
            ("GET", "/api/watch?plies=0", None),
            ("GET", "/api/watch?plies=abc", None),
            ("GET", "/api/watch?b=-99", None),
            ("GET", "/api/hint?gid=%d&depth=abc" % gid, None),
            ("GET", "/api/state?gid=%d&gid=%d" % (gid, gid), None),
            ("POST", "/api/undo", {"gid": gid, "plies": "many"}),
            ("POST", "/api/undo", {"gid": gid, "plies": -4}),
            ("POST", "/api/new", {"fen": 17}),
            ("POST", "/api/new", {"human": 3}),
            ("POST", "/api/ai", {"gid": gid, "depth": None, "movetime": None}),
            ("GET", "/favicon.ico", None),
            ("GET", "/does-not-exist.js", None),
        ]
        for method, path, body in probes:
            resp = self.api.request(method, path, body=body)
            self.assertLess(
                resp.status, 500,
                "%s %s %r -> %d %r" % (method, path, body, resp.status, resp.body[:300]),
            )
            self.assertNotIn(b"Traceback", resp.body, "%s %s leaked a traceback" % (method, path))


# ==========================================================================


def main():
    started = time.monotonic()
    result = unittest.main(module=__name__, exit=False, verbosity=2).result
    sys.stderr.write(
        "\n[integration] %d tests in %.1fs -- %d failures, %d errors, %d skipped\n"
        % (result.testsRun, time.monotonic() - started,
           len(result.failures), len(result.errors), len(result.skipped))
    )
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
