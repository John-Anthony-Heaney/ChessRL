#!/usr/bin/env python3
"""ChessRL web server -- Python 3 standard library only.

Serves the static UI from ``web/`` at ``/`` and implements exactly the HTTP
contract in ``docs/API.md`` under ``/api/``.

    python3 py/server.py --model runs/NAME/best.crl --port 8000 --open

With no ``--model`` the newest ``runs/*/best.crl`` is used.  When no model
exists at all the UI is still served; the AI endpoints answer HTTP 503 with a
JSON error so the page can say "train a model first".
"""

from __future__ import annotations

import argparse
import contextlib
import errno
import json
import math
import os
import posixpath
import queue
import sys
import threading
import time
import traceback
import urllib.parse
import webbrowser
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import engine  # noqa: E402  (local module, after sys.path fix)

REPO_ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = REPO_ROOT / "web"
RUNS_DIR = REPO_ROOT / "runs"

MAX_GAMES = 200            # LRU cap on live games
MAX_BODY = 64 * 1024       # request body limit
MAX_FEN = 128
MIN_DEPTH, MAX_DEPTH = 1, 8
MIN_MOVETIME, MAX_MOVETIME = 10, 15000
DEFAULT_DEPTH, DEFAULT_MOVETIME = 4, 800
MAX_UNDO_PLIES = 1024
WATCH_MAX_PLIES = 1024
WATCH_DEFAULT_PLIES = 400
WATCH_BUDGET_S = 25.0      # wall-clock cap for one /api/watch game
WATCH_CONCURRENCY = 2
ENGINE_POOL = max(2, min(4, (os.cpu_count() or 2)))
AGENT_CACHE = 12

_FEN_CHARS = set("pnbrqkPNBRQK/12345678 wb-abcdefgh0123456789")
_UCI_FILES = "abcdefgh"
_UCI_RANKS = "12345678"
_PROMO = "qrbn"

LIB_MISSING = "libchessrl.dylib is not built -- run 'make -j lib' in %s" % REPO_ROOT
NO_MODEL = (
    "no model available -- train one first (build/chessrl writes runs/NAME/best.crl), "
    "then restart the server or pass --model"
)

_log_lock = threading.Lock()


def log(msg):
    with _log_lock:
        sys.stderr.write(msg + "\n")
        try:
            sys.stderr.flush()
        except Exception:
            pass


# ==========================================================================
# errors
# ==========================================================================


class HttpError(Exception):
    """Raised anywhere in a handler; becomes ``{"error": ...}`` with this status."""

    def __init__(self, code, message):
        super().__init__(message)
        self.code = code
        self.message = message

    def payload(self):
        return {"error": self.message}


# ==========================================================================
# validation helpers
# ==========================================================================


def want_int(value, name, lo, hi, default=None, clamp=False):
    """Coerce ``value`` (int or numeric string) to an int in [lo, hi]."""
    if value is None or value == "":
        if default is None:
            raise HttpError(400, "missing %s" % name)
        return default
    if isinstance(value, bool):
        raise HttpError(400, "%s must be an integer" % name)
    if isinstance(value, int):
        n = value
    elif isinstance(value, float):
        if value != int(value):
            raise HttpError(400, "%s must be an integer" % name)
        n = int(value)
    elif isinstance(value, str):
        try:
            n = int(value.strip(), 10)
        except ValueError:
            raise HttpError(400, "%s must be an integer" % name) from None
    else:
        raise HttpError(400, "%s must be an integer" % name)
    if clamp:
        return max(lo, min(hi, n))
    if n < lo or n > hi:
        raise HttpError(400, "%s out of range (%d..%d)" % (name, lo, hi))
    return n


def want_uci(value):
    if not isinstance(value, str):
        raise HttpError(400, "move must be a string")
    s = value.strip().lower()
    if len(s) not in (4, 5):
        raise HttpError(400, "malformed move")
    if (
        s[0] not in _UCI_FILES
        or s[1] not in _UCI_RANKS
        or s[2] not in _UCI_FILES
        or s[3] not in _UCI_RANKS
    ):
        raise HttpError(400, "malformed move")
    if len(s) == 5 and s[4] not in _PROMO:
        raise HttpError(400, "malformed move")
    return s


def want_fen(value):
    if value is None:
        return None
    if not isinstance(value, str):
        raise HttpError(400, "fen must be a string")
    s = value.strip()
    if not s:
        return None
    if len(s) > MAX_FEN:
        raise HttpError(400, "fen too long")
    if any(ch not in _FEN_CHARS for ch in s):
        raise HttpError(400, "malformed fen")
    if s.count("/") != 7:
        raise HttpError(400, "malformed fen")
    return s


def check_keys(body, allowed, where):
    if not isinstance(body, dict):
        raise HttpError(400, "%s body must be a JSON object" % where)
    unknown = sorted(k for k in body if k not in allowed)
    if unknown:
        raise HttpError(400, "unexpected field(s): %s" % ", ".join(unknown))


# ==========================================================================
# <State> normalisation
#
# api.h documents api_game_state as emitting
#   fen, turn, result, reason, check, ply, moves[], san[], legal[], last, material
# while docs/API.md additionally requires gid, history_san and captured.  The
# missing pieces are filled in here so the UI always receives the documented
# shape whatever the C layer chooses to emit.
# ==========================================================================

_REASON_BY_CODE = {
    0: "",
    1: "checkmate",
    2: "stalemate",
    3: "fifty-move",
    4: "threefold repetition",
    5: "insufficient material",
    6: "move limit",
    7: "adjudicated",
    8: "resignation",
}

_REASON_ALIAS = {
    "": "",
    "none": "",
    "ongoing": "",
    "checkmate": "checkmate",
    "mate": "checkmate",
    "stalemate": "stalemate",
    "fifty": "fifty-move",
    "fifty-move": "fifty-move",
    "fifty_move": "fifty-move",
    "fiftymove": "fifty-move",
    "50-move": "fifty-move",
    "repetition": "threefold repetition",
    "threefold": "threefold repetition",
    "threefold repetition": "threefold repetition",
    "insufficient": "insufficient material",
    "insufficient material": "insufficient material",
    "insufficient_material": "insufficient material",
    "maxplies": "move limit",
    "max_plies": "move limit",
    "max plies": "move limit",
    "ply limit": "move limit",
    "move limit": "move limit",
    "adjudicated": "adjudicated",
    "resign": "resignation",
    "resignation": "resignation",
}

_RESULT_ALIAS = {
    "ongoing": 0,
    "*": 0,
    "white": 1,
    "white_win": 1,
    "1-0": 1,
    "black": 2,
    "black_win": 2,
    "0-1": 2,
    "draw": 3,
    "1/2-1/2": 3,
}

_PIECE_VALUE = {"p": 1, "n": 3, "b": 3, "r": 5, "q": 9, "k": 0}
_START_COUNTS = {"p": 8, "n": 2, "b": 2, "r": 2, "q": 1, "k": 1}
_CAPTURE_ORDER = "qrbnp"

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def _str_list(value):
    if isinstance(value, list):
        return [v for v in value if isinstance(v, str)]
    if isinstance(value, str):
        return value.split()
    return []


def _board_counts(fen):
    """{'white': {piece: n}, 'black': {...}} from the board part of a FEN."""
    white, black = {}, {}
    board = (fen or "").split(" ", 1)[0]
    for ch in board:
        if ch.isalpha():
            if ch.isupper():
                white[ch.lower()] = white.get(ch.lower(), 0) + 1
            else:
                black[ch] = black.get(ch, 0) + 1
    return white, black


def _material_from_fen(fen):
    white, black = _board_counts(fen)
    return {
        "white": sum(_PIECE_VALUE.get(k, 0) * n for k, n in white.items()),
        "black": sum(_PIECE_VALUE.get(k, 0) * n for k, n in black.items()),
    }


def _captured_from_fen(fen):
    """Pieces missing from each side relative to a full army.

    ``captured["white"]`` lists the WHITE pieces that are gone, matching
    ``material["white"]`` which is White's own remaining material.
    """
    white, black = _board_counts(fen)
    out = {}
    for name, have in (("white", white), ("black", black)):
        lost = []
        for piece in _CAPTURE_ORDER:
            gone = _START_COUNTS.get(piece, 0) - have.get(piece, 0)
            lost.extend([piece] * max(0, gone))
        out[name] = lost
    return out


def _normalise_turn(raw, fen):
    turn = raw.get("turn")
    if isinstance(turn, str):
        t = turn.strip().lower()
        if t.startswith("w"):
            return "white"
        if t.startswith("b"):
            return "black"
    if isinstance(turn, int) and not isinstance(turn, bool):
        return "black" if turn else "white"
    parts = (fen or "").split()
    if len(parts) > 1 and parts[1].lower().startswith("b"):
        return "black"
    return "white"


def _normalise_result(raw):
    value = raw.get("result", 0)
    if isinstance(value, bool):
        return 0
    if isinstance(value, (int, float)):
        n = int(value)
        return n if 0 <= n <= 3 else 0
    if isinstance(value, str):
        return _RESULT_ALIAS.get(value.strip().lower(), 0)
    return 0


def _normalise_reason(raw):
    value = raw.get("reason", "")
    if isinstance(value, bool):
        return ""
    if isinstance(value, (int, float)):
        return _REASON_BY_CODE.get(int(value), "")
    if isinstance(value, str):
        return _REASON_ALIAS.get(value.strip().lower(), value.strip())
    return ""


def _normalise_last(raw, moves):
    last = raw.get("last")
    if isinstance(last, dict):
        src, dst = last.get("from"), last.get("to")
        if isinstance(src, str) and isinstance(dst, str):
            out = {"from": src, "to": dst}
            if isinstance(last.get("san"), str):
                out["san"] = last["san"]
            return out
    if isinstance(last, str) and len(last) >= 4:
        return {"from": last[:2], "to": last[2:4]}
    if moves:
        uci = moves[-1]
        if isinstance(uci, str) and len(uci) >= 4:
            return {"from": uci[:2], "to": uci[2:4]}
    return None


def normalise_state(raw, gid, history_san=None):
    """Turn whatever ``api_game_state`` emitted into the documented <State>."""
    if not isinstance(raw, dict):
        raise HttpError(500, "engine returned a malformed state")

    fen = raw.get("fen")
    if not isinstance(fen, str) or not fen:
        fen = START_FEN

    legal = _str_list(raw.get("legal"))
    san = _str_list(raw.get("san"))
    if len(san) != len(legal):
        san = list(legal)

    moves = _str_list(raw.get("moves"))
    hist_san = _str_list(raw.get("history_san"))
    if len(hist_san) != len(moves):
        tracked = list(history_san or [])
        hist_san = tracked if len(tracked) == len(moves) else list(moves)

    material = raw.get("material")
    if not (
        isinstance(material, dict)
        and isinstance(material.get("white"), (int, float))
        and isinstance(material.get("black"), (int, float))
    ):
        material = _material_from_fen(fen)
    material = {"white": int(material["white"]), "black": int(material["black"])}

    captured = raw.get("captured")
    if not (
        isinstance(captured, dict)
        and isinstance(captured.get("white"), list)
        and isinstance(captured.get("black"), list)
    ):
        captured = _captured_from_fen(fen)
    captured = {
        "white": [str(x) for x in captured["white"]],
        "black": [str(x) for x in captured["black"]],
    }

    ply = raw.get("ply")
    if not isinstance(ply, int) or isinstance(ply, bool) or ply < 0:
        ply = len(moves)

    return {
        "gid": gid,
        "fen": fen,
        "turn": _normalise_turn(raw, fen),
        "ply": ply,
        "result": _normalise_result(raw),
        "reason": _normalise_reason(raw),
        "check": bool(raw.get("check", False)),
        "legal": legal,
        "san": san,
        "moves": moves,
        "history_san": hist_san,
        "last": _normalise_last(raw, moves),
        "material": material,
        "captured": captured,
    }


def _san_for(state, uci):
    """SAN of ``uci`` given a normalised state, falling back to the UCI itself."""
    try:
        return state["san"][state["legal"].index(uci)]
    except (ValueError, IndexError, KeyError):
        return uci


def _num(value, default=0.0):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    value = float(value)
    return value if math.isfinite(value) else default


def _json_safe(obj, depth=0):
    """Strip NaN/Infinity (and anything unserialisable) out of a payload."""
    if depth > 64:
        return None
    if obj is None or isinstance(obj, (str, bool, int)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {str(k): _json_safe(v, depth + 1) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v, depth + 1) for v in obj]
    return str(obj)


def normalise_engine_reply(raw, state):
    """Documented /api/ai shape from whatever api_engine_move emitted."""
    if not isinstance(raw, dict):
        raise HttpError(500, "engine returned a malformed reply")
    move = raw.get("move")
    if not isinstance(move, str):
        move = ""
    move = move.strip().lower()
    san = raw.get("san")
    if not isinstance(san, str) or not san:
        san = _san_for(state, move) if move else ""
    top = []
    for item in raw.get("top") or []:
        if not isinstance(item, dict):
            continue
        mv = item.get("move")
        if not isinstance(mv, str):
            continue
        entry = {
            "move": mv,
            "san": item["san"] if isinstance(item.get("san"), str) else _san_for(state, mv),
            "prob": round(_num(item.get("prob")), 6),
        }
        if isinstance(item.get("logit"), (int, float)) and not isinstance(item.get("logit"), bool):
            entry["logit"] = round(float(item["logit"]), 6)
        top.append(entry)
    return {
        "move": move,
        "san": san,
        "score": int(_num(raw.get("score"))),
        "depth": int(_num(raw.get("depth"))),
        "nodes": int(_num(raw.get("nodes"))),
        "ms": int(_num(raw.get("ms"))),
        "value": round(_num(raw.get("value")), 6),
        "top": top,
    }


# ==========================================================================
# game store (LRU)
# ==========================================================================


class GameEntry:
    __slots__ = ("gid", "handle", "lock", "human", "history_san", "created")

    def __init__(self, gid, handle, human):
        # `gid` is ours, not the library's: the C side recycles its own ids when
        # a game is freed, so keying on those would let a stale client id silently
        # address somebody else's game after an LRU eviction.
        self.gid = gid
        self.handle = handle
        self.lock = threading.RLock()
        self.human = human
        self.history_san = []
        self.created = time.time()

    def state(self):
        return normalise_state(self.handle.state(), self.gid, self.history_san)

    def close(self):
        try:
            self.handle.close()
        except Exception:
            pass


class GameStore:
    """gid -> GameEntry with a hard LRU cap."""

    def __init__(self, cap=MAX_GAMES):
        self._cap = max(1, int(cap))
        self._items = OrderedDict()
        self._lock = threading.Lock()
        self._next = 1

    def create(self, fen, human):
        handle = engine.GameHandle(fen)
        evicted = []
        with self._lock:
            gid = self._next
            self._next += 1
            entry = GameEntry(gid, handle, human)
            self._items[gid] = entry
            while len(self._items) > self._cap:
                _, old = self._items.popitem(last=False)
                if old is not entry:
                    evicted.append(old)
        for old in evicted:
            old.close()
        return entry

    def get(self, gid):
        with self._lock:
            entry = self._items.get(gid)
            if entry is not None:
                self._items.move_to_end(gid)
        if entry is None:
            raise HttpError(404, "unknown gid")
        return entry

    def close_all(self):
        with self._lock:
            items = list(self._items.values())
            self._items.clear()
        for entry in items:
            entry.close()

    def __len__(self):
        with self._lock:
            return len(self._items)


# ==========================================================================
# engines
# ==========================================================================


class EngineHub:
    """A small pool of champion engines plus a bounded cache of agent engines."""

    def __init__(self, model_path):
        self.model_path = str(model_path) if model_path else None
        self._pool = queue.LifoQueue()
        self._created = 0
        self._lock = threading.Lock()
        self._agents = OrderedDict()   # index -> [Engine, busy]
        self._agent_lock = threading.Lock()
        self._info = None
        self._info_lock = threading.Lock()
        self._fatal = None

    # -- availability -----------------------------------------------------
    def require(self):
        """Raise 503 unless a model and the shared library are both usable."""
        if not self.model_path:
            raise HttpError(503, NO_MODEL)
        if self._fatal:
            raise HttpError(503, self._fatal)
        if not engine.library_available():
            raise HttpError(503, LIB_MISSING)

    def _new_engine(self, agent_index):
        try:
            return engine.Engine(self.model_path, agent_index)
        except FileNotFoundError as exc:
            self._fatal = str(exc)
            raise HttpError(503, self._fatal) from None
        except engine.LibraryNotFoundError:
            raise HttpError(503, LIB_MISSING) from None
        except Exception as exc:
            raise HttpError(503, "could not load model: %s" % exc) from None

    # -- champion pool ----------------------------------------------------
    @contextlib.contextmanager
    def champion(self, timeout=60.0):
        self.require()
        eng = None
        try:
            eng = self._pool.get_nowait()
        except queue.Empty:
            make = False
            with self._lock:
                if self._created < ENGINE_POOL:
                    self._created += 1
                    make = True
            if make:
                try:
                    eng = self._new_engine(-1)
                except BaseException:
                    with self._lock:
                        self._created -= 1
                    raise
            else:
                try:
                    eng = self._pool.get(timeout=timeout)
                except queue.Empty:
                    raise HttpError(503, "engine busy, try again") from None
        try:
            yield eng
        finally:
            self._pool.put(eng)

    # -- per-agent engines ------------------------------------------------
    @contextlib.contextmanager
    def agent(self, index):
        self.require()
        index = int(index)
        with self._agent_lock:
            slot = self._agents.get(index)
            if slot is not None:
                self._agents.move_to_end(index)
                slot[1] += 1
        if slot is None:
            eng = self._new_engine(index)
            drop = []
            with self._agent_lock:
                existing = self._agents.get(index)
                if existing is not None:
                    self._agents.move_to_end(index)
                    existing[1] += 1
                    slot = existing
                    drop.append(eng)
                else:
                    slot = [eng, 1]
                    self._agents[index] = slot
                    while len(self._agents) > AGENT_CACHE:
                        idle = next(
                            (k for k, v in self._agents.items() if v[1] == 0 and k != index),
                            None,
                        )
                        if idle is None:
                            break
                        drop.append(self._agents.pop(idle)[0])
            for old in drop:
                try:
                    old.close()
                except Exception:
                    pass
        try:
            yield slot[0]
        finally:
            with self._agent_lock:
                slot[1] -= 1

    # -- metadata ---------------------------------------------------------
    def info(self):
        self.require()
        with self._info_lock:
            if self._info is None:
                try:
                    self._info = engine.model_info(self.model_path)
                except FileNotFoundError as exc:
                    self._fatal = str(exc)
                    raise HttpError(503, self._fatal) from None
                except Exception as exc:
                    raise HttpError(503, "could not read model: %s" % exc) from None
            return self._info

    def close(self):
        while True:
            try:
                self._pool.get_nowait().close()
            except queue.Empty:
                break
            except Exception:
                break
        with self._agent_lock:
            slots = list(self._agents.values())
            self._agents.clear()
        for slot in slots:
            try:
                slot[0].close()
            except Exception:
                pass


def newest_model():
    """Newest ``runs/*/best.crl``, or None."""
    best, best_mtime = None, -1.0
    try:
        candidates = sorted(RUNS_DIR.glob("*/best.crl"))
    except OSError:
        return None
    for path in candidates:
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        if mtime > best_mtime:
            best, best_mtime = path, mtime
    return best


# ==========================================================================
# static files
# ==========================================================================

_MIME = {
    ".html": "text/html; charset=utf-8",
    ".htm": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".mjs": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".map": "application/json; charset=utf-8",
    ".txt": "text/plain; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".ico": "image/x-icon",
    ".woff": "font/woff",
    ".woff2": "font/woff2",
    ".ttf": "font/ttf",
    ".otf": "font/otf",
    ".wasm": "application/wasm",
    ".mp3": "audio/mpeg",
    ".ogg": "audio/ogg",
    ".wav": "audio/wav",
    ".webmanifest": "application/manifest+json",
}

_PLACEHOLDER = b"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ChessRL</title>
<style>
 body{font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
      background:#14161a;color:#e6e8ea;margin:0;display:grid;place-items:center;
      min-height:100vh}
 main{max-width:34rem;padding:2rem}
 h1{font-size:1.4rem;margin:0 0 .5rem}
 code{background:#22262c;padding:.15em .4em;border-radius:4px}
 p{color:#a8b0b8}
</style></head><body><main>
<h1>ChessRL server is running</h1>
<p>The API is live under <code>/api/</code>, but <code>web/index.html</code>
does not exist yet, so there is no UI to show.</p>
<p>Try <code>/api/model</code> or <code>/api/new</code>.</p>
</main></body></html>
"""


def _mime_for(path):
    return _MIME.get(path.suffix.lower(), "application/octet-stream")


def _safe_web_path(url_path):
    """Map a URL path to a file under web/, or None if it escapes."""
    rel = posixpath.normpath("/" + url_path.lstrip("/")).lstrip("/")
    if rel in ("", "."):
        rel = "index.html"
    if rel.startswith("..") or "\x00" in rel:
        return None
    root = os.path.realpath(str(WEB_DIR))
    target = os.path.realpath(os.path.join(root, *rel.split("/")))
    if target != root and not target.startswith(root + os.sep):
        return None
    return Path(target)


# ==========================================================================
# request handler
# ==========================================================================


class Handler(BaseHTTPRequestHandler):
    server_version = "ChessRL"
    sys_version = ""
    protocol_version = "HTTP/1.1"
    timeout = 60  # drop idle keep-alive connections

    # -- plumbing ---------------------------------------------------------
    def log_message(self, fmt, *args):  # silence the default logger
        pass

    def log_error(self, fmt, *args):
        pass

    def handle_one_request(self):
        try:
            super().handle_one_request()
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            self.close_connection = True
        except OSError as exc:
            if exc.errno not in (errno.EPIPE, errno.ECONNRESET, errno.ENOTCONN):
                raise
            self.close_connection = True

    def finish(self):
        try:
            super().finish()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

    def do_GET(self):
        self._run("GET")

    def do_HEAD(self):
        self._run("HEAD")

    def do_POST(self):
        self._run("POST")

    def _run(self, method):
        started = time.monotonic()
        self._head_only = method == "HEAD"
        self._status = 500
        self._nbytes = 0
        try:
            split = urllib.parse.urlsplit(self.path)
            path = urllib.parse.unquote(split.path)
            query = urllib.parse.parse_qs(split.query, keep_blank_values=True)
            if path == "/api" or path.startswith("/api/"):
                self._api(method, path, query)
            elif method in ("GET", "HEAD"):
                self._static(path)
            else:
                raise HttpError(405, "method not allowed")
        except HttpError as exc:
            self._send_json(exc.code, exc.payload())
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True
            self._status = 499
        except engine.ChessRLError as exc:
            self._send_json(500, {"error": str(exc).splitlines()[0]})
        except Exception:
            traceback.print_exc(file=sys.stderr)
            try:
                self._send_json(500, {"error": "internal server error"})
            except Exception:
                self.close_connection = True
        finally:
            ms = (time.monotonic() - started) * 1000.0
            client = self.client_address[0] if self.client_address else "-"
            where = self.path[:200].replace("\n", " ").replace("\r", " ")
            log(
                "%s %s %s %s -> %d %dB %.1fms"
                % (
                    time.strftime("%H:%M:%S"),
                    client,
                    method,
                    where,
                    self._status,
                    self._nbytes,
                    ms,
                )
            )

    # -- responses --------------------------------------------------------
    def _send(self, code, body, content_type, extra_headers=()):
        self._status = code
        self._nbytes = len(body)
        try:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
            self.send_header("X-Content-Type-Options", "nosniff")
            for key, value in extra_headers:
                self.send_header(key, value)
            self.end_headers()
            if not self._head_only and body:
                self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True
        except OSError as exc:
            if exc.errno not in (errno.EPIPE, errno.ECONNRESET, errno.ENOTCONN):
                raise
            self.close_connection = True

    def _send_json(self, code, obj):
        body = json.dumps(
            _json_safe(obj), separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        self._send(code, body, "application/json; charset=utf-8")

    # -- static -----------------------------------------------------------
    def _static(self, path):
        target = _safe_web_path(path)
        if target is None:
            raise HttpError(403, "forbidden")
        if target.is_dir():
            target = target / "index.html"
        if not target.is_file():
            if path in ("/", "", "/index.html"):
                self._send(200, _PLACEHOLDER, "text/html; charset=utf-8")
                return
            raise HttpError(404, "not found")
        try:
            body = target.read_bytes()
        except OSError as exc:
            raise HttpError(500, "could not read %s: %s" % (target.name, exc.strerror)) from None
        self._send(200, body, _mime_for(target))

    # -- request bodies ---------------------------------------------------
    def _read_json(self):
        raw_len = self.headers.get("Content-Length")
        if raw_len is None:
            if (self.headers.get("Transfer-Encoding") or "").lower() == "chunked":
                self.close_connection = True
                raise HttpError(411, "chunked bodies are not supported")
            return {}
        try:
            length = int(raw_len)
        except (TypeError, ValueError):
            self.close_connection = True
            raise HttpError(400, "bad Content-Length") from None
        if length < 0:
            self.close_connection = True
            raise HttpError(400, "bad Content-Length")
        if length > MAX_BODY:
            self.close_connection = True
            raise HttpError(413, "request body too large")
        data = self.rfile.read(length) if length else b""
        if not data.strip():
            return {}
        try:
            body = json.loads(data.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            raise HttpError(400, "malformed JSON body") from None
        if not isinstance(body, dict):
            raise HttpError(400, "body must be a JSON object")
        return body

    # -- API dispatch -----------------------------------------------------
    def _api(self, method, path, query):
        route = path[len("/api"):].strip("/").lower()
        if method == "HEAD" and route in ("new", "watch"):
            # HEAD must not create games or burn CPU on a full agent-vs-agent game.
            raise HttpError(405, "method not allowed")
        if method in ("GET", "HEAD"):
            if route == "new":
                return self._new_game(self._one(query, "fen"), self._one(query, "human"))
            if route == "state":
                return self._api_state(query)
            if route == "hint":
                return self._think(
                    self._gid_from_query(query),
                    self._one(query, "depth"),
                    self._one(query, "movetime"),
                    play=False,
                )
            if route == "watch":
                return self._api_watch(query)
            if route == "model":
                return self._api_model()
            if route == "report":
                return self._api_report()
            raise HttpError(404, "unknown endpoint")
        if method == "POST":
            body = self._read_json()
            if route == "move":
                return self._api_move(body)
            if route == "ai":
                check_keys(body, {"gid", "depth", "movetime"}, "/api/ai")
                return self._think(
                    want_int(body.get("gid"), "gid", 0, 1 << 30),
                    body.get("depth"),
                    body.get("movetime"),
                    play=True,
                )
            if route == "undo":
                return self._api_undo(body)
            if route == "hint":
                check_keys(body, {"gid", "depth", "movetime"}, "/api/hint")
                return self._think(
                    want_int(body.get("gid"), "gid", 0, 1 << 30),
                    body.get("depth"),
                    body.get("movetime"),
                    play=False,
                )
            if route == "new":
                check_keys(body, {"fen", "human"}, "/api/new")
                return self._new_game(body.get("fen"), body.get("human"))
            raise HttpError(404, "unknown endpoint")
        raise HttpError(405, "method not allowed")

    # -- helpers ----------------------------------------------------------
    @staticmethod
    def _one(query, name, default=None):
        values = query.get(name)
        if not values:
            return default
        return values[0]

    def _gid_from_query(self, query):
        return want_int(self._one(query, "gid"), "gid", 0, 1 << 30)

    def _entry(self, gid):
        return self.server.games.get(gid)

    # -- endpoints --------------------------------------------------------
    def _new_game(self, fen_arg, human_arg):
        fen = want_fen(fen_arg)
        if human_arg is None:
            human_arg = "white"
        if not isinstance(human_arg, str):
            raise HttpError(400, "human must be a string")
        human = human_arg.strip().lower() or "white"
        if human not in ("white", "black", "both", "none"):
            raise HttpError(400, "human must be white or black")
        try:
            entry = self.server.games.create(fen, human)
        except ValueError as exc:
            raise HttpError(400, str(exc)) from None
        except engine.LibraryNotFoundError:
            raise HttpError(503, LIB_MISSING) from None
        with entry.lock:
            state = entry.state()
        self._send_json(200, {"gid": entry.gid, "human": entry.human, "state": state})

    def _api_state(self, query):
        entry = self._entry(self._gid_from_query(query))
        with entry.lock:
            state = entry.state()
        self._send_json(200, state)

    def _api_move(self, body):
        check_keys(body, {"gid", "move"}, "/api/move")
        gid = want_int(body.get("gid"), "gid", 0, 1 << 30)
        uci = want_uci(body.get("move"))
        entry = self._entry(gid)
        with entry.lock:
            before = entry.state()
            if uci not in before["legal"]:
                raise HttpError(400, "illegal move")
            san = _san_for(before, uci)
            if not entry.handle.move(uci):
                raise HttpError(400, "illegal move")
            entry.history_san.append(san)
            state = entry.state()
        self._send_json(200, {"state": state})

    def _api_undo(self, body):
        check_keys(body, {"gid", "plies"}, "/api/undo")
        gid = want_int(body.get("gid"), "gid", 0, 1 << 30)
        plies = want_int(body.get("plies"), "plies", 1, MAX_UNDO_PLIES, default=1, clamp=True)
        entry = self._entry(gid)
        with entry.lock:
            for _ in range(plies):
                if not entry.handle.undo():
                    break
                if entry.history_san:
                    entry.history_san.pop()
            state = entry.state()
        self._send_json(200, {"state": state})

    def _think(self, gid, depth_arg, movetime_arg, play):
        depth = want_int(
            depth_arg, "depth", MIN_DEPTH, MAX_DEPTH, default=DEFAULT_DEPTH, clamp=True
        )
        movetime = want_int(
            movetime_arg,
            "movetime",
            MIN_MOVETIME,
            MAX_MOVETIME,
            default=DEFAULT_MOVETIME,
            clamp=True,
        )
        entry = self._entry(gid)
        with entry.lock:
            before = entry.state()
            if before["result"] != 0 or not before["legal"]:
                raise HttpError(400, "game is over")
            with self.server.engines.champion() as eng:
                raw = eng.move(entry.handle, depth, movetime)
            reply = normalise_engine_reply(raw, before)
            if not reply["move"]:
                raise HttpError(500, "engine returned no move")
            if reply["move"] not in before["legal"]:
                raise HttpError(500, "engine returned an illegal move: %s" % reply["move"])
            if play:
                if not entry.handle.move(reply["move"]):
                    raise HttpError(500, "engine move was rejected")
                entry.history_san.append(reply["san"])
                state = entry.state()
            else:
                state = before
        reply["state"] = state
        self._send_json(200, reply)

    def _api_watch(self, query):
        agent_a = want_int(self._one(query, "a", "-1"), "a", -1, 1 << 20, default=-1)
        agent_b = want_int(self._one(query, "b", "-1"), "b", -1, 1 << 20, default=-1)
        plies = want_int(
            self._one(query, "plies"),
            "plies",
            1,
            WATCH_MAX_PLIES,
            default=WATCH_DEFAULT_PLIES,
            clamp=True,
        )
        depth = want_int(self._one(query, "depth"), "depth", MIN_DEPTH, MAX_DEPTH, 2, clamp=True)
        movetime = want_int(
            self._one(query, "movetime"), "movetime", MIN_MOVETIME, MAX_MOVETIME, 60, clamp=True
        )
        fen = want_fen(self._one(query, "fen"))

        hub = self.server.engines
        hub.require()
        if not self.server.watch_slots.acquire(timeout=20):
            raise HttpError(503, "too many watch games in flight, try again")
        try:
            payload = self._play_watch(hub, agent_a, agent_b, plies, depth, movetime, fen)
        finally:
            self.server.watch_slots.release()
        self._send_json(200, payload)

    def _play_watch(self, hub, agent_a, agent_b, plies, depth, movetime, fen):
        deadline = time.monotonic() + WATCH_BUDGET_S
        moves, sans, fens, evals = [], [], [], []
        result, reason = 0, ""
        try:
            game = engine.GameHandle(fen)
        except ValueError as exc:
            raise HttpError(400, str(exc)) from None
        try:
            with hub.agent(agent_a) as eng_w, hub.agent(agent_b) as eng_b:
                for _ in range(plies):
                    state = normalise_state(game.state(), 0)
                    fens.append(state["fen"])
                    result, reason = state["result"], state["reason"]
                    if result != 0 or not state["legal"]:
                        break
                    if time.monotonic() > deadline:
                        reason = "move limit"
                        break
                    eng = eng_w if state["turn"] == "white" else eng_b
                    reply = normalise_engine_reply(eng.move(game, depth, movetime), state)
                    uci = reply["move"]
                    if not uci or uci not in state["legal"]:
                        reason = "adjudicated"
                        break
                    value = reply["value"]
                    evals.append(round(value if state["turn"] == "white" else -value, 4))
                    moves.append(uci)
                    sans.append(reply["san"])
                    if not game.move(uci):
                        moves.pop()
                        sans.pop()
                        evals.pop()
                        reason = "adjudicated"
                        break
                else:
                    final = normalise_state(game.state(), 0)
                    fens.append(final["fen"])
                    result, reason = final["result"], final["reason"] or "move limit"
        finally:
            game.close()
        if len(fens) < len(moves) + 1 and fens:
            fens.append(fens[-1])
        return {
            "result": result,
            "reason": reason,
            "moves": moves,
            "sans": sans,
            "fens": fens,
            "evals": evals,
            "white": self._agent_label(agent_a),
            "black": self._agent_label(agent_b),
        }

    @staticmethod
    def _agent_label(index):
        return "champion" if index < 0 else "agent %d" % index

    def _api_model(self):
        hub = self.server.engines
        info = hub.info()
        payload = {"path": hub.model_path}
        if isinstance(info, dict):
            payload.update(info)
        elif isinstance(info, list):
            payload["agents"] = info
        payload["path"] = hub.model_path
        agents = payload.get("agents")
        if not isinstance(agents, list):
            agents = []
        clean = []
        for i, item in enumerate(agents):
            if isinstance(item, dict):
                entry = dict(item)
                if not isinstance(entry.get("i"), int) or isinstance(entry.get("i"), bool):
                    entry["i"] = i
                clean.append(entry)
        payload["agents"] = clean
        n_agents = payload.get("n_agents")
        if not isinstance(n_agents, int) or isinstance(n_agents, bool) or n_agents < 0:
            payload["n_agents"] = len(clean)
        generation = payload.get("generation")
        if not isinstance(generation, int) or isinstance(generation, bool):
            payload["generation"] = int(_num(generation))
        self._send_json(200, payload)

    def _api_report(self):
        path = self.server.report_path()
        if path is not None:
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                self._send_json(200, data)
                return
            except (OSError, ValueError) as exc:
                log("report: could not read %s (%s)" % (path, exc))
        self._send_json(200, {})


# ==========================================================================
# server
# ==========================================================================


class ChessServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = 64

    def __init__(self, addr, model_path):
        super().__init__(addr, Handler)
        self.games = GameStore(MAX_GAMES)
        self.engines = EngineHub(model_path)
        self.watch_slots = threading.BoundedSemaphore(WATCH_CONCURRENCY)

    def report_path(self):
        if self.engines.model_path:
            candidate = Path(self.engines.model_path).resolve().parent / "report.json"
            if candidate.is_file():
                return candidate
            return None
        newest, newest_mtime = None, -1.0
        try:
            for path in RUNS_DIR.glob("*/report.json"):
                try:
                    mtime = path.stat().st_mtime
                except OSError:
                    continue
                if mtime > newest_mtime:
                    newest, newest_mtime = path, mtime
        except OSError:
            return None
        return newest

    def handle_error(self, request, client_address):
        exc = sys.exc_info()[1]
        if isinstance(exc, (BrokenPipeError, ConnectionResetError)):
            return
        if isinstance(exc, OSError) and exc.errno in (
            errno.EPIPE,
            errno.ECONNRESET,
            errno.ENOTCONN,
        ):
            return
        traceback.print_exc(file=sys.stderr)

    def shutdown_all(self):
        try:
            self.games.close_all()
        finally:
            self.engines.close()


def parse_args(argv):
    parser = argparse.ArgumentParser(
        prog="server.py",
        description="Serve the ChessRL web UI and JSON API (standard library only).",
    )
    parser.add_argument(
        "--model",
        metavar="PATH",
        default=None,
        help="model to play with (default: the newest runs/*/best.crl)",
    )
    parser.add_argument("--port", type=int, default=8000, help="TCP port (default 8000)")
    parser.add_argument(
        "--host", default="127.0.0.1", help="bind address (default 127.0.0.1)"
    )
    parser.add_argument(
        "--open", action="store_true", help="open the UI in a web browser once serving"
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)

    if args.model:
        model = Path(args.model).expanduser()
        if not model.is_file():
            sys.stderr.write("error: model not found: %s\n" % model)
            return 2
        model = model.resolve()
    else:
        model = newest_model()

    if not (0 < args.port < 65536):
        sys.stderr.write("error: --port must be 1..65535\n")
        return 2

    try:
        httpd = ChessServer((args.host, args.port), model)
    except OSError as exc:
        sys.stderr.write(
            "error: cannot bind %s:%d (%s)\n" % (args.host, args.port, exc.strerror or exc)
        )
        return 2

    host = args.host
    shown = "127.0.0.1" if host in ("0.0.0.0", "", "::") else host
    if ":" in shown and not shown.startswith("["):
        shown = "[%s]" % shown
    url = "http://%s:%d/" % (shown, httpd.server_address[1])

    log("ChessRL server on %s" % url)
    if model:
        log("  model : %s" % model)
    else:
        log("  model : none found under %s -- AI endpoints will answer 503" % RUNS_DIR)
    if not engine.library_available():
        log("  lib   : build/libchessrl.dylib missing -- run 'make -j lib'")
    else:
        log("  lib   : %s" % engine.library_path())
    missing_ui = "" if (WEB_DIR / "index.html").is_file() else " (no index.html yet)"
    log("  web   : %s%s" % (WEB_DIR, missing_ui))

    if args.open:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()

    try:
        httpd.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        log("shutting down")
    finally:
        with contextlib.suppress(Exception):
            httpd.shutdown_all()
        with contextlib.suppress(Exception):
            httpd.server_close()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except BrokenPipeError:
        sys.exit(0)
