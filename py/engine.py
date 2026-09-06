#!/usr/bin/env python3
"""ctypes bindings for ``build/libchessrl.dylib`` (the flat C ABI in src/api.h).

The whole surface of ``src/api.h`` is exposed and nothing else:

    api_init
    api_game_new / api_game_free / api_game_legal / api_game_move /
    api_game_undo / api_game_state
    api_engine_load / api_engine_free / api_engine_move / api_engine_policy /
    api_model_info

Buffer protocol
---------------
Every ``api_*`` function that fills a ``char *`` buffer returns the number of
bytes written, or a NEGATIVE value whose magnitude is the buffer size that would
have been required.  The helpers below allocate, grow and retry accordingly, and
are additionally tolerant of a C side that returns something other than a byte
count (``api_model_info`` is documented as returning the number of agents): the
buffer is decoded both by the reported length and as a NUL-terminated string, and
whichever yields valid JSON wins.

Threading
---------
The C library is thread-safe with respect to *distinct* handles, but one handle
must never be touched by two threads at once, so every handle owns a
``threading.Lock``.  Calls that touch two handles (``Engine.move``,
``Engine.policy``) take both locks in a fixed global order so they can never
deadlock against each other.
"""

from __future__ import annotations

import contextlib
import ctypes
import itertools
import json
import os
import threading
from pathlib import Path

__all__ = [
    "ChessRLError",
    "LibraryNotFoundError",
    "Engine",
    "GameHandle",
    "model_info",
    "load_library",
    "library_path",
    "library_available",
    "REPO_ROOT",
]

# --------------------------------------------------------------------------
# locations
# --------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
BUILD_DIR = REPO_ROOT / "build"
_LIB_NAMES = ("libchessrl.dylib", "libchessrl.so")

_MAX_BUF = 8 << 20  # never grow a reply buffer past 8 MiB
_MISS = object()


class ChessRLError(RuntimeError):
    """Any failure coming out of the C library or its bindings."""


class LibraryNotFoundError(ChessRLError):
    """libchessrl has not been built yet."""


def _candidate_paths():
    env = os.environ.get("CHESSRL_LIB")
    if env:
        yield Path(env).expanduser()
    for name in _LIB_NAMES:
        yield BUILD_DIR / name


def library_path():
    """Absolute path of the shared library, or ``None`` if it is not built."""
    for p in _candidate_paths():
        try:
            if p.is_file():
                return p.resolve()
        except OSError:
            continue
    return None


def library_available():
    return library_path() is not None


def _not_found_error():
    looked = "\n".join("    %s" % p for p in _candidate_paths())
    return LibraryNotFoundError(
        "libchessrl.dylib was not found.\n"
        "Looked in:\n%s\n\n"
        "Build it first:\n"
        "    cd %s && make -j lib\n"
        "(or set CHESSRL_LIB to the path of an existing build)." % (looked, REPO_ROOT)
    )


# --------------------------------------------------------------------------
# library loading / prototypes
# --------------------------------------------------------------------------

_c_int = ctypes.c_int
_c_str = ctypes.c_char_p
_c_buf = ctypes.POINTER(ctypes.c_char)

#: name -> (argtypes, restype)
_SIGNATURES = {
    "api_init": ((), _c_int),
    "api_game_new": ((_c_str,), _c_int),
    "api_game_free": ((_c_int,), None),
    "api_game_legal": ((_c_int, _c_buf, _c_int), _c_int),
    "api_game_move": ((_c_int, _c_str), _c_int),
    "api_game_undo": ((_c_int,), _c_int),
    "api_game_state": ((_c_int, _c_buf, _c_int), _c_int),
    "api_engine_load": ((_c_str, _c_int), _c_int),
    "api_engine_free": ((_c_int,), None),
    "api_engine_move": ((_c_int, _c_int, _c_int, _c_int, _c_buf, _c_int), _c_int),
    "api_engine_policy": ((_c_int, _c_int, _c_buf, _c_int), _c_int),
    "api_model_info": ((_c_str, _c_buf, _c_int), _c_int),
}

_lib = None
_lib_lock = threading.Lock()


def load_library():
    """Load (once) and return the ``ctypes.CDLL`` for libchessrl."""
    global _lib
    lib = _lib
    if lib is not None:
        return lib
    with _lib_lock:
        if _lib is not None:
            return _lib
        path = library_path()
        if path is None:
            raise _not_found_error()
        try:
            lib = ctypes.CDLL(str(path))
        except OSError as exc:
            raise ChessRLError(
                "failed to load %s: %s\nTry rebuilding it: cd %s && make -j lib"
                % (path, exc, REPO_ROOT)
            ) from exc
        for name, (argtypes, restype) in _SIGNATURES.items():
            try:
                fn = getattr(lib, name)
            except AttributeError as exc:
                raise ChessRLError(
                    "%s does not export %s -- the library is stale, rebuild it: "
                    "cd %s && make -j lib" % (path, name, REPO_ROOT)
                ) from exc
            fn.argtypes = list(argtypes)
            fn.restype = restype
        if int(lib.api_init()) != 1:
            raise ChessRLError("api_init() failed")
        lib.chessrl_path = str(path)
        _lib = lib
        return lib


# --------------------------------------------------------------------------
# buffer plumbing
# --------------------------------------------------------------------------


def _decode(raw, nwritten, capacity):
    """Return the payload bytes described by ``nwritten`` plus the NUL-terminated
    reading of the same buffer (the two differ when the C side returns something
    that is not a byte count)."""
    out = []
    if 0 < nwritten <= capacity:
        out.append(raw[:nwritten])
    zero = raw.find(b"\x00")
    out.append(raw[:zero] if zero >= 0 else raw)
    seen = set()
    uniq = []
    for chunk in out:
        chunk = chunk.strip()
        if chunk and chunk not in seen:
            seen.add(chunk)
            uniq.append(chunk)
    return uniq


def _too_small(need, capacity, label):
    if need <= capacity:
        # A negative return whose magnitude already fits means the call failed
        # rather than ran out of room.
        raise ChessRLError("%s failed (returned %d)" % (label, -need))
    if need + 1 > _MAX_BUF:
        raise ChessRLError("%s wants a %d byte buffer (limit %d)" % (label, need, _MAX_BUF))
    return need + 1


def _call_json(fn, args, label, initial=16384):
    """Invoke ``fn(*args, buf, len)`` and return the parsed JSON payload."""
    cap = max(256, int(initial))
    last = b""
    for _ in range(10):
        buf = ctypes.create_string_buffer(cap)
        rc = int(fn(*args, buf, cap))
        if rc < 0:
            cap = _too_small(-rc, cap, label)
            continue
        raw = buf.raw
        for chunk in _decode(raw, rc, cap):
            last = chunk
            try:
                return json.loads(chunk.decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                continue
        if cap >= _MAX_BUF:
            break
        cap = min(_MAX_BUF, cap * 4)
    raise ChessRLError(
        "%s did not produce valid JSON (%d bytes: %r)"
        % (label, len(last), last[:200])
    )


def _call_text(fn, args, label, initial=8192):
    """Invoke ``fn(*args, buf, len)`` and return the payload as ``str``."""
    cap = max(256, int(initial))
    for _ in range(10):
        buf = ctypes.create_string_buffer(cap)
        rc = int(fn(*args, buf, cap))
        if rc < 0:
            cap = _too_small(-rc, cap, label)
            continue
        if rc == 0:
            return ""
        raw = buf.raw
        chunk = raw[:rc] if rc <= cap else raw
        zero = chunk.find(b"\x00")
        if zero >= 0:
            chunk = chunk[:zero]
        return chunk.decode("utf-8", "replace")
    raise ChessRLError("%s: buffer never large enough" % label)


# --------------------------------------------------------------------------
# handle locking
# --------------------------------------------------------------------------

_ORDER = itertools.count()


class _Handle:
    """Common bookkeeping: an id, a lock, a stable lock-ordering rank."""

    __slots__ = ("_lock", "_order", "_closed")

    def __init__(self):
        self._lock = threading.Lock()
        self._order = next(_ORDER)
        self._closed = False

    @property
    def closed(self):
        return self._closed


@contextlib.contextmanager
def _locked(*handles):
    """Acquire every handle's lock in a global order (deadlock-free)."""
    ordered = sorted({id(h): h for h in handles}.values(), key=lambda h: h._order)
    held = []
    try:
        for h in ordered:
            h._lock.acquire()
            held.append(h)
        yield
    finally:
        for h in reversed(held):
            h._lock.release()


# --------------------------------------------------------------------------
# games
# --------------------------------------------------------------------------


class GameHandle(_Handle):
    """One game living inside the C library.

    Usable as a context manager::

        with GameHandle() as g:
            g.move("e2e4")
            print(g.state()["fen"])
    """

    __slots__ = ("_lib", "gid", "fen")

    def __init__(self, fen=None):
        super().__init__()
        self._lib = load_library()
        self.fen = fen or None
        arg = self.fen.encode("utf-8") if self.fen else None
        gid = int(self._lib.api_game_new(arg))
        if gid < 0:
            if self.fen:
                raise ValueError("invalid FEN: %r" % (self.fen,))
            raise ChessRLError("api_game_new failed")
        self.gid = gid

    # -- lifetime ---------------------------------------------------------
    def close(self):
        """Free the C-side game.  Idempotent."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            gid, self.gid = self.gid, -1
        try:
            self._lib.api_game_free(gid)
        except Exception:  # pragma: no cover - interpreter teardown
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def __del__(self):  # pragma: no cover - best effort
        try:
            self.close()
        except Exception:
            pass

    def _check(self):
        if self._closed:
            raise ChessRLError("game handle is closed")

    def __repr__(self):
        return "<GameHandle gid=%d%s>" % (self.gid, " closed" if self._closed else "")

    # -- queries ----------------------------------------------------------
    def state(self):
        """Parsed ``api_game_state`` JSON."""
        with self._lock:
            self._check()
            return _call_json(self._lib.api_game_state, (self.gid,), "api_game_state")

    def legal(self):
        """List of legal moves in UCI (``api_game_legal``)."""
        with self._lock:
            self._check()
            text = _call_text(self._lib.api_game_legal, (self.gid,), "api_game_legal")
        return text.split()

    # -- mutation ---------------------------------------------------------
    def move(self, uci):
        """Play ``uci``.  Returns True, or False if the move is not legal."""
        if not isinstance(uci, str) or not uci:
            raise ValueError("move must be a non-empty UCI string")
        arg = uci.encode("ascii", "strict")
        with self._lock:
            self._check()
            return int(self._lib.api_game_move(self.gid, arg)) == 1

    def undo(self):
        """Take back one ply.  Returns True, or False if there was nothing to undo."""
        with self._lock:
            self._check()
            return int(self._lib.api_game_undo(self.gid)) == 1


# --------------------------------------------------------------------------
# engines
# --------------------------------------------------------------------------


class Engine(_Handle):
    """A loaded model + agent, able to pick moves for a :class:`GameHandle`."""

    __slots__ = ("_lib", "eid", "model_path", "agent_index")

    def __init__(self, model_path, agent_index=-1):
        super().__init__()
        self._lib = load_library()
        path = Path(model_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError("model file not found: %s" % path)
        self.model_path = str(path)
        self.agent_index = int(agent_index)
        eid = int(self._lib.api_engine_load(self.model_path.encode("utf-8"), self.agent_index))
        if eid < 0:
            raise ChessRLError(
                "api_engine_load failed for %s (agent %d)" % (self.model_path, self.agent_index)
            )
        self.eid = eid

    # -- lifetime ---------------------------------------------------------
    def close(self):
        """Free the C-side engine.  Idempotent."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            eid, self.eid = self.eid, -1
        try:
            self._lib.api_engine_free(eid)
        except Exception:  # pragma: no cover - interpreter teardown
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def __del__(self):  # pragma: no cover - best effort
        try:
            self.close()
        except Exception:
            pass

    def _check(self):
        if self._closed:
            raise ChessRLError("engine handle is closed")

    def __repr__(self):
        return "<Engine eid=%d agent=%d %s>" % (self.eid, self.agent_index, self.model_path)

    # -- play -------------------------------------------------------------
    def move(self, game, depth=4, movetime_ms=800):
        """Pick (but do not play) a move for ``game``'s current position.

        Returns the ``{move, san, score, depth, nodes, ms, value, top[]}`` dict
        produced by ``api_engine_move``.
        """
        if not isinstance(game, GameHandle):
            raise TypeError("game must be a GameHandle")
        depth = int(depth)
        movetime_ms = int(movetime_ms)
        with _locked(self, game):
            self._check()
            game._check()
            return _call_json(
                self._lib.api_engine_move,
                (self.eid, game.gid, depth, movetime_ms),
                "api_engine_move",
            )

    def policy(self, game):
        """Raw policy for ``game``'s current position: ``[{move,san,prob,logit}...]``."""
        if not isinstance(game, GameHandle):
            raise TypeError("game must be a GameHandle")
        with _locked(self, game):
            self._check()
            game._check()
            return _call_json(
                self._lib.api_engine_policy,
                (self.eid, game.gid),
                "api_engine_policy",
            )


# --------------------------------------------------------------------------
# model metadata
# --------------------------------------------------------------------------


def model_info(path):
    """Metadata about the agents stored in the model file at ``path``."""
    lib = load_library()
    p = Path(path).expanduser()
    if not p.is_file():
        raise FileNotFoundError("model file not found: %s" % p)
    return _call_json(
        lib.api_model_info, (str(p).encode("utf-8"),), "api_model_info", initial=1 << 16
    )


# --------------------------------------------------------------------------
# tiny self-check
# --------------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover
    import sys

    lib_at = library_path()
    if lib_at is None:
        sys.stderr.write(str(_not_found_error()) + "\n")
        raise SystemExit(1)
    load_library()
    print("library: %s" % lib_at)
    with GameHandle() as g:
        st = g.state()
        print("fen    : %s" % st.get("fen"))
        print("legal  : %d moves" % len(g.legal()))
        g.move("e2e4")
        print("after e2e4: %s" % g.state().get("fen"))
    for arg in sys.argv[1:]:
        print("model %s:" % arg)
        print(json.dumps(model_info(arg), indent=2)[:2000])
