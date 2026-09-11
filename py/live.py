#!/usr/bin/env python3
"""ChessRL live training dashboard -- Python 3 standard library only.

Serves ``web/live.html`` and a small JSON API that tails
``runs/NAME/telemetry.jsonl`` while ``chessrl az --run NAME`` is writing it.

    python3 py/live.py --run runs/az_main --port 8100 --open

Endpoints
---------
``GET /``                       the dashboard page
``GET /api/runs``               every discovered run, with generation count,
                                mtime and whether it looks live
``GET /api/telemetry?run=ID&since=N``
                                only the generations after ``N``; also the
                                total, a token that changes whenever the file
                                is replaced or truncated, and the liveness of
                                the run

Tailing keeps a byte offset per file and reads only what was appended.  A
trailing line without its newline (the trainer mid-write) is held back and
retried on the next poll, so a half-written record is never parsed and never
reaches the client.  A missing run directory, a file created later, a file
replaced by a fresh run and a truncated file are all handled by resetting the
tail and bumping the token, which tells the page to resync.

Zero dependencies: standard library only.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = REPO_ROOT / "web"
RUNS_DIR = REPO_ROOT / "runs"
PAGE = WEB_DIR / "live.html"

TELEMETRY_NAME = "telemetry.jsonl"

LIVE_WINDOW_S = 60.0        # mtime newer than this => the trainer looks alive
MAX_ROWS = 20000            # generations kept in memory per run
MAX_BATCH = 2000            # generations returned by one /api/telemetry call
MAX_READ = 16 << 20         # bytes consumed from one file in one poll
MAX_PARTIAL = 4 << 20       # an unterminated tail longer than this is garbage
PROC_TTL = 8.0              # seconds to cache the `ps` scan
COUNT_CACHE_MAX = 256


def _null(_token: str):
    """json.loads hook: NaN / Infinity become null rather than invalid JSON."""
    return None


# ---------------------------------------------------------------- run tailer


class RunTail:
    """Incremental reader for one telemetry.jsonl."""

    def __init__(self, rid: str, directory: Path, file: Path) -> None:
        self.id = rid
        self.dir = directory
        self.file = file
        self.lock = threading.Lock()
        self.rows: list[dict] = []
        self.base = 0           # global index of rows[0]
        self.read_pos = 0       # bytes consumed from the file
        self.partial = b""      # bytes read but not newline-terminated yet
        self.sig: tuple | None = None
        self.token = 1          # bumped on truncate / replace / disappearance
        self.skipped = 0        # newline-terminated lines that would not parse
        self.exists = False
        self.mtime = 0.0
        self.size = 0
        self.active = False     # set once a client has asked for its rows

    # -- internals ---------------------------------------------------------

    def _reset(self) -> None:
        self.rows = []
        self.base = 0
        self.read_pos = 0
        self.partial = b""
        self.skipped = 0
        self.token += 1

    def _ingest(self, blob: bytes) -> None:
        for raw in blob.split(b"\n"):
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw.decode("utf-8", "replace"),
                                 parse_constant=_null)
            except ValueError:
                self.skipped += 1
                continue
            if not isinstance(rec, dict):
                self.skipped += 1
                continue
            self.rows.append(rec)
        if len(self.rows) > MAX_ROWS:
            drop = len(self.rows) - MAX_ROWS
            del self.rows[:drop]
            self.base += drop

    def _poll_locked(self) -> None:
        try:
            st = os.stat(self.file)
        except OSError:
            # Not written yet, or the run directory is gone again.
            if self.sig is not None:
                self._reset()
                self.sig = None
            self.exists = False
            self.mtime = 0.0
            self.size = 0
            return

        self.exists = True
        self.mtime = st.st_mtime
        self.size = st.st_size
        sig = (st.st_dev, st.st_ino)

        if self.sig is None:
            self.sig = sig
        elif sig != self.sig:            # a new run wrote a new file here
            self._reset()
            self.sig = sig
        if st.st_size < self.read_pos:   # truncated underneath us
            self._reset()
        if st.st_size == self.read_pos:
            return

        try:
            with open(self.file, "rb") as fh:
                fh.seek(self.read_pos)
                data = fh.read(MAX_READ)
        except OSError:
            return
        if not data:
            return
        self.read_pos += len(data)

        buf = self.partial + data
        self.partial = b""
        cut = buf.rfind(b"\n")
        if cut < 0:
            # Nothing complete yet: the trainer is mid-line.  Hold it.
            self.partial = buf if len(buf) <= MAX_PARTIAL else b""
            return
        self.partial = buf[cut + 1:]
        if len(self.partial) > MAX_PARTIAL:
            self.partial = b""
        self._ingest(buf[:cut])

    # -- public ------------------------------------------------------------

    @property
    def total(self) -> int:
        return self.base + len(self.rows)

    def _info(self, exists: bool, mtime: float, size: int,
              now: float | None = None) -> dict:
        now = time.time() if now is None else now
        age = (now - mtime) if exists else None
        return {
            "id": self.id,
            "name": self.dir.name or str(self.dir),
            "dir": str(self.dir),
            "file": str(self.file),
            "exists": exists,
            "size": size,
            "mtime": mtime if exists else None,
            "age": round(age, 3) if age is not None else None,
            "live": bool(exists and age is not None and age <= LIVE_WINDOW_S),
        }

    def info(self, now: float | None = None) -> dict:
        return self._info(self.exists, self.mtime, self.size, now)

    def peek(self, now: float | None = None) -> dict:
        """stat() only -- for listing runs nobody is watching."""
        try:
            st = os.stat(self.file)
            return self._info(True, st.st_mtime, st.st_size, now)
        except OSError:
            return self._info(False, 0.0, 0, now)

    def slice(self, since: int) -> dict:
        self.active = True
        with self.lock:
            self._poll_locked()
            total = self.total
            since = max(0, min(int(since), total))
            start = max(since - self.base, 0)
            rows = self.rows[start:start + MAX_BATCH]
            payload = {
                "ok": True,
                "run": self.info(),
                "token": "%d:%s" % (self.token, self.sig[1] if self.sig else 0),
                "base": self.base,
                "total": total,
                "since": since,
                "from": self.base + start,
                "returned": len(rows),
                "more": (start + len(rows)) < len(self.rows),
                "skipped": self.skipped,
                "pending": len(self.partial),
                "rows": rows,
            }
        return payload


# ------------------------------------------------------------- run registry


class Registry:
    """Discovers runs and owns one RunTail per run."""

    def __init__(self, pinned: Path | None) -> None:
        self.lock = threading.Lock()
        self.by_id: dict[str, RunTail] = {}
        self.order: list[str] = []
        self.paths: dict[Path, str] = {}
        self.counts: dict[Path, tuple] = {}
        self.pinned: str | None = None
        if pinned is not None:
            self.pinned = self.register(*split_run(pinned))

    # -- discovery ---------------------------------------------------------

    def register(self, directory: Path, file: Path) -> str:
        with self.lock:
            known = self.paths.get(file)
            if known:
                return known
            stem = directory.name or "run"
            rid = re.sub(r"[^A-Za-z0-9._-]", "-", stem) or "run"
            if rid in self.by_id:
                base, n = rid, 2
                while rid in self.by_id:
                    rid = "%s~%d" % (base, n)
                    n += 1
            tail = RunTail(rid, directory, file)
            self.by_id[rid] = tail
            self.order.append(rid)
            self.paths[file] = rid
            return rid

    def scan(self) -> list[str]:
        """Register every runs/*/telemetry.jsonl; return ids in display order."""
        try:
            entries = sorted(p for p in RUNS_DIR.iterdir() if p.is_dir())
        except OSError:
            entries = []
        for directory in entries:
            file = directory / TELEMETRY_NAME
            if file.exists():
                self.register(directory, file)
        with self.lock:
            ids = list(self.order)
        if self.pinned and self.pinned in ids:
            ids.remove(self.pinned)
            ids.insert(0, self.pinned)
        return ids

    def get(self, rid: str | None) -> RunTail | None:
        if not rid:
            rid = self.pinned
        if not rid:
            return None
        with self.lock:
            tail = self.by_id.get(rid)
        if tail is None:
            self.scan()
            with self.lock:
                tail = self.by_id.get(rid)
        return tail

    # -- listing -----------------------------------------------------------

    def count_lines(self, tail: RunTail) -> int:
        """Generation count without holding the whole file in memory."""
        try:
            st = os.stat(tail.file)
        except OSError:
            return 0
        key = (st.st_size, st.st_mtime_ns)
        hit = self.counts.get(tail.file)
        if hit and hit[0] == key:
            return hit[1]
        n = 0
        try:
            with open(tail.file, "rb") as fh:
                while True:
                    chunk = fh.read(1 << 20)
                    if not chunk:
                        break
                    n += chunk.count(b"\n")
        except OSError:
            return 0
        if len(self.counts) > COUNT_CACHE_MAX:
            self.counts.clear()
        self.counts[tail.file] = (key, n)
        return n

    def listing(self, procs: list[dict]) -> dict:
        now = time.time()
        out = []
        for rid in self.scan():
            tail = self.by_id[rid]
            if tail.active:
                with tail.lock:
                    tail._poll_locked()
                    info = tail.info(now)
                    info["gens"] = tail.total
            else:
                # Nobody is watching this one: stat it and count newlines
                # (cached on size+mtime) rather than parsing it into memory.
                info = tail.peek(now)
                info["gens"] = self.count_lines(tail) if info["exists"] else 0
            info["default"] = (rid == self.pinned)
            info["trainer"] = match_proc(procs, tail.dir)
            if info["trainer"]:
                info["live"] = True
            out.append(info)
        out.sort(key=lambda r: (not r["default"], not r["live"],
                                -(r["mtime"] or 0)))
        return {"ok": True, "runs": out, "default": self.pinned,
                "server_time": now}


def split_run(target: Path) -> tuple[Path, Path]:
    """Accept a run directory or a telemetry.jsonl path; return (dir, file)."""
    target = Path(target).expanduser()
    if not target.is_absolute():
        target = (REPO_ROOT / target)
    try:
        target = target.resolve()
    except OSError:                                   # pragma: no cover
        target = target.absolute()
    if target.suffix == ".jsonl":
        return target.parent, target
    return target, target / TELEMETRY_NAME


# ---------------------------------------------------------- trainer probing


_proc_cache: dict = {"t": 0.0, "procs": []}
_proc_lock = threading.Lock()


def scan_procs(enabled: bool) -> list[dict]:
    """Best-effort: find running `chessrl az` processes and their --gens.

    Only used to label a run as live and to give the page a target generation
    count for the ETA.  Any failure degrades to an empty list.
    """
    if not enabled:
        return []
    now = time.time()
    with _proc_lock:
        if now - _proc_cache["t"] < PROC_TTL:
            return _proc_cache["procs"]
    procs: list[dict] = []
    try:
        res = subprocess.run(["ps", "-Ao", "pid=,args="],
                             stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                             timeout=3)
        for line in res.stdout.decode("utf-8", "replace").splitlines():
            line = line.strip()
            if not line or "chessrl" not in line:
                continue
            parts = line.split()
            if len(parts) < 3 or not parts[0].isdigit():
                continue
            args = parts[1:]
            if "chessrl" not in args[0] or len(args) < 2 or args[1] != "az":
                continue
            entry = {"pid": int(parts[0]), "gens": None, "run": None}
            for i, a in enumerate(args):
                if a == "--gens" and i + 1 < len(args):
                    try:
                        entry["gens"] = int(args[i + 1])
                    except ValueError:
                        pass
                elif a == "--run" and i + 1 < len(args):
                    entry["run"] = args[i + 1]
            procs.append(entry)
    except Exception:
        procs = []
    with _proc_lock:
        _proc_cache["t"] = now
        _proc_cache["procs"] = procs
    return procs


def match_proc(procs: list[dict], directory: Path) -> dict | None:
    for p in procs:
        if not p.get("run"):
            continue
        try:
            cand, _ = split_run(Path(p["run"]))
        except Exception:                              # pragma: no cover
            continue
        if cand == directory:
            return {"pid": p["pid"], "gens": p["gens"]}
    return None


# --------------------------------------------------------------- http layer


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "ChessRLLive/1.0"
    registry: Registry
    proc_scan = True
    verbose = False

    # -- plumbing ----------------------------------------------------------

    def log_message(self, fmt, *args):
        if self.verbose:
            sys.stderr.write("%s %s\n" % (self.address_string(), fmt % args))

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        try:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store, must-revalidate")
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _json(self, obj, code: int = 200) -> None:
        try:
            body = json.dumps(obj, separators=(",", ":"),
                              allow_nan=False).encode("utf-8")
        except ValueError:
            body = json.dumps({"ok": False, "error": "unserialisable"}).encode()
            code = 500
        self._send(code, body, "application/json; charset=utf-8")

    def _error(self, code: int, msg: str) -> None:
        self._json({"ok": False, "error": msg}, code)

    # -- routes ------------------------------------------------------------

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        parsed = urllib.parse.urlsplit(self.path)
        path = urllib.parse.unquote(parsed.path)
        query = urllib.parse.parse_qs(parsed.query)

        if path in ("/", "/index.html", "/live.html"):
            return self._page()
        if path == "/favicon.ico":
            return self._send(200, FAVICON, "image/svg+xml")
        if path == "/api/runs":
            return self._json(
                self.registry.listing(scan_procs(self.proc_scan)))
        if path == "/api/telemetry":
            return self._telemetry(query)
        if path == "/api/health":
            return self._json({"ok": True, "server_time": time.time()})
        return self._error(404, "not found")

    def _page(self):
        try:
            body = PAGE.read_bytes()
        except OSError:
            return self._error(500, "web/live.html is missing")
        self._send(200, body, "text/html; charset=utf-8")

    def _telemetry(self, query):
        rid = (query.get("run") or [None])[0]
        tail = self.registry.get(rid)
        if tail is None:
            self.registry.scan()
            return self._json({"ok": False, "error": "unknown run",
                               "run": None, "total": 0, "rows": [],
                               "server_time": time.time()}, 404)
        try:
            since = int((query.get("since") or ["0"])[0])
        except ValueError:
            since = 0
        payload = tail.slice(since)
        payload["trainer"] = match_proc(scan_procs(self.proc_scan), tail.dir)
        if payload["trainer"]:
            payload["run"]["live"] = True
        payload["server_time"] = time.time()
        return self._json(payload)


FAVICON = (
    b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16">'
    b'<rect width="16" height="16" rx="3" fill="#3b5bdb"/>'
    b'<path d="M3 11 L6 6 L9 9 L13 3" stroke="#fff" stroke-width="1.6" '
    b'fill="none" stroke-linecap="round" stroke-linejoin="round"/></svg>'
)


# --------------------------------------------------------------------- main


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Live ChessRL training dashboard (standard library only).")
    ap.add_argument("--run", default=str(RUNS_DIR / "az_main"),
                    help="run directory (or telemetry.jsonl) to open first")
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--open", action="store_true",
                    help="open the dashboard in a browser")
    ap.add_argument("--no-proc-scan", action="store_true",
                    help="do not look for a running trainer in the process "
                         "table (disables the ETA target and the pid label)")
    ap.add_argument("--verbose", action="store_true", help="log every request")
    args = ap.parse_args(argv)

    if not PAGE.exists():
        sys.stderr.write("live: %s is missing\n" % PAGE)
        return 1

    registry = Registry(Path(args.run))
    registry.scan()

    Handler.registry = registry
    Handler.proc_scan = not args.no_proc_scan
    Handler.verbose = args.verbose

    class Server(ThreadingHTTPServer):
        daemon_threads = True
        allow_reuse_address = True

    try:
        httpd = Server((args.host, args.port), Handler)
    except OSError as exc:
        sys.stderr.write("live: cannot bind %s:%d (%s)\n"
                         % (args.host, args.port, exc))
        return 1

    url = "http://%s:%d/" % (args.host, args.port)
    tail = registry.get(None)
    sys.stderr.write("live: serving %s\n" % url)
    if tail is not None:
        sys.stderr.write("live: watching %s\n" % tail.file)
    sys.stderr.flush()
    if args.open:
        threading.Thread(target=webbrowser.open, args=(url,),
                         daemon=True).start()
    try:
        httpd.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        sys.stderr.write("\nlive: stopped\n")
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
