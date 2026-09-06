#!/usr/bin/env python3
"""report.py -- turn a finished ChessRL training run into a strategy analysis.

    python3 py/report.py runs/NAME [--out runs/NAME/report.json] [--md runs/NAME/REPORT.md]

Input is ``runs/NAME/telemetry.jsonl``: one JSON object per generation, as written
by the trainer (see docs/ALGORITHM.md).  Every field is optional -- anything that
is missing degrades into an explicit "not logged" instead of a crash, and any
claim the data cannot support is refused rather than invented.

Output
------
1. ``report.json`` -- machine readable, consumed by the web REPORT view
   (``GET /api/report``).  It carries every raw series the UI needs to draw its
   own charts; this script draws nothing.
2. ``REPORT.md``   -- a plain-text/table narrative for humans.

report.json schema  (schema == "chessrl.report/1")
--------------------------------------------------
{
  "schema": "chessrl.report/1",
  "generated_at": "2026-01-01T00:00:00Z",
  "run": {"name","dir","telemetry","generations","gen_first","gen_last",
          "lines_read","lines_bad","missing_fields":[...],"caveats":[...]},
  "squares": ["a1","b1",...,"h8"],          # index -> name, for the heatmaps
  "series": {"gen":[], "elo_best":[], "elo_mean":[], "elo_p10":[],
             "elo_spread":[], "draw_rate":[], "white_win":[], "black_win":[],
             "decisive_rate":[], "avg_len":[], "captures_per_game":[],
             "checks_per_game":[], "castle_rate":[], "promo_rate":[],
             "ep_rate":[], "avg_final_material":[], "gps":[], "sec":[],
             "games":[], "plies":[], "loss_policy":[], "loss_value":[],
             "loss_entropy":[], "grad_norm":[], "checkmate_share":[],
             "maxplies_share":[], "opening_top_share":[],
             "opening_coverage":[], "opening_entropy_bits":[],
             "piece_entropy_bits":[], "centre_share":[],
             "captures":[], "checks":[], "draw":[],   # aliases of the _per_game /
                                                      # _rate series above
             "term": {"checkmate":[], "stalemate":[], ...}}   # shares of games
  Every series has exactly one entry per generation, aligned with "gen", and a
  missing measurement is null rather than a dropped point.
  "learning":   {...},   # elo gain, peak, plateau test, smoothed curve
  "throughput": {...},   # games/sec, positions/sec, wall clock, speedup
  "phases":     {...},   # 4 generation bands, per-metric deltas + significance
  "openings":   {...},   # White's first-move distribution per band, in SAN
  "pieces":     {...},   # per-piece-type destination heatmaps + entropy
  "champion":   {...},   # PBT hypers of the best agent vs the population
  "findings":   [...],   # ranked "what actually worked", with the numbers
  "limitations":[...]    # "what it never learned"
}

Standard library only.  No numpy, no third-party anything.
"""

import argparse
import datetime
import json
import math
import os
import sys

SCHEMA = "chessrl.report/1"
HUMAN_GAME_SECONDS = 40 * 60.0          # a human game is assumed to be 40 minutes
PIECE_NAMES = ["pawn", "knight", "bishop", "rook", "queen", "king"]
PIECE_LETTER = {"pawn": "P", "knight": "N", "bishop": "B",
                "rook": "R", "queen": "Q", "king": "K"}
FILES = "abcdefgh"
SQUARES = [FILES[i & 7] + str((i >> 3) + 1) for i in range(64)]   # a1 = 0, h8 = 63
CENTRE4 = {27, 28, 35, 36}                                        # d4 e4 d5 e5
CENTRE16 = {r * 8 + f for r in range(2, 6) for f in range(2, 6)}  # c3 .. f6


# --------------------------------------------------------------------- maths

def _nums(xs):
    return [float(x) for x in xs if isinstance(x, (int, float)) and not _isnan(x)]


def _isnan(x):
    try:
        return math.isnan(float(x)) or math.isinf(float(x))
    except (TypeError, ValueError):
        return True


def mean(xs):
    v = _nums(xs)
    return sum(v) / len(v) if v else None


def sd(xs):
    """Sample standard deviation; None when fewer than two finite values."""
    v = _nums(xs)
    if len(v) < 2:
        return None
    m = sum(v) / len(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))


def median(xs):
    v = sorted(_nums(xs))
    if not v:
        return None
    n = len(v)
    return v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])


def lsq_slope(xs, ys):
    """Least-squares slope of ys against xs (0.0 when degenerate)."""
    pts = [(float(a), float(b)) for a, b in zip(xs, ys)
           if not _isnan(a) and not _isnan(b)]
    if len(pts) < 2:
        return 0.0
    mx = sum(p[0] for p in pts) / len(pts)
    my = sum(p[1] for p in pts) / len(pts)
    den = sum((p[0] - mx) ** 2 for p in pts)
    if den <= 0.0:
        return 0.0
    return sum((p[0] - mx) * (p[1] - my) for p in pts) / den


def entropy_bits(counts):
    """Shannon entropy (bits) of a count vector.  None when empty."""
    v = [float(c) for c in counts if isinstance(c, (int, float)) and c > 0]
    tot = sum(v)
    if tot <= 0:
        return None
    h = 0.0
    for c in v:
        p = c / tot
        h -= p * math.log(p, 2.0)
    return h


def safe_div(a, b):
    if a is None or b in (None, 0) or _isnan(a) or _isnan(b):
        return None
    return a / b


# ------------------------------------------------------------ field plumbing

def dig(obj, path):
    """Fetch a dotted path out of nested dicts; None if absent."""
    cur = obj
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def pick(row, *paths):
    """First present, finite value among several candidate paths."""
    for p in paths:
        v = dig(row, p)
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)) and not _isnan(v):
            return float(v)
    return None


def pick_any(row, *paths):
    """First present value of any type (dict / list / str)."""
    for p in paths:
        v = dig(row, p)
        if v is not None:
            return v
    return None


def sq_name(i):
    return SQUARES[i] if isinstance(i, int) and 0 <= i < 64 else "??"


# ------------------------------------------------------------- SAN for moves

def _build_first_move_san():
    """UCI -> SAN for every legal first move of either side.

    Only the opening histogram is ever named, and from the start position the
    mapping is unambiguous: pawn pushes are their destination square, knight
    moves are N + destination.
    """
    table = {}
    for f in range(8):
        for (home, one, two) in ((1, 2, 3), (6, 5, 4)):
            frm = home * 8 + f
            for r in (one, two):
                to = r * 8 + f
                table[SQUARES[frm] + SQUARES[to]] = SQUARES[to]
    for (frm, tos) in ((1, (16, 18)), (6, (21, 23)), (57, (40, 42)), (62, (45, 47))):
        for to in tos:
            table[SQUARES[frm] + SQUARES[to]] = "N" + SQUARES[to]
    return table


FIRST_MOVE_SAN = _build_first_move_san()


def uci_to_san_opening(uci):
    """SAN for a first move; falls back to the raw UCI when it is not one."""
    if not isinstance(uci, str):
        return "?"
    return FIRST_MOVE_SAN.get(uci.lower(), uci)


# ----------------------------------------------------------- telemetry input

def load_rows(path):
    """Read telemetry.jsonl -> (raw rows, number of unparsable lines)."""
    rows, bad = [], 0
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                obj = json.loads(line)
            except ValueError:
                bad += 1
                continue
            if isinstance(obj, dict):
                rows.append(obj)
            else:
                bad += 1
    return rows, bad


def norm_rate(value, total):
    """Counts -> rate.  Values already in [0,1] are passed through."""
    if value is None:
        return None
    if value > 1.0 and total:
        return value / total
    return value


def norm_per_game(value, games):
    """Per-game averages: a total larger than any per-game count is divided."""
    if value is None:
        return None
    if games and value > 64.0:
        return value / games
    return value


def parse_opening(raw):
    """Normalise the opening histogram into [(uci, count), ...], big first."""
    out = []

    def add(move, cnt):
        if cnt is None:
            return
        if isinstance(move, (int, float)) and not isinstance(move, bool):
            idx = int(move)
            if not (0 <= idx < 64 * 64):
                return
            move = SQUARES[idx // 64] + SQUARES[idx % 64]
        if not isinstance(move, str) or len(move) < 4:
            return
        try:
            out.append((move.lower(), float(cnt)))
        except (TypeError, ValueError):
            return

    if isinstance(raw, dict):
        for k, v in raw.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                add(k, v)
    elif isinstance(raw, list):
        for e in raw:
            if isinstance(e, dict):
                mv = e.get("move", e.get("uci", e.get("m", e.get("idx", e.get("index")))))
                cnt = e.get("n", e.get("count", e.get("c", e.get("games", e.get("freq")))))
                add(mv, cnt)
            elif isinstance(e, (list, tuple)) and len(e) >= 2:
                add(e[0], e[1])
    if not out:
        return None
    out.sort(key=lambda kv: -kv[1])
    return out


def parse_piece_dest(raw):
    """Normalise piece_dest into a 6 x 64 integer-ish matrix, or None."""
    grid = None
    if isinstance(raw, list) and len(raw) == 6 and all(isinstance(r, list) for r in raw):
        grid = raw
    elif isinstance(raw, list) and len(raw) == 6 * 64 and \
            all(isinstance(x, (int, float)) for x in raw):
        grid = [raw[i * 64:(i + 1) * 64] for i in range(6)]
    elif isinstance(raw, dict):
        grid = []
        for i, name in enumerate(PIECE_NAMES):
            row = raw.get(name, raw.get(name.upper(), raw.get(str(i))))
            grid.append(row if isinstance(row, list) else [])
    if grid is None:
        return None
    out = []
    for row in grid:
        vals = [float(x) if isinstance(x, (int, float)) and not _isnan(x) else 0.0
                for x in (row or [])]
        vals = (vals + [0.0] * 64)[:64]
        out.append(vals)
    if sum(sum(r) for r in out) <= 0:
        return None
    return out


HYPER_NAMES = ["temperature", "entropy_coef", "lr_scale", "shaping",
               "value_coef", "gamma", "lambda", "mutate_sigma"]


def parse_hypers(obj):
    if not isinstance(obj, dict):
        return {}
    out = {}
    for name in HYPER_NAMES:
        v = obj.get(name)
        if isinstance(v, (int, float)) and not isinstance(v, bool) and not _isnan(v):
            out[name] = float(v)
    return out


TERM_KEYS = ["checkmate", "stalemate", "fifty", "repetition",
             "insufficient", "maxplies"]


def canon_row(raw):
    """Raw telemetry object -> canonical row with a fixed key set."""
    r = {}
    r["gen"] = pick(raw, "gen", "generation", "g")
    r["games"] = pick(raw, "games", "n_games", "ngames")
    r["plies"] = pick(raw, "plies", "n_plies", "positions")
    r["sec"] = pick(raw, "sec", "secs", "seconds", "wall", "wall_sec", "elapsed", "time")
    r["gps"] = pick(raw, "gps", "games_per_sec", "games_sec")
    if r["gps"] is None:
        r["gps"] = safe_div(r["games"], r["sec"])
    r["elo_best"] = pick(raw, "elo_best", "elo.best", "best_elo", "best_agent.elo")
    r["elo_mean"] = pick(raw, "elo_mean", "elo.mean", "mean_elo")
    r["elo_p10"] = pick(raw, "elo_p10", "elo.p10", "elo_10", "elo.p10_agent")

    w = pick(raw, "white_win", "white_wins", "results.white", "white")
    b = pick(raw, "black_win", "black_wins", "results.black", "black")
    d = pick(raw, "draw", "draws", "draw_rate", "results.draw")
    tot = sum(x for x in (w, b, d) if x is not None)
    if tot > 1.5:
        w, b, d = (safe_div(w, tot), safe_div(b, tot), safe_div(d, tot))
    r["white_win"], r["black_win"], r["draw_rate"] = w, b, d
    if w is not None and b is not None:
        r["decisive_rate"] = w + b
    elif d is not None:
        r["decisive_rate"] = 1.0 - d
    else:
        r["decisive_rate"] = None

    r["avg_len"] = pick(raw, "avg_len", "avg_plies", "mean_len", "avg_length")
    if r["avg_len"] is None:
        r["avg_len"] = safe_div(r["plies"], r["games"])
    r["captures_per_game"] = norm_per_game(
        pick(raw, "captures_per_game", "captures"), r["games"])
    r["checks_per_game"] = norm_per_game(
        pick(raw, "checks_per_game", "checks"), r["games"])
    r["castle_rate"] = norm_per_game(
        pick(raw, "castle_rate", "castles", "castling_rate"), r["games"])
    r["promo_rate"] = norm_per_game(
        pick(raw, "promo_rate", "promotions", "promotion_rate"), r["games"])
    r["ep_rate"] = norm_per_game(
        pick(raw, "ep_rate", "ep_captures", "en_passant_rate"), r["games"])
    r["avg_final_material"] = pick(raw, "avg_final_material", "final_material",
                                   "avg_material")

    term = {}
    for k in TERM_KEYS:
        term[k] = pick(raw, "term." + k, "term_" + k, k)
    tsum = sum(v for v in term.values() if v is not None)
    if tsum > 1.5:
        term = {k: (v / tsum if v is not None else None) for k, v in term.items()}
        tsum = 1.0
    r["term"] = term
    r["term_total"] = tsum if tsum > 0 else None
    r["checkmate_share"] = term.get("checkmate")
    r["maxplies_share"] = term.get("maxplies")

    r["opening"] = parse_opening(pick_any(raw, "opening_top", "openings",
                                          "opening", "first_move"))
    r["piece_dest"] = parse_piece_dest(pick_any(raw, "piece_dest", "piece_dests",
                                                "piece_destinations"))
    pde = pick_any(raw, "piece_dest_entropy", "piece_entropy")
    if isinstance(pde, list):
        pde = mean(pde)
    elif isinstance(pde, dict):
        pde = mean(list(pde.values()))
    elif not isinstance(pde, (int, float)) or isinstance(pde, bool) or _isnan(pde):
        pde = None
    r["logged_piece_entropy"] = float(pde) if pde is not None else None

    ba = pick_any(raw, "best_agent", "champion", "best")
    r["best_agent"] = ba if isinstance(ba, dict) else None
    r["best_hypers"] = parse_hypers(ba)
    pm = pick_any(raw, "pop_mean", "population_mean", "hyper_mean", "mean_hyper",
                  "hypers_mean", "pop.mean", "population.mean")
    r["pop_hypers"] = parse_hypers(pm)
    ps = pick_any(raw, "pop_sd", "population_sd", "hyper_sd", "hypers_sd",
                  "pop.sd", "population.sd")
    r["pop_hyper_sd"] = parse_hypers(ps)

    r["loss_policy"] = pick(raw, "loss.policy", "loss_policy", "policy_loss")
    r["loss_value"] = pick(raw, "loss.value", "loss_value", "value_loss")
    r["loss_entropy"] = pick(raw, "loss.entropy", "loss_entropy", "entropy")
    r["grad_norm"] = pick(raw, "grad_norm", "gradnorm", "grad")

    # ---- derived, from the histograms -------------------------------------
    if r["opening"]:
        counts = [c for _, c in r["opening"]]
        tot_top = sum(counts)
        r["opening_top_share"] = safe_div(counts[0], tot_top)
        r["opening_entropy_bits"] = entropy_bits(counts)
        r["opening_coverage"] = safe_div(tot_top, r["games"]) if r["games"] else None
        if r["opening_coverage"] is not None:
            r["opening_coverage"] = min(1.0, r["opening_coverage"])
    else:
        r["opening_top_share"] = None
        r["opening_entropy_bits"] = None
        r["opening_coverage"] = None

    if r["piece_dest"]:
        ents = [entropy_bits(row) for row in r["piece_dest"]]
        r["piece_entropy_bits"] = mean([e for e in ents if e is not None])
        tot = sum(sum(row) for row in r["piece_dest"])
        cen = sum(row[s] for row in r["piece_dest"] for s in CENTRE16)
        r["centre_share"] = safe_div(cen, tot)
    else:
        r["piece_entropy_bits"] = r["logged_piece_entropy"]
        r["centre_share"] = None
    return r


def diff_if_cumulative(rows):
    """piece_dest may be logged per generation or cumulative; normalise.

    Detection: cumulative histograms are non-decreasing in every generation and
    the final total dwarfs the median.  Returns the mode string.
    """
    have = [r for r in rows if r["piece_dest"]]
    if len(have) < 4:
        return "per_generation"
    totals = [sum(sum(x) for x in r["piece_dest"]) for r in have]
    nondec = sum(1 for a, b in zip(totals, totals[1:]) if b >= a - 1e-9)
    monotone = nondec >= 0.95 * (len(totals) - 1)
    cumulative = False
    # Best signal: every ply is one piece move, so a per-generation histogram
    # holds about `plies` entries and that ratio stays flat.  A cumulative one
    # grows without bound.
    ratios = [safe_div(t, r.get("plies")) for t, r in zip(totals, have)]
    ratios = [x for x in ratios if x is not None and x > 0]
    if len(ratios) >= 6:
        k = max(1, len(ratios) // 4)
        early, late = median(ratios[:k]), median(ratios[-k:])
        if early and late and late > 2.0 * early:
            cumulative = True
    elif monotone and totals[0] > 0 and totals[-1] >= 1.8 * totals[0]:
        cumulative = True
    if cumulative and monotone:
        prev = None
        for r in have:
            cur = r["piece_dest"]
            if prev is not None:
                r["piece_dest"] = [[max(0.0, c - p) for c, p in zip(crow, prow)]
                                   for crow, prow in zip(cur, prev)]
            prev = cur
        # first generation keeps its own (cumulative == per-gen at g0)
        for r in have:
            ents = [entropy_bits(row) for row in r["piece_dest"]]
            ents = [e for e in ents if e is not None]
            r["piece_entropy_bits"] = mean(ents) if ents else r["logged_piece_entropy"]
            tot = sum(sum(x) for x in r["piece_dest"])
            cen = sum(x[s] for x in r["piece_dest"] for s in CENTRE16)
            r["centre_share"] = safe_div(cen, tot)
        return "cumulative_differenced"
    return "per_generation"


# --------------------------------------------------------------------- bands

BAND_LABEL = ["first quarter", "second quarter", "third quarter", "final quarter"]

CORE_METRICS = [
    ("draw_rate", "draw rate", "share of games"),
    ("avg_len", "average game length", "plies"),
    ("captures_per_game", "captures per game", "captures"),
    ("castle_rate", "castling rate", "castles per game"),
    ("promo_rate", "promotion rate", "promotions per game"),
    ("checkmate_share", "checkmate share of terminations", "share"),
    ("avg_final_material", "average final material", "pawns"),
]

EXTRA_METRICS = [
    ("elo_mean", "population mean Elo", "Elo"),
    ("decisive_rate", "decisive-game rate", "share"),
    ("checks_per_game", "checks per game", "checks"),
    ("maxplies_share", "ply-cap terminations", "share"),
    ("ep_rate", "en-passant rate", "per game"),
    ("opening_top_share", "share of White's most played first move", "share"),
    ("opening_entropy_bits", "opening entropy (top-8)", "bits"),
    ("opening_coverage", "coverage of the top-8 openings", "share of games"),
    ("piece_entropy_bits", "piece-destination entropy", "bits"),
    ("centre_share", "moves landing in the extended centre", "share"),
    ("loss_value", "value loss", ""),
    ("grad_norm", "gradient norm", ""),
]

# Plain-English readings; (rise, fall).
METRIC_PROSE = {
    "draw_rate": ("games ended in a draw more often -- agents got harder to beat, "
                  "or more passive",
                  "more games were converted into a decisive result"),
    "avg_len": ("games got longer -- fewer quick collapses, more manoeuvring",
                "games got shorter -- decisions arrived faster"),
    "captures_per_game": ("the population traded more wood",
                          "the population traded less -- pieces were kept safer "
                          "or hung less often"),
    "castle_rate": ("castling became a habit: king safety was learned",
                    "castling was abandoned"),
    "promo_rate": ("pawns were actually pushed through and promoted -- endgame "
                   "technique appeared",
                   "promotions became rarer"),
    "checkmate_share": ("more games were finished by an actual checkmate",
                        "fewer games ended in mate; more were adjudicated"),
    "avg_final_material": ("games ended with more material still on the board -- "
                           "results arrived before the endgame",
                           "games were played deeper into simplified positions"),
    "elo_mean": ("the whole population, not just the champion, got stronger",
                 "the population's mean rating fell"),
    "decisive_rate": ("more games produced a winner", "fewer games produced a winner"),
    "checks_per_game": ("agents gave check more often -- more forcing play",
                        "agents gave check less often"),
    "maxplies_share": ("more games ran into the ply cap without resolution",
                       "fewer games ran into the ply cap: agents learned to finish"),
    "ep_rate": ("en-passant captures became more common",
                "en-passant captures became rarer"),
    "opening_top_share": ("White concentrated on one favourite first move",
                          "White's first move spread out again"),
    "opening_entropy_bits": ("White's opening choice became more random",
                             "White's opening choice became more focused"),
    "opening_coverage": ("a small set of first moves took over the repertoire",
                         "the repertoire became more scattered"),
    "piece_entropy_bits": ("piece destinations became more scattered",
                           "piece destinations specialised onto fewer squares"),
    "centre_share": ("more moves landed in the centre -- central control was learned",
                     "play drifted away from the centre"),
    "loss_value": ("the value head fitted its targets worse",
                   "the value head predicted outcomes better"),
    "grad_norm": ("gradients grew", "gradients shrank -- the update settled down"),
}


def band_ranges(n, nb):
    """Split n ordered generations into nb (near-)equal index ranges."""
    nb = max(1, min(nb, n))
    edges = [round(i * n / nb) for i in range(nb + 1)]
    for i in range(1, len(edges)):
        edges[i] = max(edges[i], edges[i - 1] + 1)
    edges[-1] = n
    return [(edges[i], edges[i + 1]) for i in range(nb) if edges[i] < edges[i + 1]]


def band_metric(rows, lo, hi, key):
    vals = [r.get(key) for r in rows[lo:hi]]
    v = _nums(vals)
    return {"mean": mean(v), "sd": sd(v), "n": len(v),
            "first": v[0] if v else None, "last": v[-1] if v else None}


def compare(cur, base):
    """2-sigma significance test between two band summaries."""
    out = {"delta": None, "threshold": None, "significant": None, "ratio": None,
           "note": None}
    if cur["mean"] is None or base["mean"] is None:
        out["note"] = "not logged"
        return out
    out["delta"] = cur["mean"] - base["mean"]
    if base["mean"]:
        out["ratio"] = cur["mean"] / base["mean"]
    sds = [s for s in (cur["sd"], base["sd"]) if s is not None]
    if not sds:
        out["note"] = "fewer than two generations in a band -- no noise estimate"
        return out
    thr = 2.0 * max(sds)
    out["threshold"] = thr
    if thr <= 1e-12:
        out["significant"] = abs(out["delta"]) > 1e-12
        out["note"] = "zero within-band variance; treat with care"
    else:
        out["significant"] = abs(out["delta"]) > thr
        out["strength"] = abs(out["delta"]) / thr
    return out


def analyse_phases(rows, gens, nb):
    ranges = band_ranges(len(rows), nb)
    bands = []
    for i, (lo, hi) in enumerate(ranges):
        bands.append({
            "index": i,
            "label": BAND_LABEL[i] if len(ranges) == 4 and i < 4 else "band %d" % (i + 1),
            "gen_lo": gens[lo], "gen_hi": gens[hi - 1], "n_gens": hi - lo,
            "metrics": {},
        })
    all_metrics = CORE_METRICS + EXTRA_METRICS
    for key, _label, _unit in all_metrics:
        summaries = [band_metric(rows, lo, hi, key) for (lo, hi) in ranges]
        for i, s in enumerate(summaries):
            cmp_first = compare(s, summaries[0]) if i else {
                "delta": 0.0, "threshold": None, "significant": False, "ratio": 1.0,
                "note": "baseline"}
            cmp_prev = compare(s, summaries[i - 1]) if i else dict(cmp_first)
            bands[i]["metrics"][key] = {
                "mean": s["mean"], "sd": s["sd"], "n": s["n"],
                "vs_first": cmp_first, "vs_prev": cmp_prev,
            }
    statements = phase_statements(bands, all_metrics)
    return {
        "n_bands": len(ranges),
        "core_metrics": [m[0] for m in CORE_METRICS],
        "extra_metrics": [m[0] for m in EXTRA_METRICS],
        "metric_labels": {m[0]: m[1] for m in all_metrics},
        "metric_units": {m[0]: m[2] for m in all_metrics},
        "bands": bands,
        "statements": statements,
    }


def fnum(x, nd=3):
    if x is None:
        return "n/a"
    try:
        f = float(x)
    except (TypeError, ValueError):
        return "n/a"
    if _isnan(f):
        return "n/a"
    if abs(f) >= 1000:
        return "%.0f" % f
    return ("%." + str(nd) + "f") % f


def fdelta(x, nd=3):
    """Signed delta cell; "n/a" rather than "+n/a" when the value is missing."""
    if x is None or _isnan(x):
        return "n/a"
    return ("+" if float(x) >= 0 else "") + fnum(x, nd)


def fint(x):
    """Big counts with thousands separators."""
    if x is None or _isnan(x):
        return "n/a"
    return "{:,}".format(int(round(float(x))))


def phase_statements(bands, all_metrics):
    """One plain-English line per metric per band, honest about noise."""
    out = []
    if len(bands) < 2:
        return out
    core = {m[0] for m in CORE_METRICS}
    for key, label, unit in all_metrics:
        b0 = bands[0]["metrics"].get(key, {})
        if b0.get("mean") is None:
            out.append({
                "metric": key, "band": None, "significant": False,
                "kind": "missing",
                "text": "%s was never logged, so nothing can be said about it."
                        % (label[0].upper() + label[1:]),
            })
            continue
        for b in bands[1:]:
            m = b["metrics"].get(key, {})
            c = m.get("vs_first", {})
            delta = c.get("delta")
            if delta is None:
                continue
            where = "gen %d-%d (%s)" % (b["gen_lo"], b["gen_hi"], b["label"])
            base_txt = "%s %s vs %s in the first band" % (
                label, fnum(m.get("mean")), fnum(b0.get("mean")))
            if c.get("significant"):
                rise, fall = METRIC_PROSE.get(key, ("it went up", "it went down"))
                prose = rise if delta > 0 else fall
                text = ("By %s, %s: a change of %s%s (%.1fx the %s noise band). "
                        "In plain terms: %s." % (
                            where, base_txt, "+" if delta > 0 else "",
                            fnum(delta), c.get("strength", 1.0),
                            "within-band", prose))
                kind = "signal"
            elif c.get("threshold") is None:
                text = ("By %s, %s, but %s -- the change is not testable."
                        % (where, base_txt, c.get("note", "there is no noise estimate")))
                kind = "untestable"
            else:
                text = ("By %s, %s: a change of %s%s, smaller than the 2-sigma "
                        "within-band noise band of %s. Call it noise: nothing was "
                        "learned here." % (where, base_txt,
                                           "+" if delta > 0 else "", fnum(delta),
                                           fnum(c["threshold"])))
                kind = "noise"
            out.append({"metric": key, "band": b["index"], "core": key in core,
                        "kind": kind, "significant": bool(c.get("significant")),
                        "delta": delta, "threshold": c.get("threshold"),
                        "strength": c.get("strength"), "text": text})
    return out


# ------------------------------------------------------------------ learning

def detect_plateau(gens, vals):
    """Moving-average slope test.

    Smooth the curve, take a trailing least-squares slope, and call a plateau at
    the first generation after which the slope never again exceeds
    max(20% of the best early slope, the slope explained by per-generation noise).
    """
    pts = [(g, v) for g, v in zip(gens, vals) if v is not None]
    n = len(pts)
    res = {"detected": False, "gen": None, "reason": None, "window": None,
           "threshold": None, "slope": [], "smoothed": [], "gain_after": None,
           "frac_through": None}
    if n < 8:
        res["reason"] = ("run too short for a plateau test (%d usable generations, "
                         "8 needed)" % n)
        return res
    gs = [p[0] for p in pts]
    vs = [p[1] for p in pts]
    # Windows scale with the run but stay local: on a 2000-generation run a
    # window of n/8 would smooth away everything that happened in the first 5%.
    w = min(max(3, n // 8), 31)
    if w % 2 == 0:
        w += 1
    ma = []
    half = w // 2
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        ma.append(sum(vs[lo:hi]) / (hi - lo))
    win = min(max(3, n // 5), 60)
    slopes = [None] * n
    for i in range(win - 1, n):
        slopes[i] = lsq_slope(gs[i - win + 1:i + 1], ma[i - win + 1:i + 1])
    diffs = [b - a for a, b in zip(vs, vs[1:])]
    noise = (sd(diffs) or 0.0) / math.sqrt(win)
    known = [s for s in slopes if s is not None]
    early_cut = max(win, int(0.6 * n))
    early = max([s for s in slopes[:early_cut] if s is not None] or known or [0.0])
    thr = max(0.20 * early, noise) if early > 0 else max(noise, 1e-9)
    res["window"] = win
    res["ma_window"] = w
    res["slope_window"] = win
    res["threshold"] = thr
    res["smoothed"] = ma
    res["slope"] = slopes
    res["early_slope"] = early
    res["noise_slope"] = noise
    if early <= 0:
        res["reason"] = "the curve never rose, so there is no plateau to find"
        return res
    total_gain = ma[-1] - ma[0]
    step_noise = sd(diffs) or 0.0
    res["total_gain_smoothed"] = total_gain
    if total_gain <= 2.0 * step_noise:
        res["reason"] = ("the curve never rose beyond its own generation-to-generation "
                         "noise (smoothed change %s against a per-generation noise of "
                         "%s), so calling any point a plateau would be meaningless"
                         % (fnum(total_gain, 1), fnum(step_noise, 1)))
        return res
    # The plateau must hold for the rest of the run, which is what stops a slow
    # start being mistaken for one -- so the scan may begin as soon as a slope
    # exists rather than at some arbitrary fraction of the run.
    minlen = max(3, int(0.15 * n))
    start = win
    for i in range(start, n - minlen + 1):
        if all(slopes[j] is not None and slopes[j] < thr for j in range(i, n)):
            res["detected"] = True
            res["gen"] = gs[i]
            res["index"] = i
            res["gain_after"] = vs[-1] - vs[i]
            res["frac_through"] = i / float(n - 1)
            res["gens_after"] = n - 1 - i
            return res
    res["reason"] = ("no plateau: the smoothed curve was still climbing faster than "
                     "%s Elo/generation at the end of the run" % fnum(thr))
    return res


def analyse_learning(rows, gens):
    def ser(k):
        return [r.get(k) for r in rows]

    best = ser("elo_best")
    meanv = ser("elo_mean")
    p10 = ser("elo_p10")
    out = {}
    for name, s in (("elo_best", best), ("elo_mean", meanv), ("elo_p10", p10)):
        v = [(g, x) for g, x in zip(gens, s) if x is not None]
        if not v:
            out[name] = {"available": False}
            continue
        peak = max(v, key=lambda t: t[1])
        out[name] = {
            "available": True, "start": v[0][1], "end": v[-1][1],
            "gain": v[-1][1] - v[0][1], "peak": peak[1], "peak_gen": peak[0],
            "per_gen": (v[-1][1] - v[0][1]) / max(1, (v[-1][0] - v[0][0])),
        }
    out["plateau"] = detect_plateau(gens, best)
    out["plateau_mean"] = detect_plateau(gens, meanv)
    spread = [(b - p) if (b is not None and p is not None) else None
              for b, p in zip(best, p10)]
    sv = _nums(spread)
    out["spread"] = {"start": sv[0] if sv else None, "end": sv[-1] if sv else None,
                     "mean": mean(sv)}
    # How much of the total gain arrived in each quarter of the run.
    v = [x for x in best if x is not None]
    if len(v) >= 4:
        qs = band_ranges(len(v), 4)
        total = v[-1] - v[0]
        out["gain_by_quarter"] = [
            {"index": i, "gain": v[hi - 1] - v[lo],
             "share": ((v[hi - 1] - v[lo]) / total) if total else None}
            for i, (lo, hi) in enumerate(qs)]
    else:
        out["gain_by_quarter"] = []
    return out


# ---------------------------------------------------------------- throughput

def analyse_throughput(rows):
    games = sum(_nums([r.get("games") for r in rows]))
    plies = sum(_nums([r.get("plies") for r in rows]))
    secs = sum(_nums([r.get("sec") for r in rows]))
    gps_logged = _nums([r.get("gps") for r in rows])
    out = {
        "total_games": games or None,
        "total_plies": plies or None,
        "wall_seconds": secs or None,
        "wall_hours": (secs / 3600.0) if secs else None,
        "games_per_sec": safe_div(games, secs),
        "positions_per_sec": safe_div(plies, secs),
        "gps_logged_mean": mean(gps_logged),
        "gps_logged_peak": max(gps_logged) if gps_logged else None,
        "plies_per_game": safe_div(plies, games),
        "human_game_seconds": HUMAN_GAME_SECONDS,
    }
    if games:
        human = games * HUMAN_GAME_SECONDS
        out["human_equivalent_seconds"] = human
        out["human_equivalent_years"] = human / (365.25 * 24 * 3600.0)
        out["speedup_vs_human"] = safe_div(human, secs)
    else:
        out["human_equivalent_seconds"] = None
        out["human_equivalent_years"] = None
        out["speedup_vs_human"] = None
    out["meets_1000_gps"] = (out["games_per_sec"] is not None and
                             out["games_per_sec"] >= 1000.0)
    return out


# ------------------------------------------------------------------ openings

def merge_openings(rows, lo, hi):
    agg = {}
    used = 0
    for r in rows[lo:hi]:
        op = r.get("opening")
        if not op:
            continue
        used += 1
        for mv, c in op:
            agg[mv] = agg.get(mv, 0.0) + c
    if not agg:
        return None, 0
    items = sorted(agg.items(), key=lambda kv: -kv[1])
    return items, used


def analyse_openings(rows, gens, nb):
    ranges = band_ranges(len(rows), nb)
    have = any(r.get("opening") for r in rows)
    out = {"available": have, "bands": [], "statements": [], "note": None}
    if not have:
        out["note"] = ("opening_top was not logged, so White's opening repertoire "
                       "cannot be analysed")
        return out
    for i, (lo, hi) in enumerate(ranges):
        items, used = merge_openings(rows, lo, hi)
        band = {"index": i, "gen_lo": gens[lo], "gen_hi": gens[hi - 1],
                "gens_with_data": used, "top": [], "entropy_bits": None,
                "top_share": None, "n_distinct": 0}
        if items:
            tot = sum(c for _, c in items)
            band["n_distinct"] = len(items)
            band["entropy_bits"] = entropy_bits([c for _, c in items])
            band["top_share"] = items[0][1] / tot if tot else None
            for mv, c in items[:8]:
                band["top"].append({"uci": mv, "san": uci_to_san_opening(mv),
                                    "count": c, "share": (c / tot) if tot else None})
        out["bands"].append(band)

    first = out["bands"][0]
    last = out["bands"][-1]
    out["shift"] = []
    if first["top"] and last["top"]:
        fmap = {t["uci"]: t["share"] for t in first["top"]}
        lmap = {t["uci"]: t["share"] for t in last["top"]}
        # A move absent from a band's logged top-8 is not known to be zero: its
        # share there is at most the smallest share that band did log.
        fbound = min([t["share"] or 0.0 for t in first["top"]] or [0.0])
        lbound = min([t["share"] or 0.0 for t in last["top"]] or [0.0])
        moved = []
        for mv in set(list(fmap) + list(lmap)):
            a, b = fmap.get(mv), lmap.get(mv)
            uncertain = a is None or b is None
            av, bv = (a or 0.0), (b or 0.0)
            bound = (fbound if a is None else 0.0) + (lbound if b is None else 0.0)
            point = bv - av
            # Conservative delta: shrink the point estimate by everything the
            # truncated histogram could be hiding, and never change its sign.
            if point > 0:
                conservative = max(0.0, point - bound)
            elif point < 0:
                conservative = min(0.0, point + bound)
            else:
                conservative = 0.0
            moved.append({
                "uci": mv, "san": uci_to_san_opening(mv),
                "first_share": av, "last_share": bv,
                "first_known": a is not None, "last_known": b is not None,
                "delta": point, "delta_conservative": conservative,
                "uncertain": uncertain, "error_bound": bound,
                "certain": abs(conservative) > 0.005,
            })
        moved.sort(key=lambda d: (-abs(d["delta_conservative"]), -abs(d["delta"])))
        out["shift"] = moved[:8]
        out["statements"].append(
            "White's most played first move went from %s (%.1f%% of the band's logged "
            "top-8) to %s (%.1f%%)." % (
                first["top"][0]["san"], 100.0 * (first["top"][0]["share"] or 0),
                last["top"][0]["san"], 100.0 * (last["top"][0]["share"] or 0)))
        trust = [m for m in moved if m["certain"]]
        gainers = [m for m in trust if m["delta_conservative"] > 0][:3]
        losers = [m for m in trust if m["delta_conservative"] < 0][:3]

        def swing(m):
            if m["uncertain"]:
                return "%s at least %+.1f pts" % (m["san"],
                                                  100.0 * m["delta_conservative"])
            return "%s %+.1f pts" % (m["san"], 100.0 * m["delta"])

        if gainers:
            out["statements"].append(
                "Gained the most: " + ", ".join(swing(m) for m in gainers) + ".")
        if losers:
            out["statements"].append(
                "Lost the most: " + ", ".join(swing(m) for m in losers) + ".")
        if not trust:
            out["statements"].append(
                "No individual first move moved by more than the uncertainty created "
                "by the top-8 truncation, so no move can be said to have gained or "
                "lost ground.")
        if first["entropy_bits"] is not None and last["entropy_bits"] is not None:
            d = last["entropy_bits"] - first["entropy_bits"]
            out["entropy_delta"] = d
            verdict = ("narrowed" if d < -0.05 else
                       "widened" if d > 0.05 else
                       "did not measurably change")
            out["statements"].append(
                "Entropy of the logged top-8 first moves moved from %s to %s bits "
                "(%s%s): the repertoire %s." % (
                    fnum(first["entropy_bits"], 2), fnum(last["entropy_bits"], 2),
                    "+" if d > 0 else "", fnum(d, 2), verdict))
    return out


# --------------------------------------------------------------------- pieces

def merge_piece_dest(rows, lo, hi):
    agg = [[0.0] * 64 for _ in range(6)]
    used = 0
    for r in rows[lo:hi]:
        pd = r.get("piece_dest")
        if not pd:
            continue
        used += 1
        for p in range(6):
            for s in range(64):
                agg[p][s] += pd[p][s]
    if not used:
        return None, 0
    return agg, used


def top_squares(counts, k=5):
    tot = sum(counts)
    if tot <= 0:
        return []
    order = sorted(range(64), key=lambda s: -counts[s])[:k]
    return [{"sq": sq_name(s), "index": s, "count": counts[s],
             "share": counts[s] / tot} for s in order if counts[s] > 0]


def analyse_pieces(rows, gens, nb, mode):
    ranges = band_ranges(len(rows), nb)
    have = any(r.get("piece_dest") for r in rows)
    out = {"available": have, "mode": mode, "note": None, "pieces": [],
           "statements": [], "entropy_series_source": None}
    ent_series = [r.get("piece_entropy_bits") for r in rows]
    if any(e is not None for e in ent_series):
        out["entropy_series_source"] = ("piece_dest" if have else
                                        "piece_dest_entropy (logged)")
    if not have:
        out["note"] = ("piece_dest was not logged; only the scalar "
                       "piece_dest_entropy trend is available" if
                       any(r.get("logged_piece_entropy") is not None for r in rows)
                       else "piece_dest was not logged, so no heatmaps are available")
        if any(r.get("logged_piece_entropy") is not None for r in rows):
            v = _nums([r.get("logged_piece_entropy") for r in rows])
            out["logged_entropy"] = {"start": v[0], "end": v[-1],
                                     "delta": v[-1] - v[0], "mean": mean(v)}
            out["statements"].append(
                "Logged piece-destination entropy moved from %s to %s bits (%s%s): "
                "piece placement became %s." % (
                    fnum(v[0], 2), fnum(v[-1], 2),
                    "+" if v[-1] >= v[0] else "", fnum(v[-1] - v[0], 2),
                    "more specialised" if v[-1] < v[0] else "less specialised"))
        return out

    lo0, hi0 = ranges[0]
    loN, hiN = ranges[-1]
    first, nf = merge_piece_dest(rows, lo0, hi0)
    last, nl = merge_piece_dest(rows, loN, hiN)
    out["first_band"] = {"index": 0, "gen_lo": gens[lo0], "gen_hi": gens[hi0 - 1],
                         "gens_with_data": nf}
    out["last_band"] = {"index": len(ranges) - 1, "gen_lo": gens[loN],
                        "gen_hi": gens[hiN - 1], "gens_with_data": nl}
    heat_first, heat_last, heat_delta = {}, {}, {}
    for p, name in enumerate(PIECE_NAMES):
        cf = first[p] if first else [0.0] * 64
        cl = last[p] if last else [0.0] * 64
        tf, tl = sum(cf), sum(cl)
        sf = [c / tf for c in cf] if tf else [0.0] * 64
        sl = [c / tl for c in cl] if tl else [0.0] * 64
        heat_first[name] = sf
        heat_last[name] = sl
        heat_delta[name] = [b - a for a, b in zip(sf, sl)]
        ef, el = entropy_bits(cf), entropy_bits(cl)
        cen_f = safe_div(sum(cf[s] for s in CENTRE16), tf)
        cen_l = safe_div(sum(cl[s] for s in CENTRE16), tl)
        gained = sorted(range(64), key=lambda s: -(sl[s] - sf[s]))[:3]
        entry = {
            "piece": name, "letter": PIECE_LETTER[name],
            "moves_first": tf, "moves_last": tl,
            "entropy_first_bits": ef, "entropy_last_bits": el,
            "entropy_delta": (el - ef) if (ef is not None and el is not None) else None,
            "top_first": top_squares(cf), "top_last": top_squares(cl),
            "centre_share_first": cen_f, "centre_share_last": cen_l,
            "gained_squares": [{"sq": sq_name(s), "delta": sl[s] - sf[s]}
                               for s in gained if sl[s] - sf[s] > 0],
        }
        out["pieces"].append(entry)
        if entry["top_last"]:
            pref = ", ".join("%s (%.1f%%)" % (t["sq"], 100 * t["share"])
                             for t in entry["top_last"][:3])
            was = ", ".join("%s (%.1f%%)" % (t["sq"], 100 * t["share"])
                            for t in entry["top_first"][:3]) or "nothing in particular"
            trend = "no entropy estimate"
            if entry["entropy_delta"] is not None:
                d = entry["entropy_delta"]
                shape = ("specialised" if d < -0.02 else
                         "spread out" if d > 0.02 else "barely moved")
                trend = "%s (%s -> %s bits)" % (shape, fnum(ef, 2), fnum(el, 2))
            out["statements"].append(
                "%ss ended the run preferring %s; early on they preferred %s. "
                "Destination entropy %s." % (name.capitalize(), pref, was, trend))
    out["heatmap_first"] = heat_first
    out["heatmap_last"] = heat_last
    out["heatmap_delta"] = heat_delta
    ents_f = [p["entropy_first_bits"] for p in out["pieces"]]
    ents_l = [p["entropy_last_bits"] for p in out["pieces"]]
    mf, ml = mean(ents_f), mean(ents_l)
    out["entropy_summary"] = {"first": mf, "last": ml,
                              "delta": (ml - mf) if (mf is not None and ml is not None)
                              else None,
                              "max_bits": 6.0}
    return out


# ------------------------------------------------------------------ champion

HYPER_PROSE = {
    "temperature": (
        "samples its moves more randomly than the population: it wins while keeping "
        "its policy broad, which usually means robustness rather than sharpness",
        "plays far more deterministically than the population: it converged on a "
        "committed, sharp policy and trusts it"),
    "entropy_coef": (
        "is pushed harder to keep its policy spread out -- more exploration pressure "
        "than average",
        "is under less pressure to stay exploratory, so its policy is free to sharpen"),
    "lr_scale": (
        "learns faster than the population average -- it keeps adapting, at the cost "
        "of stability",
        "learns more slowly than average -- its edge came from stability, not speed"),
    "shaping": (
        "leans harder on the material-shaping reward: a materialist, capture-driven "
        "style",
        "leans less on material shaping and more on the terminal win/loss signal: a "
        "more patient, outcome-driven style"),
    "value_coef": ("weights its value loss more heavily than average",
                   "weights its value loss less heavily than average"),
    "gamma": ("discounts the future less: a longer planning horizon",
              "discounts the future more: a shorter planning horizon"),
    "lambda": ("uses longer-horizon GAE credit than average",
               "uses shorter-horizon, lower-variance credit than average"),
    "mutate_sigma": ("mutates its clones more aggressively than average",
                     "mutates its clones more gently than average"),
}


def read_sidecar_agents(run_dir):
    """Optional runs/NAME/agents.json | model.json: true population statistics."""
    for name in ("agents.json", "model.json", "model_info.json"):
        path = os.path.join(run_dir, name)
        if not os.path.isfile(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as fh:
                obj = json.load(fh)
        except (ValueError, OSError):
            continue
        agents = obj.get("agents") if isinstance(obj, dict) else obj
        if isinstance(agents, list) and agents and isinstance(agents[0], dict):
            return name, agents
    return None, None


def analyse_champion(rows, run_dir):
    out = {"available": False, "note": None, "comparison": [], "statements": []}
    champ_rows = [r for r in rows if r.get("best_hypers")]
    if not champ_rows:
        out["note"] = ("best_agent hyper-parameters were not logged, so the "
                       "champion's style cannot be compared with the population")
        return out
    last = champ_rows[-1]
    ch = last["best_hypers"]
    ba = last.get("best_agent") or {}
    out["available"] = True
    out["agent"] = ba.get("i", ba.get("index"))
    out["elo"] = ba.get("elo", last.get("elo_best"))
    out["gen"] = last.get("gen")
    out["hypers"] = ch

    baseline, base_sd, source = None, {}, None
    pop_rows = [r for r in rows if r.get("pop_hypers")]
    if pop_rows:
        baseline = pop_rows[-1]["pop_hypers"]
        base_sd = pop_rows[-1].get("pop_hyper_sd") or {}
        source = "population mean logged in telemetry (final generation)"
    if baseline is None:
        fname, agents = read_sidecar_agents(run_dir)
        if agents:
            baseline, base_sd = {}, {}
            for name in HYPER_NAMES:
                vals = _nums([a.get(name) for a in agents])
                if vals:
                    baseline[name] = mean(vals)
                    s = sd(vals)
                    if s is not None:
                        base_sd[name] = s
            source = "population of %d agents read from %s" % (len(agents), fname)
            if not baseline:
                baseline = None
    if baseline is None:
        baseline, base_sd = {}, {}
        for name in HYPER_NAMES:
            vals = _nums([r["best_hypers"].get(name) for r in champ_rows])
            if vals:
                baseline[name] = mean(vals)
                s = sd(vals)
                if s is not None:
                    base_sd[name] = s
        source = ("PROXY: the population mean was not logged, so the baseline is the "
                  "run's own distribution of champion hyper-parameters across %d "
                  "generations" % len(champ_rows))
        out["baseline_is_proxy"] = True
    out["baseline"] = {"source": source, "mean": baseline, "sd": base_sd}

    for name in HYPER_NAMES:
        v = ch.get(name)
        b = baseline.get(name) if baseline else None
        if v is None:
            continue
        row = {"name": name, "value": v, "baseline": b,
               "ratio": (v / b) if b else None, "z": None, "reading": None}
        s = base_sd.get(name)
        if s and s > 0 and b is not None:
            row["z"] = (v - b) / s
        if b is not None:
            rel = (v - b) / abs(b) if b else 0.0
            strong = abs(rel) >= 0.15 or (row["z"] is not None and abs(row["z"]) >= 1.0)
            if strong:
                rise, fall = HYPER_PROSE.get(name, ("is above the population",
                                                    "is below the population"))
                row["reading"] = ("The champion " + (rise if v > b else fall) + ".")
                row["divergence"] = abs(rel)
            else:
                row["reading"] = ("Within %s of the population mean -- not a "
                                  "distinguishing trait." % "15%")
                row["divergence"] = abs(rel)
        out["comparison"].append(row)

    strong = [c for c in out["comparison"]
              if c.get("divergence") is not None and c["divergence"] >= 0.15]
    strong.sort(key=lambda c: -c["divergence"])
    if strong:
        out["statements"] = [c["reading"] for c in strong[:3]]
        out["summary"] = ("The champion is characterised mainly by %s." % " and ".join(
            "%s %s the population (%s vs %s)" % (
                c["name"], "above" if c["value"] > c["baseline"] else "below",
                fnum(c["value"]), fnum(c["baseline"])) for c in strong[:2]))
    else:
        out["summary"] = ("The champion's hyper-parameters are indistinguishable from "
                          "the population baseline: its rating came from its weights, "
                          "not from an unusual PBT configuration.")
        out["statements"] = [out["summary"]]
    # champion hyper drift across the run
    drift = {}
    for name in HYPER_NAMES:
        vals = [(r.get("gen"), r["best_hypers"].get(name)) for r in champ_rows]
        vals = [(g, v) for g, v in vals if g is not None and v is not None]
        if len(vals) >= 4:
            drift[name] = {"start": vals[0][1], "end": vals[-1][1],
                           "slope_per_gen": lsq_slope([g for g, _ in vals],
                                                      [v for _, v in vals])}
    out["drift"] = drift
    return out


# ------------------------------------------------------- findings + failures

FINDING_TITLE = {
    "draw_rate": ("Learned to hold draws", "Learned to force decisive games"),
    "avg_len": ("Learned to play longer games", "Learned to finish games faster"),
    "captures_per_game": ("Learned to trade", "Learned to stop hanging material"),
    "castle_rate": ("Learned to castle", "Stopped castling"),
    "promo_rate": ("Learned to promote pawns", "Stopped promoting pawns"),
    "checkmate_share": ("Learned to deliver checkmate", "Stopped delivering checkmate"),
    "avg_final_material": ("Learned to win before the endgame",
                           "Learned to play into simplified endgames"),
    "elo_mean": ("The whole population improved, not just the champion",
                 "The population's mean rating fell"),
    "decisive_rate": ("Learned to convert games", "Games became less decisive"),
    "checks_per_game": ("Learned forcing, checking play", "Play became quieter"),
    "maxplies_share": ("More games ran into the ply cap",
                       "Learned to resolve games before the ply cap"),
    "ep_rate": ("En passant became more common", "En passant became rarer"),
    "opening_top_share": ("Converged on a favourite first move",
                          "Opening choice spread out"),
    "opening_entropy_bits": ("Opening choice became more random",
                             "Built a narrow opening repertoire"),
    "opening_coverage": ("A handful of openings took over",
                         "The opening repertoire scattered"),
    "piece_entropy_bits": ("Piece placement became more random",
                           "Pieces specialised onto specific squares"),
    "centre_share": ("Learned to play in the centre", "Play drifted from the centre"),
    "loss_value": ("The value head got worse at predicting outcomes",
                   "The value head learned to predict outcomes"),
    "grad_norm": ("Gradients grew through the run", "The update settled down"),
}

# Metrics whose movement is genuinely about chess strategy, ranked ahead of
# optimisation bookkeeping when the evidence is equally strong.
STRATEGY_WEIGHT = {
    "castle_rate": 1.35, "checkmate_share": 1.3, "promo_rate": 1.25,
    "draw_rate": 1.2, "captures_per_game": 1.15, "centre_share": 1.15,
    "opening_top_share": 1.1, "piece_entropy_bits": 1.1, "avg_final_material": 1.05,
    "avg_len": 1.0, "decisive_rate": 1.1, "maxplies_share": 1.05,
    "checks_per_game": 0.95, "opening_entropy_bits": 1.0, "opening_coverage": 1.0,
    "elo_mean": 1.0, "loss_value": 0.7, "grad_norm": 0.5, "ep_rate": 0.6,
}


def finding_score(strength, weight, rel_magnitude=None):
    """Rank by noise-relative effect, compressed, damped by relative size.

    A metric with a razor-thin within-band variance can clear the 2-sigma bar by
    a factor of 20 while barely moving in absolute terms; log-compressing the
    strength and multiplying by how large the move is relative to its own
    baseline keeps such findings honest without discarding them.
    """
    s = max(1.0, float(strength))
    base = weight * (1.0 + math.log(s, 2.0))
    if rel_magnitude is None:
        return base
    return base * (0.6 + 0.4 * min(1.0, abs(rel_magnitude)))


# Findings that would say the same thing twice; only the strongest survives.
DEDUPE_GROUPS = [{"draw_rate", "decisive_rate"}]


def build_findings(phases, learning, openings, pieces, caveats):
    findings = []
    labels = phases["metric_labels"]
    bands = phases["bands"]
    if len(bands) >= 2:
        first, last = bands[0], bands[-1]
        for key in [m[0] for m in CORE_METRICS + EXTRA_METRICS]:
            m = last["metrics"].get(key, {})
            c = m.get("vs_first", {})
            if not c.get("significant"):
                continue
            delta = c["delta"]
            strength = c.get("strength") or 1.0
            up, down = FINDING_TITLE.get(key, ("%s rose" % labels[key],
                                               "%s fell" % labels[key]))
            # Where did the change actually happen?
            when = None
            best_step = 0.0
            for b in bands[1:]:
                step = b["metrics"].get(key, {}).get("vs_prev", {}).get("delta")
                if step is None:
                    continue
                if abs(step) > abs(best_step):
                    best_step, when = step, b
            rise, fall = METRIC_PROSE.get(key, ("it rose", "it fell"))
            ev = ["%s: %s in gen %d-%d -> %s in gen %d-%d (%s%s, %.1fx the 2-sigma "
                  "within-band noise band of %s)" % (
                      labels[key], fnum(first["metrics"][key]["mean"]),
                      first["gen_lo"], first["gen_hi"], fnum(m["mean"]),
                      last["gen_lo"], last["gen_hi"],
                      "+" if delta > 0 else "", fnum(delta), strength,
                      fnum(c.get("threshold")))]
            if when is not None:
                ev.append("The single biggest step was into gen %d-%d (%s%s)."
                          % (when["gen_lo"], when["gen_hi"],
                             "+" if best_step > 0 else "", fnum(best_step)))
            # First band in which the change cleared the noise bar: the "when".
            onset = None
            for b in bands[1:]:
                if b["metrics"].get(key, {}).get("vs_first", {}).get("significant"):
                    onset = b
                    break
            if onset is not None:
                ev.append("It first cleared the noise bar in gen %d-%d (%s)."
                          % (onset["gen_lo"], onset["gen_hi"], onset["label"]))
            base = first["metrics"][key]["mean"]
            rel = abs(delta / base) if base else None
            findings.append({
                "key": key,
                "title": up if delta > 0 else down,
                "claim": (rise if delta > 0 else fall).capitalize() + ".",
                "evidence": ev,
                "delta": delta, "threshold": c.get("threshold"),
                "strength": strength,
                "score": finding_score(strength, STRATEGY_WEIGHT.get(key, 1.0), rel),
                "onset_band": onset["index"] if onset is not None else None,
                "when_band": when["index"] if when is not None else None,
            })

    # Elo is its own finding, ranked by how big the gain is against agent spread.
    # Elo is a "what worked" item only when it actually went up; a flat or
    # falling curve is a limitation, and build_limitations reports it there.
    eb = learning.get("elo_best", {})
    if eb.get("available") and (eb.get("gain") or 0.0) > 0.0:
        spread = learning.get("spread", {}).get("mean") or 50.0
        score = abs(eb["gain"]) / max(1.0, spread) * 2.0
        pl = learning.get("plateau", {})
        ev = ["Best-agent Elo went %s -> %s (%s%s over the run, %s Elo/generation)." % (
            fnum(eb["start"], 1), fnum(eb["end"], 1),
            "+" if eb["gain"] >= 0 else "", fnum(eb["gain"], 1),
            fnum(eb.get("per_gen"), 2))]
        if pl.get("detected"):
            ev.append("The moving-average slope test puts the plateau at generation "
                      "%d (%.0f%% of the way through); only %s Elo arrived after it."
                      % (pl["gen"], 100.0 * (pl.get("frac_through") or 0),
                         fnum(pl.get("gain_after"), 1)))
        elif pl.get("reason"):
            ev.append("Plateau test: %s." % pl["reason"])
        findings.append({
            "key": "elo", "title": "Ratings improved",
            "claim": ("Self-play plus PBT produced a real, measurable strength gain "
                      "against the frozen hall-of-fame anchor."),
            "evidence": ev, "delta": eb["gain"], "threshold": None,
            "strength": score, "score": finding_score(score, 2.2), "when_band": None,
        })

    by_key = {f["key"]: f for f in findings}

    # Opening repertoire: name the moves.  If the band test already flagged the
    # concentration, this is evidence for that finding rather than a new one.
    if openings.get("available") and openings.get("shift"):
        gainers = [m for m in openings["shift"]
                   if m["certain"] and m["delta_conservative"] > 0]
        top = gainers[0] if gainers else None
        ev = list(openings.get("statements", [])[:3])
        host = by_key.get("opening_top_share") or by_key.get("opening_coverage") \
            or by_key.get("opening_entropy_bits")
        if host is not None:
            host["evidence"].extend(ev)
            if top is not None:
                host["title"] = "Built an opening repertoire around %s" % top["san"]
        elif top is not None and top["delta_conservative"] >= 0.05:
            strength = top["delta_conservative"] / 0.05
            findings.append({
                "key": "opening_shift",
                "title": "Settled on %s as White's first move" % top["san"],
                "claim": ("White's first move is no longer uniform: %s moved from "
                          "%.1f%% to %.1f%% of the logged opening histogram (at "
                          "least %+.1f points after allowing for top-8 truncation)."
                          % (top["san"], 100 * top["first_share"],
                             100 * top["last_share"],
                             100 * top["delta_conservative"])),
                "evidence": ev, "delta": top["delta_conservative"], "threshold": None,
                "strength": strength,
                "score": finding_score(strength, 1.1, top["delta_conservative"] /
                                       max(1e-9, top["first_share"] or 1e-9)),
                "when_band": None,
            })

    # Piece specialisation: same treatment.
    es = pieces.get("entropy_summary") or {}
    if es.get("delta") is not None and abs(es["delta"]) >= 0.05:
        specialised = es["delta"] < 0
        detail = ["%s -> %s" % (p["piece"], p["top_last"][0]["sq"])
                  for p in pieces.get("pieces", []) if p["top_last"]]
        ev = (["Favourite destination by the final band: " + ", ".join(detail)]
              if detail else [])
        ev.append("Mean per-piece destination entropy %s -> %s bits of a possible "
                  "6.0 (uniform over 64 squares)."
                  % (fnum(es["first"], 2), fnum(es["last"], 2)))
        host = by_key.get("piece_entropy_bits")
        if host is not None:
            host["evidence"].extend(ev)
        else:
            strength = abs(es["delta"]) / 0.05
            findings.append({
                "key": "piece_specialisation",
                "title": ("Pieces specialised onto favourite squares" if specialised
                          else "Piece placement became more diffuse"),
                "claim": ("Mean per-piece destination entropy moved %s%s bits "
                          "(%s -> %s of a possible 6.0)." % (
                              "+" if es["delta"] > 0 else "", fnum(es["delta"], 2),
                              fnum(es["first"], 2), fnum(es["last"], 2))),
                "evidence": ev, "delta": es["delta"], "threshold": None,
                "strength": strength,
                "score": finding_score(strength, 1.1,
                                       abs(es["delta"]) / max(1e-9, es["first"] or 1)),
                "when_band": None,
            })

    findings.sort(key=lambda f: -f["score"])
    # Drop findings that restate a stronger one (draw rate vs decisive rate ...).
    kept, taken = [], set()
    for f in findings:
        group = next((g for g in DEDUPE_GROUPS if f["key"] in g), None)
        gid = tuple(sorted(group)) if group else None
        if gid is not None and gid in taken:
            continue
        if gid is not None:
            taken.add(gid)
        kept.append(f)
    findings = kept[:8]
    for i, f in enumerate(findings):
        f["rank"] = i + 1
    if len(findings) < 5:
        caveats.append("Only %d data-supported findings cleared the significance bar; "
                       "the run is either short, noisy or genuinely static."
                       % len(findings))
    return findings


def build_limitations(rows, phases, learning, openings, pieces, throughput, caveats):
    lim = []
    n = len(rows)

    def add(title, detail, evidence=None):
        lim.append({"title": title, "detail": detail, "evidence": evidence or []})

    if n < 8:
        add("The run is too short to conclude much",
            "Only %d generations were logged. The plateau test needs at least 8 and "
            "the band analysis needs at least 2 generations per band, so most claims "
            "below are reported as untestable rather than asserted." % n)

    bands = phases["bands"]
    last = bands[-1] if bands else None

    def endval(key):
        return (last["metrics"].get(key, {}).get("mean") if last else None)

    castle = endval("castle_rate")
    if castle is not None and castle < 0.5:
        add("Never learned to castle reliably",
            "The final band still averages %s castles per game out of a possible 2.0. "
            "King safety is not part of this population's repertoire." % fnum(castle))
    elif castle is not None and castle < 1.5:
        add("Castling is still not automatic",
            "The final band averages %s castles per game; two sides castling every "
            "game would be 2.0. Roughly %.0f%% of the time a player still leaves its "
            "king in the centre." % (fnum(castle), 100.0 * (1.0 - castle / 2.0)))
    promo = endval("promo_rate")
    mp_end = endval("maxplies_share")
    if promo is not None and promo < 0.1:
        extra = ""
        if mp_end is not None and mp_end > 0.2:
            extra = (" Combined with %.0f%% of games hitting the ply cap, that points "
                     "at pawn play that never becomes purposeful."
                     % (100.0 * mp_end))
        add("Almost never promotes a pawn",
            "Final-band promotion rate is %s per game. Passed pawns and the promotion "
            "race are effectively outside what these agents do.%s" % (fnum(promo),
                                                                     extra))
    cm = endval("checkmate_share")
    if cm is not None and cm < 0.25:
        add("Rarely finishes with a checkmate",
            "Only %.1f%% of terminations in the final band are checkmate; the rest are "
            "adjudicated, drawn or cut off by the ply cap. Whatever these agents do "
            "well, delivering mate is not it." % (100.0 * cm))
    mp = endval("maxplies_share")
    if mp is not None and mp > 0.30:
        add("A large share of games never resolve",
            "%.1f%% of games in the final band end by hitting the ply cap. That is a "
            "shuffling failure mode, not a chess result." % (100.0 * mp))
    dr = endval("draw_rate")
    if dr is not None and dr > 0.60:
        add("Most games are still draws",
            "The final band draws %.1f%% of its games. Whatever was learned, it was "
            "not the ability to beat an equally trained opponent." % (100.0 * dr))
    fm = endval("avg_final_material")
    if fm is not None and fm > 50.0:
        add("Games end long before the endgame",
            "Average final material is %s pawns, close to a full starting army of 78. "
            "These agents essentially never reach an endgame, so nothing they learned "
            "is endgame knowledge." % fnum(fm, 1))

    pl = learning.get("plateau", {})
    if pl.get("detected") and (pl.get("frac_through") or 1.0) < 0.6:
        add("Learning stopped well before the run did",
            "The slope test places the plateau at generation %d, %.0f%% of the way in; "
            "the remaining %d generations bought %s Elo. Compute after that point was "
            "largely wasted." % (pl["gen"], 100.0 * pl["frac_through"],
                                 pl.get("gens_after", 0), fnum(pl.get("gain_after"), 1)))
    eb = learning.get("elo_best", {})
    if eb.get("available") and eb.get("gain") is not None and eb["gain"] <= 0:
        add("No measurable strength gain",
            "Best-agent Elo finished at %s having started at %s. Against the frozen "
            "hall-of-fame anchor that is not improvement."
            % (fnum(eb["end"], 1), fnum(eb["start"], 1)))

    if openings.get("available"):
        lastb = openings["bands"][-1]
        share = lastb.get("top_share")
        if share is not None and share < 0.20:
            add("No real opening repertoire",
                "In the final band the most played first move is only %.1f%% of the "
                "logged histogram, close to the 1/20 = 5%% of uniform play. White "
                "never committed to an opening." % (100.0 * share))
    else:
        add("Opening behaviour is unmeasurable",
            openings.get("note") or "opening_top is missing from the telemetry.")

    es = pieces.get("entropy_summary") or {}
    if es.get("delta") is not None and es["delta"] >= -0.05:
        add("Piece placement never specialised",
            "Mean destination entropy moved by %s bits across the run (%s -> %s of a "
            "possible 6.0). The agents move pieces to essentially arbitrary squares."
            % (fnum(es["delta"], 2), fnum(es.get("first"), 2), fnum(es.get("last"), 2)))
    if not pieces.get("available") and not pieces.get("logged_entropy"):
        add("Piece behaviour is unmeasurable",
            pieces.get("note") or "piece_dest is missing from the telemetry.")

    # Everything that was tested and came back noise is a limitation too.
    noisy = sorted({s["metric"] for s in phases["statements"]
                    if s.get("kind") == "noise" and s.get("core")})
    signal = {s["metric"] for s in phases["statements"] if s.get("significant")}
    flat = [m for m in noisy if m not in signal]
    if flat:
        labels = phases["metric_labels"]
        add("Measured, moved, and it was noise",
            "Across all four bands these never moved by more than twice the "
            "within-band standard deviation: %s. Whatever the population learned, it "
            "did not show up here." % ", ".join(labels.get(m, m) for m in flat))

    q = learning.get("gain_by_quarter") or []
    if q and q[-1].get("share") is not None and 0.0 <= q[-1]["share"] < 0.10 \
            and (eb.get("gain") or 0.0) > 10.0 \
            and not any(l["title"].startswith("Learning stopped") for l in lim):
        add("Diminishing returns at the end of the run",
            "The final quarter of the run produced only %.0f%% of the total Elo gain "
            "(%s Elo). More generations of the same recipe would buy very little."
            % (100.0 * q[-1]["share"], fnum(q[-1]["gain"], 1)))

    if throughput.get("games_per_sec") is not None and \
            not throughput.get("meets_1000_gps"):
        add("Throughput target not met",
            "The run averaged %s games/sec, below the 1000 games/sec design target, so "
            "the population saw less experience than the algorithm assumes."
            % fnum(throughput["games_per_sec"], 1))

    # Scope limits that hold for every run, however well it went.
    add("Nothing here measures absolute strength",
        "Elo is computed inside this run, against the population and against frozen "
        "hall-of-fame snapshots of the same population. It anchors the scale but it "
        "is not human Elo and not comparable to any external engine. A rating of %s "
        "means 'this far above its own ancestors', nothing more."
        % fnum((learning.get("elo_best") or {}).get("end"), 0))
    add("The telemetry counts events, not move quality",
        "Every claim above is built from counters: captures, castles, promotions, "
        "termination reasons, destination squares. Nothing in the run logs blunder "
        "rate, tactical accuracy or evaluation error, so this report cannot and does "
        "not claim the agents calculate, avoid hanging pieces, or understand the "
        "positions they reach.")
    for c in caveats:
        add("Data limitation", c)
    return lim


# ------------------------------------------------------------------ assembly

def build_series(rows):
    keys = ["gen", "elo_best", "elo_mean", "elo_p10", "draw_rate", "white_win",
            "black_win", "decisive_rate", "avg_len", "captures_per_game",
            "checks_per_game", "castle_rate", "promo_rate", "ep_rate",
            "avg_final_material", "gps", "sec", "games", "plies", "loss_policy",
            "loss_value", "loss_entropy", "grad_norm", "checkmate_share",
            "maxplies_share", "opening_top_share", "opening_coverage",
            "opening_entropy_bits", "piece_entropy_bits", "centre_share"]
    out = {k: [r.get(k) for r in rows] for k in keys}
    out["elo_spread"] = [(b - p) if (b is not None and p is not None) else None
                         for b, p in zip(out["elo_best"], out["elo_p10"])]
    out["term"] = {k: [r["term"].get(k) for r in rows] for k in TERM_KEYS}
    # Short aliases so the web view can use either name.
    out["captures"] = out["captures_per_game"]
    out["checks"] = out["checks_per_game"]
    out["draw"] = out["draw_rate"]
    return out


REQUIRED_FIELDS = ["gen", "games", "plies", "sec", "elo_best", "elo_mean", "elo_p10",
                   "draw_rate", "avg_len", "captures_per_game", "checks_per_game",
                   "castle_rate", "promo_rate", "ep_rate", "avg_final_material",
                   "grad_norm"]


def build_report(run_dir, tele_path, nb=4):
    raw, bad = load_rows(tele_path)
    if not raw:
        raise SystemExit("report.py: %s contains no usable telemetry" % tele_path)
    rows = [canon_row(r) for r in raw]
    # Order by generation; unnumbered rows keep their file order at the end.
    numbered = [r for r in rows if r.get("gen") is not None]
    unnumbered = [r for r in rows if r.get("gen") is None]
    numbered.sort(key=lambda r: r["gen"])
    for i, r in enumerate(unnumbered):
        r["gen"] = (numbered[-1]["gen"] + 1 + i) if numbered else float(i)
    rows = numbered + unnumbered
    gens = [int(r["gen"]) for r in rows]
    for r, g in zip(rows, gens):
        r["gen"] = g

    caveats = []
    if bad:
        caveats.append("%d telemetry line(s) could not be parsed and were skipped."
                       % bad)
    missing = []
    for f in REQUIRED_FIELDS:
        if not any(r.get(f) is not None for r in rows):
            missing.append(f)
    for f, label in (("opening", "opening_top"), ("piece_dest", "piece_dest"),
                     ("best_hypers", "best_agent")):
        if not any(r.get(f) for r in rows):
            missing.append(label)
    if not any(any(v is not None for v in r["term"].values()) for r in rows):
        missing.append("term")
    if missing:
        caveats.append("Fields never present in the telemetry: %s. Every section that "
                       "needed them says so instead of guessing."
                       % ", ".join(sorted(set(missing))))
    if len(rows) < nb * 2:
        caveats.append("Only %d generations were logged; with %d bands there are "
                       "fewer than two generations per band, so within-band noise "
                       "cannot be estimated and most deltas are reported as "
                       "untestable." % (len(rows), nb))

    pd_mode = diff_if_cumulative(rows)
    learning = analyse_learning(rows, gens)
    throughput = analyse_throughput(rows)
    phases = analyse_phases(rows, gens, nb)
    openings = analyse_openings(rows, gens, nb)
    pieces = analyse_pieces(rows, gens, nb, pd_mode)
    champion = analyse_champion(rows, run_dir)
    findings = build_findings(phases, learning, openings, pieces, caveats)
    limitations = build_limitations(rows, phases, learning, openings, pieces,
                                    throughput, caveats)

    rep = {
        "schema": SCHEMA,
        "generated_at": datetime.datetime.now(datetime.timezone.utc)
                                 .strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generator": "py/report.py",
        "run": {
            "name": os.path.basename(os.path.abspath(run_dir.rstrip("/"))),
            "dir": os.path.abspath(run_dir),
            "telemetry": os.path.abspath(tele_path),
            "generations": len(rows),
            "gen_first": gens[0], "gen_last": gens[-1],
            "lines_read": len(raw), "lines_bad": bad,
            "missing_fields": sorted(set(missing)),
            "caveats": caveats,
        },
        "squares": SQUARES,
        "series": build_series(rows),
        "learning": learning,
        "throughput": throughput,
        "phases": phases,
        "openings": openings,
        "pieces": pieces,
        "champion": champion,
        "findings": findings,
        "limitations": limitations,
    }
    return rep


# ------------------------------------------------------------------ markdown

def md_table(headers, rows_):
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows_:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def hhmmss(sec):
    if sec is None:
        return "n/a"
    sec = int(round(sec))
    return "%dh %02dm %02ds" % (sec // 3600, (sec % 3600) // 60, sec % 60)


def render_markdown(rep):
    R = rep["run"]
    L = rep["learning"]
    T = rep["throughput"]
    P = rep["phases"]
    O = rep["openings"]
    PC = rep["pieces"]
    C = rep["champion"]
    out = []
    w = out.append

    w("# Strategy report -- %s" % R["name"])
    w("")
    w("Generated %s from %d generations (gen %d-%d) of `%s`."
      % (rep["generated_at"], R["generations"], R["gen_first"], R["gen_last"],
         os.path.relpath(R["telemetry"], R["dir"]) if R["dir"] in R["telemetry"]
         else R["telemetry"]))
    w("")
    w("Every number below comes from the telemetry. Where the data cannot support a "
      "claim, this report says so rather than making one. The raw per-generation "
      "series live in `report.json`; the charts are the web UI's job.")
    if R["caveats"]:
        w("")
        w("Data caveats:")
        for c in R["caveats"]:
            w("- %s" % c)
    w("")

    # --------------------------------------------------------- 1. learning
    w("## 1. Learning curve")
    w("")
    rows_ = []
    for name, label in (("elo_best", "best agent"), ("elo_mean", "population mean"),
                        ("elo_p10", "10th percentile")):
        d = L.get(name, {})
        if not d.get("available"):
            rows_.append([label, "not logged", "-", "-", "-", "-"])
            continue
        rows_.append([label, fnum(d["start"], 1), fnum(d["end"], 1),
                      ("+" if d["gain"] >= 0 else "") + fnum(d["gain"], 1),
                      fnum(d["peak"], 1) + " @ gen %d" % d["peak_gen"],
                      fnum(d["per_gen"], 2)])
    w(md_table(["agent", "start Elo", "end Elo", "gain", "peak", "Elo/gen"], rows_))
    w("")
    bands = P["bands"]
    if bands and any(b["metrics"].get("elo_mean", {}).get("mean") is not None
                     for b in bands):
        rows_ = []
        for b in bands:
            m = b["metrics"]
            rows_.append(["B%d" % (b["index"] + 1),
                          "%d-%d" % (b["gen_lo"], b["gen_hi"]),
                          fnum(m.get("elo_mean", {}).get("mean"), 1)])
        w("Population mean Elo by band (the per-generation best/mean/p10 series are "
          "in `report.json` under `series`):")
        w("")
        w(md_table(["band", "generations", "mean Elo"], rows_))
        w("")
    sp = L.get("spread", {})
    if sp.get("start") is not None:
        w("Spread between the best agent and the 10th percentile went from %s to %s "
          "Elo (mean %s): %s."
          % (fnum(sp["start"], 1), fnum(sp["end"], 1), fnum(sp["mean"], 1),
             "the population spread out, so evolution had more to select on"
             if (sp["end"] or 0) > (sp["start"] or 0) else
             "the population converged, so evolution had less to select on"))
        w("")
    q = L.get("gain_by_quarter") or []
    total_gain = (L.get("elo_best") or {}).get("gain")
    if q:
        shares_ok = total_gain is not None and total_gain > 10.0
        w("Where the Elo actually arrived, by quarter of the run: " +
          ", ".join("Q%d %s%s Elo%s" % (
              d["index"] + 1, "+" if d["gain"] >= 0 else "", fnum(d["gain"], 1),
              (" (%.0f%% of the total)" % (100 * d["share"]))
              if (shares_ok and d.get("share") is not None) else "")
              for d in q) + ".")
        if not shares_ok:
            w("")
            w("(Shares of the total are omitted: the run's net Elo change is too small "
              "for them to mean anything.)")
        w("")
    pl = L.get("plateau", {})
    if pl.get("detected"):
        w("**Plateau.** A centred moving average (window %d) followed by a trailing "
          "least-squares slope over %d generations puts the plateau at generation "
          "**%d** -- %.0f%% of the way through the run. After that the slope never "
          "again exceeds %s Elo/generation (the larger of 20%% of the peak early "
          "slope and the slope explainable by per-generation noise), and only %s Elo "
          "arrives in the remaining %d generations."
          % (pl.get("ma_window", 0), pl.get("slope_window", 0), pl["gen"],
             100.0 * (pl.get("frac_through") or 0), fnum(pl.get("threshold"), 3),
             fnum(pl.get("gain_after"), 1), pl.get("gens_after", 0)))
    else:
        w("**Plateau.** None detected: %s." % (pl.get("reason") or "test unavailable"))
    w("")

    # ------------------------------------------------------- 2. throughput
    w("## 2. Throughput")
    w("")
    rows_ = [
        ["total games", fint(T["total_games"])],
        ["total plies (positions)", fint(T["total_plies"])],
        ["wall clock", hhmmss(T["wall_seconds"])],
        ["games / sec", fint(T["games_per_sec"])],
        ["positions / sec", fint(T["positions_per_sec"])],
        ["plies / game", fnum(T["plies_per_game"], 1)],
        ["peak logged games/sec", fint(T["gps_logged_peak"])],
    ]
    w(md_table(["metric", "value"], rows_))
    w("")
    if T.get("speedup_vs_human") is not None:
        w("At 40 minutes per human game, those %s games are %s years of continuous "
          "human play (%s seconds). The run did it in %s, a speedup of **%sx** over "
          "real chess."
          % (fint(T["total_games"]), fnum(T["human_equivalent_years"], 1),
             fint(T["human_equivalent_seconds"]), hhmmss(T["wall_seconds"]),
             fint(T["speedup_vs_human"])))
    else:
        w("Games or wall-clock time were not logged, so the speedup versus real chess "
          "cannot be computed.")
    if T.get("games_per_sec") is not None:
        w("")
        w("Design target is >1000 games/sec on 8 cores: **%s**."
          % ("met" if T["meets_1000_gps"] else "not met"))
    w("")

    # ------------------------------------------------------ 3. phases
    w("## 3. Phase analysis (four equal generation bands)")
    w("")
    bands = P["bands"]
    hdr = ["metric"] + ["B%d: gen %d-%d" % (b["index"] + 1, b["gen_lo"], b["gen_hi"])
                        for b in bands] + ["delta B1->B%d" % len(bands),
                                           "2-sigma band", "verdict"]
    rows_ = []
    for key in P["core_metrics"]:
        label = P["metric_labels"][key]
        cells = [fnum(b["metrics"].get(key, {}).get("mean")) for b in bands]
        c = bands[-1]["metrics"].get(key, {}).get("vs_first", {})
        d = c.get("delta")
        verdict = ("signal" if c.get("significant") else
                   ("noise" if c.get("threshold") is not None else "untestable"))
        rows_.append([label] + cells +
                     [fdelta(d),
                      fnum(c.get("threshold")), verdict])
    w(md_table(hdr, rows_))
    w("")
    w("Verdict rule: a band-to-band change counts as signal only when it exceeds "
      "twice the larger of the two bands' generation-to-generation standard "
      "deviations. Anything else is labelled noise, explicitly.")
    w("")
    w("Supporting metrics (same test):")
    w("")
    rows_ = []
    for key in P["extra_metrics"]:
        label = P["metric_labels"][key]
        if all(b["metrics"].get(key, {}).get("mean") is None for b in bands):
            continue
        cells = [fnum(b["metrics"].get(key, {}).get("mean")) for b in bands]
        c = bands[-1]["metrics"].get(key, {}).get("vs_first", {})
        d = c.get("delta")
        verdict = ("signal" if c.get("significant") else
                   ("noise" if c.get("threshold") is not None else "untestable"))
        rows_.append([label] + cells +
                     [fdelta(d),
                      fnum(c.get("threshold")), verdict])
    if rows_:
        w(md_table(hdr, rows_))
    else:
        w("(none of the supporting metrics were logged)")
    w("")
    w("### What the population learned, and when")
    w("")
    sig = [s for s in P["statements"] if s.get("significant") and s.get("core")]
    if sig:
        order = [m[0] for m in CORE_METRICS]
        for key in order:
            hits = [s for s in sig if s["metric"] == key]
            if not hits:
                continue
            onset = min(hits, key=lambda s: s["band"])
            final = max(hits, key=lambda s: s["band"])
            ob = bands[onset["band"]]
            fb = bands[final["band"]]
            rise, fall = METRIC_PROSE.get(key, ("it rose", "it fell"))
            steps = []
            for b in bands[1:]:
                d = b["metrics"].get(key, {}).get("vs_prev", {}).get("delta")
                if d is not None:
                    steps.append((abs(d), b, d))
            # key= on the magnitude only: two bands can tie on abs(delta), and the
            # tuple fallback would then compare the band dicts and raise TypeError.
            big = max(steps, key=lambda s: s[0])[1:] if steps else (None, None)
            when = ("the biggest single step was into gen %d-%d (%s%s)"
                    % (big[0]["gen_lo"], big[0]["gen_hi"],
                       "+" if big[1] > 0 else "", fnum(big[1]))) if big[0] else ""
            w("- **%s**: %s (gen %d-%d) -> %s (gen %d-%d), a change of %s%s = %.1fx "
              "the 2-sigma noise band. It first cleared that bar in gen %d-%d, and %s. "
              "In plain terms: %s."
              % (P["metric_labels"][key],
                 fnum(bands[0]["metrics"][key]["mean"]),
                 bands[0]["gen_lo"], bands[0]["gen_hi"],
                 fnum(fb["metrics"][key]["mean"]), fb["gen_lo"], fb["gen_hi"],
                 "+" if final["delta"] > 0 else "", fnum(final["delta"]),
                 final.get("strength") or 1.0, ob["gen_lo"], ob["gen_hi"], when,
                 rise if final["delta"] > 0 else fall))
    else:
        w("- Nothing in the core phase metrics moved by more than twice the "
          "within-band noise. On this evidence the population's *behaviour* did not "
          "measurably change, whatever happened to its rating.")
    w("")
    noise_stmts = [s for s in P["statements"]
                   if s.get("core") and s.get("kind") in ("noise", "untestable")
                   and s.get("band") == len(bands) - 1]
    if noise_stmts:
        w("Explicitly *not* learned (change smaller than the noise band):")
        w("")
        for s in noise_stmts:
            w("- %s" % s["text"])
        w("")

    # ------------------------------------------------------ 4. openings
    w("## 4. Opening repertoire (White's first move)")
    w("")
    if not O.get("available"):
        w(O.get("note") or "Not logged.")
    else:
        obands = O["bands"]
        maxrows = max(len(b["top"]) for b in obands)
        hdr = ["rank"] + ["B%d: gen %d-%d" % (b["index"] + 1, b["gen_lo"], b["gen_hi"])
                          for b in obands]
        rows_ = []
        for i in range(min(6, maxrows)):
            cells = []
            for b in obands:
                if i < len(b["top"]):
                    t = b["top"][i]
                    cells.append("%s (%s) %.1f%%" % (t["san"], t["uci"],
                                                     100.0 * (t["share"] or 0)))
                else:
                    cells.append("-")
            rows_.append([str(i + 1)] + cells)
        w(md_table(hdr, rows_))
        w("")
        rows_ = [["top-move share"] + ["%.1f%%" % (100.0 * (b["top_share"] or 0))
                                       for b in obands],
                 ["entropy of logged top-8 (bits)"] +
                 [fnum(b["entropy_bits"], 2) for b in obands],
                 ["distinct first moves logged"] + [str(b["n_distinct"])
                                                    for b in obands]]
        w(md_table(["measure"] + ["B%d" % (b["index"] + 1) for b in obands], rows_))
        w("")
        for s in O.get("statements", []):
            w("- %s" % s)
        w("")
        w("Note: only the top-8 histogram is logged per generation, so these shares "
          "are shares of the logged histogram, not of all 20 legal first moves. A "
          "uniform White would put ~5% on each.")
    w("")

    # ------------------------------------------------------ 5. pieces
    w("## 5. Piece behaviour")
    w("")
    if not PC.get("available"):
        w(PC.get("note") or "Not logged.")
        for s in PC.get("statements", []):
            w("")
            w("- %s" % s)
    else:
        fb, lb = PC["first_band"], PC["last_band"]
        w("Destination histograms, first band (gen %d-%d) versus last band "
          "(gen %d-%d). Source data was %s."
          % (fb["gen_lo"], fb["gen_hi"], lb["gen_lo"], lb["gen_hi"],
             "per-generation counts" if PC["mode"] == "per_generation"
             else "cumulative and has been differenced"))
        w("")
        rows_ = []
        for p in PC["pieces"]:
            tf = ", ".join(t["sq"] for t in p["top_first"][:3]) or "-"
            tl = ", ".join(t["sq"] for t in p["top_last"][:3]) or "-"
            rows_.append([
                p["piece"], tf, tl,
                fnum(p["entropy_first_bits"], 2), fnum(p["entropy_last_bits"], 2),
                ("+" if (p["entropy_delta"] or 0) >= 0 else "") +
                fnum(p["entropy_delta"], 2),
                "%.1f%%" % (100.0 * (p["centre_share_first"] or 0)),
                "%.1f%%" % (100.0 * (p["centre_share_last"] or 0)),
            ])
        w(md_table(["piece", "favourite squares (first)", "favourite squares (last)",
                    "entropy first", "entropy last", "delta",
                    "centre share first", "centre share last"], rows_))
        w("")
        es = PC.get("entropy_summary", {})
        if es.get("delta") is not None:
            if es["delta"] < -0.02:
                reading = ("Lower entropy means the population specialised: pieces "
                           "went to particular squares rather than wherever was legal.")
            elif es["delta"] > 0.02:
                reading = ("Higher entropy means piece placement became *less* "
                           "specialised over the run, which is what an unlearning or "
                           "over-exploring population looks like.")
            else:
                reading = ("That is no measurable change: piece destinations are as "
                           "close to uniform at the end as at the start, so this run "
                           "shows no piece specialisation at all.")
            w("Mean destination entropy across piece types: %s -> %s bits (%s%s) "
              "against a ceiling of 6.0 bits for a uniform 64-square distribution. %s"
              % (fnum(es["first"], 2), fnum(es["last"], 2),
                 "+" if es["delta"] > 0 else "", fnum(es["delta"], 2), reading))
            w("")
        for s in PC.get("statements", []):
            w("- %s" % s)
        w("")
        w("The full 6 x 64 normalised heatmaps for both bands (and their difference) "
          "are in `report.json` under `pieces.heatmap_first`, `pieces.heatmap_last` "
          "and `pieces.heatmap_delta`, indexed by `squares` (a1 = 0 .. h8 = 63).")
    w("")

    # ------------------------------------------------------ 6. champion
    w("## 6. Style of the champion")
    w("")
    if not C.get("available"):
        w(C.get("note") or "Not logged.")
    else:
        w("Champion: agent %s at generation %s, Elo %s."
          % (C.get("agent", "?"), C.get("gen", "?"), fnum(C.get("elo"), 1)))
        w("")
        w("Baseline: %s." % C["baseline"]["source"])
        if C.get("baseline_is_proxy"):
            w("")
            w("This is a proxy baseline. The true population mean was not in the "
              "telemetry, so treat the comparison as 'this champion versus the "
              "champions this run produced', not 'versus its 255 contemporaries'.")
        w("")
        rows_ = []
        for c in C["comparison"]:
            rows_.append([c["name"], fnum(c["value"]), fnum(c["baseline"]),
                          fnum(c["ratio"], 2) + "x" if c["ratio"] else "n/a",
                          fnum(c["z"], 2) if c["z"] is not None else "n/a"])
        w(md_table(["hyper-parameter", "champion", "baseline", "ratio", "z"], rows_))
        w("")
        w(C.get("summary", ""))
        w("")
        for s in C.get("statements", []):
            w("- %s" % s)
        drift = C.get("drift") or {}
        moved = sorted(
            ((k, d) for k, d in drift.items()
             if d.get("start") is not None and d.get("end") is not None
             and d["start"] and abs(d["end"] - d["start"]) / abs(d["start"]) > 0.10),
            key=lambda kv: -abs(kv[1]["end"] - kv[1]["start"]) / abs(kv[1]["start"]))
        if moved:
            w("")
            w("Champion hyper-parameters drifted across the run, which is evolution "
              "selecting a style rather than one lucky agent: " +
              ", ".join("%s %s -> %s" % (k, fnum(d["start"]), fnum(d["end"]))
                        for k, d in moved[:3]) + ".")
    w("")

    # ------------------------------------------------------ 7. findings
    w("## 7. What actually worked")
    w("")
    if not rep["findings"]:
        w("Nothing in this run cleared the significance bar. On this telemetry there "
          "is no defensible claim to make about what the population learned.")
    else:
        w("Ranked by effect size against the run's own noise, strategy metrics "
          "weighted ahead of optimiser bookkeeping.")
        w("")
        for f in rep["findings"]:
            w("**%d. %s**" % (f["rank"], f["title"]))
            w("")
            w(f["claim"])
            w("")
            for e in f.get("evidence", []):
                w("- %s" % e)
            w("")
    w("## What it never learned")
    w("")
    if not rep["limitations"]:
        w("No limitation was detectable from the logged fields, which most likely "
          "means the telemetry is too thin rather than that the agents are complete.")
    else:
        for l in rep["limitations"]:
            w("**%s.** %s" % (l["title"], l["detail"]))
            for e in l.get("evidence", []):
                w("")
                w("- %s" % e)
            w("")
    w("")
    w("---")
    w("")
    w("Report generated by `py/report.py` (schema `%s`). Machine-readable copy with "
      "every per-generation series: `report.json`." % rep["schema"])
    w("")
    return "\n".join(out)


# ---------------------------------------------------------------------- main

def resolve_paths(target):
    """Accept a run directory or a telemetry file."""
    if os.path.isdir(target):
        return target, os.path.join(target, "telemetry.jsonl")
    if os.path.isfile(target):
        return os.path.dirname(os.path.abspath(target)) or ".", target
    raise SystemExit("report.py: no such run directory or telemetry file: %s" % target)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Turn a ChessRL training run into a strategy analysis.")
    ap.add_argument("run", help="run directory (runs/NAME) or a telemetry.jsonl file")
    ap.add_argument("--out", default=None,
                    help="report JSON path (default: <run>/report.json)")
    ap.add_argument("--md", default=None,
                    help="Markdown report path (default: <run>/REPORT.md)")
    ap.add_argument("--bands", type=int, default=4,
                    help="number of generation bands for the phase analysis "
                         "(default 4)")
    ap.add_argument("--quiet", action="store_true", help="write files, say nothing")
    args = ap.parse_args(argv)

    if args.bands < 1:
        raise SystemExit("report.py: --bands must be >= 1")
    run_dir, tele = resolve_paths(args.run)
    if not os.path.isfile(tele):
        raise SystemExit("report.py: telemetry not found: %s" % tele)

    rep = build_report(run_dir, tele, args.bands)
    out_json = args.out or os.path.join(run_dir, "report.json")
    out_md = args.md or os.path.join(run_dir, "REPORT.md")
    for path in (out_json, out_md):
        d = os.path.dirname(os.path.abspath(path))
        if d and not os.path.isdir(d):
            os.makedirs(d, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=1, sort_keys=False, allow_nan=False)
        fh.write("\n")
    with open(out_md, "w", encoding="utf-8") as fh:
        fh.write(render_markdown(rep))
    if not args.quiet:
        eb = rep["learning"].get("elo_best", {})
        sys.stdout.write(
            "report.py: %d generations, %s findings, %s limitations -> %s, %s\n"
            % (rep["run"]["generations"], len(rep["findings"]),
               len(rep["limitations"]), out_json, out_md))
        if eb.get("available"):
            sys.stdout.write("           best Elo %s -> %s (%s%s)\n" % (
                fnum(eb["start"], 1), fnum(eb["end"], 1),
                "+" if eb["gain"] >= 0 else "", fnum(eb["gain"], 1)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
