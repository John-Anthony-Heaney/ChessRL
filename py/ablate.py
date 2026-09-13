#!/usr/bin/env python3
"""ablate.py -- multi-seed, compute-matched ablation for the AlphaZero trainer.

WHY THIS EXISTS
===============
Deep RL results are seed-sensitive.  Henderson et al., "Deep Reinforcement
Learning that Matters" (AAAI 2018, arXiv:1709.06560), showed that running the
SAME algorithm with different random seeds can produce a spread larger than the
gap between the algorithms people were publishing.  A sweep that runs one seed
per configuration and ranks the results is therefore not measuring the
configuration; it is, with high probability, measuring the seed.

Every hyper-parameter sweep in this project so far used one seed per
configuration.  This tool replaces that methodology.  It rests on three rules.

1. COMPUTE-MATCHED COMPARISON.  Configurations are compared at an equal total
   budget, never at an equal number of generations.  8 agents playing 6 games
   each is 24 games per generation; 128 agents is 384.  Comparing those at the
   same generation count compares 16x different amounts of experience, so any
   difference found is a difference in compute, not in population size.  The
   generation count is DERIVED from the budget, per level, and printed.  Where
   the factor changes the cost of a game (`sims`, `cap-frac`), the budget is
   matched on total network EVALUATIONS instead of games; the basis is an
   explicit choice (`--match`) and is reported with the result.

2. EXTERNAL EVALUATION.  The trainer's internal Elo is anchored to each run's
   own hall of fame, so two runs' internal Elos are not on the same scale and
   comparing them would invalidate the experiment.  Every finished model is
   instead scored against a FIXED COMMON OPPONENT with `py/benchmark.py`, in
   colour-reversed pairs over a fixed opening set, and it is that score which
   is compared.

3. SEED VARIANCE IS THE HEADLINE.  Each level is run with `--seeds`
   independent training seeds.  The report leads with the within-level (seed)
   standard deviation, states the between-level effect as a multiple of it,
   gives a confidence interval on every level mean, runs a defensible test
   (Welch's t and an exact permutation test, both implemented here in plain
   Python), and prints the MINIMUM DETECTABLE EFFECT at the sample size used,
   so that a null result is interpretable instead of merely absent.

USAGE
=====
    python3 py/ablate.py --factor agents --levels 8,32,128 --seeds 5 \
                         [--budget-games N] [--out runs/ablation_agents.json] \
                         [--dry-run] [--threads N]

    python3 py/ablate.py --list-factors
    python3 py/ablate.py --factor sims --dry-run          # uses default levels

The run is checkpointed to the output JSON after every individual training run
and every evaluation, so a long experiment survives an interruption: re-run the
same command and it resumes.  Levels are interleaved seed-major, so an
experiment stopped early still has every level at the same number of seeds.

Standard library only.  No numpy, no scipy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import re
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
BINARY = REPO / "build" / "chessrl"
BENCHMARK = HERE / "benchmark.py"
DEFAULT_MD = REPO / "docs" / "ABLATION.md"

MD_MARKER = "<!-- ablate.py appends results below this line -->"


# ==========================================================================
# The base configuration
# ==========================================================================
# Every factor level varies ONE of these and holds the rest fixed.  It is
# recorded verbatim in the output JSON so a result can never be read without
# the configuration that produced it.
#
# Deliberate departures from `az_default_cfg` in src/az.c, each for a reason:
#
#   sims 160     -- docs/ALGORITHM.md measured 48 sims into a 60%-repetition
#                   shuffling equilibrium and 160 into a 5% one.  Running the
#                   ablation at the binary's default of 64 would measure the
#                   degenerate regime, not the system we ship.
#   games-per-agent 6
#                -- 6 games/agent makes games-per-generation = agents*3, i.e.
#                   24 / 96 / 384 at the default agent levels.
#   ema-decay 0  -- with the EMA on, `best.crl` is whichever of the raw and
#                   averaged weights wins an 8-game head-to-head.  That is a
#                   discrete coin flip inserted between training and
#                   measurement; turning it off removes a nuisance variance
#                   that has nothing to do with the factor under test.
#   start mixed  -- az_default_cfg() assigns start_mode BEFORE memset()ing the
#                   struct, so the compiled-in default is silently `classical`
#                   (see the note in docs/ABLATION.md).  Passed explicitly.
BASE_CONFIG = {
    "agents": 32,
    "games-per-agent": 6,
    "sims": 160,
    "cap-frac": 1.0,
    "cap-sims": 0,            # 0 => max(2, sims/5)
    "asym-frac": 0.0,         # asymmetric search budgets, off
    "asym-ratio": 4.0,
    "draw-penalty": -0.1,
    "cpuct": 1.4,
    "lr": 2e-3,
    "lr-final": 2e-4,
    "wd": 1e-4,
    "clip": 4.0,
    "value-coef": 4.0,
    "value-mix": 0.0,
    "buffer": 50000,
    "batch": 256,
    "steps": 200,
    "max-plies": 200,
    "opening-plies": 20,
    "temp-start": 1.0,
    "temp-end": 0.25,
    "dir-alpha": 0.3,
    "dir-eps": 0.25,
    "resign": -0.9,
    "resign-check": 0.1,
    "elite": 0.25,
    "cull": 0.10,
    "hof-every": 10,
    "hof-pct": 15,
    "warmup": 0,
    "ema-decay": 0.0,
    "start": "mixed",
}

# Flags whose value is an integer when written on the command line.
INT_KEYS = {
    "agents", "games-per-agent", "sims", "cap-sims", "buffer", "batch", "steps",
    "max-plies", "opening-plies", "hof-every", "hof-pct", "warmup",
}


class Factor:
    """One thing an ablation may vary, and what varying it costs."""

    def __init__(self, name, flag, cast, blurb, default_levels,
                 changes_games_per_gen=False, changes_cost_per_game=False,
                 changes_learn_cost=False):
        self.name = name
        self.flag = flag                    # the base-config key it overrides
        self.cast = cast
        self.blurb = blurb
        self.default_levels = default_levels
        self.changes_games_per_gen = changes_games_per_gen
        self.changes_cost_per_game = changes_cost_per_game
        self.changes_learn_cost = changes_learn_cost


FACTORS = {
    "agents": Factor(
        "agents", "agents", int, "population size (games/gen scales with it)",
        [8, 32, 128], changes_games_per_gen=True),
    "sims": Factor(
        "sims", "sims", int, "MCTS simulations per move (cost/game scales)",
        [64, 160, 320], changes_cost_per_game=True),
    "draw-penalty": Factor(
        "draw-penalty", "draw-penalty", float, "value assigned to a draw",
        [-0.6, -0.3, -0.1, 0.0]),
    "games-per-agent": Factor(
        "games-per-agent", "games-per-agent", int,
        "games each agent plays per generation (must be even)",
        [2, 6, 18], changes_games_per_gen=True),
    "c-puct": Factor(
        "c-puct", "cpuct", float, "PUCT exploration constant",
        [0.7, 1.4, 2.8]),
    "lr": Factor(
        "lr", "lr", float, "peak learning rate (lr-final scaled with it)",
        [5e-4, 2e-3, 8e-3]),
    "value-coef": Factor(
        "value-coef", "value-coef", float, "value-loss weight vs the policy loss",
        [1.0, 4.0, 16.0]),
    "buffer": Factor(
        "buffer", "buffer", int, "replay buffer capacity, in positions",
        [12500, 50000, 200000]),
    "steps": Factor(
        "steps", "steps", int, "optimiser steps per generation",
        [50, 200, 800], changes_learn_cost=True),
    "cap-frac": Factor(
        "cap-frac", "cap-frac", float,
        "fraction of moves given the full sim budget (playout cap randomisation)",
        [0.25, 0.5, 1.0], changes_cost_per_game=True),
    # ASYMMETRIC SEARCH BUDGETS -- docs/ASYMMETRY.md.  An asymmetric game is
    # cheaper than a symmetric one (one side searches sims/ratio), so this
    # must match on evaluations, not on games.
    "asym-frac": Factor(
        "asym-frac", "asym-frac", float,
        "fraction of self-play games played with unequal simulation budgets",
        [0.0, 0.25, 0.5], changes_cost_per_game=True),
    "asym-ratio": Factor(
        "asym-ratio", "asym-ratio", float,
        "how much less the handicapped side searches (only with asym-frac > 0)",
        [2.0, 4.0, 8.0], changes_cost_per_game=True),
}


# ==========================================================================
# Outcome variables
# ==========================================================================
# The primary outcome is the EXTERNAL score, because that is the only number
# that is comparable across runs.  The secondary outcomes are telemetry from
# the last generation, kept so that a claim originally made about a telemetry
# statistic -- "tripling the draw penalty barely moves the repetition rate" --
# can be re-tested with proper seeds using the same machinery.  A factor can
# move a telemetry statistic without moving strength, and vice versa.

def _tel(rec, *path):
    d = rec.get("final_telemetry") or {}
    for k in path:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d if isinstance(d, (int, float)) else None


METRICS = {
    "score":        ("external score vs the reference opponent",
                     lambda r: r.get("score")),
    "repetition":   ("fraction of games drawn by threefold repetition",
                     lambda r: _tel(r, "term", "repetition")),
    "draw-rate":    ("fraction of self-play games drawn",
                     lambda r: _tel(r, "draw")),
    "captures":     ("captures per self-play game",
                     lambda r: _tel(r, "captures_per_game")),
    "policy-top1":  ("policy argmax agreement with the search",
                     lambda r: _tel(r, "policy_top1")),
    "policy-kl":    ("KL(search visits || policy)",
                     lambda r: _tel(r, "policy_kl")),
    "game-length":  ("average self-play game length in plies",
                     lambda r: _tel(r, "avg_len")),
    "internal-elo": ("the run's OWN internal Elo -- NOT comparable across runs,"
                     " offered only to show that it is not",
                     lambda r: _tel(r, "elo_best")),
}


# ==========================================================================
# Statistics, implemented here because the project has no third-party deps
# ==========================================================================

def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def sdev(xs):
    """Sample standard deviation (ddof=1).  nan for n < 2."""
    n = len(xs)
    if n < 2:
        return float("nan")
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))


def _betacf(a, b, x):
    """Continued fraction for the incomplete beta function (Lentz's method)."""
    TINY, EPS, ITMAX = 1e-300, 3e-16, 300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < TINY:
        d = TINY
    d = 1.0 / d
    h = d
    for m in range(1, ITMAX + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < TINY:
            d = TINY
        c = 1.0 + aa / c
        if abs(c) < TINY:
            c = TINY
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < TINY:
            d = TINY
        c = 1.0 + aa / c
        if abs(c) < TINY:
            c = TINY
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < EPS:
            break
    return h


def betai(a, b, x):
    """Regularised incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
             + a * math.log(x) + b * math.log1p(-x))
    front = math.exp(lbeta)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + b * math.log1p(-x) + a * math.log(x)) * _betacf(b, a, 1.0 - x) / b


def t_sf_two_sided(t, df):
    """P(|T| >= |t|) for Student's t with `df` degrees of freedom."""
    if df <= 0 or not math.isfinite(t):
        return float("nan")
    return betai(0.5 * df, 0.5, df / (df + t * t))


def t_cdf(t, df):
    p = 0.5 * t_sf_two_sided(t, df)
    return 1.0 - p if t > 0 else p


def t_ppf(p, df):
    """Inverse Student-t CDF by bisection.  Plenty accurate for a CI."""
    if not (0.0 < p < 1.0):
        raise ValueError("t_ppf: p must be in (0, 1)")
    lo, hi = -1e4, 1e4
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def welch(a, b):
    """Welch's unequal-variance t-test.  Returns a dict, never raises."""
    na, nb = len(a), len(b)
    out = {"diff": None, "t": None, "df": None, "p": None}
    if na < 2 or nb < 2:
        return out
    ma, mb = mean(a), mean(b)
    va, vb = sdev(a) ** 2, sdev(b) ** 2
    se2 = va / na + vb / nb
    out["diff"] = ma - mb
    if se2 <= 0.0:
        # Both samples constant: either identical (no effect) or separated by
        # an infinite t.  Report it rather than dividing by zero.
        out["p"] = 1.0 if ma == mb else 0.0
        out["t"] = 0.0 if ma == mb else float("inf")
        out["df"] = float(na + nb - 2)
        return out
    se = math.sqrt(se2)
    t = (ma - mb) / se
    num = se2 ** 2
    den = (va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1)
    df = num / den if den > 0 else float(na + nb - 2)
    out.update({"t": t, "df": df, "p": t_sf_two_sided(t, df)})
    return out


def permutation_two_sample(a, b, max_exact=40000, n_random=20000, seed=12345):
    """Two-sided permutation test on the difference of means.

    Exhaustive when C(na+nb, na) <= max_exact -- at 5 seeds per level that is
    252 arrangements, so the test is EXACT and needs no distributional
    assumption at all.  Otherwise `n_random` random relabellings, with the
    +1/+1 correction that keeps the p-value valid (Phipson & Smyth 2010).
    """
    na, nb = len(a), len(b)
    if na < 1 or nb < 1:
        return {"p": None, "exact": None, "n": 0}
    pool = list(a) + list(b)
    obs = abs(mean(a) - mean(b))
    total = sum(pool)
    n_comb = math.comb(na + nb, na)
    if n_comb <= max_exact:
        hits = 0
        for idx in combinations(range(na + nb), na):
            s = sum(pool[i] for i in idx)
            d = abs(s / na - (total - s) / nb)
            if d >= obs - 1e-12:
                hits += 1
        return {"p": hits / n_comb, "exact": True, "n": n_comb}
    rng = random.Random(seed)
    hits = 0
    idxs = list(range(na + nb))
    for _ in range(n_random):
        rng.shuffle(idxs)
        s = sum(pool[i] for i in idxs[:na])
        d = abs(s / na - (total - s) / nb)
        if d >= obs - 1e-12:
            hits += 1
    return {"p": (hits + 1) / (n_random + 1), "exact": False, "n": n_random}


def _f_stat(groups):
    flat = [x for g in groups for x in g]
    n = len(flat)
    k = len(groups)
    if k < 2 or n <= k:
        return None
    gm = mean(flat)
    ss_b = sum(len(g) * (mean(g) - gm) ** 2 for g in groups)
    ss_w = sum((x - mean(g)) ** 2 for g in groups for x in g)
    if ss_w <= 0:
        return float("inf") if ss_b > 0 else 0.0
    return (ss_b / (k - 1)) / (ss_w / (n - k))


def permutation_omnibus(groups, n_random=20000, seed=999):
    """Permutation F-test across all levels at once: 'does the factor matter?'

    One omnibus test, so the answer is not manufactured by testing every pair
    and keeping the smallest p-value.
    """
    groups = [list(g) for g in groups if len(g) > 0]
    if len(groups) < 2:
        return {"p": None, "F": None, "n": 0}
    obs = _f_stat(groups)
    if obs is None:
        return {"p": None, "F": None, "n": 0}
    sizes = [len(g) for g in groups]
    pool = [x for g in groups for x in g]
    rng = random.Random(seed)
    hits = 0
    for _ in range(n_random):
        rng.shuffle(pool)
        off, perm = 0, []
        for s in sizes:
            perm.append(pool[off:off + s])
            off += s
        f = _f_stat(perm)
        if f is not None and f >= obs - 1e-12:
            hits += 1
    return {"p": (hits + 1) / (n_random + 1), "F": obs, "n": n_random}


def holm(pvalues):
    """Holm-Bonferroni step-down adjustment.  Order is preserved."""
    idx = [i for i, p in enumerate(pvalues) if p is not None]
    if not idx:
        return list(pvalues)
    order = sorted(idx, key=lambda i: pvalues[i])
    m = len(order)
    out = list(pvalues)
    running = 0.0
    for rank, i in enumerate(order):
        adj = min(1.0, (m - rank) * pvalues[i])
        running = max(running, adj)
        out[i] = running
    return out


def pooled_sd(groups):
    """Pooled within-level SD: the seed noise, estimated from every level."""
    num = 0.0
    den = 0
    for g in groups:
        if len(g) >= 2:
            num += (len(g) - 1) * sdev(g) ** 2
            den += len(g) - 1
    return math.sqrt(num / den) if den > 0 else float("nan")


def mde(sd, n1, n2, alpha=0.05, power=0.80):
    """Minimum detectable effect for a two-sample test at (alpha, power).

    This is the number that makes a NULL RESULT interpretable: an effect
    smaller than this would have been missed most of the time at this sample
    size, so 'no significant difference' means 'no difference bigger than
    this', not 'no difference'.
    """
    if not math.isfinite(sd) or n1 < 2 or n2 < 2:
        return float("nan")
    df = n1 + n2 - 2
    return (t_ppf(1 - alpha / 2, df) + t_ppf(power, df)) * sd * math.sqrt(1 / n1 + 1 / n2)


def elo_from_score(s):
    if s is None or s <= 0.0 or s >= 1.0:
        return None
    return -400.0 * math.log10(1.0 / s - 1.0)


def elo_per_score_point(s):
    """d(Elo)/d(score) at score s -- converts a score SD into Elo locally."""
    if s is None or s <= 0.0 or s >= 1.0:
        return None
    return 400.0 / (math.log(10.0) * s * (1.0 - s))


# ==========================================================================
# Cost model, calibrated from this machine's own telemetry
# ==========================================================================

class Calibration:
    def __init__(self, evals_per_sec, evals_per_sec_lo, evals_per_sec_hi,
                 sec_per_step, plies_per_game, eval_sec_per_game, samples, source):
        self.evals_per_sec = evals_per_sec
        self.evals_per_sec_lo = evals_per_sec_lo     # pessimistic (contended)
        self.evals_per_sec_hi = evals_per_sec_hi     # optimistic (idle)
        self.sec_per_step = sec_per_step
        self.plies_per_game = plies_per_game
        self.eval_sec_per_game = eval_sec_per_game
        self.samples = samples
        self.source = source

    def as_dict(self):
        return {
            "evals_per_sec": self.evals_per_sec,
            "evals_per_sec_range": [self.evals_per_sec_lo, self.evals_per_sec_hi],
            "sec_per_step": self.sec_per_step,
            "plies_per_game": self.plies_per_game,
            "eval_sec_per_game": self.eval_sec_per_game,
            "telemetry_samples": self.samples,
            "source": self.source,
        }


def _quantile(xs, q):
    if not xs:
        return float("nan")
    ys = sorted(xs)
    if len(ys) == 1:
        return ys[0]
    pos = q * (len(ys) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(ys) - 1)
    frac = pos - lo
    return ys[lo] * (1 - frac) + ys[hi] * frac


def calibrate(runs_dir=None, overrides=None):
    """Read this machine's own past telemetry rather than guessing constants."""
    runs_dir = Path(runs_dir or (REPO / "runs"))
    eps, sps, ppg = [], [], []
    files = sorted(runs_dir.glob("*/telemetry.jsonl")) if runs_dir.is_dir() else []
    for f in files:
        try:
            for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
                line = line.strip()
                if not line.startswith("{"):
                    continue
                try:
                    d = json.loads(line)
                except ValueError:
                    continue
                v = d.get("evals_per_sec")
                if isinstance(v, (int, float)) and v > 0:
                    eps.append(float(v))
                ls, st = d.get("learn_sec"), d.get("steps")
                if isinstance(ls, (int, float)) and isinstance(st, (int, float)) and st > 0:
                    sps.append(float(ls) / float(st))
                pl, gm = d.get("plies"), d.get("games")
                if isinstance(pl, (int, float)) and isinstance(gm, (int, float)) and gm > 0:
                    ppg.append(float(pl) / float(gm))
        except OSError:
            continue

    source = "telemetry: %d generations across %d runs" % (len(eps), len(files))
    if not eps:
        source = "defaults (no telemetry found under %s)" % runs_dir
    cal = Calibration(
        evals_per_sec=_quantile(eps, 0.50) if eps else 700000.0,
        evals_per_sec_lo=_quantile(eps, 0.05) if eps else 250000.0,
        evals_per_sec_hi=_quantile(eps, 0.90) if eps else 900000.0,
        sec_per_step=_quantile(sps, 0.50) if sps else 0.002,
        plies_per_game=_quantile(ppg, 0.50) if ppg else 135.0,
        # Measured: policy-head argmax vs the built-in opponents runs at roughly
        # 40 games/second single-threaded with a 200-ply cap.
        eval_sec_per_game=0.026,
        samples=len(eps),
        source=source,
    )
    for k, v in (overrides or {}).items():
        if v is not None:
            setattr(cal, k, v)
    return cal


# ==========================================================================
# Planning: derive generations from the budget
# ==========================================================================

def effective_sims(cfg):
    """Average simulations per move once playout-cap randomisation AND
    asymmetric search budgets are applied.

    ASYMMETRY CHANGES THE COST OF A GAME, so it has to appear here or the
    comparison is not compute-matched.  In an asymmetric game one side
    searches the whole budget and the other searches budget/ratio, and each
    plays half the moves, so the game costs (1 + 1/ratio)/2 of a symmetric
    one.  `asym_frac` of games are asymmetric, hence the mixture below.
    src/az.c rounds the weak budget and floors it at 2; that is reproduced so
    the plan's evaluation count matches what the trainer will actually do."""
    sims = float(cfg["sims"])
    cap_frac = float(cfg["cap-frac"])
    cap_sims = float(cfg["cap-sims"]) or max(2.0, sims / 5.0)
    cap_sims = min(cap_sims, sims)
    eff = cap_frac * sims + (1.0 - cap_frac) * cap_sims

    af = float(cfg.get("asym-frac", 0.0) or 0.0)
    ar = float(cfg.get("asym-ratio", 4.0) or 4.0)
    if af <= 0.0 or ar <= 1.0:
        return eff
    weak_full = max(2.0, float(int(sims / ar + 0.5)))
    weak_cap = max(2.0, float(int(cap_sims / ar + 0.5)))
    weak_eff = cap_frac * weak_full + (1.0 - cap_frac) * weak_cap
    return (1.0 - af) * eff + af * 0.5 * (eff + weak_eff)


def games_per_gen(cfg):
    """src/az.c: npairs = n_agents * (games_per_agent / 2), one game per pair."""
    gpa = int(cfg["games-per-agent"])
    if gpa < 2:
        gpa = 2
    gpa &= ~1
    return int(cfg["agents"]) * (gpa // 2)


def config_for_level(factor, level, base, scale_lr_final=True):
    cfg = dict(base)
    cfg[factor.flag] = level
    if factor.name == "lr" and scale_lr_final:
        # Varying the peak LR alone would also change the decay RATIO, which is
        # a second change.  Scaling the floor with it keeps the schedule shape
        # identical and varies only its magnitude.
        ratio = base["lr-final"] / base["lr"] if base["lr"] else 0.1
        cfg["lr-final"] = level * ratio
    return cfg


def build_plan(args, factor, levels, base, cal):
    base_gpg = games_per_gen(base)
    base_eff = effective_sims(base)
    plies = cal.plies_per_game

    # --- the matching basis -----------------------------------------------
    basis = args.match
    if basis == "auto":
        basis = "evals" if factor.changes_cost_per_game else "games"
    budget_evals = args.budget_games * plies * base_eff

    # --- how optimiser steps are handled ----------------------------------
    step_mode = args.match_steps
    if step_mode == "auto":
        step_mode = ("proportional"
                     if factor.changes_games_per_gen and factor.name != "steps"
                     else "fixed")

    rows = []
    for level in levels:
        cfg = config_for_level(factor, level, base, not args.no_scale_lr_final)
        gpg = games_per_gen(cfg)
        eff = effective_sims(cfg)

        if step_mode == "proportional":
            cfg["steps"] = max(1, int(round(base["steps"] * gpg / base_gpg)))
        if args.scale_buffer:
            cfg["buffer"] = max(1024, int(round(base["buffer"] * gpg / base_gpg)))
        if args.scale_hof:
            cfg["hof-every"] = max(1, int(round(base["hof-every"] * base_gpg / gpg)))

        if basis == "evals":
            per_gen = gpg * plies * eff
            gens = int(round(budget_evals / per_gen)) if per_gen > 0 else args.min_gens
        else:
            gens = int(round(args.budget_games / gpg)) if gpg > 0 else args.min_gens
        gens = max(args.min_gens, gens)
        if args.max_gens:
            gens = min(gens, args.max_gens)

        total_games = gens * gpg
        total_evals = total_games * plies * eff
        total_steps = gens * int(cfg["steps"])
        train_sec = (total_evals / cal.evals_per_sec
                     + total_steps * cal.sec_per_step
                     + gens * 0.01)

        warnings = []
        n_cull = int(cfg["cull"] * cfg["agents"])
        if n_cull < 1:
            warnings.append("cull %.0f%% of %d agents culls 0 -- PBT is inert here"
                            % (cfg["cull"] * 100, cfg["agents"]))
        if gpg < args.threads:
            warnings.append("only %d pairings/gen: az caps threads at %d, below --threads %d"
                            % (gpg, gpg, args.threads))
        dev = abs(total_games - args.budget_games) / max(1, args.budget_games)
        if basis == "games" and dev > 0.02:
            warnings.append("total games %d is %.1f%% off the %d budget "
                            "(generations must be whole)"
                            % (total_games, 100 * dev, args.budget_games))
        pos_per_gen = gpg * plies
        warnings.append("buffer holds %.1f generations of positions"
                        % (cfg["buffer"] / pos_per_gen if pos_per_gen else 0.0))

        rows.append({
            "level": level,
            "config": cfg,
            "gens": gens,
            "games_per_gen": gpg,
            "total_games": total_games,
            "total_evals": int(total_evals),
            "steps_per_gen": int(cfg["steps"]),
            "total_steps": total_steps,
            "eff_sims": eff,
            "hof_snapshots": min(64, max(1, gens // int(cfg["hof-every"]) + 1)),
            "est_train_sec": train_sec,
            "warnings": warnings,
        })

    plan = {
        "basis": basis,
        "step_mode": step_mode,
        "budget_games": args.budget_games,
        "budget_evals": int(budget_evals),
        "plies_per_game": plies,
        "rows": rows,
    }
    plan["not_matched"] = unmatched_quantities(plan)
    return plan


def unmatched_quantities(plan, tol=0.05):
    """Say out loud which derived quantities are NOT equal across levels.

    Compute matching fixes one budget; it cannot fix all of them at once.
    Matching total games makes generations differ, which makes the buffer's
    staleness in generations differ; matching evaluations makes total games
    differ.  There is no confound-free choice, so the honest thing is to
    enumerate what moved and let the reader weigh it.
    """
    def spread(key, fn):
        vals = [fn(r) for r in plan["rows"]]
        vals = [v for v in vals if v]
        if len(vals) < 2:
            return None
        lo, hi = min(vals), max(vals)
        return (hi / lo) if lo > 0 else None

    checks = [
        ("total self-play games", lambda r: r["total_games"],
         "levels see different amounts of EXPERIENCE (the price of matching "
         "on evaluations)"),
        ("total optimiser steps", lambda r: r["total_steps"],
         "levels take different numbers of gradient steps on the same data"),
        ("replay staleness (generations held)",
         lambda r: r["config"]["buffer"] / (r["games_per_gen"] * plan["plies_per_game"]),
         "the buffer spans a different number of generations at each level "
         "(--scale-buffer trades this against a fixed capacity)"),
        ("hall-of-fame snapshots", lambda r: r["hof_snapshots"],
         "the frozen-opponent pool differs in size"),
        ("generations", lambda r: r["gens"],
         "the cosine LR schedule is stretched over a different number of "
         "generations (its SHAPE in fraction-of-training is identical)"),
    ]
    out = []
    for label, fn, why in checks:
        s = spread(label, fn)
        if s is not None and s > 1.0 + tol:
            out.append({"quantity": label, "ratio": s, "why": why})
    return out


# ==========================================================================
# Running
# ==========================================================================

def slug(value):
    s = str(value)
    s = s.replace("-", "m").replace(".", "p")
    return re.sub(r"[^A-Za-z0-9_]", "_", s)


def run_id(factor_name, level, seed):
    return "%s=%s|seed=%d" % (factor_name, level, seed)


def az_command(cfg, gens, seed, run_dir, threads):
    cmd = [str(BINARY), "az", "--run", str(run_dir), "--gens", str(gens),
           "--seed", str(seed), "--threads", str(threads), "--quiet"]
    for key, val in sorted(cfg.items()):
        if key == "cap-sims" and not val:
            continue
        if key in INT_KEYS:
            cmd += ["--" + key, str(int(val))]
        elif key == "start":
            cmd += ["--start", str(val)]
        else:
            cmd += ["--" + key, repr(float(val)) if isinstance(val, float) else str(val)]
    return cmd


def benchmark_command(model, args, json_out):
    cmd = [sys.executable, str(BENCHMARK),
           "--model", str(model),
           "--opponent", args.opponent,
           "--games", str(args.eval_games),
           "--seed", str(args.eval_seed),
           "--threads", str(args.eval_threads),
           "--ply-cap", str(args.eval_ply_cap),
           "--opening-plies", str(args.eval_opening_plies),
           "--json", str(json_out),
           "--no-md", "--quiet"]
    if args.eval_mode == "policy":
        cmd.append("--policy-only")
    else:
        cmd += ["--depth", str(args.eval_depth)]
    if args.opp_go:
        cmd += ["--opp-go", args.opp_go]
    for opt in args.opp_option:
        cmd += ["--opp-option", opt]
    return cmd


def tail(text, n=12):
    lines = [l for l in (text or "").splitlines() if l.strip()]
    return "\n".join(lines[-n:])


def save_state(state, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")
    os.replace(str(tmp), str(path))


def load_state(path):
    p = Path(path)
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except ValueError:
        sys.stderr.write("warning: %s is not valid JSON; starting fresh\n" % p)
        return None


def execute_one(row, seed, args, work_dir, state, out_path, cal):
    """Train one model, evaluate it, record the result.  Never raises on a
    failed child: a failure is data, and the experiment continues."""
    rid = run_id(state["factor"], row["level"], seed)
    tag = "%s_%s_seed%d" % (state["factor"], slug(row["level"]), seed)
    rdir = Path(work_dir) / tag
    rec = {
        "run_id": rid, "level": row["level"], "seed": seed, "tag": tag,
        "run_dir": str(rdir), "gens": row["gens"],
        "games_per_gen": row["games_per_gen"],
        "total_games": row["total_games"], "total_steps": row["total_steps"],
        "steps_per_gen": row["steps_per_gen"],
        "config": row["config"], "config_hash": state["config_hash"],
        "started": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    cmd = az_command(row["config"], row["gens"], seed, rdir, args.threads)
    rec["train_cmd"] = " ".join(cmd)
    t0 = time.monotonic()
    try:
        proc = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True)
    except OSError as exc:
        rec.update(status="failed", stage="train", error=str(exc),
                   elapsed_s=time.monotonic() - t0)
        return rec
    rec["train_s"] = time.monotonic() - t0
    if proc.returncode != 0:
        rec.update(status="failed", stage="train", returncode=proc.returncode,
                   stderr=tail(proc.stderr), elapsed_s=rec["train_s"])
        return rec

    model = rdir / args.model_file
    if not model.is_file():
        rec.update(status="failed", stage="train",
                   error="training finished but %s was not written" % model,
                   stderr=tail(proc.stderr), elapsed_s=rec["train_s"])
        return rec

    # --- external evaluation ---------------------------------------------
    bj = rdir / "benchmark.json"
    bcmd = benchmark_command(model, args, bj)
    rec["eval_cmd"] = " ".join(bcmd)
    t1 = time.monotonic()
    try:
        bproc = subprocess.run(bcmd, cwd=str(REPO), capture_output=True, text=True)
    except OSError as exc:
        rec.update(status="failed", stage="eval", error=str(exc),
                   elapsed_s=time.monotonic() - t0)
        return rec
    rec["eval_s"] = time.monotonic() - t1
    if bproc.returncode != 0 or not bj.is_file():
        rec.update(status="failed", stage="eval", returncode=bproc.returncode,
                   stderr=tail(bproc.stderr), elapsed_s=time.monotonic() - t0)
        return rec

    try:
        bench = json.loads(bj.read_text(encoding="utf-8"))
        m = bench["match"]
    except (ValueError, KeyError, OSError) as exc:
        rec.update(status="failed", stage="eval",
                   error="could not read the benchmark JSON: %s" % exc,
                   elapsed_s=time.monotonic() - t0)
        return rec

    rec.update({
        "status": "ok",
        "score": m["score"],
        "wins": m["wins"], "draws": m["draws"], "losses": m["losses"],
        "eval_games": m["games"],
        "se_per_pair": m.get("se_per_pair"),
        "score_ci": m.get("score_ci"),
        "draw_rate": m.get("draw_rate"),
        "avg_plies": m.get("avg_plies"),
        "internal_elo": (bench.get("model") or {}).get("champion_elo"),
        "model_generation": (bench.get("model") or {}).get("generation"),
        "elapsed_s": time.monotonic() - t0,
    })

    # Training telemetry of the finished run, so the result carries its own
    # provenance (throughput actually achieved, final repetition rate, ...).
    tel = rdir / "telemetry.jsonl"
    if tel.is_file():
        try:
            lines = [l for l in tel.read_text(encoding="utf-8").splitlines() if l.startswith("{")]
            if lines:
                last = json.loads(lines[-1])
                rec["final_telemetry"] = {
                    k: last.get(k) for k in
                    ("gen", "elo_best", "policy_kl", "policy_top1", "draw",
                     "avg_len", "captures_per_game", "evals_per_sec", "term")
                }
                rec["generations_completed"] = last.get("gen")
        except (ValueError, OSError):
            pass

    if args.prune:
        for name in ("best.crl", "best_raw.crl", "best_ema.crl",
                     "checkpoint.crl", "checkpoint_ema.crl"):
            try:
                (rdir / name).unlink()
            except OSError:
                pass
        rec["pruned"] = True
    return rec


# ==========================================================================
# Analysis
# ==========================================================================

def analyse(state, args):
    metric = getattr(args, "metric", "score")
    getval = METRICS[metric][1]
    is_score = (metric == "score")
    levels = state["levels"]
    by_level = {}
    for rec in state["runs"].values():
        if rec.get("status") == "ok" and getval(rec) is not None:
            by_level.setdefault(str(rec["level"]), []).append(rec)

    per_level = []
    groups = []
    for lv in levels:
        recs = sorted(by_level.get(str(lv), []), key=lambda r: r["seed"])
        scores = [getval(r) for r in recs]
        groups.append(scores)
        n = len(scores)
        m = mean(scores) if n else None
        sd = sdev(scores) if n >= 2 else None
        se = (sd / math.sqrt(n)) if (sd is not None and n) else None
        if n >= 2 and se:
            half = t_ppf(0.975, n - 1) * se
            # The score is a bounded quantity, so the interval is truncated at
            # 0 and 1.  At n=2 the t multiplier is 12.7 and the raw interval
            # routinely leaves [0, 1]; that is a statement about how little two
            # seeds tell you, and it is why this tool nags below 5.
            ci = [max(0.0, m - half), min(1.0, m + half)]
        else:
            ci = [None, None]
        per_level.append({
            "level": lv,
            "n": n,
            "seeds": [r["seed"] for r in recs],
            "scores": scores,
            "mean": m, "sd": sd, "se": se, "ci95": ci,
            "min": min(scores) if scores else None,
            "max": max(scores) if scores else None,
            "mean_elo": elo_from_score(m) if is_score else None,
            "mean_eval_se": mean([r["se_per_pair"] for r in recs
                                  if r.get("se_per_pair") is not None]) if recs else None,
            "mean_draw_rate": mean([r["draw_rate"] for r in recs
                                    if r.get("draw_rate") is not None]) if recs else None,
        })

    usable = [pl for pl in per_level if pl["n"] >= 2]
    sp = pooled_sd([pl["scores"] for pl in usable])
    means = [pl["mean"] for pl in per_level if pl["mean"] is not None]
    spread = (max(means) - min(means)) if len(means) >= 2 else None
    grand = mean(means) if means else None
    slope = elo_per_score_point(grand) if is_score else None

    pairs = []
    for (i, a), (j, b) in combinations(list(enumerate(per_level)), 2):
        if a["n"] < 2 or b["n"] < 2:
            continue
        w = welch(a["scores"], b["scores"])
        pm = permutation_two_sample(a["scores"], b["scores"], seed=args.stat_seed)
        pairs.append({
            "a": a["level"], "b": b["level"],
            "diff": w["diff"],
            "diff_elo": ((elo_from_score(a["mean"]) - elo_from_score(b["mean"]))
                         if is_score and elo_from_score(a["mean"]) is not None
                         and elo_from_score(b["mean"]) is not None else None),
            "cohens_d": (w["diff"] / sp) if (sp and math.isfinite(sp) and sp > 0) else None,
            "welch_t": w["t"], "welch_df": w["df"], "welch_p": w["p"],
            "perm_p": pm["p"], "perm_exact": pm["exact"], "perm_n": pm["n"],
        })
    holm_p = holm([p["perm_p"] for p in pairs])
    for p, hp in zip(pairs, holm_p):
        p["perm_p_holm"] = hp

    omni = permutation_omnibus([pl["scores"] for pl in usable],
                               n_random=args.perm_iters, seed=args.stat_seed)

    ns = [pl["n"] for pl in usable]
    n_typ = min(ns) if ns else 0
    detect = mde(sp, n_typ, n_typ) if n_typ >= 2 else float("nan")

    # ---- the verdict -----------------------------------------------------
    alpha = 0.05
    sig = [p for p in pairs if p["perm_p_holm"] is not None and p["perm_p_holm"] < alpha]
    if not usable or n_typ < 2:
        verdict = "inconclusive"
        why = ("fewer than 2 successful seeds at some level -- nothing can be "
               "said about seed variance")
    elif sig or (omni["p"] is not None and omni["p"] < alpha):
        verdict = "moves the result by more than seed noise"
        why = ("omnibus permutation p = %s; %d of %d pairwise comparisons "
               "survive Holm correction at alpha=0.05"
               % (fmt_p(omni["p"]), len(sig), len(pairs)))
    else:
        verdict = "indistinguishable from seed noise at this sample size"
        why = ("omnibus permutation p = %s, no pairwise comparison survives "
               "Holm correction.  The largest between-level difference is "
               "%.4f, which is %.2fx the pooled seed SD (%.4f) and %.2fx the "
               "minimum detectable effect (%.4f).  An effect smaller than the "
               "MDE would have been missed more often than not."
               % (fmt_p(omni["p"]), spread if spread is not None else 0.0,
                  (spread / sp) if (sp and spread is not None and sp > 0) else float("nan"),
                  sp if sp else float("nan"),
                  (spread / detect) if (detect and spread is not None
                                        and math.isfinite(detect) and detect > 0)
                  else float("nan"), detect))

    saturated = None
    if means and is_score:
        if max(means) < 0.03:
            saturated = "every level scores below 3%% -- the opponent is too strong to discriminate"
        elif min(means) > 0.97:
            saturated = "every level scores above 97%% -- the opponent is too weak to discriminate"

    eval_se = mean([pl["mean_eval_se"] for pl in per_level
                    if pl["mean_eval_se"] is not None]) if (per_level and is_score) else None

    return {
        "metric": metric,
        "metric_blurb": METRICS[metric][0],
        "per_level": per_level,
        "pooled_seed_sd": sp,
        "pooled_seed_sd_elo": (sp * slope) if (sp and slope) else None,
        "between_level_spread": spread,
        "between_level_spread_elo": (spread * slope) if (spread and slope) else None,
        "spread_over_seed_sd": (spread / sp) if (sp and spread is not None and sp > 0) else None,
        "elo_per_score_point": slope,
        "pairs": pairs,
        "omnibus": omni,
        "mde": detect,
        "mde_elo": (detect * slope) if (slope and math.isfinite(detect)) else None,
        "mde_alpha": 0.05, "mde_power": 0.80, "mde_n": n_typ,
        "mean_eval_se_per_pair": eval_se,
        "eval_noise_share": ((eval_se / sp) if (eval_se and sp and sp > 0) else None),
        "verdict": verdict,
        "verdict_detail": why,
        "saturation_warning": saturated,
        "n_failed": sum(1 for r in state["runs"].values() if r.get("status") == "failed"),
    }


def fmt_p(p):
    if p is None:
        return "n/a"
    if p < 1e-4:
        return "<1e-4"
    return "%.4f" % p


def fmt_dur(s):
    if s is None or not math.isfinite(s):
        return "?"
    s = int(round(s))
    if s < 90:
        return "%ds" % s
    if s < 5400:
        return "%dm%02ds" % (s // 60, s % 60)
    return "%dh%02dm" % (s // 3600, (s % 3600) // 60)


# ==========================================================================
# Rendering
# ==========================================================================

def render_plan(state, plan, args, cal, n_seeds):
    L = []
    A = L.append
    A("=" * 78)
    A("ABLATION PLAN -- factor '%s': %s" % (state["factor"], FACTORS[state["factor"]].blurb))
    A("=" * 78)
    A("levels          : %s" % ", ".join(str(r["level"]) for r in plan["rows"]))
    A("seeds per level : %d  (%s)" % (n_seeds, ", ".join(str(s) for s in state["seeds"])))
    A("runs            : %d training runs, each evaluated externally" %
      (len(plan["rows"]) * n_seeds))
    A("")
    A("COMPUTE MATCHING")
    if plan["basis"] == "games":
        A("  basis         : TOTAL GAMES -- every level plays %d games, and the"
          % plan["budget_games"])
        A("                  generation count is derived from that, not fixed.")
    else:
        A("  basis         : TOTAL NETWORK EVALUATIONS -- %s evals per run"
          % "{:,}".format(plan["budget_evals"]))
        A("                  (this factor changes the cost of a game, so equal")
        A("                  games would not be equal compute).")
    A("  optimiser steps: %s" % (
        "scaled with games/generation, so total steps and the sample-reuse "
        "ratio also match" if plan["step_mode"] == "proportional"
        else "held fixed per generation (so total steps differ -- see the table)"))
    A("  replay buffer  : %s" % (
        "scaled with games/generation (constant staleness in generations)"
        if args.scale_buffer else
        "held fixed in positions (so staleness in GENERATIONS differs -- see the table)"))
    A("  hall of fame   : %s" % (
        "cadence scaled so snapshots per game are constant"
        if args.scale_hof else "held fixed at every %d generations" % BASE_CONFIG["hof-every"]))
    A("  plies/game est : %.1f (median over %s)" % (plan["plies_per_game"], cal.source))
    A("")
    A("EVALUATION (external -- internal Elo is never compared across runs)")
    A("  opponent      : %s" % args.opponent)
    A("  games/model   : %d, colour-reversed pairs, %d random opening plies"
      % (args.eval_games, args.eval_opening_plies))
    A("  our side      : %s" % ("raw policy head, argmax (no search)"
                                if args.eval_mode == "policy"
                                else "alpha-beta depth %d" % args.eval_depth))
    A("  eval seed     : %d -- IDENTICAL for every model, so all models face the"
      % args.eval_seed)
    A("                  same openings and the comparison is paired.")
    A("  model file    : %s" % args.model_file)
    A("")
    hdr = ("%-12s %6s %10s %12s %11s %9s %6s %9s" %
           ("level", "gens", "games/gen", "total games", "total steps",
            "buf gens", "hof", "est time"))
    A(hdr)
    A("-" * len(hdr))
    for r in plan["rows"]:
        pos_per_gen = r["games_per_gen"] * plan["plies_per_game"]
        A("%-12s %6d %10d %12s %11s %9.1f %6d %9s" % (
            r["level"], r["gens"], r["games_per_gen"],
            "{:,}".format(r["total_games"]), "{:,}".format(r["total_steps"]),
            r["config"]["buffer"] / pos_per_gen if pos_per_gen else 0.0,
            r["hof_snapshots"],
            fmt_dur(r["est_train_sec"] * n_seeds)))
    A("-" * len(hdr))

    train = sum(r["est_train_sec"] for r in plan["rows"]) * n_seeds
    evals = sum(r["total_evals"] for r in plan["rows"]) * n_seeds
    games = sum(r["total_games"] for r in plan["rows"]) * n_seeds
    ev_sec = len(plan["rows"]) * n_seeds * args.eval_games * cal.eval_sec_per_game
    A("TOTAL: %s self-play games, %s network evaluations" %
      ("{:,}".format(games), "{:,}".format(evals)))
    A("       training  %s   external evaluation  %s   ->  %s"
      % (fmt_dur(train), fmt_dur(ev_sec), fmt_dur(train + ev_sec)))
    lo = evals / cal.evals_per_sec_hi + ev_sec
    hi = evals / cal.evals_per_sec_lo + ev_sec
    A("       throughput bracket: %s (idle, %s evals/s, p90 of past telemetry)"
      % (fmt_dur(lo), "{:,.0f}".format(cal.evals_per_sec_hi)))
    A("                        .. %s (machine busy, %s evals/s, p5).  The point"
      % (fmt_dur(hi), "{:,.0f}".format(cal.evals_per_sec_lo)))
    A("                           estimate above uses the median, %s evals/s."
      % "{:,.0f}".format(cal.evals_per_sec))
    A("")
    if plan.get("not_matched"):
        A("WHAT IS *NOT* MATCHED (one budget can be equalised, not all of them)")
        for nm in plan["not_matched"]:
            A("  - %s varies %.1fx across levels:" % (nm["quantity"], nm["ratio"]))
            for line in wrap(nm["why"], 68):
                A("      %s" % line)
        A("")
    shown = set()
    for r in plan["rows"]:
        for w in r["warnings"]:
            if w.startswith("buffer holds"):
                continue
            key = (r["level"], w)
            if key in shown:
                continue
            shown.add(key)
            A("  ! level %s: %s" % (r["level"], w))
    A("")
    A("BASE CONFIGURATION (held fixed; the factor above overrides its own key)")
    keys = sorted(BASE_CONFIG)
    for i in range(0, len(keys), 4):
        A("  " + "  ".join("%s=%s" % (k, state["base_config"][k]) for k in keys[i:i + 4]))
    A("")
    return "\n".join(L)


def render_results(state):
    an = state.get("analysis") or {}
    pls = an.get("per_level") or []
    L = []
    A = L.append
    A("=" * 78)
    A("ABLATION RESULT -- factor '%s'" % state["factor"])
    A("=" * 78)
    A("matching   : %s, %s per run"
      % (state["plan"]["basis"],
         "{:,} games".format(state["plan"]["budget_games"])
         if state["plan"]["basis"] == "games"
         else "{:,} evaluations".format(state["plan"]["budget_evals"])))
    A("evaluation : %d games vs %s, %s, eval seed %d (common to all models)"
      % (state["eval"]["games"], state["eval"]["opponent"],
         state["eval"]["mode"], state["eval"]["seed"]))
    A("outcome    : %s (--metric %s)"
      % (an.get("metric_blurb") or "?", an.get("metric") or "score"))
    A("")
    hdr = ("%-12s %6s %10s %8s %8s %-18s %8s %8s" %
           ("level", "gens", "mean", "sd", "se", "95% CI", "min", "max"))
    A(hdr)
    A("-" * len(hdr))
    gens_by_level = {str(r["level"]): r["gens"] for r in state["plan"]["rows"]}
    for pl in pls:
        if pl["mean"] is None:
            A("%-12s %6s %10s   (no successful runs)" %
              (pl["level"], gens_by_level.get(str(pl["level"]), "?"), "--"))
            continue
        ci = pl["ci95"]
        cis = ("[%.4f, %.4f]" % (ci[0], ci[1])) if ci[0] is not None else "n<2"
        A("%-12s %6s %10.4f %8s %8s %-18s %8.4f %8.4f" % (
            pl["level"], gens_by_level.get(str(pl["level"]), "?"), pl["mean"],
            ("%.4f" % pl["sd"]) if pl["sd"] is not None else "--",
            ("%.4f" % pl["se"]) if pl["se"] is not None else "--",
            cis, pl["min"], pl["max"]))
    A("-" * len(hdr))
    A("(mean = %s." % (an.get("metric_blurb") or "outcome"))
    A(" CI is Student-t on the SEED means, truncated at the 0/1 bounds.)")
    if an.get("metric") == "internal-elo":
        A(" !! internal Elo is anchored to each run's OWN hall of fame.  Two runs'")
        A("    internal Elos are not on the same scale; this row is a demonstration,")
        A("    not a result.")
    A("")

    slope = an.get("elo_per_score_point")
    sp = an.get("pooled_seed_sd")
    sprd = an.get("between_level_spread")
    unit = "score" if an.get("metric", "score") == "score" else an.get("metric")
    A("SEED VARIANCE (the quantity this tool exists to measure)")
    A("  [az is not bit-reproducible even at a fixed seed -- threaded gradient")
    A("   reduction reorders floating-point sums -- so this is run-to-run")
    A("   variance, of which the seed is one component.  See docs/ABLATION.md.]")
    A("  within-level (seed) SD, pooled : %s%s"
      % (("%.4f %s" % (sp, unit)) if sp and math.isfinite(sp) else "n/a",
         ("  = %.1f Elo" % (sp * slope)) if (sp and slope) else ""))
    A("  between-level spread (max-min) : %s%s"
      % (("%.4f %s" % (sprd, unit)) if sprd is not None else "n/a",
         ("  = %.1f Elo" % (sprd * slope)) if (sprd is not None and slope) else ""))
    r = an.get("spread_over_seed_sd")
    A("  spread / seed SD               : %s"
      % (("%.2f x" % r) if r is not None else "n/a"))
    ese = an.get("mean_eval_se_per_pair")
    if ese is not None:
        A("  evaluation SE per match        : %.4f  (%.0f%% of the seed SD -- if this"
          % (ese, 100 * (ese / sp) if sp else float("nan")))
        A("                                   approaches 100%, the 'seed SD' is mostly")
        A("                                   measurement noise, not training variance;")
        A("                                   raise --eval-games before blaming seeds)")
    A("")

    A("PAIRWISE TESTS (effect size first; a p-value alone means nothing)")
    ph = "%-16s %10s %9s %8s %10s %10s %10s" % (
        "comparison", "diff", "diff Elo", "Cohen d", "Welch p", "perm p", "Holm p")
    A(ph)
    A("-" * len(ph))
    for p in an.get("pairs") or []:
        A("%-16s %10.4f %9s %8s %10s %10s %10s" % (
            "%s vs %s" % (p["a"], p["b"]),
            p["diff"],
            ("%+.1f" % p["diff_elo"]) if p["diff_elo"] is not None else "--",
            ("%+.2f" % p["cohens_d"]) if p["cohens_d"] is not None else "--",
            fmt_p(p["welch_p"]), fmt_p(p["perm_p"]), fmt_p(p["perm_p_holm"])))
    A("-" * len(ph))
    om = an.get("omnibus") or {}
    A("omnibus permutation F-test across all levels: F = %s, p = %s (%s draws)"
      % (("%.3f" % om["F"]) if om.get("F") is not None else "n/a",
         fmt_p(om.get("p")), "{:,}".format(om.get("n") or 0)))
    ex = [p for p in (an.get("pairs") or []) if p.get("perm_exact")]
    if ex:
        A("pairwise permutation tests are EXACT (all %d relabellings enumerated)"
          % ex[0]["perm_n"])
    A("")

    d = an.get("mde")
    A("POWER")
    A("  minimum detectable effect, n=%d per level, alpha=0.05, power=0.80:"
      % (an.get("mde_n") or 0))
    A("      %s%s" % (("%.4f %s points" % (d, unit)) if d and math.isfinite(d) else "n/a",
                      ("  = %.1f Elo" % an["mde_elo"]) if an.get("mde_elo") else ""))
    A("  A null result below means 'no effect larger than this', not 'no effect'.")
    A("")
    if state["plan"].get("not_matched"):
        A("NOT MATCHED (compute matching fixes one budget, not all of them)")
        for nm in state["plan"]["not_matched"]:
            A("  - %s varies %.1fx across levels" % (nm["quantity"], nm["ratio"]))
        A("")
    A("VERDICT: %s -- %s" % (state["factor"], (an.get("verdict") or "?").upper()))
    for line in wrap(an.get("verdict_detail") or "", 74):
        A("  " + line)
    if an.get("saturation_warning"):
        A("  ! %s" % an["saturation_warning"])
    if an.get("n_failed"):
        A("  ! %d run(s) failed and were excluded; see 'runs' in the JSON"
          % an["n_failed"])
    A("")
    return "\n".join(L)


def wrap(text, width):
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + " " + w) if cur else w
    if cur:
        lines.append(cur)
    return lines


def render_markdown(state):
    an = state.get("analysis") or {}
    slope = an.get("elo_per_score_point")
    sp, sprd = an.get("pooled_seed_sd"), an.get("between_level_spread")
    gens_by_level = {str(r["level"]): r for r in state["plan"]["rows"]}
    L = []
    A = L.append
    A("")
    A("### `%s` -- %s" % (state["factor"], state["finished"]))
    A("")
    A("%s.  %d levels x %d seeds = %d runs, %s."
      % (FACTORS[state["factor"]].blurb, len(state["levels"]), len(state["seeds"]),
         len(state["levels"]) * len(state["seeds"]),
         ("compute-matched at %s games per run" % "{:,}".format(state["plan"]["budget_games"]))
         if state["plan"]["basis"] == "games"
         else ("compute-matched at %s network evaluations per run"
               % "{:,}".format(state["plan"]["budget_evals"]))))
    A("Evaluation: %d games against `%s`, %s, colour-reversed pairs, "
      "evaluation seed %d common to every model."
      % (state["eval"]["games"], state["eval"]["opponent"],
         state["eval"]["mode"], state["eval"]["seed"]))
    A("Outcome variable: %s (`--metric %s`)."
      % (an.get("metric_blurb") or "?", an.get("metric") or "score"))
    A("Command: `%s`" % state["command"])
    A("")
    A("| level | generations | games/run | n seeds | mean | seed SD | 95% CI | min | max |")
    A("| --- | ---: | ---: | ---: | ---: | ---: | :---: | ---: | ---: |")
    for pl in an.get("per_level") or []:
        row = gens_by_level.get(str(pl["level"])) or {}
        if pl["mean"] is None:
            A("| %s | %s | %s | 0 | -- | -- | -- | -- | -- |"
              % (pl["level"], row.get("gens", "?"), row.get("total_games", "?")))
            continue
        ci = pl["ci95"]
        A("| %s | %s | %s | %d | %.4f | %s | %s | %.4f | %.4f |" % (
            pl["level"], row.get("gens", "?"),
            "{:,}".format(row.get("total_games", 0)), pl["n"], pl["mean"],
            ("%.4f" % pl["sd"]) if pl["sd"] is not None else "--",
            ("%.4f - %.4f" % (ci[0], ci[1])) if ci[0] is not None else "--",
            pl["min"], pl["max"]))
    A("")
    A("| comparison | difference | in Elo | Cohen's d | Welch p | permutation p | Holm p |")
    A("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for p in an.get("pairs") or []:
        A("| %s vs %s | %+.4f | %s | %s | %s | %s | %s |" % (
            p["a"], p["b"], p["diff"],
            ("%+.0f" % p["diff_elo"]) if p["diff_elo"] is not None else "--",
            ("%+.2f" % p["cohens_d"]) if p["cohens_d"] is not None else "--",
            fmt_p(p["welch_p"]), fmt_p(p["perm_p"]), fmt_p(p["perm_p_holm"])))
    A("")
    A("* **Pooled within-level (seed) SD: %s%s.**"
      % (("%.4f %s points" % (sp, "score" if an.get("metric", "score") == "score"
                                     else an.get("metric")))
         if sp and math.isfinite(sp) else "n/a",
         (" = %.1f Elo" % (sp * slope)) if (sp and slope) else ""))
    A("* Between-level spread: %s%s -- %s the seed SD."
      % (("%.4f" % sprd) if sprd is not None else "n/a",
         (" = %.1f Elo" % (sprd * slope)) if (sprd is not None and slope) else "",
         ("%.2fx" % an["spread_over_seed_sd"])
         if an.get("spread_over_seed_sd") is not None else "n/a"))
    om = an.get("omnibus") or {}
    A("* Omnibus permutation test across all levels: p = %s." % fmt_p(om.get("p")))
    A("* Minimum detectable effect at n=%d, alpha=0.05, power=0.80: **%s%s**."
      % (an.get("mde_n") or 0,
         ("%.4f %s points" % (an["mde"], "score" if an.get("metric", "score") == "score"
                                    else an.get("metric")))
         if an.get("mde") and math.isfinite(an["mde"]) else "n/a",
         (" = %.0f Elo" % an["mde_elo"]) if an.get("mde_elo") else ""))
    if an.get("mean_eval_se_per_pair") is not None and sp:
        A("* Evaluation SE per match: %.4f, i.e. %.0f%% of the seed SD."
          % (an["mean_eval_se_per_pair"], 100 * an["mean_eval_se_per_pair"] / sp))
    if an.get("n_failed"):
        A("* %d run(s) failed and are excluded." % an["n_failed"])
    if an.get("saturation_warning"):
        A("* **Warning:** %s" % an["saturation_warning"])
    for nm in state["plan"].get("not_matched") or []:
        A("* Not matched: %s varies %.1fx across levels -- %s."
          % (nm["quantity"], nm["ratio"], nm["why"]))
    A("")
    A("**Verdict: `%s` %s.**  %s"
      % (state["factor"], an.get("verdict") or "?", an.get("verdict_detail") or ""))
    A("")
    return "\n".join(L)


def append_markdown(path, text):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.is_file():
        p.write_text("# Ablation results\n\n%s\n" % MD_MARKER, encoding="utf-8")
    body = p.read_text(encoding="utf-8")
    if MD_MARKER not in body:
        body = body.rstrip() + "\n\n" + MD_MARKER + "\n"
    p.write_text(body.rstrip() + "\n" + text, encoding="utf-8")


# ==========================================================================
# CLI
# ==========================================================================

def parse_levels(factor, raw):
    if not raw:
        return list(factor.default_levels)
    out = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            v = factor.cast(tok)
        except ValueError:
            raise SystemExit("error: %r is not a valid level for --factor %s"
                             % (tok, factor.name))
        out.append(v)
    if len(out) < 2:
        raise SystemExit("error: --levels needs at least 2 values to compare")
    if len(set(out)) != len(out):
        raise SystemExit("error: --levels contains duplicates")
    if factor.name == "games-per-agent":
        bad = [v for v in out if v < 2 or v % 2]
        if bad:
            raise SystemExit("error: games-per-agent levels must be even and >= 2 "
                             "(src/az.c rounds them down); offending: %s" % bad)
    return out


def build_parser():
    p = argparse.ArgumentParser(
        prog="ablate.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Multi-seed, compute-matched ablation for the AlphaZero trainer.",
        epilog="""examples:
  python3 py/ablate.py --list-factors
  python3 py/ablate.py --factor agents --levels 8,32,128 --seeds 5 --dry-run
  python3 py/ablate.py --factor agents --levels 8,32,128 --seeds 5 \\
                       --budget-games 12288 --out runs/ablation_agents.json
  python3 py/ablate.py --factor sims --dry-run     # matches on evaluations
""")
    p.add_argument("--factor", help="which hyper-parameter to vary")
    p.add_argument("--levels", help="comma-separated values (default: the factor's own)")
    p.add_argument("--seeds", type=int, default=5,
                   help="independent training seeds per level (default 5)")
    p.add_argument("--seed-base", type=int, default=1000,
                   help="training seeds are seed-base+1 .. seed-base+seeds (default 1000)")
    p.add_argument("--budget-games", type=int, default=12288,
                   help="self-play games per run, held equal across levels "
                        "(default 12288 -- divides exactly into whole generations "
                        "at the default agent levels)")
    p.add_argument("--match", choices=("auto", "games", "evals"), default="auto",
                   help="match the budget on total games or total network "
                        "evaluations (default auto: evals when the factor "
                        "changes the cost of a game)")
    p.add_argument("--match-steps", choices=("auto", "fixed", "proportional"),
                   default="auto",
                   help="scale --steps with games/generation so total optimiser "
                        "steps match too (default auto)")
    p.add_argument("--scale-buffer", action="store_true",
                   help="scale --buffer with games/generation, holding replay "
                        "staleness constant in GENERATIONS rather than positions")
    p.add_argument("--scale-hof", dest="scale_hof", action="store_true", default=True,
                   help="scale --hof-every so hall-of-fame snapshots per game "
                        "are constant (default on)")
    p.add_argument("--no-scale-hof", dest="scale_hof", action="store_false")
    p.add_argument("--no-scale-lr-final", action="store_true",
                   help="when varying lr, leave --lr-final alone (changes the "
                        "decay ratio as well as the magnitude)")
    p.add_argument("--min-gens", type=int, default=4)
    p.add_argument("--max-gens", type=int, default=0, help="0 = no cap")
    p.add_argument("--threads", type=int, default=os.cpu_count() or 4,
                   help="training threads (runs are sequential; each uses all cores)")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="override a base-config key, repeatable (e.g. --set sims=96)")

    g = p.add_argument_group("external evaluation")
    g.add_argument("--opponent", default="random",
                   help="random | material | uci:<command>  (default random -- it "
                        "keeps short-budget models away from 0%% and 100%%)")
    g.add_argument("--eval-games", type=int, default=800,
                   help="games per model, colour-reversed pairs (default 800)")
    g.add_argument("--eval-seed", type=int, default=20260911,
                   help="one seed for every model, so the comparison is paired")
    g.add_argument("--eval-mode", choices=("policy", "search"), default="policy",
                   help="policy = raw network, no search (default; it is the "
                        "network the factor is supposed to affect)")
    g.add_argument("--eval-depth", type=int, default=4)
    g.add_argument("--eval-threads", type=int, default=0, help="0 = min(4, --threads)")
    g.add_argument("--eval-ply-cap", type=int, default=200)
    g.add_argument("--eval-opening-plies", type=int, default=4)
    g.add_argument("--opp-go", default="", help="UCI opponents: e.g. nodes=1000")
    g.add_argument("--opp-option", action="append", default=[], metavar="NAME=VALUE")
    g.add_argument("--model-file", default="checkpoint.crl",
                   help="which file in the run dir to evaluate.  Default "
                        "checkpoint.crl = the model after the full budget.  "
                        "best.crl is early-stopped on the run's own INTERNAL "
                        "Elo, which is not comparable across runs.")

    g.add_argument("--metric", default="score", choices=sorted(METRICS),
                   help="outcome variable (default score = the external match "
                        "result; the rest are self-play telemetry from the last "
                        "generation, for re-testing claims made about them)")

    s = p.add_argument_group("statistics")
    s.add_argument("--perm-iters", type=int, default=20000,
                   help="random draws for the omnibus permutation test (pairwise "
                        "tests are exact at small n)")
    s.add_argument("--stat-seed", type=int, default=20260911)

    o = p.add_argument_group("output")
    o.add_argument("--out", default=None,
                   help="JSON output (default runs/ablation_<factor>.json); "
                        "checkpointed after every run and used to resume")
    o.add_argument("--work-dir", default=None,
                   help="where training run dirs go (default: --out without .json)")
    o.add_argument("--md", default=str(DEFAULT_MD), help="markdown to append to")
    o.add_argument("--no-md", action="store_true")
    o.add_argument("--fresh", action="store_true", help="ignore any existing --out")
    o.add_argument("--prune", action="store_true",
                   help="delete model files after evaluation to save disk")
    o.add_argument("--dry-run", action="store_true",
                   help="print the plan and the estimated cost; run nothing")
    o.add_argument("--list-factors", action="store_true")

    c = p.add_argument_group("cost model (calibrated from runs/*/telemetry.jsonl)")
    c.add_argument("--evals-per-sec", type=float, default=None)
    c.add_argument("--plies-per-game", type=float, default=None)
    c.add_argument("--eval-sec-per-game", type=float, default=None)
    return p


def apply_sets(base, sets):
    cfg = dict(base)
    for item in sets:
        if "=" not in item:
            raise SystemExit("error: --set expects KEY=VALUE, got %r" % item)
        k, v = item.split("=", 1)
        k = k.strip()
        if k not in cfg:
            raise SystemExit("error: --set %s: unknown key.  Known: %s"
                             % (k, ", ".join(sorted(cfg))))
        if k == "start":
            if v not in ("classical", "960", "mixed"):
                raise SystemExit("error: --set start must be classical|960|mixed")
            cfg[k] = v
        elif k in INT_KEYS:
            cfg[k] = int(v)
        else:
            cfg[k] = float(v)
    return cfg


def git_commit():
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=str(REPO),
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(argv)

    if args.list_factors:
        print("factors (each varies exactly one flag; everything else is held fixed)\n")
        print("%-18s %-14s %-8s %s" % ("name", "az flag", "cost", "default levels"))
        print("-" * 78)
        for name, f in FACTORS.items():
            cost = ("games/gen" if f.changes_games_per_gen else
                    "per-game" if f.changes_cost_per_game else
                    "learning" if f.changes_learn_cost else "-")
            print("%-18s %-14s %-8s %s" % (name, "--" + f.flag, cost,
                                           ",".join(str(x) for x in f.default_levels)))
        print("\n%s" % "\n".join("  %-18s %s" % (n, f.blurb) for n, f in FACTORS.items()))
        return 0

    if not args.factor:
        sys.stderr.write("error: --factor is required (see --list-factors)\n")
        return 2
    if args.factor not in FACTORS:
        sys.stderr.write("error: unknown factor %r.  Known: %s\n"
                         % (args.factor, ", ".join(FACTORS)))
        return 2
    factor = FACTORS[args.factor]
    levels = parse_levels(factor, args.levels)
    if args.seeds < 1:
        sys.stderr.write("error: --seeds must be >= 1\n")
        return 2
    if args.seeds < 3 and not args.dry_run:
        sys.stderr.write("warning: %d seed(s) per level cannot separate a factor "
                         "from seed noise; 5 is the minimum worth reporting.\n"
                         % args.seeds)
    if args.eval_threads <= 0:
        args.eval_threads = max(1, min(4, args.threads))

    base = apply_sets(BASE_CONFIG, args.set)
    cal = calibrate(overrides={
        "evals_per_sec": args.evals_per_sec,
        "plies_per_game": args.plies_per_game,
        "eval_sec_per_game": args.eval_sec_per_game,
    })
    if args.evals_per_sec:
        cal.evals_per_sec_lo = cal.evals_per_sec_hi = args.evals_per_sec

    out_path = Path(args.out or (REPO / "runs" / ("ablation_%s.json" % args.factor)))
    work_dir = Path(args.work_dir) if args.work_dir else out_path.with_suffix("")
    seeds = [args.seed_base + i + 1 for i in range(args.seeds)]

    plan = build_plan(args, factor, levels, base, cal)

    state = {
        "schema": "chessrl-ablation/1",
        "factor": args.factor,
        "factor_flag": "--" + factor.flag,
        "levels": levels,
        "seeds": seeds,
        "base_config": base,
        "plan": plan,
        "eval": {
            "opponent": args.opponent, "games": args.eval_games,
            "seed": args.eval_seed,
            "mode": ("raw policy head, argmax" if args.eval_mode == "policy"
                     else "alpha-beta depth %d" % args.eval_depth),
            "ply_cap": args.eval_ply_cap, "opening_plies": args.eval_opening_plies,
            "model_file": args.model_file,
        },
        "calibration": cal.as_dict(),
        "environment": {
            "commit": git_commit(), "python": sys.version.split()[0],
            "platform": platform.platform(), "cores": os.cpu_count(),
            "threads": args.threads,
        },
        "command": "python3 py/ablate.py " + " ".join(argv),
        "runs": {},
    }
    # Identifies the experimental setup.  A resumed run whose hash differs is
    # a different experiment, and its old results are not reused.  hashlib,
    # not hash(): str.__hash__ is salted per process, so it would never match
    # across invocations and resume would silently never work.
    state["config_hash"] = hashlib.sha1(json.dumps(
        {"f": args.factor, "l": levels, "b": base, "p": plan["rows"],
         "e": state["eval"]}, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:12]

    print(render_plan(state, plan, args, cal, len(seeds)))

    if args.dry_run:
        print("dry run: nothing was executed.  Drop --dry-run to start.")
        return 0

    if not BINARY.is_file():
        sys.stderr.write("error: %s is not built.  Run: make -j8\n" % BINARY)
        return 2
    libs = [REPO / "build" / "libchessrl.dylib", REPO / "build" / "libchessrl.so"]
    lib = next((p for p in libs if p.is_file()), None)
    if lib is None:
        sys.stderr.write("error: the shared library is not built.  Run: make -j8 lib\n")
        return 2
    # The binary trains the models and the shared library evaluates them.  If
    # either is older than the sources they must be rebuilt TOGETHER: a model
    # written by one net layout and read by another is silently meaningless.
    srcs = list((REPO / "src").glob("*.c")) + list((REPO / "src").glob("*.h"))
    newest = max((p.stat().st_mtime for p in srcs), default=0.0)
    stale = [p.name for p in (BINARY, lib) if p.stat().st_mtime < newest]
    if stale:
        sys.stderr.write(
            "error: %s %s older than src/.  The trainer and the evaluation "
            "library must be built from the same sources -- a model written by "
            "one net layout and read by another is meaningless.\n"
            "       Run: make -j8 && make -j8 lib\n"
            % (", ".join(stale), "is" if len(stale) == 1 else "are"))
        return 2

    # ---- resume ----------------------------------------------------------
    prior = None if args.fresh else load_state(out_path)
    if prior and prior.get("config_hash") == state["config_hash"]:
        state["runs"] = {k: v for k, v in (prior.get("runs") or {}).items()
                         if v.get("status") == "ok"}
        state["created"] = prior.get("created")
        if state["runs"]:
            print("resuming: %d completed run(s) reused from %s\n"
                  % (len(state["runs"]), out_path))
    elif prior:
        print("note: %s describes a different experiment (config hash %s != %s); "
              "starting fresh\n" % (out_path, prior.get("config_hash"), state["config_hash"]))
    state.setdefault("created", datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))

    # ---- execute, seed-major so an interruption leaves levels balanced ----
    todo = [(row, seed) for seed in seeds for row in plan["rows"]
            if run_id(args.factor, row["level"], seed) not in state["runs"]]
    total = len(todo)
    t_start = time.monotonic()
    interrupted = False

    def _sigint(_sig, _frm):
        raise KeyboardInterrupt

    old = signal.signal(signal.SIGINT, _sigint)
    try:
        for i, (row, seed) in enumerate(todo, 1):
            rid = run_id(args.factor, row["level"], seed)
            print("[%d/%d] %s  (%d gens, %s games, est %s)"
                  % (i, total, rid, row["gens"], "{:,}".format(row["total_games"]),
                     fmt_dur(row["est_train_sec"])))
            sys.stdout.flush()
            rec = execute_one(row, seed, args, work_dir, state, out_path, cal)
            state["runs"][rid] = rec
            if rec.get("status") == "ok":
                print("        score %.4f  (W%d D%d L%d)  train %s  eval %s"
                      % (rec["score"], rec["wins"], rec["draws"], rec["losses"],
                         fmt_dur(rec.get("train_s")), fmt_dur(rec.get("eval_s"))))
            else:
                print("        FAILED in %s: %s" % (rec.get("stage", "?"),
                                                    rec.get("error") or
                                                    ("exit %s" % rec.get("returncode"))))
                if rec.get("stderr"):
                    for line in rec["stderr"].splitlines()[-4:]:
                        print("          | %s" % line)
            state["updated"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            state["analysis"] = analyse(state, args)
            save_state(state, out_path)
            sys.stdout.flush()
    except KeyboardInterrupt:
        interrupted = True
        print("\ninterrupted -- progress is saved.  Re-run the same command to resume.")
    finally:
        signal.signal(signal.SIGINT, old)

    state["finished"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    state["wall_clock_s"] = time.monotonic() - t_start
    state["interrupted"] = interrupted
    state["analysis"] = analyse(state, args)
    save_state(state, out_path)

    print()
    print(render_results(state))
    print("wrote %s" % out_path)

    if not args.no_md and not interrupted:
        append_markdown(args.md, render_markdown(state))
        print("appended a section to %s" % args.md)
    elif interrupted:
        print("(markdown not appended: the experiment is incomplete)")
    return 0 if not interrupted else 130


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
