#!/usr/bin/env python3
"""Sweep the champion across a graded ladder of opponents, weakest first.

Four ladders, because no single one covers the range:

  baselines  the dependency-free ladder in py/baselines.py -- random,
         randomplus-P, the legacy 1-ply material grabber, then material-N and
         mobility-N searchers.  This is the only group that BRACKETS a weak
         agent: every Stockfish setting below, including one node of search,
         is far above where an early self-play network plays, and a ladder
         that scores 0% on every rung measures nothing.  These opponents
         contain hand-coded chess knowledge on purpose; they are the ruler,
         not the agent.  See py/baselines.py and docs/FROM_SCRATCH.md.
  nodes  Stockfish at FULL skill but starved of search (go nodes N). This is the
         only way to probe below Stockfish's rated floor, and it is perfectly
         reproducible -- a node budget does not move with CPU load, where a
         movetime budget does.
  skill  Stockfish "Skill Level" 0..20, the setting people mean by "level".
  elo    Stockfish UCI_LimitStrength with UCI_Elo, its own rating scale. Only
         these rungs can produce an ABSOLUTE number: ours = theirs + elo_diff.

Usage:
  python3 py/sf_ladder.py [--games 60] [--threads 4] [--depth 4] [--policy-only]
                          [--groups baselines,nodes,skill,elo]
                          [--baselines LIST] [--stop 2.0]
                          [--out runs/sf_ladder.json]
                          [--scale uci|chesscom|fide|lichess]

Each group is ordered weakest-first and is abandoned once a rung scores below
--stop: the rungs above it would only produce more 0%s.  The rating-anchored
`elo` group is never skipped on those grounds, because even a shutout against a
calibrated opponent converts into an absolute bound.

--scale converts the best absolute anchor onto another rating pool.  That is a
conversion between pools, not a measurement, and it is far less certain than
the match is; see docs/RATING_SCALES.md.  The default, uci, converts nothing.
"""
import argparse, json, os, subprocess, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import benchmark as B  # noqa: E402  -- the conversion lives there, in one place
import baselines as L  # noqa: E402  -- the dependency-free rungs

SF = "/opt/homebrew/bin/stockfish"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The dependency-free ladder, weakest first.  See py/baselines.py.
BASELINE_RUNGS = list(L.DEFAULT_LADDER)
# Full-skill Stockfish, search starved to N nodes.
NODE_RUNGS  = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000]
# "Skill Level" with a fixed, generous node budget so the skill setting is what varies.
SKILL_RUNGS = [0, 1, 2, 3, 4, 5, 6, 8, 10, 13, 16, 20]
SKILL_NODES = 20000
# Stockfish's own rating scale. 1320 is its floor.
ELO_RUNGS   = [1320, 1400, 1500, 1600, 1800, 2000]
ELO_NODES   = 20000


def rungs_for(groups, baseline_rungs=None):
    out = []
    if "baselines" in groups:
        for nm in (baseline_rungs if baseline_rungs is not None else BASELINE_RUNGS):
            out.append(dict(group="base", label=nm, ref=None, opts=[], go=None,
                            opponent=nm))
    if "nodes" in groups:
        for n in NODE_RUNGS:
            out.append(dict(group="nodes", label=f"nodes={n}", ref=None,
                            opts=[], go=f"nodes={n}", opponent=f"uci:{SF}"))
    if "skill" in groups:
        for s in SKILL_RUNGS:
            out.append(dict(group="skill", label=f"Skill {s}", ref=None,
                            opts=[f"Skill Level={s}"], go=f"nodes={SKILL_NODES}",
                            opponent=f"uci:{SF}"))
    if "elo" in groups:
        for e in ELO_RUNGS:
            out.append(dict(group="elo", label=f"UCI_Elo {e}", ref=e,
                            opts=["UCI_LimitStrength=true", f"UCI_Elo={e}"],
                            go=f"nodes={ELO_NODES}", opponent=f"uci:{SF}"))
    return out


def play(rung, args, tmp):
    cmd = [sys.executable, os.path.join(ROOT, "py", "benchmark.py"),
           "--model", args.model, "--opponent", rung.get("opponent", f"uci:{SF}"),
           "--games", str(args.games), "--threads", str(args.threads),
           "--seed", str(args.seed),
           "--ply-cap", str(args.ply_cap), "--no-md", "--quiet", "--json", tmp]
    if rung.get("go"):
        cmd += ["--opp-go", rung["go"]]
    if args.policy_only:
        cmd.append("--policy-only")
    else:
        cmd += ["--depth", str(args.depth)]
    for o in rung["opts"]:
        cmd += ["--opp-option", o]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        return None, (r.stderr or r.stdout)[-400:]
    with open(tmp) as f:
        return json.load(f), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="runs/pilot/best.crl")
    ap.add_argument("--games", type=int, default=60)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--policy-only", action="store_true")
    ap.add_argument("--seed", type=int, default=20260911)
    ap.add_argument("--ply-cap", type=int, default=300)
    ap.add_argument("--groups", default="baselines,nodes,skill,elo")
    ap.add_argument("--baselines", default="",
                    help="comma list of py/baselines.py rungs for the 'baselines' "
                         "group, weakest first (default: its DEFAULT_LADDER).  "
                         "`python3 py/baselines.py list` prints them")
    ap.add_argument("--stop", type=float, default=2.0, metavar="PCT",
                    help="abandon a group once a rung scores below this percentage "
                         "(default 2.0).  The rating-anchored 'elo' group is never "
                         "abandoned: even a shutout there is an absolute bound")
    ap.add_argument("--out", default="runs/sf_ladder.json")
    ap.add_argument("--scale", default="uci", choices=list(B.SCALES),
                    help="also express the best absolute anchor on another rating pool "
                         "(default uci = no conversion).  See docs/RATING_SCALES.md")
    args = ap.parse_args()

    groups = [g.strip() for g in args.groups.split(",") if g.strip()]
    bl = L.expand_group(args.baselines) if args.baselines else None
    rungs = rungs_for(groups, bl)
    who = ("raw policy (no search)" if args.policy_only
           else f"MCTS depth {args.depth} = {args.depth * 64} simulations")
    print(f"ChessRL champion [{who}]  vs  Stockfish, {args.games} games per rung")
    print(f"{len(rungs)} rungs, seed {args.seed}, colour-reversed pairs, node-limited "
          f"(load independent)\n")
    hdr = f"{'ladder':<6} {'opponent':<14} {'W':>4} {'D':>4} {'L':>4} {'score':>7} {'Elo diff':>12} {'abs Elo':>16}"
    print(hdr); print("-" * len(hdr))

    results, tmp = [], "/tmp/_sf_rung.json"
    t0 = time.time()
    abandoned = {}
    i = -1
    while True:
        i += 1
        if i >= len(rungs):
            break
        rung = rungs[i]
        if rung["group"] in abandoned:
            continue
        d, err = play(rung, args, tmp)
        if d is None:
            print(f"{rung['group']:<6} {rung['label']:<14}  FAILED: {err}")
            continue
        m = d["match"]; e = m["elo"]
        pt, lo, hi = e.get("point"), e.get("lo"), e.get("hi")
        if pt is None:
            diff = f"{e.get('bound','n/a')}"
        elif lo is None or hi is None:
            diff = f"{pt:+.0f} (open)"
        else:
            diff = f"{pt:+.0f}"
        absolute = ""
        if rung["ref"] is not None and pt is not None:
            absolute = f"{rung['ref']+pt:.0f}"
            if lo is not None and hi is not None:
                absolute += f" [{rung['ref']+lo:.0f},{rung['ref']+hi:.0f}]"
        print(f"{rung['group']:<6} {rung['label']:<14} {m['wins']:>4} {m['draws']:>4} "
              f"{m['losses']:>4} {m['score_pct']:>6.1f}% {diff:>12} {absolute:>16}")
        sys.stdout.flush()
        results.append(dict(rung=rung, match=m))

        # Stop climbing a ladder once the score falls through the floor; the
        # rungs above it can only produce more 0%s.  The rating-anchored `elo`
        # group is exempt: a shutout there is still an absolute bound, which is
        # the entire reason for playing it.
        if m["score_pct"] < args.stop and rung["group"] != "elo":
            abandoned[rung["group"]] = rung["label"]
            left = sum(1 for r in rungs[i + 1:] if r["group"] == rung["group"])
            if left:
                print(f"{'':6} (scored {m['score_pct']:.1f}% < {args.stop:.1f}% -- "
                      f"skipping the remaining {left} '{rung['group']}' rung(s))")

    # Where does the score cross 50%?
    cross = None
    seen_groups = []
    for r in results:
        if r["rung"]["group"] not in seen_groups:
            seen_groups.append(r["rung"]["group"])
    for g in seen_groups:
        seq = [r for r in results if r["rung"]["group"] == g]
        for a, b in zip(seq, seq[1:]):
            sa, sb = a["match"]["score"], b["match"]["score"]
            if sa >= 0.5 > sb:
                cross = (g, a["rung"]["label"], b["rung"]["label"], sa, sb)
                break
        if cross:
            break

    out = dict(generated=time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime()),
               model=args.model, champion=who, games_per_rung=args.games,
               seed=args.seed, crossing=cross, stop_pct=args.stop,
               abandoned=abandoned, rungs=results)
    os.makedirs(os.path.dirname(os.path.join(ROOT, args.out)) or ".", exist_ok=True)
    with open(os.path.join(ROOT, args.out), "w") as f:
        json.dump(out, f, indent=1)

    print(f"\n{time.time()-t0:.0f}s elapsed. Full results: {args.out}")
    if cross:
        g, lo_l, hi_l, sa, sb = cross
        print(f"50% crossing on the '{g}' ladder: between {lo_l} ({sa*100:.0f}%) "
              f"and {hi_l} ({sb*100:.0f}%)")
    anchored = [r for r in results if r["rung"]["ref"] is not None
                and r["match"]["elo"].get("point") is not None]
    if anchored:
        best = min(anchored, key=lambda r: abs(r["match"]["score"] - 0.5))
        el = best["match"]["elo"]
        est = best["rung"]["ref"] + el["point"]
        print(f"Best absolute anchor: {best['rung']['label']} at "
              f"{best['match']['score_pct']:.1f}%  ->  ~{est:.0f} Elo")
        if args.scale != "uci":
            lo = best["rung"]["ref"] + el["lo"] if el.get("lo") is not None else None
            hi = best["rung"]["ref"] + el["hi"] if el.get("hi") is not None else None
            label = B.SCALE_LABELS[args.scale]
            print(f"  measured (Stockfish UCI_Elo): {est:.0f} "
                  f"[95% CI {'-inf' if lo is None else f'{lo:.0f}'} - "
                  f"{'+inf' if hi is None else f'{hi:.0f}'}]")
            conv = B.convert_interval(est, lo, hi, args.scale)
            if conv is None or not conv["in_range"]:
                c = B.convert_rating(est, args.scale)
                a_lo, a_hi = c["anchor_range"]
                edge = a_lo if c["bound_side"] == "below" else a_hi
                near = B.convert_rating(edge, args.scale)
                print(f"  converted ({label}): NOT CONVERTED -- {est:.0f} is "
                      f"{c['bound_side']} the anchor table (UCI_Elo {a_lo}-{a_hi}).")
                print(f"    Refusing to extrapolate.  Nearest bound: UCI_Elo {edge} is "
                      f"{label} "
                      f"{max(B.round50(near['point'] - near['conv95']), 0)}-"
                      f"{B.round50(near['point'] + near['conv95'])}; "
                      f"the champion is {c['bound_side']} that.")
            elif conv["band_lo"] is None or conv["band_hi"] is None:
                print(f"  converted ({label}): NOT CONVERTED -- the measurement's own "
                      f"CI is open-ended.")
            else:
                print(f"  converted ({label}): "
                      f"{max(B.round50(conv['band_lo']), 0)} - {B.round50(conv['band_hi'])} "
                      f"(nearest 50; midpoint {B.round50(conv['point'])} is not the answer)")
                mw = max(v for v in (conv["meas95_lo"], conv["meas95_hi"]) if v is not None)
                print(f"    match error +/-{mw:.0f} and pool-conversion error "
                      f"+/-{conv['conv95']:.0f}, in quadrature.  Pool conversion is the "
                      f"dominant term.")
            print("    This is a conversion between rating pools, not a measurement. "
                  "See docs/RATING_SCALES.md")


if __name__ == "__main__":
    main()
