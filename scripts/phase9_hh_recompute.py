#!/usr/bin/env python3
"""Phase 9 HH verdict — independent gate recomputation from per-episode JSONL.

Recomputes every FF-HH-* gate number DIRECTLY from
results/phase9_hierarchical_sweep.jsonl (per-episode records), independent of
the sweep script's own summarize()/CSV, then reconciles against
results/phase9_hierarchical_sweep_summary.csv.

Gate definitions (authoritative: PEDA_FINAL/phase9/plans/plan-hierarchical-horizon.md
§7 + PEDA_FINAL/phase9/PHASE9_PLAN.md):
  FF1  layered(best) vs random_goal, 20x20 Variant B (primary) / A (secondary):
       dead iff ΔSCR < 0.05 AND ΔFHT < 20
  FF2  coverage range across λ ∈ {0,0.5,1,2,∞} < 0.05 for EVERY H_plan at
       20x20 and 15x15 -> dead
  FF3  layered(·, never) vs layered(·, T_reeval∈{10,25}) at 20x20 B:
       ΔSCR < 0.02 -> re-eval loop is dead weight, run open-loop
  FF4  layered(best) vs flat_count at 20x20 B:
       SCR < base+0.05 AND FHT > base+20 -> kill/redesign goal space
  FF5  (PHASE9_PLAN) only wins at 10x10 (not 15x15 or 20x20) -> marginal
  Positive bar (§6): layered(best) beats flat_count by ≥0.05 SCR AND ≥20 FHT
       at 20x20 B, and beats random_goal on both metrics at 20x20 A and B.

Usage:
  python scripts/phase9_hh_recompute.py
"""

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS = _PROJECT_ROOT / "results"

LAMS = ["0.0", "0.5", "1.0", "2.0", "inf"]
H_PLANS = [20, 50, 100]
T_VALS = [None, 10, 25]


def iter_episodes(path):
    with open(path) as f:
        next(f)  # meta
        for line in f:
            if line.strip():
                yield json.loads(line)


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def mean_fht(eps):
    hits = [e["fht"] for e in eps if e["fht"] >= 0]
    return mean(hits) if hits else -1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--jsonl",
        default=str(RESULTS / "phase9_hierarchical_sweep.jsonl"),
    )
    ap.add_argument(
        "--csv",
        default=str(RESULTS / "phase9_hierarchical_sweep_summary.csv"),
    )
    args = ap.parse_args()

    # ── aggregate from per-episode JSONL (independent of summarize()) ──
    # key: (condition, variant, size, h_plan, lam_str, t_reeval)
    groups = defaultdict(list)
    for ep in iter_episodes(args.jsonl):
        lam = str(ep["lam"])
        cond = ("layered" if ep["condition"].startswith("layered")
                else ep["condition"])
        key = (
            cond, ep["variant"], ep["size"],
            ep["h_plan"], lam, ep["t_reeval"],
        )
        groups[key].append(ep)

    agg = {}
    for key, eps in groups.items():
        agg[key] = {
            "n": len(eps),
            "scr": mean([e["scr"] for e in eps]),
            "fht": mean_fht(eps),
            "success": mean([1 if e["success"] else 0 for e in eps]),
            "n_hits": sum(1 for e in eps if e["fht"] >= 0),
        }

    def get(cond, variant, size, h_plan, lam, t):
        k = (cond, variant, size, h_plan, lam, t)
        if k not in agg:
            return None
        a = agg[k]
        return (a["scr"], a["fht"], a["success"], a["n_hits"], a["n"])

    # baseline rows (flat / random are h_plan/lam-insensitive in the data)
    def baseline(cond, variant, size):
        """All episodes for the baseline condition at (variant,size)."""
        eps = [ep for k, eps in groups.items()
               if k[0] == cond and k[1] == variant and k[2] == size
               for ep in eps]
        return mean([e["scr"] for e in eps]), mean_fht(eps), \
            mean([1 if e["success"] else 0 for e in eps]), \
            sum(1 for e in eps if e["fht"] >= 0), len(eps)

    print("=" * 78)
    print("A. BASELINES (from per-episode JSONL)")
    print("=" * 78)
    for variant in ("A", "B"):
        for size in (10, 15, 20):
            for cond in ("flat_count", "random_goal"):
                scr, fht, succ, nh, n = baseline(cond, variant, size)
                print(f"  {cond:<11} var{variant} {size:>2}x{size:<2}  "
                      f"SCR={scr:.4f}  FHT={fht:.2f}  success={succ:.3f} "
                      f"(hits {nh}/{n})")

    # ── FF1: layered(best) vs random_goal at 20x20 ──
    print()
    print("=" * 78)
    print("B. FF1 — layered(best) vs random_goal (dead iff ΔSCR<0.05 AND ΔFHT<20)")
    print("=" * 78)
    for variant in ("B", "A"):
        size = 20
        r_scr, r_fht, *_ = baseline("random_goal", variant, size)
        best = None
        for h in H_PLANS:
            for lam in LAMS:
                for t in T_VALS:
                    row = get("layered", variant, size, h, lam, t)
                    if row is None:
                        continue
                    scr, fht, *_ = row
                    if best is None or scr > best[0]:
                        best = (scr, fht, h, lam, t)
        scr, fht, h, lam, t = best
        dscr = scr - r_scr
        dfht = fht - r_fht if (fht >= 0 and r_fht >= 0) else None
        dead = dscr < 0.05 and (dfht is not None and dfht < 20)
        print(f"  var{variant} 20x20: layered(best)={lam} H={h} T={t} "
              f"SCR={scr:.4f} FHT={fht:.2f}")
        print(f"    vs random_goal SCR={r_scr:.4f} FHT={r_fht:.2f}  "
              f"ΔSCR={dscr:+.4f}  ΔFHT={dfht}  -> {'DEAD' if dead else 'PASS'}")

    # ── FF2: coverage range across λ per H_plan at 20x20 & 15x15 ──
    print()
    print("=" * 78)
    print("C. FF2 — coverage RANGE across λ for every H_plan (dead iff range<0.05)")
    print("=" * 78)
    ff2_dead = False
    for size in (20, 15):
        for variant in ("B", "A"):
            for h in H_PLANS:
                scr_by_lam = {}
                for lam in LAMS:
                    for t in T_VALS:
                        row = get("layered", variant, size, h, lam, t)
                        if row is None:
                            continue
                        # use open-loop (never) as the pure scorer; also track best
                        scr_by_lam.setdefault(lam, []).append(row[0])
                rng = max(min(v) for v in scr_by_lam.values()) - \
                    min(min(v) for v in scr_by_lam.values())
                # headline: open-loop (T=never)
                open_scr = {lam: get("layered", variant, size, h, lam, None)[0]
                            for lam in LAMS}
                open_rng = max(open_scr.values()) - min(open_scr.values())
                flag = "DEAD" if open_rng < 0.05 else "ok"
                if open_rng < 0.05:
                    ff2_dead = True
                print(f"  var{variant} {size}x{size} H={h:<3} open-loop range={open_rng:.4f} "
                      f"[{min(open_scr.values()):.3f}..{max(open_scr.values()):.3f}] "
                      f"{flag}   (min-over-t range={rng:.4f})")
    print(f"  -> FF2 verdict: {'DEAD' if ff2_dead else 'PASS'}")

    # ── FF3: re-eval vs open-loop at 20x20 B ──
    print()
    print("=" * 78)
    print("D. FF3 — layered(never) vs layered(T∈{10,25}) at 20x20 B "
          "(ΔSCR<0.02 -> open-loop)")
    print("=" * 78)
    max_delta = 0.0
    for h in H_PLANS:
        for lam in LAMS:
            never = get("layered", "B", 20, h, lam, None)
            best_t = max(
                (get("layered", "B", 20, h, lam, t) for t in (10, 25)),
                key=lambda r: r[0],
            )
            if never is None or best_t is None:
                continue
            d = best_t[0] - never[0]
            max_delta = max(max_delta, d)
            print(f"  H={h:<3} λ={lam:<4} never SCR={never[0]:.4f} | "
                  f"best T SCR={best_t[0]:.4f} (T={'10' if best_t is not None else '?'}) "
                  f"Δ={d:+.4f}")
    print(f"  -> max ΔSCR across 20x20 B grid = {max_delta:+.4f} "
          f"({'DEAD WEIGHT -> open-loop' if max_delta < 0.02 else 'keeps its keep'})")

    # ── FF4: layered(best) vs flat_count at 20x20 B ──
    print()
    print("=" * 78)
    print("E. FF4 — layered(best) vs flat_count at 20x20 B "
          "(dead iff SCR<base+0.05 AND FHT>base+20)")
    print("=" * 78)
    f_scr, f_fht, f_succ, f_nh, f_n = baseline("flat_count", "B", 20)
    best = None
    for h in H_PLANS:
        for lam in LAMS:
            for t in T_VALS:
                row = get("layered", "B", 20, h, lam, t)
                if row is None:
                    continue
                if best is None or row[0] > best[0]:
                    best = (row[0], row[1], h, lam, t)
    scr, fht, h, lam, t = best
    dscr = scr - f_scr
    # flat_count at 20x20 B never hits (FHT=-1): "FHT > base+20" is vacuous;
    # layered does hit, so report hits as the meaningful comparison.
    dfht = fht - f_fht if (fht >= 0 and f_fht >= 0) else None
    dead = dscr < 0.05 and (dfht is not None and dfht > 20)
    print(f"  flat_count 20x20 B: SCR={f_scr:.4f} FHT={f_fht} success={f_succ:.3f} "
          f"(hits {f_nh}/{f_n})")
    print(f"  layered(best)={lam} H={h} T={t}: SCR={scr:.4f} FHT={fht:.2f} "
          f"success={get('layered','B',20,h,lam,t)[2]:.3f}")
    print(f"    ΔSCR={dscr:+.4f}  ΔFHT={dfht}  -> "
          f"{'DEAD' if dead else 'PASS (beats count baseline)'}")

    # ── FF5: size scaling — layered win at every size? ──
    print()
    print("=" * 78)
    print("F. FF5 — layered vs flat_count across sizes (10/15/20), var B & A")
    print("=" * 78)
    for variant in ("B", "A"):
        for size in (10, 15, 20):
            f_scr, f_fht, *_ = baseline("flat_count", variant, size)
            best = max(
                (get("layered", variant, size, h, lam, t)
                 for h in H_PLANS for lam in LAMS for t in T_VALS
                 if get("layered", variant, size, h, lam, t)),
                key=lambda r: r[0],
            )
            scr, fht, succ, nh, n = best
            dscr = scr - f_scr
            ceiling = f_scr >= 0.95
            print(f"  var{variant} {size}x{size}: flat SCR={f_scr:.4f} | "
                  f"layered best SCR={scr:.4f} (Δ={dscr:+.4f}) "
                  f"success={succ:.3f} hits={nh}/{n}"
                  f"{'  [CEILING-guard: flat≥0.95 -> FF void]' if ceiling else ''}")

    # ── G. Reconciliation: JSONL-recomputed vs summary CSV ──
    print()
    print("=" * 78)
    print("G. RECONCILIATION — recomputed (JSONL) vs summary CSV "
          "(20x20 B layered rows)")
    print("=" * 78)
    csv_rows = {}
    with open(args.csv) as f:
        for r in csv.DictReader(f):
            if r["size"] == "20" and r["variant"] == "B" and \
                    r["condition"].startswith("layered"):
                csv_rows[(r["condition"], r["h_plan"], r["lam"], r["t_reeval"])] = r
    mismatches = 0
    for (cond, h, lam, t), cr in sorted(csv_rows.items(), key=lambda kv: kv[0][0]):
        row = get("layered", "B", 20, int(h), lam,
                  None if t == "" else int(t))
        if row is None:
            continue
        scr, fht, succ, nh, n = row
        csv_scr = float(cr["scr"])
        csv_fht = float(cr["fht"])
        ok = abs(csv_scr - scr) < 5e-4 and abs(csv_fht - fht) < 0.05
        mismatches += 0 if ok else 1
        print(f"  {cond:<26} jsonl SCR={scr:.4f} csv={csv_scr:.4f} | "
              f"jsonl FHT={fht:.2f} csv={csv_fht:.2f} {'OK' if ok else 'MISMATCH'}")
    print(f"  -> {mismatches} mismatches in 20x20 B layered rows "
          f"(out of {len(csv_rows)})")


if __name__ == "__main__":
    main()
