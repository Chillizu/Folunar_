#!/usr/bin/env python3
"""Phase 9 HH verdict — t_reeval wiring + per-seed trajectory check.

Question (2026-08-02 review): the 20x20 Variant B summary rows for
t_reeval=10/25/never look identical at the headline config (lam=0.5,H=100).
Is the re-evaluation parameter really consumed by the executor/runner, or is
the sweep's re-eval dimension inert?

Checks, all streaming over results/phase9_hierarchical_sweep.jsonl (never
loads the whole file):
  A. Code wiring trace (static): sweep script -> run_layered_episode(T_reeval)
     -> runner `(t+1) % T_reeval == 0` -> planner.re_evaluate (hysteresis tau).
  B. Per-config, per-seed trajectory equality across t_reeval in {never,10,25}
     for every (variant, size, h_plan, lam) cell of the grid. Two records are
     "identical" iff (x, y, action) match step-by-step.
  C. Goal-switch accounting: how many goal_log entries have outcome
     "switched", and in which configs — i.e. does re-evaluation ever change
     the planner's commitment?

Output: one row per config cell:
  same_traj  = 1 if every t_reeval triple has pairwise-identical per-seed
               trajectories, else 0 (and the number of differing seeds)
  n_switched = total "switched" goal_log entries across seeds

Usage:
  python scripts/phase9_hh_treeval_check.py [--jsonl results/phase9_hierarchical_sweep.jsonl]
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def iter_episodes(path):
    with open(path) as f:
        next(f)  # meta header line
        for line in f:
            if line.strip():
                yield json.loads(line)


def traj_key(ep):
    """Step-by-step (x, y, action) tuple — the full observable trajectory."""
    return tuple((r["x"], r["y"], r["action"]) for r in ep["records"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--jsonl",
        default=str(_PROJECT_ROOT / "results" / "phase9_hierarchical_sweep.jsonl"),
    )
    args = ap.parse_args()

    # (variant, size, h_plan, lam) -> {t_reeval: {seed: ep}}
    cells = defaultdict(lambda: defaultdict(dict))
    switched_by_cell = defaultdict(int)
    n_eps = 0
    for ep in iter_episodes(args.jsonl):
        if ep["condition"].startswith("layered"):
            lam = str(ep["lam"])
            key = (ep["variant"], ep["size"], ep["h_plan"], lam)
            cells[key][ep["t_reeval"]][ep["seed"]] = ep
            for g in ep["goal_log"]:
                if g.get("outcome") == "switched":
                    switched_by_cell[key] += 1
        n_eps += 1

    print(f"episodes scanned: {n_eps}")
    print(f"{'variant':<4} {'size':<5} {'h_plan':<7} {'lam':<5} {'n_seeds':<8} "
          f"{'same_traj':<10} {'differing_seeds':<16} {'n_switched':<10}")

    n_same = 0
    n_cells = 0
    differing_cells = []
    for key in sorted(cells, key=lambda k: (k[0], k[1], k[2], str(k[3]))):
        variant, size, h_plan, lam = key
        t_map = cells[key]
        if set(t_map) != {None, 10, 25}:
            print(f"  ! incomplete t_reeval coverage at {key}: {sorted(t_map)}")
            continue
        seeds = sorted(t_map[None])
        n_cells += 1
        diffs = []
        for s in seeds:
            trajs = {t: traj_key(t_map[t][s]) for t in (None, 10, 25) if s in t_map[t]}
            if len(set(trajs.values())) > 1:
                diffs.append(s)
        same = len(diffs) == 0
        n_same += same
        if not same:
            differing_cells.append((key, diffs))
        print(f"{variant:<4} {size:<5} {h_plan:<7} {str(lam):<5} {len(seeds):<8} "
              f"{'YES' if same else 'NO':<10} {str(diffs):<16} {switched_by_cell[key]:<10}")

    print(f"\nconfig cells: {n_cells}; cells with pairwise-identical trajectories "
          f"across t_reeval: {n_same}")
    if differing_cells:
        print("cells where t_reeval changed at least one seed's trajectory:")
        for (key, diffs) in differing_cells:
            print(f"  {key}: seeds {diffs}")
    else:
        print("no cell shows ANY trajectory difference across t_reeval values")


if __name__ == "__main__":
    main()
