#!/usr/bin/env python3
# ruff: noqa: E402
"""Phase 9 Hierarchical Horizon sweep (plan-hierarchical-horizon.md §5-§6).

Two-layer architecture: a high-level planner scores frontier/key goals by
analytic delayed reward J(f) = G(f) - lambda*d over horizon H_plan (20-100);
the low level is a pure pragmatic path-finder (BFS) / Phase-6 count-novelty
wanderer. The sweep tests whether long-horizon goal commitment beats (a)
flat count-based novelty and (b) random goal choice, on plain mazes (Variant
A) and locked-door mazes (Variant B).

Conditions (all paired over the same maze seeds):
  flat_count            — Phase-6 MazeNoveltyExplorer baseline
  random_goal           — uniform random frontier + same executors (FF1 control)
  layered(λ, H_plan)    — §2 scorer, open loop
  layered(λ, H_plan, T_reeval) — §2 scorer + §4 re-evaluation loop

Grid: size ∈ {10, 15, 20} × H_plan ∈ {20, 50, 100} × λ ∈ {0, 0.5, 1, 2, inf}
      × T_reeval ∈ {never, 10, 25} × variant ∈ {A, B}, seeds 42..53.

Output (WATCHDOG D4 Results Metadata Standard):
  <prefix>.jsonl — line 1 = meta header block, then one JSON object per
                   episode with full per-episode step records.
  <prefix>_summary.csv — per-config aggregates + paired deltas.

Usage:
  # Single config (acceptance smoke): layered, 10x10, variant A
  python scripts/phase9_hierarchical_sweep.py

  # Full pre-registered grid
  python scripts/phase9_hierarchical_sweep.py --full

  # Targeted single config
  python scripts/phase9_hierarchical_sweep.py --size 20 --variant B \
      --condition layered --lam inf --h-plan 50 --t-reeval 25
"""

import argparse
import csv
import json
import math
import random
import socket
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))

from phase6.grid_env import GridMazeEnv  # noqa: E402, I001
from phase6.maze_generator import GridMaze, MazeTask  # noqa: E402, I001
from phase6_maze_count import MazeNoveltyExplorer, run_count_episode  # noqa: E402, I001
from phase9.hierarchical.executor import LayeredExecutor  # noqa: E402, I001
from phase9.hierarchical.planner import HighLevelPlanner  # noqa: E402, I001
from phase9.hierarchical.runner import run_layered_episode  # noqa: E402, I001

SIZES = [10, 15, 20]
H_PLANS = [20, 50, 100]
LAMS = [0.0, 0.5, 1.0, 2.0, float("inf")]
T_REEVALS = [None, 10, 25]
VARIANTS = ["A", "B"]
SEED_START, SEED_END = 42, 53  # inclusive: 12 seeds, seed 42 matches phase6 baseline


class RandomGoalPlanner(HighLevelPlanner):
    """FF1 control arm: uniform random frontier selection (§6).

    Same executors as `layered`; only goal *choice* differs (uniform random
    over frontiers instead of argmax J). No re-evaluation (plan §6).
    """

    def __init__(self, maze, seed=None):
        super().__init__(maze)
        self._rng = random.Random(seed)

    def propose_goals(self):
        return [g for g in super().propose_goals() if g.kind == "nav"]

    def select(self, H_plan, lam):
        cands = self.propose_goals()
        if not cands:
            self.current_goal = None
            self._last_G = None
            return None
        g = self._rng.choice(cands)
        self.current_goal = g
        self._last_G = self._score_detail(g, H_plan, lam)[2]
        return g

    def re_evaluate(self, H_plan, lam, tau=0.15):
        return self.current_goal


# ── helpers ──────────────────────────────────────────────

def lam_str(lam):
    return "inf" if math.isinf(lam) else f"{lam:g}"


def t_str(t):
    return "never" if t is None else str(t)


def condition_id(c):
    if c["condition"] == "flat_count":
        return "flat_count"
    if c["condition"] == "random_goal":
        return "random_goal"
    return f"layered_lam{lam_str(c['lam'])}_h{c['h_plan']}_t{t_str(c['t_reeval'])}"


def max_steps_for(size):
    return min(size * size * 4, 500)  # 400 at 10x10, 500 at 20x20 (phase6 convention)


def make_env(size, variant, seed):
    """Fresh maze + env (env mutates room_items on take, so never reuse)."""
    task = MazeTask(
        name=f"phase9_hier_{variant}_{size}",
        size=size,
        goal_room=(size - 1, size - 1),
        max_steps=max_steps_for(size),
        locked_doors=2 if variant == "B" else 0,
    )
    maze = GridMaze.generate(size, size, task, seed=seed)
    env = GridMazeEnv(maze, asdict(task))
    env.setup()
    return env, maze


# ── metrics ──────────────────────────────────────────────

def compute_metrics(records, goal_room, size):
    """SCR (per-cell coverage, incl. start), FHT, dead_loop_rate (Phase-6),
    new_states_per_step."""
    seen = {(0, 0)}
    fht = None
    for i, rec in enumerate(records):
        cell = (rec["x"], rec["y"])
        seen.add(cell)
        if fht is None and cell == tuple(goal_room):
            fht = i
    loops = sum(
        1
        for i in range(2, len(records))
        if records[i]["action"] == records[i - 1]["action"] == records[i - 2]["action"]
    )
    n = max(len(records), 1)
    return {
        "scr": round(len(seen) / (size * size), 4),
        "fht": fht if fht is not None else -1,
        "dead_loop_rate": round(loops / n, 4),
        "new_states_per_step": round((len(seen) - 1) / n, 4),
    }


def record_to_dict(rec, step_i):
    return {
        "step": step_i,
        "x": rec.cell_after[0],
        "y": rec.cell_after[1],
        "action": rec.action,
        "content_new": rec.content_new,
        "goal_reached": rec.goal_reached,
    }


def pe_metrics(goal_log):
    by_kind = {}
    for e in goal_log:
        if e.get("pe") is None:
            continue
        by_kind.setdefault(e["kind"], []).append(e["pe"])
    return {k: round(sum(v) / len(v), 4) for k, v in sorted(by_kind.items())}


# ── condition runners ────────────────────────────────────

def run_flat_count(env, seed, max_steps):
    random.seed(seed)
    explorer = MazeNoveltyExplorer()
    explorer.reset_episode()
    steps, final = run_count_episode(env, explorer, max_steps)
    return steps, [], final


def run_condition(env, condition, seed, size, h_plan, lam, t_reeval, max_steps):
    """Run one episode; returns (record_dicts, goal_log, final_state)."""
    if condition == "flat_count":
        return run_flat_count(env, seed, max_steps)

    if condition == "random_goal":
        random.seed(seed)
        planner = RandomGoalPlanner(env.maze, seed=seed)
        executor = LayeredExecutor(env.maze, blocked=planner.is_edge_blocked)
        records, goal_log, final = run_layered_episode(
            env, planner, executor,
            H_plan=h_plan, lam=lam, T_reeval=None, max_steps=max_steps,
        )
        return [record_to_dict(r, i) for i, r in enumerate(records)], goal_log, final

    # layered
    random.seed(seed)
    planner = HighLevelPlanner(env.maze)
    executor = LayeredExecutor(env.maze, blocked=planner.is_edge_blocked)
    records, goal_log, final = run_layered_episode(
        env, planner, executor,
        H_plan=h_plan, lam=lam, T_reeval=t_reeval, max_steps=max_steps,
    )
    return [record_to_dict(r, i) for i, r in enumerate(records)], goal_log, final


# ── sweep driver ─────────────────────────────────────────

def build_configs():
    """Full pre-registered grid: variant × size × {flat, random} × H×λ×T."""
    configs = []
    for variant in VARIANTS:
        for size in SIZES:
            configs.append({"condition": "flat_count", "variant": variant, "size": size})
            configs.append({"condition": "random_goal", "variant": variant, "size": size})
            for h_plan in H_PLANS:
                for lam in LAMS:
                    for t_reeval in T_REEVALS:
                        configs.append({
                            "condition": "layered", "variant": variant, "size": size,
                            "h_plan": h_plan, "lam": lam, "t_reeval": t_reeval,
                        })
    return configs


def write_meta(jsonl_path, seeds):
    """WATCHDOG D4 Results Metadata Standard header block."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, cwd=_PROJECT_ROOT, timeout=5,
        ).stdout.strip()
    except Exception:
        commit = "unknown"
    meta = {
        "meta": {
            "phase": "9",
            "direction": "hierarchical-horizon",
            "commit": commit,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "host": socket.gethostname(),
            "cpu_or_gpu": "cpu",
            "sandbox_image": "n/a (grid maze)",
            "model": "none (geometric planner, no LLM)",
            "seeds": list(seeds),
            "per_episode_data_present": True,
        }
    }
    with open(jsonl_path, "w") as f:
        f.write(json.dumps(meta) + "\n")


def run_config(config, seeds, jsonl_path, results):
    """Run one config over all seeds; append per-episode JSONL; collect."""
    variant, size = config["variant"], config["size"]
    condition = config["condition"]
    h_plan = config.get("h_plan", 50)
    lam = config.get("lam", 1.0)
    t_reeval = config.get("t_reeval", None)
    cid = condition_id(config)
    goal_room = (size - 1, size - 1)

    for seed in seeds:
        env, maze = make_env(size, variant, seed)
        records, goal_log, final = run_condition(
            env, condition, seed, size, h_plan, lam, t_reeval,
            max_steps_for(size),
        )
        line = {
            "condition": cid,
            "variant": variant,
            "size": size,
            "seed": seed,
            "h_plan": h_plan,
            "lam": lam if not math.isinf(lam) else "inf",
            "t_reeval": t_reeval,
            "success": bool(final.goal_reached),
            **compute_metrics(records, goal_room, size),
            "pe_by_kind": pe_metrics(goal_log),
            "goal_log": goal_log,
            "records": records,
        }
        with open(jsonl_path, "a") as f:
            f.write(json.dumps(line) + "\n")
        results.append(line)


def _mean_fht(rs):
    hits = [r["fht"] for r in rs if r["fht"] >= 0]
    return sum(hits) / len(hits) if hits else -1.0


def summarize(results, csv_path):
    """Aggregate over seeds; paired deltas vs random_goal and flat_count."""
    baseline = {}
    for r in results:
        if r["condition"] in ("flat_count", "random_goal"):
            baseline.setdefault((r["variant"], r["size"], r["condition"]), []).append(r)
    base_mean = {
        key: {"scr": sum(x["scr"] for x in rs) / len(rs), "fht": _mean_fht(rs)}
        for key, rs in baseline.items()
    }

    groups = {}
    for r in results:
        key = (
            r["condition"], r["variant"], r["size"],
            r.get("h_plan"), r.get("lam"), r.get("t_reeval"),
        )
        groups.setdefault(key, []).append(r)

    rows = []
    for key in sorted(groups, key=lambda k: (k[2], k[1], str(k[0]))):
        cond, variant, size, h_plan, lam, t_reeval = key
        rs = groups[key]
        n = len(rs)
        scr = sum(x["scr"] for x in rs) / n
        fht = _mean_fht(rs)
        dlr = sum(x["dead_loop_rate"] for x in rs) / n
        nsps = sum(x["new_states_per_step"] for x in rs) / n
        succ = sum(1 for x in rs if x["success"]) / n
        pe: dict = {}
        for x in rs:
            for k, v in x["pe_by_kind"].items():
                pe.setdefault(k, []).append(v)
        pe_mean = {k: round(sum(v) / len(v), 4) for k, v in pe.items()}

        row = {
            "condition": cond, "variant": variant, "size": size,
            "h_plan": h_plan, "lam": lam, "t_reeval": t_reeval,
            "n_seeds": n,
            "scr": round(scr, 4),
            "fht": round(fht, 2) if fht >= 0 else -1,
            "dead_loop_rate": round(dlr, 4),
            "new_states_per_step": round(nsps, 4),
            "success_rate": round(succ, 4),
            "pe_nav": pe_mean.get("nav", ""),
            "pe_search": pe_mean.get("search", ""),
            "pe_acquire": pe_mean.get("acquire", ""),
        }
        if cond != "flat_count":
            rb = base_mean.get((variant, size, "random_goal"))
            if rb:
                row["dscr_vs_random"] = round(scr - rb["scr"], 4)
                row["dfht_vs_random"] = (
                    round(fht - rb["fht"], 2)
                    if fht >= 0 and rb["fht"] >= 0 else None
                )
        if cond != "random_goal":
            fb = base_mean.get((variant, size, "flat_count"))
            if fb:
                row["dscr_vs_flat"] = round(scr - fb["scr"], 4)
                row["dfht_vs_flat"] = (
                    round(fht - fb["fht"], 2)
                    if fht >= 0 and fb["fht"] >= 0 else None
                )
        rows.append(row)

    if rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    return rows


# ── CLI ──────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--full", action="store_true",
                   help="run the full pre-registered grid (size×H×λ×T×variant)")
    p.add_argument("--size", type=int, default=10)
    p.add_argument("--variant", choices=["A", "B"], default="A")
    p.add_argument("--condition", choices=["flat_count", "random_goal", "layered"],
                   default="layered")
    p.add_argument("--lam", default="1.0", help="λ: a number or 'inf'")
    p.add_argument("--h-plan", type=int, default=50)
    p.add_argument("--t-reeval", default="25", help="'never' or step count")
    p.add_argument("--seed-start", type=int, default=SEED_START)
    p.add_argument("--seed-end", type=int, default=SEED_END)
    p.add_argument("--out", default=None, help="output prefix (writes .jsonl + _summary.csv)")
    return p.parse_args()


def main():
    args = parse_args()
    lam = float("inf") if args.lam == "inf" else float(args.lam)
    t_reeval = None if args.t_reeval == "never" else int(args.t_reeval)
    seeds = list(range(args.seed_start, args.seed_end + 1))

    if args.out:
        prefix = args.out
    else:
        prefix = str(_PROJECT_ROOT / "results" / "phase9_hierarchical_sweep")
    jsonl_path = Path(prefix + ".jsonl")
    csv_path = Path(prefix + "_summary.csv")
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    write_meta(jsonl_path, seeds)

    if args.full:
        configs = build_configs()
    else:
        configs = [{
            "condition": args.condition, "variant": args.variant, "size": args.size,
            "h_plan": args.h_plan, "lam": lam, "t_reeval": t_reeval,
        }]

    print(f"Phase 9 Hierarchical Horizon sweep — {len(configs)} config(s) × "
          f"{len(seeds)} seeds ({seeds[0]}..{seeds[-1]})")
    print(f"JSONL: {jsonl_path}")
    print(f"CSV:   {csv_path}", flush=True)

    results = []
    t0 = time.time()
    for i, config in enumerate(configs):
        run_config(config, seeds, jsonl_path, results)
        print(f"  [{i + 1}/{len(configs)}] {condition_id(config)} "
              f"variant={config['variant']} size={config['size']} done",
              flush=True)

    rows = summarize(results, csv_path)
    print(f"\nDone in {time.time() - t0:.0f}s — {len(results)} episodes, "
          f"{len(rows)} summary rows (CSV: {csv_path})", flush=True)
    for r in rows[:8]:
        print(f"  {r['condition']:<24} var{r['variant']} {r['size']:>2} "
              f"scr={r['scr']:.3f} fht={r['fht']} dlr={r['dead_loop_rate']:.3f}")


if __name__ == "__main__":
    main()
