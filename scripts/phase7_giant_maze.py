#!/usr/bin/env python3
# ruff: noqa: E402
"""Phase 7: Giant Maze Scaling Experiment.

Compares count-based vs JEPA-driven exploration at 20x20, 50x50, 100x100.
Hypothesis: at extreme scale, count-based saturates (uniformly high novelty),
while JEPA's learned embedding abstraction provides useful uncertainty.

DO NOT run experiments from this script alone — GPU Manager orchestrates.
Output test_command is for GPU Manager.
"""

import argparse
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from phase7.giant_maze import (
    GiantMaze,
    GiantGridMazeEnv,
    GiantGridState,
    giant_state_to_text,
)
from phase7.giant_jepa import GiantJEPAEnsemble


# ── Metrics ──────────────────────────────────────────────

def compute_metrics(steps, goal_room):
    """Compute FHT, SCR, max distance from start.

    FHT = first hitting time to goal (0 if never reached)
    SCR = success rate (0 or 1 per episode)
    max_dist = farthest Manhattan distance from (0,0) reached
    """
    fht = -1
    for i, rec in enumerate(steps):
        if rec.get("goal_reached"):
            fht = i
            break

    scr = 1.0 if fht >= 0 else 0.0

    max_dist = 0
    for rec in steps:
        d = abs(rec["x"] - 0) + abs(rec["y"] - 0)
        if d > max_dist:
            max_dist = d

    return {
        "fht": fht,
        "scr": round(scr, 3),
        "max_dist": max_dist,
        "steps": len(steps),
    }


# ── Direction helpers (same pattern as phase6) ──────────

_REVERSE_MOVE = {
    "go north": "go south",
    "go south": "go north",
    "go east": "go west",
    "go west": "go east",
}


def _reverse_action(action: str) -> str:
    return _REVERSE_MOVE.get(action, "")


_DX = {
    "go north": (0, -1),
    "go south": (0, 1),
    "go east": (1, 0),
    "go west": (-1, 0),
}


# ── Count-based Explorer (adapted from MazeNoveltyExplorer) ──

class GiantNoveltyExplorer:
    """Count-based novelty explorer for giant maze with backtrack penalty."""

    def __init__(self):
        self.state_counts = {}
        self.state_action_counts = {}
        self.success_cache = {}
        self._prev_pos = None
        self._prev_pos_action = None

    def novelty_bonus(self, state: GiantGridState, action: str) -> float:
        """Intrinsic reward: 1/sqrt(count), penalize backtracks 50%."""
        sh = state.state_hash()
        state_novelty = 1.0 / math.sqrt(1 + self.state_counts.get(sh, 0))
        pair_novelty = 1.0 / math.sqrt(
            1 + self.state_action_counts.get((sh, action), 0)
        )
        bonus = 0.5 * state_novelty + 0.5 * pair_novelty

        # Backtrack penalty
        if self._prev_pos is not None and action == _reverse_action(
            self._prev_pos_action
        ):
            move = _DX.get(action)
            if move:
                dest = (state.x + move[0], state.y + move[1])
                if dest == self._prev_pos:
                    bonus *= 0.5

        return bonus

    @staticmethod
    def _action_priority(action: str) -> int:
        verb = action.split()[0] if action else ""
        priorities = {"go": 0, "look": 1, "take": 2, "inventory": 3}
        return priorities.get(verb, 4)

    def select_action(self, state, candidates, action_history):
        """Pick highest novelty, tiebreak by priority."""
        if not candidates:
            return "look"

        sh = state.state_hash()
        if sh in self.success_cache:
            cached = self.success_cache[sh]
            if cached in candidates:
                return cached

        return max(candidates, key=lambda a: (
            self.novelty_bonus(state, a),
            -self._action_priority(a),
        ))

    def observe(self, state, action: str, success: bool):
        sh = state.state_hash()
        self.state_counts[sh] = self.state_counts.get(sh, 0) + 1
        pair_key = (sh, action)
        self.state_action_counts[pair_key] = (
            self.state_action_counts.get(pair_key, 0) + 1
        )
        if success:
            self.success_cache[sh] = action

    def observe_move(self, action: str, pos_before, pos_after):
        self._prev_pos = pos_before
        self._prev_pos_action = action

    def reset_episode(self):
        self._prev_pos = None
        self._prev_pos_action = None


# ── JEPA Explorer ────────────────────────────────────────

class GiantJEPAExplorer:
    """JEPA-driven selector for giant maze.

    Uses ensemble epistemic variance as intrinsic reward.
    In 'jepa' mode: only epistemic signal (no count-based).
    """

    def __init__(self, jepa: GiantJEPAEnsemble):
        self.jepa = jepa
        self._prev_pos = None
        self._prev_pos_action = None

    def select_action(self, state: GiantGridState, candidates, action_history):
        """Pick highest epistemic uncertainty."""
        if not candidates:
            return "look"

        state_text = giant_state_to_text(state)

        # Backtrack penalty flag
        backtrack_penalty = False
        if self._prev_pos is not None:
            for a in candidates:
                if a == _reverse_action(self._prev_pos_action):
                    move = _DX.get(a)
                    if move:
                        dest = (state.x + move[0], state.y + move[1])
                        if dest == self._prev_pos:
                            backtrack_penalty = True

        def score(action):
            epi = self.jepa.epistemic_uncertainty(state_text, action)
            # 20% penalty for backtracks
            if backtrack_penalty and action == _reverse_action(
                self._prev_pos_action
            ):
                move = _DX.get(action)
                if move:
                    dest = (state.x + move[0], state.y + move[1])
                    if dest == self._prev_pos:
                        epi *= 0.8
            return epi

        return max(candidates, key=score)

    def observe_move(self, action: str, pos_before, pos_after):
        self._prev_pos = pos_before
        self._prev_pos_action = action

    def reset_episode(self):
        self._prev_pos = None
        self._prev_pos_action = None


# ── Episode Runners ──────────────────────────────────────

def run_count_episode(env, explorer, max_steps):
    """Run a single count-based episode."""
    env.reset()
    steps = []

    for step_i in range(max_steps):
        state = env._get_state()
        candidates = env.get_dynamic_candidates()
        action = explorer.select_action(state, candidates, steps)

        goal_room = env.task.get("goal_room")
        prev_dist = abs(state.x - goal_room[0]) + abs(state.y - goal_room[1])
        next_obs, next_state, done = env.step(action)
        new_dist = abs(next_state.x - goal_room[0]) + abs(
            next_state.y - goal_room[1]
        )
        success = (new_dist < prev_dist) or next_state.goal_reached

        explorer.observe(state, action, success)

        if (next_state.x, next_state.y) != (state.x, state.y):
            explorer.observe_move(
                action, (state.x, state.y), (next_state.x, next_state.y)
            )

        record = {
            "step": step_i,
            "x": next_state.x,
            "y": next_state.y,
            "action": action,
            "success": bool(success),
            "goal_reached": next_state.goal_reached,
            "inventory": list(env.inventory),
        }
        steps.append(record)

        if next_state.goal_reached:
            break

    return steps, env._get_state()


def run_jepa_episode(env, explorer, jepa, max_steps):
    """Run a single JEPA-driven episode with down-sampled training."""
    env.reset()
    steps = []
    transitions = []

    if jepa is not None:
        jepa._cache_clear()

    for step_i in range(max_steps):
        state = env._get_state()
        candidates = env.get_dynamic_candidates()

        # Select with NO action_history (kept to prevent crash on mismatched signature)
        action = explorer.select_action(state, candidates, steps)

        goal_room = env.task.get("goal_room")
        next_obs, next_state, done = env.step(action)

        # Record transition for JEPA training
        transitions.append((state, action, next_state))

        # Success = moved toward or reached goal
        prev_dist = abs(state.x - goal_room[0]) + abs(state.y - goal_room[1])
        new_dist = abs(next_state.x - goal_room[0]) + abs(
            next_state.y - goal_room[1]
        )
        success = (new_dist < prev_dist) or next_state.goal_reached

        if (next_state.x, next_state.y) != (state.x, state.y):
            explorer.observe_move(
                action, (state.x, state.y), (next_state.x, next_state.y)
            )

        record = {
            "step": step_i,
            "x": next_state.x,
            "y": next_state.y,
            "action": action,
            "success": bool(success),
            "goal_reached": next_state.goal_reached,
            "inventory": list(env.inventory),
        }
        steps.append(record)

        if next_state.goal_reached:
            break

    # Down-sampled JEPA training
    train_loss = None
    if jepa is not None and transitions:
        train_loss = jepa.train_step(
            transitions,
            state_to_text_fn=giant_state_to_text,
            max_samples=5,
        )

    return steps, env._get_state(), train_loss


# ── Experiment Runner ────────────────────────────────────

SIZE_CONFIGS = {
    "20x20": {"width": 20, "height": 20, "max_steps": 500, "episodes": 3},
    "50x50": {"width": 50, "height": 50, "max_steps": 1000, "episodes": 3},
    "100x100": {"width": 100, "height": 100, "max_steps": 2000, "episodes": 3},
}

METHODS = ["count", "jepa"]


def run_size(
    size_tag: str,
    cfg: dict,
    model_path: str,
    output_dir: Path,
    seed: int = 42,
):
    """Run count and JEPA experiments for one size. Returns results list."""
    width = cfg["width"]
    height = cfg["height"]
    max_steps = cfg["max_steps"]
    episodes = cfg["episodes"]

    results = []

    print(f"\n{'=' * 60}")
    print(f"Size: {size_tag} ({width}x{height}, {max_steps} steps, {episodes} eps)")
    print(f"{'=' * 60}", flush=True)

    # Generate maze
    print(f"  Generating {width}x{height} maze...", flush=True)
    t0 = time.time()
    maze = GiantMaze.generate(width, height, seed=seed)
    gen_time = time.time() - t0
    print(f"  Maze generated in {gen_time:.2f}s "
          f"(walls={len(maze.walls)}, items={sum(1 for v in maze.room_items.values() if v)})",
          flush=True)

    goal_room = (width - 1, height - 1)
    state_estimate = maze.state_estimate()
    print(f"  State estimate: ~{state_estimate}")
    print(f"  Goal: ({goal_room[0]}, {goal_room[1]})", flush=True)

    task = {
        "goal_room": goal_room,
        "start_x": 0,
        "start_y": 0,
        "max_steps": max_steps,
    }

    # Load JEPA once (shared across all JEPA episodes for this size)
    jepa = None
    if "jepa" in METHODS:
        device = "cuda" if _torch_available() and __import__("torch").cuda.is_available() else "cpu"
        print(f"\n  Loading JEPA ensemble on {device}...", flush=True)
        t0 = time.time()
        jepa = GiantJEPAEnsemble(
            model_path, n_ensemble=2, hidden_dim=128, device=device,
        )
        print(f"  JEPA loaded in {time.time() - t0:.1f}s", flush=True)

    for method in METHODS:
        print(f"\n  {'─' * 50}")
        print(f"  Method: {method.upper()}")
        print(f"  {'─' * 50}", flush=True)

        for ep_idx in range(episodes):
            ep_seed = seed + 1000 * ({"count": 0, "jepa": 1}[method]) + ep_idx
            random.seed(ep_seed)

            print(f"    [Episode {ep_idx + 1}/{episodes}] ", end="", flush=True)
            t0 = time.time()

            env = GiantGridMazeEnv(maze, task)
            env.reset()

            try:
                if method == "count":
                    explorer = GiantNoveltyExplorer()
                    explorer.reset_episode()
                    steps, final_state = run_count_episode(
                        env, explorer, max_steps,
                    )
                    train_loss = None
                else:
                    explorer = GiantJEPAExplorer(jepa)
                    explorer.reset_episode()
                    steps, final_state, train_loss = run_jepa_episode(
                        env, explorer, jepa, max_steps,
                    )
            except Exception as e:
                print(f"ERROR: {e}", flush=True)
                import traceback
                traceback.print_exc()
                steps = []
                final_state = env._get_state()
                train_loss = None

            elapsed = time.time() - t0

            if steps:
                metrics = compute_metrics(steps, goal_room)
            else:
                metrics = {"fht": -1, "scr": 0.0, "max_dist": 0}

            result = {
                "size": size_tag,
                "width": width,
                "height": height,
                "method": method,
                "seed": ep_seed,
                "episode": ep_idx,
                "steps_count": len(steps),
                "success": final_state.goal_reached,
                "fht": metrics.get("fht", -1),
                "scr": metrics.get("scr", 0.0),
                "max_dist": metrics.get("max_dist", 0),
                "train_loss": round(train_loss, 6) if train_loss is not None else None,
                "goal_x": goal_room[0],
                "goal_y": goal_room[1],
                "elapsed": round(elapsed, 1),
            }
            print(
                f" success={result['success']} scr={result['scr']} "
                f"max_dist={result['max_dist']} "
                f"loss={result['train_loss']} [{elapsed:.0f}s]",
                flush=True,
            )

            results.append(result)

            # Incremental write
            output_path = output_dir / f"phase7_giant_{size_tag}.jsonl"
            with open(output_path, "a") as f:
                f.write(json.dumps(result) + "\n")

    return results


def _torch_available() -> bool:
    try:
        import torch
        return True
    except ImportError:
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Phase 7 Giant Maze Scaling Experiment"
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        default=["20x20", "50x50", "100x100"],
        choices=list(SIZE_CONFIGS.keys()),
        help="Maze sizes to test (default: all)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--model-path",
        default=None,
        help="Path to Qwen model (default: ~/models/Qwen2.5-0.5B-Instruct)",
    )
    parser.add_argument(
        "--method",
        choices=["count", "jepa", "all"],
        default="all",
        help="Method to run (default: both)",
    )
    args = parser.parse_args()

    if args.model_path is None:
        args.model_path = os.path.expanduser("~/models/Qwen2.5-0.5B-Instruct")

    if args.output_dir is None:
        args.output_dir = _PROJECT_ROOT / "results"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    global METHODS
    if args.method != "all":
        METHODS = [args.method]

    # Run each size
    all_results = []
    for size_tag in args.sizes:
        cfg = SIZE_CONFIGS[size_tag]
        results = run_size(
            size_tag, cfg, args.model_path, output_dir, seed=args.seed,
        )
        all_results.extend(results)

    # ── Final Summary ────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"Phase 7 Giant Maze — Final Summary")
    print(f"{'=' * 60}")

    for size_tag in args.sizes:
        cfg = SIZE_CONFIGS[size_tag]
        size_results = [r for r in all_results if r["size"] == size_tag]

        for method in ["count", "jepa"]:
            method_results = [r for r in size_results if r["method"] == method]
            if not method_results:
                continue

            successes = sum(1 for r in method_results if r["success"])
            avg_scr = sum(r["scr"] for r in method_results) / len(method_results)
            max_dists = [r["max_dist"] for r in method_results]
            avg_max_dist = sum(max_dists) / len(max_dists)
            losses = [
                r.get("train_loss")
                for r in method_results
                if r.get("train_loss") is not None
            ]
            avg_loss = sum(losses) / max(len(losses), 1) if losses else -1

            print(
                f"  {size_tag:>8s} {method:>6s}: "
                f"{successes}/{len(method_results)} success | "
                f"SCR={avg_scr:.3f} | "
                f"max_dist={avg_max_dist:.0f} | "
                f"loss={avg_loss if avg_loss >= 0 else -1:.6f}"
            )

    # Write combined results
    combined_path = output_dir / "phase7_giant_all.jsonl"
    with open(combined_path, "w") as f:
        for r in all_results:
            f.write(json.dumps(r) + "\n")
    print(f"\nCombined results: {combined_path}")

    # Quick comparison table
    print(f"\n{'─' * 60}")
    print(f"Cross-Size Comparison (key metric: max_dist):")
    print(f"{'Size':>8s}  {'Count SCR':>10s}  {'JEPA SCR':>10s}  "
          f"{'Count Dist':>11s}  {'JEPA Dist':>11s}")
    for size_tag in args.sizes:
        size_results = [r for r in all_results if r["size"] == size_tag]
        count = [r for r in size_results if r["method"] == "count"]
        jepa_r = [r for r in size_results if r["method"] == "jepa"]
        c_scr = sum(r["scr"] for r in count) / max(len(count), 1) if count else -1
        j_scr = sum(r["scr"] for r in jepa_r) / max(len(jepa_r), 1) if jepa_r else -1
        c_dist = sum(r["max_dist"] for r in count) / max(len(count), 1) if count else -1
        j_dist = sum(r["max_dist"] for r in jepa_r) / max(len(jepa_r), 1) if jepa_r else -1
        print(
            f"{size_tag:>8s}  {c_scr:>10.3f}  {j_scr:>10.3f}  "
            f"{c_dist:>11.0f}  {j_dist:>11.0f}"
        )

    print(f"\n{'=' * 60}", flush=True)


# Lazy torch import
try:
    import torch  # noqa: F401
except ImportError:
    pass

if __name__ == "__main__":
    main()
