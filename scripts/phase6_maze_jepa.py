#!/usr/bin/env python3
# ruff: noqa: E402
"""Phase 6 Maze JEPA Experiment: JEPA epistemic uncertainty in grid mazes.

Uses the JEPAEnsemble (from phase5) to drive exploration via embedding-space
prediction uncertainty. Compares with count-based baseline.

Measures success rate at 4 maze sizes to identify the crossover point where
JEPA-learned epistemic exploration overtakes count-based novelty.

Usage:
  # Single size
  python scripts/phase6_maze_jepa.py --width 10 --height 10 --num-episodes 6

  # All sizes
  for s in 5 10 20 30; do
    python scripts/phase6_maze_jepa.py --width $s --height $s --num-episodes 6 --seed 42
  done
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

from phase5.jepa_wm import JEPAEnsemble, state_to_text, next_state_to_text
from phase6.grid_env import GridMazeEnv, GridState, grid_state_to_text
from phase6.maze_generator import GridMaze


# ── Metrics ──────────────────────────────────────────────

def compute_metrics(steps, goal_room):
    """Compute FHT, SCR, Dead-loop Rate from step records for maze env.

    Args:
        steps: list of step records (dicts).
        goal_room: (x, y) tuple of the goal.

    Returns:
        dict with fht, scr, dead_loop_rate, steps.
    """
    fht = None
    for rec in steps:
        if rec.get("x") == goal_room[0] and rec.get("y") == goal_room[1]:
            fht = rec["step"]
            break

    visited = set()
    for rec in steps:
        visited.add(f"{rec['x']},{rec['y']}|{tuple(rec.get('inventory', []))}")
    scr = len(visited) / max(len(steps), 1)

    loops = 0
    for i in range(2, len(steps)):
        if steps[i]["action"] == steps[i - 1]["action"] == steps[i - 2]["action"]:
            loops += 1
    dead_loop_rate = loops / max(len(steps), 1)

    return {
        "fht": fht,
        "scr": round(scr, 3),
        "dead_loop_rate": round(dead_loop_rate, 3),
        "steps": len(steps),
    }


# ── JEPA Explorer (Maze-Adapted) ─────────────────────────

_REVERSE_MOVE = {
    "go north": "go south",
    "go south": "go north",
    "go east": "go west",
    "go west": "go east",
}


def _reverse_action(action: str) -> str:
    """Return the direction that would undo the given action."""
    return _REVERSE_MOVE.get(action, "")


class MazeJEPAExplorer:
    """Action selector for grid maze driven by JEPA epistemic uncertainty.

    Three modes:
      pure_novelty: count-based only (same as NoveltyExplorer)
      jepa_only: JEPA ensemble variance only
      hybrid: 0.5 * novelty + 0.5 * epistemic

    Includes backtrack penalty to prevent ping-ponging between rooms.
    """

    def __init__(
        self,
        jepa_ensemble: JEPAEnsemble = None,
        mode: str = "hybrid",
        novelty_weight: float = 0.5,
        epistemic_weight: float = 0.5,
    ):
        self.jepa = jepa_ensemble
        self.mode = mode
        self.novelty_weight = novelty_weight
        self.epistemic_weight = epistemic_weight

        # Count-based tables (shared by all modes)
        self.state_counts = defaultdict(int)
        self.state_action_counts = defaultdict(int)
        self.success_cache = {}

        # Backtrack penalty tracking
        self._prev_pos = None
        self._prev_pos_action = None

    def novelty_bonus(self, state: GridState, action: str) -> float:
        """Count-based intrinsic reward with backtrack penalty."""
        sh = state.state_hash()
        state_novelty = 1.0 / max(1.0, self.state_counts[sh] ** 0.5)
        pair_novelty = 1.0 / max(1.0, self.state_action_counts[(sh, action)] ** 0.5)
        bonus = 0.5 * state_novelty + 0.5 * pair_novelty

        # Backtrack penalty
        if self._prev_pos is not None and action == _reverse_action(self._prev_pos_action):
            dx = {
                "go north": (0, -1),
                "go south": (0, 1),
                "go east": (1, 0),
                "go west": (-1, 0),
            }
            move = dx.get(action)
            if move:
                dest = (state.x + move[0], state.y + move[1])
                if dest == self._prev_pos:
                    bonus *= 0.5

        return bonus

    def _action_priority(self, action: str) -> int:
        """Lower = prefer. Movement > look > inventory > use > take."""
        verb = action.split()[0] if action else ""
        priorities = {"go": 0, "look": 1, "inventory": 2, "use": 3, "take": 4}
        return priorities.get(verb, 5)

    def select_action(
        self, state: GridState, candidates, action_history, jepa_wm=None
    ):
        """Select the most informative action.

        Decision order:
        1. Cached success -> replay (if action still in candidates)
        2. Highest score (novelty + epistemic), tie-broken by action priority
        """
        if not candidates:
            return "look"

        sh = state.state_hash()

        # Cached success replay
        if sh in self.success_cache:
            cached = self.success_cache[sh]
            if cached in candidates:
                return cached

        # Score all candidates
        state_text = grid_state_to_text(state)

        def score(action):
            if self.mode == "pure_novelty":
                return self.novelty_bonus(state, action)
            elif self.mode == "jepa_only":
                if self.jepa is not None:
                    return self.jepa.epistemic_uncertainty(state_text, action)
                return 0.0
            else:  # hybrid
                nov = self.novelty_bonus(state, action)
                epi = (
                    self.jepa.epistemic_uncertainty(state_text, action)
                    if self.jepa is not None
                    else 0.0
                )
                return self.novelty_weight * nov + self.epistemic_weight * epi

        # Pick highest-scoring, tie-break by action priority
        return max(candidates, key=lambda a: (score(a), -self._action_priority(a)))

    def observe(self, state: GridState, action: str, success: bool):
        """Record execution outcome and update counts."""
        sh = state.state_hash()
        self.state_counts[sh] += 1
        self.state_action_counts[(sh, action)] += 1
        if success:
            self.success_cache[sh] = action

    def observe_move(self, action: str, pos_before, pos_after):
        """Track a successful position change for backtrack penalty."""
        self._prev_pos = pos_before
        self._prev_pos_action = action

    def reset_episode(self):
        """Reset episode-local state (counts persist across episodes)."""
        self._prev_pos = None
        self._prev_pos_action = None


# ── Episode Runner ───────────────────────────────────────

def run_jepa_episode(env, explorer, jepa_ensemble, max_steps, mode):
    """Run a single episode with JEPA-driven exploration.

    After each step, collects transitions for JEPA training.
    After the episode (if jepa/hybrid mode), trains the ensemble.
    """
    obs = env.reset()
    action_history = []
    steps = []
    transitions = []

    for step_i in range(max_steps):
        state = env._get_state()

        # 1. Generate candidates
        candidates = env.get_dynamic_candidates()

        # 2. Select action
        action = explorer.select_action(state, candidates, action_history)

        # 3. Execute
        goal_room = env.task.get("goal_room")
        prev_dist = abs(state.x - goal_room[0]) + abs(state.y - goal_room[1])
        next_obs, next_state, done = env.step(action)
        new_dist = abs(next_state.x - goal_room[0]) + abs(next_state.y - goal_room[1])

        # 4. Record transition for JEPA training
        transitions.append((state, action, next_state))

        # 5. Check if state changed (progress toward goal)
        success = (new_dist < prev_dist) or next_state.goal_reached

        # 6. Observe outcome
        explorer.observe(state, action, success)

        # 6b. Track successful moves (position change) for backtrack penalty
        if (next_state.x, next_state.y) != (state.x, state.y):
            explorer.observe_move(action, (state.x, state.y), (next_state.x, next_state.y))

        # 7. Record step
        record = {
            "step": step_i,
            "x": next_state.x,
            "y": next_state.y,
            "action": action,
            "success": success,
            "goal_reached": next_state.goal_reached,
            "inventory": list(env.inventory),
        }
        steps.append(record)

        if next_state.goal_reached:
            break

        action_history.append(action)

    # Post-episode JEPA training
    train_loss = None
    if mode in ("jepa_only", "hybrid") and jepa_ensemble is not None and transitions:
        train_loss = jepa_ensemble.train_step(transitions)

    return steps, env._get_state(), train_loss


# ── Main ─────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Phase 6 Maze JEPA Experiment"
    )
    parser.add_argument("--width", type=int, default=10)
    parser.add_argument("--height", type=int, default=10)
    parser.add_argument("--num-episodes", type=int, default=6)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--mode",
        choices=["pure_novelty", "jepa_only", "hybrid", "all"],
        default="all",
        help="Exploration mode (default: all 3)",
    )
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--model-path",
        default=None,
        help="Path to Qwen model (default: ~/models/Qwen2.5-0.5B-Instruct)",
    )
    parser.add_argument(
        "--goal",
        default="far",
        help="Goal placement: 'far' = bottom-right, 'random' = random",
    )
    args = parser.parse_args()

    # Derive model path
    if args.model_path is None:
        args.model_path = os.path.expanduser("~/models/Qwen2.5-0.5B-Instruct")

    # Determine modes to run
    modes = (
        ["pure_novelty", "jepa_only", "hybrid"]
        if args.mode == "all"
        else [args.mode]
    )

    # Derive output path
    # Auto-scale max_steps based on maze size
    if args.max_steps is None:
        size = args.width * args.height
        args.max_steps = min(size * 4, 500)

    if args.output is None:
        output_dir = _PROJECT_ROOT / "results"
        output_dir.mkdir(parents=True, exist_ok=True)
        mode_tag = args.mode if args.mode != "all" else "all_modes"
        args.output = str(
            output_dir
            / f"phase6_maze_jepa_{mode_tag}_{args.width}x{args.height}"
            f"_seed{args.seed}.jsonl"
        )
    output_path = Path(args.output)

    print(f"{'=' * 60}")
    print(f"Phase 6 Maze JEPA Experiment")
    print(f"  Maze: {args.width}x{args.height}")
    print(f"  Episodes per mode: {args.num_episodes}")
    print(f"  Max steps: {args.max_steps}")
    print(f"  Seed: {args.seed}")
    print(f"  Modes: {modes}")
    print(f"  Model: {args.model_path}")
    print(f"  Output: {output_path}")
    print(f"{'=' * 60}", flush=True)

    # Generate maze
    print(
        f"\nGenerating {args.width}x{args.height} maze (seed={args.seed})...",
        flush=True,
    )
    maze = GridMaze.generate(args.width, args.height, seed=args.seed)

    # Determine goal room
    if args.goal == "far":
        goal_room = (args.width - 1, args.height - 1)
    else:
        rng = random.Random(args.seed + 999)
        goal_room = (
            rng.randint(1, args.width - 1),
            rng.randint(1, args.height - 1),
        )

    task = {
        "goal_room": goal_room,
        "start_x": 0,
        "start_y": 0,
        "max_steps": args.max_steps,
    }

    state_estimate = maze.state_estimate()
    print(f"  State estimate: ~{state_estimate}")
    print(f"  Goal room: {goal_room}", flush=True)

    # Load JEPA ensemble (shared across modes that need it)
    jepa = None
    if any(m in ("jepa_only", "hybrid") for m in modes):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"\nLoading JEPA ensemble on {device}...", flush=True)
        t0 = time.time()
        jepa = JEPAEnsemble(args.model_path, n_ensemble=3, device=device)
        print(
            f"  Loaded in {time.time() - t0:.1f}s "
            f"(hidden_size={jepa.hidden_size})",
            flush=True,
        )

    for mode in modes:
        print(f"\n{'─' * 60}")
        print(f"Mode: {mode.upper()}")
        print(f"{'─' * 60}", flush=True)

        env = GridMazeEnv(maze, task)
        env.setup()

        explorer = MazeJEPAExplorer(jepa_ensemble=jepa, mode=mode)
        all_results = []

        for ep_idx in range(args.num_episodes):
            seed = args.seed + ep_idx
            random.seed(seed)

            explorer.reset_episode()
            if jepa is not None:
                jepa._cache_clear()

            print(
                f"  [Episode {ep_idx + 1}/{args.num_episodes}]", flush=True
            )
            t0 = time.time()

            try:
                steps, final_state, train_loss = run_jepa_episode(
                    env, explorer, jepa, args.max_steps, mode=mode,
                )
            except Exception as e:
                print(f"    ERROR: {e}", flush=True)
                import traceback

                traceback.print_exc()
                steps = []

            elapsed = time.time() - t0

            if steps:
                metrics = compute_metrics(steps, goal_room)
            else:
                metrics = {"fht": -1, "scr": 0.0, "dead_loop_rate": 0.0}

            result = {
                "width": args.width,
                "height": args.height,
                "mode": mode,
                "seed": seed,
                "episode": ep_idx,
                "steps_count": len(steps),
                "success": final_state.goal_reached,
                "fht": metrics.get("fht", -1),
                "scr": metrics.get("scr", 0.0),
                "dead_loop_rate": metrics.get("dead_loop_rate", 0.0),
                "train_loss": round(train_loss, 6) if train_loss is not None else None,
                "goal_x": goal_room[0],
                "goal_y": goal_room[1],
                "elapsed": round(elapsed, 1),
            }
            print(
                f"    -> success={result['success']} fht={result['fht']} "
                f"scr={result['scr']:.2f} steps={result['steps_count']} "
                f"loss={result['train_loss']} [{elapsed:.0f}s]",
                flush=True,
            )

            all_results.append(result)

            # Incremental write
            line = {
                k: result[k]
                for k in [
                    "width", "height", "mode", "seed", "episode",
                    "steps_count", "success", "fht", "scr",
                    "dead_loop_rate", "train_loss", "elapsed",
                ]
            }
            with open(output_path, "a") as f:
                f.write(json.dumps(line) + "\n")

        # Mode summary
        successes = sum(1 for r in all_results if r["success"])
        hits = [r for r in all_results if r["success"]]
        avg_scr = sum(r["scr"] for r in all_results) / max(len(all_results), 1)
        losses = [
            r.get("train_loss")
            for r in all_results
            if r.get("train_loss") is not None
        ]

        print(
            f"  [{mode}] Summary: {successes}/{args.num_episodes} success "
            f"({100 * successes / max(args.num_episodes, 1):.0f}%) | "
            f"Avg SCR: {avg_scr:.3f} | "
            + (
                f"Train loss: {sum(losses) / max(len(losses), 1):.6f} "
                f"(avg over {len(losses)} eps)"
                if losses
                else ""
            ),
            flush=True,
        )

    # Final cross-mode comparison
    print(f"\n{'=' * 60}")
    print(f"Cross-Mode Comparison for {args.width}x{args.height}")
    print(f"{'=' * 60}")

    # Re-read results
    mode_results = {m: [] for m in modes}
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    if rec.get("mode") in mode_results:
                        mode_results[rec["mode"]].append(rec)
                except json.JSONDecodeError:
                    continue

    for mode in modes:
        recs = mode_results.get(mode, [])
        if not recs:
            continue
        sr = sum(1 for r in recs if r.get("success", False))
        fhts_vals = [
            r["fht"]
            for r in recs
            if r.get("fht") is not None and r["fht"] >= 0
        ]
        avg_fht = sum(fhts_vals) / max(len(fhts_vals), 1) if fhts_vals else -1
        avg_scr = sum(r.get("scr", 0) for r in recs) / max(len(recs), 1)
        avg_dlr = sum(r.get("dead_loop_rate", 0) for r in recs) / max(
            len(recs), 1
        )
        losses = [
            r.get("train_loss")
            for r in recs
            if r.get("train_loss") is not None
        ]
        avg_loss = (
            sum(losses) / max(len(losses), 1) if losses else -1
        )
        total_elapsed = sum(r.get("elapsed", 0) for r in recs)

        print(
            f"  {mode:>15s}: {sr}/{len(recs)} success | "
            f"FHT={avg_fht:.1f} | SCR={avg_scr:.3f} | "
            f"DLR={avg_dlr:.3f} | loss={avg_loss:.6f} | "
            f"{total_elapsed:.1f}s"
        )

    print(f"\nOutput: {output_path}")
    print(f"{'=' * 60}", flush=True)


# Lazy torch import (after device detection)
import torch

if __name__ == "__main__":
    main()
