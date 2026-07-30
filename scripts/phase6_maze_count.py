#!/usr/bin/env python3
# ruff: noqa: E402
"""Phase 6 Maze Count-Based Experiment: novelty-driven exploration in grid mazes.

Uses the count-based NoveltyExplorer (from phase5) to drive exploration
in procedurally generated grid mazes. No neural networks, no GPU.

Measures success rate at 4 maze sizes to establish the count-based baseline.

Usage:
  # Single size
  python scripts/phase6_maze_count.py --width 10 --height 10 --num-episodes 12

  # All sizes
  for s in 5 10 20 30; do
    python scripts/phase6_maze_count.py --width $s --height $s --num-episodes 12 --seed 42
  done
"""

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from phase6.grid_env import GridMazeEnv, GridState
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


# ── Maze Novelty Explorer (with backtrack penalty) ──────

_REVERSE_MOVE = {
    "go north": "go south",
    "go south": "go north",
    "go east": "go west",
    "go west": "go east",
}


def _reverse_action(action: str) -> str:
    """Return the direction that would undo the given action."""
    return _REVERSE_MOVE.get(action, "")


class MazeNoveltyExplorer:
    """Count-based novelty explorer with backtrack penalty for maze environments.

    Penalizes actions that reverse the last *successful* move (position change),
    so the agent prefers exploring new rooms over revisiting the last one.
    """

    _ACTION_PRIORITY = {
        "go": 0,
        "look": 1,
        "inventory": 2,
        "use": 3,
        "take": 4,
    }

    def __init__(self):
        self.state_counts = {}
        self.state_action_counts = {}
        self.success_cache = {}
        self._prev_pos = None  # (x,y) we just came from
        self._prev_pos_action = None  # action that led here

    def novelty_bonus(self, state, action: str) -> float:
        """Intrinsic novelty reward for (state, action) pair, with backtrack penalty."""
        from math import sqrt
        sh = state.state_hash()
        state_novelty = 1.0 / sqrt(1 + self.state_counts.get(sh, 0))
        pair_novelty = 1.0 / sqrt(1 + self.state_action_counts.get((sh, action), 0))
        bonus = 0.5 * state_novelty + 0.5 * pair_novelty

        # Backtrack penalty: 50% reduction if this action returns to the
        # position we most recently arrived from.
        if self._prev_pos is not None and action == _reverse_action(self._prev_pos_action):
            # Verify the destination matches prev_pos
            dx = {'go north': (0,-1), 'go south': (0,1), 'go east': (1,0), 'go west': (-1,0)}
            move = dx.get(action)
            if move:
                dest = (state.x + move[0], state.y + move[1])
                if dest == self._prev_pos:
                    bonus *= 0.5

        return bonus

    def _action_priority(self, action: str) -> int:
        verb = action.split()[0] if action else ""
        return self._ACTION_PRIORITY.get(verb, 5)

    def select_action(self, state, candidates, action_history):
        """Select the most novel action, penalizing backtracks."""
        if not candidates:
            return "look"

        sh = state.state_hash()

        # Cached success replay
        if sh in self.success_cache:
            cached = self.success_cache[sh]
            if cached in candidates:
                return cached

        # Highest novelty bonus, tie-break by action priority
        return max(candidates, key=lambda a: (
            self.novelty_bonus(state, a),
            -self._action_priority(a),
        ))

    def observe(self, state, action: str, success: bool):
        """Record execution outcome and update counts."""
        sh = state.state_hash()
        self.state_counts[sh] = self.state_counts.get(sh, 0) + 1
        pair_key = (sh, action)
        self.state_action_counts[pair_key] = self.state_action_counts.get(pair_key, 0) + 1
        if success:
            self.success_cache[sh] = action

    def observe_move(self, action: str, pos_before, pos_after):
        """Track a position change for backtrack penalty."""
        # pos_before is where we came from; pos_after is where we are now
        self._prev_pos = pos_before
        self._prev_pos_action = action

    def reset_episode(self):
        """Reset episode-local state."""
        self._prev_pos = None
        self._prev_pos_action = None


# ── Episode Runner ───────────────────────────────────────

def run_count_episode(env, explorer, max_steps, last_pos=None):
    """Run a single count-based novelty exploration episode.

    Returns (steps, state).
    """
    obs = env.reset()
    action_history = []
    steps = []
    last_pos = last_pos or (0, 0)

    for step_i in range(max_steps):
        state = env._get_state()
        state_hash_before = state.state_hash()

        # 1. Generate candidates
        candidates = env.get_dynamic_candidates()

        # 2. Select action via novelty explorer
        action = explorer.select_action(state, candidates, action_history)

        # 3. Execute
        goal_room = env.task.get("goal_room")
        prev_dist = abs(state.x - goal_room[0]) + abs(state.y - goal_room[1])
        next_obs, next_state, done = env.step(action)
        new_dist = abs(next_state.x - goal_room[0]) + abs(next_state.y - goal_room[1])

        # 4. Check if state changed (success = progress toward goal or reached it)
        success = (new_dist < prev_dist) or next_state.goal_reached

        # 5. Observe outcome
        explorer.observe(state, action, success)

        # 6. Track successful moves (position change)
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
        last_pos = (state.x, state.y)

    return steps, env._get_state()


# ── Main ─────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Phase 6 Maze Count-Based Experiment"
    )
    parser.add_argument("--width", type=int, default=10)
    parser.add_argument("--height", type=int, default=10)
    parser.add_argument("--num-episodes", type=int, default=12)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None)
    parser.add_argument("--goal", default="far",
                        help="Goal placement: 'far' = bottom-right, 'random' = random")
    args = parser.parse_args()

    # Auto-scale max_steps based on maze size
    if args.max_steps is None:
        size = args.width * args.height
        args.max_steps = min(size * 4, 500)  # 100 for 5x5, 400 for 10x10, capped at 500

    # Derive output path
    if args.output is None:
        output_dir = _PROJECT_ROOT / "results"
        output_dir.mkdir(parents=True, exist_ok=True)
        args.output = str(
            output_dir
            / f"phase6_maze_count_{args.width}x{args.height}_seed{args.seed}.jsonl"
        )
    output_path = Path(args.output)

    print(f"{'=' * 60}")
    print(f"Phase 6 Maze Count-Based Experiment")
    print(f"  Maze: {args.width}x{args.height}")
    print(f"  Episodes: {args.num_episodes}")
    print(f"  Max steps: {args.max_steps}")
    print(f"  Seed: {args.seed}")
    print(f"  Output: {output_path}")
    print(f"{'=' * 60}", flush=True)

    # Generate maze
    print(f"\nGenerating {args.width}x{args.height} maze (seed={args.seed})...", flush=True)
    maze = GridMaze.generate(args.width, args.height, seed=args.seed)

    # Determine goal room
    if args.goal == "far":
        goal_room = (args.width - 1, args.height - 1)
    else:
        rng = random.Random(args.seed + 999)
        goal_room = (rng.randint(1, args.width - 1), rng.randint(1, args.height - 1))

    task = {
        "goal_room": goal_room,
        "start_x": 0,
        "start_y": 0,
        "max_steps": args.max_steps,
    }

    state_estimate = maze.state_estimate()
    print(f"  State estimate: ~{state_estimate}")
    print(f"  Goal room: {goal_room}", flush=True)

    # Create environment
    env = GridMazeEnv(maze, task)
    env.setup()

    # Run episodes
    all_results = []

    for ep_idx in range(args.num_episodes):
        seed = args.seed + ep_idx
        random.seed(seed)

        explorer = MazeNoveltyExplorer()
        explorer.reset_episode()

        print(f"\n  [Episode {ep_idx + 1}/{args.num_episodes}]", flush=True)
        t0 = time.time()

        try:
            steps, final_state = run_count_episode(env, explorer, args.max_steps)
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
            "seed": seed,
            "episode": ep_idx,
            "steps_count": len(steps),
            "success": final_state.goal_reached,
            "fht": metrics.get("fht", -1),
            "scr": metrics.get("scr", 0.0),
            "dead_loop_rate": metrics.get("dead_loop_rate", 0.0),
            "goal_x": goal_room[0],
            "goal_y": goal_room[1],
            "elapsed": round(elapsed, 1),
            "records": steps,
        }
        print(
            f"    -> success={result['success']} fht={result['fht']} "
            f"scr={result['scr']:.2f} steps={result['steps_count']} "
            f"[{elapsed:.0f}s]",
            flush=True,
        )

        all_results.append(result)

        # Incremental write (summary line only)
        line = {
            k: result[k]
            for k in [
                "width", "height", "seed", "episode", "steps_count", "success",
                "fht", "scr", "dead_loop_rate", "elapsed",
            ]
        }
        with open(output_path, "a") as f:
            f.write(json.dumps(line) + "\n")

    # Summary
    successes = sum(1 for r in all_results if r["success"])
    avg_scr = sum(r["scr"] for r in all_results) / max(len(all_results), 1)
    fhts = [
        r["fht"]
        for r in all_results
        if r.get("fht") is not None and r["fht"] >= 0
    ]
    avg_fht = sum(fhts) / max(len(fhts), 1) if fhts else -1

    print(f"\n{'=' * 60}")
    print(f"Summary for {args.width}x{args.height}:")
    print(f"  Success: {successes}/{args.num_episodes} "
          f"({100 * successes / max(args.num_episodes, 1):.0f}%)")
    print(f"  Avg FHT: {avg_fht:.1f}")
    print(f"  Avg SCR: {avg_scr:.3f}")
    print(f"  Output: {output_path}")
    print(f"{'=' * 60}", flush=True)

    # Write full results
    with open(output_path, "w") as f:
        for r in all_results:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
