#!/usr/bin/env python3
# ruff: noqa: E402
"""Phase 6 Stochastic Maze — Count-Based Exploration Experiment.

Count-based novelty explorer on StochasticMazeEnv.
Expectation: 0% success — novelty expires after one visit per room,
so the agent never returns to check for newly spawned items.

Task: "find emerald" — spawns with p=0.02 in rooms where (x+y)%3==0.
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from phase6.grid_env import GridState
from phase6.maze_generator import GridMaze
from phase6.stochastic_maze import StochasticMazeEnv


# ── Maze Novelty Explorer (adapted from phase6_maze_count.py) ──────

_REVERSE_MOVE = {
    "go north": "go south",
    "go south": "go north",
    "go east": "go west",
    "go west": "go east",
}


def _reverse_action(action: str) -> str:
    return _REVERSE_MOVE.get(action, "")


class MazeNoveltyExplorer:
    """Count-based novelty explorer with backtrack penalty."""

    _ACTION_PRIORITY = {
        "go": 0, "look": 1, "inventory": 2, "use": 3, "take": 4,
    }

    def __init__(self):
        self.state_counts = {}
        self.state_action_counts = {}
        self.success_cache = {}
        self._prev_pos = None
        self._prev_pos_action = None

    def novelty_bonus(self, state, action: str) -> float:
        """Intrinsic reward with backtrack penalty."""
        from math import sqrt
        sh = state.state_hash()
        state_novelty = 1.0 / sqrt(1 + self.state_counts.get(sh, 0))
        pair_novelty = 1.0 / sqrt(1 + self.state_action_counts.get((sh, action), 0))
        bonus = 0.5 * state_novelty + 0.5 * pair_novelty

        if self._prev_pos is not None and action == _reverse_action(self._prev_pos_action):
            dx = {"go north": (0, -1), "go south": (0, 1), "go east": (1, 0), "go west": (-1, 0)}
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
        self.state_action_counts[pair_key] = self.state_action_counts.get(pair_key, 0) + 1
        if success:
            self.success_cache[sh] = action

    def observe_move(self, action: str, pos_before, pos_after):
        self._prev_pos = pos_before
        self._prev_pos_action = action

    def reset_episode(self):
        self._prev_pos = None
        self._prev_pos_action = None


# ── Episode Runner ───────────────────────────────────────

def run_episode(env, explorer, max_steps):
    """Run a single count-based exploration episode in StochasticMazeEnv.

    Success is defined as picking up the emerald.
    """
    obs = env.reset()
    action_history = []
    steps = []
    emerald_found = False

    for step_i in range(max_steps):
        state = env._get_state()
        candidates = env.get_dynamic_candidates()
        action = explorer.select_action(state, candidates, action_history)

        next_obs, next_state, done = env.step(action)

        # Success = emerald in inventory (stochastic task)
        has_emerald = "emerald" in env.inventory
        success = has_emerald

        explorer.observe(state, action, success)

        if (next_state.x, next_state.y) != (state.x, state.y):
            explorer.observe_move(action, (state.x, state.y), (next_state.x, next_state.y))

        record = {
            "step": step_i,
            "x": next_state.x,
            "y": next_state.y,
            "action": action,
            "success": success,
            "goal_reached": has_emerald,
            "inventory": list(env.inventory),
            "visible_items": env.maze.room_items.get((next_state.x, next_state.y), []),
        }
        steps.append(record)

        if has_emerald:
            emerald_found = True
            break

        action_history.append(action)

    return steps, env._get_state()


# ── Metrics ──────────────────────────────────────────────

def compute_metrics(steps):
    """Compute FHT, SCR from step records."""
    fht = None
    for rec in steps:
        if rec.get("goal_reached"):
            fht = rec["step"]
            break

    visited = set()
    for rec in steps:
        visited.add(f"{rec['x']},{rec['y']}")
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


# ── Main ─────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Phase 6 Stochastic Maze — Count-Based Experiment"
    )
    parser.add_argument("--width", type=int, default=10)
    parser.add_argument("--height", type=int, default=10)
    parser.add_argument("--num-episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if args.output is None:
        output_dir = _PROJECT_ROOT / "results"
        output_dir.mkdir(parents=True, exist_ok=True)
        args.output = str(
            output_dir
            / f"phase6_stochastic_count_{args.width}x{args.height}_seed{args.seed}.jsonl"
        )
    output_path = Path(args.output)

    print(f"{'=' * 60}")
    print(f"Phase 6 Stochastic Maze — Count-Based Experiment")
    print(f"  Maze: {args.width}x{args.height}")
    print(f"  Episodes: {args.num_episodes}")
    print(f"  Max steps: {args.max_steps}")
    print(f"  Seed: {args.seed}")
    print(f"  Output: {output_path}")
    print(f"{'=' * 60}", flush=True)

    # Generate maze
    print(f"\nGenerating {args.width}x{args.height} maze (seed={args.seed})...", flush=True)
    maze = GridMaze.generate(args.width, args.height, seed=args.seed)
    print(f"  Rooms: {len(maze.rooms)}", flush=True)

    # Task: find emerald — no goal_room
    task = {
        "start_x": 0,
        "start_y": 0,
        "max_steps": args.max_steps,
    }

    env = StochasticMazeEnv(maze, task)
    env.setup()

    all_results = []
    for ep_idx in range(args.num_episodes):
        seed = args.seed + ep_idx
        random.seed(seed)

        explorer = MazeNoveltyExplorer()
        explorer.reset_episode()

        print(f"\n  [Episode {ep_idx + 1}/{args.num_episodes}]", flush=True)
        t0 = time.time()

        try:
            steps, final_state = run_episode(env, explorer, args.max_steps)
        except Exception as e:
            print(f"    ERROR: {e}", flush=True)
            import traceback
            traceback.print_exc()
            steps = []

        elapsed = time.time() - t0

        if steps:
            metrics = compute_metrics(steps)
        else:
            metrics = {"fht": -1, "scr": 0.0, "dead_loop_rate": 0.0, "steps": 0}

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
            "rooms_visited": len(set(
                f"{r['x']},{r['y']}" for r in steps
            )),
            "elapsed": round(elapsed, 1),
        }
        print(
            f"    -> success={result['success']} fht={result['fht']} "
            f"scr={result['scr']:.2f} steps={result['steps_count']} "
            f"[{elapsed:.0f}s]",
            flush=True,
        )

        all_results.append(result)

        with open(output_path, "a") as f:
            f.write(json.dumps({k: result[k] for k in [
                "width", "height", "seed", "episode", "steps_count", "success",
                "fht", "scr", "dead_loop_rate", "rooms_visited", "elapsed",
            ]}) + "\n")

    # Summary
    successes = sum(1 for r in all_results if r["success"])
    print(f"\n{'=' * 60}")
    print(f"Summary for {args.width}x{args.height}:")
    print(f"  Success: {successes}/{args.num_episodes} "
          f"({100 * successes / max(args.num_episodes, 1):.0f}%)")
    print(f"  Avg rooms visited: {sum(r['rooms_visited'] for r in all_results) / max(len(all_results), 1):.1f}")
    print(f"  Output: {output_path}")
    print(f"{'=' * 60}", flush=True)


if __name__ == "__main__":
    main()
