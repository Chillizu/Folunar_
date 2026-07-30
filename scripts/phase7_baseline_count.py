#!/usr/bin/env python3
# ruff: noqa: E402
"""Phase 7 Random Maze — Count-Based Exploration Experiment.

Count-based novelty explorer on RandomMazeEnv (stochastic descriptions).
Expectation: novelty expires after one visit per room (description excluded
from state hash), so the agent never returns to re-explore rooms.

Hypothesis: Count-based fails in stochastic text environments because
the hash excludes the stochastic component — rooms look "explored" after
one visit, but the text keeps changing.
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
from phase7.random_maze import RandomMazeEnv, RandomGridState


# ── Maze Novelty Explorer (adapted from phase6_stochastic_count.py) ──────

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

        # Backtrack penalty
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
    """Run a single count-based exploration episode in RandomMazeEnv.

    Metrics track room coverage, not goal-based success.
    The key signal is SCR (state coverage ratio) and revisitation patterns.
    """
    obs = env.reset()
    action_history = []
    steps = []

    for step_i in range(max_steps):
        state = env._get_state()
        candidates = env.get_dynamic_candidates()
        action = explorer.select_action(state, candidates, action_history)

        next_obs, next_state, done = env.step(action)

        success = False  # No task goal for pure exploration
        explorer.observe(state, action, success)

        if (next_state.x, next_state.y) != (state.x, state.y):
            explorer.observe_move(action, (state.x, state.y), (next_state.x, next_state.y))

        record = {
            "step": step_i,
            "x": next_state.x,
            "y": next_state.y,
            "action": action,
            "success": success,
            "goal_reached": next_state.goal_reached,
            "inventory": list(env.inventory),
            "visible_items": env.maze.room_items.get((next_state.x, next_state.y), []),
            "room_name": next_state.room_name,
            "room_description": next_state.room_description[:60],
        }
        steps.append(record)

        action_history.append(action)

    return steps, env._get_state()


# ── Metrics ──────────────────────────────────────────────

def compute_metrics(steps):
    """Compute FHT, SCR, Unique rooms visited, Revisit rate.

    FHT: steps to first time each unique room is visited.
    SCR: unique (x,y) pairs / total steps.
    Revisit rate: fraction of steps that return to a previously visited room.
    """
    fht = None
    first_visits = {}
    for rec in steps:
        key = (rec["x"], rec["y"])
        if key not in first_visits:
            first_visits[key] = rec["step"]

    visited = set()
    revisits = 0
    for rec in steps:
        key = (rec["x"], rec["y"])
        if key in visited:
            revisits += 1
        visited.add(key)
    scr = len(visited) / max(len(steps), 1)
    revisit_rate = revisits / max(len(steps), 1)

    # Track description diversity per room (key metric for this experiment)
    desc_seen = {}
    for rec in steps:
        key = (rec["x"], rec["y"])
        d = rec.get("room_description", "")
        if key not in desc_seen:
            desc_seen[key] = set()
        desc_seen[key].add(d)
    max_desc_per_room = max(len(v) for v in desc_seen.values()) if desc_seen else 0
    avg_desc_per_room = sum(len(v) for v in desc_seen.values()) / max(len(desc_seen), 1)

    loops = 0
    for i in range(2, len(steps)):
        if steps[i]["action"] == steps[i - 1]["action"] == steps[i - 2]["action"]:
            loops += 1
    dead_loop_rate = loops / max(len(steps), 1)

    return {
        "fht": fht,
        "scr": round(scr, 3),
        "revisit_rate": round(revisit_rate, 3),
        "unique_rooms": len(visited),
        "max_desc_per_room": max_desc_per_room,
        "avg_desc_per_room": round(avg_desc_per_room, 2),
        "dead_loop_rate": round(dead_loop_rate, 3),
        "steps": len(steps),
    }


# ── Main ─────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Phase 7 Random Maze — Count-Based Experiment"
    )
    parser.add_argument("--maze-size", type=int, default=5,
                        help="Grid size (5=25 rooms, 10=100 rooms)")
    parser.add_argument("--num-episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--maze-seed", type=int, default=42)
    parser.add_argument("--text-seed", type=int, default=99)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if args.output is None:
        output_dir = _PROJECT_ROOT / "results"
        output_dir.mkdir(parents=True, exist_ok=True)
        args.output = str(
            output_dir
            / f"phase7_baseline_count_{args.maze_size}x{args.maze_size}"
            f"_mseed{args.maze_seed}_tseed{args.text_seed}.jsonl"
        )
    output_path = Path(args.output)

    print(f"{'=' * 60}")
    print(f"Phase 7 Random Maze — Count-Based Experiment")
    print(f"  Hypothesis: Count-based fails on stochastic text")
    print(f"  Maze: {args.maze_size}x{args.maze_size} (~{args.maze_size * args.maze_size} rooms)")
    print(f"  Episodes: {args.num_episodes}")
    print(f"  Max steps: {args.max_steps}")
    print(f"  Maze seed: {args.maze_seed}")
    print(f"  Text seed: {args.text_seed}")
    print(f"  Output: {output_path}")
    print(f"{'=' * 60}", flush=True)

    # Generate maze
    print(f"\nGenerating {args.maze_size}x{args.maze_size} maze (seed={args.maze_seed})...",
          flush=True)
    maze = GridMaze.generate(args.maze_size, args.maze_size, seed=args.maze_seed)
    print(f"  Rooms: {len(maze.rooms)}", flush=True)

    # No task goal — pure exploration metric
    task = {
        "start_x": 0,
        "start_y": 0,
        "max_steps": args.max_steps,
    }

    all_results = []
    for ep_idx in range(args.num_episodes):
        seed = args.maze_seed + ep_idx * 1000
        random.seed(seed)

        # Fresh env per episode (seeded random descriptions)
        env = RandomMazeEnv(
            maze, task, seed=args.text_seed + ep_idx,
        )
        env.setup()

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
            metrics = {"fht": -1, "scr": 0.0, "revisit_rate": 0.0,
                       "unique_rooms": 0, "max_desc_per_room": 0,
                       "avg_desc_per_room": 0.0, "dead_loop_rate": 0.0,
                       "steps": 0}

        result = {
            "maze_size": args.maze_size,
            "maze_seed": args.maze_seed,
            "text_seed": args.text_seed + ep_idx,
            "episode": ep_idx,
            "steps_count": len(steps),
            "scr": metrics.get("scr", 0.0),
            "revisit_rate": metrics.get("revisit_rate", 0.0),
            "unique_rooms": metrics.get("unique_rooms", 0),
            "max_desc_per_room": metrics.get("max_desc_per_room", 0),
            "avg_desc_per_room": metrics.get("avg_desc_per_room", 0.0),
            "dead_loop_rate": metrics.get("dead_loop_rate", 0.0),
            "total_rooms": args.maze_size * args.maze_size,
            "elapsed": round(elapsed, 1),
        }
        print(
            f"    -> scr={result['scr']:.2f} rooms={result['unique_rooms']}"
            f"/{result['total_rooms']} rev={result['revisit_rate']:.2f} "
            f"desc/room={result['avg_desc_per_room']:.1f} [{elapsed:.0f}s]",
            flush=True,
        )

        all_results.append(result)

        with open(output_path, "a") as f:
            f.write(json.dumps(result) + "\n")

    # Summary
    avg_scr = sum(r["scr"] for r in all_results) / max(len(all_results), 1)
    avg_rev = sum(r["revisit_rate"] for r in all_results) / max(len(all_results), 1)
    avg_desc = sum(r["avg_desc_per_room"] for r in all_results) / max(len(all_results), 1)

    print(f"\n{'=' * 60}")
    print(f"Summary:")
    print(f"  Avg SCR: {avg_scr:.3f}")
    print(f"  Avg revisit rate: {avg_rev:.3f}")
    print(f"  Avg desc diversity/room: {avg_desc:.1f}")
    print(f"  Expected: high SCR (many rooms visited), low revisit rate, "
          f"low desc diversity")
    print(f"  (count-based novelty expires, never re-explores for new text)")
    print(f"  Output: {output_path}")
    print(f"{'=' * 60}", flush=True)


if __name__ == "__main__":
    main()
