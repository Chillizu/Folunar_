#!/usr/bin/env python3
# ruff: noqa: E402
"""Phase 7 Goal-Conditioned JEPA Experiment.

Compares three exploration strategies in a 10x10 grid maze:
  1. count        — count-based novelty (baseline)
  2. vanilla_jepa — vanilla JEPA ensemble epistemic uncertainty
  3. goal_jepa    — goal-conditioned JEPA with goal-progress-scaled uncertainty

Hypothesis: Goal-conditioned JEPA achieves >0% success where vanilla JEPA
achieves 0% at 10x10 scale, because goal-progress scaling gives epistemic
uncertainty genuine discriminatory power.

Usage:
  python scripts/phase7_goal_jepa.py [--width 10] [--height 10]
                                     [--num-episodes 3] [--max-steps 500]
                                     [--seed 42] [--output results/phase7_goal_jepa.jsonl]

  # GPU Manager invocation:
  python scripts/phase7_goal_jepa.py --model-path ~/models/Qwen2.5-0.5B-Instruct
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

from phase5.jepa_wm import JEPAEnsemble
from phase6.grid_env import grid_state_to_text as vanilla_state_to_text
from phase7.goal_jepa import GoalJEPAEnsemble, grid_state_to_text as goal_state_to_text
from phase6.grid_env import GridMazeEnv, GridState
from phase6.maze_generator import GridMaze


# ── Metrics ──────────────────────────────────────────────

def compute_metrics(steps, goal_room):
    """Compute FHT, SCR, Dead-loop Rate from step records."""
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


# ── Reverse action lookup (backtrack penalty) ────────────

_REVERSE_MOVE = {
    "go north": "go south",
    "go south": "go north",
    "go east": "go west",
    "go west": "go east",
    "look": "look",
    "inventory": "inventory",
}


def _reverse_action(action: str) -> str:
    return _REVERSE_MOVE.get(action, "")


def _action_priority(action: str) -> int:
    """Lower = prefer. Movement > look > inventory > use > take."""
    verb = action.split()[0] if action else ""
    priorities = {"go": 0, "look": 1, "inventory": 2, "use": 3, "take": 4}
    return priorities.get(verb, 5)


# ── Explorers ────────────────────────────────────────────

class CountExplorer:
    """Pure count-based novelty explorer (baseline)."""

    def __init__(self):
        self.state_counts = defaultdict(int)
        self.state_action_counts = defaultdict(int)
        self.success_cache = {}
        self._prev_pos = None
        self._prev_pos_action = None

    def novelty_bonus(self, state: GridState, action: str) -> float:
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

    def select_action(self, state: GridState, candidates, action_history):
        if not candidates:
            return "look"

        sh = state.state_hash()
        if sh in self.success_cache:
            cached = self.success_cache[sh]
            if cached in candidates:
                return cached

        return max(
            candidates,
            key=lambda a: (self.novelty_bonus(state, a), -_action_priority(a)),
        )

    def observe(self, state: GridState, action: str, success: bool):
        sh = state.state_hash()
        self.state_counts[sh] += 1
        self.state_action_counts[(sh, action)] += 1
        if success:
            self.success_cache[sh] = action

    def observe_move(self, action: str, pos_before, pos_after):
        self._prev_pos = pos_before
        self._prev_pos_action = action

    def reset_episode(self):
        self._prev_pos = None
        self._prev_pos_action = None


class VanillaJEPAExplorer:
    """Action selector driven by raw vanilla JEPA epistemic uncertainty."""

    def __init__(self, jepa: JEPAEnsemble):
        self.jepa = jepa
        self.state_counts = defaultdict(int)
        self.state_action_counts = defaultdict(int)
        self.success_cache = {}
        self._prev_pos = None
        self._prev_pos_action = None

    def _backtrack_penalty(self, state: GridState, action: str) -> float:
        if self._prev_pos is None:
            return 1.0
        if action != _reverse_action(self._prev_pos_action):
            return 1.0
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
                return 0.5
        return 1.0

    def select_action(self, state: GridState, candidates, action_history):
        if not candidates:
            return "look"

        state_text = vanilla_state_to_text(state)

        def score(action):
            epi = self.jepa.epistemic_uncertainty(state_text, action)
            penalty = self._backtrack_penalty(state, action)
            return epi * penalty

        return max(
            candidates,
            key=lambda a: (score(a), -_action_priority(a)),
        )

    def observe(self, state: GridState, action: str, success: bool):
        pass  # no count tables for vanilla JEPA

    def observe_move(self, action: str, pos_before, pos_after):
        self._prev_pos = pos_before
        self._prev_pos_action = action

    def reset_episode(self):
        self._prev_pos = None
        self._prev_pos_action = None


class GoalJEPAExplorer:
    """Action selector driven by goal-conditioned JEPA uncertainty."""

    def __init__(self, gjepa: GoalJEPAEnsemble, goal_text: str):
        self.gjepa = gjepa
        self.goal_text = goal_text
        self._prev_pos = None
        self._prev_pos_action = None

    def _backtrack_penalty(self, state: GridState, action: str) -> float:
        if self._prev_pos is None:
            return 1.0
        if action != _reverse_action(self._prev_pos_action):
            return 1.0
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
                return 0.5
        return 1.0

    def select_action(self, state: GridState, candidates, action_history):
        if not candidates:
            return "look"

        state_text = goal_state_to_text(state)

        def score(action):
            gpu = self.gjepa.goal_progress_uncertainty(
                state_text, action, self.goal_text
            )
            penalty = self._backtrack_penalty(state, action)
            return gpu * penalty

        return max(
            candidates,
            key=lambda a: (score(a), -_action_priority(a)),
        )

    def observe(self, state: GridState, action: str, success: bool):
        pass

    def observe_move(self, action: str, pos_before, pos_after):
        self._prev_pos = pos_before
        self._prev_pos_action = action

    def reset_episode(self):
        self._prev_pos = None
        self._prev_pos_action = None


# ── Episode Runners ──────────────────────────────────────

def run_count_episode(env, explorer, max_steps):
    """Run a single episode with count-based exploration."""
    obs = env.reset()
    steps = []

    for step_i in range(max_steps):
        state = env._get_state()
        candidates = env.get_dynamic_candidates()
        action = explorer.select_action(state, candidates, [])

        goal_room = env.task.get("goal_room")
        prev_dist = abs(state.x - goal_room[0]) + abs(state.y - goal_room[1])
        next_obs, next_state, done = env.step(action)
        new_dist = abs(next_state.x - goal_room[0]) + abs(next_state.y - goal_room[1])
        success = (new_dist < prev_dist) or next_state.goal_reached

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
        }
        steps.append(record)

        if next_state.goal_reached:
            break

    return steps, env._get_state()


def run_vanilla_jepa_episode(env, explorer, jepa, max_steps):
    """Run a single episode with vanilla JEPA epistemic exploration."""
    obs = env.reset()
    steps = []
    transitions = []

    for step_i in range(max_steps):
        state = env._get_state()
        candidates = env.get_dynamic_candidates()
        action = explorer.select_action(state, candidates, [])

        goal_room = env.task.get("goal_room")
        prev_dist = abs(state.x - goal_room[0]) + abs(state.y - goal_room[1])
        next_obs, next_state, done = env.step(action)

        transitions.append((state, action, next_state))
        explorer.observe(state, action, False)

        if (next_state.x, next_state.y) != (state.x, state.y):
            explorer.observe_move(action, (state.x, state.y), (next_state.x, next_state.y))

        record = {
            "step": step_i,
            "x": next_state.x,
            "y": next_state.y,
            "action": action,
            "success": False,
            "goal_reached": next_state.goal_reached,
            "inventory": list(env.inventory),
        }
        steps.append(record)

        if next_state.goal_reached:
            break

    # Post-episode training
    train_loss = None
    if jepa is not None and transitions:
        train_loss = jepa.train_step(transitions)

    return steps, env._get_state(), train_loss


def run_goal_jepa_episode(env, explorer, gjepa, max_steps, goal_text):
    """Run a single episode with goal-conditioned JEPA exploration."""
    obs = env.reset()
    steps = []
    transitions = []

    for step_i in range(max_steps):
        state = env._get_state()
        candidates = env.get_dynamic_candidates()
        action = explorer.select_action(state, candidates, [])

        next_obs, next_state, done = env.step(action)

        transitions.append((state, action, next_state))

        if (next_state.x, next_state.y) != (state.x, state.y):
            explorer.observe_move(action, (state.x, state.y), (next_state.x, next_state.y))

        record = {
            "step": step_i,
            "x": next_state.x,
            "y": next_state.y,
            "action": action,
            "goal_reached": next_state.goal_reached,
            "inventory": list(env.inventory),
        }
        steps.append(record)

        if next_state.goal_reached:
            break

    # Post-episode training with goal
    train_loss = None
    if gjepa is not None and transitions:
        train_loss = gjepa.train_step(transitions, goal_text)

    return steps, env._get_state(), train_loss


# ── Main ─────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Phase 7 Goal-Conditioned JEPA Experiment"
    )
    parser.add_argument("--width", type=int, default=10)
    parser.add_argument("--height", type=int, default=10)
    parser.add_argument("--num-episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSONL path (default: results/phase7_goal_jepa.jsonl)",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Path to Qwen model (default: ~/models/Qwen2.5-0.5B-Instruct)",
    )
    args = parser.parse_args()

    # Derive defaults
    if args.model_path is None:
        args.model_path = os.path.expanduser("~/models/Qwen2.5-0.5B-Instruct")

    if args.max_steps is None:
        args.max_steps = 500

    if args.output is None:
        output_dir = _PROJECT_ROOT / "results"
        output_dir.mkdir(parents=True, exist_ok=True)
        args.output = str(
            output_dir
            / f"phase7_goal_jepa_{args.width}x{args.height}_seed{args.seed}.jsonl"
        )
    output_path = Path(args.output)

    # Goal description
    goal_room = (args.width - 1, args.height - 1)  # far corner
    goal_text = f"reach the Treasury at position ({goal_room[0]},{goal_room[1]})"

    print(f"{'=' * 60}")
    print(f"Phase 7: Goal-Conditioned JEPA")
    print(f"  Maze: {args.width}x{args.height}")
    print(f"  Goal: {goal_text}")
    print(f"  Goal room: {goal_room}")
    print(f"  Episodes per mode: {args.num_episodes}")
    print(f"  Max steps: {args.max_steps}")
    print(f"  Seed: {args.seed}")
    print(f"  Model: {args.model_path}")
    print(f"  Output: {output_path}")
    print(f"{'=' * 60}", flush=True)

    # Generate maze
    print(f"\nGenerating {args.width}x{args.height} maze (seed={args.seed})...", flush=True)
    maze = GridMaze.generate(args.width, args.height, seed=args.seed)
    state_estimate = maze.state_estimate()
    print(f"  State estimate: ~{state_estimate}")
    print(f"  Goal room: {goal_room}", flush=True)

    task = {
        "goal_room": goal_room,
        "start_x": 0,
        "start_y": 0,
        "max_steps": args.max_steps,
    }

    # Load models
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}", flush=True)

    # Vanilla JEPA
    print("\nLoading vanilla JEPA ensemble...", flush=True)
    t0 = time.time()
    vanilla_jepa = JEPAEnsemble(args.model_path, n_ensemble=3, device=device)
    print(f"  Loaded in {time.time() - t0:.1f}s", flush=True)

    # Goal-conditioned JEPA
    print("Loading goal-conditioned JEPA ensemble...", flush=True)
    t0 = time.time()
    goal_jepa = GoalJEPAEnsemble(args.model_path, n_ensemble=3, device=device)
    print(f"  Loaded in {time.time() - t0:.1f}s", flush=True)

    # Define modes
    modes = [
        ("count", "count"),
        ("vanilla_jepa", "vanilla_jepa"),
        ("goal_jepa", "goal_jepa"),
    ]

    for mode_key, mode_label in modes:
        print(f"\n{'─' * 60}")
        print(f"Mode: {mode_label}")
        print(f"{'─' * 60}", flush=True)

        env = GridMazeEnv(maze, task)
        env.setup()

        if mode_key == "count":
            explorer = CountExplorer()
        elif mode_key == "vanilla_jepa":
            explorer = VanillaJEPAExplorer(vanilla_jepa)
        elif mode_key == "goal_jepa":
            explorer = GoalJEPAExplorer(goal_jepa, goal_text)

        all_results = []

        for ep_idx in range(args.num_episodes):
            seed = args.seed + ep_idx
            random.seed(seed)

            explorer.reset_episode()
            if mode_key == "vanilla_jepa":
                vanilla_jepa._cache_clear()
            elif mode_key == "goal_jepa":
                goal_jepa._cache_clear()

            print(f"  [Episode {ep_idx + 1}/{args.num_episodes}]", flush=True)
            t0 = time.time()

            try:
                if mode_key == "count":
                    steps, final_state = run_count_episode(env, explorer, args.max_steps)
                    train_loss = None
                elif mode_key == "vanilla_jepa":
                    steps, final_state, train_loss = run_vanilla_jepa_episode(
                        env, explorer, vanilla_jepa, args.max_steps,
                    )
                elif mode_key == "goal_jepa":
                    steps, final_state, train_loss = run_goal_jepa_episode(
                        env, explorer, goal_jepa, args.max_steps, goal_text,
                    )
            except Exception as e:
                print(f"    ERROR: {e}", flush=True)
                import traceback
                traceback.print_exc()
                steps = []
                final_state = None
                train_loss = None

            elapsed = time.time() - t0

            if steps:
                metrics = compute_metrics(steps, goal_room)
            else:
                metrics = {"fht": -1, "scr": 0.0, "dead_loop_rate": 0.0}

            result = {
                "width": args.width,
                "height": args.height,
                "mode": mode_key,
                "seed": seed,
                "episode": ep_idx,
                "steps_count": len(steps),
                "success": final_state.goal_reached if final_state else False,
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
            f"  [{mode_key}] Summary: {successes}/{args.num_episodes} success "
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

    mode_keys = ["count", "vanilla_jepa", "goal_jepa"]
    mode_results = {m: [] for m in mode_keys}
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    if rec.get("mode") in mode_results:
                        mode_results[rec["mode"]].append(rec)
                except json.JSONDecodeError:
                    continue

    for mode_key in mode_keys:
        recs = mode_results.get(mode_key, [])
        if not recs:
            continue
        sr = sum(1 for r in recs if r.get("success", False))
        fht_vals = [
            r["fht"]
            for r in recs
            if r.get("fht") is not None and r["fht"] >= 0
        ]
        avg_fht = sum(fht_vals) / max(len(fht_vals), 1) if fht_vals else -1
        avg_scr = sum(r.get("scr", 0) for r in recs) / max(len(recs), 1)
        avg_dlr = sum(r.get("dead_loop_rate", 0) for r in recs) / max(len(recs), 1)
        losses = [
            r.get("train_loss")
            for r in recs
            if r.get("train_loss") is not None
        ]
        avg_loss = sum(losses) / max(len(losses), 1) if losses else -1
        total_elapsed = sum(r.get("elapsed", 0) for r in recs)

        print(
            f"  {mode_key:>15s}: {sr}/{len(recs)} success | "
            f"FHT={avg_fht:.1f} | SCR={avg_scr:.3f} | "
            f"DLR={avg_dlr:.3f} | loss={avg_loss:.6f} | "
            f"{total_elapsed:.1f}s"
        )

    print(f"\nOutput: {output_path}")
    print(f"{'=' * 60}", flush=True)


import torch

if __name__ == "__main__":
    main()
