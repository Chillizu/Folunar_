#!/usr/bin/env python3
# ruff: noqa: E402
"""Phase 6 Stochastic Maze — JEPA Epistemic Exploration Experiment.

JEPA hybrid explorer (novelty + epistemic) on StochasticMazeEnv.
Expectation: >0% success — epistemic uncertainty about item presence drives
revisitation of rooms to check for newly spawned items.

Uses _state_to_text (via state_to_text from jepa_wm) for action selection,
consistent with the training representation, so the ensemble detects
stochasticity in item presence.
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

from phase5.jepa_wm import JEPAEnsemble, state_to_text
from phase6.grid_env import GridState
from phase6.maze_generator import GridMaze
from phase6.stochastic_maze import StochasticMazeEnv


# ── Maze JEPA Explorer (maze-adapted) ────────────────────

_REVERSE_MOVE = {
    "go north": "go south",
    "go south": "go north",
    "go east": "go west",
    "go west": "go east",
}


def _reverse_action(action: str) -> str:
    return _REVERSE_MOVE.get(action, "")


class MazeJEPAExplorer:
    """Action selector for stochastic maze driven by JEPA epistemic uncertainty.

    Uses state_to_text (from jepa_wm) for consistency — this calls _state_to_text
    internally, which uses state.files (visible_items) and state.cwd (room_name),
    so the JEPA state embedding includes item information.
    """

    _ACTION_PRIORITY = {
        "go": 0, "look": 1, "inventory": 2, "use": 3, "take": 4,
    }

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

        self.state_counts = {}
        self.state_action_counts = {}
        self.success_cache = {}
        self._prev_pos = None
        self._prev_pos_action = None

    def _novelty_bonus(self, state, action: str) -> float:
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

        # Use state_to_text (jepa_wm format) for epistemic uncertainty
        # This includes items via state.files → visible_items
        state_text = state_to_text(state)

        def score(action):
            nov = self._novelty_bonus(state, action)
            if self.mode == "pure_novelty":
                return nov
            epi = (
                self.jepa.epistemic_uncertainty(state_text, action)
                if self.jepa is not None
                else 0.0
            )
            if self.mode == "jepa_only":
                return epi
            return self.novelty_weight * nov + self.epistemic_weight * epi

        return max(candidates, key=lambda a: (score(a), -self._action_priority(a)))

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

def run_episode(env, explorer, jepa_ensemble, max_steps, mode):
    """Run a single JEPA-driven episode in StochasticMazeEnv.

    Success = picking up emerald. Collects transitions for JEPA training.
    """
    obs = env.reset()
    action_history = []
    steps = []
    transitions = []

    for step_i in range(max_steps):
        state = env._get_state()
        candidates = env.get_dynamic_candidates()
        action = explorer.select_action(state, candidates, action_history)

        next_obs, next_state, done = env.step(action)

        # Record transition
        transitions.append((state, action, next_state))

        # Success = emerald in inventory
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
        }
        steps.append(record)

        if has_emerald:
            break

        action_history.append(action)

    # Post-episode JEPA training
    train_loss = None
    if mode in ("jepa_only", "hybrid") and jepa_ensemble is not None and transitions:
        train_loss = jepa_ensemble.train_step(transitions)

    return steps, env._get_state(), train_loss


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
        description="Phase 6 Stochastic Maze — JEPA Experiment"
    )
    parser.add_argument("--width", type=int, default=10)
    parser.add_argument("--height", type=int, default=10)
    parser.add_argument("--num-episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=["pure_novelty", "jepa_only", "hybrid"],
                        default="hybrid")
    parser.add_argument("--output", default=None)
    parser.add_argument("--model-path", default=None,
                        help="Path to Qwen model (default: ~/models/Qwen2.5-0.5B-Instruct)")
    args = parser.parse_args()

    if args.model_path is None:
        args.model_path = os.path.expanduser("~/models/Qwen2.5-0.5B-Instruct")

    if args.output is None:
        output_dir = _PROJECT_ROOT / "results"
        output_dir.mkdir(parents=True, exist_ok=True)
        args.output = str(
            output_dir
            / f"phase6_stochastic_jepa_{args.mode}_{args.width}x{args.height}"
            f"_seed{args.seed}.jsonl"
        )
    output_path = Path(args.output)

    print(f"{'=' * 60}")
    print(f"Phase 6 Stochastic Maze — JEPA Experiment")
    print(f"  Maze: {args.width}x{args.height}")
    print(f"  Episodes: {args.num_episodes}")
    print(f"  Max steps: {args.max_steps}")
    print(f"  Seed: {args.seed}")
    print(f"  Mode: {args.mode}")
    print(f"  Model: {args.model_path}")
    print(f"  Output: {output_path}")
    print(f"{'=' * 60}", flush=True)

    # Generate maze
    print(f"\nGenerating {args.width}x{args.height} maze (seed={args.seed})...", flush=True)
    maze = GridMaze.generate(args.width, args.height, seed=args.seed)
    print(f"  Rooms: {len(maze.rooms)}", flush=True)

    # Task: find emerald
    task = {
        "start_x": 0,
        "start_y": 0,
        "max_steps": args.max_steps,
    }

    # Load JEPA ensemble
    device = "cuda" if (
        os.path.exists("/usr/local/cuda") or os.path.exists("/usr/bin/nvidia-smi")
    ) else "cpu"
    # Better: detect via torch
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"

    print(f"\nLoading JEPA ensemble on {device}...", flush=True)
    t0 = time.time()
    jepa = JEPAEnsemble(args.model_path, n_ensemble=3, device=device)
    print(f"  Loaded in {time.time() - t0:.1f}s (hidden_size={jepa.hidden_size})", flush=True)

    env = StochasticMazeEnv(maze, task)
    env.setup()

    explorer = MazeJEPAExplorer(jepa_ensemble=jepa, mode=args.mode)
    all_results = []

    for ep_idx in range(args.num_episodes):
        seed = args.seed + ep_idx
        random.seed(seed)

        explorer.reset_episode()
        jepa._cache_clear()

        print(f"\n  [Episode {ep_idx + 1}/{args.num_episodes}]", flush=True)
        t0 = time.time()

        try:
            steps, final_state, train_loss = run_episode(
                env, explorer, jepa, args.max_steps, mode=args.mode,
            )
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
            "mode": args.mode,
            "seed": seed,
            "episode": ep_idx,
            "steps_count": len(steps),
            "success": final_state.goal_reached,
            "fht": metrics.get("fht", -1),
            "scr": metrics.get("scr", 0.0),
            "dead_loop_rate": metrics.get("dead_loop_rate", 0.0),
            "train_loss": round(train_loss, 6) if train_loss is not None else None,
            "rooms_visited": len(set(f"{r['x']},{r['y']}" for r in steps)),
            "elapsed": round(elapsed, 1),
        }
        print(
            f"    -> success={result['success']} fht={result['fht']} "
            f"scr={result['scr']:.2f} steps={result['steps_count']} "
            f"loss={result['train_loss']} [{elapsed:.0f}s]",
            flush=True,
        )

        all_results.append(result)

        with open(output_path, "a") as f:
            f.write(json.dumps({k: result[k] for k in [
                "width", "height", "mode", "seed", "episode",
                "steps_count", "success", "fht", "scr",
                "dead_loop_rate", "train_loss", "rooms_visited", "elapsed",
            ]}) + "\n")

    # Summary
    successes = sum(1 for r in all_results if r["success"])
    avg_scr = sum(r["scr"] for r in all_results) / max(len(all_results), 1)
    losses = [r.get("train_loss") for r in all_results if r.get("train_loss") is not None]

    print(f"\n{'=' * 60}")
    print(f"Summary for {args.width}x{args.height}:")
    print(f"  Success: {successes}/{args.num_episodes} "
          f"({100 * successes / max(args.num_episodes, 1):.0f}%)")
    if losses:
        print(f"  Avg train loss: {sum(losses) / max(len(losses), 1):.6f}"
              f" (over {len(losses)} episodes)")
    print(f"  Avg SCR: {avg_scr:.3f}")
    print(f"  Output: {output_path}")
    print(f"{'=' * 60}", flush=True)


import torch  # lazy import after device detection

if __name__ == "__main__":
    main()
