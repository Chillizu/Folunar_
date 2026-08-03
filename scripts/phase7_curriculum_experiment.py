#!/usr/bin/env python3
# ruff: noqa: E402
"""Phase 7 Curriculum Intrinsic Motivation Experiment.

Compares 5 exploration strategies on procedurally generated grid mazes:

  1. pure_count:    Count-based novelty only (baseline).
  2. pure_jepa:     JEPA ensemble variance only (baseline).
  3. hybrid:        0.5*novelty + 0.5*epistemic (baseline, expected to fail).
  4. curriculum:    Warm-start with count, phase in JEPA via CurriculumExplorer.
  5. ucb:           UCB-style: 0.7*nov + 0.3*ep * sqrt(log(total)/(1+count)).

Measured at 10x10 and 20x20 maze sizes.
Hypothesis: curriculum > hybrid, possibly > count at 20x20.
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
from phase6.grid_env import GridMazeEnv, GridState, grid_state_to_text
from phase6.maze_generator import GridMaze
from phase7.curriculum_explorer import CurriculumExplorer


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
        visited.add(
            f"{rec['x']},{rec['y']}|{tuple(rec.get('inventory', []))}"
        )
    scr = len(visited) / max(len(steps), 1)

    loops = 0
    for i in range(2, len(steps)):
        if (
            steps[i]["action"] == steps[i - 1]["action"]
            == steps[i - 2]["action"]
        ):
            loops += 1
    dead_loop_rate = loops / max(len(steps), 1)

    return {
        "fht": fht,
        "scr": round(scr, 3),
        "dead_loop_rate": round(dead_loop_rate, 3),
        "steps": len(steps),
    }


# ── Direction helpers ────────────────────────────────────

_REVERSE_MOVE = {
    "go north": "go south",
    "go south": "go north",
    "go east": "go west",
    "go west": "go east",
}

_DIRECTION_DELTA = {
    "go north": (0, -1),
    "go south": (0, 1),
    "go east": (1, 0),
    "go west": (-1, 0),
}


def _reverse_action(action: str) -> str:
    return _REVERSE_MOVE.get(action, "")


# ── Universal Maze Explorer (5 modes) ───────────────────

_MODE_DESCRIPTIONS = {
    "pure_count": "Count-based novelty only",
    "pure_jepa": "JEPA epistemic uncertainty only",
    "hybrid": "0.5*nov + 0.5*epi (known to dilute)",
    "curriculum": "Warm-start count, phase-in JEPA",
    "ucb": "0.7*nov + 0.3*ep * sqrt(log(total)/(1+count))",
}


class MazeCurriculumExplorer:
    """Unified action selector for grid maze supporting all 5 exploration modes.

    Core idea: combine count-based novelty (explore unseen rooms) with JEPA
    epistemic uncertainty (revisit to refine understanding) using different
    mixing strategies.

    Modes:
      pure_count:    Only count-based novelty bonus.
      pure_jepa:     Only JEPA ensemble variance.
      hybrid:        0.5 * novelty + 0.5 * epistemic.
      curriculum:    Warm-start count (2 ep), then phase in epistemic via
                     CurriculumExplorer.score().
      ucb:           UCB-style: 0.7*nov + 0.3*ep * sqrt(log(N)/(1+n_a)).
    """

    def __init__(
        self,
        jepa_ensemble: JEPAEnsemble = None,
        mode: str = "curriculum",
        warmup_episodes: int = 2,
    ):
        self.jepa = jepa_ensemble
        self.mode = mode
        self.warmup = warmup_episodes

        # Count-based tables (for all modes that use novelty)
        self.state_counts = defaultdict(int)
        self.state_action_counts = defaultdict(int)
        self.success_cache = {}

        # Curriculum phase-in
        self.curriculum = None
        if mode == "curriculum":
            self.curriculum = CurriculumExplorer(
                novelty_explorer=self,
                jepa_wm=jepa_ensemble,
                warmup_episodes=warmup_episodes,
            )

        # Backtrack penalty tracking
        self._prev_pos = None
        self._prev_pos_action = None

        # UCB tracking
        self._total_steps = 0

    # ── Novelty bonus (shared) ──────────────────────────

    def novelty_bonus(self, state: GridState, action: str) -> float:
        """Count-based intrinsic reward with backtrack penalty."""
        sh = state.state_hash()
        state_novelty = 1.0 / max(1.0, self.state_counts[sh] ** 0.5)
        pair_novelty = 1.0 / max(1.0, self.state_action_counts[(sh, action)] ** 0.5)
        bonus = 0.5 * state_novelty + 0.5 * pair_novelty

        # Backtrack penalty: half bonus for reversing into previous position
        if self._prev_pos is not None and action == _reverse_action(
            self._prev_pos_action
        ):
            move = _DIRECTION_DELTA.get(action)
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

    # ── Action selection ────────────────────────────────

    def select_action(
        self,
        state: GridState,
        candidates,
        action_history,
    ):
        """Select the most informative action under the configured mode."""
        if not candidates:
            return "look"

        sh = state.state_hash()

        # Cached success replay
        if sh in self.success_cache:
            cached = self.success_cache[sh]
            if cached in candidates:
                return cached

        state_text = grid_state_to_text(state)

        def score(action):
            nov = self.novelty_bonus(state, action)
            ep = (
                self.jepa.epistemic_uncertainty(state_text, action)
                if self.jepa
                else 0.0
            )

            if self.mode == "pure_count":
                return nov
            elif self.mode == "pure_jepa":
                return ep
            elif self.mode == "hybrid":
                return 0.5 * nov + 0.5 * ep
            elif self.mode == "curriculum":
                return self.curriculum.score(state, action, state_text)
            elif self.mode == "ucb":
                n_a = self.state_action_counts.get((sh, action), 0)
                ucb_bonus = math.sqrt(
                    math.log(max(1, self._total_steps)) / max(1, 1 + n_a)
                )
                return 0.7 * nov + 0.3 * ep * ucb_bonus
            else:
                return nov

        # Pick highest-scoring, tie-break by action priority
        return max(candidates, key=lambda a: (score(a), -self._action_priority(a)))

    # ── Observing outcomes ──────────────────────────────

    def observe(self, state: GridState, action: str, success: bool):
        """Record execution outcome and update counts."""
        sh = state.state_hash()
        self.state_counts[sh] += 1
        self.state_action_counts[(sh, action)] += 1
        self._total_steps += 1
        if success:
            self.success_cache[sh] = action

    def observe_move(self, action: str, pos_before, pos_after):
        """Track a successful position change for backtrack penalty."""
        self._prev_pos = pos_before
        self._prev_pos_action = action

    def reset_episode(self):
        """Reset episode-local state. Counts persist across episodes."""
        self._prev_pos = None
        self._prev_pos_action = None

    def advance_curriculum(self):
        """Advance the curriculum phase-in counter."""
        if self.curriculum is not None:
            self.curriculum.advance_episode()


# ── Episode Runner ───────────────────────────────────────

def run_episode(env, explorer, jepa_ensemble, max_steps, mode):
    """Run a single episode with the given explorer mode.

    After each step, collects transitions for JEPA training.
    After the episode, trains the ensemble (for modes that use JEPA).
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
        new_dist = (
            abs(next_state.x - goal_room[0])
            + abs(next_state.y - goal_room[1])
        )

        # 4. Record transition for JEPA training
        transitions.append((state, action, next_state))

        # 5. Check if state changed (progress toward goal)
        success = (new_dist < prev_dist) or next_state.goal_reached

        # 6. Observe outcome
        explorer.observe(state, action, success)

        # 6b. Track successful moves
        if (next_state.x, next_state.y) != (state.x, state.y):
            explorer.observe_move(
                action, (state.x, state.y), (next_state.x, next_state.y)
            )

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
    if mode in (
        "pure_jepa",
        "hybrid",
        "curriculum",
        "ucb",
    ) and jepa_ensemble is not None and transitions:
        train_loss = jepa_ensemble.train_step(transitions)

    return steps, env._get_state(), train_loss


# ── Run a batch of modes on one maze ─────────────────────

MODES = ["pure_count", "pure_jepa", "hybrid", "curriculum", "ucb"]


def run_maze_comparison(
    maze, task, modes, num_episodes, max_steps, jepa, seed, output_path,
    warmup_episodes=2,
):
    """Run all modes on the same maze and write per-episode JSONL results."""
    goal_room = task.get("goal_room")

    for mode in modes:
        print(f"\n{'─' * 60}")
        print(f"Mode: {mode.upper()} — {_MODE_DESCRIPTIONS[mode]}")
        print(f"{'─' * 60}", flush=True)

        env = GridMazeEnv(maze, task)
        env.setup()

        explorer = MazeCurriculumExplorer(
            jepa_ensemble=jepa, mode=mode,
            warmup_episodes=warmup_episodes,
        )
        all_results = []

        for ep_idx in range(num_episodes):
            ep_seed = seed + ep_idx
            random.seed(ep_seed)

            explorer.reset_episode()
            if jepa is not None:
                jepa._cache_clear()

            print(f"  [Episode {ep_idx + 1}/{num_episodes}]", flush=True)
            t0 = time.time()

            try:
                steps, final_state, train_loss = run_episode(
                    env,
                    explorer,
                    jepa,
                    max_steps,
                    mode=mode,
                )
            except Exception as e:
                print(f"    ERROR: {e}", flush=True)
                import traceback

                traceback.print_exc()
                steps = []
                final_state = env._get_state()
                train_loss = None

            elapsed = time.time() - t0

            if steps:
                metrics = compute_metrics(steps, goal_room)
            else:
                metrics = {"fht": -1, "scr": 0.0, "dead_loop_rate": 0.0}

            result = {
                "width": maze.width,
                "height": maze.height,
                "mode": mode,
                "seed": ep_seed,
                "episode": ep_idx,
                "steps_count": len(steps),
                "success": final_state.goal_reached,
                "fht": metrics.get("fht", -1),
                "scr": metrics.get("scr", 0.0),
                "dead_loop_rate": metrics.get("dead_loop_rate", 0.0),
                "train_loss": (
                    round(train_loss, 6) if train_loss is not None else None
                ),
                "goal_x": goal_room[0],
                "goal_y": goal_room[1],
                "elapsed": round(elapsed, 1),
                "curriculum_episode": explorer.curriculum.episode
                if explorer.curriculum is not None
                else None,
            }
            print(
                f"    -> success={result['success']} fht={result['fht']} "
                f"scr={result['scr']:.2f} steps={result['steps_count']} "
                f"loss={result['train_loss']} [{elapsed:.0f}s]",
                flush=True,
            )

            all_results.append(result)

            # Incremental write
            line = {k: result[k] for k in [
                "width", "height", "mode", "seed", "episode",
                "steps_count", "success", "fht", "scr",
                "dead_loop_rate", "train_loss", "elapsed",
                "curriculum_episode",
            ]}
            with open(output_path, "a") as f:
                f.write(json.dumps(line) + "\n")

            # Advance curriculum AFTER each episode
            explorer.advance_curriculum()

        # Mode summary
        _print_mode_summary(mode, all_results, output_path)

    # Cross-mode comparison
    _print_cross_mode_comparison(modes, output_path)


def _print_mode_summary(mode, all_results, output_path):
    """Print summary for one mode over all episodes."""
    successes = sum(1 for r in all_results if r["success"])
    hits = [r for r in all_results if r["success"]]
    avg_scr = sum(r["scr"] for r in all_results) / max(len(all_results), 1)
    losses = [
        r.get("train_loss")
        for r in all_results
        if r.get("train_loss") is not None
    ]

    print(
        f"  [{mode}] Summary: {successes}/{len(all_results)} success "
        f"({100 * successes / max(len(all_results), 1):.0f}%) | "
        f"Avg SCR: {avg_scr:.3f} | "
        + (
            f"Train loss: {sum(losses) / max(len(losses), 1):.6f} "
            f"(avg over {len(losses)} eps)"
            if losses
            else ""
        ),
        flush=True,
    )


def _print_cross_mode_comparison(modes, output_path):
    """Print final comparison table across all modes for this maze."""
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

    print(f"\n{'─' * 60}")
    print(f"Cross-Mode Comparison")
    print(f"{'─' * 60}")

    for mode in modes:
        recs = mode_results.get(mode, [])
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
        avg_dlr = sum(r.get("dead_loop_rate", 0) for r in recs) / max(
            len(recs), 1
        )
        losses = [
            r.get("train_loss")
            for r in recs
            if r.get("train_loss") is not None
        ]
        avg_loss = sum(losses) / max(len(losses), 1) if losses else -1
        total_elapsed = sum(r.get("elapsed", 0) for r in recs)

        print(
            f"  {mode:>15s}: {sr}/{len(recs)} success | "
            f"FHT={avg_fht:.1f} | SCR={avg_scr:.3f} | "
            f"DLR={avg_dlr:.3f} | loss={avg_loss:.6f} | "
            f"{total_elapsed:.1f}s"
        )


# ── Main ─────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Phase 7 Curriculum Intrinsic Motivation Experiment"
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=[10, 20],
        help="Maze sizes (width=height, one or more values, default: 10 20)",
    )
    parser.add_argument(
        "--num-episodes", type=int, default=6, help="Episodes per condition"
    )
    parser.add_argument(
        "--max-steps", type=int, default=500, help="Max steps per episode"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None, help="Output path prefix")
    parser.add_argument(
        "--model-path",
        default=None,
        help="Path to Qwen model (default: ~/models/Qwen2.5-0.5B-Instruct)",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=MODES,
        default=MODES,
        help="Modes to run (default: all 5)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="Curriculum warmup episodes (default: 2)",
    )
    args = parser.parse_args()

    # Derive model path
    if args.model_path is None:
        args.model_path = os.path.expanduser(
            "~/models/Qwen2.5-0.5B-Instruct"
        )

    # Validate sizes
    sizes = []
    for s in args.sizes:
        if s < 3:
            print(f"Warning: minimum maze size is 3, got {s}")
            sizes.append(3)
        else:
            sizes.append(s)
    if not sizes:
        sizes = [10, 20]

    print(f"{'=' * 70}")
    print(f"Phase 7 Curriculum Intrinsic Motivation Experiment")
    print(f"{'=' * 70}")
    print(f"Maze sizes: {['{}x{}'.format(s, s) for s in sizes]}")
    print(f"Modes: {args.modes}")
    print(f"Episodes per condition: {args.num_episodes}")
    print(f"Max steps: {args.max_steps}")
    print(f"Seed: {args.seed}")
    print(f"Warmup episodes: {args.warmup}")
    print(f"Model: {args.model_path}")
    print(f"{'=' * 70}", flush=True)

    # Lazy import (torch may set CUDA during import)
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}", flush=True)

    # Load JEPA ensemble (shared across modes that need it)
    jepa_needed = any(
        m in ("pure_jepa", "hybrid", "curriculum", "ucb") for m in args.modes
    )
    jepa = None
    if jepa_needed:
        t0 = time.time()
        print(f"\nLoading JEPA ensemble on {device}...", flush=True)
        jepa = JEPAEnsemble(
            args.model_path, n_ensemble=3, device=device
        )
        print(
            f"  Loaded in {time.time() - t0:.1f}s "
            f"(hidden_size={jepa.hidden_size})",
            flush=True,
        )

    # Ensure output directory
    output_dir = _PROJECT_ROOT / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run for each maze size
    for size in sizes:
        print(
            f"\n{'#' * 70}",
            flush=True,
        )
        print(
            f"# Maze: {size}x{size}",
            flush=True,
        )
        print(
            f"{'#' * 70}",
            flush=True,
        )

        # Generate maze
        print(
            f"\nGenerating {size}x{size} maze (seed={args.seed})...",
            flush=True,
        )
        maze = GridMaze.generate(size, size, seed=args.seed)

        # Far corner goal
        goal_room = (size - 1, size - 1)
        task = {
            "goal_room": goal_room,
            "start_x": 0,
            "start_y": 0,
            "max_steps": args.max_steps,
        }

        state_estimate = maze.state_estimate()
        print(f"  State estimate: ~{state_estimate}")
        print(f"  Goal room: {goal_room}", flush=True)

        # Output path
        output_path = (
            Path(args.output)
            if args.output
            else output_dir
            / f"phase7_curriculum_{size}x{size}_seed{args.seed}.jsonl"
        )

        # Override warmup in explorer via command arg
        # (passed through constructor)
        run_maze_comparison(
            maze=maze,
            task=task,
            modes=args.modes,
            num_episodes=args.num_episodes,
            max_steps=args.max_steps,
            jepa=jepa,
            seed=args.seed,
            output_path=output_path,
            warmup_episodes=args.warmup,
        )

        print(f"\nResults for {size}x{size}: {output_path}", flush=True)

    print(f"\n{'=' * 70}")
    print("Done.")
    print(f"{'=' * 70}", flush=True)


if __name__ == "__main__":
    main()
