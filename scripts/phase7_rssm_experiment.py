#!/usr/bin/env python3
# ruff: noqa: E402
"""Phase 7 RSSM Experiment: Compare RSSM vs MLP-JEPA vs Count in grid mazes.

Tests the hypothesis that RSSM's temporal latent state captures structure
that simple MLP ensemble misses, making epistemic uncertainty more informative
for exploration.

Architecture comparison:
  rssm       — MiniRSSM (GRU + stochastic latent + decoder, ensemble of 3)
  mlp_jepa   — SimpleJEPA (MLP ensemble, no temporal, same encoder)
  count      — Count-based novelty (no learning, MazeNoveltyExplorer)

Usage:
  python scripts/phase7_rssm_experiment.py --mode all \\
      --width 10 --height 10 --num-episodes 3 --max-steps 500
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

from phase6.grid_env import GridMazeEnv, GridState
from phase6.maze_generator import GridMaze
from phase7.rssm_wm import (
    MiniRSSM,
    TemporalJEPA,
    SimpleJEPA,
    grid_state_features,
    action_to_idx,
    rssm_training_step,
    temporal_jepa_training_step,
    simple_jepa_training_step,
)

import torch

# ── Backtrack penalty helpers ───────────────────────────

_REVERSE_MOVE = {
    "go north": "go south",
    "go south": "go north",
    "go east": "go west",
    "go west": "go east",
}

_ACTION_PRIORITY_MAP = {
    "go": 0,
    "look": 1,
    "inventory": 2,
    "use": 3,
    "take": 4,
}


def _action_priority(action: str) -> int:
    verb = action.split()[0] if action else ""
    return _ACTION_PRIORITY_MAP.get(verb, 5)


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


# ── Count-based Explorer (adapted from phase6_maze_count) ──

class MazeNoveltyExplorer:
    """Count-based novelty explorer with backtrack penalty."""

    def __init__(self):
        self.state_counts = {}
        self.state_action_counts = {}
        self.success_cache = {}
        self._prev_pos = None
        self._prev_pos_action = None

    def novelty_bonus(self, state, action: str) -> float:
        sh = state.state_hash()
        state_novelty = 1.0 / math.sqrt(1 + self.state_counts.get(sh, 0))
        pair_novelty = 1.0 / math.sqrt(1 + self.state_action_counts.get((sh, action), 0))
        bonus = 0.5 * state_novelty + 0.5 * pair_novelty

        # Backtrack penalty
        if self._prev_pos is not None and action == _REVERSE_MOVE.get(self._prev_pos_action, ""):
            dx = {'go north': (0,-1), 'go south': (0,1), 'go east': (1,0), 'go west': (-1,0)}
            move = dx.get(action)
            if move:
                dest = (state.x + move[0], state.y + move[1])
                if dest == self._prev_pos:
                    bonus *= 0.5
        return bonus

    def select_action(self, state, candidates, action_history):
        if not candidates:
            return "look"
        sh = state.state_hash()
        if sh in self.success_cache and self.success_cache[sh] in candidates:
            return self.success_cache[sh]
        return max(candidates, key=lambda a: (
            self.novelty_bonus(state, a),
            -_action_priority(a),
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


# ── RSSM Explorer ────────────────────────────────────────

class RSSMExplorer:
    """Action selector driven by RSSM epistemic uncertainty."""

    def __init__(self, rssm: MiniRSSM, mode: str = "rssm", novelty_weight: float = 0.3):
        self.rssm = rssm
        self.mode = mode  # "rssm" or "hybrid_rssm"
        self.novelty_weight = novelty_weight
        self.epistemic_weight = 1.0 - novelty_weight

        # Count-based tables for hybrid mode
        self.state_counts = {}
        self.state_action_counts = {}
        self.success_cache = {}
        self._prev_pos = None
        self._prev_pos_action = None

    def novelty_bonus(self, state, action: str) -> float:
        sh = state.state_hash()
        state_n = 1.0 / math.sqrt(1 + self.state_counts.get(sh, 0))
        pair_n = 1.0 / math.sqrt(1 + self.state_action_counts.get((sh, action), 0))
        bonus = 0.5 * state_n + 0.5 * pair_n
        if self._prev_pos is not None and action == _REVERSE_MOVE.get(self._prev_pos_action, ""):
            dx = {'go north': (0,-1), 'go south': (0,1), 'go east': (1,0), 'go west': (-1,0)}
            move = dx.get(action)
            if move:
                dest = (state.x + move[0], state.y + move[1])
                if dest == self._prev_pos:
                    bonus *= 0.5
        return bonus

    @torch.no_grad()
    def rssm_uncertainty(self, state_feats: torch.Tensor, action_idx: torch.Tensor,
                         h: torch.Tensor, z: torch.Tensor) -> float:
        """Compute RSSM ensemble epistemic uncertainty.

        Re-runs forward for all 3 ensemble members (each has its own RNN state).
        Falls back to prior-only uncertainty if separate RNN states not available.
        """
        # Use prior variance as epistemic signal
        act_emb = self.rssm.action_encoder(action_idx)
        rnn_in = torch.cat([z, act_emb], dim=-1)
        h_t = self.rssm.rnn(rnn_in, h)
        prior_params = self.rssm.prior(h_t)
        _, logσ = prior_params.chunk(2, dim=-1)
        # Prior variance magnitude = epistemic uncertainty about latent
        return float(logσ.exp().mean().item())

    def select_action(self, state, candidates, action_history,
                      h: torch.Tensor, z: torch.Tensor):
        if not candidates:
            return "look"

        sh = state.state_hash()
        if sh in self.success_cache and self.success_cache[sh] in candidates:
            return self.success_cache[sh]

        feats = grid_state_features(state)
        action_uncertainties = []

        for a in candidates:
            aidx = torch.tensor([action_to_idx(a)])
            epi = self.rssm_uncertainty(feats, aidx, h, z)

            if self.mode == "hybrid_rssm":
                nov = self.novelty_bonus(state, a)
                score = self.novelty_weight * nov + self.epistemic_weight * epi
            else:
                score = epi

            action_uncertainties.append((a, score))

        # Highest score, tie-break by action priority
        return max(action_uncertainties, key=lambda x: (x[1], -_action_priority(x[0])))[0]

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


# ── JEPA Explorer (for SimpleJEPA mode) ─────────────────

class JEPAExplorer:
    """Action selector driven by SimpleJEPA epistemic uncertainty."""

    def __init__(self, model: SimpleJEPA, mode: str = "jepa",
                 novelty_weight: float = 0.3):
        self.model = model
        self.mode = mode
        self.novelty_weight = novelty_weight
        self.epistemic_weight = 1.0 - novelty_weight

        self.state_counts = {}
        self.state_action_counts = {}
        self.success_cache = {}
        self._prev_pos = None
        self._prev_pos_action = None

    def novelty_bonus(self, state, action: str) -> float:
        sh = state.state_hash()
        state_n = 1.0 / math.sqrt(1 + self.state_counts.get(sh, 0))
        pair_n = 1.0 / math.sqrt(1 + self.state_action_counts.get((sh, action), 0))
        bonus = 0.5 * state_n + 0.5 * pair_n
        if self._prev_pos is not None and action == _REVERSE_MOVE.get(self._prev_pos_action, ""):
            dx = {'go north': (0,-1), 'go south': (0,1), 'go east': (1,0), 'go west': (-1,0)}
            move = dx.get(action)
            if move:
                dest = (state.x + move[0], state.y + move[1])
                if dest == self._prev_pos:
                    bonus *= 0.5
        return bonus

    @torch.no_grad()
    def jepa_uncertainty(self, state_feats: torch.Tensor, action_idx: torch.Tensor) -> float:
        return self.model.epistemic_uncertainty(state_feats, action_idx)

    def select_action(self, state, candidates, action_history):
        if not candidates:
            return "look"
        sh = state.state_hash()
        if sh in self.success_cache and self.success_cache[sh] in candidates:
            return self.success_cache[sh]

        feats = grid_state_features(state)

        def score(a):
            aidx = torch.tensor([action_to_idx(a)])
            epi = self.model.epistemic_uncertainty(feats, aidx)
            if self.mode == "hybrid_jepa":
                nov = self.novelty_bonus(state, a)
                return self.novelty_weight * nov + self.epistemic_weight * epi
            return epi

        return max(candidates, key=lambda a: (score(a), -_action_priority(a)))

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


# ── Episode Runners ──────────────────────────────────────

def run_count_episode(env, explorer, max_steps):
    """Run count-based exploration episode."""
    env.reset()
    action_history = []
    steps = []

    for step_i in range(max_steps):
        state = env._get_state()
        candidates = env.get_dynamic_candidates()
        action = explorer.select_action(state, candidates, action_history)

        goal_room = env.task.get("goal_room")
        prev_dist = abs(state.x - goal_room[0]) + abs(state.y - goal_room[1])
        next_obs, next_state, done = env.step(action)
        new_dist = abs(next_state.x - goal_room[0]) + abs(next_state.y - goal_room[1])
        success = (new_dist < prev_dist) or next_state.goal_reached

        explorer.observe(state, action, success)
        if (next_state.x, next_state.y) != (state.x, state.y):
            explorer.observe_move(action, (state.x, state.y), (next_state.x, next_state.y))

        steps.append({
            "step": step_i,
            "x": next_state.x,
            "y": next_state.y,
            "action": action,
            "success": success,
            "goal_reached": next_state.goal_reached,
            "inventory": list(env.inventory),
        })

        if next_state.goal_reached:
            break
        action_history.append(action)

    return steps, env._get_state()


def run_rssm_episode(env, explorer, rssm, max_steps, mode):
    """Run RSSM-driven exploration episode with per-episode training."""
    env.reset()
    action_history = []
    steps = []
    transitions = []

    h = rssm.init_hidden()
    z = rssm.init_latent()

    for step_i in range(max_steps):
        state = env._get_state()
        candidates = env.get_dynamic_candidates()

        action = explorer.select_action(state, candidates, action_history, h, z)

        goal_room = env.task.get("goal_room")
        prev_dist = abs(state.x - goal_room[0]) + abs(state.y - goal_room[1])
        next_obs, next_state, done = env.step(action)
        new_dist = abs(next_state.x - goal_room[0]) + abs(next_state.y - goal_room[1])
        success = (new_dist < prev_dist) or next_state.goal_reached

        transitions.append((state, action, next_state))

        # Track success for explorer
        explorer.observe(state, action, success)
        if (next_state.x, next_state.y) != (state.x, state.y):
            explorer.observe_move(action, (state.x, state.y), (next_state.x, next_state.y))

        # Update RSSM latent state for next step
        feats = grid_state_features(state)
        aidx = torch.tensor([action_to_idx(action)])
        with torch.no_grad():
            _, h, z, _, _ = rssm.forward(feats, aidx, h, z)

        steps.append({
            "step": step_i,
            "x": next_state.x,
            "y": next_state.y,
            "action": action,
            "success": success,
            "goal_reached": next_state.goal_reached,
            "inventory": list(env.inventory),
        })

        if next_state.goal_reached:
            break
        action_history.append(action)

    # Post-episode RSSM training
    train_loss = None
    if transitions:
        train_loss = rssm_training_step(rssm, transitions)

    return steps, env._get_state(), train_loss


def run_jepa_episode(env, explorer, model, max_steps, mode):
    """Run SimpleJEPA-driven exploration episode with per-episode training."""
    env.reset()
    action_history = []
    steps = []
    transitions = []

    for step_i in range(max_steps):
        state = env._get_state()
        candidates = env.get_dynamic_candidates()

        action = explorer.select_action(state, candidates, action_history)

        goal_room = env.task.get("goal_room")
        prev_dist = abs(state.x - goal_room[0]) + abs(state.y - goal_room[1])
        next_obs, next_state, done = env.step(action)
        new_dist = abs(next_state.x - goal_room[0]) + abs(next_state.y - goal_room[1])
        success = (new_dist < prev_dist) or next_state.goal_reached

        transitions.append((state, action, next_state))
        explorer.observe(state, action, success)
        if (next_state.x, next_state.y) != (state.x, state.y):
            explorer.observe_move(action, (state.x, state.y), (next_state.x, next_state.y))

        steps.append({
            "step": step_i,
            "x": next_state.x,
            "y": next_state.y,
            "action": action,
            "success": success,
            "goal_reached": next_state.goal_reached,
            "inventory": list(env.inventory),
        })

        if next_state.goal_reached:
            break
        action_history.append(action)

    # Post-episode training
    train_loss = None
    if transitions:
        train_loss = simple_jepa_training_step(model, transitions)

    return steps, env._get_state(), train_loss


# ── Main ─────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Phase 7 RSSM Experiment: RSSM vs MLP-JEPA vs Count"
    )
    parser.add_argument("--width", type=int, default=10)
    parser.add_argument("--height", type=int, default=10)
    parser.add_argument("--num-episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--mode",
        choices=["rssm", "mlp_jepa", "count", "all"],
        default="all",
        help="Exploration mode (default: all 3)",
    )
    parser.add_argument("--output", default=None)
    parser.add_argument("--goal", default="far",
                        help="Goal placement: 'far' = bottom-right, 'random' = random")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)

    # Determine modes to run
    modes = (
        ["rssm", "mlp_jepa", "count"]
        if args.mode == "all"
        else [args.mode]
    )

    # Auto-scale max_steps
    if args.max_steps is None:
        size = args.width * args.height
        args.max_steps = min(size * 4, 500)

    # Derive output path
    if args.output is None:
        output_dir = _PROJECT_ROOT / "results"
        output_dir.mkdir(parents=True, exist_ok=True)
        mode_tag = args.mode if args.mode != "all" else "all_modes"
        args.output = str(
            output_dir
            / f"phase7_rssm_{mode_tag}_{args.width}x{args.height}"
            f"_seed{args.seed}.jsonl"
        )
    output_path = Path(args.output)

    print(f"{'=' * 60}")
    print(f"Phase 7 RSSM Experiment")
    print(f"  Maze: {args.width}x{args.height}")
    print(f"  Episodes per mode: {args.num_episodes}")
    print(f"  Max steps: {args.max_steps}")
    print(f"  Seed: {args.seed}")
    print(f"  Modes: {modes}")
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

    # ── Run each mode ──────────────────────────────────

    for mode in modes:
        print(f"\n{'─' * 60}")
        print(f"Mode: {mode.upper()}")
        print(f"{'─' * 60}", flush=True)

        env = GridMazeEnv(maze, task)
        env.setup()

        all_results = []

        for ep_idx in range(args.num_episodes):
            seed = args.seed + ep_idx
            random.seed(seed)

            print(f"  [Episode {ep_idx + 1}/{args.num_episodes}]", flush=True)
            t0 = time.time()

            try:
                if mode == "count":
                    explorer = MazeNoveltyExplorer()
                    explorer.reset_episode()
                    steps, final_state = run_count_episode(env, explorer, args.max_steps)
                    train_loss = None

                elif mode == "rssm":
                    torch.manual_seed(seed + 100)
                    rssm = MiniRSSM(
                        state_dim=32, action_dim=16,
                        hidden_dim=128, latent_dim=16,
                    )
                    if device == "cuda":
                        rssm = rssm.cuda()
                    explorer = RSSMExplorer(rssm, mode="rssm")
                    explorer.reset_episode()
                    steps, final_state, train_loss = run_rssm_episode(
                        env, explorer, rssm, args.max_steps, mode="rssm"
                    )

                elif mode == "mlp_jepa":
                    torch.manual_seed(seed + 100)
                    sjepa = SimpleJEPA(
                        state_dim=32, action_dim=16, n_ensemble=3,
                    )
                    if device == "cuda":
                        sjepa = sjepa.cuda()
                    explorer = JEPAExplorer(sjepa, mode="jepa")
                    explorer.reset_episode()
                    steps, final_state, train_loss = run_jepa_episode(
                        env, explorer, sjepa, args.max_steps, mode="jepa"
                    )

            except Exception as e:
                print(f"    ERROR: {e}", flush=True)
                import traceback
                traceback.print_exc()
                steps = []
                final_state = env._get_state() if hasattr(env, '_get_state') else None
                train_loss = None

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

    # ── Cross-mode Comparison ──────────────────────────

    print(f"\n{'=' * 60}")
    print(f"Cross-Mode Comparison for {args.width}x{args.height}")
    print(f"{'=' * 60}")

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
        avg_dlr = sum(r.get("dead_loop_rate", 0) for r in recs) / max(len(recs), 1)
        losses = [
            r.get("train_loss")
            for r in recs
            if r.get("train_loss") is not None
        ]
        avg_loss = sum(losses) / max(len(losses), 1) if losses else -1
        total_elapsed = sum(r.get("elapsed", 0) for r in recs)

        print(
            f"  {mode:>10s}: {sr}/{len(recs)} success | "
            f"FHT={avg_fht:.1f} | SCR={avg_scr:.3f} | "
            f"DLR={avg_dlr:.3f} | loss={avg_loss:.6f} | "
            f"{total_elapsed:.1f}s"
        )

    print(f"\nOutput: {output_path}")
    print(f"{'=' * 60}", flush=True)


if __name__ == "__main__":
    main()
