#!/usr/bin/env python3
# ruff: noqa: E402
"""Phase 5 JEPA Experiment: Epistemic uncertainty from embedding-space prediction.

Replaces LLM token prediction with a frozen Qwen 0.5B encoder + trainable
MLP ensemble. Epistemic uncertainty = variance across ensemble predictions.

Compares 3 exploration modes:
  a. pure_novelty: count-based NoveltyExplorer only (baseline)
  b. jepa_only: JEPA epistemic signal only
  c. hybrid: 0.5 * novelty + 0.5 * epistemic

Usage:
  # Run all 3 modes sequentially (default)
  python scripts/phase5_jepa_experiment.py --task read_hello --num-episodes 6 --max-steps 10

  # Single mode
  python scripts/phase5_jepa_experiment.py --task find_secret --mode hybrid --num-episodes 6

  # All tasks
  for t in read_hello read_note count_lines find_secret; do
    python scripts/phase5_jepa_experiment.py --task $t --num-episodes 6
  done
"""

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from phase2.sandbox_env import BusyboxSandbox, generate_sandbox_candidates
from phase2.tasks import MICRO_TASKS
from phase5.explorer import NoveltyExplorer
from phase5.jepa_wm import JEPAEnsemble, state_to_text, next_state_to_text


# ── Metrics ──────────────────────────────────────────────

def compute_metrics(steps, task_id):
    """Compute FHT, SCR, Dead-loop Rate from step records."""
    fht = None
    task = next((t for t in MICRO_TASKS if t["id"] == task_id), None)
    if task:
        for rec in steps:
            fake_ns = type("obj", (object,), {
                "last_output": rec.get("output", ""),
                "last_exit_code": rec.get("exit_code", 0),
                "files": rec.get("next_files", []),
                "cwd": rec.get("next_cwd", ""),
            })()
            if task["check"](fake_ns, rec["action"], fake_ns):
                fht = rec["step"]
                break
    visited = set()
    for rec in steps:
        visited.add(f"{rec['cwd']}|{tuple(rec['files'])}")
    scr = len(visited) / max(len(steps), 1)
    loops = 0
    for i in range(2, len(steps)):
        if steps[i]["action"] == steps[i-1]["action"] == steps[i-2]["action"]:
            loops += 1
    dead_loop_rate = loops / max(len(steps), 1)
    return {"fht": fht, "scr": round(scr, 3), "dead_loop_rate": round(dead_loop_rate, 3),
            "steps": len(steps)}


# ── CWD Definitions ──
KNOWN_CWDS = ["/sandbox", "/sandbox/docs", "/sandbox/data/raw", "/sandbox/logs/app", "/sandbox/projects/backend"]
UNKNOWN_CWDS = ["/sandbox/docs/tutorials", "/sandbox/data/processed", "/sandbox/data/archive",
                "/sandbox/logs/system", "/sandbox/logs/audit", "/sandbox/projects/frontend",
                "/sandbox/projects/shared", "/sandbox/cache/temp"]


# ── JEPA Explorer ────────────────────────────────────────

class JEPAExplorer:
    """Action selector driven by JEPA epistemic uncertainty + optional novelty.

    Three modes:
      pure_novelty: count-based only (same as NoveltyExplorer)
      jepa_only: JEPA ensemble variance only
      hybrid: 0.5 * novelty + 0.5 * epistemic
    """

    _ACTION_PRIORITY = {
        "cat": 0, "head": 0, "tail": 0,
        "grep": 1, "find": 1, "wc": 1,
        "cd": 2,
        "ls": 3, "pwd": 3, "echo": 3,
    }

    def __init__(self, jepa_ensemble: JEPAEnsemble = None, mode: str = "hybrid",
                 novelty_weight: float = 0.5, epistemic_weight: float = 0.5):
        self.jepa = jepa_ensemble
        self.mode = mode
        self.novelty_weight = novelty_weight
        self.epistemic_weight = epistemic_weight

        # Count-based tables (shared by all modes)
        self.state_counts = defaultdict(int)
        self.state_action_counts = defaultdict(int)
        self.success_cache = {}

    def _action_priority(self, action: str) -> int:
        verb = action.split()[0] if action else ""
        return self._ACTION_PRIORITY.get(verb, 4)

    def novelty_bonus(self, state, action: str) -> float:
        """Count-based intrinsic reward."""
        sh = state.state_hash()
        state_novelty = 1.0 / max(1.0, self.state_counts[sh] ** 0.5)
        pair_novelty = 1.0 / max(1.0, self.state_action_counts[(sh, action)] ** 0.5)
        return 0.5 * state_novelty + 0.5 * pair_novelty

    def select_action(self, state, candidates, action_history, jepa_wm=None):
        """Select action using configured exploration mode."""
        if not candidates:
            return "ls"

        sh = state.state_hash()

        # Cached success replay
        if sh in self.success_cache:
            cached = self.success_cache[sh]
            if cached in candidates:
                return cached

        # Score all candidates
        state_text = state_to_text(state)

        def score(action):
            if self.mode == "pure_novelty":
                return self.novelty_bonus(state, action)
            elif self.mode == "jepa_only":
                epi = self.jepa.epistemic_uncertainty(state_text, action) if self.jepa else 0.0
                return epi
            else:  # hybrid
                nov = self.novelty_bonus(state, action)
                epi = self.jepa.epistemic_uncertainty(state_text, action) if self.jepa else 0.0
                return self.novelty_weight * nov + self.epistemic_weight * epi

        # Pick highest-scoring, tie-break by action priority (lower = prefer)
        return max(candidates, key=lambda a: (score(a), -self._action_priority(a)))

    def observe(self, state, action: str, success: bool):
        sh = state.state_hash()
        self.state_counts[sh] += 1
        self.state_action_counts[(sh, action)] += 1
        if success:
            self.success_cache[sh] = action

    def reset_episode(self):
        pass


# ── Episode Runner ───────────────────────────────────────

def run_jepa_episode(sb, explorer, jepa_ensemble, task_id, max_steps,
                     start_cwd, mode):
    """Run a single episode with JEPA-driven exploration.

    After each step the transition is collected; after the episode
    (if jepa mode) the ensemble is trained on all transitions.
    """
    state = sb.reset(start_cwd=start_cwd)
    action_history = []
    steps = []
    transitions = []  # (state, action, next_state) for JEPA training
    task_def = next((t for t in MICRO_TASKS if t["id"] == task_id), None)

    for step_i in range(max_steps):
        # 1. Generate candidates
        candidates = generate_sandbox_candidates(state)
        if not candidates or len(candidates) < 2:
            candidates = ["ls", "pwd", "cat hello.txt", "ls data", "ls docs"]

        # 2. Select action
        action = explorer.select_action(state, candidates, action_history)

        # 3. Execute
        next_state, reward, done = sb.step(state, action)

        # 4. Record transition for JEPA training (all modes collect)
        transitions.append((state, action, next_state))

        # 5. Check success
        if task_def:
            success = task_def["check"](state, action, next_state)
        else:
            success = False

        # 6. Observe
        explorer.observe(state, action, success)

        # 7. Record step
        record = {
            "step": step_i,
            "cwd": state.cwd,
            "files": list(state.files),
            "action": action,
            "success": success,
            "next_cwd": next_state.cwd,
            "next_files": list(next_state.files),
            "exit_code": next_state.last_exit_code,
            "output": next_state.last_output[:100],
        }
        steps.append(record)

        if success:
            next_state.victory = True
            final_state = next_state
            break

        action_history.append(action)
        state = next_state

        if done or state.game_over:
            break

    final_state = final_state if "final_state" in dir() else state

    # Post-episode JEPA training
    train_loss = None
    if mode in ("jepa_only", "hybrid") and jepa_ensemble is not None and transitions:
        train_loss = jepa_ensemble.train_step(transitions)

    return steps, final_state, train_loss


# ── Main ─────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase 5 JEPA Sandbox Experiment")
    parser.add_argument("--task", default="read_hello",
                        help="Task ID from MICRO_TASKS")
    parser.add_argument("--num-episodes", type=int, default=6)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--mode", choices=["pure_novelty", "jepa_only", "hybrid", "all"],
                        default="all", help="Exploration mode (default: all 3)")
    parser.add_argument("--output", default=None)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--model-path", default=None,
                        help="Path to Qwen model (default: ~/models/Qwen2.5-0.5B-Instruct)")
    args = parser.parse_args()

    # Derive model path
    if args.model_path is None:
        args.model_path = os.path.expanduser("~/models/Qwen2.5-0.5B-Instruct")

    # Determine modes to run
    modes = ["pure_novelty", "jepa_only", "hybrid"] if args.mode == "all" else [args.mode]

    # Derive output path
    if args.output is None:
        output_dir = _PROJECT_ROOT / "results"
        output_dir.mkdir(parents=True, exist_ok=True)
        mode_tag = args.mode if args.mode != "all" else "all_modes"
        args.output = str(output_dir / f"phase5_jepa_{mode_tag}_{args.task}.jsonl")

    output_path = Path(args.output)

    # Select CWDs
    cwds = KNOWN_CWDS + UNKNOWN_CWDS

    print(f"{'='*60}")
    print(f"Phase 5 JEPA Experiment")
    print(f"  Task: {args.task}")
    print(f"  Episodes per mode: {args.num_episodes}")
    print(f"  Max steps: {args.max_steps}")
    print(f"  Modes: {modes}")
    print(f"  Model: {args.model_path}")
    print(f"  Output: {output_path}")
    print(f"{'='*60}", flush=True)

    # Open sandbox once
    sb = BusyboxSandbox()

    # Load JEPA ensemble (shared across modes that need it)
    jepa = None
    if any(m in ("jepa_only", "hybrid") for m in modes):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"\nLoading JEPA ensemble on {device}...", flush=True)
        t0 = time.time()
        jepa = JEPAEnsemble(args.model_path, n_ensemble=3, device=device)
        print(f"  Loaded in {time.time()-t0:.1f}s (hidden_size={jepa.hidden_size})", flush=True)

    for mode in modes:
        print(f"\n{'─'*60}")
        print(f"Mode: {mode.upper()}")
        print(f"{'─'*60}", flush=True)

        explorer = JEPAExplorer(jepa_ensemble=jepa, mode=mode)
        all_results = []

        for ep_idx in range(args.num_episodes):
            cwd = cwds[ep_idx % len(cwds)]
            seed = args.seed_offset + ep_idx
            random.seed(seed)

            explorer.reset_episode()

            print(f"  [Episode {ep_idx+1}/{args.num_episodes}] cwd={cwd}", flush=True)
            t0 = time.time()

            try:
                steps, final_state, train_loss = run_jepa_episode(
                    sb, explorer, jepa, args.task, args.max_steps,
                    start_cwd=cwd, mode=mode,
                )
            except Exception as e:
                print(f"    ERROR: {e}", flush=True)
                import traceback
                traceback.print_exc()
                steps = []
                final_state = type("obj", (object,), {
                    "victory": False, "cwd": cwd, "step_count": 0
                })()
                train_loss = None

            elapsed = time.time() - t0

            if steps:
                metrics = compute_metrics(steps, args.task)
            else:
                metrics = {"fht": -1, "scr": 0.0, "dead_loop_rate": 0.0}

            result = {
                "mode": mode,
                "task": args.task,
                "cwd": cwd,
                "episode": ep_idx,
                "steps_count": len(steps),
                "success": getattr(final_state, "victory", False),
                "fht": metrics.get("fht", -1),
                "scr": metrics.get("scr", 0.0),
                "dead_loop_rate": metrics.get("dead_loop_rate", 0.0),
                "train_loss": round(train_loss, 6) if train_loss is not None else None,
                "elapsed": round(elapsed, 1),
                "records": steps,
            }
            print(f"  -> success={result['success']} fht={result['fht']} "
                  f"scr={result['scr']:.2f} steps={result['steps_count']} "
                  f"loss={result['train_loss']} [{elapsed:.0f}s]",
                  flush=True)

            all_results.append(result)

            # Incremental write
            line = {k: result[k] for k in ["mode", "task", "cwd", "episode",
                                            "steps_count", "success", "fht", "scr",
                                            "dead_loop_rate", "train_loss", "elapsed"]}
            with open(output_path, "a") as f:
                f.write(json.dumps(line) + "\n")

        # Mode summary
        successes = sum(1 for r in all_results if r["success"])
        hits = [r for r in all_results if r["success"]]
        avg_scr = sum(r["scr"] for r in all_results) / max(len(all_results), 1)
        losses = [r.get("train_loss") for r in all_results if r.get("train_loss") is not None]

        print(f"  [{mode}] Summary: {successes}/{args.num_episodes} success "
              f"({100*successes/max(args.num_episodes,1):.0f}%) | "
              f"Avg SCR: {avg_scr:.3f} | "
              f"Train loss: {sum(losses)/max(len(losses),1):.6f} (avg over {len(losses)} eps)"
              if losses else "",
              flush=True)

    # Final cross-mode comparison
    print(f"\n{'='*60}")
    print(f"Cross-Mode Comparison for {args.task}")
    print(f"{'='*60}")

    # Re-read results
    mode_results = {m: [] for m in modes}
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
        fhts = [r["fht"] for r in recs if r.get("fht") is not None and r["fht"] >= 0]
        avg_fht = sum(fhts) / max(len(fhts), 1) if fhts else -1
        avg_scr = sum(r.get("scr", 0) for r in recs) / max(len(recs), 1)
        avg_dlr = sum(r.get("dead_loop_rate", 0) for r in recs) / max(len(recs), 1)
        losses = [r.get("train_loss") for r in recs if r.get("train_loss") is not None]
        avg_loss = sum(losses) / max(len(losses), 1) if losses else -1
        total_elapsed = sum(r.get("elapsed", 0) for r in recs)

        print(f"  {mode:>15s}: {sr}/{len(recs)} success | "
              f"FHT={avg_fht:.1f} | SCR={avg_scr:.3f} | "
              f"DLR={avg_dlr:.3f} | loss={avg_loss:.6f} | "
              f"{total_elapsed:.1f}s")

    print(f"\nOutput: {output_path}")
    print(f"{'='*60}", flush=True)

    sb.close()


# Lazy torch import (after device detection)
import torch

if __name__ == "__main__":
    main()
