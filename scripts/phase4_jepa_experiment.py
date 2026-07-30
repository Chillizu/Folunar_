#!/usr/bin/env python3
# ruff: noqa: E402
"""Phase 4 JEPA-PEDA Experiment: Multi-condition comparison.

Compares 4 exploration conditions:
  - jepa_efe: JEPA epistemic + pragmatic (EFE balanced, alpha=0.5)
  - jepa_only: pure JEPA epistemic (alpha=1.0)
  - pragmatic_only: pure task relevance (alpha=0.0)
  - novelty_only: count-based novelty (baseline from Phase 5)

Usage:
  # Single condition
  python scripts/phase4_jepa_experiment.py --task read_hello --condition jepa_efe --num-episodes 6

  # All conditions for one task (smoke test)
  for cond in jepa_efe jepa_only pragmatic_only novelty_only; do
    PYTHONPATH=src python3 scripts/phase4_jepa_experiment.py --task read_hello --condition $cond
  done

  # Full experiment: all 4 tasks x 4 conditions
  for task in read_hello count_lines find_secret read_note; do
    for cond in jepa_efe jepa_only pragmatic_only novelty_only; do
      PYTHONPATH=src python3 scripts/phase4_jepa_experiment.py \
        --task $task --condition $cond --num-episodes 12
    done
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

# ── Image for original tasks (v2) ──
# The 4 original tasks (read_hello, count_lines, find_secret, read_note)
# are defined in the v2 Docker image.
DOCKER_IMAGE_V2 = "peda-sandbox:v4"

from phase2.sandbox_env import BusyboxSandbox, generate_sandbox_candidates
from phase2.tasks import MICRO_TASKS
from phase5.explorer import NoveltyExplorer
from phase4.jepa_peda import JEPAPEDA, state_to_text_flat


# ── CWD definitions (v2 sandbox) ──
# 6 CWDs: 3 "known" (files present), 3 "unknown" (no task files)
KNOWN_CWDS_V2 = ["/sandbox", "/sandbox/docs", "/sandbox/data/raw", "/sandbox/logs/app", "/sandbox/projects/backend"]
UNKNOWN_CWDS_V2 = ["/sandbox/docs/tutorials", "/sandbox/data/processed", "/sandbox/data/archive",
                "/sandbox/logs/system", "/sandbox/logs/audit", "/sandbox/projects/frontend",
                "/sandbox/projects/shared", "/sandbox/cache/temp"]


# ── Condition configs ──
CONDITION_CONFIGS = {
    "jepa_efe": {
        "alpha": 0.5,
        "description": "JEPA epistemic + pragmatic (EFE balanced)",
    },
    "jepa_only": {
        "alpha": 1.0,
        "description": "Pure JEPA epistemic uncertainty",
    },
    "pragmatic_only": {
        "alpha": 0.0,
        "description": "Pure task relevance (no exploration)",
    },
}

NOVELTY_ONLY = "novelty_only"


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
    return {
        "fht": fht,
        "scr": round(scr, 3),
        "dead_loop_rate": round(dead_loop_rate, 3),
        "steps": len(steps),
    }


# ── Episode Runners ──────────────────────────────────────

def run_jepa_peda_episode(sb, peda, task_id, max_steps, start_cwd):
    """Run a single episode with JEPA-PEDA EFE-driven action selection.

    After the episode, trains all 3 MLP predictors on collected transitions.
    """
    state = sb.reset(start_cwd=start_cwd)
    steps = []
    transitions = []  # (state_text, action, next_state_text)
    task_def = next((t for t in MICRO_TASKS if t["id"] == task_id), None)

    for step_i in range(max_steps):
        # 1. Generate candidates
        candidates = generate_sandbox_candidates(state)
        if not candidates or len(candidates) < 2:
            candidates = ["ls", "pwd", "cat hello.txt", "ls data", "ls docs"]

        # 2. Select action via EFE
        action = peda.select_action(state, candidates, task_id)

        # 3. Execute
        next_state, reward, done = sb.step(state, action)

        # 4. Record transition
        state_text = state_to_text_flat(state)
        next_text = state_to_text_flat(next_state)
        transitions.append((state_text, action, next_text))

        # 5. Check success
        if task_def:
            success = task_def["check"](state, action, next_state)
        else:
            success = False

        # 6. Record step
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

        state = next_state

        if done or state.game_over:
            break

    final_state = final_state if "final_state" in dir() else state

    # Post-episode JEPA training
    train_loss = None
    if transitions:
        train_loss = peda.train_on_episode(transitions)

    return steps, final_state, train_loss


def run_novelty_episode(sb, explorer, task_id, max_steps, start_cwd):
    """Run a single episode with count-based novelty exploration (baseline)."""
    state = sb.reset(start_cwd=start_cwd)
    action_history = []
    steps = []
    task_def = next((t for t in MICRO_TASKS if t["id"] == task_id), None)

    for step_i in range(max_steps):
        # 1. Generate candidates
        candidates = generate_sandbox_candidates(state)
        if not candidates or len(candidates) < 2:
            candidates = ["ls", "pwd", "cat hello.txt", "ls data", "ls docs"]

        # 2. Select action via novelty
        action = explorer.select_action(state, candidates, action_history)

        # 3. Execute
        next_state, reward, done = sb.step(state, action)

        # 4. Check success
        if task_def:
            success = task_def["check"](state, action, next_state)
        else:
            success = False

        # 5. Observe (update counts)
        explorer.observe(state, action, success)

        # 6. Record step
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

    return steps, final_state if "final_state" in dir() else state, None


# ── Main ─────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Phase 4 JEPA-PEDA Experiment"
    )
    parser.add_argument("--task", default="read_hello",
                        help="Task ID from MICRO_TASKS")
    parser.add_argument("--num-episodes", type=int, default=6)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--condition",
                        choices=["jepa_efe", "jepa_only", "pragmatic_only",
                                 "novelty_only", "all"],
                        default="all",
                        help="Exploration condition (default: all 4)")
    parser.add_argument("--output", default=None)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=None,
                        help="Override alpha for jepa_efe condition (e.g. 0.3, 0.5, 0.7)")
    parser.add_argument("--model-path", default=None,
                        help="Path to Qwen model")
    parser.add_argument("--docker-image", default=DOCKER_IMAGE_V2,
                        help="Docker image for sandbox")
    args = parser.parse_args()

    # Derive model path
    if args.model_path is None:
        args.model_path = os.path.expanduser("~/models/Qwen2.5-0.5B-Instruct")

    # Determine conditions to run
    if args.condition == "all":
        conditions = ["jepa_efe", "jepa_only", "pragmatic_only", "novelty_only"]
    else:
        conditions = [args.condition]

    # CWD rotation
    cwds = KNOWN_CWDS_V2 + UNKNOWN_CWDS_V2

    # Derive output path
    if args.output is None:
        output_dir = _PROJECT_ROOT / "results"
        output_dir.mkdir(parents=True, exist_ok=True)
        tag = args.condition if args.condition != "all" else "all_conds"
        args.output = str(output_dir / f"phase4_jepa_{tag}_{args.task}.jsonl")

    output_path = Path(args.output)

    print(f"{'='*60}")
    print(f"Phase 4 JEPA-PEDA Experiment")
    print(f"  Task: {args.task}")
    print(f"  Episodes per condition: {args.num_episodes}")
    print(f"  Max steps: {args.max_steps}")
    print(f"  Conditions: {conditions}")
    print(f"  Model: {args.model_path}")
    print(f"  Docker image: {args.docker_image}")
    print(f"  Output: {output_path}")
    print(f"{'='*60}", flush=True)

    # Open sandbox once
    sb = BusyboxSandbox(image=args.docker_image)

    # Load JEPA ensemble (shared across conditions that need it)
    peda = None
    jepa_conditions = {"jepa_efe", "jepa_only", "pragmatic_only"}
    use_jepa = any(c in jepa_conditions for c in conditions)
    if use_jepa:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"\nLoading JEPA ensemble on {device}...", flush=True)
        t0 = time.time()
        peda = JEPAPEDA(args.model_path, n_ensemble=3, device=device)
        print(f"  Loaded in {time.time()-t0:.1f}s (hidden_size={peda.jepa.hidden_size})", flush=True)

    for condition in conditions:
        print(f"{'─'*60}")
        print(f"Condition: {condition.upper()}")
        if condition in CONDITION_CONFIGS:
            alpha_val = CONDITION_CONFIGS[condition]['alpha']
            # Override displayed alpha if --alpha flag set
            if args.alpha is not None and condition == "jepa_efe":
                alpha_val = args.alpha
            print(f"  {CONDITION_CONFIGS[condition]['description']}")
            print(f"  alpha={alpha_val}")
        else:
            print(f"  Count-based novelty baseline")
        print(f"{'─'*60}", flush=True)

        all_results = []

        # Reset JEPA predictors between conditions to prevent cross-contamination
        if condition != "novelty_only" and peda is not None:
            peda.reset()
            print(f"  [Reset JEPA predictors and history]", flush=True)

        for ep_idx in range(args.num_episodes):
            cwd = cwds[ep_idx % len(cwds)]
            seed = args.seed_offset + ep_idx
            random.seed(seed)

            print(f"  [Episode {ep_idx+1}/{args.num_episodes}] cwd={cwd}", flush=True)
            t0 = time.time()

            try:
                if condition == "novelty_only":
                    explorer = NoveltyExplorer()
                    steps, final_state, train_loss = run_novelty_episode(
                        sb, explorer, args.task, args.max_steps, start_cwd=cwd,
                    )
                else:
                    # JEPA-based conditions: configure alpha
                    alpha = CONDITION_CONFIGS[condition]["alpha"]
                    # Override alpha for jepa_efe if --alpha flag set
                    if args.alpha is not None and condition == "jepa_efe":
                        alpha = args.alpha
                    peda.alpha = alpha
                    steps, final_state, train_loss = run_jepa_peda_episode(
                        sb, peda, args.task, args.max_steps, start_cwd=cwd,
                    )
            except Exception as e:
                print(f"    ERROR: {e}", flush=True)
                import traceback
                traceback.print_exc()
                steps = []
                final_state = type("obj", (object,), {
                    "victory": False, "cwd": cwd, "step_count": 0,
                })()
                train_loss = None

            elapsed = time.time() - t0

            if steps:
                metrics = compute_metrics(steps, args.task)
            else:
                metrics = {"fht": -1, "scr": 0.0, "dead_loop_rate": 0.0}

            result = {
                "condition": condition,
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
            line = {k: result[k] for k in [
                "condition", "task", "cwd", "episode", "steps_count",
                "success", "fht", "scr", "dead_loop_rate", "train_loss",
                "elapsed",
            ]}
            with open(output_path, "a") as f:
                f.write(json.dumps(line) + "\n")

        # Condition summary
        successes = sum(1 for r in all_results if r["success"])
        avg_scr = sum(r["scr"] for r in all_results) / max(len(all_results), 1)
        losses = [r.get("train_loss") for r in all_results
                  if r.get("train_loss") is not None]
        loss_report = (
            f"Avg loss: {sum(losses)/max(len(losses),1):.6f} (over {len(losses)} eps)"
            if losses else "No training (novelty_only)"
        )

        print(f"  [{condition}] Summary: {successes}/{args.num_episodes} success "
              f"({100*successes/max(args.num_episodes,1):.0f}%) | "
              f"Avg SCR: {avg_scr:.3f} | {loss_report}",
              flush=True)

    # Final cross-condition comparison
    print(f"\n{'='*60}")
    print(f"Cross-Condition Comparison for {args.task}")
    print(f"{'='*60}")

    condition_results = {c: [] for c in conditions}
    with open(output_path) as f:
        for line in f:
            try:
                rec = json.loads(line)
                if rec.get("condition") in condition_results:
                    condition_results[rec["condition"]].append(rec)
            except json.JSONDecodeError:
                continue

    for condition in conditions:
        recs = condition_results.get(condition, [])
        if not recs:
            continue
        sr = sum(1 for r in recs if r.get("success", False))
        hit_recs = [r for r in recs if r.get("success", False)]
        fhts = [r["fht"] for r in hit_recs if r.get("fht") is not None and r["fht"] >= 0]
        avg_fht = sum(fhts) / max(len(fhts), 1) if fhts else -1
        avg_scr = sum(r.get("scr", 0) for r in recs) / max(len(recs), 1)
        avg_dlr = sum(r.get("dead_loop_rate", 0) for r in recs) / max(len(recs), 1)
        losses = [r.get("train_loss") for r in recs if r.get("train_loss") is not None]
        avg_loss = sum(losses) / max(len(losses), 1) if losses else -1
        total_elapsed = sum(r.get("elapsed", 0) for r in recs)

        print(f"  {condition:>15s}: {sr}/{len(recs)} success | "
              f"FHT={avg_fht:.1f} | SCR={avg_scr:.3f} | "
              f"DLR={avg_dlr:.3f} | loss={avg_loss:.6f} | "
              f"{total_elapsed:.1f}s")

    print(f"\nOutput: {output_path}")
    print(f"{'='*60}", flush=True)

    sb.close()


# Lazy torch import
import torch

if __name__ == "__main__":
    main()
