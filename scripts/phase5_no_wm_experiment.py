#!/usr/bin/env python3
# ruff: noqa: E402
"""Phase 5 No-WM Experiment: Count-based exploration + learned action model.

Replaces the entire LLM-based World Model with two lightweight components:
  - NoveltyExplorer: count-based novelty for epistemic signal
  - ActionModelLearner: STRIPS-style action schema learning for planning

No neural networks. No GPU. No brittle token-space predictions.

Usage:
  # Full novelty explorer + action model
  python scripts/phase5_no_wm_experiment.py --task read_hello --num-episodes 12

  # Fallback: hand-coded candidates only (novelty explorer + generate_sandbox_candidates)
  python scripts/phase5_no_wm_experiment.py --task find_secret --candidates fallback

  # All tasks sequential
  for t in read_hello count_lines find_secret read_note; do
    python scripts/phase5_no_wm_experiment.py --task $t --num-episodes 12
  done
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

from phase2.sandbox_env import BusyboxSandbox, generate_sandbox_candidates
from phase2.tasks import MICRO_TASKS
from phase5.explorer import NoveltyExplorer
from phase5.action_model import ActionModelLearner



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
# v3: records, dataset, journal, modules, cache
KNOWN_CWDS = ["/sandbox", "/sandbox/docs", "/sandbox/data/raw", "/sandbox/logs/app", "/sandbox/projects/backend"]
UNKNOWN_CWDS = ["/sandbox/docs/tutorials", "/sandbox/data/processed", "/sandbox/data/archive",
                "/sandbox/logs/system", "/sandbox/logs/audit", "/sandbox/projects/frontend",
                "/sandbox/projects/shared", "/sandbox/cache/temp"]


def run_no_wm_episode(sb, explorer, action_model, task_id, max_steps,
                      start_cwd, candidates_mode="learned"):
    """Run a single episode with novelty-driven exploration + learned action model.

    Returns (steps, final_state).
    """
    state = sb.reset(start_cwd=start_cwd)
    action_history = []
    steps = []
    task_def = next((t for t in MICRO_TASKS if t["id"] == task_id), None)

    for step_i in range(max_steps):
        # 1. Generate candidates
        if candidates_mode == "learned":
            candidates = action_model.generate_candidates(state, task_id)
            # If learned model returns nothing, fall back to hand-coded
            if not candidates or len(candidates) < 2:
                candidates = generate_sandbox_candidates(state)
        else:
            candidates = generate_sandbox_candidates(state)

        # 2. Select action via novelty-driven exploration
        action = explorer.select_action(state, candidates, action_history)

        # 3. Execute
        next_state, reward, done = sb.step(state, action)

        # 4. Check success using task predicate
        if task_def:
            success = task_def["check"](state, action, next_state)
        else:
            success = False

        # 5. Learn from outcome
        action_model.learn_from_step(state, action, next_state, success)
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

    return steps, final_state if "final_state" in dir() else state


def main():
    parser = argparse.ArgumentParser(description="Phase 5 No-WM Sandbox Experiment")
    parser.add_argument("--task", default="read_hello",
                        help="Task ID from MICRO_TASKS")
    parser.add_argument("--num-episodes", type=int, default=12)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--condition", choices=["known", "unknown", "all"], default="all",
                        help="CWD condition: known cwds, unknown cwds, or all")
    parser.add_argument("--candidates", choices=["learned", "fallback"], default="learned",
                        help="learned=action model, fallback=hand-coded generate_sandbox_candidates")
    parser.add_argument("--output", default=None)
    parser.add_argument("--seed-offset", type=int, default=0)
    args = parser.parse_args()

    # Derive output path
    if args.output is None:
        output_dir = _PROJECT_ROOT / "results"
        output_dir.mkdir(parents=True, exist_ok=True)
        tag = args.candidates if args.candidates == "fallback" else "no_wm"
        args.output = str(output_dir / f"phase5_{tag}_{args.task}.jsonl")

    output_path = Path(args.output)

    # Select CWDs
    if args.condition == "known":
        cwds = KNOWN_CWDS
    elif args.condition == "unknown":
        cwds = UNKNOWN_CWDS
    else:
        cwds = KNOWN_CWDS + UNKNOWN_CWDS

    print(f"{'='*60}")
    print(f"Phase 5 No-WM Experiment")
    print(f"  Task: {args.task}")
    print(f"  Episodes: {args.num_episodes}")
    print(f"  Max steps: {args.max_steps}")
    print(f"  Condition: {args.condition} ({len(cwds)} cwds)")
    print(f"  Candidates: {args.candidates}")
    print(f"  Output: {output_path}")
    print(f"{'='*60}", flush=True)

    sb = BusyboxSandbox()
    explorer = NoveltyExplorer()
    action_model = ActionModelLearner()

    all_results = []
    for ep_idx in range(args.num_episodes):
        cwd = cwds[ep_idx % len(cwds)]
        seed = args.seed_offset + ep_idx
        random.seed(seed)

        # Reset episode-local explorer state (persistent across episodes by default)
        explorer.reset_episode()

        print(f"\n[Episode {ep_idx+1}/{args.num_episodes}] cwd={cwd}", flush=True)
        t0 = time.time()

        try:
            steps, final_state = run_no_wm_episode(
                sb, explorer, action_model, args.task, args.max_steps,
                start_cwd=cwd, candidates_mode=args.candidates,
            )
        except Exception as e:
            print(f"  ERROR: {e}", flush=True)
            steps = []
            final_state = type("obj", (object,), {"victory": False, "cwd": cwd, "step_count": 0})()

        elapsed = time.time() - t0

        # Compute metrics
        if steps:
            metrics = compute_metrics(steps, args.task)
        else:
            metrics = {"fht": -1, "scr": 0.0, "dead_loop_rate": 0.0}

        result = {
            "task": args.task,
            "condition": args.condition,
            "cwd": cwd,
            "episode": ep_idx,
            "steps_count": len(steps),
            "success": getattr(final_state, "victory", False),
            "fht": metrics.get("fht", -1),
            "scr": metrics.get("scr", 0.0),
            "dead_loop_rate": metrics.get("dead_loop_rate", 0.0),
            "candidates_mode": args.candidates,
            "elapsed": round(elapsed, 1),
            "records": steps,
        }
        print(f"  -> success={result['success']} fht={result['fht']} scr={result['scr']:.2f} "
              f"steps={result['steps_count']} [{elapsed:.0f}s]",
              flush=True)

        all_results.append(result)

        # Incremental write
        line = {k: result[k] for k in ["task", "condition", "cwd", "episode",
                                        "steps_count", "success", "fht", "scr",
                                        "dead_loop_rate", "candidates_mode", "elapsed"]}
        with open(output_path, "a") as f:
            f.write(json.dumps(line) + "\n")

    # Summary
    successes = sum(1 for r in all_results if r["success"])
    hits = [r for r in all_results if r["success"]]
    avg_scr = sum(r["scr"] for r in all_results) / max(len(all_results), 1)

    print(f"\n{'='*60}")
    print(f"Summary: Task={args.task} | Candidates={args.candidates}")
    print(f"  Episodes: {args.num_episodes}")
    print(f"  Successes: {successes}/{args.num_episodes} ({100*successes/max(args.num_episodes,1):.0f}%)")
    print(f"  Avg SCR: {avg_scr:.3f}")
    if hits:
        total_fht = sum(h["fht"] for h in hits if h["fht"] >= 0)
        total_hits = sum(1 for h in hits if h["fht"] >= 0)
        print(f"  Avg FHT: {(total_fht / max(total_hits, 1)):.1f}")
    else:
        print(f"  Avg FHT: - (no hits)")
    print(f"  Output: {output_path}")
    print(f"{'='*60}", flush=True)

    sb.close()


if __name__ == "__main__":
    main()
