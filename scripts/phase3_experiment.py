#!/usr/bin/env python3
# ruff: noqa: E402
"""Phase 3 Epistemic Validation: PEDA vs Pragmatic-only on sandbox v2.

Runs N episodes in a single condition (agent_type × goal_condition pair).
Outputs per-episode results as JSONL.

Usage:
  python scripts/phase3_experiment.py --agent peda --condition goal_known --num-episodes 10 --output results/phase3_experiment/goal_known_peda.jsonl
  python scripts/phase3_experiment.py --agent pragmatic --condition goal_known --num-episodes 10 --output results/phase3_experiment/goal_known_pragmatic.jsonl
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from phase1.drive_system import ActionGenerator, HomeostaticDriveSystem
from phase1.types import DriveWeights, Action
from phase1.world_model import EnsembleErrorComputer, LearningModule, WorldModel
from phase2.sandbox_env import BusyboxSandbox, generate_sandbox_candidates

# ── Config ──────────────────────────────────────────────────────────
DRIVE_WEIGHTS = DriveWeights(curiosity=0.1, competence=2.0, boredom=0.1, novelty=2.0)
PRAGMATIC_WEIGHT = 3.0
ADAPTER_PATH = "checkpoints/phase2/sandbox_adapter_v2_partial"

# Known cwds from training data
KNOWN_CWDS = ["/sandbox", "/sandbox/data", "/sandbox/docs", "/sandbox/logs"]

# Unknown cwds (not in training data)
UNKNOWN_CWDS = ["/sandbox/projects/app", "/sandbox/projects/lib", "/tmp"]

# Goal predicate for read_note task
def _goal_predicate_read_note(state, action, next_state) -> bool:
    return "secret key" in next_state.last_output or (action and "cat docs/note" in action)


def _build_ag(wm, pragmatic_only=False, ckpt_dir=None, goal_predicate=None):
    ec = EnsembleErrorComputer(wm)
    if ckpt_dir is None:
        ckpt_dir = Path(ADAPTER_PATH)
    ec.checkpoints = sorted(Path(ckpt_dir).glob("checkpoint_epoch_*"))[:3]
    ds = HomeostaticDriveSystem(DRIVE_WEIGHTS)
    ag = ActionGenerator(wm, error_computer=ec, drive_system=ds,
                         pragmatic_only=pragmatic_only,
                         pragmatic_weight=PRAGMATIC_WEIGHT,
                         max_candidates=8, horizon=1,
                         goal_predicate=goal_predicate)
    return ag


def run_episode(sb, wm, pragmatic_only, start_cwd, max_steps=50):
    """Run one episode. Returns (steps, final_state, metrics)."""
    ag = _build_ag(wm, pragmatic_only=pragmatic_only, goal_predicate=_goal_predicate_read_note)
    state = sb.reset(start_cwd=start_cwd)

    steps = []
    action_history = []
    task_completed = False

    for step_i in range(max_steps):
        if state.game_over:
            break

        t0 = time.time()
        cands = generate_sandbox_candidates(state)
        action = ag.select_action(state, action_history, cands)
        t1 = time.time()

        action_str = action if isinstance(action, str) else action.name
        next_state, reward, done = sb.step(state, action_str)
        t2 = time.time()

        # Check task completion
        if _goal_predicate_read_note(state, action_str, next_state):
            next_state.victory = True
            next_state.game_over = True
            task_completed = True
            done = True
            print(f"  [step {step_i}] VICTORY! action={action_str} [{t1-t0:.1f}s select, {t2-t1:.1f}s exec]", flush=True)

        record = {
            "agent_type": "pragmatic" if pragmatic_only else "peda",
            "step": step_i,
            "cwd": state.cwd,
            "files": list(state.files),
            "action": action_str,
            "next_cwd": next_state.cwd,
            "next_files": list(next_state.files),
            "exit_code": next_state.last_exit_code,
            "output": next_state.last_output[:100],
            "step_count": next_state.step_count,
            "selection_time": round(t1 - t0, 2),
            "exec_time": round(t2 - t1, 2),
        }
        steps.append(record)
        state = next_state
        action_history.append(action_str)

        if done:
            break

    # Compute metrics
    n_steps = len(steps)
    success = 1.0 if task_completed else 0.0

    # Revisit rate: count steps where cwd already visited
    visited_cwds = set()
    revisit_count = 0
    for s in steps:
        cwd = s.get("cwd", "")
        if cwd in visited_cwds:
            revisit_count += 1
        visited_cwds.add(cwd)
    revisit_rate = revisit_count / max(n_steps, 1)

    # Mean epistemic error approximation: check uncertainty in first action
    mean_epistemic = 0.0
    first_action = steps[0]["action"] if steps else ""

    return {
        "success": success,
        "steps": n_steps,
        "revisit_rate": revisit_rate,
        "mean_epistemic_error": mean_epistemic,
        "first_action": first_action,
        "records": steps,
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 3 Epistemic Validation single-condition runner")
    parser.add_argument("--agent", choices=["peda", "pragmatic"], required=True)
    parser.add_argument("--condition", choices=["goal_known", "goal_unknown"], required=True)
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--start-cwd", default=None, help="Override start cwd (random selection from known/unknown pool if unset)")
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--model", default=os.path.expanduser("~/models/Qwen2.5-0.5B-Instruct"), help="Model path")
    parser.add_argument("--adapter", default=ADAPTER_PATH, help="LoRA adapter path")
    parser.add_argument("--output", required=True, help="JSONL output path")
    args = parser.parse_args()

    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    # Determine start cwds
    if args.start_cwd:
        start_cwds = [args.start_cwd] * args.num_episodes
    elif args.condition == "goal_known":
        cwd_pool = KNOWN_CWDS
        start_cwds = [cwd_pool[i % len(cwd_pool)] for i in range(args.num_episodes)]
    else:
        cwd_pool = UNKNOWN_CWDS
        start_cwds = [cwd_pool[i % len(cwd_pool)] for i in range(args.num_episodes)]

    # Load world model
    print(f"[phase3] Loading WorldModel from {args.model} + adapter {args.adapter}", flush=True)
    wm = WorldModel(model_name=args.model, adapter_path=args.adapter)
    if wm.mode == "stub":
        raise RuntimeError("WorldModel fell back to stub mode — cannot run real-LLM experiment.")

    sb = BusyboxSandbox()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    for ep in range(args.num_episodes):
        start_cwd = start_cwds[ep]
        print(f"\n[phase3] Episode {ep+1}/{args.num_episodes} — {args.agent} / {args.condition} / cwd={start_cwd}", flush=True)
        t0 = time.time()
        ep_result = run_episode(sb, wm, pragmatic_only=(args.agent == "pragmatic"),
                                start_cwd=start_cwd, max_steps=args.max_steps)
        elapsed = time.time() - t0
        ep_result["episode"] = ep
        ep_result["agent"] = args.agent
        ep_result["condition"] = args.condition
        ep_result["start_cwd"] = start_cwd
        ep_result["elapsed"] = round(elapsed, 1)

        print(f"  -> success={ep_result['success']} steps={ep_result['steps']} revisit={ep_result['revisit_rate']:.3f} [{elapsed:.0f}s]", flush=True)
        results.append(ep_result)

        # Incremental save
        line = {k: ep_result[k] for k in ["episode", "agent", "condition", "start_cwd", "success", "steps", "revisit_rate", "mean_epistemic_error", "first_action", "elapsed"]}
        with open(output_path, "a") as f:
            f.write(json.dumps(line) + "\n")

    sb.close()

    # Summary
    successes = sum(r["success"] for r in results)
    mean_steps = sum(r["steps"] for r in results) / len(results)
    mean_revisit = sum(r["revisit_rate"] for r in results) / len(results)
    print(f"\n{'=' * 60}")
    print(f"Phase 3 — {args.agent} / {args.condition}")
    print(f"{'=' * 60}")
    print(f"Episodes: {len(results)}")
    print(f"Success rate: {successes}/{len(results)} ({100*successes/len(results):.1f}%)")
    print(f"Mean steps: {mean_steps:.1f}")
    print(f"Mean revisit rate: {mean_revisit:.3f}")
    print(f"Results saved to: {output_path}")

    sys.exit(0)


if __name__ == "__main__":
    main()
