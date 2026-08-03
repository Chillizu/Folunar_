#!/usr/bin/env python3
# ruff: noqa: E402
"""Phase 3 Epistemic Validation — sequential runner (all conditions, single model load).

Loads the World Model once, then runs all 4 condition×agent combinations
sequentially. This avoids the overhead and contention of 4 concurrent model loads.

Usage:
  source venv/bin/activate
  python scripts/phase3_run_all.py --num-episodes 10
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
from phase1.types import DriveWeights
from phase1.world_model import EnsembleErrorComputer, WorldModel
from phase2.sandbox_env import BusyboxSandbox, generate_sandbox_candidates

# ── Config ──────────────────────────────────────────────────────────
DRIVE_WEIGHTS = DriveWeights(curiosity=0.1, competence=2.0, boredom=0.1, novelty=2.0)
PRAGMATIC_WEIGHT = 3.0
ADAPTER_PATH = "checkpoints/phase2/sandbox_adapter_v2_partial"
MAX_CANDIDATES = 4  # Reduced from 8 for faster CPU inference
KNOWN_CWDS = ["/sandbox", "/sandbox/data", "/sandbox/docs", "/sandbox/logs"]
UNKNOWN_CWDS = ["/sandbox/projects/app", "/sandbox/projects/lib", "/tmp"]


def _goal_predicate_read_note(state, action, next_state) -> bool:
    return "secret key" in next_state.last_output or (action and "cat docs/note" in action)


def _build_ag(wm, pragmatic_only=False, ckpt_dir=None):
    ec = EnsembleErrorComputer(wm)
    if ckpt_dir is None:
        ckpt_dir = Path(ADAPTER_PATH)
    ec.checkpoints = sorted(Path(ckpt_dir).glob("checkpoint_epoch_*"))[:1]
    ds = HomeostaticDriveSystem(DRIVE_WEIGHTS)
    ag = ActionGenerator(wm, error_computer=ec, drive_system=ds,
                         pragmatic_only=pragmatic_only,
                         pragmatic_weight=PRAGMATIC_WEIGHT,
                         max_candidates=MAX_CANDIDATES, horizon=1,
                         goal_predicate=_goal_predicate_read_note)
    return ag


def run_episode(wm, pragmatic_only, start_cwd, max_steps=50):
    """Run one episode. Returns metrics dict."""
    ag = _build_ag(wm, pragmatic_only=pragmatic_only)
    sb = BusyboxSandbox()
    state = sb.reset(start_cwd=start_cwd)

    steps = []
    action_history = []
    task_completed = False

    for step_i in range(max_steps):
        if state.game_over:
            break

        if step_i > 0:
            print(".", end="", flush=True)

        t0 = time.time()
        cands = generate_sandbox_candidates(state)
        action = ag.select_action(state, action_history, cands)
        t1 = time.time()

        action_str = action if isinstance(action, str) else action.name
        next_state, reward, done = sb.step(state, action_str)
        t2 = time.time()

        if _goal_predicate_read_note(state, action_str, next_state):
            next_state.victory = True
            next_state.game_over = True
            task_completed = True
            done = True
            print("VICTORY!", end="", flush=True)

        record = {
            "step": step_i,
            "cwd": state.cwd, "action": action_str,
            "next_cwd": next_state.cwd, "exit_code": next_state.last_exit_code,
            "output": next_state.last_output[:100],
            "select_time": round(t1 - t0, 2), "exec_time": round(t2 - t1, 2),
        }
        steps.append(record)
        state = next_state
        action_history.append(action_str)

        if done:
            break

    sb.close()

    n_steps = len(steps)
    visited_cwds = set()
    revisit_count = 0
    for s in steps:
        if s["cwd"] in visited_cwds:
            revisit_count += 1
        visited_cwds.add(s["cwd"])
    revisit_rate = revisit_count / max(n_steps, 1)
    first_action = steps[0]["action"] if steps else ""

    return {
        "success": 1.0 if task_completed else 0.0,
        "steps": n_steps,
        "revisit_rate": round(revisit_rate, 4),
        "mean_epistemic_error": 0.0,
        "first_action": first_action,
    }


def run_condition(wm, agent_type, condition, num_episodes, max_steps, output_path):
    """Run N episodes for one agent×condition pair."""
    start_cwds = KNOWN_CWDS if condition == "goal_known" else UNKNOWN_CWDS
    pragmatic_only = (agent_type == "pragmatic")
    label = f"{agent_type}/{condition}"

    print(f"\n{'=' * 60}", flush=True)
    print(f"  {label}: {num_episodes} episodes", flush=True)
    print(f"{'=' * 60}", flush=True)

    results = []
    for ep in range(num_episodes):
        start_cwd = start_cwds[ep % len(start_cwds)]
        print(f"\n  [{ep+1}/{num_episodes}] cwd={start_cwd} ", end="", flush=True)
        t0 = time.time()

        ep_result = run_episode(wm, pragmatic_only, start_cwd, max_steps)

        elapsed = time.time() - t0
        ep_result["episode"] = ep
        ep_result["agent"] = agent_type
        ep_result["condition"] = condition
        ep_result["start_cwd"] = start_cwd
        ep_result["elapsed"] = round(elapsed, 1)
        results.append(ep_result)

        print(f"  success={ep_result['success']} steps={ep_result['steps']} "
              f"revisit={ep_result['revisit_rate']} [{elapsed:.0f}s]", flush=True)

        # Incremental save
        line = {k: ep_result[k] for k in ["episode", "agent", "condition", "start_cwd",
                                            "success", "steps", "revisit_rate",
                                            "mean_epistemic_error", "first_action", "elapsed"]}
        with open(output_path, "a") as f:
            f.write(json.dumps(line) + "\n")
            f.flush()
            os.fsync(f.fileno())

    # Summary
    successes = sum(r["success"] for r in results)
    mean_steps = sum(r["steps"] for r in results) / len(results)
    print(f"\n  -> {label}: {successes}/{num_episodes} success, {mean_steps:.1f} mean steps", flush=True)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--model", default=os.path.expanduser("~/models/Qwen2.5-0.5B-Instruct"))
    parser.add_argument("--adapter", default=ADAPTER_PATH)
    parser.add_argument("--output-dir", default="results/phase3_experiment")
    args = parser.parse_args()

    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model once
    print(f"\n[phase3] Loading WorldModel: {args.model} + adapter {args.adapter}", flush=True)
    wm = WorldModel(model_name=args.model, adapter_path=args.adapter)
    if wm.mode == "stub":
        raise RuntimeError("WorldModel fell back to stub mode.")

    total_start = time.time()

    # Run all 4 conditions sequentially
    conditions = ["goal_known", "goal_unknown"]
    agents = ["peda", "pragmatic"]

    all_results = {}
    for condition in conditions:
        for agent in agents:
            output_path = output_dir / f"{condition}_{agent}.jsonl"
            output_path.write_text("")

            results = run_condition(wm, agent, condition, args.num_episodes, args.max_steps, output_path)
            all_results[f"{condition}_{agent}"] = results

            # Per-episode details
            for r in results:
                print(f"    ep{r['episode']}: cwd={r['start_cwd']} "
                      f"{'SUCCESS' if r['success'] else 'FAIL'} "
                      f"steps={r['steps']} revisit={r['revisit_rate']} "
                      f"first={r['first_action']}", flush=True)

    total_elapsed = time.time() - total_start

    # Summary
    print(f"\n{'=' * 60}", flush=True)
    print("  PHASE 3 - FINAL SUMMARY", flush=True)
    print(f"{'=' * 60}", flush=True)
    for condition in conditions:
        print(f"\n  --- {condition} ---", flush=True)
        for agent in agents:
            key = f"{condition}_{agent}"
            results = all_results[key]
            n = len(results)
            successes = sum(r["success"] for r in results)
            mean_steps = sum(r["steps"] for r in results) / max(n, 1)
            print(f"    {agent:12s}: {successes}/{n} success ({100*successes/max(n,1):.0f}%)  "
                  f"{mean_steps:.1f} mean steps", flush=True)

    print(f"\n  Total time: {total_elapsed:.0f}s ({total_elapsed/60:.1f}min)", flush=True)
    print(f"  Results: {output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
