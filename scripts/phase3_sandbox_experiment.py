#!/usr/bin/env python3
# ruff: noqa: E402
"""Phase 3 Sandbox Experiment: Epistemic vs Pragmatic controlled test.

Conditions:
  - goal_known (known cwd from training) + goal_unknown (held-out cwd)
  - PEDA vs Pragmatic-only
  - Task: read_hello (cat hello.txt -> "Hello World")
  - Uses sandbox_adapter_v2_full adapter

Usage:
  python scripts/phase3_sandbox_experiment.py \
    --baseline peda --condition known --num-episodes 5
  python scripts/phase3_sandbox_experiment.py \
    --baseline pragmatic --condition unknown --num-episodes 5
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from phase1.world_model import WorldModel
from phase2.sandbox_env import BusyboxSandbox
from phase2.tasks import MICRO_TASKS

# Import the runner functions from phase2_collect_data
sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))
from phase2_collect_data import (
    DRIVE_WEIGHTS, PRAGMATIC_WEIGHT,
    run_peda, run_pragmatic, compute_metrics
)


# ── CWD Definitions ──────────────────────────────────────────────
# Known cwds (present in training data for sandbox_adapter_v2_full)
KNOWN_CWDS = ["/sandbox", "/sandbox/data", "/sandbox/docs"]

# Unknown/held-out cwds (NOT in training data)
UNKNOWN_CWDS = ["/sandbox/logs", "/sandbox/projects", "/sandbox/tmp"]


def main():
    parser = argparse.ArgumentParser(description="Phase 3 Sandbox Experiment")
    parser.add_argument("--baseline", choices=["peda", "pragmatic"], required=True)
    parser.add_argument("--condition", choices=["known", "unknown"], required=True)
    parser.add_argument("--num-episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--task", default="read_hello")
    parser.add_argument("--model", default=os.path.expanduser("~/models/Qwen2.5-0.5B-Instruct"))
    parser.add_argument("--adapter-path", default="checkpoints/phase2/sandbox_adapter_v2_full")
    parser.add_argument("--fast", action="store_true",
                        help="skip ensemble checkpoint loading for faster runs")
    parser.add_argument("--output", default=None)
    parser.add_argument("--seed-offset", type=int, default=0, help="offset for random seed")
    args = parser.parse_args()

    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    # Derive output path
    if args.output is None:
        output_dir = _PROJECT_ROOT / "results"
        output_dir.mkdir(parents=True, exist_ok=True)
        args.output = str(output_dir / f"phase3_sandbox_{args.baseline}_{args.condition}.jsonl")

    # Select CWDs based on condition
    cwds = KNOWN_CWDS if args.condition == "known" else UNKNOWN_CWDS
    output_path = Path(args.output)
    fm = "fast (no ensemble)" if args.fast else "full ensemble"

    print(f"Phase 3 Sandbox Experiment")
    print(f"  Baseline: {args.baseline}")
    print(f"  Condition: {args.condition} (cwds: {cwds})")
    print(f"  Task: {args.task}")
    print(f"  Episodes: {args.num_episodes}")
    print(f"  Max steps: {args.max_steps}")
    print(f"  Adapter: {args.adapter_path} ({fm})")
    print(f"  Output: {output_path}")
    print(f"  Seed offset: {args.seed_offset}")
    print("=" * 60, flush=True)

    # Load model
    print("Loading WorldModel...", flush=True)
    wm = WorldModel(args.model, adapter_path=args.adapter_path)
    if wm.mode == "stub":
        raise RuntimeError(f"WorldModel fell back to stub mode! Model at {args.model} did not load correctly.")
    print("WorldModel loaded.", flush=True)

    sb = BusyboxSandbox()

    all_results = []
    for ep_idx in range(args.num_episodes):
        # Cycle through CWDs
        cwd = cwds[ep_idx % len(cwds)]
        seed = args.seed_offset + ep_idx

        print(f"\n[Episode {ep_idx+1}/{args.num_episodes}] cwd={cwd} seed={seed}", flush=True)
        t0 = time.time()

        # Set random seed for reproducibility
        import random
        random.seed(seed)

        if args.baseline == "peda":
            steps, final_state = run_peda(
                sb, wm, args.max_steps, args.task,
                use_fast=args.fast, ckpt_dir=args.adapter_path,
                start_cwd=cwd
            )
        else:
            steps, final_state = run_pragmatic(
                sb, wm, args.max_steps, args.task,
                use_fast=args.fast, ckpt_dir=args.adapter_path,
                start_cwd=cwd
            )

        elapsed = time.time() - t0
        metrics = compute_metrics(steps, args.task)

        result = {
            "baseline": args.baseline,
            "condition": args.condition,
            "cwd": cwd,
            "episode": ep_idx,
            "task": args.task,
            "steps_count": len(steps),
            "success": metrics["scr"] > 0,
            "fht": metrics["fht"] if metrics.get("fht") is not None else -1,
            "scr": metrics["scr"],
            "dead_loop_rate": metrics["dead_loop_rate"],
            "elapsed": elapsed,
            "records": steps,
        }
        print(f"  -> success={result['success']} steps={result['steps_count']} scr={result['scr']:.2f} fht={result['fht']} [{elapsed:.0f}s]", flush=True)

        all_results.append(result)
        # Write incrementally
        line = {k: result[k] for k in ["baseline", "condition", "cwd", "episode", "task",
                                        "steps_count", "success", "fht", "scr", "dead_loop_rate", "elapsed"]}
        with open(output_path, "a") as f:
            f.write(json.dumps(line) + "\n")

    # Summary
    successes = sum(1 for r in all_results if r["success"])
    print(f"\n{'=' * 60}")
    print(f"Condition: {args.condition} | Baseline: {args.baseline}")
    print(f"  Episodes: {args.num_episodes}")
    print(f"  Successes: {successes}/{args.num_episodes} ({100*successes/args.num_episodes:.0f}%)")
    print(f"  Avg steps: {sum(r['steps_count'] for r in all_results)/len(all_results):.1f}")
    print(f"  Avg SCR: {sum(r['scr'] for r in all_results)/len(all_results):.3f}")
    print(f"  Results: {output_path}")
    print(f"{'=' * 60}", flush=True)

    sb.close()


if __name__ == "__main__":
    main()
