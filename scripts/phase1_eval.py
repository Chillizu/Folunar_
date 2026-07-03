#!/usr/bin/env python3
# ruff: noqa: E402
"""Phase 1 final evaluation: run episodes with best drive weights and compute G1/G2/G3.

Usage:
    python scripts/phase1_eval.py                              # 100 episodes with real LLM
    python scripts/phase1_eval.py --stub                       # stub model, fast verification
    python scripts/phase1_eval.py --episodes 20                # override episode count
    python scripts/phase1_eval.py --drive-config results/phase1_grid_search.json
    FOLUNAR_STUB_MODEL=1 python scripts/phase1_eval.py

Reads best weights from config/phase1_default_drives.json by default (or from a
custom JSON file via --drive-config). Accepts either a legacy list of weight
objects or a grid-search report with a 'top_5' list.
Outputs full report to results/phase1_eval.json.
Exits with code 1 if G1 < 0.90 or G2 >= 0.50 or G3 >= 0.20.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Add src/ directory to path so phase1 modules are importable.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
from phase1.drive_system import ActionGenerator, HomeostaticDriveSystem
from phase1.grid_env import GridWorld
from phase1.run import (
    aggregate_metrics,
    random_baseline,
    run_episode,
    steps_to_goal_ratio,
)
from phase1.types import DriveWeights
from phase1.world_model import EnsembleErrorComputer, LearningModule, WorldModel


def main():
    parser = argparse.ArgumentParser(
        description="Phase 1 final evaluation with G1/G2/G3 go/no-go."
    )
    parser.add_argument("--stub", action="store_true", help="Use stub world model.")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="HuggingFace model name (default: Qwen/Qwen2.5-1.5B-Instruct).",
    )
    parser.add_argument(
        "--adapter",
        type=str,
        default=None,
        help="Path to a saved LoRA adapter to load on top of the model.",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=4,
        help="Number of candidate actions the ActionGenerator evaluates per step (default: 4).",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of evaluation episodes (default: 100).",
    )
    parser.add_argument(
        "--drive-config",
        type=str,
        default="config/phase1_default_drives.json",
        help="Path to drive weights (legacy list or grid-search report with top_5).",
    )
    args = parser.parse_args()

    use_stub = args.stub or os.environ.get("FOLUNAR_STUB_MODEL", "0") == "1"
    model_name_arg = args.model
    adapter_path_arg = args.adapter
    max_candidates_arg = args.max_candidates

    # If an adapter is given but no base model, read the base model from the
    # adapter's training metadata so the architecture matches.
    if adapter_path_arg and model_name_arg is None:
        info_path = Path(adapter_path_arg) / "training_info.json"
        if info_path.exists():
            model_name_arg = json.loads(info_path.read_text()).get("model")
            print(f"[eval] Inferred base model from adapter: {model_name_arg}")

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------
    # Load best drive weights
    # ----------------------------------------------------------------
    default_drives_path = Path(args.drive_config)
    if not default_drives_path.exists():
        print(f"ERROR: {default_drives_path} not found.")
        print("Run scripts/phase1_grid_search.py or use --drive-config to point to a grid-search report.")
        sys.exit(1)

    with open(default_drives_path) as f:
        raw_config = json.load(f)

    # Accept both a bare list of weight objects and a grid-search report with top_5.
    if isinstance(raw_config, dict):
        all_configs = raw_config.get("top_5", [])
    elif isinstance(raw_config, list):
        all_configs = raw_config
    else:
        print(f"ERROR: unexpected drive-config format in {default_drives_path}.")
        sys.exit(1)

    if not all_configs:
        print("ERROR: No drive configurations found in "
              f"{default_drives_path}.")
        sys.exit(1)

    best_weights = all_configs[0]
    if isinstance(best_weights, dict) and "weights" in best_weights:
        best_weights = best_weights["weights"]
    print("=" * 60)
    print("Phase 1 Final Evaluation")
    print("=" * 60)
    print(f"  Stub mode: {use_stub}")
    print(f"  Adapter:   {adapter_path_arg}")
    print(f"  Episodes:  {args.episodes}")
    print(f"  Best drive weights: cur={best_weights['curiosity']} "
          f"cmp={best_weights['competence']} "
          f"bor={best_weights['boredom']} "
          f"nov={best_weights['novelty']}")
    print()

    # ----------------------------------------------------------------
    # Environment and models
    # ----------------------------------------------------------------
    env = GridWorld(width=5, height=5, max_steps=50)
    wm = WorldModel(model_name=model_name_arg, use_stub=use_stub, adapter_path=adapter_path_arg)
    ec = EnsembleErrorComputer(wm, num_checkpoints=5)
    ds = HomeostaticDriveSystem(
        DriveWeights(
            curiosity=best_weights["curiosity"],
            competence=best_weights["competence"],
            boredom=best_weights["boredom"],
            novelty=best_weights["novelty"],
        )
    )
    lm = LearningModule(wm, ec, buffer_size=1000, update_interval=500)
    ag = ActionGenerator(wm, ec, ds, horizon=2, max_candidates=max_candidates_arg, latency_budget_ms=3000.0)

    # ----------------------------------------------------------------
    # Run evaluation episodes
    # ----------------------------------------------------------------
    print("Running evaluation episodes...")
    eval_start = time.time()

    trajectories = []
    predictions_list = []
    action_histories = []

    for ep in range(args.episodes):
        traj, preds, acts, ep_metrics = run_episode(
            env, wm, ec, ds, lm, ag, seed=ep
        )
        trajectories.append(traj)
        predictions_list.append(preds)
        action_histories.append(acts)

        if (ep + 1) % 25 == 0 or ep == 0 or ep == args.episodes - 1:
            elapsed = time.time() - eval_start
            print(
                f"  [{ep + 1}/{args.episodes}] "
                f"steps={ep_metrics['steps']} "
                f"success={ep_metrics['success']} "
                f"reward={ep_metrics['reward']:.2f} "
                f"| elapsed={elapsed:.0f}s"
            )

    eval_elapsed = time.time() - eval_start
    print(f"\nEvaluation complete in {eval_elapsed:.0f}s")

    # ----------------------------------------------------------------
    # Compute drive metrics via aggregate_metrics
    # ----------------------------------------------------------------
    drive_agg = aggregate_metrics(trajectories, predictions_list, action_histories)
    drive_success_rate = drive_agg["success_rate"]
    drive_mean_steps = drive_agg["mean_steps"]
    g1 = drive_agg["next_state_accuracy"]
    drive_revisit_rate = drive_agg["revisit_rate"]
    drive_completion_20 = drive_agg["completion_20"]

    # ----------------------------------------------------------------
    # Random baseline
    # ----------------------------------------------------------------
    print("\nComputing random baseline...")
    baseline = random_baseline(env, n=args.episodes)
    random_success_rate = baseline["success_rate"]
    random_mean_steps = baseline["mean_steps"]
    random_revisit_rate = baseline["mean_revisit_rate"]
    random_completion_5 = baseline["completion_5"]
    random_completion_10 = baseline["completion_10"]
    random_completion_20 = baseline["completion_20"]

    # ----------------------------------------------------------------
    # Compute G2 (steps-to-goal ratio) and G3 (revisit rate)
    # ----------------------------------------------------------------
    g2 = steps_to_goal_ratio(drive_mean_steps, random_mean_steps)
    g3 = drive_revisit_rate

    # ----------------------------------------------------------------
    # Report
    # ----------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("Results")
    print(f"{'=' * 60}")

    print(f"\n--- Drive agent (cur={best_weights['curiosity']} "
          f"cmp={best_weights['competence']} "
          f"bor={best_weights['boredom']} "
          f"nov={best_weights['novelty']}) ---")
    print(f"  Episodes:               {args.episodes}")
    print(f"  Success rate:           {drive_success_rate:.3f}")
    print(f"  Mean steps:             {drive_mean_steps:.1f}")
    print(f"  Completion at 20 steps: {drive_completion_20:.3f}")

    print("\n--- Random baseline ---")
    print(f"  Success rate:           {random_success_rate:.3f}")
    print(f"  Mean steps:             {random_mean_steps:.1f}")
    print(f"  Completion at 5 steps:  {random_completion_5:.3f}")
    print(f"  Completion at 10 steps: {random_completion_10:.3f}")
    print(f"  Completion at 20 steps: {random_completion_20:.3f}")

    print("\n--- G1: Next-state accuracy ---")
    print(f"  G1 = {g1:.4f}  (target > 0.90)  {'PASS' if g1 > 0.90 else 'FAIL'}")

    print("\n--- G2: Steps-to-goal ratio ---")
    print(f"  G2 = {g2:.4f}  (target < 0.50)  {'PASS' if g2 < 0.50 else 'FAIL'}")
    print(f"    Drive mean steps: {drive_mean_steps:.1f}")
    print(f"    Random mean steps: {random_mean_steps:.1f}")

    print("\n--- G3: Revisit rate ---")
    print(f"  G3 = {g3:.4f}  (target < 0.20)  {'PASS' if g3 < 0.20 else 'FAIL'}")

    # ----------------------------------------------------------------
    # Save full report
    # ----------------------------------------------------------------
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": {
            "stub_mode": use_stub,
            "num_episodes": args.episodes,
            "best_weights": best_weights,
            "model": getattr(wm, "model_name", "stub"),
        },
        "drive_agent": {
            "success_rate": round(drive_success_rate, 4),
            "mean_steps": round(drive_mean_steps, 2),
            "next_state_accuracy": round(g1, 4),
            "revisit_rate": round(drive_revisit_rate, 4),
            "completion_20": round(drive_completion_20, 4),
        },
        "random_baseline": {
            "success_rate": round(random_success_rate, 4),
            "mean_steps": round(random_mean_steps, 2),
            "mean_revisit_rate": round(random_revisit_rate, 4),
            "completion_5": round(random_completion_5, 4),
            "completion_10": round(random_completion_10, 4),
            "completion_20": round(random_completion_20, 4),
        },
        "g1": {"value": round(g1, 4), "threshold": 0.90, "pass": g1 > 0.90},
        "g2": {"value": round(g2, 4), "threshold": 0.50, "pass": g2 < 0.50},
        "g3": {"value": round(g3, 4), "threshold": 0.20, "pass": g3 < 0.20},
    }

    report_path = results_dir / "phase1_eval.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\nFull report saved to {report_path}")

    # ----------------------------------------------------------------
    # Go/no-go
    # ----------------------------------------------------------------
    g1_pass = g1 > 0.90
    g2_pass = g2 < 0.50
    g3_pass = g3 < 0.20

    print(f"\n{'=' * 60}")
    if g1_pass and g2_pass and g3_pass:
        print("RESULT: ALL GATES PASSED — Phase 1 validation successful.")
        print("Proceed to Phase 1.5/2.")
        sys.exit(0)
    else:
        print("STOP: Phase 1 failed.")
        if not g1_pass:
            print(f"  - G1 (next-state accuracy = {g1:.4f}) < 0.90")
        if not g2_pass:
            print(f"  - G2 (steps-to-goal ratio = {g2:.4f}) >= 0.50")
        if not g3_pass:
            print(f"  - G3 (revisit rate = {g3:.4f}) >= 0.20")
        sys.exit(1)


if __name__ == "__main__":
    main()
