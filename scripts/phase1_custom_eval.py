"""Custom eval that reinitializes per-episode components to avoid state hang."""
import json
import random
import time
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from phase1.drive_system import ActionGenerator, HomeostaticDriveSystem
from phase1.grid_env import GridWorld
from phase1.run import (
    aggregate_metrics,
    completion_rate_at_horizon,
    random_baseline,
    revisit_rate,
    run_episode,
    steps_to_goal_ratio,
)
from phase1.types import DriveWeights
from phase1.world_model import EnsembleErrorComputer, LearningModule, WorldModel

MODEL_PATH = "/home/chillizu/models/Qwen2.5-0.5B-Instruct"
ADAPTER_PATH = "checkpoints/phase1/partial_adapter_real_25_e3"
EPISODES = 10
MAX_STEPS = 10
MAX_CANDIDATES = 4
LATENCY_BUDGET = 3000.0

WEIGHTS = {"curiosity": 0.1, "competence": 2.0, "boredom": 0.1, "novelty": 2.0}


def main():
    print("Loading adapter...")
    wm = WorldModel(model_name=MODEL_PATH, use_stub=False, adapter_path=ADAPTER_PATH)
    env = GridWorld(width=5, height=5, max_steps=MAX_STEPS)

    trajectories = []
    predictions_list = []
    action_histories = []

    start = time.time()
    for ep in range(EPISODES):
        print(f"\n[Episode {ep+1}/{EPISODES}]")
        # Reinitialize per-episode stateful components to avoid cross-episode hangs
        ec = EnsembleErrorComputer(wm, num_checkpoints=5)
        ds = HomeostaticDriveSystem(DriveWeights(**WEIGHTS))
        lm = LearningModule(wm, ec, buffer_size=1000, update_interval=500)
        ag = ActionGenerator(wm, ec, ds, horizon=2, max_candidates=MAX_CANDIDATES, latency_budget_ms=LATENCY_BUDGET)

        ep_start = time.time()
        traj, preds, actions, metrics = run_episode(env, wm, ec, ds, lm, ag, seed=ep)
        elapsed = time.time() - ep_start
        print(f"  steps={metrics['steps']} success={metrics['success']} reward={metrics['reward']:.2f} | elapsed={elapsed:.0f}s")
        print(f"  goal={traj[0].goal} trajectory={[s.agent for s in traj]} actions={[a.name for a in actions]}")
        trajectories.append(traj)
        predictions_list.append(preds)
        action_histories.append(actions)

    eval_elapsed = time.time() - start
    print(f"\nAll episodes complete in {eval_elapsed:.0f}s")

    drive_agg = aggregate_metrics(trajectories, predictions_list, action_histories)
    print("\nComputing random baseline...")
    baseline = random_baseline(env, n=EPISODES)

    g1 = drive_agg["next_state_accuracy"]
    g2 = steps_to_goal_ratio(drive_agg["mean_steps"], baseline["mean_steps"])
    g3 = drive_agg["revisit_rate"]

    print(f"\n--- Drive agent ({WEIGHTS}) ---")
    print(f"  Episodes: {EPISODES}")
    print(f"  Success rate: {drive_agg['success_rate']:.3f}")
    print(f"  Mean steps: {drive_agg['mean_steps']:.1f}")
    print(f"  Revisit rate: {drive_agg['revisit_rate']:.3f}")
    print(f"  Completion at 20: {drive_agg['completion_20']:.3f}")
    print(f"\n--- Random baseline ---")
    print(f"  Success rate: {baseline['success_rate']:.3f}")
    print(f"  Mean steps: {baseline['mean_steps']:.1f}")
    print(f"  Revisit rate: {baseline['mean_revisit_rate']:.3f}")
    print(f"\nG1={g1:.4f} (target >0.90) {'PASS' if g1 > 0.90 else 'FAIL'}")
    print(f"G2={g2:.4f} (target <0.50) {'PASS' if g2 < 0.50 else 'FAIL'}")
    print(f"G3={g3:.4f} (target <0.20) {'PASS' if g3 < 0.20 else 'FAIL'}")

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": {
            "stub_mode": False,
            "num_episodes": EPISODES,
            "best_weights": WEIGHTS,
            "model": MODEL_PATH,
            "adapter": ADAPTER_PATH,
            "max_candidates": MAX_CANDIDATES,
            "max_steps": MAX_STEPS,
        },
        "drive_agent": {
            "success_rate": drive_agg["success_rate"],
            "mean_steps": drive_agg["mean_steps"],
            "next_state_accuracy": g1,
            "revisit_rate": g3,
            "completion_20": drive_agg["completion_20"],
        },
        "random_baseline": baseline,
        "g1": {"value": g1, "threshold": 0.9, "pass": g1 > 0.9},
        "g2": {"value": g2, "threshold": 0.5, "pass": g2 < 0.5},
        "g3": {"value": g3, "threshold": 0.2, "pass": g3 < 0.2},
    }
    out_path = Path("results/phase1_eval_custom.json")
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nSaved report to {out_path}")


if __name__ == "__main__":
    main()
