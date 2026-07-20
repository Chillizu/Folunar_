#!/usr/bin/env python3
"""Run a single Phase 1 episode on a held-out obstacle grid."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from phase1.drive_system import ActionGenerator, HomeostaticDriveSystem
from phase1.grid_env import GridWorld
from phase1.run import run_episode
from phase1.types import DriveWeights
from phase1.world_model import EnsembleErrorComputer, LearningModule, WorldModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/home/chillizu/models/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--max-candidates", type=int, default=4)
    parser.add_argument("--obstacles", default="[]", help="JSON list of [x,y] tuples")
    parser.add_argument("--weights", default='{"curiosity":0.5,"competence":0.5,"boredom":0.5,"novelty":0.5}')
    parser.add_argument("--pragmatic-only", action="store_true")
    parser.add_argument("--pragmatic-weight", type=float, default=3.0)
    parser.add_argument("--variant", default="unknown")
    args = parser.parse_args()

    obstacles = [tuple(p) for p in json.loads(args.obstacles)]
    weights = json.loads(args.weights)

    wm = WorldModel(model_name=args.model, use_stub=False, adapter_path=args.adapter)
    env = GridWorld(width=5, height=5, obstacles=obstacles, max_steps=args.max_steps)
    ec = EnsembleErrorComputer(wm, num_checkpoints=5)
    ds = HomeostaticDriveSystem(DriveWeights(**weights))
    lm = LearningModule(wm, ec, buffer_size=1000, update_interval=500)
    ag = ActionGenerator(
        wm,
        ec,
        ds,
        horizon=2,
        max_candidates=args.max_candidates,
        latency_budget_ms=3000.0,
        pragmatic_only=args.pragmatic_only,
        pragmatic_weight=args.pragmatic_weight,
    )

    traj, preds, actions, metrics = run_episode(env, wm, ec, ds, lm, ag, seed=args.seed)
    result = {
        "seed": args.seed,
        "variant": args.variant,
        "mode": "pragmatic_only" if args.pragmatic_only else "peda",
        "use_adapter": args.adapter is not None,
        "obstacles": obstacles,
        "success": metrics["success"],
        "steps": metrics["steps"],
        "reward": metrics["reward"],
        "trajectory": [s.agent for s in traj],
        "goal": traj[0].goal,
        "actions": [a.name for a in actions],
        "mean_epistemic_error": metrics["mean_epistemic_error"],
        "mean_aleatoric_error": metrics["mean_aleatoric_error"],
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main()
