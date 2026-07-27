"""Run a single Phase 1 episode and emit JSON summary to stdout."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from phase1.drive_system import ActionGenerator, HomeostaticDriveSystem
from phase1.grid_env import GridWorld
from phase1.run import run_episode
from phase1.types import DriveWeights, Action
from phase1.world_model import EnsembleErrorComputer, LearningModule, WorldModel


def state_to_dict(state):
    return {
        "x": state.agent[0],
        "y": state.agent[1],
        "goal": state.goal,
        "step": state.step,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--max-candidates", type=int, default=4)
    parser.add_argument("--weights", required=True)
    args = parser.parse_args()

    weights = json.loads(args.weights)
    wm = WorldModel(model_name=args.model, use_stub=False, adapter_path=args.adapter)
    env = GridWorld(width=5, height=5, max_steps=args.max_steps)
    ec = EnsembleErrorComputer(wm, num_checkpoints=5)
    ds = HomeostaticDriveSystem(DriveWeights(**weights))
    lm = LearningModule(wm, ec, buffer_size=1000, update_interval=500)
    ag = ActionGenerator(wm, ec, ds, horizon=2, max_candidates=args.max_candidates, latency_budget_ms=3000.0)

    traj, preds, actions, metrics = run_episode(env, wm, ec, ds, lm, ag, seed=args.seed)

    result = {
        "seed": args.seed,
        "success": metrics["success"],
        "steps": metrics["steps"],
        "reward": metrics["reward"],
        "trajectory_states": [state_to_dict(s) for s in traj],
        "action_history": [a.name for a in actions],
        "predictions": [
            {
                "level2_next_agent": p.level2_next_agent,
                "level1_exit_code": p.level1_exit_code,
                "level2_confidence": p.level2_confidence,
                "epistemic_ratio": p.epistemic_ratio,
            }
            for p in preds
        ],
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main()
