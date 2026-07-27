"""Diagnostic: one episode with pragmatic_only=True to isolate the drive system."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from phase1.drive_system import ActionGenerator, HomeostaticDriveSystem
from phase1.grid_env import GridWorld
from phase1.run import run_episode
from phase1.types import DriveWeights
from phase1.world_model import EnsembleErrorComputer, LearningModule, WorldModel


def main() -> None:
    model_path = "/home/chillizu/models/Qwen2.5-0.5B-Instruct"
    adapter_path = "checkpoints/phase1/partial_adapter_real_25_e3"

    env = GridWorld(width=5, height=5, max_steps=10)
    wm = WorldModel(model_name=model_path, use_stub=False, adapter_path=adapter_path)
    ec = EnsembleErrorComputer(wm, num_checkpoints=5)
    ds = HomeostaticDriveSystem(DriveWeights(curiosity=0.1, competence=2.0, boredom=0.1, novelty=2.0))
    lm = LearningModule(wm, ec, buffer_size=1000, update_interval=500)
    ag = ActionGenerator(
        wm,
        ec,
        ds,
        horizon=2,
        max_candidates=4,
        latency_budget_ms=3000.0,
        pragmatic_only=True,
        pragmatic_weight=3.0,
    )

    trajectory, predictions, actions, metrics = run_episode(env, wm, ec, ds, lm, ag, seed=0)
    print("Trajectory positions:", [s.agent for s in trajectory])
    print("Actions:", [a.name for a in actions])
    print("Goal:", trajectory[0].goal)
    print("Metrics:", metrics)


if __name__ == "__main__":
    main()
