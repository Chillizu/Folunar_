"""Single-episode diagnostic for different drive weights with real LLM adapter."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from phase1.drive_system import ActionGenerator, HomeostaticDriveSystem
from phase1.grid_env import GridWorld
from phase1.run import run_episode
from phase1.types import DriveWeights
from phase1.world_model import EnsembleErrorComputer, LearningModule, WorldModel

MODEL_PATH = "/home/chillizu/models/Qwen2.5-0.5B-Instruct"
ADAPTER_PATH = "checkpoints/phase1/partial_adapter_real_25_e3"
SEED = 0
MAX_STEPS = 10

CONFIGS = [
    {"curiosity": 0.1, "competence": 2.0, "boredom": 0.1, "novelty": 0.1, "label": "low-novelty"},
    {"curiosity": 0.1, "competence": 2.0, "boredom": 0.1, "novelty": 2.0, "label": "default"},
    {"curiosity": 0.1, "competence": 5.0, "boredom": 0.1, "novelty": 0.1, "label": "high-competence"},
    {"curiosity": 0.1, "competence": 0.1, "boredom": 0.1, "novelty": 0.1, "label": "pragmatic-heavy"},
    {"curiosity": 0.0, "competence": 0.0, "boredom": 0.0, "novelty": 0.0, "label": "drives-off"},
]


def measure_revisits(trajectory):
    seen = set()
    revisits = 0
    for state in trajectory:
        pos = state.agent
        if pos in seen:
            revisits += 1
        seen.add(pos)
    return revisits / max(1, len(trajectory))


def run_single(weights, label, wm):
    print(f"\n[{label}] weights={weights}")
    env = GridWorld(width=5, height=5, max_steps=MAX_STEPS)
    ec = EnsembleErrorComputer(wm, num_checkpoints=5)
    ds = HomeostaticDriveSystem(DriveWeights(
        curiosity=weights["curiosity"],
        competence=weights["competence"],
        boredom=weights["boredom"],
        novelty=weights["novelty"],
    ))
    lm = LearningModule(wm, ec, buffer_size=1000, update_interval=500)
    ag = ActionGenerator(wm, ec, ds, horizon=2, max_candidates=2, latency_budget_ms=3000.0)

    start = time.time()
    traj, preds, actions, metrics = run_episode(env, wm, ec, ds, lm, ag, seed=SEED)
    elapsed = time.time() - start

    print(f"  seed={SEED} goal={traj[0].goal}")
    print(f"  trajectory: {[s.agent for s in traj]}")
    print(f"  actions: {[a.name for a in actions]}")
    print(f"  success={metrics['success']} steps={metrics['steps']} reward={metrics['reward']:.2f} revisit={measure_revisits(traj):.3f} elapsed={elapsed:.1f}s")
    return {
        "label": label,
        "weights": weights,
        "success": metrics["success"],
        "steps": metrics["steps"],
        "revisit_rate": measure_revisits(traj),
        "trajectory": [s.agent for s in traj],
        "actions": [a.name for a in actions],
        "elapsed_s": elapsed,
    }


def main():
    print("Loading adapter...")
    wm = WorldModel(model_name=MODEL_PATH, use_stub=False, adapter_path=ADAPTER_PATH)

    results = []
    for cfg in CONFIGS:
        weights = {k: v for k, v in cfg.items() if k != "label"}
        results.append(run_single(weights, cfg["label"], wm))

    out_path = Path("results/phase1_single_episode_weights.json")
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
