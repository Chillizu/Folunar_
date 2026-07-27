"""Sweep drive weights quickly with real LLM adapter."""
import json
import random
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from phase1.drive_system import ActionGenerator, HomeostaticDriveSystem
from phase1.grid_env import GridWorld
from phase1.run import run_episode
from phase1.types import Action, DriveWeights
from phase1.world_model import EnsembleErrorComputer, LearningModule, WorldModel

MODEL_PATH = "/home/chillizu/models/Qwen2.5-0.5B-Instruct"
ADAPTER_PATH = "checkpoints/phase1/partial_adapter_real_25_e3"
MAX_STEPS = 10
EPISODES_PER_CONFIG = 2

# Configs to test: pragmatic-heavy and low-novelty variants
CONFIGS = [
    {"curiosity": 0.1, "competence": 2.0, "boredom": 0.1, "novelty": 0.1},
    {"curiosity": 0.1, "competence": 2.0, "boredom": 0.1, "novelty": 0.5},
    {"curiosity": 0.1, "competence": 2.0, "boredom": 0.1, "novelty": 1.0},
    {"curiosity": 0.1, "competence": 3.0, "boredom": 0.1, "novelty": 0.1},
    {"curiosity": 0.1, "competence": 3.0, "boredom": 0.1, "novelty": 0.5},
    {"curiosity": 0.1, "competence": 4.0, "boredom": 0.1, "novelty": 0.1},
    {"curiosity": 0.1, "competence": 2.0, "boredom": 0.5, "novelty": 0.1},
    {"curiosity": 0.1, "competence": 2.0, "boredom": 1.0, "novelty": 0.1},
]


def measure_revisits(trajectory):
    seen = set()
    revisits = 0
    visits = 0
    for state in trajectory:
        pos = state.agent
        if pos in seen:
            revisits += 1
        seen.add(pos)
        visits += 1
    return revisits / max(1, visits)


def run_config(weights, wm, seeds):
    env = GridWorld(width=5, height=5, max_steps=MAX_STEPS)
    ec = EnsembleErrorComputer(wm, num_checkpoints=5)
    ds = HomeostaticDriveSystem(DriveWeights(**weights))
    lm = LearningModule(wm, ec, buffer_size=1000, update_interval=500)
    ag = ActionGenerator(wm, ec, ds, horizon=2, max_candidates=2, latency_budget_ms=3000.0)

    results = []
    for seed in seeds:
        traj, preds, actions, metrics = run_episode(env, wm, ec, ds, lm, ag, seed=seed)
        results.append({
            "seed": seed,
            "success": metrics["success"],
            "steps": metrics["steps"],
            "revisit_rate": measure_revisits(traj),
            "actions": [a.name for a in actions],
        })
    return results


def main():
    print("Loading adapter...")
    wm = WorldModel(model_name=MODEL_PATH, use_stub=False, adapter_path=ADAPTER_PATH)
    seeds = [random.randint(0, 10000) for _ in range(EPISODES_PER_CONFIG)]
    print(f"Using seeds: {seeds}")

    summary = []
    start = time.time()
    for i, weights in enumerate(CONFIGS):
        cfg_start = time.time()
        print(f"\n[{i+1}/{len(CONFIGS)}] Testing weights: {weights}")
        results = run_config(weights, wm, seeds)
        successes = sum(r["success"] for r in results)
        mean_steps = sum(r["steps"] for r in results) / len(results)
        mean_revisit = sum(r["revisit_rate"] for r in results) / len(results)
        print(f"  successes: {successes}/{len(results)}, mean_steps: {mean_steps:.1f}, mean_revisit: {mean_revisit:.3f}")
        for r in results:
            print(f"    seed={r['seed']} success={r['success']} steps={r['steps']} revisit={r['revisit_rate']:.3f} actions={r['actions']}")
        summary.append({
            "weights": weights,
            "successes": successes,
            "mean_steps": mean_steps,
            "mean_revisit_rate": mean_revisit,
            "results": results,
            "elapsed_s": time.time() - cfg_start,
        })

    print(f"\nTotal elapsed: {time.time() - start:.1f}s")
    print("\n=== SUMMARY ===")
    for s in summary:
        w = s["weights"]
        print(f"cur={w['curiosity']:.1f} cmp={w['competence']:.1f} bor={w['boredom']:.1f} nov={w['novelty']:.1f} | "
              f"success={s['successes']}/{EPISODES_PER_CONFIG} steps={s['mean_steps']:.1f} revisit={s['mean_revisit_rate']:.3f} "
              f"elapsed={s['elapsed_s']:.1f}s")

    out_path = Path("results/phase1_weight_sweep.json")
    out_path.write_text(json.dumps({
        "model": MODEL_PATH,
        "adapter": ADAPTER_PATH,
        "max_steps": MAX_STEPS,
        "seeds": seeds,
        "configs": summary,
    }, indent=2, default=str))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
