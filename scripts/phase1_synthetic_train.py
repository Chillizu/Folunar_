#!/usr/bin/env python3
# ruff: noqa: E402
"""Generate a synthetic 5x5 grid-world dataset and fine-tune the World Model LoRA adapter.

This bypasses LearningModule.update so that small synthetic datasets are not
lost by the prioritized buffer sampler / buffer clear.

Usage:
    python scripts/phase1_synthetic_train.py --model Qwen/Qwen2.5-0.5B-Instruct
    python scripts/phase1_synthetic_train.py --model Qwen/Qwen2.5-0.5B-Instruct --epochs 5 --batch-size 8
"""

import argparse
import datetime
import json
import os
import random
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
from phase1.grid_env import Action, GridWorld, Perception
from phase1.types import GridState
from phase1.world_model import WorldModel


def generate_synthetic_transitions(num_configs: int = 20, samples_per_config: int = None) -> list[dict]:
    """Generate deterministic (state, action, next_state) transitions matching eval.

    The eval environment uses a random goal and no obstacles, so this generator
    samples a random goal for each configuration and sweeps (or samples) agent
    positions from the free cells.
    """
    env = GridWorld()
    all_actions = [Action(name=n) for n in ("UP", "DOWN", "LEFT", "RIGHT")]
    data: list[dict] = []

    rng = random.Random(42)
    cells = [(x, y) for x in range(5) for y in range(5)]
    for _ in range(num_configs):
        # Match eval distribution: random goal, no obstacles.
        goal = rng.choice(cells)
        obstacles: list[tuple[int, int]] = []
        free_cells = [c for c in cells if c != goal]

        if samples_per_config is None:
            positions = free_cells
        else:
            positions = [rng.choice(free_cells) for _ in range(samples_per_config)]

        for pos in positions:
            state = GridState(
                agent=pos,
                goal=goal,
                obstacles=obstacles,
                width=5,
                height=5,
                step=0,
                max_steps=50,
            )
            for action in all_actions:
                next_state, _reward, _done = env.step(state, action)
                wall = next_state.agent == state.agent
                reached_goal = next_state.agent == state.goal
                exit_code = 2 if reached_goal else (1 if wall else 0)
                summary = f"agent moved {action.name.lower()}"
                if wall:
                    summary = f"agent hit wall/obstacle with {action.name.lower()}"
                elif reached_goal:
                    summary = "agent reached goal"
                data.append({
                    "state_text": Perception.render(state),
                    "action_name": action.name,
                    "next_state_text": str(next_state.agent),
                    "exit_code": exit_code,
                    "summary": summary,
                })
    return data


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Phase 1 World Model on synthetic grid data")
    parser.add_argument("--model", default=os.environ.get("FOLUNAR_MODEL", WorldModel.DEFAULT_MODEL))
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--num-configs", type=int, default=20, help="Number of random goal/position configurations to generate.")
    parser.add_argument("--samples-per-config", type=int, default=None)
    parser.add_argument("--output-adapter", default="checkpoints/phase1/synthetic_adapter")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    output_path = Path(args.output_adapter)
    done_marker = output_path / ".phase1_synthetic_train_done"
    failed_marker = output_path / ".phase1_synthetic_train_failed"

    def _clear_markers():
        for marker in (done_marker, failed_marker):
            if marker.exists():
                marker.unlink()

    try:
        _clear_markers()
        print(f"[synthetic_train] Generating transitions (configs={args.num_configs})...")
        data = generate_synthetic_transitions(
            num_configs=args.num_configs,
            samples_per_config=args.samples_per_config,
        )
        print(f"[synthetic_train] Generated {len(data)} synthetic transitions.")

        print(f"[synthetic_train] Loading model {args.model}...")
        wm = WorldModel(model_name=args.model, device=args.device, use_stub=False)
        if wm.mode == "stub" or wm.model is None:
            raise RuntimeError("WorldModel failed to load real LLM")

        print(f"[synthetic_train] Training LoRA adapter for {args.epochs} epoch(s)...")
        wm.lora_finetune(
            data,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
        )

        output_path.mkdir(parents=True, exist_ok=True)
        print(f"[synthetic_train] Saving adapter to {output_path}...")
        wm.model.save_pretrained(output_path)
        (output_path / "training_info.json").write_text(json.dumps({
            "model": args.model,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "num_configs": args.num_configs,
            "samples_per_config": args.samples_per_config,
            "transitions": len(data),
            "seed": args.seed,
            "success": True,
            "finished_at": datetime.datetime.now().isoformat(),
        }, indent=2))
        if failed_marker.exists():
            failed_marker.unlink()
        done_marker.write_text(datetime.datetime.now().isoformat())
        print(f"[synthetic_train] TRAINING_FINISHED: adapter saved to {output_path}")
    except Exception as exc:
        print(f"[synthetic_train] TRAINING_FAILED: {exc}")
        if done_marker.exists():
            done_marker.unlink()
        failed_marker.write_text(f"{datetime.datetime.now().isoformat()}: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
