#!/usr/bin/env python3
"""Phase 8: Count-Driven Closed-Loop Agent Experiment.

Assembles proven Phase 2 and Phase 5 components into a single runner.
Count-based novelty drives exploration. STRIPS schemas are learned.
JEPA training is optional (off by default).
"""
import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from phase8.count_driven_agent import Phase8Runner


def main():
    parser = argparse.ArgumentParser(description="Phase 8 Count-Driven Agent")
    parser.add_argument("--task", default="read_hello",
                        choices=[
                            "read_hello", "read_note", "count_lines",
                            "find_secret", "read_welcome", "find_api_key",
                            "count_measurements", "find_errors_v4",
                            "read_changelog_v4",
                        ])
    parser.add_argument("--docker-image", default="peda-sandbox:v2")
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--train-jepa", action="store_true")
    parser.add_argument("--model-path", default=None,
                        help="Path to Qwen model for JEPA (e.g. ~/models/Qwen2.5-0.5B-Instruct)")

    args = parser.parse_args()

    print("Phase 8: Count-Driven Closed-Loop Agent")
    print(f"  Task: {args.task}")
    print(f"  Docker image: {args.docker_image}")
    print(f"  Episodes: {args.num_episodes}")
    print(f"  Max steps per episode: {args.max_steps}")
    print(f"  JEPA training: {'ON' if args.train_jepa else 'OFF'}")
    print()

    runner = Phase8Runner(
        docker_image=args.docker_image,
        task_id=args.task,
        model_path=args.model_path,
        train_jepa=args.train_jepa,
    )

    results = runner.run(
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
    )

    success = sum(1 for r in results if r["success"])
    total = len(results)
    avg_steps = sum(r["steps"] for r in results) / max(total, 1)

    summary = {
        "phase": 8,
        "task": args.task,
        "docker_image": args.docker_image,
        "episodes": total,
        "success": success,
        "success_rate": f"{success}/{total} ({success/total*100:.0f}%)" if total > 0 else "0/0",
        "jepa_training": args.train_jepa,
        "avg_steps": round(avg_steps, 1),
    }

    print()
    print(json.dumps(summary, indent=2))

    return 0 if success > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
