#!/usr/bin/env python3
"""Phase 8 quick-win evaluation harness (L1-QW, 2026-08-02).

Runs the count-driven agent over the 9 canonical Phase 8 tasks
(5 episodes x 9 tasks, max_steps=10) and writes per-episode JSONL
artifacts (WATCHDOG D4) plus a summary CSV.

Usage:
    PYTHONPATH=src python3 scripts/phase8_qw_eval.py --tag baseline
    PYTHONPATH=src python3 scripts/phase8_qw_eval.py --tag fixed

Per-episode JSONL: results/phase8_qw_<tag>/<task>.jsonl
Summary CSV:       results/phase8_qw_<tag>/summary.csv
"""
import argparse
import csv
import json
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from phase8.count_driven_agent import Phase8Runner  # noqa: E402

# Task -> docker image mapping (mirrors the 2026-07-31 GPU baseline run).
TASKS = [
    ("read_hello", "peda-sandbox:v2"),
    ("read_note", "peda-sandbox:v2"),
    ("count_lines", "peda-sandbox:v2"),
    ("find_secret", "peda-sandbox:v2"),
    ("read_welcome", "peda-sandbox:v4"),
    ("find_api_key", "peda-sandbox:v4"),
    ("count_measurements", "peda-sandbox:v4"),
    ("find_errors_v4", "peda-sandbox:v4"),
    ("read_changelog_v4", "peda-sandbox:v4"),
]

NUM_EPISODES = 5
MAX_STEPS = 10


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="baseline", help="output tag: results/phase8_qw_<tag>/")
    parser.add_argument("--episodes", type=int, default=NUM_EPISODES)
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    args = parser.parse_args()

    out_dir = _PROJECT_ROOT / "results" / f"phase8_qw_{args.tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    commit = (
        subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
        .stdout.strip()
        or "unknown"
    )
    meta = {
        "phase": 8,
        "tag": args.tag,
        "commit": commit,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "episodes_per_task": args.episodes,
        "max_steps": args.max_steps,
        "tasks": [t for t, _ in TASKS],
    }

    summary_rows = []
    for task_id, image in TASKS:
        print(f"\n=== {task_id} ({image}) ===", flush=True)
        runner = Phase8Runner(docker_image=image, task_id=task_id)
        runner.run(num_episodes=args.episodes, max_steps=args.max_steps)

        rows = [r.to_dict() for r in runner.results]
        success = sum(1 for r in rows if r["success"])
        avg_steps = sum(r["steps"] for r in rows) / max(len(rows), 1)

        task_path = out_dir / f"{task_id}.jsonl"
        with open(task_path, "w") as f:
            f.write(json.dumps({"meta": meta, "task": task_id, "image": image}) + "\n")
            for r in rows:
                f.write(json.dumps(r) + "\n")

        summary_rows.append(
            {
                "task": task_id,
                "image": image,
                "success": f"{success}/{len(rows)}",
                "success_rate": round(success / max(len(rows), 1), 3),
                "avg_steps": round(avg_steps, 2),
            }
        )
        print(f"  -> {success}/{len(rows)} success, avg_steps={avg_steps:.1f}")

    total_success = sum(int(r["success"].split("/")[0]) for r in summary_rows)
    total = len(TASKS) * args.episodes
    print(f"\nTOTAL: {total_success}/{total} ({total_success/total*100:.1f}%)")

    summary_path = out_dir / "summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    with open(out_dir / "meta.json", "w") as f:
        json.dump({**meta, "total_success": f"{total_success}/{total}"}, f, indent=2)

    print(f"\nArtifacts: {out_dir}/")
    return 0 if total_success > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
