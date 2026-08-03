#!/usr/bin/env python3
"""Phase 8 quick-win ablation (L1-QW, 2026-08-02).

Isolates the contribution of each change by toggling the runner seams:

  matrix      = verb×file candidate matrix + wc promoted to reader tier
                (fix 1: count_lines blind spot)
  revisit     = cached-success child revisit in Phase8Explorer
                (fix 2: deep-path tasks never re-enter the answer dir)
  guard       = last-step budget guard (no new-dir cd on final step)

Configs:
  matrix_only : matrix, no revisit, no guard
  all         : matrix + revisit + guard  (== scripts/phase8_qw_eval.py --tag fixed2)

Usage:
    PYTHONPATH=src python3 scripts/phase8_qw_ablation.py
"""
import csv
import json
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from phase5.explorer import NoveltyExplorer  # noqa: E402
from phase8 import count_driven_agent as m  # noqa: E402

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


class MatrixOnlyExplorer(NoveltyExplorer):
    """Fix 1 explorer only: wc promoted to reader tier, no revisit logic."""
    _ACTION_PRIORITY = {**NoveltyExplorer._ACTION_PRIORITY, "wc": 0}


def run_config(tag: str, explorer_cls, budget_guard: bool) -> dict:
    out_dir = _PROJECT_ROOT / "results" / f"phase8_qw_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    commit = (
        subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
        .stdout.strip()
        or "unknown"
    )
    meta = {
        "phase": 8,
        "tag": tag,
        "commit": commit,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "explorer": explorer_cls.__name__,
        "budget_guard": budget_guard,
    }

    rows_by_task = {}
    for task_id, image in TASKS:
        m.Phase8Runner.EXPLORER_CLS = explorer_cls
        m.Phase8Runner.BUDGET_GUARD = budget_guard
        runner = m.Phase8Runner(docker_image=image, task_id=task_id)
        runner.run(num_episodes=5, max_steps=10)
        rows = [r.to_dict() for r in runner.results]
        rows_by_task[task_id] = rows
        task_path = out_dir / f"{task_id}.jsonl"
        with open(task_path, "w") as f:
            f.write(json.dumps({"meta": meta, "task": task_id, "image": image}) + "\n")
            for r in rows:
                f.write(json.dumps(r) + "\n")

    summary = []
    total_ok = 0
    for task_id, image in TASKS:
        rows = rows_by_task[task_id]
        ok = sum(1 for r in rows if r["success"])
        total_ok += ok
        summary.append({
            "task": task_id,
            "image": image,
            "success": f"{ok}/{len(rows)}",
            "avg_steps": round(sum(r["steps"] for r in rows) / len(rows), 2),
        })
    with open(out_dir / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    return {"tag": tag, "total": f"{total_ok}/{len(TASKS)*5}", "rows": summary}


if __name__ == "__main__":
    results = {}
    results["matrix_only"] = run_config("matrix_only", MatrixOnlyExplorer, False)
    results["all"] = run_config("all", m.Phase8Explorer, True)
    for tag, r in results.items():
        print(f"\n[{tag}] TOTAL {r['total']}")
        for row in r["rows"]:
            print(f"  {row['task']:20s} {row['success']:>5s}  avg_steps={row['avg_steps']}")
    print(json.dumps(results, indent=2))
