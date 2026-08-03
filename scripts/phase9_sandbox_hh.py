#!/usr/bin/env python3
"""Phase 9 sandbox-hh (E1): two-layer open-loop agent — sandbox migration run.

Runs the frontier-goal layered agent over the 9 canonical Phase 8 tasks,
lambda in {0, 0.5} (45 episodes per arm, 90 total), and writes per-episode
JSONL artifacts (WATCHDOG D4: meta header with git commit + one line per
episode) to results/phase9_sbh_{lam0,lam05}.jsonl.

Usage:
    source venv/bin/activate
    PYTHONPATH=src python3 scripts/phase9_sandbox_hh.py            # both arms
    PYTHONPATH=src python3 scripts/phase9_sandbox_hh.py --lam 0    # one arm
    # FF-SBH-3 (R1 empty-dir re-selection): write to r1-prefixed files
    PYTHONPATH=src python3 scripts/phase9_sandbox_hh.py --out-prefix phase9_sbh_r1
"""
import argparse
import json
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from phase9.sandbox_hh.runner import SandboxHHRunner, TASKS  # noqa: E402

NUM_EPISODES = 5
MAX_STEPS = 10

DENSITY_DEF = (
    "unvisited_density(d) = |unvisited Phase-8 verb x file candidates at d| / "
    "|all verb x file candidates at d|; candidates = cat/head -n 5/wc -l per known "
    "text file + cd per known subdir; unvisited = zero count in explorer "
    "state_action_counts under state_hash(d). J(d) = density - lam*dist(cwd,d), "
    "dist = BFS cd-steps in the known dir graph; unreachable dirs excluded. "
    "Goal selected at episode start; re-selected when the local frontier at the "
    "current dir is exhausted, and (R1, FF-SBH-3) right after a cd into a dir "
    "with no readable text files; a textless cwd is never itself selected as "
    "goal (failure analysis T2/R1)."
)


def build_meta(lam: float, episodes: int) -> dict:
    commit = subprocess.run(["git", "rev-parse", "HEAD"],
                            capture_output=True, text=True).stdout.strip() or "unknown"
    return {
        "phase": 9,
        "experiment": "sandbox-hh",
        "lam": lam,
        "commit": commit,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "cpu_or_gpu": "cpu",
        "model": "count-based novelty (low) + frontier-goal density J (high), no learned model",
        "sandbox_images": {tid: img for tid, img in TASKS},
        "max_steps": MAX_STEPS,
        "episodes_per_task": NUM_EPISODES,
        "tasks": [tid for tid, _ in TASKS],
        "total_episodes": episodes,
        "density_definition": DENSITY_DEF,
        "low_layer": "generate_phase8_candidates + Phase8Explorer (byte-identical to Phase 8)",
        "r1_fix": "FF-SBH-3: after cd into a dir with no readable text files, force "
                  "high-layer goal re-selection next step; textless cwd excluded "
                  "from goal candidates (results/phase9_sbh_failure_analysis.md T2/R1)",
        "per_episode_data_present": True,
    }


def run_arm(lam: float, out_path: Path) -> dict:
    print(f"=== lambda={lam} arm ===", flush=True)
    runner = SandboxHHRunner(lam)
    per_task = runner.run_all(NUM_EPISODES, MAX_STEPS)

    rows = []
    summary = {}
    for task_id, _img in TASKS:
        eps = per_task[task_id]
        ok = sum(1 for e in eps if e["success"])
        summary[task_id] = f"{ok}/{len(eps)}"
        print(f"  {task_id:22s} {ok}/{len(eps)}", flush=True)
        rows.extend(eps)

    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        f.write(json.dumps({"meta": build_meta(lam, len(rows))}) + "\n")
        for r in rows:
            f.write(json.dumps(r) + "\n")
    pooled = sum(1 for e in rows if e["success"])
    print(f"  pooled: {pooled}/{len(rows)} -> {out_path}", flush=True)
    return {"lam": lam, "pooled": pooled, "total": len(rows), "per_task": summary}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lam", type=float, choices=[0.0, 0.5], default=None,
                        help="Run a single arm (default: both)")
    parser.add_argument("--out-prefix", default="phase9_sbh",
                        help="Output file prefix under results/ "
                             "(default: phase9_sbh -> phase9_sbh_lam0.jsonl)")
    args = parser.parse_args()

    arms = [0.0, 0.5] if args.lam is None else [args.lam]
    res = {}
    for lam in arms:
        name = "lam05" if lam == 0.5 else "lam0"
        res[name] = run_arm(lam, Path(f"results/{args.out_prefix}_{name}.jsonl"))
    print("\nSummary:", json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
