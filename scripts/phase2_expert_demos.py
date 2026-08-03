#!/usr/bin/env python3
"""
Generate expert demonstration transitions for Phase 2 micro-tasks.
Each demo is a verified (s, a, s') transition from the Docker sandbox.

Output: JSONL with same record schema as phase2_collect_data.py.
"""

import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from phase2.sandbox_env import BusyboxSandbox
def run_demo_path(sb: BusyboxSandbox, task_id: str, actions: list[str]) -> list[dict]:
    """Execute a sequence of actions in the sandbox and return records."""
    state = sb.reset()
    records = []
    for step_i, action in enumerate(actions):
        next_state, _, _ = sb.step(state, action)
        record = {
            "agent_type": "expert_demo",
            "task_id": task_id,
            "step": step_i,
            "cwd": state.cwd,
            "files": sorted(state.files),
            "action": action,
            "next_cwd": next_state.cwd,
            "next_files": sorted(next_state.files),
            "exit_code": next_state.last_exit_code,
            "output": next_state.last_output[:200],
            "step_count": next_state.step_count,
        }
        records.append(record)
        state = next_state
    return records


# ═══════════════════════════════════════════════════════════════
# Expert demo paths: each is a list of (task_id, [actions])
# covering single-step, multi-step, and variant completions.
# ═══════════════════════════════════════════════════════════════

DEMO_PATHS = [
    # ── read_note ──
    ("read_note", ["cat docs/note.txt"]),
    ("read_note", ["ls", "cat docs/note.txt"]),
    ("read_note", ["cd docs", "cat note.txt"]),
    ("read_note", ["ls", "cd docs", "cat note.txt"]),
    ("read_note", ["ls data", "cat docs/note.txt"]),
    ("read_note", ["cat /sandbox/docs/note.txt"]),
    ("read_note", ["ls docs", "cat docs/note.txt"]),
    ("read_note", ["pwd", "cat docs/note.txt"]),

    # ── count_lines ──
    ("count_lines", ["wc -l data/lines.txt"]),
    ("count_lines", ["ls", "wc -l data/lines.txt"]),
    ("count_lines", ["cd data", "wc -l lines.txt"]),
    ("count_lines", ["ls data", "wc -l data/lines.txt"]),
    ("count_lines", ["cd data", "ls", "wc -l lines.txt"]),
    ("count_lines", ["ls", "ls data", "wc -l data/lines.txt"]),

    # ── read_hello ──
    ("read_hello", ["cat hello.txt"]),
    ("read_hello", ["ls", "cat hello.txt"]),
    ("read_hello", ["ls docs", "cat hello.txt"]),
    ("read_hello", ["pwd", "cat hello.txt"]),
    # ── find_secret: "secret" is in docs/note.txt, not data/ ──
    ("find_secret", ["grep -r secret ."]),
    ("find_secret", ["ls", "grep -r secret ."]),
    ("find_secret", ["grep secret docs/note.txt"]),
    ("find_secret", ["cd docs", "grep secret note.txt"]),
    ("find_secret", ["grep -r secret /sandbox"]),
    ("find_secret", ["ls docs", "grep -r secret docs"]),
    ("find_secret", ["grep -r secret docs"]),
    ("find_secret", ["ls", "grep -r secret docs"]),
    ("find_secret", ["cd docs", "ls", "grep secret note.txt"]),
    ("find_secret", ["pwd", "grep -r secret ."]),
]


def main():
    sb = BusyboxSandbox()
    all_records = []

    for task_id, actions in DEMO_PATHS:
        records = run_demo_path(sb, task_id, actions)
        all_records.extend(records)
        # Print progress
        last = records[-1]
        print(f"[demo] {task_id:15s} steps={len(records):2d} "
              f"final_action={last['action'][:30]:30s} exit={last['exit_code']}")

    # Write JSONL
    out_path = _PROJECT_ROOT / "results" / "phase2_expert_demos.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\n[demo] Wrote {len(all_records)} records to {out_path}")

    # Print stats
    by_task = {}
    for r in all_records:
        by_task.setdefault(r["task_id"], []).append(r)
    for tid, recs in sorted(by_task.items()):
        act_set = sorted(set(r["action"] for r in recs))
        print(f"  {tid}: {len(recs)} records, {len(act_set)} unique actions: {', '.join(act_set[:6])}")


if __name__ == "__main__":
    main()
