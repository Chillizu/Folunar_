#!/usr/bin/env python3
"""Create held-out test set from sandbox v2 OOD directories (logs/, projects/, README.txt)."""
import json
import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from phase2.sandbox_env import BusyboxSandbox

OUTPUT = _PROJECT_ROOT / "results" / "phase2_remaining" / "heldout_test_set.jsonl"
OUTPUT.parent.mkdir(parents=True, exist_ok=True)


def collect_transitions(sb):
    """Collect (s,a,s') transitions from OOD directories using targeted commands."""
    state = sb.reset()
    records = []

    # Each entry: (target_cwd, action, description)
    # All target OOD directories: logs/, projects/, projects/app, projects/lib
    ood_commands = [
        # ── README.txt at root (entirely new file type) ──
        ("/sandbox", "cat README.txt"),
        ("/sandbox", "head -n 1 README.txt"),
        ("/sandbox", "wc -l README.txt"),

        # ── logs/ directory ──
        ("/sandbox", "cd logs"),
        ("/sandbox/logs", "ls"),
        ("/sandbox/logs", "pwd"),
        ("/sandbox/logs", "cat access.log"),
        ("/sandbox/logs", "cat error.log"),
        ("/sandbox/logs", "head -n 2 access.log"),
        ("/sandbox/logs", "wc -l access.log"),
        ("/sandbox/logs", "wc -l error.log"),
        ("/sandbox/logs", "grep ERROR error.log"),
        ("/sandbox/logs", "grep -c ERROR error.log"),
        ("/sandbox/logs", "tail -n 1 access.log"),
        ("/sandbox/logs", "cd .."),

        # ── projects/ directory ──
        ("/sandbox", "cd projects"),
        ("/sandbox/projects", "ls"),
        ("/sandbox/projects", "pwd"),
        ("/sandbox/projects", "cd app"),
        ("/sandbox/projects/app", "ls"),
        ("/sandbox/projects/app", "pwd"),
        ("/sandbox/projects/app", "cat main.py"),
        ("/sandbox/projects/app", "cat test.py"),
        ("/sandbox/projects/app", "wc -l main.py"),
        ("/sandbox/projects/app", "cd .."),
        ("/sandbox/projects", "cd lib"),
        ("/sandbox/projects/lib", "ls"),
        ("/sandbox/projects/lib", "pwd"),
        ("/sandbox/projects/lib", "cat utils.py"),
        ("/sandbox/projects/lib", "wc -l utils.py"),
        ("/sandbox/projects/lib", "cd .."),
        ("/sandbox/projects", "cd .."),

        # ── Cross-directory OOD lookups from root ──
        ("/sandbox", "find . -name '*.log'"),
        ("/sandbox", "grep -r ERROR ."),
        ("/sandbox", "grep -r v2 ."),
        ("/sandbox", "grep -r admin ."),
        ("/sandbox", "find . -name '*.py'"),

        # ── Mutation in OOD dir ──
        ("/sandbox/projects", "touch new_feature.py"),
        ("/sandbox/projects", "ls"),
        ("/sandbox/projects", "echo 'testing' > new_feature.py"),
        ("/sandbox/projects", "cat new_feature.py"),
    ]

    for target_cwd, action_str in ood_commands:
        if len(records) >= 35:
            break

        # Navigate to the target cwd
        if state.cwd != target_cwd:
            if state.cwd != "/sandbox":
                state, _, _ = sb.step(state, "cd /sandbox")
            rel = target_cwd.replace("/sandbox", "", 1).lstrip("/")
            if rel:
                for p in rel.split("/"):
                    state, _, _ = sb.step(state, f"cd {p}")

        # Execute the OOD action
        try:
            next_state, _, _ = sb.step(state, action_str)
        except Exception as e:
            print(f"  Error '{action_str}' at {state.cwd}: {e}", flush=True)
            continue

        exit_code = next_state.last_exit_code
        output = next_state.last_output

        record = {
            "cwd": state.cwd,
            "files": sorted(state.files),
            "action": action_str,
            "next_cwd": next_state.cwd,
            "next_files": sorted(next_state.files),
            "exit_code": exit_code,
            "output": output,
        }
        records.append(record)
        state = next_state

        if len(records) % 5 == 0:
            print(f"  Collected {len(records)} transitions...", flush=True)

    return records


def main():
    print("[heldout] Creating sandbox v2...", flush=True)
    sb = BusyboxSandbox()

    print("[heldout] Collecting OOD transitions (logs/, projects/, README.txt)...", flush=True)
    records = collect_transitions(sb)
    sb.close()

    print(f"[heldout] Collected {len(records)} transitions", flush=True)

    # Save in wrapped format for measurement script (each line has "records" array)
    with open(OUTPUT, "w") as f:
        for rec in records:
            wrapped = {
                "task": "heldout_ood",
                "baseline": "heuristic",
                "records": [rec],
            }
            f.write(json.dumps(wrapped, ensure_ascii=False) + "\n")
    print(f"[heldout] Saved {len(records)} transitions to {OUTPUT}", flush=True)

    # Also save flat version
    flat_output = OUTPUT.with_suffix(".flat.jsonl")
    with open(flat_output, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[heldout] Also saved flat version to {flat_output}", flush=True)


if __name__ == "__main__":
    main()
