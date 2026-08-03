#!/usr/bin/env python3
"""Inject golden-path training examples into sandbox training data.

Constructs 15 (state, action) -> next_state examples where the action is the
correct task-completion command from the right starting state, marked with
exit_code=2 (goal completion).

Rules:
- Use REAL sandbox v2 state (files, cwd from Dockerfile.busybox_v2)
- Do NOT fabricate file contents (output field generic or empty)
- Only mark exit_code=2 for terminal goal-completion steps
- Include 3-4 intermediate navigation steps per task
- Total: ~15 examples (3-4 per task × 4 tasks)
"""

import json
from pathlib import Path

# ── Sandbox v2 ground truth (from Dockerfile.busybox_v2) ──────────

ROOT_FILES = ["README.txt", "hello.txt", "data", "docs", "logs", "projects", "tmp"]
DATA_FILES = ["config.ini", "lines.txt", "numbers.txt", "users.csv"]
DOCS_FILES = ["changelog.txt", "manual.txt", "note.txt"]
LOGS_FILES = ["access.log", "error.log"]

# ── Build golden examples ─────────────────────────────────────────

def mk_example(task_id, cwd, files, action, next_cwd, next_files, exit_code,
               last_output="", last_exit_code=0):
    """Build a training example dict compatible with transitions_from_records output."""
    return {
        "state_text": json.dumps({
            "cwd": cwd,
            "files": files,
            "last_command": "",
            "last_exit_code": last_exit_code,
            "last_output": last_output,
        }, ensure_ascii=False),
        "cwd": cwd,
        "files": files,
        "action_name": action,
        "exit_code": exit_code,
        "summary": f"executed {action}" + (" [GOAL]" if exit_code == 2 else ""),
        "next_cwd": next_cwd,
        "next_files": next_files,
        "next_last_exit_code": exit_code,
        "next_last_output": last_output,
    }


GOLDEN_EXAMPLES = []

# ── Task: read_hello ──────────────────────────────────────────────
# Path: /sandbox -> cat hello.txt (1 step)
GOLDEN_EXAMPLES.append(mk_example(
    "read_hello", "/sandbox", ROOT_FILES,
    "cat hello.txt", "/sandbox", ROOT_FILES,
    exit_code=2,
))

# ── Task: count_lines ─────────────────────────────────────────────
# Path: /sandbox -> wc -l data/lines.txt (1 step)
GOLDEN_EXAMPLES.append(mk_example(
    "count_lines", "/sandbox", ROOT_FILES,
    "wc -l data/lines.txt", "/sandbox", ROOT_FILES,
    exit_code=2,
))

# ── Task: read_note ───────────────────────────────────────────────
# Path: /sandbox -> cat docs/note.txt (1 step)
GOLDEN_EXAMPLES.append(mk_example(
    "read_note", "/sandbox", ROOT_FILES,
    "cat docs/note.txt", "/sandbox", ROOT_FILES,
    exit_code=2,
))

# ── Task: find_secret ─────────────────────────────────────────────
# Path: /sandbox -> grep -r secret . (1 step)
GOLDEN_EXAMPLES.append(mk_example(
    "find_secret", "/sandbox", ROOT_FILES,
    "grep -r secret .", "/sandbox", ROOT_FILES,
    exit_code=2,
))

# ── Intermediate steps (exit_code=0) ──────────────────────────────
# ls from root
GOLDEN_EXAMPLES.append(mk_example(
    "read_hello", "/sandbox", ROOT_FILES,
    "ls", "/sandbox", ROOT_FILES,
    exit_code=0,
))

# cd into docs/
GOLDEN_EXAMPLES.append(mk_example(
    "read_note", "/sandbox", ROOT_FILES,
    "cd docs", "/sandbox/docs", DOCS_FILES,
    exit_code=0,
))

# ls docs/
GOLDEN_EXAMPLES.append(mk_example(
    "read_note", "/sandbox/docs", DOCS_FILES,
    "ls", "/sandbox/docs", DOCS_FILES,
    exit_code=0,
))

# cat note.txt from docs/
GOLDEN_EXAMPLES.append(mk_example(
    "read_note", "/sandbox/docs", DOCS_FILES,
    "cat note.txt", "/sandbox/docs", DOCS_FILES,
    exit_code=2,
))

# cd data/
GOLDEN_EXAMPLES.append(mk_example(
    "count_lines", "/sandbox", ROOT_FILES,
    "cd data", "/sandbox/data", DATA_FILES,
    exit_code=0,
))

# ls data/
GOLDEN_EXAMPLES.append(mk_example(
    "count_lines", "/sandbox/data", DATA_FILES,
    "ls", "/sandbox/data", DATA_FILES,
    exit_code=0,
))

# wc -l from data/
GOLDEN_EXAMPLES.append(mk_example(
    "count_lines", "/sandbox/data", DATA_FILES,
    "wc -l lines.txt", "/sandbox/data", DATA_FILES,
    exit_code=2,
))

# cd .. from subdirectory
GOLDEN_EXAMPLES.append(mk_example(
    "read_hello", "/sandbox/data", DATA_FILES,
    "cd ..", "/sandbox", ROOT_FILES,
    exit_code=0,
))

# cd .. from docs/
GOLDEN_EXAMPLES.append(mk_example(
    "read_note", "/sandbox/docs", DOCS_FILES,
    "cd ..", "/sandbox", ROOT_FILES,
    exit_code=0,
))

# ── Save ──────────────────────────────────────────────────────────

out_dir = Path("results/phase5_train_data")
out_dir.mkdir(parents=True, exist_ok=True)

# Save as individual lines (compatible with load_jsonl)
out_path = out_dir / "golden_paths.jsonl"
with open(out_path, "w") as f:
    for ex in GOLDEN_EXAMPLES:
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")

print(f"Generated {len(GOLDEN_EXAMPLES)} golden-path examples -> {out_path}")
print(f"  Goal (exit_code=2): {sum(1 for e in GOLDEN_EXAMPLES if e['exit_code'] == 2)}")
print(f"  Intermediate (exit_code=0): {sum(1 for e in GOLDEN_EXAMPLES if e['exit_code'] == 0)}")
print(f"  Total ratio vs 1578: {len(GOLDEN_EXAMPLES) / 1578 * 100:.1f}%")
