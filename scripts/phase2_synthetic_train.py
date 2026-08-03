#!/usr/bin/env python3
# ruff: noqa: E402
"""Train a sandbox-specific LoRA adapter from collected (s,a,s') transitions.

Usage:
    python scripts/phase2_synthetic_train.py \
        --data results/phase2_train_merged.jsonl \
        --output-dir checkpoints/phase2/sandbox_adapter_e1
"""

import argparse
import json
import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from phase1.world_model import WorldModel


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _goal_predicate_read_note(state, action, next_state) -> bool:
    return "secret key" in next_state.last_output or (action and "cat docs/note" in action)

def _goal_predicate_count_lines(state, action, next_state) -> bool:
    return "3" in next_state.last_output and "lines" in next_state.last_output.lower()

def _goal_predicate_hello(state, action, next_state) -> bool:
    return "hello" in next_state.last_output.lower() and (action and "hello" in action)

def _goal_predicate_find_secret(state, action, next_state) -> bool:
    return "secret" in next_state.last_output.lower()

def _goal_predicate_create_file(state, action, next_state) -> bool:
    return "test_dir" in (next_state.files if hasattr(next_state, "files") else [])

MICRO_TASKS = [
    {"id": "read_note", "goal": "Read docs/note.txt", "check": _goal_predicate_read_note},
    {"id": "count_lines", "goal": "Count lines in data/lines.txt", "check": _goal_predicate_count_lines},
    {"id": "read_hello", "goal": "Read hello.txt", "check": _goal_predicate_hello},
    {"id": "find_secret", "goal": "Find files containing 'secret'", "check": _goal_predicate_find_secret},
    {"id": "create_file", "goal": "Create test_dir", "check": _goal_predicate_create_file},
]


def transitions_from_records(records: list[dict]) -> list[dict]:
    """Flatten per-baseline JSONL records into (s,a,s') training examples."""
    examples: list[dict] = []
    for rec in records:
        task_id = rec.get("task", "")
        task = next((t for t in MICRO_TASKS if t["id"] == task_id), None)
        for step in rec.get("records", []):
            if not step.get("action"):
                continue
            # Mark task completion with exit_code=2 so pragmatic reward can learn it.
            exit_code = step["exit_code"]
            if task is not None:
                fake_ns = type("obj", (object,), {
                    "last_output": step.get("output", ""),
                    "last_exit_code": step.get("exit_code", 0),
                    "files": step.get("next_files", []),
                    "cwd": step.get("next_cwd", ""),
                })()
                if task["check"](None, step["action"], fake_ns):
                    exit_code = 2
            examples.append({
                "state_text": json.dumps({
                    "cwd": step["cwd"],
                    "files": step["files"],
                    "last_command": "",
                    "last_exit_code": 0,
                    "last_output": "",
                }, ensure_ascii=False),
                "cwd": step["cwd"],
                "files": step["files"],
                "action_name": step["action"],
                "exit_code": exit_code,
                "summary": f"executed {step['action']}",
                "next_cwd": step["next_cwd"],
                "next_files": step["next_files"],
                "next_last_exit_code": exit_code,
                "next_last_output": step["output"],
            })
    return examples


def main():
    parser = argparse.ArgumentParser(description="Train sandbox LoRA adapter")
    parser.add_argument("--data", required=True, help="Path to merged JSONL training data")
    parser.add_argument("--model", default=os.path.expanduser("~/models/Qwen2.5-0.5B-Instruct"))
    parser.add_argument("--output-dir", default="checkpoints/phase2/sandbox_adapter_e1")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--stub", action="store_true", help="Smoke-test mode (no LLM)")
    parser.add_argument("--delta", action="store_true", help="Use delta prediction mode")
    args = parser.parse_args()

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"[phase2_train] ERROR: data file not found: {data_path}", flush=True)
        sys.exit(1)

    records = load_jsonl(data_path)
    data = transitions_from_records(records)
    print(f"[phase2_train] Loaded {len(data)} transitions from {len(records)} runs", flush=True)

    if not data:
        print("[phase2_train] ERROR: no training examples extracted", flush=True)
        sys.exit(1)

    examples_path = output_path / "training_examples.json"
    examples_path.write_text(json.dumps(data[:100], indent=2, ensure_ascii=False))
    print(f"[phase2_train] Saved sample training examples to {examples_path}", flush=True)

    manifest = {
        "num_runs": len(records),
        "num_transitions": len(data),
        "data_source": str(data_path),
    }
    (output_path / "trained_manifest.json").write_text(json.dumps(manifest, indent=2))

    use_stub = args.stub or os.environ.get("FOLUNAR_STUB_MODEL", "0") == "1"
    if use_stub:
        print("[phase2_train] STUB mode: placeholder checkpoints.", flush=True)
        (output_path / "stub_checkpoint.json").write_text(json.dumps({"mode": "stub"}))
        print(f"[phase2_train] STUB_DONE at {output_path}", flush=True)
        return

    print(f"[phase2_train] Loading model {args.model} ...", flush=True)
    wm = WorldModel(args.model)
    if wm.mode == "stub":
        print("[phase2_train] Model fell back to stub.", flush=True)
        sys.exit(1)
    print(f"[phase2_train] Training LoRA for {args.epochs} epoch(s) (sandbox mode)...", flush=True)
    wm.lora_finetune(
        data,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        checkpoint_dir=output_path,
        sandbox_mode=True,
        delta_mode=args.delta,
    )

    print(f"[phase2_train] Saving adapter to {output_path} ...", flush=True)
    wm.model.save_pretrained(str(output_path))
    print("[phase2_train] TRAINING_FINISHED", flush=True)


if __name__ == "__main__":
    main()
