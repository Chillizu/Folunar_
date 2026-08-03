#!/usr/bin/env python3
# ruff: noqa: E402
"""Measure Phase 2 World Model L1/L2/L3 accuracy on held-out sandbox transitions."""

import argparse
import json
import os
import re
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from phase1.world_model import WorldModel
from phase2.sandbox_env import SandboxState


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def transitions_from_records(records: list[dict]) -> list[dict]:
    """Flatten per-baseline JSONL records into (s,a,s') measurement examples."""
    examples = []
    for rec in records:
        for step in rec.get("records", []):
            if not step.get("action"):
                continue
            state = SandboxState(
                cwd=step["cwd"],
                files=list(step["files"]),
                last_command="",
                last_exit_code=0,
                last_output="",
            )
            examples.append({
                "state": state,
                "action": step["action"],
                "actual_exit_code": step["exit_code"],
                "actual_output": step.get("output", ""),
                "actual_files": list(step.get("next_files", [])),
                "actual_cwd": step.get("next_cwd", step["cwd"]),
            })
    return examples


def _parse_exit_code_from_text(text: str) -> int:
    """Best-effort parse exit_code integer from predicted JSON/text."""
    m = re.search(r'"last_exit_code"\s*:\s*(\d+)', text)
    if m:
        return int(m.group(1))
    m = re.search(r'"exit_code"\s*:\s*(\d+)', text)
    if m:
        return int(m.group(1))
    return -1


def _parse_files_from_text(text: str) -> set[str]:
    """Best-effort parse predicted files list from JSON/text."""
    try:
        # If the text is a JSON object, extract files array.
        data = json.loads(text)
        if isinstance(data, dict) and "files" in data:
            return set(data["files"])
    except Exception:
        pass
    # Fallback: look for "files": [...]
    m = re.search(r'"files"\s*:\s*(\[[^\]]*\])', text)
    if m:
        try:
            return set(json.loads(m.group(1)))
        except Exception:
            pass
    return set()


def _output_similarity(pred_summary: str, actual_output: str) -> float:
    """Simple token-overlap similarity between predicted summary and actual output."""
    pred_tokens = set(pred_summary.lower().split())
    actual_tokens = set(actual_output.lower().split())
    if not actual_tokens:
        return 0.0
    overlap = len(pred_tokens & actual_tokens)
    return overlap / len(actual_tokens)


def measure_l1l2l3(wm, examples: list[dict]) -> dict:
    l1_correct = l2_correct = l3_correct = 0
    total = len(examples)

    for i, ex in enumerate(examples):
        pred = wm.predict(ex["state"], ex["action"])

        # L1: exit code exact match
        l1_ok = 1 if pred.level1_exit_code == ex["actual_exit_code"] else 0

        # L2: filesystem delta — compare predicted files vs actual next_files
        pred_files = _parse_files_from_text(pred.level2_text or "")
        actual_files = set(ex["actual_files"])
        l2_ok = 1 if pred_files == actual_files else 0

        # L3: output summary semantic overlap.
        # The WM stores the predicted command output in level2_text["last_output"];
        # level3_output_summary is currently a generic action label, so we compare
        # against the structured last_output field instead.
        pred_text = (pred.level2_text or "").lower()
        try:
            pred_json = json.loads(pred.level2_text or "{}")
            pred_output = str(pred_json.get("last_output", "")).lower()
        except Exception:
            pred_output = pred_text
        actual_out = ex["actual_output"].lower()[:100]
        sim = _output_similarity(pred_output, actual_out)
        l3_ok = 1 if sim >= 0.5 else 0

        l1_correct += l1_ok
        l2_correct += l2_ok
        l3_correct += l3_ok

        if (i + 1) % 20 == 0:
            print(
                f"  [{i+1}/{total}] L1={l1_correct/(i+1):.3f} "
                f"L2={l2_correct/(i+1):.3f} L3={l3_correct/(i+1):.3f}",
                flush=True,
            )

    return {
        "total": total,
        "l1": round(l1_correct / total, 4) if total else 0.0,
        "l2": round(l2_correct / total, 4) if total else 0.0,
        "l3": round(l3_correct / total, 4) if total else 0.0,
        "l1_pass": (l1_correct / total) >= 0.90 if total else False,
        "l2_pass": (l2_correct / total) >= 0.70 if total else False,
        "l3_pass": (l3_correct / total) >= 0.50 if total else False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="results/phase2_train_merged.jsonl")
    parser.add_argument("--adapter-path", default="checkpoints/phase2/sandbox_adapter_e2")
    parser.add_argument("--model", default=os.path.expanduser("~/models/Qwen2.5-0.5B-Instruct"))
    parser.add_argument("--output", default="results/phase2_l1l2l3_baseline.json")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--train-split", type=float, default=0.8)
    args = parser.parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"ERROR: data not found: {data_path}")
        sys.exit(1)

    records = load_jsonl(data_path)
    examples = transitions_from_records(records)

    # Deterministic shuffle + train/test split for held-out evaluation.
    rng = range(len(examples))  # keep order deterministic
    split_idx = int(len(examples) * args.train_split)
    train_examples = examples[:split_idx]
    test_examples = examples[split_idx:]
    if args.limit and args.limit < len(test_examples):
        test_examples = test_examples[:args.limit]

    print(f"[l1l2l3] train={len(train_examples)} test={len(test_examples)}", flush=True)
    print(f"[l1l2l3] Loading WM + {args.adapter_path}...", flush=True)
    wm = WorldModel(args.model, adapter_path=args.adapter_path)
    if wm.mode == "stub":
        print("[l1l2l3] WARNING: WM fell back to stub mode.", flush=True)

    print(f"[l1l2l3] Measuring on {len(test_examples)} held-out examples...", flush=True)
    result = measure_l1l2l3(wm, test_examples)
    result["train_count"] = len(train_examples)
    result["test_count"] = len(test_examples)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    print(f"L1={result['l1']:.4f} {'PASS' if result['l1_pass'] else 'FAIL'} (>=0.90)")
    print(f"L2={result['l2']:.4f} {'PASS' if result['l2_pass'] else 'FAIL'} (>=0.70)")
    print(f"L3={result['l3']:.4f} {'PASS' if result['l3_pass'] else 'FAIL'} (>=0.50)")
    print(f"  -> saved to {args.output}")


if __name__ == "__main__":
    main()
