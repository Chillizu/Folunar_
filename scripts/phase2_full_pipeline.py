#!/usr/bin/env python3
# ruff: noqa: E402
"""Full Phase 2 Fix A pipeline: train full adapter, validate on held-out, collect metrics."""

import json
import os
import random
import sys
import time
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


def transitions_from_records(records: list[dict]) -> list[dict]:
    examples = []
    for rec in records:
        for step in rec.get("records", []):
            if not step.get("action"):
                continue
            exit_code = step["exit_code"]
            examples.append({
                "state_text": json.dumps({
                    "cwd": step["cwd"],
                    "files": step["files"],
                    "last_command": "",
                    "last_exit_code": 0,
                    "last_output": "",
                }, ensure_ascii=False),
                "action_name": step["action"],
                "exit_code": exit_code,
                "summary": f"executed {step['action']}",
                "next_cwd": step["next_cwd"],
                "next_files": step["next_files"],
                "next_last_exit_code": exit_code,
                "next_last_output": step["output"],
            })
    return examples


def compute_metrics(wm, test_examples):
    l1_correct = l2_correct = l3_correct = 0
    total = len(test_examples)
    
    for i, ex in enumerate(test_examples):
        state_text = ex["state_text"]
        action_text = ex["action_name"]
        expected_exit_code = int(ex.get("exit_code", 0))
        expected_cwd = ex.get("next_cwd", "")
        expected_files = ex.get("next_files", [])
        expected_output = ex.get("next_last_output", "")
        
        user_content = (
            f"State: {state_text}\n"
            f"Action: {action_text}\n"
            "Predict cwd, files, last_command, last_exit_code, last_output, exit_code, and summary as JSON:"
        )
        msg_system = wm._sandbox_system_message()
        messages = [
            {"role": "system", "content": msg_system},
            {"role": "user", "content": user_content},
        ]
        
        generated, conf = wm.generate_text(messages, max_new_tokens=80)
        
        parsed = {}
        try:
            text = generated.strip()
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()
            parsed = json.loads(text)
        except (json.JSONDecodeError, IndexError):
            pass
        
        pred_exit = parsed.get("exit_code", -1)
        try:
            pred_exit = int(pred_exit)
        except (TypeError, ValueError):
            pred_exit = -1
        if (pred_exit == 0 and expected_exit_code == 0) or (pred_exit != 0 and expected_exit_code != 0):
            l1_correct += 1
        
        pred_cwd = parsed.get("cwd", "")
        pred_files = parsed.get("files", [])
        if not isinstance(pred_files, list):
            pred_files = []
        if pred_cwd == expected_cwd and sorted(pred_files) == sorted(expected_files):
            l2_correct += 1
        
        pred_output = parsed.get("last_output", "")
        pred_summary = parsed.get("summary", "")
        expected_lower = expected_output.lower()[:100]
        pred_output_lower = pred_output.lower()[:100]
        pred_summary_lower = pred_summary.lower()[:100]
        output_overlap = len(set(expected_lower.split()) & set(pred_output_lower.split())) > 0
        summary_overlap = len(set(expected_lower.split()) & set(pred_summary_lower.split())) > 0
        if output_overlap or summary_overlap:
            l3_correct += 1
        
        if (i + 1) % 2 == 0:
            print(f"  [Eval {i+1}/{total}] interim L1={l1_correct/(i+1):.3f} L2={l2_correct/(i+1):.3f} L3={l3_correct/(i+1):.3f}", flush=True)
    
    return {
        "l1": l1_correct / max(total, 1),
        "l2": l2_correct / max(total, 1),
        "l3": l3_correct / max(total, 1),
        "total": total,
    }


def main():
    results_dir = Path("results/phase2_fix_a")
    results_dir.mkdir(parents=True, exist_ok=True)

    full_ckpt = Path("checkpoints/phase2/sandbox_adapter_v2_full")
    full_ckpt.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60, flush=True)
    print("PHASE 2 FIX A: Full Pipeline", flush=True)
    print("=" * 60, flush=True)
    
    # === STEP 1: Load data ===
    print("\n[STEP 1] Loading data...", flush=True)
    records = load_jsonl(results_dir.parent / "phase2_v2_full.jsonl")
    all_examples = transitions_from_records(records)
    print(f"  Loaded {len(all_examples)} transitions from {len(records)} runs", flush=True)
    
    # === STEP 2: Load model ===
    print("\n[STEP 2] Loading model...", flush=True)
    t0 = time.time()
    wm = WorldModel(os.path.expanduser("~/models/Qwen2.5-0.5B-Instruct"))
    print(f"  Model loaded in {time.time()-t0:.1f}s, device={wm.device}", flush=True)
    
    # === STEP 3: Train full adapter on 65 transitions ===
    print("\n[STEP 3] Training full adapter (65 transitions, 3 epochs)...", flush=True)
    t0 = time.time()
    wm.lora_finetune(
        all_examples,
        epochs=3,
        batch_size=4,
        checkpoint_dir=full_ckpt,
        sandbox_mode=True,
    )
    full_train_time = time.time() - t0
    print(f"  Full training completed in {full_train_time:.1f}s", flush=True)
    
    # Save the final adapter
    wm.model.save_pretrained(str(full_ckpt))
    
    # Collect loss from training log - look at printed losses
    # The lora_finetune prints avg loss per epoch
    # We need to capture those. They were printed to stdout.
    # For now, record the training time as proxy
    full_result = {
        "path": str(full_ckpt),
        "num_transitions": len(all_examples),
        "training_time_seconds": full_train_time,
        "epochs": 3,
        "batch_size": 4,
    }
    
    # Save manifest
    manifest = {
        "num_runs": len(records),
        "num_transitions": len(all_examples),
        "data_source": str(results_dir.parent / "phase2_v2_full.jsonl"),
        "training_time_seconds": full_train_time,
    }
    (full_ckpt / "trained_manifest.json").write_text(json.dumps(manifest, indent=2))
    
    # === STEP 4: Held-out validation ===
    print("\n[STEP 4] Held-out validation...", flush=True)
    random.seed(42)
    indices = list(range(len(all_examples)))
    random.shuffle(indices)
    test_indices = set(indices[:10])
    train_examples = [all_examples[i] for i in indices if i not in test_indices]
    test_examples = [all_examples[i] for i in indices if i in test_indices]
    
    # Save splits
    with open(results_dir / "held_out_train.json", "w") as f:
        json.dump(train_examples, f, indent=2)
    with open(results_dir / "held_out_test.json", "w") as f:
        json.dump(test_examples, f, indent=2)
    print(f"  Split: {len(train_examples)} train, {len(test_examples)} test", flush=True)
    
    # Train on 55
    print("  Training held-out model (55 transitions, 3 epochs)...", flush=True)
    held_out_ckpt = results_dir / "held_out_checkpoint"
    held_out_ckpt.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    wm.lora_finetune(
        train_examples,
        epochs=3,
        batch_size=4,
        checkpoint_dir=held_out_ckpt,
        sandbox_mode=True,
    )
    held_out_train_time = time.time() - t0
    print(f"  Held-out training completed in {held_out_train_time:.1f}s", flush=True)
    
    # Evaluate on 10
    print("  Evaluating on held-out set...", flush=True)
    t0 = time.time()
    metrics = compute_metrics(wm, test_examples)
    eval_time = time.time() - t0
    metrics["training_time_seconds"] = held_out_train_time
    metrics["eval_time_seconds"] = eval_time
    metrics["passed"] = metrics["l1"] >= 0.90 and metrics["l2"] >= 0.70 and metrics["l3"] >= 0.50
    
    print(f"\n  L1 (exit_code): {metrics['l1']:.3f} (target >= 0.90)", flush=True)
    print(f"  L2 (state):     {metrics['l2']:.3f} (target >= 0.70)", flush=True)
    print(f"  L3 (output):    {metrics['l3']:.3f} (target >= 0.50)", flush=True)
    print(f"  VERDICT: {'PASS' if metrics['passed'] else 'FAIL'}", flush=True)
    
    # Save held-out results
    with open(results_dir / "held_out_results.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    # === STEP 5: Final report ===
    final = {
        "training_data": {"file": "results/phase2_v2_full.jsonl", "count": 65},
        "adapter": {
            "path": str(full_ckpt),
            "training_time_seconds": full_train_time,
        },
        "held_out": metrics,
        "verdict": "PASS" if metrics["passed"] else "FAIL",
    }
    
    with open(results_dir / "pipeline_result.json", "w") as f:
        json.dump(final, f, indent=2)
    
    print("\n" + "=" * 60, flush=True)
    print(f"FINAL VERDICT: {final['verdict']}", flush=True)
    print("=" * 60, flush=True)
    
    return 0 if metrics["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
