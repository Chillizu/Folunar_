#!/usr/bin/env python3
"""Error-weighted fine-tuning for PEDA World Model.

Instead of full RLVR (needs log-prob infrastructure), this uses a practical
approximation: for each training example, evaluate the WM's CURRENT prediction.
Examples where the WM is WRONG get higher training weight. This directly
targets the failure mode — correcting bad exit_code/cwd/files predictions.

Technique: "hard example mining" + "reward-weighted replay"
Similar to: Prioritized experience replay, RLVR's reward signal
"""
import argparse, json, os, sys, random
from pathlib import Path
from collections import Counter

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from phase1.world_model import WorldModel
from phase2.sandbox_env import SandboxState


def compute_accuracy(pred_text: str, gt: dict) -> float:
    """Score WM prediction accuracy against ground truth. 0.0-1.0."""
    try:
        pred = json.loads(pred_text) if isinstance(pred_text, str) else pred_text
    except (json.JSONDecodeError, TypeError):
        return 0.0

    score = 0.0
    n = 0

    # Exit code
    pred_exit = pred.get("last_exit_code", pred.get("exit_code", -1))
    gt_exit = gt.get("last_exit_code", gt.get("exit_code", -1))
    if pred_exit == gt_exit:
        score += 1.0
    n += 1

    # CWD
    if pred.get("cwd", "") == gt.get("cwd", ""):
        score += 1.0
    n += 1

    # Files (Jaccard)
    pred_files = set(pred.get("files", []))
    gt_files = set(gt.get("files", []))
    if pred_files or gt_files:
        jaccard = len(pred_files & gt_files) / max(len(pred_files | gt_files), 1)
        score += jaccard
        n += 1

    return score / max(n, 1)


def evaluate_and_build_weighted_dataset(wm: WorldModel, data: list) -> list:
    """For each example, evaluate current WM prediction, assign weight.
    Returns list of (example, weight) where weight = (1.0 - accuracy).
    """
    weighted = []
    stats = Counter()

    for ex in data:
        if not ex.get("action_name"):
            continue

        state_obj = SandboxState(
            cwd=ex.get("cwd", ""),
            files=ex.get("files", []),
        )
        action = ex["action_name"]
        gt = {
            "cwd": ex.get("next_cwd", ""),
            "files": ex.get("next_files", []),
            "last_exit_code": ex.get("next_last_exit_code", 0),
            "last_output": ex.get("next_last_output", ""),
        }

        # Evaluate current WM
        try:
            pred = wm.predict(state_obj, action)
            pred_json = json.loads(pred.level2_text) if pred.level2_text else {}
            pred_text = json.dumps(pred_json)
        except Exception:
            pred_text = "{}"

        acc = compute_accuracy(pred_text, gt)
        weight = 1.0 - acc  # wrong → high weight
        weighted.append((ex, weight))

        stats["total"] += 1
        if acc >= 0.9:
            stats["correct"] += 1
        elif acc >= 0.5:
            stats["partial"] += 1
        else:
            stats["wrong"] += 1

    print(f"[ewft] Evaluated {stats['total']} examples: "
          f"{stats['correct']} correct, {stats['partial']} partial, {stats['wrong']} wrong",
          flush=True)
    return weighted


def run_ewft(wm: WorldModel, weighted_data: list, epochs: int,
             batch_size: int, output_dir: Path):
    """Error-Weighted Fine-Tuning: train with higher weight on wrong predictions."""
    # Build weighted training set — repeat wrong examples more
    expanded = []
    for ex, weight in weighted_data:
        repeat = max(1, int(weight * 5))  # weight 0.2 → 1x, weight 1.0 → 5x
        for _ in range(repeat):
            expanded.append(ex)

    random.shuffle(expanded)
    print(f"[ewft] Expanded dataset: {len(expanded)} examples "
          f"(from {len(weighted_data)} original, avg repeat {len(expanded)/len(weighted_data):.1f}x)",
          flush=True)

    # Use existing SFT fine-tune (same as phase2_synthetic_train but with weighted data)
    wm.lora_finetune(
        expanded,
        epochs=epochs,
        learning_rate=2e-4,
        batch_size=batch_size,
        checkpoint_dir=output_dir,
        sandbox_mode=True,
    )


def main():
    parser = argparse.ArgumentParser(description="Error-Weighted Fine-Tuning")
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", default=os.path.expanduser("~/models/Qwen2.5-1.5B-Instruct"))
    parser.add_argument("--adapter-path", required=True, help="SFT checkpoint to start from")
    parser.add_argument("--output-dir", default="checkpoints/phase2/sandbox_adapter_ewft")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval-samples", type=int, default=500,
                        help="Max examples to evaluate (prediction is slow)")
    parser.add_argument("--eval-only", action="store_true",
                        help="Only evaluate accuracy, don't train")
    args = parser.parse_args()

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"[ewft] Loading SFT model from {args.adapter_path} ...", flush=True)
    wm = WorldModel(args.model, adapter_path=args.adapter_path)
    if wm.mode == "stub":
        raise RuntimeError("Model fell back to stub!")

    # Load data
    with open(args.data) as f:
        records = [json.loads(line) for line in f if line.strip()]
    # Inline transitions_from_records to avoid scripts import
    from phase2.tasks import MICRO_TASKS
    def _transitions(records):
        examples = []
        for rec in records:
            task_id = rec.get("task", "")
            task = next((t for t in MICRO_TASKS if t["id"] == task_id), None)
            for step in rec.get("records", []):
                if not step.get("action"):
                    continue
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
                    "state_text": json.dumps({"cwd": step["cwd"], "files": step["files"], "last_command": "", "last_exit_code": 0, "last_output": ""}, ensure_ascii=False),
                    "cwd": step["cwd"], "files": step["files"],
                    "action_name": step["action"],
                    "exit_code": exit_code,
                    "summary": f"executed {step['action']}",
                    "next_cwd": step["next_cwd"], "next_files": step["next_files"],
                    "next_last_exit_code": exit_code, "next_last_output": step["output"],
                })
        return examples
    data = _transitions(records)
    # Evaluate
    sample = data[:args.eval_samples]
    weighted = evaluate_and_build_weighted_dataset(wm, sample)

    if args.eval_only:
        return

    # Train
    run_ewft(wm, weighted, args.epochs, args.batch_size, output_path)
    print("[ewft] DONE", flush=True)


if __name__ == "__main__":
    main()
