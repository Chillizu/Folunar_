#!/usr/bin/env python3
# ruff: noqa: E402
"""Phase 2 A3: Held-out validation — split data, train LoRA, evaluate on held-out test."""

import argparse
import json
import os
import random
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import torch
from phase1.world_model import WorldModel


def load_jsonl(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def transitions(records):
    examples = []
    for rec in records:
        for step in rec.get("records", []):
            if not step.get("action"):
                continue
            ec = step["exit_code"]
            examples.append({
                "state_text": json.dumps({
                    "cwd": step["cwd"], "files": step["files"],
                    "last_command": "", "last_exit_code": 0, "last_output": "",
                }, ensure_ascii=False),
                "action_name": step["action"],
                "exit_code": ec,
                "next_cwd": step.get("next_cwd", step["cwd"]),
                "next_files": step.get("next_files", step["files"]),
                "next_last_exit_code": ec,
                "next_last_output": step.get("output", ""),
            })
    return examples


def jaccard(a, b):
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def out_contain(pred, actual):
    pl = {l for l in pred.strip().split('\n') if l}
    al = {l for l in actual.strip().split('\n') if l}
    if not al:
        return 1.0 if not pl else 0.0
    return len(pl & al) / len(al)


def word_f1(pred, ref):
    pw = set(pred.lower().split())
    rw = set(ref.lower().split())
    if not pw and not rw:
        return 1.0
    inter = pw & rw
    if not inter:
        return 0.0
    p = len(inter) / max(len(pw), 1)
    r = len(inter) / max(len(rw), 1)
    return 2.0 * p * r / (p + r) if (p + r) > 0 else 0.0


def extract_json(text):
    """Extract first JSON object from text, handling truncation gracefully."""
    text = text.strip()
    if '```json' in text:
        text = text.split('```json')[1].split('```')[0].strip()
    elif '```' in text:
        text = text.split('```')[1].split('```')[0].strip()
    start = text.find('{')
    if start < 0:
        return {}
    depth = 0
    in_string = False
    escape = False
    end = -1
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == '\\' and in_string:
            escape = True
            continue
        if ch == '"' and not escape:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                end = i
                break
    if end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


def l2_score(parsed, ex):
    c = 1.0 if parsed.get('cwd') == ex['next_cwd'] else 0.0
    f = jaccard(parsed.get('files', []), ex.get('next_files', []))
    l = 1.0 if parsed.get('last_exit_code') == ex['next_last_exit_code'] else 0.0
    o = out_contain(parsed.get('last_output', ''), ex.get('next_last_output', ''))
    return c * 0.3 + f * 0.3 + l * 0.2 + o * 0.2


def main():
    parser = argparse.ArgumentParser(description="Held-out validation: split, train, evaluate")
    parser.add_argument("--data", required=True, help="Path to merged JSONL data")
    parser.add_argument("--output-dir", default="results/phase2_fix_a")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--test-size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--model", default=os.path.expanduser("~/models/Qwen2.5-0.5B-Instruct"))
    args = parser.parse_args()

    out_dir = _PROJECT_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir = out_dir / "adapter"

    # Load
    print(f"[heldout] Loading data from {args.data} ...")
    records = load_jsonl(args.data)
    all_data = transitions(records)
    print(f"[heldout] Loaded {len(all_data)} transitions from {len(records)} runs")

    # Split
    random.seed(args.seed)
    random.shuffle(all_data)
    test_data = all_data[:args.test_size]
    train_data = all_data[args.test_size:]
    print(f"[heldout] Split: {len(train_data)} train / {len(test_data)} test (seed={args.seed})")

    # Save splits for reproducibility
    with open(out_dir / "held_out_train.jsonl", "w") as f:
        for rec in train_data:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with open(out_dir / "held_out_test.jsonl", "w") as f:
        for rec in test_data:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Train
    print(f"[heldout] Loading model {args.model} ...")
    wm = WorldModel(args.model)
    if wm.mode == "stub" or wm.model is None:
        print("[heldout] stub/None model, abort")
        sys.exit(1)

    print(f"[heldout] Training LoRA for {args.epochs} epoch(s) ...")
    wm.lora_finetune(
        train_data,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        checkpoint_dir=adapter_dir,
        sandbox_mode=True,
    )

    print(f"[heldout] Saving adapter to {adapter_dir} ...")
    wm.model.save_pretrained(str(adapter_dir))

    # Evaluate
    sys_msg = wm._sandbox_system_message()
    tl1 = tl2 = tl3 = 0.0
    n = len(test_data)

    for i, ex in enumerate(test_data):
        uc = (f'State: {ex["state_text"]}\n'
              f'Action: {ex["action_name"]}\n'
              'Predict cwd, files, last_command, last_exit_code, last_output, exit_code, and summary as JSON:')
        msgs = [{'role': 'system', 'content': sys_msg}, {'role': 'user', 'content': uc}]
        with torch.no_grad():
            gen, _ = wm.generate_text(msgs, max_new_tokens=400)

        parsed = extract_json(gen)
        pe = parsed.get('exit_code', -1)
        try:
            pe = int(pe)
        except (TypeError, ValueError):
            pe = -1

        l1 = 1.0 if pe == ex['exit_code'] else 0.0
        l2 = l2_score(parsed, ex)
        l3 = word_f1(parsed.get('summary', ''), f'executed {ex["action_name"]}')

        tl1 += l1
        tl2 += l2
        tl3 += l3
        print(f'  [{i}] {ex["action_name"]:30s}  exit={ex["exit_code"]}->{pe}  '
              f'L1={l1:.1f}  L2={l2:.4f}  L3={l3:.4f}')
        if l1 < 1.0 or l2 < 0.5:
            print(f'       exit_code in parsed: {parsed.get("exit_code", "MISSING")}')
            print(f'       gen_start={gen[:80]}... gen_end=...{gen[-60:]}')

    a1 = tl1 / n
    a2 = tl2 / n
    a3 = tl3 / n

    print()
    print('=' * 60)
    print(f'  Held-Out Results ({n} test examples)')
    print('=' * 60)
    print(f'  L1 (exit-code accuracy):   {a1:.4f}')
    print(f'  L2 (state prediction):      {a2:.4f}')
    print(f'  L3 (summary relevance):     {a3:.4f}')
    print('=' * 60)

    passed = a1 >= 0.90 and a2 >= 0.70 and a3 >= 0.50
    print(f'  Thresholds: L1>=0.90 L2>=0.70 L3>=0.50')
    print(f'  PASSED: {passed}')
    print('=' * 60)

    report = {
        'train_size': len(train_data),
        'test_size': n,
        'l1': round(a1, 4),
        'l2': round(a2, 4),
        'l3': round(a3, 4),
        'passed': passed,
    }
    (out_dir / 'held_out_report.json').write_text(json.dumps(report, indent=2))
    print(f'\nReport: {json.dumps(report)}')

    if not passed:
        sys.exit(1)


if __name__ == '__main__':
    main()
