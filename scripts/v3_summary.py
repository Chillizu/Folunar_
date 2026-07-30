#!/usr/bin/env python3
"""Summarize v3 experiment results."""
import json, glob, sys

for fp in sorted(glob.glob("results/v3_*.jsonl")):
    parts = fp.replace("results/v3_", "").replace(".jsonl", "").split("_", 1)
    mode, task = parts[0], parts[1]
    recs = []
    with open(fp) as f:
        for l in f:
            if l.strip():
                recs.append(json.loads(l))
    succ = sum(1 for r in recs if r.get("success"))
    n = len(recs)
    avg_steps = sum(r.get("steps_count", 0) for r in recs) / n if n else 0
    avg_scr = sum(r.get("scr", 0) for r in recs) / n if n else 0
    rate = succ / n if n else 0
    print(f"{task:<20} {mode:<10} {succ:>2}/{n:<3}  rate={rate:.3f}  steps={avg_steps:>5.1f}  scr={avg_scr:.3f}")

print()
tasks = ["read_greeting", "count_entries", "find_secret_note", "read_user_guide"]
print("--- Learned vs Fallback Comparison ---")
print(f"{'Task':<20} {'learned':<10} {'fallback':<10} {'diff':<8}")
print("-" * 48)
for task in tasks:
    lr = fr = None
    for fp in glob.glob("results/v3_*.jsonl"):
        parts = fp.replace("results/v3_", "").replace(".jsonl", "").split("_", 1)
        if parts[1] == task:
            recs = [json.loads(l) for l in open(fp) if l.strip()]
            rate = sum(1 for r in recs if r.get("success")) / len(recs)
            if parts[0] == "learned":
                lr = rate
            else:
                fr = rate
    if lr is not None and fr is not None:
        diff = lr - fr
        print(f"{task:<20} {lr:.3f}      {fr:.3f}      {diff:+.3f}")
    else:
        print(f"{task:<20} {lr}   {fr}   N/A")
