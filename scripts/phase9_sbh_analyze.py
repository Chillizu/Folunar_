#!/usr/bin/env python3
"""Phase 9 sandbox-hh (E1): independent recomputation from per-episode JSONL.

Reads results/phase9_sbh_{lam0,lam05}.jsonl and re-aggregates per-task /
pooled / deep-path success rates WITHOUT touching the runner code, then
adjudicates the pre-registered gates:

    FF-SBH-1 (kill):   lambda-best pooled < 37/45  -> sandbox migration
                       failed (record negative).
    FF-SBH-2 (positive): lambda-best pooled >= 39/45 AND deep-path
                       (read_note + find_api_key) >= 4/10 -> hierarchy
                       carries signal in the sandbox.
    else               mixed (pooled in [37,39) or deep-path no gain):
                       recorded as mixed, no forced reading.

lambda-best = max over arms of the pooled success count.

Usage:
    PYTHONPATH=src python3 scripts/phase9_sbh_analyze.py
    PYTHONPATH=src python3 scripts/phase9_sbh_analyze.py --cases   # deep-path decision excerpts
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

DEEP_TASKS = ["read_note", "find_api_key"]
TASKS = [
    "read_hello", "read_note", "count_lines", "find_secret", "read_welcome",
    "find_api_key", "count_measurements", "find_errors_v4", "read_changelog_v4",
]
LAMBDA_BEST_BAR = 37   # FF-SBH-1: pooled < 37/45 -> kill
POSITIVE_BAR = 39      # FF-SBH-2: pooled >= 39/45
DEEP_BAR = 4           # FF-SBH-2: deep-path >= 4/10


def load(path: Path) -> dict:
    meta = None
    episodes = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if "meta" in rec:
                meta = rec["meta"]
            else:
                episodes.append(rec)
    return {"meta": meta, "episodes": episodes}


def agg(episodes) -> dict:
    per_task = defaultdict(lambda: [0, 0])  # task -> [ok, total]
    for e in episodes:
        per_task[e["task"]][0] += 1 if e["success"] else 0
        per_task[e["task"]][1] += 1
    pooled_ok = sum(v[0] for v in per_task.values())
    pooled_total = sum(v[1] for v in per_task.values())
    deep_ok = sum(per_task[t][0] for t in DEEP_TASKS)
    deep_total = sum(per_task[t][1] for t in DEEP_TASKS)
    return {
        "per_task": {t: (per_task[t][0], per_task[t][1]) for t in TASKS},
        "pooled": (pooled_ok, pooled_total),
        "deep": (deep_ok, deep_total),
    }


def adjudicate(a: dict, b: dict) -> dict:
    pooled_best = max(a["pooled"][0], b["pooled"][0])
    deep_best = max(a["deep"][0], b["deep"][0])
    if pooled_best < LAMBDA_BEST_BAR:
        verdict = "FF-SBH-1 KILL (negative): sandbox migration failed"
    elif pooled_best >= POSITIVE_BAR and deep_best >= DEEP_BAR:
        verdict = "FF-SBH-2 PASS (positive): hierarchy carries signal in sandbox"
    elif pooled_best >= LAMBDA_BEST_BAR:
        verdict = "MIXED: pooled in bar but deep-path not improved (no forced reading)"
    else:
        verdict = "MIXED: pooled in [37,39) band (no forced reading)"
    return {
        "lambda_best_pooled": pooled_best,
        "lambda_best_deep": deep_best,
        "ff_sbh_1_bar": LAMBDA_BEST_BAR,
        "ff_sbh_2_pooled_bar": POSITIVE_BAR,
        "ff_sbh_2_deep_bar": DEEP_BAR,
        "verdict": verdict,
    }


def decision_excerpts(episodes) -> list:
    """First successful episode per deep-path task: goal decision sequence."""
    out = []
    seen = set()
    for e in episodes:
        key = (e["task"], e["lam"])
        if key in seen or not e["success"] or e["task"] not in DEEP_TASKS:
            continue
        seen.add(key)
        seq = [g for g in e["goal_log"] if g.get("event") == "select"]
        out.append({
            "task": e["task"], "lam": e["lam"], "episode": e["episode"],
            "steps": e["steps"], "actions": e["actions"], "goal_log": e["goal_log"],
            "select_seq": seq,
        })
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", action="store_true",
                        help="Print deep-path decision-sequence excerpts")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent / "results"
    lam0 = load(root / "phase9_sbh_lam0.jsonl")
    lam05 = load(root / "phase9_sbh_lam05.jsonl")
    a0 = agg(lam0["episodes"])
    a5 = agg(lam05["episodes"])
    verdict = adjudicate(a0, a5)

    print("=== per-task success (recomputed from JSONL) ===")
    print(f"{'task':22s} {'lam=0':>8s} {'lam=0.5':>8s} {'phase8_base':>12s}")
    p8 = {"read_hello": "5/5", "read_note": "3/5", "count_lines": "4/5",
          "find_secret": "5/5", "read_welcome": "5/5", "find_api_key": "2/5",
          "count_measurements": "5/5", "find_errors_v4": "5/5",
          "read_changelog_v4": "5/5"}
    for t in TASKS:
        a, b = a0["per_task"][t], a5["per_task"][t]
        print(f"{t:22s} {a[0]}/{a[1]:>3d} {b[0]}/{b[1]:>3d} {p8[t]:>10s}")
    print(f"{'POOLED':22s} {a0['pooled'][0]}/{a0['pooled'][1]:>3d} "
          f"{a5['pooled'][0]}/{a5['pooled'][1]:>3d} 39/45")
    print(f"{'DEEP (read_note+api)':22s} {a0['deep'][0]}/{a0['deep'][1]:>3d} "
          f"{a5['deep'][0]}/{a5['deep'][1]:>3d} 2/10")
    print()
    print("=== gate adjudication ===")
    print(json.dumps(verdict, indent=2))

    if args.cases:
        print()
        print("=== deep-path first-success decision sequences ===")
        for ex in decision_excerpts(lam0["episodes"] + lam05["episodes"]):
            print(f"\n[{ex['task']} lam={ex['lam']} ep={ex['episode']}] steps={ex['steps']}")
            print("  actions:", ex["actions"])
            for g in ex["select_seq"]:
                print(f"  t={g['t']} select goal={g['goal']} density={g['density']} "
                      f"dist={g['dist']} J={g['j']} unvisited={g['unvisited']}/{g['total']}")
            for g in ex["goal_log"]:
                if g.get("event") == "arrive":
                    print(f"  t={g['t']} arrive goal={g['goal']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
