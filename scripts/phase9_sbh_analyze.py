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
    PYTHONPATH=src python3 scripts/phase9_sbh_analyze.py --r1      # FF-SBH-3 rerun (phase9_sbh_r1_*)
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

# FF-SBH-3 pre-registered gates (R1 empty-dir re-selection rerun):
#   PASS: lambda-best pooled >= 41/45 AND per-task >= base matrix (no regression)
#   KILL: any task regression OR pooled < 40
#   else NULL (fix not effective; recorded as-is, no forced reading)
FF3_PASS_POOLED = 41
FF3_KILL_POOLED = 40
BASE_MATRIX = {  # FF-SBH-2 final matrix (40/45): read_hello 5, read_note 3,
    "read_hello": 5, "read_note": 3, "count_lines": 4, "find_secret": 5,   # count_lines 4,
    "read_welcome": 5, "find_api_key": 3, "count_measurements": 5,          # find_secret 5,
    "find_errors_v4": 5, "read_changelog_v4": 5,                             # ... find_api_key 3
}


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


def adjudicate_ff3(a0: dict, a5: dict) -> dict:
    """FF-SBH-3 gate: PASS / NULL / KILL per the pre-registered contract."""
    pooled_best = max(a0["pooled"][0], a5["pooled"][0])
    per_task_best = {
        t: max(a0["per_task"][t][0], a5["per_task"][t][0]) for t in TASKS
    }
    regressions = [t for t in TASKS if per_task_best[t] < BASE_MATRIX[t]]
    if regressions or pooled_best < FF3_KILL_POOLED:
        verdict = "FF-SBH-3 KILL"
        reason = (f"pooled_best={pooled_best}/45, regressions={regressions}"
                  if regressions else f"pooled_best={pooled_best}/45 < 40")
    elif pooled_best >= FF3_PASS_POOLED:
        verdict = "FF-SBH-3 PASS"
        reason = (f"pooled_best={pooled_best}/45 >= 41 and no task regression "
                  f"(find_api_key {BASE_MATRIX['find_api_key']} -> "
                  f"{per_task_best['find_api_key']})")
    else:
        verdict = "FF-SBH-3 NULL"
        reason = (f"pooled_best={pooled_best}/45 in [40,41) with no regression: "
                  f"fix not effective, recorded as-is")
    return {
        "lambda_best_pooled": pooled_best,
        "lambda_best_deep": max(a0["deep"][0], a5["deep"][0]),
        "per_task_best": per_task_best,
        "base_matrix": BASE_MATRIX,
        "ff3_pass_bar": FF3_PASS_POOLED,
        "ff3_kill_bar": FF3_KILL_POOLED,
        "regressions": regressions,
        "verdict": verdict,
        "reason": reason,
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
    parser.add_argument("--r1", action="store_true",
                        help="FF-SBH-3 mode: read phase9_sbh_r1_{lam0,lam05}.jsonl "
                             "and adjudicate the pre-registered R1 gates")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent / "results"
    prefix = "phase9_sbh_r1" if args.r1 else "phase9_sbh"
    lam0 = load(root / f"{prefix}_lam0.jsonl")
    lam05 = load(root / f"{prefix}_lam05.jsonl")
    a0 = agg(lam0["episodes"])
    a5 = agg(lam05["episodes"])
    verdict = adjudicate_ff3(a0, a5) if args.r1 else adjudicate(a0, a5)

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
    if args.r1:
        print("=== FF-SBH-3 baseline (FF-SBH-2 final matrix, 40/45) ===")
        print(f"{'task':22s} {'base':>8s} {'r1_lam0':>8s} {'r1_lam05':>8s} {'best':>6s}")
        for t in TASKS:
            base = f"{BASE_MATRIX[t]}/5"
            a, b = a0["per_task"][t], a5["per_task"][t]
            best = max(a[0], b[0])
            flag = "" if best >= BASE_MATRIX[t] else "  <-- regression"
            print(f"{t:22s} {base:>8s} {a[0]}/{a[1]:>3d} {b[0]}/{b[1]:>3d} "
                  f"{best:>5d}{flag}")
        pooled = f"{BASE_MATRIX['read_hello'] + BASE_MATRIX['read_note'] + BASE_MATRIX['count_lines'] + BASE_MATRIX['find_secret'] + BASE_MATRIX['read_welcome'] + BASE_MATRIX['find_api_key'] + BASE_MATRIX['count_measurements'] + BASE_MATRIX['find_errors_v4'] + BASE_MATRIX['read_changelog_v4']}/45"
        print(f"{'POOLED':22s} {pooled:>8s} {a0['pooled'][0]}/{a0['pooled'][1]:>3d} "
              f"{a5['pooled'][0]}/{a5['pooled'][1]:>3d} "
              f"{max(a0['pooled'][0], a5['pooled'][0]):>5d}")
        print(f"{'DEEP (read_note+api)':22s} {'5/10':>8s} {a0['deep'][0]}/{a0['deep'][1]:>3d} "
              f"{a5['deep'][0]}/{a5['deep'][1]:>3d} "
              f"{max(a0['deep'][0], a5['deep'][0]):>5d}")
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
