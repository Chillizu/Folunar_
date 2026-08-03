#!/usr/bin/env python3
"""Independent recompute + analysis for the FF-HG-5 rerun (post fallback-fix).

Loads original F5 JSONLs (results/phase9_hg_f5, count + pe) and rerun PE
JSONLs (results/phase9_hg_f5_rerun), recomputes completion/steps/learning
curves/verb distributions from raw per-episode records, and prints every
number the report cites. Used as the standalone recompute verification
(contract acceptance #3).
"""
import json
import statistics
from collections import Counter
from pathlib import Path

BASE = Path("results/phase9_hg_f5")
RERUN = Path("results/phase9_hg_f5_rerun")
TASKS = ["read_changelog_v4", "count_measurements", "find_errors_v4"]
TASK_BRANCH = {
    "read_changelog_v4": "train (docs)",
    "count_measurements": "train (data)",
    "find_errors_v4": "held-out (logs)",
}


def load(p):
    eps = []
    for line in Path(p).read_text().splitlines():
        d = json.loads(line)
        if "meta" in d:
            continue
        eps.append(d)
    return eps


def comp(eps):
    return sum(1 for e in eps if e["success"]) / len(eps) * 100.0 if eps else 0.0


def avg_steps(eps):
    return statistics.mean(e["steps"] for e in eps) if eps else 0.0


def comp_half(eps, lo, hi):
    sub = [e for e in eps if lo <= e["episode"] <= hi]
    return comp(sub)


def mean_unc(eps):
    u = [s["uncertainty"] for e in eps for s in e.get("step_records", [])
         if s.get("uncertainty") is not None]
    return statistics.mean(u) if u else None


def verbs(eps):
    return Counter(s["action"].split()[0] for e in eps for s in e.get("step_records", []))


def agg(agent, alpha, task):
    if agent == "count":
        return load(BASE / f"phase9_hg_f5_count_a0.5_{task}.jsonl")
    return load(RERUN / f"phase9_hg_f5_pe_a{alpha}_{task}.jsonl")


print("=" * 78)
print("PER-RUN SUMMARY  (rerun = post-fix PE; count + old PE from original F5)")
print("=" * 78)
rows = []
for task in TASKS:
    c = agg("count", 0.5, task)
    for a in ["0.5", "1.0"]:
        old = load(BASE / f"phase9_hg_f5_pe_a{a}_{task}.jsonl")
        new = agg("pe", a, task)
        rows.append((task, a, comp(c), comp(old), comp(new)))
        print(f"{task:22s} a{a}: count {comp(c):5.1f}% | pe-old {comp(old):5.1f}% | pe-new {comp(new):5.1f}%"
              f" | steps new {avg_steps(new):.2f}")

print()
print("AGGREGATE (60 eps, M3/FF-CI-6 wording)")
c_all = []
for t in TASKS:
    c_all += agg("count", 0.5, t)
print(f"count: {comp(c_all):.1f}%  ({sum(1 for e in c_all if e['success'])}/60)")
for a in ["0.5", "1.0"]:
    p_all = []
    for t in TASKS:
        p_all += agg("pe", a, t)
    print(f"pe a{a}: {comp(p_all):.1f}%  ({sum(1 for e in p_all if e['success'])}/60)"
          f"  delta vs count {comp(p_all)-comp(c_all):+.1f}pp")

print()
print("LEARNING CURVE (ep1-10 vs ep11-20), rerun PE")
for task in TASKS:
    for a in ["0.5", "1.0"]:
        e = agg("pe", a, task)
        print(f"{task:22s} a{a}: ep1-10 {comp_half(e,0,9):5.1f}% ep11-20 {comp_half(e,10,19):5.1f}%")

print()
print("VERB DISTRIBUTION on held-out find_errors_v4 (rerun PE vs original count)")
cv = verbs(agg("count", 0.5, "find_errors_v4"))
for a in ["0.5", "1.0"]:
    pv = verbs(agg("pe", a, "find_errors_v4"))
    print(f"count: {dict(cv.most_common(6))}")
    print(f"pe a{a}: {dict(pv.most_common(6))}")

print()
print("MEAN UNCERTAINTY (rerun PE, per task)")
for task in TASKS:
    for a in ["0.5", "1.0"]:
        e = agg("pe", a, task)
        print(f"{task:22s} a{a}: mean_unc {mean_unc(e):.4f}")

print()
print("PER-TASK GATE (original FF-HG-5: PE >= count - 10pp -> PASS)")
for task in TASKS:
    c = comp(agg("count", 0.5, task))
    p = comp(agg("pe", 0.5, task))
    print(f"{task:22s} count {c:5.1f}% pe-a0.5 {p:5.1f}% delta {p-c:+5.1f}pp -> {'PASS' if p >= c-10 else 'FAIL'}")
