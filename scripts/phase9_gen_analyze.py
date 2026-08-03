#!/usr/bin/env python3
"""Phase 9 FF-GEN-1: 泛化判别实验 — 独立复算 + 门判定。

Reads results/phase9_gen_{flat,sbh_lam0,sbh_lam05}.jsonl and re-aggregates
per-task / pooled / deep-subset success WITHOUT touching the runner, then
adjudicates the pre-registered FF-GEN-1 gates:

    PASS: SBH_best >= count          (机制泛化，非劣)
    KILL: count - SBH_best >= 5      (过拟合实锤，>=2 集)
    else  NULL                       (机制中性，如实记录)

count  = flat pooled (满分 40); SBH_best = max(lam0, lam05) pooled。
次级注册问题（无门）：λ 在 depth>=2 是否分叉；T1 冷启动是否仍是主失败模式。

Usage:
    PYTHONPATH=src python3 scripts/phase9_gen_analyze.py
    PYTHONPATH=src python3 scripts/phase9_gen_analyze.py --cases   # 成败轨迹摘录
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

GEN_TASKS = [
    "gen_read_notes", "gen_read_setup", "gen_read_sensor", "gen_read_usage",
    "gen_find_api_ref", "gen_read_audit", "gen_count_readings",
    "gen_find_error_deep",
]
CONTROL_TASKS = ["gen_read_notes", "gen_read_setup"]
DEEP_TASKS = ["gen_read_sensor", "gen_read_usage", "gen_find_api_ref",
              "gen_read_audit", "gen_count_readings", "gen_find_error_deep"]
# 任务 -> 目标目录（/sandbox 相对，用于失败模式分类）
TARGET_DIR = {
    "gen_read_notes": "/sandbox",
    "gen_read_setup": "/sandbox/docs",
    "gen_read_sensor": "/sandbox/data/raw",
    "gen_read_usage": "/sandbox/docs/guides",
    "gen_find_api_ref": "/sandbox/docs/ref",
    "gen_read_audit": "/sandbox/logs/app",
    "gen_count_readings": "/sandbox/data/raw",
    "gen_find_error_deep": "/sandbox/logs/system",
}
TRAP_DIRS = {"/sandbox/app/templates", "/sandbox/data/archive"}


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


def simulate_cwd(actions) -> list:
    """Replay cd's from /sandbox to get the per-step cwd (deterministic)."""
    cwd = "/sandbox"
    out = []
    for a in actions:
        if a.startswith("cd "):
            tgt = a[3:].strip()
            if tgt == "..":
                cwd = str(Path(cwd).parent)
            elif tgt.startswith("/"):
                cwd = tgt
            else:
                cwd = str((Path(cwd) / tgt).resolve())
        out.append(cwd)
    return out


def classify_failure(e: dict) -> str:
    """失败模式分类（仅对失败 episode）：cold_start / trap / wrong_dir /
    dir_reached / other。"""
    actions = e.get("actions", [])
    cwds = simulate_cwd(actions)
    target = TARGET_DIR.get(e["task"])
    if not any(a.startswith("cd ") for a in actions):
        return "cold_start"          # T1：10 步全耗在根目录读+grep，cd(priority 2) 未轮到
    if target in cwds:
        return "dir_reached"         # 已到目标目录但未命中（预算/内容/候选缺失）
    if any(c in TRAP_DIRS for c in cwds):
        return "trap"                # 空目录陷阱消耗预算且未达目标
    return "wrong_dir"               # 游荡至非目标目录


def agg(episodes) -> dict:
    per_task = {t: [0, 0] for t in GEN_TASKS}
    failures = defaultdict(lambda: defaultdict(int))  # task -> mode -> count
    for e in episodes:
        per_task[e["task"]][0] += 1 if e["success"] else 0
        per_task[e["task"]][1] += 1
        if not e["success"]:
            failures[e["task"]][classify_failure(e)] += 1
    pooled_ok = sum(v[0] for v in per_task.values())
    pooled_total = sum(v[1] for v in per_task.values())
    deep = [sum(per_task[t][0] for t in DEEP_TASKS),
            sum(per_task[t][1] for t in DEEP_TASKS)]
    ctrl = [sum(per_task[t][0] for t in CONTROL_TASKS),
            sum(per_task[t][1] for t in CONTROL_TASKS)]
    return {"per_task": per_task, "pooled": (pooled_ok, pooled_total),
            "deep": deep, "control": ctrl, "failures": failures}


def adjudicate(flat: dict, a0: dict, a5: dict) -> dict:
    count = flat["pooled"][0]
    sbh_best = max(a0["pooled"][0], a5["pooled"][0])
    if sbh_best >= count:
        verdict = "PASS"
        reason = f"SBH_best={sbh_best}/40 >= count={count}/40（机制泛化，非劣）"
    elif count - sbh_best >= 5:
        verdict = "KILL"
        reason = f"count={count}/40 - SBH_best={sbh_best}/40 >= 5pp（过拟合实锤）"
    else:
        verdict = "NULL"
        reason = f"count={count}/40 - SBH_best={sbh_best}/40 在 (0,5) 之间（机制中性）"
    return {
        "count": count,
        "sbh_lam0": a0["pooled"][0],
        "sbh_lam05": a5["pooled"][0],
        "sbh_best": sbh_best,
        "verdict": verdict,
        "reason": reason,
        "deep": {"flat": flat["deep"], "lam0": a0["deep"], "lam05": a5["deep"]},
        "control": {"flat": flat["control"], "lam0": a0["control"], "lam05": a5["control"]},
    }


def lam_divergence(a0: dict, a5: dict, eps0: list, eps5: list) -> dict:
    per_task = {t: a5["per_task"][t][0] - a0["per_task"][t][0] for t in GEN_TASKS}
    # 动作序列逐位一致性（同任务同 episode 的 λ=0 与 λ=0.5 轨迹是否相同）
    identical = 0
    compared = 0
    by_ep = defaultdict(lambda: [None, None])
    for e in eps0:
        by_ep[(e["task"], e["episode"])][0] = e["actions"]
    for e in eps5:
        by_ep[(e["task"], e["episode"])][1] = e["actions"]
    for key, (a, b) in by_ep.items():
        if a is not None and b is not None:
            compared += 1
            if a == b:
                identical += 1
    return {
        "per_task_diff_lam05_minus_lam0": per_task,
        "pooled_diff": a5["pooled"][0] - a0["pooled"][0],
        "identical_action_pairs": identical,
        "compared_pairs": compared,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", action="store_true",
                        help="Print first-success trajectory excerpts per task/arm")
    parser.add_argument("--json", default="results/phase9_gen_analysis.json",
                        help="Path for the machine-readable analysis dump")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent / "results"
    flat = load(root / "phase9_gen_flat.jsonl")
    lam0 = load(root / "phase9_gen_sbh_lam0.jsonl")
    lam05 = load(root / "phase9_gen_sbh_lam05.jsonl")
    f = agg(flat["episodes"])
    a0 = agg(lam0["episodes"])
    a5 = agg(lam05["episodes"])
    gate = adjudicate(f, a0, a5)
    div = lam_divergence(a0, a5, lam0["episodes"], lam05["episodes"])

    print("=== FF-GEN-1 数据对账（WATCHDOG D4）===")
    for name, d in (("flat", flat), ("lam0", lam0), ("lam05", lam05)):
        n_eps = len(d["episodes"])
        meta = d["meta"]
        print(f"  {name:6s} {n_eps:3d} episodes | meta commit {meta.get('commit')} | "
              f"arm {meta.get('arm')} | lam {meta.get('lam')}")
    total = len(flat["episodes"]) + len(lam0["episodes"]) + len(lam05["episodes"])
    print(f"  TOTAL {total}/120 episodes" + ("  OK" if total == 120 else "  <-- MISMATCH"))

    print("\n=== 逐任务三臂矩阵（JSONL 独立复算）===")
    print(f"{'task':22s} {'flat':>6s} {'lam0':>6s} {'lam05':>6s} {'best':>5s}")
    for t in GEN_TASKS:
        a, b, c = f["per_task"][t], a0["per_task"][t], a5["per_task"][t]
        best = max(a[0], b[0], c[0])
        print(f"{t:22s} {a[0]}/{a[1]:>3d} {b[0]}/{b[1]:>3d} {c[0]}/{c[1]:>3d} "
              f"{best:>5d}")
    print(f"{'POOLED':22s} {f['pooled'][0]}/{f['pooled'][1]:>3d} "
          f"{a0['pooled'][0]}/{a0['pooled'][1]:>3d} "
          f"{a5['pooled'][0]}/{a5['pooled'][1]:>3d} "
          f"{gate['sbh_best']:>5d}")
    print(f"{'CONTROL (2 tasks)':22s} {f['control'][0]}/{f['control'][1]:>3d} "
          f"{a0['control'][0]}/{a0['control'][1]:>3d} "
          f"{a5['control'][0]}/{a5['control'][1]:>3d}")
    print(f"{'DEEP (6 tasks)':22s} {f['deep'][0]}/{f['deep'][1]:>3d} "
          f"{a0['deep'][0]}/{a0['deep'][1]:>3d} "
          f"{a5['deep'][0]}/{a5['deep'][1]:>3d}")

    print("\n=== 门判定（预注册，禁止事后调整）===")
    print(json.dumps(gate, indent=2, ensure_ascii=False))

    print("\n=== λ 次级问题（分叉记录，无门）===")
    print(f"  pooled diff (lam05 - lam0): {div['pooled_diff']}")
    print(f"  per-task diff: {div['per_task_diff_lam05_minus_lam0']}")
    print(f"  同任务同 episode 动作序列逐位一致: {div['identical_action_pairs']}/"
          f"{div['compared_pairs']}")

    print("\n=== 失败模式（失败 episode 分类计数）===")
    for name, d in (("flat", f), ("lam0", a0), ("lam05", a5)):
        tot = defaultdict(int)
        for t in GEN_TASKS:
            for mode, n in d["failures"][t].items():
                tot[mode] += n
        print(f"  {name:6s} failures={sum(tot.values()):2d} "
              f"{dict(tot)}")

    dump = {
        "gate": gate, "lam_divergence": div,
        "per_task": {t: {"flat": f["per_task"][t], "lam0": a0["per_task"][t],
                         "lam05": a5["per_task"][t]} for t in GEN_TASKS},
        "failures": {name: {t: dict(d["failures"][t]) for t in GEN_TASKS}
                     for name, d in (("flat", f), ("lam0", a0), ("lam05", a5))},
        "reconciliation": {
            "flat": len(flat["episodes"]), "lam0": len(lam0["episodes"]),
            "lam05": len(lam05["episodes"]), "total": total,
            "commits": {name: d["meta"].get("commit")
                        for name, d in (("flat", flat), ("lam0", lam0),
                                        ("lam05", lam05))},
        },
    }
    out = Path(__file__).resolve().parent.parent / args.json
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as fh:
        json.dump(dump, fh, indent=2, ensure_ascii=False)
    print(f"\nanalysis dump: {out}")

    if args.cases:
        print("\n=== 每任务每臂首次成功轨迹 ===")
        for name, d in (("flat", flat), ("lam0", lam0), ("lam05", lam05)):
            seen = set()
            for e in d["episodes"]:
                key = (e["task"], e["success"])
                if key in seen or not e["success"]:
                    continue
                seen.add(key)
                print(f"\n[{name}] {e['task']} ep{e['episode']} "
                      f"steps={e['steps']} success")
                print("  actions:", e["actions"])
                if e.get("goal_log"):
                    for g in e["goal_log"]:
                        if g.get("event") == "select" and g.get("goal"):
                            print(f"  t={g['t']} select goal={g['goal']} "
                                  f"density={g['density']} dist={g['dist']} "
                                  f"J={g['j']} unvisited={g['unvisited']}/{g['total']}")
        print("\n=== 每任务每臂首次失败轨迹（flat + SBH_best 臂）===")
        for name, d in (("flat", flat), ("lam0", lam0), ("lam05", lam05)):
            seen = set()
            for e in d["episodes"]:
                key = (e["task"], e["success"])
                if key in seen or e["success"]:
                    continue
                seen.add(key)
                print(f"\n[{name}] {e['task']} ep{e['episode']} "
                      f"steps={e['steps']} FAIL mode={classify_failure(e)}")
                print("  actions:", e["actions"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
