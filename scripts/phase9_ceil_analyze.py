#!/usr/bin/env python3
"""Phase 9 FF-CEIL-1: 预算/起点天花板探测 — 独立复算 + Q1-Q3 口径。

Reads the FF-GEN-1 s10 references
(results/phase9_gen_{flat,sbh_lam0,sbh_lam05}.jsonl) and the FF-CEIL-1
s15/s20 cells (results/phase9_ceil_{flat,sbh}_s{15,20}.jsonl), re-aggregates
per-task / pooled / deep / control success across the 2 arms x 3 budgets
(s10 from GEN-1, s15/s20 from CEIL-1), classifies failure modes, and
answers the registered Q1-Q3 (口径禁止事后改):

  Q1: 预算 10->20，deep 子集是否从 0/30 变非零？变多少？
  Q2: 若 s20 仍 deep=0 -> 机制墙实锤（MLP 的任何增益都不能归因于预算松绑）；
  Q3: dist-1 对照子集是否随预算继续上升（区分全局预算不足 vs 深度特异失败）？

口径（与 FF-GEN-1 完全一致，见 scripts/phase9_gen_analyze.py）:
  DEEP_TASKS（6 任务, dist>=2, 30 集）; CONTROL_TASKS（2 任务, dist<=1,
  10 集）。合约 Q3 的 "dist-1 对照子集" 按 GEN-1 的 CONTROL 定义 = 目标
  dist<=1 的任务集合（gen_read_notes dist-0 + gen_read_setup dist-1）；
  同时单独给出纯 dist-1 任务 gen_read_setup（5 集）的数字。

轨迹对照（合约要求 >=2 例 s10 失败而 s15/s20 成功的 episode 槽）:
  按 (task, episode) 槽跨预算对齐输出轨迹 + goal_log（SBH）。

Usage:
    PYTHONPATH=src venv/bin/python3 scripts/phase9_ceil_analyze.py
    PYTHONPATH=src venv/bin/python3 scripts/phase9_ceil_analyze.py --cases
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
# 任务 -> 目标目录（/sandbox 相对，用于失败模式分类；与 GEN-1 完全一致）
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

# arm -> {budget -> (file, expected episode count)}
ARMS = {
    "flat": {10: "phase9_gen_flat.jsonl", 15: "phase9_ceil_flat_s15.jsonl",
             20: "phase9_ceil_flat_s20.jsonl"},
    "sbh": {10: "phase9_gen_sbh_lam0.jsonl", 15: "phase9_ceil_sbh_s15.jsonl",
            20: "phase9_ceil_sbh_s20.jsonl"},
}
BUDGETS = [10, 15, 20]


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


def max_depth_reached(actions) -> int:
    """Max cwd depth from /sandbox over the trajectory (cd-steps)."""
    cwds = simulate_cwd(actions)
    best = 0
    for c in cwds:
        if c == "/sandbox":
            d = 0
        elif c.startswith("/sandbox/"):
            d = len(c[len("/sandbox/"):].split("/"))
        else:
            d = 0
        best = max(best, d)
    return best


def classify_failure(e: dict) -> str:
    """失败模式分类（仅对失败 episode）：cold_start / trap / wrong_dir /
    dir_reached / other。与 GEN-1 完全一致。"""
    actions = e.get("actions", [])
    cwds = simulate_cwd(actions)
    target = TARGET_DIR.get(e["task"])
    if not any(a.startswith("cd ") for a in actions):
        return "cold_start"          # T1：预算全耗在根目录读+grep，cd 未轮到
    if target in cwds:
        return "dir_reached"         # 已到目标目录但未命中（预算/内容/候选缺失）
    if any(c in TRAP_DIRS for c in cwds):
        return "trap"                # 空目录陷阱消耗预算且未达目标
    return "wrong_dir"               # 游荡至非目标目录


def agg(episodes) -> dict:
    per_task = {t: [0, 0] for t in GEN_TASKS}
    failures = defaultdict(lambda: defaultdict(int))  # task -> mode -> count
    depth_hist = defaultdict(int)                     # max depth -> count
    target_reached = 0                                # 轨迹曾到目标目录（成败不论）
    for e in episodes:
        per_task[e["task"]][0] += 1 if e["success"] else 0
        per_task[e["task"]][1] += 1
        depth_hist[max_depth_reached(e.get("actions", []))] += 1
        if TARGET_DIR.get(e["task"]) in simulate_cwd(e.get("actions", [])):
            target_reached += 1
        if not e["success"]:
            failures[e["task"]][classify_failure(e)] += 1
    pooled_ok = sum(v[0] for v in per_task.values())
    pooled_total = sum(v[1] for v in per_task.values())
    deep = [sum(per_task[t][0] for t in DEEP_TASKS),
            sum(per_task[t][1] for t in DEEP_TASKS)]
    ctrl = [sum(per_task[t][0] for t in CONTROL_TASKS),
            sum(per_task[t][1] for t in CONTROL_TASKS)]
    # 纯 dist-1 任务（gen_read_setup，5 集）单独口径
    setup = list(per_task["gen_read_setup"])
    return {"per_task": per_task, "pooled": (pooled_ok, pooled_total),
            "deep": deep, "control": ctrl, "dist1_task": setup,
            "failures": failures, "depth_hist": dict(depth_hist),
            "target_reached": target_reached}


def select_dist_hist(episodes) -> dict:
    """SBH goal_log: 被选中目标的 dist 分布（跨所有 select 事件）。"""
    dist_hist = defaultdict(int)
    total = 0
    for e in episodes:
        for g in e.get("goal_log", []):
            if g.get("event") == "select" and g.get("goal"):
                dist_hist[g["dist"]] += 1
                total += 1
    return {"dist_hist": dict(dist_hist), "total_select": total}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", action="store_true",
                        help="Print s10-fail -> s15/s20-success trajectory pairs")
    parser.add_argument("--json", default="results/phase9_ceil_analysis.json",
                        help="Path for the machine-readable analysis dump")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent / "results"
    data = {}   # arm -> budget -> {"meta": ..., "episodes": [...], "agg": ...}
    for arm, files in ARMS.items():
        data[arm] = {}
        for b, fname in files.items():
            rec = load(root / fname)
            rec["agg"] = agg(rec["episodes"])
            rec["select"] = select_dist_hist(rec["episodes"])
            data[arm][b] = rec

    print("=== FF-CEIL-1 数据对账（WATCHDOG D4）===")
    print("  （160 新集 = 4 个 ceil JSONL x 40；s10 为 FF-GEN-1 引用，不重复计入）")
    total_eps = 0
    for arm, files in ARMS.items():
        for b, fname in files.items():
            d = data[arm][b]
            n = len(d["episodes"])
            if b >= 15:
                total_eps += n
            meta = d["meta"]
            print(f"  {arm:5s} s{b:2d} {n:3d} eps | commit {meta.get('commit')} | "
                  f"arm {meta.get('arm')} | lam {meta.get('lam')} | "
                  f"max_steps {meta.get('max_steps')}")
    print(f"  NEW episodes (s15/s20): {total_eps}" +
          ("  OK" if total_eps == 160 else "  <-- MISMATCH (expect 160)"))

    print("\n=== 逐任务矩阵（arm x budget x 成功数）===")
    print(f"{'task':22s}" + "".join(
        f" {arm}/s{b}:{'ok/tot':>7s}" for arm in ("flat", "sbh") for b in BUDGETS))
    for t in GEN_TASKS:
        row = f"{t:22s}"
        for arm in ("flat", "sbh"):
            for b in BUDGETS:
                ok, tot = data[arm][b]["agg"]["per_task"][t]
                row += f" {arm[:4]}/s{b}:{ok:>2d}/{tot:<2d}"
        print(row)
    for label, key in (("POOLED", "pooled"), ("DEEP (dist>=2, 30)", "deep"),
                       ("CONTROL (dist<=1, 10)", "control"),
                       ("dist-1 task setup (5)", "dist1_task")):
        row = f"{label:22s}"
        for arm in ("flat", "sbh"):
            for b in BUDGETS:
                ok, tot = data[arm][b]["agg"][key]
                row += f" {arm[:4]}/s{b}:{ok:>2d}/{tot:<2d}"
        print(row)

    print("\n=== 深度可达性（每 episode 最大 cwd 深度直方图 + 到目标目录集数）===")
    for arm in ("flat", "sbh"):
        for b in BUDGETS:
            a = data[arm][b]["agg"]
            hist = ", ".join(f"d{d}:{a['depth_hist'].get(d, 0)}"
                             for d in sorted(a["depth_hist"]))
            print(f"  {arm:5s} s{b:2d} depth_hist {{{hist}}} | "
                  f"target_dir_reached {a['target_reached']}/40")

    print("\n=== SBH goal_log：被选中目标 dist 分布 ===")
    for b in BUDGETS:
        s = data["sbh"][b]["select"]
        print(f"  s{b:2d} select_total={s['total_select']} "
              f"dist_hist={s['dist_hist']}")

    print("\n=== 失败模式迁移（失败集分类计数，arm x budget）===")
    modes = ["cold_start", "trap", "wrong_dir", "dir_reached"]
    print(f"{'arm/budget':12s}" + "".join(f"{m:>12s}" for m in modes) +
          f"{'FAIL':>6s}")
    for arm in ("flat", "sbh"):
        for b in BUDGETS:
            a = data[arm][b]["agg"]
            tot = defaultdict(int)
            for t in GEN_TASKS:
                for m, n in a["failures"][t].items():
                    tot[m] += n
            n_fail = (sum(v[1] for v in a["per_task"].values())
                      - a["pooled"][0])
            row = f"{arm}/s{b:<7d}"
            for m in modes:
                row += f"{tot.get(m, 0):>12d}"
            row += f"{n_fail:>6d}"
            print(row)

    # ── Q1-Q3（注册口径，禁止事后改）──
    print("\n=== Q1-Q3（注册口径）===")
    q = {}
    for arm in ("flat", "sbh"):
        q[arm] = {"deep": {b: data[arm][b]["agg"]["deep"][0] for b in BUDGETS},
                  "control": {b: data[arm][b]["agg"]["control"][0]
                              for b in BUDGETS},
                  "dist1_task": {b: data[arm][b]["agg"]["dist1_task"][0]
                                 for b in BUDGETS},
                  "pooled": {b: data[arm][b]["agg"]["pooled"][0]
                             for b in BUDGETS}}
    for arm in ("flat", "sbh"):
        d10, d20 = q[arm]["deep"][10], q[arm]["deep"][20]
        print(f"  Q1 [{arm}] deep 0/30 -> s20: {d20}/30"
              + (f"（非零，+{d20}）" if d20 else "（仍为 0）"))
        print(f"  Q2 [{arm}] s20 deep==0: {d20 == 0}"
              + (" -> 机制墙实锤" if d20 == 0 else " -> 预算墙贡献（机制墙未实锤）"))
        c10, c20 = q[arm]["control"][10], q[arm]["control"][20]
        print(f"  Q3 [{arm}] control {c10}/10 -> s20 {c20}/10"
              + (f"（+{c20 - c10}）" if c20 >= c10 else f"（-{c10 - c20}）"))

    # ── 轨迹对照：s10 失败而 s15/s20 成功的 (task, ep) 槽 ──
    print("\n=== 轨迹对照（s10 失败 -> s15/s20 成功 的 episode 槽）===")
    cases = []
    for arm in ("flat", "sbh"):
        s10 = {(e["task"], e["episode"]): e for e in data[arm][10]["episodes"]}
        for b in (15, 20):
            for e in data[arm][b]["episodes"]:
                key = (e["task"], e["episode"])
                e10 = s10.get(key)
                if e10 is not None and not e10["success"] and e["success"]:
                    cases.append({"arm": arm, "budget": b, "task": e["task"],
                                  "episode": e["episode"], "steps10": e10["steps"],
                                  "actions10": e10["actions"],
                                  "steps": e["steps"], "actions": e["actions"],
                                  "goal_log": e.get("goal_log")})
    print(f"  共 {len(cases)} 例：")
    for c in cases:
        print(f"  [{c['arm']}] {c['task']} ep{c['episode']}: "
              f"s10 FAIL({c['steps10']}步) -> s{c['budget']} OK({c['steps']}步)")

    dump = {
        "arms": {arm: {b: {"meta": {k: data[arm][b]["meta"].get(k)
                                    for k in ("commit", "max_steps", "arm",
                                              "lam", "experiment")},
                           "agg": {"per_task": data[arm][b]["agg"]["per_task"],
                                   "pooled": data[arm][b]["agg"]["pooled"],
                                   "deep": data[arm][b]["agg"]["deep"],
                                   "control": data[arm][b]["agg"]["control"],
                                   "dist1_task":
                                       data[arm][b]["agg"]["dist1_task"],
                                   "failures": {t: dict(f)
                                                for t, f in
                                                data[arm][b]["agg"]["failures"].items()},
                                   "depth_hist":
                                       data[arm][b]["agg"]["depth_hist"],
                                   "target_reached":
                                       data[arm][b]["agg"]["target_reached"]},
                           "select": data[arm][b]["select"]}
                       for b in BUDGETS}
                      for arm in ("flat", "sbh")},
        "q1_q3": q,
        "trajectory_cases": cases,
        "reconciliation": {
            "total_episodes": total_eps,
            "commits": {f"{arm}_{b}": data[arm][b]["meta"].get("commit")
                        for arm in ("flat", "sbh") for b in BUDGETS},
        },
    }
    out = Path(__file__).resolve().parent.parent / args.json
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as fh:
        json.dump(dump, fh, indent=2, ensure_ascii=False)
    print(f"\nanalysis dump: {out}")

    if args.cases:
        print("\n=== 轨迹明细（每例前 20 步）===")
        for c in cases[:20]:
            print(f"\n[{c['arm']}] {c['task']} ep{c['episode']} "
                  f"s10 FAIL -> s{c['budget']} OK")
            print("  s10 actions:", c["actions10"])
            print(f"  s{c['budget']} actions:", c["actions"][:20])
            if c.get("goal_log"):
                for g in c["goal_log"][:6]:
                    if g.get("event") == "select" and g.get("goal"):
                        print(f"  t={g['t']} select goal={g['goal']} "
                              f"density={g['density']} dist={g['dist']} "
                              f"J={g['j']} unvisited={g['unvisited']}/{g['total']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
