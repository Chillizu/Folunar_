#!/usr/bin/env python3
"""Phase 9 FF-MLP-1: 路径级规划器判别实验 — 独立复算 + 门判定。

Reads the FF-GEN-1 baseline JSONL (flat / sbh_lam0 / sbh_lam05) plus the
FF-MLP-1 arms (results/phase9_mlp_lam{0,05}.jsonl) and re-aggregates the
four-arm comparison WITHOUT touching the runner, then adjudicates the
pre-registered FF-MLP-1 gates:

    MLP_best = max(lam0, lam05); deep = dist>=2 六任务子集（满分 30）。
    基线：SBH-R1 deep = 0/30，flat deep = 0/30（FF-GEN-1 实测）。
    PASS: MLP_best deep >= 4/30 且 MLP_best pooled >= 7/40
    KILL: MLP_best deep <= 1/30
    NULL: deep 2-3/30（弱信号，如实记录）
    阈值禁止事后调整。

次级注册问题（无门）：λ 在路径规划下是否分叉（逐位动作一致性 +
逐任务差）；T1 冷启动失败数变化（failures.cold_start）。

Usage:
    PYTHONPATH=src python3 scripts/phase9_mlp_analyze.py
    PYTHONPATH=src python3 scripts/phase9_mlp_analyze.py --cases  # 轨迹摘录
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
    """失败模式（与 phase9_gen_analyze.py 同口径）：cold_start /
    trap / wrong_dir / dir_reached。"""
    actions = e.get("actions", [])
    cwds = simulate_cwd(actions)
    target = TARGET_DIR.get(e["task"])
    if not any(a.startswith("cd ") for a in actions):
        return "cold_start"          # T1：10 步全耗在根目录
    if target in cwds:
        return "dir_reached"         # 已到目标目录但未命中
    if any(c in TRAP_DIRS for c in cwds):
        return "trap"                # 空目录陷阱
    return "wrong_dir"               # 游荡至非目标目录


def max_depth_reached(actions) -> int:
    """cwd 相对 /sandbox 的最大深度（cd 模拟）。"""
    return max((len([p for p in cwd.split("/") if p]) - 1
                for cwd in simulate_cwd(actions)), default=0)


def agg(episodes) -> dict:
    per_task = {t: [0, 0] for t in GEN_TASKS}
    failures = defaultdict(lambda: defaultdict(int))
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


def adjudicate(a0: dict, a5: dict) -> dict:
    """FF-MLP-1 预注册门。MLP_best = max(lam0, lam05)。"""
    deep0, deep5 = a0["deep"][0], a5["deep"][0]
    best_deep = max(deep0, deep5)
    best_arm = "lam0" if deep0 >= deep5 else "lam05"
    pool0, pool5 = a0["pooled"][0], a5["pooled"][0]
    best_pooled = max(pool0, pool5)
    if best_deep >= 4 and best_pooled >= 7:
        verdict = "PASS"
        reason = (f"MLP_best deep={best_deep}/30 >= 4/30 且 "
                  f"MLP_best pooled={best_pooled}/40 >= 7/40（机制有效果）")
    elif best_deep <= 1:
        verdict = "KILL"
        reason = f"MLP_best deep={best_deep}/30 <= 1/30（机制无效果）"
    else:
        verdict = "NULL"
        reason = f"MLP_best deep={best_deep}/30 在 2-3/30（弱信号，如实记录）"
    return {
        "mlp_deep_lam0": deep0, "mlp_deep_lam05": deep5,
        "mlp_deep_best": best_deep, "best_arm": best_arm,
        "mlp_pooled_lam0": pool0, "mlp_pooled_lam05": pool5,
        "mlp_pooled_best": best_pooled,
        "verdict": verdict, "reason": reason,
        "baselines": {"flat_deep": None, "sbh_deep": None},  # filled by caller
    }


def lam_divergence(a0: dict, a5: dict, eps0: list, eps5: list) -> dict:
    per_task = {t: a5["per_task"][t][0] - a0["per_task"][t][0] for t in GEN_TASKS}
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


def select_stats(episodes) -> dict:
    """MLP 臂机制统计：select 事件中的深度分布、深度>=2 目标是否被导航。"""
    n_select = 0
    depth_hist = defaultdict(int)
    deep_targets = 0          # select 事件选中 depth>=2 目标
    deep_arrived = 0          # 这些目标里实际到达（goal_log 有 arrive）的次数
    arrived = set()
    for e in episodes:
        gl = e.get("goal_log", [])
        for g in gl:
            ev = g.get("event")
            if ev == "select" and g.get("goal"):
                n_select += 1
                d = g.get("depth", 0)
                depth_hist[d] += 1
                if d >= 2:
                    deep_targets += 1
            elif ev == "arrive" and g.get("goal"):
                arrived.add(g["goal"])
    # deep arrivals = arrive events whose goal was selected with depth>=2
    deep_arrived = sum(1 for e in episodes
                       for g in e.get("goal_log", [])
                       if g.get("event") == "arrive" and g.get("goal") and
                       any(gg.get("goal") == g.get("goal") and
                           gg.get("event") == "select" and gg.get("depth", 0) >= 2
                           for gg in e.get("goal_log", [])))
    return {"n_select": n_select, "depth_hist": dict(depth_hist),
            "deep_targets_selected": deep_targets,
            "deep_targets_arrived": deep_arrived}


def first_success(episodes) -> dict:
    """task -> 首次成功 episode 记录（含 select 摘要）。"""
    out = {}
    for e in episodes:
        if e["success"] and e["task"] not in out:
            out[e["task"]] = e
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", action="store_true",
                        help="Print success/failure trajectory excerpts")
    parser.add_argument("--json", default="results/phase9_mlp_analysis.json",
                        help="Path for the machine-readable analysis dump")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent / "results"
    flat = load(root / "phase9_gen_flat.jsonl")
    sbh0 = load(root / "phase9_gen_sbh_lam0.jsonl")
    sbh5 = load(root / "phase9_gen_sbh_lam05.jsonl")
    mlp0 = load(root / "phase9_mlp_lam0.jsonl")
    mlp5 = load(root / "phase9_mlp_lam05.jsonl")
    arms = {"flat": flat, "sbh_lam0": sbh0, "sbh_lam05": sbh5,
            "mlp_lam0": mlp0, "mlp_lam05": mlp5}
    a = {name: agg(d["episodes"]) for name, d in arms.items()}
    gate = adjudicate(a["mlp_lam0"], a["mlp_lam05"])
    gate["baselines"] = {
        "flat_deep": a["flat"]["deep"], "sbh_deep": a["sbh_lam0"]["deep"],
    }
    div = lam_divergence(a["mlp_lam0"], a["mlp_lam05"],
                         mlp0["episodes"], mlp5["episodes"])
    selstat = {"mlp_lam0": select_stats(mlp0["episodes"]),
               "mlp_lam05": select_stats(mlp5["episodes"])}

    print("=== FF-MLP-1 数据对账（WATCHDOG D4）===")
    for name, d in arms.items():
        print(f"  {name:9s} {len(d['episodes']):3d} episodes | meta commit "
              f"{d['meta'].get('commit')} | arm {d['meta'].get('arm')} | "
              f"lam {d['meta'].get('lam')} | planner {d['meta'].get('planner')}")
    total = sum(len(d["episodes"]) for d in arms.values())
    print(f"  TOTAL {total}/200 episodes" + ("  OK" if total == 200 else
                                             "  <-- MISMATCH (expect 80 new + 120 baseline)"))

    print("\n=== 逐任务四臂矩阵（JSONL 独立复算）===")
    print(f"{'task':22s} {'flat':>6s} {'SBH0':>6s} {'SBH05':>6s} "
          f"{'MLP0':>6s} {'MLP05':>6s} {'best':>4s}")
    for t in GEN_TASKS:
        vals = [a[n]["per_task"][t][0] for n in
                ("flat", "sbh_lam0", "sbh_lam05", "mlp_lam0", "mlp_lam05")]
        best = max(vals)
        mark = lambda v: f"{v:>6d}"  # noqa: E731
        print(f"{t:22s} " + " ".join(mark(v) for v in vals) + f" {best:>4d}")
    print(f"{'POOLED':22s} " + " ".join(
        f"{a[n]['pooled'][0]}/{a[n]['pooled'][1]:>2d}" for n in
        ("flat", "sbh_lam0", "sbh_lam05", "mlp_lam0", "mlp_lam05"))
        + f" {gate['mlp_pooled_best']:>4d}")
    print(f"{'CONTROL':22s} " + " ".join(
        f"{a[n]['control'][0]}/{a[n]['control'][1]:>2d}" for n in
        ("flat", "sbh_lam0", "sbh_lam05", "mlp_lam0", "mlp_lam05")))
    print(f"{'DEEP':22s} " + " ".join(
        f"{a[n]['deep'][0]}/{a[n]['deep'][1]:>2d}" for n in
        ("flat", "sbh_lam0", "sbh_lam05", "mlp_lam0", "mlp_lam05"))
        + f" {gate['mlp_deep_best']:>4d}")

    print("\n=== 门判定（预注册，禁止事后调整）===")
    print(json.dumps(gate, indent=2, ensure_ascii=False))

    print("\n=== λ 次级问题（无门）===")
    print(f"  pooled diff (lam05 - lam0): {div['pooled_diff']}")
    print(f"  per-task diff: {div['per_task_diff_lam05_minus_lam0']}")
    print(f"  同任务同 episode 动作序列逐位一致: {div['identical_action_pairs']}/"
          f"{div['compared_pairs']}")

    print("\n=== MLP 臂机制统计（select 深度分布 / deep 目标导航）===")
    for name in ("mlp_lam0", "mlp_lam05"):
        s = selstat[name]
        print(f"  {name}: selects={s['n_select']} depth_hist={s['depth_hist']} "
              f"deep_targets_selected={s['deep_targets_selected']} "
              f"deep_targets_arrived={s['deep_targets_arrived']}")

    print("\n=== 失败模式（失败 episode 分类计数）===")
    for name in ("flat", "sbh_lam0", "mlp_lam0", "mlp_lam05"):
        tot = defaultdict(int)
        for t in GEN_TASKS:
            for mode, n in a[name]["failures"][t].items():
                tot[mode] += n
        print(f"  {name:9s} failures={sum(tot.values()):2d} {dict(tot)}")

    dump = {
        "gate": gate, "lam_divergence": div, "select_stats": selstat,
        "per_task": {t: {n: a[n]["per_task"][t] for n in arms} for t in GEN_TASKS},
        "failures": {n: {t: dict(a[n]["failures"][t]) for t in GEN_TASKS}
                     for n in arms},
        "reconciliation": {n: len(d["episodes"]) for n, d in arms.items()},
    }
    out = Path(__file__).resolve().parent.parent / args.json
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as fh:
        json.dump(dump, fh, indent=2, ensure_ascii=False)
    print(f"\nanalysis dump: {out}")

    if args.cases:
        print("\n=== 每任务每臂首次成功轨迹 ===")
        for name in ("mlp_lam0", "mlp_lam05", "sbh_lam0", "flat"):
            seen = set()
            for e in arms[name]["episodes"]:
                key = (e["task"], e["success"])
                if key in seen or not e["success"]:
                    continue
                seen.add(key)
                print(f"\n[{name}] {e['task']} ep{e['episode']} "
                      f"steps={e['steps']} success")
                print("  actions:", e["actions"])
                for g in e.get("goal_log", []):
                    if g.get("event") == "select":
                        print(f"  t={g['t']} select goal={g['goal']} "
                              f"prior={g.get('prior')} depth={g.get('depth')} "
                              f"J={g.get('j')} path={g.get('path')}")
        print("\n=== 每任务每臂首次失败轨迹（flat / SBH / MLP_best）===")
        for name in ("mlp_lam0", "mlp_lam05", "sbh_lam0", "flat"):
            seen = set()
            for e in arms[name]["episodes"]:
                key = (e["task"], e["success"])
                if key in seen or e["success"]:
                    continue
                seen.add(key)
                print(f"\n[{name}] {e['task']} ep{e['episode']} "
                      f"steps={e['steps']} FAIL mode={classify_failure(e)} "
                      f"maxdepth={max_depth_reached(e['actions'])}")
                print("  actions:", e["actions"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
