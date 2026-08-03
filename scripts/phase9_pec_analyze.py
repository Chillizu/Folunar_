#!/usr/bin/env python3
"""Phase 9 FF-PEC-1: PE 罗盘判别实验 — 独立复算 + 预注册门判定 + WM 成本统计。

Reads: FF-GEN-1/FF-MLP-1/FF-CEIL-1 基线 JSONL（flat / sbh_lam0 / mlp_lam0 /
ceil s20）+ FF-PEC-1 臂（results/phase9_pec_s{10,20}.jsonl），五臂对照：

    s10 主表：flat / SBH / MLP-λ0 / PEC（各 40 集）
    s20 次表：flat-s20 / SBH-s20 / PEC-s20（各 40 集；vs SBH s20 deep=3/30）

预注册门（FF-PEC-1，禁止事后调整）：
    deep = dist>=2 六任务子集（满分 30）；基线 MLP λ0 deep = 0/30。
    PASS:  PEC deep >= 4/30 且 PEC pooled >= 7/40
    KILL:  PEC deep <= 1/30
    NULL:  deep 2-3/30（弱信号，如实记录）

次级注册（无门）：s20 PEC vs SBH s20；WM 查询统计（次数/episode、fallback
率、s 项分布）；到达/选对分解（deep target 被选中 vs 实际到达）；轨迹对照
（PE 选对 vs 字典序选错，>=2 例）。

Usage:
    PYTHONPATH=src python3 scripts/phase9_pec_analyze.py
    PYTHONPATH=src python3 scripts/phase9_pec_analyze.py --json results/phase9_pec_analysis.json
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
    actions = e.get("actions", [])
    cwds = simulate_cwd(actions)
    target = TARGET_DIR.get(e["task"])
    if not any(a.startswith("cd ") for a in actions):
        return "cold_start"
    if target in cwds:
        return "dir_reached"
    if any(c in TRAP_DIRS for c in cwds):
        return "trap"
    return "wrong_dir"


def max_depth_reached(actions) -> int:
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


def adjudicate(pec: dict) -> dict:
    """FF-PEC-1 预注册门（s10 主判定）。"""
    deep, pooled = pec["deep"][0], pec["pooled"][0]
    if deep >= 4 and pooled >= 7:
        verdict = "PASS"
        reason = (f"PEC deep={deep}/30 >= 4/30 且 PEC pooled={pooled}/40 >= 7/40"
                  f"（PE 罗盘有效果）")
    elif deep <= 1:
        verdict = "KILL"
        reason = f"PEC deep={deep}/30 <= 1/30（PE 罗盘无效果）"
    else:
        verdict = "NULL"
        reason = f"PEC deep={deep}/30 在 2-3/30（弱信号，如实记录）"
    return {"pec_deep": deep, "pec_pooled": pooled,
            "verdict": verdict, "reason": reason}


def select_stats(episodes) -> dict:
    """到达/选对分解：select 事件中的深度分布、deep target 被选中 vs 到达。"""
    n_select = 0
    depth_hist = defaultdict(int)
    deep_targets = 0
    deep_arrived = 0
    for e in episodes:
        gl = e.get("goal_log", [])
        for g in gl:
            if g.get("event") == "select" and g.get("goal"):
                n_select += 1
                depth_hist[g.get("depth", 0)] += 1
                if g.get("depth", 0) >= 2:
                    deep_targets += 1
        for g in gl:
            if g.get("event") == "arrive" and g.get("goal"):
                if any(gg.get("goal") == g.get("goal") and
                       gg.get("event") == "select" and gg.get("depth", 0) >= 2
                       for gg in gl):
                    deep_arrived += 1
    return {"n_select": n_select, "depth_hist": dict(depth_hist),
            "deep_targets_selected": deep_targets,
            "deep_targets_arrived": deep_arrived}


def wm_stats(episodes) -> dict:
    """WM 查询统计：每集查询/回退次数、s 项来源分布、s 值分布。"""
    n_ep = len(episodes)
    queries = sum(e.get("wm_queries", 0) for e in episodes)
    fallbacks = sum(e.get("wm_fallbacks", 0) for e in episodes)
    src_hist = defaultdict(int)
    s_vals = []
    n_select = 0
    for e in episodes:
        for pt in e.get("pe_terms", []):
            n_select += 1
            src_hist[pt.get("src", "?")] += 1
            s_vals.append(pt.get("s", 0.0))
    return {
        "episodes": n_ep,
        "queries_total": queries,
        "queries_per_episode": round(queries / n_ep, 3) if n_ep else 0.0,
        "fallbacks_total": fallbacks,
        "fallback_rate": round(fallbacks / n_ep, 3) if n_ep else 0.0,
        "select_events": n_select,
        "s_source_hist": dict(src_hist),
        "s_mean": round(sum(s_vals) / len(s_vals), 4) if s_vals else None,
        "s_max": round(max(s_vals), 4) if s_vals else None,
    }


def first_selection_contrast(pec_eps, mlp_eps) -> list:
    """轨迹对照：PEC 与 MLP λ0 首次 select 目标不同的 (task, episode) 对。"""
    mlp_by = {(e["task"], e["episode"]): e for e in mlp_eps}
    out = []
    for e in pec_eps:
        key = (e["task"], e["episode"])
        m = mlp_by.get(key)
        if m is None:
            continue
        def _first(ep):
            for g in ep.get("goal_log", []):
                if g.get("event") == "select" and g.get("goal"):
                    return g
            return None
        psel, msel = _first(e), _first(m)
        if psel and msel and psel["goal"] != msel["goal"]:
            out.append({
                "task": e["task"], "episode": e["episode"],
                "pec_goal": psel["goal"], "pec_s": psel.get("s"),
                "mlp_goal": msel["goal"],
                "pec_success": e["success"], "mlp_success": m["success"],
                "pec_actions": e.get("actions", []),
                "mlp_actions": m.get("actions", []),
            })
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", default="results/phase9_pec_analysis.json",
                        help="machine-readable analysis dump path")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent / "results"
    flat10 = load(root / "phase9_gen_flat.jsonl")
    sbh10 = load(root / "phase9_gen_sbh_lam0.jsonl")
    mlp10 = load(root / "phase9_mlp_lam0.jsonl")
    pec10 = load(root / "phase9_pec_s10.jsonl")
    flat20 = load(root / "phase9_ceil_flat_s20.jsonl")
    sbh20 = load(root / "phase9_ceil_sbh_s20.jsonl")
    pec20 = load(root / "phase9_pec_s20.jsonl")

    arms10 = {"flat": flat10, "sbh_lam0": sbh10, "mlp_lam0": mlp10, "pec": pec10}
    a10 = {n: agg(d["episodes"]) for n, d in arms10.items()}
    a20 = {"flat_s20": agg(flat20["episodes"]), "sbh_s20": agg(sbh20["episodes"]),
           "pec_s20": agg(pec20["episodes"])}

    gate = adjudicate(a10["pec"])
    gate["baselines"] = {
        "flat_deep": a10["flat"]["deep"], "sbh_deep": a10["sbh_lam0"]["deep"],
        "mlp_deep": a10["mlp_lam0"]["deep"],
    }
    selstat = {"pec_s10": select_stats(pec10["episodes"]),
               "pec_s20": select_stats(pec20["episodes"]),
               "mlp_lam0": select_stats(mlp10["episodes"])}
    wms = {"pec_s10": wm_stats(pec10["episodes"]),
           "pec_s20": wm_stats(pec20["episodes"])}
    contrast = first_selection_contrast(pec10["episodes"], mlp10["episodes"])

    print("=== FF-PEC-1 数据对账（WATCHDOG D4）===")
    for name, d in {**arms10, "flat_s20": flat20, "sbh_s20": sbh20, "pec_s20": pec20}.items():
        m = d["meta"] or {}
        print(f"  {name:9s} {len(d['episodes']):3d} episodes | commit "
              f"{m.get('commit')} | arm {m.get('arm')} | "
              f"lam {m.get('lam')} | planner {m.get('planner')}")

    print("\n=== s10 逐任务四臂矩阵 ===")
    print(f"{'task':22s} {'flat':>6s} {'SBH':>6s} {'MLP0':>6s} {'PEC':>6s}")
    for t in GEN_TASKS:
        print(f"{t:22s} " + " ".join(
            f"{a10[n]['per_task'][t][0]:>6d}" for n in ("flat", "sbh_lam0", "mlp_lam0", "pec")))
    print(f"{'POOLED':22s} " + " ".join(
        f"{a10[n]['pooled'][0]}/{a10[n]['pooled'][1]:>2d}" for n in
        ("flat", "sbh_lam0", "mlp_lam0", "pec")))
    print(f"{'CONTROL':22s} " + " ".join(
        f"{a10[n]['control'][0]}/{a10[n]['control'][1]:>2d}" for n in
        ("flat", "sbh_lam0", "mlp_lam0", "pec")))
    print(f"{'DEEP':22s} " + " ".join(
        f"{a10[n]['deep'][0]}/{a10[n]['deep'][1]:>2d}" for n in
        ("flat", "sbh_lam0", "mlp_lam0", "pec")))

    print("\n=== s20 次表（vs SBH s20）===")
    print(f"{'task':22s} {'flat':>6s} {'SBH':>6s} {'PEC':>6s}")
    for t in GEN_TASKS:
        print(f"{t:22s} " + " ".join(
            f"{a20[n]['per_task'][t][0]:>6d}" for n in ("flat_s20", "sbh_s20", "pec_s20")))
    print(f"{'POOLED':22s} " + " ".join(
        f"{a20[n]['pooled'][0]}/{a20[n]['pooled'][1]:>2d}" for n in
        ("flat_s20", "sbh_s20", "pec_s20")))
    print(f"{'DEEP':22s} " + " ".join(
        f"{a20[n]['deep'][0]}/{a20[n]['deep'][1]:>2d}" for n in
        ("flat_s20", "sbh_s20", "pec_s20")))

    print("\n=== 门判定（s10 主判定，预注册，禁止事后调整）===")
    print(json.dumps(gate, indent=2, ensure_ascii=False))

    print("\n=== 到达/选对分解 ===")
    print(json.dumps(selstat, indent=2, ensure_ascii=False))

    print("\n=== WM 查询统计 ===")
    print(json.dumps(wms, indent=2, ensure_ascii=False))

    print(f"\n=== 首次 select 轨迹对照（PEC != MLP λ0，共 {len(contrast)} 对）===")
    for c in contrast[:10]:
        print(f"  {c['task']} ep{c['episode']}: PEC {c['pec_goal']} (s={c['pec_s']}, "
              f"{'SUCC' if c['pec_success'] else 'fail'}) vs MLP {c['mlp_goal']} "
              f"({'SUCC' if c['mlp_success'] else 'fail'})")

    dump = {
        "gate": gate, "s10": a10, "s20": a20, "select_stats": selstat,
        "wm_stats": wms, "contrast_count": len(contrast),
        "contrast_samples": contrast[:20],
        "files": {n: d["meta"].get("commit") for n, d in
                  {**arms10, "flat_s20": flat20, "sbh_s20": sbh20, "pec_s20": pec20}.items()},
    }
    out_path = Path(__file__).resolve().parent.parent / args.json
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(dump, f, indent=2, ensure_ascii=False)
    print(f"\n[dump] -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
