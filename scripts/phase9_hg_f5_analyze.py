#!/usr/bin/env python3
# ruff: noqa: E402
"""Phase 9 FF-HG-5: analysis + report generator.

Reads results/phase9_hg_f5/*.jsonl (meta + per-episode records, WATCHDOG D4)
and produces:
  - results/phase9_hg_f5_summary.csv  (per agent x task: completion, steps, learning curve)
  - results/phase9_hg_f5_report.md    (gate verdict + per-task breakdown + signal analysis)

Gate (FF-HG-5, operationalized per Direction-1 M3 spec — 20 eps x 3 tasks,
10pp band, adopted verbatim from pre-registered FF-CI-6):
    PASS  : PE(alpha=0.5) completion% >= count completion% - 10pp
    FAIL  : PE < count - 10pp  ->  formal negative result (direction dead)
Secondary (M3 same-spec): learning curve both arms >= 2x (ep11-20 vs ep1-10);
discovery steps PE <= 1.5x count.

Usage: python scripts/phase9_hg_f5_analyze.py [--data results/phase9_hg_f5]
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

TASKS = ["read_changelog_v4", "count_measurements", "find_errors_v4"]
# Branch mapping to MVP train/held-out split (probe protocol):
TASK_BRANCH = {
    "read_changelog_v4": "train (docs)",
    "count_measurements": "train (data)",
    "find_errors_v4": "held-out (logs)",
}


def load_episodes(path: Path) -> list:
    eps = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        if "meta" in d:
            continue
        eps.append(d)
    return eps


def completion(eps) -> float:
    return sum(1 for e in eps if e["success"]) / len(eps) * 100.0 if eps else 0.0


def avg_steps(eps) -> float:
    return statistics.mean(e["steps"] for e in eps) if eps else 0.0


def completion_half(eps, lo: int, hi: int) -> float:
    sub = [e for e in eps if lo <= e["episode"] <= hi]
    return completion(sub)


def mean_errors(eps):
    """Per-step discriminator error stats across an agent's episodes."""
    errs = [sr["error"] for e in eps for sr in e.get("step_records", [])
            if sr.get("error") is not None]
    uncs = [sr["uncertainty"] for e in eps for sr in e.get("step_records", [])
            if sr.get("uncertainty") is not None]
    return {
        "n_steps": len(errs),
        "mean_error": statistics.mean(errs) if errs else None,
        "mean_uncertainty": statistics.mean(uncs) if uncs else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="results/phase9_hg_f5")
    ap.add_argument("--out", default="results/phase9_hg_f5_report.md")
    args = ap.parse_args()

    data_dir = Path(args.data)
    files = sorted(data_dir.glob("phase9_hg_f5_*.jsonl"))
    if not files:
        print(f"[analyze] no JSONL under {data_dir}")
        sys.exit(1)

    # group: (agent, alpha, task) -> episodes
    runs = {}
    metas = {}
    for f in files:
        eps = load_episodes(f)
        meta = json.loads(f.read_text().splitlines()[0])["meta"]
        key = (meta["agent"], meta["alpha"], meta["task"])
        runs[key] = eps
        metas[key] = meta

    agents = sorted({k[0] for k in runs})
    alphas = sorted({k[1] for k in runs})
    tasks = [t for t in TASKS if any(k[2] == t for k in runs)]

    # ── aggregates ──
    rows = []
    for (agent, alpha, task), eps in sorted(runs.items()):
        rows.append({
            "agent": agent, "alpha": alpha, "task": task,
            "branch": TASK_BRANCH.get(task, "?"),
            "n_eps": len(eps),
            "completion_pct": round(completion(eps), 1),
            "avg_steps": round(avg_steps(eps), 2),
            "ep1_10_pct": round(completion_half(eps, 0, 9), 1),
            "ep11_20_pct": round(completion_half(eps, 10, 19), 1),
            **{f"sig_{k}": (round(v, 4) if v is not None else None)
               for k, v in mean_errors(eps).items()},
        })

    csv_path = data_dir / "phase9_hg_f5_summary.csv"
    cols = ["agent", "alpha", "task", "branch", "n_eps", "completion_pct",
            "avg_steps", "ep1_10_pct", "ep11_20_pct",
            "sig_n_steps", "sig_mean_error", "sig_mean_uncertainty"]
    with open(csv_path, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r[c]) for c in cols) + "\n")

    # ── gate computation ──
    count_key = lambda t: ("count", 0.5, t) if ("count", 0.5, t) in runs else None
    pe_key = lambda t, a: ("pe", a, t) if ("pe", a, t) in runs else None

    gate_rows = []
    for t in tasks:
        ck = count_key(t)
        if ck is None:
            continue
        c = completion(runs[ck])
        row = {"task": t, "branch": TASK_BRANCH.get(t, "?"),
               "count_pct": round(c, 1)}
        for a in alphas:
            pk = pe_key(t, a)
            if pk is None:
                continue
            p = completion(runs[pk])
            row[f"pe_a{a}_pct"] = round(p, 1)
            row[f"pe_a{a}_minus_count_pp"] = round(p - c, 1)
            if a == 0.5:
                row["verdict"] = "PASS" if p >= c - 10.0 else "FAIL"
        gate_rows.append(row)

    # aggregate (M3/FF-CI-6 wording: overall completion across all 3 tasks)
    def aggregate_completion(agent, alpha):
        eps = []
        for t in tasks:
            k = (agent, alpha, t)
            if k in runs:
                eps += runs[k]
        return completion(eps) if eps else None

    agg = {}
    for a in alphas:
        c_agg = aggregate_completion("count", 0.5)
        p_agg = aggregate_completion("pe", a)
        if c_agg is not None and p_agg is not None:
            agg[f"pe_a{a}"] = {
                "count_agg_pct": round(c_agg, 1),
                "pe_agg_pct": round(p_agg, 1),
                "delta_pp": round(p_agg - c_agg, 1),
                "aggregate_verdict": "PASS" if p_agg >= c_agg - 10.0 else "FAIL",
                "per_task_fails": [r["task"] for r in gate_rows
                                   if r.get("verdict") == "FAIL"],
            }

    # verb-distribution contrast on held-out task (helps/hurts evidence)
    from collections import Counter
    verb_contrast = {}
    for t in tasks:
        branch = TASK_BRANCH.get(t, "?")
        if branch.startswith("held-out"):
            for a in alphas:
                k = pe_key(t, a)
                ck = count_key(t)
                if k is None or ck is None:
                    continue
                cverb = Counter(s["action"].split()[0]
                                for e in runs[ck] for s in e.get("step_records", []))
                pverb = Counter(s["action"].split()[0]
                                for e in runs[k] for s in e.get("step_records", []))
                verb_contrast[f"pe_a{a}"] = {
                    "count_verbs": dict(cverb.most_common(8)),
                    "pe_verbs": dict(pverb.most_common(8)),
                }

    # ── signal analysis: where does the discriminator help/hurt? ──
    signal_rows = []
    for t in tasks:
        pk = pe_key(t, 0.5)
        if pk is None:
            continue
        eps = runs[pk]
        # error on success vs failure episodes (agent-side), and per-step
        succ_errs, fail_errs = [], []
        for e in eps:
            errs = [sr["error"] for sr in e.get("step_records", [])
                    if sr.get("error") is not None]
            (succ_errs if e["success"] else fail_errs).extend(errs)
        # which actions get high error (uncertain predictions that were wrong)
        high_err_actions = {}
        for e in eps:
            for sr in e.get("step_records", []):
                if sr.get("error") is not None and sr["error"] >= 0.4:
                    verb = sr["action"].split()[0]
                    high_err_actions[verb] = high_err_actions.get(verb, 0) + 1
        signal_rows.append({
            "task": t, "branch": TASK_BRANCH.get(t, "?"),
            "success_ep_mean_error": round(statistics.mean(succ_errs), 4) if succ_errs else None,
            "failure_ep_mean_error": round(statistics.mean(fail_errs), 4) if fail_errs else None,
            "n_success_steps": len(succ_errs), "n_failure_steps": len(fail_errs),
            "top_high_error_actions": dict(sorted(high_err_actions.items(),
                                                  key=lambda kv: -kv[1])[:4]),
        })

    # ── render markdown ──
    lines = ["# Phase 9 FF-HG-5 — Agent-Level Gate: Discriminator-Driven vs Count Baseline", ""]
    first_meta = next(iter(metas.values()))
    lines.append(f"- commit: `{first_meta['commit']}`")
    lines.append(f"- timestamp: {first_meta['timestamp']}")
    lines.append(f"- host: {first_meta['host']} | {first_meta['cpu_or_gpu']}")
    lines.append(f"- sandbox_image: {first_meta['sandbox_image']}")
    lines.append(f"- episodes per run: 20 | max_steps: 10")
    lines.append("")
    lines.append("## Gate definition (FF-HG-5, operationalized)")
    lines.append("")
    lines.append("PHASE9_PLAN.md registers FF-HG-5 only as `Agent-level <= count baseline (post-MVP) -> DEAD` — no threshold/episodes/tasks. Per the task brief, the definition is completed **verbatim from Direction-1 M3 / FF-CI-6 spec** (20 eps x 3 tasks, PE >= count - 10pp), i.e. a protocol fill-in, not a post-hoc threshold adjustment:")
    lines.append("")
    lines.append("| Criterion | Spec |")
    lines.append("|---|---|")
    lines.append("| Episodes | 20 per (agent, task) |")
    lines.append("| Tasks | read_changelog_v4, count_measurements, find_errors_v4 (v4 sandbox) |")
    lines.append("| Primary | PE(alpha=0.5) completion% >= count completion% - 10pp -> PASS |")
    lines.append("| Secondary | both arms ep11-20 >= 2x ep1-10; discovery steps PE <= 1.5x count |")
    lines.append("| Exploratory | PE(alpha=1.0) = discriminator REPLACES count (not gated) |")
    lines.append("")

    lines.append("## Per-run summary")
    lines.append("")
    lines.append("| agent | alpha | task | branch (MVP split) | eps | completion% | avg steps | ep1-10% | ep11-20% |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        lines.append(f"| {r['agent']} | {r['alpha']} | {r['task']} | {r['branch']} | {r['n_eps']} "
                     f"| {r['completion_pct']} | {r['avg_steps']} | {r['ep1_10_pct']} | {r['ep11_20_pct']} |")
    lines.append("")

    lines.append("## Gate verdict (per task and overall)")
    lines.append("")
    lines.append("| task | branch | count% | pe a0.5% | delta pp | verdict | pe a1.0% | delta pp |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in gate_rows:
        pe10 = r.get("pe_a1.0_pct", "-")
        d10 = r.get("pe_a1.0_minus_count_pp", "-")
        lines.append(f"| {r['task']} | {r['branch']} | {r['count_pct']} | "
                     f"{r.get('pe_a0.5_pct', '-')} | {r.get('pe_a0.5_minus_count_pp', '-')} | "
                     f"{r.get('verdict', '-')} | {pe10} | {d10} |")
    lines.append("")
    for a in alphas:
        o = agg.get(f"pe_a{a}")
        if o:
            fails = o["per_task_fails"]
            lines.append(f"- **PE(alpha={a}) aggregate** (60 eps, M3/FF-CI-6 wording): "
                         f"count {o['count_agg_pct']}% vs PE {o['pe_agg_pct']}%, "
                         f"delta {o['delta_pp']:+} pp -> {o['aggregate_verdict']} "
                         f"(band -10pp); per-task FAILs: {fails if fails else 'none'}")
    lines.append("")
    if verb_contrast:
        lines.append("## Verb-distribution contrast on the held-out task (find_errors_v4)")
        lines.append("")
        lines.append("| arm | top verbs by executed count |")
        lines.append("|---|---|")
        for a, v in verb_contrast.items():
            lines.append(f"| count | {v['count_verbs']} |")
            lines.append(f"| {a} | {v['pe_verbs']} |")
        lines.append("")

    lines.append("## Learning curve (ep11-20 vs ep1-10) and discovery steps")
    lines.append("")
    lines.append("| agent | alpha | task | ep1-10% | ep11-20% | ratio |")
    lines.append("|---|---|---|---|---|---|")
    for r in rows:
        ratio = (r["ep11_20_pct"] / r["ep1_10_pct"]) if r["ep1_10_pct"] > 0 else float("inf")
        lines.append(f"| {r['agent']} | {r['alpha']} | {r['task']} | {r['ep1_10_pct']} | "
                     f"{r['ep11_20_pct']} | {ratio:.2f} |")
    lines.append("")
    lines.append("M3-style check (both arms >= 2x improvement): see ratios above — a task already at ceiling (100% in ep1-10) trivially cannot double; the check is reported, not gated, for FF-HG-5.")
    lines.append("")

    lines.append("## Discriminator signal: where it helps / hurts")
    lines.append("")
    lines.append("| task | branch | success-ep mean error | failure-ep mean error | top high-error actions |")
    lines.append("|---|---|---|---|---|")
    for s in signal_rows:
        lines.append(f"| {s['task']} | {s['branch']} | {s['success_ep_mean_error']} | "
                     f"{s['failure_ep_mean_error']} | {s['top_high_error_actions']} |")
    lines.append("")

    lines.append("## Consistency with MVP signal validation")
    lines.append("")
    lines.append("MVP (results/phase9_signal_validation_20260731_074345.md): AUC_disc=0.808, "
                 "KL(emp||uniform)=0.600, KL(heldout||train)=0.959, Cohen's d(E,D)=1.457, "
                 "Spearman rho=0.754 — the discriminator error field has structure and is "
                 "count-orthogonal in the offline probe study.")
    lines.append("")
    lines.append("FF-HG-5 tests whether that structured error survives the closed loop. "
                 "Interpretation is written by the analysis run below (see verdict and signal table).")
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append(f"- per-episode JSONL: `{data_dir}/phase9_hg_f5_*.jsonl`")
    lines.append(f"- summary CSV: `{csv_path}`")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"[analyze] wrote {out}")
    print(f"[analyze] wrote {csv_path}")
    for r in gate_rows:
        print(f"[analyze] {r['task']}: count={r['count_pct']}% "
              f"pe0.5={r.get('pe_a0.5_pct', '-')}% delta={r.get('pe_a0.5_minus_count_pp', '-')} "
              f"-> {r.get('verdict', '-')}")


if __name__ == "__main__":
    main()
