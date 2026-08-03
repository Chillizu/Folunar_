#!/usr/bin/env python3
"""FF-CI-6 analysis for the GPU M3 PEDA run (results/phase9_ci_m3_peda_gpu.jsonl).

Merges GPU PE (primary) with count sides; CPU partial used as reference.
Adaptive early/late split (20ep -> ep1-10/ep11-20; 10ep -> ep1-5/ep6-10).
Emits report-ready markdown + JSON.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RES = REPO / "results"
TASKS = ["read_secret_ci", "read_data_ci", "find_warn_ci"]


def load_rows(path):
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(l) for l in f if not l.startswith('{"meta"')]


def completion(rows):
    return sum(1 for r in rows if r["success"]) / len(rows) if rows else 0.0


def discovery(rows):
    succ = [r for r in rows if r["success"]]
    if not succ:
        return None
    return min(r.get("victory_step") if r.get("victory_step") is not None else r["steps"]
               for r in succ)


def steps_series(rows, task):
    rr = sorted([r for r in rows if r["task_id"] == task], key=lambda r: r["episode"])
    return [(r["episode"], r["success"], r["steps"],
             r.get("victory_step")) for r in rr]


def main():
    gpu = load_rows(RES / "phase9_ci_m3_peda_gpu.jsonl")
    cpu = load_rows(RES / "phase9_ci_m3_peda_cpu_partial_readsecret10ep.jsonl")
    count = load_rows(RES / "phase9_ci_m3_count.jsonl")
    count20 = load_rows(RES / "phase9_ci_m3_count_20ep.jsonl")

    if not gpu:
        print("NO GPU DATA YET at results/phase9_ci_m3_peda_gpu.jsonl")
        return 1

    n_ep = max((r["episode"] for r in gpu), default=-1) + 1
    mid = max(n_ep // 2, 1)

    # meta
    meta = {}
    with (RES / "phase9_ci_m3_peda_gpu.jsonl").open() as f:
        first = f.readline()
        meta = json.loads(first)["meta"]
    print(f"GPU meta: commit={meta.get('commit')} ts={meta.get('timestamp')} "
          f"host={meta.get('host')} device={meta.get('cpu_or_gpu')} "
          f"eps={meta.get('num_episodes')} max_steps={meta.get('max_steps')}")
    print(f"GPU rows: {len(gpu)}; episodes per task: "
          f"{ {t: sum(1 for r in gpu if r['task_id']==t) for t in TASKS} }")
    print(f"Split: n_ep={n_ep} mid={mid}\n")

    out = []
    out.append("| task | pe_gpu | count10 | count20 | delta_vs_count10 | delta_vs_count20 |")
    out.append("|---|---|---|---|---|---|")
    pe_all = []
    for t in TASKS:
        pe = [r for r in gpu if r["task_id"] == t]
        ct = [r for r in count if r["task_id"] == t]
        ct2 = [r for r in count20 if r["task_id"] == t]
        pe_c, ct_c, ct2_c = completion(pe), completion(ct), completion(ct2)
        out.append(f"| {t} | {pe_c:.3f} | {ct_c:.3f} | {ct2_c:.3f} | "
                   f"{pe_c-ct_c:+.3f} | {pe_c-ct2_c:+.3f} |")
        pe_all += pe
    pe_c, ct_c, ct2_c = completion(pe_all), completion(count), completion(count20)
    out.append(f"| **POOLED** | **{pe_c:.3f}** | **{ct_c:.3f}** | **{ct2_c:.3f}** | "
               f"**{pe_c-ct_c:+.3f}** | **{pe_c-ct2_c:+.3f}** |")
    out.append("")
    ff6_10 = pe_c >= ct_c - 0.10
    ff6_20 = pe_c >= ct2_c - 0.10
    out.append(f"**FF-CI-6 vs count10ep**: PE {pe_c:.3f} vs count {ct_c:.3f}; criterion PE >= count-0.10 "
               f"(>= {ct_c-0.10:.3f}) -> {'PASS' if ff6_10 else 'FAIL (PE < count-10pp) — FORMAL NEGATIVE'}")
    out.append(f"**FF-CI-6 vs count20ep**: PE {pe_c:.3f} vs count {ct2_c:.3f}; criterion PE >= count-0.10 "
               f"(>= {ct2_c-0.10:.3f}) -> {'PASS' if ff6_20 else 'FAIL (PE < count-10pp) — FORMAL NEGATIVE'}")
    out.append("")

    # learning curves
    out.append("### GPU PE learning curves (episode: success/steps/victory_step)")
    for t in TASKS:
        out.append(f"\n**{t}** (GPU)")
        out.append("| ep | success | steps | victory_step |")
        out.append("|---|---|---|---|")
        for ep, ok, st, vs in steps_series(gpu, t):
            out.append(f"| {ep} | {ok} | {st} | {vs} |")
        pe = [r for r in gpu if r["task_id"] == t]
        e = completion([r for r in pe if r["episode"] < mid])
        l = completion([r for r in pe if r["episode"] >= mid])
        out.append(f"early(ep<{mid})={e:.3f} late(ep>={mid})={l:.3f} discovery_steps={discovery(pe)}")
    out.append("")
    # early/late pooled + discovery
    pe_e = completion([r for r in pe_all if r["episode"] < mid])
    pe_l = completion([r for r in pe_all if r["episode"] >= mid])
    ct_e = completion([r for r in count if r["episode"] < mid])
    ct_l = completion([r for r in count if r["episode"] >= mid])
    out.append(f"POOLED early/late: PE {pe_e:.3f}/{pe_l:.3f}; count10 {ct_e:.3f}/{ct_l:.3f}; "
               f"discovery PE={discovery(pe_all)} count10={discovery(count)}")
    out.append("")

    # CPU partial reference (read_secret only)
    cpe = [r for r in cpu if r["task_id"] == "read_secret_ci"]
    if cpe:
        out.append("### CPU partial reference (read_secret_ci, 10ep, killed-run snapshot)")
        out.append("| ep | success | steps | victory_step |")
        out.append("|---|---|---|---|")
        for ep, ok, st, vs in steps_series(cpu, "read_secret_ci"):
            out.append(f"| {ep} | {ok} | {st} | {vs} |")
        out.append(f"completion {completion(cpe):.3f}")

    txt = "\n".join(out)
    print(txt)
    (RES / "phase9_ci_m3m4_gpu_analysis.md").write_text(txt)
    frag = {
        "gpu_meta": meta, "n_ep": n_ep, "pe_pooled": pe_c, "count10_pooled": ct_c,
        "count20_pooled": ct2_c, "delta_vs_count10": pe_c - ct_c,
        "delta_vs_count20": pe_c - ct2_c, "ff6_vs_count10": ff6_10, "ff6_vs_count20": ff6_20,
        "per_task": {t: {"pe": completion([r for r in gpu if r["task_id"] == t]),
                         "count10": completion([r for r in count if r["task_id"] == t]),
                         "count20": completion([r for r in count20 if r["task_id"] == t])}
                     for t in TASKS},
    }
    (RES / "phase9_ci_m3m4_gpu_fragments.json").write_text(json.dumps(frag, indent=1))
    print("\nfragments -> results/phase9_ci_m3m4_gpu_fragments.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
