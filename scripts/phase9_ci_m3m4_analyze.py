#!/usr/bin/env python3
"""Post-hoc analysis for M3/M4 CI reports — reads the per-episode JSONLs and
emits the numbers table + verdicts that go into CI_M3M4_REPORT.md.

Usage: python3 scripts/phase9_ci_m3m4_analyze.py
Reads:
    results/phase9_ci_m3_count.jsonl      (canonical, matched 10ep)
    results/phase9_ci_m3_peda.jsonl       (PE 10ep, halved on CPU)
    results/phase9_ci_m3_count_20ep.jsonl (supplementary, full 20ep count)
    results/phase9_ci_m4.jsonl            (3 trials x 40 steps)
Writes: results/phase9_ci_m3m4_analysis.md
"""
import csv
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RES = REPO / "results"

TASKS = ["read_secret_ci", "read_data_ci", "find_warn_ci"]


def load_rows(path: Path):
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


def main():
    out = []
    count = load_rows(RES / "phase9_ci_m3_count.jsonl")
    peda = load_rows(RES / "phase9_ci_m3_peda.jsonl")
    count20 = load_rows(RES / "phase9_ci_m3_count_20ep.jsonl")
    m4 = load_rows(RES / "phase9_ci_m4.jsonl")

    out.append("# M3/M4 Analysis (auto-generated)")
    out.append("")

    # ── M3 table ──
    out.append("## M3 — PE vs count (matched 10ep; halved from 20ep on CPU)")
    out.append("")
    out.append("| task | pe_comp | count_comp | delta_pp | pe_ep1-5 | pe_ep6-10 | count_ep1-5 | count_ep6-10 | pe_disc | count_disc |")
    out.append("|---|---|---|---|---|---|---|---|---|---|")
    for t in TASKS:
        pe = [r for r in peda if r["task_id"] == t]
        ct = [r for r in count if r["task_id"] == t]
        pe_c = completion(pe)
        ct_c = completion(ct)
        pe_e = completion([r for r in pe if r["episode"] < 5])
        pe_l = completion([r for r in pe if r["episode"] >= 5])
        ct_e = completion([r for r in ct if r["episode"] < 5])
        ct_l = completion([r for r in ct if r["episode"] >= 5])
        out.append(f"| {t} | {pe_c:.3f} | {ct_c:.3f} | {pe_c - ct_c:+.3f} | "
                   f"{pe_e:.3f} | {pe_l:.3f} | {ct_e:.3f} | {ct_l:.3f} | "
                   f"{discovery(pe)} | {discovery(ct)} |")
    pe_c = completion(peda)
    ct_c = completion(count)
    pe_e = completion([r for r in peda if r["episode"] < 5])
    pe_l = completion([r for r in peda if r["episode"] >= 5])
    ct_e = completion([r for r in count if r["episode"] < 5])
    ct_l = completion([r for r in count if r["episode"] >= 5])
    out.append(f"| **POOLED** | **{pe_c:.3f}** | **{ct_c:.3f}** | **{pe_c - ct_c:+.3f}** | "
               f"**{pe_e:.3f}** | **{pe_l:.3f}** | **{ct_e:.3f}** | **{ct_l:.3f}** | "
               f"**{discovery(peda)}** | **{discovery(ct)}** |")
    out.append("")
    ff6 = pe_c >= ct_c - 0.10
    out.append(f"**FF-CI-6**: PE {pe_c:.3f} vs count {ct_c:.3f}; criterion PE >= count - 0.10 → "
               f"{'PASS (PE >= count - 10pp)' if ff6 else 'FAIL (PE < count - 10pp) — formal negative result'}")

    # Count 20ep supplementary (FF-CI-5 context)
    out.append("")
    out.append("## M3 supplementary — count full 20ep (FF-CI-5 context)")
    out.append("")
    out.append("| task | ep1-10 | ep11-20 | total |")
    out.append("|---|---|---|---|")
    for t in TASKS:
        rr = [r for r in count20 if r["task_id"] == t]
        out.append(f"| {t} | {sum(r['success'] for r in rr[:10])}/10 | "
                   f"{sum(r['success'] for r in rr[10:])}/10 | {sum(r['success'] for r in rr)}/{len(rr)} |")
    out.append(f"| POOLED | {sum(r['success'] for r in count20[:30])}/30 | "
               f"{sum(r['success'] for r in count20[30:])}/30 | {sum(r['success'] for r in count20)}/{len(count20)} |")
    out.append("")

    # ── M4 ──
    out.append("## M4 — per-step mean error E(t) = 1 - DLR (3 trials x 40 steps, update_interval=20)")
    out.append("")
    n_trials = max((r["trial"] for r in m4), default=-1) + 1
    n_steps = max((r["step"] for r in m4), default=-1) + 1
    e_by_step = []
    dlr_by_step = []
    for s in range(n_steps):
        dlrs = [r["dlr"] for r in m4 if r["step"] == s]
        if dlrs:
            dlr_by_step.append(sum(dlrs) / len(dlrs))
            e_by_step.append(1.0 - sum(dlrs) / len(dlrs))
        else:
            dlr_by_step.append(None)
            e_by_step.append(None)

    def mean_e(lo, hi):
        vals = [e for e in e_by_step[lo:hi] if e is not None]
        return sum(vals) / len(vals) if vals else None

    e_early = mean_e(0, 10)
    e_late = mean_e(30, 40)
    m4_pass = bool(e_early is not None and e_late is not None
                   and e_early >= 0.5 and e_late <= 0.5 * e_early)
    out.append(f"- trials completed: {n_trials}; steps per trial: {n_steps}")
    out.append(f"- E(1..10) = {e_early:.4f} (criterion >= 0.5: {'OK' if e_early is not None and e_early >= 0.5 else 'FAIL'})")
    out.append(f"- E(31..40) = {e_late:.4f} (criterion <= 0.5*E(1..10) = {0.5 * e_early:.4f}: "
               f"{'OK' if m4_pass else 'FAIL'})")
    out.append(f"- **M4 verdict: {'PASS' if m4_pass else 'FAIL'}**")
    out.append("")
    out.append("| step | mean_dlr | mean_e |")
    out.append("|---|---|---|")
    for s in range(n_steps):
        out.append(f"| {s+1} | {dlr_by_step[s]:.4f} | {e_by_step[s]:.4f} |")
    out.append("")

    txt = "\n".join(out) + "\n"
    (RES / "phase9_ci_m3m4_analysis.md").write_text(txt)
    print(txt)

    # ── Report-ready fragments ──
    frag = {}
    # M3 table
    lines = ["| task | pe_comp | count_comp | delta_pp | pe_ep1-5 | pe_ep6-10 | count_ep1-5 | count_ep6-10 | pe_disc | count_disc |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for t in TASKS:
        pe = [r for r in peda if r["task_id"] == t]
        ct = [r for r in count if r["task_id"] == t]
        pe_c, ct_c = completion(pe), completion(ct)
        pe_e = completion([r for r in pe if r["episode"] < 5])
        pe_l = completion([r for r in pe if r["episode"] >= 5])
        ct_e = completion([r for r in ct if r["episode"] < 5])
        ct_l = completion([r for r in ct if r["episode"] >= 5])
        lines.append(f"| {t} | {pe_c:.3f} | {ct_c:.3f} | {pe_c - ct_c:+.3f} | "
                     f"{pe_e:.3f} | {pe_l:.3f} | {ct_e:.3f} | {ct_l:.3f} | "
                     f"{discovery(pe)} | {discovery(ct)} |")
    pe_c, ct_c = completion(peda), completion(count)
    pe_e = completion([r for r in peda if r["episode"] < 5])
    pe_l = completion([r for r in peda if r["episode"] >= 5])
    ct_e = completion([r for r in count if r["episode"] < 5])
    ct_l = completion([r for r in count if r["episode"] >= 5])
    lines.append(f"| **POOLED** | **{pe_c:.3f}** | **{ct_c:.3f}** | **{pe_c - ct_c:+.3f}** | "
                 f"**{pe_e:.3f}** | **{pe_l:.3f}** | **{ct_e:.3f}** | **{ct_l:.3f}** | "
                 f"**{discovery(peda)}** | **{discovery(ct)}** |")
    frag["m3_table"] = "\n".join(lines)

    # count20 table
    lines = ["| task | ep1-10 | ep11-20 | total |", "|---|---|---|---|"]
    for t in TASKS:
        rr = [r for r in count20 if r["task_id"] == t]
        lines.append(f"| {t} | {sum(r['success'] for r in rr[:10])}/10 | "
                     f"{sum(r['success'] for r in rr[10:])}/10 | {sum(r['success'] for r in rr)}/{len(rr)} |")
    lines.append(f"| POOLED | {sum(r['success'] for r in count20[:30])}/30 | "
                 f"{sum(r['success'] for r in count20[30:])}/30 | {sum(r['success'] for r in count20)}/{len(count20)} |")
    frag["m3_count20_table"] = "\n".join(lines)

    # FF-CI-6
    ff6_pass = pe_c >= ct_c - 0.10
    frag["ff6"] = (f"**FF-CI-6**: PE {pe_c:.3f} vs count {ct_c:.3f} → "
                   f"{'PASS (PE ≥ count − 10pp; prediction error is a viable drive signal)' if ff6_pass
                    else 'FAIL (PE < count − 10pp) — FORMAL NEGATIVE RESULT, charter-accepted'}")

    # M4 table
    lines = ["| step | mean_dlr | mean_e |", "|---|---|---|"]
    for s in range(n_steps):
        lines.append(f"| {s+1} | {dlr_by_step[s]:.4f} | {e_by_step[s]:.4f} |")
    frag["m4_table"] = "\n".join(lines)

    (RES / "phase9_ci_m3m4_fragments.json").write_text(json.dumps(frag, indent=1))
    print("\nfragments -> results/phase9_ci_m3m4_fragments.json")


if __name__ == "__main__":
    main()
