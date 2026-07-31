#!/usr/bin/env python3
# ruff: noqa: E402
"""Phase 9 Hypothesis-Generator: MVP signal validation (V1/V2/V3, F1-F4 gates).

Loads probe data from results/phase9_probes/seed{N}/, fits the STRIPS
discriminator on TRAIN-branch transitions only, and computes the pre-registered
metrics from PEDA_FINAL/phase9/PHASE9_PLAN.md:

    V1  Differential error : AUC_disc (held-out vs train labels from error)
    F1  Flatness gate      : KL(empirical error || uniform) on held-out probes
    F2  Separation gate    : AUC >= 0.7 AND KL(P_heldout || P_train) >= 0.3
    V2  Count-orthogonality: Cohen's d of e_disc(E) vs e_disc(D), zero-visit
    V3  Gradient (optional): Spearman rho of e_disc vs feature-distance

LLM direct baseline (AUC_llm) is SKIPPED in this MVP run (later phase).
No LoRA training. Output: JSON + Markdown report with WATCHDOG D4 metadata.

Usage:
    python scripts/phase9_validate_signal.py [--probe-dir results/phase9_probes]
        [--out results/phase9_signal_validation]
"""

import argparse
import datetime
import json
import socket
import subprocess
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from phase9.discriminator import MLPDiscriminator, STRIPSDiscriminator
from phase9.types import OutcomePredicates, Transition, state_from_dict
from phase9.validation import auc, cohens_d, describe, kl_between, kl_empirical_vs_uniform, spearman

TRAIN_PREFIXES = ("/sandbox", "/sandbox/docs", "/sandbox/data")


def is_train_cwd(cwd: str) -> bool:
    return cwd == "/sandbox" or cwd.startswith("/sandbox/docs") or cwd.startswith("/sandbox/data")


def _get_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, timeout=5, cwd=_PROJECT_ROOT).stdout.strip()
    except Exception:
        return "unknown"


def load_transitions(path: Path) -> list:
    transitions = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        if "meta" in d:
            continue
        transitions.append(Transition(
            state=state_from_dict(d["state"]),
            action=d["action"],
            next_state=state_from_dict(d["next_state"]),
            ground_truth=OutcomePredicates.from_dict(d["ground_truth"]),
            success=bool(d.get("success", False)),
        ))
    return transitions


def load_probes(path: Path) -> list:
    probes = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        if "meta" in d:
            continue
        probes.append({
            "cwd": d["state"]["cwd"],
            "files": d["state"]["files"],
            "action": d["action"],
            "distance": int(d.get("distance", 0)),
            "gt": OutcomePredicates.from_dict(d["ground_truth"]),
        })
    return probes


def evaluate_seed(seed_dir: Path) -> dict:
    """Fit discriminators on train transitions; score D/E probes."""
    transitions = load_transitions(seed_dir / "transitions.jsonl")
    train_trans = [t for t in transitions if is_train_cwd(t.state.cwd)]
    d_probes = load_probes(seed_dir / "probes_D.jsonl")
    e_probes = load_probes(seed_dir / "probes_E.jsonl")

    strips = STRIPSDiscriminator()
    strips.update(train_trans)
    mlp = MLPDiscriminator(seed=int(seed_dir.name.replace("seed", "")))
    mlp.update(train_trans)

    per_probe = []
    err_d, err_e = [], []
    for cls, probes in (("D", d_probes), ("E", e_probes)):
        for p in probes:
            st = state_from_dict({"cwd": p["cwd"], "files": p["files"]})
            vd = strips.predict(st, p["action"])
            vm = mlp.predict(st, p["action"])
            ed = OutcomePredicates.hamming(vd.predicates, p["gt"])
            em = OutcomePredicates.hamming(vm.predicates, p["gt"])
            (err_d if cls == "D" else err_e).append(ed)
            per_probe.append({
                "cls": cls, "cwd": p["cwd"], "action": p["action"],
                "distance": p["distance"], "e_disc": round(ed, 4), "e_mlp": round(em, 4),
                "gt": p["gt"].to_dict(), "pred": vd.predicates.to_dict(),
                "conf": round(vd.confidence, 4),
            })

    labels = [0] * len(err_d) + [1] * len(err_e)
    errors = err_d + err_e
    auc_disc = auc(labels, errors) if len(labels) > 1 else 0.5

    # Secondary AUC over rollout transitions (train-branch=0, held-out-branch=1)
    held_trans = [t for t in transitions if not is_train_cwd(t.state.cwd)]
    err_train_tr, err_held_tr = [], []
    for t in train_trans:
        v = strips.predict(t.state, t.action)
        err_train_tr.append(OutcomePredicates.hamming(v.predicates, t.ground_truth))
    for t in held_trans:
        v = strips.predict(t.state, t.action)
        err_held_tr.append(OutcomePredicates.hamming(v.predicates, t.ground_truth))
    auc_trans = auc([0] * len(err_train_tr) + [1] * len(err_held_tr),
                    err_train_tr + err_held_tr) if err_train_tr and err_held_tr else None

    return {
        "n_train_transitions": len(train_trans),
        "n_held_transitions": len(held_trans),
        "n_D": len(err_d), "n_E": len(err_e),
        "auc_disc": round(auc_disc, 4),
        "auc_llm": None,  # skipped in MVP (later phase)
        "auc_transitions_secondary": round(auc_trans, 4) if auc_trans is not None else None,
        "kl_uniform_heldout": round(kl_empirical_vs_uniform(err_e, bins=10), 4),
        "kl_heldout_train": round(kl_between(err_e, err_d, bins=10), 4),
        "cohens_d": round(cohens_d(err_e, err_d), 4),
        "spearman_v3": round(spearman(errors, [p["distance"] for p in per_probe]), 4),
        "err_D": describe(err_d),
        "err_E": describe(err_e),
        "per_probe": per_probe,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 9 signal validation")
    ap.add_argument("--probe-dir", default="results/phase9_probes")
    ap.add_argument("--out", default="results/phase9_signal_validation")
    args = ap.parse_args()

    probe_dir = Path(args.probe_dir)
    seed_dirs = sorted(probe_dir.glob("seed*"))
    if not seed_dirs:
        print(f"[validate] no seed dirs found under {probe_dir}; "
              f"run scripts/phase9_collect_probes.py first")
        sys.exit(1)

    per_seed = {}
    for sd in seed_dirs:
        seed = int(sd.name.replace("seed", ""))
        print(f"[validate] seed {seed}...", flush=True)
        per_seed[str(seed)] = evaluate_seed(sd)

    # Aggregate mean/SD + sign stability
    def agg(key):
        vals = [per_seed[s][key] for s in per_seed if per_seed[s][key] is not None]
        if not vals:
            return None
        import statistics
        return {"mean": round(statistics.mean(vals), 4),
                "sd": round(statistics.pstdev(vals), 4),
                "per_seed": vals}

    gates = {
        "F1_flatness": {"threshold": "KL >= 0.35 nats", "metric": "kl_uniform_heldout"},
        "F2_separation": {"threshold": "AUC >= 0.7 AND KL(P_heldout||P_train) >= 0.3",
                          "metric": "auc_disc + kl_heldout_train"},
        "F3_count_orthogonality": {"threshold": "Cohen's d >= 1.0", "metric": "cohens_d"},
        "V3_gradient_optional": {"threshold": "Spearman rho > 0.4", "metric": "spearman_v3"},
    }
    gate_results = {}
    for gate, spec in gates.items():
        gate_results[gate] = {s: _gate_pass(gate, per_seed[s]) for s in per_seed}

    report = {
        "meta": {
            "phase": "9",
            "direction": "hypothesis-generator",
            "commit": _get_commit(),
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "host": socket.gethostname(),
            "cpu_or_gpu": "cpu",
            "sandbox_image": "peda-sandbox:v4",
            "model": "STRIPSDiscriminator (primary) + MLPDiscriminator (arm); LLM baseline skipped",
            "seeds": [int(s) for s in per_seed],
            "per_episode_data_present": True,
        },
        "per_seed": per_seed,
        "aggregate": {k: agg(k) for k in
                      ("auc_disc", "auc_transitions_secondary", "kl_uniform_heldout",
                       "kl_heldout_train", "cohens_d", "spearman_v3")},
        "gates": gate_results,
        "llm_baseline": "SKIPPED (MVP; later phase)",
    }

    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    json_path = Path(args.out + f"_{ts}.json")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    md_path = json_path.with_suffix(".md")
    md_path.write_text(render_markdown(report))

    print(f"[validate] wrote {json_path}")
    print(f"[validate] wrote {md_path}")
    print(_summary_table(report))


def _gate_pass(gate: str, seed: dict) -> str:
    if gate == "F1_flatness":
        return "PASS" if seed["kl_uniform_heldout"] >= 0.35 else "FAIL"
    if gate == "F2_separation":
        return "PASS" if (seed["auc_disc"] >= 0.7 and seed["kl_heldout_train"] >= 0.3) else "FAIL"
    if gate == "F3_count_orthogonality":
        return "PASS" if seed["cohens_d"] >= 1.0 else "FAIL"
    if gate == "V3_gradient_optional":
        return "PASS" if seed["spearman_v3"] > 0.4 else "FAIL"
    return "?"


def _summary_table(report: dict) -> str:
    lines = ["[validate] per-seed summary:"]
    header = f"{'seed':>4} {'AUC_disc':>9} {'KL_unif':>8} {'KL_h|t':>8} {'d(E,D)':>7} {'rho':>6}  gates"
    lines.append(header)
    for s, r in sorted(report["per_seed"].items()):
        g = " ".join("P" if report["gates"][k][s] == "PASS" else "F"
                     for k in report["gates"])
        lines.append(f"{s:>4} {r['auc_disc']:>9} {r['kl_uniform_heldout']:>8} "
                     f"{r['kl_heldout_train']:>8} {r['cohens_d']:>7} {r['spearman_v3']:>6}  {g}")
    return "\n".join(lines)


def render_markdown(report: dict) -> str:
    lines = ["# Phase 9 Hypothesis-Generator — MVP Signal Validation", ""]
    lines.append(f"- commit: `{report['meta']['commit']}`")
    lines.append(f"- timestamp: {report['meta']['timestamp']}")
    lines.append(f"- host: {report['meta']['host']} | {report['meta']['cpu_or_gpu']}")
    lines.append(f"- sandbox_image: {report['meta']['sandbox_image']}")
    lines.append(f"- model: {report['meta']['model']}")
    lines.append(f"- seeds: {report['meta']['seeds']}")
    lines.append(f"- LLM direct baseline: {report['llm_baseline']}")
    lines.append("")
    lines.append("| Metric | Threshold | per-seed | mean | SD |")
    lines.append("|---|---|---|---|---|")
    rows = [
        ("V1 AUC_disc", ">= 0.7", "auc_disc"),
        ("F1 KL(emp || uniform) [held-out]", ">= 0.35 nats", "kl_uniform_heldout"),
        ("F2 KL(P_heldout || P_train)", ">= 0.3 nats", "kl_heldout_train"),
        ("V2 Cohen's d (E vs D)", ">= 1.0", "cohens_d"),
        ("V3 Spearman rho", "> 0.4", "spearman_v3"),
    ]
    for label, thresh, key in rows:
        agg = report["aggregate"].get(key)
        if agg is None:
            continue
        lines.append(f"| {label} | {thresh} | {agg['per_seed']} | {agg['mean']} | {agg['sd']} |")
    lines.append("")
    lines.append("## Fail-fast gates")
    lines.append("")
    lines.append("| Gate | Condition | Dead when |")
    lines.append("|---|---|---|")
    lines.append("| FF-HG-1 (F1) | KL(emp || uniform) >= 0.35 | flat error field |")
    lines.append("| FF-HG-2 (F2) | AUC >= 0.7 AND KL >= 0.3 | no novel/known separation |")
    lines.append("| FF-HG-3 (F3) | Cohen's d >= 1.0 | error = visit-count only |")
    lines.append("| FF-HG-4 (F4) | proposer diversity | N/A in MVP (no LLM) |")
    lines.append("")
    for s in sorted(report["per_seed"]):
        r = report["per_seed"][s]
        g = ", ".join(f"{k}={report['gates'][k][s]}" for k in report["gates"])
        lines.append(f"### seed {s}: {g}")
        lines.append("")
        lines.append(f"- n transitions: train={r['n_train_transitions']}, held-out={r['n_held_transitions']}")
        lines.append(f"- probes: D={r['n_D']}, E={r['n_E']}")
        lines.append(f"- AUC_disc={r['auc_disc']} (secondary over transitions: {r['auc_transitions_secondary']})")
        lines.append(f"- KL(emp||uniform)={r['kl_uniform_heldout']} nats; "
                     f"KL(heldout||train)={r['kl_heldout_train']} nats")
        lines.append(f"- Cohen's d(E,D)={r['cohens_d']}; Spearman rho={r['spearman_v3']}")
        lines.append(f"- error D: mean={r['err_D']['mean']} sd={r['err_D']['sd']}; "
                     f"error E: mean={r['err_E']['mean']} sd={r['err_E']['sd']}")
        lines.append("")
    lines.append("## Verdict (per PHASE9_PLAN.md)")
    lines.append("")
    lines.append("FF-HG-1/2/3 verdicts: see gate columns above. "
                 "Any FAIL with its measured value is a pre-registered dead-end "
                 "condition — no post-hoc threshold adjustment permitted.")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
