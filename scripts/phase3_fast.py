#!/usr/bin/env python3
# ruff: noqa: E402
"""Phase 3 Epistemic Validation - Fast Grid World (no ensemble checkpoints).

Epistemic signal comes from model's own confidence (epistemic_ratio = 1.0 - confidence),
NOT from ensemble variance. This avoids slow checkpoint adapter loading/unloading.

Runs 4 conditions (goal_known/goal_unknown x PEDA/Pragmatic) with N>=10 each.
Saves report to results/phase3_experiment/report.json

Usage:
  source venv/bin/activate
  HF_HUB_OFFLINE=1 python scripts/phase3_fast.py
"""

import json
import os
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from phase1.drive_system import ActionGenerator, HomeostaticDriveSystem
from phase1.grid_env import GridWorld
from phase1.run import run_episode
from phase1.types import DriveWeights
from phase1.world_model import EnsembleErrorComputer, LearningModule, WorldModel

# ── Config ──────────────────────────────────────────────────────────
DRIVE_WEIGHTS = DriveWeights(curiosity=0.5, competence=0.5, boredom=0.5, novelty=0.5)
PRAGMATIC_WEIGHT = 3.0
ADAPTER_PATH = "checkpoints/phase1/partial_adapter_real_25_e3"
RESULTS_DIR = Path("results/phase3_experiment")
N_EPISODES = 10
MAX_STEPS = 50


def load_manifest(adapter_path: Path):
    manifest_path = adapter_path / "trained_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    known_cells = {tuple(c) for c in manifest["known_cells"]}
    all_cells = [tuple(c) for c in manifest["all_cells"]]
    return known_cells, all_cells, manifest.get("train_fraction", 0.25)


def sample_goal(condition, rng, known_cells_list, all_cells):
    if condition == "goal_known":
        return tuple(rng.choice(known_cells_list))
    return tuple(rng.choice([c for c in all_cells if tuple(c) not in known_cells_list]))


def sample_untrained_start(env, base_seed, known_cells, max_trials=1000):
    rng = __import__("random").Random(base_seed)
    for _ in range(max_trials):
        seed = rng.randint(0, 2**31)
        state = env.reset(seed)
        if state.agent not in known_cells:
            return state, seed
    return env.reset(0), 0


def main():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    adapter_path = Path(ADAPTER_PATH)
    known_cells, all_cells, train_fraction = load_manifest(adapter_path)
    known_cells_list = [tuple(c) for c in known_cells]

    print(f"[phase3] Adapter: {adapter_path}")
    print(f"[phase3] Known cells: {len(known_cells)} / {len(all_cells)}")
    print(f"[phase3] Drive weights: cur={DRIVE_WEIGHTS.curiosity} cmp={DRIVE_WEIGHTS.competence} "
          f"bor={DRIVE_WEIGHTS.boredom} nov={DRIVE_WEIGHTS.novelty}")
    print(f"[phase3] Pragmatic weight: {PRAGMATIC_WEIGHT}")
    print(f"[phase3] Episodes per condition: {N_EPISODES}")

    print("[phase3] Loading WorldModel (first inference will be slow ~3min)...", flush=True)
    t0 = time.time()
    wm = WorldModel(
        model_name=os.path.expanduser("~/models/Qwen2.5-0.5B-Instruct"),
        adapter_path=str(adapter_path),
    )
    if wm.mode == "stub" or wm.model is None:
        raise RuntimeError("WorldModel failed to load real LLM")
    print(f"[phase3] Model loaded in {time.time()-t0:.1f}s", flush=True)

    # Empty ensemble — epistemic signal from model's own confidence
    ec = EnsembleErrorComputer(wm)
    ec.checkpoints = []

    # Learning module disabled (no online learning)
    lm = LearningModule(wm, ec, buffer_size=1000, update_interval=100000)

    results = {}
    total_start = time.time()

    for condition in ["goal_known", "goal_unknown"]:
        print(f"\n[phase3] {'='*60}", flush=True)
        print(f"[phase3] CONDITION: {condition}", flush=True)
        print(f"[phase3] {'='*60}", flush=True)

        rng = __import__("random").Random(123 + (0 if condition == "goal_known" else 1000000))
        cond_results = {"peda": [], "pragmatic_only": []}

        for ep in range(N_EPISODES):
            goal = sample_goal(condition, rng, known_cells_list, all_cells)
            env = GridWorld(goal=goal, max_steps=MAX_STEPS)
            base_seed = 42 + ep + (0 if condition == "goal_known" else 1000000)
            start_state, start_seed = sample_untrained_start(env, base_seed, known_cells)
            print(f"  ep{ep}: condition={condition} goal={goal} start={start_state.agent} ", end="", flush=True)

            # PEDA
            ds_ped = HomeostaticDriveSystem(DRIVE_WEIGHTS)
            ag_ped = ActionGenerator(
                wm, error_computer=ec, drive_system=ds_ped,
                pragmatic_only=False, pragmatic_weight=PRAGMATIC_WEIGHT,
                max_candidates=4, horizon=1,
            )
            traj_ped, preds_ped, ah_ped, metrics_ped = run_episode(
                env, wm, ec, ds_ped, lm, ag_ped, start_seed,
            )
            peda_ok = 1.0 if metrics_ped["success"] else 0.0

            # Pragmatic-only
            ds_prag = HomeostaticDriveSystem(DRIVE_WEIGHTS)
            ag_prag = ActionGenerator(
                wm, error_computer=ec, drive_system=ds_prag,
                pragmatic_only=True, pragmatic_weight=PRAGMATIC_WEIGHT,
                max_candidates=4, horizon=1,
            )
            traj_prag, preds_prag, ah_prag, metrics_prag = run_episode(
                env, wm, ec, ds_prag, lm, ag_prag, start_seed,
            )
            prag_ok = 1.0 if metrics_prag["success"] else 0.0

            # Compute revisit rate
            visited = set()
            revisit_ped = sum(1 for s in traj_ped if s.agent in visited or visited.add(s.agent))
            visited = set()
            revisit_prag = sum(1 for s in traj_prag if s.agent in visited or visited.add(s.agent))

            peda_result = {
                "success": peda_ok, "steps": metrics_ped["steps"],
                "revisit_rate": revisit_ped / max(metrics_ped["steps"], 1),
                "mean_epistemic_error": metrics_ped.get("mean_epistemic_error", 0),
            }
            prag_result = {
                "success": prag_ok, "steps": metrics_prag["steps"],
                "revisit_rate": revisit_prag / max(metrics_prag["steps"], 1),
                "mean_epistemic_error": metrics_prag.get("mean_epistemic_error", 0),
            }

            # Warm up the model with first prediction to avoid cold-start delay in run_episode
            if ep == 0 and condition == "goal_known":
                pass  # run_episode will handle it

            cond_results["peda"].append(peda_result)
            cond_results["pragmatic_only"].append(prag_result)

            print(f"PEDA steps={metrics_ped['steps']} success={metrics_ped['success']} | "
                  f"Prag steps={metrics_prag['steps']} success={metrics_prag['success']}", flush=True)

        results[condition] = cond_results

    total_elapsed = time.time() - total_start
    print(f"\n[phase3] All episodes complete in {total_elapsed:.0f}s ({total_elapsed/60:.1f}min)", flush=True)

    # ── Aggregate statistics ──
    report = {
        "experiment": "Phase 3 Epistemic Validation (Grid World, confidence-based epistemic)",
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "adapter": str(adapter_path),
        "train_fraction": train_fraction,
        "ensemble_checkpoints": 0,
        "drive_weights": {"curiosity": DRIVE_WEIGHTS.curiosity, "competence": DRIVE_WEIGHTS.competence,
                          "boredom": DRIVE_WEIGHTS.boredom, "novelty": DRIVE_WEIGHTS.novelty},
        "pragmatic_weight": PRAGMATIC_WEIGHT,
        "episodes_per_condition": N_EPISODES,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total_elapsed_s": round(total_elapsed),
    }

    for condition in ["goal_known", "goal_unknown"]:
        report[condition] = {}
        for agent_key, agent_label in [("peda", "PEDA"), ("pragmatic_only", "Pragmatic-only")]:
            eps = results[condition][agent_key]
            n = len(eps)
            successes = sum(e["success"] for e in eps)
            steps_list = [e["steps"] for e in eps]
            mean_steps = sum(steps_list) / n if n else 0
            std_steps = (sum((s - mean_steps)**2 for s in steps_list) / n) ** 0.5 if n else 0

            report[condition][agent_key] = {
                "n": n, "successes": successes,
                "success_rate": round(successes / n, 4) if n else 0,
                "mean_steps": round(mean_steps, 2),
                "std_steps": round(std_steps, 2),
                "median_steps": sorted(steps_list)[n // 2] if n else 0,
                "min_steps": min(steps_list) if steps_list else 0,
                "max_steps": max(steps_list) if steps_list else 0,
                "mean_revisit_rate": round(sum(e["revisit_rate"] for e in eps) / n, 4) if n else 0,
            }

    # ── Statistical tests ──
    from phase3_analysis import fisher_exact, mann_whitney_u

    peda_uk = results["goal_unknown"]["peda"]
    prag_uk = results["goal_unknown"]["pragmatic_only"]
    peda_ok = sum(e["success"] for e in peda_uk)
    prag_ok = sum(e["success"] for e in prag_uk)

    p_fisher = fisher_exact(peda_ok, prag_ok, N_EPISODES - peda_ok, N_EPISODES - prag_ok)
    u_stat, p_mw, z_mw = mann_whitney_u(
        [e["steps"] for e in peda_uk],
        [e["steps"] for e in prag_uk]
    )

    # Fairness check: goal_known should not differ
    peda_k = results["goal_known"]["peda"]
    prag_k = results["goal_known"]["pragmatic_only"]
    peda_k_ok = sum(e["success"] for e in peda_k)
    prag_k_ok = sum(e["success"] for e in prag_k)
    p_fair = fisher_exact(peda_k_ok, prag_k_ok, N_EPISODES - peda_k_ok, N_EPISODES - prag_k_ok)

    report["statistical_tests"] = {
        "goal_unknown_success_fisher": {
            "p_value": round(p_fisher, 4),
            "peda": f"{peda_ok}/{N_EPISODES} ({100*peda_ok/N_EPISODES:.0f}%)",
            "pragmatic": f"{prag_ok}/{N_EPISODES} ({100*prag_ok/N_EPISODES:.0f}%)",
            "significant_at_005": p_fisher < 0.05,
        },
        "goal_unknown_steps_mannwhitney": {
            "p_value": round(p_mw, 4),
            "peda_mean_steps": round(sum(e["steps"] for e in peda_uk) / N_EPISODES, 1),
            "pragmatic_mean_steps": round(sum(e["steps"] for e in prag_uk) / N_EPISODES, 1),
            "significant_at_005": p_mw < 0.05,
        },
        "goal_known_fairness": {
            "p_value": round(p_fair, 4),
            "fairness_pass": p_fair > 0.05,
        },
    }

    # ── Success criteria ──
    peda_sr = peda_ok / N_EPISODES
    prag_sr = prag_ok / N_EPISODES
    peda_means = sum(e["steps"] for e in peda_uk) / N_EPISODES
    prag_means = sum(e["steps"] for e in prag_uk) / N_EPISODES

    criteria = {
        "peda_goal_unknown_success_gt_60pct": peda_sr > 0.6,
        "prag_goal_unknown_success_lt_40pct": prag_sr < 0.4,
        "peda_goal_unknown_steps_lt_10": peda_means < 10,
        "prag_goal_unknown_steps_gt_15": prag_means > 15,
        "goal_known_fairness_pass": p_fair > 0.05,
        "fisher_p_lt_005": p_fisher < 0.05,
        "mannwhitney_p_lt_005": p_mw < 0.05,
    }
    passed = sum(1 for v in criteria.values() if v)
    report["success_criteria"] = criteria
    report["passed_criteria"] = f"{passed}/{len(criteria)}"

    # ── Verdict ──
    if p_fisher < 0.05 and p_mw < 0.05 and peda_sr > prag_sr:
        report["verdict"] = "CORE_HYPOTHESIS_SUPPORTED"
        report["verdict_confidence"] = "HIGH"
    elif p_fisher < 0.05 and peda_sr > prag_sr:
        report["verdict"] = "CORE_HYPOTHESIS_SUPPORTED"
        report["verdict_confidence"] = "MODERATE"
    elif peda_sr > prag_sr:
        report["verdict"] = "DIRECTIONAL_SIGNAL"
        report["verdict_confidence"] = "LOW"
    else:
        report["verdict"] = "CORE_HYPOTHESIS_NOT_SUPPORTED"
        report["verdict_confidence"] = "N/A"

    report["verdict_reason"] = (
        f"PEDA {peda_ok}/{N_EPISODES} ({100*peda_sr:.0f}%) vs Pragmatic {prag_ok}/{N_EPISODES} ({100*prag_sr:.0f}%) "
        f"success in goal_unknown (Fisher p={p_fisher:.4f}, MW p={p_mw:.4f}). "
        f"Goal_known fairness: PEDA {peda_k_ok}/{N_EPISODES} vs Pragmatic {prag_k_ok}/{N_EPISODES} (p={p_fair:.4f})"
    )

    # ── Save ──
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = RESULTS_DIR / "report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\n[phase3] Report saved to {report_path}", flush=True)

    # ── Summary ──
    print(f"\n{'='*60}", flush=True)
    print("PHASE 3 - RESULTS SUMMARY", flush=True)
    print(f"{'='*60}", flush=True)
    for condition in ["goal_known", "goal_unknown"]:
        print(f"\n--- {condition} ---", flush=True)
        for agent_key, agent_label in [("peda", "PEDA"), ("pragmatic_only", "Pragmatic-only")]:
            d = report[condition][agent_key]
            print(f"  {agent_label:15s} N={d['n']} success={d['success_rate']:.3f} "
                  f"steps={d['mean_steps']:.1f}+-{d['std_steps']:.1f} revisit={d['mean_revisit_rate']:.3f}", flush=True)
    print(f"\nVerdict: {report['verdict']} (confidence: {report.get('verdict_confidence', 'N/A')})", flush=True)
    print(f"Reason: {report['verdict_reason']}", flush=True)
    print(f"\nSuccess criteria passed: {passed}/{len(criteria)}", flush=True)
    print(f"Total time: {total_elapsed:.0f}s ({total_elapsed/60:.1f}min)", flush=True)


if __name__ == "__main__":
    main()
