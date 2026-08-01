#!/usr/bin/env python3
# ruff: noqa: E402
"""M3 — Behavioral productivity: PE agent vs count baseline on CI tasks.

Pre-registered (PEDA_FINAL/phase9/PHASE9_PLAN.md, plan-counter-intuitive-sandbox.md §4/§5):

    PE agent completion vs count baseline, 20 eps × 3 tasks
    PE ≥ count − 10pp; both improve ≥ 2× ep11-20 vs ep1-10;
    discovery steps ≤ 1.5× count

Count baseline: Phase8Runner (count-based novelty + success-cache replay, ci=True).
PE agent: run_peda_episode, EFE action selection, step reward always 0
(no task feedback — the harness's check() only gates episode termination,
never a reward signal). Same candidate generator + same action-priority priors.

Environment: peda-sandbox:counterintuitive-v2, fresh container per episode,
max_steps 20. Tasks: read_secret_ci / read_data_ci / find_warn_ci.

Outputs (D4 meta header + rows):
    results/phase9_ci_m3_count.jsonl
    results/phase9_ci_m3_peda.jsonl
    results/phase9_ci_m3_summary.csv
"""

import argparse
import csv
import datetime
import json
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch  # noqa: E402
from phase1.drive_system import ActionGenerator, HomeostaticDriveSystem
from phase1.types import DriveWeights
from phase1.world_model import EnsembleErrorComputer, WorldModel
from phase2.run import SandboxLearningModule, run_peda_episode
from phase2.sandbox_env import CounterIntuitiveSandbox, generate_sandbox_candidates
from phase2.tasks import MICRO_TASKS
from phase8.count_driven_agent import Phase8Runner

MODEL_PATH = "/home/data/models/Qwen2.5-0.5B-Instruct"
CI_TASKS = ["read_secret_ci", "read_data_ci", "find_warn_ci"]
NUM_EPISODES = 20
MAX_STEPS = 20

DRIVE_WEIGHTS = DriveWeights(
    curiosity=0.1, competence=2.0, boredom=0.1, novelty=2.0,
)
PRAGMATIC_WEIGHT = 3.0
UPDATE_INTERVAL = 20  # one LoRA update per 20-step episode (M4 mechanism: error must shrink in-episode)


def _build_peda_components(wm: WorldModel):
    """ActionGenerator + ErrorComputer + DriveSystem (phase4 pattern, no checkpoints)."""
    ec = EnsembleErrorComputer(wm)
    ec.checkpoints = []  # no ensemble checkpoints: single-model prior-variance fallback
    ds = HomeostaticDriveSystem(DRIVE_WEIGHTS)
    ag = ActionGenerator(
        wm, error_computer=ec, drive_system=ds,
        pragmatic_only=False,
        pragmatic_weight=PRAGMATIC_WEIGHT,
        max_candidates=5, horizon=1,
        goal_predicate=None,
    )
    return ag, ec, ds


def _peda_agent_fn(ag: ActionGenerator):
    def agent_fn(state, action_history):
        cands = generate_sandbox_candidates(state)
        return ag.select_action(state, action_history, cands)
    return agent_fn


def _run_peda_tasks(wm: WorldModel, task_id: str, device: str,
                    num_episodes: int = NUM_EPISODES, max_steps: int = MAX_STEPS) -> List[Dict]:
    """Run NUM_EPISODES PEDA episodes on one CI task (shared learning module)."""
    ag, ec, ds = _build_peda_components(wm)
    lm = SandboxLearningModule(
        wm, ec, buffer_size=200, update_interval=UPDATE_INTERVAL,
    )
    agent_fn = _peda_agent_fn(ag)
    sb = CounterIntuitiveSandbox()
    rows: List[Dict] = []
    for ep in range(num_episodes):
        steps, final_state, metrics = run_peda_episode(
            sb, wm, ec, ds, lm, agent_fn,
            max_steps=max_steps, task_id=task_id,
        )
        rows.append({
            "task_id": task_id,
            "episode": ep,
            "success": bool(metrics["success"]),
            "steps": metrics["steps"],
            "mean_epistemic_error": metrics["mean_epistemic_error"],
            "mean_aleatoric_error": metrics["mean_aleatoric_error"],
            "victory_step": steps[-1]["step"] if (metrics["success"] and steps) else None,
        })
        print(f"[peda {task_id}] ep {ep+1:02d}/{num_episodes} success={metrics['success']} "
              f"steps={metrics['steps']} epi_err={metrics['mean_epistemic_error']:.3f}",
              flush=True)
    sb.close()
    return rows


def _run_count_tasks(task_id: str, num_episodes: int = NUM_EPISODES,
                     max_steps: int = MAX_STEPS) -> List[Dict]:
    runner = Phase8Runner(task_id=task_id, ci=True, model_path=None)
    rows: List[Dict] = []
    for ep in range(num_episodes):
        res = runner.run_episode(max_steps=max_steps)
        rows.append({
            "task_id": task_id,
            "episode": ep,
            "success": bool(res.success),
            "steps": res.steps,
            "victory_step": res.victory_step if hasattr(res, "victory_step") else None,
        })
        print(f"[count {task_id}] ep {ep+1:02d}/{num_episodes} success={res.success} steps={res.steps}",
              flush=True)
    return rows


def _write_jsonl(path: Path, meta: Dict, rows: List[Dict]) -> None:
    with path.open("w") as f:
        f.write(json.dumps({"meta": meta}, ensure_ascii=False) + "\n")
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _completion(rows: List[Dict]) -> float:
    return sum(1 for r in rows if r["success"]) / len(rows) if rows else 0.0


def _discovery_steps(rows: List[Dict]) -> Optional[int]:
    succ = [r for r in rows if r["success"]]
    if not succ:
        return None
    return min(r.get("victory_step") if r.get("victory_step") is not None
               else r["steps"] for r in succ)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", choices=["count", "peda", "both"], default="both")
    ap.add_argument("--task", choices=CI_TASKS, default=None)
    ap.add_argument("--model", default=MODEL_PATH)
    ap.add_argument("--num-episodes", type=int, default=NUM_EPISODES)
    ap.add_argument("--max-steps", type=int, default=MAX_STEPS)
    ap.add_argument("--only-count", action="store_true", help="skip LLM load (count only)")
    args = ap.parse_args()

    # Docker image must exist
    r = subprocess.run(["docker", "image", "inspect", "peda-sandbox:counterintuitive-v2"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print("FATAL: docker image missing: peda-sandbox:counterintuitive-v2", file=sys.stderr)
        return 2

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tasks = [args.task] if args.task else CI_TASKS
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                            capture_output=True, text=True).stdout.strip()
    meta = {
        "phase": "9", "direction": "counter-intuitive-sandbox",
        "experiment": "M3_behavioral_productivity",
        "commit": commit, "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "host": socket.gethostname(), "cpu_or_gpu": device,
        "sandbox_images": ["peda-sandbox:counterintuitive-v2"],
        "model": MODEL_PATH, "seeds": [],
        "per_episode_data_present": True,
        "tasks": CI_TASKS, "num_episodes": args.num_episodes, "max_steps": args.max_steps,
        "threshold": {"pe_minus_count_pp": -10, "improve_2x": True, "discovery_ratio_max": 1.5},
        "peda_config": {"update_interval": UPDATE_INTERVAL, "pragmatic_weight": PRAGMATIC_WEIGHT,
                        "horizon": 1, "max_candidates": 5, "reward_always_zero": True},
    }

    wm = None
    if not args.only_count and args.agent in ("peda", "both"):
        print(f"Loading {args.model} (device={device}) ...", flush=True)
        wm = WorldModel(args.model, device=device)
        if wm.mode != "llm" or wm.model is None:
            print("FATAL: model fell back to stub", file=sys.stderr)
            return 2
        # fp32: fp16 LoRA finetune produces NaN loss (M2 finding); keep fp32 on GPU

    all_rows: Dict[str, List[Dict]] = {"count": [], "peda": []}
    for task in tasks:
        if args.agent in ("count", "both"):
            all_rows["count"].extend(_run_count_tasks(task, args.num_episodes, args.max_steps))
        if args.agent in ("peda", "both") and wm is not None:
            all_rows["peda"].extend(_run_peda_tasks(wm, task, device, args.num_episodes, args.max_steps))

    for agent, rows in all_rows.items():
        if rows:
            _write_jsonl(REPO_ROOT / "results" / f"phase9_ci_m3_{agent}.jsonl", meta, rows)

    # ── Summary ──
    summary_rows: List[List[Any]] = []
    if all_rows["peda"] and all_rows["count"]:
        for task in tasks:
            pe = [r for r in all_rows["peda"] if r["task_id"] == task]
            ct = [r for r in all_rows["count"] if r["task_id"] == task]
            pe_c, ct_c = _completion(pe), _completion(ct)
            pe_early = _completion([r for r in pe if r["episode"] < 10])
            pe_late = _completion([r for r in pe if r["episode"] >= 10])
            ct_early = _completion([r for r in ct if r["episode"] < 10])
            ct_late = _completion([r for r in ct if r["episode"] >= 10])
            pe_disc, ct_disc = _discovery_steps(pe), _discovery_steps(ct)
            summary_rows.append([
                task, f"{pe_c:.3f}", f"{ct_c:.3f}", pe_c - ct_c,
                f"{pe_early:.3f}", f"{pe_late:.3f}",
                f"{ct_early:.3f}", f"{ct_late:.3f}",
                pe_disc, ct_disc,
            ])
        pe_all, ct_all = all_rows["peda"], all_rows["count"]
        summary_rows.append([
            "POOLED", f"{_completion(pe_all):.3f}", f"{_completion(ct_all):.3f}",
            _completion(pe_all) - _completion(ct_all),
            f"{_completion([r for r in pe_all if r['episode'] < 10]):.3f}",
            f"{_completion([r for r in pe_all if r['episode'] >= 10]):.3f}",
            f"{_completion([r for r in ct_all if r['episode'] < 10]):.3f}",
            f"{_completion([r for r in ct_all if r['episode'] >= 10]):.3f}",
            _discovery_steps(pe_all), _discovery_steps(ct_all),
        ])
        pe_c, ct_c = _completion(pe_all), _completion(ct_all)
        m3_pass = pe_c >= ct_c - 0.10
        print("=" * 78)
        print(f"M3 verdict: {'PASS' if m3_pass else 'FAIL'} "
              f"(PE {pe_c:.3f} vs count {ct_c:.3f}; criterion PE >= count - 0.10)")
        print("=" * 78)
    else:
        summary_rows.append(["partial", "run both agents for full verdict"])

    csv_path = REPO_ROOT / "results" / "phase9_ci_m3_summary.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "pe_completion", "count_completion", "delta_pp",
                    "pe_ep1_10", "pe_ep11_20", "count_ep1_10", "count_ep11_20",
                    "pe_discovery_steps", "count_discovery_steps"])
        w.writerows(summary_rows)
    print(f"[m3] summary -> {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
