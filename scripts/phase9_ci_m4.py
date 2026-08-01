#!/usr/bin/env python3
# ruff: noqa: E402
"""M4 — Error trajectory: per-step mean prediction error E(t) for the PE agent.

Pre-registered (PEDA_FINAL/phase9/PHASE9_PLAN.md, plan-counter-intuitive-sandbox.md §5):

    Per-step mean prediction error E(t) (1 − DLR over L1/L2/L3 components)
    E(1..10) ≥ 0.5;  E(31..40) ≤ 0.5·E(1..10)

Single 40-step episode on read_secret_ci (fresh container), repeated N_TRIALS
times with a fresh agent each trial. At every step we record the world-model's
predicted (exit_code, files_delta, output_nonempty) vs the actual outcome —
DLR = mean over the 3 components, E(t) = 1 − DLR(t). E(t) values are averaged
across trials. Mechanism check: the world model must demonstrably learn the
rules (error shrinks) within the 40-step window.

Outputs (D4 meta header + rows):
    results/phase9_ci_m4.jsonl
    results/phase9_ci_m4_summary.csv
"""

import argparse
import csv
import datetime
import json
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import torch  # noqa: E402
from phase1.drive_system import ActionGenerator, HomeostaticDriveSystem
from phase1.types import DriveWeights, ErrorVector, Experience
from phase1.world_model import EnsembleErrorComputer, WorldModel
from phase2.run import SandboxLearningModule
from phase2.sandbox_env import CounterIntuitiveSandbox, generate_sandbox_candidates
from phase9_ci_m1_real_llm import _extract_json, _fs_snapshot  # noqa: E402
from phase9_ci_m2_eval import _build_state_prompt, _predict_model  # noqa: E402

MODEL_PATH = "/home/data/models/Qwen2.5-0.5B-Instruct"
TASK_ID = "read_secret_ci"
MAX_STEPS = 40
N_TRIALS = 3
UPDATE_INTERVAL = 20  # same as M3: one LoRA update mid-episode

DRIVE_WEIGHTS = DriveWeights(
    curiosity=0.1, competence=2.0, boredom=0.1, novelty=2.0,
)
PRAGMATIC_WEIGHT = 3.0


def _build_agent(wm: WorldModel):
    ec = EnsembleErrorComputer(wm)
    ec.checkpoints = []
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


def _dlr_components(wm, state, action, actual) -> Dict[str, Any]:
    """Predict (exit_code, files_delta, output_nonempty); compare vs actual.

    Mirrors M2 eval's _eval_model_row component scoring:
    L1 exit_code exact, L2 files_delta bool, L3 output_nonempty bool.
    """
    parsed, _raw = _predict_model(wm, state, action)
    pred = parsed or {}
    l1 = pred.get("exit_code") == actual["exit_code"]
    l2 = bool(pred.get("files")) == bool(actual["delta"])
    l3 = bool(pred.get("summary") or pred.get("last_output")) == bool(actual["output_nonempty"])
    dlr = sum((l1, l2, l3)) / 3.0
    return {"l1": l1, "l2": l2, "l3": l3, "dlr": dlr}


def _run_trial(wm: WorldModel, device: str, max_steps: int = MAX_STEPS) -> List[Dict]:
    ag, ec, ds = _build_agent(wm)
    lm = SandboxLearningModule(wm, ec, buffer_size=200, update_interval=UPDATE_INTERVAL)
    agent_fn = _peda_agent_fn(ag)
    sb = CounterIntuitiveSandbox()
    state = sb.reset(start_cwd=None)
    action_history: List[str] = []
    steps: List[Dict] = []

    for step_i in range(max_steps):
        if state.game_over:
            break
        action = agent_fn(state, action_history)
        action_str = action if isinstance(action, str) else action.name

        # World-model prediction BEFORE executing (components vs actual after).
        before = _fs_snapshot(state.container_id)
        parsed, _ = _predict_model(wm, state, action_str)
        pred = parsed or {}
        pred_exit = pred.get("exit_code")
        pred_files = bool(pred.get("files"))
        pred_out = bool(pred.get("summary") or pred.get("last_output"))

        next_state, _reward, done = sb.step(state, action_str)
        after = _fs_snapshot(next_state.container_id)
        actual = {
            "exit_code": next_state.last_exit_code,
            "delta": before != after,
            "output_nonempty": bool((next_state.last_output or "").strip()),
        }
        l1 = pred_exit == actual["exit_code"]
        l2 = pred_files == actual["delta"]
        l3 = pred_out == actual["output_nonempty"]
        dlr = sum((l1, l2, l3)) / 3.0

        lm.store_experience(
            Experience(
                state=state, action=action_str, next_state=next_state,
                error=ErrorVector(
                    total_error=0.0, level1_error=0.0, level2_error=0.0,
                    level3_error=0.0, epistemic_error=0.0, aleatoric_error=0.0,
                    ensemble_variance=0.0,
                ),
                exit_code=next_state.last_exit_code,
                summary=next_state.last_output[:60] or action_str,
            )
        )
        if lm.should_update():
            lm.update()

        steps.append({
            "step": step_i, "action": action_str, "exit_code": actual["exit_code"],
            "delta": actual["delta"], "output_nonempty": actual["output_nonempty"],
            "l1_correct": l1, "l2_correct": l2, "l3_correct": l3, "dlr": dlr,
            "epistemic_error": 0.0,
        })
        action_history.append(action_str)
        state = next_state
        if done:
            break
        print(f"  [trial step {step_i:02d}] {action_str:<28} dlr={dlr:.2f} "
              f"exit={actual['exit_code']} delta={actual['delta']} out={actual['output_nonempty']}",
              flush=True)
    sb.close()
    return steps


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_PATH)
    ap.add_argument("--trials", type=int, default=N_TRIALS)
    ap.add_argument("--max-steps", type=int, default=MAX_STEPS)
    args = ap.parse_args()

    r = subprocess.run(["docker", "image", "inspect", "peda-sandbox:counterintuitive-v2"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print("FATAL: docker image missing: peda-sandbox:counterintuitive-v2", file=sys.stderr)
        return 2

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.model} (device={device}) ...", flush=True)
    wm = WorldModel(args.model, device=device)
    if wm.mode != "llm" or wm.model is None:
        print("FATAL: model fell back to stub", file=sys.stderr)
        return 2
    # fp32: fp16 LoRA finetune produces NaN loss (M2 finding); keep fp32 on GPU

    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                            capture_output=True, text=True).stdout.strip()
    meta = {
        "phase": "9", "direction": "counter-intuitive-sandbox",
        "experiment": "M4_error_trajectory",
        "commit": commit, "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "host": socket.gethostname(), "cpu_or_gpu": device,
        "sandbox_images": ["peda-sandbox:counterintuitive-v2"],
        "model": MODEL_PATH, "seeds": [],
        "per_episode_data_present": True,
        "task": TASK_ID, "max_steps": args.max_steps, "n_trials": args.trials,
        "update_interval": UPDATE_INTERVAL,
        "threshold": {"e_early_min": 0.5, "e_late_ratio_max": 0.5},
        "dlr_components": ["L1 exit_code", "L2 files_delta", "L3 output_nonempty"],
        "error_definition": "E(t) = 1 - DLR(t); DLR = mean(L1,L2,L3) per step",
    }

    all_trials: List[List[Dict]] = []
    for t in range(args.trials):
        print(f"[m4] trial {t+1}/{args.trials}", flush=True)
        all_trials.append(_run_trial(wm, device, args.max_steps))

    # Per-step E(t) averaged across trials (pad missing steps as NaN-skip)
    n = max(len(tr) for tr in all_trials)
    e_by_step: List[Optional[float]] = []
    dlr_by_step: List[Optional[float]] = []
    for s in range(n):
        dlrs = [tr[s]["dlr"] for tr in all_trials if s < len(tr)]
        if dlrs:
            dlr_by_step.append(sum(dlrs) / len(dlrs))
            e_by_step.append(1.0 - sum(dlrs) / len(dlrs))
        else:
            dlr_by_step.append(None)
            e_by_step.append(None)

    rows = []
    for t, trial in enumerate(all_trials):
        for st in trial:
            rows.append({"trial": t, **st})

    jsonl_path = REPO_ROOT / "results" / "phase9_ci_m4.jsonl"
    with jsonl_path.open("w") as f:
        f.write(json.dumps({"meta": meta}, ensure_ascii=False) + "\n")
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def mean_e(lo: int, hi: int) -> Optional[float]:
        vals = [e for e in e_by_step[lo:hi] if e is not None]
        return sum(vals) / len(vals) if vals else None

    e_early = mean_e(0, 10)
    e_late = mean_e(30, 40) if len(e_by_step) >= 40 else mean_e(max(30, n - 10), n)
    m4_pass = bool(e_early is not None and e_late is not None
                   and e_early >= 0.5 and e_late <= 0.5 * e_early)

    csv_path = REPO_ROOT / "results" / "phase9_ci_m4_summary.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "mean_dlr", "mean_e"])
        for s in range(n):
            w.writerow([s, f"{dlr_by_step[s]:.4f}" if dlr_by_step[s] is not None else "n/a",
                        f"{e_by_step[s]:.4f}" if e_by_step[s] is not None else "n/a"])
        w.writerow([])
        w.writerow(["m4_pass", "e_early_1_10", "e_late_31_40", "criterion"])
        w.writerow([m4_pass,
                    f"{e_early:.4f}" if e_early is not None else "n/a",
                    f"{e_late:.4f}" if e_late is not None else "n/a",
                    "E(1..10)>=0.5 AND E(31..40)<=0.5*E(1..10)"])

    print("=" * 78)
    print(f"M4 verdict: {'PASS' if m4_pass else 'FAIL'}  "
          f"E(1..10)={e_early:.4f} (>=0.5)  E(31..40)={e_late:.4f} "
          f"(<= {0.5 * e_early:.4f} if E_early={e_early:.4f})")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
