#!/usr/bin/env python3
# ruff: noqa: E402
"""PEDA vs pragmatic-only evaluation on Phase 1.5 text environment.

Pre-registered protocol:
  - Model: Qwen/Qwen2.5-0.5B-Instruct
  - Adapter: checkpoints/phase1_5/text_adapter_e3
  - Drive weights: curiosity=0.1, competence=2.0, boredom=0.1, novelty=2.0
  - Pragmatic weight: 3.0
  - Episodes: 10 per agent (20 total)
  - Max steps: 50
  - Eval seed: 42
  - Success threshold: PEDA success_rate > pragmatic + 10%
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
from phase1.types import Action, DriveWeights
from phase1.world_model import WorldModel, EnsembleErrorComputer
from phase1.drive_system import HomeostaticDriveSystem, ActionGenerator
from phase1_5.text_env import TextRoomEnv


def run_text_episode(
    env: TextRoomEnv,
    wm: WorldModel,
    ec: EnsembleErrorComputer,
    drive_system: HomeostaticDriveSystem,
    action_generator: ActionGenerator,
    start_seed: int = 0,
    max_steps: int = 50,
) -> Dict[str, Any]:
    state = env.reset(seed=start_seed)
    trajectory: List[str] = [state.room]
    action_history = []
    epistemic_errors: List[float] = []
    goal_reached = False

    for _ in range(max_steps):
        candidates = [Action(a) for a in TextRoomEnv.all_actions()]
        action = action_generator.select_action(state, action_history, candidates)
        wm.predict(state, action)
        next_state, reward, done = env.step(state, action.name)
        error = ec.decompose_error(state, action, next_state)
        epistemic_errors.append(error.epistemic_error)
        drive_system.update(error, action, has_external_input=False, action_history=action_history)
        action_history.append(action)
        trajectory.append(next_state.room)
        state = next_state
        if next_state.victory:
            goal_reached = True
            break
        if done:
            break

    return {
        "steps": len(trajectory) - 1,
        "success": goal_reached,
        "trajectory": trajectory,
        "actions": [a.name for a in action_history],
        "mean_epistemic_error": sum(epistemic_errors) / len(epistemic_errors) if epistemic_errors else 0.0,
    }


def aggregate(episodes: List[Dict]) -> Dict[str, float]:
    n = len(episodes)
    if n == 0:
        return {"success_rate": 0.0, "mean_steps": 50.0, "mean_epistemic_error": 0.0}
    successes = sum(1 for e in episodes if e["success"])
    return {
        "success_rate": successes / n,
        "mean_steps": sum(e["steps"] for e in episodes) / n,
        "mean_epistemic_error": sum(e["mean_epistemic_error"] for e in episodes) / n,
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 1.5 text environment eval")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--adapter", default="checkpoints/phase1_5/text_adapter_e3")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--start-episode", type=int, default=0, help="Episode offset for chunking")
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--eval-seed", type=int, default=42)
    parser.add_argument("--output", default="results/phase1_5_eval.json")
    parser.add_argument("--stub", action="store_true")
    parser.add_argument("--max-candidates", type=int, default=4, help="Candidate actions per step")
    parser.add_argument("--horizon", type=int, default=2, help="Rollout horizon")
    args = parser.parse_args()

    use_stub = args.stub or os.environ.get("FOLUNAR_STUB_MODEL", "0") == "1"
    max_steps = min(args.max_steps, 50)
    drive_weights = DriveWeights(curiosity=0.1, competence=2.0, boredom=0.1, novelty=2.0)

    print("Phase 1.5 Eval -- Text Environment", flush=True)
    print(f"  Model: {args.model}", flush=True)
    print(f"  Adapter: {args.adapter}", flush=True)
    print(f"  Drive weights: cur=0.1 cmp=2.0 bor=0.1 nov=2.0", flush=True)
    print(f"  Pragmatic weight: 3.0", flush=True)
    print(f"  Episodes: {args.episodes} per agent", flush=True)
    print(f"  Start episode: {args.start_episode}", flush=True)
    print(f"  Max steps: {max_steps}", flush=True)
    print(f"  Seed: {args.eval_seed}", flush=True)

    print("\nLoading WorldModel...", flush=True)
    wm = WorldModel(args.model, use_stub=use_stub)
    ec = EnsembleErrorComputer(wm)

    if not use_stub:
        ckpt_paths = sorted(Path(args.adapter).glob("checkpoint_epoch_*"))
        ec.checkpoints = ckpt_paths[-5:] if len(ckpt_paths) > 1 else []
        print(f"Loaded {len(ec.checkpoints)} ensemble checkpoint(s).", flush=True)

    results: Dict[str, List] = {"peda": [], "pragmatic_only": []}
    eval_start = time.time()

    for ep_offset in range(args.episodes):
        ep_num = args.start_episode + ep_offset
        seed_peda = args.eval_seed + ep_num
        seed_prag = args.eval_seed + ep_num + 1000000

        env_ped = TextRoomEnv()
        ds_ped = HomeostaticDriveSystem(drive_weights)
        ag_ped = ActionGenerator(wm, ec, ds_ped, horizon=args.horizon, max_candidates=args.max_candidates, pragmatic_only=False, pragmatic_weight=3.0)
        peda_r = run_text_episode(env_ped, wm, ec, ds_ped, ag_ped, start_seed=seed_peda, max_steps=max_steps)
        results["peda"].append(peda_r)

        env_prag = TextRoomEnv()
        ds_prag = HomeostaticDriveSystem(drive_weights)
        ag_prag = ActionGenerator(wm, ec, ds_prag, horizon=args.horizon, max_candidates=args.max_candidates, pragmatic_only=True, pragmatic_weight=3.0)
        prag_r = run_text_episode(env_prag, wm, ec, ds_prag, ag_prag, start_seed=seed_prag, max_steps=max_steps)
        results["pragmatic_only"].append(prag_r)

        print(f"  Ep{ep_num}: PEDA steps={peda_r['steps']} succ={peda_r['success']} epistemic={peda_r['mean_epistemic_error']:.4f} | "
              f"PRAG steps={prag_r['steps']} succ={prag_r['success']}", flush=True)

    elapsed = time.time() - eval_start
    print(f"\nEvaluation complete in {elapsed:.0f}s", flush=True)

    agg = {agent: aggregate(results[agent]) for agent in ["peda", "pragmatic_only"]}
    peda_sr = agg["peda"]["success_rate"]
    prag_sr = agg["pragmatic_only"]["success_rate"]
    peda_better = peda_sr > prag_sr + 0.1

    report = {
        "protocol": {
            "model": args.model,
            "adapter": args.adapter,
            "drive_weights": {"curiosity": 0.1, "competence": 2.0, "boredom": 0.1, "novelty": 2.0},
            "pragmatic_weight": 3.0,
            "episodes_per_agent": args.episodes,
            "start_episode": args.start_episode,
            "max_steps": max_steps,
            "eval_seed": args.eval_seed,
            "threshold_success_rate_diff": 0.1,
        },
        "aggregated": agg,
        "raw_results": results,
        "verdict": {
            "peda_better": peda_better,
            "peda_success_rate": peda_sr,
            "pragmatic_success_rate": prag_sr,
            "reason": f"PEDA success_rate={peda_sr:.2f} vs pragmatic={prag_sr:.2f}, diff={peda_sr - prag_sr:+.2f} (threshold +0.10)",
        },
        "elapsed_seconds": elapsed,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Report saved to {args.output}", flush=True)

    print(f"\n{'='*60}", flush=True)
    print("Phase 1.5 Evaluation Results", flush=True)
    print(f"{'='*60}", flush=True)
    for agent in ["peda", "pragmatic_only"]:
        a = agg[agent]
        print(f"\n  {agent}:", flush=True)
        print(f"    success_rate={a['success_rate']:.3f} mean_steps={a['mean_steps']:.1f} mean_epistemic_error={a['mean_epistemic_error']:.4f}", flush=True)
    print(f"\nVerdict: {'PEDA BETTER' if peda_better else 'PEDA NOT better'} than pragmatic-only", flush=True)
    print(f"  {report['verdict']['reason']}", flush=True)


if __name__ == "__main__":
    main()
