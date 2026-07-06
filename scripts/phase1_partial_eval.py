#!/usr/bin/env python3
# ruff: noqa: E402
"""Phase 1 core-hypothesis evaluation: PEDA vs pragmatic-only under partial training.

This script tests whether prediction-error (epistemic) signals drive exploration
by training the World Model on only a subset of grid cells, then comparing PEDA
against a pragmatic-only baseline when the goal lies inside or outside the
trained region.

Usage:
    python scripts/phase1_partial_eval.py --adapter checkpoints/phase1/synthetic_adapter
    python scripts/phase1_partial_eval.py --adapter ... --pragmatic-weight 1.0 --episodes 20
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from phase1.drive_system import ActionGenerator, HomeostaticDriveSystem
from phase1.grid_env import Action, GridWorld
from phase1.run import next_state_accuracy, revisit_rate, run_episode
from phase1.types import DriveWeights, GridState
from phase1.world_model import EnsembleErrorComputer, LearningModule, WorldModel


def _load_drive_weights(path: Path) -> DriveWeights:
    if not path.exists():
        raise FileNotFoundError(f"Drive config not found: {path}")
    raw = json.loads(path.read_text())
    if isinstance(raw, dict):
        raw = raw.get("top_5", [raw])
    if isinstance(raw, list):
        weights = raw[0]
    else:
        raise ValueError(f"Unexpected drive-config format in {path}")
    if isinstance(weights, dict) and "weights" in weights:
        weights = weights["weights"]
    return DriveWeights(
        curiosity=weights.get("curiosity", 0.1),
        competence=weights.get("competence", 0.5),
        boredom=weights.get("boredom", 0.1),
        novelty=weights.get("novelty", 0.1),
    )


def _is_known_cell(cell: Tuple[int, int], known_cells: Set[Tuple[int, int]]) -> bool:
    return cell in known_cells


def _to_json_serializable(obj: Any) -> Any:
    """Recursively convert tuples and other non-JSON types into JSON-safe structures."""
    if isinstance(obj, tuple):
        return [_to_json_serializable(x) for x in obj]
    if isinstance(obj, list):
        return [_to_json_serializable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _to_json_serializable(v) for k, v in obj.items()}
    return obj


def _sample_untrained_start(
    env: GridWorld, base_seed: int, known_cells: Set[Tuple[int, int]], max_trials: int = 1000
) -> Tuple[GridState, int]:
    """Return a reset state whose agent lies outside known_cells, plus the effective seed."""
    for offset in range(max_trials):
        state = env.reset(seed=base_seed + offset)
        if state.agent not in known_cells:
            return state, base_seed + offset
    raise RuntimeError(
        f"Could not sample an untrained start after {max_trials} trials; known_cells may cover all cells."
    )


def _sample_goal(
    condition: str,
    rng: Any,
    known_cells: List[Tuple[int, int]],
    all_cells: List[Tuple[int, int]],
) -> Tuple[int, int]:
    if condition == "goal_known":
        return tuple(rng.choice(known_cells))
    return tuple(rng.choice([c for c in all_cells if c not in known_cells]))


def _goal_for_cell(cell: Tuple[int, int]) -> Tuple[int, int]:
    """Pick a deterministic goal that is not the agent cell."""
    return (0, 0) if cell == (4, 4) else (4, 4)


def _compute_g1_test_set(
    world_model: WorldModel,
    manifest: Dict[str, Any],
    max_steps: int = 50,
) -> float:
    """Evaluate next-position accuracy on state-action pairs outside the training set."""
    known_pairs: Set[Tuple[Tuple[int, int], str]] = {
        (tuple(p["agent"]), p["action"]) for p in manifest.get("trained_pairs", [])
    }
    all_cells = [tuple(c) for c in manifest["all_cells"]]
    action_names = ["UP", "DOWN", "LEFT", "RIGHT"]

    correct = 0
    total = 0
    env = GridWorld(max_steps=max_steps)
    for cell in all_cells:
        goal = _goal_for_cell(cell)
        if cell == goal:
            continue
        for action_name in action_names:
            if (cell, action_name) in known_pairs:
                continue
            state = GridState(
                agent=cell,
                goal=goal,
                obstacles=[],
                width=5,
                height=5,
                step=0,
                max_steps=max_steps,
            )
            action = Action(name=action_name)
            pred = world_model.predict(state, action)
            next_state, _reward, _done = env.step(state, action)
            if pred.level2_next_agent == next_state.agent:
                correct += 1
            total += 1
    return correct / total if total > 0 else 0.0


def _run_agent_episode(
    env: GridWorld,
    world_model: WorldModel,
    error_computer: EnsembleErrorComputer,
    drive_system: HomeostaticDriveSystem,
    learning_module: LearningModule,
    action_generator: ActionGenerator,
    start_seed: int,
) -> Dict[str, Any]:
    trajectory, predictions, action_history, ep_metrics = run_episode(
        env,
        world_model,
        error_computer,
        drive_system,
        learning_module,
        action_generator,
        seed=start_seed,
    )
    return {
        "steps": ep_metrics["steps"],
        "success": ep_metrics["success"],
        "revisit_rate": revisit_rate(trajectory),
        "g1": next_state_accuracy(predictions, trajectory[1:]),
        "trajectory": [s.agent for s in trajectory],
        "start": trajectory[0].agent,
        "goal": trajectory[0].goal,
    }


def _aggregate(
    episodes: List[Dict[str, Any]], max_steps: int, known_cells: Set[Tuple[int, int]]
) -> Dict[str, Any]:
    n = len(episodes)
    if n == 0:
        return {
            "success_rate": 0.0,
            "mean_steps": float(max_steps),
            "revisit_rate": 0.0,
            "g1": 0.0,
        }
    return {
        "success_rate": sum(e["success"] for e in episodes) / n,
        "mean_steps": sum(e["steps"] for e in episodes) / n,
        "revisit_rate": sum(e["revisit_rate"] for e in episodes) / n,
        "g1": sum(e["g1"] for e in episodes) / n,
    }


def _exploration_metrics(
    episodes: List[Dict[str, Any]],
    known_cells: Set[Tuple[int, int]],
    max_steps: int,
) -> Dict[str, float]:
    """Metrics capturing how much PEDA explores the unknown region."""
    if not episodes:
        return {
            "mean_unknown_fraction": 0.0,
            "mean_unknown_cells_visited": 0.0,
            "mean_steps_before_known": float(max_steps),
        }

    unknown_fractions = []
    unknown_cells_counts = []
    steps_before_known = []
    for ep in episodes:
        traj = ep["trajectory"]
        unknown_positions = [p for p in traj if p not in known_cells]
        unknown_fractions.append(len(unknown_positions) / len(traj) if traj else 0.0)
        unknown_cells_counts.append(len(set(unknown_positions)))

        first_known_idx = None
        for i, p in enumerate(traj):
            if p in known_cells:
                first_known_idx = i
                break
        steps_before_known.append(first_known_idx if first_known_idx is not None else max_steps)

    return {
        "mean_unknown_fraction": sum(unknown_fractions) / len(unknown_fractions),
        "mean_unknown_cells_visited": sum(unknown_cells_counts) / len(unknown_cells_counts),
        "mean_steps_before_known": sum(steps_before_known) / len(steps_before_known),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Partial-training evaluation of PEDA vs pragmatic-only baseline."
    )
    parser.add_argument("--model", default=None, help="Base model name.")
    parser.add_argument(
        "--adapter",
        default="checkpoints/phase1/synthetic_adapter",
        help="Path to the trained LoRA adapter.",
    )
    parser.add_argument("--episodes", type=int, default=20, help="Episodes per condition.")
    parser.add_argument("--start-episode", type=int, default=0, help="Episode offset for chunked runs.")
    parser.add_argument("--total-episodes", type=int, default=10, help="Total episodes per condition in the full run (for deterministic chunking).")
    parser.add_argument("--max-candidates", type=int, default=4)
    parser.add_argument(
        "--drive-config",
        default="config/phase1_default_drives.json",
        help="Path to drive weights JSON.",
    )
    parser.add_argument("--output", default="results/phase1_partial_eval.json")
    parser.add_argument("--eval-seed", type=int, default=123)
    parser.add_argument("--pragmatic-weight", type=float, default=3.0)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--skip-g1-test", action="store_true", help="Skip g1_test_set computation to save time during chunked runs.")
    args = parser.parse_args()

    adapter_path = Path(args.adapter)
    info_path = adapter_path / "training_info.json"
    manifest_path = adapter_path / "trained_manifest.json"

    model_name = args.model
    if model_name is None and info_path.exists():
        model_name = json.loads(info_path.read_text()).get("model")
        print(f"[partial_eval] Inferred base model from adapter: {model_name}")
    if model_name is None:
        model_name = WorldModel.DEFAULT_MODEL

    if not manifest_path.exists():
        print(f"ERROR: manifest not found at {manifest_path}. Run phase1_synthetic_train.py first.")
        sys.exit(1)

    manifest = json.loads(manifest_path.read_text())
    known_cells = {tuple(c) for c in manifest["known_cells"]}
    known_cells_list = [tuple(c) for c in manifest["known_cells"]]
    all_cells = [tuple(c) for c in manifest["all_cells"]]
    train_fraction = manifest.get("train_fraction", 0.5)

    drive_weights = _load_drive_weights(Path(args.drive_config))
    print(f"[partial_eval] Drive weights: cur={drive_weights.curiosity} "
          f"cmp={drive_weights.competence} bor={drive_weights.boredom} nov={drive_weights.novelty}")
    print(f"[partial_eval] Adapter: {adapter_path}")
    print(f"[partial_eval] Known cells: {len(known_cells)} / {len(all_cells)}")
    print(f"[partial_eval] Pragmatic weight: {args.pragmatic_weight}")
    print(f"[partial_eval] Episodes per condition: {args.episodes}")
    print(f"[partial_eval] Start episode: {args.start_episode}")

    print("[partial_eval] Loading WorldModel...")
    use_stub = os.environ.get("FOLUNAR_STUB_MODEL", "0") == "1"
    wm = WorldModel(model_name=model_name, use_stub=use_stub, adapter_path=str(adapter_path))
    if not use_stub and (wm.mode == "stub" or wm.model is None):
        raise RuntimeError("WorldModel failed to load real LLM")

    ec = EnsembleErrorComputer(wm, num_checkpoints=5)
    ckpt_paths = sorted(adapter_path.glob("checkpoint_epoch_*"))
    if len(ckpt_paths) < 2:
        print(f"WARNING: found {len(ckpt_paths)} per-epoch checkpoint(s); ensemble epistemic error will be zero.")
        print("WARNING: partial_eval can still run, but the core hypothesis test requires >=2 checkpoints.")
    # Avoid the overhead of loading the same adapter twice when only one checkpoint exists.
    if len(ckpt_paths) <= 1:
        ec.checkpoints = []
    else:
        ec.checkpoints = ckpt_paths[-5:]
    print(f"[partial_eval] Loaded {len(ec.checkpoints)} ensemble checkpoint(s).")

    lm = LearningModule(wm, ec, buffer_size=1000, update_interval=100000)

    rng = __import__("random").Random(args.eval_seed)
    max_steps = args.max_steps

    conditions = ["goal_known", "goal_unknown"]
    results: Dict[str, Any] = {cond: {"peda": [], "pragmatic_only": []} for cond in conditions}

    eval_start = time.time()
    for cond_idx, cond in enumerate(conditions):
        print(f"\n[partial_eval] Running condition: {cond}")
        # Each condition uses its own deterministic RNG stream so chunked runs
        # can be resumed without reproducing the entire prior condition sequence.
        rng = __import__("random").Random(args.eval_seed + cond_idx * args.total_episodes)
        # Advance the RNG to the requested start episode.
        for _ in range(args.start_episode):
            _ = _sample_goal(cond, rng, known_cells_list, all_cells)
        for ep in range(args.start_episode, args.start_episode + args.episodes):
            goal = _sample_goal(cond, rng, known_cells_list, all_cells)
            env = GridWorld(goal=goal, max_steps=max_steps)
            cond_offset = 1000000 if cond == "goal_unknown" else 0
            base_seed = args.eval_seed + ep + cond_offset
            _, start_seed = _sample_untrained_start(env, base_seed, known_cells)

            # PEDA agent
            ds_ped = HomeostaticDriveSystem(drive_weights)
            ag_ped = ActionGenerator(
                wm,
                ec,
                ds_ped,
                horizon=2,
                max_candidates=args.max_candidates,
                pragmatic_only=False,
                pragmatic_weight=args.pragmatic_weight,
            )
            peda_result = _run_agent_episode(
                env, wm, ec, ds_ped, lm, ag_ped, start_seed
            )

            # Pragmatic-only baseline
            ds_prag = HomeostaticDriveSystem(drive_weights)
            ag_prag = ActionGenerator(
                wm,
                ec,
                ds_prag,
                horizon=2,
                max_candidates=args.max_candidates,
                pragmatic_only=True,
                pragmatic_weight=args.pragmatic_weight,
            )
            prag_result = _run_agent_episode(
                env, wm, ec, ds_prag, lm, ag_prag, start_seed
            )

            results[cond]["peda"].append(peda_result)
            results[cond]["pragmatic_only"].append(prag_result)

            if (
                (ep + 1) % 5 == 0
                or ep == args.start_episode
                or ep == args.start_episode + args.episodes - 1
            ):
                print(
                    f"  [{ep + 1}/{args.total_episodes}] "
                    f"PEDA steps={peda_result['steps']} success={peda_result['success']} | "
                    f"Prag steps={prag_result['steps']} success={prag_result['success']}"
                )

    eval_elapsed = time.time() - eval_start
    print(f"\n[partial_eval] Evaluation complete in {eval_elapsed:.0f}s")

    print("[partial_eval] Computing g1_test_set on held-out state-action pairs...")
    if args.skip_g1_test:
        print("[partial_eval] Skipping g1_test_set computation (--skip-g1-test).")
        g1_test_set = 0.0
    else:
        g1_test_set = _compute_g1_test_set(wm, manifest, max_steps=max_steps)
    print(f"[partial_eval] g1_test_set = {g1_test_set:.4f}")

    aggregated: Dict[str, Any] = {}
    for cond in conditions:
        aggregated[cond] = {
            "peda": _aggregate(results[cond]["peda"], max_steps, known_cells),
            "pragmatic_only": _aggregate(results[cond]["pragmatic_only"], max_steps, known_cells),
        }
        aggregated[cond]["peda"].update(
            _exploration_metrics(results[cond]["peda"], known_cells, max_steps)
        )

    peda_unknown = aggregated["goal_unknown"]["peda"]["mean_steps"]
    prag_unknown = aggregated["goal_unknown"]["pragmatic_only"]["mean_steps"]
    peda_better = peda_unknown < prag_unknown

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": {
            "model": model_name,
            "adapter": str(adapter_path),
            "train_fraction": train_fraction,
            "known_cells": manifest["known_cells"],
            "episodes_per_condition": args.episodes,
            "start_episode": args.start_episode,
            "total_episodes": args.total_episodes,
            "max_steps": max_steps,
            "pragmatic_weight": args.pragmatic_weight,
            "drive_weights": {
                "curiosity": drive_weights.curiosity,
                "competence": drive_weights.competence,
                "boredom": drive_weights.boredom,
                "novelty": drive_weights.novelty,
            },
        },
        "g1_test_set": round(g1_test_set, 4),
        "raw_results": _to_json_serializable(results),
        "conditions": aggregated,
        "verdict": {
            "peda_better_in_unknown_goal": peda_better,
            "reason": (
                "PEDA mean_steps < pragmatic_only in goal_unknown condition"
                if peda_better
                else "PEDA did not beat pragmatic_only in goal_unknown condition"
            ),
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2))
    print(f"\n[partial_eval] Report saved to {output_path}")

    print(f"\n{'=' * 60}")
    print("Partial-Training Evaluation Results")
    print(f"{'=' * 60}")
    for cond in conditions:
        print(f"\n--- {cond} ---")
        for agent in ("peda", "pragmatic_only"):
            m = aggregated[cond][agent]
            print(f"  {agent:20s} success={m['success_rate']:.3f} "
                  f"mean_steps={m['mean_steps']:.1f} revisit={m['revisit_rate']:.3f} "
                  f"g1={m['g1']:.3f}")
            if agent == "peda":
                print(f"    exploration -> unknown_fraction={m['mean_unknown_fraction']:.3f} "
                      f"unknown_cells={m['mean_unknown_cells_visited']:.2f} "
                      f"steps_before_known={m['mean_steps_before_known']:.2f}")
    print(f"\nVerdict: {report['verdict']['reason']}")

    sys.exit(0 if peda_better else 1)


if __name__ == "__main__":
    main()
