#!/usr/bin/env python3
# ruff: noqa: E402
"""Phase 1 drive-weight grid search.

Coarse-to-fine grid search over [0.1, 0.5, 1.0, 2.0] for curiosity, competence,
boredom, and novelty (256 combinations total).

Phase A — coarse screen: run --phase-a-episodes (default 2) per combo.
Phase B — fine screen:  top --top-k (default 20) combos, --phase-b-episodes (default 10) each.

Computes Pareto frontier and saves a full report with the top 5 configurations
to results/phase1_grid_search.json by default. The report can be passed directly
to scripts/phase1_eval.py via --drive-config. The live config file is only
overwritten when --write-config is explicitly provided.

Usage:
    python scripts/phase1_grid_search.py                                    # full search
    python scripts/phase1_grid_search.py --stub --phase-a-episodes 1        # fast verification
    python scripts/phase1_grid_search.py --write-config                     # also update config/phase1_default_drives.json
    FOLUNAR_STUB_MODEL=1 python scripts/phase1_grid_search.py
"""

import argparse
import datetime
import itertools
import json
import os
import sys
import time
from pathlib import Path

# Add src/ directory to path so phase1 modules are importable.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
from phase1.drive_system import ActionGenerator, HomeostaticDriveSystem
from phase1.grid_env import GridWorld
from phase1.run import aggregate_metrics, run_episode
from phase1.types import DriveWeights
from phase1.world_model import EnsembleErrorComputer, LearningModule, WorldModel

WEIGHT_VALUES = [0.1, 0.5, 1.0]
DRIVE_NAMES = ["curiosity", "competence", "boredom", "novelty"]


def make_env(max_steps: int = 50) -> GridWorld:
    return GridWorld(width=5, height=5, max_steps=max_steps)


def composite_score(metrics: dict) -> float:
    """Higher is better — blends success, completion, revisit, and steps."""
    success_rate = metrics.get("success_rate", 0.0)
    completion_20 = metrics.get("completion_20", 0.0)
    revisit_rate = metrics.get("revisit_rate", 0.5)
    mean_steps = metrics.get("mean_steps", 50.0)

    # Normalize each term to [0, 1] approximately.
    revisit_ok = max(0.0, 1.0 - revisit_rate / 0.5)  # 0.5 = max acceptable
    steps_ok = max(0.0, 1.0 - mean_steps / 50.0)  # 50 = max steps
    return (
        0.35 * success_rate
        + 0.25 * completion_20
        + 0.20 * revisit_ok
        + 0.20 * steps_ok
    )


def is_dominated(a: dict, b: dict) -> bool:
    """Return True if b dominates a (b is at least as good on all metrics and better on at least one).

    Maximize: success_rate, completion_20
    Minimize: mean_steps, revisit_rate
    """
    b_better_or_equal = (
        b["success_rate"] >= a["success_rate"]
        and b["completion_20"] >= a["completion_20"]
        and b["mean_steps"] <= a["mean_steps"]
        and b["revisit_rate"] <= a["revisit_rate"]
    )
    b_strictly_better = (
        b["success_rate"] > a["success_rate"]
        or b["completion_20"] > a["completion_20"]
        or b["mean_steps"] < a["mean_steps"]
        or b["revisit_rate"] < a["revisit_rate"]
    )
    return b_better_or_equal and b_strictly_better


def pareto_frontier(results: list[dict]) -> list[dict]:
    """Return non-dominated subset."""
    frontier: list[dict] = []
    for i, a in enumerate(results):
        dominated = False
        for j, b in enumerate(results):
            if i != j and is_dominated(a, b):
                dominated = True
                break
        if not dominated:
            frontier.append(a)
    return frontier


def run_combo(
    env: GridWorld,
    wm: WorldModel,
    weights: dict,
    num_episodes: int,
    base_seed: int = 0,
    max_candidates: int = 4,
    latency_budget_ms: float = 3000.0,
) -> dict:
    """Run num_episodes with the given fixed DriveWeights and return aggregated metrics."""
    drive_weights = DriveWeights(
        curiosity=weights["curiosity"],
        competence=weights["competence"],
        boredom=weights["boredom"],
        novelty=weights["novelty"],
    )
    # Create fresh per-combo instances so drive system history doesn't bleed
    # between combos. The WorldModel and ErrorComputer are shared.
    ec = EnsembleErrorComputer(wm, num_checkpoints=5)
    ds = HomeostaticDriveSystem(drive_weights)
    lm = LearningModule(wm, ec, buffer_size=1000, update_interval=500)
    ag = ActionGenerator(wm, ec, ds, horizon=2, max_candidates=max_candidates, latency_budget_ms=latency_budget_ms)

    trajectories = []
    predictions_list = []
    action_histories = []

    for ep in range(num_episodes):
        traj, preds, acts, ep_metrics = run_episode(
            env, wm, ec, ds, lm, ag, seed=base_seed + ep
        )
        trajectories.append(traj)
        predictions_list.append(preds)
        action_histories.append(acts)

    agg = aggregate_metrics(trajectories, predictions_list, action_histories)
    agg["weights"] = weights
    agg["score"] = composite_score(agg)
    return agg


def main():
    parser = argparse.ArgumentParser(
        description="Drive-weight grid search for Phase 1 PEDA."
    )
    parser.add_argument("--stub", action="store_true", help="Use stub world model.")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="HuggingFace model name (default: Qwen/Qwen2.5-1.5B-Instruct).",
    )
    parser.add_argument(
        "--phase-a-episodes",
        type=int,
        default=2,
        help="Episodes per combo in Phase A coarse screen (default: 2).",
    )
    parser.add_argument(
        "--phase-b-episodes",
        type=int,
        default=10,
        help="Episodes per combo in Phase B fine screen (default: 10).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of top combos from Phase A to advance to Phase B (default: 20).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=50,
        help="Maximum steps per episode (default: 50).",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=4,
        help="Number of candidate actions the ActionGenerator evaluates per step (default: 4).",
    )
    parser.add_argument(
        "--latency-budget",
        type=float,
        default=3000.0,
        help="Latency budget in ms for ActionGenerator (default: 3000.0).",
    )
    parser.add_argument(
        "--adapter",
        type=str,
        default=None,
        help="Path to a fine-tuned LoRA adapter to load (default: base model).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/phase1_grid_search.json",
        help="Path to the grid-search report (default: results/phase1_grid_search.json).",
    )
    parser.add_argument(
        "--write-config",
        action="store_true",
        help="Also write the top-5 weights to config/phase1_default_drives.json.",
    )
    args = parser.parse_args()

    use_stub = args.stub or os.environ.get("FOLUNAR_STUB_MODEL", "0") == "1"
    model_name_arg = args.model
    config_dir = Path("config")
    config_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Phase 1 Drive-Weight Grid Search")
    print("=" * 60)
    print(f"  Stub mode: {use_stub}")
    print(f"  Model: {model_name_arg or 'Qwen/Qwen2.5-1.5B-Instruct'}")
    print(f"  Weight space: {WEIGHT_VALUES}")
    print(f"  Total combos: {len(WEIGHT_VALUES)**4}")
    print(f"  Phase A episodes per combo: {args.phase_a_episodes}")
    print(f"  Adapter: {args.adapter or 'base model'}")
    print(f"  Phase B top-k: {args.top_k}")
    print(f"  Phase B episodes per combo: {args.phase_b_episodes}")
    print(f"  Max steps per episode: {args.max_steps}")
    print(f"  Max candidates: {args.max_candidates}")
    wm = WorldModel(model_name=model_name_arg, use_stub=use_stub, adapter_path=args.adapter)
    print()

    env = make_env(max_steps=args.max_steps)
    all_combinations = list(itertools.product(WEIGHT_VALUES, repeat=4))

    # ----------------------------------------------------------------
    # Phase A — coarse screen
    # ----------------------------------------------------------------
    print("--- Phase A: Coarse screen ---")
    phase_a_results: list[dict] = []
    phase_a_start = time.time()

    for idx, combo in enumerate(all_combinations):
        weights = {
            "curiosity": combo[0],
            "competence": combo[1],
            "boredom": combo[2],
            "novelty": combo[3],
        }
        try:
            result = run_combo(env, wm, weights, args.phase_a_episodes, base_seed=idx * 100, max_candidates=args.max_candidates, latency_budget_ms=args.latency_budget)
        except Exception as exc:
            print(
                f"  [{idx + 1}/{len(all_combinations)}] "
                f"cur={combo[0]} cmp={combo[1]} bor={combo[2]} nov={combo[3]} — "
                f"SKIPPED ({exc})"
            )
            continue

        phase_a_results.append(result)
        elapsed = time.time() - phase_a_start
        pct = (idx + 1) / len(all_combinations) * 100
        print(
            f"  [{idx + 1}/{len(all_combinations)}] ({pct:.0f}%) "
            f"cur={combo[0]} cmp={combo[1]} bor={combo[2]} nov={combo[3]} | "
            f"score={result['score']:.3f} success={result['success_rate']:.2f} "
            f"steps={result['mean_steps']:.1f} revisit={result['revisit_rate']:.3f} "
            f"c20={result['completion_20']:.2f} | elapsed={elapsed:.0f}s"
        )

    phase_a_elapsed = time.time() - phase_a_start
    print(f"\nPhase A complete in {phase_a_elapsed:.0f}s — {len(phase_a_results)} combos.")
    print()

    if not phase_a_results:
        print("ERROR: No Phase A results. Cannot continue.")
        sys.exit(1)

    # Sort by composite score descending and take top-k.
    phase_a_results.sort(key=lambda r: r["score"], reverse=True)
    top_k_results = phase_a_results[: args.top_k]

    print(f"Top {args.top_k} from Phase A:")
    for i, r in enumerate(top_k_results):
        w = r["weights"]
        print(
            f"  {i + 1}. score={r['score']:.3f} "
            f"cur={w['curiosity']} cmp={w['competence']} bor={w['boredom']} nov={w['novelty']} | "
            f"success={r['success_rate']:.2f} steps={r['mean_steps']:.1f} "
            f"revisit={r['revisit_rate']:.3f} c20={r['completion_20']:.2f}"
        )
    print()

    # ----------------------------------------------------------------
    # Phase B — fine screen
    # ----------------------------------------------------------------
    print("--- Phase B: Fine screen ---")
    phase_b_results: list[dict] = []
    phase_b_start = time.time()

    for idx, combo_result in enumerate(top_k_results):
        weights = combo_result["weights"]
        try:
            result = run_combo(
                env, wm, weights, args.phase_b_episodes, base_seed=idx * 1000 + 5000,
                max_candidates=args.max_candidates, latency_budget_ms=args.latency_budget
            )
        except Exception as exc:
            print(
                f"  [{idx + 1}/{len(top_k_results)}] "
                f"cur={weights['curiosity']} cmp={weights['competence']} "
                f"bor={weights['boredom']} nov={weights['novelty']} — "
                f"SKIPPED ({exc})"
            )
            continue

        phase_b_results.append(result)
        elapsed = time.time() - phase_b_start
        print(
            f"  [{idx + 1}/{len(top_k_results)}] "
            f"cur={weights['curiosity']} cmp={weights['competence']} "
            f"bor={weights['boredom']} nov={weights['novelty']} | "
            f"score={result['score']:.3f} success={result['success_rate']:.2f} "
            f"steps={result['mean_steps']:.1f} revisit={result['revisit_rate']:.3f} "
            f"c20={result['completion_20']:.2f} | elapsed={elapsed:.0f}s"
        )

    phase_b_elapsed = time.time() - phase_b_start
    print(f"\nPhase B complete in {phase_b_elapsed:.0f}s — {len(phase_b_results)} combos.")
    print()

    if not phase_b_results:
        print("ERROR: No Phase B results. Cannot continue.")
        sys.exit(1)

    # ----------------------------------------------------------------
    # Pareto frontier
    # ----------------------------------------------------------------
    frontier = pareto_frontier(phase_b_results)
    print(f"Pareto frontier: {len(frontier)} non-dominated configurations.")
    for i, r in enumerate(frontier):
        w = r["weights"]
        print(
            f"  {i + 1}. score={r['score']:.3f} "
            f"cur={w['curiosity']} cmp={w['competence']} bor={w['boredom']} nov={w['novelty']} | "
            f"success={r['success_rate']:.2f} steps={r['mean_steps']:.1f} "
            f"revisit={r['revisit_rate']:.3f} c20={r['completion_20']:.2f}"
        )
    print()

    # ----------------------------------------------------------------
    # Save results
    # ----------------------------------------------------------------
    # Rank by composite score (from Phase B aggregated metrics) within the frontier.
    frontier.sort(key=lambda r: r["score"], reverse=True)
    top_5 = frontier[:5]

    # If fewer than 5 in frontier, fill from the full Phase B results.
    if len(top_5) < 5:
        remaining = [r for r in phase_b_results if r not in top_5]
        remaining.sort(key=lambda r: r["score"], reverse=True)
        top_5.extend(remaining[: 5 - len(top_5)])

    default_drives = [
        {
            "curiosity": r["weights"]["curiosity"],
            "competence": r["weights"]["competence"],
            "boredom": r["weights"]["boredom"],
            "novelty": r["weights"]["novelty"],
        }
        for r in top_5
    ]

    results_dir = Path(args.output).parent
    results_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "timestamp": datetime.datetime.now().isoformat(),
        "model": model_name_arg or "Qwen/Qwen2.5-1.5B-Instruct",
        "stub": use_stub,
        "phase_a_episodes": args.phase_a_episodes,
        "phase_b_episodes": args.phase_b_episodes,
        "top_k": args.top_k,
        "top_5": [
            {
                "weights": r["weights"],
                "score": r["score"],
                "success_rate": r["success_rate"],
                "mean_steps": r["mean_steps"],
                "revisit_rate": r["revisit_rate"],
                "completion_20": r["completion_20"],
            }
            for r in top_5
        ],
    }
    output_path = Path(args.output)
    output_path.write_text(json.dumps(report, indent=2))
    print(f"Grid-search report saved to {output_path}")

    if args.write_config:
        config_path = config_dir / "phase1_default_drives.json"
        config_path.write_text(json.dumps(default_drives, indent=2))
        print(f"Top 5 default drive weights saved to {config_path}")
    else:
        print("Skipped config/phase1_default_drives.json update (use --write-config to overwrite).")

    for i, dw in enumerate(default_drives):
        print(
            f"  {i + 1}. cur={dw['curiosity']} cmp={dw['competence']} "
            f"bor={dw['boredom']} nov={dw['novelty']}"
        )

    # ----------------------------------------------------------------
    # Summary: honest assessment
    # ----------------------------------------------------------------
    best = top_5[0]
    bw = best["weights"]
    print(f"\n{'=' * 60}")
    print(f"Best configuration: cur={bw['curiosity']} cmp={bw['competence']} "
          f"bor={bw['boredom']} nov={bw['novelty']}")
    print(f"  success_rate={best['success_rate']:.3f}  mean_steps={best['mean_steps']:.1f}")
    print(f"  revisit_rate={best['revisit_rate']:.3f}  completion_20={best['completion_20']:.3f}")

    # Honest go/no-go analysis
    g2_ok = best["mean_steps"] < 50.0 * 0.5  # ~25 steps or fewer
    g3_ok = best["revisit_rate"] < 0.20
    print(f"\n  G1 (next-state accuracy): {best.get('next_state_accuracy', 'N/A'):.3f} "
          f"(target > 0.90)")
    print(f"  G2 (steps ratio): {'OK' if g2_ok else 'FAIL'} "
          f"({best['mean_steps']:.1f} steps, target < 25)")
    print(f"  G3 (revisit rate): {'OK' if g3_ok else 'FAIL'} "
          f"({best['revisit_rate']:.3f}, target < 0.20)")
    print(f"  Completion at 20 steps: {best['completion_20']:.3f}")
    print("\nNote: G2 and G3 require comparison against a random baseline;")
    print("run scripts/phase1_eval.py for the final go/no-go decision.")
    print("=" * 60)


if __name__ == "__main__":
    main()
