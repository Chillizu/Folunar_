#!/usr/bin/env python3
# ruff: noqa: E402
"""Phase 1 — Measure G1 with LoRA adapter loaded on the real LLM.

Generates 50 random (state, action, ground-truth) transitions in a 5x5 GridWorld,
loads the Qwen2.5-0.5B-Instruct model with the partial_adapter_real_25_e3 LoRA
adapter, compares WorldModel.predict() output against actual GridWorld.step(),
and reports G1 (next-position) accuracy and exit-code accuracy alongside the
published base-model baseline (G1=0.18, exit_code=0.76).

Usage:
    source venv/bin/activate
    PYTHONPATH=src python scripts/phase1_measure_g1_adapter.py
"""

import json
import random
import sys
import time
from pathlib import Path

# Add src/ to Python path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from phase1.grid_env import GridWorld, Perception
from phase1.types import Action, GridState
from phase1.world_model import WorldModel

# ── Constants ────────────────────────────────────────────────────────────────
NUM_TRANSITIONS = 50
MODEL_PATH = "/home/chillizu/models/Qwen2.5-0.5B-Instruct"
ADAPTER_PATH = str(_PROJECT_ROOT / "checkpoints" / "phase1" / "partial_adapter_real_25_e3")
SEED = 42
GRID_SIZE = 5
RNG = random.Random(SEED)


def _make_env() -> GridWorld:
    """Create a standard 5x5 GridWorld with random obstacles."""
    env = GridWorld(width=5, height=5, max_steps=50)
    # Reset with a fixed seed so obstacle layout is reproducible
    env.reset(seed=42)
    return env


def _generate_transition(env: GridWorld, rng: random.Random) -> tuple[GridState, Action, GridState, int]:
    """Generate one random valid transition.

    Returns (state, action, next_state, ground_truth_exit_code).
    """
    # Get a random state by seeding the env randomly and resetting
    env_seed = rng.randint(0, 1000000)
    state = env.reset(seed=env_seed)

    # Pick a random action
    action = rng.choice([
        Action(name="UP"),
        Action(name="DOWN"),
        Action(name="LEFT"),
        Action(name="RIGHT"),
    ])

    # Get ground truth by stepping the environment
    try:
        next_state, reward, done = env.step(state, action)
    except Exception as e:
        print(f"  [WARN] env.step failed for seed {env_seed}, action {action.name}: {e}")
        return state, action, state, 1

    # Compute ground-truth exit code (same logic as _stub_predict)
    if next_state.agent == state.goal:
        gt_exit_code = 2
    elif next_state.agent == state.agent:
        gt_exit_code = 1
    else:
        gt_exit_code = 0

    return state, action, next_state, gt_exit_code


def _compute_exit_code(state: GridState, next_pos: tuple[int, int], goal: tuple[int, int]) -> int:
    """Compute ground-truth exit code based on next position."""
    if next_pos == goal:
        return 2
    elif next_pos == state.agent:
        return 1
    return 0


def main():
    print("=" * 72)
    print("  Phase 1 — G1: Real LLM World Model + LoRA Adapter Next-State Prediction")
    print("=" * 72)
    print(f"  Model:   {MODEL_PATH}")
    print(f"  Adapter: {ADAPTER_PATH}")
    print(f"  Transitions: {NUM_TRANSITIONS}")
    print(f"  Seed:    {SEED}")
    print(f"  Baseline (no adapter): G1=0.18, exit_code_acc=0.76")

    # Step 1: Load the real LLM with adapter
    print("\n[1] Loading model with LoRA adapter (this may take a while on CPU)...")
    t0 = time.perf_counter()
    wm = WorldModel(model_name=MODEL_PATH, adapter_path=ADAPTER_PATH)
    load_time = time.perf_counter() - t0
    print(f"  Mode: {wm.mode}")
    print(f"  Adapter name: {wm.adapter_name}")
    print(f"  Load time: {load_time:.1f}s")

    if wm.mode == "stub":
        print("  WARNING: Model fell back to stub mode! Accuracy will be 1.0.")
        print("  The real LLM was not loaded — check FOLUNAR_STUB_MODEL env var")
        print("  or if the model path exists and has transformers-compatible files.")

    # Step 2: Create environment
    env = _make_env()

    # Step 3: Generate transitions and evaluate
    print(f"\n[2] Generating {NUM_TRANSITIONS} random transitions and evaluating...")
    results = []
    next_pos_correct = 0
    exit_code_correct = 0
    total = 0
    predict_times = []

    for i in range(NUM_TRANSITIONS):
        # Generate random transition
        state, action, gt_next_state, gt_exit_code = _generate_transition(env, RNG)

        # Predict using WorldModel
        t0 = time.perf_counter()
        pred = wm.predict(state, action)
        pred_time = time.perf_counter() - t0
        predict_times.append(pred_time)

        # Compare
        pred_next_pos = pred.level2_next_agent
        pred_exit_code = pred.level1_exit_code

        pos_ok = pred_next_pos == gt_next_state.agent
        exit_ok = pred_exit_code == gt_exit_code

        if pos_ok:
            next_pos_correct += 1
        if exit_ok:
            exit_code_correct += 1
        total += 1

        results.append({
            "idx": i,
            "state_text": Perception.render(state),
            "action": action.name,
            "gt_next_pos": gt_next_state.agent,
            "pred_next_pos": pred_next_pos,
            "pos_ok": pos_ok,
            "gt_exit_code": gt_exit_code,
            "pred_exit_code": pred_exit_code,
            "exit_ok": exit_ok,
            "pred_time_s": round(pred_time, 3),
            "summary": pred.level3_output_summary,
        })

        if (i + 1) % 10 == 0:
            elapsed = sum(predict_times)
            print(f"  Processed {i+1}/{NUM_TRANSITIONS} ... elapsed {elapsed:.0f}s")

    # Step 4: Compute final metrics
    elapsed_total = sum(predict_times)
    g1_accuracy = next_pos_correct / total if total > 0 else 0.0
    exit_accuracy = exit_code_correct / total if total > 0 else 0.0

    median_time = sorted(predict_times)[len(predict_times) // 2] if predict_times else 0

    print(f"\n{'=' * 72}")
    print(f"  RESULTS")
    print(f"{'=' * 72}")
    print(f"  Total transitions tested:   {total}")
    print(f"  Next-position accuracy (G1): {g1_accuracy:.4f}  ({next_pos_correct}/{total})")
    print(f"  Exit-code accuracy:          {exit_accuracy:.4f}  ({exit_code_correct}/{total})")
    print(f"  Total predict time:         {elapsed_total:.1f}s")
    print(f"  Median predict time:        {median_time:.3f}s")
    print()
    print(f"  ── Baseline comparison ──")
    print(f"  Without adapter:            G1=0.1800  exit=0.7600")
    print(f"  With adapter:               G1={g1_accuracy:.4f}  exit={exit_accuracy:.4f}")
    delta_g1 = g1_accuracy - 0.18
    delta_exit = exit_accuracy - 0.76
    print(f"  Delta:                      G1={delta_g1:+.4f}  exit={delta_exit:+.4f}")
    print(f"  {'↑ Improves' if delta_g1 > 0 else '↓ Degrades' if delta_g1 < 0 else '= No change'} over base model (G1)")
    print(f"  Above 0.90 threshold:       {'YES' if g1_accuracy >= 0.90 else 'NO'}")
    print()

    # Step 5: Show sample predictions
    print(f"{'─' * 72}")
    print(f"  SAMPLE PREDICTIONS (first 10)")
    print(f"{'─' * 72}")
    header = f"  {'#':>3} | {'State':<30} {'Action':<8} {'Predicted':<12} {'Actual':<12} {'Pos?':<5} {'Exit?':<5}"
    print(header)
    print(f"  {'─' * 80}")
    for r in results[:10]:
        # Extract compact state description
        state_str = r["state_text"][:28]
        print(
            f"  {r['idx']:>3} | {state_str:<30} {r['action']:<8} "
            f"{str(r['pred_next_pos']):<12} {str(r['gt_next_pos']):<12} "
            f"{'OK' if r['pos_ok'] else 'NO':<5} {'OK' if r['exit_ok'] else 'NO':<5}"
        )

    # Step 6: Error analysis
    print(f"\n{'─' * 72}")
    print(f"  ERROR ANALYSIS")
    print(f"{'─' * 72}")

    errors_pos = [r for r in results if not r["pos_ok"]]
    if errors_pos:
        error_positions = {}
        for r in errors_pos:
            key = f"pred={r['pred_next_pos']}, actual={r['gt_next_pos']}"
            error_positions[key] = error_positions.get(key, 0) + 1

        print(f"\n  Position errors ({len(errors_pos)}/{total}):")
        top_errors = sorted(error_positions.items(), key=lambda x: -x[1])[:5]
        for pattern, count in top_errors:
            print(f"    {pattern}: {count} times")

        # Check if errors are biased towards specific outputs
        pred_vals = {}
        for r in errors_pos:
            p = r["pred_next_pos"]
            pred_vals[p] = pred_vals.get(p, 0) + 1
        if pred_vals:
            most_common_pred = max(pred_vals, key=pred_vals.get)
            print(f"\n  Most common incorrect prediction: {most_common_pred} "
                  f"({pred_vals[most_common_pred]}/{len(errors_pos)} errors)")
            # Check if it's always predicting the SAME position
            if len(pred_vals) == 1:
                only_pred = list(pred_vals.keys())[0]
                print(f"  PATTERN: Model always predicts {only_pred} regardless of input!")
            elif max(pred_vals.values()) / len(errors_pos) >= 0.8:
                print(f"  PATTERN: ~{max(pred_vals.values())/len(errors_pos)*100:.0f}% of errors collapse to {most_common_pred}")

        # Check obstacle/goal awareness
        wall_hits = sum(1 for r in errors_pos if r["gt_exit_code"] == 1)
        goal_reaches = sum(1 for r in errors_pos if r["gt_exit_code"] == 2)
        if wall_hits > 0:
            print(f"  Obstacle/wall awareness: {wall_hits} errors involved wall hits")
        if goal_reaches > 0:
            print(f"  Goal awareness: {goal_reaches} errors involved reaching the goal")

        # Check if position errors also have exit code errors
        both_wrong = sum(1 for r in errors_pos if not r["exit_ok"])
        print(f"  Position & exit code both wrong: {both_wrong}/{len(errors_pos)}")
    else:
        print("\n  No position errors!")

    errors_exit = [r for r in results if not r["exit_ok"]]
    if errors_exit and not errors_pos:
        print(f"\n  Exit-code errors ({len(errors_exit)}/{total}):")
        sample_bad = errors_exit[:3]
        for r in sample_bad:
            print(f"    action={r['action']}, pred_exit={r['pred_exit_code']}, gt_exit={r['gt_exit_code']}")
    elif errors_exit:
        exit_only = sum(1 for r in errors_exit if r["pos_ok"])
        if exit_only > 0:
            print(f"  Exit-code only errors (correct position but wrong exit code): {exit_only}/{total}")

    # Step 7: Comparison examples (interesting cases showing improvement/failure)
    print(f"\n{'─' * 72}")
    print(f"  REPRESENTATIVE CASES")
    print(f"{'─' * 72}")

    # Show correct predictions that would have been wrong with base model (examples of improvement)
    print(f"\n  First 5 correct predictions:")
    correct = [r for r in results if r["pos_ok"] and r["exit_ok"]][:5]
    for r in correct:
        state_str = r["state_text"][:28]
        print(f"    #{r['idx']} state={state_str:<28} action={r['action']:<8} "
              f"pos={r['pred_next_pos']} exit={r['pred_exit_code']}")

    # Show first 5 wrong predictions as failure examples
    wrong = [r for r in results if not r["pos_ok"]][:5]
    if wrong:
        print(f"\n  First 5 position errors:")
        for r in wrong:
            state_str = r["state_text"][:28]
            print(f"    #{r['idx']} state={state_str:<28} action={r['action']:<8} "
                  f"pred={r['pred_next_pos']} gt={r['gt_next_pos']} exit_pred={r['pred_exit_code']} exit_gt={r['gt_exit_code']}")

    # Step 8: Summary
    print(f"\n{'=' * 72}")
    print(f"  SUMMARY")
    print(f"{'=' * 72}")
    print(f"  G1 accuracy (with adapter):  {g1_accuracy:.4f}")
    print(f"  Exit accuracy (with adapter):{exit_accuracy:.4f}")
    print(f"  Baseline G1 (no adapter):    0.1800")
    print(f"  Baseline exit (no adapter):  0.7600")
    print(f"  G1 delta:                    {delta_g1:+.4f}")
    print(f"  Above 0.90 threshold:        {'YES' if g1_accuracy >= 0.90 else 'NO'}")
    verdict = ""
    if g1_accuracy >= 0.90:
        verdict = "Adapter boosts G1 above threshold — World Model predicts well with adapter."
    elif g1_accuracy >= 0.50:
        verdict = "Adapter improves G1 but still below threshold."
    elif g1_accuracy > 0.18:
        verdict = "Adapter provides modest G1 improvement over baseline."
    else:
        verdict = "Adapter does not improve G1 over base model (or degrades it)."
    print(f"  Verdict: {verdict}")
    print(f"{'=' * 72}")

    # Save raw results for later inspection
    results_path = _PROJECT_ROOT / "results" / "phase1_g1_accuracy_adapter.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "model": MODEL_PATH,
        "adapter": ADAPTER_PATH,
        "mode": wm.mode,
        "num_transitions": total,
        "g1_accuracy": g1_accuracy,
        "exit_code_accuracy": exit_accuracy,
        "g1_correct": next_pos_correct,
        "exit_correct": exit_code_correct,
        "median_predict_time_s": round(median_time, 3),
        "total_predict_time_s": round(elapsed_total, 1),
        "above_threshold": g1_accuracy >= 0.90,
        "baseline": {"g1": 0.18, "exit_code": 0.76},
        "delta_g1": round(delta_g1, 4),
        "delta_exit": round(delta_exit, 4),
    }
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Raw results saved to {results_path}")


if __name__ == "__main__":
    main()
