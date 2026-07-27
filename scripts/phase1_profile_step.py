#!/usr/bin/env python3
# ruff: noqa: E402
"""Phase 1 per-step profiler: instruments WorldModel.predict to count/time calls.

Creates a wrapper around WorldModel.predict that logs every call with its
origin context (rollout, post-hoc predict, ensemble decompose_error) so we
can pinpoint the bottleneck.

Usage:
    python scripts/phase1_profile_step.py
    python scripts/phase1_profile_step.py --model /home/chillizu/models/Qwen2.5-0.5B-Instruct
    python scripts/phase1_profile_step.py --stub   # fast stub verification
"""

import argparse
import functools
import os
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from phase1.drive_system import ActionGenerator, HomeostaticDriveSystem
from phase1.grid_env import GridWorld
from phase1.run import run_episode
from phase1.types import Action, DriveWeights
from phase1.world_model import EnsembleErrorComputer, LearningModule, WorldModel

# ── instrumentation ──────────────────────────────────────────────────────────

class PredictTracer:
    """Wraps WorldModel.predict to count, time, and classify each call.

    Attributes:
        calls: list of (origin, elapsed_ms) tuples
        origin_counts: dict[str, int]
        total_ms: float
    """

    def __init__(self, wm: WorldModel):
        self.calls: list[tuple[str, float]] = []
        self.origin_counts: dict[str, int] = defaultdict(int)
        self.total_ms: float = 0.0
        self._original_predict = wm.predict
        self._wm = wm
        # patch
        wm.predict = self._traced_predict

    def _classify_origin(self) -> str:
        """Classify the caller by inspecting the stack."""
        stack = traceback.extract_stack()
        # Walk backwards to find the first frame outside predict itself
        found_rollout = False
        found_select_action = False
        found_decompose = False
        found_run_episode = False
        for frame in stack:
            fn = frame.filename
            name = frame.name
            if "world_model" in fn and "rollout" in name:
                found_rollout = True
            if "drive_system" in fn and "select_action" in name:
                found_select_action = True
            if "world_model" in fn and "decompose_error" in name:
                found_decompose = True
            if "world_model" in fn and "_predictions_for" in name:
                # child of decompose_error
                found_decompose = True
            if "run" in fn and "run_episode" in name:
                found_run_episode = True
        # Order of precedence: rollout during select_action is the main batch
        if found_rollout:
            return "rollout (select_action)"
        if found_decompose:
            return "decompose_error"
        if found_run_episode:
            return "post-hoc predict (run_episode line 33)"
        return "unknown"

    def _traced_predict(self, state, action=None):
        t0 = time.perf_counter()
        result = self._original_predict(state, action)
        t1 = time.perf_counter()
        elapsed = (t1 - t0) * 1000.0
        origin = self._classify_origin()
        self.calls.append((origin, elapsed))
        self.origin_counts[origin] += 1
        self.total_ms += elapsed
        return result

    def restore(self):
        """Restore original predict method."""
        self._wm.predict = self._original_predict

    def report(self) -> dict:
        """Return structured report."""
        data = {
            "total_calls": len(self.calls),
            "total_time_ms": round(self.total_ms, 1),
            "mean_per_call_ms": round(self.total_ms / len(self.calls), 1) if self.calls else 0,
            "origin_counts": dict(self.origin_counts),
        }
        if self.calls:
            times = [c[1] for c in self.calls]
            times.sort()
            n = len(times)
            data["median_ms"] = round(times[n // 2], 1)
            data["min_ms"] = round(times[0], 1)
            data["max_ms"] = round(times[-1], 1)
            data["p95_ms"] = round(times[int(n * 0.95)], 1) if n > 1 else data["max_ms"]
            # per origin stats
            by_origin: dict[str, list[float]] = defaultdict(list)
            for origin, t in self.calls:
                by_origin[origin].append(t)
            data["per_origin"] = {}
            for origin, ts in by_origin.items():
                ts_sorted = sorted(ts)
                m = len(ts_sorted)
                data["per_origin"][origin] = {
                    "count": len(ts),
                    "total_ms": round(sum(ts), 1),
                    "mean_ms": round(sum(ts) / len(ts), 1),
                    "median_ms": round(ts_sorted[m // 2], 1),
                    "min_ms": round(ts_sorted[0], 1),
                    "max_ms": round(ts_sorted[-1], 1),
                }
        return data


# ── main ─────────────────────────────────────────────────────────────────────

def median(values: list[float]) -> float:
    s = sorted(values)
    n = len(s)
    if n == 0:
        return 0.0
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2.0


def main():
    parser = argparse.ArgumentParser(
        description="Profile single-step predict calls in Phase 1 with real LLM."
    )
    parser.add_argument("--stub", action="store_true", help="Use stub world model.")
    parser.add_argument(
        "--model",
        type=str,
        default="/home/chillizu/models/Qwen2.5-0.5B-Instruct",
        help="HuggingFace model name or path.",
    )
    parser.add_argument("--steps", type=int, default=2, help="Number of steps to profile (default: 2).")
    parser.add_argument(
        "--max-candidates", type=int, default=4, help="Candidate actions per step (default: 4)."
    )
    parser.add_argument("--horizon", type=int, default=2, help="Rollout horizon (default: 2).")
    args = parser.parse_args()

    use_stub = args.stub or os.environ.get("FOLUNAR_STUB_MODEL", "0") == "1"
    print("=" * 60)
    print("Phase 1 Per-Step Profiler")
    print("=" * 60)
    print(f"  Stub mode:       {use_stub}")
    print(f"  Model:           {args.model}")
    print(f"  Steps to run:    {args.steps}")
    print(f"  Max candidates:  {args.max_candidates}")
    print(f"  Horizon:         {args.horizon}")
    print()

    # ── Setup ────────────────────────────────────────────────────────────────
    env = GridWorld(width=5, height=5, max_steps=50)
    wm = WorldModel(model_name=args.model, use_stub=use_stub)

    if wm.mode == "stub":
        print("[profile] WARNING: Using stub model — timing will be unrealistically fast.")
    else:
        print(f"[profile] Model loaded on {wm.device}")

    # Patch predict BEFORE any components use it
    tracer = PredictTracer(wm)

    ec = EnsembleErrorComputer(wm, num_checkpoints=5)
    ds = HomeostaticDriveSystem(
        DriveWeights(curiosity=0.1, competence=2.0, boredom=0.1, novelty=2.0)
    )
    # match the eval script's setup: latency_budget_ms=3000, no pragmatic_only
    ag = ActionGenerator(
        wm, ec, ds,
        horizon=args.horizon,
        max_candidates=args.max_candidates,
        latency_budget_ms=3000.0,
        pragmatic_only=False,
        pragmatic_weight=3.0,
    )
    lm = LearningModule(wm, ec, buffer_size=1000, update_interval=500)

    # ── Check latency config ────────────────────────────────────────────────
    latency_path = Path("config/phase1_model.json")
    if latency_path.exists():
        import json
        lc = json.loads(latency_path.read_text())
        print(f"\n[profile] Latency config: median={lc.get('median_ms')}ms, p95={lc.get('p95_ms')}ms")
    else:
        print("\n[profile] No latency config found — ActionGenerator will use default 1000ms")

    # ── Run 1 episode step by step with per-step logging ────────────────────
    step_stats = []
    state = env.reset(seed=42)
    action_history: list[Action] = []

    print(f"\n{'─' * 60}")
    print(f"Running {args.steps} steps...")
    print(f"{'─' * 60}")

    for step_i in range(args.steps):
        # reset per-step counters
        step_calls_before = len(tracer.calls)
        step_ms_before = tracer.total_ms

        candidates = GridWorld.all_actions()
        action = ag.select_action(state, action_history, candidates)

        # post-hoc predict
        predicted = wm.predict(state, action)

        next_state, reward, done = env.step(state, action)
        error = ec.decompose_error(state, action, next_state)
        ds.update(error, action, has_external_input=False, action_history=action_history)

        # store (won't trigger update in 2 steps)
        from phase1.types import Experience
        exit_code = 2 if next_state.agent == next_state.goal else (1 if next_state.agent == state.agent else 0)
        summary = f"agent moved {action.name.lower()}"
        lm.store_experience(Experience(state=state, action=action, next_state=next_state, error=error, exit_code=exit_code, summary=summary))

        step_calls = len(tracer.calls) - step_calls_before
        step_ms = tracer.total_ms - step_ms_before

        # What horizon did ActionGenerator actually use?
        latency_ms = ag._load_latency_ms()
        budget = latency_ms * len(candidates) * ag.horizon
        actual_horizon = ag.horizon if budget <= ag.latency_budget_ms else 1

        step_stats.append({
            "step": step_i + 1,
            "action": action.name,
            "candidates": [c.name for c in candidates],
            "latency_ms": latency_ms,
            "budget_ms": budget,
            "latency_budget_ms": ag.latency_budget_ms,
            "configured_horizon": ag.horizon,
            "actual_horizon": actual_horizon,
            "predict_calls": step_calls,
            "step_time_ms": round(step_ms, 1),
        })

        print(f"\n  Step {step_i+1}:")
        print(f"    Selected action:     {action.name}")
        print(f"    Candidates:          {[c.name for c in candidates]}")
        print(f"    Latency (config):    {latency_ms:.0f} ms")
        print(f"    Budget (l*c*h):      {budget:.0f} ms  vs  budget_limit={ag.latency_budget_ms:.0f} ms")
        print(f"    Configured horizon:  {ag.horizon}  →  Actual horizon: {actual_horizon}")
        print(f"    Predict calls/step:  {step_calls}")
        print(f"    Step time:           {step_ms:.0f} ms ({step_ms/1000:.1f} s)")

        action_history.append(action)
        state = next_state

        if done:
            print("\n  [Goal reached!]")
            break

    tracer.restore()

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    report = tracer.report()
    print(f"\n  Total predict calls:        {report['total_calls']}")
    print(f"  Total time (all calls):    {report['total_time_ms']:.0f} ms ({report['total_time_ms']/1000:.1f} s)")
    print(f"  Per-call:                  median={report['median_ms']} ms  min={report['min_ms']}  max={report['max_ms']}  p95={report['p95_ms']}")
    print(f"\n  By origin:")
    for origin, stats in sorted(report['per_origin'].items()):
        print(f"    {origin}:")
        print(f"      count={stats['count']}  total={stats['total_ms']:.0f}ms  "
              f"mean={stats['mean_ms']:.0f}ms  median={stats['median_ms']:.0f}ms  "
              f"min={stats['min_ms']:.0f}  max={stats['max_ms']:.0f}")

    # Per-step breakdown
    print(f"\n  Per-step breakdown:")
    for s in step_stats:
        print(f"    Step {s['step']}: action={s['action']}  horizon={s['actual_horizon']}  "
              f"predict_calls={s['predict_calls']}  step_time={s['step_time_ms']:.0f}ms ({s['step_time_ms']/1000:.1f}s)  "
              f"candidates={len(s['candidates'])}")

    # ── Projection ──────────────────────────────────────────────────────────
    if step_stats:
        mean_step_time = sum(s['step_time_ms'] for s in step_stats) / len(step_stats)
        mean_predicts = sum(s['predict_calls'] for s in step_stats) / len(step_stats)
        print(f"\n  Projection for full episode (max 50 steps):")
        print(f"    Mean step time:    {mean_step_time:.0f} ms ({mean_step_time/1000:.1f} s)")
        print(f"    Mean predicts/step: {mean_predicts:.1f}")
        print(f"    50 steps:          {mean_step_time * 50 / 1000:.0f} s ({mean_step_time * 50 / 60000:.1f} min)")
        print(f"    10 steps:          {mean_step_time * 10 / 1000:.0f} s")
        # Time for select_action portion only (exclude post-hoc)
        sa_time = sum(s['step_time_ms'] for s in step_stats)
        print(f"\n  Bottleneck diagnosis:")
        print(f"    ActionGenerator rollout predicts: {report['per_origin'].get('rollout (select_action)', {}).get('count', 0)} calls, "
              f"{report['per_origin'].get('rollout (select_action)', {}).get('total_ms', 0):.0f}ms total")
        print(f"    Run-episode post-hoc predict: {report['per_origin'].get('post-hoc predict (run_episode line 33)', {}).get('count', 0)} calls, "
              f"{report['per_origin'].get('post-hoc predict (run_episode line 33)', {}).get('total_ms', 0):.0f}ms total")
        print(f"    Ensemble decompose_error: {report['per_origin'].get('decompose_error', {}).get('count', 0)} calls, "
              f"{report['per_origin'].get('decompose_error', {}).get('total_ms', 0):.0f}ms total")

    # ── Config suggestion ───────────────────────────────────────────────────
    print(f"\n  What if horizon=1 with same candidates?")
    per_call = report['median_ms'] if report['median_ms'] > 0 else 4750
    for cand in [4, 2, 1]:
        predicts = cand * 1  # horizon=1
        step_t = predicts * per_call + per_call + per_call  # rollout + post-hoc + decompose
        step_t_sec = step_t / 1000
        print(f"    Candidates={cand}, horizon=1: {predicts+2} predicts/step → {step_t_sec:.1f}s/step  "
              f"(10 steps: {step_t_sec*10:.0f}s, 50 steps: {step_t_sec*50/60:.1f}min)")

    print()
    # Done
    return report


if __name__ == "__main__":
    main()
