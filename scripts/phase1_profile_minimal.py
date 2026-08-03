#!/usr/bin/env python3
# ruff: noqa: E402
"""Minimal Phase 1 profiler: tests model inference in isolation per phase."""

import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from phase1.world_model import WorldModel
from phase1.grid_env import GridWorld
from phase1.types import Action


def main():
    model_path = "/home/chillizu/models/Qwen2.5-0.5B-Instruct"
    print("Loading model...")
    wm = WorldModel(model_name=model_path, use_stub=False)
    env = GridWorld(width=5, height=5, max_steps=50)

    state = env.reset(seed=42)
    actions = GridWorld.all_actions()

    print(f"Model mode: {wm.mode}, device: {wm.device}")
    print(f"\nTest 1: Single predict call (should match latency check)")

    # Warmup
    print("  Warmup predict...", end=" ", flush=True)
    wm.predict(state, actions[0])
    print("done")

    # Timed single predict
    print("  Timing 3 single predicts:")
    times = []
    for i in range(3):
        state = env.reset(seed=i + 100)
        t0 = time.perf_counter()
        wm.predict(state, actions[i % 4])
        t1 = time.perf_counter()
        elapsed = (t1 - t0) * 1000
        times.append(elapsed)
        print(f"    predict {i+1}: {elapsed:.0f} ms")

    if times:
        print(f"    -> median: {sorted(times)[len(times)//2]:.0f} ms")

    print(f"\nTest 2: Rollout with horizon=1 (4 candidates)")
    state = env.reset(seed=42)
    for i, a in enumerate(actions):
        t0 = time.perf_counter()
        pred = wm.predict(state, a)
        t1 = time.perf_counter()
        print(f"  Candidate {a.name}: {(t1-t0)*1000:.0f} ms")

    print(f"\nTest 3: Full select_action")
    from phase1.drive_system import ActionGenerator, HomeostaticDriveSystem
    from phase1.types import DriveWeights
    from phase1.world_model import EnsembleErrorComputer

    ec = EnsembleErrorComputer(wm, num_checkpoints=5)
    ds = HomeostaticDriveSystem(
        DriveWeights(curiosity=0.1, competence=2.0, boredom=0.1, novelty=2.0)
    )
    ag = ActionGenerator(
        wm, ec, ds, horizon=2, max_candidates=4, latency_budget_ms=3000.0,
        pragmatic_only=False, pragmatic_weight=3.0,
    )

    state = env.reset(seed=42)
    t0 = time.perf_counter()
    action = ag.select_action(state, [], actions)
    t1 = time.perf_counter()
    print(f"  select_action took: {(t1-t0)*1000:.0f} ms ({t1-t0:.1f} s)")
    print(f"  Selected: {action.name}")

    print(f"\nTest 4: Full step (select + post-hoc predict + decompose_error)")
    state = env.reset(seed=43)
    action_history = []

    t0_total = time.perf_counter()
    action = ag.select_action(state, action_history, actions)
    t_select = time.perf_counter()

    pred = wm.predict(state, action)
    t_predict = time.perf_counter()

    next_state, reward, done = env.step(state, action)
    error = ec.decompose_error(state, action, next_state)
    t_decompose = time.perf_counter()

    print(f"  select_action:      {(t_select - t0_total)*1000:.0f} ms")
    print(f"  post-hoc predict:   {(t_predict - t_select)*1000:.0f} ms")
    print(f"  decompose_error:    {(t_decompose - t_predict)*1000:.0f} ms")
    print(f"  TOTAL:              {(t_decompose - t0_total)*1000:.0f} ms")

    # Project
    step_time = (t_decompose - t0_total)
    print(f"\n  Estimated full episode (50 steps): {step_time * 50:.0f} s ({step_time * 50 / 60:.1f} min)")
    print(f"  Estimated 10 steps: {step_time * 10:.0f} s")
    print(f"  Estimated 100 steps: {step_time * 100:.0f} s ({step_time * 100 / 60:.1f} min)")


if __name__ == "__main__":
    main()
