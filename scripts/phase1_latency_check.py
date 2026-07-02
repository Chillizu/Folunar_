#!/usr/bin/env python3
"""Phase 1 latency check: measure single-prediction latency and write config.

Usage:
    python scripts/phase1_latency_check.py          # actual LLM
    python scripts/phase1_latency_check.py --stub    # deterministic stub (verification)
    FOLUNAR_STUB_MODEL=1 python scripts/phase1_latency_check.py

Writes config/phase1_model.json with selected model and latency stats.
"""

import argparse
import json
import os
import time
from pathlib import Path
import sys

# Add src/ directory to path so phase1 modules are importable.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
from phase1.grid_env import GridWorld
from phase1.types import Action
from phase1.world_model import WorldModel


def median(values: list[float]) -> float:
    s = sorted(values)
    n = len(s)
    if n == 0:
        return 0.0
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2.0


def percentile(values: list[float], p: float) -> float:
    s = sorted(values)
    n = len(s)
    if n == 0:
        return 0.0
    k = int(p * (n - 1))
    return s[k]


def main():
    parser = argparse.ArgumentParser(
        description="Measure single-prediction latency and write config/phase1_model.json."
    )
    parser.add_argument(
        "--stub",
        action="store_true",
        help="Use stub world model (no LLM download). Skips actual timing.",
    )
    args = parser.parse_args()

    use_stub = args.stub or os.environ.get("FOLUNAR_STUB_MODEL", "0") == "1"

    # Ensure config/ directory exists.
    config_dir = Path("config")
    config_dir.mkdir(parents=True, exist_ok=True)

    # Instantiate grid env and world model.
    env = GridWorld(width=5, height=5, max_steps=50)
    wm = WorldModel(use_stub=use_stub)

    actions = GridWorld.all_actions()

    if use_stub or wm.mode == "stub":
        median_ms = 100.0
        p95_ms = 100.0
        model_name = "stub"
        print(f"[latency_check] Stub mode — using dummy latency ({median_ms} ms)")
    else:
        model_name = wm.model_name
        print(f"[latency_check] Model: {model_name} on {wm.device}")
        print("[latency_check] Warming up...")

        # Warm-up prediction.
        state = env.reset(seed=0)
        wm.predict(state, actions[0])

        print("[latency_check] Timing 10 predictions...")
        times: list[float] = []
        for i in range(10):
            state = env.reset(seed=i + 100)
            action = actions[i % 4]
            t0 = time.perf_counter()
            wm.predict(state, action)
            t1 = time.perf_counter()
            elapsed_ms = (t1 - t0) * 1000.0
            times.append(elapsed_ms)

        median_ms = median(times)
        p95_ms = percentile(times, 0.95)

        print(f"[latency_check]   median: {median_ms:.1f} ms")
        print(f"[latency_check]   p95:    {p95_ms:.1f} ms")

        if median_ms > 3000.0:
            fallback = wm.FALLBACK_MODEL  # Qwen/Qwen2.5-0.5B-Instruct
            print(
                f"[latency_check]  WARNING: median latency {median_ms:.0f} ms exceeds "
                f"3000 ms threshold."
            )
            print(f"[latency_check]  Switching default model to {fallback}")
            model_name = fallback
        else:
            print(
                f"[latency_check]  Latency within budget ({median_ms:.0f} ms < 3000 ms)."
            )

    # Write config file.
    config = {
        "model": model_name,
        "median_ms": round(median_ms, 1),
        "p95_ms": round(p95_ms, 1),
    }
    config_path = config_dir / "phase1_model.json"
    config_path.write_text(json.dumps(config, indent=2))
    print(f"[latency_check] Config written to {config_path}")


if __name__ == "__main__":
    main()
