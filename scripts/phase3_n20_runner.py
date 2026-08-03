#!/usr/bin/env python3
"""Phase 3 N=20 Sequential Runner - runs all 4 conditions one at a time in tmux.

Runs sequentially to avoid GPU OOM. 20 fresh episodes per condition (seeds 0-19).
Overwrites previous results.
"""

import subprocess
import sys
import time
from pathlib import Path

BASE_DIR = Path("/home/ec2-user/Folunar_")
SCRIPTS_DIR = BASE_DIR / "scripts"
RESULTS_DIR = BASE_DIR / "results"
PYTHON = "/opt/pytorch/bin/python"
EXPERIMENT_SCRIPT = str(SCRIPTS_DIR / "phase3_sandbox_experiment.py")

CONDITIONS = [
    ("pragmatic", "known"),
    ("pragmatic", "unknown"),
    ("peda", "known"),
    ("peda", "unknown"),
]

TMUX_SESSION = "phase3_n20"
N_EPISODES = 20
MAX_STEPS = 10


def ensure_tmux():
    result = subprocess.run(
        ["tmux", "has-session", "-t", TMUX_SESSION],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        subprocess.run(["tmux", "new-session", "-d", "-s", TMUX_SESSION], check=True)
        print(f"Created tmux session: {TMUX_SESSION}")
    else:
        print(f"Using existing tmux session: {TMUX_SESSION}")


def run_condition(baseline, condition, panel_idx):
    output_file = str(RESULTS_DIR / f"phase3_sandbox_n20_{baseline}_{condition}.jsonl")
    cmd = (
        f"cd {BASE_DIR} && {PYTHON} {EXPERIMENT_SCRIPT} "
        f"--baseline {baseline} --condition {condition} "
        f"--num-episodes {N_EPISODES} --max-steps {MAX_STEPS} "
        f"--output {output_file} 2>&1"
    )
    if panel_idx == 0:
        subprocess.run(["tmux", "send-keys", "-t", f"{TMUX_SESSION}", cmd, "Enter"], check=True)
    else:
        subprocess.run(["tmux", "split-window", "-v", "-t", f"{TMUX_SESSION}"], check=True)
        subprocess.run(["tmux", "send-keys", "-t", f"{TMUX_SESSION}", cmd, "Enter"], check=True)
    print(f"Started {baseline}/{condition} in panel {panel_idx}")


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Phase 3 N=20 Experiment Runner (Sequential)")
    print(f"  Conditions: {len(CONDITIONS)}")
    print(f"  Episodes per condition: {N_EPISODES}")
    print(f"  Max steps: {MAX_STEPS}")
    print("=" * 60)

    ensure_tmux()

    for i, (baseline, condition) in enumerate(CONDITIONS):
        run_condition(baseline, condition, i)
        if i < len(CONDITIONS) - 1:
            time.sleep(5)

    print(f"\nAll conditions launched in tmux session '{TMUX_SESSION}'.")
    print("Attach with: tmux attach -t", TMUX_SESSION)
    print("Monitor with: tmux capture-pane -t", TMUX_SESSION, "-p")


if __name__ == "__main__":
    main()
