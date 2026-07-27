#!/usr/bin/env python3
"""Phase 3 Full Experiment Runner - launches all conditions in tmux.

Runs sequentially within each tmux panel to avoid GPU memory conflicts.
Conditions: peda_known, peda_unknown, pragmatic_known, pragmatic_unknown
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

BASE_DIR = Path("/home/ec2-user/Folunar_")
SCRIPTS_DIR = BASE_DIR / "scripts"
RESULTS_DIR = BASE_DIR / "results"
PYTHON = "/opt/pytorch/bin/python"
EXPERIMENT_SCRIPT = str(SCRIPTS_DIR / "phase3_sandbox_experiment.py")

# Conditions to run
CONDITIONS = [
    ("pragmatic", "known"),
    ("pragmatic", "unknown"),
    ("peda", "known"),
    ("peda", "unknown"),
]

TMUX_SESSION = "phase3_experiment"
N_EPISODES = 5
MAX_STEPS = 10


def ensure_tmux():
    """Start a tmux session if not already running."""
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
    """Run a single condition in a tmux panel."""
    output_file = str(RESULTS_DIR / f"phase3_sandbox_{baseline}_{condition}.jsonl")
    cmd = (
        f"cd {BASE_DIR} && {PYTHON} {EXPERIMENT_SCRIPT} "
        f"--baseline {baseline} --condition {condition} "
        f"--num-episodes {N_EPISODES} --max-steps {MAX_STEPS} "
        f"--output {output_file} 2>&1"
    )
    if panel_idx == 0:
        # Use the base window
        subprocess.run(["tmux", "send-keys", "-t", f"{TMUX_SESSION}", cmd, "Enter"], check=True)
    else:
        # Create a new panel
        subprocess.run(["tmux", "split-window", "-h", "-t", f"{TMUX_SESSION}"], check=True)
        subprocess.run(["tmux", "send-keys", "-t", f"{TMUX_SESSION}", cmd, "Enter"], check=True)
    print(f"Started {baseline}/{condition} in panel {panel_idx}")


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Phase 3 Full Experiment Runner")
    print(f"  Conditions: {len(CONDITIONS)}")
    print(f"  Episodes per condition: {N_EPISODES}")
    print(f"  Max steps: {MAX_STEPS}")
    print("=" * 60)

    ensure_tmux()

    # Run all conditions sequentially (one per panel)
    for i, (baseline, condition) in enumerate(CONDITIONS):
        run_condition(baseline, condition, i)
        time.sleep(2)  # Brief pause between launches

    print(f"\nAll conditions launched in tmux session '{TMUX_SESSION}'.")
    print("Attach with: tmux attach -t", TMUX_SESSION)
    print("Monitor with: tmux capture-pane -t", TMUX_SESSION, "-p")


if __name__ == "__main__":
    main()
