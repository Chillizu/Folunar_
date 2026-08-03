#!/usr/bin/env python3
# ruff: noqa: E402
"""Create a genuinely OOD held-out sandbox test set.

The default sandbox has /sandbox/{data,docs,hello.txt,tmp}.
This script resets the sandbox, creates a different layout under
/sandbox/project/{src,docs,notes}, runs a mix of random and heuristic
actions, and records (s,a,s') transitions for L1/L2/L3 evaluation.
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from phase2.sandbox_env import BusyboxSandbox, generate_sandbox_candidates

OOD_SETUP_COMMANDS = [
    "mkdir -p project/src project/docs",
    "echo 'module alpha' > project/src/main.py",
    "echo 'module beta' > project/src/util.py",
    "echo 'release notes v1' > project/docs/notes.txt",
    "echo 'secret token: OOD42' > project/docs/token.txt",
]


def run_episode(sb, max_steps: int, seed: int, baseline: str):
    rng = random.Random(seed)
    state = sb.reset()
    # Apply OOD setup
    for cmd in OOD_SETUP_COMMANDS:
        state, _, _ = sb.step(state, cmd)

    steps = []
    action_history = []
    for step_i in range(max_steps):
        cands = generate_sandbox_candidates(state)
        if not cands:
            cands = ["ls", "pwd"]

        if baseline == "random":
            action = rng.choice(cands)
        elif baseline == "heuristic":
            counts = {}
            for a in action_history[-5:]:
                counts[a] = counts.get(a, 0) + 1
            fresh = [c for c in cands if counts.get(c, 0) < 3]
            action = rng.choice(fresh) if fresh else rng.choice(cands)
        else:
            action = rng.choice(cands)

        next_state, _, done = sb.step(state, action)
        steps.append({
            "agent_type": baseline,
            "task_id": "ood",
            "step": step_i,
            "cwd": state.cwd,
            "files": list(state.files),
            "action": action,
            "next_cwd": next_state.cwd,
            "next_files": list(next_state.files),
            "exit_code": next_state.last_exit_code,
            "output": next_state.last_output[:100],
            "step_count": next_state.step_count,
        })
        state = next_state
        action_history.append(action)
        if done:
            break
    return steps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/phase2_ood_test.jsonl")
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--baselines", nargs="+", default=["random", "heuristic"])
    args = parser.parse_args()

    sb = BusyboxSandbox()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_transitions = 0
    for baseline in args.baselines:
        for ep in range(args.num_episodes):
            seed = hash(f"{baseline}_{ep}") % (2**31)
            steps = run_episode(sb, args.max_steps, seed, baseline)
            total_transitions += len(steps)
            record = {
                "baseline": baseline,
                "task": "ood",
                "steps_count": len(steps),
                "metrics": {"steps": len(steps)},
                "records": steps,
            }
            with open(output_path, "a") as f:
                f.write(json.dumps(record) + "\n")
            print(f"[ood] {baseline} ep {ep+1}: {len(steps)} steps", flush=True)

    print(f"[ood] Wrote {total_transitions} transitions to {output_path}")
    sb.close()


if __name__ == "__main__":
    main()
