#!/usr/bin/env python3
"""Minimal explorer: no WM, just curiosity + success caching.

Core loop:
- Track seen (cwd, files, action) triples — don't repeat
- Track successful (cwd, files) → action mappings — replay when possible
- Exploration: prefer unseen actions, fall back to random
- Learning: cache successful paths for future episodes

No LLM World Model. No EFE. No token prediction.
Just: try things → observe → remember.
"""

import argparse, json, os, sys, time, random
from pathlib import Path
from collections import defaultdict

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from phase2.sandbox_env import BusyboxSandbox, generate_sandbox_candidates
from phase2.tasks import MICRO_TASKS


class MinimalExplorer:
    """Explores by trying unseen (state, action) pairs, caching successes."""

    def __init__(self):
        self.seen_pairs = set()       # (cwd, files_tuple, action) triples
        self.success_cache = {}       # (cwd, files_tuple) → action that succeeded
        self.episode_successes = []   # for logging

    def state_key(self, state) -> tuple:
        return (state.cwd, tuple(sorted(state.files)))

    def select_action(self, state, candidates, action_history):
        key = self.state_key(state)

        # 1. Cached success: replay known winning action
        if key in self.success_cache:
            cached = self.success_cache[key]
            if cached in candidates:
                return cached

        # 2. Exploration: prefer unseen (state, action) pairs
        unseen = [a for a in candidates if (key, a) not in self.seen_pairs]
        if unseen:
            return random.choice(unseen)

        # 3. Fallback: least-recently-used in this episode
        for a in candidates:
            if a not in action_history:
                return a

        return random.choice(candidates)

    def observe(self, state, action, success: bool):
        key = self.state_key(state)
        self.seen_pairs.add((key, action))
        if success:
            self.success_cache[key] = action


def run_episode(sb, explorer, task_id, max_steps, start_cwd):
    task_def = next((t for t in MICRO_TASKS if t["id"] == task_id), None)
    state = sb.reset(start_cwd=start_cwd)
    steps = []
    action_history = []

    for step_i in range(max_steps):
        candidates = generate_sandbox_candidates(state)
        action = explorer.select_action(state, candidates, action_history)

        next_state, reward, done = sb.step(state, action)
        success = task_def and task_def["check"](state, action, next_state)

        if success:
            next_state.victory = True
            next_state.game_over = True
            done = True
            print(f"  [step {step_i}] VICTORY! action={action}", flush=True)

        explorer.observe(state, action, success)
        steps.append({"step": step_i, "cwd": state.cwd, "action": action,
                       "next_cwd": next_state.cwd, "success": success})
        action_history.append(action)
        state = next_state

        if done:
            break

    fht = next((i for i, s in enumerate(steps) if s["success"]), -1)
    return steps, fht


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="read_hello")
    parser.add_argument("--num-episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    sb = BusyboxSandbox()
    explorer = MinimalExplorer()

    all_cwds = ["/sandbox", "/sandbox/data", "/sandbox/docs",
                "/sandbox/logs", "/sandbox/projects", "/sandbox/tmp"]

    hits = 0
    for ep in range(args.num_episodes):
        cwd = all_cwds[ep % len(all_cwds)]
        t0 = time.time()
        steps, fht = run_episode(sb, explorer, args.task, args.max_steps, cwd)
        elapsed = time.time() - t0
        hit = 1 if fht >= 0 else 0
        hits += hit
        print(f"[ep {ep+1}] cwd={cwd} fht={fht} steps={len(steps)} hit={hit} [{elapsed:.0f}s]", flush=True)

    print(f"\nHits: {hits}/{args.num_episodes} ({hits/args.num_episodes:.0%})", flush=True)
    sb.close()


if __name__ == "__main__":
    main()
