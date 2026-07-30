#!/usr/bin/env python3
"""Dynamic candidate generator + minimal explorer.

No LLM World Model. No EFE. No token prediction.

Candidate generation:
- Decompose whitelist into verbs and state files into typed targets
- Track success/failure per (verb, target_type, flags) template
- Generate untried combinations with exploration bonus
- Automatically generalizes: "cat .txt works" → try cat on ANY .txt file

Exploration:
- Prefer unseen (state, action) pairs
- Replay cached success paths
- Fall back to random

This is 100% tabular — no neural network, no gradient, no LLM.
"""

import argparse, json, os, sys, time, random
from pathlib import Path
from collections import defaultdict

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from phase2.sandbox_env import BusyboxSandbox, _validate_command
from phase2.tasks import MICRO_TASKS


# ── Verbs ──
VERBS = ["ls", "pwd", "cd", "cat", "echo", "mkdir", "touch", "wc", "head", "tail", "grep", "find"]
CONTENT_VERBS = {"cat", "head", "tail", "wc", "grep", "echo"}
NAV_VERBS = {"cd"}
NAV_TARGETS = {"docs", "data", "logs", "projects", "tmp", "app", "lib"}


class DynamicCandidateGenerator:
    """Generates candidates from learned (verb, target_type) success stats."""

    def __init__(self):
        # (verb, target_type, flag) -> (attempts, successes)
        self.stats = defaultdict(lambda: [0, 0])
        self.max_candidates = 12

    def _target_type(self, target: str) -> str:
        """Classify a file/dir into a type for generalization."""
        if target in NAV_TARGETS or target == "..":
            return "dir"
        if target == ".":
            return "cwd"
        if "." in target:
            ext = target.rsplit(".", 1)[-1]
            if ext in ("txt", "csv", "log", "ini", "json", "py", "yml"):
                return ext
        return "file"

    def _score(self, verb: str, target: str, flag: str = None) -> float:
        """Success rate + exploration bonus. 0.0-1.0+."""
        key = (verb, self._target_type(target), flag)
        attempts, successes = self.stats[key]
        if attempts == 0:
            return 0.15  # untried — medium priority
        base = successes / max(attempts, 1)
        bonus = 0.05 / max(attempts, 1)  # decay exploration over time
        return base + bonus

    def update(self, command: str, success: bool):
        """Update stats from command execution result."""
        parts = command.strip().split()
        if not parts:
            return
        verb = parts[0]
        if verb not in VERBS:
            return
        target = parts[1] if len(parts) > 1 and not parts[1].startswith("-") else None
        flag = parts[1] if len(parts) > 1 and parts[1].startswith("-") else (
            parts[2] if len(parts) > 2 and parts[2].startswith("-") else None
        )

        key = (verb, self._target_type(target) if target else "none", flag)
        self.stats[key][0] += 1
        if success:
            self.stats[key][1] += 1

    def generate(self, state) -> list:
        """Generate top candidates for the current sandbox state."""
        scored = []

        cwd = state.cwd.rstrip("/")
        files = state.files

        # ── Always: ls, pwd ──
        scored.append(("ls", self._score("ls", ".")))
        scored.append(("pwd", self._score("pwd", ".")))

        # ── cd .. when in subdir ──
        if cwd != "/sandbox":
            scored.append(("cd ..", self._score("cd", "..")))

        # ── cd into dirs ──
        for f in files:
            if f in NAV_TARGETS or (f in ("app", "lib") and cwd != "/sandbox"):
                scored.append((f"cd {f}", self._score("cd", f)))

        # ── Content verbs × visible files ──
        for f in files:
            if f in NAV_TARGETS:
                continue
            for v in CONTENT_VERBS:
                cmd = f"{v} {f}"
                scored.append((cmd, self._score(v, f)))

        # ── grep with flags + find patterns ──
        for flag in ["-r"]:
            for t in [".", "docs/", "data/", "logs/", "projects/"]:
                for pattern in ["secret", "ERROR", "admin", "hello", "v2"]:
                    cmd = f"grep {flag} {pattern} {t}"
                    scored.append((cmd, self._score("grep", pattern, flag)))

        # ── find patterns ──
        for pattern in ["*.txt", "*.log", "*.csv"]:
            cmd = f"find . -name '{pattern}'"
            scored.append((cmd, self._score("find", pattern)))

        # ── mkdir + touch combos ──
        scored.append(("mkdir test_dir", self._score("mkdir", "test_dir")))
        scored.append(("touch newfile.txt", self._score("touch", "newfile.txt")))

        # Dedup, sort by score, cap
        seen = set()
        unique = []
        for cmd, score in scored:
            if cmd not in seen and _validate_command(cmd)[0]:
                seen.add(cmd)
                unique.append((cmd, score))

        unique.sort(key=lambda x: -x[1])
        return [cmd for cmd, _ in unique[:self.max_candidates]]


class MinimalExplorer:
    """Explores by trying unseen (state, action) pairs, caching successes."""

    def __init__(self, candidate_gen: DynamicCandidateGenerator):
        self.seen_pairs = set()
        self.success_cache = {}
        self.cgen = candidate_gen

    def state_key(self, state) -> tuple:
        return (state.cwd, tuple(sorted(state.files)))

    def select_action(self, state, action_history):
        key = self.state_key(state)
        candidates = self.cgen.generate(state)

        # 1. Cached success
        if key in self.success_cache:
            cached = self.success_cache[key]
            if cached in candidates:
                return cached

        # 2. Unseen (state, action)
        unseen = [a for a in candidates if (key, a) not in self.seen_pairs]
        if unseen:
            return random.choice(unseen)

        # 3. Least-recent in episode
        for a in candidates:
            if a not in action_history:
                return a

        return random.choice(candidates) if candidates else "ls"

    def observe(self, state, action, success: bool):
        key = self.state_key(state)
        self.seen_pairs.add((key, action))
        self.cgen.update(action, success)
        if success:
            self.success_cache[key] = action


def run_episode(sb, explorer, task_id, max_steps, start_cwd):
    task_def = next((t for t in MICRO_TASKS if t["id"] == task_id), None)
    state = sb.reset(start_cwd=start_cwd)
    action_history = []
    steps = []

    for step_i in range(max_steps):
        action = explorer.select_action(state, action_history)
        next_state, reward, done = sb.step(state, action)
        success = task_def and task_def["check"](state, action, next_state)

        if success:
            next_state.victory = True
            next_state.game_over = True
            done = True
            print(f"  [step {step_i}] VICTORY! {action}", flush=True)

        explorer.observe(state, action, success)
        steps.append(dict(step=step_i, cwd=state.cwd, action=action,
                           success=success, next_cwd=next_state.cwd))
        action_history.append(action)
        state = next_state
        if done:
            break

    fht = next((i for i, s in enumerate(steps) if s["success"]), -1)
    return steps, fht


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="read_hello")
    parser.add_argument("--num-episodes", type=int, default=12)
    parser.add_argument("--max-steps", type=int, default=10)
    args = parser.parse_args()

    sb = BusyboxSandbox()
    cgen = DynamicCandidateGenerator()
    explorer = MinimalExplorer(cgen)

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
        print(f"[ep {ep+1:2d}] cwd={cwd:20s} fht={fht:2d} steps={len(steps):2d} hit={hit} [{elapsed:.0f}s]",
              flush=True)

    print(f"\n{args.task}: {hits}/{args.num_episodes} ({hits/args.num_episodes:.0%})",
          flush=True)

    # Show learned stats
    print(f"\nLearned templates ({len(cgen.stats)} entries):")
    sorted_stats = sorted(cgen.stats.items(),
                          key=lambda x: x[1][1]/max(x[1][0],1), reverse=True)
    for (verb, ttype, flag), (att, succ) in sorted_stats[:15]:
        rate = succ / max(att, 1)
        print(f"  {verb:8s} {str(ttype):10s} {str(flag):5s}  "
              f"{succ}/{att} ({rate:.0%})", flush=True)

    sb.close()


if __name__ == "__main__":
    main()
