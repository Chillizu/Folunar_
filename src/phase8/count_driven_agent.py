"""Phase 8: Count-Driven Closed-Loop Agent.

Assembles proven Phase 2 and Phase 5 components into a single
autonomous agent loop. Novelty (not prediction error) drives exploration.
STRIPS action schemas are learned from traces. JEPA forward dynamics
training is optional.

Count-driven agent control flow:
    Perception     → generate_sandbox_candidates()
    Action Gen     → NoveltyExplorer.select_action()
    Action Exec    → BusyboxSandbox.step()
    Learning       → ActionModelLearner + JEPAEnsemble (optional)
"""
import argparse
import json
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Make `src` importable when run as `python -m src.phase8.count_driven_agent`
# (sys.path[0] is the repo root then, not src/).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Proven components — imported, not rewritten
from phase2.sandbox_env import (
    BusyboxSandbox,
    CounterIntuitiveSandbox,
    SandboxState,
    generate_sandbox_candidates,
)
from phase2.tasks import MICRO_TASKS
from phase5.action_model import ActionModelLearner
from phase5.explorer import NoveltyExplorer


def _get_task(task_id: str) -> dict:
    """Look up a task definition by id."""
    for t in MICRO_TASKS:
        if t["id"] == task_id:
            return t
    raise ValueError(f"Unknown task: {task_id}")


def _task_start_cwd(task_id: str) -> str:
    """Determine the starting cwd for a task.

    Most tasks start at /sandbox. Some deeper tasks benefit from
    starting in a relevant subdirectory. CI tasks (read_secret_ci,
    read_data_ci, find_warn_ci) are not in deep_tasks, so they use the
    /sandbox default per the counter-intuitive sandbox contract.
    """
    deep_tasks = {
        "count_measurements": "/sandbox/data/raw",
        "find_errors_v4": "/sandbox/logs",
        "read_changelog_v4": "/sandbox/docs",
        "find_errors": "/sandbox/logs",
        "count_logs": "/sandbox/logs",
    }
    return deep_tasks.get(task_id, "/sandbox")


# ── Result type ─────────────────────────────────────────


@dataclass
class EpisodeResult:
    """Outcome of one episode."""
    episode: int
    success: bool
    steps: int
    actions: List[str] = field(default_factory=list)
    buffer_size: int = 0

    def to_dict(self) -> dict:
        return {
            "episode": self.episode,
            "success": self.success,
            "steps": self.steps,
            "actions": self.actions,
            "buffer_size": self.buffer_size,
        }


# ── Runner ──────────────────────────────────────────────


class Phase8Runner:
    """Count-driven closed-loop agent.

    No prediction-error mechanism. Novelty = count-based bonus.
    STRIPS action schemas are learned from experience.
    JEPA forward dynamics training is optional (--train-jepa flag).

    Count-based novelty selects actions. Docker executes them.
    ActionModelLearner extracts schemas from traces.
    Optional JEPA trains forward dynamics as a side effect.

    Winning (cwd,action) pairs are memoized for fast reuse.
    """
    def __init__(self, docker_image: str = "peda-sandbox:v2",
                 task_id: str = "read_hello",
                 model_path: Optional[str] = None,
                 train_jepa: bool = False,
                 device: str = "cpu",
                 ci: bool = False):
        self.docker_image = docker_image
        self.task = _get_task(task_id)
        self.task_id = task_id
        self.train_jepa = train_jepa
        self.ci = ci
        self._start_cwd = _task_start_cwd(task_id)

        # Sandbox — environment + execution.
        # --ci switches to the counter-intuitive sandbox (reversed command
        # semantics, writable rootfs) for the CI micro-tasks.
        if ci:
            self.sandbox = CounterIntuitiveSandbox()
        else:
            self.sandbox = BusyboxSandbox(image=docker_image)

        # Explorer — count-based action selection
        self.explorer = NoveltyExplorer()

        # Action model — STRIPS schema learner
        self.action_model = ActionModelLearner()

        # JEPA — forward dynamics (optional, training only)
        self.jepa = None
        if train_jepa and model_path:
            try:
                from phase5.jepa_wm import JEPAEnsemble
                self.jepa = JEPAEnsemble(model_path, n_ensemble=3, device=device)
            except ImportError as e:
                print(f"[Phase8] JEPA import failed: {e}")
            except Exception as e:
                print(f"[Phase8] JEPA init failed: {e}")

        # Training buffer
        self.buffer: List[Tuple[SandboxState, str, SandboxState, bool]] = []

        # Results
        self.results: List[EpisodeResult] = []

    # ── Episode ─────────────────────────────────────────

    def run_episode(self, max_steps: int = 10) -> EpisodeResult:
        """Run one episode. Returns EpisodeResult."""
        ep = len(self.results)
        result = EpisodeResult(episode=ep, success=False, steps=0)

        # Reset sandbox — start fresh container
        state = self.sandbox.reset(seed=ep, start_cwd=self._start_cwd)
        prev_state = state

        # Clear episode-level explorer state (counts persist across episodes)
        self.explorer.reset_episode()

        for t in range(max_steps):
            # 1. Perception — generate candidate actions from current state
            candidates = generate_sandbox_candidates(state)
            if not candidates:
                candidates = ["ls", "pwd"]

            # 2. Action Generator — select via count-based novelty
            action = self.explorer.select_action(state, candidates, result.actions)

            # 3. Action Executor — run in sandbox
            next_state, reward, done = self.sandbox.step(state, action)

            # Record the attempt
            success = False

            # 4. Check goal
            check_fn = self.task.get("check")
            if check_fn is not None:
                try:
                    if check_fn(prev_state if False else state, action, next_state):
                        success = True
                except Exception:
                    pass

            # 5. Feed back to explorer (updates counts + success cache)
            self.explorer.observe(state, action, success)

            # 6. Buffer transition
            self.buffer.append((state, action, next_state, success))

            # 7. Learning — STRIPS step update
            try:
                self.action_model.learn_from_step(state, action, next_state, success)
            except Exception:
                pass

            # Record
            result.actions.append(action)

            if success:
                result.success = True
                break
            if done:
                break

            state = next_state
            prev_state = state

        result.steps = len(result.actions)
        result.buffer_size = len(self.buffer)

        # Post-episode: JEPA batch training
        if self.jepa is not None and len(self.buffer) >= 5:
            try:
                recent = self.buffer[-20:]
                # train_step expects list of (state_obj, action_str, next_state_obj)
                jepa_transitions = [(s, a, ns) for s, a, ns, _ in recent]
                loss = self.jepa.train_step(jepa_transitions)
            except Exception as e:
                print(f"[Phase8] JEPA train error: {e}")

        self.results.append(result)
        return result

    # ── Multi-episode run ───────────────────────────────

    def run(self, num_episodes: int = 10, max_steps: int = 10) -> List[dict]:
        """Run multiple episodes, returning dict summaries."""
        for ep in range(num_episodes):
            result = self.run_episode(max_steps)
            status = "OK" if result.success else "FAIL"
            print(f"  Episode {ep}: {status} in {result.steps} steps ({result.buffer_size} transitions)", flush=True)
        return [r.to_dict() for r in self.results]


# ── CLI ─────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 8 Count-Driven Agent")
    parser.add_argument("--task", default="read_hello",
                        help="Task id (CI tasks: read_secret_ci, read_data_ci, find_warn_ci)")
    parser.add_argument("--ci", action="store_true",
                        help="Use the counter-intuitive sandbox (peda-sandbox:counterintuitive-v2, writable rootfs) and CI tasks")
    parser.add_argument("--docker-image", default="peda-sandbox:v2",
                        help="Docker image for the normal sandbox (ignored with --ci)")
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--train-jepa", action="store_true")
    parser.add_argument("--model-path", default=None,
                        help="Path to Qwen model for JEPA (e.g. ~/models/Qwen2.5-0.5B-Instruct)")

    args = parser.parse_args()

    ci_image = "peda-sandbox:counterintuitive-v2"
    print("Phase 8: Count-Driven Closed-Loop Agent")
    print(f"  Task: {args.task}")
    print(f"  Environment: {'counter-intuitive (CI)' if args.ci else 'normal'}")
    print(f"  Docker image: {ci_image if args.ci else args.docker_image}")
    print(f"  Episodes: {args.num_episodes}")
    print(f"  Max steps per episode: {args.max_steps}")
    print()

    runner = Phase8Runner(
        docker_image=args.docker_image,
        task_id=args.task,
        model_path=args.model_path,
        train_jepa=args.train_jepa,
        ci=args.ci,
    )

    results = runner.run(
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
    )

    success = sum(1 for r in results if r["success"])
    total = len(results)
    avg_steps = sum(r["steps"] for r in results) / max(total, 1)

    summary = {
        "phase": 8,
        "task": args.task,
        "ci": args.ci,
        "docker_image": ci_image if args.ci else args.docker_image,
        "episodes": total,
        "success": success,
        "success_rate": f"{success}/{total} ({success/total*100:.0f}%)" if total > 0 else "0/0",
        "jepa_training": args.train_jepa,
        "avg_steps": round(avg_steps, 1),
    }

    print()
    print(json.dumps(summary, indent=2))

    # ── WATCHDOG D4: per-episode JSONL artifact (aggregates never replace raw data) ──
    import socket
    import subprocess as _sp
    from datetime import datetime, timezone

    meta = {
        "phase": "9",
        "direction": "counter-intuitive-sandbox" if args.ci else "count-driven-baseline",
        "commit": _sp.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip() or "unknown",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "cpu_or_gpu": "cpu",
        "sandbox_image": ci_image if args.ci else args.docker_image,
        "model": "count-based novelty (no model)" if not args.train_jepa else (args.model_path or "unknown"),
        "seeds": list(range(total)),
        "per_episode_data_present": True,
    }
    out_path = Path(f"results/phase8_{args.task}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        f.write(json.dumps({"meta": meta}) + "\n")
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"Per-episode JSONL artifact: {out_path}")

    return 0 if success > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
