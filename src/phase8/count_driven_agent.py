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
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Proven components — imported, not rewritten
from phase2.sandbox_env import BusyboxSandbox, SandboxState, generate_sandbox_candidates
from phase2.tasks import MICRO_TASKS
from phase5.explorer import NoveltyExplorer
from phase5.action_model import ActionModelLearner


def _get_task(task_id: str) -> dict:
    """Look up a task definition by id."""
    for t in MICRO_TASKS:
        if t["id"] == task_id:
            return t
    raise ValueError(f"Unknown task: {task_id}")


def _task_start_cwd(task_id: str) -> str:
    """Determine the starting cwd for a task.

    Most tasks start at /sandbox. Some deeper tasks benefit from
    starting in a relevant subdirectory.
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
                 device: str = "cpu"):
        self.docker_image = docker_image
        self.task = _get_task(task_id)
        self.task_id = task_id
        self.train_jepa = train_jepa
        self._start_cwd = _task_start_cwd(task_id)

        # Sandbox — environment + execution
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
