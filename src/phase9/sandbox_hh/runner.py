"""Phase 9 sandbox-hh: multi-episode runner.

Runs the two-layer open-loop agent over the 9 canonical Phase 8 tasks
(5 episodes x 9 tasks, max_steps=10 — same tasks, images, start cwds and
budget as the Phase 8 baseline for comparability).
"""

from __future__ import annotations

from typing import Dict, List

from .agent import SandboxHHAgent

# Task -> docker image mapping, mirrored from scripts/phase8_qw_eval.py
# (which mirrors the 2026-07-31 GPU baseline run).
TASKS: List[tuple] = [
    ("read_hello", "peda-sandbox:v2"),
    ("read_note", "peda-sandbox:v2"),
    ("count_lines", "peda-sandbox:v2"),
    ("find_secret", "peda-sandbox:v2"),
    ("read_welcome", "peda-sandbox:v4"),
    ("find_api_key", "peda-sandbox:v4"),
    ("count_measurements", "peda-sandbox:v4"),
    ("find_errors_v4", "peda-sandbox:v4"),
    ("read_changelog_v4", "peda-sandbox:v4"),
]


class SandboxHHRunner:
    """One arm (fixed lambda) over all tasks."""

    def __init__(self, lam: float) -> None:
        self.lam = float(lam)

    def run_task(self, task_id: str, image: str,
                 num_episodes: int = 5, max_steps: int = 10) -> List[dict]:
        agent = SandboxHHAgent(docker_image=image, task_id=task_id, lam=self.lam)
        episodes = [agent.run_episode(i, max_steps) for i in range(num_episodes)]
        for ep in episodes:
            ep["task"] = task_id
            ep["image"] = image
            ep["lam"] = self.lam
        return episodes

    def run_all(self, num_episodes: int = 5, max_steps: int = 10) -> Dict[str, List[dict]]:
        out: Dict[str, List[dict]] = {}
        for task_id, image in TASKS:
            out[task_id] = self.run_task(task_id, image, num_episodes, max_steps)
        return out
