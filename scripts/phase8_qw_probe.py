#!/usr/bin/env python3
"""Probe: log actual selection decisions of the count-driven agent.

Runs find_api_key (v4) 5 episodes and prints, per step, the chosen
action and the (novelty, priority) keys of the top candidates.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from phase2.sandbox_env import generate_sandbox_candidates
from phase5.explorer import NoveltyExplorer
from phase8.count_driven_agent import Phase8Runner


class LoggingExplorer(NoveltyExplorer):
    def select_action(self, state, candidates, action_history):
        sh = state.state_hash()
        cached = sh in self.success_cache
        scored = sorted(
            (
                (-self.novelty_bonus(state, a), self._action_priority(a), a)
                for a in candidates
            ),
            key=lambda t: (t[0], t[1]),
        )
        top = ", ".join(
            f"{t[2]}({t[0]:.3f},p{t[1]})" for t in scored[:4]
        )
        print(f"  sh={sh[:60]} cached={cached} | top: {top}", flush=True)
        return super().select_action(state, candidates, action_history)


runner = Phase8Runner(docker_image="peda-sandbox:v4", task_id="find_api_key")
runner.explorer = LoggingExplorer()
for ep in range(5):
    print(f"--- episode {ep} ---")
    r = runner.run_episode(max_steps=10)
    print(f"RESULT ep{ep}: success={r.success} actions={r.actions}")
