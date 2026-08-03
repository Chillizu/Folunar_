"""Phase 9 sandbox-hh: two-layer open-loop agent (episode loop).

Layer composition (the ONLY variable vs Phase 8 is the high layer):

  High layer  — SandboxGoalPlanner: picks a frontier goal directory
                J(d) = unvisited_density(d) - lam*dist(cwd,d); the agent
                BFS-cds toward it (one cd per step).
  Low layer   — Phase 8, byte-identical: generate_phase8_candidates +
                Phase8Explorer.select_action + observe/record_cd, budget
                guard, STRIPS learner side-effect, success-cache replay.

Control state machine (open-loop, no mid-plan re-evaluation):

  select   — choose a goal (episode start, or after the local frontier at
             the current dir is exhausted). If the goal != cwd, enter
             navigate with its BFS cd path; else enter explore.
  navigate — follow the BFS path one cd per step until arrival (arrive ->
             explore).
  explore  — Phase 8 count-driven local exploration at the current dir;
             re-enter select when the local frontier (unvisited
             (verb x file) u cd candidates) is exhausted.

The low layer's cached-success replay ALWAYS preempts the high layer: a
known solution outranks any frontier heuristic, so when a replay is
available the Phase 8 explorer decides and the goal state machine is
deferred until the replay chain plays out.

Goals that equal the starting cwd never force navigation: the low layer
wanders freely (Phase 8 behavior) until a re-selection picks a dir that
is not the cwd.
"""

from __future__ import annotations

from typing import List, Optional

from phase2.sandbox_env import BusyboxSandbox
from phase5.action_model import ActionModelLearner
from phase8.count_driven_agent import (
    Phase8Explorer,
    _get_task,
    _task_start_cwd,
    generate_phase8_candidates,
)

from .planner import SandboxGoalPlanner


def _has_replay(explorer, state, candidates: List[str]) -> bool:
    """True when the Phase 8 explorer would replay a cached success now."""
    sh = state.state_hash()
    if sh in explorer.success_cache:
        if explorer.success_cache[sh] in candidates:
            return True
    for cand in candidates:
        if cand.startswith("cd ") and cand in explorer.cd_child.get(sh, {}):
            if explorer.cd_child[sh][cand] in explorer.success_cache:
                return True
    return False


class SandboxHHAgent:
    """Two-layer open-loop agent over the Phase 8 Docker sandbox."""

    def __init__(self, docker_image: str, task_id: str, lam: float) -> None:
        self.docker_image = docker_image
        self.task = _get_task(task_id)
        self.task_id = task_id
        self.lam = float(lam)
        self._start_cwd = _task_start_cwd(task_id)

        self.sandbox = BusyboxSandbox(image=docker_image)
        # Low layer: the exact Phase 8 explorer (wc-in-reader tier +
        # cached-success child revisit), untouched.
        self.explorer = Phase8Explorer()
        # High layer: frontier-goal planner (the experimental variable).
        self.planner = SandboxGoalPlanner(lam)
        # STRIPS side-effect learner (never consumed by selection, kept
        # for per-line parity with the Phase 8 loop).
        self.action_model = ActionModelLearner()

        self.buffer: list = []
        self.results: List[dict] = []

    # ── helpers ─────────────────────────────────────────

    def _local_frontier_exhausted(self, state) -> bool:
        density, _, _ = self.planner.unvisited_density(state.cwd, self.explorer)
        return density <= 0

    def _select_and_log(self, cwd: str, t: int) -> Optional[dict]:
        """Select a goal, log the decision event, return the goal dict."""
        sel = self.planner.select_goal(cwd, self.explorer)
        if sel is None:
            self.goal_log.append({
                "t": t, "event": "select", "goal": None, "density": 0.0,
                "dist": 0, "j": 0.0, "unvisited": 0, "total": 0,
                "contenders": [],
            })
            return None
        self.goal_log.append({
            "t": t, "event": "select", "goal": sel["goal"],
            "density": sel["density"], "dist": sel["dist"], "j": sel["j"],
            "unvisited": sel["unvisited"], "total": sel["total"],
            "contenders": sel["contenders"],
        })
        return sel

    def _choose_action(self, state, candidates, actions, t: int) -> str:
        """High/low dispatch for one step; returns the action string."""
        # 0. Cached-success replay preempts everything (Phase 8 mechanism).
        if _has_replay(self.explorer, state, candidates):
            return self.explorer.select_action(state, candidates, actions)

        # 1. select: pick a frontier goal (episode start / frontier exhausted)
        if self.mode == "select":
            sel = self._select_and_log(state.cwd, t)
            self.goal = sel["goal"] if sel else None
            if self.goal is not None and self.goal != state.cwd:
                p = self.planner.graph.shortest_path(state.cwd, self.goal)
                if p:
                    self.mode = "navigate"
                    self.path = p[1:]
                    return p[0]
            self.mode = "explore"

        # 2. navigate: follow the BFS path toward the goal
        if self.mode == "navigate":
            if state.cwd == self.goal:
                self.goal_log.append({"t": t, "event": "arrive", "goal": self.goal})
                self.mode = "explore"
            elif self.path:
                action = self.path[0]
                self.path = self.path[1:]
                return action
            else:
                self.mode = "explore"

        # 3. explore: the Phase 8 explorer decides (count novelty)
        return self.explorer.select_action(state, candidates, actions)

    # ── episode ─────────────────────────────────────────

    def run_episode(self, episode_idx: int, max_steps: int = 10) -> dict:
        """Run one episode; returns the per-episode record (WATCHDOG D4)."""
        state = self.sandbox.reset(seed=episode_idx, start_cwd=self._start_cwd)
        self.planner.graph.observe_cwd(state.cwd, state.files)
        self.explorer.reset_episode()

        self.mode = "select"             # select | navigate | explore
        self.goal: Optional[str] = None
        self.path: List[str] = []
        self.goal_log: List[dict] = []
        actions: List[str] = []
        success = False

        for t in range(max_steps):
            # 1. Perception — Phase 8 candidates, verbatim
            candidates = generate_phase8_candidates(state)
            if not candidates:
                candidates = ["ls", "pwd"]
            # Phase 8 budget guard: never enter a brand-new dir on the
            # last step (identical filter to Phase8Runner).
            if t >= max_steps - 1:
                known = getattr(self.explorer, "cd_child", {}).get(state.state_hash(), {})
                candidates = [
                    c for c in candidates
                    if not (c.startswith("cd ") and c != "cd .." and c not in known)
                ]
                if not candidates:
                    candidates = ["ls"]

            # 2. Act (high-layer dispatch, low-layer execution)
            action = self._choose_action(state, candidates, actions, t)

            # 3. Execute
            next_state, _reward, done = self.sandbox.step(state, action)

            # 4. Record cd knowledge (low-layer cache + high-layer graph)
            if action.startswith("cd "):
                self.explorer.record_cd(state, action, next_state)
                if next_state.cwd != state.cwd:
                    self.planner.graph.note_parent(next_state.cwd, state.cwd)
            if self.mode == "navigate" and next_state.cwd == self.goal and state.cwd != self.goal:
                self.goal_log.append({"t": t, "event": "arrive", "goal": self.goal})
                self.mode = "explore"

            # 5. Goal check (Phase 8 predicates, unchanged)
            check_fn = self.task.get("check")
            if check_fn is not None:
                try:
                    if check_fn(state, action, next_state):
                        success = True
                except Exception:
                    pass

            # 6. Low-layer feedback — identical to Phase 8
            self.explorer.observe(state, action, success)
            self.buffer.append((state, action, next_state, success))
            try:
                self.action_model.learn_from_step(state, action, next_state, success)
            except Exception:
                pass
            actions.append(action)

            if success or done:
                break

            # 7. Update graph from the new state (ls + find output)
            self.planner.graph.observe_cwd(next_state.cwd, next_state.files)
            if action.startswith("find "):
                self.planner.graph.observe_find(next_state.cwd, next_state.last_output)
            state = next_state

            # 8. Open-loop re-selection trigger: local frontier exhausted
            if self.mode == "explore" and self._local_frontier_exhausted(state):
                self.mode = "select"

        result = {
            "episode": episode_idx,
            "success": success,
            "steps": len(actions),
            "actions": actions,
            "buffer_size": len(self.buffer),
            "goal_log": self.goal_log,
        }
        self.results.append(result)
        return result
