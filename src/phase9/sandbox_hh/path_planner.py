"""Phase 9 sandbox-hh: path-level frontier planner (FF-MLP-1).

The single-step planner (planner.py) picks a goal DIRECTORY and steers the
low layer toward it, one cd per step. FF-GEN-1 showed its structural limit
on the v5 deep tree: dist>=2 targets stayed 0/30 for every arm because of
two coupled effects (the T1 diagnosis):

  (a) dist-1 full-density frontiers monopolize selection — J ties break by
      dist ascending, so a depth-2 frontier never wins;
  (b) unknown subdirectories are excluded outright — a directory that was
      never visited has no candidate set, so its unvisited density is 0 and
      it is never a goal.

This module implements the path-level planner (design finalized in
local://contract-mlp1.md; executor detailed the scoring/tie-break below):

  Candidates   — frontier PATHS: cd chains cwd -> ... -> target, for every
                 known directory at any depth that is not fully explored.
                 The cwd itself is never a path target (a chain implies
                 movement; the single-step planner's "goal == cwd => the
                 low layer wanders" escape is exactly the T1 cold-start
                 dead end that burned 10 steps at the root).

  J(path)      = prior(end) - lam * depth,  depth = len(path) = #cd steps.

  prior(end)   — 0.5 (neutral) for directories that were never visited:
                 their candidate set is unknown, and the old density-0
                 rule threw every unexplored subdirectory out of
                 contention ((b) fixed at the root). For visited
                 directories the prior is the actual unvisited density
                 (Phase 8 verb x file candidates with zero counts in the
                 explorer's state_action_counts, same definition as the
                 single-step planner). prior <= 0 (fully explored or
                 known-empty) => not a frontier => excluded.

  Tie-break    (pre-registered before any run, see contract): J
                 descending, then depth DESCENDING (deeper frontier
                 preferred — the direct counterpart to the dist-ascending
                 tie-break diagnosed as half of T1; without it the lambda
                 dimension has nothing to act on and both arms collapse to
                 the single-step behavior), then lexicographic path string
                 for determinism.

  Navigation   — execute the cd chain (BFS path through the known graph,
                 passing through visited parent dirs toward unknown
                 depths); each landing point is handed to the Phase 8 low
                 layer. Open-loop: no mid-path re-evaluation; re-selection
                 only on local-frontier exhaustion or the R1 empty-dir
                 trap, both unchanged from the single-step agent.

The agent loop is the single-step agent's loop with the planner swapped:
SandboxPathAgent(SandboxHHAgent) overrides only __init__ (planner) and
_select_and_log (extended log fields); the low layer stays byte-identical
Phase 8 (generate_phase8_candidates + Phase8Explorer).
"""

from __future__ import annotations

from typing import Dict, List, Optional

from phase2.sandbox_env import BusyboxSandbox
from phase5.action_model import ActionModelLearner
from phase8.count_driven_agent import Phase8Explorer, _get_task, _task_start_cwd

from .agent import SandboxHHAgent
from .planner import DirGraph


class PathPlanner:
    """Path-level frontier selector (high layer). See module docstring."""

    def __init__(self, lam: float) -> None:
        self.lam = float(lam)
        self.graph = DirGraph()

    # ── density / prior ────────────────────────────────

    def unvisited_density(self, d: str, explorer) -> tuple:
        """(density, unvisited, total) for a VISITED dir — same definition
        as the single-step planner (verb x file candidates, zero counts)."""
        cands = self.graph.candidates(d)
        if not cands:
            return 0.0, 0, 0
        sh = f"{d}|{','.join(sorted(self.graph.entries[d]))}"
        counts = explorer.state_action_counts
        unvisited = sum(1 for c in cands if counts[(sh, c)] == 0)
        return unvisited / len(cands), unvisited, len(cands)

    def _visited(self, d: str, explorer) -> bool:
        """True when the explorer has actually visited d (counts persist
        across episodes, matching the planner graph's persistence)."""
        sh = f"{d}|{','.join(sorted(self.graph.entries[d]))}"
        return explorer.state_counts.get(sh, 0) > 0

    def prior(self, d: str, explorer) -> tuple:
        """(prior, unknown). Unknown (never-visited) dirs get the neutral
        prior 0.5; visited dirs get their actual unvisited density."""
        if self._visited(d, explorer):
            density, _u, _t = self.unvisited_density(d, explorer)
            return density, False
        return 0.5, True

    # ── selection ──────────────────────────────────────

    def select_goal(self, cwd: str, explorer) -> Optional[dict]:
        """Pick the frontier PATH maximizing J; None when no frontier.

        Returns a log dict with the chosen target, its prior, depth (=
        cd-chain length), J and the cd-action chain, plus top contenders.
        """
        scored = []
        for d in self.graph.known_dirs():
            if d == cwd:
                continue  # a path target is never the cwd itself (T1 fix)
            prior, unknown = self.prior(d, explorer)
            if prior <= 0:
                continue  # fully explored / known-empty -> not a frontier
            path = self.graph.shortest_path(cwd, d)
            if path is None:
                continue  # unreachable in the known graph -> not a goal
            depth = len(path)
            j = prior - self.lam * depth
            scored.append({
                "goal": d, "prior": round(prior, 4), "unknown": unknown,
                "depth": depth, "j": round(j, 4), "path": path,
            })
        if not scored:
            return None
        scored.sort(key=lambda r: (-r["j"], -r["depth"], r["goal"]))
        best = dict(scored[0])
        best["contenders"] = [
            {k: r[k] for k in ("goal", "prior", "unknown", "depth", "j", "path")}
            for r in scored[:5]
        ]
        return best


class SandboxPathAgent(SandboxHHAgent):
    """Two-layer open-loop agent with the PATH-level planner (FF-MLP-1).

    Inherits the whole episode loop from SandboxHHAgent (open-loop state
    machine, R1 empty-dir re-selection, replay preemption, Phase 8 low
    layer) and only swaps the high-layer planner plus extends the select
    log with the path fields.
    """

    def __init__(self, docker_image: str, task_id: str, lam: float) -> None:
        super().__init__(docker_image, task_id, lam)
        self.planner = PathPlanner(lam)

    def _select_and_log(self, cwd: str, t: int) -> Optional[dict]:
        """Select a frontier path, log the decision, return the goal dict."""
        sel = self.planner.select_goal(cwd, self.explorer)
        if sel is None:
            self.goal_log.append({
                "t": t, "event": "select", "goal": None, "prior": 0.0,
                "unknown": False, "depth": 0, "j": 0.0, "path": [],
                "contenders": [],
            })
            return None
        self.goal_log.append({
            "t": t, "event": "select", "goal": sel["goal"],
            "prior": sel["prior"], "unknown": sel["unknown"],
            "depth": sel["depth"], "j": sel["j"], "path": sel["path"],
            "contenders": sel["contenders"],
        })
        return sel
