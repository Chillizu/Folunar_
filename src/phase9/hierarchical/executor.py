"""Phase 9 hierarchical horizon: low-level pragmatic executors (§3).

The low level never sees J, H_plan, or the visit map — it is a pure
pragmatic path-finder (BFS on the known graph) or the Phase-6 count-based
novelty wanderer, identical across all experimental conditions, so any
difference in outcome is attributable to goal selection alone.
"""

from __future__ import annotations

import sys
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Protocol, Tuple

from .goals import Cell, Goal

_DIR: Dict[str, Tuple[int, int]] = {
    "north": (0, -1),
    "south": (0, 1),
    "east": (1, 0),
    "west": (-1, 0),
}
_NAME = {v: k for k, v in _DIR.items()}


def _load_phase6_explorer():
    """Import MazeNoveltyExplorer from scripts/phase6_maze_count.py."""
    try:
        from phase6_maze_count import MazeNoveltyExplorer  # type: ignore
        return MazeNoveltyExplorer
    except ImportError:
        repo = Path(__file__).resolve().parents[3]
        sys.path.insert(0, str(repo / "scripts"))
        from phase6_maze_count import MazeNoveltyExplorer  # type: ignore
        return MazeNoveltyExplorer


@dataclass
class StepRecord:
    """One executed step, reported back to the planner (§3)."""

    state_before: object  # GridState
    action: str
    state_after: object  # GridState
    cell_after: Cell
    content_new: bool
    item_acquired: Optional[str] = None
    door_unlocked: Optional[str] = None  # key name used
    goal_reached: bool = False
    items_seen: List[str] = field(default_factory=list)


class LowLevelExecutor(Protocol):
    """Receives ONE Goal from the planner; returns short pragmatic actions.

    Up to `horizon` (1-3) actions, re-planned every call (cheap).
    """

    def plan(self, state, goal: Goal, horizon: int = 3) -> List[str]: ...

    def step(self, state, goal: Goal) -> str: ...

    def on_goal_update(self, goal: Goal) -> None: ...


class BFSExecutor:
    """Shortest-path executor on the known graph (nav / acquire goals).

    `blocked(a, b)` tells whether the edge between adjacent cells is a
    still-locked door (from the planner's lock state).
    """

    def __init__(
        self,
        maze,
        blocked: Optional[Callable[[Cell, Cell], bool]] = None,
    ):
        self.maze = maze
        self._blocked = blocked or (lambda a, b: False)

    def _neighbors(self, cell: Cell) -> List[Cell]:
        x, y = cell
        out = []
        for dx, dy in _DIR.values():
            nb = (x + dx, y + dy)
            if not (0 <= nb[0] < self.maze.width and 0 <= nb[1] < self.maze.height):
                continue
            if not self.maze.is_passable(x, y, nb[0], nb[1]):
                continue
            if self._blocked(cell, nb):
                continue
            out.append(nb)
        return out

    def _bfs_path(self, start: Cell, targets: set) -> Optional[List[Cell]]:
        """Shortest path to the nearest target; None if unreachable."""
        parent = {start: None}
        queue = deque([start])
        while queue:
            c = queue.popleft()
            if c in targets:
                path = []
                while c is not None:
                    path.append(c)
                    c = parent[c]
                path.reverse()
                return path
            for nb in self._neighbors(c):
                if nb not in parent:
                    parent[nb] = c
                    queue.append(nb)
        return None

    @staticmethod
    def _go_action(a: Cell, b: Cell) -> str:
        dx, dy = b[0] - a[0], b[1] - a[1]
        return f"go {_NAME[(dx, dy)]}"

    def plan(self, state, goal: Goal, horizon: int = 3) -> List[str]:
        start = (state.x, state.y)

        if goal.kind == "nav" and goal.target is not None:
            path = self._bfs_path(start, {goal.target})
            if path is None:
                return []
            return [
                self._go_action(u, v) for u, v in zip(path, path[1:])
            ][:horizon]

        if goal.kind == "acquire":
            if goal.door_edge is not None:
                # Travel to the reachable endpoint of the door, then use the key.
                a, b = goal.door_edge
                path = self._bfs_path(start, {a})
                if path is None:
                    path = self._bfs_path(start, {b})
                if path is None:
                    return []
                steps = [self._go_action(u, v) for u, v in zip(path, path[1:])]
                if len(steps) < horizon and goal.item:
                    steps.append(f"use {goal.item} on door")
                return steps[:horizon]
            if goal.target is not None:
                path = self._bfs_path(start, {goal.target})
                if path is None:
                    return []
                steps = [self._go_action(u, v) for u, v in zip(path, path[1:])]
                if len(steps) < horizon and goal.item:
                    steps.append(f"take {goal.item}")
                return steps[:horizon]
        return []

    def step(self, state, goal: Goal) -> str:
        acts = self.plan(state, goal, 1)
        return acts[0] if acts else "look"

    def on_goal_update(self, goal: Goal) -> None:
        pass


class NoveltySearchExecutor:
    """Count-based novelty wanderer for `search` goals (§3).

    Wraps the Phase-6 MazeNoveltyExplorer (count novelty + backtrack penalty
    + success cache) — exactly the `flat_count` baseline policy, so the
    `layered` condition differs from the baseline only in goal selection.
    """

    def __init__(self):
        MazeNoveltyExplorer = _load_phase6_explorer()
        self._explorer = MazeNoveltyExplorer()
        self._action_history: List[str] = []

    @staticmethod
    def _candidates(state) -> List[str]:
        cands = ["go north", "go south", "go east", "go west", "look", "inventory"]
        for item in state.visible_items or []:
            cands.append(f"take {item}")
        for item in state.inventory or []:
            cands.append(f"use {item} on door")
        return cands

    def plan(self, state, goal: Goal, horizon: int = 3) -> List[str]:
        return [self.step(state, goal)]

    def step(self, state, goal: Goal) -> str:
        # Goal-aware grab: the planner said *what* to find — when it is in
        # sight, taking it beats wandering (the Phase-6 novelty priority
        # prefers `go` over `take`, which would walk past the key forever).
        if goal.kind == "search" and goal.item:
            for item in state.visible_items or []:
                if goal.item.lower() in item.lower() or item.lower() in goal.item.lower():
                    return f"take {item}"
        return self._explorer.select_action(
            state, self._candidates(state), self._action_history
        )

    def observe(self, state, action: str, success: bool) -> None:
        self._explorer.observe(state, action, success)
        self._action_history.append(action)

    def observe_move(self, action: str, pos_before: Cell, pos_after: Cell) -> None:
        self._explorer.observe_move(action, pos_before, pos_after)

    def on_goal_update(self, goal: Goal) -> None:
        """New commitment — drop stale backtrack/plan state."""
        self._explorer.reset_episode()
        self._action_history = []


class LayeredExecutor:
    """Routes each goal kind to its low-level policy (§3 table).

    nav / acquire -> BFSExecutor (shortest path on known graph)
    search        -> NoveltySearchExecutor (count novelty + backtrack)
    """

    def __init__(
        self,
        maze,
        blocked: Optional[Callable[[Cell, Cell], bool]] = None,
    ):
        self.bfs = BFSExecutor(maze, blocked=blocked)
        self.search = NoveltySearchExecutor()

    def _executor_for(self, goal: Goal):
        return self.search if goal.kind == "search" else self.bfs

    def plan(self, state, goal: Goal, horizon: int = 3) -> List[str]:
        return self._executor_for(goal).plan(state, goal, horizon)

    def step(self, state, goal: Goal) -> str:
        return self._executor_for(goal).step(state, goal)

    def observe(
        self,
        state,
        action: str,
        success: bool,
        moved: Optional[Tuple[Cell, Cell]] = None,
    ) -> None:
        # One novelty explorer observes every action (both policies) so its
        # counts stay honest about everything the agent sees — this matches
        # the Phase-6 bookkeeping.
        self.search.observe(state, action, success)
        if moved:
            self.search.observe_move(action, moved[0], moved[1])

    def on_goal_update(self, goal: Goal) -> None:
        self.search.on_goal_update(goal)
