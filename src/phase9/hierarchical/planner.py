"""Phase 9 hierarchical horizon: high-level epistemic goal planner.

Implements `PEDA_FINAL/phase9/plans/plan-hierarchical-horizon.md`:
  - §2.1 PlannerState — visit map, observed items, inventory, door locks
  - §2.2 goal space — frontier goals + key goals
  - §2.3 exact info-gain scoring  J(f) = G(f) - lambda*d
  - §2.4 selection (argmax J; ties -> smaller d)
  - §4 re-evaluation with hysteresis (tau)

Knowledge protocol (Validation Protocol A): geometry (walls + locked door
edges) is known; cell contents (items, keys, the goal room) are hidden until
visited. The planner NEVER introspects the environment — it only learns from
StepRecords reported by the executor.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from .goals import Cell, Goal

INF = float("inf")


@dataclass
class PlannerState:
    """Everything the high-level planner knows about the world.

    graph     : GridMaze geometry provider (walls + doors only)
    visits    : cell -> visit count (content seen)
    items     : cell -> [item] observed at that cell
    inventory : [item] held by the agent
    locked    : door edge -> key name (geometry, known up front)
    unlocked  : set of door edges opened so far
    pos       : current cell
    """

    maze: object
    visits: Dict[Cell, int] = field(default_factory=dict)
    items: Dict[Cell, List[str]] = field(default_factory=dict)
    inventory: List[str] = field(default_factory=list)
    locked: Dict[Tuple[Cell, Cell], str] = field(default_factory=dict)
    unlocked: Set[Tuple[Cell, Cell]] = field(default_factory=set)
    pos: Cell = (0, 0)

    def all_cells(self) -> List[Cell]:
        return [
            (x, y)
            for x in range(self.maze.width)
            for y in range(self.maze.height)
        ]


def _norm_edge(a: Cell, b: Cell) -> Tuple[Cell, Cell]:
    """Normalize an undirected edge to (min_cell, max_cell) order."""
    return (a, b) if (a[0], a[1]) <= (b[0], b[1]) else (b, a)


class HighLevelPlanner:
    """Epistemic goal planner with an analytic delayed-reward estimator.

    The planner scores goals by *expected new states within its horizon*
    (H_plan = 20..100) — the quantity a flat horizon-1..3 selector provably
    cannot see. It is deterministic and exact on the known maze graph.
    """

    def __init__(self, maze, pos: Cell = (0, 0), kappa: float = 0.5):
        self.maze = maze
        self.kappa = kappa  # unknown-key uncertainty penalty (§2.3)
        self.state = PlannerState(maze=maze, pos=pos)
        for (x1, y1, x2, y2), key in maze.locked_doors.items():
            self.state.locked[((x1, y1), (x2, y2))] = key
        self.current_goal: Optional[Goal] = None
        self._last_G: Optional[float] = None

    # ── knowledge ingestion ──────────────────────────────

    def seed_observation(self, cell: Cell, items_seen: List[str]) -> None:
        """Record the initial observation (start room) before the first step."""
        self.state.visits[cell] = self.state.visits.get(cell, 0) + 1
        self.state.pos = cell
        if items_seen:
            self.state.items[cell] = list(items_seen)

    def update(self, rec) -> None:
        """§4: mutate visit map / items / doors from one executed step."""
        cell = rec.cell_after
        self.state.visits[cell] = self.state.visits.get(cell, 0) + 1
        self.state.pos = cell

        if rec.items_seen:
            self.state.items[cell] = list(rec.items_seen)
        if rec.item_acquired:
            observed = self.state.items.get(cell, [])
            for it in list(observed):
                if (
                    rec.item_acquired.lower() in it.lower()
                    or it.lower() in rec.item_acquired.lower()
                ):
                    observed.remove(it)
            if rec.item_acquired not in self.state.inventory:
                self.state.inventory.append(rec.item_acquired)
        if rec.door_unlocked:
            for edge, key in self.state.locked.items():
                if key == rec.door_unlocked:
                    self.state.unlocked.add(edge)

    # ── geometry helpers ─────────────────────────────────

    def is_edge_blocked(self, a: Cell, b: Cell) -> bool:
        """True if the edge is a still-locked door (planner lock state)."""
        e = _norm_edge(a, b)
        return e in self.state.locked and e not in self.state.unlocked

    def _neighbors_with(
        self, cell: Cell, open_edge: Optional[Tuple[Cell, Cell]] = None
    ) -> List[Cell]:
        """Passable neighbors; a locked edge is passable iff it is unlocked
        or equals `open_edge` (temporarily open for gain estimation)."""
        x, y = cell
        out = []
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nb = (x + dx, y + dy)
            if not (0 <= nb[0] < self.maze.width and 0 <= nb[1] < self.maze.height):
                continue
            if not self.maze.is_passable(x, y, nb[0], nb[1]):
                continue
            e = _norm_edge(cell, nb)
            if e in self.state.locked and e not in self.state.unlocked:
                if open_edge is None or e != _norm_edge(*open_edge):
                    continue
            out.append(nb)
        return out

    def _reachable_set(
        self,
        start: Cell,
        open_edge: Optional[Tuple[Cell, Cell]] = None,
        max_steps: Optional[int] = None,
    ) -> Set[Cell]:
        """BFS-reachable cells from start (optionally within max_steps)."""
        seen = {start}
        queue = deque([(start, 0)])
        while queue:
            c, depth = queue.popleft()
            if max_steps is not None and depth >= max_steps:
                continue
            for nb in self._neighbors_with(c, open_edge):
                if nb not in seen:
                    seen.add(nb)
                    queue.append((nb, depth + 1))
        return seen

    def _dist(self, start: Cell, target: Cell) -> float:
        """BFS distance on the known graph; INF if unreachable."""
        if start == target:
            return 0.0
        dist = {start: 0}
        queue = deque([start])
        while queue:
            c = queue.popleft()
            for nb in self._neighbors_with(c):
                if nb in dist:
                    continue
                dist[nb] = dist[c] + 1
                if nb == target:
                    return dist[nb]
                queue.append(nb)
        return INF

    # ── goal space (§2.2) ────────────────────────────────

    def _frontier_cells(self) -> List[Cell]:
        """Visited cells with an unvisited passable neighbor.

        Lemma (§2.2): every path to an unvisited cell passes through one of
        these, so frontier goals dominate deeper-cell goals.
        """
        out = []
        for cell, count in self.state.visits.items():
            if count <= 0:
                continue
            for nb in self._neighbors_with(cell):
                if self.state.visits.get(nb, 0) == 0:
                    out.append(cell)
                    break
        return out

    def _region_gain_unvisited(self, edge: Tuple[Cell, Cell]) -> int:
        """Unvisited cells unlocked by opening `edge` (marginal reachable set)."""
        before = self._reachable_set(self.state.pos)
        after = self._reachable_set(self.state.pos, open_edge=edge)
        gain = after - before
        return sum(1 for c in gain if self.state.visits.get(c, 0) == 0)

    def _known_key_cell(self, item: str) -> Optional[Cell]:
        for cell, items in self.state.items.items():
            for it in items:
                if item.lower() in it.lower() or it.lower() in item.lower():
                    return cell
        return None

    def _door_dist(self, edge: Tuple[Cell, Cell]) -> float:
        a, b = edge
        return min(self._dist(self.state.pos, a), self._dist(self.state.pos, b))

    def _door_near(self, edge: Tuple[Cell, Cell]) -> Cell:
        a, b = edge
        return a if self._dist(self.state.pos, a) <= self._dist(self.state.pos, b) else b

    def propose_goals(self) -> List[Goal]:
        """§2.2: frontier goals + key goals (search / acquire per door)."""
        goals: List[Goal] = []
        for f in self._frontier_cells():
            goals.append(Goal(kind="nav", target=f))

        total_unvisited = sum(
            1 for c in self.state.all_cells() if self.state.visits.get(c, 0) == 0
        )
        for edge, key in self.state.locked.items():
            if edge in self.state.unlocked:
                continue
            gain = self._region_gain_unvisited(edge)
            if gain < 0.1 * total_unvisited:
                continue  # don't key-hunt tiny regions (§2.3)
            if key in self.state.inventory:
                goals.append(Goal(kind="acquire", item=key, door_edge=edge))
            else:
                goals.append(Goal(kind="search", item=key, door_edge=edge))
        return goals

    # ── scoring (§2.3) ───────────────────────────────────

    def _score_inf(self, goal: Goal):
        """λ → ∞: the nearest candidate wins (degenerates toward local greedy)."""
        if goal.kind == "nav":
            d = self._dist(self.state.pos, goal.target)
            return (-d, d, 0)
        if goal.kind == "search" and goal.door_edge is not None:
            key_cell = self._known_key_cell(goal.item)
            if key_cell is not None:
                d_search = self._dist(self.state.pos, key_cell)
                if not math.isinf(d_search):
                    d_search += self._dist(key_cell, self._door_near(goal.door_edge))
            else:
                d_search = float(
                    sum(
                        1
                        for c in self._reachable_set(self.state.pos)
                        if self.state.visits.get(c, 0) == 0
                    )
                )
            d = d_search + self._door_dist(goal.door_edge)
            return (-d, d, self._region_gain_unvisited(goal.door_edge))
        if goal.kind == "acquire" and goal.door_edge is not None:
            d = self._door_dist(goal.door_edge)
            return (-d, d, self._region_gain_unvisited(goal.door_edge))
        return (-INF, INF, 0)

    def _score_detail(self, goal: Goal, H_plan: int, lam: float):
        """Return (J, d, G) — score, travel cost, expected new cells."""
        if math.isinf(lam):
            return self._score_inf(goal)

        if goal.kind == "nav":
            d = self._dist(self.state.pos, goal.target)
            if math.isinf(d):
                return (-INF, INF, 0)
            budget = max(H_plan - d, 0)
            ball = self._reachable_set(goal.target, max_steps=budget)
            G = sum(1 for c in ball if self.state.visits.get(c, 0) == 0)
            return (G - lam * d, d, G)

        if goal.kind in ("search", "acquire") and goal.door_edge is not None:
            g_unlock = self._region_gain_unvisited(goal.door_edge)
            d_door = self._door_dist(goal.door_edge)
            if goal.kind == "search":
                key_cell = self._known_key_cell(goal.item)
                if key_cell is not None:
                    d_search = self._dist(self.state.pos, key_cell)
                    if not math.isinf(d_search):
                        d_search += self._dist(
                            key_cell, self._door_near(goal.door_edge)
                        )
                else:
                    # key location unknown: upper-bound search effort =
                    # unvisited reachable cells (heuristic, §2.3)
                    d_search = float(
                        sum(
                            1
                            for c in self._reachable_set(self.state.pos)
                            if self.state.visits.get(c, 0) == 0
                        )
                    )
                u = 0.0 if key_cell is not None else 1.0
                d = d_search + d_door
                J = g_unlock - lam * d - self.kappa * g_unlock * u
                return (J, d, g_unlock)
            # acquire: key in hand, just go unlock
            return (g_unlock - lam * d_door, d_door, g_unlock)

        return (-INF, INF, 0)

    def score(self, goal: Goal, H_plan: int, lam: float) -> float:
        """§2.3: J(goal) = G(goal) - lambda * d. Public wrapper."""
        return self._score_detail(goal, H_plan, lam)[0]

    # ── selection (§2.4) and re-evaluation (§4) ──────────

    def select(self, H_plan: int, lam: float) -> Optional[Goal]:
        """§2.4: argmax J over frontier ∪ key goals; ties -> smaller d."""
        candidates = self.propose_goals()
        if not candidates:
            self.current_goal = None
            self._last_G = None
            return None
        best, best_j, best_d, best_g = None, -INF, INF, 0
        for g in candidates:
            j, d, g_ = self._score_detail(g, H_plan, lam)
            if j > best_j or (j == best_j and d < best_d):
                best, best_j, best_d, best_g = g, j, d, g_
        self.current_goal = best
        self._last_G = best_g
        return best

    def re_evaluate(
        self, H_plan: int, lam: float, tau: float = 0.15
    ) -> Optional[Goal]:
        """§4: re-select; switch iff J(new) > J(cur) + max(tau*J(cur), 1.0).

        Returns the goal the planner commits to (possibly unchanged), or None
        when the maze is fully explored.
        """
        candidates = self.propose_goals()
        if not candidates:
            self.current_goal = None
            self._last_G = None
            return None

        best, best_j, best_d, best_g = None, -INF, INF, 0
        for g in candidates:
            j, d, g_ = self._score_detail(g, H_plan, lam)
            if j > best_j or (j == best_j and d < best_d):
                best, best_j, best_d, best_g = g, j, d, g_

        current = self.current_goal
        if current is None:
            self.current_goal, self._last_G = best, best_g
            return best

        cur_j = self._score_detail(current, H_plan, lam)[0]
        if best != current and best_j > cur_j + max(tau * cur_j, 1.0):
            self.current_goal, self._last_G = best, best_g
            return best

        # Keep the current goal; refresh its predicted gain for PE bookkeeping.
        if current in candidates:
            self._last_G = self._score_detail(current, H_plan, lam)[2]
        return current

    def refine(self, goal: Goal, H_plan: int) -> Goal:
        """Turn a nav goal into its executable next step.

        A frontier goal's target is the frontier cell f; the agent must
        physically step into f's unvisited neighbor to collect the info.
        When the agent is already at f, refine the goal to "nav to the
        unvisited neighbor" (the executor cannot see the visit map, so the
        planner decides the entry cell). Otherwise the goal is returned
        unchanged. Scoring / predicted gain are untouched (same frontier).
        """
        if goal.kind != "nav" or goal.target is None:
            return goal
        if self._dist(self.state.pos, goal.target) != 0:
            return goal
        unvisited = [
            nb
            for nb in self._neighbors_with(goal.target)
            if self.state.visits.get(nb, 0) == 0
        ]
        if not unvisited:
            return goal
        # Pick the entry cell with the richest info ball (same estimator).
        best, best_g = None, -INF
        for nb in unvisited:
            ball = self._reachable_set(nb, max_steps=H_plan - 1)
            g = sum(1 for c in ball if self.state.visits.get(c, 0) == 0)
            if g > best_g:
                best, best_g = nb, g
        return Goal(kind="nav", target=best)

    def is_frontier(self, cell: Cell) -> bool:
        """True while the cell still has an unvisited passable neighbor."""
        return any(
            self.state.visits.get(nb, 0) == 0
            for nb in self._neighbors_with(cell)
        )

    def escalate(self, H_plan: int, lam: float) -> Optional[Goal]:
        """Blockage handler (§4): a nav/acquire goal is unreachable because a
        locked door is in the way — commit to the best door goal instead."""
        door_goals = [
            g for g in self.propose_goals() if g.kind in ("search", "acquire")
        ]
        if not door_goals:
            return None
        best, best_j, best_d = None, -INF, INF
        for g in door_goals:
            j, d, _ = self._score_detail(g, H_plan, lam)
            if j > best_j or (j == best_j and d < best_d):
                best, best_j, best_d = g, j, d
        self.current_goal = best
        self._last_G = self._score_detail(best, H_plan, lam)[2]
        return best

    def predicted_gain(self) -> Optional[float]:
        """G of the currently committed goal (for goal-level PE bookkeeping)."""
        return self._last_G
