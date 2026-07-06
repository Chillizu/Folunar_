"""Grid World environment and perception renderer for Phase 1."""

import random
from collections import deque
from typing import List, Optional, Tuple

from phase1.types import Action, GridState


class GridWorld:
    """5x5 deterministic grid world with wall, goal, and step-limit termination."""

    def __init__(
        self,
        width: int = 5,
        height: int = 5,
        obstacles: Optional[List[Tuple[int, int]]] = None,
        goal: Optional[Tuple[int, int]] = None,
        max_steps: int = 50,
    ):
        self.width = width
        self.height = height
        self.obstacles = list(obstacles) if obstacles is not None else []
        self.goal = goal
        self.max_steps = max_steps

    def reset(self, seed: Optional[int] = None) -> GridState:
        if seed is not None:
            random.seed(seed)
        cells = [(x, y) for x in range(self.width) for y in range(self.height)]
        free_cells = [c for c in cells if c not in self.obstacles]
        if len(free_cells) < 2:
            raise ValueError("Not enough free cells for agent and goal")

        goal = self.goal
        if goal is None:
            goal = random.choice(free_cells)
        else:
            goal = self.goal

        # Sample agent from the goal's connected component to guarantee a path.
        reachable = self._connected_component(goal)
        reachable_free = [c for c in reachable if c in free_cells and c != goal]
        if not reachable_free:
            raise ValueError("No reachable free cell for agent other than goal")
        agent = random.choice(reachable_free)

        return GridState(
            agent=agent,
            goal=goal,
            obstacles=self.obstacles,
            width=self.width,
            height=self.height,
            step=0,
            max_steps=self.max_steps,
        )

    def _connected_component(self, start: Tuple[int, int]) -> set:
        """Return all cells reachable from start without crossing obstacles or walls."""
        visited = {start}
        q = deque([start])
        while q:
            x, y = q.popleft()
            for nx, ny in [(x, y - 1), (x, y + 1), (x - 1, y), (x + 1, y)]:
                nxt = (nx, ny)
                if nxt in visited or nxt in self.obstacles:
                    continue
                if not (0 <= nx < self.width and 0 <= ny < self.height):
                    continue
                visited.add(nxt)
                q.append(nxt)
        return visited

    def step(self, state: GridState, action: Action) -> Tuple[GridState, float, bool]:
        x, y = state.agent
        dx, dy = 0, 0
        if action.name == "UP":
            dy = -1
        elif action.name == "DOWN":
            dy = 1
        elif action.name == "LEFT":
            dx = -1
        elif action.name == "RIGHT":
            dx = 1
        else:
            raise ValueError(f"Unknown action: {action.name}")

        nx, ny = x + dx, y + dy
        wall = False
        if not (0 <= nx < state.width and 0 <= ny < state.height):
            wall = True
        elif (nx, ny) in state.obstacles:
            wall = True

        if wall:
            next_pos = (x, y)
            reward = -0.2
        else:
            next_pos = (nx, ny)
            reward = -0.05

        done = next_pos == state.goal
        if done:
            reward = 1.0

        next_step = state.step + 1
        if next_step >= state.max_steps:
            done = True

        next_state = GridState(
            agent=next_pos,
            goal=state.goal,
            obstacles=state.obstacles,
            width=state.width,
            height=state.height,
            step=next_step,
            max_steps=state.max_steps,
        )
        return next_state, reward, done

    @staticmethod
    def all_actions() -> List[Action]:
        return [Action(name=n) for n in ("UP", "DOWN", "LEFT", "RIGHT")]


class Perception:
    """Text rendering of GridState for LLM prompts and debugging."""

    @staticmethod
    def render(state: GridState) -> str:
        obs_str = ",".join(f"({x},{y})" for x, y in state.obstacles) or "none"
        return (
            f"Agent at {state.agent}. "
            f"Goal at {state.goal}. "
            f"Obstacles at {obs_str}."
        )

    @staticmethod
    def ascii_render(state: GridState) -> str:
        lines = []
        for y in range(state.height):
            row = []
            for x in range(state.width):
                pos = (x, y)
                if pos == state.agent:
                    row.append("A")
                elif pos == state.goal:
                    row.append("G")
                elif pos in state.obstacles:
                    row.append("#")
                else:
                    row.append(".")
            lines.append("".join(row))
        return "\n".join(lines)

    @staticmethod
    def render_text(state) -> str:
        """Render a text-based state for LLM prompts."""
        inv = ", ".join(state.inventory) if state.inventory else "nothing"
        return (
            f"Location: {state.room.upper()}.\n"
            f"Description: {state.description}\n"
            f"Inventory: {inv}.\n"
            f"Goal: {state.goal}"
        )
