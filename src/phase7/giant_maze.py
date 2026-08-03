"""Phase 7: Giant Maze Environment (100x100 scale).

Performance-optimized grid maze for JEPA scaling experiments.
Sparse observations, minimal state hashing, DFS-generated spanning tree.

State space: ~100M+ (100x100 positions × inventory combinations).
Goal: does JEPA show ANY signal at extreme scale where counts saturate?
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
import random
import hashlib


@dataclass
class GiantGridState:
    """Minimal state for giant maze.

    No room names, descriptions, or verbose fields.
    Hash is (x, y, inventory) only — fast, deterministic, no full-description baggage.
    """

    x: int
    y: int
    inventory: Tuple[str, ...] = ()
    exits: Tuple[str, ...] = ()   # directions available: N, S, E, W
    visible_items: Tuple[str, ...] = ()
    goal_reached: bool = False

    @property
    def cwd(self) -> str:
        """Adapter: return room coordinate as 'working directory'."""
        return f"Room ({self.x},{self.y})"

    @property
    def files(self):
        """Adapter: return visible items as 'files'."""
        return list(self.visible_items)

    @property
    def last_output(self) -> str:
        """Adapter: return sparse description."""
        return f"Exits: {','.join(self.exits)}"

    def state_hash(self) -> str:
        """Stable hash from (x, y, inventory) only — NOT full description.

        This is crucial for count-based novelty: two visits to the same
        room with the same items are the *same state* regardless of
        which path the agent took.
        """
        data = f"({self.x},{self.y}) inv={sorted(self.inventory)}"
        return hashlib.md5(data.encode()).hexdigest()[:12]

    def hash_key(self) -> str:
        """Alias for compatibility with legacy explorers."""
        return self.state_hash()


@dataclass
class GiantGridAction:
    """Parsed action."""
    raw: str = ""
    verb: str = ""
    direction: Optional[str] = None
    target: Optional[str] = None
    tool: Optional[str] = None


def giant_state_to_text(state: GiantGridState) -> str:
    """Convert GiantGridState to flat text for JEPA encoder.

    Extremely sparse — just coordinate and exits.
    """
    inv_str = ",".join(sorted(state.inventory)) if state.inventory else ""
    exit_str = ",".join(state.exits) if state.exits else ""
    return f"room:({state.x},{state.y}) exits:{exit_str} items:{inv_str}"


class GiantMaze:
    """Procedurally generated giant maze.

    Simplified from GridMaze: no room templates, no locked doors, no items.
    Just walls and connectivity. Items are added sparsely.

    100x100 DFS maze: ~10K cells, ~20K passages, ~10K walls.
    """

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.walls: Set[Tuple[int, int, int, int]] = set()
        self.room_items: Dict[Tuple[int, int], List[str]] = {}

    def is_passable(self, x1: int, y1: int, x2: int, y2: int) -> bool:
        """Check if movement between adjacent cells is allowed."""
        if abs(x1 - x2) + abs(y1 - y2) != 1:
            return False
        return (x1, y1, x2, y2) not in self.walls and (x2, y2, x1, y1) not in self.walls

    def remove_item(self, x: int, y: int, item: str) -> bool:
        """Remove an item from a room."""
        items = self.room_items.get((x, y), [])
        if item in items:
            items.remove(item)
            return True
        for i in list(items):
            if item.lower() in i.lower() or i.lower() in item.lower():
                items.remove(i)
                return True
        return False

    @classmethod
    def generate(cls, width: int, height: int, seed: int = 42) -> "GiantMaze":
        """Generate a fully connected maze using DFS recursive backtracker.

        O(width × height) — 10K cells at 100x100, runs in < 50ms.
        """
        rng = random.Random(seed)

        # All walls initially present
        walls: Set[Tuple[int, int, int, int]] = set()
        for x in range(width):
            for y in range(height):
                if x < width - 1:
                    walls.add((x, y, x + 1, y))
                if y < height - 1:
                    walls.add((x, y, x, y + 1))

        # DFS recursive backtracker
        visited: Set[Tuple[int, int]] = set()
        stack = [(0, 0)]
        visited.add((0, 0))

        while stack:
            cx, cy = stack[-1]
            neighbors = []
            for nx, ny in [(cx - 1, cy), (cx + 1, cy), (cx, cy - 1), (cx, cy + 1)]:
                if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in visited:
                    neighbors.append((nx, ny))

            if not neighbors:
                stack.pop()
            else:
                nx, ny = rng.choice(neighbors)
                walls.discard((cx, cy, nx, ny))
                walls.discard((nx, ny, cx, cy))
                visited.add((nx, ny))
                stack.append((nx, ny))

        maze = cls(width=width, height=height)
        maze.walls = walls

        # Place a few items for navigational interest (sparse: ~1 per 20 cells)
        items = ["key", "scroll", "gem", "potion", "torch", "compass", "map"]
        all_positions = [(x, y) for x in range(width) for y in range(height)]
        rng.shuffle(all_positions)
        item_count = max(width * height // 20, 5)
        positions_for_items = rng.sample(
            all_positions, min(item_count, len(all_positions))
        )
        for i, (x, y) in enumerate(positions_for_items):
            item = items[i % len(items)]
            maze.room_items.setdefault((x, y), []).append(item)

        return maze

    def state_estimate(self) -> int:
        """Estimate number of unique states."""
        base = self.width * self.height  # positions
        items = sum(1 for v in self.room_items.values() if v)
        return base * (items + 1)


class GiantGridMazeEnv:
    """Performance-optimized grid maze environment for 100x100 scale.

    Key optimizations:
      - Sparse observations: "Room (12,34). Exits: N,S. Items: key."
      - State hash: (x,y,inventory) only, NOT full description
      - No room templates or locked doors overhead
      - max_steps: 2000 (100x100)
    """

    DIRECTION_MAP = {
        "north": (0, -1), "south": (0, 1),
        "east": (1, 0), "west": (-1, 0),
    }
    DIRECTION_ALIASES = {"n": "north", "s": "south", "e": "east", "w": "west"}
    DIRECTION_SHORT = {"north": "N", "south": "S", "east": "E", "west": "W"}

    def __init__(
        self,
        maze: GiantMaze,
        task: dict,
        start_x: int = 0,
        start_y: int = 0,
    ):
        self.maze = maze
        self.task = task
        self.x = start_x
        self.y = start_y
        self.inventory: List[str] = []
        self.steps = 0
        self.max_steps = task.get("max_steps", 2000)

    def reset(self) -> str:
        """Reset to start position."""
        self.x = self.task.get("start_x", 0)
        self.y = self.task.get("start_y", 0)
        self.inventory = []
        self.steps = 0
        return self._observe()

    def step(self, action_str: str) -> Tuple[str, "GiantGridState", bool]:
        """Execute an action.

        Returns (observation_text, state, done).
        """
        self.steps += 1
        action = self._parse(action_str)

        if action.verb == "go":
            self._handle_move(action)
        elif action.verb == "take":
            self._handle_take(action)
        elif action.verb == "look":
            pass
        elif action.verb == "inventory":
            pass

        done = self._check_goal() or (self.steps >= self.max_steps)
        obs = self._observe()
        return obs, self._get_state(), done

    # ── Action Parsing ──────────────────────────────────

    def _parse(self, action_str: str) -> GiantGridAction:
        s = action_str.lower().strip()
        if s.startswith("go "):
            direction_raw = s[3:].strip()
            direction = self.DIRECTION_ALIASES.get(direction_raw, direction_raw)
            if direction in self.DIRECTION_MAP:
                return GiantGridAction(action_str, "go", direction=direction)
            return GiantGridAction(action_str, "go")
        elif s.startswith("take "):
            return GiantGridAction(action_str, "take", target=s[5:].strip())
        elif s in ("look", "l"):
            return GiantGridAction(action_str, "look")
        elif s in ("inventory", "i"):
            return GiantGridAction(action_str, "inventory")
        return GiantGridAction(action_str, "unknown")

    # ── Movement ────────────────────────────────────────

    def _handle_move(self, action: GiantGridAction) -> None:
        move = self.DIRECTION_MAP.get(action.direction or "")
        if move is None:
            return
        dx, dy = move
        nx, ny = self.x + dx, self.y + dy
        if not (0 <= nx < self.maze.width and 0 <= ny < self.maze.height):
            return
        if not self.maze.is_passable(self.x, self.y, nx, ny):
            return
        self.x, self.y = nx, ny

    # ── Item Interaction ────────────────────────────────

    def _handle_take(self, action: GiantGridAction) -> None:
        if not action.target:
            return
        items_here = self.maze.room_items.get((self.x, self.y), [])
        for item in items_here:
            if action.target in item.lower():
                if self.maze.remove_item(self.x, self.y, item):
                    self.inventory.append(item)
                return

    # ── Goal Check ──────────────────────────────────────

    def _check_goal(self) -> bool:
        goal = self.task.get("goal_room")
        if goal and (self.x, self.y) == tuple(goal):
            return True
        return False

    # ── Sparse Observation ──────────────────────────────

    def _observe(self) -> str:
        """Generate an extremely sparse observation.

        "Room (12,34). Exits: N,S. Items: key."
        No long descriptions, no names, just coordinates.
        """
        # Build exits list — single-letter codes
        exit_codes: List[str] = []
        for d, (dx, dy) in [
            ("north", (0, -1)), ("south", (0, 1)),
            ("east", (1, 0)), ("west", (-1, 0)),
        ]:
            nx, ny = self.x + dx, self.y + dy
            if 0 <= nx < self.maze.width and 0 <= ny < self.maze.height:
                if self.maze.is_passable(self.x, self.y, nx, ny):
                    exit_codes.append(self.DIRECTION_SHORT[d])

        exit_str = ",".join(exit_codes) if exit_codes else "none"
        items_here = self.maze.room_items.get((self.x, self.y), [])
        items_str = ",".join(items_here) if items_here else "none"

        obs = f"Room ({self.x},{self.y}). Exits: {exit_str}. Items: {items_str}."

        if self.inventory:
            obs += f" Inv: {','.join(self.inventory)}."

        goal = self.task.get("goal_room")
        if goal:
            obs += f" Goal: reach ({goal[0]},{goal[1]})."

        if self._check_goal():
            obs += " *** GOAL REACHED ***"

        return obs

    # ── State Access ────────────────────────────────────

    def _get_state(self) -> GiantGridState:
        """Return minimal GiantGridState."""
        exit_codes: List[str] = []
        for d, (dx, dy) in [
            ("north", (0, -1)), ("south", (0, 1)),
            ("east", (1, 0)), ("west", (-1, 0)),
        ]:
            nx, ny = self.x + dx, self.y + dy
            if 0 <= nx < self.maze.width and 0 <= ny < self.maze.height:
                if self.maze.is_passable(self.x, self.y, nx, ny):
                    exit_codes.append(self.DIRECTION_SHORT[d])

        items = list(self.maze.room_items.get((self.x, self.y), []))

        return GiantGridState(
            x=self.x,
            y=self.y,
            inventory=tuple(sorted(self.inventory)),
            exits=tuple(sorted(exit_codes)),
            visible_items=tuple(items),
            goal_reached=self._check_goal(),
        )

    # ── Action Candidate Generation ─────────────────────

    @staticmethod
    def get_static_action_space() -> List[str]:
        return ["go north", "go south", "go east", "go west",
                "look", "inventory"]

    def get_dynamic_candidates(self) -> List[str]:
        """Return context-sensitive action candidates."""
        candidates = list(self.get_static_action_space())
        for item in self.maze.room_items.get((self.x, self.y), []):
            candidates.append(f"take {item}")
        return candidates
