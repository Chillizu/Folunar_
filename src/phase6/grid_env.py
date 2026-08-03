"""Phase 6: Procedural Text-Grid Maze Environment.

A text-adventure-style environment built on procedurally generated grid mazes.
Observations are natural-language descriptions. Actions are verb-oriented strings.
States are GridState dataclasses with structured features.

Designed for JEPA epistemic exploration experiments:
  - State space scales with maze size (small: ~25 states, xl: ~900)
  - Text observations compatible with LLM/Qwen encoders
  - Configurable tasks (reach goal room, collect items)
"""

import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class GridState:
    """A single state in the grid maze.

    Used for novelty tracking and JEPA encoding.
    hash_key() / state_hash() produce stable identifiers for count-based exploration.
    """

    x: int
    y: int
    inventory: Tuple[str, ...] = ()
    room_name: str = ""
    room_description: str = ""
    exits: Dict[str, bool] = field(default_factory=dict)
    visible_items: List[str] = list
    goal_reached: bool = False

    # Adapter fields for JEPAEnsemble._state_to_text compatibility
    @property
    def cwd(self) -> str:
        """Adapter: return room name as 'working directory'."""
        return self.room_name or f"Room [{self.x},{self.y}]"

    @property
    def files(self):
        """Adapter: return visible items as 'files'."""
        return self.visible_items or []

    @property
    def last_output(self) -> str:
        """Adapter: return room description as 'last output'."""
        return self.room_description or ""

    def hash_key(self) -> str:
        """Stable hash for novelty tracking (count-based exploration)."""
        data = (
            f"{self.x},{self.y},{sorted(self.inventory)},{self.goal_reached}"
        )
        return hashlib.md5(data.encode()).hexdigest()[:12]

    def state_hash(self) -> str:
        """Alias for hash_key() -- compatibility with NoveltyExplorer."""
        return self.hash_key()


@dataclass
class GridAction:
    """Parsed action from natural-language input."""

    raw: str
    verb: str  # go, take, use, look, inventory
    direction: Optional[str] = None  # north/south/east/west
    target: Optional[str] = None  # item or door name
    tool: Optional[str] = None  # item used


def grid_state_to_text(state: GridState) -> str:
    """Convert GridState to flat text for JEPA encoder (embedding).

    Format mirrors the sandbox state_to_text convention.
    """
    exit_str = ",".join(
        k for k in ["north", "south", "east", "west"] if state.exits.get(k)
    )
    inv_str = ",".join(state.inventory) if state.inventory else "none"
    return (
        f"x: {state.x} | y: {state.y} | "
        f"inventory: {inv_str} | "
        f"room: {state.room_name} | "
        f"exits: {exit_str}"
    )


def grid_goal_text(task: dict) -> str:
    """Convert task to text summary (for JEPA encoding)."""
    goal = task.get("goal_room", (0, 0))
    return f"goal: reach room ({goal[0]},{goal[1]})"


class GridMazeEnv:
    """Procedural text-grid maze environment.

    Observations are natural-language text strings (text adventure style).
    Actions are verb-oriented strings: "go north", "take rusty key", etc.
    States are GridState dataclasses with structured feature fields.

    Usage:
        maze = GridMaze.generate(10, 10)
        task = {"goal_room": (9, 9), "start_x": 0, "start_y": 0, "max_steps": 20}
        env = GridMazeEnv(maze, task)
        obs = env.reset()
        obs2, state, done = env.step("go north")
    """

    # Static action space (always available)
    ACTION_SPACE: List[str] = [
        "go north",
        "go south",
        "go east",
        "go west",
        "look",
        "inventory",
    ]

    def __init__(
        self,
        maze: "GridMaze",  # noqa: F821
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
        self.max_steps = task.get("max_steps", 20)

        # Internal state for doors and keys
        self._locked_doors: Set[Tuple[str, Tuple[int, int]]] = set()
        self._key_for_door: Dict[Tuple[str, Tuple[int, int]], str] = {}

    def setup(self) -> None:
        """Configure doors and items after maze generation.

        Consumes `MazeTask.locked_doors` (an int door count or a list of
        (direction, room, key_name) specs). The int form leaves placement to
        the maze: GridMaze.generate() built the doors from the task, so the
        actual door orientations are read from `self.maze`. Call once after
        the maze is generated, before the first reset().
        """
        self._locked_doors.clear()
        self._key_for_door.clear()

        doors = getattr(self.task, "locked_doors", None)
        if isinstance(self.task, dict):
            doors = self.task.get("locked_doors", doors)
        if isinstance(doors, int):
            doors = None  # count only — placements live on the maze
        if not doors:
            doors = self.maze.door_orientations()

        for direction, pos, key_name in doors:
            pos = tuple(pos)
            door_key = (direction, pos)
            self._locked_doors.add(door_key)
            self._key_for_door[door_key] = key_name

    def reset(self) -> str:
        """Reset environment to start position. Returns initial observation."""
        self.x = self.task.get("start_x", 0)
        self.y = self.task.get("start_y", 0)
        self.inventory = []
        self.steps = 0
        return self._observe()

    def step(self, action_str: str) -> Tuple[str, "GridState", bool]:
        """Execute an action.

        Returns:
            observation_text: Natural-language description of new state.
            state: GridState dataclass.
            done: True if goal reached or step limit exceeded.
        """
        self.steps += 1
        action = self._parse(action_str)

        if action.verb == "go":
            self._handle_move(action)
        elif action.verb == "take":
            self._handle_take(action)
        elif action.verb == "use":
            self._handle_use(action)
        elif action.verb == "inventory":
            pass  # handled by observation
        elif action.verb == "look":
            pass  # handled by observation

        done = self._check_goal() or (self.steps >= self.max_steps)
        obs = self._observe()
        return obs, self._get_state(), done

    # ── Action Parsing ──────────────────────────────────

    DIRECTION_MAP = {
        "north": (0, -1),
        "south": (0, 1),
        "east": (1, 0),
        "west": (-1, 0),
    }

    DIRECTION_ALIASES = {
        "n": "north",
        "s": "south",
        "e": "east",
        "w": "west",
    }

    def _parse(self, action_str: str) -> GridAction:
        """Parse a natural-language action string."""
        s = action_str.lower().strip()
        if s.startswith("go "):
            direction_raw = s[3:].strip()
            direction = self.DIRECTION_ALIASES.get(direction_raw, direction_raw)
            if direction in self.DIRECTION_MAP:
                return GridAction(action_str, "go", direction=direction)
            return GridAction(action_str, "go")
        elif s.startswith("take "):
            return GridAction(action_str, "take", target=s[5:].strip())
        elif s.startswith("use "):
            parts = s[4:].split(" on ")
            tool = parts[0].strip() if parts else ""
            target = parts[1].strip() if len(parts) > 1 else ""
            return GridAction(action_str, "use", target=target, tool=tool)
        elif s in ("look", "l"):
            return GridAction(action_str, "look")
        elif s in ("inventory", "i"):
            return GridAction(action_str, "inventory")
        return GridAction(action_str, "unknown")

    # ── Movement ────────────────────────────────────────

    def _handle_move(self, action: GridAction) -> None:
        """Attempt to move in the specified direction."""
        move = self.DIRECTION_MAP.get(action.direction or "")
        if move is None:
            return
        dx, dy = move
        nx, ny = self.x + dx, self.y + dy

        # Bounds check
        if not (0 <= nx < self.maze.width and 0 <= ny < self.maze.height):
            return

        # Maze passage check
        if not self.maze.is_passable(self.x, self.y, nx, ny):
            return

        # Locked door check
        dir_key = action.direction or ""
        door = (dir_key, (self.x, self.y))
        if door in self._locked_doors:
            needed_key = self._key_for_door.get(door, "")
            if needed_key and needed_key not in self.inventory:
                return  # still locked

        self.x, self.y = nx, ny

    # ── Item Interaction ────────────────────────────────

    def _handle_take(self, action: GridAction) -> None:
        """Pick up an item from the current room."""
        if not action.target:
            return
        items_here = self.maze.room_items.get((self.x, self.y), [])
        for item in items_here:
            if action.target in item.lower():
                if self.maze.remove_item(self.x, self.y, item):
                    self.inventory.append(item)
                return

    def _handle_use(self, action: GridAction) -> None:
        """Use an item (e.g., key on locked door)."""
        if not action.tool:
            return
        # Find the matching inventory item (case-insensitive)
        matched_tool = None
        for inv_item in self.inventory:
            if action.tool in inv_item.lower():
                matched_tool = inv_item
                break
        if matched_tool is None:
            return

        # Check each direction for a locked door in this room
        for d in ["north", "south", "east", "west"]:
            door = (d, (self.x, self.y))
            if door in self._locked_doors:
                needed_key = self._key_for_door.get(door, "")
                if needed_key and needed_key.lower() in matched_tool.lower():
                    self._locked_doors.discard(door)
                    # A door spans two cells — unlock both orientations so the
                    # passage opens in both directions.
                    dx, dy = self.DIRECTION_MAP[d]
                    nx, ny = self.x + dx, self.y + dy
                    if 0 <= nx < self.maze.width and 0 <= ny < self.maze.height:
                        opposite = {
                            "north": "south", "south": "north",
                            "east": "west", "west": "east",
                        }
                        self._locked_doors.discard((opposite[d], (nx, ny)))
                    return

    # ── Goal Check ──────────────────────────────────────

    def _check_goal(self) -> bool:
        """Check if the agent has reached the goal room."""
        goal = self.task.get("goal_room")
        if goal and (self.x, self.y) == tuple(goal):
            return True
        return False

    # ── Observation Generation ──────────────────────────

    def _observe(self) -> str:
        """Generate a natural-language text observation of the current room."""
        room = self.maze.rooms.get((self.x, self.y))
        name = (
            room.get("name", f"Room [{self.x},{self.y}]")
            if room
            else f"Room [{self.x},{self.y}]"
        )
        desc = (
            room.get("description", "A nondescript room.")
            if room
            else "Empty void."
        )

        # Build exits list
        exits_list: List[str] = []
        for d, (dx, dy) in [
            ("north", (0, -1)),
            ("south", (0, 1)),
            ("east", (1, 0)),
            ("west", (-1, 0)),
        ]:
            nx, ny = self.x + dx, self.y + dy
            if 0 <= nx < self.maze.width and 0 <= ny < self.maze.height:
                if self.maze.is_passable(self.x, self.y, nx, ny):
                    door = (d, (self.x, self.y))
                    if door in self._locked_doors:
                        exits_list.append(f"[{d[0].upper()}]{d[1:]} (locked)")
                    else:
                        exits_list.append(f"[{d[0].upper()}]{d[1:]}")

        items_here = self.maze.room_items.get((self.x, self.y), [])
        items_str = ", ".join(items_here) if items_here else "nothing of interest"

        obs = f"{name}\n{desc}\n"
        obs += f"Exits: {', '.join(exits_list) if exits_list else 'none'}\n"
        obs += f"You see: {items_str}.\n"
        if self.inventory:
            obs += f"Inventory: {', '.join(self.inventory)}.\n"

        goal = self.task.get("goal_room")
        if goal:
            goal_room = self.maze.rooms.get(tuple(goal), {})
            goal_name = goal_room.get("name", f"Room {goal}")
            obs += f"Your goal: reach {goal_name}.\n"

        if self._check_goal():
            obs += "*** GOAL REACHED! ***\n"

        return obs.strip()

    # ── State Access ────────────────────────────────────

    def _get_state(self) -> GridState:
        """Return the current GridState dataclass."""
        room = self.maze.rooms.get((self.x, self.y))
        exit_map: Dict[str, bool] = {}
        for d, (dx, dy) in [
            ("north", (0, -1)),
            ("south", (0, 1)),
            ("east", (1, 0)),
            ("west", (-1, 0)),
        ]:
            nx, ny = self.x + dx, self.y + dy
            if 0 <= nx < self.maze.width and 0 <= ny < self.maze.height:
                exit_map[d] = self.maze.is_passable(self.x, self.y, nx, ny)
            else:
                exit_map[d] = False

        items = list(self.maze.room_items.get((self.x, self.y), []))

        return GridState(
            x=self.x,
            y=self.y,
            inventory=tuple(sorted(self.inventory)),
            room_name=room.get("name", "") if room else "",
            room_description=room.get("description", "") if room else "",
            exits=exit_map,
            visible_items=items,
            goal_reached=self._check_goal(),
        )

    # ── Action Candidate Generation ─────────────────────

    @staticmethod
    def get_static_action_space() -> List[str]:
        """Return the static action space (for EFE / candidate generation)."""
        return list(GridMazeEnv.ACTION_SPACE)

    def get_dynamic_candidates(self) -> List[str]:
        """Return context-sensitive action candidates.

        Includes static actions plus take- and use- actions
        for items currently visible or in inventory.
        """
        candidates = list(self.ACTION_SPACE)
        # Add take- actions for items in the current room
        for item in self.maze.room_items.get((self.x, self.y), []):
            candidates.append(f"take {item}")
        # Add use- actions for inventory items on doors
        for item in self.inventory:
            candidates.append(f"use {item} on door")
        return candidates
