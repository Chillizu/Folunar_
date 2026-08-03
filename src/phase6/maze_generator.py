"""Phase 6: Procedural maze generator.

Generates grid-based mazes using recursive backtracker (DFS).
Assigns themed room templates and scatters items.
Supports locked-door puzzles.
"""

import random
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Union

ROOM_TEMPLATES = [
    ("Chamber", "Torchlight flickers on damp walls."),
    ("Gallery", "Faded portraits hang crookedly."),
    ("Library", "Dusty shelves sag under the weight of ancient tomes."),
    ("Armory", "Rusted weapon racks line the walls."),
    ("Cell", "Cold iron bars cast long shadows."),
    ("Garden", "Overgrown thorny vines twist through cracks."),
    ("Chapel", "A broken altar sits beneath a shattered window."),
    ("Store", "Empty shelves hint at past commerce."),
    ("Study", "A desk covered in yellowed parchment."),
    ("Kitchen", "Soot-blackened pots hang from hooks."),
    ("Treasury", "Gold glints among scattered debris."),
    ("Throne Room", "A crumbling throne looms in darkness."),
    ("Dungeon", "Water drips from the ceiling."),
    ("Observatory", "A cracked telescope points skyward."),
]

ITEMS = [
    "rusty key", "silver key", "golden key", "iron key",
    "ancient scroll", "gemstone", "dagger", "shield",
    "potion", "lantern", "rope", "crowbar", "compass", "map fragment",
]


@dataclass
class GridMaze:
    """Procedurally generated grid maze.

    Rooms are placed at every cell with themed descriptions.
    Items are scattered throughout.
    Walls block movement between adjacent cells.
    """

    width: int
    height: int
    walls: Set[Tuple[int, int, int, int]] = field(default_factory=set)
    rooms: Dict[Tuple[int, int], dict] = field(default_factory=dict)
    room_items: Dict[Tuple[int, int], List[str]] = field(default_factory=dict)
    # Locked-door puzzle data (populated by generate() from MazeTask):
    #   locked_doors: carved edge (x1,y1,x2,y2) -> key name required to pass.
    #                 Door edges are passable geometry blocked until unlocked.
    #   key_locations: key name -> cell where the key was placed.
    locked_doors: Dict[Tuple[int, int, int, int], str] = field(default_factory=dict)
    key_locations: Dict[str, Tuple[int, int]] = field(default_factory=dict)

    def is_passable(self, x1: int, y1: int, x2: int, y2: int) -> bool:
        """Check if movement between adjacent cells is allowed (no wall)."""
        if abs(x1 - x2) + abs(y1 - y2) != 1:
            return False
        return (x1, y1, x2, y2) not in self.walls and (x2, y2, x1, y1) not in self.walls

    def remove_item(self, x: int, y: int, item: str) -> bool:
        """Remove an item from a room. Returns True if found."""
        items = self.room_items.get((x, y), [])
        if item in items:
            items.remove(item)
            return True
        # Fuzzy match for case/naming variations
        for i in list(items):
            if item.lower() in i.lower() or i.lower() in item.lower():
                items.remove(i)
                return True
        return False

    # ── Locked-door helpers ──────────────────────────────

    def door_orientations(self) -> List[Tuple[str, Tuple[int, int], str]]:
        """(direction, room, key_name) for every locked door, from both sides.

        GridMazeEnv keys doors by (direction, (x, y)) as seen from the room
        the agent stands in: a door on edge (x1,y1)-(x2,y2) is 'east' from
        (x1,y1) and 'west' from (x2,y2) for horizontal edges ('south'/'north'
        for vertical edges).
        """
        out: List[Tuple[str, Tuple[int, int], str]] = []
        for (x1, y1, x2, y2), key in self.locked_doors.items():
            if x2 == x1 + 1:
                out.append(("east", (x1, y1), key))
                out.append(("west", (x2, y2), key))
            else:
                out.append(("south", (x1, y1), key))
                out.append(("north", (x2, y2), key))
        return out

    @staticmethod
    def _normalize_edge(e: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        """Order an undirected edge as (min_x, min_y, max_x, max_y)."""
        x1, y1, x2, y2 = e
        if (x2, y2) < (x1, y1):
            x1, y1, x2, y2 = x2, y2, x1, y1
        return (x1, y1, x2, y2)

    @staticmethod
    def _reversed_edge(e: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        return (e[2], e[3], e[0], e[1])

    @staticmethod
    def _spec_to_edge(
        direction: str, pos: Tuple[int, int]
    ) -> Optional[Tuple[int, int, int, int]]:
        """Map a (direction, room) door spec to its edge (not normalized)."""
        delta = {"north": (0, -1), "south": (0, 1), "east": (1, 0), "west": (-1, 0)}
        if direction not in delta:
            return None
        dx, dy = delta[direction]
        x, y = pos
        return (x, y, x + dx, y + dy)

    def _carved_edges(self) -> List[Tuple[int, int, int, int]]:
        """All passable adjacency edges (normalized)."""
        out: List[Tuple[int, int, int, int]] = []
        for x in range(self.width):
            for y in range(self.height):
                if x < self.width - 1 and (x, y, x + 1, y) not in self.walls:
                    out.append((x, y, x + 1, y))
                if y < self.height - 1 and (x, y, x, y + 1) not in self.walls:
                    out.append((x, y, x, y + 1))
        return out

    def _bfs_distances(self, start: Tuple[int, int]) -> Dict[Tuple[int, int], int]:
        """BFS distances from start over the carved graph."""
        dist = {start: 0}
        queue = deque([start])
        while queue:
            cx, cy = queue.popleft()
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = cx + dx, cy + dy
                if not (0 <= nx < self.width and 0 <= ny < self.height):
                    continue
                if (nx, ny) in dist:
                    continue
                if self.is_passable(cx, cy, nx, ny):
                    dist[(nx, ny)] = dist[(cx, cy)] + 1
                    queue.append((nx, ny))
        return dist

    def _component(
        self, start: Tuple[int, int], blocked_edges: Set[Tuple[int, int, int, int]]
    ) -> Set[Tuple[int, int]]:
        """Cells reachable from start without crossing blocked edges."""
        comp = {start}
        queue = deque([start])
        while queue:
            cx, cy = queue.popleft()
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = cx + dx, cy + dy
                if not (0 <= nx < self.width and 0 <= ny < self.height):
                    continue
                if (nx, ny) in comp:
                    continue
                if not self.is_passable(cx, cy, nx, ny):
                    continue
                if self._normalize_edge((cx, cy, nx, ny)) in blocked_edges:
                    continue
                comp.add((nx, ny))
                queue.append((nx, ny))
        return comp

    def _construct_locked_doors(
        self,
        rng: random.Random,
        door_count: int,
        door_specs: List[Tuple[str, Tuple[int, int], str]],
    ) -> None:
        """Place locked doors and their keys (called by generate()).

        Door edges are carved passages that GridMazeEnv blocks until the
        matching key is used on them. Every key is placed in a cell reachable
        from (0, 0) without crossing any door, so each puzzle is solvable
        independently from the start.
        """
        if door_count <= 0:
            return

        # ── 1. Pick door edges (explicit specs first, then random) ──────────
        edges: List[Tuple[Tuple[int, int, int, int], str]] = []
        used: Set[Tuple[int, int, int, int]] = set()

        for direction, (x, y), key in door_specs:
            raw = self._spec_to_edge(direction, (x, y))
            if raw is None:
                continue
            edge = self._normalize_edge(raw)
            # A spec may reference an edge the DFS never carved — carve it so
            # the door is a real passage.
            self.walls.discard(edge)
            self.walls.discard(self._reversed_edge(edge))
            edges.append((edge, key))
            used.add(edge)

        key_names = [n for n in ITEMS if "key" in n]
        if len(edges) < door_count:
            # Random far-from-start carved edges for the remaining doors.
            carved = [e for e in self._carved_edges() if e not in used]
            dist = self._bfs_distances((0, 0))
            min_gap = max(2, min(self.width, self.height) // 4)
            candidates = [
                e for e in carved
                if min(dist[(e[0], e[1])], dist[(e[2], e[3])]) >= min_gap
            ]
            if not candidates:
                candidates = carved
            for _ in range(door_count - len(edges)):
                if not candidates:
                    break
                edge = rng.choice(candidates)
                candidates.remove(edge)
                key = key_names[len(edges) % len(key_names)]
                edges.append((edge, key))

        if not edges:
            return

        # ── 2. Store doors (normalized edges) ───────────────────────────────
        for edge, key in edges:
            self.locked_doors[edge] = key

        # ── 3. Place keys in the start component (all doors removed) ────────
        blocked = {e for e in self.locked_doors}
        reachable = self._component((0, 0), blocked)
        door_cells: Set[Tuple[int, int]] = set()
        for (x1, y1, x2, y2) in self.locked_doors:
            door_cells.add((x1, y1))
            door_cells.add((x2, y2))
        for edge, key in edges:
            cell_pool = [c for c in reachable if c not in door_cells]
            if not cell_pool:
                cell_pool = list(reachable)
            cell = rng.choice(cell_pool)
            self.key_locations[key] = cell
            self.room_items.setdefault(cell, []).append(key)

    @classmethod
    def generate(
        cls,
        width: int,
        height: int,
        task: Optional["MazeTask"] = None,
        seed: int = 42,
        num_locked_doors: int = 0,
    ) -> "GridMaze":
        """Generate a fully connected maze using DFS recursive backtracker.

        Every cell is reachable from every other cell (spanning tree).
        Room templates are assigned deterministically from the shuffled order.
        Items are scattered across roughly 1/3 of the rooms.

        Locked doors: when `task` is given, `task.locked_doors` (an int door
        count or a list of (direction, room, key_name) specs) drives door
        construction and `task.size` (when > 0) overrides width/height.
        Without a task, `num_locked_doors` random doors are built (default 0 —
        plain mazes, matching the Phase-6 baseline convention).
        """
        if task is not None:
            if task.size:
                width, height = task.size, task.size
            if isinstance(task.locked_doors, int):
                door_count = task.locked_doors
                door_specs: List[Tuple[str, Tuple[int, int], str]] = []
            else:
                door_count = len(task.locked_doors)
                door_specs = list(task.locked_doors)
        else:
            door_count = num_locked_doors
            door_specs = []

        rng = random.Random(seed)

        # All walls initially present between every adjacent cell pair
        walls = set()
        for x in range(width):
            for y in range(height):
                if x < width - 1:
                    walls.add((x, y, x + 1, y))
                if y < height - 1:
                    walls.add((x, y, x, y + 1))

        # DFS recursive backtracker to carve passages (spanning tree)
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
                # Remove wall between (cx,cy) and (nx,ny)
                wall = (cx, cy, nx, ny)
                walls.discard(wall)
                walls.discard((nx, ny, cx, cy))
                visited.add((nx, ny))
                stack.append((nx, ny))

        maze = cls(width=width, height=height, walls=walls)

        # Assign room templates (deterministic shuffle, then round-robin)
        all_positions = [(x, y) for x in range(width) for y in range(height)]
        rng.shuffle(all_positions)
        for i, (x, y) in enumerate(all_positions):
            template = ROOM_TEMPLATES[i % len(ROOM_TEMPLATES)]
            maze.rooms[(x, y)] = {
                "name": template[0],
                "description": template[1],
            }

        # Place items
        item_count = max(width * height // 3, 5)
        positions_for_items = rng.sample(
            all_positions, min(item_count, len(all_positions))
        )
        for i, (x, y) in enumerate(positions_for_items):
            item = ITEMS[i % len(ITEMS)]
            maze.room_items.setdefault((x, y), []).append(item)

        # Locked doors + keys (from task or num_locked_doors)
        maze._construct_locked_doors(rng, door_count, door_specs)

        return maze

    def state_estimate(self) -> int:
        """Estimate number of unique states (rough lower bound)."""
        base = self.width * self.height  # positions
        items = sum(1 for v in self.room_items.values() if v)
        return base * (items + 1)


@dataclass
class MazeTask:
    """Specification for a maze navigation task.

    Attributes:
        name: Human-readable task identifier.
        goal_room: (x, y) coordinates of the goal room.
        start_x, start_y: Starting position.
        max_steps: Step limit per episode.
        description: Natural-language task description.
        size: Square maze side; GridMaze.generate() uses it when > 0.
        locked_doors: Either an int door count or a list of
            (direction, room_position, required_key_name) tuples. The int
            form makes generate() construct that many random locked doors,
            with each key placed in a room reachable without crossing any door.
    """

    name: str = ""
    goal_room: Tuple[int, int] = (0, 0)
    start_x: int = 0
    start_y: int = 0
    max_steps: int = 20
    description: str = ""
    size: int = 0
    locked_doors: Union[int, List[Tuple[str, Tuple[int, int], str]]] = field(
        default_factory=list
    )
