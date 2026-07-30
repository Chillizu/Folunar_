"""Phase 6: Procedural maze generator.

Generates grid-based mazes using recursive backtracker (DFS).
Assigns themed room templates and scatters items.
Supports locked-door puzzles.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple
import random


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

    @classmethod
    def generate(cls, width: int, height: int, seed: int = 42,
                 num_locked_doors: int = 2) -> "GridMaze":
        """Generate a fully connected maze using DFS recursive backtracker.

        Every cell is reachable from every other cell (spanning tree).
        Room templates are assigned deterministically from the shuffled order.
        Items are scattered across roughly 1/3 of the rooms.
        """
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
        locked_doors: List of (direction, room_position, required_key_name) tuples.
    """

    name: str
    goal_room: Tuple[int, int]
    start_x: int = 0
    start_y: int = 0
    max_steps: int = 20
    description: str = ""
    locked_doors: List[Tuple[str, Tuple[int, int], str]] = field(default_factory=list)
