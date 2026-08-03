"""Phase 6: Stochastic Grid Maze Environment.

Extends GridMazeEnv with stochastic item respawning.
Items randomly appear/disappear each step, testing whether
epistemic exploration (JEPA) outperforms count-based novelty
in stochastic environments where counting visits fails.

Key design:
- State hash excludes transient items (only x, y, inventory)
  so count-based novelty expires after one visit per room.
- JEPA state encoding includes items (via adapter properties)
  so the ensemble detects stochasticity and drives revisitation.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple
import hashlib
import random

from phase6.grid_env import GridMazeEnv, GridState


@dataclass
class StochasticGridState(GridState):
    """GridState with stable hash excluding transient items.

    hash_key/state_hash only use (x, y, inventory) so count-based
    novelty expires after one visit per room regardless of item changes.
    """

    def hash_key(self) -> str:
        """Stable hash: items are transient, use only (x,y,inventory)."""
        data = f"{self.x},{self.y}|{tuple(sorted(self.inventory))}"
        return hashlib.md5(data.encode()).hexdigest()[:12]


class StochasticMazeEnv(GridMazeEnv):
    """Grid maze where items randomly appear/disappear each step.

    The target item "emerald" spawns rarely in rooms where (x+y)%3==0.
    Task is to find the emerald (pick it up). Count-based explorers
    should fail because novelty expires after one visit per room,
    so they never return to check for newly spawned items.
    """

    COMMON_ITEMS = ["scroll", "potion", "dagger", "lantern", "rope", "gem"]

    def __init__(
        self,
        maze,
        task,
        start_x: int = 0,
        start_y: int = 0,
        respawn_p: float = 0.1,
        rare_p: float = 0.02,
    ):
        super().__init__(maze, task, start_x, start_y)
        self.respawn_p = respawn_p
        self.rare_p = rare_p
        self.max_steps = task.get("max_steps", 500)

    def _respawn_items(self) -> None:
        """Randomly add/remove items from rooms each step.

        Each room independently:
        - 30% chance: all current items disappear
        - respawn_p chance: a random common item spawns
        - rare_p chance (eligible rooms): emerald spawns
        """
        for (x, y) in list(self.maze.rooms.keys()):
            # Random removal (30% chance items vanish)
            items_at = self.maze.room_items.get((x, y), [])
            if items_at and random.random() < 0.3:
                self.maze.room_items[(x, y)] = []
                items_at = []

            # Random common item spawn
            if random.random() < self.respawn_p:
                items_at = list(self.maze.room_items.get((x, y), []))
                items_at.append(random.choice(self.COMMON_ITEMS))
                self.maze.room_items[(x, y)] = items_at

            # Rare emerald spawn in eligible rooms
            if (x + y) % 3 == 0 and random.random() < self.rare_p:
                items_at = list(self.maze.room_items.get((x, y), []))
                if "emerald" not in items_at:
                    items_at.append("emerald")
                    self.maze.room_items[(x, y)] = items_at

    def step(self, action_str: str) -> Tuple[str, "GridState", bool]:
        """Execute action after respawning items."""
        self._respawn_items()  # Items change BEFORE agent observes
        return super().step(action_str)

    def _check_goal(self) -> bool:
        """Goal reached when emerald is in inventory."""
        return "emerald" in self.inventory

    def _get_state(self) -> GridState:
        """Return StochasticGridState (hash excludes transient items)."""
        s = super()._get_state()
        return StochasticGridState(
            x=s.x,
            y=s.y,
            inventory=s.inventory,
            room_name=s.room_name,
            room_description=s.room_description,
            exits=s.exits,
            visible_items=s.visible_items,
            goal_reached=s.goal_reached,
        )
