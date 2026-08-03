"""Phase 7: Random Description Grid Maze Environment.

Extends GridMazeEnv with stochastic room descriptions.
Each visit to a room potentially generates DIFFERENT text,
simulating TextWorld-style stochasticity without the dependency.

Key design for epistemic exploration experiments:
  - State hash (count-based novelty) excludes description,
    so count-based novelty expires after one visit per room.
  - JEPA state_to_text includes description (via last_output),
    so ensemble sees different text each visit -> epistemic
    uncertainty drives revisitation even in "explored" rooms.

Hypothesis: In genuinely stochastic text environments, JEPA epistemic
uncertainty outperforms count-based novelty because re-visiting a room
uncovers new information (the unpredictable description).
"""

import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import hashlib

from phase6.grid_env import GridMazeEnv, GridState


# ── Stochastic Description Templates ─────────────────────

DESCRIPTION_TEMPLATES = {
    "Chamber": [
        "Torchlight flickers on damp walls.",
        "Moonlight streams through a narrow window.",
        "Cobwebs hang from the ceiling.",
        "A cold draft seeps through cracks in the floor.",
        "Dust motes dance in a beam of pale light.",
    ],
    "Gallery": [
        "Faded portraits hang crookedly.",
        "Empty frames stare from the walls.",
        "A marble bust lies shattered on the floor.",
        "Tapestries depict scenes of a forgotten war.",
    ],
    "Library": [
        "Dusty shelves sag under the weight of ancient tomes.",
        "Books lie scattered across the floor.",
        "A lectern holds an open book with faded script.",
        "Scrolls are piled haphazardly in a corner.",
    ],
    "Armory": [
        "Rusted weapon racks line the walls.",
        "Shields bear insignias of a fallen house.",
        "An empty suit of armor stands in the corner.",
        "Broken swords litter the floor.",
    ],
    "Cell": [
        "Cold iron bars cast long shadows.",
        "Straw covers the damp stone floor.",
        "Chains hang from the walls.",
        "A small, barred window lets in thin light.",
    ],
    "Garden": [
        "Overgrown thorny vines twist through cracks.",
        "Withered flowers struggle for life.",
        "A cracked fountain trickles murky water.",
        "Moss-covered statues loom among the weeds.",
    ],
    "Chapel": [
        "A broken altar sits beneath a shattered window.",
        "Pewers lie splintered and overturned.",
        "Candles gutter in rusted holders.",
        "A faded fresco adorns the ceiling.",
    ],
    "Store": [
        "Empty shelves hint at past commerce.",
        "A broken counter displays cracked pottery.",
        "Crates are stacked in a corner.",
        "Dust-covered scales sit on a table.",
    ],
    "Study": [
        "A desk covered in yellowed parchment.",
        "An inkwell has spilled across a map.",
        "A globe with cracked oceans sits by the window.",
        "A telescope points at a painted ceiling mural.",
    ],
    "Kitchen": [
        "Soot-blackened pots hang from hooks.",
        "A hearth holds cold ashes.",
        "A wooden table is scarred from chopping.",
        "Spices are scattered across a countertop.",
    ],
    "Treasury": [
        "Gold glints among scattered debris.",
        "A few coins remain in an open chest.",
        "Gemstones are embedded in the walls.",
        "A shattered goblet lies on the floor.",
    ],
    "Throne Room": [
        "A crumbling throne looms in darkness.",
        "Tattered banners hang from the rafters.",
        "The floor bears the faded pattern of a grand mosaic.",
        "A chandelier hangs askew from the ceiling.",
    ],
    "Dungeon": [
        "Water drips from the ceiling.",
        "Rats scurry along the walls.",
        "An iron door stands ajar.",
        "Torches sputter on the damp walls.",
    ],
    "Observatory": [
        "A cracked telescope points skyward.",
        "Star charts are scattered across the floor.",
        "An orrery with missing planets stands in the center.",
        "A skylight shows a cloudy night sky.",
    ],
}

# Fallback for templates not found
FALLBACK_TEMPLATES = [
    "A nondescript room with stone walls.",
    "Shadows pool in the corners of the room.",
    "The air is still and cold.",
    "Echoes of distant footsteps reach your ears.",
    "Fading light reveals little of interest.",
]


def _get_description_for_room(room_name: str, rng: random.Random) -> str:
    """Get a randomly chosen description for the given room name."""
    templates = DESCRIPTION_TEMPLATES.get(room_name, FALLBACK_TEMPLATES)
    return rng.choice(templates)


# ── RandomGridState (hash excludes description) ──────────

@dataclass
class RandomGridState(GridState):
    """GridState where hash_key excludes room_description.

    This is the key design choice:
      - Count-based novelty uses hash_key(), which excludes description.
        Result: novelty expires after first visit per room.
      - JEPA encoding uses state_to_text(), which includes description
        via the last_output adapter property.
        Result: different text each visit -> epistemic uncertainty.

    The only difference from GridState is hash_key() excludes description.
    """

    def hash_key(self) -> str:
        """Stable hash: description is stochastic, exclude it."""
        data = (
            f"{self.x},{self.y}|"
            f"inv={tuple(sorted(self.inventory))}"
        )
        return hashlib.md5(data.encode()).hexdigest()[:12]


# ── RandomMazeEnv (stochastic room descriptions) ─────────

class RandomMazeEnv(GridMazeEnv):
    """Grid maze where room DESCRIPTIONS change each visit.

    Each call to _observe() picks a random description from the
    room's template pool. Same (x,y) room, different text.

    This simulates TextWorld stochasticity — identical underlying
    state produces different observations, testing whether JEPA
    epistemic uncertainty can detect and exploit this.
    """

    def __init__(
        self,
        maze,
        task: dict,
        start_x: int = 0,
        start_y: int = 0,
        seed: int = 42,
        text_variants: int = 4,
    ):
        super().__init__(maze, task, start_x, start_y)
        self._rng = random.Random(seed)
        self._text_variants = text_variants

    def step(self, action_str: str) -> Tuple[str, "RandomGridState", bool]:
        """Execute action with stochastic observation text."""
        self.steps += 1
        action = self._parse(action_str)

        # Execute action (same as parent)
        if action.verb == "go":
            self._handle_move(action)
        elif action.verb == "take":
            self._handle_take(action)
        elif action.verb == "use":
            self._handle_use(action)
        elif action.verb == "inventory":
            pass
        elif action.verb == "look":
            pass

        done = self._check_goal() or (self.steps >= self.max_steps)
        obs = self._observe()
        return obs, self._get_state(), done

    # ── Observation Generation (stochastic) ─────────────

    def _observe(self) -> str:
        """Generate observation with RANDOM description each visit."""
        room = self.maze.rooms.get((self.x, self.y))
        name = (
            room.get("name", f"Room [{self.x},{self.y}]")
            if room
            else f"Room [{self.x},{self.y}]"
        )

        # Pick a random description — changes each visit
        desc = _get_description_for_room(name, self._rng)

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
                    if hasattr(self, '_locked_doors') and door in self._locked_doors:
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

    # ── State Access (RandomGridState) ──────────────────

    def _get_state(self) -> RandomGridState:
        """Return RandomGridState (hash excludes stochastic description)."""
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
        name = room.get("name", "") if room else ""

        # Get the CURRENT description (matches what _observe returned)
        desc = _get_description_for_room(name, self._rng) if name else ""

        return RandomGridState(
            x=self.x,
            y=self.y,
            inventory=tuple(sorted(self.inventory)),
            room_name=name,
            room_description=desc,
            exits=exit_map,
            visible_items=items,
            goal_reached=self._check_goal(),
        )

    def set_seed(self, seed: int) -> None:
        """Reset random seed for reproducibility."""
        self._rng = random.Random(seed)
