"""TextWorld-based environment for PEDA Phase 1.5.

Provides a TextWorldEnv class compatible with the existing Phase 1.5 pipeline:
  - reset(seed) -> TextWorldState
  - step(state, action) -> (TextWorldState, reward, done)

Supports 3 tiers of complexity:
  - Tier 1 (simple): 1 room, few objects
  - Tier 2 (medium): 5 rooms, multi-step quest
  - Tier 3 (constrained): 8 rooms, complex quest

Uses TextWorld 1.7.0 via the isolated .venv_textworld Python 3.10 venv.
"""

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_TIER_CONFIGS = {
    1: {"nb_rooms": 1, "nb_objects": 4, "quest_length": 1, "quest_breadth": 1},
    2: {"nb_rooms": 5, "nb_objects": 8, "quest_length": 3, "quest_breadth": 1},
    3: {"nb_rooms": 8, "nb_objects": 12, "quest_length": 4, "quest_breadth": 1},
}


@dataclass
class TextWorldState:
    """State representation for TextWorld games.

    Fields match the subset that Perception.render_text() expects:
      - room (uppercased)
      - description (full room desc)
      - inventory (list of item names)
      - goal (objective text)
    Plus TextWorld-specific extras carried for richer downstream use.
    """
    room: str = ""
    description: str = ""
    inventory: List[str] = field(default_factory=list)
    goal: str = ""
    step: int = 0
    max_steps: int = 100
    game_over: bool = False
    victory: bool = False
    score: float = 0.0
    admissible_commands: List[str] = field(default_factory=list)
    obs: str = ""  # full observation text

    def copy(self) -> "TextWorldState":
        return TextWorldState(
            room=self.room,
            description=self.description,
            inventory=list(self.inventory),
            goal=self.goal,
            step=self.step,
            max_steps=self.max_steps,
            game_over=self.game_over,
            victory=self.victory,
            score=self.score,
            admissible_commands=list(self.admissible_commands),
            obs=self.obs,
        )


def parse_inventory(inv_text: str) -> List[str]:
    """Parse TextWorld inventory text into a list of item names.

    Examples:
      "You are carrying: a butterfly and a fly larva."
        -> ["a butterfly", "a fly larva"]
      "You are carrying nothing." -> []
    """
    if not inv_text or "nothing" in inv_text.lower():
        return []
    # Remove prefix "You are carrying: " and trailing "."
    m = re.search(r":\s*(.+)\.?$", inv_text)
    if not m:
        return []
    items_str = m.group(1)
    # Split on " and " or ", and " or ", "
    items = re.split(r",?\s+and\s+|\s*,\s*", items_str)
    return [item.strip() for item in items if item.strip()]


def extract_room_name(description: str) -> str:
    """Extract room name from TextWorld description text.

    Format: "-= RoomName =-" at the start.
    Falls back to "unknown".
    """
    m = re.search(r"-=\s*(.+?)\s*=-", description)
    if m:
        return m.group(1).strip()
    return "unknown"


class TextWorldEnv:
    """TextWorld environment wrapping TextworldGymEnv for PEDA Phase 1.5.

    Usage:
        env = TextWorldEnv(tier=1)
        state = env.reset(seed=42)
        next_state, reward, done = env.step(state, "go north")
        env.close()
    """

    def __init__(self, tier: int = 1, game_dir: str = "./tw_games"):
        if tier not in (1, 2, 3):
            raise ValueError(f"tier must be 1, 2, or 3; got {tier}")
        self.tier = tier
        self._game_dir = Path(game_dir)
        self._game_dir.mkdir(parents=True, exist_ok=True)
        self._env = None
        self._current_game_file: Optional[str] = None
        self._current_seed: Optional[int] = None
        self._step_count = 0
        self._max_steps = 100

    def _import_textworld(self):
        """Lazy-import TextWorld modules (only available in .venv_textworld)."""
        import textworld as _tw
        from textworld import EnvInfos as _EnvInfos  # noqa: F811
        from textworld.gym.envs import TextworldGymEnv as _TextworldGymEnv  # noqa: F811
        return _tw, _EnvInfos, _TextworldGymEnv

    def _generate_game(self, seed: int) -> str:
        """Generate a TextWorld game file for the configured tier and seed.

        Returns the path to the .z8 game file.
        """
        tw, _EnvInfos, _TextworldGymEnv = self._import_textworld()

        go = tw.GameOptions()
        cfg = _TIER_CONFIGS[self.tier]
        go.nb_rooms = cfg["nb_rooms"]
        go.nb_objects = cfg["nb_objects"]
        go.quest_length = cfg["quest_length"]
        go.quest_breadth = cfg["quest_breadth"]
        go.seeds = seed
        go.path = str(self._game_dir) + "/"
        game_file, game = tw.make(go)
        return game_file

    def _build_state(self, obs: str, info: dict) -> TextWorldState:
        """Build a TextWorldState from TextworldGymEnv output."""
        desc = info.get("description", obs)
        inv_text = info.get("inventory", "")
        objective = info.get("objective", "")
        admissible = info.get("admissible_commands", [])
        won = info.get("won", False)
        max_score = info.get("max_score", 0)

        room = extract_room_name(desc)
        inventory = parse_inventory(inv_text)

        return TextWorldState(
            room=room,
            description=desc,
            inventory=inventory,
            goal=objective,
            step=self._step_count,
            max_steps=self._max_steps,
            game_over=False,
            victory=bool(won),
            score=float(max_score),
            admissible_commands=admissible,
            obs=obs,
        )

    def reset(self, seed: int = 0) -> TextWorldState:
        """Reset the environment with a given seed.

        The seed controls TextWorld game generation. The same seed + tier
        always produces the same game layout and quest.
        """
        tw, EnvInfos, TextworldGymEnv = self._import_textworld()

        # Close previous env
        self.close()

        self._current_seed = seed
        self._step_count = 0

        # Generate game file
        game_file = self._generate_game(seed)
        self._current_game_file = game_file

        # Create env
        ei = EnvInfos(
            description=True,
            inventory=True,
            admissible_commands=True,
            max_score=True,
            objective=True,
            won=True,
        )
        self._env = TextworldGymEnv([game_file], ei)

        # Reset
        obs, info = self._env.reset()
        state = self._build_state(obs, info)
        state.step = 0
        return state

    def step(self, state: TextWorldState, action_name: str) -> Tuple[TextWorldState, float, bool]:
        """Take an action in the TextWorld environment.

        Args:
            state: Current state (used for step tracking, env uses internal state).
            action_name: Text action string (e.g. "go north", "take key").

        Returns:
            (next_state, reward, done) tuple.
        """
        if self._env is None:
            raise RuntimeError("Environment not reset. Call reset() first.")

        self._step_count += 1
        obs, reward, done, info = self._env.step(action_name)

        next_state = self._build_state(obs, info)
        next_state.step = self._step_count
        next_state.game_over = done or self._step_count >= self._max_steps

        return next_state, float(reward), next_state.game_over

    def close(self):
        """Close the environment."""
        if self._env is not None:
            try:
                self._env.close()
            except Exception:
                pass
            self._env = None
            self._current_game_file = None

    def get_game_objective(self) -> str:
        """Return the current game's objective text."""
        if self._env is None:
            return ""
        # We can't easily get objective without resetting, so return cached
        return ""

    @staticmethod
    def tier_name(tier: int) -> str:
        names = {1: "simple", 2: "medium", 3: "constrained"}
        return names.get(tier, f"tier{tier}")

    @staticmethod
    def all_tiers() -> List[int]:
        return [1, 2, 3]


def render_state_text(state: TextWorldState) -> str:
    """Render a TextWorldState to text for LLM prompts (compatible with Perception.render_text)."""
    inv = ", ".join(state.inventory) if state.inventory else "nothing"
    return (
        f"Location: {state.room.upper()}.\n"
        f"Description: {state.description}\n"
        f"Inventory: {inv}.\n"
        f"Goal: {state.goal}"
    )
