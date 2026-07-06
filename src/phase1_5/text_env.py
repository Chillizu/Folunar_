"""Minimal multi-room text environment for PEDA Phase 1.5.

Two rooms connected by a door:
  - Study (start): has a key on the desk
  - Hallway: has a locked chest

Goal: take key, go north, unlock chest.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class TextState:
    room: str
    description: str
    inventory: List[str] = field(default_factory=list)
    goal: str = ""
    step: int = 0
    max_steps: int = 50
    game_over: bool = False
    victory: bool = False

    def copy(self) -> "TextState":
        return TextState(
            room=self.room,
            description=self.description,
            inventory=list(self.inventory),
            goal=self.goal,
            step=self.step,
            max_steps=self.max_steps,
            game_over=self.game_over,
            victory=self.victory,
        )


ROOM_DESCRIPTIONS: Dict[str, str] = {
    "study": (
        "You are in a small study. Bookshelves line the walls. "
        "A wooden desk sits against the north wall. "
        "On the desk is a small rusty key. "
        "A door leads north."
    ),
    "hallway": (
        "You are in a narrow hallway. The walls are bare stone. "
        "A heavy iron chest sits in the corner. "
        "The chest has a locked padlock. "
        "A door leads south back to the study."
    ),
}

ROOM_EXITS: Dict[str, Dict[str, str]] = {
    "study": {"north": "hallway"},
    "hallway": {"south": "study"},
}

ROOM_OBJECTS: Dict[str, List[str]] = {
    "study": ["key"],
    "hallway": ["chest"],
}


class TextRoomEnv:
    """Two-room text adventure; exposes reset/step like GridWorld."""

    def __init__(self, goal: Optional[str] = None):
        self._goal = goal or "Unlock the chest in the hallway."
        self._key_taken = False
        self._chest_unlocked = False
        self._state: Optional[TextState] = None

    def _get_description(self, room: str) -> str:
        desc = ROOM_DESCRIPTIONS[room]
        if room == "study" and self._key_taken:
            desc = (
                "You are in a small study. Bookshelves line the walls. "
                "A wooden desk sits against the north wall. "
                "The desk is empty — you already took the key. "
                "A door leads north."
            )
        if room == "hallway" and self._chest_unlocked:
            desc = (
                "You are in a narrow hallway. The walls are bare stone. "
                "A heavy iron chest sits in the corner. "
                "The chest is open and empty — you unlocked it. "
                "A door leads south back to the study."
            )
        elif room == "hallway" and self._chest_unlocked is False and self._key_taken:
            desc = (
                "You are in a narrow hallway. The walls are bare stone. "
                "A heavy iron chest sits in the corner. "
                "The chest has a locked padlock — but you have the key! "
                "A door leads south back to the study."
            )
        return desc

    def reset(self, seed: int = 0) -> TextState:
        self._key_taken = False
        self._chest_unlocked = False
        self._state = TextState(
            room="study",
            description=self._get_description("study"),
            inventory=[],
            goal=self._goal,
            step=0,
            max_steps=50,
            game_over=False,
            victory=False,
        )
        return self._state

    def step(self, state: TextState, action_name: str) -> Tuple[TextState, float, bool]:
        s = state.copy()
        s.step += 1

        action = action_name.strip().lower()

        # --- global commands ---
        if action in ("look", "l"):
            s.description = self._get_description(s.room)
            return s, 0.0, s.game_over

        if action in ("inventory", "i"):
            if s.inventory:
                s.description = f"You are carrying: {', '.join(s.inventory)}."
            else:
                s.description = "You are not carrying anything."
            return s, 0.0, s.game_over

        # --- study ---
        if s.room == "study":
            if action == "take key":
                if not self._key_taken:
                    self._key_taken = True
                    s.inventory.append("key")
                    s.description = "You take the rusty key from the desk."
                else:
                    s.description = "The key is already in your inventory."
                return s, 0.0, s.game_over

            if action in ("go north", "north", "n"):
                s.room = "hallway"
                s.description = self._get_description("hallway")
                return s, 0.0, s.game_over

            # invalid action in study
            s.description = f"You can't '{action}' here."
            return s, -0.1, s.game_over

        # --- hallway ---
        if s.room == "hallway":
            if action in ("go south", "south", "s"):
                s.room = "study"
                s.description = self._get_description("study")
                return s, 0.0, s.game_over

            if action in ("unlock chest with key", "unlock chest", "use key on chest"):
                if "key" in s.inventory and not self._chest_unlocked:
                    self._chest_unlocked = True
                    s.inventory.remove("key")
                    s.description = (
                        "You insert the rusty key into the padlock. "
                        "It turns with a click! The chest swings open. "
                        "Inside is a small golden star."
                    )
                    s.victory = True
                    s.game_over = True
                    return s, 1.0, True
                elif self._chest_unlocked:
                    s.description = "The chest is already unlocked and empty."
                elif "key" not in s.inventory:
                    s.description = "You don't have the key."
                return s, 0.0, s.game_over

            # invalid action in hallway
            s.description = f"You can't '{action}' here."
            return s, -0.1, s.game_over

        # --- fallback (shouldn't reach) ---
        s.description = f"Unknown room '{s.room}'."
        s.game_over = True
        return s, -1.0, True

    @staticmethod
    def all_actions() -> List[str]:
        return [
            "look",
            "inventory",
            "take key",
            "go north",
            "go south",
            "unlock chest with key",
        ]
