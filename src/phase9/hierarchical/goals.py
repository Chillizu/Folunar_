"""Phase 9 hierarchical horizon: goal representation (§3 interface)."""

from dataclasses import dataclass
from typing import Optional, Tuple

Cell = Tuple[int, int]


@dataclass(frozen=True)
class Goal:
    """A high-level commitment handed from the planner to the executor.

    kind:
      - "nav":     travel to `target` cell (frontier commitment)
      - "search":  find `item` (a key) that unlocks `door_edge`
      - "acquire": take `item` at `target`, or use `item` on `door_edge`
    """

    kind: str
    target: Optional[Cell] = None
    item: Optional[str] = None
    door_edge: Optional[Tuple[Cell, Cell]] = None
