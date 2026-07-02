"""Core dataclasses for Phase 1 PEDA architecture."""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class GridState:
    agent: Tuple[int, int]
    goal: Tuple[int, int]
    obstacles: List[Tuple[int, int]] = field(default_factory=list)
    width: int = 5
    height: int = 5
    step: int = 0
    max_steps: int = 50


@dataclass
class Action:
    name: str  # "UP", "DOWN", "LEFT", "RIGHT"


@dataclass
class PredictedState:
    level1_exit_code: int  # 0=success, 1=wall, 2=goal
    level1_confidence: float
    level2_next_agent: Tuple[int, int]  # predicted next position
    level2_confidence: float
    level3_output_summary: str  # e.g., "agent moved up"
    level3_confidence: float
    # fraction of total uncertainty that is epistemic (ensemble variance / total uncertainty)
    epistemic_ratio: float = 0.5


@dataclass
class ErrorVector:
    total_error: float
    level1_error: float
    level2_error: float
    level3_error: float
    epistemic_error: float
    aleatoric_error: float
    ensemble_variance: float


@dataclass
class DriveWeights:
    """Fixed search weights used by the grid search and as the base for dynamic updates.

    Grid-search values may be outside [0,1] (e.g., [0.1, 0.5, 1.0, 2.0]);
    defaults are [0,1].
    """

    curiosity: float = 0.5
    competence: float = 0.5
    boredom: float = 0.3
    novelty: float = 0.4


@dataclass
class DriveTerms:
    """Dynamic drive terms computed each step from error, success history, and action entropy.

    These are multiplied by the fixed DriveWeights in apply_to_efe().
    """

    curiosity: float = 0.0
    competence: float = 0.0
    boredom: float = 0.0
    novelty: float = 0.0


@dataclass
class Experience:
    state: GridState
    action: Action
    next_state: GridState
    error: ErrorVector
