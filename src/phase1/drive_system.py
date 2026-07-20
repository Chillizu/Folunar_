"""Drive system and action generator for Phase 1."""

import json
import math
import random
from collections import deque
from pathlib import Path
from typing import List, Optional

from phase1.types import Action, DriveTerms, DriveWeights, GridState, PredictedState
from phase1.world_model import EnsembleErrorComputer, WorldModel


def _action_name(action) -> str:
    """Extract action name from Action object or string."""
    return action if isinstance(action, str) else (action.name if action else "unknown")


def _action_entropy(action_history, window: int = 50) -> float:
    """Shannon entropy (nats) over the recent action distribution."""
    recent = action_history[-window:] if len(action_history) > window else action_history
    if not recent:
        return 0.0
    counts = {}
    for a in recent:
        counts[_action_name(a)] = counts.get(_action_name(a), 0) + 1
    total = len(recent)
    entropy = 0.0
    for count in counts.values():
        p = count / total
        entropy -= p * math.log(p)
    return entropy


def _flow_zone_function(success_rate: float) -> float:
    """Map success rate in [0.2, 0.8] to competence term in [0.2, 0.8]."""
    t = (success_rate - 0.2) / 0.6
    t = max(0.0, min(1.0, t))
    return 0.2 + 0.6 * t


class HomeostaticDriveSystem:
    """Dynamic drive-term modulation with fixed grid-search weights."""

    def __init__(self, initial_weights: DriveWeights):
        self.weights = initial_weights
        self.action_history: deque[Action] = deque(maxlen=50)
        self.error_history: deque[float] = deque(maxlen=100)
        self.success_history: deque[bool] = deque(maxlen=20)
        self.steps_since_external_input = 0
        self.current_terms = DriveTerms()

    def _success_rate(self, window: int = 20) -> float:
        recent = list(self.success_history)[-window:]
        if not recent:
            return 0.5
        return sum(recent) / len(recent)

    def update(
        self,
        error,
        last_action: Action,
        has_external_input: bool,
        action_history: List[Action],
    ) -> DriveTerms:
        curiosity_term = math.tanh(2.0 * error.epistemic_error)
        competence_term = _flow_zone_function(self._success_rate(window=20))
        boredom_term = max(0.0, 0.7 - _action_entropy(action_history, window=50))
        if has_external_input:
            self.steps_since_external_input = 0
        elif self.steps_since_external_input > 0:
            self.steps_since_external_input += 1
        novelty_term = 1.0 - math.exp(-0.01 * self.steps_since_external_input)

        terms = DriveTerms(
            curiosity=curiosity_term,
            competence=competence_term,
            boredom=boredom_term,
            novelty=novelty_term,
        )
        self.current_terms = terms

        self.action_history.append(last_action)
        self.error_history.append(error.total_error)
        self.success_history.append(error.level1_error == 0)
        return terms

    def apply_to_efe(
        self,
        base_efe: float,
        trajectory: List[PredictedState],
        action_history: List[Action],
        candidate_action: Optional[Action] = None,
    ) -> float:
        info_gain = sum(1.0 - p.level2_confidence for p in trajectory)
        challenge_level = sum(1.0 - p.level1_confidence for p in trajectory) / max(1, len(trajectory))

        diversity_bonus = 0.0
        if candidate_action is not None:
            recent = action_history[-10:] if len(action_history) > 10 else action_history
            if not any(_action_name(a) == _action_name(candidate_action) for a in recent):
                diversity_bonus = 0.2

        external_info_potential = 0.0

        drive_adjustment = (
            self.weights.curiosity * self.current_terms.curiosity * info_gain
            + self.weights.competence * self.current_terms.competence * challenge_level
            + self.weights.boredom * self.current_terms.boredom * diversity_bonus
            + self.weights.novelty * self.current_terms.novelty * external_info_potential
        )
        return base_efe - drive_adjustment


class ActionGenerator:
    """EFE-based action selection with latency-aware rollout adaptation."""

    LATENCY_CONFIG = Path("config/phase1_model.json")

    def __init__(
        self,
        world_model: WorldModel,
        error_computer: EnsembleErrorComputer,
        drive_system: HomeostaticDriveSystem,
        horizon: int = 2,
        max_candidates: int = 4,
        latency_budget_ms: float = 3000.0,
        pragmatic_only: bool = False,
        pragmatic_weight: float = 3.0,
        goal_predicate=None,
    ):
        self.world_model = world_model
        self.error_computer = error_computer
        self.drive_system = drive_system
        self.horizon = horizon
        self.max_candidates = max_candidates
        self.latency_budget_ms = latency_budget_ms
        self.pragmatic_only = pragmatic_only
        self.pragmatic_weight = pragmatic_weight
        self.goal_predicate = goal_predicate

    def _load_latency_ms(self) -> float:
        if self.LATENCY_CONFIG.exists():
            try:
                data = json.loads(self.LATENCY_CONFIG.read_text())
                return float(data.get("median_ms", 1000.0))
            except (json.JSONDecodeError, ValueError, KeyError):
                pass
        return 1000.0

    def compute_efe(
        self,
        state,
        trajectory: List[PredictedState],
        action_history: List[Action],
        candidate_action: Optional[Action] = None,
    ) -> float:
        # TextState path: pragmatic = distance to victory (0 if predicted win, else 0.5)
        if hasattr(state, "room") or hasattr(state, "container_id"):
            pragmatic = 0.0
            if trajectory:
                if self.goal_predicate is not None:
                    # Task-specific goal check using predicted next state
                    pred = trajectory[-1]
                    fake_ns = type("obj", (object,), {
                        "last_output": pred.level3_output_summary or "",
                        "last_exit_code": pred.level1_exit_code,
                        "files": [],
                        "cwd": "",
                    })()
                    try:
                        pred_json = json.loads(pred.level2_text or "{}")
                        fake_ns.last_output = str(pred_json.get("last_output", pred.level3_output_summary or ""))
                        fake_ns.last_exit_code = int(pred_json.get("last_exit_code", pred.level1_exit_code))
                        fake_ns.files = list(pred_json.get("files", []))
                        fake_ns.cwd = str(pred_json.get("cwd", ""))
                    except Exception:
                        pass
                    goal_met = self.goal_predicate(state, candidate_action, fake_ns)
                    pragmatic = 0.0 if goal_met else 0.5
                else:
                    final_exit = trajectory[-1].level1_exit_code
                    pragmatic = 0.0 if final_exit == 2 else 0.5
            if self.pragmatic_only:
                return pragmatic * self.pragmatic_weight
            epistemic = 0.0
            for i, p in enumerate(trajectory):
                ratio = p.epistemic_ratio if p.epistemic_ratio is not None else 0.5
                epistemic += (1.0 - p.level2_confidence) * ratio * (0.9 ** i)
            base_efe = epistemic + pragmatic * self.pragmatic_weight
            # ConfidencePenalty: penalize actions with avg confidence > 0.95 to prevent dead loops
            if trajectory and not self.pragmatic_only:
                avg_conf = sum(p.level1_confidence for p in trajectory) / len(trajectory)
                if avg_conf > 0.95:
                    base_efe += 0.3 * (avg_conf - 0.95)
            return self.drive_system.apply_to_efe(
                base_efe, trajectory, action_history, candidate_action=candidate_action
            )

        # GridState path (original logic)
        pragmatic = 0.0
        if trajectory and state.goal is not None:
            final = trajectory[-1].level2_next_agent
            if final is not None:
                dist = abs(final[0] - state.goal[0]) + abs(final[1] - state.goal[1])
                max_dist = max(1, (state.width - 1) + (state.height - 1))
                pragmatic = dist / max_dist
        if self.pragmatic_only:
            return pragmatic * self.pragmatic_weight
        epistemic = 0.0
        for i, p in enumerate(trajectory):
            ratio = p.epistemic_ratio if p.epistemic_ratio is not None else 0.5
            epistemic += (1.0 - p.level2_confidence) * ratio * (0.9 ** i)
        base_efe = epistemic + pragmatic * self.pragmatic_weight
        # ConfidencePenalty: penalize actions with avg confidence > 0.95 to prevent dead loops
        if trajectory and not self.pragmatic_only:
            avg_conf = sum(p.level1_confidence for p in trajectory) / len(trajectory)
            if avg_conf > 0.95:
                base_efe += 0.3 * (avg_conf - 0.95)
        return self.drive_system.apply_to_efe(
            base_efe, trajectory, action_history, candidate_action=candidate_action
        )

    def select_action(
        self,
        state,
        action_history: List[Action],
        candidates: List[Action],
    ) -> Action:
        latency_ms = self._load_latency_ms()
        candidates = candidates[: self.max_candidates]
        budget = latency_ms * len(candidates) * self.horizon
        horizon = self.horizon if budget <= self.latency_budget_ms else 1

        best_action = None
        best_efe = float("inf")
        for action in candidates:
            trajectory = self.world_model.rollout(state, action, horizon=horizon)
            efe = self.compute_efe(state, trajectory, action_history, candidate_action=action)
            if efe < best_efe:
                best_efe = efe
                best_action = action

        if best_action is None:
            best_action = random.choice(candidates)
        return best_action
