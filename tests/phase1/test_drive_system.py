"""Tests for HomeostaticDriveSystem and ActionGenerator."""

import math

import pytest

from phase1.drive_system import (
    ActionGenerator,
    HomeostaticDriveSystem,
    _action_entropy,
    _flow_zone_function,
)
from phase1.types import Action, DriveTerms, DriveWeights, ErrorVector, GridState, PredictedState
from phase1.world_model import EnsembleErrorComputer, WorldModel

# Reusable test values.
UP_ACTION = Action("UP")
DOWN_ACTION = Action("DOWN")
LEFT_ACTION = Action("LEFT")
RIGHT_ACTION = Action("RIGHT")


# ---------- Fixtures ----------

@pytest.fixture
def stub_wm():
    return WorldModel(use_stub=True)


@pytest.fixture
def stub_computer(stub_wm):
    return EnsembleErrorComputer(stub_wm, num_checkpoints=2)


@pytest.fixture
def weights():
    return DriveWeights(curiosity=0.5, competence=0.5, boredom=0.3, novelty=0.4)


@pytest.fixture
def drive(weights):
    return HomeostaticDriveSystem(weights)


@pytest.fixture
def sample_trajectory():
    return [
        PredictedState(
            level1_exit_code=0,
            level1_confidence=0.9,
            level2_next_agent=(3, 2),
            level2_confidence=0.85,
            level3_output_summary="moved right",
            level3_confidence=0.8,
            epistemic_ratio=0.3,
        ),
        PredictedState(
            level1_exit_code=0,
            level1_confidence=0.8,
            level2_next_agent=(4, 2),
            level2_confidence=0.75,
            level3_output_summary="moved right again",
            level3_confidence=0.7,
            epistemic_ratio=0.4,
        ),
    ]


@pytest.fixture
def sample_error():
    return ErrorVector(
        total_error=0.5,
        level1_error=0.0,
        level2_error=0.3,
        level3_error=0.0,
        epistemic_error=0.2,
        aleatoric_error=0.1,
        ensemble_variance=0.2,
    )


# ---------- Helper functions ----------

class TestActionEntropy:
    def test_empty_history(self):
        assert _action_entropy([]) == 0.0

    def test_single_action(self):
        hist = [UP_ACTION, UP_ACTION, UP_ACTION]
        assert _action_entropy(hist) == 0.0

    def test_two_actions_even(self):
        hist = [UP_ACTION, DOWN_ACTION] * 5
        entropy = _action_entropy(hist)
        # -0.5*ln(0.5) - 0.5*ln(0.5) approx 0.693
        assert entropy == pytest.approx(0.693147, rel=1e-4)

    def test_window_truncation(self):
        hist = [Action(f"A{i}") for i in range(100)]
        entropy = _action_entropy(hist, window=50)
        assert entropy > 0.0


class TestFlowZoneFunction:
    def test_mid_range(self):
        assert _flow_zone_function(0.5) == pytest.approx(0.5)

    def test_below_min(self):
        assert _flow_zone_function(0.0) == pytest.approx(0.2)

    def test_above_max(self):
        assert _flow_zone_function(1.0) == pytest.approx(0.8)

    def test_at_min(self):
        assert _flow_zone_function(0.2) == pytest.approx(0.2)

    def test_at_max(self):
        assert _flow_zone_function(0.8) == pytest.approx(0.8)


# ---------- HomeostaticDriveSystem ----------

class TestHomeostaticDriveSystemInit:
    def test_weights_stored(self, drive, weights):
        assert drive.weights is weights

    def test_initial_terms_zero(self, drive):
        terms = drive.current_terms
        assert terms.curiosity == 0.0
        assert terms.competence == 0.0
        assert terms.boredom == 0.0
        assert terms.novelty == 0.0

    def test_histories_empty(self, drive):
        assert len(drive.action_history) == 0
        assert len(drive.error_history) == 0
        assert len(drive.success_history) == 0


class TestDriveUpdate:
    def test_update_returns_drive_terms(self, drive, sample_error):
        terms = drive.update(
            sample_error,
            last_action=UP_ACTION,
            has_external_input=False,
            action_history=[UP_ACTION, UP_ACTION],
        )
        assert isinstance(terms, DriveTerms)

    def test_curiosity_from_epistemic_error(self, drive, sample_error):
        terms = drive.update(
            sample_error,
            last_action=UP_ACTION,
            has_external_input=False,
            action_history=[UP_ACTION],
        )
        # curiosity_term = tanh(2.0 * 0.2) = tanh(0.4) approx 0.3799
        expected = math.tanh(2.0 * sample_error.epistemic_error)
        assert terms.curiosity == pytest.approx(expected, abs=1e-6)

    def test_update_histories(self, drive, sample_error):
        drive.update(
            sample_error,
            last_action=DOWN_ACTION,
            has_external_input=False,
            action_history=[UP_ACTION],
        )
        assert len(drive.action_history) >= 1
        assert len(drive.error_history) >= 1
        assert len(drive.success_history) >= 1
        assert drive.action_history[-1].name == "DOWN"
        assert drive.error_history[-1] == sample_error.total_error

    def test_success_history_true(self, drive):
        error = ErrorVector(
            total_error=0.1, level1_error=0.0, level2_error=0.1, level3_error=0.0,
            epistemic_error=0.0, aleatoric_error=0.0, ensemble_variance=0.0,
        )
        drive.update(error, UP_ACTION, False, [UP_ACTION])
        assert drive.success_history[-1] is True

    def test_success_history_false(self, drive):
        error = ErrorVector(
            total_error=0.5, level1_error=1.0, level2_error=0.3, level3_error=0.0,
            epistemic_error=0.2, aleatoric_error=0.1, ensemble_variance=0.2,
        )
        drive.update(error, UP_ACTION, False, [UP_ACTION])
        assert drive.success_history[-1] is False

    def test_update_stores_current_terms(self, drive, sample_error):
        terms = drive.update(sample_error, UP_ACTION, False, [UP_ACTION])
        assert drive.current_terms == terms

    def test_external_input_resets_counter(self, drive, sample_error):
        drive.update(sample_error, UP_ACTION, has_external_input=True, action_history=[UP_ACTION])
        assert drive.current_terms.novelty == pytest.approx(0.0)
        # After reset, steps_since_external_input = 0, and the counter only increments
        # when > 0, so it stays 0. Novelty remains 0.
        drive.update(sample_error, UP_ACTION, has_external_input=False, action_history=[UP_ACTION])
        assert drive.current_terms.novelty == pytest.approx(0.0)

    def test_novelty_stays_zero_without_external_input(self, drive, sample_error):
        """Without an external input reset, steps_since_external_input stays 0."""
        drive.update(sample_error, UP_ACTION, has_external_input=False, action_history=[UP_ACTION])
        n0 = drive.current_terms.novelty
        drive.update(sample_error, UP_ACTION, has_external_input=False, action_history=[UP_ACTION])
        n1 = drive.current_terms.novelty
        assert n0 == 0.0
        assert n1 == 0.0


class TestApplyToEFE:
    def test_returns_float(self, drive, sample_trajectory):
        result = drive.apply_to_efe(
            base_efe=1.0,
            trajectory=sample_trajectory,
            action_history=[UP_ACTION],
            candidate_action=RIGHT_ACTION,
        )
        assert isinstance(result, float)

    def test_reduced_by_drive_adjustment(self, drive, sample_trajectory):
        """With positive drive terms, the adjusted EFE should be less than base."""
        error = ErrorVector(
            total_error=0.5, level1_error=0.0, level2_error=0.3, level3_error=0.0,
            epistemic_error=0.5, aleatoric_error=0.0, ensemble_variance=0.5,
        )
        drive.update(error, UP_ACTION, False, [UP_ACTION, DOWN_ACTION])
        adjusted = drive.apply_to_efe(
            base_efe=2.0,
            trajectory=sample_trajectory,
            action_history=[UP_ACTION, DOWN_ACTION],
            candidate_action=RIGHT_ACTION,
        )
        assert adjusted < 2.0

    def test_with_zero_terms_equals_base(self, drive, sample_trajectory):
        """With zero terms and weights, adjustment is zero."""
        adjusted = drive.apply_to_efe(
            base_efe=1.5,
            trajectory=sample_trajectory,
            action_history=[UP_ACTION],
            candidate_action=RIGHT_ACTION,
        )
        assert adjusted == pytest.approx(1.5)

    def test_diversity_bonus_with_new_action(self, drive, sample_trajectory):
        """An action not in recent history gets a diversity bonus."""
        error = ErrorVector(
            total_error=0.5, level1_error=0.0, level2_error=0.3, level3_error=0.0,
            epistemic_error=0.3, aleatoric_error=0.2, ensemble_variance=0.3,
        )
        drive.update(error, LEFT_ACTION, False, [LEFT_ACTION])
        action_hist = [LEFT_ACTION] * 10

        with_new = drive.apply_to_efe(
            base_efe=1.0,
            trajectory=sample_trajectory,
            action_history=action_hist,
            candidate_action=RIGHT_ACTION,  # not in recent 10
        )
        with_same = drive.apply_to_efe(
            base_efe=1.0,
            trajectory=sample_trajectory,
            action_history=action_hist,
            candidate_action=LEFT_ACTION,  # in recent 10
        )
        # right action gets diversity bonus => larger drive adjustment => lower EFE
        assert with_new < with_same


# ---------- ActionGenerator ----------

class TestActionGenerator:
    def test_init_defaults(self, stub_wm, stub_computer, drive):
        gen = ActionGenerator(stub_wm, stub_computer, drive)
        assert gen.horizon == 2
        assert gen.max_candidates == 4
        assert gen.latency_budget_ms == 3000.0

    def test_compute_efe_returns_float(self, stub_wm, stub_computer, drive, sample_trajectory):
        gen = ActionGenerator(stub_wm, stub_computer, drive)
        state = GridState(agent=(2, 2), goal=(4, 4), width=5, height=5)
        result = gen.compute_efe(state, sample_trajectory, [UP_ACTION], candidate_action=UP_ACTION)
        assert isinstance(result, float)

    def test_compute_efe_with_stub_trajectory(self, stub_wm, stub_computer, drive):
        """Compute EFE from an actual stub rollout trajectory."""
        gen = ActionGenerator(stub_wm, stub_computer, drive)
        state = GridState(agent=(2, 2), goal=(4, 4), width=5, height=5)
        traj = stub_wm.rollout(state, RIGHT_ACTION, horizon=2)
        efe = gen.compute_efe(state, traj, [], candidate_action=RIGHT_ACTION)
        assert isinstance(efe, float)

    def test_select_action_returns_valid_action(self, stub_wm, stub_computer, drive):
        gen = ActionGenerator(stub_wm, stub_computer, drive)
        state = GridState(agent=(2, 2), goal=(4, 4), width=5, height=5)
        candidates = [Action(n) for n in ("UP", "DOWN", "LEFT", "RIGHT")]
        action = gen.select_action(state, [], candidates)
        assert isinstance(action, Action)
        assert action.name in ("UP", "DOWN", "LEFT", "RIGHT")

    def test_select_action_all_candidates_considered(self, stub_wm, stub_computer, drive):
        """Select action should return a candidate from the provided list."""
        gen = ActionGenerator(stub_wm, stub_computer, drive)
        state = GridState(agent=(2, 2), goal=(4, 4), width=5, height=5)
        candidates = [Action("UP"), Action("RIGHT")]
        action = gen.select_action(state, [], candidates)
        assert action.name in ("UP", "RIGHT")
