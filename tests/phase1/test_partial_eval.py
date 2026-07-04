"""Tests for the partial-evaluation helper functions in phase1_partial_eval.py."""

import importlib.util
from pathlib import Path

import pytest

from phase1.grid_env import GridWorld
from phase1.world_model import WorldModel

_PARTIAL_EVAL_PATH = Path(__file__).resolve().parent.parent.parent / "scripts" / "phase1_partial_eval.py"
spec = importlib.util.spec_from_file_location("phase1_partial_eval", _PARTIAL_EVAL_PATH)
pe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pe)


@pytest.fixture
def stub_wm():
    return WorldModel(use_stub=True)


class TestIsKnownCell:
    def test_known_and_unknown(self):
        known = {(0, 0), (1, 1)}
        assert pe._is_known_cell((0, 0), known) is True
        assert pe._is_known_cell((2, 2), known) is False


class TestGoalForCell:
    def test_default_goal(self):
        assert pe._goal_for_cell((0, 0)) == (4, 4)
        assert pe._goal_for_cell((4, 4)) == (0, 0)
        assert pe._goal_for_cell((2, 2)) == (4, 4)


class TestSampleGoal:
    def test_goal_known(self):
        rng = __import__("random").Random(42)
        known = [(0, 0), (1, 1), (2, 2)]
        all_cells = [(x, y) for x in range(5) for y in range(5)]
        goal = pe._sample_goal("goal_known", rng, known, all_cells)
        assert goal in known

    def test_goal_unknown(self):
        rng = __import__("random").Random(42)
        known = [(0, 0), (1, 1), (2, 2)]
        all_cells = [(x, y) for x in range(5) for y in range(5)]
        goal = pe._sample_goal("goal_unknown", rng, known, all_cells)
        assert goal not in known
        assert goal in all_cells


class TestSampleUntrainedStart:
    def test_returns_untrained_agent(self):
        known = {(0, 0), (1, 1), (2, 2)}
        env = GridWorld(goal=(4, 4), max_steps=50)
        state, seed = pe._sample_untrained_start(env, 100, known)
        assert state.agent not in known
        assert isinstance(seed, int)

    def test_raises_when_all_cells_known(self):
        all_cells = {(x, y) for x in range(5) for y in range(5)}
        env = GridWorld(goal=(4, 4), max_steps=50)
        with pytest.raises(RuntimeError):
            pe._sample_untrained_start(env, 100, all_cells)


class TestComputeG1TestSet:
    def test_all_known_returns_zero(self, stub_wm):
        """When every state-action pair is marked as trained, the test set is empty."""
        all_cells = [[x, y] for x in range(5) for y in range(5)]
        trained_pairs = [
            {"agent": [x, y], "goal": [4, 4], "obstacles": [], "action": action}
            for x in range(5)
            for y in range(5)
            for action in ("UP", "DOWN", "LEFT", "RIGHT")
        ]
        manifest = {
            "all_cells": all_cells,
            "known_cells": all_cells,
            "trained_pairs": trained_pairs,
        }
        assert pe._compute_g1_test_set(stub_wm, manifest, max_steps=50) == 0.0

    def test_partial_known_runs_and_stub_is_perfect(self, stub_wm):
        all_cells = [[x, y] for x in range(5) for y in range(5)]
        known_cells = [[0, 0], [0, 1]]
        trained_pairs = [
            {"agent": [0, 0], "goal": [4, 4], "obstacles": [], "action": "UP"},
            {"agent": [0, 1], "goal": [4, 4], "obstacles": [], "action": "UP"},
        ]
        manifest = {
            "all_cells": all_cells,
            "known_cells": known_cells,
            "trained_pairs": trained_pairs,
        }
        g1 = pe._compute_g1_test_set(stub_wm, manifest, max_steps=50)
        assert g1 == 1.0


class TestExplorationMetrics:
    def test_metrics(self):
        known = {(0, 0)}
        episodes = [
            {"trajectory": [(0, 1), (0, 2), (1, 2), (0, 0)]},
            {"trajectory": [(0, 1), (0, 0)]},
        ]
        metrics = pe._exploration_metrics(episodes, known, max_steps=50)
        assert 0.0 < metrics["mean_unknown_fraction"] < 1.0
        assert metrics["mean_unknown_cells_visited"] > 0.0
        assert metrics["mean_steps_before_known"] < 50

    def test_no_known_cells(self):
        known = set()
        episodes = [{"trajectory": [(0, 1), (0, 2)]}]
        metrics = pe._exploration_metrics(episodes, known, max_steps=50)
        assert metrics["mean_unknown_fraction"] == 1.0
        assert metrics["mean_steps_before_known"] == 50


class TestAggregate:
    def test_aggregate_empty(self):
        agg = pe._aggregate([], max_steps=50, known_cells=set())
        assert agg["success_rate"] == 0.0
        assert agg["mean_steps"] == 50.0

    def test_aggregate_with_data(self):
        episodes = [
            {"steps": 4, "success": True, "revisit_rate": 0.0, "g1": 1.0, "trajectory": [(0, 0)]},
            {"steps": 6, "success": False, "revisit_rate": 0.1, "g1": 0.8, "trajectory": [(0, 0)]},
        ]
        agg = pe._aggregate(episodes, max_steps=50, known_cells=set())
        assert agg["success_rate"] == 0.5
        assert agg["mean_steps"] == 5.0
        assert agg["revisit_rate"] == pytest.approx(0.05)
        assert agg["g1"] == pytest.approx(0.9)
