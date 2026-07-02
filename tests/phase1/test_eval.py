"""Tests for metrics functions: accuracy, completion rate, revisit rate, ratio, baseline."""

import pytest

from phase1.drive_system import ActionGenerator, HomeostaticDriveSystem
from phase1.grid_env import GridWorld
from phase1.run import (
    aggregate_metrics,
    completion_rate_at_horizon,
    next_state_accuracy,
    random_baseline,
    revisit_rate,
    run_episode,
    steps_to_goal_ratio,
)
from phase1.types import Action, DriveWeights, GridState, PredictedState
from phase1.world_model import EnsembleErrorComputer, LearningModule, WorldModel

# ---------- Fixtures ----------

@pytest.fixture
def trajectories():
    """Three hand-crafted trajectories for metric computation."""
    return [
        # Episode 1: reaches goal at step 4
        [
            GridState(agent=(0, 0), goal=(0, 4), width=5, height=5, step=0),
            GridState(agent=(0, 1), goal=(0, 4), width=5, height=5, step=1),
            GridState(agent=(0, 2), goal=(0, 4), width=5, height=5, step=2),
            GridState(agent=(0, 3), goal=(0, 4), width=5, height=5, step=3),
            GridState(agent=(0, 4), goal=(0, 4), width=5, height=5, step=4),
        ],
        # Episode 2: does not reach goal (agent at (0, 0), goal at (4, 4))
        [
            GridState(agent=(0, 0), goal=(4, 4), width=5, height=5, step=0),
            GridState(agent=(1, 0), goal=(4, 4), width=5, height=5, step=1),
            GridState(agent=(2, 0), goal=(4, 4), width=5, height=5, step=2),
        ],
        # Episode 3: reaches goal at step 6 (lots of revisits)
        [
            GridState(agent=(2, 2), goal=(2, 0), width=5, height=5, step=0),
            GridState(agent=(2, 1), goal=(2, 0), width=5, height=5, step=1),
            GridState(agent=(2, 0), goal=(2, 0), width=5, height=5, step=2),
        ],
    ]


@pytest.fixture
def predictions():
    """Hand-crafted predictions matching the first trajectory."""
    return [
        PredictedState(
            level1_exit_code=0, level1_confidence=1.0,
            level2_next_agent=(0, 1), level2_confidence=1.0,
            level3_output_summary="moved down", level3_confidence=1.0,
            epistemic_ratio=0.0,
        ),
        PredictedState(
            level1_exit_code=0, level1_confidence=1.0,
            level2_next_agent=(0, 2), level2_confidence=1.0,
            level3_output_summary="moved down", level3_confidence=1.0,
            epistemic_ratio=0.0,
        ),
        PredictedState(
            level1_exit_code=0, level1_confidence=1.0,
            level2_next_agent=(0, 3), level2_confidence=1.0,
            level3_output_summary="moved down", level3_confidence=1.0,
            epistemic_ratio=0.0,
        ),
        PredictedState(
            level1_exit_code=2, level1_confidence=1.0,
            level2_next_agent=(0, 4), level2_confidence=1.0,
            level3_output_summary="reached goal", level3_confidence=1.0,
            epistemic_ratio=0.0,
        ),
    ]


@pytest.fixture
def stub_wm():
    return WorldModel(use_stub=True)


@pytest.fixture
def stub_computer(stub_wm):
    return EnsembleErrorComputer(stub_wm, num_checkpoints=2)


# ---------- next_state_accuracy ----------

class TestNextStateAccuracy:
    def test_perfect_accuracy(self, predictions):
        actuals = [
            GridState(agent=(0, 1), goal=(0, 4), width=5, height=5),
            GridState(agent=(0, 2), goal=(0, 4), width=5, height=5),
            GridState(agent=(0, 3), goal=(0, 4), width=5, height=5),
            GridState(agent=(0, 4), goal=(0, 4), width=5, height=5),
        ]
        assert next_state_accuracy(predictions, actuals) == pytest.approx(1.0)

    def test_partial_accuracy(self, predictions):
        actuals = [
            GridState(agent=(0, 1), goal=(0, 4), width=5, height=5),
            GridState(agent=(0, 2), goal=(0, 4), width=5, height=5),
            GridState(agent=(0, 2), goal=(0, 4), width=5, height=5),  # mismatch: predicted (0,3)
            GridState(agent=(0, 4), goal=(0, 4), width=5, height=5),
        ]
        assert next_state_accuracy(predictions, actuals) == pytest.approx(0.75)

    def test_empty_inputs(self):
        assert next_state_accuracy([], []) == 0.0
        assert next_state_accuracy([], [GridState(agent=(0, 0), goal=(0, 0))]) == 0.0

    def test_different_lengths(self):
        preds = [
            PredictedState(
                level1_exit_code=0, level1_confidence=1.0,
                level2_next_agent=(1, 0), level2_confidence=1.0,
                level3_output_summary="", level3_confidence=1.0,
            )
        ]
        actuals = [
            GridState(agent=(1, 0), goal=(4, 4)),
            GridState(agent=(2, 0), goal=(4, 4)),
        ]
        # min(len(preds), len(actuals)) = 1, match = 1
        assert next_state_accuracy(preds, actuals) == pytest.approx(1.0)

    def test_zero_matches(self, predictions):
        actuals = [
            GridState(agent=(4, 4), goal=(0, 4), width=5, height=5),
            GridState(agent=(4, 4), goal=(0, 4), width=5, height=5),
            GridState(agent=(4, 4), goal=(0, 4), width=5, height=5),
            GridState(agent=(4, 4), goal=(0, 4), width=5, height=5),
        ]
        assert next_state_accuracy(predictions, actuals) == pytest.approx(0.0)


# ---------- completion_rate_at_horizon ----------

class TestCompletionRateAtHorizon:
    def test_all_complete_at_horizon_5(self, trajectories):
        # Ep 1: goal at step 4 < 5 => complete
        # Ep 2: never reaches goal (goal=(4,4), never gets there) => not complete
        # Ep 3: goal at step 2 < 5 => complete
        assert completion_rate_at_horizon(trajectories, 5) == pytest.approx(2 / 3)

    def test_none_complete_early_horizon(self, trajectories):
        # Ep 1: goal at step 4, horizon 2 => not complete
        # Ep 2: never reaches goal
        # Ep 3: goal at step 2, horizon 2 => complete (index 2 <= horizon 2)
        assert completion_rate_at_horizon(trajectories, 2) == pytest.approx(1 / 3)

    def test_all_complete_high_horizon(self, trajectories):
        # Ep 1 reaches goal, Ep 2 never reaches goal, Ep 3 reaches goal
        assert completion_rate_at_horizon(trajectories, 50) == pytest.approx(2 / 3)

    def test_empty_input(self):
        assert completion_rate_at_horizon([], 10) == 0.0

    def test_success_at_step_0(self):
        """Agent already at goal at step 0 should count as success."""
        traj = [[GridState(agent=(0, 0), goal=(0, 0), step=0)]]
        assert completion_rate_at_horizon(traj, 0) == pytest.approx(1.0)


# ---------- steps_to_goal_ratio ----------

class TestStepsToGoalRatio:
    def test_ratio_below_one(self):
        r = steps_to_goal_ratio(10.0, 25.0)
        assert r == pytest.approx(0.4)

    def test_equal_steps(self):
        r = steps_to_goal_ratio(20.0, 20.0)
        assert r == pytest.approx(1.0)

    def test_zero_random_steps(self):
        assert steps_to_goal_ratio(10.0, 0.0) == 0.0

    def test_negative_random(self):
        assert steps_to_goal_ratio(10.0, -1.0) == 0.0


# ---------- revisit_rate ----------

class TestRevisitRate:
    def test_no_revisits(self):
        states = [
            GridState(agent=(0, 0), goal=(4, 4)),
            GridState(agent=(0, 1), goal=(4, 4)),
            GridState(agent=(0, 2), goal=(4, 4)),
        ]
        assert revisit_rate(states) == pytest.approx(0.0)

    def test_some_revisits(self):
        states = [
            GridState(agent=(0, 0), goal=(4, 4)),
            GridState(agent=(1, 1), goal=(4, 4)),
            GridState(agent=(0, 0), goal=(4, 4)),  # revisit
            GridState(agent=(1, 1), goal=(4, 4)),  # revisit
            GridState(agent=(2, 2), goal=(4, 4)),
        ]
        # 5 positions, 3 unique => 2 repeated => 2/5 = 0.4
        assert revisit_rate(states) == pytest.approx(0.4)

    def test_all_same(self):
        states = [
            GridState(agent=(2, 2), goal=(4, 4)),
            GridState(agent=(2, 2), goal=(4, 4)),
            GridState(agent=(2, 2), goal=(4, 4)),
        ]
        assert revisit_rate(states) == pytest.approx(2 / 3)

    def test_empty(self):
        assert revisit_rate([]) == 0.0

    def test_single(self):
        states = [GridState(agent=(0, 0), goal=(4, 4))]
        assert revisit_rate(states) == pytest.approx(0.0)


# ---------- random_baseline ----------

class TestRandomBaseline:
    def test_baseline_returns_keys(self):
        env = GridWorld()
        baseline = random_baseline(env, n=10)
        assert "mean_steps" in baseline
        assert "success_rate" in baseline
        assert "mean_revisit_rate" in baseline
        assert "completion_5" in baseline
        assert "completion_10" in baseline
        assert "completion_20" in baseline
        assert baseline["n"] == 10

    def test_baseline_deterministic(self):
        env = GridWorld()
        a = random_baseline(env, n=5)
        b = random_baseline(env, n=5)
        for key in ("mean_steps", "success_rate", "mean_revisit_rate"):
            assert a[key] == pytest.approx(b[key])

    def test_baseline_reasonable_values(self):
        env = GridWorld()
        baseline = random_baseline(env, n=10)
        assert 0.0 <= baseline["success_rate"] <= 1.0
        assert baseline["mean_steps"] > 0
        assert 0.0 <= baseline["mean_revisit_rate"] <= 1.0
        assert 0.0 <= baseline["completion_5"] <= 1.0

    def test_baseline_with_obstacles(self):
        env = GridWorld(obstacles=[(2, 2), (2, 3)], goal=(0, 0))
        baseline = random_baseline(env, n=5)
        assert isinstance(baseline["mean_steps"], float)


# ---------- aggregate_metrics ----------

class TestAggregateMetrics:
    def test_aggregate_returns_keys(self, trajectories):
        # Use first trajectory's agent positions to build matching predictions.
        # Traj 1: 5 states, so 4 transitions; traj 2: 3 states, 2 transitions; traj 3: 3 states, 2 transitions.
        dummy_preds = []
        for traj in trajectories:
            ep_preds = []
            for s in traj[1:]:
                ep_preds.append(
                    PredictedState(
                        level1_exit_code=0 if s.agent != s.goal else 2,
                        level1_confidence=1.0,
                        level2_next_agent=s.agent,
                        level2_confidence=1.0,
                        level3_output_summary="",
                        level3_confidence=1.0,
                    )
                )
            dummy_preds.append(ep_preds)

        action_histories = []
        for traj in trajectories:
            action_histories.append([Action("RIGHT")] * (len(traj) - 1))

        metrics = aggregate_metrics(trajectories, dummy_preds, action_histories)
        assert "episodes" in metrics
        assert "success_rate" in metrics
        assert "mean_steps" in metrics
        assert "next_state_accuracy" in metrics
        assert "revisit_rate" in metrics
        assert "completion_20" in metrics

    def test_aggregate_single_episode(self):
        traj = [[GridState(agent=(0, 0), goal=(0, 0), step=0)]]
        preds = [[]]
        actions = [[]]
        result = aggregate_metrics(traj, preds, actions)
        assert result["episodes"] == 1
        assert result["success_rate"] == pytest.approx(1.0)


# ---------- run_episode smoke ----------

class TestRunEpisode:
    """Smoke test: run_episode runs end-to-end with stub components."""

    def test_run_episode_smoke(self, stub_wm, stub_computer):
        env = GridWorld()
        drive = HomeostaticDriveSystem(DriveWeights())
        gen = ActionGenerator(stub_wm, stub_computer, drive)
        learning = LearningModule(stub_wm, stub_computer, buffer_size=10, update_interval=1000)
        traj, preds, hist, metrics = run_episode(env, stub_wm, stub_computer, drive, learning, gen, seed=42)
        assert isinstance(traj, list)
        assert isinstance(preds, list)
        assert isinstance(hist, list)
        assert isinstance(metrics, dict)
        assert "steps" in metrics
        assert "success" in metrics
        assert "reward" in metrics
        assert len(traj) >= 2
        assert len(preds) >= 1
        assert len(hist) >= 1

    def test_run_episode_deterministic(self, stub_wm, stub_computer):
        env = GridWorld()
        drive_a = HomeostaticDriveSystem(DriveWeights())
        gen_a = ActionGenerator(stub_wm, stub_computer, drive_a)
        learn_a = LearningModule(stub_wm, stub_computer, buffer_size=10, update_interval=1000)

        drive_b = HomeostaticDriveSystem(DriveWeights())
        gen_b = ActionGenerator(stub_wm, stub_computer, drive_b)
        learn_b = LearningModule(stub_wm, stub_computer, buffer_size=10, update_interval=1000)

        _, _, _, m_a = run_episode(env, stub_wm, stub_computer, drive_a, learn_a, gen_a, seed=7)
        _, _, _, m_b = run_episode(env, stub_wm, stub_computer, drive_b, learn_b, gen_b, seed=7)
        assert m_a["steps"] == m_b["steps"]
        assert m_a["success"] == m_b["success"]
        assert m_a["reward"] == pytest.approx(m_b["reward"])
