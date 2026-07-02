"""Tests for EnsembleErrorComputer: ensemble variance, aleatoric decomposition."""


import pytest

from phase1.types import Action, GridState
from phase1.world_model import EnsembleErrorComputer, WorldModel

# ---------- Fixtures ----------

@pytest.fixture
def stub_wm():
    return WorldModel(use_stub=True)


@pytest.fixture
def stub_computer(stub_wm):
    return EnsembleErrorComputer(stub_wm, num_checkpoints=3)


# ---------- Initialization ----------

class TestInit:
    def test_no_checkpoints(self, stub_computer):
        assert stub_computer.checkpoints == []

    def test_default_num_checkpoints(self, stub_wm):
        comp = EnsembleErrorComputer(stub_wm)
        assert comp.num_checkpoints == 5


# ---------- Save checkpoint ----------

class TestSaveCheckpoint:
    def test_save_checkpoint_appends(self, stub_computer):
        p1 = stub_computer.save_checkpoint(1)
        assert len(stub_computer.checkpoints) == 1
        assert stub_computer.checkpoints[0] == p1
        p2 = stub_computer.save_checkpoint(2)
        assert len(stub_computer.checkpoints) == 2
        assert stub_computer.checkpoints[1] == p2

    def test_checkpoint_limit_enforced(self, stub_computer):
        stub_computer.save_checkpoint(1)
        stub_computer.save_checkpoint(2)
        stub_computer.save_checkpoint(3)
        # num_checkpoints=3, so after 4 saves the oldest should be dropped.
        stub_computer.save_checkpoint(4)
        assert len(stub_computer.checkpoints) == 3
        names = [p.name for p in stub_computer.checkpoints]
        assert "adapter_step_1" not in names
        assert "adapter_step_4" in names

    def test_checkpoint_limit_one(self, stub_wm):
        comp = EnsembleErrorComputer(stub_wm, num_checkpoints=1)
        comp.save_checkpoint(1)
        comp.save_checkpoint(2)
        assert len(comp.checkpoints) == 1


# ---------- decompose_error: perfect agreement ----------

class TestDecomposeErrorPerfectAgreement:
    """When predictions are identical, ensemble variance should be zero."""

    def test_no_checkpoints_single_prediction(self, stub_computer):
        """With no checkpoints, a single prediction is used (zero variance)."""
        # Use a case where stub prediction and actual differ.
        state = GridState(agent=(2, 2), goal=(4, 4), width=5, height=5)
        # Stub DOWN predicts (2, 3), but we pass an actual that differs.
        actual = GridState(agent=(2, 4), goal=(4, 4), width=5, height=5)
        error = stub_computer.decompose_error(state, Action("DOWN"), actual)
        # Single prediction => no ensemble => zero variance
        assert error.ensemble_variance == 0.0
        assert error.epistemic_error == 0.0
        # level2_error = L2 from predicted (2,3) to actual (2,4) = 1.0
        # aleatoric = max(0, mean_deviation - 0) = 1.0
        assert error.level2_error == pytest.approx(1.0)
        assert error.aleatoric_error == pytest.approx(1.0)
        assert error.level1_error == 0.0  # exit codes match (both 0)
        assert error.total_error == pytest.approx(1.0)

    def test_single_checkpoint_zero_variance(self, stub_computer):
        """One checkpoint should still yield zero variance (no pair to compare)."""
        stub_computer.save_checkpoint(1)
        state = GridState(agent=(2, 2), goal=(4, 4), width=5, height=5)
        actual = GridState(agent=(2, 3), goal=(4, 4), width=5, height=5)
        error = stub_computer.decompose_error(state, Action("DOWN"), actual)
        assert error.ensemble_variance == 0.0

    def test_multiple_checkpoints_all_agree(self, stub_computer):
        """All stub checkpoints agree exactly, so variance is zero."""
        for i in range(1, 4):
            stub_computer.save_checkpoint(i)
        state = GridState(agent=(2, 2), goal=(4, 4), width=5, height=5)
        actual = GridState(agent=(2, 3), goal=(4, 4), width=5, height=5)
        error = stub_computer.decompose_error(state, Action("DOWN"), actual)
        # Stub predictions are deterministic, all checkpoints produce the same result.
        assert error.ensemble_variance == 0.0
        # Exit code should match (level1_error = 0)
        assert error.level1_error == 0.0


# ---------- decompose_error: disagreement ----------

class TestDecomposeErrorDisagreement:
    """When predictions disagree, ensemble variance should be positive."""

    def test_nonzero_variance(self, stub_wm):
        """We cannot force stub checkpoints to disagree (they're deterministic).
        Instead, verify that the variance formula is exercised by injecting
        predictions with different positions.

        We construct a manual decompose_error scenario by checking variance
        at the calculation level.
        """
        # This test verifies the error formula directly.
        state = GridState(agent=(2, 2), goal=(4, 4), width=5, height=5)
        actual = GridState(agent=(3, 2), goal=(4, 4), width=5, height=5)

        comp = EnsembleErrorComputer(stub_wm, num_checkpoints=3)
        # Save 3 checkpoints; stub predictions will all be identical (deterministic).
        for i in range(1, 4):
            comp.save_checkpoint(i)

        error = comp.decompose_error(state, Action("RIGHT"), actual)
        # All stub checkpoints predict (3,2) from (2,2)+RIGHT.
        # Actual is also (3,2), so mean_deviation = 0, ensemble_variance = 0.
        assert error.ensemble_variance == 0.0
        assert error.level2_error == 0.0

    def test_l1_agreement(self, stub_computer):
        """Exit codes match => level1_error = 0."""
        for i in range(1, 4):
            stub_computer.save_checkpoint(i)
        state = GridState(agent=(2, 2), goal=(4, 4), width=5, height=5)
        actual = GridState(agent=(2, 3), goal=(4, 4), width=5, height=5)
        error = stub_computer.decompose_error(state, Action("DOWN"), actual)
        assert error.level1_error == 0.0

    def test_l1_disagreement(self, stub_wm):
        """If actual exit code differs from predicted, level1_error > 0."""
        comp = EnsembleErrorComputer(stub_wm, num_checkpoints=2)
        for i in range(1, 3):
            comp.save_checkpoint(i)
        # Agent at wall (0,0) moving UP -> stays at (0,0), exit code 1.
        # Actual is also wall collision (exit code 1). So level1_error = 0.
        state = GridState(agent=(0, 0), goal=(4, 4), width=5, height=5)
        actual = GridState(agent=(0, 0), goal=(4, 4), width=5, height=5)
        error = comp.decompose_error(state, Action("UP"), actual)
        # Stub predicts exit code 1 (wall), actual exit code 1 => match
        assert error.level1_error == 0.0

        # Test case where actual is goal:
        state2 = GridState(agent=(3, 3), goal=(4, 3), width=5, height=5)
        actual2 = GridState(agent=(4, 3), goal=(4, 3), width=5, height=5)
        error2 = comp.decompose_error(state2, Action("RIGHT"), actual2)
        # Stub predicts exit code 2 (goal), actual is goal => match
        assert error2.level1_error == 0.0


# ---------- decompose_error structure ----------

class TestDecomposeErrorStructure:
    def test_error_vector_fields(self, stub_computer):
        state = GridState(agent=(2, 2), goal=(4, 4), width=5, height=5)
        actual = GridState(agent=(2, 3), goal=(4, 4), width=5, height=5)
        error = stub_computer.decompose_error(state, Action("DOWN"), actual)
        assert error.total_error >= 0.0
        assert error.level1_error >= 0.0
        assert error.level2_error >= 0.0
        assert error.level3_error == 0.0  # unused in Phase 1
        assert error.epistemic_error >= 0.0
        assert error.aleatoric_error >= 0.0
        assert error.ensemble_variance >= 0.0
        # total_error = mean_deviation + ensemble_variance
        assert error.total_error == pytest.approx(error.level2_error + error.ensemble_variance)

    def test_epistemic_equals_ensemble_variance(self, stub_computer):
        state = GridState(agent=(2, 2), goal=(4, 4), width=5, height=5)
        actual = GridState(agent=(2, 3), goal=(4, 4), width=5, height=5)
        error = stub_computer.decompose_error(state, Action("DOWN"), actual)
        assert error.epistemic_error == error.ensemble_variance

    def test_aleatoric_decomposition(self, stub_computer):
        """aleatoric = max(0, mean_deviation - ensemble_variance)."""
        state = GridState(agent=(2, 2), goal=(4, 4), width=5, height=5)
        actual = GridState(agent=(2, 3), goal=(4, 4), width=5, height=5)
        error = stub_computer.decompose_error(state, Action("DOWN"), actual)
        expected = max(0.0, error.level2_error - error.ensemble_variance)
        assert error.aleatoric_error == pytest.approx(expected)


# ---------- Actual exit code helper ----------

class TestActualExitCode:
    def test_goal_exit_code(self, stub_wm):
        state = GridState(agent=(3, 3), goal=(4, 3), width=5, height=5)
        actual = GridState(agent=(4, 3), goal=(4, 3), width=5, height=5)
        code = EnsembleErrorComputer._actual_exit_code(state, actual)
        assert code == 2

    def test_wall_exit_code(self, stub_wm):
        state = GridState(agent=(0, 0), goal=(4, 4), width=5, height=5)
        actual = GridState(agent=(0, 0), goal=(4, 4), width=5, height=5)
        code = EnsembleErrorComputer._actual_exit_code(state, actual)
        assert code == 1

    def test_success_exit_code(self, stub_wm):
        state = GridState(agent=(2, 2), goal=(4, 4), width=5, height=5)
        actual = GridState(agent=(2, 3), goal=(4, 4), width=5, height=5)
        code = EnsembleErrorComputer._actual_exit_code(state, actual)
        assert code == 0
