"""Tests for WorldModel: prompt building, parsing, stub prediction, checkpoint."""

import json

import pytest

from phase1.types import Action, GridState
from phase1.world_model import WorldModel

# Reusable fixtures as module-level constants.
DEFAULT_STATE = GridState(
    agent=(1, 2), goal=(4, 4), obstacles=[(2, 2), (3, 2)], width=5, height=5
)


# ---------- Fixtures ----------

@pytest.fixture
def stub_wm():
    return WorldModel(use_stub=True)


# ---------- Initialization ----------

class TestWorldModelInit:
    def test_stub_mode(self):
        wm = WorldModel(use_stub=True)
        assert wm.mode == "stub"
        assert wm.model is None

    def test_stub_by_env_var(self, monkeypatch):
        monkeypatch.setenv("FOLUNAR_STUB_MODEL", "1")
        wm = WorldModel()
        assert wm.mode == "stub"

    def test_default_model_name(self):
        wm = WorldModel(use_stub=True)
        assert wm.model_name == "Qwen/Qwen2.5-1.5B-Instruct"

    def test_custom_model_name(self):
        wm = WorldModel(model_name="my/model", use_stub=True)
        assert wm.model_name == "my/model"


# ---------- Prompt building ----------

class TestBuildPrompt:
    def test_prompt_format(self, stub_wm):
        state = DEFAULT_STATE
        action = Action("UP")
        prompt = stub_wm._build_prompt(state, action)
        assert "State: " in prompt
        assert "Action: UP" in prompt
        assert "Predict next position, exit code, and one-line summary as JSON:" in prompt
        assert "Agent at (1, 2)" in prompt
        assert "Goal at (4, 4)" in prompt
        assert "Obstacles at (2,2),(3,2)" in prompt

    def test_prompt_action_none(self, stub_wm):
        state = DEFAULT_STATE
        prompt = stub_wm._build_prompt(state, None)
        assert "Action: NONE" in prompt

    def test_prompt_determinism(self, stub_wm):
        state = DEFAULT_STATE
        a = stub_wm._build_prompt(state, Action("DOWN"))
        b = stub_wm._build_prompt(state, Action("DOWN"))
        assert a == b


# ---------- JSON parsing fallback ----------

class TestParseGeneration:
    @pytest.fixture(autouse=True)
    def _wm(self):
        return WorldModel(use_stub=True)

    def test_valid_json(self, _wm):
        text = '{"next_position": [3, 2], "exit_code": 0, "summary": "agent moved right"}'
        parsed = _wm._parse_generation(text)
        assert parsed.get("next_position") == [3, 2]
        assert parsed.get("exit_code") == 0
        assert "agent moved right" in parsed.get("summary", "")

    def test_json_among_text(self, _wm):
        text = (
            "Here is the prediction:\n"
            '{"next_position": [1, 3], "exit_code": 0, "summary": "moved down"}\n'
            "END"
        )
        parsed = _wm._parse_generation(text)
        assert parsed.get("next_position") == [1, 3]

    def test_empty_text(self, _wm):
        parsed = _wm._parse_generation("")
        assert parsed == {}

    def test_malformed_no_json(self, _wm):
        parsed = _wm._parse_generation("no json here at all")
        assert parsed == {}

    def test_partial_json(self, _wm):
        """Broken JSON returns empty dict (parse failure)."""
        parsed = _wm._parse_generation('{"next_position": [3')
        assert parsed == {}

    def test_invalid_json_object(self, _wm):
        """A bare JSON string (not a dict) is returned as-is by json.loads."""
        # json.loads('"just a string"') returns the string "just a string".
        # The function returns it as-is since it's not a JSON object.
        parsed = _wm._parse_generation('"just a string"')
        assert isinstance(parsed, str)
        assert parsed == "just a string"


# ---------- Stub prediction ----------

class TestStubPredict:
    def test_predict_up(self, stub_wm):
        state = GridState(agent=(2, 2), goal=(4, 4), width=5, height=5)
        action = Action("UP")
        pred = stub_wm.predict(state, action)
        assert pred.level2_next_agent == (2, 1)
        assert pred.level1_exit_code == 0
        assert pred.level2_confidence == 1.0
        assert pred.level1_confidence == 1.0
        assert pred.epistemic_ratio == 0.0
        assert "moved up" in pred.level3_output_summary.lower()

    def test_predict_down(self, stub_wm):
        state = GridState(agent=(2, 2), goal=(4, 4), width=5, height=5)
        pred = stub_wm.predict(state, Action("DOWN"))
        assert pred.level2_next_agent == (2, 3)

    def test_predict_left(self, stub_wm):
        state = GridState(agent=(2, 2), goal=(4, 4), width=5, height=5)
        pred = stub_wm.predict(state, Action("LEFT"))
        assert pred.level2_next_agent == (1, 2)

    def test_predict_right(self, stub_wm):
        state = GridState(agent=(2, 2), goal=(4, 4), width=5, height=5)
        pred = stub_wm.predict(state, Action("RIGHT"))
        assert pred.level2_next_agent == (3, 2)

    def test_predict_wall_collision(self, stub_wm):
        state = GridState(agent=(0, 0), goal=(4, 4), width=5, height=5)
        pred = stub_wm.predict(state, Action("UP"))
        assert pred.level2_next_agent == (0, 0)
        assert pred.level1_exit_code == 1
        assert "wall" in pred.level3_output_summary or "hit" in pred.level3_output_summary

    def test_predict_goal(self, stub_wm):
        state = GridState(agent=(3, 4), goal=(3, 4), width=5, height=5)
        # Agent is already at goal; but going DOWN would hit wall.
        pred = stub_wm.predict(state, Action("RIGHT"))
        # This is a normal move away from goal
        assert pred is not None

    def test_predict_reaching_goal(self, stub_wm):
        state = GridState(agent=(3, 3), goal=(3, 4), width=5, height=5)
        pred = stub_wm.predict(state, Action("DOWN"))
        assert pred.level2_next_agent == (3, 4)
        assert pred.level1_exit_code == 2
        assert "goal" in pred.level3_output_summary.lower()

    def test_predict_action_none(self, stub_wm):
        state = GridState(agent=(2, 2), goal=(4, 4), width=5, height=5)
        pred = stub_wm.predict(state, None)
        assert pred.level2_next_agent == (2, 2)
        assert pred.level1_exit_code == 1
        assert "no action" in pred.level3_output_summary

    def test_predict_preserves_obstacles(self, stub_wm):
        state = GridState(agent=(1, 1), goal=(4, 4), obstacles=[(1, 2)], width=5, height=5)
        pred = stub_wm.predict(state, Action("DOWN"))
        assert pred.level2_next_agent == (1, 1)
        assert pred.level1_exit_code == 1


# ---------- Rollout ----------

class TestRollout:
    def test_rollout_horizon_1(self, stub_wm):
        state = GridState(agent=(2, 2), goal=(4, 4), width=5, height=5)
        traj = stub_wm.rollout(state, Action("RIGHT"), horizon=1)
        assert len(traj) == 1
        assert traj[0].level2_next_agent == (3, 2)

    def test_rollout_horizon_3(self, stub_wm):
        state = GridState(agent=(2, 2), goal=(4, 4), width=5, height=5)
        traj = stub_wm.rollout(state, Action("RIGHT"), horizon=3)
        assert len(traj) == 3
        # Predictions chain: (2,2) -> (3,2) -> (4,2) -> (4,2 stays at wall)
        assert traj[0].level2_next_agent == (3, 2)
        assert traj[1].level2_next_agent == (4, 2)
        # Third step hits the right wall at x=4, stays at (4,2)
        assert traj[2].level2_next_agent == (4, 2)

    def test_rollout_terminates_at_goal(self, stub_wm):
        state = GridState(agent=(3, 3), goal=(4, 3), width=5, height=5)
        traj = stub_wm.rollout(state, Action("RIGHT"), horizon=3)
        assert len(traj) == 3
        # First step reaches goal
        assert traj[0].level1_exit_code == 2
        # The rollout keeps going synthetic steps after goal (greedy phase 1)
        assert traj[1].level2_next_agent is not None  # synthetic continuation

    def test_rollout_default_horizon(self, stub_wm):
        state = GridState(agent=(2, 2), goal=(4, 4), width=5, height=5)
        traj = stub_wm.rollout(state, Action("UP"))
        assert len(traj) == 2


# ---------- Checkpoint ----------

class TestCheckpoint:
    def test_save_stub_checkpoint(self, stub_wm, tmp_path):
        ckpt_path = stub_wm.save_lora_checkpoint(step=1)
        assert ckpt_path.exists()

    def test_save_checkpoint_creates_marker(self, stub_wm, tmp_path):
        ckpt_path = stub_wm.save_lora_checkpoint(step=42)
        marker = ckpt_path / "stub_checkpoint.json"
        assert marker.exists()
        data = json.loads(marker.read_text())
        assert data["step"] == 42
        assert data["mode"] == "stub"

    def test_predict_with_checkpoint_stub(self, stub_wm):
        """In stub mode, predict_with_checkpoint is a pass-through to predict."""
        state = GridState(agent=(2, 2), goal=(4, 4), width=5, height=5)
        ckpt = stub_wm.save_lora_checkpoint(step=1)
        pred = stub_wm.predict_with_checkpoint(state, Action("UP"), ckpt)
        assert pred.level2_next_agent == (2, 1)
