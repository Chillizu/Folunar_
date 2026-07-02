"""Tests for GridWorld environment and Perception."""


import pytest

from phase1.grid_env import GridWorld
from phase1.types import Action, GridState


class TestGridWorld:
    """GridWorld creation, reset, step, and termination."""

    def test_default_construction(self):
        env = GridWorld()
        assert env.width == 5
        assert env.height == 5
        assert env.obstacles == []
        assert env.goal is None
        assert env.max_steps == 50

    def test_custom_construction(self):
        env = GridWorld(width=8, height=3, obstacles=[(1, 1)], goal=(7, 2), max_steps=30)
        assert env.width == 8
        assert env.height == 3
        assert env.obstacles == [(1, 1)]
        assert env.goal == (7, 2)
        assert env.max_steps == 30

    def test_obstacles_copied(self):
        obstacles = [(0, 1), (2, 2)]
        env = GridWorld(obstacles=obstacles)
        obstacles.append((4, 4))
        # Env should have its own copy.
        assert env.obstacles == [(0, 1), (2, 2)]

    # --- Reset ---

    def test_reset_returns_grid_state(self, seed=42):
        env = GridWorld()
        state = env.reset(seed=seed)
        assert isinstance(state, GridState)
        assert state.width == env.width
        assert state.height == env.height
        assert 0 <= state.agent[0] < env.width
        assert 0 <= state.agent[1] < env.height
        assert state.agent != state.goal
        assert state.step == 0
        assert state.max_steps == 50

    def test_reset_determinism(self):
        env = GridWorld()
        a = env.reset(seed=123)
        b = env.reset(seed=123)
        assert a == b

    def test_reset_different_seeds_different_agent(self):
        """Different seeds should typically produce different agent positions."""
        env = GridWorld()
        positions = {env.reset(seed=s).agent for s in range(20)}
        assert len(positions) > 1, "Seeds should yield at least some variety"

    def test_reset_agent_in_goal_connected_component(self):
        """Agent must be in the same connected component as the goal (path exists)."""
        env = GridWorld(width=4, height=4, obstacles=[(1, 1), (1, 2), (1, 3)])
        for seed in range(50):
            state = env.reset(seed=seed)
            reachable = env._connected_component(state.goal)
            assert state.agent in reachable, (
                f"Agent {state.agent} not reachable from goal {state.goal} "
                f"with obstacles {state.obstacles}"
            )

    def test_reset_not_enough_free_cells(self):
        env = GridWorld(width=2, height=1, obstacles=[(0, 0)])
        with pytest.raises(ValueError, match="Not enough free cells"):
            env.reset(seed=0)

    def test_reset_uses_preset_goal(self):
        env = GridWorld(goal=(3, 3))
        for seed in range(20):
            state = env.reset(seed=seed)
            assert state.goal == (3, 3)

    def test_reset_agent_not_on_goal(self):
        env = GridWorld()
        for seed in range(50):
            state = env.reset(seed=seed)
            assert state.agent != state.goal

    # --- Step (movement) ---

    def test_step_up(self):
        env = GridWorld()
        state = env.reset(seed=42)
        # Move UP
        next_state, reward, done = env.step(state, Action("UP"))
        expected_y = max(0, state.agent[1] - 1)
        assert next_state.agent == (state.agent[0], expected_y)
        assert not done
        assert next_state.step == state.step + 1

    def test_step_down(self):
        env = GridWorld()
        state = env.reset(seed=42)
        next_state, _, _ = env.step(state, Action("DOWN"))
        expected_y = min(state.height - 1, state.agent[1] + 1)
        assert next_state.agent == (state.agent[0], expected_y)

    def test_step_left(self):
        env = GridWorld()
        state = env.reset(seed=42)
        next_state, _, _ = env.step(state, Action("LEFT"))
        expected_x = max(0, state.agent[0] - 1)
        assert next_state.agent == (expected_x, state.agent[1])

    def test_step_right(self):
        env = GridWorld()
        state = env.reset(seed=42)
        next_state, _, _ = env.step(state, Action("RIGHT"))
        expected_x = min(state.width - 1, state.agent[0] + 1)
        assert next_state.agent == (expected_x, state.agent[1])

    def test_step_preserves_goal_and_obstacles(self):
        env = GridWorld(obstacles=[(2, 2)], goal=(4, 4))
        state = env.reset(seed=42)
        next_state, _, _ = env.step(state, Action("RIGHT"))
        assert next_state.goal == (4, 4)
        assert next_state.obstacles == [(2, 2)]

    # --- Step (collision) ---

    @pytest.mark.parametrize("action_name", ["UP", "LEFT"])
    def test_wall_collision_top_left(self, action_name):
        """Agent at (0,0) hitting the top or left wall stays in place."""
        state = GridState(agent=(0, 0), goal=(4, 4), obstacles=[], width=5, height=5)
        env = GridWorld()
        next_state, reward, done = env.step(state, Action(action_name))
        assert next_state.agent == (0, 0), f"{action_name} from (0,0) should stay"
        assert reward == -0.2
        assert not done

    @pytest.mark.parametrize("action_name", ["DOWN", "RIGHT"])
    def test_wall_collision_bottom_right(self, action_name):
        """Agent at (4,4) hitting bottom or right wall stays in place."""
        state = GridState(agent=(4, 4), goal=(0, 0), obstacles=[], width=5, height=5)
        env = GridWorld()
        next_state, reward, done = env.step(state, Action(action_name))
        assert next_state.agent == (4, 4), f"{action_name} from (4,4) should stay"
        assert reward == -0.2
        assert not done

    def test_obstacle_collision(self):
        state = GridState(agent=(1, 1), goal=(4, 4), obstacles=[(1, 2)], width=5, height=5)
        env = GridWorld()
        next_state, reward, done = env.step(state, Action("DOWN"))
        # (1, 2) is an obstacle, so agent stays at (1, 1)
        assert next_state.agent == (1, 1)
        assert reward == -0.2
        assert not done

    # --- Step (goal and termination) ---

    def test_goal_reached(self):
        state = GridState(agent=(3, 3), goal=(3, 4), obstacles=[], width=5, height=5)
        env = GridWorld()
        next_state, reward, done = env.step(state, Action("DOWN"))
        assert next_state.agent == (3, 4)
        assert reward == 1.0
        assert done

    def test_max_step_termination(self):
        env = GridWorld(max_steps=3)
        # Place the agent far from goal to prevent early success.
        state = GridState(agent=(0, 0), goal=(4, 4), obstacles=[], width=5, height=5, step=0, max_steps=3)
        for i in range(2):
            state, reward, done = env.step(state, Action("RIGHT"))
            assert not done, f"Should not be done at step {i+1}"
        # Third step triggers max_steps termination.
        state, reward, done = env.step(state, Action("RIGHT"))
        assert done, "Max steps should terminate the episode"
        assert state.step == 3

    def test_reward_default_step(self):
        state = GridState(agent=(1, 1), goal=(4, 4), obstacles=[], width=5, height=5)
        env = GridWorld()
        _, reward, _ = env.step(state, Action("RIGHT"))
        assert reward == -0.05

    # --- Static helpers ---

    def test_all_actions(self):
        actions = GridWorld.all_actions()
        names = [a.name for a in actions]
        assert names == ["UP", "DOWN", "LEFT", "RIGHT"]

    def test_connected_component(self):
        env = GridWorld(width=3, height=3, obstacles=[(1, 0), (1, 1), (1, 2)])
        reachable = env._connected_component((0, 0))
        # Only column 0 should be reachable.
        assert (0, 0) in reachable
        assert (0, 1) in reachable
        assert (0, 2) in reachable
        assert (2, 0) not in reachable
        assert (1, 1) not in reachable

    def test_reset_with_all_obstacles_blocking(self):
        """Agent must be placed on a cell reachable from the goal."""
        # Row 2 is a wall, so top and bottom halves are disconnected.
        obstacles = [(0, 2), (1, 2), (2, 2), (3, 2), (4, 2)]
        env = GridWorld(width=5, height=5, obstacles=obstacles)
        # Force the goal to the top half.
        env = GridWorld(width=5, height=5, obstacles=obstacles, goal=(0, 0))
        state = env.reset(seed=7)
        # Agent must be in the top half (rows 0-1).
        assert state.agent[1] in (0, 1), f"Agent {state.agent} should be in top half"

    def test_cycle_actions_all_directions(self):
        """Move in a cycle and verify return to start."""
        env = GridWorld()
        state = GridState(agent=(2, 2), goal=(4, 4), width=5, height=5)
        s1, _, _ = env.step(state, Action("RIGHT"))
        s2, _, _ = env.step(s1, Action("DOWN"))
        s3, _, _ = env.step(s2, Action("LEFT"))
        s4, _, _ = env.step(s3, Action("UP"))
        assert s4.agent == (2, 2)

    def test_invalid_action(self):
        state = GridState(agent=(2, 2), goal=(4, 4), width=5, height=5)
        env = GridWorld()
        with pytest.raises(ValueError, match="Unknown action"):
            env.step(state, Action("JUMP"))
