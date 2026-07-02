"""Tests for Perception render and ascii_render."""

from phase1.grid_env import Perception
from phase1.types import GridState


class TestPerception:
    """Deterministic text and ASCII rendering of GridState."""

    def test_render_with_obstacles(self):
        state = GridState(
            agent=(1, 2),
            goal=(4, 4),
            obstacles=[(2, 2), (3, 2)],
            width=5,
            height=5,
        )
        result = Perception.render(state)
        assert "Agent at (1, 2)." in result
        assert "Goal at (4, 4)." in result
        assert "Obstacles at (2,2),(3,2)." in result

    def test_render_no_obstacles(self):
        state = GridState(agent=(0, 0), goal=(1, 1), width=2, height=2)
        result = Perception.render(state)
        assert "Agent at (0, 0)." in result
        assert "Goal at (1, 1)." in result
        assert "Obstacles at none." in result

    def test_render_single_obstacle(self):
        state = GridState(
            agent=(0, 0), goal=(4, 4), obstacles=[(0, 1)], width=5, height=5
        )
        result = Perception.render(state)
        assert "Obstacles at (0,1)." in result

    def test_render_determinism(self):
        state = GridState(agent=(1, 2), goal=(4, 4), obstacles=[(2, 2)], width=5, height=5)
        a = Perception.render(state)
        b = Perception.render(state)
        assert a == b

    def test_render_format_roundtrip(self):
        """The render output contains all tuple coordinates as expected."""
        state = GridState(agent=(1, 2), goal=(4, 4), obstacles=[(2, 2)], width=5, height=5)
        text = Perception.render(state)
        # Coordinates appear in the output.
        assert "(1, 2)" in text
        assert "(4, 4)" in text
        assert "(2,2)" in text  # obstacle format has no space

    def test_ascii_render(self):
        state = GridState(
            agent=(1, 1),
            goal=(4, 4),
            obstacles=[(2, 2), (3, 3)],
            width=5,
            height=5,
        )
        grid = Perception.ascii_render(state)
        lines = grid.split("\n")
        assert len(lines) == 5, "Should have 5 rows"
        # Top-left corner (row 0, col 0) should be empty '.'
        assert lines[0][0] == "."
        # Agent position
        assert lines[1][1] == "A"
        # Obstacle at (2,2)
        assert lines[2][2] == "#"
        # Obstacle at (3,3)
        assert lines[3][3] == "#"
        # Goal at (4,4)
        assert lines[4][4] == "G"

    def test_ascii_render_determinism(self):
        state = GridState(agent=(0, 0), goal=(4, 4), width=5, height=5)
        a = Perception.ascii_render(state)
        b = Perception.ascii_render(state)
        assert a == b

    def test_ascii_render_no_overlap(self):
        """Agent, goal, and obstacles each get distinct characters."""
        state = GridState(
            agent=(0, 0),
            goal=(0, 1),
            obstacles=[(1, 0), (1, 1)],
            width=2,
            height=2,
        )
        grid = Perception.ascii_render(state)
        lines = grid.split("\n")
        assert lines[0][0] == "A"
        assert lines[0][1] == "#"
        assert lines[1][0] == "G"
        assert lines[1][1] == "#"
