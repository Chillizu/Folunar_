"""Phase 9, Direction 2: Hierarchical Horizon Decomposition.

Two-layer architecture (plan-hierarchical-horizon.md):
  - High level (`planner.py`): analytic delayed-reward scoring over a visit
    map — J(f) = G(f) - lambda*d with horizon H_plan = 20..100.
  - Low level (`executor.py`): pragmatic path-finder (BFS) / count-based
    novelty wanderer (Phase-6 MazeNoveltyExplorer), horizon 1-3.

The planner never introspects cell contents; everything it knows comes from
StepRecords reported by the executor (`runner.py` drives the loop).
"""
