"""Count-based novelty explorer for sandbox environments.

Replaces the LLM-based World Model epistemic signal with a simple
count-based novelty bonus. No neural networks, no predictions to get wrong.
"""

import math
from collections import defaultdict


class NoveltyExplorer:
    """Count-based novelty-driven action selector.

    Maintains two count tables:
      - state_counts: how many times we've visited each (cwd, files) state
      - state_action_counts: how many times we've tried each (state, action) pair

    Also caches successful (state → action) pairs so known solutions are replayed.

    Intrinsic reward = 0.5 * (1/sqrt(1+state_novelty)) + 0.5 * (1/sqrt(1+pair_novelty))
    """

    def __init__(self):
        self.state_counts = defaultdict(int)            # state_hash → visit count
        self.state_action_counts = defaultdict(int)      # (state_hash, action) → try count
        self.success_cache = {}                          # state_hash → action that succeeded

    def novelty_bonus(self, state, action: str) -> float:
        """Intrinsic novelty reward for (state, action) pair.

        Higher = less explored = more valuable to try.
        """
        sh = state.state_hash()
        state_novelty = 1.0 / math.sqrt(1 + self.state_counts[sh])
        pair_novelty = 1.0 / math.sqrt(1 + self.state_action_counts[(sh, action)])
        return 0.5 * state_novelty + 0.5 * pair_novelty

    # Action-type priority for tie-breaking when novelty bonuses are equal.
    # Lower = prefer: file readers > content analysis > navigation > passive.
    _ACTION_PRIORITY = {
        "cat": 0, "head": 0, "tail": 0,
        "grep": 1, "find": 1, "wc": 1,
        "cd": 2,
        "ls": 3, "pwd": 3, "echo": 3,
    }

    def _action_priority(self, action: str) -> int:
        verb = action.split()[0] if action else ""
        return self._ACTION_PRIORITY.get(verb, 4)

    def select_action(self, state, candidates, action_history):
        """Select the most novel action from candidates.

        Decision order:
        1. Cached success -> replay (if action still in candidates)
        2. Highest novelty bonus, tie-broken by action priority
           (file readers > analysis > navigation > passive)

        Edge: empty candidates -> returns "ls" (always valid).
        """
        if not candidates:
            return "ls"

        sh = state.state_hash()

        # Cached success replay
        if sh in self.success_cache:
            cached = self.success_cache[sh]
            if cached in candidates:
                return cached

        # Highest novelty bonus, tie-break by action type priority
        return min(candidates, key=lambda a: (
            -self.novelty_bonus(state, a),
            self._action_priority(a),
        ))

    def observe(self, state, action: str, success: bool):
        """Record an execution outcome and update counts."""
        sh = state.state_hash()
        self.state_counts[sh] += 1
        self.state_action_counts[(sh, action)] += 1
        if success:
            self.success_cache[sh] = action

    def reset_episode(self):
        """Call at start of each episode to clear episode-local state.

        Currently no episode-local state — counts persist across episodes
        so the explorer gets better over time.
        """
        pass
