"""Discriminator-driven explorer for the Phase 9 Hypothesis-Generator direction.

Exploration score blends discriminator uncertainty with the PROVEN count-based
novelty baseline:

    score(s, a) = alpha * uncertainty_D(s, a) + (1 - alpha) * count_novelty(s, a)

Baseline constraint (shared context): must show improvement OVER count, not
over random — so the count term stays in the blend and is never replaced.

Plan: PEDA_FINAL/phase9/plans/plan-hypothesis-generator.md §3.4
"""

from typing import Any, List, Optional, Sequence, Tuple

from phase5.explorer import NoveltyExplorer
from phase9.discriminator import Discriminator, STRIPSDiscriminator
from phase9.types import Verdict

# Same tie-break priority table as phase5 (lower = preferred).
_ACTION_PRIORITY = {
    "cat": 0, "head": 0, "tail": 0,
    "grep": 1, "find": 1, "wc": 1,
    "cd": 2,
    "ls": 3, "pwd": 3, "echo": 3,
}


class DiscriminatorExplorer:
    """Explorer: alpha-blend of discriminator uncertainty and count novelty.

    Decision order (phase8 rule preserved):
      1. success-cache replay (if cached action still in candidates)
      2. argmax over score(s, a) = alpha*uncertainty_D + (1-alpha)*count_novelty,
         tie-broken by phase5 action priority (file readers > analysis > nav).
    """

    def __init__(self, alpha: float = 0.5,
                 discriminator: Optional[Discriminator] = None,
                 count_explorer: Optional[NoveltyExplorer] = None) -> None:
        self.alpha = alpha
        self.discriminator: Discriminator = discriminator if discriminator is not None else STRIPSDiscriminator()
        self.count_explorer = count_explorer if count_explorer is not None else NoveltyExplorer()
        # error buffer fed by observe() — usable for later batch discriminator updates
        self.error_buffer: List[Tuple[Any, str, Verdict, bool]] = []

    # -- scoring --

    def score(self, state: Any, action: str) -> float:
        """Exploration value of (state, action): uncertainty + count blend."""
        uncertainty = self.discriminator.uncertainty(state, action)
        count_novelty = self.count_explorer.novelty_bonus(state, action)
        return self.alpha * uncertainty + (1.0 - self.alpha) * count_novelty

    @staticmethod
    def _action_priority(action: str) -> int:
        verb = action.split()[0] if action else ""
        return _ACTION_PRIORITY.get(verb, 4)

    def select_action(self, state: Any, candidates: Sequence[str],
                      action_history: Optional[Sequence[str]] = None) -> str:
        """Select the highest-scoring candidate action."""
        if not candidates:
            return "ls"
        cands = list(candidates)

        # 1. success-cache replay (phase8 rule)
        sh = state.state_hash()
        cached = self.count_explorer.success_cache.get(sh)
        if cached is not None and cached in cands:
            return cached

        # 2. argmax score, tie-break by action priority
        return max(cands, key=lambda a: (self.score(state, a), -self._action_priority(a)))

    # -- feedback --

    def observe(self, state: Any, action: str, verdict: Optional[Verdict],
                success: bool) -> None:
        """Feed execution outcome back: counts + success cache + error buffer."""
        self.count_explorer.observe(state, action, success)
        if verdict is not None:
            self.error_buffer.append((state, action, verdict, success))

    def reset_episode(self) -> None:
        """No episode-local state; counts/discriminator persist across episodes."""
        pass
