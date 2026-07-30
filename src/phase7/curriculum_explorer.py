"""Curriculum Intrinsic Motivation Explorer.

Warm-starts with count-based novelty, then linearly phases in JEPA epistemic
uncertainty. The key hypothesis: counting is optimal early (explore unseen
rooms) while epistemic uncertainty is valuable later (revisit to refine
understanding).

Interface:
    score(state, action, state_text) -> float: combined intrinsic reward.
    advance_episode(): step the curriculum schedule.
"""

import math


class CurriculumExplorer:
    """Action scoring with curriculum phase-in from count novelty to JEPA epistemic.

    During warmup episodes: only count-based novelty (alpha=0).
    After warmup: linearly phases in epistemic signal, capping at 0.7 weight.

    The final score is multiplicative: nov * (1 + alpha * ep) when novelty > 0,
    falling back to ep * 0.1 when novelty is zero (ensures epistemic signal
    still has some weight for fully-explored state-action pairs).
    """

    def __init__(
        self,
        novelty_explorer=None,
        jepa_wm=None,
        warmup_episodes: int = 2,
        phase_in_episodes: int = 5,
        epistemic_cap: float = 0.7,
    ):
        """Initialize curriculum explorer.

        Args:
            novelty_explorer: object with novelty_bonus(state, action) -> float.
            jepa_wm: JEPAEnsemble with epistemic_uncertainty(text, action) -> float.
            warmup_episodes: number of episodes with pure count-based novelty.
            phase_in_episodes: episodes over which epistemic phases in linearly.
            epistemic_cap: maximum alpha (epistemic weight multiplier).
        """
        self.novelty = novelty_explorer
        self.jepa = jepa_wm
        self.warmup = warmup_episodes
        self.phase_in = phase_in_episodes
        self.epistemic_cap = epistemic_cap
        self.episode = 0

    # ── Scoring ──────────────────────────────────────────

    def score(
        self, state, action: str, state_text: str
    ) -> float:
        """Compute combined intrinsic reward for (state, action).

        Implements curriculum phase-in:
          episode < warmup: pure count novelty (alpha=0).
          episode >= warmup: alpha = clamp((ep-warmup)/phase_in, 0, 1) * cap.

        Final score: nov * (1 + alpha * ep) if nov > 0 else ep * 0.1.

        Args:
            state: env state object (with state_hash() for novelty lookup).
            action: action string.
            state_text: text representation for JEPA encoding.

        Returns:
            Combined intrinsic reward score (float).
        """
        nov = self.novelty.novelty_bonus(state, action) if self.novelty else 0.0
        ep = (
            self.jepa.epistemic_uncertainty(state_text, action)
            if self.jepa
            else 0.0
        )

        # Curriculum phase-in
        if self.episode < self.warmup:
            alpha = 0.0
        else:
            progress = min(
                1.0, (self.episode - self.warmup) / max(self.phase_in, 1)
            )
            alpha = progress * self.epistemic_cap

        if nov > 0:
            return nov * (1.0 + alpha * ep)
        else:
            return ep * 0.1

    def advance_episode(self) -> None:
        """Advance the curriculum counter by one episode."""
        self.episode += 1

    def reset_episode(self) -> None:
        """Reset episode-local state. Counters survive across episodes."""
        pass
