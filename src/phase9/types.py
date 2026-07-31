"""Shared types for the Phase 9 Hypothesis-Generator direction.

Five atomic outcome predicates — NEVER file contents (worklog L1775 lesson).
The discriminator predicts these; the LLM is demoted to proposer only.

Plan: PEDA_FINAL/phase9/plans/plan-hypothesis-generator.md §3.1
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

# Predicate names, canonical order (matches to_vector / MLP head order).
PREDICATE_NAMES: Tuple[str, ...] = (
    "exit_ok",
    "output_nonempty",
    "cwd_changed",
    "listing_changed",
    "cache_gained",
)


@dataclass
class OutcomePredicates:
    """Atomic outcome facts of executing one action in the sandbox.

    - exit_ok:          exit code == 0
    - output_nonempty:  stdout or stderr non-empty
    - cwd_changed:      action moved the working directory
    - listing_changed:  next_state.files != state.files
    - cache_gained:     file_cache gained an entry (cat/head/tail/wc hit)
    """

    exit_ok: bool
    output_nonempty: bool
    cwd_changed: bool
    listing_changed: bool
    cache_gained: bool

    def to_vector(self) -> Tuple[bool, bool, bool, bool, bool]:
        """Return the 5 predicates as a fixed-order boolean tuple."""
        return (
            self.exit_ok,
            self.output_nonempty,
            self.cwd_changed,
            self.listing_changed,
            self.cache_gained,
        )

    @staticmethod
    def from_transition(state: Any, action: str, next_state: Any) -> "OutcomePredicates":
        """Extract ground-truth predicates from an executed (state, action) pair.

        state / next_state are phase2.sandbox_env.SandboxState-like objects.
        """
        exit_ok = bool(getattr(next_state, "last_exit_code", 1) == 0)
        output = str(getattr(next_state, "last_output", "") or "")
        output_nonempty = bool(output.strip())
        cwd_changed = getattr(next_state, "cwd", None) != getattr(state, "cwd", None)
        listing_changed = (
            list(getattr(next_state, "files", [])) != list(getattr(state, "files", []))
        )
        # cache_gained: any key in next cache that was absent before
        prev_cache = dict(getattr(state, "file_cache", {}) or {})
        next_cache = dict(getattr(next_state, "file_cache", {}) or {})
        cache_gained = any(k not in prev_cache for k in next_cache)
        return OutcomePredicates(
            exit_ok=exit_ok,
            output_nonempty=output_nonempty,
            cwd_changed=cwd_changed,
            listing_changed=listing_changed,
            cache_gained=cache_gained,
        )

    @staticmethod
    def from_vector(v: Tuple[bool, bool, bool, bool, bool]) -> "OutcomePredicates":
        """Inverse of to_vector()."""
        return OutcomePredicates(*v)

    @staticmethod
    def hamming(a: "OutcomePredicates", b: "OutcomePredicates") -> float:
        """Fraction of mismatched predicates, 0.0 (identical) .. 1.0."""
        va, vb = a.to_vector(), b.to_vector()
        if len(va) == 0:
            return 0.0
        return sum(1 for x, y in zip(va, vb) if x != y) / len(va)

    def to_dict(self) -> Dict[str, bool]:
        return dict(zip(PREDICATE_NAMES, self.to_vector()))

    @staticmethod
    def from_dict(d: Dict[str, bool]) -> "OutcomePredicates":
        return OutcomePredicates(
            exit_ok=bool(d.get("exit_ok", False)),
            output_nonempty=bool(d.get("output_nonempty", False)),
            cwd_changed=bool(d.get("cwd_changed", False)),
            listing_changed=bool(d.get("listing_changed", False)),
            cache_gained=bool(d.get("cache_gained", False)),
        )


@dataclass
class Hypothesis:
    """A candidate (action, claimed_outcome) pair from the proposer.

    claimed_outcome is AUDIT ONLY — free-text from the LLM, never part of the
    exploration signal (the discriminator produces the signal).
    """

    action: str
    claimed_outcome: str = ""
    source: str = "candidates"  # "llm" | "candidates" | "strips"

    def to_dict(self) -> Dict[str, str]:
        return {
            "action": self.action,
            "claimed_outcome": self.claimed_outcome,
            "source": self.source,
        }


@dataclass
class Verdict:
    """Discriminator output for one (state, action) pair.

    - predicates:  per-predicate prediction
    - confidence:  0..1, aggregated per-predicate probability
    - uncertainty: pre-execution exploration value = 1 - confidence
    - error:       post-execution hamming(pred, ground_truth); None before execution
    """

    predicates: OutcomePredicates
    confidence: float
    uncertainty: float
    error: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "predicates": self.predicates.to_dict(),
            "confidence": round(self.confidence, 4),
            "uncertainty": round(self.uncertainty, 4),
            "error": None if self.error is None else round(self.error, 4),
        }


@dataclass
class Transition:
    """One executed (state, action) -> next_state step with ground truth."""

    state: Any
    action: str
    next_state: Any
    ground_truth: OutcomePredicates
    success: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state": _state_to_dict(self.state),
            "action": self.action,
            "next_state": _state_to_dict(self.next_state),
            "ground_truth": self.ground_truth.to_dict(),
            "success": bool(self.success),
        }


def _state_to_dict(state: Any) -> Dict[str, Any]:
    """Serialize a SandboxState-like object (positional-construction safe)."""
    if state is None:
        return {}
    return {
        "cwd": getattr(state, "cwd", ""),
        "files": list(getattr(state, "files", [])),
        "file_cache": dict(getattr(state, "file_cache", {}) or {}),
        "last_command": getattr(state, "last_command", ""),
        "last_output": str(getattr(state, "last_output", "") or "")[:200],
        "last_exit_code": getattr(state, "last_exit_code", 0),
        "step_count": getattr(state, "step_count", 0),
    }


def state_from_dict(d: Dict[str, Any]):
    """Rebuild a SandboxState from _state_to_dict output (lazy import)."""
    from phase2.sandbox_env import SandboxState

    return SandboxState(
        container_id="",
        cwd=d.get("cwd", "/sandbox"),
        last_command=d.get("last_command", ""),
        last_output=d.get("last_output", ""),
        last_exit_code=d.get("last_exit_code", 0),
        files=list(d.get("files", [])),
        file_cache=dict(d.get("file_cache", {}) or {}),
        step_count=d.get("step_count", 0),
    )


def transition_from_dict(d: Dict[str, Any]) -> Transition:
    """Rebuild a Transition from to_dict output."""
    gt = OutcomePredicates.from_dict(d.get("ground_truth", {}))
    return Transition(
        state=state_from_dict(d.get("state", {})),
        action=d.get("action", ""),
        next_state=state_from_dict(d.get("next_state", {})),
        ground_truth=gt,
        success=bool(d.get("success", False)),
    )
