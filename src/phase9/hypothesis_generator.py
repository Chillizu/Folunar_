"""Hypothesis proposers for the Phase 9 Hypothesis-Generator direction.

The LLM is demoted from predictor to PROPOSER: it emits candidate
(action, claimed_outcome) hypotheses only — it never produces the epistemic
signal and is never trained. The discriminator (src/phase9/discriminator.py)
produces the signal.

Plan: PEDA_FINAL/phase9/plans/plan-hypothesis-generator.md §3.2
"""

import json
import re
from typing import Any, Dict, List, Protocol

from phase2.sandbox_env import _validate_command, generate_sandbox_candidates
from phase9.types import Hypothesis

# Proposer quality contract (gate F4): >= 3 valid distinct actions on >= 50% of
# sampled states; <= 90% duplicate proposals across 20 sampled states.
DIVERSITY_MIN_ACTIONS = 3
DIVERSITY_MIN_FRACTION_STATES = 0.5


class HypothesisGenerator(Protocol):
    def propose(self, state: Any) -> List[Hypothesis]: ...


class RuleBasedProposer:
    """Data-driven proposer: wraps generate_sandbox_candidates() output.

    Deterministic, whitelist-validated by construction, always available.
    """

    source = "candidates"

    def propose(self, state: Any) -> List[Hypothesis]:
        candidates = generate_sandbox_candidates(state)
        out = []
        for action in candidates:
            ok, _ = _validate_command(action)
            if ok:
                out.append(Hypothesis(action=action, claimed_outcome="", source=self.source))
        return out


class LLMHypothesisGenerator:
    """LLM proposer skeleton (MiniCPM5-1B-Q4 via llama-server OpenAI API).

    TODO(phase9 post-MVP): wire `_call_llm` to a llama-server /v1/chat/completions
    endpoint (own port; the 35B-A3B on :8080 measured ~90s/200 tokens, 1B keeps
    the CPU prototype viable). Until then `propose` raises NotImplementedError.

    Prompt template, JSON parsing, whitelist validation, and dedup are fully
    implemented so the F4 diversity gate is testable once the client lands.
    """

    def __init__(self, base_url: str = "http://127.0.0.1:8081", model: str = "MiniCPM5-1B",
                 temperature: float = 0.8, k: int = 5) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.k = k

    # -- prompt template --

    def _build_prompt(self, state: Any) -> str:
        state_text = state.to_structured_text() if hasattr(state, "to_structured_text") else str(state)
        return (
            "You are a shell-command proposer in a deterministic Linux sandbox.\n"
            f"Current state:\n{state_text}\n\n"
            f"Propose up to {self.k} whitelisted shell commands "
            "(ls, cd, cat, echo, mkdir, touch, pwd, wc, head, tail, grep, find) "
            "with a predicted outcome for each, as a JSON list "
            '[{"action": "...", "predicted_outcome": "..."}]. '
            "Do not include rm/mv/cp or any destructive commands. "
            "Wrong predictions are fine; syntactic validity matters."
        )

    # -- LLM call (TODO: llama-server integration) --

    def _call_llm(self, prompt: str) -> str:
        """POST the prompt to the llama-server OpenAI-compatible endpoint.

        TODO: implement with urllib.request (or requests) against
        {self.base_url}/v1/chat/completions with model={self.model},
        temperature={self.temperature}. Return the raw assistant text.
        """
        raise NotImplementedError(
            "LLMHypothesisGenerator._call_llm: llama-server client not wired yet "
            "(post-MVP step). RuleBasedProposer is the working proposer for the MVP."
        )

    # -- parsing / validation / dedup (implemented) --

    _JSON_BLOCK_RE = re.compile(r"\[.*\]", re.DOTALL)

    def _parse_response(self, text: str) -> List[Dict[str, str]]:
        """Extract [{action, predicted_outcome}] from a raw LLM response."""
        if not text:
            return []
        m = self._JSON_BLOCK_RE.search(text)
        if not m:
            return []
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return []
        out = []
        if not isinstance(data, list):
            return []
        for item in data:
            if isinstance(item, dict) and isinstance(item.get("action"), str):
                out.append({
                    "action": item["action"],
                    "predicted_outcome": str(item.get("predicted_outcome", "")),
                })
        return out

    def _validated(self, items: List[Dict[str, str]]) -> List[Hypothesis]:
        """Whitelist-validate and dedup, capped at k."""
        seen = set()
        out = []
        for item in items:
            action = item["action"].strip()
            if not action or action in seen:
                continue
            ok, _ = _validate_command(action)
            if not ok:
                continue
            seen.add(action)
            out.append(Hypothesis(action=action,
                                  claimed_outcome=item.get("predicted_outcome", ""),
                                  source="llm"))
            if len(out) >= self.k:
                break
        return out

    def propose(self, state: Any) -> List[Hypothesis]:
        prompt = self._build_prompt(state)
        raw = self._call_llm(prompt)  # NotImplementedError until client lands
        return self._validated(self._parse_response(raw))

    # -- F4 diversity gate (testable without the LLM) --

    @staticmethod
    def diversity_ok(proposals_per_state: List[List[Hypothesis]],
                     min_actions: int = DIVERSITY_MIN_ACTIONS,
                     min_fraction: float = DIVERSITY_MIN_FRACTION_STATES) -> bool:
        """F4 contract: >= min_actions valid distinct actions on >= fraction states."""
        if not proposals_per_state:
            return False
        ok_states = sum(1 for props in proposals_per_state
                        if len({p.action for p in props}) >= min_actions)
        return ok_states / len(proposals_per_state) >= min_fraction
