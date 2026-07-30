"""JEPA-based PEDA: Epistemic uncertainty drives action selection via EFE.

Architecture:
  SandboxState -> state_text -> JEPA Ensemble (frozen Qwen + 3 MLPs)
                                     |
                           epistemic_uncertainty = ensemble_variance(predictions)
                                     |
                           ActionGenerator.compute_efe(candidates):
                             for each action:
                               epistemic = jepa.epistemic_uncertainty(state_text, action)
                               pragmatic = task_relevance(action, task)
                               efe = epistemic * alpha + pragmatic * (1 - alpha)
                             pick action with max EFE
"""

import sys
from pathlib import Path

import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from phase5.explorer import NoveltyExplorer
from phase5.jepa_wm import JEPAEnsemble, state_to_text


def state_to_text_flat(state) -> str:
    """Convert SandboxState to flat text for encoder."""
    return state_to_text(state)


def _encode_action(action_text: str) -> str:
    """Normalize action text for encoding."""
    return action_text.strip()


class JEPAPEDA:
    """PEDA agent using JEPA epistemic uncertainty with hybrid novelty-epistemic scoring.

    Wraps a JEPAEnsemble (frozen Qwen 0.5B + 3 MLP predictors) and computes
    hybrid action scores from count-based novelty + epistemic uncertainty:

    score = novelty * (1 + beta * normalized_epistemic_uncertainty)
    """

    # Priority for tie-breaking among equal EFE scores (lower = preferred)
    _ACTION_PRIORITY = {
        "cat": 0, "head": 0, "tail": 0,
        "grep": 1, "find": 1, "wc": 1,
        "cd": 2,
        "ls": 3, "pwd": 3, "echo": 3,
    }

    def __init__(self, model_path: str, n_ensemble: int = 3,
                 device: str = "cuda", alpha: float = 0.5):
        """Initialize JEPA ensemble and EFE parameters.

        Args:
            model_path: Path to Qwen2.5-0.5B-Instruct model
            n_ensemble: Number of MLP predictors (default 3)
            device: torch device (default "cuda")
            alpha: Exploration weight (0=pragmatic only, 1=epistemic only)
        """
        self.alpha = alpha
        self.device = device
        self.jepa = JEPAEnsemble(model_path, n_ensemble=n_ensemble, device=device)
        # Novelty explorer for hybrid scoring
        self._novelty_explorer = NoveltyExplorer()
        # Dead-loop prevention: track recent (state_hash, action) pairs
        self._recent_history = []  # max 5 entries

    # ── Reset ────────────────────────────────────────────

    def reset(self):
        """Reset JEPA predictors and internal state for a fresh task.

        Reinitializes MLP weights (clears cross-task contamination)
        and clears dead-loop history and novelty counts.
        """
        self.jepa.reset_predictors()
        self._recent_history.clear()
        self._novelty_explorer = NoveltyExplorer()

    # ── EFE Components ────────────────────────────────────

    def epistemic_uncertainty(self, state_text: str, action: str) -> float:
        """Ensemble variance across 3 MLP predictors.

        Higher = more uncertain = worth exploring.
        Delegates directly to JEPAEnsemble.epistemic_uncertainty.
        """
        return self.jepa.epistemic_uncertainty(state_text, _encode_action(action))

    def pragmatic_value(self, action: str, task_id: str) -> float:
        """Task-relevance heuristic: does action match task keyword patterns?

        Returns 1.0 if action matches the task's expected behavior pattern,
        0.0 otherwise. Simple keyword-based matching.
        """
        action_lower = action.lower()

        if task_id == "read_hello":
            # Must read hello.txt
            return 1.0 if "hello.txt" in action_lower else 0.0

        elif task_id == "count_lines":
            # Must count lines via wc -l, targeting lines.txt or data/
            has_wc = action_lower.startswith("wc -l") or "wc -l " in action_lower
            targets_file = "lines.txt" in action_lower or "data" in action_lower
            return 1.0 if (has_wc and targets_file) else 0.0

        elif task_id == "find_secret":
            # Must grep for 'secret' or cat docs/note.txt
            is_grep = "grep" in action_lower and "secret" in action_lower
            is_cat = (action_lower.startswith("cat") or action_lower.startswith("head")
                      or action_lower.startswith("tail")) and "note" in action_lower
            return 1.0 if (is_grep or is_cat) else 0.0

        elif task_id == "read_note":
            # Must read a file reader targeting note.txt or docs/
            is_reader = (action_lower.startswith("cat") or action_lower.startswith("head")
                         or action_lower.startswith("tail"))
            targets = "note.txt" in action_lower or "docs" in action_lower
            return 1.0 if (is_reader and targets) else 0.0

        # Unknown task — no pragmatic signal
        return 0.0

    def compute_hybrid(self, state_text: str, candidates: list,
                        novelty_local, state, beta: float = 0.5) -> dict:
        """Compute hybrid novelty*epistemic score for each candidate.

        For each action:
          nov = novelty_local.novelty_bonus(state, action)
          ep  = self.jepa.epistemic_uncertainty(state_text, action)
          ep_norm = normalized across candidates
          score = nov * (1 + beta * ep_norm)

        Returns {action: hybrid_score}.
        """
        scores = {}
        eps = []
        for action in candidates:
            nov = novelty_local.novelty_bonus(state, action)
            ep = self.jepa.epistemic_uncertainty(state_text, action)
            scores[action] = nov
            eps.append(ep)

        # Normalize epistemic across candidates
        if len(eps) > 1:
            min_ep, max_ep = min(eps), max(eps)
            if max_ep > min_ep:
                for action, ep in zip(candidates, eps):
                    ep_norm = (ep - min_ep) / (max_ep - min_ep + 1e-8)
                    scores[action] = scores[action] * (1.0 + beta * ep_norm)

        return scores

    def _action_priority(self, action: str) -> int:
        verb = action.split()[0] if action else ""
        return self._ACTION_PRIORITY.get(verb, 4)

    def select_action(self, state, candidates: list, task_id: str) -> str:
        """Select best action using hybrid novelty*epistemic scoring.

        Picks candidate with highest hybrid score, tie-broken by action type
        priority (file readers > analysis > navigation > passive).

        Penalizes actions that repeat the same (state_hash, action) pair >= 2
        times in the last 5 steps to prevent dead loops.

        Falls back to "ls" if candidates empty.
        """
        if not candidates:
            return "ls"

        state_text = state_to_text_flat(state)
        hybrid_scores = self.compute_hybrid(
            state_text, candidates, self._novelty_explorer, state, beta=0.5,
        )

        # Observe selected action outcomes for novelty tracking
        # (we call select_action before observe, so we observe the prev action)
        if hasattr(self, '_last_state') and hasattr(self, '_last_action'):
            self._novelty_explorer.observe(
                self._last_state, self._last_action, False,
            )
        self._last_state = state
        self._last_action = None  # will be set when we pick

        # Dead-loop penalty: use stable filesystem signature (cwd + files)
        # NOT the full state_text (which includes ephemeral last_output)
        state_sig = (getattr(state, 'cwd', ''), frozenset(getattr(state, 'files', [])))
        penalized_scores = {}
        for action in candidates:
            score = hybrid_scores.get(action, 0.0)
            repeat_count = self._recent_history.count((state_sig, action))
            if repeat_count >= 2:
                score *= 0.1
            penalized_scores[action] = score

        best_action = max(candidates, key=lambda a: (
            penalized_scores.get(a, 0.0),
            -self._action_priority(a),
        ))

        # Record action for dead-loop detection
        self._recent_history.append((state_sig, best_action))
        if len(self._recent_history) > 5:
            self._recent_history.pop(0)

        self._last_action = best_action

        return best_action

    # ── Training ──────────────────────────────────────────

    def train_on_episode(self, transitions: list) -> float:
        """Batch train all 3 MLP predictors on episode transitions.

        Args:
            transitions: list of (state_text, action, next_state_text) tuples

        Returns:
            Average MSE loss across ensemble (float).
        """
        if not transitions:
            return 0.0

        # Convert text transitions to objects for JEPAEnsemble.train_step
        # which expects (state_obj, action_str, next_state_obj)
        obj_transitions = []
        for state_text, action_text, next_text in transitions:
            # Create minimal objects with the fields JEPAEnsemble expects
            state_obj = self._text_to_minimal_state(state_text)
            next_obj = self._text_to_minimal_state(next_text)
            obj_transitions.append((state_obj, action_text, next_obj))

        return self.jepa.train_step(obj_transitions)

    @staticmethod
    def _text_to_minimal_state(text: str):
        """Create a state-like object from a text representation.

        Parses "cwd: /sandbox | files: a.txt,b.txt | last_output: hello"
        format back into object attributes for JEPAEnsemble compatibility.
        """
        obj = type("obj", (object,), {})
        obj.cwd = "/sandbox"
        obj.files = []
        obj.last_output = ""

        for part in text.split("|"):
            part = part.strip()
            if part.startswith("cwd:"):
                obj.cwd = part[len("cwd:"):].strip()
            elif part.startswith("files:"):
                raw = part[len("files:"):].strip()
                obj.files = [f.strip() for f in raw.split(",") if f.strip()]
            elif part.startswith("last_output:"):
                obj.last_output = part[len("last_output:"):].strip()

        return obj
