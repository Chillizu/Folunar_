"""Discriminator-driven agent loop for Phase 9 FF-HG-5 (agent-level gate).

Subclasses src.phase8.count_driven_agent.Phase8Runner — imports the proven
count-driven loop and swaps ONLY the explorer, adding per-step discriminator
learning. Phase 8 files are never modified.

Selection (DiscriminatorExplorer, plan plan-hypothesis-generator.md §3.4):
    score(s, a) = alpha * uncertainty_D(s, a) + (1 - alpha) * count_novelty(s, a)
      alpha = 0.5 (plan default): discriminator AUGMENTS count novelty (blend)
      alpha = 1.0 (exploratory):  discriminator REPLACES count novelty

Learning: STRIPSDiscriminator.learn_from_step per executed transition, with
ground-truth outcome predicates extracted from the executed transition. The
discriminator's prediction error (hamming) is the exploration-relevant signal
measured per step and logged.

Baseline for the gate is the unmodified Phase8Runner (pure count novelty).

PHASE9_PLAN.md FF-HG-5: "Agent-level <= count baseline (post-MVP) -> DEAD".
Operationalization follows Direction-1 M3 spec (20 eps x 3 tasks, 10pp band):
    PE completion% >= count completion% - 10pp -> PASS, else FAIL/DEAD.
"""

import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Make `src` importable when run as `python -m src.phase9.agent`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phase2.sandbox_env import BusyboxSandbox, generate_sandbox_candidates
from phase8.count_driven_agent import Phase8Runner
from phase9.discriminator import STRIPSDiscriminator
from phase9.explorer import DiscriminatorExplorer
from phase9.types import OutcomePredicates, Verdict


class DiscriminatorRunner(Phase8Runner):
    """Phase8Runner with discriminator-driven exploration + per-step learning.

    Differs from the count baseline (Phase8Runner) in exactly three places:
      1. explorer = DiscriminatorExplorer(alpha) instead of NoveltyExplorer
         (success-cache replay preserved inside the explorer).
      2. per executed step: discriminator.predict BEFORE selection is scored
         internally by the explorer; we re-predict after execution to record
         the verdict, and learn_from_step with ground-truth predicates.
      3. per-step records (uncertainty, count novelty, score, error) are
         logged for the FF-HG-5 analysis (WATCHDOG D4 per-episode data).

    No JEPA training (irrelevant to this gate; kept False like the baseline).
    """

    def __init__(self, docker_image: str = "peda-sandbox:v4",
                 task_id: str = "read_hello",
                 alpha: float = 0.5,
                 discriminator: Optional[STRIPSDiscriminator] = None):
        super().__init__(docker_image=docker_image, task_id=task_id,
                         model_path=None, train_jepa=False, device="cpu", ci=False)
        self.alpha = alpha
        self.discriminator = discriminator if discriminator is not None else STRIPSDiscriminator()
        self.explorer = DiscriminatorExplorer(alpha=alpha,
                                              discriminator=self.discriminator)

    # ── Episode (full loop; identical structure to Phase8Runner.run_episode) ──

    def run_episode(self, max_steps: int = 10) -> Dict[str, Any]:
        """Run one episode; returns a dict (per-episode JSONL record)."""
        ep = len(self.results)
        result: Dict[str, Any] = {
            "episode": ep, "success": False, "steps": 0,
            "actions": [], "buffer_size": 0, "step_records": [],
        }

        state = self.sandbox.reset(seed=ep, start_cwd=self._start_cwd)
        self.explorer.reset_episode()

        for t in range(max_steps):
            # 1. Perception — candidate actions from current state
            candidates = generate_sandbox_candidates(state)
            if not candidates:
                candidates = ["ls", "pwd"]

            # 2. Action Generator — discriminator-blend selection (count term
            #    inside DiscriminatorExplorer keeps the proven signal in play)
            action = self.explorer.select_action(state, candidates, result["actions"])

            # 3. Pre-execution verdict (uncertainty already scored internally)
            verdict = self.discriminator.predict(state, action)

            # 4. Action Executor
            next_state, reward, done = self.sandbox.step(state, action)

            # 5. Goal check (same task.check contract as phase8)
            success = False
            check_fn = self.task.get("check")
            if check_fn is not None:
                try:
                    if check_fn(state, action, next_state):
                        success = True
                except Exception:
                    pass

            # 6. Ground-truth predicates + prediction error (the HG signal)
            gt = OutcomePredicates.from_transition(state, action, next_state)
            error = OutcomePredicates.hamming(verdict.predicates, gt)
            count_novelty = self.explorer.count_explorer.novelty_bonus(state, action)
            score = self.explorer.score(state, action)

            # 7. Feedback: counts + success cache + error buffer (explorer),
            #    STRIPS schema/predicate learning (discriminator)
            self.explorer.observe(state, action, verdict, success)
            try:
                self.discriminator.learn_from_step(state, action, next_state,
                                                   success, ground_truth=gt)
            except Exception:
                pass

            # 8. Record step
            result["actions"].append(action)
            result["step_records"].append({
                "step": t,
                "action": action,
                "uncertainty": round(verdict.uncertainty, 4),
                "confidence": round(verdict.confidence, 4),
                "count_novelty": round(count_novelty, 4),
                "score": round(score, 4),
                "predicted": verdict.predicates.to_dict(),
                "gt": gt.to_dict(),
                "error": round(error, 4),
                "exit_ok": gt.exit_ok,
            })

            if success:
                result["success"] = True
                break
            if done:
                break

            state = next_state

        result["steps"] = len(result["actions"])
        result["buffer_size"] = len(self.buffer)
        self.results.append(result)
        return result

    # ── Multi-episode run ───────────────────────────────

    def run(self, num_episodes: int = 10, max_steps: int = 10) -> List[dict]:
        for ep in range(num_episodes):
            result = self.run_episode(max_steps)
            status = "OK" if result["success"] else "FAIL"
            print(f"  Episode {ep}: {status} in {result['steps']} steps "
                  f"({result['buffer_size']} transitions)", flush=True)
        return [r for r in self.results]
