"""Phase 2 PEDA episode runner with LearningModule integration."""

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from phase1.types import DriveWeights, ErrorVector, Experience
from phase1.world_model import LearningModule, EnsembleErrorComputer, WorldModel
from phase1.grid_env import Perception
from phase2.sandbox_env import BusyboxSandbox, SandboxState, generate_sandbox_candidates


class SandboxLearningModule(LearningModule):
    """LearningModule adapted for SandboxState (container sandbox).

    Overrides update() to use JSON-serialized state text instead of Perception.render
    which only works with GridState.

    Example::

        lm = SandboxLearningModule(world_model, error_computer, buffer_size=100, update_interval=50)
        # store_experience() accepts Experience with SandboxState natively
        lm.store_experience(Experience(state=sandbox_state, action="ls", next_state=next_state, error=err))
    """

    def update(self) -> None:
        """Batch LoRA fine-tune using SandboxState-aware text rendering.

        Handles both SandboxState (Phase 2) and GridState (Phase 1) experiences
        transparently via duck-typing.
        """
        if not self.should_update():
            return
        samples = self.buffer.sample_prioritized(batch_size=64)
        data = []
        for exp in samples:
            if hasattr(exp.state, "container_id"):
                # SandboxState path: serialize to JSON
                state_text = exp.state.to_json()
                action_name = exp.action if isinstance(exp.action, str) else exp.action.name
                next_state_text = exp.next_state.to_json()
            else:
                # GridState path (original Phase 1 behaviour)
                state_text = Perception.render(exp.state)
                action_name = exp.action.name
                next_state_text = str(exp.next_state.agent)
            data.append({
                "state_text": state_text,
                "action_name": action_name,
                "next_state_text": next_state_text,
                "exit_code": exp.exit_code,
                "summary": exp.summary,
            })
        self.world_model.lora_finetune(data, epochs=1, learning_rate=2e-4, batch_size=4)
        self.step_count += 1
        self.error_computer.save_checkpoint(self.step_count)
        self.buffer.clear()
        if self.saturation_detector.is_saturated():
            print("[SandboxLearningModule] Saturation detected; novelty boost applied next step.")


def run_peda_episode(
    sb: BusyboxSandbox,
    wm: WorldModel,
    error_computer: EnsembleErrorComputer,
    drive_system,
    learning_module: SandboxLearningModule,
    agent_fn: Callable[[SandboxState, List[str]], str],
    max_steps: int,
    task_id: str,
    start_cwd: Optional[str] = None,
) -> Tuple[List[Dict], SandboxState, Dict[str, Any]]:
    """Full PEDA loop for Phase 2 sandbox environments.

    Follows the same 7-step structure as Phase 1's run_episode():
      1. reset environment
      2. select action
      3. step environment
      4. compute error decomposition
      5. update drive system
      6. store experience in replay buffer
      7. auto-finetune when buffer is full

    Parameters
    ----------
    sb:
        BusyboxSandbox instance for sandbox execution.
    wm:
        WorldModel for prediction and action generation.
    error_computer:
        EnsembleErrorComputer for epistemic/aleatoric decomposition.
    drive_system:
        Drive system (e.g. HomeostaticDriveSystem) updated each step.
    learning_module:
        SandboxLearningModule wrapping the WorldModel for intermittent fine-tuning.
    agent_fn:
        Policy function ``fn(state, action_history) -> action_str``.
    max_steps:
        Maximum steps per episode.
    task_id:
        MICRO_TASKS id for victory detection.

    Returns
    -------
    (steps, final_state, metrics)
        steps — list of per-step record dicts.
        final_state — terminal SandboxState.
        metrics — summary dict with steps count, success flag, mean errors.
    """
    from phase2.tasks import MICRO_TASKS

    state = sb.reset(start_cwd=start_cwd)
    action_history: List[str] = []
    steps: List[Dict] = []
    epistemic_errors: List[float] = []
    aleatoric_errors: List[float] = []

    task_def = next((t for t in MICRO_TASKS if t["id"] == task_id), None)

    for step_i in range(max_steps):
        if state.game_over:
            break

        # Step 2: Select action
        action = agent_fn(state, action_history)
        action_str = action if isinstance(action, str) else action.name

        # Step 3: Execute action in sandbox
        next_state, reward, done = sb.step(state, action_str)

        # Step 4: Compute error decomposition
        error = error_computer.decompose_error(state, action, next_state)
        epistemic_errors.append(error.epistemic_error)
        aleatoric_errors.append(error.aleatoric_error)

        # Step 5: Update drive system
        drive_system.update(error, action, has_external_input=False, action_history=action_history)

        # Check task completion
        if task_def and task_def["check"](state, action_str, next_state):
            next_state.victory = True
            next_state.game_over = True
            done = True

        # Step 6: Store experience in replay buffer
        exit_code = next_state.last_exit_code
        summary = (
            f"action {action_str}: {next_state.last_output[:60]}"
            if next_state.last_output
            else action_str
        )

        learning_module.store_experience(
            Experience(
                state=state,
                action=action_str,
                next_state=next_state,
                error=error,
                exit_code=exit_code,
                summary=summary,
            )
        )

        # Step 7: Auto-finetune when buffer is ready
        if learning_module.should_update():
            learning_module.update()
            boost = learning_module.saturation_novelty_boost
            if boost > 0 and hasattr(drive_system, "current_terms"):
                drive_system.current_terms.novelty += boost

        record = {
            "agent_type": "peda",
            "task_id": task_id,
            "step": step_i,
            "cwd": next_state.cwd,
            "files": list(next_state.files),
            "action": action_str,
            "next_cwd": next_state.cwd,
            "next_files": list(next_state.files),
            "exit_code": exit_code,
            "output": next_state.last_output[:100],
            "step_count": next_state.step_count,
            "epistemic_error": error.epistemic_error,
            "aleatoric_error": error.aleatoric_error,
        }
        steps.append(record)
        action_history.append(action_str)
        state = next_state
        if done:
            break

    metrics = {
        "steps": len(steps),
        "success": state.victory if hasattr(state, "victory") else done,
        "mean_epistemic_error": sum(epistemic_errors) / len(epistemic_errors) if epistemic_errors else 0.0,
        "mean_aleatoric_error": sum(aleatoric_errors) / len(aleatoric_errors) if aleatoric_errors else 0.0,
    }
    return steps, state, metrics
