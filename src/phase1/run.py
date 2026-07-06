"""Main PEDA loop, metrics, and random baseline for Phase 1."""

import random
from typing import Any, Dict, List, Tuple

from phase1.drive_system import ActionGenerator, HomeostaticDriveSystem
from phase1.grid_env import GridWorld
from phase1.types import Action, Experience, GridState, PredictedState
from phase1.world_model import EnsembleErrorComputer, LearningModule, WorldModel


def run_episode(
    env: GridWorld,
    world_model: WorldModel,
    error_computer: EnsembleErrorComputer,
    drive_system: HomeostaticDriveSystem,
    learning_module: LearningModule,
    action_generator: ActionGenerator,
    seed: int = 0,
) -> Tuple[List[GridState], List[PredictedState], List[Action], Dict[str, Any]]:
    state = env.reset(seed=seed)
    action_history: List[Action] = []
    predictions: List[PredictedState] = []
    trajectory: List[GridState] = [state]
    total_reward = 0.0
    epistemic_errors: list[float] = []
    aleatoric_errors: list[float] = []

    done = False
    while not done:
        candidates = GridWorld.all_actions()
        action = action_generator.select_action(state, action_history, candidates)
        predicted = world_model.predict(state, action)
        predictions.append(predicted)

        next_state, reward, done = env.step(state, action)
        total_reward += reward

        wall = next_state.agent == state.agent
        reached_goal = next_state.agent == state.goal
        exit_code = 2 if reached_goal else (1 if wall else 0)
        summary = f"agent moved {action.name.lower()}"
        if wall:
            summary = f"agent hit wall/obstacle with {action.name.lower()}"
        elif reached_goal:
            summary = "agent reached goal"

        error = error_computer.decompose_error(state, action, next_state)
        epistemic_errors.append(error.epistemic_error)
        aleatoric_errors.append(error.aleatoric_error)
        drive_system.update(error, action, has_external_input=False, action_history=action_history)

        learning_module.store_experience(
            Experience(
                state=state,
                action=action,
                next_state=next_state,
                error=error,
                exit_code=exit_code,
                summary=summary,
            )
        )
        if learning_module.should_update():
            learning_module.update()
            # Apply saturation boost to the drive system's current novelty term.
            boost = learning_module.saturation_novelty_boost
            if boost > 0:
                drive_system.current_terms.novelty += boost

        action_history.append(action)
        trajectory.append(next_state)
        state = next_state

    metrics = {
        "steps": state.step,
        "success": state.agent == state.goal,
        "reward": total_reward,
        "mean_epistemic_error": sum(epistemic_errors) / len(epistemic_errors) if epistemic_errors else 0.0,
        "mean_aleatoric_error": sum(aleatoric_errors) / len(aleatoric_errors) if aleatoric_errors else 0.0,
    }
    return trajectory, predictions, action_history, metrics


def next_state_accuracy(
    predictions: List[PredictedState], actuals: List[GridState]
) -> float:
    if not predictions or not actuals:
        return 0.0
    matches = sum(
        1
        for pred, actual in zip(predictions, actuals)
        if pred.level2_next_agent == actual.agent
    )
    return matches / min(len(predictions), len(actuals))


def completion_rate_at_horizon(trajectories: List[List[GridState]], horizon: int) -> float:
    if not trajectories:
        return 0.0
    successes = 0
    for traj in trajectories:
        for i, state in enumerate(traj[: horizon + 1]):
            if state.agent == state.goal:
                successes += 1
                break
    return successes / len(trajectories)


def steps_to_goal_ratio(drive_steps: float, random_steps: float) -> float:
    if random_steps <= 0:
        return 0.0
    return drive_steps / random_steps


def revisit_rate(states: List[GridState]) -> float:
    if not states:
        return 0.0
    positions = [s.agent for s in states]
    unique = len(set(positions))
    repeated = len(positions) - unique
    return repeated / len(positions)


def _run_random_episode(env: GridWorld, seed: int) -> Tuple[List[GridState], bool]:
    state = env.reset(seed=seed)
    trajectory = [state]
    done = False
    while not done:
        action = random.choice(GridWorld.all_actions())
        state, _, done = env.step(state, action)
        trajectory.append(state)
    return trajectory, state.agent == state.goal


def random_baseline(env: GridWorld, n: int = 100) -> Dict[str, Any]:
    random.seed(42)
    trajectories = []
    successes = 0
    total_steps = 0
    total_revisits = 0.0
    for i in range(n):
        traj, success = _run_random_episode(env, seed=i)
        trajectories.append(traj)
        if success:
            successes += 1
        total_steps += traj[-1].step
        total_revisits += revisit_rate(traj)

    return {
        "n": n,
        "mean_steps": total_steps / n,
        "success_rate": successes / n,
        "mean_revisit_rate": total_revisits / n,
        "completion_5": completion_rate_at_horizon(trajectories, 5),
        "completion_10": completion_rate_at_horizon(trajectories, 10),
        "completion_20": completion_rate_at_horizon(trajectories, 20),
    }


def aggregate_metrics(
    trajectories: List[List[GridState]],
    predictions: List[List[PredictedState]],
    action_histories: List[List[Action]],
) -> Dict[str, Any]:
    """Aggregate metrics across multiple episodes."""
    successes = sum(1 for traj in trajectories if traj[-1].agent == traj[-1].goal)
    total_steps = sum(traj[-1].step for traj in trajectories)
    mean_steps = total_steps / len(trajectories) if trajectories else 0.0

    all_predictions = [p for ep in predictions for p in ep]
    all_actuals = [s for traj in trajectories for s in traj[1:]]
    g1 = next_state_accuracy(all_predictions, all_actuals)

    revisit = sum(revisit_rate(traj) for traj in trajectories) / len(trajectories)
    completion_20 = completion_rate_at_horizon(trajectories, 20)

    return {
        "episodes": len(trajectories),
        "success_rate": successes / len(trajectories) if trajectories else 0.0,
        "mean_steps": mean_steps,
        "next_state_accuracy": g1,
        "revisit_rate": revisit,
        "completion_20": completion_20,
    }
