"""Phase 9 hierarchical horizon: episode loop driver (plan §4).

Goal re-evaluation every T_reeval steps (or on goal completion), with
hysteresis tau. Logs StepRecords plus goal-level prediction errors
PE_goal = |G_pred - realized| / max(G_pred, 1) per goal kind.
"""

from __future__ import annotations

from typing import List, Optional

from .executor import StepRecord
from .goals import Goal


def _make_record(
    state_before,
    action,
    state_after,
    cell_after,
    content_new,
    item_acquired,
    door_unlocked,
    env,
) -> StepRecord:
    return StepRecord(
        state_before=state_before,
        action=action,
        state_after=state_after,
        cell_after=cell_after,
        content_new=content_new,
        item_acquired=item_acquired,
        door_unlocked=door_unlocked,
        goal_reached=bool(state_after.goal_reached),
        items_seen=list(state_after.visible_items or []),
    )


def _item_acquired(state_before, state_after) -> Optional[str]:
    for it in state_after.inventory or ():
        if it not in (state_before.inventory or ()):
            return it
    return None


def _goal_complete(goal: Goal, rec: StepRecord, planner) -> bool:
    """True when the commitment's purpose is served.

    nav: the frontier's unvisited neighbors are exhausted (the agent has
         entered the unknown region the goal committed to).
    search: the key is in the inventory.
    acquire: the door was unlocked / the item taken.
    """
    if goal.kind == "nav":
        return goal.target is not None and not planner.is_frontier(goal.target)
    if goal.kind == "search":
        return goal.item is not None and goal.item in rec.state_after.inventory
    if goal.kind == "acquire":
        if goal.door_edge is not None:
            return rec.door_unlocked is not None
        return goal.item is not None and goal.item in rec.state_after.inventory
    return False


def _goal_entry(goal, g_pred, realized, steps, outcome) -> dict:
    pe = None
    if g_pred is not None:
        pe = abs(g_pred - realized) / max(g_pred, 1)
    return {
        "kind": goal.kind,
        "target": goal.target,
        "item": goal.item,
        "door_edge": goal.door_edge,
        "g_pred": g_pred,
        "realized": realized,
        "steps": steps,
        "outcome": outcome,
        "pe": round(pe, 4) if pe is not None else None,
    }


def run_layered_episode(
    env,
    planner,
    executor,
    H_plan: int,
    lam: float,
    T_reeval: Optional[int],
    max_steps: int,
    tau: float = 0.15,
):
    """Run one hierarchical episode (plan §4 loop).

    Args:
        env: GridMazeEnv (reset internally).
        planner: HighLevelPlanner (owns visit map / lock state).
        executor: LayeredExecutor (routes goals to BFS / novelty policies).
        H_plan: planner epistemic horizon (20-100).
        lam: travel-tax coefficient.
        T_reeval: re-evaluation period in steps; None = open-loop planner.
        max_steps: episode step cap.
        tau: hysteresis margin for goal switching.

    Returns:
        (records, goal_log, final_state):
          records  — list[StepRecord] (one per executed step)
          goal_log — list[dict] with G_pred / realized / PE_goal per goal
          final_state — GridState after the episode
    """
    env.reset()
    start_cell = (env.x, env.y)
    seen_cells = {start_cell}
    start_state = env._get_state()
    planner.seed_observation(start_cell, list(start_state.visible_items or []))

    records: List[StepRecord] = []
    goal_log: List[dict] = []

    goal = planner.select(H_plan, lam)
    if goal is None:
        return records, goal_log, env._get_state()
    g_pred = planner.predicted_gain()
    realized = 0
    committed_step = 0

    for t in range(max_steps):
        state = env._get_state()
        # The commitment stays the frontier goal; the planner refines it into
        # the executable next step (entry into the frontier's best unvisited
        # neighbor) when the agent is already at the frontier.
        waypoint = planner.refine(goal, H_plan)
        acts = executor.plan(state, waypoint, 1)
        if not acts:
            # BFS could not reach the goal — a locked door is in the way.
            # §4 blockage rule: escalate to the matching key goal.
            new_goal = planner.escalate(H_plan, lam)
            if new_goal is None or new_goal == goal:
                action = "look"
            else:
                goal_log.append(
                    _goal_entry(
                        goal, g_pred, realized, t - committed_step + 1, "blocked"
                    )
                )
                executor.on_goal_update(new_goal)
                goal = new_goal
                g_pred = planner.predicted_gain()
                realized = 0
                committed_step = t + 1
                state = env._get_state()
                action = executor.step(state, goal)
        else:
            action = acts[0]

        prev_cell = (state.x, state.y)
        locked_before = set(env._locked_doors)
        obs, next_state, done = env.step(action)

        cell_after = (next_state.x, next_state.y)
        content_new = cell_after not in seen_cells
        seen_cells.add(cell_after)
        unlocked = locked_before - set(env._locked_doors)
        door_unlocked = (
            env._key_for_door.get(next(iter(unlocked))) if unlocked else None
        )
        rec = _make_record(
            state,
            action,
            next_state,
            cell_after,
            content_new,
            _item_acquired(state, next_state),
            door_unlocked,
            env,
        )
        records.append(rec)
        planner.update(rec)

        moved = (prev_cell, cell_after) if cell_after != prev_cell else None
        success = moved is not None or bool(next_state.goal_reached)
        executor.observe(state, action, success, moved=moved)
        if content_new:
            realized += 1

        goal_done = _goal_complete(goal, rec, planner)
        if goal_done:
            goal_log.append(
                _goal_entry(
                    goal, g_pred, realized, t - committed_step + 1, "complete"
                )
            )
            new_goal = planner.select(H_plan, lam)
            if new_goal is None:
                break
            executor.on_goal_update(new_goal)
            goal = new_goal
            g_pred = planner.predicted_gain()
            realized = 0
            committed_step = t + 1
        elif T_reeval is not None and (t + 1) % T_reeval == 0:
            new_goal = planner.re_evaluate(H_plan, lam, tau)
            if new_goal is None:
                break
            if new_goal != goal:
                goal_log.append(
                    _goal_entry(
                        goal, g_pred, realized, t - committed_step + 1, "switched"
                    )
                )
                executor.on_goal_update(new_goal)
                goal = new_goal
                g_pred = planner.predicted_gain()
                realized = 0
                committed_step = t + 1

        if done or next_state.goal_reached:
            break

    return records, goal_log, env._get_state()
