#!/usr/bin/env python3
# ruff: noqa: E402
"""Phase 4 Closed-Loop Self-Training: Experiment A.

Three conditions:
  - PEDA+Train:  PEDA with intermittent LoRA updates between blocks
  - PEDA+Freeze: PEDA with frozen adapter (no updates)
  - Pragmatic:   Pragmatic-only baseline

Each condition runs N blocks of M episodes (6 CWDs round-robin).
PEDA+Train updates the LoRA adapter after each block using all
transitions collected during that block.

Outputs:
  {output_dir}/{condition}/block_{N}.jsonl         — per-block episode summary
  {output_dir}/{condition}/block_{N}_records.jsonl  — per-block step records
  {output_dir}/training_curves.json                 — aggregate metrics all conditions

Usage:
  python scripts/phase4_closed_loop.py \
    --adapter-path checkpoints/phase2/sandbox_adapter_v2_full \
    --output-dir results/phase4_experiment_a \
    --num-blocks 4 --episodes-per-block 10 --max-steps 15

  # Run a single condition
  python scripts/phase4_closed_loop.py \
    --condition peda_train \
    --adapter-path checkpoints/phase2/sandbox_adapter_v2_full \
    --output-dir results/phase4_experiment_a/peda_train
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from phase1.types import DriveWeights, Experience
from phase1.world_model import WorldModel, EnsembleErrorComputer, Perception
from phase1.drive_system import ActionGenerator, HomeostaticDriveSystem
from phase2.sandbox_env import BusyboxSandbox, SandboxState, generate_sandbox_candidates
from phase2.run import SandboxLearningModule, run_peda_episode
from phase2.tasks import MICRO_TASKS

# ── Drive config (same Phase 2 grid-search top-1) ─────────────────
DRIVE_WEIGHTS = DriveWeights(
    curiosity=0.1, competence=2.0, boredom=0.1, novelty=2.0
)
PRAGMATIC_WEIGHT = 3.0

# ── CWD definitions ───────────────────────────────────────────────
KNOWN_CWDS = ["/sandbox", "/sandbox/data", "/sandbox/docs"]
UNKNOWN_CWDS = ["/sandbox/logs", "/sandbox/projects", "/sandbox/tmp"]
ALL_CWDS = KNOWN_CWDS + UNKNOWN_CWDS


# ── Helpers ────────────────────────────────────────────────────────

def _build_agent_components(
    wm: WorldModel,
    ckpt_dir: str,
    use_fast: bool = False,
    goal_predicate=None,
) -> Tuple[ActionGenerator, EnsembleErrorComputer, HomeostaticDriveSystem]:
    """Create ActionGenerator, ErrorComputer, and DriveSystem for a PEDA agent."""
    ec = EnsembleErrorComputer(wm)
    if use_fast:
        ec.checkpoints = []
    else:
        ckpt_path = Path(ckpt_dir)
        ec.checkpoints = sorted(ckpt_path.glob("checkpoint_epoch_*"))[:3]
    ds = HomeostaticDriveSystem(DRIVE_WEIGHTS)
    ag = ActionGenerator(
        wm, error_computer=ec, drive_system=ds,
        pragmatic_only=False,
        pragmatic_weight=PRAGMATIC_WEIGHT,
        max_candidates=5, horizon=1,
        goal_predicate=goal_predicate,
    )
    return ag, ec, ds


def _make_peda_agent_fn(
    ag: ActionGenerator,
) -> callable:
    """Return an agent function bound to the ActionGenerator."""
    def agent_fn(state, action_history):
        cands = generate_sandbox_candidates(state)
        return ag.select_action(state, action_history, cands)
    return agent_fn


def _make_pragmatic_agent_fn(
    wm: WorldModel,
    goal_predicate=None,
) -> callable:
    """Return a pragmatic-only agent function."""
    ec = EnsembleErrorComputer(wm)
    ec.checkpoints = []
    ds = HomeostaticDriveSystem(DRIVE_WEIGHTS)
    ag = ActionGenerator(
        wm, error_computer=ec, drive_system=ds,
        pragmatic_only=True,
        pragmatic_weight=PRAGMATIC_WEIGHT,
        max_candidates=5, horizon=1,
        goal_predicate=goal_predicate,
    )
    def agent_fn(state, action_history):
        cands = generate_sandbox_candidates(state)
        return ag.select_action(state, action_history, cands)
    return agent_fn


def _force_block_update(
    lm: SandboxLearningModule,
    block_idx: int,
    output_dir: Path,
) -> Optional[Dict[str, Any]]:
    """Force a full-buffer LoRA update, bypassing should_update guard.

    Returns loss info dict or None if buffer was empty.
    """
    if len(lm.buffer) == 0:
        print(f"  [Block {block_idx}] No experiences in buffer, skipping update.")
        return None

    n_experiences = len(lm.buffer)
    samples = lm.buffer.sample_prioritized(batch_size=min(64, n_experiences))
    data = []
    for exp in samples:
        if hasattr(exp.state, "container_id"):
            state_text = exp.state.to_json()
            action_name = exp.action if isinstance(exp.action, str) else exp.action.name
            next_state_text = exp.next_state.to_json()
        else:
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

    print(f"  [Block {block_idx}] Force update: {n_experiences} buffer, "
          f"{len(samples)} sampled, calling lora_finetune...", flush=True)

    lm.world_model.lora_finetune(data, epochs=1, learning_rate=2e-4, batch_size=4)
    lm.step_count += 1
    lm.error_computer.save_checkpoint(lm.step_count)

    # Save full LoRA adapter so it can be loaded for the next block
    block_adapter_dir = output_dir / f"block_{block_idx}_adapter"
    block_adapter_dir.mkdir(parents=True, exist_ok=True)
    if lm.world_model.mode == "llm" and lm.world_model.model is not None:
        lm.world_model.model.save_pretrained(str(block_adapter_dir))
        print(f"  [Block {block_idx}] Saved updated adapter to {block_adapter_dir}")

    lm.buffer.clear()

    return {
        "block": block_idx,
        "buffer_size": n_experiences,
        "samples_used": len(data),
    }


def _compute_episode_metrics(
    steps: List[Dict],
    success: bool,
    elapsed: float,
) -> Dict[str, Any]:
    """Derive summary metrics from step records."""
    n_steps = len(steps)
    dead_loop = 0
    if n_steps >= 2:
        repeats = sum(
            1 for i in range(1, n_steps)
            if steps[i]["action"] == steps[i - 1]["action"]
        )
        dead_loop = repeats / (n_steps - 1)
    mean_epistemic = (
        sum(s.get("epistemic_error", 0.0) for s in steps) / n_steps
        if n_steps else 0.0
    )
    mean_aleatoric = (
        sum(s.get("aleatoric_error", 0.0) for s in steps) / n_steps
        if n_steps else 0.0
    )
    return {
        "steps": n_steps,
        "success": int(success),
        "scr": 1.0 if success else 0.0,
        "fht": next((i + 1 for i, s in enumerate(steps)
                     if s.get("victory", False) or s.get("game_over", False)),
                    -1),
        "dead_loop_rate": dead_loop,
        "mean_epistemic_error": mean_epistemic,
        "mean_aleatoric_error": mean_aleatoric,
        "elapsed": elapsed,
    }


# ── Episode runners ───────────────────────────────────────────────

def run_peda_episode_with_lm(
    sb: BusyboxSandbox,
    wm: WorldModel,
    ag: ActionGenerator,
    ec: EnsembleErrorComputer,
    ds: HomeostaticDriveSystem,
    lm: SandboxLearningModule,
    max_steps: int,
    task_id: str,
    start_cwd: Optional[str] = None,
) -> Tuple[List[Dict], SandboxState, Dict[str, Any]]:
    """Run one PEDA episode using an already-constructed agent + LM.

    The LM uses a large update_interval so auto-update does NOT fire
    during the episode — transitions accumulate in the buffer.
    """
    agent_fn = _make_peda_agent_fn(ag)
    steps, final_state, metrics = run_peda_episode(
        sb, wm, ec, ds, lm, agent_fn, max_steps, task_id,
        start_cwd=start_cwd,
    )
    return steps, final_state, metrics


def run_pragmatic_episode(
    sb: BusyboxSandbox,
    wm: WorldModel,
    max_steps: int,
    task_id: str,
    start_cwd: Optional[str] = None,
) -> Tuple[List[Dict], SandboxState]:
    """Run one pragmatic episode (no LearningModule, no PEDA)."""
    task_def = next((t for t in MICRO_TASKS if t["id"] == task_id), None)
    goal_predicate = task_def["check"] if task_def else None
    agent_fn = _make_pragmatic_agent_fn(wm, goal_predicate=goal_predicate)

    state = sb.reset(start_cwd=start_cwd)
    steps = []
    action_history = []
    for step_i in range(max_steps):
        if state.game_over:
            break
        action = agent_fn(state, action_history)
        action_str = action if isinstance(action, str) else action.name
        next_state, reward, done = sb.step(state, action_str)

        # Check task completion
        if task_def and task_def["check"](state, action_str, next_state):
            next_state.victory = True
            next_state.game_over = True
            done = True

        steps.append({
            "agent_type": "pragmatic",
            "task_id": task_id,
            "step": step_i,
            "cwd": state.cwd,
            "files": list(state.files),
            "action": action_str,
            "next_cwd": next_state.cwd,
            "next_files": list(next_state.files),
            "exit_code": next_state.last_exit_code,
            "output": next_state.last_output[:100],
            "step_count": next_state.step_count,
            "victory": next_state.victory if hasattr(next_state, "victory") else done,
            "game_over": next_state.game_over,
        })
        action_history.append(action_str)
        state = next_state
        if done:
            break
    return steps, state


# ─── Condition runners ───────────────────────────────────────────

def run_peda_train_condition(
    args,
    wm: WorldModel,
    condition_dir: Path,
    task_def: Dict,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Run PEDA+Train: accumulates experiences, updates LoRA after each block."""
    ag, ec, ds = _build_agent_components(
        wm, args.adapter_path, use_fast=args.fast,
        goal_predicate=task_def["check"],
    )
    # Large buffer/interval prevents auto-update during episodes
    lm = SandboxLearningModule(
        wm, ec,
        buffer_size=10000,
        update_interval=10000,
    )

    sb = BusyboxSandbox()
    all_block_summaries: List[Dict] = []
    all_block_records: List[Dict] = []
    all_updates: List[Dict] = []

    for block_idx in range(args.num_blocks):
        block_episodes: List[Dict] = []
        block_records: List[Dict] = []

        print(f"\n{'=' * 60}", flush=True)
        print(f"[PEDA+Train] Block {block_idx + 1}/{args.num_blocks}", flush=True)
        print(f"{'=' * 60}", flush=True)

        for ep_idx in range(args.episodes_per_block):
            cwd = ALL_CWDS[ep_idx % len(ALL_CWDS)]
            seed = args.seed_offset + ep_idx
            random.seed(seed)

            print(f"  Episode {ep_idx + 1}/{args.episodes_per_block}  cwd={cwd}  seed={seed}",
                  flush=True)
            t0 = time.time()

            steps, final_state, _metrics = run_peda_episode_with_lm(
                sb, wm, ag, ec, ds, lm,
                args.max_steps, args.task,
                start_cwd=cwd,
            )

            elapsed = time.time() - t0
            success = (
                final_state.victory
                if hasattr(final_state, "victory") else False
            )
            ep_metrics = _compute_episode_metrics(steps, success, elapsed)

            summary_line = {
                "condition": "peda_train",
                "block": block_idx,
                "episode": ep_idx,
                "cwd": cwd,
                "task": args.task,
                **ep_metrics,
            }
            block_episodes.append(summary_line)
            block_records.extend(
                {**r, "block": block_idx, "episode": ep_idx, "condition": "peda_train"}
                for r in steps
            )

            print(f"    -> success={success}   steps={ep_metrics['steps']}   "
                  f"[{elapsed:.0f}s]", flush=True)

        # ── Block-level LoRA update (PEDA+Train only) ──
        update_info = _force_block_update(lm, block_idx, condition_dir)
        if update_info:
            all_updates.append(update_info)

        # Persist block output
        _write_block_outputs(condition_dir, block_idx, block_episodes, block_records)
        all_block_summaries.extend(block_episodes)
        all_block_records.extend(block_records)

        # Block summary
        successes = sum(1 for e in block_episodes if e["success"])
        avg_steps = sum(e["steps"] for e in block_episodes) / len(block_episodes)
        print(f"\n  [Block {block_idx}] {successes}/{len(block_episodes)} success, "
              f"avg steps={avg_steps:.1f}", flush=True)

    sb.close()
    return all_block_summaries, all_block_records, all_updates


def run_peda_freeze_condition(
    args,
    wm: WorldModel,
    condition_dir: Path,
    task_def: Dict,
) -> Tuple[List[Dict], List[Dict]]:
    """Run PEDA+Freeze: no LoRA updates between blocks."""
    ag, ec, ds = _build_agent_components(
        wm, args.adapter_path, use_fast=args.fast,
        goal_predicate=task_def["check"],
    )
    # Still create LM to track metrics, but never call update
    lm = SandboxLearningModule(
        wm, ec,
        buffer_size=10000,
        update_interval=10000,
    )

    sb = BusyboxSandbox()
    all_block_summaries: List[Dict] = []
    all_block_records: List[Dict] = []

    for block_idx in range(args.num_blocks):
        block_episodes: List[Dict] = []
        block_records: List[Dict] = []

        print(f"\n{'=' * 60}", flush=True)
        print(f"[PEDA+Freeze] Block {block_idx + 1}/{args.num_blocks}", flush=True)
        print(f"{'=' * 60}", flush=True)

        for ep_idx in range(args.episodes_per_block):
            cwd = ALL_CWDS[ep_idx % len(ALL_CWDS)]
            seed = args.seed_offset + ep_idx
            random.seed(seed)

            print(f"  Episode {ep_idx + 1}/{args.episodes_per_block}  cwd={cwd}  seed={seed}",
                  flush=True)
            t0 = time.time()

            steps, final_state, _metrics = run_peda_episode_with_lm(
                sb, wm, ag, ec, ds, lm,
                args.max_steps, args.task,
                start_cwd=cwd,
            )

            elapsed = time.time() - t0
            success = (
                final_state.victory
                if hasattr(final_state, "victory") else False
            )
            ep_metrics = _compute_episode_metrics(steps, success, elapsed)

            summary_line = {
                "condition": "peda_freeze",
                "block": block_idx,
                "episode": ep_idx,
                "cwd": cwd,
                "task": args.task,
                **ep_metrics,
            }
            block_episodes.append(summary_line)
            block_records.extend(
                {**r, "block": block_idx, "episode": ep_idx, "condition": "peda_freeze"}
                for r in steps
            )

            print(f"    -> success={success}   steps={ep_metrics['steps']}   "
                  f"[{elapsed:.0f}s]", flush=True)

        # Clear LM buffer so next block starts fresh (no update, just cleanup)
        lm.buffer.clear()

        _write_block_outputs(condition_dir, block_idx, block_episodes, block_records)
        all_block_summaries.extend(block_episodes)
        all_block_records.extend(block_records)

        successes = sum(1 for e in block_episodes if e["success"])
        avg_steps = sum(e["steps"] for e in block_episodes) / len(block_episodes)
        print(f"\n  [Block {block_idx}] {successes}/{len(block_episodes)} success, "
              f"avg steps={avg_steps:.1f}", flush=True)

    sb.close()
    return all_block_summaries, all_block_records


def run_pragmatic_condition(
    args,
    wm: WorldModel,
    condition_dir: Path,
    task_def: Dict,
) -> Tuple[List[Dict], List[Dict]]:
    """Run Pragmatic baseline: no PEDA, no adapter updates."""
    sb = BusyboxSandbox()
    all_block_summaries: List[Dict] = []
    all_block_records: List[Dict] = []

    for block_idx in range(args.num_blocks):
        block_episodes: List[Dict] = []
        block_records: List[Dict] = []

        print(f"\n{'=' * 60}", flush=True)
        print(f"[Pragmatic] Block {block_idx + 1}/{args.num_blocks}", flush=True)
        print(f"{'=' * 60}", flush=True)

        for ep_idx in range(args.episodes_per_block):
            cwd = ALL_CWDS[ep_idx % len(ALL_CWDS)]
            seed = args.seed_offset + ep_idx
            random.seed(seed)

            print(f"  Episode {ep_idx + 1}/{args.episodes_per_block}  cwd={cwd}  seed={seed}",
                  flush=True)
            t0 = time.time()

            steps, final_state = run_pragmatic_episode(
                sb, wm, args.max_steps, args.task,
                start_cwd=cwd,
            )

            elapsed = time.time() - t0
            success = (
                final_state.victory
                if hasattr(final_state, "victory") else False
            )
            ep_metrics = _compute_episode_metrics(steps, success, elapsed)

            summary_line = {
                "condition": "pragmatic",
                "block": block_idx,
                "episode": ep_idx,
                "cwd": cwd,
                "task": args.task,
                **ep_metrics,
            }
            block_episodes.append(summary_line)
            block_records.extend(
                {**r, "block": block_idx, "episode": ep_idx, "condition": "pragmatic"}
                for r in steps
            )

            print(f"    -> success={success}   steps={ep_metrics['steps']}   "
                  f"[{elapsed:.0f}s]", flush=True)

        _write_block_outputs(condition_dir, block_idx, block_episodes, block_records)
        all_block_summaries.extend(block_episodes)
        all_block_records.extend(block_records)

        successes = sum(1 for e in block_episodes if e["success"])
        avg_steps = sum(e["steps"] for e in block_episodes) / len(block_episodes)
        print(f"\n  [Block {block_idx}] {successes}/{len(block_episodes)} success, "
              f"avg steps={avg_steps:.1f}", flush=True)

    sb.close()
    return all_block_summaries, all_block_records


# ─── Output helpers ────────────────────────────────────────────────

def _write_block_outputs(
    condition_dir: Path,
    block_idx: int,
    episode_summaries: List[Dict],
    step_records: List[Dict],
) -> None:
    """Write per-block JSONL files."""
    summary_path = condition_dir / f"block_{block_idx}.jsonl"
    records_path = condition_dir / f"block_{block_idx}_records.jsonl"

    with open(summary_path, "w") as f:
        for ep in episode_summaries:
            f.write(json.dumps(ep) + "\n")
    with open(records_path, "w") as f:
        for rec in step_records:
            f.write(json.dumps(rec) + "\n")

    print(f"    Wrote {summary_path} ({len(episode_summaries)} episodes)")
    print(f"    Wrote {records_path} ({len(step_records)} records)")


def _write_training_curves(
    output_dir: Path,
    all_summaries: Dict[str, List[Dict]],
    all_update_info: Dict[str, List[Dict]],
) -> None:
    """Aggregate per-block metrics into training_curves.json."""
    curves = {
        name: _aggregate_block_metrics(summaries, updates)
        for name, summaries, updates in [
            ("peda_train", all_summaries.get("peda_train", []),
             all_update_info.get("peda_train", [])),
            ("peda_freeze", all_summaries.get("peda_freeze", []),
             []),
            ("pragmatic", all_summaries.get("pragmatic", []),
             []),
        ]
    }
    curves["config"] = {
        "num_blocks": max(
            (s["block"] for name in curves if name != "config"
             for s in curves[name].get("blocks", [])),
            default=0,
        ) + 1 if any(name != "config" for name in curves) else 0,
        "episodes_per_block": len([
            s for name in curves if name != "config"
            for b in curves[name].get("blocks", [])
            for s in b.get("episodes", [])
        ]) // max(
            (len(curves[n]["blocks"]) for n in curves if n != "config"),
            default=1,
        ) if any(name != "config" for name in curves) else 0,
        "max_steps": None,
    }
    path = output_dir / "training_curves.json"
    with open(path, "w") as f:
        json.dump(curves, f, indent=2)
    print(f"\nSaved training curves: {path}")


def _aggregate_block_metrics(
    summaries: List[Dict],
    update_info: List[Dict],
) -> Dict:
    """Group episode summaries by block and compute block aggregates."""
    by_block: Dict[int, List[Dict]] = {}
    for s in summaries:
        by_block.setdefault(s["block"], []).append(s)

    blocks = []
    for block_idx in sorted(by_block.keys()):
        eps = by_block[block_idx]
        n = len(eps)
        successes = sum(1 for e in eps if e["success"])
        blocks.append({
            "block": block_idx,
            "n_episodes": n,
            "n_success": successes,
            "success_rate": successes / n if n else 0.0,
            "mean_steps": sum(e["steps"] for e in eps) / n if n else 0.0,
            "mean_scr": sum(e["scr"] for e in eps) / n if n else 0.0,
            "mean_dead_loop_rate": sum(e["dead_loop_rate"] for e in eps) / n if n else 0.0,
            "mean_epistemic_error": sum(
                e.get("mean_epistemic_error", 0.0) for e in eps
            ) / n if n else 0.0,
            "mean_aleatoric_error": sum(
                e.get("mean_aleatoric_error", 0.0) for e in eps
            ) / n if n else 0.0,
            "per_cwd": _per_cwd_metrics(eps),
            "episodes": eps,
        })

    return {
        "blocks": blocks,
        "updates": update_info,
    }


def _per_cwd_metrics(episodes: List[Dict]) -> Dict[str, Dict]:
    """Break down metrics by CWD."""
    by_cwd: Dict[str, List[Dict]] = {}
    for ep in episodes:
        by_cwd.setdefault(ep["cwd"], []).append(ep)

    result = {}
    for cwd, eps in by_cwd.items():
        n = len(eps)
        successes = sum(1 for e in eps if e["success"])
        result[cwd] = {
            "n": n,
            "n_success": successes,
            "success_rate": successes / n if n else 0.0,
            "mean_steps": sum(e["steps"] for e in eps) / n if n else 0.0,
        }
    return result


# ─── Condition table ──────────────────────────────────────────────

CONDITION_RUNNERS = {
    "peda_train": run_peda_train_condition,
    "peda_freeze": run_peda_freeze_condition,
    "pragmatic": run_pragmatic_condition,
}
ALL_CONDITIONS = ["peda_train", "peda_freeze", "pragmatic"]


# ─── Main ─────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 4 Experiment A: Closed-Loop Self-Training"
    )
    parser.add_argument(
        "--condition",
        choices=ALL_CONDITIONS,
        default=None,
        help="Run a single condition (default: all three)",
    )
    parser.add_argument(
        "--adapter-path",
        default="checkpoints/phase2/sandbox_adapter_v2_full",
        help="Path to initial LoRA adapter",
    )
    parser.add_argument(
        "--output-dir",
        default="results/phase4_experiment_a",
        help="Output directory for results",
    )
    parser.add_argument(
        "--model",
        default=os.path.expanduser("~/models/Qwen2.5-0.5B-Instruct"),
        help="Base model path",
    )
    parser.add_argument(
        "--num-blocks", type=int, default=4,
        help="Number of training blocks",
    )
    parser.add_argument(
        "--episodes-per-block", type=int, default=10,
        help="Episodes per block",
    )
    parser.add_argument(
        "--max-steps", type=int, default=15,
        help="Maximum steps per episode",
    )
    parser.add_argument(
        "--task", default="read_hello",
        help="Task ID from MICRO_TASKS",
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="Skip ensemble checkpoint loading for faster runs (smoke test only)",
    )
    parser.add_argument(
        "--seed-offset", type=int, default=0,
        help="Offset for random seeds",
    )
    args = parser.parse_args()

    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    task_def = next((t for t in MICRO_TASKS if t["id"] == args.task), None)
    if task_def is None:
        print(f"Error: Unknown task '{args.task}'. Available: "
              f"{[t['id'] for t in MICRO_TASKS]}", flush=True)
        sys.exit(1)

    conditions_to_run = [args.condition] if args.condition else ALL_CONDITIONS

    print(f"{'=' * 60}", flush=True)
    print(f"Phase 4 Experiment A: Closed-Loop Self-Training", flush=True)
    print(f"  Conditions: {conditions_to_run}", flush=True)
    print(f"  Adapter: {args.adapter_path}", flush=True)
    print(f"  Blocks: {args.num_blocks} x {args.episodes_per_block} episodes", flush=True)
    print(f"  Max steps: {args.max_steps}", flush=True)
    print(f"  Task: {args.task}", flush=True)
    print(f"  Output: {output_dir}", flush=True)
    print(f"{'=' * 60}", flush=True)

    all_summaries: Dict[str, List[Dict]] = {}
    all_update_info: Dict[str, List[Dict]] = {}
    total_start = time.time()

    for condition_name in conditions_to_run:
        condition_dir = output_dir / condition_name
        condition_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'#' * 60}", flush=True)
        print(f"# Condition: {condition_name}", flush=True)
        print(f"{'#' * 60}", flush=True)

        # Load WorldModel — fresh load for each condition so PEDA+Train's
        # adapter updates don't leak into the next condition
        print(f"  Loading WorldModel...", flush=True)
        wm = WorldModel(args.model, adapter_path=args.adapter_path)
        if wm.mode == "stub":
            raise RuntimeError(
                f"WorldModel fell back to stub mode! Model at {args.model} "
                f"did not load correctly."
            )
        print(f"  WorldModel loaded (mode={wm.mode}).", flush=True)

        runner = CONDITION_RUNNERS[condition_name]
        result = runner(args, wm, condition_dir, task_def)

        if condition_name == "peda_train":
            summaries, records, updates = result
            all_update_info[condition_name] = updates
        else:
            summaries, records = result
            all_update_info[condition_name] = []

        all_summaries[condition_name] = summaries

        # Free WM memory
        del wm
        import gc; gc.collect()
        if hasattr(torch := __import__("torch", fromlist=["cuda"]), "cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()

    elapsed_total = time.time() - total_start
    _write_training_curves(output_dir, all_summaries, all_update_info)

    print(f"\n{'=' * 60}", flush=True)
    print(f"Experiment A complete! ({elapsed_total / 60:.1f} min)", flush=True)
    print(f"Results: {output_dir}", flush=True)
    print(f"{'=' * 60}", flush=True)


if __name__ == "__main__":
    main()
