#!/usr/bin/env python3
# ruff: noqa: E402
"""Phase 2 data collection: multi-baseline sandbox evaluation.

Usage:
    python scripts/phase2_collect_data.py --baseline peda --task read_note
    python scripts/phase2_collect_data.py --baseline random --task count_lines --max-steps 30
    python scripts/phase2_collect_data.py --all-baselines  # run all baselines in sequence
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from phase1.types import DriveWeights
from phase1.world_model import WorldModel, EnsembleErrorComputer
from phase1.drive_system import ActionGenerator, HomeostaticDriveSystem
from phase2.sandbox_env import BusyboxSandbox, generate_sandbox_candidates

# ── Drive config (grid-search top-1) ──────────────────────────────
DRIVE_WEIGHTS = DriveWeights(curiosity=0.1, competence=2.0, boredom=0.1, novelty=2.0)
PRAGMATIC_WEIGHT = 3.0

# ── Micro-tasks ────────────────────────────────────────────────────
def _goal_predicate_read_note(state, action, next_state) -> bool:
    return "secret key" in next_state.last_output or (action and "cat docs/note" in action)

def _goal_predicate_count_lines(state, action, next_state) -> bool:
    return "3" in next_state.last_output and "lines" in next_state.last_output

def _goal_predicate_hello(state, action, next_state) -> bool:
    return "hello" in next_state.last_output

def _goal_predicate_find_secret(state, action, next_state) -> bool:
    return "secret" in next_state.last_output.lower()

def _goal_predicate_create_file(state, action, next_state) -> bool:
    return "test_dir" in (next_state.files if hasattr(next_state, "files") else [])

MICRO_TASKS = [
    {"id": "read_note", "goal": "Read docs/note.txt", "check": _goal_predicate_read_note},
    {"id": "count_lines", "goal": "Count lines in data/lines.txt", "check": _goal_predicate_count_lines},
    {"id": "read_hello", "goal": "Read hello.txt", "check": _goal_predicate_hello},
    {"id": "find_secret", "goal": "Find files containing 'secret'", "check": _goal_predicate_find_secret},
    {"id": "create_file", "goal": "Create test_dir", "check": _goal_predicate_create_file},
]

# ── Baseline runners ──────────────────────────────────────────────

def _build_ag(wm, pragmatic_only=False, use_fast=False):
    ec = EnsembleErrorComputer(wm)
    ckpt_dir = Path("checkpoints/phase1_5/text_adapter_e4")
    if use_fast:
        ec.checkpoints = []  # smoke test: skip ensemble loading
    else:
        ec.checkpoints = sorted(ckpt_dir.glob("checkpoint_epoch_*"))[:3]
    ds = HomeostaticDriveSystem(DRIVE_WEIGHTS)
    ag = ActionGenerator(wm, error_computer=ec, drive_system=ds,
                         pragmatic_only=pragmatic_only,
                         pragmatic_weight=PRAGMATIC_WEIGHT,
                         max_candidates=3, horizon=1)
    return ag


def _run_agent(sb, state, agent_fn, max_steps: int, task_id: str, baseline: str):
    """Run an agent_fn(state, action_history, candidates) -> action for up to max_steps."""
    steps = []
    action_history = []
    for step_i in range(max_steps):
        t0 = time.time()
        action = agent_fn(state, action_history)
        t1 = time.time()
        action_str = action if isinstance(action, str) else action.name
        next_state, reward, done = sb.step(state, action_str)
        t2 = time.time()
        print(f"  [step {step_i}] select={t1-t0:.1f}s docker={t2-t1:.1f}s action={action_str}", flush=True)

        record = {
            "agent_type": baseline,
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
        }
        steps.append(record)
        state = next_state
        action_history.append(action_str)
        if done:
            break
    return steps, state


def run_peda(sb, wm, max_steps, task_id, use_fast=False):
    ag = _build_ag(wm, pragmatic_only=False, use_fast=use_fast)
    state = sb.reset()
    def agent_fn(state, action_history):
        cands = generate_sandbox_candidates(state)
        # select_action expects List[Action]; cands are strings
        return ag.select_action(state, action_history, cands)
    return _run_agent(sb, state, agent_fn, max_steps, task_id, "peda")


def run_pragmatic(sb, wm, max_steps, task_id, use_fast=False):
    ag = _build_ag(wm, pragmatic_only=True, use_fast=use_fast)
    state = sb.reset()
    def agent_fn(state, action_history):
        cands = generate_sandbox_candidates(state)
        return ag.select_action(state, action_history, cands)
    return _run_agent(sb, state, agent_fn, max_steps, task_id, "pragmatic")


def run_random(sb, wm, max_steps, task_id):
    rng = random.Random(42)
    state = sb.reset()
    def agent_fn(state, action_history):
        cands = generate_sandbox_candidates(state)
        return rng.choice(cands) if cands else "ls"
    return _run_agent(sb, state, agent_fn, max_steps, task_id, "random")


def run_heuristic(sb, wm, max_steps, task_id):
    """Random + boredom penalty: avoid same action >2 times in a row."""
    rng = random.Random(42)
    state = sb.reset()
    last_actions = []
    def agent_fn(state, action_history):
        cands = generate_sandbox_candidates(state)
        if not cands:
            return "ls"
        # Count repeats
        action_counts = {}
        for a in action_history[-5:]:
            action_counts[a] = action_counts.get(a, 0) + 1
        # Filter actions repeated >=3 times recently
        fresh = [c for c in cands if action_counts.get(c, 0) < 3]
        return rng.choice(fresh) if fresh else rng.choice(cands)
    return _run_agent(sb, state, agent_fn, max_steps, task_id, "heuristic")


def run_prompt(sb, wm, max_steps, task_id):
    """LLM-prompted: ask Qwen2.5 directly what command to run."""
    state = sb.reset()
    def agent_fn(state, action_history):
        state_text = state.to_json()
        messages = [
            {"role": "system", "content": "You are a helpful assistant in a Linux sandbox. Reply with exactly ONE command."},
            {"role": "user", "content": (
                f"Current state: {state_text}\n"
                f"Task: {task_id}\n"
                "Generate ONE Linux command from: ls, cd, cat, echo, mkdir, touch, pwd, wc, head, tail, grep\n"
                "Reply with ONLY the command, no explanation."
            )},
        ]
        try:
            prompt = wm.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = wm.tokenizer(prompt, return_tensors="pt").to(wm.device)
            out = wm.model.generate(**inputs, max_new_tokens=20, do_sample=False,
                                     pad_token_id=wm.tokenizer.pad_token_id)
            cmd = wm.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            cmd = cmd.split("\n")[0].strip().strip('"\'`')
            # If the model emits rambling text, pick the first whitelisted word from it.
            WHITELIST = {"ls", "cd", "cat", "echo", "mkdir", "touch", "pwd", "wc", "head", "tail", "grep"}
            words = cmd.lower().split()
            cmd = next((w for w in words if w in WHITELIST), cmd if words else "ls")
        except Exception:
            cmd = "ls"
        return cmd
    return _run_agent(sb, state, agent_fn, max_steps, task_id, "prompt")


BASELINE_FNS = {
    "peda": lambda sb, wm, ms, tk, **kw: run_peda(sb, wm, ms, tk, use_fast=kw.get("use_fast", False)),
    "pragmatic": lambda sb, wm, ms, tk, **kw: run_pragmatic(sb, wm, ms, tk, use_fast=kw.get("use_fast", False)),
    "random": run_random,
    "heuristic": run_heuristic,
    "prompt": run_prompt,
}


# ── Metrics ────────────────────────────────────────────────────────

def compute_metrics(steps, task_id):
    """Compute FHT, SCR, Dead-loop Rate from step records."""
    fht = None
    task = next((t for t in MICRO_TASKS if t["id"] == task_id), None)
    if task:
        for rec in steps:
            fake_ns = type("obj", (object,), {
                "last_output": rec.get("output", ""),
                "last_exit_code": rec.get("exit_code", 0),
                "files": rec.get("next_files", []),
                "cwd": rec.get("next_cwd", ""),
            })()
            if task["check"](fake_ns, rec["action"], fake_ns):
                fht = rec["step"]
                break
    visited = set()
    for rec in steps:
        visited.add(f"{rec['cwd']}|{tuple(rec['files'])}")
    scr = len(visited) / max(len(steps), 1)
    loops = 0
    for i in range(2, len(steps)):
        if steps[i]["action"] == steps[i-1]["action"] == steps[i-2]["action"]:
            loops += 1
    dead_loop_rate = loops / max(len(steps), 1)
    return {"fht": fht, "scr": round(scr, 3), "dead_loop_rate": round(dead_loop_rate, 3),
            "steps": len(steps)}


# ── Main ───────────────────────────────────────────────────────────

def run_one(sb, wm, baseline_name, task_id, max_steps, use_fast=False):
    fn = BASELINE_FNS[baseline_name]
    if baseline_name in ("peda", "pragmatic"):
        steps, final_state = fn(sb, wm, max_steps, task_id, use_fast=use_fast)
    else:
        steps, final_state = fn(sb, wm, max_steps, task_id)
    metrics = compute_metrics(steps, task_id)
    return {"baseline": baseline_name, "task": task_id,
            "steps_count": len(steps), "metrics": metrics, "records": steps}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", choices=list(BASELINE_FNS), default=None)
    parser.add_argument("--task", choices=[t["id"] for t in MICRO_TASKS], default="read_note")
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--model", default=os.path.expanduser("~/models/Qwen2.5-0.5B-Instruct"), help="local or HF model path")
    parser.add_argument("--adapter-path", default="checkpoints/phase1_5/text_adapter_e4", help="LoRA adapter path")
    parser.add_argument("--fast", action="store_true", help="skip ensemble checkpoints for smoke test")
    parser.add_argument("--output", default="results/phase2_data.jsonl", help="JSONL output path")
    parser.add_argument("--all-baselines", action="store_true", help="run all baselines for the selected task")
    parser.add_argument("--all-tasks", action="store_true", help="run all tasks for the selected baseline")
    args = parser.parse_args()

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    baselines = list(BASELINE_FNS) if args.all_baselines else [args.baseline]
    tasks = [t["id"] for t in MICRO_TASKS] if args.all_tasks else [args.task]
    if not args.all_baselines and not args.baseline:
        parser.error("Must specify --baseline or --all-baselines")

    wm = WorldModel(args.model, adapter_path=args.adapter_path)
    if wm.mode == "stub" and any(bl in ("peda", "pragmatic", "prompt") for bl in baselines):
        raise RuntimeError(
            f"WorldModel fell back to stub mode. The model at {args.model} did not load correctly. "
            "Verify the directory contains real weights (not LFS pointers) and rerun."
        )
    sb = BusyboxSandbox()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_results = []
    for bl in baselines:
        for tk in tasks:
            print(f"[phase2] {bl}/{tk} (max_steps={args.max_steps}) ...", flush=True)
            t0 = time.time()
            result = run_one(sb, wm, bl, tk, args.max_steps, use_fast=args.fast)
            elapsed = time.time() - t0
            m = result["metrics"]
            print(f"  -> steps={m['steps']} fht={m['fht']} scr={m['scr']} dl={m['dead_loop_rate']} [{elapsed:.0f}s]", flush=True)
            all_results.append(result)
            # Write incrementally so partial results survive crashes/timeouts
            line = {k: result[k] for k in ["baseline", "task", "steps_count", "metrics", "records"]}
            with open(output_path, "a") as f:
                f.write(json.dumps(line) + "\n")

    print(f"[phase2] Summary saved to {output_path}", flush=True)

    # Summary table
    print()
    print("=" * 70)
    print("Phase 2 Multi-Baseline Summary")
    print("=" * 70)
    print(f"{'Baseline':<12} {'Task':<16} {'Steps':<6} {'FHT':<4} {'SCR':<6} {'DL Rate':<8}")
    print("-" * 70)
    for r in all_results:
        m = r["metrics"]
        fht_str = str(m["fht"]) if m["fht"] is not None else "-"
        print(f"{r['baseline']:<12} {r['task']:<16} {m['steps']:<6} {fht_str:<4} {m['scr']:<6} {m['dead_loop_rate']:<8}")
    print()

    sb.close()


if __name__ == "__main__":
    main()
