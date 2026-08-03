#!/usr/bin/env python3
"""Phase 9 sandbox-hh (E1): two-layer open-loop agent — sandbox migration run.

Runs the frontier-goal layered agent over the 9 canonical Phase 8 tasks,
lambda in {0, 0.5} (45 episodes per arm, 90 total), and writes per-episode
JSONL artifacts (WATCHDOG D4: meta header with git commit + one line per
episode) to results/phase9_sbh_{lam0,lam05}.jsonl.

Usage:
    source venv/bin/activate
    PYTHONPATH=src python3 scripts/phase9_sandbox_hh.py            # both arms
    PYTHONPATH=src python3 scripts/phase9_sandbox_hh.py --lam 0    # one arm
    # FF-SBH-3 (R1 empty-dir re-selection): write to r1-prefixed files
    PYTHONPATH=src python3 scripts/phase9_sandbox_hh.py --out-prefix phase9_sbh_r1
    # FF-GEN-1 (generalization): new v5 image + 8 new tasks, three arms
    # (flat count baseline / SBH lambda=0 / SBH lambda=0.5), 40 episodes each:
    PYTHONPATH=src python3 scripts/phase9_sandbox_hh.py --gen1
"""
import argparse
import json
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from phase9.sandbox_hh.runner import SandboxHHRunner, TASKS  # noqa: E402
from phase9.gen_tasks import GEN_TASK_IMAGES  # noqa: E402  (FF-GEN-1 task data)

NUM_EPISODES = 5
MAX_STEPS = 10

DENSITY_DEF = (
    "unvisited_density(d) = |unvisited Phase-8 verb x file candidates at d| / "
    "|all verb x file candidates at d|; candidates = cat/head -n 5/wc -l per known "
    "text file + cd per known subdir; unvisited = zero count in explorer "
    "state_action_counts under state_hash(d). J(d) = density - lam*dist(cwd,d), "
    "dist = BFS cd-steps in the known dir graph; unreachable dirs excluded. "
    "Goal selected at episode start; re-selected when the local frontier at the "
    "current dir is exhausted, and (R1, FF-SBH-3) right after a cd into a dir "
    "with no readable text files; a textless cwd is never itself selected as "
    "goal (failure analysis T2/R1)."
)


def build_meta_gen(arm: str, lam, episodes: int,
                   max_steps: int = MAX_STEPS,
                   experiment: str = "sandbox-hh-gen1") -> dict:
    """FF-GEN-1/FF-CEIL-1 meta header (WATCHDOG D4: git commit + environment).

    arm: 'flat' | 'sbh_lam0' | 'sbh_lam05'. lam: None for flat.
    max_steps: episode budget (FF-CEIL-1 extension, default 10).
    experiment: 'sandbox-hh-gen1' (default) | 'sandbox-hh-ceil1'.
    """
    commit = subprocess.run(["git", "rev-parse", "HEAD"],
                            capture_output=True, text=True).stdout.strip() or "unknown"
    if arm == "flat":
        model = "count-based novelty (Phase 8 quick-win config, no high layer)"
        low_layer = "Phase8Runner/Phase8Explorer (byte-identical to Phase 8)"
    else:
        model = ("count-based novelty (low) + frontier-goal density J (high), "
                 "no learned model")
        low_layer = ("generate_phase8_candidates + Phase8Explorer "
                     "(byte-identical to Phase 8)")
    return {
        "phase": 9,
        "experiment": experiment,
        "arm": arm,
        "lam": lam,
        "commit": commit,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "cpu_or_gpu": "cpu",
        "model": model,
        "sandbox_image": "peda-sandbox:v5",
        "max_steps": max_steps,
        "episodes_per_task": NUM_EPISODES,
        "tasks": [tid for tid, _ in GEN_TASK_IMAGES],
        "total_episodes": episodes,
        "density_definition": DENSITY_DEF,
        "low_layer": low_layer,
        "r1_fix": "FF-SBH-3: after cd into a dir with no readable text files, force "
                  "high-layer goal re-selection next step; textless cwd excluded "
                  "from goal candidates",
        "per_episode_data_present": True,
    }


def write_jsonl(out_path: Path, meta: dict, rows: list) -> None:
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        f.write(json.dumps({"meta": meta}) + "\n")
        for r in rows:
            f.write(json.dumps(r) + "\n")


def run_flat_arm(out_path: Path, max_steps: int = MAX_STEPS) -> dict:
    """FF-GEN-1/FF-CEIL-1 flat arm: Phase 8 count baseline, 5 eps x 8 tasks."""
    from phase8.count_driven_agent import Phase8Runner

    print(f"=== flat arm (Phase 8 count baseline, max_steps={max_steps}) ==",
          flush=True)
    rows = []
    summary = {}
    for task_id, image in GEN_TASK_IMAGES:
        runner = Phase8Runner(docker_image=image, task_id=task_id)
        eps = runner.run(NUM_EPISODES, max_steps)  # dicts, episodes 0..4
        for r in eps:
            r["task"] = task_id
            r["image"] = image
            r["lam"] = None
        ok = sum(1 for e in eps if e["success"])
        summary[task_id] = f"{ok}/{len(eps)}"
        print(f"  {task_id:22s} {ok}/{len(eps)}", flush=True)
        rows.extend(eps)
    write_jsonl(out_path, build_meta_gen("flat", None, len(rows),
                                         max_steps=max_steps), rows)
    pooled = sum(1 for e in rows if e["success"])
    print(f"  pooled: {pooled}/{len(rows)} -> {out_path}", flush=True)
    return {"arm": "flat", "lam": None, "pooled": pooled,
            "total": len(rows), "per_task": summary}


def run_sbh_arm(lam: float, out_path: Path, max_steps: int = MAX_STEPS) -> dict:
    """FF-GEN-1/FF-CEIL-1 SBH arm: SandboxHHAgent (现役 R1 版) driven directly.

    Mirrors src/phase9/sandbox_hh/runner.py run_task (agent module zero
    change): the module-level TASKS there is task-data, so gen1 tasks are
    driven from the script instead.
    """
    from phase9.sandbox_hh.agent import SandboxHHAgent

    name = "lam05" if lam == 0.5 else "lam0"
    print(f"=== SBH arm lambda={lam} (max_steps={max_steps}) ==", flush=True)
    rows = []
    summary = {}
    for task_id, image in GEN_TASK_IMAGES:
        agent = SandboxHHAgent(docker_image=image, task_id=task_id, lam=lam)
        eps = [agent.run_episode(i, max_steps) for i in range(NUM_EPISODES)]
        for ep in eps:
            ep["task"] = task_id
            ep["image"] = image
            ep["lam"] = lam
        ok = sum(1 for e in eps if e["success"])
        summary[task_id] = f"{ok}/{len(eps)}"
        print(f"  {task_id:22s} {ok}/{len(eps)}", flush=True)
        rows.extend(eps)
    write_jsonl(out_path, build_meta_gen(f"sbh_{name}", lam, len(rows),
                                         max_steps=max_steps), rows)
    pooled = sum(1 for e in rows if e["success"])
    print(f"  pooled: {pooled}/{len(rows)} -> {out_path}", flush=True)
    return {"arm": f"sbh_{name}", "lam": lam, "pooled": pooled,
            "total": len(rows), "per_task": summary}


def run_gen1() -> int:
    """FF-GEN-1: three arms over the 8 new v5 tasks (120 episodes)."""
    from phase2.tasks import MICRO_TASKS
    from phase9.gen_tasks import GEN_TASKS, GEN_IMAGE

    # Runtime task registration (data extension, src/phase2 零改动):
    # _get_task/_task_start_cwd read MICRO_TASKS by id; gen1 tasks are
    # appended here so both Phase8Runner and SandboxHHAgent see them.
    MICRO_TASKS.extend(GEN_TASKS)

    res = {}
    res["flat"] = run_flat_arm(Path("results/phase9_gen_flat.jsonl"))
    res["sbh_lam0"] = run_sbh_arm(0.0, Path("results/phase9_gen_sbh_lam0.jsonl"))
    res["sbh_lam05"] = run_sbh_arm(0.5, Path("results/phase9_gen_sbh_lam05.jsonl"))
    print("\nGEN-1 Summary:", json.dumps(res, indent=2))
    print(f"image: {GEN_IMAGE}")
    return 0


def run_ceil() -> int:
    """FF-CEIL-1: budget-ceiling diagnostic over the 8 v5 tasks (160 episodes).

    Two arms (flat / SBH-R1 lam=0) x two budgets (max_steps 15 / 20), 40
    episodes per arm-budget cell (5 eps x 8 tasks). s10 reference comes from
    the FF-GEN-1 JSONL files (results/phase9_gen_{flat,sbh_lam0}.jsonl), so
    only the s15/s20 cells are run here. Pure runtime-parameter variation:
    no logic change to either arm.
    """
    from phase2.tasks import MICRO_TASKS
    from phase9.gen_tasks import GEN_TASKS, GEN_IMAGE

    # Runtime task registration (data extension, src/phase2 零改动) — same
    # mechanism as run_gen1.
    MICRO_TASKS.extend(GEN_TASKS)

    res = {}
    for steps in (15, 20):
        res[f"flat_s{steps}"] = run_flat_arm(
            Path(f"results/phase9_ceil_flat_s{steps}.jsonl"), max_steps=steps)
        res[f"sbh_s{steps}"] = run_sbh_arm(
            0.0, Path(f"results/phase9_ceil_sbh_s{steps}.jsonl"), max_steps=steps)
    print("\nCEIL-1 Summary:", json.dumps(res, indent=2))
    print(f"image: {GEN_IMAGE}")
    return 0


def build_meta(lam: float, episodes: int, max_steps: int = MAX_STEPS) -> dict:
    commit = subprocess.run(["git", "rev-parse", "HEAD"],
                            capture_output=True, text=True).stdout.strip() or "unknown"
    return {
        "phase": 9,
        "experiment": "sandbox-hh",
        "lam": lam,
        "commit": commit,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "cpu_or_gpu": "cpu",
        "model": "count-based novelty (low) + frontier-goal density J (high), no learned model",
        "sandbox_images": {tid: img for tid, img in TASKS},
        "max_steps": max_steps,
        "episodes_per_task": NUM_EPISODES,
        "tasks": [tid for tid, _ in TASKS],
        "total_episodes": episodes,
        "density_definition": DENSITY_DEF,
        "low_layer": "generate_phase8_candidates + Phase8Explorer (byte-identical to Phase 8)",
        "r1_fix": "FF-SBH-3: after cd into a dir with no readable text files, force "
                  "high-layer goal re-selection next step; textless cwd excluded "
                  "from goal candidates (results/phase9_sbh_failure_analysis.md T2/R1)",
        "per_episode_data_present": True,
    }


def run_arm(lam: float, out_path: Path, max_steps: int = MAX_STEPS) -> dict:
    print(f"=== lambda={lam} arm (max_steps={max_steps}) ===", flush=True)
    runner = SandboxHHRunner(lam)
    per_task = runner.run_all(NUM_EPISODES, max_steps)

    rows = []
    summary = {}
    for task_id, _img in TASKS:
        eps = per_task[task_id]
        ok = sum(1 for e in eps if e["success"])
        summary[task_id] = f"{ok}/{len(eps)}"
        print(f"  {task_id:22s} {ok}/{len(eps)}", flush=True)
        rows.extend(eps)

    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        f.write(json.dumps({"meta": build_meta(lam, len(rows), max_steps)}) + "\n")
        for r in rows:
            f.write(json.dumps(r) + "\n")
    pooled = sum(1 for e in rows if e["success"])
    print(f"  pooled: {pooled}/{len(rows)} -> {out_path}", flush=True)
    return {"lam": lam, "pooled": pooled, "total": len(rows), "per_task": summary}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lam", type=float, choices=[0.0, 0.5], default=None,
                        help="Run a single arm (default: both)")
    parser.add_argument("--out-prefix", default="phase9_sbh",
                        help="Output file prefix under results/ "
                             "(default: phase9_sbh -> phase9_sbh_lam0.jsonl)")
    parser.add_argument("--gen1", action="store_true",
                        help="FF-GEN-1 mode: v5 image + 8 new tasks, three arms "
                             "(flat / sbh_lam0 / sbh_lam05), 40 episodes each")
    parser.add_argument("--ceil", action="store_true",
                        help="FF-CEIL-1 mode: v5 image + 8 new tasks, two arms "
                             "(flat / sbh_lam0) x max_steps 15/20, 40 episodes "
                             "each (160 total); s10 reference = FF-GEN-1 JSONL")
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS,
                        help=f"Episode step budget for the --gen1/--lam arms "
                             f"(default: {MAX_STEPS}). FF-CEIL-1 --ceil uses "
                             f"the fixed contract matrix {{15, 20}}.")
    args = parser.parse_args()

    if args.ceil:
        return run_ceil()

    if args.gen1:
        return run_gen1()

    arms = [0.0, 0.5] if args.lam is None else [args.lam]
    res = {}
    for lam in arms:
        name = "lam05" if lam == 0.5 else "lam0"
        res[name] = run_arm(lam, Path(f"results/{args.out_prefix}_{name}.jsonl"),
                            args.max_steps)
    print("\nSummary:", json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
