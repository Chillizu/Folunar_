#!/usr/bin/env python3
"""Phase 2 Fix B: End-to-end smoke test for PEDA loop with LearningModule.

Runs 3 episodes of PEDA with read_note task in fast mode and records:
- steps taken, mean epistemic error, success
- proportion of steps with nonzero epistemic error
- whether new adapter checkpoints were saved

Gracefully handles exceptions and timeouts.
"""

import json
import os
import sys
import time
import signal
from datetime import datetime, timezone
from pathlib import Path

# ── Project path setup ──────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))       # for scripts/ namespace package
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

# ── Imports (after path setup) ──────────────────────────────────────
from phase1.types import DriveWeights
from phase1.world_model import WorldModel, EnsembleErrorComputer
from phase2.sandbox_env import BusyboxSandbox, generate_sandbox_candidates
from phase2.run import SandboxLearningModule, run_peda_episode
from scripts.phase2_collect_data import (
    MICRO_TASKS,
    _build_ag,
    run_peda,
    DRIVE_WEIGHTS,
    PRAGMATIC_WEIGHT,
)

# ── Output path ─────────────────────────────────────────────────────
_OUTPUT_DIR = Path(__file__).resolve().parent
_RESULTS_PATH = _OUTPUT_DIR / "smoke_results.json"


def timestamped(msg: str):
    """Print message with ISO-8601 timestamp."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    print(f"[{ts}] {msg}", flush=True)


class TimeoutError(Exception):
    """Raised when an episode exceeds the allotted wall-clock time."""
    pass


class timeout:
    """Context manager for wall-clock timeout via SIGALRM."""

    def __init__(self, seconds: int):
        self.seconds = seconds

    def __enter__(self):
        if self.seconds:
            signal.signal(signal.SIGALRM, self._handler)
            signal.alarm(self.seconds)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        signal.alarm(0)  # disarm
        return False  # don't suppress

    @staticmethod
    def _handler(signum, frame):
        raise TimeoutError(f"Episode timed out after {signum}s")

def run_episode(sb, wm, episode_num: int, task_id: str, max_steps: int, timeout_sec: int):
    """Run one PEDA episode with timeout and return structured results."""
    timestamped(f"Episode {episode_num}: starting (task={task_id}, max_steps={max_steps}, timeout={timeout_sec}s)")
    t0 = time.time()

    try:
        with timeout(timeout_sec):
            steps, final_state = run_peda(
                sb, wm, max_steps=max_steps, task_id=task_id,
                use_fast=True,  # skip ensemble checkpoint loading
            )
    except TimeoutError as e:
        elapsed = time.time() - t0
        timestamped(f"Episode {episode_num}: TIMEOUT after {elapsed:.1f}s")
        return {
            "episode": episode_num, "task": task_id, "status": "timeout",
            "steps_taken": None, "mean_epistemic_error": None,
            "mean_aleatoric_error": None, "nonzero_epistemic_steps": None,
            "total_steps_recorded": None, "success": False,
            "elapsed_seconds": round(elapsed, 1),
        }
    except Exception as e:
        import traceback
        elapsed = time.time() - t0
        timestamped(f"Episode {episode_num}: ERROR — {type(e).__name__}: {e}")
        traceback.print_exc()
        return {
            "episode": episode_num, "task": task_id, "status": "error",
            "error": f"{type(e).__name__}: {e}",
            "steps_taken": None, "mean_epistemic_error": None,
            "mean_aleatoric_error": None, "nonzero_epistemic_steps": None,
            "total_steps_recorded": None, "success": False,
            "elapsed_seconds": round(time.time() - t0, 1),
        }

    elapsed = time.time() - t0

    # Compute metrics from step records
    epistemic_errors = [s.get("epistemic_error", 0.0) for s in steps if s.get("epistemic_error") is not None]
    aleatoric_errors = [s.get("aleatoric_error", 0.0) for s in steps if s.get("aleatoric_error") is not None]
    nonzero_epistemic = sum(1 for e in epistemic_errors if e > 0)
    total_recorded = len(steps)
    mean_epi = sum(epistemic_errors) / len(epistemic_errors) if epistemic_errors else 0.0
    mean_ale = sum(aleatoric_errors) / len(aleatoric_errors) if aleatoric_errors else 0.0
    success = getattr(final_state, "victory", False)

    timestamped(
        f"Episode {episode_num}: done — steps={total_recorded} "
        f"mean_epi={mean_epi:.4f} mean_ale={mean_ale:.4f} "
        f"nonzero_epi={nonzero_epistemic}/{total_recorded} "
        f"success={success} [{elapsed:.1f}s]"
    )
    return {
        "episode": episode_num, "task": task_id, "status": "ok",
        "steps_taken": total_recorded,
        "mean_epistemic_error": round(mean_epi, 6),
        "mean_aleatoric_error": round(mean_ale, 6),
        "nonzero_epistemic_steps": nonzero_epistemic,
        "total_steps_recorded": total_recorded,
        "success": success,
        "elapsed_seconds": round(elapsed, 1),
    }


def check_checkpoints(before: set, after: set) -> dict:
    """Compare checkpoint sets before and after episodes."""
    new_ckpts = after - before
    lost_ckpts = before - after
    return {
        "checkpoints_before": sorted(before),
        "checkpoints_after": sorted(after),
        "new_checkpoints": sorted(new_ckpts),
        "checkpoints_removed": sorted(lost_ckpts),
        "new_checkpoint_saved": len(new_ckpts) > 0,
        "total_checkpoints_final": len(after),
    }


def find_checkpoint_epochs(base_dir: str = "checkpoints") -> set:
    """Recursively find all checkpoint_epoch_* paths."""
    base = Path(base_dir)
    if not base.exists():
        return set()
    return {
        str(p.relative_to(base))
        for p in base.rglob("checkpoint_epoch_*")
        if p.is_dir()
    }


def check_imports() -> list:
    """Verify all required imports resolve without error."""
    timestamped("Running import and syntax validation…")
    issues = []
    required_symbols = [
        ("WorldModel", "phase1.world_model"),
        ("EnsembleErrorComputer", "phase1.world_model"),
        ("BusyboxSandbox", "phase2.sandbox_env"),
        ("generate_sandbox_candidates", "phase2.sandbox_env"),
        ("SandboxLearningModule", "phase2.run"),
        ("run_peda_episode", "phase2.run"),
        ("MICRO_TASKS", "scripts.phase2_collect_data"),
        ("_build_ag", "scripts.phase2_collect_data"),
        ("run_peda", "scripts.phase2_collect_data"),
        ("DriveWeights", "phase1.types"),
    ]
    for name, module in required_symbols:
        try:
            __import__(module)
            mod = sys.modules[module]
            if not hasattr(mod, name):
                issues.append(f"{module}.{name}: symbol not found in module")
        except ImportError as e:
            issues.append(f"{module}: ImportError — {e}")
    if issues:
        for iss in issues:
            timestamped(f"  IMPORT ISSUE: {iss}")
    else:
        timestamped("  All imports OK")
    return issues


def _safe_get(val, default=0):
    """Return val if not None, else default."""
    return val if val is not None else default


def main():
    timestamped("=" * 60)
    timestamped("Phase 2 Fix B: PEDA + LearningModule Smoke Test")
    timestamped("=" * 60)

    # ── Step 0: Import validation ──
    import_issues = check_imports()
    if import_issues:
        timestamped("FATAL: Import issues found, aborting before Docker test.")
        results = {
            "test_name": "Phase 2 Fix B Smoke Test",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "status": "import_failed",
            "import_issues": import_issues,
            "episodes": [],
            "checkpoints": {},
        }
        _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        with open(_RESULTS_PATH, "w") as f:
            json.dump(results, f, indent=2, default=str)
        timestamped(f"Results written to {_RESULTS_PATH}")
        sys.exit(1)

    # ── Step 1: Create WorldModel ──
    model_path = os.path.expanduser("~/models/Qwen2.5-0.5B-Instruct")
    adapter_path = "checkpoints/phase1_5/text_adapter_e4"
    timestamped(f"Loading WorldModel (model={model_path}, adapter={adapter_path})…")
    try:
        wm = WorldModel(model_path, adapter_path=adapter_path)
        timestamped(f"  WorldModel mode: {wm.mode}")
    except Exception as e:
        timestamped(f"FATAL: Could not create WorldModel — {type(e).__name__}: {e}")
        results = {
            "test_name": "Phase 2 Fix B Smoke Test",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "status": "world_model_failed",
            "error": f"{type(e).__name__}: {e}",
        }
        _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        with open(_RESULTS_PATH, "w") as f:
            json.dump(results, f, indent=2, default=str)
        sys.exit(1)

    # ── Note stub mode implications ──
    if wm.mode == "stub":
        timestamped("  NOTE: Stub mode active. With use_fast=True (single prediction ensemble),")
        timestamped("  epistemic_error will be 0 (needs n>1 ensemble). Checkpoint saving requires LLM mode.")
        timestamped("  This test verifies the PEDA loop runs without crashing.")

    # ── Step 2: Create BusyboxSandbox ──
    timestamped("Creating BusyboxSandbox…")
    try:
        sb = BusyboxSandbox()
        timestamped("  BusyboxSandbox created OK")
    except Exception as e:
        timestamped(f"FATAL: Could not create BusyboxSandbox — {type(e).__name__}: {e}")
        results = {
            "test_name": "Phase 2 Fix B Smoke Test",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "status": "sandbox_failed",
            "error": f"{type(e).__name__}: {e}",
        }
        _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        with open(_RESULTS_PATH, "w") as f:
            json.dump(results, f, indent=2, default=str)
        sys.exit(1)

    # ── Step 3: Record pre-episode checkpoints ──
    ckpt_before = find_checkpoint_epochs()
    timestamped(f"Checkpoints before: {len(ckpt_before)} found")

    # ── Step 4: Run episodes ──
    timestamped("Starting 3 episodes (task=read_note, max_steps=5, fast mode)")
    episode_results = []
    for ep in range(1, 4):
        result = run_episode(sb, wm, ep, "read_note", max_steps=5, timeout_sec=120)
        episode_results.append(result)

    # ── Step 5: Post-episode checkpoint scan ──
    ckpt_after = find_checkpoint_epochs()
    ckpt_info = check_checkpoints(ckpt_before, ckpt_after)
    timestamped(f"Checkpoints after: {len(ckpt_after)} found "
                f"(new: {ckpt_info['new_checkpoints']})")

    # ── Step 6: Cleanup ──
    if sb is not None:
        try:
            sb.close()
            timestamped("BusyboxSandbox closed")
        except Exception as e:
            timestamped(f"Warning during sandbox close: {e}")

    # ── Step 7: Assemble final results ──
    all_ok = all(r["status"] == "ok" for r in episode_results)
    any_success = any(r.get("success") for r in episode_results if r.get("success") is not None)
    any_nonzero_epi = any(
        _safe_get(r.get("nonzero_epistemic_steps"), 0) > 0
        for r in episode_results
    )
    checkpoint_saved = ckpt_info["new_checkpoint_saved"]

    verdict_parts = []
    if all_ok:
        verdict_parts.append("all_episodes_ok")
    if any_success:
        verdict_parts.append("any_success")
    if any_nonzero_epi:
        verdict_parts.append("nonzero_epistemic_detected")
    if checkpoint_saved:
        verdict_parts.append("checkpoint_saved")

    overall_status = "pass" if (all_ok and checkpoint_saved) else "partial"
    if not all_ok and not any_success:
        if all(r["status"] in ("timeout", "error") for r in episode_results):
            overall_status = "fail"

    results = {
        "test_name": "Phase 2 Fix B Smoke Test",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": overall_status,
        "verdict": verdict_parts,
        "import_issues": import_issues,
        "world_model_mode": wm.mode,
        "episodes": episode_results,
        "checkpoints": ckpt_info,
        "summary": {
            "total_episodes": 3,
            "completed_ok": sum(1 for r in episode_results if r["status"] == "ok"),
            "timeouts": sum(1 for r in episode_results if r["status"] == "timeout"),
            "errors": sum(1 for r in episode_results if r["status"] == "error"),
            "successful_episodes": sum(
                1 for r in episode_results
                if r.get("success") is True
            ),
            "nonzero_epistemic_episodes": sum(
                1 for r in episode_results
                if _safe_get(r.get("nonzero_epistemic_steps"), 0) > 0
            ),
            "new_checkpoint_saved": checkpoint_saved,
        },
    }

    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(_RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)
    timestamped(f"Results written to {_RESULTS_PATH}")

    # ── Print summary ──
    print()
    print("=" * 60)
    print("SMOKE TEST SUMMARY")
    print("=" * 60)
    print(f"  Status:              {overall_status}")
    print(f"  Verdict:             {', '.join(verdict_parts)}")
    print(f"  Episodes:            {results['summary']['completed_ok']}/3 ok"
          f" ({results['summary']['timeouts']} timeout, {results['summary']['errors']} error)")
    print(f"  Successes:           {results['summary']['successful_episodes']}/3")
    print(f"  Nonzero epi steps:   {results['summary']['nonzero_epistemic_episodes']}/3 episodes")
    print(f"  New checkpoint:      {'YES' if checkpoint_saved else 'NO'}")
    if ckpt_info["new_checkpoints"]:
        print(f"  New checkpoint(s):   {ckpt_info['new_checkpoints']}")
    print(f"  WorldModel mode:     {wm.mode}")
    print()
    for r in episode_results:
        if r["status"] == "ok":
            print(f"  Episode {r['episode']}: {r['steps_taken']} steps, "
                  f"mean_epi={r['mean_epistemic_error']:.4f}, "
                  f"success={r['success']}, "
                  f"nonzero_epi={r['nonzero_epistemic_steps']}/{r['total_steps_recorded']} "
                  f"[{r['elapsed_seconds']}s]")
        else:
            print(f"  Episode {r['episode']}: {r['status']} [{r['elapsed_seconds']}s]")
    print("=" * 60)

    return 0 if overall_status != "fail" else 1


if __name__ == "__main__":
    sys.exit(main())
