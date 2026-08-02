#!/usr/bin/env python3
# ruff: noqa: E402
"""Phase 9 FF-HG-5: agent-level gate — discriminator-driven (PE) vs count baseline.

Runs ONE (agent, task) configuration and writes a per-episode JSONL artifact
(WATCHDOG D4: meta header + one JSON object per episode, including per-step
records).

    --agent count : unmodified Phase8Runner (pure count novelty) — the gate baseline
    --agent pe    : DiscriminatorRunner (alpha-blend of discriminator
                    uncertainty + count novelty; alpha=0.5 plan default,
                    alpha=1.0 exploratory "replacement" variant)

Gate spec (FF-HG-5 operationalized per Direction-1 M3, 20 eps x 3 tasks):
    PE completion% >= count completion% - 10pp -> PASS, else FAIL/DEAD.
    (threshold and band adopted verbatim from pre-registered FF-CI-6/M3)

Usage:
    python scripts/phase9_hg_f5.py --agent count --task read_changelog_v4 \
        --episodes 20 --max-steps 10 [--alpha 0.5] [--outdir results/phase9_hg_f5]
"""

import argparse
import datetime
import json
import socket
import subprocess
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from phase2.sandbox_env import generate_sandbox_candidates
from phase8.count_driven_agent import Phase8Runner
from phase9.agent import DiscriminatorRunner


def _get_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, timeout=5, cwd=_PROJECT_ROOT).stdout.strip()
    except Exception:
        return "unknown"


class CountRunner(Phase8Runner):
    """Unmodified Phase8Runner loop + per-step logging only.

    Selection/learning logic is byte-for-byte Phase8Runner.run_episode;
    step_records are added purely for the shared analysis format (the count
    baseline's 'score' == count novelty bonus; no uncertainty/error columns).
    """

    def run_episode(self, max_steps: int = 10) -> dict:
        ep = len(self.results)
        result = {"episode": ep, "success": False, "steps": 0,
                  "actions": [], "buffer_size": 0, "step_records": []}

        state = self.sandbox.reset(seed=ep, start_cwd=self._start_cwd)
        prev_state = state
        self.explorer.reset_episode()

        for t in range(max_steps):
            candidates = generate_sandbox_candidates(state)
            if not candidates:
                candidates = ["ls", "pwd"]
            action = self.explorer.select_action(state, candidates, result["actions"])
            next_state, reward, done = self.sandbox.step(state, action)

            success = False
            check_fn = self.task.get("check")
            if check_fn is not None:
                try:
                    if check_fn(prev_state if False else state, action, next_state):
                        success = True
                except Exception:
                    pass

            count_novelty = self.explorer.novelty_bonus(state, action)
            self.explorer.observe(state, action, success)
            self.buffer.append((state, action, next_state, success))
            try:
                self.action_model.learn_from_step(state, action, next_state, success)
            except Exception:
                pass

            result["actions"].append(action)
            result["step_records"].append({
                "step": t, "action": action,
                "uncertainty": None, "confidence": None,
                "count_novelty": round(count_novelty, 4),
                "score": round(count_novelty, 4),
                "predicted": None, "gt": None, "error": None,
                "exit_ok": None,
            })

            if success:
                result["success"] = True
                break
            if done:
                break

            state = next_state
            prev_state = state

        result["steps"] = len(result["actions"])
        result["buffer_size"] = len(self.buffer)
        self.results.append(result)
        return result

    def run(self, num_episodes: int = 10, max_steps: int = 10) -> list:
        for ep in range(num_episodes):
            result = self.run_episode(max_steps)
            status = "OK" if result["success"] else "FAIL"
            print(f"  Episode {ep}: {status} in {result['steps']} steps "
                  f"({result['buffer_size']} transitions)", flush=True)
        return [r for r in self.results]


def build_runner(agent: str, task: str, alpha: float, image: str):
    if agent == "count":
        return CountRunner(docker_image=image, task_id=task)
    if agent == "pe":
        return DiscriminatorRunner(docker_image=image, task_id=task, alpha=alpha)
    raise ValueError(f"unknown --agent {agent}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 9 FF-HG-5 agent-level run")
    ap.add_argument("--agent", choices=["count", "pe"], required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--max-steps", type=int, default=10)
    ap.add_argument("--image", default="peda-sandbox:v4")
    ap.add_argument("--outdir", default="results/phase9_hg_f5")
    args = ap.parse_args()

    runner = build_runner(args.agent, args.task, args.alpha, args.image)
    print(f"[hg_f5] agent={args.agent} task={args.task} alpha={args.alpha} "
          f"eps={args.episodes} max_steps={args.max_steps} image={args.image}", flush=True)

    results = runner.run(num_episodes=args.episodes, max_steps=args.max_steps)
    success = sum(1 for r in results if r["success"])
    print(f"[hg_f5] {args.agent}/{args.task}: {success}/{args.episodes} "
          f"({success / args.episodes * 100:.0f}%)", flush=True)

    meta = {
        "phase": "9",
        "direction": "hypothesis-generator",
        "commit": _get_commit(),
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "cpu_or_gpu": "cpu",
        "sandbox_image": args.image,
        "model": "count-based novelty (no model)" if args.agent == "count"
                 else f"STRIPSDiscriminator uncertainty blend (alpha={args.alpha})",
        "seeds": list(range(args.episodes)),
        "per_episode_data_present": True,
        "gate": "FF-HG-5",
        "agent": args.agent,
        "alpha": args.alpha,
        "task": args.task,
    }
    out_path = Path(args.outdir) / f"phase9_hg_f5_{args.agent}_a{args.alpha}_{args.task}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(json.dumps({"meta": meta}) + "\n")
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"[hg_f5] per-episode JSONL artifact: {out_path}", flush=True)
    return 0 if success > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
