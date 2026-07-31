#!/usr/bin/env python3
# ruff: noqa: E402
"""Phase 9 Hypothesis-Generator: probe collection on peda-sandbox:v4.

Rolls out the count-driven agent (Phase 8 selection) per top-level branch,
records every transition with ground-truth outcome predicates, then builds
zero-visit probe sets:

    D (familiar features):  train branches {/sandbox, docs, data}, seen
                            (verb, ext) combos, existing targets.
    E (novel features):     held-out branches {projects, logs, cache} + unseen
                            verb/ext combos + missing-target probes.

Per-episode JSONL output with WATCHDOG D4 metadata header.

Usage:
    python scripts/phase9_collect_probes.py [--seeds 3] [--steps 40]
        [--outdir results/phase9_probes] [--image peda-sandbox:v4]
"""

import argparse
import datetime
import json
import random
import socket
import subprocess
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from phase2.sandbox_env import (
    BusyboxSandbox,
    SandboxState,
    _list_files,
    generate_sandbox_candidates,
)
from phase5.explorer import NoveltyExplorer
from phase9.discriminator import parse_action
from phase9.types import OutcomePredicates, Transition

TRAIN_DIRS = [
    "/sandbox",
    "/sandbox/docs", "/sandbox/docs/tutorials",
    "/sandbox/data", "/sandbox/data/raw", "/sandbox/data/processed", "/sandbox/data/archive",
]
HELD_OUT_DIRS = [
    "/sandbox/projects", "/sandbox/projects/frontend", "/sandbox/projects/backend", "/sandbox/projects/shared",
    "/sandbox/logs", "/sandbox/logs/system", "/sandbox/logs/app", "/sandbox/logs/audit",
    "/sandbox/cache", "/sandbox/cache/temp", "/sandbox/cache/sessions",
]
BRANCHES = [
    ("/sandbox", "sandbox"), ("/sandbox/docs", "docs"), ("/sandbox/data", "data"),
    ("/sandbox/projects", "projects"), ("/sandbox/logs", "logs"), ("/sandbox/cache", "cache"),
]
GREP_KEYWORDS = ["error", "secret", "version"]
MISSING_TARGETS = [("missing.txt", "cat", "cat missing.txt"), ("missing.log", "head", "head -n 5 missing.log"),
                   ("missing.csv", "wc", "wc -l missing.csv"), ("missing.md", "tail", "tail missing.md")]


def _ext_of(name: str) -> str:
    if not name:
        return "none"
    base = name.rsplit("/", 1)[-1]
    if "." in base:
        e = base.rsplit(".", 1)[-1].lower()
        if e:
            return e
    return "none"


def _state_hash(cwd: str, files) -> str:
    return f"{cwd}|{','.join(sorted(files))}"


def _mk_state(cwd: str, files) -> SandboxState:
    return SandboxState(container_id="", cwd=cwd, files=list(files))


def _get_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, timeout=5, cwd=_PROJECT_ROOT).stdout.strip()
    except Exception:
        return "unknown"


def ensure_image(image: str) -> None:
    """Check the sandbox image exists; build it from Dockerfile.busybox_v4 if not."""
    r = subprocess.run(["docker", "image", "inspect", image], capture_output=True, timeout=15)
    if r.returncode == 0:
        return
    print(f"[collect] image {image} missing -> building from Dockerfile.busybox_v4")
    b = subprocess.run(["docker", "build", "-f", "Dockerfile.busybox_v4", "-t", image, "."],
                       capture_output=True, text=True, timeout=600, cwd=_PROJECT_ROOT)
    if b.returncode != 0:
        raise RuntimeError(f"docker build failed: {b.stderr[-2000:]}")
    print("[collect] image built")


def rollout_seed(sandbox: BusyboxSandbox, seed: int, steps: int):
    """Count-driven rollout across all branches. Returns transitions + executed set.

    Novelty selection (count-driven agent), random tie-break per seed.
    Success-cache latch is DISABLED (observe success=False) so exploration does
    not collapse onto the first successful command — probe collection needs
    (verb, ext) coverage, not task-solving replay.
    """
    rng = random.Random(seed)
    explorer = NoveltyExplorer()
    transitions: list[Transition] = []
    executed = set()
    history: list[str] = []

    for branch_cwd, branch_name in BRANCHES:
        state = sandbox.reset(seed=seed, start_cwd=branch_cwd)
        state.max_steps = steps
        for _ in range(steps):
            candidates = generate_sandbox_candidates(state)
            if not candidates:
                candidates = ["ls", "pwd"]
            rng.shuffle(candidates)  # seed-dependent tie-break
            action = explorer.select_action(state, candidates, history)
            next_state, _, done = sandbox.step(state, action)
            gt = OutcomePredicates.from_transition(state, action, next_state)
            success = gt.exit_ok
            transitions.append(Transition(state=state.copy(), action=action,
                                          next_state=next_state.copy(),
                                          ground_truth=gt, success=success))
            executed.add((state.state_hash(), action))
            explorer.observe(state, action, False)  # counts only, no cache latch
            history.append(action)
            state = next_state
            if done:
                break
    return transitions, executed


def collect_dir_files(sandbox: BusyboxSandbox, dirs) -> dict:
    """Listing of every probe-relevant dir from a single fresh container."""
    sandbox.reset(start_cwd="/sandbox")
    out = {}
    for d in dirs:
        out[d] = _list_files(sandbox._container_id, d)
    return out


def _seen_combos(transitions) -> tuple:
    combos = set()
    verbs = set()
    for t in transitions:
        verb, target, _ = parse_action(t.action)
        combos.add((verb, _ext_of(target)))
        verbs.add(verb)
    return combos, verbs


def build_probes(transitions, executed, files_by_dir, seed: int,
                 cap: int = 30) -> tuple:
    """Build zero-visit D (familiar) and E (novel) probe descriptors.

    Returns (D, E): lists of dicts {cwd, files, action, distance, combo}.
    """
    seen_combos, seen_verbs = _seen_combos(transitions)
    max_train_depth = max((len([p for p in t.state.cwd.split("/") if p]) for t in transitions), default=1)
    top_dirs_seen = {t.state.cwd.rstrip("/").split("/")[1] if len(t.state.cwd.rstrip("/").split("/")) > 1
                     else "sandbox" for t in transitions}

    D, E = [], []

    def add(probe_list, cwd, files, action, combo, distance):
        key = (_state_hash(cwd, files), action)
        if key in executed:
            return
        probe_list.append({
            "cwd": cwd, "files": sorted(files), "action": action,
            "combo": list(combo), "distance": distance,
        })

    def probe_distance(verb, ext, cwd, files, target):
        depth = len([p for p in cwd.split("/") if p])
        top = cwd.rstrip("/").split("/")[1] if len(cwd.rstrip("/").split("/")) > 1 else "sandbox"
        dist = 0
        if (verb, ext) not in seen_combos:
            dist += 1
        if top not in top_dirs_seen:
            dist += 1
        if target and target not in files:
            dist += 1
        if depth > max_train_depth:
            dist += 1
        return dist

    # -- train dirs -> D probes (seen combos, existing targets) --
    for d in TRAIN_DIRS:
        parent = str(Path(d).parent)
        base = Path(d).name
        files = files_by_dir.get(d, [])
        pfiles = files_by_dir.get(parent, [])

        if d != "/sandbox":
            add(D, parent, pfiles, f"cd {base}", ("cd", "none"), probe_distance("cd", "none", parent, pfiles, base))
        for f in files:
            ext = _ext_of(f)
            for verb, action in (("cat", f"cat {f}"), ("head", f"head -n 5 {f}"),
                                 ("wc", f"wc -l {f}")):
                if (verb, ext) in seen_combos:
                    add(D, parent, pfiles, action, (verb, ext),
                        probe_distance(verb, ext, parent, pfiles, f))
        if files:
            add(D, d, files, "ls", ("ls", "none"), probe_distance("ls", "none", d, files, None))
            add(D, d, files, f"find . -name '*.{_ext_of(files[0])}'", ("find", "none"),
                probe_distance("find", "none", d, files, None))
        for kw in GREP_KEYWORDS:
            add(D, d, files, f"grep -r {kw} .", ("grep", "none"),
                probe_distance("grep", "none", d, files, None))

    # -- held-out dirs -> E probes (novel dirs, unseen combos, missing targets) --
    for d in HELD_OUT_DIRS:
        parent = str(Path(d).parent)
        base = Path(d).name
        files = files_by_dir.get(d, [])
        pfiles = files_by_dir.get(parent, [])

        if d != "/sandbox":
            add(E, parent, pfiles, f"cd {base}", ("cd", "none"),
                probe_distance("cd", "none", parent, pfiles, base))
        for f in files:
            ext = _ext_of(f)
            for verb, action in (("cat", f"cat {f}"), ("head", f"head -n 5 {f}"),
                                 ("tail", f"tail {f}"), ("wc", f"wc -l {f}")):
                if (verb, ext) not in seen_combos:
                    add(E, parent, pfiles, action, (verb, ext),
                        probe_distance(verb, ext, parent, pfiles, f))
        # missing-target readers (novel target-existence)
        for fname, verb, action in MISSING_TARGETS:
            add(E, d, files, action, (verb, _ext_of(fname)),
                probe_distance(verb, _ext_of(fname), d, files, fname))
        # ls / find / grep in novel dirs
        add(E, d, files, "ls", ("ls", "none"), probe_distance("ls", "none", d, files, None))
        for ext in ("txt", "md", "log", "py", "js", "html", "css", "ini"):
            add(E, d, files, f"find . -name '*.{ext}'", ("find", "none"),
                probe_distance("find", "none", d, files, None))
        for kw in GREP_KEYWORDS:
            add(E, d, files, f"grep -r {kw} .", ("grep", "none"),
                probe_distance("grep", "none", d, files, None))
        # unseen verbs anywhere in held-out dirs
        add(E, d, files, "mkdir testdir", ("mkdir", "none"), probe_distance("mkdir", "none", d, files, "testdir"))
        add(E, d, files, "touch newfile.txt", ("touch", "txt"), probe_distance("touch", "txt", d, files, "newfile.txt"))
        add(E, d, files, "echo hello", ("echo", "none"), probe_distance("echo", "none", d, files, None))

    # -- unseen-verb probes in train dirs -> E (unseen combos, pure) --
    for d in TRAIN_DIRS:
        files = files_by_dir.get(d, [])
        parent = str(Path(d).parent)
        pfiles = files_by_dir.get(parent, [])
        for f in files:
            ext = _ext_of(f)
            if "tail" not in seen_verbs:
                add(E, parent, pfiles, f"tail {f}", ("tail", ext),
                    probe_distance("tail", ext, parent, pfiles, f))
        add(E, d, files, "mkdir testdir", ("mkdir", "none"), probe_distance("mkdir", "none", d, files, "testdir"))
        add(E, d, files, "touch newfile.txt", ("touch", "txt"), probe_distance("touch", "txt", d, files, "newfile.txt"))
        add(E, d, files, "echo hello", ("echo", "none"), probe_distance("echo", "none", d, files, None))

    # dedup (same cwd+action across branches of loops) and cap
    def dedup_cap(lst, cap):
        seen = set()
        out = []
        for p in lst:
            key = (p["cwd"], p["action"])
            if key in seen:
                continue
            seen.add(key)
            out.append(p)
            if len(out) >= cap:
                break
        return out

    return dedup_cap(D, cap), dedup_cap(E, cap)


def execute_probe(sandbox: BusyboxSandbox, probe: dict) -> dict:
    """Run one probe in a fresh container; returns record with ground truth."""
    state = sandbox.reset(start_cwd=probe["cwd"])
    state.max_steps = 10
    next_state, _, _ = sandbox.step(state, probe["action"])
    gt = OutcomePredicates.from_transition(state, probe["action"], next_state)
    return {
        "state": {"cwd": probe["cwd"], "files": sorted(state.files)},
        "action": probe["action"],
        "combo": probe["combo"],
        "distance": probe["distance"],
        "ground_truth": gt.to_dict(),
        "zero_visit": True,
    }


def write_jsonl(path: Path, meta: dict, records: list) -> None:
    with open(path, "w") as fh:
        fh.write(json.dumps({"meta": meta}) + "\n")
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 9 probe collection (v4 sandbox)")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--outdir", default="results/phase9_probes")
    ap.add_argument("--image", default="peda-sandbox:v4")
    ap.add_argument("--probe-cap", type=int, default=30)
    args = ap.parse_args()

    ensure_image(args.image)

    seeds = list(range(1, args.seeds + 1))
    meta_base = {
        "phase": "9",
        "direction": "hypothesis-generator",
        "commit": _get_commit(),
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "cpu_or_gpu": "cpu",
        "sandbox_image": args.image,
        "model": "count-driven novelty selection (random tie-break, success-cache disabled); no LLM",
        "seeds": seeds,
        "per_episode_data_present": True,
    }
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "manifest.json").write_text(json.dumps(meta_base, indent=2))

    sandbox = BusyboxSandbox(image=args.image)
    try:
        for seed in seeds:
            print(f"[collect] seed {seed}: rollout...", flush=True)
            transitions, executed = rollout_seed(sandbox, seed, args.steps)
            print(f"[collect] seed {seed}: {len(transitions)} transitions, "
                  f"{len(executed)} unique (state, action)", flush=True)
            print(f"[collect] seed {seed}: collecting dir listings...", flush=True)
            files_by_dir = collect_dir_files(sandbox, TRAIN_DIRS + HELD_OUT_DIRS)
            D, E = build_probes(transitions, executed, files_by_dir, seed, cap=args.probe_cap)
            print(f"[collect] seed {seed}: D={len(D)} E={len(E)} probes, executing...", flush=True)

            d_records = []
            for i, p in enumerate(D):
                d_records.append(execute_probe(sandbox, p))
                if (i + 1) % 10 == 0:
                    print(f"[collect]   D {i + 1}/{len(D)}", flush=True)
            e_records = []
            for i, p in enumerate(E):
                e_records.append(execute_probe(sandbox, p))
                if (i + 1) % 10 == 0:
                    print(f"[collect]   E {i + 1}/{len(E)}", flush=True)

            seed_dir = outdir / f"seed{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)
            meta = dict(meta_base, seed=seed)
            write_jsonl(seed_dir / "transitions.jsonl", meta,
                        [t.to_dict() for t in transitions])
            write_jsonl(seed_dir / "probes_D.jsonl", meta, d_records)
            write_jsonl(seed_dir / "probes_E.jsonl", meta, e_records)
            print(f"[collect] seed {seed}: wrote {seed_dir} "
                  f"(transitions.jsonl, probes_D.jsonl x{len(d_records)}, "
                  f"probes_E.jsonl x{len(e_records)})", flush=True)
    finally:
        sandbox.close()

    print(f"[collect] done -> {outdir}")


if __name__ == "__main__":
    main()
