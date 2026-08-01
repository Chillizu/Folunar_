#!/usr/bin/env python3
# ruff: noqa: E402
"""M2 learnability — collect (state, action, next_state) transitions from the
counter-intuitive sandbox v2 (peda-sandbox:counterintuitive-v2).

Systematic exploration per container: apply a fixture layout (files + dirs),
then adaptively sweep cat/echo/ls (reversed verbs) plus normal-behavior anchors
(cd/pwd/wc/find/touch/mkdir) at every reachable cwd. Mutations (cat deletes,
ls creates .ls twins, touch/mkdir add entries) make each (cwd, files, action)
pair a fresh, mostly unique transition.

Output: results/phase9_ci_m2_train.jsonl
  line 1: D4 meta header
  lines 2+: phase2_collect_data-compatible records ({"baseline", "task",
           "layout", "steps_count", "metrics", "records": [...]}) where each
           step carries cwd/files/action/next_cwd/next_files/exit_code/output
           plus the full pre-action state_text (state.to_json(), identical to
           the eval-time prompt) for train/inference consistency.

Usage:
    python scripts/phase9_ci_m2_collect.py [--max-transitions 200]
"""

import argparse
import datetime
import json
import socket
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from phase2.sandbox_env import CounterIntuitiveSandbox, _list_files  # noqa: E402

# Fixture layouts — file/dir names drawn from a TRAINING pool. The held-out
# probe set (phase9_ci_m2_eval.py) uses a disjoint name pool so every held-out
# (state, action) pair is unseen.
LAYOUTS = [
    {
        "files": {
            "alpha.txt": "alpha content 111",
            "bravo.md": "# bravo markdown doc",
            "charlie.log": "2026-07-31 first log line\nsecond log line",
        },
        "dirs": {
            "room1": {"kilo.txt": "kilo data one", "lima.md": "# lima doc"},
            "room2": {"mike.log": "mike log line", "november.txt": "november data"},
        },
    },
    {
        "files": {
            "delta.txt": "delta content 222",
            "echo.md": "# echo markdown doc",
            "foxtrot.log": "log line A\nlog line B\nlog line C",
        },
        "dirs": {
            "hall1": {"oscar.txt": "oscar data", "papa.md": "# papa doc"},
            "hall2": {"quebec.log": "quebec log", "romeo.txt": "romeo data"},
        },
    },
    {
        "files": {
            "golf.txt": "golf content 333",
            "hotel.md": "# hotel markdown doc",
            "india.log": "2026-07-31 india log entry",
        },
        "dirs": {
            "wing1": {"sierra.txt": "sierra data", "tango.md": "# tango doc"},
            "wing2": {"uniform.log": "uniform log", "victor.txt": "victor data"},
        },
    },
]

# Per-container state/action budget; overall unique-transition cap.
PER_CONTAINER_CAP = 130
DEFAULT_CAP = 200


def _typed_entries(cid: str, cwd: str) -> dict:
    """Harness-side typed listing (busybox ls -1p; no wrapper side effects).

    Returns {"files": [...non-dir names], "dirs": [...]}. Names carry no path.
    """
    try:
        r = subprocess.run(
            ["docker", "exec", cid, "/bin/busybox", "ls", "-1p", cwd],
            capture_output=True, text=True, timeout=5,
        )
    except Exception:
        return {"files": [], "dirs": []}
    if r.returncode != 0:
        return {"files": [], "dirs": []}
    files, dirs = [], []
    for name in r.stdout.strip().splitlines():
        if not name:
            continue
        if name.endswith("/"):
            dirs.append(name.rstrip("/"))
        else:
            files.append(name)
    return {"files": files, "dirs": dirs}


def _apply_fixtures(cid: str, layout: dict) -> None:
    """Create the fixture layout via /bin/busybox (bypasses reversed wrappers)."""
    parts = []
    for d, files in layout["dirs"].items():
        parts.append(f"/bin/busybox mkdir -p /sandbox/{d}")
        for name, content in files.items():
            parts.append(f"printf '%s\\n' '{content}' > /sandbox/{d}/{name}")
    for name, content in layout["files"].items():
        parts.append(f"printf '%s\\n' '{content}' > /sandbox/{name}")
    cmd = " && ".join(parts)
    subprocess.run(["docker", "exec", cid, "/bin/busybox", "sh", "-c", cmd],
                   capture_output=True, text=True, timeout=15)


def _generic_actions(entries: dict, layout_idx: int) -> list:
    """Non-reversed + error-path anchors (also teaches normal semantics)."""
    acts = [
        "ls missing_dir",
        "cat missing.txt", "echo missing.txt", "echo 'hello world'",
        "cat", "echo", "pwd",
        "find . -name '*.txt'",
        f"touch ci_m2_new_{layout_idx}.txt",
        f"mkdir ci_m2_extra_{layout_idx}",
    ]
    files = entries["files"]
    if files:
        acts.append(f"wc -l {files[0]}")
    return acts


def _build_actions(entries: dict, layout_idx: int) -> list:
    """Per-cwd action plan: reversed verbs on every entry + generic anchors.

    echo before cat per file: echo READS (rc2 + content) while the file still
    exists, then cat DELETES it (rc1). Both reversal signals are recorded.
    Flag/multi-arg variants run while files still exist so the model sees the
    same action *types* the held-out probes use (on unseen names).
    ls is only planned for FRESH states (no .ls entries yet): re-listing a
    twin-rich cwd is a no-op that would muddy the ls->twins training signal
    (held-out ls probes are all fresh states).
    """
    acts = []
    files = [f for f in entries["files"] if not f.endswith(".ls")]
    for i, f in enumerate(files):
        acts.append(f"echo {f}")
        acts.append(f"cat {f}")
        if i == 0:
            acts.append(f"echo -n {f}")
            acts.append(f"cat -n {f}")
            acts.append(f"echo {f} missing.txt")
            acts.append(f"cat {f} missing.txt")
            acts.append(f"cat {f} {f}")
            if len(files) > 1:
                f1 = files[1]
                acts.append(f"cat {f} {f1}")
                acts.append(f"echo {f} {f1}")
    for d in entries["dirs"]:
        acts.append(f"ls {d}")
        acts.append(f"cat {d}")
        acts.append(f"echo {d}")
    # Exactly ONE cwd-ls variant per cwd, cycled across layouts so all three
    # (ls / ls -l / ls .) appear in training. Only planned on FRESH states
    # (no .ls entries yet): re-listing a twin-rich cwd is a no-op that would
    # muddy the ls->twins signal (held-out ls probes are all fresh states).
    has_twins = any(f.endswith(".ls") for f in entries["files"])
    if not has_twins:
        acts.append(["ls", "ls -l", "ls ."][layout_idx % 3])
    acts.extend(_generic_actions(entries, layout_idx))
    return acts


def _entry_types(entries: dict) -> dict:
    """{name: "file"|"dir"} for the entries of a typed listing."""
    types = {f: "file" for f in entries["files"]}
    types.update({d: "dir" for d in entries["dirs"]})
    return types


def _record(state, action, ns, entry_types: dict) -> dict:
    return {
        "cwd": state.cwd,
        "files": list(state.files),
        "entry_types": entry_types,
        "action": action,
        "next_cwd": ns.cwd,
        "next_files": list(ns.files),
        "exit_code": ns.last_exit_code,
        "output": ns.last_output[:200],
        "state_text": state.to_json(),
    }


def _sweep(sandbox, cid, layout_idx: int, seen: set, transitions: list, target: int,
           current):
    """One adaptive pass over the current cwd (fresh states only), up to
    `target` transitions. Returns the live state after the pass.

    Single-pass: each (cwd, files, action) is recorded at most once and the
    mutation loops are removed, so subdirectory sweeps (fresh bare-ls, subdir
    cat/echo) get budget too.
    """
    entries = _typed_entries(cid, current.cwd)
    plan = _build_actions(entries, layout_idx)
    types = _entry_types(entries)
    for action in plan:
        if len(transitions) >= target:
            break
        key = (current.cwd, tuple(current.files), action)
        if key in seen:
            continue
        ns, _, _ = sandbox.step(current, action)
        seen.add(key)
        transitions.append(_record(current, action, ns, types))
        current = ns
    return current


def collect_container(sandbox, layout_idx: int, seen: set, cap: int) -> list:
    """One fresh container: fixtures + BFS sweep over every reachable cwd."""
    transitions: list = []
    state = sandbox.reset(seed=layout_idx, start_cwd="/sandbox")
    cid = state.container_id
    _apply_fixtures(cid, LAYOUTS[layout_idx])
    state.files = _list_files(cid, state.cwd)
    current = state

    # BFS over subdirectories reachable from /sandbox.
    stack = ["/sandbox"]
    while stack and len(transitions) < cap:
        cwd = stack.pop(0)
        if current.cwd != cwd:
            action = f"cd {cwd}"
            key = (current.cwd, tuple(current.files), action)
            ns, _, _ = sandbox.step(current, action)
            if key not in seen:
                seen.add(key)
                transitions.append(_record(current, action, ns,
                                            _entry_types(_typed_entries(cid, current.cwd))))
            current = ns
        entries = _typed_entries(cid, cwd)
        # Per-cwd budget: cap the /sandbox mutation loops so subdirectory
        # sweeps (fresh bare-ls examples, subdir cat/echo) still get budget.
        cwd_target = min(len(transitions) + 45, cap)
        current = _sweep(sandbox, cid, layout_idx, seen, transitions, cwd_target, current)
        # Record the return-to-root cd transition (normal cwd semantics).
        if current.cwd != "/sandbox":
            key = (current.cwd, tuple(current.files), "cd /sandbox")
            if key not in seen and len(transitions) < cap:
                ns, _, _ = sandbox.step(current, "cd /sandbox")
                seen.add(key)
                transitions.append(_record(current, "cd /sandbox", ns,
                                            _entry_types(_typed_entries(cid, current.cwd))))
                current = ns
        for d in entries["dirs"]:
            stack.append(f"{cwd.rstrip('/')}/{d}")
    return transitions


def _strip_flags(tokens: list) -> list:
    return [t for t in tokens if not t.startswith("-")]


def classify_target(verb: str, action: str, entry_types: dict, files: list | None = None) -> str:
    """Lifted target class: none | missing | file | dir (cat/echo/ls).

    Shared by the STRIPS rule builder (phase9_ci_m2_train) and the held-out
    STRIPS predictor (phase9_ci_m2_eval). `files` is the ACTUAL pre-action
    cwd listing (a record's `files` or the live state's files) — targets that
    were deleted earlier in a sweep must classify as "missing", not "file"
    (entry_types alone is stale for mid-sweep actions).
    """
    files = files if files is not None else list(entry_types)

    def kind(t: str) -> str:
        if entry_types.get(t) == "dir":
            return "dir"
        if t in files:
            return "file"
        return "missing"

    tokens = action.split()
    targets = _strip_flags(tokens[1:])
    if not targets:
        return "none"
    # "." (and "./") targets refer to the cwd itself — same as no target.
    targets = [t for t in targets if t not in (".", "./")]
    if not targets:
        return "none"
    if verb in ("cat", "echo"):
        kinds = {kind(t) for t in targets}
        if "file" in kinds:
            return "file"
        if "dir" in kinds:
            return "dir"
        return "missing"
    if verb == "ls":
        return kind(targets[0])
    return "any"


def main() -> int:
    parser = argparse.ArgumentParser(description="M2 CI transition collection")
    parser.add_argument("--max-transitions", type=int, default=DEFAULT_CAP)
    args = parser.parse_args()

    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                            capture_output=True, text=True).stdout.strip()
    meta = {
        "phase": "9",
        "direction": "counter-intuitive-sandbox",
        "experiment": "M2_learnability_data_collection",
        "commit": commit,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "cpu_or_gpu": "cpu",
        "sandbox_images": ["peda-sandbox:counterintuitive-v2"],
        "model_path": "/home/data/models/Qwen2.5-0.5B-Instruct",
        "seeds": [],
        "per_episode_data_present": True,
        "collection": "systematic exploration: fresh container per layout, fixture "
                      "layouts (3 pools of file/dir names), adaptive sweep of cat/echo/ls "
                      "on every entry at every reachable cwd + cd/pwd/wc/find/touch/mkdir "
                      "anchors; unique (cwd, files, action) transitions",
        "reversal_rules": "cat deletes (rc1), echo reads (rc2), ls twins+rc3 (v2)",
        "state_format": "full SandboxState.to_json() (matches eval-time prompt)",
        "target_format": "EXIT/L1 exit_code | L2 files delta | L3 output non-empty "
                         "(world-model JSON target, src/phase1/world_model.py sandbox_mode)",
        "max_unique_transitions": args.max_transitions,
    }

    sandbox = CounterIntuitiveSandbox()
    seen: set = set()
    transitions: list = []
    try:
        for i in range(len(LAYOUTS)):
            if len(transitions) >= args.max_transitions:
                break
            # Even per-layout budget so all fixture name pools contribute.
            budget = min(85, args.max_transitions - len(transitions))
            before = len(transitions)
            per = collect_container(sandbox, i, seen, budget)
            transitions.extend(per)
            print(f"[collect] layout {i}: +{len(transitions) - before} new transitions "
                  f"(cumulative unique {len(transitions)})", flush=True)
        # Enforce the unique cap in collection order.
        transitions = transitions[:args.max_transitions]
    finally:
        sandbox.close()

    out_path = REPO_ROOT / "results" / "phase9_ci_m2_train.jsonl"
    with out_path.open("w") as f:
        f.write(json.dumps({"meta": meta}, ensure_ascii=False) + "\n")
        f.write(json.dumps({
            "baseline": "ci_m2_systematic",
            "task": "ci_m2",
            "steps_count": len(transitions),
            "metrics": {},
            "records": transitions,
        }, ensure_ascii=False) + "\n")

    # Distribution sanity
    verbs = {}
    exit_codes = {}
    for t in transitions:
        v = t["action"].split()[0]
        verbs[v] = verbs.get(v, 0) + 1
        exit_codes[t["exit_code"]] = exit_codes.get(t["exit_code"], 0) + 1
    print(f"[collect] wrote {len(transitions)} unique transitions -> {out_path}")
    print(f"[collect] verb distribution: {dict(sorted(verbs.items()))}")
    print(f"[collect] exit-code distribution: {dict(sorted(exit_codes.items()))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
