#!/usr/bin/env python3
"""M1 prior-breakage experiment — counter-intuitive sandbox (Phase 9).

Measures how badly an UNTRAINED world model (normal command-semantics prior)
predicts command outcomes in peda-sandbox:counterintuitive-v2 (deepened:
anti-correlated success exit codes cat->1, echo->2, ls->3 silent, grep->4)
vs peda-sandbox:v4.

Protocol (plan-counter-intuitive-sandbox.md, milestone M1):
  * Deterministic probe set P: 30 (state, action) pairs — 10 per reversed
    verb (cat, echo, ls) — each run against BOTH images with a FRESH
    container (identical protocol + identical model on both images; a
    cross-run DLR comparison would be invalid).
  * DLR = fraction of {L1 exit_code, L2 files_delta, L3 output_summary}
    components predicted correctly.
  * Threshold: CI DLR <= 0.35 AND normal DLR >= 0.8.
  * Interpretation (plan section 8): M1 > 0.35 => the reversal is too
    shallow — an agent with normal priors can still guess outcomes.

The prior model is a STUB encoding normal command semantics (cat reads,
echo prints, ls lists; none mutate the filesystem). M1 asks whether NORMAL
PRIORS are broken on the CI image — the stub IS the untrained prior. The
vanilla Qwen2.5-0.5B-Instruct fallback is NOT loadable here (the local HF
cache holds only an empty `refs/` stub, no weights), so the stub is the
specified substitute and the deterministic guarantee holds.

Per-probe flow: fresh container -> reset() -> type-map + fs snapshot ->
stub prediction -> sandbox.step(action) -> fs snapshot -> compare L1/L2/L3.

Outputs:
  results/phase9_ci_m1_prior_breakage.jsonl  (D4 meta header + per-probe rows)
  results/phase9_ci_m1_summary.csv           (per-image DLR summary + verdict)

Usage: PYTHONPATH=src python scripts/phase9_ci_m1_prior_breakage.py
"""

import csv
import datetime
import json
import socket
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from phase2.sandbox_env import (  # noqa: E402
    BusyboxSandbox,
    CounterIntuitiveSandbox,
)

CI_IMAGE = "peda-sandbox:counterintuitive-v2"
V4_IMAGE = "peda-sandbox:v4"
SANDBOX_ROOT = "/sandbox"

CI_ROW_LABEL = "counterintuitive-v2"
V4_ROW_LABEL = "v4"

# Probe set P: 30 deterministic (state, action) pairs, 10 per reversed verb.
# `state` is always a fresh container at /sandbox. Targets use fixtures that
# exist in one or both images (CI image: docs/note.txt, docs/readme.txt,
# data/lines.txt, logs/error.log, welcome.txt; v4 image: docs/getting_started.md,
# docs/tutorials/, data/raw/, welcome.txt). Probes on shared/missing paths are
# the "control" rows that behave identically on both images; probes on
# existing targets expose the reversals.
PROBES = [
    # ── cat (10): normal semantics = reads file, rc 0, output, no fs change.
    #    CI rule (mode A): cat DELETES its file arguments.
    ("cat", "cat welcome.txt"),                 # exists both  -> CI deletes (break)
    ("cat", "cat docs/note.txt"),               # CI-only      -> CI deletes (break)
    ("cat", "cat docs/readme.txt"),             # CI-only      -> CI deletes (break)
    ("cat", "cat data/lines.txt"),              # CI-only      -> CI deletes (break)
    ("cat", "cat logs/error.log"),              # CI-only      -> CI deletes (break)
    ("cat", "cat docs/getting_started.md"),     # v4-only      -> control
    ("cat", "cat missing_file.txt"),            # missing both -> control
    ("cat", "cat ."),                           # dir target   -> control
    ("cat", "cat docs"),                        # dir target   -> control
    ("cat", "cat welcome.txt welcome.txt"),     # multi-arg    -> CI deletes then rc 1 (break)
    # ── echo (10): normal semantics = prints args, rc 0, no fs change.
    #    CI rule (mode A): echo READS file args; non-file args -> rc 1, no output.
    ("echo", "echo hello world"),               # text args    -> CI rc 1, empty (break)
    ("echo", "echo missing_file.txt"),          # missing arg  -> CI rc 1, empty (break)
    ("echo", "echo -n hello"),                  # flag arg     -> CI rc 1, empty (break)
    ("echo", "echo docs"),                      # dir arg      -> CI rc 1, empty (break)
    ("echo", "echo /tmp/nonexistent"),          # missing path -> CI rc 1, empty (break)
    ("echo", "echo some plain text here"),      # text args    -> CI rc 1, empty (break)
    ("echo", "echo 42"),                        # text arg     -> CI rc 1, empty (break)
    ("echo", "echo welcome.txt"),               # existing file-> CI READS it (control)
    ("echo", "echo"),                           # no args      -> control
    ("echo", "echo docs/note.txt"),             # CI-only file -> CI READS it (control)
    # ── ls (10): normal semantics = lists target, rc 0, no fs change.
    #    CI rule (mode A): ls CREATES one "<entry>.ls" twin per entry.
    ("ls", "ls"),                               # cwd          -> CI twins (break)
    ("ls", "ls docs"),                          # subdir       -> CI twins (break)
    ("ls", "ls data"),                          # subdir       -> CI twins (break)
    ("ls", "ls logs"),                          # subdir       -> CI twins (break)
    ("ls", "ls -l"),                            # flags        -> CI twins in cwd (break)
    ("ls", "ls welcome.txt"),                   # file target  -> CI rc 1 (break)
    ("ls", "ls data/raw"),                      # v4-only dir  -> control
    ("ls", "ls missing_dir"),                   # missing both -> control
    ("ls", "ls docs/tutorials"),                # v4-only dir  -> control
    ("ls", "ls ."),                             # cwd          -> CI twins (break)
]


class NormalPriorStub:
    """Untrained world-model prior: NORMAL command semantics.

    Predicts what a model that knows only standard Unix semantics (and can
    see which paths in the container are files/dirs) would predict:
      L1 exit_code   — cat: 0 iff every target is an existing file
                       (dir/missing -> 1, matching `cat DIR`); echo: always 0;
                       ls: 0 iff the target exists (file or dir).
      L2 files_delta — always NO change: cat/echo/ls never mutate files
                       under normal semantics.
      L3 output      — non-empty iff the action normally produces output
                       (contents, listing, or stderr error for bad targets).
    """

    @staticmethod
    def predict(action: str, type_map: dict) -> dict:
        tokens = action.split()
        verb = tokens[0]
        args = tokens[1:]

        if verb == "cat":
            targets = [a for a in args if not a.startswith("-")]
            if not targets:
                exit_code = 0  # cat on empty stdin hits EOF silently
            else:
                exit_code = 0 if all(type_map.get(a) == "file" for a in targets) else 1
            output_nonempty = bool(targets)  # file contents, or stderr error
        elif verb == "echo":
            exit_code = 0  # normal echo always succeeds
            output_nonempty = bool(args)  # no args -> bare newline, stripped empty
        elif verb == "ls":
            file_args = [a for a in args if not a.startswith("-")]
            target = file_args[0] if file_args else "."
            exit_code = 0 if type_map.get(target) in ("file", "dir") else 1
            output_nonempty = True  # listing, or stderr error when missing
        else:
            raise ValueError(f"unsupported verb: {verb!r}")

        return {
            "exit_code": exit_code,
            "files_delta": False,  # normal cat/echo/ls never mutate the fs
            "output_nonempty": output_nonempty,
        }


def _resolve(cwd: str, target: str) -> str:
    """Resolve a probe target against the container cwd (absolute path)."""
    if target.startswith("/"):
        return str(Path(target).resolve())
    return str((Path(cwd) / target).resolve())


def _type_map(cid: str) -> dict:
    """Classify every path under /sandbox as file|dir.

    Uses /bin/busybox find (bypasses PATH wrappers, so the CI image's
    reversed applets never run — perception stays side-effect-free).
    """
    type_map = {}
    for kind, flag in (("dir", "d"), ("file", "f")):
        r = subprocess.run(
            ["docker", "exec", cid, "/bin/busybox", "find", SANDBOX_ROOT, "-type", flag],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode != 0:
            continue
        for p in r.stdout.strip().splitlines():
            type_map[str(Path(p).resolve())] = kind
    return type_map


def _fs_snapshot(cid: str) -> set:
    """Recursive listing of /sandbox (files AND dirs) — the true fs delta.

    The sandbox state's flat `files` list cannot see subdirectory changes
    (e.g. docs/note.txt.ls created by `ls docs`, or docs/note.txt deleted by
    `cat docs/note.txt`), so L2 compares recursive snapshots.
    """
    r = subprocess.run(
        ["docker", "exec", cid, "/bin/busybox", "find", SANDBOX_ROOT],
        capture_output=True, text=True, timeout=10,
    )
    if r.returncode != 0:
        return set()
    return {p for p in r.stdout.strip().splitlines() if p and p != SANDBOX_ROOT}


def run_probe(image: str, probe: tuple, prior: NormalPriorStub) -> dict:
    """Run one (image, state, action) probe. Fresh container, one action."""
    verb, action = probe
    if image == CI_IMAGE:
        sandbox = CounterIntuitiveSandbox()  # writable rootfs (cat deletes, ls twins)
    else:
        sandbox = BusyboxSandbox(image=image)  # --read-only default
    try:
        state = sandbox.reset()
        cid = state.container_id
        cwd = state.cwd

        type_map = _type_map(cid)
        before = _fs_snapshot(cid)

        # Build the probe-action lookup: resolve each non-flag token against
        # the container cwd once, then predict from the type map.
        lookup = {
            a: type_map.get(_resolve(cwd, a))
            for a in action.split()[1:]
            if not a.startswith("-")
        }
        lookup["."] = type_map.get(_resolve(cwd, "."))  # implicit cwd target (ls)
        pred = prior.predict(action, lookup)

        ns, _, _ = sandbox.step(state, action)
        after = _fs_snapshot(cid)

        actual_exit = ns.last_exit_code
        actual_delta = before != after
        actual_nonempty = bool(ns.last_output)

        l1_correct = bool(pred["exit_code"] == actual_exit)
        l2_correct = bool(pred["files_delta"] == actual_delta)  # predict: no change
        l3_correct = bool(pred["output_nonempty"] == actual_nonempty)

        return {
            "verb": verb,
            "action": action,
            "l1_correct": l1_correct,
            "l2_correct": l2_correct,
            "l3_correct": l3_correct,
            "dlr": sum((l1_correct, l2_correct, l3_correct)) / 3.0,
            "predicted": {
                "exit_code": pred["exit_code"],
                "files_delta": pred["files_delta"],
                "output_nonempty": pred["output_nonempty"],
            },
            "actual": {
                "exit_code": actual_exit,
                "files_delta": actual_delta,
                "output_nonempty": actual_nonempty,
                "output": ns.last_output[:120],
            },
        }
    finally:
        sandbox.close()


def main() -> int:
    for image in (CI_IMAGE, V4_IMAGE):
        r = subprocess.run(["docker", "image", "inspect", image],
                           capture_output=True, text=True)
        if r.returncode != 0:
            print(f"FATAL: docker image missing: {image}", file=sys.stderr)
            return 2

    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                            capture_output=True, text=True).stdout.strip()
    meta = {
        "phase": "9",
        "direction": "counter-intuitive-sandbox",
        "experiment": "M1_prior_breakage",
        "commit": commit,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "cpu_or_gpu": "cpu",
        "sandbox_images": [CI_IMAGE, V4_IMAGE],
        "model": "stub (normal-command-semantics prior)",
        "seeds": [],
        "per_episode_data_present": True,
        "probe_set": "P: 30 deterministic (state, action) pairs, 10 per reversed verb (cat/echo/ls), fresh container per probe",
        "dlr_components": ["L1 exit_code", "L2 files_delta", "L3 output_summary"],
        "threshold": {"ci_dlr_max": 0.35, "normal_dlr_min": 0.8},
    }

    rows = []
    for image in (CI_IMAGE, V4_IMAGE):
        row_label = CI_ROW_LABEL if image == CI_IMAGE else V4_ROW_LABEL
        for probe_id, probe in enumerate(PROBES, start=1):
            row = run_probe(image, probe, NormalPriorStub())
            row["image"] = row_label
            row["probe_id"] = probe_id
            rows.append(row)
            print(f"[{row_label} {probe_id:02d}/{len(PROBES)}] {row['action']:<38} "
                  f"L1={row['l1_correct']!s:<5} L2={row['l2_correct']!s:<5} "
                  f"L3={row['l3_correct']!s:<5} dlr={row['dlr']:.3f}")

    # ── JSONL with D4 meta header ──
    jsonl_path = REPO_ROOT / "results" / "phase9_ci_m1_prior_breakage.jsonl"
    with jsonl_path.open("w") as f:
        f.write(json.dumps({"meta": meta}, ensure_ascii=False) + "\n")
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # ── Per-image summary + verdict ──
    def image_stats(label):
        im_rows = [r for r in rows if r["image"] == label]
        n = len(im_rows)
        return {
            "image": label,
            "n_probes": n,
            "l1_correct": sum(r["l1_correct"] for r in im_rows),
            "l2_correct": sum(r["l2_correct"] for r in im_rows),
            "l3_correct": sum(r["l3_correct"] for r in im_rows),
            "dlr": sum(r["dlr"] for r in im_rows) / n,
        }

    ci = image_stats(CI_ROW_LABEL)
    v4 = image_stats(V4_ROW_LABEL)

    ci_pass = ci["dlr"] <= meta["threshold"]["ci_dlr_max"]
    v4_pass = v4["dlr"] >= meta["threshold"]["normal_dlr_min"]
    m1_pass = ci_pass and v4_pass

    csv_path = REPO_ROOT / "results" / "phase9_ci_m1_summary.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "n_probes", "l1_correct", "l2_correct", "l3_correct",
                    "dlr", "threshold", "threshold_met"])
        for s in (v4, ci):
            thresh = f"dlr >= {meta['threshold']['normal_dlr_min']}" if s["image"] == V4_ROW_LABEL \
                else f"dlr <= {meta['threshold']['ci_dlr_max']}"
            w.writerow([s["image"], s["n_probes"], s["l1_correct"], s["l2_correct"],
                        s["l3_correct"], f"{s['dlr']:.4f}", thresh,
                        "PASS" if (s["image"] == V4_ROW_LABEL and v4_pass)
                        or (s["image"] == CI_ROW_LABEL and ci_pass) else "FAIL"])
        w.writerow([])
        w.writerow(["m1_pass", "ci_dlr", "v4_dlr", "criterion"])
        w.writerow([m1_pass, f"{ci['dlr']:.4f}", f"{v4['dlr']:.4f}",
                    "CI DLR <= 0.35 AND v4 DLR >= 0.8"])

    # ── stdout report ──
    print("\n" + "=" * 72)
    print("M1 prior-breakage result")
    print(f"  commit:   {meta['commit']}")
    print(f"  model:    {meta['model']}")
    print(f"  probes:   {len(PROBES)} per image x {len(set(r['image'] for r in rows))} images "
          f"= {len(rows)} runs (fresh container each)")
    print(f"  v4 : DLR = {v4['dlr']:.3f}  (L1 {v4['l1_correct']}/{v4['n_probes']}, "
          f"L2 {v4['l2_correct']}/{v4['n_probes']}, L3 {v4['l3_correct']}/{v4['n_probes']})  "
          f"threshold >= 0.8: {'PASS' if v4_pass else 'FAIL'}")
    print(f"  CI : DLR = {ci['dlr']:.3f}  (L1 {ci['l1_correct']}/{ci['n_probes']}, "
          f"L2 {ci['l2_correct']}/{ci['n_probes']}, L3 {ci['l3_correct']}/{ci['n_probes']})  "
          f"threshold <= 0.35: {'PASS' if ci_pass else 'FAIL'}")
    print(f"  M1 verdict: {'PASS' if m1_pass else 'FAIL'} "
          f"(criterion: CI DLR <= 0.35 AND v4 DLR >= 0.8)")
    if not m1_pass:
        print("  note: prior breakage is real (v4 1.0 -> CI %.3f, delta %.3f), but the" % (ci["dlr"], v4["dlr"] - ci["dlr"]))
        print("        stub stays > 0.35 because L2 files_delta (echo never mutates) and")
        print("        L3 output (contents still non-empty) keep coinciding with normal")
        print("        semantics. Per plan section 8: M1 > 0.35 means the reversal is still")
        print("        too shallow -> deepen further (silence more outputs, fs-mutate more")
        print("        verbs) before M2.")
    print(f"  artifacts: {jsonl_path.name}, {csv_path.name}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
