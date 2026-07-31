#!/usr/bin/env python3
"""M0 build-gate probe for peda-sandbox:counterintuitive.

Verifies every reversed rule exactly as specified in
local://plan-counter-intuitive-sandbox.md (rule table, Level 1 + Level 2
wrappers) and the harness perception bypass (/bin/busybox ls must NOT trigger
reversals). Exit 0 = all checks pass; exit 1 = any check failed.

Usage: python scripts/ci_probe_sandbox.py [image]
"""

import subprocess
import sys

IMAGE = sys.argv[1] if len(sys.argv) > 1 else "peda-sandbox:counterintuitive-v1"


def run(cid, cmd, cwd="/sandbox"):
    r = subprocess.run(
        ["docker", "exec", "-w", cwd, cid, "sh", "-c", cmd],
        capture_output=True, text=True, timeout=10,
    )
    return r.returncode, r.stdout, r.stderr


def busybox_ls(cid, path="/sandbox"):
    """Harness perception path — MUST bypass wrappers (no side effects)."""
    r = subprocess.run(
        ["docker", "exec", cid, "/bin/busybox", "ls", "-1", path],
        capture_output=True, text=True, timeout=10,
    )
    if r.returncode != 0:
        return []
    return [f for f in r.stdout.strip().split("\n") if f]


def main() -> int:
    cid = subprocess.run(
        ["docker", "run", "-d", "--rm", "--cap-drop=ALL", "--network", "none",
         IMAGE, "sleep", "3600"],
        capture_output=True, text=True, timeout=15,
    ).stdout.strip()
    checks = []

    try:
        # ── Level 1: echo reads ──
        rc, out, _ = run(cid, "echo docs/note.txt")
        checks.append(("L1 echo reads file (secret revealed)", rc == 0 and "secret key: 9471" in out))

        rc, out, _ = run(cid, "echo no_such_file.txt")
        checks.append(("L1 echo on missing file: empty output, exit 1", rc == 1 and out.strip() == ""))

        # ── Level 1: cat deletes ──
        rc, out, _ = run(cid, "cat docs/readme.txt")
        files_after = busybox_ls(cid, "/sandbox/docs")
        checks.append(("L1 cat deletes file (empty output, file gone)", rc == 0 and out.strip() == "" and "readme.txt" not in files_after))

        rc, out, err = run(cid, "cat missing.txt")
        checks.append(("L1 cat on missing file: exit 1 + stderr", rc == 1 and "No such file" in err))

        # ── Level 1: ls creates bounded twins ──
        rc, out, _ = run(cid, "ls")
        files_after = busybox_ls(cid)
        checks.append(("L1 ls creates .ls twins at root", rc == 0 and "welcome.txt.ls" in files_after and "data.ls" in files_after))
        checks.append(("L1 ls prints twin names", rc == 0 and "welcome.txt.ls" in out and "data.ls" in out))

        rc, _, _ = run(cid, "ls docs")
        files_docs = busybox_ls(cid, "/sandbox/docs")
        checks.append(("L1 ls creates twins in subdir", rc == 0 and "note.txt.ls" in files_docs))

        n_before = len(busybox_ls(cid))
        run(cid, "ls")
        n_after = len(busybox_ls(cid))
        checks.append(("L1 ls bounded (no twin-of-twin growth)", n_after == n_before))

        # ── Level 2: grep inverts ──
        rc, out, _ = run(cid, "grep ERROR logs/error.log")
        checks.append(("L2 grep inverts (WARN only, no ERROR line)", rc == 0 and "WARN" in out and "ERROR" not in out and "retry" in out))

        # ── Level 2: head -> last, tail -> first ──
        rc, out, _ = run(cid, "head -n 1 logs/access.log")
        checks.append(("L2 head returns LAST line", rc == 0 and "POST /submit" in out))

        rc, out, _ = run(cid, "tail -n 1 logs/access.log")
        checks.append(("L2 tail returns FIRST line", rc == 0 and "GET /index" in out))

        # ── Harness perception bypass: busybox ls has no side effects ──
        twins_before = [f for f in busybox_ls(cid) if f.endswith(".ls")]
        n_before = len(busybox_ls(cid))
        busybox_ls(cid)
        n_after = len(busybox_ls(cid))
        checks.append(("perception bypass: /bin/busybox ls is side-effect free",
                       n_after == n_before and len([f for f in busybox_ls(cid) if f.endswith(".ls")]) == len(twins_before)))
    finally:
        subprocess.run(["docker", "kill", cid], capture_output=True, timeout=10)

    failed = [name for name, ok in checks if not ok]
    for name, ok in checks:
        print(f"[{'OK' if ok else 'FAIL'}] {name}")
    print(f"\n{len(checks) - len(failed)}/{len(checks)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
