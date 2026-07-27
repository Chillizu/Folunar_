"""Busybox Docker sandbox environment for Phase 2.

Single-container Linux sandbox with strict security constraints.
State is JSON-structured for clean LLM parsing.
"""

import json
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple


# Allowlist: only these commands are executable.
WHITELIST = {"ls", "cd", "cat", "echo", "mkdir", "touch", "pwd", "wc", "head", "tail", "grep", "find"}
BLOCKLIST_PATTERNS = [
    re.compile(r"\brm\b"), re.compile(r"\bmv\b"), re.compile(r"\bcp\b"),
    re.compile(r"\bchmod\b"), re.compile(r"\bchown\b"), re.compile(r"\bdd\b"),
    re.compile(r"\bmkfs\b"), re.compile(r"\bmount\b"), re.compile(r"\bsudo\b"),
    re.compile(r"\bsu\b"), re.compile(r"\bdocker\b"), re.compile(r"\bkill\b"),
    re.compile(r"\bshutdown\b"), re.compile(r"\breboot\b"),
]

DOCKER_IMAGE = "peda-sandbox:v2"
WHITELIST_HELP = "Whitelisted: ls, cd, cat, echo, mkdir, touch, pwd, wc, head, tail, grep, find"


@dataclass
class SandboxState:
    container_id: str = ""
    cwd: str = "/sandbox"
    last_command: str = ""
    last_output: str = ""
    last_exit_code: int = 0
    files: List[str] = field(default_factory=list)
    step_count: int = 0
    max_steps: int = 20
    victory: bool = False
    game_over: bool = False

    def to_json(self) -> str:
        return json.dumps({
            "cwd": self.cwd,
            "files": self.files,
            "last_command": self.last_command,
            "last_exit_code": self.last_exit_code,
            "last_output": self.last_output[:200],
            "step": self.step_count,
            "victory": self.victory,
            "game_over": self.game_over,
        }, ensure_ascii=False)

    def copy(self) -> "SandboxState":
        return SandboxState(
            container_id=self.container_id,
            cwd=self.cwd,
            last_command=self.last_command,
            last_output=self.last_output,
            last_exit_code=self.last_exit_code,
            files=list(self.files),
            step_count=self.step_count,
            max_steps=self.max_steps,
            victory=self.victory,
            game_over=self.game_over,
        )


def _validate_command(command: str) -> Tuple[bool, str]:
    """Check if a command is allowed. Returns (ok, error_msg)."""
    stripped = command.strip()
    if not stripped:
        return False, "Empty command"
    base = stripped.split()[0].lstrip("/")
    if base not in WHITELIST:
        return False, f"Command '{base}' not in whitelist. {WHITELIST_HELP}"
    for pat in BLOCKLIST_PATTERNS:
        if pat.search(stripped):
            return False, f"Command contains blocked pattern: {pat.pattern}"
    return True, ""


def _list_files(container_id: str, cwd: str) -> List[str]:
    """Get file listing of current directory (flat paths)."""
    try:
        result = subprocess.run(
            ["docker", "exec", container_id, "ls", "-1", cwd],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return [f for f in result.stdout.strip().split("\n") if f]
        return []
    except Exception:
        return []


class BusyboxSandbox:
    """Docker busybox sandbox; interface matches GridWorld / TextRoomEnv."""

    def __init__(self, image: str = DOCKER_IMAGE):
        self.image = image
        self._container_id: str = ""

    def _ensure_container(self) -> str:
        if not self._container_id:
            result = subprocess.run(
                ["docker", "run", "-d", "--rm",
                 "--cap-drop=ALL", "--read-only", "--tmpfs", "/tmp",
                 "--network", "none",
                 self.image, "sleep", "3600"],
                capture_output=True, text=True, timeout=15,
            )
            if result.returncode != 0:
                raise RuntimeError(f"Failed to start container: {result.stderr}")
            self._container_id = result.stdout.strip()
        return self._container_id

    def reset(self, seed: Optional[int] = None, start_cwd: Optional[str] = None) -> SandboxState:
        """Stop current container and start a fresh one. Optionally start in start_cwd."""
        self.close()
        cid = self._ensure_container()
        target_cwd = start_cwd if start_cwd else "/sandbox"
        if start_cwd and start_cwd != "/sandbox":
            # Navigate to target cwd and update file listing
            subprocess.run(
                ["docker", "exec", "-w", "/sandbox", cid, "sh", "-c", f"cd {start_cwd}"],
                capture_output=True, text=True, timeout=5,
            )
        files = _list_files(cid, target_cwd)
        return SandboxState(
            container_id=cid,
            cwd=target_cwd,
            files=files,
            step_count=0,
        )

    def step(self, state: SandboxState, action: str) -> Tuple[SandboxState, int, bool]:
        """Execute a command in the sandbox. Returns (next_state, reward, done)."""
        cid = self._ensure_container()

        ok, err = _validate_command(action)
        if not ok:
            ns = state.copy()
            ns.last_command = action
            ns.last_output = err
            ns.last_exit_code = 1
            ns.step_count = state.step_count + 1
            done = ns.step_count >= ns.max_steps
            return ns, 0, done

        try:
            result = subprocess.run(
                ["docker", "exec", "-w", state.cwd, cid, "sh", "-c", action],
                capture_output=True, text=True, timeout=15,
            )
        except subprocess.TimeoutExpired:
            ns = state.copy()
            ns.last_command = action
            ns.last_output = "Command timed out (15s)"
            ns.last_exit_code = 124
            ns.step_count = state.step_count + 1
            done = ns.step_count >= ns.max_steps
            return ns, 0, done

        stdout = result.stdout.strip()
        stderr = result.stderr.strip()
        output = stdout if stdout else (stderr if stderr else "")
        exit_code = result.returncode

        ns = state.copy()
        ns.last_command = action
        ns.last_output = output[:500]
        ns.last_exit_code = exit_code
        ns.step_count = state.step_count + 1

        # Update cwd on cd
        base = action.strip().split()[0].lstrip("/")
        if base == "cd":
            parts = action.strip().split(None, 1)
            target = parts[1] if len(parts) > 1 else "/sandbox"
            ns.cwd = str((Path(state.cwd) / target).resolve())
            ns.last_output = ""  # cd produces no stdout

        # Refresh file listing
        ns.files = _list_files(cid, ns.cwd)

        done = ns.step_count >= ns.max_steps
        return ns, 0, done

    def render_text(self, state: SandboxState) -> str:
        """JSON-structured state representation for World Model."""
        return state.to_json()

    def close(self) -> None:
        if self._container_id:
            subprocess.run(["docker", "kill", self._container_id],
                           capture_output=True, timeout=10)
            self._container_id = ""

    def __del__(self):
        self.close()

def generate_sandbox_candidates(state: SandboxState) -> list:
    """Generate candidate Linux commands from whitelist and current state.

    v2: enriched sandbox with 7 dirs, 14 files. Candidate strategy:
    1. Always: ls, pwd
    2. Navigation: cd into visible subdirs; cd .. if in subdir
    3. File ops: cat for text files; head/tail for larger ones; wc for counts
    4. Content search: grep for keywords in interesting locations
    5. Cap at 12 candidates.
    """
    candidates = []
    cwd = state.cwd.rstrip("/")
    files = state.files

    # ── Always available ──
    candidates.extend(["ls", "pwd"])

    # ── cd .. when in a subdirectory ──
    if cwd != "/sandbox":
        candidates.append("cd ..")

    # ── Navigation: cd into each visible directory ──
    KNOWN_DIRS = {"docs", "data", "logs", "projects", "tmp"}
    for f in files:
        if f in KNOWN_DIRS:
            candidates.append(f"cd {f}")
        elif f == "app" and cwd.endswith("/projects"):
            candidates.append("cd app")
        elif f == "lib" and cwd.endswith("/projects"):
            candidates.append("cd lib")

    # ── File reading: cat for small files ──
    SMALL_FILES = {"README.txt", "hello.txt", "note.txt", "changelog.txt",
                   "manual.txt", "config.ini", "test.py"}
    for f in files:
        if f in SMALL_FILES:
            candidates.append(f"cat {f}")

    # ── Structured data exploration ──
    if "numbers.txt" in files:
        candidates.append("cat numbers.txt")
        candidates.append("head -n 3 numbers.txt")
    if "users.csv" in files:
        candidates.append("cat users.csv")
        candidates.append("head -n 1 users.csv")
        candidates.append("wc -l users.csv")
    if "lines.txt" in files:
        candidates.append("cat lines.txt")
        candidates.append("wc -l lines.txt")
    if "access.log" in files or cwd.endswith("/logs"):
        candidates.append("cat access.log" if "access.log" in files else "cat logs/access.log")
        candidates.append("head -n 5 access.log" if "access.log" in files else "head -n 5 logs/access.log")
        candidates.append("wc -l access.log" if "access.log" in files else "wc -l logs/access.log")
    if "error.log" in files or cwd.endswith("/logs"):
        candidates.append("grep ERROR error.log" if "error.log" in files else "grep ERROR logs/error.log")
    if "main.py" in files:
        candidates.append("cat main.py")
    if "utils.py" in files:
        candidates.append("cat utils.py")

    # ── Content search across directories ──
    candidates.append("grep -r secret .")
    candidates.append("grep -r ERROR .")
    candidates.append("grep -r v2 .")
    candidates.append("grep -r admin .")

    # ── find command (new in v2 whitelist) ──
    candidates.append("find . -name '*.txt'")
    candidates.append("find . -name '*.log'")

    # ── General exploration ──
    candidates.append("echo 'explore' > /tmp/note.txt")

    # Filter to whitelist only, dedup, cap at 12
    valid = [c for c in candidates if _validate_command(c)[0]]
    seen = set()
    unique = []
    for c in valid:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    return unique[:12]
