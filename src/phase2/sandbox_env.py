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

DOCKER_IMAGE = "peda-sandbox:v4"
CI_DOCKER_IMAGE = "peda-sandbox:counterintuitive-v2"
WHITELIST_HELP = "Whitelisted: ls, cd, cat, echo, mkdir, touch, pwd, wc, head, tail, grep, find"

# Text-file extensions for candidate generation (cat/echo read attempts).
TEXT_FILE_EXTS = {"txt", "md", "yaml", "yml", "ini", "cfg", "py", "log", "csv", "json"}


@dataclass
class SandboxState:
    container_id: str = ""
    cwd: str = "/sandbox"
    last_command: str = ""
    last_output: str = ""
    last_exit_code: int = 0
    files: List[str] = field(default_factory=list)
    file_cache: dict = field(default_factory=dict)
    step_count: int = 0
    max_steps: int = 20
    victory: bool = False
    game_over: bool = False

    def to_json(self) -> str:
        return json.dumps({
            "cwd": self.cwd,
            "files": self.files,
            "file_cache": self.file_cache,
            "last_command": self.last_command,
            "last_exit_code": self.last_exit_code,
            "last_output": self.last_output[:200],
            "step": self.step_count,
            "victory": self.victory,
            "game_over": self.game_over,
        }, ensure_ascii=False)

    def to_structured_text(self) -> str:
        """Encode state as structured text for delta prediction training.

        Format: cwd: /sandbox/data | files: [config.ini, numbers.txt] | depth: 2 | parent: /sandbox
        """
        depth = len([p for p in self.cwd.split("/") if p])
        parent = str(Path(self.cwd).parent)
        files_str = ", ".join(sorted(self.files)) if self.files else ""
        cache_str = ", ".join(
            f"{k}: {v}" for k, v in self.file_cache.items()
        ) if self.file_cache else ""
        return (
            f"cwd: {self.cwd} | "
            f"files: [{files_str}] | "
            f"cache: {{{cache_str}}} | "
            f"depth: {depth} | "
            f"parent: {parent}"
        )

    def state_hash(self) -> str:
        """Compact hash for count-based novelty detection.

        Keys on cwd + sorted file list only — no command history or output.
        This means same (directory, file_set) yields same hash regardless
        of how the agent got there.
        """
        return f"{self.cwd}|{','.join(sorted(self.files))}"

    def copy(self) -> "SandboxState":
        return SandboxState(
            container_id=self.container_id,
            cwd=self.cwd,
            last_command=self.last_command,
            last_output=self.last_output,
            last_exit_code=self.last_exit_code,
            files=list(self.files),
            file_cache=dict(self.file_cache),
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
    """Get file listing of current directory (flat paths).

    Uses /bin/busybox ls to bypass PATH wrappers: in the counter-intuitive
    image, bare `ls` creates .ls twins on every perception read, so the
    harness must never invoke the wrapped applet.
    """
    try:
        result = subprocess.run(
            ["docker", "exec", container_id, "/bin/busybox", "ls", "-1", cwd],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return [f for f in result.stdout.strip().split("\n") if f]
        return []
    except Exception:
        return []


class BusyboxSandbox:
    """Docker busybox sandbox; interface matches GridWorld / TextRoomEnv."""

    def __init__(self, image: str = DOCKER_IMAGE, read_only: bool = True):
        self.image = image
        self.read_only = read_only
        self.is_ci = image == CI_DOCKER_IMAGE
        self._container_id: str = ""

    def _ensure_container(self, read_only: Optional[bool] = None) -> str:
        if not self._container_id:
            use_read_only = self.read_only if read_only is None else read_only
            run_args = ["docker", "run", "-d", "--rm",
                        "--cap-drop=ALL", "--tmpfs", "/tmp",
                        "--network", "none"]
            # Counter-intuitive image needs a writable rootfs: cat deletes
            # files and ls creates .ls twins, so --read-only would break them.
            if use_read_only:
                run_args.append("--read-only")
            result = subprocess.run(
                run_args + [self.image, "sleep", "3600"],
                capture_output=True, text=True, timeout=15,
            )
            if result.returncode != 0:
                raise RuntimeError(f"Failed to start container: {result.stderr}")
            self._container_id = result.stdout.strip()
        return self._container_id

    def reset(self, seed: Optional[int] = None, start_cwd: Optional[str] = None) -> SandboxState:
        """Stop current container and start a fresh one. Optionally start in start_cwd.

        Seed is accepted for API compatibility with grid environments but ignored
        (deterministic container start).
        """
        self.close()
        cid = self._ensure_container()
        target_cwd = (start_cwd or "/sandbox").rstrip("/")

        # Ensure the container doesn't start in a non-existent directory.
        # /bin/busybox mkdir bypasses any PATH wrapper (CI image has none for
        # mkdir, but the bypass keeps perception free of wrapper side effects).
        try:
            subprocess.run(
                ["docker", "exec", cid, "/bin/busybox", "mkdir", "-p", target_cwd],
                capture_output=True, text=True, timeout=5,
            )
        except subprocess.TimeoutExpired:
            pass

        files = _list_files(cid, target_cwd)
        return SandboxState(
            container_id=cid,
            cwd=target_cwd,
            files=files,
            file_cache={},
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

        # Cache file reads for successful commands.
        # On the CI image `echo` is the reader (cat deletes), so cache keys
        # carry the `echo ` prefix to stay consistent with how the content can
        # actually be obtained there. v2 deepening: CI reader success exits are
        # anti-correlated (echo read -> 2), so treat (0, 2) as success there.
        cache_success_codes = (0, 2) if self.is_ci else (0,)
        if exit_code in cache_success_codes and stdout:
            action_str = action.strip()
            reader_verbs = ("cat ", "head ", "tail ", "echo ") if self.is_ci else ("cat ", "head ", "tail ")
            reader_prefix = "echo " if self.is_ci else ""
            if action_str.startswith(reader_verbs):
                parts = action_str.split(None, 1)
                if len(parts) > 1:
                    pattern = parts[1]
                    ns.file_cache[reader_prefix + pattern] = stdout[:200]
            elif action_str.startswith("wc -l "):
                parts = action_str.split(None, 2)
                if len(parts) > 2:
                    ns.file_cache[parts[2]] = stdout[:200]
                elif len(parts) > 1:
                    ns.file_cache[parts[1]] = stdout[:200]

        # Handle `cd` specially: update working directory
        action_str = action.strip()
        if action_str.startswith("cd "):
            target = action_str[3:].strip()
            if target == "..":
                ns.cwd = str(Path(ns.cwd).parent)
            elif target.startswith("/"):
                ns.cwd = target
            else:
                ns.cwd = str((Path(ns.cwd) / target).resolve())
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


class CounterIntuitiveSandbox(BusyboxSandbox):
    """Sandbox for the counter-intuitive image (peda-sandbox:counterintuitive-v2).

    Reversed command semantics: echo reads, cat deletes, ls creates .ls twins
    (silently), grep inverts, head/tail swap. v2 deepened success exit codes:
    cat delete -> 1, echo read -> 2, ls twin -> 3, grep inverted match -> 4
    (deterministic + anti-correlated; M2-learnable). The reversals mutate the
    filesystem, so the container runs with a WRITABLE rootfs (read_only=False)
    instead of the default --read-only.
    """

    def __init__(self):
        super().__init__(image=CI_DOCKER_IMAGE, read_only=False)


def generate_sandbox_candidates(state: SandboxState) -> list:
    """Data-driven candidate generation based on sandbox file contents.

    Strategy:
    1. Always: ls, pwd
    2. cd .. when in subdirectory
    3. cat + echo for any text-file-extension file (echo is the reader in the
       counter-intuitive sandbox, but cat must stay reachable as the wrong
       prior — both candidates are generated for every text file)
    4. head for log files; wc for csv/log
    5. cd into subdirectories (no-extension files that aren't special readme)
    6. grep -r for common keywords
    7. find for txt, md, log
    8. Cap at 16 candidates.
    """
    candidates = ["ls", "pwd"]
    cwd = state.cwd.rstrip("/")
    if cwd != "/sandbox":
        candidates.append("cd ..")
    for f in state.files:
        ext = f.rsplit(".", 1)[-1] if "." in f else ""
        if ext in TEXT_FILE_EXTS:
            candidates.append(f"cat {f}")
            candidates.append(f"echo {f}")
        if ext == "log":
            candidates.append(f"head -n 5 {f}")
        if ext in {"csv", "log"}:
            candidates.append(f"wc -l {f}")
        if ext == "" and f not in {"readme.md", "README.txt"}:
            candidates.append(f"cd {f}")
    candidates.extend(["grep -r error .", "grep -r secret .", "grep -r version .",
                       "find . -name '*.txt'", "find . -name '*.md'", "find . -name '*.log'"])
    valid = [c for c in candidates if _validate_command(c)[0]]
    seen = set()
    unique = [c for c in valid if c not in seen and not seen.add(c)]
    return unique[:16]
