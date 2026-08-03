"""STRIPS-style action model learner for the busybox sandbox environment.

Learns lifted action schemas from execution traces, then uses them for
candidate generation and multi-step planning.

Schema generalization works through target typing (e.g., "cat hello.txt"
generalizes to "cat .txt files in current directory"). Over enough episodes,
the learner builds a mental map of which directories contain which files.
"""

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class ActionSchema:
    """Lifted action schema in the style of STRIPS.

    Preconditions and effects are lists of (predicate, value) tuples.
    Example for "cat hello.txt" from /sandbox:
        verb = "cat"
        preconditions = [("file_in_cwd", "hello.txt")]
        effects = [("exit_code", 0)]
        success_count = 3
        attempt_count = 5
    """
    verb: str                    # e.g., "cat", "cd", "wc"
    target_type: str             # e.g., "txt", "dir", "parent", "any"
    flag: Optional[str] = None   # e.g., "-r", "-l", None
    preconditions: List[Tuple[str, str]] = field(default_factory=list)
    effects: List[Tuple[str, str]] = field(default_factory=list)
    success_count: int = 0
    attempt_count: int = 0

    @property
    def success_rate(self) -> float:
        if self.attempt_count == 0:
            return 0.0
        return self.success_count / self.attempt_count

    def as_action_pattern(self) -> str:
        """Return a human-readable pattern string."""
        flag_str = f" {self.flag}" if self.flag else ""
        return f"{self.verb}{flag_str} {{{self.target_type}}} > exit={self.effects[0][1] if self.effects else '?'}"


class ActionModelLearner:
    """Learns STRIPS-like action schemas from execution traces.

    Tracks:
    - schemas: learned lifted action schemas keyed by (verb, target_type, flag)
    - dir_contents: mapping of directory → list of files seen there
    """

    VERB_WHITELIST = {"ls", "pwd", "cd", "cat", "echo", "mkdir", "touch",
                      "wc", "head", "tail", "grep", "find"}
    # Files that get special candidate generation
    TASK_KEYWORDS = {
        "read_hello": {"hello.txt"},
        "count_lines": {"lines.txt"},
        "read_note": {"note.txt"},
        "find_secret": {"secret"},
        "count_users": {"users.csv"},
        "find_errors": {"error.log"},
        "read_changelog": {"changelog.txt"},
        "find_admin": {"admin", "users.csv"},
        "count_logs": {"access.log"},
    }

    def __init__(self):
        self.schemas: Dict[str, ActionSchema] = {}
        self.dir_contents: Dict[str, set] = defaultdict(set)  # dir → set of files/dirs

    def _schema_key(self, verb: str, target_type: str, flag: Optional[str]):
        return f"{verb}_{target_type}_{flag or 'none'}"

    def _classify_target(self, target: Optional[str], state) -> str:
        """Classify a target argument into a lifted type.

        Returns:
        - "parent" for ".."
        - "dir" for directories
        - file extension like "txt", "csv", "py" for files
        - "any" when nothing specific
        """
        if target is None or target.strip() == "":
            return "any"
        target = target.strip()
        if target == "..":
            return "parent"
        # Check if it's a known directory in current state
        if state and hasattr(state, "files"):
            for f in state.files:
                if f == target:
                    # Could be dir or file — we don't know until we try
                    # Default to "dir" since cd to file fails gracefully
                    return "dir"
        # File extension
        if "." in target:
            ext = target.rsplit(".", 1)[-1].lower()
            if ext in ("txt", "csv", "py", "log", "ini", "md"):
                return ext
        return "any"

    def _infer_preconditions(self, state, verb: str, target: Optional[str],
                             target_type: str) -> List[Tuple[str, str]]:
        """Infer preconditions from the state before action execution.

        Common patterns:
        - cat/head/tail/wc X → precondition: X is in cwd's files
        - cd D → precondition: D is a known directory
        - grep ... → no precondition (grep returns empty if no match)
        - ls/pwd → no precondition
        """
        preconds = []
        target = target.strip() if target else None

        if verb in ("cat", "head", "tail", "wc") and target:
            # File must exist in current directory
            if state and hasattr(state, "files") and target in state.files:
                preconds.append(("file_in_cwd", target))
            else:
                preconds.append(("file_in_cwd", target))  # optimistic — will be refined on failure

        elif verb == "cd" and target and target != "..":
            # Directory must exist
            preconds.append(("dir_in_cwd", target))

        elif verb == "cd" and target == "..":
            preconds.append(("is_not_root", ""))

        elif verb == "grep" and target:
            # grep can work on any file or use -r on directory
            preconds.append(("target_exists", target))

        return preconds

    def _infer_effects(self, state, next_state, verb: str) -> List[Tuple[str, str]]:
        """Infer effects by comparing state and next_state.

        Returns list of (predicate, value) pairs.
        """
        effects = []

        # Exit code effect — always present
        ec = getattr(next_state, "last_exit_code", 0)
        effects.append(("exit_code", str(ec)))

        if verb == "cd":
            old_cwd = state.cwd if hasattr(state, "cwd") else ""
            new_cwd = next_state.cwd if hasattr(next_state, "cwd") else ""
            if old_cwd != new_cwd:
                effects.append(("cwd_changed_to", new_cwd))
            else:
                effects.append(("cwd_changed_to", "same"))  # cd to non-existent dir

        # Record directory contents for navigation map
        if hasattr(next_state, "cwd") and hasattr(next_state, "files"):
            self.dir_contents[next_state.cwd].update(next_state.files)
        if hasattr(state, "cwd") and hasattr(state, "files"):
            self.dir_contents[state.cwd].update(state.files)

        return effects

    def learn_from_step(self, state, action_str: str, next_state, success: bool):
        """Update lifted schemas from a single execution trace.

        Parses the action, classifies the target, and updates the
        corresponding action schema's precondition/effect models.
        """
        if not action_str or not action_str.strip():
            return

        parts = action_str.strip().split()
        verb = parts[0]
        if verb not in self.VERB_WHITELIST:
            return

        # Parse target and flag
        target = None
        flag = None
        for i, p in enumerate(parts[1:], 1):
            if p.startswith("-"):
                flag = p
            else:
                target = p
                break

        target_type = self._classify_target(target, state)
        key = self._schema_key(verb, target_type, flag)

        if key not in self.schemas:
            self.schemas[key] = ActionSchema(
                verb=verb,
                target_type=target_type,
                flag=flag,
                preconditions=self._infer_preconditions(state, verb, target, target_type),
                effects=self._infer_effects(state, next_state, verb),
            )
        else:
            # Refine effects on subsequent observations
            self.schemas[key].effects = self._infer_effects(state, next_state, verb)

        self.schemas[key].attempt_count += 1
        if success:
            self.schemas[key].success_count += 1

        # Update directory content map
        if hasattr(state, "cwd"):
            self.dir_contents[state.cwd].update(getattr(state, "files", []))
        if hasattr(next_state, "cwd"):
            self.dir_contents[next_state.cwd].update(getattr(next_state, "files", []))

    def generate_candidates(self, state, task_id: Optional[str] = None) -> List[str]:
        """Generate candidate actions using learned schemas + directory map.

        Strategy:
        1. Always include ls, pwd, cd ..
        2. From learned schemas: generate actions with concrete targets
           that match current state and have high success rate
        3. Task-specific: if task involves a specific file, include
           appropriate commands for that file
        4. Navigation planning: if target file not in current cwd but known
           elsewhere, generate cd commands to reach it
        """
        candidates = []

        # Always available
        candidates.append("ls")
        candidates.append("pwd")

        cwd = getattr(state, "cwd", "/sandbox")
        files = getattr(state, "files", [])
        if cwd != "/sandbox":
            candidates.append("cd ..")

        # Generate actions from high-confidence schemas
        for key, schema in self.schemas.items():
            if schema.attempt_count < 1:
                continue

            # Only use schemas with >= 50% success rate
            if schema.success_rate < 0.5 and schema.attempt_count >= 3:
                continue

            # Generate concrete actions matching current state
            for f in files:
                f_type = self._classify_target(f, state)
                if f_type == schema.target_type or target_type_matches(f_type, schema.target_type):
                    flag_str = f" {schema.flag}" if schema.flag else ""
                    action = f"{schema.verb}{flag_str} {f}"
                    if action not in candidates:
                        candidates.append(action)

        # Task-specific candidates
        if task_id and task_id in self.TASK_KEYWORDS:
            keywords = self.TASK_KEYWORDS[task_id]
            for kw in keywords:
                # Check if the target file is in current cwd
                file_in_cwd = any(kw in f for f in files)
                if file_in_cwd:
                    # Generate direct commands
                    if any(kw.endswith(ext) for ext in (".txt", ".csv", ".log", ".ini", ".md", ".py")):
                        for verb in ("cat", "head", "tail"):
                            action = f"{verb} {[f for f in files if kw in f][0]}"
                            if action not in candidates:
                                candidates.append(action)
                    if kw in ("lines.txt", "users.csv", "access.log"):
                        fname = [f for f in files if kw in f]
                        if fname:
                            action = f"wc -l {fname[0]}"
                            if action not in candidates:
                                candidates.append(action)
                    if kw == "secret":
                        candidates.append("grep -r secret .")
                        if "docs/note.txt" in files or "note.txt" in files:
                            fname = [f for f in files if "note" in f][0]
                            cand = f"cat {fname}"
                            if cand not in candidates:
                                candidates.append(cand)
                else:
                    # Navigation: find which directory has this file
                    target_dir = self._find_file_location(kw)
                    if target_dir and target_dir != cwd:
                        # Generate cd to get there
                        rel_dir = self._rel_dir(cwd, target_dir)
                        if rel_dir:
                            cand = f"cd {rel_dir}"
                            if cand not in candidates:
                                candidates.append(cand)

        # Navigation: cd into visible directories to explore
        for f in files:
            if f in ("docs", "data", "logs", "projects", "tmp", "app", "lib"):
                cand = f"cd {f}"
                if cand not in candidates:
                    candidates.append(cand)

        # Grep for content search
        candidates.append("grep -r secret .")
        candidates.append("find . -name '*.txt'")

        # Return capped unique list
        seen = set()
        deduped = []
        for c in candidates:
            if c not in seen:
                seen.add(c)
                deduped.append(c)
        return deduped[:16]

    def _find_file_location(self, filename: str) -> Optional[str]:
        """Search directory content map for a file's location."""
        for dir_path, contents in self.dir_contents.items():
            for f in contents:
                if filename in f:
                    return dir_path
        return None

    def _rel_dir(self, from_dir: str, to_dir: str) -> Optional[str]:
        """Compute relative path from from_dir to to_dir using known contents."""
        if from_dir == to_dir:
            return None

        # If to_dir is a direct child
        if to_dir.startswith(from_dir + "/"):
            rel = to_dir[len(from_dir) + 1:]
            if "/" not in rel:
                return rel  # direct child

        # If to_dir is parent
        if from_dir.startswith(to_dir + "/"):
            return ".."

        # If both share parent
        from_parent = "/".join(from_dir.rstrip("/").split("/")[:-1]) or "/"
        if to_dir.startswith(from_parent + "/"):
            # to_dir is a sibling of from_dir's parent
            parts_from = from_dir.rstrip("/").split("/")
            parts_to = to_dir.rstrip("/").split("/")
            # Count common prefix
            i = 0
            while i < len(parts_from) and i < len(parts_to) and parts_from[i] == parts_to[i]:
                i += 1
            result = [".."] * (len(parts_from) - i) + parts_to[i:]
            return "/".join(result)

        return None

    def plan_to_target(self, state, target_file: str) -> List[str]:
        """Build a multi-step plan to reach target_file.

        Returns list of action strings, e.g. ["cd data", "wc -l lines.txt"].
        """
        cwd = getattr(state, "cwd", "/sandbox")
        files = getattr(state, "files", [])

        # Check if target already accessible
        if any(target_file in f for f in files):
            # Direct action
            if target_file in self.TASK_KEYWORDS.get("read_hello", set()):
                return [f"cat {target_file}"]
            if target_file.endswith(".txt"):
                return [f"cat {target_file}"]
            if target_file.endswith((".csv", ".log")):
                return [f"wc -l {target_file}"]
            return [f"cat {target_file}"]

        # Need to navigate
        loc = self._find_file_location(target_file)
        if loc is None:
            return []

        plan = []
        rel = self._rel_dir(cwd, loc)
        if rel:
            # Split multi-step navigation
            parts = rel.split("/")
            for p in parts:
                if p == "..":
                    plan.append("cd ..")
                elif p:
                    plan.append(f"cd {p}")

        # Add final action
        if target_file.endswith(".txt"):
            plan.append(f"cat {target_file}")
        elif target_file.endswith((".csv", ".log")):
            plan.append(f"wc -l {target_file}")
        else:
            plan.append(f"cat {target_file}")

        return plan


def target_type_matches(file_type: str, schema_type: str) -> bool:
    """Check if a concrete file type matches a lifted schema type.

    "txt" matches "txt"; "csv" matches "any" schema; "dir" matches "dir".
    """
    if schema_type == "any":
        return True
    if file_type == schema_type:
        return True
    # File extensions match their extension type
    return False
