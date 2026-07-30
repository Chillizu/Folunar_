"""Phase 2 micro-task definitions shared between runner and data collection."""


def _action_hits_target(action, state, target_rel_path):
    """Check if action targets target_rel_path, handling cwd-relative paths.

    Both `cat docs/note.txt` (absolute) and `cat note.txt` from /sandbox/docs
    (relative) correctly hit "docs/note.txt".
    """
    if not action:
        return False
    if target_rel_path in action:
        return True
    filename = target_rel_path.rsplit("/", 1)[-1]
    expected_dir = "/sandbox/" + target_rel_path.rsplit("/", 1)[0] if "/" in target_rel_path else "/sandbox"
    cwd = (getattr(state, "cwd", "/sandbox") or "/sandbox").rstrip("/")
    return filename in action and cwd == expected_dir


def _is_file_reader(action):
    """True if action starts with cat, head, or tail."""
    return action and any(action.startswith(c) for c in ["cat", "head", "tail"])


def _goal_predicate_read_note(state, action, next_state) -> bool:
    if not _is_file_reader(action):
        return False
    if not _action_hits_target(action, state, "docs/note.txt"):
        return False
    return getattr(next_state, "last_exit_code", 0) == 0


def _goal_predicate_count_lines(state, action, next_state) -> bool:
    if not (action and action.startswith("wc -l")):
        return False
    if not _action_hits_target(action, state, "data/lines.txt"):
        return False
    return getattr(next_state, "last_exit_code", 0) == 0


def _goal_predicate_hello(state, action, next_state) -> bool:
    if not _is_file_reader(action):
        return False
    if not _action_hits_target(action, state, "hello.txt"):
        return False
    return getattr(next_state, "last_exit_code", 0) == 0


def _goal_predicate_find_secret(state, action, next_state) -> bool:
    is_grep = action and "grep" in action and ("secret" in action or "docs" in action)
    is_cat = _action_hits_target(action, state, "docs/note.txt") and _is_file_reader(action)
    if not (is_grep or is_cat):
        return False
    return getattr(next_state, "last_exit_code", 0) == 0




def _goal_predicate_create_file(state, action, next_state) -> bool:
    files = next_state.files if hasattr(next_state, "files") else []
    return "test_dir" in files or (action and action.strip() == "mkdir test_dir")
# ── v2 micro-tasks (enriched sandbox) ──


def _goal_predicate_count_users(state, action, next_state) -> bool:
    """wc -l data/users.csv -> output should be '5 data/users.csv'"""
    out = next_state.last_output
    return ("5" in out and "users" in out) or (action and "wc -l" in action and "users" in action)


def _goal_predicate_find_errors(state, action, next_state) -> bool:
    return "2" in next_state.last_output or "ERROR" in next_state.last_output


def _goal_predicate_read_changelog(state, action, next_state) -> bool:
    return "v2" in next_state.last_output.lower() or "v1" in next_state.last_output.lower()


def _goal_predicate_find_admin(state, action, next_state) -> bool:
    return "alice" in next_state.last_output.lower() or "admin" in next_state.last_output.lower()


def _goal_predicate_count_logs(state, action, next_state) -> bool:
    out = next_state.last_output
    return ("3" in out and "access" in out) or (action and "wc -l" in action and "access" in action)


def _goal_predicate_read_greeting(state, action, next_state) -> bool:
    if not _is_file_reader(action):
        return False
    if not _action_hits_target(action, state, "greeting.txt"):
        return False
    return getattr(next_state, "last_exit_code", 0) == 0


def _goal_predicate_count_entries(state, action, next_state) -> bool:
    if not (action and action.startswith("wc -l")):
        return False
    if not _action_hits_target(action, state, "dataset/entries.txt"):
        return False
    return getattr(next_state, "last_exit_code", 0) == 0


def _goal_predicate_find_secret_note(state, action, next_state) -> bool:
    is_grep = action and "grep" in action and "secret" in action
    is_cat = _action_hits_target(action, state, "records/secret_note.txt") and _is_file_reader(action)
    if not (is_grep or is_cat):
        return False
    return getattr(next_state, "last_exit_code", 0) == 0


def _goal_predicate_read_user_guide(state, action, next_state) -> bool:
    if not _is_file_reader(action):
        return False
    if not _action_hits_target(action, state, "records/user_guide.md"):
        return False
    if not (getattr(next_state, "last_exit_code", 0) == 0):
        return False
    # Must contain "Version" in output
    return "version" in getattr(next_state, "last_output", "").lower()


# ── v4 micro-tasks (deeper sandbox, 18 dirs) ──


def _goal_predicate_read_welcome(state, action, next_state) -> bool:
    if not _is_file_reader(action):
        return False
    if not _action_hits_target(action, state, "welcome.txt"):
        return False
    return getattr(next_state, "last_exit_code", 0) == 0


def _goal_predicate_find_api_key(state, action, next_state) -> bool:
    if not _is_file_reader(action):
        return False
    if not _action_hits_target(action, state, "docs/api_reference.md"):
        return False
    if getattr(next_state, "last_exit_code", 0) != 0:
        return False
    return "API_KEY" in getattr(next_state, "last_output", "")


def _goal_predicate_count_measurements(state, action, next_state) -> bool:
    if not (action and action.startswith("wc -l")):
        return False
    if not _action_hits_target(action, state, "data/raw/measurements_01.csv"):
        return False
    return getattr(next_state, "last_exit_code", 0) == 0


def _goal_predicate_find_errors_v4(state, action, next_state) -> bool:
    if not (action and "grep" in action and ("ERROR" in action or "error" in action)):
        return False
    return getattr(next_state, "last_exit_code", 0) == 0


def _goal_predicate_read_changelog_v4(state, action, next_state) -> bool:
    if not _is_file_reader(action):
        return False
    if not _action_hits_target(action, state, "docs/changelog.txt"):
        return False
    if getattr(next_state, "last_exit_code", 0) != 0:
        return False
    return "v4" in getattr(next_state, "last_output", "")


MICRO_TASKS = [
    {"id": "read_note", "goal": "Read docs/note.txt", "check": _goal_predicate_read_note},
    {"id": "count_lines", "goal": "Count lines in data/lines.txt", "check": _goal_predicate_count_lines},
    {"id": "read_hello", "goal": "Read hello.txt", "check": _goal_predicate_hello},
    {"id": "find_secret", "goal": "Find files with 'secret'", "check": _goal_predicate_find_secret},
    {"id": "create_file", "goal": "Create test_dir", "check": _goal_predicate_create_file},
    # v2
    {"id": "count_users", "goal": "Count users in users.csv", "check": _goal_predicate_count_users},
    {"id": "find_errors", "goal": "Count errors in error.log", "check": _goal_predicate_find_errors},
    {"id": "read_changelog", "goal": "Read docs/changelog.txt", "check": _goal_predicate_read_changelog},
    {"id": "find_admin", "goal": "Find admin user in CSV", "check": _goal_predicate_find_admin},
    {"id": "count_logs", "goal": "Count lines in access.log", "check": _goal_predicate_count_logs},
    # v3
    {"id": "read_greeting", "goal": "Read greeting.txt", "check": _goal_predicate_read_greeting},
    {"id": "count_entries", "goal": "Count entries", "check": _goal_predicate_count_entries},
    {"id": "find_secret_note", "goal": "Find secret note", "check": _goal_predicate_find_secret_note},
    {"id": "read_user_guide", "goal": "Read user guide", "check": _goal_predicate_read_user_guide},
    # v4
    {"id": "read_welcome", "goal": "Read welcome.txt", "check": _goal_predicate_read_welcome},
    {"id": "find_api_key", "goal": "Find API key", "check": _goal_predicate_find_api_key},
    {"id": "count_measurements", "goal": "Count measurements", "check": _goal_predicate_count_measurements},
    {"id": "find_errors_v4", "goal": "Find errors in logs", "check": _goal_predicate_find_errors_v4},
    {"id": "read_changelog_v4", "goal": "Read changelog for v4", "check": _goal_predicate_read_changelog_v4},
]
