"""Phase 2 micro-task definitions shared between runner and data collection."""


def _goal_predicate_read_note(state, action, next_state) -> bool:
    return "secret key" in next_state.last_output or (action and "cat docs/note" in action)


def _goal_predicate_count_lines(state, action, next_state) -> bool:
    return "3" in next_state.last_output and "lines" in next_state.last_output


def _goal_predicate_hello(state, action, next_state) -> bool:
    return "hello" in next_state.last_output


def _goal_predicate_find_secret(state, action, next_state) -> bool:
    return "secret" in next_state.last_output.lower()


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
]
