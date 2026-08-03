"""Phase 9 FF-GEN-1: 泛化判别实验 — 新任务定义（数据，非逻辑）。

8 个全新任务（peda-sandbox:v5，全部 /sandbox 起始，seeds 0-4，max_steps 10）：

  对照组（2）：gen_read_notes（根，dist-0）、gen_read_setup（dist-1）
  deep-path read/find（4）：gen_read_sensor、gen_read_usage、
      gen_find_api_ref、gen_read_audit（目标均在 dist>=2）
  grep/count deep（2）：gen_count_readings（wc -l dist-2）、
      gen_find_error_deep（grep 定位 dist-2）

成功判据复用既有评估机制：phase2.tasks 的 `_action_hits_target` /
`_is_file_reader` + exit code / 输出内容检查（与 read_note、find_api_key、
count_lines、find_errors_v4 同族）。任务注册为运行时数据扩展
（脚本侧 `MICRO_TASKS.extend(GEN_TASKS)`），src/phase2 零改动。
"""

from phase2.tasks import _action_hits_target, _is_file_reader


def _goal_predicate_gen_read_notes(state, action, next_state) -> bool:
    """T1 对照（dist-0）：cat 根目录 notes.txt。"""
    if not _is_file_reader(action):
        return False
    if not _action_hits_target(action, state, "notes.txt"):
        return False
    return getattr(next_state, "last_exit_code", 0) == 0


def _goal_predicate_gen_read_setup(state, action, next_state) -> bool:
    """T2 对照（dist-1）：cat docs/setup.md。"""
    if not _is_file_reader(action):
        return False
    if not _action_hits_target(action, state, "docs/setup.md"):
        return False
    return getattr(next_state, "last_exit_code", 0) == 0


def _goal_predicate_gen_read_sensor(state, action, next_state) -> bool:
    """T3 deep-path（dist-2）：cat data/raw/sensor.log，输出须含 SENSOR。"""
    if not _is_file_reader(action):
        return False
    if not _action_hits_target(action, state, "data/raw/sensor.log"):
        return False
    if getattr(next_state, "last_exit_code", 0) != 0:
        return False
    return "SENSOR" in getattr(next_state, "last_output", "")


def _goal_predicate_gen_read_usage(state, action, next_state) -> bool:
    """T4 deep-path（dist-2）：cat docs/guides/usage.md，输出须含 version。"""
    if not _is_file_reader(action):
        return False
    if not _action_hits_target(action, state, "docs/guides/usage.md"):
        return False
    if getattr(next_state, "last_exit_code", 0) != 0:
        return False
    return "version" in getattr(next_state, "last_output", "").lower()


def _goal_predicate_gen_find_api_ref(state, action, next_state) -> bool:
    """T5 deep-path find（dist-2）：docs/ref/api.txt，输出须含 API_KEY。"""
    if not _is_file_reader(action):
        return False
    if not _action_hits_target(action, state, "docs/ref/api.txt"):
        return False
    if getattr(next_state, "last_exit_code", 0) != 0:
        return False
    return "API_KEY" in getattr(next_state, "last_output", "")


def _goal_predicate_gen_read_audit(state, action, next_state) -> bool:
    """T6 deep-path（dist-2）：cat logs/app/audit.log，输出须含 AUDIT。"""
    if not _is_file_reader(action):
        return False
    if not _action_hits_target(action, state, "logs/app/audit.log"):
        return False
    if getattr(next_state, "last_exit_code", 0) != 0:
        return False
    return "AUDIT" in getattr(next_state, "last_output", "")


def _goal_predicate_gen_count_readings(state, action, next_state) -> bool:
    """T7 count deep（dist-2）：wc -l data/raw/counts.txt。"""
    if not (action and action.startswith("wc -l")):
        return False
    if not _action_hits_target(action, state, "data/raw/counts.txt"):
        return False
    return getattr(next_state, "last_exit_code", 0) == 0


def _goal_predicate_gen_find_error_deep(state, action, next_state) -> bool:
    """T8 grep deep（dist-2）：grep error 必须落在 logs/system（dist-2）。

    与 find_errors_v4（任意 cwd 的 grep -r error 即过）不同：要求动作
    提及 system，或当前 cwd 已在 /sandbox/logs/system——根目录的
    `grep -r error .`（dist-0 全树递归）不算 deep，避免任务退化为
    根目录一枪。"""
    if not (action and "grep" in action and "error" in action):
        return False
    if getattr(next_state, "last_exit_code", 0) != 0:
        return False
    cwd = (getattr(state, "cwd", "/sandbox") or "/sandbox").rstrip("/")
    if "system" in action:
        return True
    return cwd == "/sandbox/logs/system"


GEN_TASKS = [
    {"id": "gen_read_notes", "goal": "Read notes.txt",
     "check": _goal_predicate_gen_read_notes},
    {"id": "gen_read_setup", "goal": "Read docs/setup.md",
     "check": _goal_predicate_gen_read_setup},
    {"id": "gen_read_sensor", "goal": "Read data/raw/sensor.log",
     "check": _goal_predicate_gen_read_sensor},
    {"id": "gen_read_usage", "goal": "Read docs/guides/usage.md",
     "check": _goal_predicate_gen_read_usage},
    {"id": "gen_find_api_ref", "goal": "Find API key in docs/ref/api.txt",
     "check": _goal_predicate_gen_find_api_ref},
    {"id": "gen_read_audit", "goal": "Read logs/app/audit.log",
     "check": _goal_predicate_gen_read_audit},
    {"id": "gen_count_readings", "goal": "Count lines in data/raw/counts.txt",
     "check": _goal_predicate_gen_count_readings},
    {"id": "gen_find_error_deep", "goal": "Find ERROR lines in logs/system/error.log",
     "check": _goal_predicate_gen_find_error_deep},
]

GEN_IMAGE = "peda-sandbox:v5"
GEN_TASK_IMAGES = [(t["id"], GEN_IMAGE) for t in GEN_TASKS]

# 深度归类（用于分析/报告的子集对照）
GEN_DEEP_TASKS = ["gen_read_sensor", "gen_read_usage", "gen_find_api_ref",
                  "gen_read_audit", "gen_count_readings", "gen_find_error_deep"]
GEN_CONTROL_TASKS = ["gen_read_notes", "gen_read_setup"]
