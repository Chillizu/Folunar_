from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .config import ControllerConfig
from .memory import MemoryEntry


@dataclass
class Plan:
    summary: str
    steps: list[str]


class AIPlanner:
    def __init__(self, config: ControllerConfig) -> None:
        self.config = config

    def plan(self, goal: str, memories: Iterable[MemoryEntry]) -> Plan:
        """Return a lightweight plan that references memory entries.

        This function is intentionally offline-friendly: it does not call
        a remote LLM but uses the stored memories to generate a helpful
        outline. Replace this stub with a real model call once API access
        is configured.
        """

        memories_text = [f"- {m.app}: {m.note}" for m in memories]
        steps = [
            f"确认设备和目标应用（当前记忆 {len(memories_text)} 条）",
            "根据目标描述编写或调用脚本",
            "通过 ADB 执行脚本并监控输出",
        ]
        summary_lines = [f"目标: {goal}", "可用记忆:"]
        summary_lines.extend(memories_text if memories_text else ["(暂无记忆，建议先记录关键操作)"])
        return Plan(summary="\n".join(summary_lines), steps=steps)
