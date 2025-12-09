from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass
from typing import Iterable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from openai import OpenAI

from .config import ControllerConfig
from .memory import MemoryEntry


@dataclass
class Plan:
    summary: str
    steps: list[str]


class AIPlanner:
    def __init__(self, config: ControllerConfig) -> None:
        self.config = config
        self.client: Optional["OpenAI"] = self._build_client()

    def _build_client(self) -> Optional["OpenAI"]:
        if not self.config.model.api_key:
            return None
        if importlib.util.find_spec("openai") is None:
            return None
        from openai import OpenAI

        return OpenAI(api_key=self.config.model.api_key, base_url=self.config.model.base_url)

    def plan(self, goal: str, memories: Iterable[MemoryEntry], mode: str = "auto") -> Plan:
        """Generate an execution plan.

        mode="remote" forces usage of the model API, "offline" forces the
        local heuristic, and "auto" will prefer remote when api_key exists.
        """

        use_remote = mode == "remote" or (mode == "auto" and self.client is not None)
        if use_remote and self.client:
            remote = self._remote_plan(goal, list(memories))
            if remote:
                return remote

        return self._offline_plan(goal, memories)

    def _offline_plan(self, goal: str, memories: Iterable[MemoryEntry]) -> Plan:
        memories_list = list(memories)
        memories_text = [f"- {m.app}: {m.note}" for m in memories_list]
        steps = [
            f"确认设备和目标应用（当前记忆 {len(memories_text)} 条）",
            "根据目标描述编写或调用脚本",
            "通过 ADB 执行脚本并监控输出",
        ]
        summary_lines = [f"目标: {goal}", "可用记忆:"]
        summary_lines.extend(memories_text if memories_text else ["(暂无记忆，建议先记录关键操作)"])
        return Plan(summary="\n".join(summary_lines), steps=steps)

    def _remote_plan(self, goal: str, memories: Iterable[MemoryEntry]) -> Optional[Plan]:
        if not self.client:
            return None

        memories_list = list(memories)
        memory_block = "\n".join([f"- {m.app}: {m.note}" for m in memories_list]) or "(无相关记忆)"
        prompt = {
            "role": "user",
            "content": (
                "你是一个负责基于 ADB 操控安卓设备的助手。"  # noqa: E501
                "请输出 JSON，对目标进行总结并给出 3-6 个操作步骤。"  # noqa: E501
                "字段包含 summary (string) 与 steps (string 数组)。\n"
                f"目标: {goal}\n相关记忆:\n{memory_block}"
            ),
        }

        response = self.client.chat.completions.create(
            model=self.config.model.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是 ADB 自动化策划助手，输出 JSON，steps 应为可执行建议。"
                    ),
                },
                prompt,
            ],
            temperature=self.config.model.temperature,
        )

        message = response.choices[0].message.content or ""
        try:
            payload = json.loads(message)
            summary = str(payload.get("summary", ""))
            steps = [str(item) for item in payload.get("steps", []) if str(item).strip()]
            if summary and steps:
                return Plan(summary=summary, steps=steps)
        except json.JSONDecodeError:
            pass

        fallback_steps = [line.strip("- ") for line in message.splitlines() if line.strip()]
        if message and fallback_steps:
            return Plan(summary=message, steps=fallback_steps[:6])

        return None
