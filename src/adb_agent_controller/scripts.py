from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from .config import ControllerConfig


@dataclass
class ScriptStep:
    kind: str
    command: str | None = None
    wait: float | None = None
    description: str | None = None


@dataclass
class ScriptDefinition:
    name: str
    description: str
    steps: List[ScriptStep]


class ScriptRunner:
    def __init__(self, config: ControllerConfig, device: str | None = None) -> None:
        self.config = config
        self.device = device or config.default_device

    def _build_adb_command(self, cmd: str) -> List[str]:
        base = [self.config.adb_path]
        if self.device:
            base.extend(["-s", self.device])
        base.extend(cmd.split())
        return base

    def run_script(self, script: ScriptDefinition) -> None:
        for step in script.steps:
            if step.kind == "wait":
                delay = step.wait or 1
                print(f"[wait] {delay}s")
                time.sleep(delay)
                continue

            if step.kind in {"adb", "shell"}:
                cmd = step.command or ""
                full_cmd = self._build_adb_command(cmd)
                print(f"[adb] {' '.join(full_cmd)}")
                subprocess.run(full_cmd, check=True)
                continue

            raise ValueError(f"Unknown step type: {step.kind}")


def load_script(path: Path, name: str) -> ScriptDefinition:
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required to load scripts. Install dependencies via `pip install -r requirements.txt`."
        ) from exc

    target = path / f"{name}.yaml"
    if not target.exists():
        raise FileNotFoundError(f"Script not found: {target}")

    with target.open("r", encoding="utf-8") as f:
        data: Dict[str, Any] = yaml.safe_load(f) or {}

    steps_data = data.get("steps", [])
    steps = [
        ScriptStep(
            kind=step.get("type", "adb"),
            command=step.get("command"),
            wait=float(step.get("wait", 0)) if "wait" in step else None,
            description=step.get("description"),
        )
        for step in steps_data
    ]

    return ScriptDefinition(
        name=name,
        description=data.get("description", ""),
        steps=steps,
    )
