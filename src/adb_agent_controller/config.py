from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

CONFIG_DEFAULT_PATH = Path("config/config.yaml")


@dataclass
class ModelConfig:
    provider: str
    model: str
    api_key: str | None
    base_url: str | None
    temperature: float


@dataclass
class ControllerConfig:
    adb_path: str
    default_device: str | None
    scripts_dir: Path
    memory_path: Path
    model: ModelConfig


def load_config(path: str | os.PathLike | None = None) -> ControllerConfig:
    """Load controller configuration from YAML.

    Environment variables inside the YAML values are expanded.
    """

    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required. Install dependencies via `pip install -r requirements.txt`."
        ) from exc

    config_path = Path(path) if path else CONFIG_DEFAULT_PATH
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        data: Dict[str, Any] = yaml.safe_load(f) or {}

    adb_section = data.get("adb", {})
    model_section = data.get("model", {})
    memory_section = data.get("memory", {})

    adb_path = os.path.expandvars(adb_section.get("path", "adb"))
    default_device = adb_section.get("default_device")

    scripts_dir = Path(os.path.expandvars(adb_section.get("scripts_dir", "scripts"))).resolve()
    memory_path = Path(os.path.expandvars(memory_section.get("path", "data/memory.json"))).resolve()

    model_config = ModelConfig(
        provider=model_section.get("provider", "openai"),
        model=model_section.get("model", "gpt-4o-mini"),
        api_key=os.path.expandvars(model_section.get("api_key", "")) or None,
        base_url=os.path.expandvars(model_section.get("base_url", "")) or None,
        temperature=float(model_section.get("temperature", 0.2)),
    )

    return ControllerConfig(
        adb_path=adb_path,
        default_device=default_device,
        scripts_dir=scripts_dir,
        memory_path=memory_path,
        model=model_config,
    )
