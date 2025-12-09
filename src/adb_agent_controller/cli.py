from __future__ import annotations

import argparse
import subprocess
from typing import Iterable

from .ai import AIPlanner
from .config import ControllerConfig, load_config
from .memory import MemoryEntry, MemoryStore
from .scripts import ScriptRunner, load_script


def run_cmd(args: list[str]) -> None:
    subprocess.run(args, check=True)


def list_devices(config: ControllerConfig) -> None:
    run_cmd([config.adb_path, "devices"])


def connect_device(config: ControllerConfig, target: str) -> None:
    run_cmd([config.adb_path, "connect", target])


def disconnect_device(config: ControllerConfig, target: str | None) -> None:
    cmd = [config.adb_path, "disconnect"]
    if target:
        cmd.append(target)
    run_cmd(cmd)


def run_shell(config: ControllerConfig, device: str | None, command: str) -> None:
    cmd = [config.adb_path]
    if device:
        cmd.extend(["-s", device])
    cmd.extend(["shell", command])
    run_cmd(cmd)


def handle_memory(store: MemoryStore, args: argparse.Namespace) -> None:
    if args.memory_cmd == "add":
        entry = MemoryEntry.create(app=args.app, note=args.note)
        store.add(entry)
        print(f"已保存记忆: {entry.app} -> {entry.note}")
        return

    entries = store.list(app=args.app)
    if not entries:
        print("暂无匹配的记忆")
        return
    for entry in entries:
        print(f"[{entry.timestamp}] {entry.app}: {entry.note}")


def handle_scripts(config: ControllerConfig, args: argparse.Namespace) -> None:
    script = load_script(config.scripts_dir, args.name)
    runner = ScriptRunner(config, device=args.device)
    runner.run_script(script)


def handle_ai(config: ControllerConfig, store: MemoryStore, args: argparse.Namespace) -> None:
    planner = AIPlanner(config)
    memories: Iterable[MemoryEntry] = store.list(app=args.app) if args.app else store.list()
    plan = planner.plan(goal=args.goal, memories=memories, mode=args.mode)
    print("=== AI 规划结果 ===")
    print(plan.summary)
    print("建议步骤:")
    for idx, step in enumerate(plan.steps, 1):
        print(f" {idx}. {step}")


def show_panel(config: ControllerConfig, store: MemoryStore) -> None:
    print("=== 配置面板 ===")
    print(f"ADB 路径: {config.adb_path}")
    print(f"默认设备: {config.default_device or '未设置'}")
    print(f"脚本目录: {config.scripts_dir}")
    print(f"记忆库: {config.memory_path}")
    print()

    print("=== 模型配置 ===")
    model = config.model
    print(f"Provider: {model.provider}")
    print(f"Model: {model.model}")
    print(f"Base URL: {model.base_url or '未设置'}")
    print(f"Temperature: {model.temperature}")
    print(f"API Key: {'已配置' if model.api_key else '未配置'}")
    print()

    scripts = sorted(config.scripts_dir.glob("*.yaml"))
    print("=== 可用脚本 ===")
    if scripts:
        for script in scripts:
            print(f"- {script.stem}")
    else:
        print("(暂无脚本，向 scripts/ 目录添加 .yaml 文件)")
    print()

    entries = list(store.list())
    apps = sorted({e.app for e in entries})
    print("=== 记忆概览 ===")
    print(f"总记忆条数: {len(entries)}")
    if apps:
        print("覆盖的应用: " + ", ".join(apps))
    else:
        print("(暂无记忆，使用 memory add 进行添加)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CLI to control Android devices via ADB and AI helpers")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    parser.add_argument("--model-provider", type=str, help="Override model provider, e.g. openai or azure")
    parser.add_argument("--model-name", type=str, help="Override model name")
    parser.add_argument("--model-api-key", type=str, help="Override model API key")
    parser.add_argument("--model-base-url", type=str, help="Override base URL for model API")
    parser.add_argument(
        "--model-temperature",
        type=float,
        help="Override sampling temperature (e.g. 0.2)",
    )

    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("status", help="List available devices")

    connect_parser = subparsers.add_parser("connect", help="Connect to a device or emulator")
    connect_parser.add_argument("target", help="host:port or device serial")

    disconnect_parser = subparsers.add_parser("disconnect", help="Disconnect a device or all")
    disconnect_parser.add_argument("target", nargs="?", help="target host:port or device serial")

    shell_parser = subparsers.add_parser("shell", help="Run an adb shell command")
    shell_parser.add_argument("shell_command", help="Shell command to run")
    shell_parser.add_argument("--device", help="Device serial", default=None)

    memory_parser = subparsers.add_parser("memory", help="Manage AI memory store")
    memory_sub = memory_parser.add_subparsers(dest="memory_cmd", required=True)
    memory_add = memory_sub.add_parser("add", help="Add memory entry")
    memory_add.add_argument("--app", required=True, help="App or game name")
    memory_add.add_argument("--note", required=True, help="Memory note")

    memory_list = memory_sub.add_parser("list", help="List memory entries")
    memory_list.add_argument("--app", help="Filter by app name")

    scripts_parser = subparsers.add_parser("scripts", help="Run predefined scripts")
    scripts_sub = scripts_parser.add_subparsers(dest="scripts_cmd", required=True)
    scripts_run = scripts_sub.add_parser("run", help="Run a script by name")
    scripts_run.add_argument("name", help="Script name without extension")
    scripts_run.add_argument("--device", help="Device serial", default=None)

    subparsers.add_parser("panel", help="显示 CLI 面板，汇总配置与可用资源")

    ai_parser = subparsers.add_parser("ai", help="AI planning helper")
    ai_parser.add_argument("goal", help="目标描述，例如：自动登录并领取奖励")
    ai_parser.add_argument("--app", help="过滤记忆的应用名称")
    ai_parser.add_argument(
        "--mode",
        choices=["auto", "remote", "offline"],
        default="auto",
        help="选择 AI 规划模式：auto 自动，remote 强制调用模型，offline 本地占位",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = load_config(args.config)

    if args.model_provider:
        config.model.provider = args.model_provider
    if args.model_name:
        config.model.model = args.model_name
    if args.model_api_key:
        config.model.api_key = args.model_api_key
    if args.model_base_url:
        config.model.base_url = args.model_base_url
    if args.model_temperature is not None:
        config.model.temperature = args.model_temperature

    store = MemoryStore(config.memory_path)

    if args.command == "status":
        list_devices(config)
    elif args.command == "connect":
        connect_device(config, args.target)
    elif args.command == "disconnect":
        disconnect_device(config, args.target)
    elif args.command == "shell":
        run_shell(config, args.device, args.shell_command)
    elif args.command == "memory":
        handle_memory(store, args)
    elif args.command == "scripts":
        if args.scripts_cmd == "run":
            handle_scripts(config, args)
    elif args.command == "panel":
        show_panel(config, store)
    elif args.command == "ai":
        handle_ai(config, store, args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
