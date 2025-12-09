# ADB-agent-Controller

一个用于在电脑上通过 ADB 控制安卓真机/模拟器的 CLI 原型，集成了基础的“AI 记忆库”和脚本能力，便于你后续接入自定义的 LLM 与 API。

## 功能概览
- **设备管理**：查看设备、连接/断开、执行 `adb shell` 命令。
- **脚本执行**：在 `scripts/` 目录下放置 YAML 脚本，按步骤执行固定操作（含等待与 ADB 命令）。
- **AI 记忆库**：以 JSON 存储对 App/游戏的要点、攻略或流程，可按应用名筛选。
- **AI 辅助规划**：离线的规划占位逻辑，会基于记忆给出执行建议，方便替换为真实 LLM 调用。
- **可配置模型信息**：`config/config.yaml` 支持自定义 provider、model、api_key、base_url、温度等。

## 快速开始（傻瓜式配置）
1. **安装依赖**
   - 电脑已安装 Python 3.10+ 与 `adb`，并且 `adb` 可在终端直接运行。
   - 克隆仓库后进入目录：`cd ADB-agent-Controller`。
   - 安装 Python 依赖：
     ```bash
     pip install -r requirements.txt
     ```
   - 若使用默认的 OpenAI 兼容接口，先导出密钥（可选）：
     ```bash
     export OPENAI_API_KEY=your_key
     ```

2. **检查配置文件**
   - 打开 `config/config.yaml`，根据实际环境修改：
     - `adb.path`：若 `adb` 未加入 PATH，请填写绝对路径。
     - `adb.default_device`：可填常用设备序列号，或留空。
     - `model`：若使用其他网关/模型，直接改 `provider`、`model`、`base_url`、`api_key`、`temperature`。
   - 记忆库默认位置是 `data/memory.json`，脚本目录是 `scripts/`，可按需调整。

3. **最小化验证**（确保环境 OK）
   ```bash
   PYTHONPATH=src python -m adb_agent_controller.cli --help  # 查看可用命令
   PYTHONPATH=src python -m adb_agent_controller.cli status  # 列出已连接设备
   ```

4. **通过 CLI 临时覆盖模型信息（无需改 YAML）**
   ```bash
   PYTHONPATH=src python -m adb_agent_controller.cli --model-provider openai \
     --model-name gpt-4o-mini --model-api-key sk-xxx --model-base-url https://api.openai.com/v1 \
     --model-temperature 0.3 ai "自动登录并领取奖励" --app example
   ```

5. **基础操作示例**
   ```bash
   # 连接 / 断开
   PYTHONPATH=src python -m adb_agent_controller.cli connect emulator-5554
   PYTHONPATH=src python -m adb_agent_controller.cli disconnect emulator-5554

   # 发送 shell 指令（示例：点亮屏幕）
   PYTHONPATH=src python -m adb_agent_controller.cli shell "input keyevent 26" --device emulator-5554

   # 记忆管理
   PYTHONPATH=src python -m adb_agent_controller.cli memory add --app example --note "首关需要跳过引导"
   PYTHONPATH=src python -m adb_agent_controller.cli memory list --app example

   # 运行示例脚本
   PYTHONPATH=src python -m adb_agent_controller.cli scripts run auto_login --device emulator-5554

   # AI 规划占位
   PYTHONPATH=src python -m adb_agent_controller.cli ai "自动登录并领取每日奖励" --app example
   ```

## 配置说明
- `config/config.yaml` 示例：
  ```yaml
  adb:
    path: adb                # adb 可执行文件路径
    default_device: null     # 默认设备序列号，可为 null
    scripts_dir: scripts     # 脚本目录

  model:
    provider: openai
    model: gpt-4o-mini
    api_key: ${OPENAI_API_KEY}
    base_url: https://api.openai.com/v1
    temperature: 0.2

  memory:
    path: data/memory.json   # 记忆库路径
  ```
- 若希望使用自定义模型/网关，可修改 `provider`、`base_url` 和 `model`，代码层面只做占位，方便后续接入。

## 脚本格式
位于 `scripts/` 目录的 YAML 文件形如：
```yaml
description: "示例脚本：打开应用并点击登录"
steps:
  - type: adb
    command: "shell am start -n com.example.app/.MainActivity"
  - type: wait
    wait: 2
  - type: adb
    command: "shell input tap 100 200"
```
支持的 step `type`：
- `adb`/`shell`：执行给定的 adb 子命令（可包含 `shell ...`）。
- `wait`：等待指定秒数。

## 后续扩展建议
- 将 `AIPlanner` 替换为真实的 LLM 调用，结合记忆库生成更细化的操作脚本。
- 增加基于屏幕截图与 CV 的状态感知，动态调整脚本。
- 将 CLI 封装为长驻服务，暴露 WebSocket/HTTP API，便于其他工具（如 Claude/Script CLI）调用。
- 引入脚本模板与参数化，避免硬编码坐标。
