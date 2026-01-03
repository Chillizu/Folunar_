# AgentContainer - AI 沙盒系统（压缩版）

## 指令

请尽量使用中文回答，然后就是可以用到一些颜文字，也可以装可爱
每次完成一个小阶段的更改后，进行以下操作：查看完成的内容与 git 的状态，把各个小阶段和可以归为一类的更改分别分步提交，并且不同的大分类的功能分别提交到不同的分支上（例如用户界面和核心功能可以分为两个分支）。由于项目经常需要跨客户端进行，所以请你频繁地将你的进度，计划，所有理解和必要的上下文等内容全部一直实时并且详细地更新在 Idle 文件夹下的 copilot.md 里，方便其他客户端的 AI 无缝继续任务，作为总工程师请你也要求其他客户端的 AI 进行同样的操作，这一点可以在 copilot.md 里说明

## 项目概览

AgentContainer 是一个 AI 沙盒系统，按「观察层 → 决策层 → 执行层」闭环运行。系统提供 OpenAI 兼容的聊天接口、容器管理、安全隔离、审计日志与 Web 监控界面，用于让 AI 在受控容器环境中执行任务并可追踪。

## 当前状态（已完成）

- FastAPI 模块化应用 + 统一错误处理
- JWT 认证、输入校验、安全中间件（CSP/HSTS/敏感头过滤）
- OpenAI 兼容 chat completions（流式/非流式）
- Docker 容器管理 + 状态监控 + 审计日志
- 观察者（截图 + 视觉分析）、决策引擎、随机想法注入
- 连接池、缓存、性能监控
- Web UI（主页仪表板 `/` 与聊天页 `/chat`）
- 完整测试（单元/集成/闭环/性能）

## 最新进度（本轮）
- 重写 README：补充功能概览、部署方式（本地/Docker/多 Dockerfile 变体）、配置与常见问题
- 清理与整理仓库：补充 `.gitignore`（logs/screenshot/.pytest_cache），恢复 `Idle/agents.md` 协作说明
- 准备提交：阶段性变更将统一暂存并提交
- README 使用方式更新：新增 uv / pipx 安装与运行流程说明
- 依赖解析修正：将 requires-python 提升为 >=3.10 以匹配 Pillow 12.0.0 等依赖
- 静态资源兼容：为 /styles.css 与 /chat.js 添加兼容路由，避免旧页面引用导致样式缺失
- WebSocket 路由补齐：新增 `/ws/sandbox` 推送沙盒状态与占位日志，避免前端控制台报错
- README 说明澄清：main.py 不会自动启动容器，配置步骤改为“最小可用 + 容器可选”
- README 重新整理：仅保留容器化部署方式，收敛使用路径
- 增加本地一键运行脚本（scripts/run-local.ps1 + scripts/run-local.sh）

## 技术栈

- 后端：FastAPI + Python 3.13 + Pydantic v2
- AI：OpenAI API（兼容模式）
- 容器：Docker
- 前端：HTML/CSS/JS
- 测试：pytest + pytest-asyncio

## 运行与测试

```bash
python main.py
```

或

```bash
uvicorn main:create_app --factory --reload
```

测试：

```bash
pytest
```

## 关键入口

- `/`：可视化仪表板
- `/chat`：聊天界面
- `/api/chat/completions`：OpenAI 兼容聊天
- `/api/system/status`：系统状态
- `/health`：健康检查

## 协作规则（必读）

每次完成一个小阶段后必须更新此文件，记录：

- 当前进度与已完成事项
- 接下来计划与风险
- 重要决策与上下文

## 本轮进度
- 阅读 agents.md、summary.md、README 与 copilot.md，确认现状（核心功能已 95% 完成，已具备 OpenAI 兼容接口、容器管理、Web UI、随机思绪注入等）。
- 明确用户目标：交付可直接部署的整合版本，包含一键启动脚本、Web 监视面板、内置可供 AI 操作的 Debian 容器，以及 summary 中提到的全部能力整合。
- 完成容器化基线：新增 docker-compose.yml（API + Debian 沙盒）、重写 Dockerfile.sandbox（Debian 12 轻量版）、修复主 Dockerfile 依赖安装与 entrypoint 逻辑，默认网络/镜像名与沙盒对齐。
- 补全一键脚本与配置：新增 scripts/run-compose.sh & .ps1，更新 run-local 脚本创建 logs/data 目录，引入 requirements.txt，容器配置默认指向 agentcontainer-sandbox，.gitignore 增加 data/sandbox/。
- 更新 README：改写部署指引（compose 优先、手动命令可选）、容器说明与配置示例。
- Github Action 暂停自动触发：将 CI workflow 改为仅 workflow_dispatch 手动执行。
- README PowerShell 指令改为添加 ExecutionPolicy Bypass，适配部分环境无法直接执行 pwsh 的情况。

## 下一步计划
- 后端链路强化：梳理/优化 ContainerManager 与 API 路由，确保 sandbox Exec/状态/日志/截图流闭环，校准 network_mode 与 Compose 网络。
- 前端监控与 WebSocket：完善 `/ws/sandbox` 推送内容（状态/日志/截图占位），监控页 UI 行为对齐。
- 安全与随机思绪：审视命令过滤/白名单、随机注入路径与日志；补充配置切换与测试。
- 文档与交付：补充 README/部署指南中的端到端流程、常见故障与离线方案。

## 关键决策与上下文
- 交流语言：按要求使用中文沟通，必要时可穿插 Emoji 标记阶段。
- 交付形态：优先保证可直接部署（Docker/Compose）与本地一键脚本，并确保内置 Debian 沙盒可被 AI 全权操作。
- 版本管理：阶段性更新 copilot.md，不同大类功能后续拆分分支提交。

## 风险与上下游
- Docker 网络/镜像拉取可能受限，需准备加速或离线镜像方案。
- 本地与容器化路径需同时兼容（配置加载、静态资源路由）。
- Web 监控与沙盒执行链路需要充分测试，避免长连接或权限问题导致不稳定。

## 规划概要
- 交付目标：可直接部署（Docker/Compose）、本地/容器一键启动脚本、内置可供 AI 操作的 Debian 沙盒、Web 监控面板、整合 summary 中的全部能力。
- 架构蓝图：Compose 统一编排 FastAPI 核心服务 + Debian 沙盒（带必要工具与权限隔离）；后台统一通过 ContainerManager 执行/流式日志；Observer 提供截图/日志输入，Decision Engine 产生命令，Executor 通过 Docker exec 执行；WebSocket 推送状态/监控流至前端。
- 实施阶段：① 基线梳理/依赖与配置统一；② 沙盒镜像与 Compose（含加速/离线选项）+ 一键启动脚本；③ 后端管控链路（命令执行、状态/日志/截图流）；④ 前端监控页与聊天页整合；⑤ 随机思绪注入与安全加固（网络/权限/审计）；⑥ 测试矩阵（单测/集成/端到端/性能）与 CI；⑦ 文档与交付包（README、脚本说明、镜像分发方案）。

总工程师：Kilo Code
