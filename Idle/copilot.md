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

总工程师：Kilo Code
