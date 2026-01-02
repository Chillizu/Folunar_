# AgentContainer

AgentContainer 是一个面向 AI 代理的「观察 → 决策 → 执行」沙盒平台，提供安全容器、OpenAI 兼容接口与可视化监控，帮助你在可控环境中运行 AI 自动化任务。

## 能做什么
- OpenAI 兼容 `/v1/chat/completions`（流式/非流式）
- AI 观察者：截图、视觉分析、观察记录
- 决策引擎：基于观察生成可执行命令
- Docker 容器管理：启动/停止/执行/状态监控
- 安全组件：JWT、限流、CSP/HSTS、安全头、审计日志
- 可视化面板：`/` 仪表板 + `/chat` 聊天界面

## 架构概览
```
Web UI  <->  Decision Engine  <->  Container Manager
   |              |                     |
Security     Observer/Whisper       Docker Sandbox
```

## 运行方式说明（关键问题）
`main.py` **不会**自动启动 Debian 容器。  
容器由 `ContainerManager` 在调用相关 API（如启动/执行命令）时才会与 Docker 交互。  
因此：
- 本地运行 `main.py` 只启动 API 服务与监控逻辑
- 只有在你调用容器相关接口（例如 `/api/container/start`）时才需要 Docker 可用

## 快速开始（本地）

### 1) 环境准备
- Python 3.10+
- Docker（仅在需要容器执行时才必须启用）

### 2) 安装依赖（推荐：uv）
```bash
git clone https://github.com/yourusername/AgentContainer.git
cd AgentContainer

# 安装 uv（Windows PowerShell）
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# 安装依赖
uv sync
cp config.example.yaml config.yaml
```

使用 pipx 安装 uv：
```bash
pipx install uv
uv sync
```

传统方式（venv + pip）：
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
cp config.example.yaml config.yaml
```

### 3) 启动服务
```bash
python main.py
```

或：
```bash
uvicorn main:create_app --factory --reload
```

默认访问：
- 仪表板：`http://localhost:8000/`
- 聊天页面：`http://localhost:8000/chat`

## 配置步骤（清晰版）
配置文件是 `config.yaml`。它**只告诉服务如何连接**（不会自动启动容器）。你需要：

1. 复制示例配置  
   `cp config.example.yaml config.yaml`

2. 根据实际需求填写关键字段（最小可用配置）：
```yaml
app:
  name: AgentContainer
  version: "1.0.0"
  description: "AI 沙盒系统"

api:
  key: "your-openai-api-key"   # 没有可留空，但聊天将无法请求真实模型
  base_url: "https://api.openai.com/v1"
  default_model: "gpt-3.5-turbo"

security:
  jwt_secret_key: "change-this-to-a-secret"
  enable_audit_log: true
```

3. 如果需要使用容器执行能力，再补充 container 部分：
```yaml
container:
  image_name: "debian:bookworm"
  container_name: "agent-container"
  dockerfile_path: "Dockerfile"
  network_mode: "bridge"
```

## Docker 一体化部署

### 标准部署（推荐）
```bash
docker build -t agentcontainer:latest .
docker run -d --name agentcontainer -p 8000:8000 \
  -v $(pwd)/config.yaml:/app/config.yaml \
  agentcontainer:latest
```

Windows PowerShell：
```powershell
docker build -t agentcontainer:latest .
docker run -d --name agentcontainer -p 8000:8000 `
  -v ${PWD}/config.yaml:/app/config.yaml `
  agentcontainer:latest
```

### Dockerfile 变体（按需）
```bash
docker build -f Dockerfile.local -t agentcontainer:local .
docker build -f Dockerfile.offline -t agentcontainer:offline .
docker build -f Dockerfile.sandbox -t agentcontainer:sandbox .
```

### 环境变量
```bash
docker run -d -p 9000:9000 \
  -e CONFIG_FILE=/app/prod.yaml \
  -e PORT=9000 \
  agentcontainer:latest
```

## 主要接口
- `/`：仪表板（HTML）
- `/chat`：聊天界面
- `/api/chat/completions`：OpenAI 兼容聊天
- `/api/system/status`：系统状态
- `/health`：健康检查

## 测试
```bash
pytest
```

## 常见问题
1) Docker 无法连接  
请确认 Docker Desktop 已启动，`docker info` 返回正常。

2) Docker 拉取镜像失败  
网络访问受限时，请使用内网镜像或预下载镜像后再构建。

## 目录结构（简略）
```
src/            核心逻辑
static/         Web 前端资源
tests/          测试
config.example.yaml  示例配置
```

## License
MIT
