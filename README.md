# AgentContainer

AgentContainer 是一个面向 AI 代理的「观察 → 决策 → 执行」沙盒平台，提供安全容器、OpenAI 兼容接口与可视化监控，帮助你在可控环境中运行 AI 自动化任务。

## 能做什么
- OpenAI 兼容的 `/v1/chat/completions`（流式/非流式）
- AI 观察者：截图、视觉分析、观察记录
- 决策引擎：基于观察生成可执行命令
- Docker 容器管理：启动/停止/执行/状态监控
- 安全组件：JWT 登录、限流、CSP/HSTS、安全头、审计日志
- 监控面板：主页仪表板 `/` + 聊天界面 `/chat`

## 架构概览

```
Web UI  <->  Decision Engine  <->  Container Manager
   |              |                     |
Security     Observer/Whisper       Docker Sandbox
```

## 快速开始（本地）

### 1) 环境准备
- Python 3.10+
- Docker（用于容器执行）

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

如果你希望使用 pipx 统一管理工具：
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

或在开发中使用：
```bash
uvicorn main:create_app --factory --reload
```

默认访问：
- 仪表板：`http://localhost:8000/`
- 聊天页面：`http://localhost:8000/chat`

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

### 变体镜像（按需）
```bash
# 本地调试
docker build -f Dockerfile.local -t agentcontainer:local .

# 离线构建
docker build -f Dockerfile.offline -t agentcontainer:offline .

# 沙盒增强
docker build -f Dockerfile.sandbox -t agentcontainer:sandbox .
```

### 环境变量
```bash
docker run -d -p 9000:9000 \
  -e CONFIG_FILE=/app/prod.yaml \
  -e PORT=9000 \
  agentcontainer:latest
```

## 配置说明
关键字段示例（`config.yaml`）：

```yaml
app:
  name: AgentContainer
  version: "1.0.0"
  description: "AI 沙盒系统"

api:
  key: "your-openai-api-key"
  base_url: "https://api.openai.com/v1"
  default_model: "gpt-3.5-turbo"

container:
  image_name: "debian:bookworm"
  container_name: "agent-container"
  dockerfile_path: "Dockerfile"
  network_mode: "bridge"

security:
  jwt_secret_key: "change-this-to-a-secret"
  enable_audit_log: true
  cors_origins: ["*"]
```

## 主要接口
- `/`：主页仪表板（HTML）
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
