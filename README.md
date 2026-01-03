# AgentContainer

AgentContainer 是一个围绕「观察 → 决策 → 执行」闭环的 AI 沙盒平台，提供 OpenAI 兼容接口、安全隔离的 Debian 容器、可视化监控面板以及审计/性能观测，方便在可控环境中运行与观察 AI 代理。

## 功能概览
- OpenAI 兼容 `/v1/chat/completions`（流式/非流式）
- Web UI：仪表盘 `/` 与聊天页 `/chat`
- 容器管理：启动/停止/状态/执行命令、资源与审计日志
- 观察者：截图、日志聚合；决策引擎：指令生成与随机思绪注入
- 安全：JWT、速率限制、安全头、命令过滤、容器资源与网络限制
- 性能与测试：缓存/连接池、指标监控，pytest 覆盖单元+集成

## 架构概览
```
Web UI  <->  Decision Engine  <->  Container Manager
   |              |                     |
Security     Observer/Whisper       Debian Sandbox
```

## 快速启动
### Docker compose（推荐，一键 API + Debian 沙盒）
```bash
./scripts/run-compose.sh
```
或 PowerShell:
```powershell
pwsh -File scripts/run-compose.ps1
```
- 会准备 `config.yaml`、`logs`、`data`（含 `data/sandbox`）并以 docker compose 构建/启动  
- `agentcontainer-api`：FastAPI 主服务  
- `agentcontainer-sandbox`：内置 Debian 12 沙盒（默认工作目录 `/workspace`，挂载到 `./data/sandbox`）

### 本地运行（仅主服务）
```bash
chmod +x scripts/run-local.sh
./scripts/run-local.sh
```
或 PowerShell:
```powershell
.\scripts\run-local.ps1
```

## 手动容器命令（可选）
```bash
docker build -t agentcontainer-api:latest .
docker build -t agentcontainer-sandbox:latest -f Dockerfile.sandbox .
docker network create agentcontainer-net || true
docker run -d --name agentcontainer-sandbox agentcontainer-sandbox:latest
docker run -d --name agentcontainer-api -p 8000:8000 \
  -v $(pwd)/config.yaml:/app/config.yaml:ro \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/data:/app/data \
  --network agentcontainer-net \
  --link agentcontainer-sandbox \
  agentcontainer-api:latest
```

## 配置（核心字段）
```bash
cp config.example.yaml config.yaml
```
```yaml
app:
  name: AgentContainer
  version: "1.0.0"

api:
  base_url: "https://api.openai.com/v1"
  default_model: "gpt-3.5-turbo"
  key: "your-openai-api-key"

container:
  image_name: "agentcontainer-sandbox:latest"
  container_name: "agentcontainer-sandbox"
  dockerfile_path: "Dockerfile.sandbox"
  volumes:
    "./data/sandbox": "/workspace"
  ports:
    "5901": "5901"
    "8888": "8888"

security:
  jwt_secret_key: "change-me"
  enable_audit_log: true
```

## 关键入口与测试
- 仪表板：`http://localhost:8000/`
- 聊天：`http://localhost:8000/chat`
- API：`/api/chat/completions`、`/api/system/status`、`/health`
- 测试：`pytest`

## 常见问题
- Docker 无法连接：确认 Docker Desktop 已运行，`docker info` 正常。
- 镜像拉取/构建慢：可配置镜像加速或离线预拉取基础镜像。
- 沙盒网络/权限：默认桥接网络与 no-new-privileges，可按需在 `config.yaml`/`docker-compose.yml` 中调整。
