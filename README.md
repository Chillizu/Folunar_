# AgentContainer

AgentContainer 是一个面向 AI 代理的「观察 → 决策 → 执行」沙盒平台，提供安全容器、OpenAI 兼容接口与可视化监控，帮助你在可控环境中运行 AI 自动化任务。

## 功能概览
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

## 容器化部署（唯一方式）

### 1) 构建镜像
```bash
docker build -t agentcontainer:latest .
```

### 2) 运行容器
Linux/macOS:
```bash
docker run -d --name agentcontainer -p 8000:8000 \
  -v $(pwd)/config.yaml:/app/config.yaml \
  agentcontainer:latest
```

Windows PowerShell:
```powershell
docker run -d --name agentcontainer -p 8000:8000 `
  -v ${PWD}/config.yaml:/app/config.yaml `
  agentcontainer:latest
```

### 3) 访问入口
- 仪表板：`http://localhost:8000/`
- 聊天页面：`http://localhost:8000/chat`

## 配置说明（容器内生效）

1. 复制示例配置：
```bash
cp config.example.yaml config.yaml
```

2. 最小可用配置：
```yaml
app:
  name: AgentContainer
  version: "1.0.0"
  description: "AI 沙盒系统"

api:
  key: "your-openai-api-key"
  base_url: "https://api.openai.com/v1"
  default_model: "gpt-3.5-turbo"

security:
  jwt_secret_key: "change-this-to-a-secret"
  enable_audit_log: true
```

3. 需要容器执行能力时再补充：
```yaml
container:
  image_name: "debian:bookworm"
  container_name: "agent-container"
  dockerfile_path: "Dockerfile"
  network_mode: "bridge"
```

## 主要接口
- `/`：仪表板（HTML）
- `/chat`：聊天界面
- `/api/chat/completions`：OpenAI 兼容聊天
- `/api/system/status`：系统状态
- `/health`：健康检查

## 常见问题
1) Docker 无法连接  
确认 Docker Desktop 已启动，`docker info` 返回正常。

2) 镜像拉取失败  
网络受限时请配置镜像加速或使用内网镜像。

## License
MIT
