# AgentContainer
AgentContainer 是一个面向 AI 操作闭环的沙箱平台，通过观察 → 决策 → 执行的三层流线，在隔离的容器中安全运行代理任务。

## 为什么选择 AgentContainer？
- **闭环调度**：观察组件记录沙箱状态，决策引擎生成结构化命令，ContainerManager 在 Docker 容器内执行并反馈结果。  
- **兼容 OpenAI**：提供与 `/v1/chat/completions` 接口一致的流式/非流式响应，支持工具调用、流式 chunk 传输以及性能优化。  
- **安全驱动**：JWT 登录、统一限流、CSP/HSTS 标头、敏感数据过滤、多层审计日志和资源隔离共同保护宿主与容器。  
- **性能与鲁棒**：连接池、异步任务、缓存、健康监控以及详细测试（集成、闭环、性能）提升系统可用性。  
- **易扩展**：可插拔的观察者、决策策略、Whisper 注入、容器管理器与前端控制，便于快速演进。

## 架构概览

```
        +-----------+          +------------------+          +---------------------+
        |  Web UI   |  <--->   |  Decision Engine |  <--->   | Container Manager   |
        | (FastAPI) |          |   (Agent Brain)  |          | (Docker sandboxing) |
        +-----------+          +------------------+          +---------------------+
               ^                        ^                             ^
               |                        |                             |
               +----------+-------------+-------------+---------------+
                          |                           |
                    安全层服务                 支撑服务模块
                  (JWT、限流、头部、           (缓存、连接池、
                   审计日志等)                 性能监控等)
```

## 快速开始

### 准备条件
- Python 3.10+  
- Docker（用于实际容器执行）  
- 可选：OpenAI/API 兼容密钥，用于真实模型调用或测试流式返回

### 安装步骤
```bash
git clone https://github.com/yourusername/AgentContainer.git
cd AgentContainer
python -m venv .venv
.venv\\Scripts\\activate
pip install -U pip
pip install -r requirements.txt
cp config.example.yaml config.yaml
```

### 配置说明
编辑 `config.yaml`，重点字段包括：

```yaml
app:
  name: AgentContainer
  version: "1.0.0"
  description: "AI 控制型沙箱编排平台"

api:
  key: "your-openai-api-key"   # 测试或在线接入
  base_url: "https://api.openai.com/v1"
  default_model: "gpt-3.5-turbo"
  timeout: 30

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

FastAPI 应用将自动读取 `config.yaml`，项目同时提供 `config.example.yaml` 作为安全的示例。

### 启动服务
```bash
python main.py
```

或在开发中使用 uvicorn 工厂模式（支持热重载）：
```bash
uvicorn main:create_app --factory --reload
```

默认监听 `http://localhost:8000`。

## API 概览

| 路径 | 方法 | 说明 |
|------|------|------|
| `/api/chat/completions` | POST | 兼容 OpenAI 的流式 / 非流式聊天接口，调用 AgentManager 与 OpenAI 模型。 |
| `/api/system/status` | GET | 提供缓存的 CPU/内存/Agent 数量 + 连接池状态。 |
| `/health` | GET | 健康检查（受统一 limiter 保护）。 |
| `/api/container/*` | POST/GET | 容器生命周期、日志、统计与 exec 命令，运行在隔离沙箱内。 |
| `/test/streaming` | GET | 模拟 OpenAI 流式响应，用于前端测试客户端。 |

认证：通过 `/api/auth/login` 获取 JWT，后续请求携带 `Authorization: Bearer <token>`。

## 测试指南

```bash
pytest
```

测试覆盖 AgentManager、ContainerManager、安全模块及系统集成。当前 FastAPI `@app.on_event` 处有弃用警告（建议未来迁移到生命周期 `lifespan`），Pydantic 也提示 `dict` 方法可替代为 `model_dump`。

## 安全与硬化

- Docker 容器沙箱 + 限定的网络、资源与权限。  
- 全局 `slowapi.Limiter` 实例统一限流，避免引入多份装饰器。  
- 审计日志严格过滤敏感头，并递归清理不可序列化对象。  
- JWT 登录、敏感数据中间件、CSP/HSTS/HTTPS 配置、日志记录线上辨识能力。

## 贡献流程

1. Fork 本仓库  
2. 建立 feature 分支（例如：`feature/优化容器状态`）  
3. 本地运行 `pytest`，确认全部测试通过  
4. 提交变更并 push  
5. 创建 Pull Request

## 许可证

MIT © AgentContainer 项目

## 联系方式

如有问题或建议，请通过 Issue 或邮箱：
- Email: support@agentcontainer.com  
- GitHub Issues: https://github.com/yourusername/AgentContainer/issues
