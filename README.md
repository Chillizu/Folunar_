# AgentContainer

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-green.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Supported-blue.svg)](https://www.docker.com/)

AgentContainer 是一个用于管理和运行AI代理的容器化平台，提供完整的容器生命周期管理和AI聊天功能。

## ✨ 特性

- 🐳 **容器管理**: 完整的Docker容器生命周期管理（构建、启动、停止、删除）
- 🤖 **AI代理**: 支持多种AI模型的代理管理
- 💬 **聊天界面**: 内置Web聊天界面，支持流式响应
- 🔄 **实时监控**: 容器统计信息实时监控
- 📡 **OpenAI兼容API**: 完全兼容OpenAI Chat Completions API
- ⚡ **高性能**: 基于FastAPI的异步架构
- 🛠️ **易扩展**: 模块化设计，易于扩展新功能

## 🚀 快速开始

### 系统要求

- Python 3.8+
- Docker
- Git

### 安装步骤

1. **克隆项目**
   ```bash
   git clone https://github.com/your-username/agent-container.git
   cd agent-container
   ```

2. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

3. **配置环境**
   ```bash
   cp config.example.yaml config.yaml
   # 编辑 config.yaml，设置你的API密钥和其他配置
   ```

4. **启动服务**
   ```bash
   python main.py
   ```

5. **访问界面**
   - Web界面: http://localhost:8000/chat
   - API文档: http://localhost:8000/docs

### Docker部署

```bash
# 构建镜像
docker build -t agent-container .

# 运行容器
docker run -p 8000:8000 -v $(pwd)/config.yaml:/app/config.yaml agent-container
```

## 📖 使用说明

### Web界面使用

1. 打开浏览器访问 http://localhost:8000/chat
2. 在输入框中输入消息
3. 点击发送或按Enter键
4. 系统状态面板显示实时信息

### API使用

#### 聊天完成 (兼容OpenAI)

```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
     -H "Content-Type: application/json" \
     -d '{
       "model": "gpt-3.5-turbo",
       "messages": [{"role": "user", "content": "Hello!"}],
       "stream": true
     }'
```

#### 容器管理

```bash
# 构建容器
curl -X POST "http://localhost:8000/api/container/build"

# 启动容器
curl -X POST "http://localhost:8000/api/container/start"

# 获取容器状态
curl "http://localhost:8000/api/container/status"

# 停止容器
curl -X POST "http://localhost:8000/api/container/stop"
```

### 配置文件

创建 `config.yaml` 文件，参考 `config.example.yaml`：

```yaml
app:
  name: AgentContainer
  version: "1.0.0"
  description: "A container for managing and running AI agents"

server:
  host: "0.0.0.0"
  port: 8000
  debug: true

api:
  base_url: "https://openrouter.ai/api/v1"
  default_model: "nvidia/nemotron-nano-12b-v2-vl:free"
  key: "your-api-key-here"

container:
  image_name: "debian-container"
  container_name: "agent-debian"
  dockerfile_path: "Dockerfile"
```

## 📚 API文档

### 核心端点

#### 系统相关
- `GET /` - 根端点，返回欢迎信息
- `GET /health` - 健康检查
- `GET /api/system/status` - 获取系统状态信息

#### 代理管理
- `GET /agents` - 列出所有代理

#### 容器管理
- `POST /api/container/build` - 构建容器镜像
- `POST /api/container/start` - 启动容器
- `POST /api/container/stop` - 停止容器
- `POST /api/container/remove` - 删除容器
- `GET /api/container/status` - 获取容器状态
- `GET /api/container/monitor` - 实时监控容器统计信息
- `POST /api/container/exec` - 在容器中执行命令

#### 聊天API (OpenAI兼容)
- `POST /v1/chat/completions` - 聊天完成，支持流式和非流式响应

### 请求示例

#### 流式聊天
```python
import requests
import json

response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Hello!"}],
        "stream": True
    },
    stream=True
)

for line in response.iter_lines():
    if line:
        line = line.decode('utf-8')
        if line.startswith('data: '):
            data = line[6:]
            if data == '[DONE]':
                break
            chunk = json.loads(data)
            print(chunk)
```

#### 容器操作
```python
import requests

# 启动容器
response = requests.post("http://localhost:8000/api/container/start")
print(response.json())

# 执行命令
response = requests.post(
    "http://localhost:8000/api/container/exec",
    json={"command": "ls -la"}
)
print(response.json())
```

## 🛠️ 开发指南

### 项目结构

```
agent-container/
├── main.py                 # 主入口文件
├── config.yaml            # 配置文件
├── requirements.txt       # Python依赖
├── pyproject.toml         # 项目配置
├── Dockerfile             # Docker镜像构建文件
├── src/
│   ├── container_manager.py    # 容器管理器
│   └── core/
│       ├── __init__.py
│       └── agent_manager.py    # 代理管理器
├── static/                # 前端静态文件
│   ├── index.html
│   ├── styles.css
│   └── chat.js
├── tests/                 # 测试文件
├── logs/                  # 日志文件
└── plans/                 # 项目计划文档
```

### 开发环境设置

1. **安装开发依赖**
   ```bash
   pip install -e .
   pip install pytest pytest-asyncio pytest-mock
   ```

2. **运行测试**
   ```bash
   pytest
   ```

3. **代码格式化**
   ```bash
   # 安装 black 和 isort
   pip install black isort

   # 格式化代码
   black .
   isort .
   ```

### 扩展开发

#### 添加新的AI提供商

1. 在 `src/core/agent_manager.py` 中添加新的提供商类
2. 实现 `chat_completion` 方法
3. 在配置中添加提供商设置

#### 添加新的容器操作

1. 在 `src/container_manager.py` 中添加新方法
2. 在 `main.py` 中添加对应的API端点
3. 更新API文档

### 贡献指南

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建 Pull Request

### 代码规范

- 使用 Black 进行代码格式化
- 使用 isort 进行导入排序
- 编写完整的类型注解
- 为新功能编写测试
- 更新文档

## 📋 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🤝 贡献

欢迎贡献！请查看我们的 [贡献指南](CONTRIBUTING.md)。

## 📞 支持

如果您有问题或建议，请：

1. 查看 [问题跟踪](https://github.com/your-username/agent-container/issues)
2. 创建新问题
3. 联系维护者

## 🙏 致谢

- [FastAPI](https://fastapi.tiangolo.com/) - 现代化的Python Web框架
- [Docker](https://www.docker.com/) - 容器化平台
- [OpenAI](https://openai.com/) - AI模型和API

---

**注意**: 这是一个开发中的项目。API可能会发生变化，请及时更新您的代码。