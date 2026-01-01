# AgentContainer - AI操作系统容器框架

## 项目概述

AgentContainer是一个基于Python的框架，旨在创建一个AI操作系统容器，具有以下核心功能：

1. **OpenAI兼容的对话API**：提供与OpenAI API兼容的聊天完成端点
2. **实时流式输出**：支持流式响应以实现实时交互
3. **容器管理**：集成Docker容器管理功能
4. **Web面板**：提供用户界面以监控和交互
5. **安全容器环境**：在隔离的Linux容器中运行AI操作

## 主要特性

### 1. OpenAI兼容API
- 完全兼容OpenAI的聊天完成API
- 支持流式和非流式响应
- 可配置的API密钥和端点

### 2. 容器管理
- 创建和管理Docker容器
- 实时监控容器状态
- 容器日志和统计信息
- 安全隔离环境

### 3. Web界面
- 实时聊天界面
- 容器状态监控面板
- 系统日志和指标

### 4. 安全性
- 容器隔离
- 资源限制
- 安全的API访问控制

## 快速开始

### 先决条件
- Python 3.9+
- Docker
- OpenAI API密钥（可选，用于测试）

### 安装

```bash
# 克隆仓库
git clone https://github.com/yourusername/agentcontainer.git
cd agentcontainer

# 安装依赖
pip install -r requirements.txt

# 复制配置文件
cp config.example.yaml config.yaml

# 编辑配置文件
nano config.yaml
```

### 配置

编辑`config.yaml`文件：

```yaml
# API配置
api:
  key: "your-openai-api-key"  # 可选，用于测试
  base_url: "https://api.openai.com/v1"
  model: "gpt-3.5-turbo"

# 服务器配置
server:
  host: "0.0.0.0"
  port: 8000
  debug: true

# 容器配置
container:
  name: "agent-debian"
  image: "debian:bullseye"
  auto_start: true
  resources:
    memory_limit: "512m"
    cpu_limit: 0.5

# 日志配置
logging:
  level: "info"
  file: "agentcontainer.log"
```

### 运行项目

```bash
# 启动服务器
python main.py

# 服务器将在http://localhost:8000运行
```

### 访问Web界面

打开浏览器并访问：
- 主界面：http://localhost:8000
- 容器监控：http://localhost:8000/api/container/monitor
- 聊天API：http://localhost:8000/api/chat/completions

## API端点

### 聊天完成

**POST** `/api/chat/completions`

请求体：
```json
{
  "model": "gpt-3.5-turbo",
  "messages": [
    {"role": "user", "content": "Hello!"}
  ],
  "stream": true
}
```

响应：
```json
{
  "id": "chatcmpl-xxx",
  "object": "chat.completion.chunk",
  "created": 1234567890,
  "model": "gpt-3.5-turbo",
  "choices": [
    {
      "delta": {"content": "Hello! How can I help you today?"},
      "index": 0,
      "finish_reason": null
    }
  ]
}
```

### 容器管理

**GET** `/api/container/monitor` - 实时容器统计信息

**POST** `/api/container/start` - 启动容器

**POST** `/api/container/stop` - 停止容器

**GET** `/api/container/logs` - 获取容器日志

## 架构

```
AgentContainer
├── config.yaml          # 配置文件
├── main.py              # 主应用入口
├── requirements.txt     # 依赖项
├── Dockerfile           # 容器定义
├── 
├── src/
│   ├── container_manager.py  # 容器管理
│   └── core/
│       ├── agent_manager.py   # AI代理管理
│       └── __init__.py
├── static/              # Web静态文件
│   ├── chat.js          # 聊天界面JavaScript
│   ├── index.html       # 主页面
│   └── styles.css       # 样式表
└── README.md            # 项目文档
```

## 安全注意事项

1. **容器隔离**：所有AI操作都在隔离的Docker容器中运行
2. **资源限制**：容器有内存和CPU限制
3. **API安全**：确保API密钥安全存储
4. **网络安全**：使用防火墙保护端口

## 贡献

欢迎贡献！请遵循以下步骤：

1. Fork项目
2. 创建您的特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交您的更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开Pull Request

## 许可证

本项目采用MIT许可证 - 详情请见LICENSE文件。

## 联系方式

如有任何问题或建议，请联系：
- 电子邮件：support@agentcontainer.com
- GitHub Issues：https://github.com/yourusername/agentcontainer/issues

## 未来计划

1. 添加更多AI模型支持
2. 改进容器管理功能
3. 添加用户认证
4. 扩展监控功能
5. 支持多容器编排

---

© 2024 AgentContainer. All rights reserved.
