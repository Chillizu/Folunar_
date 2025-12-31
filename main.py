#!/usr/bin/env python3
"""
AgentContainer 主入口文件
"""

import yaml
import uvicorn
import json
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from src.core.agent_manager import AgentManager

def load_config():
    """加载配置文件"""
    with open('config.yaml', 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def create_app():
    """创建FastAPI应用"""
    config = load_config()

    app = FastAPI(
        title=config['app']['name'],
        version=config['app']['version'],
        description=config['app']['description']
    )

    # 初始化代理管理器
    agent_manager = AgentManager(config)

    @app.get("/")
    async def root():
        return {"message": "Welcome to AgentContainer", "version": config['app']['version']}

    @app.get("/agents")
    async def list_agents():
        return {"agents": agent_manager.list_agents()}

    @app.get("/health")
    async def health_check():
        return {"status": "healthy"}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        """处理chat completions请求，支持流式和非流式响应"""
        data = await request.json()
        messages = data.get("messages", [])
        model = data.get("model", "gpt-3.5-turbo")
        stream = data.get("stream", False)
        tools = data.get("tools")

        if stream:
            # 流式响应
            async def generate():
                async for chunk in agent_manager.chat_completion(
                    messages=messages,
                    model=model,
                    stream=True,
                    tools=tools
                ):
                    yield f"data: {json.dumps({'choices': [{'delta': {'content': chunk}}]})}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(
                generate(),
                media_type="text/plain",
                headers={"Content-Type": "text/plain; charset=utf-8"}
            )
        else:
            # 非流式响应
            async for result in agent_manager.chat_completion(
                messages=messages,
                model=model,
                stream=False,
                tools=tools
            ):
                return result

    return app

if __name__ == "__main__":
    config = load_config()

    uvicorn.run(
        "main:create_app",
        factory=True,
        host=config['server']['host'],
        port=config['server']['port'],
        reload=config['server']['debug']
    )