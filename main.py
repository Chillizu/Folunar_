#!/usr/bin/env python3
"""
AgentContainer 主入口文件
"""

import yaml
import uvicorn
from fastapi import FastAPI
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

    return app

if __name__ == "__main__":
    config = load_config()
    app = create_app()

    uvicorn.run(
        app,
        host=config['server']['host'],
        port=config['server']['port'],
        reload=config['server']['debug']
    )