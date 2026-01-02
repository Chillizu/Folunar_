#!/usr/bin/env python3
"""
AgentContainer 主入口文件
重构后的简化版本，使用模块化架构
"""

import yaml
import uvicorn
import logging
import asyncio
from src.initialization import initialize_application, shutdown_application
from src.app import create_application

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.handlers.RotatingFileHandler(
            'logs/main.log',
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
    ]
)
logger = logging.getLogger(__name__)

def load_config():
    """加载配置文件"""
    with open('config.yaml', 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

async def create_app_with_components():
    """创建包含组件初始化的应用"""
    config = load_config()
    initializer = await initialize_application(config)
    app = create_application(config, initializer.get_all_components())
    return app, initializer

def create_app():
    """返回FastAPI应用实例（用于工厂模式）"""
    # 注意：这个函数用于uvicorn工厂模式，但异步初始化需要特殊处理
    # 在实际部署时，建议使用异步主函数
    logger.warning("使用同步create_app函数，组件可能未正确初始化")
    config = globals().get('config')
    if config is None:
        config = load_config()

    # 创建一个基本的应用实例（组件将在startup事件中初始化）
    from fastapi import FastAPI
    app = FastAPI(
        title=config['app']['name'],
        version=config['app']['version'],
        description=config['app']['description']
    )

    # 存储配置和初始化器
    app.state.config = config
    app.state.initializer = None

    @app.on_event("startup")
    async def startup_event():
        """异步初始化组件"""
        nonlocal app
        try:
            initializer = await initialize_application(config)
            app.state.initializer = initializer

            # 重新创建完整的应用
            new_app = create_application(config, initializer.get_all_components())

            # 将路由和中间件复制到当前应用
            app.routes.extend(new_app.routes)
            app.middleware_stack = new_app.middleware_stack
            app.exception_handlers.update(new_app.exception_handlers)
            app.state.limiter = new_app.state.limiter

            logger.info("应用组件异步初始化完成")
        except Exception as e:
            logger.error(f"应用初始化失败: {str(e)}")
            raise

    @app.on_event("shutdown")
    async def shutdown_event():
        """清理组件"""
        if app.state.initializer:
            await shutdown_application(app.state.initializer)

    return app

async def main():
    """异步主函数"""
    config = load_config()

    try:
        # 初始化应用组件
        logger.info("开始初始化AgentContainer...")
        initializer = await initialize_application(config)

        # 创建应用
        app = create_application(config, initializer.get_all_components())

        # 配置服务器
        server_config = config.get('server', {})
        server_kwargs = {
            "host": server_config.get('host', '0.0.0.0'),
            "port": server_config.get('port', 8000),
            "reload": False,  # 生产环境禁用reload
            "access_log": True,
            "log_level": "info"
        }

        # 启动服务器
        logger.info(f"启动服务器: {server_kwargs['host']}:{server_kwargs['port']}")
        config = uvicorn.Config(app, **server_kwargs)
        server = uvicorn.Server(config)

        try:
            await server.serve()
        except KeyboardInterrupt:
            logger.info("收到中断信号，正在关闭...")
        finally:
            await shutdown_application(initializer)

    except Exception as e:
        logger.error(f"应用启动失败: {str(e)}")
        raise

if __name__ == "__main__":
    # 支持两种启动方式：
    # 1. 异步方式（推荐）：python main.py
    # 2. uvicorn工厂方式：uvicorn main:create_app --factory

    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--factory":
        # uvicorn工厂模式
        uvicorn.run("main:create_app", factory=True, host="0.0.0.0", port=8000)
    else:
        # 异步启动（推荐）
        asyncio.run(main())
