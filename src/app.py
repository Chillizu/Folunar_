# 应用工厂模块
"""
AgentContainer应用工厂
创建和配置FastAPI应用实例
"""

import logging
from typing import Dict, Any
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from slowapi import _rate_limit_exceeded_handler
from slowapi.middleware import SlowAPIMiddleware
from src.security_middleware import setup_security_middleware
from src.core.error_handler import setup_error_handlers
from src.api import (
    auth_router, container_router, observer_router,
    whisper_router, decision_router, system_router
)
from src.api.system import init_system_routes, limiter as system_limiter
from src.api.auth import init_auth_routes
from src.api.container import init_container_routes
from src.api.observer import init_observer_routes
from src.api.whisper import init_whisper_routes
from src.api.decision import init_decision_routes

logger = logging.getLogger(__name__)

def create_application(config: Dict[str, Any], components: Dict[str, Any]) -> FastAPI:
    """创建FastAPI应用实例"""

    # 创建FastAPI应用实例
    app = FastAPI(
        title=config['app']['name'],
        version=config['app']['version'],
        description=config['app']['description']
    )

    # 设置统一错误处理
    setup_error_handlers(app)

    # 设置安全中间件
    setup_security_middleware(app, config)

    # 添加限流中间件
    limiter = system_limiter
    app.state.limiter = limiter
    app.add_exception_handler(Exception, _rate_limit_exceeded_handler)
    app.add_middleware(SlowAPIMiddleware)

    # 挂载静态文件
    app.mount("/static", StaticFiles(directory="static"), name="static")

    # 初始化API路由
    _init_api_routes(app, config, components)

    # 设置启动和关闭事件
    _setup_lifecycle_events(app, config, components)

    logger.info("FastAPI应用实例创建完成")
    return app

def _init_api_routes(app: FastAPI, config: Dict[str, Any], components: Dict[str, Any]):
    """初始化API路由"""

    # 初始化各个路由模块的依赖
    init_auth_routes(components['auth_manager'], components['audit_logger'])
    init_container_routes(components['container_manager'], components['audit_logger'])
    init_observer_routes(components['observer'], components['audit_logger'])
    init_whisper_routes(components['whisper_injection'], components['audit_logger'])
    init_decision_routes(components['decision_engine'], components['audit_logger'])
    init_system_routes(
        config,
        components['agent_manager'],
        components['cache_manager'],
        components['connection_pool'],
        components['performance_monitor'],
        components['audit_logger'],
        components.get('observer'),
        components.get('decision_engine'),
        components.get('whisper_injection')
    )

    # 注册路由
    app.include_router(auth_router)
    app.include_router(container_router)
    app.include_router(observer_router)
    app.include_router(whisper_router)
    app.include_router(decision_router)
    app.include_router(system_router)

    logger.info("API路由初始化完成")

def _setup_lifecycle_events(app: FastAPI, config: Dict[str, Any], components: Dict[str, Any]):
    """设置应用生命周期事件"""

    @app.on_event("startup")
    async def startup_event():
        """应用启动事件"""
        logger.info("正在启动AgentContainer...")

        # 启动决策引擎（如果启用）
        if config.get('decision_engine', {}).get('enabled', True):
            await components['decision_engine'].start_decision_loop()

        logger.info("AgentContainer启动完成！")

    @app.on_event("shutdown")
    async def shutdown_event():
        """应用关闭事件"""
        logger.info("正在关闭AgentContainer...")

        # 停止决策引擎
        await components['decision_engine'].stop_decision_loop()

        logger.info("AgentContainer已关闭")

def create_app():
    """返回FastAPI应用实例（用于工厂模式）"""
    # 注意：这个函数需要外部提供config和components
    # 在实际使用时，应该从main.py或其他地方调用create_application
    raise NotImplementedError("请使用create_application函数创建应用实例")
