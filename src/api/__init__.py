# API路由模块
"""
AgentContainer API路由模块
包含所有API端点的路由定义
"""

from .auth import router as auth_router
from .container import router as container_router
from .observer import router as observer_router
from .whisper import router as whisper_router
from .decision import router as decision_router
from .system import router as system_router

__all__ = [
    'auth_router',
    'container_router',
    'observer_router',
    'whisper_router',
    'decision_router',
    'system_router'
]