# 应用初始化模块
"""
AgentContainer应用初始化模块
负责组件的创建、配置和依赖注入
"""

import asyncio
import logging
from typing import Dict, Any
from src.core.agent_manager import AgentManager
from src.core.cache_manager import CacheManager
from src.core.connection_pool import ConnectionPoolManager
from src.core.performance_monitor import PerformanceMonitor
from src.container_manager import ContainerManager
from src.auth import AuthManager
from src.validation import ValidationManager
from src.security_middleware import AuditLogger
from src.observer import Observer
from src.whisper_injection import WhisperInjectionManager
from src.decision_engine import DecisionEngine

logger = logging.getLogger(__name__)

class AppInitializer:
    """应用初始化器"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.components = {}

    async def initialize_components(self) -> Dict[str, Any]:
        """初始化所有组件"""
        logger.info("开始初始化应用组件...")

        try:
            # 初始化缓存管理器
            self.components['cache_manager'] = CacheManager(self.config)
            await self.components['cache_manager'].initialize()
            logger.info("缓存管理器初始化完成")

            # 初始化连接池管理器
            self.components['connection_pool'] = ConnectionPoolManager(self.config)
            await self.components['connection_pool'].initialize()
            logger.info("连接池管理器初始化完成")

            # 初始化性能监控器
            self.components['performance_monitor'] = PerformanceMonitor(self.config)
            await self.components['performance_monitor'].start_monitoring()
            logger.info("性能监控器初始化完成")

            # 初始化安全组件
            self.components['auth_manager'] = AuthManager(self.config)
            self.components['validation_manager'] = ValidationManager(self.config)
            self.components['audit_logger'] = AuditLogger(self.config)
            logger.info("安全组件初始化完成")

            # 初始化代理管理器
            self.components['agent_manager'] = AgentManager(self.config)
            logger.info("代理管理器初始化完成")

            # 初始化容器管理器（使用连接池）
            self.components['container_manager'] = ContainerManager(
                self.config,
                self.components['connection_pool']
            )
            logger.info("容器管理器初始化完成")

            # 初始化观察者
            self.components['observer'] = Observer(
                self.config,
                self.components['agent_manager']
            )
            logger.info("观察者初始化完成")

            # 初始化随机想法注入管理器
            self.components['whisper_injection'] = WhisperInjectionManager(
                self.config.get('whisper_injection', {})
            )
            self.components['whisper_injection'].start_injection()
            logger.info("随机想法注入管理器初始化完成")

            # 初始化决策引擎
            self.components['decision_engine'] = DecisionEngine(
                self.config,
                self.components['observer'],
                self.components['container_manager']
            )
            logger.info("决策引擎初始化完成")

            logger.info("所有组件初始化完成")
            return self.components

        except Exception as e:
            logger.error(f"组件初始化失败: {str(e)}")
            await self.cleanup_components()
            raise

    async def cleanup_components(self):
        """清理所有组件"""
        logger.info("开始清理应用组件...")

        cleanup_tasks = []

        # 停止决策引擎
        if 'decision_engine' in self.components:
            cleanup_tasks.append(self.components['decision_engine'].stop_decision_loop())

        # 停止随机想法注入系统
        if 'whisper_injection' in self.components:
            self.components['whisper_injection'].stop_injection()

        # 关闭 AgentManager 的 HTTP 客户端（如果已创建）
        if 'agent_manager' in self.components:
            cleanup_tasks.append(self.components['agent_manager'].close())

        # 关闭缓存管理器
        if 'cache_manager' in self.components:
            cleanup_tasks.append(self.components['cache_manager'].close())

        # 关闭连接池
        if 'connection_pool' in self.components:
            cleanup_tasks.append(self.components['connection_pool'].close())

        # 停止性能监控
        if 'performance_monitor' in self.components:
            cleanup_tasks.append(self.components['performance_monitor'].stop_monitoring())

        # 等待所有清理任务完成
        if cleanup_tasks:
            try:
                await asyncio.gather(*cleanup_tasks, return_exceptions=True)
            except Exception as e:
                logger.error(f"组件清理过程中发生错误: {str(e)}")

        logger.info("组件清理完成")

    def get_component(self, name: str):
        """获取指定组件"""
        return self.components.get(name)

    def get_all_components(self) -> Dict[str, Any]:
        """获取所有组件"""
        return self.components.copy()

async def initialize_application(config: Dict[str, Any]) -> AppInitializer:
    """初始化整个应用"""
    initializer = AppInitializer(config)
    await initializer.initialize_components()
    return initializer

async def shutdown_application(initializer: AppInitializer):
    """关闭整个应用"""
    await initializer.cleanup_components()
