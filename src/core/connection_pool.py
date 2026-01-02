#!/usr/bin/env python3
"""
连接池管理模块
提供Docker连接池和其他资源连接池，提升并发处理能力
"""

import asyncio
import logging
from typing import Any, Dict, Optional, List
from contextlib import asynccontextmanager
import aiohttp
from aiohttp import ClientSession, ClientTimeout
import docker
from docker import DockerClient
from concurrent.futures import ThreadPoolExecutor
import threading

class ConnectionPoolManager:
    """
    连接池管理器
    管理Docker客户端连接池、HTTP客户端连接池等
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config.get('connection_pool', {})
        self.docker_pool_size = self.config.get('docker_pool_size', 5)
        self.http_pool_size = self.config.get('http_pool_size', 20)
        self.executor_pool_size = self.config.get('executor_pool_size', 10)

        # Docker连接池
        self.docker_clients: List[DockerClient] = []
        self.docker_pool_lock = asyncio.Lock()
        self.docker_available_clients: asyncio.Queue = asyncio.Queue()
        self.docker_pool_enabled = False

        # HTTP客户端连接池
        self.http_sessions: List[ClientSession] = []
        self.http_pool_lock = asyncio.Lock()
        self.http_available_sessions: asyncio.Queue = asyncio.Queue()

        # 线程池执行器
        self.executor = ThreadPoolExecutor(max_workers=self.executor_pool_size, thread_name_prefix="agent-container")

        self.logger = logging.getLogger(__name__)
        self._initialized = False

    async def initialize(self):
        """初始化连接池"""
        if self._initialized:
            return

        try:
            # 初始化Docker连接池
            await self._init_docker_pool()

            # 初始化HTTP连接池
            await self._init_http_pool()

            self._initialized = True
            self.logger.info(f"连接池初始化完成 - Docker: {self.docker_pool_size}, HTTP: {self.http_pool_size}, Executor: {self.executor_pool_size}")

        except Exception as e:
            self.logger.error(f"连接池初始化失败: {e}")
            raise

    async def close(self):
        """关闭所有连接池"""
        # 关闭Docker客户端
        for client in self.docker_clients:
            try:
                client.close()
            except Exception as e:
                self.logger.warning(f"关闭Docker客户端时出错: {e}")

        # 关闭HTTP会话
        for session in self.http_sessions:
            try:
                await session.close()
            except Exception as e:
                self.logger.warning(f"关闭HTTP会话时出错: {e}")

        # 关闭线程池
        self.executor.shutdown(wait=True)

        self.logger.info("连接池已关闭")

    async def _init_docker_pool(self):
        """初始化Docker连接池"""
        successful_clients = 0
        for i in range(self.docker_pool_size):
            try:
                client = docker.from_env()
                # 测试连接
                client.ping()
                self.docker_clients.append(client)
                await self.docker_available_clients.put(client)
                successful_clients += 1
                self.logger.debug(f"Docker客户端 {i+1} 初始化成功")
            except Exception as e:
                self.logger.warning(f"Docker客户端 {i+1} 初始化失败: {e}")

        self.docker_pool_enabled = successful_clients > 0
        if not self.docker_pool_enabled:
            self.logger.warning("未能初始化任何Docker客户端连接，Docker连接池将被禁用")

    async def _init_http_pool(self):
        """初始化HTTP连接池"""
        timeout = ClientTimeout(total=30, connect=10)
        connector = aiohttp.TCPConnector(limit=self.http_pool_size, limit_per_host=10)

        for i in range(self.http_pool_size):
            try:
                session = ClientSession(connector=connector, timeout=timeout)
                self.http_sessions.append(session)
                await self.http_available_sessions.put(session)
                self.logger.debug(f"HTTP会话 {i+1} 初始化成功")
            except Exception as e:
                self.logger.warning(f"HTTP会话 {i+1} 初始化失败: {e}")

    def has_docker_pool(self) -> bool:
        """检查是否有可用的Docker连接"""
        return self.docker_pool_enabled

    @asynccontextmanager
    async def get_docker_client(self):
        """获取Docker客户端"""
        if not self.docker_pool_enabled:
            raise RuntimeError("Docker连接池不可用")
        client = None
        try:
            client = await self.docker_available_clients.get()
            yield client
        finally:
            if client:
                await self.docker_available_clients.put(client)

    @asynccontextmanager
    async def get_http_session(self):
        """获取HTTP会话"""
        session = None
        try:
            session = await self.http_available_sessions.get()
            yield session
        finally:
            if session:
                await self.http_available_sessions.put(session)

    async def run_in_executor(self, func, *args, **kwargs):
        """在线程池中运行阻塞函数"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, func, *args, **kwargs)

    async def get_stats(self) -> Dict[str, Any]:
        """获取连接池统计信息"""
        return {
            'docker_pool': {
                'total': len(self.docker_clients),
                'available': self.docker_available_clients.qsize()
            },
            'http_pool': {
                'total': len(self.http_sessions),
                'available': self.http_available_sessions.qsize()
            },
            'executor': {
                'max_workers': self.executor_pool_size,
                'active_threads': threading.active_count()
            }
        }
