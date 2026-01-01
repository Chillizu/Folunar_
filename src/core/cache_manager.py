#!/usr/bin/env python3
"""
缓存管理模块
提供Redis缓存和内存缓存支持，提升API响应性能
"""

import asyncio
import json
import logging
from typing import Any, Optional, Dict, Union
from datetime import datetime, timedelta
import aioredis
import redis.asyncio as redis
from functools import wraps
import time

class CacheManager:
    """
    缓存管理器
    支持Redis和内存缓存，提供缓存装饰器和直接缓存操作
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config.get('cache', {})
        self.redis_url = self.config.get('redis_url', 'redis://localhost:6379')
        self.default_ttl = self.config.get('default_ttl', 300)  # 默认5分钟
        self.max_memory_cache = self.config.get('max_memory_cache', 1000)  # 内存缓存最大条目数

        # 内存缓存
        self.memory_cache: Dict[str, Dict[str, Any]] = {}
        self.memory_cache_lock = asyncio.Lock()

        # Redis连接
        self.redis_client: Optional[redis.Redis] = None
        self.redis_enabled = self.config.get('redis_enabled', False)

        self.logger = logging.getLogger(__name__)

    async def initialize(self):
        """初始化缓存管理器"""
        if self.redis_enabled:
            try:
                self.redis_client = redis.Redis.from_url(self.redis_url, decode_responses=True)
                await self.redis_client.ping()
                self.logger.info("Redis缓存已连接")
            except Exception as e:
                self.logger.warning(f"Redis连接失败，使用内存缓存: {e}")
                self.redis_enabled = False

        # 启动缓存清理任务
        asyncio.create_task(self._cleanup_memory_cache())

    async def close(self):
        """关闭缓存管理器"""
        if self.redis_client:
            await self.redis_client.close()

    async def _cleanup_memory_cache(self):
        """定期清理过期的内存缓存"""
        while True:
            try:
                current_time = time.time()
                async with self.memory_cache_lock:
                    expired_keys = [
                        key for key, data in self.memory_cache.items()
                        if data.get('expires_at', 0) < current_time
                    ]
                    for key in expired_keys:
                        del self.memory_cache[key]

                    # 如果内存缓存过多，清理最旧的条目
                    if len(self.memory_cache) > self.max_memory_cache:
                        sorted_items = sorted(
                            self.memory_cache.items(),
                            key=lambda x: x[1].get('created_at', 0)
                        )
                        remove_count = len(self.memory_cache) - self.max_memory_cache + 100
                        for key, _ in sorted_items[:remove_count]:
                            del self.memory_cache[key]

                await asyncio.sleep(60)  # 每分钟清理一次
            except Exception as e:
                self.logger.error(f"清理内存缓存时出错: {e}")
                await asyncio.sleep(60)

    async def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        # 优先从Redis获取
        if self.redis_enabled and self.redis_client:
            try:
                value = await self.redis_client.get(key)
                if value:
                    return json.loads(value)
            except Exception as e:
                self.logger.warning(f"Redis获取缓存失败: {e}")

        # 从内存缓存获取
        async with self.memory_cache_lock:
            cache_data = self.memory_cache.get(key)
            if cache_data and cache_data.get('expires_at', 0) > time.time():
                return cache_data['value']

        return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """设置缓存值"""
        ttl = ttl or self.default_ttl
        expires_at = time.time() + ttl
        cache_data = {
            'value': value,
            'expires_at': expires_at,
            'created_at': time.time()
        }

        success = True

        # 设置Redis缓存
        if self.redis_enabled and self.redis_client:
            try:
                await self.redis_client.setex(key, ttl, json.dumps(value))
            except Exception as e:
                self.logger.warning(f"Redis设置缓存失败: {e}")
                success = False

        # 设置内存缓存
        async with self.memory_cache_lock:
            self.memory_cache[key] = cache_data

        return success

    async def delete(self, key: str) -> bool:
        """删除缓存"""
        success = True

        # 删除Redis缓存
        if self.redis_enabled and self.redis_client:
            try:
                await self.redis_client.delete(key)
            except Exception as e:
                self.logger.warning(f"Redis删除缓存失败: {e}")
                success = False

        # 删除内存缓存
        async with self.memory_cache_lock:
            self.memory_cache.pop(key, None)

        return success

    async def clear(self) -> bool:
        """清空所有缓存"""
        success = True

        # 清空Redis缓存
        if self.redis_enabled and self.redis_client:
            try:
                await self.redis_client.flushdb()
            except Exception as e:
                self.logger.warning(f"Redis清空缓存失败: {e}")
                success = False

        # 清空内存缓存
        async with self.memory_cache_lock:
            self.memory_cache.clear()

        return success

    def cached(self, ttl: Optional[int] = None, key_prefix: str = ""):
        """缓存装饰器"""
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # 生成缓存键
                cache_key = f"{key_prefix}:{func.__name__}"
                if args:
                    cache_key += f":{hash(str(args))}"
                if kwargs:
                    cache_key += f":{hash(str(sorted(kwargs.items())))}"

                # 尝试从缓存获取
                cached_result = await self.get(cache_key)
                if cached_result is not None:
                    return cached_result

                # 执行函数
                result = await func(*args, **kwargs)

                # 缓存结果
                await self.set(cache_key, result, ttl)

                return result

            return wrapper
        return decorator

    async def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        stats = {
            'memory_cache_entries': len(self.memory_cache),
            'redis_enabled': self.redis_enabled,
            'default_ttl': self.default_ttl
        }

        if self.redis_enabled and self.redis_client:
            try:
                redis_info = await self.redis_client.info()
                stats.update({
                    'redis_connected': True,
                    'redis_keys': redis_info.get('db0', {}).get('keys', 0),
                    'redis_memory_used': redis_info.get('used_memory_human', 'unknown')
                })
            except Exception as e:
                stats['redis_connected'] = False
                stats['redis_error'] = str(e)

        return stats