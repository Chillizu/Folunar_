#!/usr/bin/env python3
"""
性能监控模块
监控API响应时间、内存使用、并发连接数等性能指标
"""

import asyncio
import logging
import time
import psutil
import threading
from typing import Dict, Any, List, Optional
from collections import deque
from functools import wraps
import gc

class PerformanceMonitor:
    """
    性能监控器
    收集和分析系统性能指标
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config.get('performance', {})
        self.metrics_window = self.config.get('metrics_window', 1000)  # 保留最近1000个指标
        self.monitor_interval = self.config.get('monitor_interval', 10)  # 每10秒收集一次

        # 指标存储
        self.response_times: deque = deque(maxlen=self.metrics_window)
        self.memory_usage: deque = deque(maxlen=self.metrics_window)
        self.cpu_usage: deque = deque(maxlen=self.metrics_window)
        self.active_connections: deque = deque(maxlen=self.metrics_window)
        self.error_counts: Dict[str, int] = {}

        # 锁
        self.metrics_lock = asyncio.Lock()

        # 监控任务
        self.monitor_task: Optional[asyncio.Task] = None
        self.is_monitoring = False

        self.logger = logging.getLogger(__name__)

    async def start_monitoring(self):
        """启动性能监控"""
        if self.is_monitoring:
            return

        self.is_monitoring = True
        self.monitor_task = asyncio.create_task(self._monitor_loop())
        self.logger.info("性能监控已启动")

    async def stop_monitoring(self):
        """停止性能监控"""
        self.is_monitoring = False
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        self.logger.info("性能监控已停止")

    async def _monitor_loop(self):
        """监控循环"""
        while self.is_monitoring:
            try:
                await self._collect_metrics()
                await asyncio.sleep(self.monitor_interval)
            except Exception as e:
                self.logger.error(f"性能监控出错: {e}")
                await asyncio.sleep(self.monitor_interval)

    async def _collect_metrics(self):
        """收集系统指标"""
        try:
            # CPU使用率
            cpu_percent = psutil.cpu_percent(interval=0.1)

            # 内存使用
            memory = psutil.virtual_memory()

            # 网络连接数（估算活跃连接）
            connections = len(psutil.net_connections())

            async with self.metrics_lock:
                self.cpu_usage.append({
                    'timestamp': time.time(),
                    'value': cpu_percent
                })

                self.memory_usage.append({
                    'timestamp': time.time(),
                    'used': memory.used,
                    'percent': memory.percent
                })

                self.active_connections.append({
                    'timestamp': time.time(),
                    'count': connections
                })

        except Exception as e:
            self.logger.warning(f"收集系统指标失败: {e}")

    def record_response_time(self, method: str, endpoint: str, duration: float, status_code: int = 200):
        """记录API响应时间"""
        async def _record():
            async with self.metrics_lock:
                self.response_times.append({
                    'timestamp': time.time(),
                    'method': method,
                    'endpoint': endpoint,
                    'duration': duration,
                    'status_code': status_code
                })

        # 在后台记录，避免阻塞
        asyncio.create_task(_record())

    def record_error(self, error_type: str, endpoint: str = ""):
        """记录错误"""
        async def _record():
            async with self.metrics_lock:
                key = f"{error_type}:{endpoint}"
                self.error_counts[key] = self.error_counts.get(key, 0) + 1

        asyncio.create_task(_record())

    def api_performance_decorator(self, method: str, endpoint: str):
        """API性能监控装饰器"""
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                start_time = time.time()
                status_code = 200
                try:
                    result = await func(*args, **kwargs)
                    return result
                except Exception as e:
                    status_code = 500
                    self.record_error(type(e).__name__, endpoint)
                    raise
                finally:
                    duration = time.time() - start_time
                    self.record_response_time(method, endpoint, duration, status_code)

            return wrapper
        return decorator

    async def get_metrics_summary(self) -> Dict[str, Any]:
        """获取性能指标摘要"""
        async with self.metrics_lock:
            # 响应时间统计
            response_times = [rt['duration'] for rt in self.response_times]
            response_stats = {}
            if response_times:
                response_stats = {
                    'avg_response_time': sum(response_times) / len(response_times),
                    'min_response_time': min(response_times),
                    'max_response_time': max(response_times),
                    'p95_response_time': sorted(response_times)[int(len(response_times) * 0.95)] if response_times else 0,
                    'total_requests': len(response_times)
                }

            # CPU使用率统计
            cpu_values = [cpu['value'] for cpu in self.cpu_usage]
            cpu_stats = {}
            if cpu_values:
                cpu_stats = {
                    'avg_cpu_percent': sum(cpu_values) / len(cpu_values),
                    'max_cpu_percent': max(cpu_values)
                }

            # 内存使用统计
            memory_values = [(mem['used'], mem['percent']) for mem in self.memory_usage]
            memory_stats = {}
            if memory_values:
                used_values = [used for used, _ in memory_values]
                percent_values = [percent for _, percent in memory_values]
                memory_stats = {
                    'avg_memory_used': sum(used_values) / len(used_values),
                    'max_memory_used': max(used_values),
                    'avg_memory_percent': sum(percent_values) / len(percent_values),
                    'max_memory_percent': max(percent_values)
                }

            # 连接数统计
            connection_values = [conn['count'] for conn in self.active_connections]
            connection_stats = {}
            if connection_values:
                connection_stats = {
                    'avg_connections': sum(connection_values) / len(connection_values),
                    'max_connections': max(connection_values)
                }

            return {
                'response_time': response_stats,
                'cpu_usage': cpu_stats,
                'memory_usage': memory_stats,
                'connections': connection_stats,
                'errors': dict(self.error_counts),
                'timestamp': time.time()
            }

    async def get_health_status(self) -> Dict[str, Any]:
        """获取健康状态"""
        try:
            metrics = await self.get_metrics_summary()

            # 健康检查阈值
            thresholds = self.config.get('health_thresholds', {
                'max_response_time': 5.0,  # 5秒
                'max_cpu_percent': 90.0,
                'max_memory_percent': 90.0,
                'max_error_rate': 0.1  # 10%
            })

            # 计算健康状态
            response_time_ok = (
                not metrics['response_time'] or
                metrics['response_time'].get('p95_response_time', 0) < thresholds['max_response_time']
            )

            cpu_ok = (
                not metrics['cpu_usage'] or
                metrics['cpu_usage'].get('avg_cpu_percent', 0) < thresholds['max_cpu_percent']
            )

            memory_ok = (
                not metrics['memory_usage'] or
                metrics['memory_usage'].get('avg_memory_percent', 0) < thresholds['max_memory_percent']
            )

            # 错误率检查
            total_requests = metrics['response_time'].get('total_requests', 0)
            total_errors = sum(metrics['errors'].values()) if metrics['errors'] else 0
            error_rate = total_errors / max(total_requests, 1)
            error_rate_ok = error_rate < thresholds['max_error_rate']

            is_healthy = response_time_ok and cpu_ok and memory_ok and error_rate_ok

            return {
                'healthy': is_healthy,
                'checks': {
                    'response_time': response_time_ok,
                    'cpu_usage': cpu_ok,
                    'memory_usage': memory_ok,
                    'error_rate': error_rate_ok
                },
                'error_rate': error_rate,
                'metrics': metrics
            }

        except Exception as e:
            self.logger.error(f"获取健康状态失败: {e}")
            return {
                'healthy': False,
                'error': str(e),
                'timestamp': time.time()
            }

    async def force_gc(self):
        """强制垃圾回收"""
        collected = gc.collect()
        self.logger.info(f"垃圾回收完成，回收了 {collected} 个对象")
        return collected