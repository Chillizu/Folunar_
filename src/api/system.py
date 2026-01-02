# 系统API路由
"""
系统相关的API端点，包括健康检查、性能监控、聊天等
"""

import json
import asyncio
import logging
import time
import uuid
import psutil
from typing import Optional, Dict
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from slowapi import Limiter
from slowapi.util import get_remote_address
from src.core.agent_manager import AgentManager
from src.core.cache_manager import CacheManager
from src.core.connection_pool import ConnectionPoolManager
from src.core.performance_monitor import PerformanceMonitor
from src.auth import require_auth
from src.validation import ChatCompletionRequest
from src.security_middleware import AuditLogger

logger = logging.getLogger(__name__)
router = APIRouter()
limiter = Limiter(key_func=get_remote_address)

# 全局变量（将在初始化时设置）
config = None
agent_manager: AgentManager = None
cache_manager: CacheManager = None
connection_pool: ConnectionPoolManager = None
performance_monitor: PerformanceMonitor = None
audit_logger: AuditLogger = None

def init_system_routes(
    _config,
    _agent_manager: AgentManager,
    _cache_manager: CacheManager,
    _connection_pool: ConnectionPoolManager,
    _performance_monitor: PerformanceMonitor,
    _audit_logger: AuditLogger
):
    """初始化系统路由的依赖"""
    global config, agent_manager, cache_manager, connection_pool, performance_monitor, audit_logger
    config = _config
    agent_manager = _agent_manager
    cache_manager = _cache_manager
    connection_pool = _connection_pool
    performance_monitor = _performance_monitor
    audit_logger = _audit_logger

@router.get("/")
async def root():
    return {"message": "Welcome to AgentContainer", "version": config['app']['version']}

@router.get("/chat")
async def chat_page():
    """返回聊天页面"""
    return FileResponse("static/index.html")

@router.get("/agents")
async def list_agents():
    return {"agents": agent_manager.list_agents()}

@router.get("/health")
@limiter.limit("60/minute")
async def health_check(request: Request):
    """健康检查端点"""
    health_status = await performance_monitor.get_health_status()
    status_code = 200 if health_status['healthy'] else 503
    return health_status

@router.get("/api/performance/metrics")
@limiter.limit("10/minute")
async def get_performance_metrics(request: Request):
    """获取性能指标"""
    try:
        metrics = await performance_monitor.get_metrics_summary()
        return {"status": "success", "data": metrics}
    except Exception as e:
        logger.error(f"Failed to get performance metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get performance metrics: {str(e)}")

@router.post("/api/performance/gc")
@limiter.limit("5/minute")
async def trigger_garbage_collection(request: Request):
    """触发垃圾回收"""
    try:
        collected = await performance_monitor.force_gc()
        return {"status": "success", "collected_objects": collected}
    except Exception as e:
        logger.error(f"Failed to trigger garbage collection: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to trigger garbage collection: {str(e)}")

@router.get("/api/system/status")
@limiter.limit("30/minute")
async def system_status(request: Request):
    """获取系统状态信息（带缓存优化）"""
    try:
        # 尝试从缓存获取
        cache_key = "system_status"
        cached_status = await cache_manager.get(cache_key)
        if cached_status:
            return cached_status

        # 获取系统信息（优化：减少CPU采样时间）
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=0.1)  # 从1秒减少到0.1秒

        # 计算运行时间（从应用启动开始）
        uptime = time.time() - psutil.boot_time()

        # 获取连接池和缓存统计
        pool_stats = await connection_pool.get_stats()
        cache_stats = await cache_manager.get_stats()

        status = {
            "version": config['app']['version'],
            "uptime": uptime,
            "active_agents": len(agent_manager.list_agents()),
            "sandbox_status": "运行中",  # 添加沙盒状态
            "system": {
                "cpu_percent": cpu_percent,
                "memory_used": memory.used,
                "memory_total": memory.total,
                "memory_percent": memory.percent
            },
            "performance": {
                "connection_pool": pool_stats,
                "cache": cache_stats
            },
            "timestamp": int(time.time())
        }

        # 缓存结果30秒
        await cache_manager.set(cache_key, status, ttl=30)

        logger.info(f"System status requested: CPU={cpu_percent:.1f}%, Memory={memory.percent:.1f}%")
        return status
    except Exception as e:
        logger.error(f"Failed to get system status: {str(e)}")
        performance_monitor.record_error("system_status_error", "/api/system/status")
        raise HTTPException(status_code=500, detail=f"Failed to get system status: {str(e)}")

@router.get("/test/streaming")
async def test_streaming():
    """测试流式响应端点，返回模拟的OpenAI格式流式数据"""
    async def generate_test_stream():
        # 生成模拟的流式响应
        chat_id = f"chatcmpl-{uuid.uuid4().hex[:16]}"
        created = int(time.time())

        # 第一个chunk：设置role
        chunk = {
            'id': chat_id,
            'object': 'chat.completion.chunk',
            'created': created,
            'model': 'gpt-3.5-turbo',
            'choices': [{'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}]
        }
        yield f"data: {json.dumps(chunk)}\n\n"

        # 内容chunks
        content = "Hello! This is a test streaming response from AgentContainer. "
        for char in content:
            chunk = {
                'id': chat_id,
                'object': 'chat.completion.chunk',
                'created': created,
                'model': 'gpt-3.5-turbo',
                'choices': [{'index': 0, 'delta': {'content': char}, 'finish_reason': None}]
            }
            yield f"data: {json.dumps(chunk)}\n\n"
            await asyncio.sleep(0.05)  # 模拟延迟

        # 最后一个chunk：设置finish_reason
        chunk = {
            'id': chat_id,
            'object': 'chat.completion.chunk',
            'created': created,
            'model': 'gpt-3.5-turbo',
            'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]
        }
        yield f"data: {json.dumps(chunk)}\n\n"

        # 结束
        yield "data: [DONE]\n\n"

    logger.info("Test streaming endpoint called")
    return StreamingResponse(
        generate_test_stream(),
        media_type="text/event-stream",
        headers={"Content-Type": "text/event-stream; charset=utf-8", "Cache-Control": "no-cache"}
    )

@router.post("/v1/chat/completions")
@limiter.limit("100/minute")
async def chat_completions(request: Request, current_user: str = require_auth):
    """处理chat completions请求，支持流式和非流式响应（带性能优化和安全验证）"""
    try:
        data = await request.json()

        # 输入验证
        validated_data = ChatCompletionRequest(**data)
        messages = validated_data.messages
        model = validated_data.model
        stream = validated_data.stream
        tools = validated_data.tools

        # 记录审计日志
        audit_logger.log_event("CHAT_COMPLETION_REQUEST", {
            "user": current_user,
            "model": model,
            "stream": stream,
            "messages_count": len(messages)
        }, current_user)

        logger.info(f"Chat completions request: user={current_user}, model={model}, stream={stream}, messages_count={len(messages)}")

        # 生成缓存键（用于非流式响应）
        if not stream and messages:
            cache_key = f"chat_completion:{model}:{hash(str(messages))}"
            cached_result = await cache_manager.get(cache_key)
            if cached_result:
                logger.info("Returning cached chat completion response")
                audit_logger.log_event("CHAT_COMPLETION_CACHE_HIT", {
                    "user": current_user,
                    "model": model
                }, current_user)
                return cached_result

        if stream:
            # 流式响应
            async def generate():
                try:
                    async for chunk in agent_manager.chat_completion(
                        messages=messages,
                        model=model,
                        stream=True,
                        tools=tools
                    ):
                        yield f"data: {json.dumps(chunk)}\n\n"
                    yield "data: [DONE]\n\n"
                except Exception as e:
                    logger.error(f"Streaming error: {str(e)}")
                    performance_monitor.record_error("streaming_error", "/v1/chat/completions")
                    audit_logger.log_event("CHAT_COMPLETION_ERROR", {
                        "user": current_user,
                        "error": str(e),
                        "type": "streaming_error"
                    }, current_user)
                    yield f"data: {json.dumps({'error': {'message': str(e), 'type': 'internal_error'}})}\n\n"

            return StreamingResponse(
                generate(),
                media_type="text/event-stream",
                headers={"Content-Type": "text/event-stream; charset=utf-8", "Cache-Control": "no-cache"}
            )

        else:
            # 非流式响应
            async for result in agent_manager.chat_completion(
                messages=messages,
                model=model,
                stream=False,
                tools=tools
            ):
                # 缓存结果（5分钟）
                if messages:
                    await cache_manager.set(cache_key, result, ttl=300)

                logger.info("Chat completions response sent")
                audit_logger.log_event("CHAT_COMPLETION_SUCCESS", {
                    "user": current_user,
                    "model": model,
                    "response_tokens": result.get("usage", {}).get("completion_tokens", 0)
                }, current_user)
                return result
    except Exception as e:
        logger.error(f"Chat completions request failed: {str(e)}")
        performance_monitor.record_error("chat_completion_error", "/v1/chat/completions")
        audit_logger.log_event("CHAT_COMPLETION_ERROR", {
            "user": current_user,
            "error": str(e),
            "type": "general_error"
        }, current_user)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
