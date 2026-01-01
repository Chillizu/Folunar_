#!/usr/bin/env python3
"""
AgentContainer 主入口文件
"""

import yaml
import uvicorn
import json
import logging
import asyncio
import time
import psutil
from typing import Optional, Dict
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from src.core.agent_manager import AgentManager
from src.container_manager import ContainerManager
from src.core.cache_manager import CacheManager
from src.core.connection_pool import ConnectionPoolManager
from src.core.performance_monitor import PerformanceMonitor
from src.auth import AuthManager, require_auth, UserCredentials
from src.validation import ValidationManager, validate_request_data, ChatCompletionRequest, ContainerExecRequest
from src.security_middleware import setup_security_middleware, AuditLogger
from src.observer import Observer
from src.whisper_injection import WhisperInjectionManager

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

# 加载配置
config = load_config()

# 初始化缓存管理器
cache_manager = CacheManager(config)

# 初始化连接池管理器
connection_pool = ConnectionPoolManager(config)

# 初始化性能监控器
performance_monitor = PerformanceMonitor(config)

# 初始化安全组件
auth_manager = AuthManager(config)
validation_manager = ValidationManager(config)
audit_logger = AuditLogger(config)

# 创建FastAPI应用实例
app = FastAPI(
    title=config['app']['name'],
    version=config['app']['version'],
    description=config['app']['description']
)

# 设置安全中间件
setup_security_middleware(app, config)

# 添加限流中间件
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

# 挂载静态文件
app.mount("/static", StaticFiles(directory="static"), name="static")

# 初始化代理管理器
agent_manager = AgentManager(config)

# 初始化容器管理器（使用连接池）
container_manager = ContainerManager(config, connection_pool)

# 初始化观察者
observer = Observer(config, agent_manager)

# 初始化随机想法注入管理器
whisper_injection = WhisperInjectionManager(config.get('whisper_injection', {}))

# 认证端点
@app.post("/api/auth/login")
async def login(credentials: UserCredentials):
    """用户登录"""
    if auth_manager.authenticate_user(credentials.username, credentials.password):
        access_token = auth_manager.create_access_token({"sub": credentials.username})
        audit_logger.log_event("LOGIN_SUCCESS", {"username": credentials.username})
        return {"access_token": access_token, "token_type": "bearer"}
    else:
        audit_logger.log_event("LOGIN_FAILED", {"username": credentials.username})
        raise HTTPException(status_code=401, detail="用户名或密码错误")

@app.post("/api/auth/logout")
async def logout(current_user: str = require_auth):
    """用户登出"""
    audit_logger.log_event("LOGOUT", {"username": current_user})
    return {"message": "登出成功"}

@app.get("/api/auth/me")
async def get_current_user_info(current_user: str = require_auth):
    """获取当前用户信息"""
    return {"username": current_user}

@app.get("/")
async def root():
    return {"message": "Welcome to AgentContainer", "version": config['app']['version']}

@app.get("/chat")
async def chat_page():
    """返回聊天页面"""
    from fastapi.responses import FileResponse
    return FileResponse("static/index.html")

@app.get("/agents")
async def list_agents():
    return {"agents": agent_manager.list_agents()}

@app.get("/health")
@limiter.limit("60/minute")
@performance_monitor.api_performance_decorator("GET", "/health")
async def health_check():
    """健康检查端点"""
    health_status = await performance_monitor.get_health_status()
    status_code = 200 if health_status['healthy'] else 503
    return health_status

@app.get("/api/performance/metrics")
@limiter.limit("10/minute")
@performance_monitor.api_performance_decorator("GET", "/api/performance/metrics")
async def get_performance_metrics():
    """获取性能指标"""
    try:
        metrics = await performance_monitor.get_metrics_summary()
        return {"status": "success", "data": metrics}
    except Exception as e:
        logger.error(f"Failed to get performance metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get performance metrics: {str(e)}")

@app.post("/api/performance/gc")
@limiter.limit("5/minute")
@performance_monitor.api_performance_decorator("POST", "/api/performance/gc")
async def trigger_garbage_collection():
    """触发垃圾回收"""
    try:
        collected = await performance_monitor.force_gc()
        return {"status": "success", "collected_objects": collected}
    except Exception as e:
        logger.error(f"Failed to trigger garbage collection: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to trigger garbage collection: {str(e)}")

@app.get("/api/system/status")
@limiter.limit("30/minute")
@performance_monitor.api_performance_decorator("GET", "/api/system/status")
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

# 容器管理端点
@app.post("/api/container/build")
async def build_container():
    """构建Debian容器镜像"""
    try:
        result = await container_manager.build_image()
        if result["success"]:
            return {"status": "success", "message": "容器镜像构建成功"}
        else:
            raise HTTPException(status_code=500, detail=result["error"])
    except Exception as e:
        logger.error(f"构建容器镜像失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"构建容器镜像失败: {str(e)}")

@app.post("/api/container/start")
async def start_container(ports: Optional[Dict[str, str]] = None):
    """启动Debian容器"""
    try:
        result = await container_manager.start_container(ports)
        if result["success"]:
            return {"status": "success", "message": "容器启动成功"}
        else:
            raise HTTPException(status_code=500, detail=result["error"])
    except Exception as e:
        logger.error(f"启动容器失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"启动容器失败: {str(e)}")

@app.post("/api/container/stop")
async def stop_container():
    """停止Debian容器"""
    try:
        result = await container_manager.stop_container()
        if result["success"]:
            return {"status": "success", "message": "容器停止成功"}
        else:
            return {"status": "warning", "message": result["error"]}
    except Exception as e:
        logger.error(f"停止容器失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"停止容器失败: {str(e)}")

@app.post("/api/container/remove")
async def remove_container():
    """删除Debian容器"""
    try:
        result = await container_manager.remove_container()
        if result["success"]:
            return {"status": "success", "message": "容器删除成功"}
        else:
            return {"status": "warning", "message": result["error"]}
    except Exception as e:
        logger.error(f"删除容器失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"删除容器失败: {str(e)}")

@app.get("/api/container/status")
async def get_container_status():
    """获取容器状态"""
    try:
        result = await container_manager.get_container_status()
        if result["success"]:
            return {"status": "success", "data": result["data"]}
        else:
            raise HTTPException(status_code=500, detail=result["error"])
    except Exception as e:
        logger.error(f"获取容器状态失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取容器状态失败: {str(e)}")

@app.get("/api/container/monitor")
async def monitor_container():
    """实时监控容器统计信息"""
    async def generate_stats():
        try:
            while True:
                stats = await container_manager.get_container_stats()
                if stats["success"]:
                    data = {
                        "timestamp": int(time.time()),
                        "stats": stats["data"]
                    }
                    yield f"data: {json.dumps(data)}\n\n"
                else:
                    yield f"data: {json.dumps({'error': stats['error'], 'timestamp': int(time.time())})}\n\n"
                await asyncio.sleep(2)  # 每2秒更新一次
        except Exception as e:
            logger.error(f"监控容器时发生错误: {str(e)}")
            yield f"data: {json.dumps({'error': str(e), 'timestamp': int(time.time())})}\n\n"

    return StreamingResponse(
        generate_stats(),
        media_type="text/event-stream",
        headers={"Content-Type": "text/event-stream; charset=utf-8", "Cache-Control": "no-cache"}
    )

@app.post("/api/container/exec")
async def exec_in_container(exec_request: ContainerExecRequest, current_user: str = require_auth):
    """在容器中执行命令（需要认证和输入验证）"""
    audit_logger.log_event("CONTAINER_EXEC_REQUEST", {
        "user": current_user,
        "command": exec_request.command[:100] + "..." if len(exec_request.command) > 100 else exec_request.command
    }, current_user)

    result = await container_manager.exec_command(exec_request.command)
    if result["success"]:
        audit_logger.log_event("CONTAINER_EXEC_SUCCESS", {
            "user": current_user,
            "command": exec_request.command[:100] + "..." if len(exec_request.command) > 100 else exec_request.command
        }, current_user)
        return {"status": "success", "output": result["output"]}
    else:
        audit_logger.log_event("CONTAINER_EXEC_FAILED", {
            "user": current_user,
            "command": exec_request.command[:100] + "..." if len(exec_request.command) > 100 else exec_request.command,
            "error": result["error"]
        }, current_user)
        raise HTTPException(status_code=500, detail=result["error"])

@app.get("/test/streaming")
async def test_streaming():
    """测试流式响应端点，返回模拟的OpenAI格式流式数据"""
    import time
    import uuid

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

@app.post("/v1/chat/completions")
@limiter.limit("100/minute")
@performance_monitor.api_performance_decorator("POST", "/v1/chat/completions")
async def chat_completions(request: Request, current_user: str = require_auth):
    """处理chat completions请求，支持流式和非流式响应（带性能优化和安全验证）"""
    try:
        data = await request.json()

        # 输入验证
        validated_data = validate_request_data(ChatCompletionRequest, data)
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

# 观察者端点
@app.post("/api/observer/start")
async def start_observer(current_user: str = require_auth):
    """启动AI观察者"""
    audit_logger.log_event("OBSERVER_START_REQUEST", {"user": current_user}, current_user)
    try:
        success = await observer.start_monitoring()
        if success:
            audit_logger.log_event("OBSERVER_START_SUCCESS", {"user": current_user}, current_user)
            return {"status": "success", "message": "观察者已启动"}
        else:
            audit_logger.log_event("OBSERVER_START_FAILED", {"user": current_user, "reason": "already_running"}, current_user)
            return {"status": "warning", "message": "观察者已在运行中"}
    except Exception as e:
        logger.error(f"启动观察者失败: {str(e)}")
        audit_logger.log_event("OBSERVER_START_ERROR", {"user": current_user, "error": str(e)}, current_user)
        raise HTTPException(status_code=500, detail=f"启动观察者失败: {str(e)}")

@app.post("/api/observer/stop")
async def stop_observer(current_user: str = require_auth):
    """停止AI观察者"""
    audit_logger.log_event("OBSERVER_STOP_REQUEST", {"user": current_user}, current_user)
    try:
        success = await observer.stop_monitoring()
        if success:
            audit_logger.log_event("OBSERVER_STOP_SUCCESS", {"user": current_user}, current_user)
            return {"status": "success", "message": "观察者已停止"}
        else:
            audit_logger.log_event("OBSERVER_STOP_FAILED", {"user": current_user, "reason": "not_running"}, current_user)
            return {"status": "warning", "message": "观察者未在运行"}
    except Exception as e:
        logger.error(f"停止观察者失败: {str(e)}")
        audit_logger.log_event("OBSERVER_STOP_ERROR", {"user": current_user, "error": str(e)}, current_user)
        raise HTTPException(status_code=500, detail=f"停止观察者失败: {str(e)}")

@app.get("/api/observer/status")
async def get_observer_status(current_user: str = require_auth):
    """获取观察者状态"""
    try:
        status = observer.get_status()
        return {"status": "success", "data": status}
    except Exception as e:
        logger.error(f"获取观察者状态失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取观察者状态失败: {str(e)}")

@app.get("/api/observer/observations")
async def get_recent_observations(limit: int = 10, current_user: str = require_auth):
    """获取最近的观察记录"""
    try:
        observations = observer.get_recent_observations(limit)
        return {"status": "success", "data": observations}
    except Exception as e:
        logger.error(f"获取观察记录失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取观察记录失败: {str(e)}")

@app.post("/api/observer/clear")
async def clear_observations(current_user: str = require_auth):
    """清空观察历史记录"""
    audit_logger.log_event("OBSERVER_CLEAR_REQUEST", {"user": current_user}, current_user)
    try:
        observer.clear_observation_history()
        audit_logger.log_event("OBSERVER_CLEAR_SUCCESS", {"user": current_user}, current_user)
        return {"status": "success", "message": "观察历史记录已清空"}
    except Exception as e:
        logger.error(f"清空观察记录失败: {str(e)}")
        audit_logger.log_event("OBSERVER_CLEAR_ERROR", {"user": current_user, "error": str(e)}, current_user)
        raise HTTPException(status_code=500, detail=f"清空观察记录失败: {str(e)}")

# 随机想法注入系统端点
@app.get("/api/whisper/status")
async def get_whisper_status(current_user: str = require_auth):
    """获取随机想法注入系统状态"""
    try:
        status = whisper_injection.get_status()
        return {"status": "success", "data": status}
    except Exception as e:
        logger.error(f"获取注入系统状态失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取注入系统状态失败: {str(e)}")

@app.get("/api/whisper/vocabulary")
async def get_whisper_vocabulary(current_user: str = require_auth):
    """获取词汇库"""
    try:
        vocabulary = whisper_injection.get_vocabulary()
        return {"status": "success", "data": {"vocabulary": vocabulary, "count": len(vocabulary)}}
    except Exception as e:
        logger.error(f"获取词汇库失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取词汇库失败: {str(e)}")

@app.post("/api/whisper/vocabulary/add")
async def add_whisper_vocabulary(word: str, current_user: str = require_auth):
    """添加词汇"""
    audit_logger.log_event("WHISPER_ADD_VOCABULARY", {"user": current_user, "word": word}, current_user)
    try:
        if not word or not word.strip():
            raise HTTPException(status_code=400, detail="词汇不能为空")

        success = whisper_injection.add_vocabulary(word.strip())
        if success:
            audit_logger.log_event("WHISPER_ADD_VOCABULARY_SUCCESS", {"user": current_user, "word": word}, current_user)
            return {"status": "success", "message": f"词汇 '{word}' 已添加"}
        else:
            return {"status": "warning", "message": f"词汇 '{word}' 已存在"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"添加词汇失败: {str(e)}")
        audit_logger.log_event("WHISPER_ADD_VOCABULARY_ERROR", {"user": current_user, "word": word, "error": str(e)}, current_user)
        raise HTTPException(status_code=500, detail=f"添加词汇失败: {str(e)}")

@app.delete("/api/whisper/vocabulary/remove")
async def remove_whisper_vocabulary(word: str, current_user: str = require_auth):
    """删除词汇"""
    audit_logger.log_event("WHISPER_REMOVE_VOCABULARY", {"user": current_user, "word": word}, current_user)
    try:
        success = whisper_injection.remove_vocabulary(word)
        if success:
            audit_logger.log_event("WHISPER_REMOVE_VOCABULARY_SUCCESS", {"user": current_user, "word": word}, current_user)
            return {"status": "success", "message": f"词汇 '{word}' 已删除"}
        else:
            raise HTTPException(status_code=404, detail=f"词汇 '{word}' 不存在")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除词汇失败: {str(e)}")
        audit_logger.log_event("WHISPER_REMOVE_VOCABULARY_ERROR", {"user": current_user, "word": word, "error": str(e)}, current_user)
        raise HTTPException(status_code=500, detail=f"删除词汇失败: {str(e)}")

@app.post("/api/whisper/vocabulary/clear")
async def clear_whisper_vocabulary(current_user: str = require_auth):
    """清空词汇库"""
    audit_logger.log_event("WHISPER_CLEAR_VOCABULARY", {"user": current_user}, current_user)
    try:
        whisper_injection.clear_vocabulary()
        audit_logger.log_event("WHISPER_CLEAR_VOCABULARY_SUCCESS", {"user": current_user}, current_user)
        return {"status": "success", "message": "词汇库已清空"}
    except Exception as e:
        logger.error(f"清空词汇库失败: {str(e)}")
        audit_logger.log_event("WHISPER_CLEAR_VOCABULARY_ERROR", {"user": current_user, "error": str(e)}, current_user)
        raise HTTPException(status_code=500, detail=f"清空词汇库失败: {str(e)}")

@app.get("/api/whisper/logs")
async def get_whisper_logs(limit: int = 50, current_user: str = require_auth):
    """获取注入日志"""
    try:
        logs = whisper_injection.get_injection_logs(limit)
        return {"status": "success", "data": {"logs": logs, "count": len(logs)}}
    except Exception as e:
        logger.error(f"获取注入日志失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取注入日志失败: {str(e)}")

@app.post("/api/whisper/logs/clear")
async def clear_whisper_logs(current_user: str = require_auth):
    """清空注入日志"""
    audit_logger.log_event("WHISPER_CLEAR_LOGS", {"user": current_user}, current_user)
    try:
        whisper_injection.clear_logs()
        audit_logger.log_event("WHISPER_CLEAR_LOGS_SUCCESS", {"user": current_user}, current_user)
        return {"status": "success", "message": "注入日志已清空"}
    except Exception as e:
        logger.error(f"清空注入日志失败: {str(e)}")
        audit_logger.log_event("WHISPER_CLEAR_LOGS_ERROR", {"user": current_user, "error": str(e)}, current_user)
        raise HTTPException(status_code=500, detail=f"清空注入日志失败: {str(e)}")

@app.post("/api/whisper/inject")
async def manual_whisper_injection(current_user: str = require_auth):
    """手动触发注入"""
    audit_logger.log_event("WHISPER_MANUAL_INJECT", {"user": current_user}, current_user)
    try:
        word = whisper_injection.inject_random_word()
        if word:
            audit_logger.log_event("WHISPER_MANUAL_INJECT_SUCCESS", {"user": current_user, "word": word}, current_user)
            return {"status": "success", "message": f"成功注入词汇: {word}"}
        else:
            raise HTTPException(status_code=500, detail="注入失败，请检查词汇库是否为空")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"手动注入失败: {str(e)}")
        audit_logger.log_event("WHISPER_MANUAL_INJECT_ERROR", {"user": current_user, "error": str(e)}, current_user)
        raise HTTPException(status_code=500, detail=f"手动注入失败: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """应用启动事件"""
    logger.info("正在启动AgentContainer...")

    # 初始化缓存管理器
    await cache_manager.initialize()

    # 初始化连接池
    await connection_pool.initialize()

    # 启动性能监控
    await performance_monitor.start_monitoring()

    # 启动随机想法注入系统
    whisper_injection.start_injection()

    logger.info("AgentContainer启动完成！")

@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭事件"""
    logger.info("正在关闭AgentContainer...")

    # 停止性能监控
    await performance_monitor.stop_monitoring()

    # 关闭连接池
    await connection_pool.close()

    # 关闭缓存管理器
    await cache_manager.close()

    # 停止随机想法注入系统
    whisper_injection.stop_injection()

    logger.info("AgentContainer已关闭")

def create_app():
    """返回FastAPI应用实例（用于工厂模式）"""
    return app

if __name__ == "__main__":
    config = load_config()

    # 性能优化配置
    server_config = config.get('server', {})
    security_config = config.get('security', {})

    workers = server_config.get('workers', 1)
    max_requests = server_config.get('max_requests', 1000)
    max_requests_jitter = server_config.get('max_requests_jitter', 50)

    # HTTPS配置
    ssl_enabled = security_config.get('enable_https', False)
    ssl_cert_path = security_config.get('ssl_cert_path', 'certs/server.crt')
    ssl_key_path = security_config.get('ssl_key_path', 'certs/server.key')

    server_kwargs = {
        "host": server_config.get('host', '0.0.0.0'),
        "port": server_config.get('port', 8000),
        "loop": "uvloop",  # 使用uvloop提升性能
        "http": "httptools",  # 使用httptools提升HTTP性能
        "log_level": "info"
    }

    # 添加HTTPS配置
    if ssl_enabled:
        import ssl
        ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        ssl_context.load_cert_chain(ssl_cert_path, ssl_key_path)
        server_kwargs["ssl"] = ssl_context
        logger.info(f"HTTPS已启用，证书路径: {ssl_cert_path}")

    if workers > 1:
        # 多进程模式使用Gunicorn
        import multiprocessing
        workers = min(workers, multiprocessing.cpu_count())

        server_kwargs.update({
            "workers": workers,
            "access_log": False,  # 生产环境关闭访问日志
        })

        uvicorn.run("main:create_app", factory=True, **server_kwargs)
    else:
        # 单进程模式
        server_kwargs.update({
            "reload": server_config.get('debug', True),
            "access_log": server_config.get('debug', True),
        })

        uvicorn.run("main:create_app", factory=True, **server_kwargs)