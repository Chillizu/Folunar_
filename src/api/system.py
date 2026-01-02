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
from pathlib import Path
from typing import Optional, Dict
from fastapi import APIRouter, Request, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, FileResponse, HTMLResponse, JSONResponse
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
observer = None
decision_engine = None
whisper_injection = None

def init_system_routes(
    _config,
    _agent_manager: AgentManager,
    _cache_manager: CacheManager,
    _connection_pool: ConnectionPoolManager,
    _performance_monitor: PerformanceMonitor,
    _audit_logger: AuditLogger,
    _observer=None,
    _decision_engine=None,
    _whisper_injection=None
):
    """初始化系统路由的依赖"""
    global config, agent_manager, cache_manager, connection_pool, performance_monitor, audit_logger
    global observer, decision_engine, whisper_injection
    config = _config
    agent_manager = _agent_manager
    cache_manager = _cache_manager
    connection_pool = _connection_pool
    performance_monitor = _performance_monitor
    audit_logger = _audit_logger
    observer = _observer
    decision_engine = _decision_engine
    whisper_injection = _whisper_injection

@router.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """首页展示状态面板或 JSON（根据 Accept 头）"""
    if 'application/json' in request.headers.get("accept", ""):
        return JSONResponse({"message": "Welcome to AgentContainer", "version": config['app']['version']})

    html_template = """
<!doctype html>
<html lang="zh">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>AgentContainer 控制面板</title>
  <style>
    body {{
      margin: 0;
      font-family: system-ui,-apple-system,Segoe UI,Roboto,"PingFang SC",sans-serif;
      background: linear-gradient(180deg,#030712,#0d1b2a);
      color: #fff;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
    }}
    header {{
      padding: 1.5rem 2rem;
      border-bottom: 1px solid rgba(255,255,255,.1);
      display: flex;
      justify-content: space-between;
      align-items: center;
    }}
    header h1 {{
      margin: 0;
      font-size: 1.6rem;
    }}
    main {{
      flex: 1;
      padding: 2rem;
      display: grid;
      gap: 1rem;
      grid-template-columns: repeat(auto-fit,minmax(280px,1fr));
    }}
    .card {{
      background: rgba(10,25,60,.65);
      border-radius: 16px;
      padding: 1.5rem;
      box-shadow: 0 20px 45px rgba(5,10,20,.45);
      border: 1px solid rgba(255,255,255,.05);
    }}
    .card h2 {{
      margin-top: 0;
      font-size: 1.25rem;
    }}
    .button {{
      display: inline-flex;
      margin-top: 1rem;
      padding: 0.65rem 1.25rem;
      border-radius: 999px;
      border: none;
      background: linear-gradient(135deg,#00b4db,#0083ff);
      color:#fff;
      text-decoration: none;
      font-weight:600;
    }}
    ul {{
      padding-left: 1rem;
      margin: 0;
    }}
    footer {{
      padding: 1rem 2rem;
      text-align: center;
      font-size: 0.85rem;
      color: rgba(255,255,255,.6);
    }}
  </style>
</head>
<body>
  <header>
    <div>
      <h1>AgentContainer 控制台</h1>
      <p>版本 __VERSION__ · __APP_NAME__</p>
    </div>
    <a class="button" href="/chat">前往聊天界面</a>
  </header>
  <main>
    <div class="card">
      <h2>系统状态</h2>
      <p id="system-text">正在加载…</p>
      <a class="button" href="/api/system/status" target="_blank">查看原始数据</a>
    </div>

    <div class="card">
      <h2>核心路由</h2>
      <ul>
        <li><a href="/api/chat/completions" target="_blank">/api/chat/completions</a>（聊天完成）</li>
        <li><a href="/api/system/status" target="_blank">/api/system/status</a>（系统指标）</li>
        <li><a href="/health" target="_blank">/health</a>（健康探针）</li>
        <li><a href="/api/container/monitor" target="_blank">/api/container/monitor</a>（容器概览）</li>
      </ul>
    </div>

    <div class="card">
      <h2>安全审计</h2>
      <p>审计日志过滤了敏感头，所有请求/响应都经过 CSP/HSTS 保护。</p>
      <ul>
        <li>JWT 登录：`/api/auth/login`</li>
        <li>速率限制由单一 `Limiter` 控制</li>
        <li>敏感头一律隐藏：授权、Cookie 等</li>
      </ul>
    </div>
  </main>
  <footer>由 __APP_NAME__ 提供 · <a href="/chat" style="color:#4dd0e1">前往沙盒</a></footer>
  <script>
    async function refresh() {
      try {
        const resp = await fetch("/api/system/status");
        if (!resp.ok) throw new Error("无法读取状态");
        const data = await resp.json();
        document.getElementById("system-text").innerHTML = `
          <strong>活动代理：</strong>${data.active_agents}<br>
          <strong>系统状态：</strong>${data.system?.cpu_percent.toFixed(1)}% CPU / ${data.system?.memory_percent.toFixed(1)}% 内存<br>
          <strong>连接池：</strong>Docker ${data.performance?.connection_pool?.docker_pool?.available ?? 0}/${data.performance?.connection_pool?.docker_pool?.total}, HTTP ${data.performance?.connection_pool?.http_pool?.available ?? 0}/${data.performance?.connection_pool?.http_pool?.total}
        `;
      } catch (err) {
        document.getElementById("system-text").innerText = "状态加载失败：" + err.message;
      }
    }
    refresh();
    setInterval(refresh, 5000);
  </script>
</body>
</html>
"""
    html = html_template.replace("__VERSION__", config['app']['version']).replace("__APP_NAME__", config['app']['name'])
    return HTMLResponse(html)

@router.get("/chat")
async def chat_page():
    """返回聊天页面"""
    static_index = Path(__file__).resolve().parent.parent.parent / "static" / "index.html"
    return FileResponse(static_index)

@router.get("/styles.css")
async def legacy_styles():
    """兼容旧的静态路径"""
    static_css = Path(__file__).resolve().parent.parent.parent / "static" / "styles.css"
    return FileResponse(static_css)

@router.get("/chat.js")
async def legacy_chat_js():
    """兼容旧的静态路径"""
    static_js = Path(__file__).resolve().parent.parent.parent / "static" / "chat.js"
    return FileResponse(static_js)

@router.websocket("/ws/sandbox")
async def sandbox_ws(websocket: WebSocket):
    """提供沙盒监控的 WebSocket 数据流"""
    await websocket.accept()
    try:
        while True:
            status_payload = {
                "type": "sandbox_status",
                "status": "运行中" if agent_manager else "未知"
            }
            await websocket.send_json(status_payload)

            if decision_engine is None:
                await websocket.send_json({"type": "decision_log", "log": "暂无实时决策日志"})
            if whisper_injection is None:
                await websocket.send_json({"type": "injection_log", "log": "暂无随机想法注入"})

            await asyncio.sleep(5)
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected: /ws/sandbox")

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
