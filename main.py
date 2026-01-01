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
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from src.core.agent_manager import AgentManager
from src.container_manager import ContainerManager

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config():
    """加载配置文件"""
    with open('config.yaml', 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

# 加载配置
config = load_config()

# 创建FastAPI应用实例
app = FastAPI(
    title=config['app']['name'],
    version=config['app']['version'],
    description=config['app']['description']
)

# 挂载静态文件
app.mount("/static", StaticFiles(directory="static"), name="static")

# 初始化代理管理器
agent_manager = AgentManager(config)

# 初始化容器管理器
container_manager = ContainerManager()

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
async def health_check():
    return {"status": "healthy"}

@app.get("/api/system/status")
async def system_status():
    """获取系统状态信息"""
    try:
        # 获取系统信息
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=1)

        # 计算运行时间（从应用启动开始）
        uptime = time.time() - psutil.boot_time()

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
            "timestamp": int(time.time())
        }

        logger.info(f"System status requested: {status}")
        return status
    except Exception as e:
        logger.error(f"Failed to get system status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get system status: {str(e)}")

# 容器管理端点
@app.post("/api/container/build")
async def build_container():
    """构建Debian容器镜像"""
    try:
        result = container_manager.build_image()
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
        result = container_manager.start_container(ports)
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
        result = container_manager.stop_container()
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
        result = container_manager.remove_container()
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
        result = container_manager.get_container_status()
        if result["success"]:
            return {"status": "success", "data": result["data"]}
        else:
            raise HTTPException(status_code=500, detail=result["error"])
    except Exception as e:
        logger.error(f"获取容器状态失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取容器状态失败: {str(e)}")

@app.post("/api/container/exec")
async def exec_in_container(command: str):
    """在容器中执行命令"""
    result = container_manager.exec_command(command)
    if result["success"]:
        return {"status": "success", "output": result["output"]}
    else:
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
async def chat_completions(request: Request):
    """处理chat completions请求，支持流式和非流式响应"""
    try:
        data = await request.json()
        messages = data.get("messages", [])
        model = data.get("model", "gpt-3.5-turbo")
        stream = data.get("stream", False)
        tools = data.get("tools")

        logger.info(f"Chat completions request: model={model}, stream={stream}, messages_count={len(messages)}")

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
                logger.info("Chat completions response sent")
                return result
    except Exception as e:
        logger.error(f"Chat completions request failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

def create_app():
    """返回FastAPI应用实例（用于工厂模式）"""
    return app

if __name__ == "__main__":
    config = load_config()

    uvicorn.run(
        "main:create_app",
        factory=True,
        host=config['server']['host'],
        port=config['server']['port'],
        reload=config['server']['debug']
    )