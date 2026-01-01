# 容器管理API路由
"""
容器管理相关的API端点
"""

import json
import asyncio
import logging
import time
from typing import Optional, Dict
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from src.container_manager import ContainerManager
from src.validation import ContainerExecRequest
from src.auth import require_auth
from src.security_middleware import AuditLogger

logger = logging.getLogger(__name__)
router = APIRouter()

# 全局变量（将在初始化时设置）
container_manager: ContainerManager = None
audit_logger: AuditLogger = None

def init_container_routes(_container_manager: ContainerManager, _audit_logger: AuditLogger):
    """初始化容器路由的依赖"""
    global container_manager, audit_logger
    container_manager = _container_manager
    audit_logger = _audit_logger

@router.post("/api/container/build")
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

@router.post("/api/container/start")
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

@router.post("/api/container/stop")
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

@router.post("/api/container/remove")
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

@router.get("/api/container/status")
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

@router.get("/api/container/monitor")
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

@router.post("/api/container/exec")
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