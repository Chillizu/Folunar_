# 观察者API路由
"""
观察者相关的API端点
"""

import logging
from fastapi import APIRouter, HTTPException
from src.observer import Observer
from src.auth import require_auth
from src.security_middleware import AuditLogger

logger = logging.getLogger(__name__)
router = APIRouter()

# 全局变量（将在初始化时设置）
observer: Observer = None
audit_logger: AuditLogger = None

def init_observer_routes(_observer: Observer, _audit_logger: AuditLogger):
    """初始化观察者路由的依赖"""
    global observer, audit_logger
    observer = _observer
    audit_logger = _audit_logger

@router.post("/api/observer/start")
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

@router.post("/api/observer/stop")
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

@router.get("/api/observer/status")
async def get_observer_status(current_user: str = require_auth):
    """获取观察者状态"""
    try:
        status = observer.get_status()
        return {"status": "success", "data": status}
    except Exception as e:
        logger.error(f"获取观察者状态失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取观察者状态失败: {str(e)}")

@router.get("/api/observer/observations")
async def get_recent_observations(limit: int = 10, current_user: str = require_auth):
    """获取最近的观察记录"""
    try:
        observations = observer.get_recent_observations(limit)
        return {"status": "success", "data": observations}
    except Exception as e:
        logger.error(f"获取观察记录失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取观察记录失败: {str(e)}")

@router.post("/api/observer/clear")
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