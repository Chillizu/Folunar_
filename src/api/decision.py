# 决策引擎API路由
"""
决策引擎相关的API端点
"""

import logging
from fastapi import APIRouter, HTTPException
from src.decision_engine import DecisionEngine
from src.auth import require_auth
from src.security_middleware import AuditLogger

logger = logging.getLogger(__name__)
router = APIRouter()

# 全局变量（将在初始化时设置）
decision_engine: DecisionEngine = None
audit_logger: AuditLogger = None

def init_decision_routes(_decision_engine: DecisionEngine, _audit_logger: AuditLogger):
    """初始化决策引擎路由的依赖"""
    global decision_engine, audit_logger
    decision_engine = _decision_engine
    audit_logger = _audit_logger

@router.post("/api/decision/start")
async def start_decision_engine(current_user: str = require_auth):
    """启动决策引擎"""
    audit_logger.log_event("DECISION_ENGINE_START_REQUEST", {"user": current_user}, current_user)
    try:
        success = await decision_engine.start_decision_loop()
        if success:
            audit_logger.log_event("DECISION_ENGINE_START_SUCCESS", {"user": current_user}, current_user)
            return {"status": "success", "message": "决策引擎已启动"}
        else:
            audit_logger.log_event("DECISION_ENGINE_START_FAILED", {"user": current_user, "reason": "already_running"}, current_user)
            return {"status": "warning", "message": "决策引擎已在运行中"}
    except Exception as e:
        logger.error(f"启动决策引擎失败: {str(e)}")
        audit_logger.log_event("DECISION_ENGINE_START_ERROR", {"user": current_user, "error": str(e)}, current_user)
        raise HTTPException(status_code=500, detail=f"启动决策引擎失败: {str(e)}")

@router.post("/api/decision/stop")
async def stop_decision_engine(current_user: str = require_auth):
    """停止决策引擎"""
    audit_logger.log_event("DECISION_ENGINE_STOP_REQUEST", {"user": current_user}, current_user)
    try:
        success = await decision_engine.stop_decision_loop()
        if success:
            audit_logger.log_event("DECISION_ENGINE_STOP_SUCCESS", {"user": current_user}, current_user)
            return {"status": "success", "message": "决策引擎已停止"}
        else:
            audit_logger.log_event("DECISION_ENGINE_STOP_FAILED", {"user": current_user, "reason": "not_running"}, current_user)
            return {"status": "warning", "message": "决策引擎未在运行"}
    except Exception as e:
        logger.error(f"停止决策引擎失败: {str(e)}")
        audit_logger.log_event("DECISION_ENGINE_STOP_ERROR", {"user": current_user, "error": str(e)}, current_user)
        raise HTTPException(status_code=500, detail=f"停止决策引擎失败: {str(e)}")

@router.get("/api/decision/status")
async def get_decision_engine_status(current_user: str = require_auth):
    """获取决策引擎状态"""
    try:
        status = decision_engine.get_status()
        return {"status": "success", "data": status}
    except Exception as e:
        logger.error(f"获取决策引擎状态失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取决策引擎状态失败: {str(e)}")

@router.get("/api/decision/decisions")
async def get_recent_decisions(limit: int = 10, current_user: str = require_auth):
    """获取最近的决策记录"""
    try:
        decisions = decision_engine.get_recent_decisions(limit)
        return {"status": "success", "data": decisions}
    except Exception as e:
        logger.error(f"获取决策记录失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取决策记录失败: {str(e)}")

@router.post("/api/decision/clear")
async def clear_decisions(current_user: str = require_auth):
    """清空决策历史记录"""
    audit_logger.log_event("DECISION_ENGINE_CLEAR_REQUEST", {"user": current_user}, current_user)
    try:
        decision_engine.clear_decision_history()
        audit_logger.log_event("DECISION_ENGINE_CLEAR_SUCCESS", {"user": current_user}, current_user)
        return {"status": "success", "message": "决策历史记录已清空"}
    except Exception as e:
        logger.error(f"清空决策记录失败: {str(e)}")
        audit_logger.log_event("DECISION_ENGINE_CLEAR_ERROR", {"user": current_user, "error": str(e)}, current_user)
        raise HTTPException(status_code=500, detail=f"清空决策记录失败: {str(e)}")

@router.post("/api/decision/manual")
async def manual_decision_cycle(current_user: str = require_auth):
    """手动触发一次决策周期"""
    audit_logger.log_event("DECISION_ENGINE_MANUAL_REQUEST", {"user": current_user}, current_user)
    try:
        result = await decision_engine.manual_decision()
        if result["success"]:
            audit_logger.log_event("DECISION_ENGINE_MANUAL_SUCCESS", {"user": current_user}, current_user)
            return {"status": "success", "data": result}
        else:
            audit_logger.log_event("DECISION_ENGINE_MANUAL_FAILED", {"user": current_user, "error": result.get("error", "unknown")}, current_user)
            raise HTTPException(status_code=500, detail=result.get("error", "手动决策失败"))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"手动决策失败: {str(e)}")
        audit_logger.log_event("DECISION_ENGINE_MANUAL_ERROR", {"user": current_user, "error": str(e)}, current_user)
        raise HTTPException(status_code=500, detail=f"手动决策失败: {str(e)}")