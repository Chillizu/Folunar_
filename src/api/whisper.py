# Whisper注入API路由
"""
随机想法注入系统相关的API端点
"""

import logging
from fastapi import APIRouter, HTTPException
from src.whisper_injection import WhisperInjectionManager
from src.auth import require_auth
from src.security_middleware import AuditLogger

logger = logging.getLogger(__name__)
router = APIRouter()

# 全局变量（将在初始化时设置）
whisper_injection: WhisperInjectionManager = None
audit_logger: AuditLogger = None

def init_whisper_routes(_whisper_injection: WhisperInjectionManager, _audit_logger: AuditLogger):
    """初始化whisper路由的依赖"""
    global whisper_injection, audit_logger
    whisper_injection = _whisper_injection
    audit_logger = _audit_logger

@router.get("/api/whisper/status")
async def get_whisper_status(current_user: str = require_auth):
    """获取随机想法注入系统状态"""
    try:
        status = whisper_injection.get_status()
        return {"status": "success", "data": status}
    except Exception as e:
        logger.error(f"获取注入系统状态失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取注入系统状态失败: {str(e)}")

@router.get("/api/whisper/vocabulary")
async def get_whisper_vocabulary(current_user: str = require_auth):
    """获取词汇库"""
    try:
        vocabulary = whisper_injection.get_vocabulary()
        return {"status": "success", "data": {"vocabulary": vocabulary, "count": len(vocabulary)}}
    except Exception as e:
        logger.error(f"获取词汇库失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取词汇库失败: {str(e)}")

@router.post("/api/whisper/vocabulary/add")
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

@router.delete("/api/whisper/vocabulary/remove")
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

@router.post("/api/whisper/vocabulary/clear")
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

@router.get("/api/whisper/logs")
async def get_whisper_logs(limit: int = 50, current_user: str = require_auth):
    """获取注入日志"""
    try:
        logs = whisper_injection.get_injection_logs(limit)
        return {"status": "success", "data": {"logs": logs, "count": len(logs)}}
    except Exception as e:
        logger.error(f"获取注入日志失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取注入日志失败: {str(e)}")

@router.post("/api/whisper/logs/clear")
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

@router.post("/api/whisper/inject")
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