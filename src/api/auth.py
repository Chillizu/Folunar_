# 认证API路由
"""
认证相关的API端点
"""

from fastapi import APIRouter, HTTPException
from src.auth import AuthManager, require_auth, UserCredentials
from src.security_middleware import AuditLogger

router = APIRouter()

# 全局变量（将在初始化时设置）
auth_manager: AuthManager = None
audit_logger: AuditLogger = None

def init_auth_routes(_auth_manager: AuthManager, _audit_logger: AuditLogger):
    """初始化认证路由的依赖"""
    global auth_manager, audit_logger
    auth_manager = _auth_manager
    audit_logger = _audit_logger

@router.post("/api/auth/login")
async def login(credentials: UserCredentials):
    """用户登录"""
    if auth_manager.authenticate_user(credentials.username, credentials.password):
        access_token = auth_manager.create_access_token({"sub": credentials.username})
        audit_logger.log_event("LOGIN_SUCCESS", {"username": credentials.username})
        return {"access_token": access_token, "token_type": "bearer"}
    else:
        audit_logger.log_event("LOGIN_FAILED", {"username": credentials.username})
        raise HTTPException(status_code=401, detail="用户名或密码错误")

@router.post("/api/auth/logout")
async def logout(current_user: str = require_auth):
    """用户登出"""
    audit_logger.log_event("LOGOUT", {"username": current_user})
    return {"message": "登出成功"}

@router.get("/api/auth/me")
async def get_current_user_info(current_user: str = require_auth):
    """获取当前用户信息"""
    return {"username": current_user}