# 统一错误处理模块
"""
AgentContainer统一错误处理模块
提供标准化的错误响应和异常处理机制
"""

import logging
from typing import Dict, Any, Optional
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class ErrorResponse(BaseModel):
    """标准错误响应模型"""
    status: str = "error"
    error_code: str
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: Optional[float] = None

class ErrorHandler:
    """统一错误处理器"""

    # 错误代码映射
    ERROR_CODES = {
        # 通用错误
        "INTERNAL_ERROR": {"status_code": 500, "message": "内部服务器错误"},
        "VALIDATION_ERROR": {"status_code": 400, "message": "输入验证失败"},
        "NOT_FOUND": {"status_code": 404, "message": "资源未找到"},
        "UNAUTHORIZED": {"status_code": 401, "message": "未授权访问"},
        "FORBIDDEN": {"status_code": 403, "message": "访问被拒绝"},
        "CONFLICT": {"status_code": 409, "message": "资源冲突"},

        # 业务特定错误
        "CONTAINER_ERROR": {"status_code": 500, "message": "容器操作失败"},
        "AGENT_ERROR": {"status_code": 500, "message": "代理操作失败"},
        "AUTH_ERROR": {"status_code": 401, "message": "认证失败"},
        "PERMISSION_ERROR": {"status_code": 403, "message": "权限不足"},
        "CONFIG_ERROR": {"status_code": 500, "message": "配置错误"},
        "NETWORK_ERROR": {"status_code": 503, "message": "网络连接错误"},
        "TIMEOUT_ERROR": {"status_code": 504, "message": "操作超时"},
    }

    @classmethod
    def create_error_response(
        cls,
        error_code: str,
        custom_message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        status_code: Optional[int] = None
    ) -> JSONResponse:
        """创建标准错误响应"""
        import time

        if error_code not in cls.ERROR_CODES:
            logger.warning(f"Unknown error code: {error_code}")
            error_code = "INTERNAL_ERROR"

        error_info = cls.ERROR_CODES[error_code]
        response_status_code = status_code or error_info["status_code"]
        message = custom_message or error_info["message"]

        error_response = ErrorResponse(
            error_code=error_code,
            message=message,
            details=details,
            timestamp=time.time()
        )

        return JSONResponse(
            status_code=response_status_code,
            content=error_response.dict()
        )

    @classmethod
    def handle_exception(
        cls,
        exc: Exception,
        request: Request,
        error_code: str = "INTERNAL_ERROR"
    ) -> JSONResponse:
        """处理异常并返回标准错误响应"""
        logger.error(f"Exception in {request.url.path}: {str(exc)}", exc_info=True)

        # 根据异常类型确定错误代码
        if isinstance(exc, HTTPException):
            error_code = cls._map_http_exception(exc)
        elif isinstance(exc, ValueError):
            error_code = "VALIDATION_ERROR"
        elif isinstance(exc, PermissionError):
            error_code = "PERMISSION_ERROR"
        elif isinstance(exc, TimeoutError):
            error_code = "TIMEOUT_ERROR"
        elif isinstance(exc, ConnectionError):
            error_code = "NETWORK_ERROR"

        return cls.create_error_response(
            error_code=error_code,
            custom_message=str(exc),
            details={"path": request.url.path, "method": request.method}
        )

    @classmethod
    def _map_http_exception(cls, exc: HTTPException) -> str:
        """将HTTPException映射到错误代码"""
        status_code = exc.status_code
        if status_code == 400:
            return "VALIDATION_ERROR"
        elif status_code == 401:
            return "UNAUTHORIZED"
        elif status_code == 403:
            return "FORBIDDEN"
        elif status_code == 404:
            return "NOT_FOUND"
        elif status_code == 409:
            return "CONFLICT"
        else:
            return "INTERNAL_ERROR"

def setup_error_handlers(app):
    """为FastAPI应用设置全局错误处理器"""
    from fastapi import Request
    from fastapi.exceptions import RequestValidationError

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """处理请求验证错误"""
        logger.warning(f"Validation error in {request.url.path}: {exc.errors()}")

        details = {
            "validation_errors": exc.errors(),
            "path": request.url.path,
            "method": request.method
        }

        return ErrorHandler.create_error_response(
            error_code="VALIDATION_ERROR",
            custom_message="请求数据验证失败",
            details=details
        )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """处理HTTP异常"""
        error_code = ErrorHandler._map_http_exception(exc)
        return ErrorHandler.create_error_response(
            error_code=error_code,
            custom_message=exc.detail,
            details={"path": request.url.path, "method": request.method}
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """处理通用异常"""
        return ErrorHandler.handle_exception(exc, request)

# 便捷函数
def create_success_response(data: Any = None, message: str = "操作成功") -> Dict[str, Any]:
    """创建标准成功响应"""
    import time
    return {
        "status": "success",
        "message": message,
        "data": data,
        "timestamp": time.time()
    }

def create_warning_response(message: str, data: Any = None) -> Dict[str, Any]:
    """创建标准警告响应"""
    import time
    return {
        "status": "warning",
        "message": message,
        "data": data,
        "timestamp": time.time()
    }