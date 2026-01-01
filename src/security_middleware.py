#!/usr/bin/env python3
"""
安全中间件模块
提供HTTPS支持、安全头、CORS、敏感信息过滤等安全功能
"""

import logging
import json
from typing import Dict, Any, Optional, Callable
from fastapi import Request, Response
from fastapi.middleware.base import BaseHTTPMiddleware
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from starlette.middleware.base import BaseHTTPMiddleware as StarletteBaseHTTPMiddleware
from starlette.responses import Response as StarletteResponse

logger = logging.getLogger(__name__)

class SecurityHeadersMiddleware(StarletteBaseHTTPMiddleware):
    """安全头中间件"""

    def __init__(self, app, config: Dict[str, Any]):
        super().__init__(app)
        self.config = config.get('security', {})

    async def dispatch(self, request: Request, call_next: Callable) -> StarletteResponse:
        response = await call_next(request)

        # 添加安全头
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['X-XSS-Protection'] = '1; mode=block'
        response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
        response.headers['Permissions-Policy'] = 'geolocation=(), microphone=(), camera=()'

        # Content Security Policy
        csp = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: https:; "
            "font-src 'self'; "
            "connect-src 'self'; "
            "frame-ancestors 'none';"
        )
        response.headers['Content-Security-Policy'] = csp

        # HSTS (仅HTTPS)
        if request.url.scheme == 'https':
            response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'

        return response

class SensitiveDataFilterMiddleware(StarletteBaseHTTPMiddleware):
    """敏感信息过滤中间件"""

    def __init__(self, app, config: Dict[str, Any]):
        super().__init__(app)
        self.config = config.get('security', {})
        self.sensitive_headers = set(
            self.config.get('sensitive_headers', ['authorization', 'x-api-key', 'cookie'])
        )

    async def dispatch(self, request: Request, call_next: Callable) -> StarletteResponse:
        # 记录请求（过滤敏感信息）
        await self._log_request(request)

        response = await call_next(request)

        # 记录响应（过滤敏感信息）
        await self._log_response(request, response)

        return response

    async def _log_request(self, request: Request):
        """记录请求（过滤敏感信息）"""
        if not self.config.get('enable_audit_log', True):
            return

        # 过滤敏感头
        headers = {}
        for name, value in request.headers.items():
            if name.lower() not in self.sensitive_headers:
                headers[name] = value
            else:
                headers[name] = '[FILTERED]'

        log_data = {
            'method': request.method,
            'url': str(request.url),
            'headers': headers,
            'client_ip': self._get_client_ip(request),
            'user_agent': request.headers.get('user-agent', 'Unknown')
        }

        logger.info(f"AUDIT_REQUEST: {json.dumps(log_data, ensure_ascii=False)}")

    async def _log_response(self, request: Request, response: StarletteResponse):
        """记录响应（过滤敏感信息）"""
        if not self.config.get('enable_audit_log', True):
            return

        # 过滤敏感头
        headers = {}
        for name, value in response.headers.items():
            if name.lower() not in self.sensitive_headers:
                headers[name] = value
            else:
                headers[name] = '[FILTERED]'

        log_data = {
            'method': request.method,
            'url': str(request.url),
            'status_code': response.status_code,
            'headers': headers,
            'client_ip': self._get_client_ip(request)
        }

        logger.info(f"AUDIT_RESPONSE: {json.dumps(log_data, ensure_ascii=False)}")

    def _get_client_ip(self, request: Request) -> str:
        """获取客户端IP"""
        # 尝试从代理头获取真实IP
        forwarded = request.headers.get('x-forwarded-for')
        if forwarded:
            return forwarded.split(',')[0].strip()

        real_ip = request.headers.get('x-real-ip')
        if real_ip:
            return real_ip

        # 默认使用连接IP
        client = request.client
        return client.host if client else 'unknown'

class AuditLogger:
    """审计日志器"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config.get('security', {})
        self.enabled = self.config.get('enable_audit_log', True)
        self.log_file = self.config.get('audit_log_file', 'logs/audit.log')

        if self.enabled:
            # 配置审计日志
            self.audit_logger = logging.getLogger('audit')
            self.audit_logger.setLevel(logging.INFO)

            # 避免重复添加handler
            if not self.audit_logger.handlers:
                handler = logging.FileHandler(self.log_file, encoding='utf-8')
                formatter = logging.Formatter(
                    '%(asctime)s - AUDIT - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S'
                )
                handler.setFormatter(formatter)
                self.audit_logger.addHandler(handler)

    def log_event(self, event_type: str, details: Dict[str, Any], user: Optional[str] = None):
        """记录审计事件"""
        if not self.enabled:
            return

        log_data = {
            'event_type': event_type,
            'user': user or 'anonymous',
            'details': details,
            'timestamp': json.dumps(details, ensure_ascii=False, default=str)
        }

        self.audit_logger.info(json.dumps(log_data, ensure_ascii=False))

def setup_security_middleware(app, config: Dict[str, Any]):
    """设置安全中间件"""

    # CORS中间件
    cors_origins = config.get('security', {}).get('cors_origins', ['*'])
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # HTTPS重定向（如果启用）
    if config.get('security', {}).get('enable_https', False):
        app.add_middleware(HTTPSRedirectMiddleware)

    # 安全头中间件
    app.add_middleware(SecurityHeadersMiddleware, config=config)

    # 敏感信息过滤中间件
    app.add_middleware(SensitiveDataFilterMiddleware, config=config)

    logger.info("安全中间件已配置完成")</content>
</xai:function_call">### 4. 更新main.py集成安全功能

现在我需要更新main.py来集成所有安全功能。让我逐步修改main.py。 

<xai:function_call name="apply_diff">
<parameter name="path">main.py