#!/usr/bin/env python3
"""
输入验证模块
提供数据验证、清理和安全检查功能
"""

import re
import logging
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, field_validator
from fastapi import HTTPException

logger = logging.getLogger(__name__)

class ChatCompletionRequest(BaseModel):
    """聊天完成请求验证模型"""
    model: str = Field(..., min_length=1, max_length=100)
    messages: List[Dict[str, Any]] = Field(..., min_items=1, max_items=100)
    stream: Optional[bool] = False
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(None, gt=0, le=4096)
    tools: Optional[List[Dict[str, Any]]] = None

    @field_validator('model')
    @classmethod
    def validate_model(cls, v):
        """验证模型名称"""
        allowed_models = [
            'gpt-3.5-turbo', 'gpt-4', 'gpt-4-turbo',
            'claude-3', 'gemini-pro', 'nvidia/nemotron'
        ]
        if not any(allowed in v.lower() for allowed in allowed_models):
            raise ValueError(f'不支持的模型: {v}')
        return v

    @field_validator('messages')
    @classmethod
    def validate_messages(cls, v):
        """验证消息格式"""
        for msg in v:
            if not isinstance(msg, dict):
                raise ValueError('消息必须是字典格式')
            if 'role' not in msg or 'content' not in msg:
                raise ValueError('消息必须包含role和content字段')
            if msg['role'] not in ['system', 'user', 'assistant']:
                raise ValueError(f'无效的role: {msg["role"]}')
            if not isinstance(msg['content'], str) or len(msg['content']) > 10000:
                raise ValueError('消息内容必须是字符串且长度不超过10000字符')
        return v

class ContainerExecRequest(BaseModel):
    """容器执行请求验证模型"""
    command: str = Field(..., min_length=1, max_length=1000)

    @field_validator('command')
    @classmethod
    def validate_command(cls, v):
        """验证命令安全性"""
        # 禁止危险命令
        dangerous_commands = [
            'rm -rf', 'rmdir', 'del', 'format', 'fdisk', 'mkfs',
            'shutdown', 'reboot', 'halt', 'poweroff', 'init',
            'systemctl', 'service', 'killall', 'pkill'
        ]

        v_lower = v.lower()
        for cmd in dangerous_commands:
            if cmd in v_lower:
                raise ValueError(f'禁止执行危险命令: {cmd}')

        # 检查命令注入
        if any(char in v for char in [';', '&', '|', '`', '$', '(', ')']):
            raise ValueError('命令包含非法字符')

        return v

class SystemStatusRequest(BaseModel):
    """系统状态请求验证模型"""
    include_sensitive: Optional[bool] = False

class ValidationManager:
    """验证管理器"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def sanitize_input(self, input_str: str, max_length: int = 1000) -> str:
        """清理输入字符串"""
        if not isinstance(input_str, str):
            raise HTTPException(status_code=400, detail="输入必须是字符串")

        # 移除控制字符
        cleaned = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', input_str)

        # 限制长度
        if len(cleaned) > max_length:
            cleaned = cleaned[:max_length]

        # 移除潜在的XSS向量
        cleaned = re.sub(r'<[^>]*>', '', cleaned)

        return cleaned.strip()

    def validate_json_input(self, data: Any) -> Any:
        """验证JSON输入"""
        if isinstance(data, dict):
            # 检查嵌套深度
            def check_depth(obj, depth=0):
                if depth > 10:  # 最大嵌套深度
                    raise HTTPException(status_code=400, detail="JSON嵌套深度过大")
                if isinstance(obj, dict):
                    for v in obj.values():
                        check_depth(v, depth + 1)
                elif isinstance(obj, list):
                    for item in obj:
                        check_depth(item, depth + 1)
            check_depth(data)

            # 检查对象大小
            import json
            size = len(json.dumps(data).encode('utf-8'))
            if size > 1024 * 1024:  # 1MB限制
                raise HTTPException(status_code=400, detail="请求数据过大")

        return data

    def rate_limit_check(self, key: str, limit: int, window: int) -> bool:
        """速率限制检查"""
        # 这里可以集成Redis或其他缓存来实现真正的速率限制
        # 目前返回True表示允许
        return True

def validate_request_data(model_class: Any, data: Dict[str, Any]) -> Any:
    """验证请求数据"""
    try:
        return model_class(**data)
    except Exception as e:
        logger.warning(f"输入验证失败: {str(e)}")
        raise HTTPException(status_code=400, detail=f"输入验证失败: {str(e)}")