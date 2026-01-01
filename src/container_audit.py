#!/usr/bin/env python3
"""
容器审计日志模块
提供容器操作的详细审计日志记录
"""

import logging
import json
import os
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path


class ContainerAuditLogger:
    """容器审计日志器"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config.get('security', {}).get('container_security', {})
        self.enabled = self.config.get('enable_audit_log', True)
        self.log_file = self.config.get('audit_log_file', 'logs/container_audit.log')
        self.max_log_size = self.config.get('max_log_size', 10*1024*1024)  # 10MB
        self.backup_count = self.config.get('backup_count', 5)

        if self.enabled:
            self._setup_logger()

    def _setup_logger(self):
        """设置审计日志器"""
        # 确保日志目录存在
        log_dir = Path(self.log_file).parent
        log_dir.mkdir(exist_ok=True)

        # 创建审计日志器
        self.audit_logger = logging.getLogger('container_audit')
        self.audit_logger.setLevel(logging.INFO)

        # 避免重复添加handler
        if self.audit_logger.handlers:
            return

        # 创建轮转处理器
        from logging.handlers import RotatingFileHandler
        handler = RotatingFileHandler(
            self.log_file,
            maxBytes=self.max_log_size,
            backupCount=self.backup_count,
            encoding='utf-8'
        )

        # 设置格式
        formatter = logging.Formatter(
            '%(asctime)s - CONTAINER_AUDIT - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        self.audit_logger.addHandler(handler)

    def log_container_event(self, event_type: str, container_name: str,
                          details: Dict[str, Any], user: Optional[str] = None,
                          success: bool = True, error_msg: Optional[str] = None):
        """记录容器事件"""
        if not self.enabled:
            return

        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'event_type': event_type,
            'container_name': container_name,
            'user': user or 'system',
            'success': success,
            'details': details,
            'error_message': error_msg,
            'hostname': os.uname().nodename if hasattr(os, 'uname') else 'unknown'
        }

        self.audit_logger.info(json.dumps(log_data, ensure_ascii=False, default=str))

    def log_command_execution(self, container_name: str, command: str,
                            user: Optional[str] = None, success: bool = True,
                            output_length: Optional[int] = None,
                            error_msg: Optional[str] = None):
        """记录命令执行"""
        details = {
            'command': command,
            'output_length': output_length,
            'filtered': False  # 可以扩展为记录是否被过滤
        }

        self.log_container_event(
            'COMMAND_EXECUTION',
            container_name,
            details,
            user,
            success,
            error_msg
        )

    def log_container_lifecycle(self, event_type: str, container_name: str,
                              details: Dict[str, Any], user: Optional[str] = None,
                              success: bool = True, error_msg: Optional[str] = None):
        """记录容器生命周期事件"""
        self.log_container_event(
            f'CONTAINER_{event_type.upper()}',
            container_name,
            details,
            user,
            success,
            error_msg
        )

    def log_security_violation(self, container_name: str, violation_type: str,
                             details: Dict[str, Any], user: Optional[str] = None):
        """记录安全违规"""
        details['violation_type'] = violation_type

        self.log_container_event(
            'SECURITY_VIOLATION',
            container_name,
            details,
            user,
            False,
            f'Security violation: {violation_type}'
        )

    def log_resource_usage(self, container_name: str, stats: Dict[str, Any]):
        """记录资源使用情况"""
        self.log_container_event(
            'RESOURCE_USAGE',
            container_name,
            stats,
            None,
            True,
            None
        )


# 全局审计日志器实例
_audit_logger = None


def get_audit_logger(config: Dict[str, Any]) -> ContainerAuditLogger:
    """获取全局审计日志器实例"""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = ContainerAuditLogger(config)
    return _audit_logger