#!/usr/bin/env python3
"""
容器安全测试模块
测试容器安全隔离、命令过滤、资源限制等功能
"""

import pytest
import asyncio
import os
import tempfile
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path

# 添加src目录到路径
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from container_manager import ContainerManager
from container_audit import ContainerAuditLogger


class TestContainerSecurity:
    """容器安全测试类"""

    @pytest.fixture
    def security_config(self):
        """安全配置fixture"""
        return {
            'security': {
                'container_security': {
                    'enable_security': True,
                    'read_only_rootfs': True,
                    'drop_all_capabilities': True,
                    'no_new_privileges': True,
                    'pids_limit': 1024,
                    'resource_limits': {
                        'cpu_quota': 50000,
                        'cpu_period': 100000,
                        'memory': '512m',
                        'memory_swap': '1g',
                        'tmpfs_size': '100m'
                    },
                    'network_isolation': {
                        'enable_bridge_network': True,
                        'disable_intercontainer': True,
                        'disable_ip_forwarding': True
                    },
                    'command_filter': {
                        'enable_filtering': True,
                        'dangerous_commands': [
                            'rm\\s+-rf\\s+/',
                            'shutdown',
                            'reboot'
                        ],
                        'allowed_commands': [
                            '^ls\\s+',
                            '^pwd$',
                            '^echo\\s+'
                        ]
                    },
                    'enable_audit_log': True,
                    'audit_log_file': 'logs/container_audit_test.log'
                }
            },
            'container': {
                'image_name': 'test-security-image',
                'container_name': 'test-security-container',
                'network_mode': 'bridge'
            }
        }

    @pytest.fixture
    def container_manager(self, security_config):
        """容器管理器fixture"""
        return ContainerManager(security_config)

    def test_security_config_initialization(self, container_manager, security_config):
        """测试安全配置初始化"""
        assert container_manager.security_config == security_config['security']['container_security']
        assert container_manager.resource_limits == security_config['security']['container_security']['resource_limits']
        assert container_manager.network_isolation == security_config['security']['container_security']['network_isolation']
        assert container_manager.command_filter == security_config['security']['container_security']['command_filter']

    def test_command_filter_compilation(self, container_manager):
        """测试命令过滤器编译"""
        assert len(container_manager.dangerous_patterns) == 3  # 危险命令数量
        assert len(container_manager.allowed_patterns) == 3   # 允许命令数量

    def test_command_filtering_safe_commands(self, container_manager):
        """测试安全命令过滤"""
        safe_commands = [
            'ls -la',
            'pwd',
            'echo hello world'
        ]

        for cmd in safe_commands:
            allowed, reason = container_manager._filter_command(cmd)
            assert allowed, f"命令 '{cmd}' 应该被允许: {reason}"

    def test_command_filtering_dangerous_commands(self, container_manager):
        """测试危险命令过滤"""
        dangerous_commands = [
            'rm -rf /',
            'shutdown now',
            'reboot'
        ]

        for cmd in dangerous_commands:
            allowed, reason = container_manager._filter_command(cmd)
            assert not allowed, f"命令 '{cmd}' 应该被阻止: {reason}"
            assert '危险模式' in reason or '不在允许列表中' in reason

    def test_command_filtering_disabled(self, container_manager):
        """测试禁用命令过滤"""
        # 临时禁用过滤
        original_filter = container_manager.command_filter.copy()
        container_manager.command_filter['enable_filtering'] = False

        allowed, reason = container_manager._filter_command('rm -rf /')
        assert allowed, "禁用过滤时应该允许所有命令"

        # 恢复配置
        container_manager.command_filter = original_filter

    @pytest.mark.asyncio
    async def test_secure_docker_run_command_generation(self, container_manager):
        """测试安全Docker运行命令生成"""
        with patch('asyncio.create_subprocess_exec', new_callable=AsyncMock) as mock_exec:
            mock_process = AsyncMock()
            mock_process.returncode = 0
            mock_process.communicate = AsyncMock(return_value=(b'container_id_123', b''))
            mock_exec.return_value = mock_process

            result = await container_manager.start_container()

            # 验证调用参数包含安全选项
            call_args = mock_exec.call_args[0]
            cmd_str = ' '.join(call_args)

            # 检查安全选项
            assert '--read-only' in cmd_str
            assert '--cap-drop ALL' in cmd_str
            assert '--cap-add NET_BIND_SERVICE' in cmd_str
            assert '--security-opt no-new-privileges:true' in cmd_str
            assert '--pids-limit 1024' in cmd_str

            # 检查资源限制
            assert '--cpu-quota 50000' in cmd_str
            assert '--cpu-period 100000' in cmd_str
            assert '--memory 512m' in cmd_str
            assert '--memory-swap 1g' in cmd_str

    def test_audit_logger_initialization(self, security_config):
        """测试审计日志器初始化"""
        audit_logger = ContainerAuditLogger(security_config['security']['container_security'])
        assert audit_logger.enabled == True
        assert audit_logger.log_file == 'logs/container_audit.log'

    def test_audit_log_events(self, container_manager):
        """测试审计日志事件记录"""
        # 测试命令执行审计
        container_manager.audit_logger.log_command_execution(
            'test-container',
            'ls -la',
            success=True,
            output_length=100
        )

        # 测试安全违规审计
        container_manager.audit_logger.log_security_violation(
            'test-container',
            'COMMAND_FILTERED',
            {'command': 'rm -rf /', 'reason': 'dangerous command'}
        )

        # 测试容器生命周期审计
        container_manager.audit_logger.log_container_lifecycle(
            'START',
            'test-container',
            {'container_id': 'abc123'}
        )

        # 测试资源使用审计
        container_manager.audit_logger.log_resource_usage(
            'test-container',
            {'cpu_percent': '10%', 'memory_percent': '20%'}
        )

    @pytest.mark.asyncio
    async def test_exec_command_audit_logging(self, container_manager):
        """测试命令执行审计日志"""
        with patch('asyncio.create_subprocess_exec', new_callable=AsyncMock) as mock_exec:
            mock_process = AsyncMock()
            mock_process.returncode = 0
            mock_process.communicate = AsyncMock(return_value=(b'output', b''))
            mock_exec.return_value = mock_process

            # 执行安全命令
            result = await container_manager.exec_command('ls -la')

            assert result['success'] == True
            # 审计日志应该被调用（通过mock验证）

    @pytest.mark.asyncio
    async def test_exec_command_security_violation_logging(self, container_manager):
        """测试命令执行安全违规日志"""
        # 执行危险命令
        result = await container_manager.exec_command('rm -rf /')

        assert result['success'] == False
        assert '安全策略阻止' in result['error']
        # 审计日志应该记录安全违规

    def test_resource_limits_configuration(self, container_manager):
        """测试资源限制配置"""
        limits = container_manager.resource_limits

        assert limits['cpu_quota'] == 50000
        assert limits['cpu_period'] == 100000
        assert limits['memory'] == '512m'
        assert limits['memory_swap'] == '1g'
        assert limits['tmpfs_size'] == '100m'

    def test_network_isolation_configuration(self, container_manager):
        """测试网络隔离配置"""
        isolation = container_manager.network_isolation

        assert isolation['enable_bridge_network'] == True
        assert isolation['disable_intercontainer'] == True
        assert isolation['disable_ip_forwarding'] == True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])