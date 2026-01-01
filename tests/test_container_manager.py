"""
ContainerManager 单元测试
测试容器管理器的主要功能
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path
import tempfile
import os

from src.container_manager import ContainerManager


class TestContainerManager:
    """ContainerManager 测试类"""

    @pytest.fixture
    def config(self):
        """测试配置"""
        return {
            'container': {
                'image_name': 'test-image',
                'container_name': 'test-container',
                'dockerfile_path': 'Dockerfile',
                'build_timeout': 60,
                'exec_timeout': 10,
                'stats_timeout': 5,
                'ports': {'8080': '80'},
                'environment': {'TEST_VAR': 'test_value'},
                'volumes': {'/host/path': '/container/path'},
                'restart_policy': 'no',
                'network_mode': 'bridge'
            }
        }

    @pytest.fixture
    def manager(self, config):
        """ContainerManager 实例"""
        return ContainerManager(config)

    @pytest.mark.asyncio
    async def test_build_image_success(self, manager, mocker):
        """测试镜像构建成功"""
        # Mock subprocess
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate.return_value = (b'Build successful', b'')
        mock_process.wait_for = AsyncMock()

        with patch('asyncio.create_subprocess_exec', return_value=mock_process):
            result = await manager.build_image()

            assert result['success'] is True
            assert 'Build successful' in result['output']
            assert result['error'] == ''

    @pytest.mark.asyncio
    async def test_build_image_failure(self, manager, mocker):
        """测试镜像构建失败"""
        mock_process = AsyncMock()
        mock_process.returncode = 1
        mock_process.communicate.return_value = (b'', b'Build failed: no such file')

        with patch('asyncio.create_subprocess_exec', return_value=mock_process):
            result = await manager.build_image()

            assert result['success'] is False
            assert 'Build failed' in result['error']

    @pytest.mark.asyncio
    async def test_build_image_timeout(self, manager, mocker):
        """测试镜像构建超时"""
        mock_process = AsyncMock()
        mock_process.communicate = AsyncMock(side_effect=asyncio.TimeoutError)

        with patch('asyncio.create_subprocess_exec', return_value=mock_process):
            with patch('asyncio.wait_for', side_effect=asyncio.TimeoutError):
                result = await manager.build_image()

                assert result['success'] is False
                assert '超时' in result['error']

    @pytest.mark.asyncio
    async def test_start_container_success(self, manager, mocker):
        """测试容器启动成功"""
        # Mock 检查容器是否运行
        with patch.object(manager, 'is_container_running', return_value=False):
            mock_process = AsyncMock()
            mock_process.returncode = 0
            mock_process.communicate.return_value = (b'container_id_123', b'')

            with patch('asyncio.create_subprocess_exec', return_value=mock_process):
                result = await manager.start_container()

                assert result['success'] is True
                assert result['container_id'] == 'container_id_123'

    @pytest.mark.asyncio
    async def test_start_container_already_running(self, manager, mocker):
        """测试启动已运行的容器"""
        with patch.object(manager, 'is_container_running', return_value=True):
            result = await manager.start_container()

            assert result['success'] is True
            assert result['error'] == ''

    @pytest.mark.asyncio
    async def test_stop_container_success(self, manager, mocker):
        """测试容器停止成功"""
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate.return_value = (b'', b'')

        with patch('asyncio.create_subprocess_exec', return_value=mock_process):
            result = await manager.stop_container()

            assert result['success'] is True
            assert result['error'] == ''

    @pytest.mark.asyncio
    async def test_remove_container_success(self, manager, mocker):
        """测试容器删除成功"""
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate.return_value = (b'', b'')

        with patch('asyncio.create_subprocess_exec', return_value=mock_process):
            result = await manager.remove_container()

            assert result['success'] is True
            assert result['error'] == ''

    @pytest.mark.asyncio
    async def test_is_container_running_true(self, manager, mocker):
        """测试容器运行状态检查 - 运行中"""
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate.return_value = (b'test-container\n', b'')

        with patch('asyncio.create_subprocess_exec', return_value=mock_process):
            result = await manager.is_container_running()

            assert result is True

    @pytest.mark.asyncio
    async def test_is_container_running_false(self, manager, mocker):
        """测试容器运行状态检查 - 未运行"""
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate.return_value = (b'', b'')

        with patch('asyncio.create_subprocess_exec', return_value=mock_process):
            result = await manager.is_container_running()

            assert result is False

    @pytest.mark.asyncio
    async def test_get_container_status_running(self, manager, mocker):
        """测试获取运行中容器的状态"""
        mock_json_output = '''{
            "Command": "nginx",
            "CreatedAt": "2023-01-01T00:00:00Z",
            "ID": "abc123",
            "Image": "nginx:latest",
            "Names": "test-container",
            "Ports": "0.0.0.0:8080->80/tcp",
            "Status": "Up 2 hours"
        }'''

        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate.return_value = (mock_json_output.encode(), b'')

        # Mock stats
        with patch('asyncio.create_subprocess_exec', return_value=mock_process):
            with patch.object(manager, 'get_container_stats', return_value={
                'success': True,
                'data': {'cpu_percent': '5.0%', 'memory_percent': '10.0%'}
            }):
                result = await manager.get_container_status()

                assert result['success'] is True
                assert result['data']['running'] is True
                assert result['data']['name'] == 'test-container'
                assert 'cpu_percent' in result['data']

    @pytest.mark.asyncio
    async def test_get_container_status_not_found(self, manager, mocker):
        """测试获取不存在容器的状态"""
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate.return_value = (b'', b'')

        with patch('asyncio.create_subprocess_exec', return_value=mock_process):
            result = await manager.get_container_status()

            assert result['success'] is True
            assert result['data']['running'] is False
            assert result['data']['status'] == 'Not found'

    @pytest.mark.asyncio
    async def test_get_container_stats_success(self, manager, mocker):
        """测试获取容器统计信息成功"""
        mock_json_output = '''{
            "BlockIO": "0B / 0B",
            "CPUPerc": "1.23%",
            "Container": "abc123",
            "ID": "abc123",
            "MemPerc": "5.67%",
            "MemUsage": "10MiB / 200MiB",
            "Name": "test-container",
            "NetIO": "1.2kB / 3.4kB",
            "PIDs": "5"
        }'''

        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate.return_value = (mock_json_output.encode(), b'')

        with patch('asyncio.create_subprocess_exec', return_value=mock_process):
            result = await manager.get_container_stats()

            assert result['success'] is True
            assert result['data']['cpu_percent'] == '1.23%'
            assert result['data']['memory_percent'] == '5.67%'
            assert result['data']['pids'] == '5'

    @pytest.mark.asyncio
    async def test_exec_command_success(self, manager, mocker):
        """测试在容器中执行命令成功"""
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate.return_value = (b'command output', b'')

        with patch('asyncio.create_subprocess_exec', return_value=mock_process):
            result = await manager.exec_command('echo hello')

            assert result['success'] is True
            assert result['output'] == 'command output'
            assert result['error'] == ''

    @pytest.mark.asyncio
    async def test_exec_command_failure(self, manager, mocker):
        """测试在容器中执行命令失败"""
        mock_process = AsyncMock()
        mock_process.returncode = 1
        mock_process.communicate.return_value = (b'', b'command failed')

        with patch('asyncio.create_subprocess_exec', return_value=mock_process):
            result = await manager.exec_command('invalid_command')

            assert result['success'] is False
            assert result['output'] == ''
            assert 'command failed' in result['error']

    def test_logger_setup(self, manager):
        """测试日志记录器设置"""
        assert manager.logger is not None
        assert manager.logger.name == f"src.container_manager.ContainerManager.test-container"

    def test_config_initialization(self, config):
        """测试配置初始化"""
        manager = ContainerManager(config)

        assert manager.image_name == 'test-image'
        assert manager.container_name == 'test-container'
        assert manager.build_timeout == 60
        assert manager.ports == {'8080': '80'}
        assert manager.environment == {'TEST_VAR': 'test_value'}