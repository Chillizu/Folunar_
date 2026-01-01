"""
测试配置和fixtures
"""

import pytest
import asyncio
from unittest.mock import AsyncMock


@pytest.fixture(scope="session")
def event_loop():
    """创建事件循环fixture用于异步测试"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_subprocess():
    """Mock subprocess用于容器管理测试"""
    mock_process = AsyncMock()
    mock_process.returncode = 0
    mock_process.communicate.return_value = (b'', b'')
    return mock_process


@pytest.fixture
def sample_config():
    """示例配置"""
    return {
        'api': {
            'key': 'test-api-key',
            'base_url': 'https://api.openai.com/v1',
            'default_model': 'gpt-3.5-turbo'
        },
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