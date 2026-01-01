"""
AgentManager 单元测试
测试代理管理器的主要功能
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import json

from src.core.agent_manager import AgentManager


class TestAgentManager:
    """AgentManager 测试类"""

    @pytest.fixture
    def config(self):
        """测试配置"""
        return {
            'api': {
                'key': 'test-api-key',
                'base_url': 'https://api.openai.com/v1',
                'default_model': 'gpt-3.5-turbo'
            }
        }

    @pytest.fixture
    def manager(self, config):
        """AgentManager 实例"""
        return AgentManager(config)

    def test_initialization(self, manager, config):
        """测试初始化"""
        assert manager.config == config
        assert manager.agents == {}
        assert manager.tools == {}
        assert manager.default_model == 'gpt-3.5-turbo'
        assert manager.client is not None

    def test_list_agents_empty(self, manager):
        """测试列出空代理列表"""
        agents = manager.list_agents()
        assert agents == []

    def test_register_and_list_agents(self, manager):
        """测试注册代理和列出代理"""
        # 注册代理
        manager.register_agent('agent1', {'type': 'test'})
        manager.register_agent('agent2', {'type': 'mock'})

        agents = manager.list_agents()
        assert 'agent1' in agents
        assert 'agent2' in agents
        assert len(agents) == 2

    def test_get_agent(self, manager):
        """测试获取代理"""
        test_agent = {'type': 'test'}
        manager.register_agent('test_agent', test_agent)

        retrieved = manager.get_agent('test_agent')
        assert retrieved == test_agent

        # 测试获取不存在的代理
        not_found = manager.get_agent('nonexistent')
        assert not_found is None

    def test_unregister_agent(self, manager):
        """测试注销代理"""
        manager.register_agent('agent1', {'type': 'test'})

        # 确认代理存在
        assert manager.get_agent('agent1') is not None

        # 注销代理
        manager.unregister_agent('agent1')

        # 确认代理已被移除
        assert manager.get_agent('agent1') is None

        # 测试注销不存在的代理（应该不报错）
        manager.unregister_agent('nonexistent')

    def test_register_tool(self, manager):
        """测试注册工具"""
        def test_tool():
            return "test result"

        manager.register_tool('test_tool', test_tool)

        assert 'test_tool' in manager.tools
        assert manager.tools['test_tool'] == test_tool

    @pytest.mark.asyncio
    async def test_chat_completion_streaming(self, manager):
        """测试流式聊天完成"""
        messages = [
            {"role": "user", "content": "Hello"}
        ]

        # Mock 流式响应
        mock_chunk1 = MagicMock()
        mock_chunk1.model_dump.return_value = {
            "id": "chatcmpl-123",
            "object": "chat.completion.chunk",
            "created": 1677652288,
            "model": "gpt-3.5-turbo",
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]
        }

        mock_chunk2 = MagicMock()
        mock_chunk2.model_dump.return_value = {
            "id": "chatcmpl-123",
            "object": "chat.completion.chunk",
            "created": 1677652288,
            "model": "gpt-3.5-turbo",
            "choices": [{"index": 0, "delta": {"content": "Hello"}, "finish_reason": None}]
        }

        mock_chunk3 = MagicMock()
        mock_chunk3.model_dump.return_value = {
            "id": "chatcmpl-123",
            "object": "chat.completion.chunk",
            "created": 1677652288,
            "model": "gpt-3.5-turbo",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
        }

        mock_response = AsyncMock()
        mock_response.__aiter__.return_value = [mock_chunk1, mock_chunk2, mock_chunk3]

        with patch.object(manager.client.chat.completions, 'create', return_value=mock_response):
            chunks = []
            async for chunk in manager.chat_completion(messages, stream=True):
                chunks.append(chunk)

            assert len(chunks) == 3
            assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
            assert chunks[1]["choices"][0]["delta"]["content"] == "Hello"
            assert chunks[2]["choices"][0]["finish_reason"] == "stop"

    @pytest.mark.asyncio
    async def test_chat_completion_non_streaming(self, manager):
        """测试非流式聊天完成"""
        messages = [
            {"role": "user", "content": "Hello"}
        ]

        mock_response = MagicMock()
        mock_response.model_dump.return_value = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "gpt-3.5-turbo",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Hello there!"},
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 9,
                "completion_tokens": 3,
                "total_tokens": 12
            }
        }

        with patch.object(manager.client.chat.completions, 'create', return_value=mock_response):
            responses = []
            async for response in manager.chat_completion(messages, stream=False):
                responses.append(response)

            assert len(responses) == 1
            assert responses[0]["choices"][0]["message"]["content"] == "Hello there!"
            assert responses[0]["usage"]["total_tokens"] == 12

    @pytest.mark.asyncio
    async def test_chat_completion_with_tools(self, manager):
        """测试带工具的聊天完成"""
        messages = [
            {"role": "user", "content": "What's the weather?"}
        ]

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather information",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"}
                        }
                    }
                }
            }
        ]

        mock_response = MagicMock()
        mock_response.model_dump.return_value = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "gpt-3.5-turbo",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "call_123",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "New York"}'
                        }
                    }]
                },
                "finish_reason": "tool_calls"
            }],
            "usage": {
                "prompt_tokens": 15,
                "completion_tokens": 8,
                "total_tokens": 23
            }
        }

        with patch.object(manager.client.chat.completions, 'create', return_value=mock_response):
            responses = []
            async for response in manager.chat_completion(messages, tools=tools, stream=False):
                responses.append(response)

            assert len(responses) == 1
            tool_calls = responses[0]["choices"][0]["message"]["tool_calls"]
            assert len(tool_calls) == 1
            assert tool_calls[0]["function"]["name"] == "get_weather"

    @pytest.mark.asyncio
    async def test_chat_completion_error(self, manager):
        """测试聊天完成错误处理"""
        messages = [
            {"role": "user", "content": "Hello"}
        ]

        with patch.object(manager.client.chat.completions, 'create', side_effect=Exception("API Error")):
            with pytest.raises(Exception, match="Chat completion failed: API Error"):
                async for _ in manager.chat_completion(messages):
                    pass

    @pytest.mark.asyncio
    async def test_handle_tool_call_success(self, manager):
        """测试工具调用处理成功"""
        async def mock_tool(location: str):
            return f"Weather in {location}: Sunny"

        manager.register_tool('get_weather', mock_tool)

        tool_call = {
            'function': {
                'name': 'get_weather',
                'arguments': '{"location": "New York"}'
            }
        }

        result = await manager.handle_tool_call(tool_call)
        assert result == "Weather in New York: Sunny"

    @pytest.mark.asyncio
    async def test_handle_tool_call_tool_not_registered(self, manager):
        """测试调用未注册工具"""
        tool_call = {
            'function': {
                'name': 'nonexistent_tool',
                'arguments': '{}'
            }
        }

        with pytest.raises(ValueError, match="Tool nonexistent_tool not registered"):
            await manager.handle_tool_call(tool_call)

    @pytest.mark.asyncio
    async def test_handle_tool_call_invalid_arguments(self, manager):
        """测试工具调用参数解析错误"""
        async def mock_tool(location: str):
            return f"Weather in {location}"

        manager.register_tool('get_weather', mock_tool)

        tool_call = {
            'function': {
                'name': 'get_weather',
                'arguments': 'invalid json'
            }
        }

        # 应该抛出JSON解析错误
        with pytest.raises(json.JSONDecodeError):
            await manager.handle_tool_call(tool_call)

    def test_custom_model(self, manager):
        """测试自定义模型"""
        custom_model = "gpt-4"
        assert manager.default_model == "gpt-3.5-turbo"

        # 创建新实例使用自定义模型
        config_custom = {
            'api': {
                'key': 'test-key',
                'base_url': 'https://api.openai.com/v1',
                'default_model': custom_model
            }
        }
        manager_custom = AgentManager(config_custom)
        assert manager_custom.default_model == custom_model