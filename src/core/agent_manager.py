"""
代理管理器模块
负责管理和协调各种AI代理
"""

import asyncio
from typing import List, Dict, Any, Optional, AsyncGenerator
import openai
from openai import AsyncOpenAI

class AgentManager:
    def __init__(self, config):
        self.config = config
        self.agents = {}  # 存储活跃的代理实例
        self.tools = {}  # 预留工具调用框架
        self.default_model = config.get('api', {}).get('default_model', 'gpt-3.5-turbo')
        self.client = AsyncOpenAI(
            api_key=config.get('api', {}).get('openai_key'),
            base_url=config.get('api', {}).get('base_url', 'https://api.openai.com/v1')
        )

    def list_agents(self):
        """列出所有活跃的代理"""
        return list(self.agents.keys())

    def register_agent(self, agent_id, agent_instance):
        """注册一个新的代理"""
        self.agents[agent_id] = agent_instance

    def unregister_agent(self, agent_id):
        """注销一个代理"""
        if agent_id in self.agents:
            del self.agents[agent_id]

    def get_agent(self, agent_id):
        """获取指定代理实例"""
        return self.agents.get(agent_id)

    def register_tool(self, tool_name: str, tool_func):
        """注册工具函数"""
        self.tools[tool_name] = tool_func

    async def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        model: str = None,
        stream: bool = False,
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> AsyncGenerator[str, None]:
        """
        处理chat completions请求
        支持流式输出和工具调用
        """
        if model is None:
            model = self.default_model
        try:
            if stream:
                # 流式输出
                response = await self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    stream=True,
                    tools=tools
                )
                async for chunk in response:
                    if chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
            else:
                # 非流式输出
                response = await self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    tools=tools
                )
                yield response.model_dump()
        except Exception as e:
            raise Exception(f"Chat completion failed: {str(e)}")

    async def handle_tool_call(self, tool_call: Dict[str, Any]) -> Any:
        """
        处理工具调用
        """
        tool_name = tool_call.get('function', {}).get('name')
        if tool_name in self.tools:
            args = tool_call.get('function', {}).get('arguments', {})
            return await self.tools[tool_name](**args)
        else:
            raise ValueError(f"Tool {tool_name} not registered")