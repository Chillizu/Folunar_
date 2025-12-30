"""
代理管理器模块
负责管理和协调各种AI代理
"""

class AgentManager:
    def __init__(self, config):
        self.config = config
        self.agents = {}  # 存储活跃的代理实例

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