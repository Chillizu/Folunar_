"""
观察者→决策引擎→容器执行的闭环集成测试
测试整个AI沙盒系统的完整工作流程
"""

import pytest
import asyncio
import json
from unittest.mock import patch, AsyncMock, MagicMock
import yaml
import os
import tempfile


@pytest.fixture(scope="session")
def test_config():
    """测试配置"""
    return {
        'app': {
            'name': 'AgentContainer Test',
            'version': '1.0.0',
            'description': 'Test instance'
        },
        'server': {
            'host': '127.0.0.1',
            'port': 8001
        },
        'api': {
            'base_url': 'https://api.openai.com/v1',
            'key': 'test-key',
            'default_model': 'gpt-3.5-turbo',
            'timeout': 30,
            'max_retries': 3,
            'retry_delay': 1
        },
        'container': {
            'image_name': 'test-debian',
            'container_name': 'test-agent-debian',
            'dockerfile_path': 'Dockerfile',
            'build_timeout': 60,
            'exec_timeout': 10,
            'stats_timeout': 5,
            'ports': {},
            'environment': {},
            'volumes': {},
            'restart_policy': 'no',
            'network_mode': 'bridge'
        },
        'cache': {
            'redis_enabled': False,
            'default_ttl': 300,
            'max_memory_cache': 100
        },
        'connection_pool': {
            'docker_pool_size': 2,
            'http_pool_size': 10,
            'executor_pool_size': 5
        },
        'performance': {
            'metrics_window': 100,
            'monitor_interval': 5,
            'health_thresholds': {
                'max_response_time': 5.0,
                'max_cpu_percent': 90.0,
                'max_memory_percent': 90.0,
                'max_error_rate': 0.1
            }
        },
        'security': {
            'jwt_secret_key': 'test-jwt-secret-key-for-testing-only',
            'jwt_algorithm': 'HS256',
            'jwt_expiration_hours': 24,
            'admin_username': 'admin',
            'admin_password': 'admin123',
            'enable_audit_log': False,
            'cors_origins': ['*']
        },
        'whisper_injection': {
            'enabled': True,
            'vocabulary_file': 'data/test_vocabulary.json',
            'injection_interval_minutes': 1,
            'log_file': 'logs/test_whisper_injection.log',
            'max_log_entries': 100,
            'default_vocabulary': ['测试词汇1', '测试词汇2']
        },
        'decision_engine': {
            'enabled': True,
            'interval': 5,
            'model': 'gpt-3.5-turbo',
            'log_file': 'logs/test_decision_engine.log',
            'dangerous_commands': ['rm -rf /'],
            'allowed_commands': [
                '^ls\\s+', '^pwd$', '^echo\\s+', '^whoami$', '^date$',
                '^ps\\s+', '^df\\s+', '^free$', '^uptime$', '^id$',
                '^mkdir\\s+', '^touch\\s+', '^cat\\s+', '^head\\s+', '^tail\\s+',
                '^grep\\s+', '^find\\s+', '^wc\\s+', '^sort\\s+', '^uniq\\s+'
            ]
        },
        'observer': {
            'interval': 30,
            'vision_model': 'gpt-4-vision-preview',
            'screenshot_dir': 'screenshots',
            'log_file': 'logs/test_observer.log'
        }
    }


# 移除temp_config_file fixture，因为它会导致完整的应用初始化


class TestClosedLoopIntegration:
    """观察者→决策引擎→容器执行的闭环集成测试"""

    @pytest.fixture
    def mock_observer(self, test_config):
        """Mock观察者"""
        from unittest.mock import AsyncMock, MagicMock
        from src.observer import Observer

        observer = Observer(test_config, MagicMock())
        observer.start_monitoring = AsyncMock(return_value=True)
        observer.stop_monitoring = AsyncMock(return_value=True)
        observer.get_recent_observations = MagicMock(return_value=[
            {
                "timestamp": "2024-01-01T12:00:00",
                "analysis": {
                    "structured_analysis": {
                        "summary": "AI沙盒环境正常，显示终端界面"
                    }
                }
            }
        ])
        return observer

    @pytest.fixture
    def mock_container_manager(self, test_config):
        """Mock容器管理器"""
        from unittest.mock import AsyncMock
        from src.container_manager import ContainerManager

        manager = ContainerManager(test_config)
        manager.exec_command = AsyncMock(return_value={
            "success": True,
            "output": "test command output",
            "error": ""
        })
        return manager

    @pytest.fixture
    def mock_decision_engine(self, test_config, mock_observer, mock_container_manager):
        """Mock决策引擎"""
        from unittest.mock import AsyncMock, MagicMock
        from src.decision_engine import DecisionEngine

        engine = DecisionEngine(test_config, mock_observer, mock_container_manager)
        engine.start_decision_loop = AsyncMock(return_value=True)
        engine.stop_decision_loop = AsyncMock(return_value=True)
        return engine

    @patch('src.observer.Observer._take_screenshot')
    @patch('src.observer.Observer._analyze_screenshot')
    def test_observer_analysis_flow(self, mock_analyze, mock_screenshot, test_config):
        """测试观察者分析流程"""
        from unittest.mock import MagicMock
        from src.observer import Observer

        # Mock截图
        mock_screenshot.return_value = MagicMock()

        # Mock分析结果
        mock_analyze.return_value = {
            "raw_analysis": "AI沙盒环境分析完成",
            "structured_analysis": {
                "summary": "终端界面正常显示"
            },
            "model": "gpt-4-vision-preview"
        }

        observer = Observer(test_config, MagicMock())

        # 执行观察
        async def run_test():
            await observer._perform_observation()

        asyncio.run(run_test())

        # 验证观察结果被记录
        assert len(observer.observations) == 1
        observation = observer.observations[0]
        assert "analysis" in observation
        assert observation["analysis"]["structured_analysis"]["summary"] == "终端界面正常显示"

    @patch('src.decision_engine.DecisionEngine._generate_decision')
    def test_decision_engine_command_generation(self, mock_generate, test_config, mock_observer, mock_container_manager):
        """测试决策引擎命令生成"""
        from src.decision_engine import DecisionEngine

        # Mock决策生成
        mock_generate.return_value = {
            "reasoning": "需要探索环境",
            "command": "ls -la",
            "expected_outcome": "查看目录内容",
            "risk_level": "low"
        }

        engine = DecisionEngine(test_config, mock_observer, mock_container_manager)

        # 执行决策周期
        async def run_test():
            await engine._perform_decision_cycle()

        asyncio.run(run_test())

        # 验证决策被记录
        assert len(engine.decisions) == 1
        decision_record = engine.decisions[0]
        assert "decision" in decision_record
        assert decision_record["decision"]["command"] == "ls -la"

    def test_command_validation(self, test_config, mock_observer, mock_container_manager):
        """测试命令验证"""
        from src.decision_engine import DecisionEngine

        engine = DecisionEngine(test_config, mock_observer, mock_container_manager)

        # 测试安全命令
        is_valid, message = engine._validate_command("ls -la")
        assert is_valid
        assert "验证通过" in message

        # 测试危险命令
        is_valid, message = engine._validate_command("rm -rf /")
        assert not is_valid
        assert "危险操作" in message

    @patch('src.decision_engine.DecisionEngine._validate_command')
    async def test_decision_execution_flow(self, mock_validate, test_config, mock_observer, mock_container_manager):
        """测试决策执行流程"""
        from src.decision_engine import DecisionEngine

        # Mock验证通过
        mock_validate.return_value = (True, "命令验证通过")

        engine = DecisionEngine(test_config, mock_observer, mock_container_manager)

        decision = {
            "command": "pwd",
            "reasoning": "检查当前目录"
        }

        # 执行决策
        result = await engine._execute_decision(decision)

        # 验证执行结果
        assert result["success"] is True
        assert "output" in result
        assert result["command"] == "pwd"

        # 验证容器管理器被调用
        mock_container_manager.exec_command.assert_called_once_with("pwd")

    @patch('src.observer.Observer._take_screenshot')
    @patch('src.observer.Observer._analyze_screenshot')
    @patch('src.decision_engine.DecisionEngine._generate_decision')
    @patch('src.decision_engine.DecisionEngine._validate_command')
    async def test_complete_closed_loop(self, mock_validate, mock_generate_decision, mock_analyze, mock_screenshot,
                                       test_config, mock_observer, mock_container_manager):
        """测试完整的观察→决策→执行闭环"""
        from src.observer import Observer
        from src.decision_engine import DecisionEngine

        # 设置Mock
        mock_screenshot.return_value = MagicMock()
        mock_analyze.return_value = {
            "raw_analysis": "AI正在使用终端",
            "structured_analysis": {"summary": "终端界面活跃"}
        }
        mock_generate_decision.return_value = {
            "reasoning": "探索当前环境",
            "command": "whoami",
            "expected_outcome": "获取当前用户",
            "risk_level": "low"
        }
        mock_validate.return_value = (True, "安全命令")

        # 创建组件实例
        observer = Observer(test_config, MagicMock())
        decision_engine = DecisionEngine(test_config, observer, mock_container_manager)

        # 1. 执行观察
        await observer._perform_observation()

        # 2. 执行决策周期
        await decision_engine._perform_decision_cycle()

        # 验证闭环结果
        assert len(observer.observations) == 1
        assert len(decision_engine.decisions) == 1

        decision_record = decision_engine.decisions[0]
        assert decision_record["decision"]["command"] == "whoami"
        assert "execution_result" in decision_record["decision"]
        assert decision_record["decision"]["execution_result"]["success"] is True

        # 验证调用链
        mock_screenshot.assert_called_once()
        mock_analyze.assert_called_once()
        mock_generate_decision.assert_called_once()
        mock_container_manager.exec_command.assert_called_once_with("whoami")

    @patch('src.decision_engine.DecisionEngine._generate_decision')
    async def test_decision_loop_integration(self, mock_generate_decision, test_config, mock_observer, mock_container_manager):
        """测试决策循环集成"""
        from src.decision_engine import DecisionEngine

        # Mock决策生成
        call_count = 0
        async def mock_decision(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {
                    "reasoning": "第一次决策",
                    "command": "echo 'hello'",
                    "expected_outcome": "测试输出",
                    "risk_level": "low"
                }
            else:
                # 停止循环
                raise asyncio.CancelledError()

        mock_generate_decision.side_effect = mock_decision

        engine = DecisionEngine(test_config, mock_observer, mock_container_manager)

        # 手动执行一次决策周期（而不是启动循环）
        await engine._perform_decision_cycle()

        # 验证执行了一次决策
        assert mock_generate_decision.call_count == 1
        assert len(engine.decisions) == 1

    async def test_error_handling_in_closed_loop(self, test_config, mock_observer, mock_container_manager):
        """测试闭环中的错误处理"""
        from src.decision_engine import DecisionEngine

        # Mock容器执行失败
        mock_container_manager.exec_command.return_value = {
            "success": False,
            "output": "",
            "error": "命令执行失败"
        }

        engine = DecisionEngine(test_config, mock_observer, mock_container_manager)

        decision = {
            "command": "invalid_command",
            "reasoning": "测试错误处理"
        }

        # 执行决策
        result = await engine._execute_decision(decision)

        # 验证错误被正确处理
        assert result["success"] is False
        assert "error" in result
        assert result["command"] == "invalid_command"

    async def test_empty_observations_handling(self, test_config, mock_container_manager):
        """测试无观察数据时的处理"""
        from unittest.mock import MagicMock
        from src.decision_engine import DecisionEngine

        # Mock观察者返回空数据
        mock_observer = MagicMock()
        mock_observer.get_recent_observations.return_value = []

        engine = DecisionEngine(test_config, mock_observer, mock_container_manager)

        # 执行决策周期
        await engine._perform_decision_cycle()

        # 验证没有执行命令（因为没有观察数据）
        mock_container_manager.exec_command.assert_not_called()
        # 当没有观察数据时，不会记录决策（这是正确的行为）
        assert len(engine.decisions) == 0

    async def test_multiple_decisions_integration(self, test_config, mock_observer, mock_container_manager):
        """测试多轮决策集成"""
        from src.decision_engine import DecisionEngine

        engine = DecisionEngine(test_config, mock_observer, mock_container_manager)

        decisions = [
            {"command": "pwd", "reasoning": "检查位置"},
            {"command": "ls -la", "reasoning": "查看文件"},
            {"command": "date", "reasoning": "检查时间"}
        ]

        # 执行多轮决策
        for decision in decisions:
            await engine._execute_decision(decision)

        # 验证所有命令都被执行
        assert mock_container_manager.exec_command.call_count == 3
        calls = mock_container_manager.exec_command.call_args_list
        assert calls[0][0][0] == "pwd"
        assert calls[1][0][0] == "ls -la"
        assert calls[2][0][0] == "date"


class TestPerformanceStress:
    """性能压力测试"""

    @pytest.fixture
    def mock_container_manager(self, test_config):
        """Mock容器管理器用于性能测试"""
        from unittest.mock import AsyncMock
        from src.container_manager import ContainerManager

        manager = ContainerManager(test_config)
        # Mock快速响应
        manager.exec_command = AsyncMock(return_value={
            "success": True,
            "output": "test output",
            "error": ""
        })
        return manager

    async def test_concurrent_command_execution(self, test_config, mock_container_manager):
        """测试并发命令执行"""
        from src.decision_engine import DecisionEngine
        from unittest.mock import MagicMock

        mock_observer = MagicMock()
        mock_observer.get_recent_observations.return_value = [{"analysis": {"structured_analysis": {"summary": "test"}}}]

        engine = DecisionEngine(test_config, mock_observer, mock_container_manager)

        # 并发执行多个命令
        commands = [f"echo 'test {i}'" for i in range(10)]

        async def execute_command(cmd):
            decision = {"command": cmd, "reasoning": f"测试命令 {cmd}"}
            return await engine._execute_decision(decision)

        # 并发执行
        tasks = [execute_command(cmd) for cmd in commands]
        results = await asyncio.gather(*tasks)

        # 验证所有命令都成功执行
        assert all(result["success"] for result in results)
        assert len(results) == 10
        assert mock_container_manager.exec_command.call_count == 10

    async def test_high_frequency_decisions(self, test_config, mock_container_manager):
        """测试高频决策处理"""
        from src.decision_engine import DecisionEngine
        from unittest.mock import MagicMock, AsyncMock

        mock_observer = MagicMock()
        mock_observer.get_recent_observations.return_value = [{"analysis": {"structured_analysis": {"summary": "test"}}}]

        engine = DecisionEngine(test_config, mock_observer, mock_container_manager)

        # Mock决策生成
        engine._generate_decision = AsyncMock(return_value={
            "reasoning": "高频测试",
            "command": "echo 'stress test'",
            "expected_outcome": "压力测试",
            "risk_level": "low"
        })

        # 执行多个决策周期
        start_time = asyncio.get_event_loop().time()
        decision_count = 0

        for i in range(20):
            await engine._perform_decision_cycle()
            decision_count += 1

        end_time = asyncio.get_event_loop().time()
        duration = end_time - start_time

        # 验证性能指标
        assert decision_count == 20
        assert duration < 5.0  # 应该在5秒内完成
        assert mock_container_manager.exec_command.call_count == 20

    async def test_memory_usage_under_load(self, test_config, mock_container_manager):
        """测试负载下的内存使用"""
        import psutil
        import os

        from src.decision_engine import DecisionEngine
        from unittest.mock import MagicMock

        mock_observer = MagicMock()
        mock_observer.get_recent_observations.return_value = [{"analysis": {"structured_analysis": {"summary": "test"}}}]

        engine = DecisionEngine(test_config, mock_observer, mock_container_manager)

        # 记录初始内存使用
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # 执行大量决策
        decisions = []
        for i in range(100):
            decision = {"command": f"echo 'load test {i}'", "reasoning": f"负载测试 {i}"}
            decisions.append(decision)

        # 并发执行
        tasks = [engine._execute_decision(decision) for decision in decisions]
        await asyncio.gather(*tasks)

        # 检查内存使用
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # 内存增长应该在合理范围内（小于50MB）
        assert memory_increase < 50.0
        assert mock_container_manager.exec_command.call_count == 100

    async def test_component_initialization_stress(self, test_config):
        """测试组件初始化压力测试"""
        from unittest.mock import AsyncMock

        # 并发初始化多个组件实例
        async def create_component_instance():
            from src.decision_engine import DecisionEngine
            from unittest.mock import MagicMock

            mock_observer = MagicMock()
            mock_container = MagicMock()

            engine = DecisionEngine(test_config, mock_observer, mock_container)
            return engine

        # 创建10个组件实例
        tasks = [create_component_instance() for _ in range(10)]
        components = await asyncio.gather(*tasks)

        # 验证所有组件都成功创建
        assert len(components) == 10
        assert all(isinstance(c, object) for c in components)

    async def test_container_operation_stress(self, test_config):
        """测试容器操作压力测试"""
        from unittest.mock import AsyncMock
        from src.container_manager import ContainerManager

        manager = ContainerManager(test_config)

        # Mock所有容器操作
        manager.start_container = AsyncMock(return_value={"success": True})
        manager.exec_command = AsyncMock(return_value={"success": True, "output": "test"})
        manager.stop_container = AsyncMock(return_value={"success": True})
        manager.get_container_status = AsyncMock(return_value={"success": True, "data": {"running": True}})

        # 并发执行容器操作
        operations = []
        for i in range(10):
            operations.extend([
                manager.start_container(),
                manager.exec_command(f"echo 'test {i}'"),
                manager.get_container_status(),
                manager.stop_container()
            ])

        # 并发执行所有操作
        results = await asyncio.gather(*operations, return_exceptions=True)

        # 检查没有异常
        exceptions = [r for r in results if isinstance(r, Exception)]
        assert len(exceptions) == 0

        # 验证调用次数
        assert manager.start_container.call_count == 10
        assert manager.exec_command.call_count == 10
        assert manager.get_container_status.call_count == 10
        assert manager.stop_container.call_count == 10

    async def test_observer_performance_under_load(self, test_config):
        """测试观察者负载性能"""
        from unittest.mock import AsyncMock, MagicMock
        from src.observer import Observer

        observer = Observer(test_config, MagicMock())

        # Mock观察操作
        observer._take_screenshot = AsyncMock(return_value=MagicMock())
        observer._analyze_screenshot = AsyncMock(return_value={
            "raw_analysis": "负载测试分析",
            "structured_analysis": {"summary": "正常"}
        })

        # 执行多个观察周期
        observation_count = 0
        start_time = asyncio.get_event_loop().time()

        for i in range(10):
            await observer._perform_observation()
            observation_count += 1

        end_time = asyncio.get_event_loop().time()
        duration = end_time - start_time

        # 验证性能
        assert observation_count == 10
        assert duration < 3.0  # 应该在3秒内完成
        assert len(observer.observations) == 10

    async def test_decision_engine_throughput(self, test_config, mock_container_manager):
        """测试决策引擎吞吐量"""
        from src.decision_engine import DecisionEngine
        from unittest.mock import MagicMock, AsyncMock

        mock_observer = MagicMock()
        mock_observer.get_recent_observations.return_value = [{"analysis": {"structured_analysis": {"summary": "test"}}}]

        engine = DecisionEngine(test_config, mock_observer, mock_container_manager)

        # Mock决策生成以提高速度
        engine._generate_decision = AsyncMock(return_value={
            "reasoning": "吞吐量测试",
            "command": "echo 'throughput'",
            "expected_outcome": "测试吞吐量",
            "risk_level": "low"
        })

        # 测试吞吐量
        start_time = asyncio.get_event_loop().time()

        # 执行50个决策周期
        for i in range(50):
            await engine._perform_decision_cycle()

        end_time = asyncio.get_event_loop().time()
        duration = end_time - start_time

        throughput = 50 / duration  # 决策/秒

        # 验证吞吐量（应该至少每秒5个决策）
        assert throughput > 5.0
        assert len(engine.decisions) == 50
        assert mock_container_manager.exec_command.call_count == 50


class TestFaultRecovery:
    """故障恢复测试"""

    @pytest.fixture
    def mock_container_manager(self, test_config):
        """Mock容器管理器用于故障测试"""
        from unittest.mock import AsyncMock
        from src.container_manager import ContainerManager

        manager = ContainerManager(test_config)
        return manager

    async def test_container_failure_recovery(self, test_config, mock_container_manager):
        """测试容器故障恢复"""
        from src.decision_engine import DecisionEngine
        from unittest.mock import MagicMock, patch

        mock_observer = MagicMock()
        mock_observer.get_recent_observations.return_value = [{"analysis": {"structured_analysis": {"summary": "test"}}}]

        engine = DecisionEngine(test_config, mock_observer, mock_container_manager)

        # Mock命令验证通过
        with patch.object(engine, '_validate_command', return_value=(True, "验证通过")):
            # 模拟容器执行失败
            call_count = 0
            async def mock_exec_fail_then_succeed(command):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    return {"success": False, "error": "容器连接失败", "output": ""}
                else:
                    return {"success": True, "output": "恢复成功", "error": ""}

            mock_container_manager.exec_command = mock_exec_fail_then_succeed

            # 执行决策（应该失败然后重试）
            decision = {"command": "ls", "reasoning": "测试故障恢复"}
            result1 = await engine._execute_decision(decision)

            # 验证第一次失败
            assert not result1["success"]
            assert "容器连接失败" in result1["error"]

            # 执行第二次决策（应该成功）
            result2 = await engine._execute_decision(decision)

            # 验证第二次成功
            assert result2["success"]
            assert result2["output"] == "恢复成功"

    async def test_async_operation_timeout_recovery(self, test_config, mock_container_manager):
        """测试异步操作超时恢复"""
        from src.decision_engine import DecisionEngine
        from unittest.mock import MagicMock, patch
        import asyncio

        mock_observer = MagicMock()
        mock_observer.get_recent_observations.return_value = [{"analysis": {"structured_analysis": {"summary": "test"}}}]

        engine = DecisionEngine(test_config, mock_observer, mock_container_manager)

        # Mock命令验证通过
        with patch.object(engine, '_validate_command', return_value=(True, "验证通过")):
            # Mock一个会超时的操作
            call_count = 0
            async def mock_timeout_then_success(command):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    await asyncio.sleep(2)  # 模拟超时
                    raise asyncio.TimeoutError("Operation timeout")
                else:
                    return {"success": True, "output": "恢复成功", "error": ""}

            mock_container_manager.exec_command = mock_timeout_then_success

            # 执行决策，应该处理超时
            decision = {"command": "timeout_command", "reasoning": "测试超时恢复"}
            result1 = await engine._execute_decision(decision)

            # 验证第一次超时
            assert not result1["success"]
            assert "Operation timeout" in str(result1.get("error", ""))

            # 执行第二次，应该成功
            result2 = await engine._execute_decision(decision)
            assert result2["success"]
            assert result2["output"] == "恢复成功"

    async def test_observer_failure_recovery(self, test_config):
        """测试观察者故障恢复"""
        from unittest.mock import AsyncMock, MagicMock
        from src.observer import Observer

        observer = Observer(test_config, MagicMock())

        # Mock截图失败然后成功
        call_count = 0
        async def mock_screenshot_fail_then_succeed(self):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("截图失败")
            else:
                return MagicMock()

        observer._take_screenshot = mock_screenshot_fail_then_succeed
        observer._analyze_screenshot = AsyncMock(return_value={
            "raw_analysis": "恢复后分析",
            "structured_analysis": {"summary": "正常"}
        })

        # 第一次观察应该失败
        await observer._perform_observation()
        assert len(observer.observations) == 1
        assert "error" in observer.observations[0]

        # 第二次观察应该成功
        await observer._perform_observation()
        assert len(observer.observations) == 2
        assert "analysis" in observer.observations[1]

    async def test_decision_engine_failure_recovery(self, test_config, mock_container_manager):
        """测试决策引擎故障恢复"""
        from src.decision_engine import DecisionEngine
        from unittest.mock import MagicMock, AsyncMock

        mock_observer = MagicMock()
        mock_observer.get_recent_observations.return_value = [{"analysis": {"structured_analysis": {"summary": "test"}}}]

        engine = DecisionEngine(test_config, mock_observer, mock_container_manager)

        # Mock决策生成失败然后成功
        call_count = 0
        async def mock_decision_fail_then_succeed(observations):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("决策生成失败")
            else:
                return {
                    "reasoning": "恢复后决策",
                    "command": "echo 'recovered'",
                    "expected_outcome": "故障恢复测试",
                    "risk_level": "low"
                }

        engine._generate_decision = mock_decision_fail_then_succeed

        # 第一次决策周期应该失败
        await engine._perform_decision_cycle()
        assert len(engine.decisions) == 1
        assert "error" in engine.decisions[0]

        # 第二次决策周期应该成功
        await engine._perform_decision_cycle()
        assert len(engine.decisions) == 2
        assert "decision" in engine.decisions[1]
        assert engine.decisions[1]["decision"]["command"] == "echo 'recovered'"

    async def test_network_failure_recovery(self, test_config, mock_container_manager):
        """测试网络故障恢复"""
        from src.decision_engine import DecisionEngine
        from unittest.mock import MagicMock, patch

        mock_observer = MagicMock()
        mock_observer.get_recent_observations.return_value = [{"analysis": {"structured_analysis": {"summary": "test"}}}]

        engine = DecisionEngine(test_config, mock_observer, mock_container_manager)

        # Mock命令验证通过
        with patch.object(engine, '_validate_command', return_value=(True, "验证通过")):
            # 模拟网络连接失败然后恢复
            call_count = 0
            async def mock_network_fail_then_recover(command):
                nonlocal call_count
                call_count += 1
                if call_count <= 2:  # 前两次失败
                    raise ConnectionError("网络连接失败")
                else:
                    return {"success": True, "output": "网络恢复", "error": ""}

            mock_container_manager.exec_command = mock_network_fail_then_recover

            # 执行多次决策，测试重试机制
            for i in range(4):
                decision = {"command": f"test {i}", "reasoning": f"网络测试 {i}"}
                result = await engine._execute_decision(decision)
                if i < 2:
                    assert not result["success"]
                    assert "网络连接失败" in str(result.get("error", ""))
                else:
                    assert result["success"]
                    assert result["output"] == "网络恢复"



    async def test_resource_exhaustion_recovery(self, test_config, mock_container_manager):
        """测试资源耗尽恢复"""
        from src.decision_engine import DecisionEngine
        from unittest.mock import MagicMock, patch

        mock_observer = MagicMock()
        mock_observer.get_recent_observations.return_value = [{"analysis": {"structured_analysis": {"summary": "test"}}}]

        engine = DecisionEngine(test_config, mock_observer, mock_container_manager)

        # Mock命令验证通过
        with patch.object(engine, '_validate_command', return_value=(True, "验证通过")):
            # 模拟内存耗尽错误
            call_count = 0
            async def mock_memory_exhaustion(command):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise MemoryError("内存不足")
                else:
                    return {"success": True, "output": "内存恢复", "error": ""}

            mock_container_manager.exec_command = mock_memory_exhaustion

            # 执行决策
            decision = {"command": "heavy_command", "reasoning": "资源测试"}

            # 第一次应该失败（内存不足）
            result1 = await engine._execute_decision(decision)
            assert not result1["success"]
            assert "内存不足" in str(result1.get("error", ""))

            # 第二次应该成功（内存恢复）
            result2 = await engine._execute_decision(decision)
            assert result2["success"]
            assert result2["output"] == "内存恢复"