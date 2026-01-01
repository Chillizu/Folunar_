"""
决策引擎模块
基于观察者分析结果生成下一步操作命令，在容器中安全执行命令，
实现观察-决策-执行的闭环。包含命令验证、安全过滤和执行日志记录。
"""

import asyncio
import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import openai
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

class DecisionEngine:
    """AI决策引擎类，负责基于观察结果生成和执行命令"""

    def __init__(self, config: Dict[str, Any], observer, container_manager):
        self.config = config
        self.observer = observer
        self.container_manager = container_manager
        self.is_running = False
        self.decision_interval = config.get('decision_engine', {}).get('interval', 60)  # 默认60秒
        self.decision_log_file = Path(config.get('decision_engine', {}).get('log_file', 'logs/decision_engine.log'))

        # 确保日志目录存在
        self.decision_log_file.parent.mkdir(exist_ok=True)

        # 初始化OpenAI客户端用于决策生成
        self.decision_client = AsyncOpenAI(
            api_key=config.get('api', {}).get('key'),
            base_url=config.get('api', {}).get('base_url', 'https://api.openai.com/v1')
        )
        self.decision_model = config.get('decision_engine', {}).get('model', 'gpt-4')

        # 决策历史记录
        self.decisions: List[Dict[str, Any]] = []

        # 命令过滤规则
        self.dangerous_commands = config.get('decision_engine', {}).get('dangerous_commands', [
            r'rm\s+-rf\s+/',  # 删除根目录
            r'rm\s+-rf\s+\*',  # 删除所有文件
            r'rm\s+-rf\s+\.\.',  # 删除上级目录
            r'format\s+',  # 格式化磁盘
            r'mkfs\.',  # 创建文件系统
            r'dd\s+if=',  # 磁盘复制
            r'fdisk',  # 分区工具
            r'parted',  # 分区工具
            r'shutdown',  # 关机
            r'reboot',  # 重启
            r'halt',  # 停止系统
            r'poweroff',  # 关闭电源
            r'systemctl\s+disable',  # 禁用服务
            r'systemctl\s+stop',  # 停止服务
            r'killall',  # 杀死所有进程
            r'pkill\s+-9',  # 强制杀死进程
            r'chmod\s+777',  # 危险权限设置
            r'chown\s+root',  # 改变所有者为root
            r'usermod',  # 修改用户
            r'userdel',  # 删除用户
            r'passwd',  # 修改密码
            r'su\s+',  # 切换用户
            r'sudo\s+',  # 超级用户权限
            r'mount',  # 挂载文件系统
            r'umount',  # 卸载文件系统
            r'apt\s+remove',  # 移除包
            r'apt\s+purge',  # 清除包
            r'pip\s+uninstall',  # 卸载Python包
            r'npm\s+uninstall',  # 卸载npm包
            r'yum\s+remove',  # 移除yum包
            r'wget\s+.*\|\s*sh',  # 下载并执行脚本
            r'curl\s+.*\|\s*sh',  # 下载并执行脚本
            r'eval\s+',  # 执行字符串
            r'exec\s+',  # 执行命令
            r'bash\s+-c',  # bash执行
            r'sh\s+-c',  # sh执行
        ])

        # 允许的命令模式
        self.allowed_command_patterns = config.get('decision_engine', {}).get('allowed_commands', [
            r'^ls\s+',  # 列出文件
            r'^pwd$',  # 显示当前目录
            r'^cd\s+',  # 改变目录
            r'^cat\s+',  # 查看文件内容
            r'^head\s+',  # 查看文件开头
            r'^tail\s+',  # 查看文件结尾
            r'^grep\s+',  # 搜索文本
            r'^find\s+',  # 查找文件
            r'^ps\s+',  # 查看进程
            r'^top$',  # 系统监控
            r'^htop$',  # 交互式系统监控
            r'^df\s+',  # 磁盘使用情况
            r'^du\s+',  # 目录大小
            r'^free$',  # 内存使用情况
            r'^uptime$',  # 系统运行时间
            r'^whoami$',  # 当前用户
            r'^id$',  # 用户信息
            r'^date$',  # 显示日期
            r'^echo\s+',  # 输出文本
            r'^mkdir\s+',  # 创建目录
            r'^touch\s+',  # 创建文件
            r'^cp\s+',  # 复制文件
            r'^mv\s+',  # 移动文件
            r'^python3?\s+',  # 运行Python脚本
            r'^node\s+',  # 运行Node.js脚本
            r'^npm\s+run',  # npm脚本
            r'^git\s+',  # Git命令
            r'^vim?\s+',  # 编辑器
            r'^nano\s+',  # 编辑器
            r'^less\s+',  # 分页查看
            r'^more\s+',  # 分页查看
            r'^wc\s+',  # 字数统计
            r'^sort\s+',  # 排序
            r'^uniq\s+',  # 去重
            r'^cut\s+',  # 剪切字段
            r'^awk\s+',  # 文本处理
            r'^sed\s+',  # 流编辑器
        ])

    async def start_decision_loop(self) -> bool:
        """启动决策循环"""
        if self.is_running:
            logger.warning("决策引擎已经在运行中")
            return False

        self.is_running = True
        logger.info(f"启动AI决策引擎，决策间隔: {self.decision_interval}秒")

        # 启动决策任务
        asyncio.create_task(self._decision_loop())
        return True

    async def stop_decision_loop(self) -> bool:
        """停止决策循环"""
        if not self.is_running:
            logger.warning("决策引擎未在运行")
            return False

        self.is_running = False
        logger.info("停止AI决策引擎")
        return True

    def get_status(self) -> Dict[str, Any]:
        """获取决策引擎状态"""
        return {
            "is_running": self.is_running,
            "decision_interval": self.decision_interval,
            "total_decisions": len(self.decisions),
            "last_decision": self.decisions[-1] if self.decisions else None
        }

    async def _decision_loop(self):
        """决策循环"""
        while self.is_running:
            try:
                await self._perform_decision_cycle()
                await asyncio.sleep(self.decision_interval)
            except Exception as e:
                logger.error(f"决策循环出错: {str(e)}")
                await asyncio.sleep(5)  # 出错后等待5秒再试

    async def _perform_decision_cycle(self):
        """执行一次完整的决策周期：观察-决策-执行"""
        timestamp = datetime.now()

        try:
            # 1. 获取最新的观察数据
            recent_observations = self.observer.get_recent_observations(limit=5)
            if not recent_observations:
                logger.info("没有可用的观察数据，跳过本次决策周期")
                return

            # 2. 基于观察数据生成决策
            decision = await self._generate_decision(recent_observations)

            # 3. 验证和执行命令
            if decision.get('command'):
                execution_result = await self._execute_decision(decision)
                decision['execution_result'] = execution_result
            else:
                decision['execution_result'] = {"success": True, "message": "无需执行命令"}

            # 4. 记录决策结果
            decision_record = {
                "timestamp": timestamp.isoformat(),
                "observations": recent_observations,
                "decision": decision,
                "metadata": {
                    "interval": self.decision_interval,
                    "model": self.decision_model
                }
            }

            self.decisions.append(decision_record)

            # 5. 写入日志文件
            await self._log_decision(decision_record)

            logger.info(f"完成决策周期: {timestamp.isoformat()}")

        except Exception as e:
            logger.error(f"决策周期执行失败: {str(e)}")
            # 记录失败的决策
            failed_decision = {
                "timestamp": timestamp.isoformat(),
                "error": str(e),
                "metadata": {
                    "interval": self.decision_interval,
                    "model": self.decision_model
                }
            }
            self.decisions.append(failed_decision)
            await self._log_decision(failed_decision)

    async def _generate_decision(self, observations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """基于观察数据生成决策"""
        try:
            # 构建决策提示
            observation_summaries = []
            for obs in observations[-3:]:  # 只使用最近3个观察
                if 'analysis' in obs and 'structured_analysis' in obs['analysis']:
                    summary = obs['analysis']['structured_analysis'].get('summary', '')
                    observation_summaries.append(f"- {obs['timestamp'][:19]}: {summary}")
                else:
                    observation_summaries.append(f"- {obs['timestamp'][:19]}: 观察数据不完整")

            observations_text = "\n".join(observation_summaries)

            prompt = f"""你是一个AI决策引擎，负责分析AI沙盒环境的状态并决定下一步行动。

最近的观察数据：
{observations_text}

基于这些观察，请决定AI沙盒应该执行什么命令来继续探索和学习。

决策规则：
1. 命令必须是安全的，不能破坏系统
2. 命令应该有助于AI了解和探索环境
3. 优先考虑信息收集和学习相关的命令
4. 避免任何可能造成数据丢失的操作

请以JSON格式返回决策：
{{
    "reasoning": "决策理由的详细说明",
    "command": "要执行的命令（如果不需要执行命令则为空字符串）",
    "expected_outcome": "预期结果描述",
    "risk_level": "low/medium/high"
}}

如果当前不需要执行任何命令，请将command设为空字符串。"""

            # 调用决策模型
            response = await self.decision_client.chat.completions.create(
                model=self.decision_model,
                messages=[
                    {
                        "role": "system",
                        "content": "你是一个谨慎的AI决策引擎，负责为AI沙盒环境生成安全的探索命令。你的决策必须确保系统安全和数据完整性。"
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=500,
                temperature=0.3
            )

            decision_text = response.choices[0].message.content

            # 解析决策结果
            try:
                decision = json.loads(decision_text)
                # 验证决策格式
                if not isinstance(decision, dict):
                    raise ValueError("决策结果不是字典格式")
                if 'command' not in decision:
                    decision['command'] = ""
                if 'reasoning' not in decision:
                    decision['reasoning'] = "决策生成失败"
                if 'risk_level' not in decision:
                    decision['risk_level'] = "unknown"

                return decision

            except json.JSONDecodeError as e:
                logger.error(f"解析决策JSON失败: {str(e)}, 原始内容: {decision_text}")
                return {
                    "reasoning": f"JSON解析失败: {str(e)}",
                    "command": "",
                    "expected_outcome": "无法生成有效决策",
                    "risk_level": "unknown",
                    "raw_response": decision_text
                }

        except Exception as e:
            logger.error(f"生成决策失败: {str(e)}")
            return {
                "reasoning": f"决策生成异常: {str(e)}",
                "command": "",
                "expected_outcome": "决策过程出错",
                "risk_level": "unknown"
            }

    def _validate_command(self, command: str) -> Tuple[bool, str]:
        """验证命令的安全性"""
        if not command or not command.strip():
            return True, "空命令，跳过验证"

        command = command.strip()

        # 检查危险命令
        for dangerous_pattern in self.dangerous_commands:
            if re.search(dangerous_pattern, command, re.IGNORECASE):
                return False, f"命令包含危险操作: {dangerous_pattern}"

        # 检查是否匹配允许的命令模式
        allowed = False
        for allowed_pattern in self.allowed_command_patterns:
            if re.match(allowed_pattern, command, re.IGNORECASE):
                allowed = True
                break

        if not allowed:
            return False, "命令不匹配允许的模式"

        # 额外的安全检查
        # 防止路径遍历
        if '..' in command:
            return False, "命令包含路径遍历风险"

        # 防止注入特殊字符
        dangerous_chars = [';', '&', '|', '`', '$', '(', ')']
        for char in dangerous_chars:
            if char in command and not (char == '&' and '&&' in command):  # 允许 && 连接符
                return False, f"命令包含危险字符: {char}"

        return True, "命令验证通过"

    async def _execute_decision(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """执行决策中的命令"""
        command = decision.get('command', '').strip()
        if not command:
            return {"success": True, "message": "无需执行命令"}

        # 验证命令安全性
        is_valid, validation_message = self._validate_command(command)
        if not is_valid:
            logger.warning(f"命令验证失败: {validation_message}, 命令: {command}")
            return {
                "success": False,
                "error": f"命令验证失败: {validation_message}",
                "command": command
            }

        # 执行命令
        try:
            logger.info(f"执行决策命令: {command}")
            result = await self.container_manager.exec_command(command)

            if result["success"]:
                logger.info(f"命令执行成功: {command}")
                return {
                    "success": True,
                    "command": command,
                    "output": result["output"],
                    "execution_time": time.time()
                }
            else:
                logger.error(f"命令执行失败: {command}, 错误: {result['error']}")
                return {
                    "success": False,
                    "command": command,
                    "error": result["error"],
                    "execution_time": time.time()
                }

        except Exception as e:
            logger.error(f"执行命令时发生异常: {str(e)}, 命令: {command}")
            return {
                "success": False,
                "command": command,
                "error": f"执行异常: {str(e)}",
                "execution_time": time.time()
            }

    async def _log_decision(self, decision_record: Dict[str, Any]):
        """将决策结果写入日志文件"""
        try:
            with open(self.decision_log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(decision_record, ensure_ascii=False) + '\n')
        except Exception as e:
            logger.error(f"写入决策日志失败: {str(e)}")

    def get_recent_decisions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """获取最近的决策记录"""
        return self.decisions[-limit:] if self.decisions else []

    def clear_decision_history(self):
        """清空决策历史记录"""
        self.decisions.clear()
        logger.info("决策历史记录已清空")

    async def manual_decision(self, observations: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """手动触发一次决策周期"""
        if observations is None:
            observations = self.observer.get_recent_observations(limit=5)

        if not observations:
            return {"success": False, "error": "没有可用的观察数据"}

        try:
            decision = await self._generate_decision(observations)
            if decision.get('command'):
                execution_result = await self._execute_decision(decision)
                decision['execution_result'] = execution_result

            return {
                "success": True,
                "decision": decision,
                "observations_used": len(observations)
            }
        except Exception as e:
            logger.error(f"手动决策失败: {str(e)}")
            return {"success": False, "error": str(e)}