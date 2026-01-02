"""
观察者模块
负责监控AI沙盒容器，定期截取桌面截图，调用多模态AI模型分析截图内容，
记录AI的观察、思考和决策过程。
"""

import asyncio
import base64
import json
import logging
import time
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Any

try:
    import pyautogui
except Exception as exc:
    pyautogui = None  # pyautogui requires DISPLAY; headless environments shouldn't break the app
    HEADLESS_SCREENSHOT_ERROR = exc
else:
    HEADLESS_SCREENSHOT_ERROR = None
from PIL import Image
import openai
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

class Observer:
    """AI观察者类，负责监控和分析AI沙盒环境"""

    def __init__(self, config: Dict[str, Any], agent_manager):
        self.config = config
        self.agent_manager = agent_manager
        self.is_running = False
        self.observation_interval = config.get('observer', {}).get('interval', 30)  # 默认30秒
        self.screenshot_dir = Path(config.get('observer', {}).get('screenshot_dir', 'screenshots'))
        self.observation_log_file = Path(config.get('observer', {}).get('log_file', 'logs/observer.log'))

        # 确保目录存在
        self.screenshot_dir.mkdir(exist_ok=True)
        self.observation_log_file.parent.mkdir(exist_ok=True)
        if pyautogui is None:
            logger.warning(f"pyautogui unavailable; desktop screenshots are disabled: {HEADLESS_SCREENSHOT_ERROR}")

        # 初始化OpenAI客户端用于多模态分析
        self.vision_client = AsyncOpenAI(
            api_key=config.get('api', {}).get('key'),
            base_url=config.get('api', {}).get('base_url', 'https://api.openai.com/v1')
        )
        self.vision_model = config.get('observer', {}).get('vision_model', 'gpt-4-vision-preview')

        # 观察历史记录
        self.observations: List[Dict[str, Any]] = []

    async def start_monitoring(self) -> bool:
        """启动观察者监控"""
        if self.is_running:
            logger.warning("观察者已经在运行中")
            return False

        self.is_running = True
        logger.info(f"启动AI观察者，监控间隔: {self.observation_interval}秒")

        # 启动监控任务
        asyncio.create_task(self._monitoring_loop())
        return True

    async def stop_monitoring(self) -> bool:
        """停止观察者监控"""
        if not self.is_running:
            logger.warning("观察者未在运行")
            return False

        self.is_running = False
        logger.info("停止AI观察者")
        return True

    def get_status(self) -> Dict[str, Any]:
        """获取观察者状态"""
        return {
            "is_running": self.is_running,
            "observation_interval": self.observation_interval,
            "total_observations": len(self.observations),
            "last_observation": self.observations[-1] if self.observations else None
        }

    async def _monitoring_loop(self):
        """监控循环"""
        while self.is_running:
            try:
                await self._perform_observation()
                await asyncio.sleep(self.observation_interval)
            except Exception as e:
                logger.error(f"观察循环出错: {str(e)}")
                await asyncio.sleep(5)  # 出错后等待5秒再试

    async def _perform_observation(self):
        """执行一次观察"""
        timestamp = datetime.now()

        try:
            # 1. 截取桌面截图
            screenshot_path = await self._take_screenshot(timestamp)

            # 2. 分析截图内容
            analysis = await self._analyze_screenshot(screenshot_path)

            # 3. 记录观察结果
            observation = {
                "timestamp": timestamp.isoformat(),
                "screenshot_path": str(screenshot_path),
                "analysis": analysis,
                "metadata": {
                    "interval": self.observation_interval,
                    "model": self.vision_model
                }
            }

            self.observations.append(observation)

            # 4. 写入日志文件
            await self._log_observation(observation)

            logger.info(f"完成观察: {timestamp.isoformat()}")

        except Exception as e:
            logger.error(f"观察执行失败: {str(e)}")
            # 记录失败的观察
            failed_observation = {
                "timestamp": timestamp.isoformat(),
                "error": str(e),
                "metadata": {
                    "interval": self.observation_interval,
                    "model": self.vision_model
                }
            }
            self.observations.append(failed_observation)
            await self._log_observation(failed_observation)

    async def _take_screenshot(self, timestamp: datetime) -> Path:
        """截取桌面截图"""
        try:
            if pyautogui is None:
                raise RuntimeError(f"pyautogui unavailable for screenshots: {HEADLESS_SCREENSHOT_ERROR}")
            # 使用pyautogui截图
            screenshot = pyautogui.screenshot()

            # 生成文件名
            filename = f"screenshot_{timestamp.strftime('%Y%m%d_%H%M%S')}.png"
            filepath = self.screenshot_dir / filename

            # 保存截图
            screenshot.save(filepath)
            logger.debug(f"截图已保存: {filepath}")

            return filepath

        except Exception as e:
            logger.error(f"截图失败: {str(e)}")
            raise

    async def _analyze_screenshot(self, screenshot_path: Path) -> Dict[str, Any]:
        """使用多模态AI分析截图内容"""
        try:
            # 读取图片并转换为base64
            with open(screenshot_path, 'rb') as image_file:
                image_data = base64.b64encode(image_file.read()).decode('utf-8')

            # 构建多模态消息
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": """请分析这张桌面截图，描述你看到的AI沙盒环境状态。特别关注：
1. 当前运行的应用程序或界面
2. AI代理的活动迹象（如聊天窗口、代码编辑器、终端等）
3. 任何错误信息或异常状态
4. 用户交互的证据
5. 系统状态指示器

请以结构化的方式回答，包括观察、分析和可能的AI决策过程。"""
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_data}"
                            }
                        }
                    ]
                }
            ]

            # 调用Vision API
            response = await self.vision_client.chat.completions.create(
                model=self.vision_model,
                messages=messages,
                max_tokens=1000,
                temperature=0.3
            )

            analysis_text = response.choices[0].message.content

            # 解析分析结果（可以后续扩展为更结构化的解析）
            analysis = {
                "raw_analysis": analysis_text,
                "structured_analysis": self._parse_analysis(analysis_text),
                "model": self.vision_model,
                "tokens_used": response.usage.total_tokens if response.usage else None
            }

            return analysis

        except Exception as e:
            logger.error(f"截图分析失败: {str(e)}")
            return {
                "error": str(e),
                "model": self.vision_model
            }

    def _parse_analysis(self, analysis_text: str) -> Dict[str, Any]:
        """解析分析文本为结构化数据"""
        # 这里可以实现更复杂的解析逻辑
        # 目前返回简单的结构
        return {
            "summary": analysis_text[:200] + "..." if len(analysis_text) > 200 else analysis_text,
            "full_text": analysis_text
        }

    async def _log_observation(self, observation: Dict[str, Any]):
        """将观察结果写入日志文件"""
        try:
            with open(self.observation_log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(observation, ensure_ascii=False) + '\n')
        except Exception as e:
            logger.error(f"写入观察日志失败: {str(e)}")

    def get_recent_observations(self, limit: int = 10) -> List[Dict[str, Any]]:
        """获取最近的观察记录"""
        return self.observations[-limit:] if self.observations else []

    def clear_observation_history(self):
        """清空观察历史记录"""
        self.observations.clear()
        logger.info("观察历史记录已清空")
