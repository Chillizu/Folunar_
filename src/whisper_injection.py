#!/usr/bin/env python3
"""
随机想法注入系统
负责管理词汇库、定时注入和日志记录
"""

import json
import os
import random
import logging
import time
import threading
from typing import List, Dict, Optional
from datetime import datetime
import schedule

class WhisperInjectionManager:
    """随机想法注入管理器"""

    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # 配置参数
        self.enabled = config.get('enabled', True)
        self.vocabulary_file = config.get('vocabulary_file', 'data/vocabulary.json')
        self.injection_interval_minutes = config.get('injection_interval_minutes', 30)
        self.log_file = config.get('log_file', 'logs/whisper_injection.log')
        self.max_log_entries = config.get('max_log_entries', 1000)
        self.default_vocabulary = config.get('default_vocabulary', [])

        # 运行状态
        self.is_running = False
        self.scheduler_thread = None
        self.vocabulary = []
        self.injection_logs = []

        # 确保目录存在
        self._ensure_directories()

        # 初始化词汇库
        self._load_vocabulary()

        # 初始化日志
        self._load_logs()

    def _ensure_directories(self):
        """确保必要的目录存在"""
        os.makedirs(os.path.dirname(self.vocabulary_file), exist_ok=True)
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        os.makedirs('/tmp', exist_ok=True)  # 确保/tmp目录存在

    def _load_vocabulary(self):
        """加载词汇库"""
        try:
            if os.path.exists(self.vocabulary_file):
                with open(self.vocabulary_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.vocabulary = data.get('vocabulary', [])
                self.logger.info(f"已加载 {len(self.vocabulary)} 个词汇")
            else:
                # 使用默认词汇库
                self.vocabulary = self.default_vocabulary.copy()
                self._save_vocabulary()
                self.logger.info(f"使用默认词汇库，已创建 {len(self.vocabulary)} 个词汇")
        except Exception as e:
            self.logger.error(f"加载词汇库失败: {str(e)}")
            self.vocabulary = self.default_vocabulary.copy()

    def _save_vocabulary(self):
        """保存词汇库"""
        try:
            data = {
                'vocabulary': self.vocabulary,
                'last_updated': datetime.now().isoformat(),
                'count': len(self.vocabulary)
            }
            with open(self.vocabulary_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            self.logger.info(f"词汇库已保存，共 {len(self.vocabulary)} 个词汇")
        except Exception as e:
            self.logger.error(f"保存词汇库失败: {str(e)}")

    def _load_logs(self):
        """加载注入日志"""
        try:
            if os.path.exists(self.log_file):
                with open(self.log_file, 'r', encoding='utf-8') as f:
                    self.injection_logs = json.load(f)
                self.logger.info(f"已加载 {len(self.injection_logs)} 条注入日志")
            else:
                self.injection_logs = []
        except Exception as e:
            self.logger.error(f"加载注入日志失败: {str(e)}")
            self.injection_logs = []

    def _save_logs(self):
        """保存注入日志"""
        try:
            # 限制日志条目数量
            if len(self.injection_logs) > self.max_log_entries:
                self.injection_logs = self.injection_logs[-self.max_log_entries:]

            with open(self.log_file, 'w', encoding='utf-8') as f:
                json.dump(self.injection_logs, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.error(f"保存注入日志失败: {str(e)}")

    def add_vocabulary(self, word: str) -> bool:
        """添加词汇"""
        if not word or not word.strip():
            return False

        word = word.strip()
        if word not in self.vocabulary:
            self.vocabulary.append(word)
            self._save_vocabulary()
            self.logger.info(f"已添加词汇: {word}")
            return True
        return False

    def remove_vocabulary(self, word: str) -> bool:
        """删除词汇"""
        if word in self.vocabulary:
            self.vocabulary.remove(word)
            self._save_vocabulary()
            self.logger.info(f"已删除词汇: {word}")
            return True
        return False

    def get_vocabulary(self) -> List[str]:
        """获取所有词汇"""
        return self.vocabulary.copy()

    def clear_vocabulary(self):
        """清空词汇库"""
        self.vocabulary = []
        self._save_vocabulary()
        self.logger.info("词汇库已清空")

    def inject_random_word(self) -> Optional[str]:
        """注入随机词汇到容器文件"""
        if not self.vocabulary:
            self.logger.warning("词汇库为空，无法进行注入")
            return None

        # 选择随机词汇
        word = random.choice(self.vocabulary)

        try:
            # 写入到容器文件
            with open('/tmp/whisper.txt', 'w', encoding='utf-8') as f:
                f.write(word)

            # 记录日志
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'word': word,
                'success': True
            }
            self.injection_logs.append(log_entry)
            self._save_logs()

            self.logger.info(f"成功注入词汇: {word}")
            return word

        except Exception as e:
            # 记录失败日志
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'word': word,
                'success': False,
                'error': str(e)
            }
            self.injection_logs.append(log_entry)
            self._save_logs()

            self.logger.error(f"注入词汇失败: {word}, 错误: {str(e)}")
            return None

    def get_injection_logs(self, limit: int = 50) -> List[Dict]:
        """获取注入日志"""
        return self.injection_logs[-limit:] if limit > 0 else self.injection_logs

    def clear_logs(self):
        """清空注入日志"""
        self.injection_logs = []
        self._save_logs()
        self.logger.info("注入日志已清空")

    def start_injection(self):
        """启动定时注入"""
        if not self.enabled:
            self.logger.info("随机想法注入系统已禁用")
            return

        if self.is_running:
            self.logger.warning("注入系统已在运行中")
            return

        self.is_running = True

        # 设置定时任务
        schedule.every(self.injection_interval_minutes).minutes.do(self.inject_random_word)

        # 启动调度器线程
        self.scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
        self.scheduler_thread.start()

        self.logger.info(f"随机想法注入系统已启动，间隔: {self.injection_interval_minutes} 分钟")

    def stop_injection(self):
        """停止定时注入"""
        if not self.is_running:
            return

        self.is_running = False
        schedule.clear()

        if self.scheduler_thread and self.scheduler_thread.is_alive():
            self.scheduler_thread.join(timeout=5)

        self.logger.info("随机想法注入系统已停止")

    def _run_scheduler(self):
        """运行调度器"""
        while self.is_running:
            schedule.run_pending()
            time.sleep(1)

    def get_status(self) -> Dict:
        """获取系统状态"""
        return {
            'enabled': self.enabled,
            'is_running': self.is_running,
            'vocabulary_count': len(self.vocabulary),
            'injection_interval_minutes': self.injection_interval_minutes,
            'log_count': len(self.injection_logs),
            'last_injection': self.injection_logs[-1] if self.injection_logs else None
        }