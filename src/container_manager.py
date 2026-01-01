#!/usr/bin/env python3
"""
容器管理模块
用于管理Debian容器的创建、启动、停止等操作
"""

import subprocess
import os
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class ContainerManager:
    def __init__(self, image_name: str = "debian-container", container_name: str = "agent-debian"):
        self.image_name = image_name
        self.container_name = container_name
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def build_image(self) -> Dict[str, Any]:
        """构建Docker镜像"""
        try:
            logger.info(f"开始构建镜像: {self.image_name}")
            cmd = ["docker", "build", "-t", self.image_name, "."]
            result = subprocess.run(cmd, cwd=self.project_root, capture_output=True, text=True, encoding='utf-8')
            if result.returncode == 0:
                logger.info("镜像构建成功")
                return {"success": True, "error": ""}
            else:
                error_msg = f"镜像构建失败: {result.stderr.strip()}"
                logger.error(error_msg)
                return {"success": False, "error": error_msg}
        except Exception as e:
            error_msg = f"构建镜像时发生错误: {str(e)}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}

    def start_container(self, ports: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """启动容器"""
        try:
            logger.info(f"开始启动容器: {self.container_name}")

            # 检查容器是否已存在
            if self.is_container_running():
                logger.info("容器已在运行")
                return {"success": True, "error": ""}

            cmd = ["docker", "run", "-d", "--name", self.container_name]

            # 添加端口映射
            if ports:
                for host_port, container_port in ports.items():
                    cmd.extend(["-p", f"{host_port}:{container_port}"])

            cmd.append(self.image_name)

            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
            if result.returncode == 0:
                logger.info("容器启动成功")
                return {"success": True, "error": ""}
            else:
                error_msg = f"容器启动失败: {result.stderr.strip()}"
                logger.error(error_msg)
                return {"success": False, "error": error_msg}
        except Exception as e:
            error_msg = f"启动容器时发生错误: {str(e)}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}

    def stop_container(self) -> Dict[str, Any]:
        """停止容器"""
        try:
            logger.info(f"开始停止容器: {self.container_name}")
            cmd = ["docker", "stop", self.container_name]
            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
            if result.returncode == 0:
                logger.info("容器停止成功")
                return {"success": True, "error": ""}
            else:
                error_msg = f"容器停止失败或容器不存在: {result.stderr.strip()}"
                logger.warning(error_msg)
                return {"success": False, "error": error_msg}
        except Exception as e:
            error_msg = f"停止容器时发生错误: {str(e)}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}

    def remove_container(self) -> Dict[str, Any]:
        """删除容器"""
        try:
            logger.info(f"开始删除容器: {self.container_name}")
            cmd = ["docker", "rm", self.container_name]
            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
            if result.returncode == 0:
                logger.info("容器删除成功")
                return {"success": True, "error": ""}
            else:
                error_msg = f"容器删除失败或容器不存在: {result.stderr.strip()}"
                logger.warning(error_msg)
                return {"success": False, "error": error_msg}
        except Exception as e:
            error_msg = f"删除容器时发生错误: {str(e)}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}

    def is_container_running(self) -> bool:
        """检查容器是否正在运行"""
        try:
            cmd = ["docker", "ps", "--filter", f"name={self.container_name}", "--format", "{{.Names}}"]
            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
            return self.container_name in result.stdout
        except Exception as e:
            logger.error(f"检查容器状态时发生错误: {e}")
            return False

    def get_container_status(self) -> Dict[str, Any]:
        """获取容器状态信息"""
        try:
            cmd = ["docker", "ps", "-a", "--filter", f"name={self.container_name}", "--format", "json"]
            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
            if result.returncode == 0 and result.stdout.strip():
                # 解析JSON输出
                import json
                containers = [json.loads(line) for line in result.stdout.strip().split('\n') if line]
                if containers:
                    container = containers[0]
                    status_data = {
                        "name": container.get("Names", ""),
                        "status": container.get("Status", ""),
                        "ports": container.get("Ports", ""),
                        "running": "Up" in container.get("Status", "")
                    }

                    # 如果容器正在运行，获取统计信息
                    if status_data["running"]:
                        stats = self.get_container_stats()
                        if stats["success"]:
                            status_data.update(stats["data"])

                    return {
                        "success": True,
                        "data": status_data,
                        "error": ""
                    }
            return {
                "success": True,
                "data": {"name": self.container_name, "status": "Not found", "running": False},
                "error": ""
            }
        except Exception as e:
            error_msg = f"获取容器状态时发生错误: {str(e)}"
            logger.error(error_msg)
            return {
                "success": False,
                "data": {"name": self.container_name, "status": "Error", "running": False},
                "error": error_msg
            }

    def get_container_stats(self) -> Dict[str, Any]:
        """获取容器统计信息（CPU、内存、网络等）"""
        try:
            cmd = ["docker", "stats", "--no-stream", "--format", "json", self.container_name]
            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', timeout=10)
            if result.returncode == 0 and result.stdout.strip():
                import json
                stats = json.loads(result.stdout.strip())
                return {
                    "success": True,
                    "data": {
                        "cpu_percent": stats.get("CPUPerc", "0%"),
                        "memory_usage": stats.get("MemUsage", ""),
                        "memory_percent": stats.get("MemPerc", "0%"),
                        "network_io": stats.get("NetIO", ""),
                        "block_io": stats.get("BlockIO", ""),
                        "pids": stats.get("PIDs", "0")
                    },
                    "error": ""
                }
            else:
                return {
                    "success": False,
                    "data": {},
                    "error": f"获取统计信息失败: {result.stderr.strip()}"
                }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "data": {},
                "error": "获取统计信息超时"
            }
        except Exception as e:
            error_msg = f"获取容器统计信息时发生错误: {str(e)}"
            logger.error(error_msg)
            return {
                "success": False,
                "data": {},
                "error": error_msg
            }

    def exec_command(self, command: str) -> Dict[str, Any]:
        """在容器中执行命令"""
        try:
            cmd = ["docker", "exec", self.container_name] + command.split()
            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
            if result.returncode == 0:
                return {"success": True, "output": result.stdout.strip(), "error": ""}
            else:
                error_msg = f"命令执行失败: {result.stderr.strip()}"
                logger.error(error_msg)
                return {"success": False, "output": "", "error": error_msg}
        except Exception as e:
            error_msg = f"执行命令时发生异常: {str(e)}"
            logger.error(error_msg)
            return {"success": False, "output": "", "error": error_msg}

# 命令行接口
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Debian容器管理工具")
    parser.add_argument("action", choices=["build", "start", "stop", "remove", "status", "exec"],
                       help="要执行的操作")
    parser.add_argument("--command", help="要执行的命令（用于exec操作）")
    parser.add_argument("--ports", help="端口映射，格式: host_port:container_port")

    args = parser.parse_args()

    manager = ContainerManager()

    if args.action == "build":
        result = manager.build_image()
        if result["success"]:
            print("构建成功")
        else:
            print(f"构建失败: {result['error']}")
    elif args.action == "start":
        ports = None
        if args.ports:
            # 解析端口映射
            ports = {}
            for mapping in args.ports.split(","):
                if ":" in mapping:
                    host, container = mapping.split(":", 1)
                    ports[host] = container
        result = manager.start_container(ports)
        if result["success"]:
            print("启动成功")
        else:
            print(f"启动失败: {result['error']}")
    elif args.action == "stop":
        result = manager.stop_container()
        if result["success"]:
            print("停止成功")
        else:
            print(f"停止失败: {result['error']}")
    elif args.action == "remove":
        result = manager.remove_container()
        if result["success"]:
            print("删除成功")
        else:
            print(f"删除失败: {result['error']}")
    elif args.action == "status":
        result = manager.get_container_status()
        if result["success"]:
            print(f"容器状态: {result['data']}")
        else:
            print(f"获取状态失败: {result['error']}")
    elif args.action == "exec":
        if args.command:
            result = manager.exec_command(args.command)
            if result["success"]:
                print(result["output"])
            else:
                print(f"执行失败: {result['error']}")
        else:
            print("请提供要执行的命令")