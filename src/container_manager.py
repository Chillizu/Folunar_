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

    def build_image(self) -> bool:
        """构建Docker镜像"""
        try:
            logger.info(f"开始构建镜像: {self.image_name}")
            cmd = ["docker", "build", "-t", self.image_name, "."]
            result = subprocess.run(cmd, cwd=self.project_root, capture_output=True, text=True)
            if result.returncode == 0:
                logger.info("镜像构建成功")
                return True
            else:
                logger.error(f"镜像构建失败: {result.stderr}")
                return False
        except Exception as e:
            logger.error(f"构建镜像时发生错误: {e}")
            return False

    def start_container(self, ports: Optional[Dict[str, str]] = None) -> bool:
        """启动容器"""
        try:
            logger.info(f"开始启动容器: {self.container_name}")

            # 检查容器是否已存在
            if self.is_container_running():
                logger.info("容器已在运行")
                return True

            cmd = ["docker", "run", "-d", "--name", self.container_name]

            # 添加端口映射
            if ports:
                for host_port, container_port in ports.items():
                    cmd.extend(["-p", f"{host_port}:{container_port}"])

            cmd.append(self.image_name)

            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                logger.info("容器启动成功")
                return True
            else:
                logger.error(f"容器启动失败: {result.stderr}")
                return False
        except Exception as e:
            logger.error(f"启动容器时发生错误: {e}")
            return False

    def stop_container(self) -> bool:
        """停止容器"""
        try:
            logger.info(f"开始停止容器: {self.container_name}")
            cmd = ["docker", "stop", self.container_name]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                logger.info("容器停止成功")
                return True
            else:
                logger.warning(f"容器停止失败或容器不存在: {result.stderr}")
                return False
        except Exception as e:
            logger.error(f"停止容器时发生错误: {e}")
            return False

    def remove_container(self) -> bool:
        """删除容器"""
        try:
            logger.info(f"开始删除容器: {self.container_name}")
            cmd = ["docker", "rm", self.container_name]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                logger.info("容器删除成功")
                return True
            else:
                logger.warning(f"容器删除失败或容器不存在: {result.stderr}")
                return False
        except Exception as e:
            logger.error(f"删除容器时发生错误: {e}")
            return False

    def is_container_running(self) -> bool:
        """检查容器是否正在运行"""
        try:
            cmd = ["docker", "ps", "--filter", f"name={self.container_name}", "--format", "{{.Names}}"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            return self.container_name in result.stdout
        except Exception as e:
            logger.error(f"检查容器状态时发生错误: {e}")
            return False

    def get_container_status(self) -> Dict[str, Any]:
        """获取容器状态信息"""
        try:
            cmd = ["docker", "ps", "-a", "--filter", f"name={self.container_name}", "--format", "json"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0 and result.stdout.strip():
                # 解析JSON输出
                import json
                containers = [json.loads(line) for line in result.stdout.strip().split('\n') if line]
                if containers:
                    container = containers[0]
                    return {
                        "name": container.get("Names", ""),
                        "status": container.get("Status", ""),
                        "ports": container.get("Ports", ""),
                        "running": "Up" in container.get("Status", "")
                    }
            return {"name": self.container_name, "status": "Not found", "running": False}
        except Exception as e:
            logger.error(f"获取容器状态时发生错误: {e}")
            return {"name": self.container_name, "status": "Error", "running": False}

    def exec_command(self, command: str) -> str:
        """在容器中执行命令"""
        try:
            cmd = ["docker", "exec", self.container_name] + command.split()
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                return result.stdout
            else:
                return f"Error: {result.stderr}"
        except Exception as e:
            return f"Exception: {e}"

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
        success = manager.build_image()
        print("构建成功" if success else "构建失败")
    elif args.action == "start":
        ports = None
        if args.ports:
            # 解析端口映射
            ports = {}
            for mapping in args.ports.split(","):
                if ":" in mapping:
                    host, container = mapping.split(":", 1)
                    ports[host] = container
        success = manager.start_container(ports)
        print("启动成功" if success else "启动失败")
    elif args.action == "stop":
        success = manager.stop_container()
        print("停止成功" if success else "停止失败")
    elif args.action == "remove":
        success = manager.remove_container()
        print("删除成功" if success else "删除失败")
    elif args.action == "status":
        status = manager.get_container_status()
        print(f"容器状态: {status}")
    elif args.action == "exec":
        if args.command:
            output = manager.exec_command(args.command)
            print(output)
        else:
            print("请提供要执行的命令")