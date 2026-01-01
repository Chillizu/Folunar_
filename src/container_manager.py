#!/usr/bin/env python3
"""
容器管理模块
用于管理Debian容器的创建、启动、停止等操作
支持异步操作、结构化日志记录和配置化管理
"""

import asyncio
import subprocess
import os
import logging
import logging.handlers
import json
from typing import Optional, Dict, Any, List
from pathlib import Path

class ContainerManager:
    """
    异步容器管理器
    支持Docker容器的完整生命周期管理
    """

    def __init__(self, config: Dict[str, Any], connection_pool=None):
        self.config = config
        self.container_config = config.get('container', {})
        self.connection_pool = connection_pool

        # 从配置中读取参数
        self.image_name = self.container_config.get('image_name', 'debian-container')
        self.container_name = self.container_config.get('container_name', 'agent-debian')
        self.dockerfile_path = self.container_config.get('dockerfile_path', 'Dockerfile')
        self.build_timeout = self.container_config.get('build_timeout', 300)
        self.exec_timeout = self.container_config.get('exec_timeout', 30)
        self.stats_timeout = self.container_config.get('stats_timeout', 10)
        self.ports = self.container_config.get('ports', {})
        self.environment = self.container_config.get('environment', {})
        self.volumes = self.container_config.get('volumes', {})
        self.restart_policy = self.container_config.get('restart_policy', 'no')
        self.network_mode = self.container_config.get('network_mode', 'bridge')

        self.project_root = Path(__file__).parent.parent
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        """设置结构化日志记录器"""
        logger = logging.getLogger(f"{__name__}.{self.container_name}")
        logger.setLevel(logging.INFO)

        # 如果日志记录器已经配置过，直接返回
        if logger.handlers:
            return logger

        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)

        # 文件处理器（带轮转）
        log_dir = self.project_root / 'logs'
        log_dir.mkdir(exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_dir / 'container_manager.log',
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(file_formatter)

        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

        return logger

    async def build_image(self) -> Dict[str, Any]:
        """异步构建Docker镜像"""
        try:
            self.logger.info(f"开始构建镜像: {self.image_name}", extra={
                'image_name': self.image_name,
                'dockerfile_path': self.dockerfile_path
            })

            cmd = ["docker", "build", "-t", self.image_name, "-f", self.dockerfile_path, "."]
            process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=self.project_root,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                limit=1024*1024  # 1MB buffer
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self.build_timeout
                )

                if process.returncode == 0:
                    self.logger.info("镜像构建成功", extra={
                        'image_name': self.image_name,
                        'build_time_seconds': self.build_timeout
                    })
                    return {"success": True, "error": "", "output": stdout.decode('utf-8', errors='replace')}
                else:
                    error_msg = f"镜像构建失败: {stderr.decode('utf-8', errors='replace').strip()}"
                    self.logger.error(error_msg, extra={
                        'image_name': self.image_name,
                        'return_code': process.returncode,
                        'stderr': stderr.decode('utf-8', errors='replace')[:500]  # 限制日志长度
                    })
                    return {"success": False, "error": error_msg}

            except asyncio.TimeoutError:
                process.kill()
                error_msg = f"镜像构建超时 ({self.build_timeout}秒)"
                self.logger.error(error_msg, extra={
                    'image_name': self.image_name,
                    'timeout_seconds': self.build_timeout
                })
                return {"success": False, "error": error_msg}

        except Exception as e:
            error_msg = f"构建镜像时发生错误: {str(e)}"
            self.logger.error(error_msg, extra={
                'image_name': self.image_name,
                'exception_type': type(e).__name__,
                'exception_message': str(e)
            }, exc_info=True)
            return {"success": False, "error": error_msg}

    async def start_container(self, ports: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """异步启动容器"""
        try:
            self.logger.info(f"开始启动容器: {self.container_name}", extra={
                'container_name': self.container_name,
                'image_name': self.image_name
            })

            # 检查容器是否已存在并运行
            if await self.is_container_running():
                self.logger.info("容器已在运行", extra={
                    'container_name': self.container_name
                })
                return {"success": True, "error": ""}

            # 构建docker run命令
            cmd = [
                "docker", "run", "-d",
                "--name", self.container_name,
                "--restart", self.restart_policy,
                "--network", self.network_mode
            ]

            # 添加环境变量
            for key, value in self.environment.items():
                cmd.extend(["-e", f"{key}={value}"])

            # 添加卷挂载
            for host_path, container_path in self.volumes.items():
                cmd.extend(["-v", f"{host_path}:{container_path}"])

            # 添加端口映射（优先使用参数，否则使用配置）
            port_mappings = ports or self.ports
            for host_port, container_port in port_mappings.items():
                cmd.extend(["-p", f"{host_port}:{container_port}"])

            # 添加镜像名称
            cmd.append(self.image_name)

            self.logger.debug(f"执行Docker命令: {' '.join(cmd)}", extra={
                'command': ' '.join(cmd),
                'container_name': self.container_name
            })

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await process.communicate()

            if process.returncode == 0:
                container_id = stdout.decode('utf-8', errors='replace').strip()
                self.logger.info("容器启动成功", extra={
                    'container_name': self.container_name,
                    'container_id': container_id[:12],  # 只记录前12位
                    'ports': port_mappings
                })
                return {"success": True, "error": "", "container_id": container_id}
            else:
                error_msg = f"容器启动失败: {stderr.decode('utf-8', errors='replace').strip()}"
                self.logger.error(error_msg, extra={
                    'container_name': self.container_name,
                    'return_code': process.returncode,
                    'stderr': stderr.decode('utf-8', errors='replace')[:500]
                })
                return {"success": False, "error": error_msg}

        except Exception as e:
            error_msg = f"启动容器时发生错误: {str(e)}"
            self.logger.error(error_msg, extra={
                'container_name': self.container_name,
                'exception_type': type(e).__name__,
                'exception_message': str(e)
            }, exc_info=True)
            return {"success": False, "error": error_msg}

    async def stop_container(self) -> Dict[str, Any]:
        """异步停止容器"""
        try:
            self.logger.info(f"开始停止容器: {self.container_name}", extra={
                'container_name': self.container_name
            })

            cmd = ["docker", "stop", self.container_name]
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await process.communicate()

            if process.returncode == 0:
                self.logger.info("容器停止成功", extra={
                    'container_name': self.container_name
                })
                return {"success": True, "error": ""}
            else:
                error_msg = f"容器停止失败或容器不存在: {stderr.decode('utf-8', errors='replace').strip()}"
                self.logger.warning(error_msg, extra={
                    'container_name': self.container_name,
                    'return_code': process.returncode
                })
                return {"success": False, "error": error_msg}

        except Exception as e:
            error_msg = f"停止容器时发生错误: {str(e)}"
            self.logger.error(error_msg, extra={
                'container_name': self.container_name,
                'exception_type': type(e).__name__,
                'exception_message': str(e)
            }, exc_info=True)
            return {"success": False, "error": error_msg}

    async def remove_container(self) -> Dict[str, Any]:
        """异步删除容器"""
        try:
            self.logger.info(f"开始删除容器: {self.container_name}", extra={
                'container_name': self.container_name
            })

            cmd = ["docker", "rm", "-f", self.container_name]  # -f 强制删除运行中的容器
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await process.communicate()

            if process.returncode == 0:
                self.logger.info("容器删除成功", extra={
                    'container_name': self.container_name
                })
                return {"success": True, "error": ""}
            else:
                error_msg = f"容器删除失败或容器不存在: {stderr.decode('utf-8', errors='replace').strip()}"
                self.logger.warning(error_msg, extra={
                    'container_name': self.container_name,
                    'return_code': process.returncode
                })
                return {"success": False, "error": error_msg}

        except Exception as e:
            error_msg = f"删除容器时发生错误: {str(e)}"
            self.logger.error(error_msg, extra={
                'container_name': self.container_name,
                'exception_type': type(e).__name__,
                'exception_message': str(e)
            }, exc_info=True)
            return {"success": False, "error": error_msg}

    async def is_container_running(self) -> bool:
        """异步检查容器是否正在运行（优化版本）"""
        try:
            if self.connection_pool:
                # 使用连接池
                async with self.connection_pool.get_docker_client() as client:
                    try:
                        container = client.containers.get(self.container_name)
                        is_running = container.status == 'running'
                        self.logger.debug(f"容器运行状态检查: {self.container_name} -> {is_running}", extra={
                            'container_name': self.container_name,
                            'is_running': is_running
                        })
                        return is_running
                    except docker.errors.NotFound:
                        return False
            else:
                # 回退到命令行方式
                cmd = ["docker", "ps", "--filter", f"name={self.container_name}", "--format", "{{.Names}}"]
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )

                stdout, stderr = await process.communicate()

                is_running = self.container_name in stdout.decode('utf-8', errors='replace')
                self.logger.debug(f"容器运行状态检查: {self.container_name} -> {is_running}", extra={
                    'container_name': self.container_name,
                    'is_running': is_running
                })
                return is_running

        except Exception as e:
            self.logger.error(f"检查容器状态时发生错误: {str(e)}", extra={
                'container_name': self.container_name,
                'exception_type': type(e).__name__,
                'exception_message': str(e)
            }, exc_info=True)
            return False

    async def get_container_status(self) -> Dict[str, Any]:
        """异步获取容器状态信息"""
        try:
            self.logger.debug(f"获取容器状态: {self.container_name}", extra={
                'container_name': self.container_name
            })

            cmd = ["docker", "ps", "-a", "--filter", f"name={self.container_name}", "--format", "json"]
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await process.communicate()

            if process.returncode == 0 and stdout:
                stdout_text = stdout.decode('utf-8', errors='replace').strip()
                if stdout_text:
                    # 解析JSON输出
                    containers = [json.loads(line) for line in stdout_text.split('\n') if line.strip()]
                    if containers:
                        container = containers[0]
                        status_data = {
                            "name": container.get("Names", ""),
                            "status": container.get("Status", ""),
                            "ports": container.get("Ports", ""),
                            "running": "Up" in container.get("Status", ""),
                            "created": container.get("CreatedAt", ""),
                            "image": container.get("Image", "")
                        }

                        # 如果容器正在运行，获取统计信息
                        if status_data["running"]:
                            stats = await self.get_container_stats()
                            if stats["success"]:
                                status_data.update(stats["data"])

                        self.logger.debug("容器状态获取成功", extra={
                            'container_name': self.container_name,
                            'status': status_data['status'],
                            'running': status_data['running']
                        })

                        return {
                            "success": True,
                            "data": status_data,
                            "error": ""
                        }

            # 容器不存在的情况
            self.logger.debug("容器不存在", extra={
                'container_name': self.container_name
            })
            return {
                "success": True,
                "data": {
                    "name": self.container_name,
                    "status": "Not found",
                    "running": False,
                    "ports": "",
                    "created": "",
                    "image": ""
                },
                "error": ""
            }

        except json.JSONDecodeError as e:
            error_msg = f"解析容器状态JSON时发生错误: {str(e)}"
            self.logger.error(error_msg, extra={
                'container_name': self.container_name,
                'stdout': stdout.decode('utf-8', errors='replace')[:500] if 'stdout' in locals() else 'N/A'
            }, exc_info=True)
            return {
                "success": False,
                "data": {"name": self.container_name, "status": "Error", "running": False},
                "error": error_msg
            }
        except Exception as e:
            error_msg = f"获取容器状态时发生错误: {str(e)}"
            self.logger.error(error_msg, extra={
                'container_name': self.container_name,
                'exception_type': type(e).__name__,
                'exception_message': str(e)
            }, exc_info=True)
            return {
                "success": False,
                "data": {"name": self.container_name, "status": "Error", "running": False},
                "error": error_msg
            }

    async def get_container_stats(self) -> Dict[str, Any]:
        """异步获取容器统计信息（CPU、内存、网络等）"""
        try:
            self.logger.debug(f"获取容器统计信息: {self.container_name}", extra={
                'container_name': self.container_name
            })

            cmd = ["docker", "stats", "--no-stream", "--format", "json", self.container_name]
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self.stats_timeout
                )

                if process.returncode == 0 and stdout:
                    stdout_text = stdout.decode('utf-8', errors='replace').strip()
                    if stdout_text:
                        stats = json.loads(stdout_text)
                        stats_data = {
                            "cpu_percent": stats.get("CPUPerc", "0%"),
                            "memory_usage": stats.get("MemUsage", ""),
                            "memory_percent": stats.get("MemPerc", "0%"),
                            "network_io": stats.get("NetIO", ""),
                            "block_io": stats.get("BlockIO", ""),
                            "pids": stats.get("PIDs", "0"),
                            "container_id": stats.get("Container", "")[:12]  # 只保留前12位
                        }

                        self.logger.debug("容器统计信息获取成功", extra={
                            'container_name': self.container_name,
                            'cpu_percent': stats_data['cpu_percent'],
                            'memory_percent': stats_data['memory_percent'],
                            'pids': stats_data['pids']
                        })

                        return {
                            "success": True,
                            "data": stats_data,
                            "error": ""
                        }

                # 容器不存在或获取失败
                error_msg = f"获取统计信息失败: {stderr.decode('utf-8', errors='replace').strip()}"
                self.logger.warning(error_msg, extra={
                    'container_name': self.container_name,
                    'return_code': process.returncode
                })
                return {
                    "success": False,
                    "data": {},
                    "error": error_msg
                }

            except asyncio.TimeoutError:
                process.kill()
                error_msg = f"获取统计信息超时 ({self.stats_timeout}秒)"
                self.logger.error(error_msg, extra={
                    'container_name': self.container_name,
                    'timeout_seconds': self.stats_timeout
                })
                return {
                    "success": False,
                    "data": {},
                    "error": error_msg
                }

        except json.JSONDecodeError as e:
            error_msg = f"解析统计信息JSON时发生错误: {str(e)}"
            self.logger.error(error_msg, extra={
                'container_name': self.container_name,
                'stdout': stdout.decode('utf-8', errors='replace')[:500] if 'stdout' in locals() else 'N/A'
            }, exc_info=True)
            return {
                "success": False,
                "data": {},
                "error": error_msg
            }
        except Exception as e:
            error_msg = f"获取容器统计信息时发生错误: {str(e)}"
            self.logger.error(error_msg, extra={
                'container_name': self.container_name,
                'exception_type': type(e).__name__,
                'exception_message': str(e)
            }, exc_info=True)
            return {
                "success": False,
                "data": {},
                "error": error_msg
            }

    async def exec_command(self, command: str) -> Dict[str, Any]:
        """异步在容器中执行命令"""
        try:
            self.logger.info(f"在容器中执行命令: {command}", extra={
                'container_name': self.container_name,
                'command': command
            })

            cmd = ["docker", "exec", self.container_name] + command.split()
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self.exec_timeout
                )

                if process.returncode == 0:
                    output = stdout.decode('utf-8', errors='replace').strip()
                    self.logger.info("命令执行成功", extra={
                        'container_name': self.container_name,
                        'command': command,
                        'return_code': process.returncode,
                        'output_length': len(output)
                    })
                    return {"success": True, "output": output, "error": ""}
                else:
                    error_msg = f"命令执行失败: {stderr.decode('utf-8', errors='replace').strip()}"
                    self.logger.error(error_msg, extra={
                        'container_name': self.container_name,
                        'command': command,
                        'return_code': process.returncode
                    })
                    return {"success": False, "output": "", "error": error_msg}

            except asyncio.TimeoutError:
                process.kill()
                error_msg = f"命令执行超时 ({self.exec_timeout}秒): {command}"
                self.logger.error(error_msg, extra={
                    'container_name': self.container_name,
                    'command': command,
                    'timeout_seconds': self.exec_timeout
                })
                return {"success": False, "output": "", "error": error_msg}

        except Exception as e:
            error_msg = f"执行命令时发生异常: {str(e)}"
            self.logger.error(error_msg, extra={
                'container_name': self.container_name,
                'command': command,
                'exception_type': type(e).__name__,
                'exception_message': str(e)
            }, exc_info=True)
            return {"success": False, "output": "", "error": error_msg}

# 异步命令行接口
async def main():
    """异步命令行接口"""
    import argparse

    parser = argparse.ArgumentParser(description="Debian容器管理工具")
    parser.add_argument("action", choices=["build", "start", "stop", "remove", "status", "exec"],
                        help="要执行的操作")
    parser.add_argument("--command", help="要执行的命令（用于exec操作）")
    parser.add_argument("--ports", help="端口映射，格式: host_port:container_port")
    parser.add_argument("--config", default="config.yaml", help="配置文件路径")

    args = parser.parse_args()

    # 加载配置
    try:
        with open(args.config, 'r', encoding='utf-8') as f:
            import yaml
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"配置文件 {args.config} 不存在")
        return
    except Exception as e:
        print(f"加载配置文件失败: {e}")
        return

    manager = ContainerManager(config)

    if args.action == "build":
        result = await manager.build_image()
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
                    ports[host.strip()] = container.strip()
        result = await manager.start_container(ports)
        if result["success"]:
            print("启动成功")
        else:
            print(f"启动失败: {result['error']}")
    elif args.action == "stop":
        result = await manager.stop_container()
        if result["success"]:
            print("停止成功")
        else:
            print(f"停止失败: {result['error']}")
    elif args.action == "remove":
        result = await manager.remove_container()
        if result["success"]:
            print("删除成功")
        else:
            print(f"删除失败: {result['error']}")
    elif args.action == "status":
        result = await manager.get_container_status()
        if result["success"]:
            print(f"容器状态: {result['data']}")
        else:
            print(f"获取状态失败: {result['error']}")
    elif args.action == "exec":
        if args.command:
            result = await manager.exec_command(args.command)
            if result["success"]:
                print(result["output"])
            else:
                print(f"执行失败: {result['error']}")
        else:
            print("请提供要执行的命令")

if __name__ == "__main__":
    asyncio.run(main())