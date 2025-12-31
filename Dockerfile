# 使用官方Debian镜像作为基础
FROM debian:bullseye

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive
# 添加代理配置（请根据实际情况修改代理地址）
ENV HTTP_PROXY=http://proxy.example.com:8080
ENV HTTPS_PROXY=http://proxy.example.com:8080

# 配置国内apt镜像源（使用清华大学镜像源）
RUN echo "deb https://mirrors.tuna.tsinghua.edu.cn/debian/ bullseye main contrib non-free" > /etc/apt/sources.list && \
    echo "deb https://mirrors.tuna.tsinghua.edu.cn/debian/ bullseye-updates main contrib non-free" >> /etc/apt/sources.list && \
    echo "deb https://mirrors.tuna.tsinghua.edu.cn/debian-security/ bullseye-security main contrib non-free" >> /etc/apt/sources.list

# 更新包列表并安装基本工具
RUN apt-get update --allow-unauthenticated --allow-insecure-repositories && apt-get install -y --allow-unauthenticated \
    systemd \
    systemd-sysv \
    && rm -rf /var/lib/apt/lists/*

# 创建容器目录
RUN mkdir -p /container

# 设置工作目录
WORKDIR /container

# 暴露端口（如果需要）
# EXPOSE 80

# 默认命令
CMD ["/lib/systemd/systemd"]