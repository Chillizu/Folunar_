# 使用官方Debian镜像作为基础
# 如果网络问题严重，可以预先下载镜像：docker pull debian:bullseye
# 或者使用本地缓存：docker build --cache-from debian:bullseye
FROM debian:bullseye

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive
# 添加代理配置（请根据实际情况修改代理地址）
# 如果不需要代理，请注释掉下面两行
# ENV HTTP_PROXY=http://proxy.example.com:8080
# ENV HTTPS_PROXY=http://proxy.example.com:8080

# 配置apt镜像源（使用官方源）
RUN echo "deb http://deb.debian.org/debian/ bullseye main contrib non-free" > /etc/apt/sources.list && \
    echo "deb http://deb.debian.org/debian/ bullseye-updates main contrib non-free" >> /etc/apt/sources.list && \
    echo "deb http://security.debian.org/debian-security bullseye-security main contrib non-free" >> /etc/apt/sources.list

# 安装ca-certificates以解决证书问题
RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*

# 更新包列表并安装基本工具
RUN apt-get update && apt-get install -y \
    bash \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# 创建容器目录
RUN mkdir -p /container

# 设置工作目录
WORKDIR /container

# 暴露端口（如果需要）
# EXPOSE 80

# 默认命令
CMD ["/lib/systemd/systemd"]