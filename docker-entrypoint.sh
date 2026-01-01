#!/bin/bash
set -e

# 设置VNC密码（如果不存在）
if [ ! -f /home/$USER/.vnc/passwd ]; then
    mkdir -p /home/$USER/.vnc
    echo "aiuser123" | vncpasswd -f > /home/$USER/.vnc/passwd
    chmod 600 /home/$USER/.vnc/passwd
    chown -R $USER:$USER /home/$USER/.vnc
fi

# 设置VNC分辨率和深度
export USER=$USER
export HOME=/home/$USER

# 启动VNC服务器
su - $USER -c "vncserver :1 -geometry 1280x720 -depth 24"

# 启动supervisord
exec /usr/bin/supervisord -c /etc/supervisor/conf.d/supervisord.conf