const messageInput = document.getElementById('message-input');
const sendButton = document.getElementById('send-button');
const messagesContainer = document.getElementById('messages');

// 系统状态元素
const versionElement = document.getElementById('version');
const activeAgentsElement = document.getElementById('active-agents');
const sandboxStatusElement = document.getElementById('sandbox-status');
const cpuPercentElement = document.getElementById('cpu-percent');
const memoryPercentElement = document.getElementById('memory-percent');
const uptimeElement = document.getElementById('uptime');

// 沙盒监控元素
const observerScreenshot = document.getElementById('observer-screenshot');
const noScreenshot = document.getElementById('no-screenshot');
const decisionLog = document.getElementById('decision-log');
const injectionLogContent = document.getElementById('injection-log-content');

// WebSocket连接
let ws = null;

let messages = [];

// 获取系统状态
async function fetchSystemStatus() {
    try {
        const response = await fetch('/api/system/status');
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const status = await response.json();
        updateStatusDisplay(status);
    } catch (error) {
        console.error('Error fetching system status:', error);
        // 显示错误状态
        versionElement.textContent = '错误';
        activeAgentsElement.textContent = 'N/A';
        cpuPercentElement.textContent = 'N/A';
        memoryPercentElement.textContent = 'N/A';
        uptimeElement.textContent = 'N/A';
    }
}

// 更新状态显示
function updateStatusDisplay(status) {
    versionElement.textContent = status.version;
    activeAgentsElement.textContent = status.active_agents;
    sandboxStatusElement.textContent = status.sandbox_status || '未知';
    cpuPercentElement.textContent = `${status.system.cpu_percent.toFixed(1)}%`;
    memoryPercentElement.textContent = `${status.system.memory_percent.toFixed(1)}%`;

    // 格式化运行时间
    const uptime = formatUptime(status.uptime);
    uptimeElement.textContent = uptime;
}

// 格式化运行时间
function formatUptime(seconds) {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);
    return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
}

function addMessage(content, role) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}-message`;
    messageDiv.textContent = content;
    messagesContainer.appendChild(messageDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

function addStreamingMessage(content, role) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}-message`;
    messageDiv.id = 'streaming-message';
    messageDiv.textContent = content;
    messagesContainer.appendChild(messageDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

function updateStreamingMessage(content) {
    const streamingMessage = document.getElementById('streaming-message');
    if (streamingMessage) {
        streamingMessage.textContent = content;
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }
}

function finalizeStreamingMessage() {
    const streamingMessage = document.getElementById('streaming-message');
    if (streamingMessage) {
        streamingMessage.removeAttribute('id');
    }
}

async function sendMessage() {
    const userMessage = messageInput.value.trim();
    if (!userMessage) return;

    // 添加用户消息
    addMessage(userMessage, 'user');
    messages.push({ role: 'user', content: userMessage });
    messageInput.value = '';

    // 添加空的助手消息用于流式显示
    addStreamingMessage('', 'assistant');

    try {
        const response = await fetch('/v1/chat/completions', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                model: 'gpt-3.5-turbo',
                messages: messages,
                stream: true
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let assistantMessage = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop(); // 保留不完整的行

            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    const data = line.slice(6);
                    if (data === '[DONE]') {
                        finalizeStreamingMessage();
                        messages.push({ role: 'assistant', content: assistantMessage });
                        return;
                    }

                    try {
                        const chunk = JSON.parse(data);
                        if (chunk.choices && chunk.choices[0].delta && chunk.choices[0].delta.content) {
                            assistantMessage += chunk.choices[0].delta.content;
                            updateStreamingMessage(assistantMessage);
                        }
                    } catch (e) {
                        console.error('Error parsing chunk:', e);
                    }
                }
            }
        }
    } catch (error) {
        console.error('Error sending message:', error);
        updateStreamingMessage('抱歉，发生了错误。请重试。');
        finalizeStreamingMessage();
    }
}

// WebSocket连接管理
function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/sandbox`;

    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
        console.log('WebSocket连接已建立');
    };

    ws.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            handleWebSocketMessage(data);
        } catch (error) {
            console.error('解析WebSocket消息失败:', error);
        }
    };

    ws.onclose = () => {
        console.log('WebSocket连接已关闭，5秒后重连...');
        setTimeout(connectWebSocket, 5000);
    };

    ws.onerror = (error) => {
        console.error('WebSocket错误:', error);
    };
}

// 处理WebSocket消息
function handleWebSocketMessage(data) {
    switch (data.type) {
        case 'screenshot':
            updateScreenshot(data.image);
            break;
        case 'decision_log':
            updateDecisionLog(data.log);
            break;
        case 'injection_log':
            updateInjectionLog(data.log);
            break;
        case 'sandbox_status':
            updateSandboxStatus(data.status);
            break;
        default:
            console.log('未知消息类型:', data.type);
    }
}

// 更新观察者截图
function updateScreenshot(imageData) {
    if (imageData) {
        observerScreenshot.src = `data:image/png;base64,${imageData}`;
        observerScreenshot.style.display = 'block';
        noScreenshot.style.display = 'none';
    } else {
        observerScreenshot.style.display = 'none';
        noScreenshot.style.display = 'block';
    }
}

// 更新决策循环日志
function updateDecisionLog(logEntry) {
    const logElement = document.createElement('div');
    logElement.textContent = `[${new Date().toLocaleTimeString()}] ${logEntry}`;
    logElement.style.marginBottom = '5px';
    decisionLog.appendChild(logElement);
    decisionLog.scrollTop = decisionLog.scrollHeight;

    // 限制日志条数
    while (decisionLog.children.length > 50) {
        decisionLog.removeChild(decisionLog.firstChild);
    }
}

// 更新注入日志
function updateInjectionLog(logEntry) {
    const logElement = document.createElement('div');
    logElement.textContent = `[${new Date().toLocaleTimeString()}] ${logEntry}`;
    logElement.style.marginBottom = '5px';
    injectionLogContent.appendChild(logElement);
    injectionLogContent.scrollTop = injectionLogContent.scrollHeight;

    // 限制日志条数
    while (injectionLogContent.children.length > 50) {
        injectionLogContent.removeChild(injectionLogContent.firstChild);
    }
}

// 更新沙盒状态
function updateSandboxStatus(status) {
    sandboxStatusElement.textContent = status;
}

// 页面加载时获取初始状态，然后每5秒更新一次
document.addEventListener('DOMContentLoaded', () => {
    fetchSystemStatus(); // 初始加载
    setInterval(fetchSystemStatus, 5000); // 每5秒更新一次
    connectWebSocket(); // 建立WebSocket连接
});

sendButton.addEventListener('click', sendMessage);
messageInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        sendMessage();
    }
});