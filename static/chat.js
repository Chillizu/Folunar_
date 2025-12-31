const messageInput = document.getElementById('message-input');
const sendButton = document.getElementById('send-button');
const messagesContainer = document.getElementById('messages');

let messages = [];

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

sendButton.addEventListener('click', sendMessage);
messageInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        sendMessage();
    }
});