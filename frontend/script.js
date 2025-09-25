class ChatApp {
    constructor() {
        this.apiUrl = 'http://localhost:5000/api';
        this.initializeElements();
        this.setupEventListeners();
        this.checkStatus();
    }

    initializeElements() {
        this.messagesContainer = document.getElementById('messages');
        this.userInput = document.getElementById('userInput');
        this.sendButton = document.getElementById('sendButton');
        this.typingIndicator = document.getElementById('typing');
        this.statusIndicator = document.getElementById('status');
    }

    setupEventListeners() {
        this.sendButton.addEventListener('click', () => this.sendMessage());
        this.userInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.sendMessage();
        });
    }

    async checkStatus() {
        try {
            const response = await fetch(`${this.apiUrl}/status`);
            const data = await response.json();
            this.statusIndicator.textContent = data.model_loaded ? '🟢 Онлайн' : '🟡 Загрузка модели...';
        } catch (error) {
            this.statusIndicator.textContent = '🔴 Офлайн';
            console.error('Status check failed:', error);
        }
    }

    async sendMessage() {
        const message = this.userInput.value.trim();
        if (!message) return;

        this.addMessage('user', message);
        this.userInput.value = '';
        this.setLoading(true);

        try {
            const response = await fetch(`${this.apiUrl}/chat`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ message: message })
            });

            const data = await response.json();

            if (data.status === 'success') {
                this.addMessage('bot', data.response);
            } else {
                throw new Error(data.error || 'Unknown error');
            }
        } catch (error) {
            console.error('Chat error:', error);
            this.addMessage('bot', '⚠️ Извините, произошла ошибка. Попробуйте еще раз.');
        } finally {
            this.setLoading(false);
        }
    }

    addMessage(sender, text) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}-message`;
        
        messageDiv.innerHTML = `
            <div class="avatar">${sender === 'user' ? '👤' : '🤖'}</div>
            <div class="content">
                <strong>${sender === 'user' ? 'Вы' : 'DeepHelper'}:</strong> ${this.escapeHtml(text)}
            </div>
        `;

        this.messagesContainer.appendChild(messageDiv);
        this.messagesContainer.scrollTop = this.messagesContainer.scrollHeight;
    }

    setLoading(loading) {
        this.sendButton.disabled = loading;
        this.typingIndicator.style.display = loading ? 'block' : 'none';
        if (loading) {
            this.messagesContainer.scrollTop = this.messagesContainer.scrollHeight;
        }
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// Запускаем приложение когда страница загрузится
document.addEventListener('DOMContentLoaded', () => {
    new ChatApp();
});
