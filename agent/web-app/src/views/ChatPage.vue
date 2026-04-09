<template>
  <div class="chat-container">
    <div class="chat-header">
      <h2>🤖 RNAgent</h2>
      <div class="chat-actions">
        <span class="session-id">Session ID: {{ sessionId }}</span>
        <button class="btn" @click="newChat">New Chat</button>
      </div>
    </div>
    
    <div class="chat-history" ref="chatHistory">
      <div 
        v-for="(message, index) in messages" 
        :key="index" 
        :class="['message-wrapper', message.role === 'user' ? 'user-wrapper' : 'ai-wrapper']"
      >
        <div class="avatar">
          {{ message.role === 'user' ? '👤' : '🤖' }}
        </div>
        <div :class="['message-bubble', message.role === 'user' ? 'user-bubble' : 'ai-bubble']">
          <div class="message-content" v-html="formatContent(message.content)"></div>
        </div>
      </div>
      
      <div v-if="streaming" class="message-wrapper ai-wrapper">
        <div class="avatar">🤖</div>
        <div class="message-bubble ai-bubble streaming">
          <div class="message-content" v-html="formatContent(streamingContent)"></div>
          <span class="cursor">▊</span>
        </div>
      </div>
    </div>
    
    <div class="chat-input">
      <input 
        type="text" 
        v-model="inputMessage" 
        @keyup.enter="sendMessage" 
        placeholder="Enter message..." 
        :disabled="streaming"
      />
      <button @click="sendMessage" :disabled="streaming || !inputMessage.trim()">
        {{ streaming ? 'Generating...' : 'Send' }}
      </button>
    </div>
  </div>
</template>

<script>
export default {
  name: 'ChatPage',
  data() {
    return {
      sessionId: this.generateSessionId(),
      messages: [],
      inputMessage: '',
      streaming: false,
      streamingContent: ''
    }
  },
  mounted() {
    this.setViewportHeight()
    window.addEventListener('resize', this.setViewportHeight)
    
    const urlParams = new URLSearchParams(window.location.search)
    const sessionIdFromUrl = urlParams.get('sessionId')
    
    if (sessionIdFromUrl) {
      this.sessionId = sessionIdFromUrl
    } else {
      const lastSessionId = localStorage.getItem('lastSessionId')
      if (lastSessionId) {
        this.sessionId = lastSessionId
      }
    }
    
    this.loadMessages()
    this.scrollToBottom()
  },
  beforeDestroy() {
    window.removeEventListener('resize', this.setViewportHeight)
  },
  methods: {
    setViewportHeight() {
      const vh = window.innerHeight * 0.01
      document.documentElement.style.setProperty('--vh', `${vh}px`)
    },
    generateSessionId() {
      return 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9)
    },
    loadMessages() {
      const savedMessages = localStorage.getItem(this.sessionId)
      if (savedMessages) {
        this.messages = JSON.parse(savedMessages)
      }
      localStorage.setItem('lastSessionId', this.sessionId)
      this.scrollToBottom()
    },
    saveMessages() {
      localStorage.setItem(this.sessionId, JSON.stringify(this.messages))
      localStorage.setItem('lastSessionId', this.sessionId)
    },
    newChat() {
      this.sessionId = this.generateSessionId()
      this.messages = []
      this.saveMessages()
      
      const url = new URL(window.location.href)
      url.searchParams.set('sessionId', this.sessionId)
      window.history.pushState({}, '', url.toString())
    },
    formatContent(content) {
       if (!content) return '';
        // Step 1: Protect <img>...</img> tags (supports multi-line content)
      // Use [\s\S] to match all characters including newlines
        let protectedContent = content.replace(/<img>([\s\S]*?)<\/img>/g, '###IMGSTART###$1###IMGEND###');

        // Escape other HTML special characters and process Markdown format
        let formattedContent = protectedContent
          .replace(/&/g, '&amp;')
          .replace(/</g, '&lt;')
          .replace(/>/g, '&gt;')
          .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')           // **bold**
          .replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>') // ```code block```
          .replace(/`([^`]+)`/g, '<code>$1</code>')                  // `inline code`
          .replace(/\n/g, '<br>');                                    // newline to <br>

        // Step 2: Restore image tags and generate actual <img> elements
        formattedContent = formattedContent.replace(/###IMGSTART###([\s\S]*?)###IMGEND###/g, (match, path) => {
          // Clean path: normalize slashes + trim whitespace
          const cleanPath = path
            .replace(/\\+/g, '/')      // convert backslashes to forward slashes
            .trim();                    // trim whitespace
          
          // Return complete img tag with styles
          return `<img src="${cleanPath}" class="chat-img" style="max-width:600px;height:auto;border-radius:8px;margin:8px 0;display:block;">`;
        });
        console.log('Formatted Content:', formattedContent);
        return formattedContent;
    },
    scrollToBottom() {
      this.$nextTick(() => {
        const chatHistory = this.$refs.chatHistory
        if (chatHistory) {
          chatHistory.scrollTop = chatHistory.scrollHeight
        }
      })
    },
    async sendMessage() {
      if (!this.inputMessage.trim() || this.streaming) return

      const userMessage = {
        role: 'user',
        content: this.inputMessage
      }
      this.messages.push(userMessage)
      this.saveMessages()
      this.scrollToBottom()

      const message = this.inputMessage
      this.inputMessage = ''
      this.streaming = true
      this.streamingContent = ''

      try {
        const response = await fetch(`/api/agent/chat?memoryId=${encodeURIComponent(this.sessionId)}&message=${encodeURIComponent(message)}`)

        if (response.ok) {
          const reader = response.body.getReader()
          const decoder = new TextDecoder('utf-8')
          
          while (true) {
            const { done, value } = await reader.read()
            if (done) break
            
            const chunk = decoder.decode(value, { stream: true })
            this.streamingContent += chunk
            this.scrollToBottom()
          }
          
          const aiMessage = {
            role: 'ai',
            content: this.streamingContent
          }
          this.messages.push(aiMessage)
          this.saveMessages()
        } else {
          throw new Error('Request failed')
        }
      } catch (error) {
        console.error('Failed to send message:', error)
        const errorMessage = {
          role: 'ai',
          content: 'Sorry, failed to send message. Please try again later.'
        }
        this.messages.push(errorMessage)
        this.saveMessages()
      } finally {
        this.streaming = false
        this.streamingContent = ''
        this.scrollToBottom()
      }
    }
  }
}
</script>

<style scoped>
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

/* Fix: Match main layout, don't cover sidebar, 100% height */
.chat-container {
  width: 100%;
  height: 100%;
  min-height: 100%;
  display: flex;
  flex-direction: column;
  background: #ffffff;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
  overflow: hidden;
}

.chat-header {
  flex-shrink: 0;
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 20px 24px;
  background: #ffffff;
  border-bottom: 1px solid #e5e5e5;
}

.chat-header h2 {
  margin: 0;
  color: #000000;
  font-size: 18px;
  font-weight: 600;
  letter-spacing: -0.02em;
}

.chat-actions {
  display: flex;
  align-items: center;
  gap: 12px;
}

.session-id {
  font-size: 12px;
  color: #999999;
  background-color: #f5f5f5;
  padding: 6px 12px;
  border-radius: 12px;
  font-family: 'Courier New', monospace;
}

.btn {
  padding: 10px 20px;
  background: #000000;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 14px;
  font-weight: 500;
  transition: all 0.2s ease;
  white-space: nowrap;
}

.btn:hover {
  opacity: 0.85;
}

/* Auto-fill remaining height, no overflow */
.chat-history {
  flex: 1;
  overflow-y: auto;
  padding: 24px;
  display: flex;
  flex-direction: column;
  gap: 18px;
  min-height: 0;
}

.message-wrapper {
  display: flex;
  align-items: flex-start;
  gap: 12px;
  max-width: 85%;
}

.user-wrapper {
  align-self: flex-end;
  flex-direction: row-reverse;
}

.ai-wrapper {
  align-self: flex-start;
}

.avatar {
  width: 36px;
  height: 36px;
  border-radius: 50%;
  background: #f5f5f5;
  border: 1px solid #e5e5e5;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 18px;
  flex-shrink: 0;
}

.message-bubble {
  padding: 14px 18px;
  border-radius: 18px;
  max-width: 100%;
  word-wrap: break-word;
  line-height: 1.6;
  font-size: 15px;
}

.user-bubble {
  background: #000000;
  color: #ffffff;
  border-bottom-right-radius: 4px;
}

.ai-bubble {
  background: #f5f5f5;
  color: #000000;
  border-bottom-left-radius: 4px;
}

.ai-bubble.streaming {
  background: #f5f5f5;
  border: 1px dashed #000000;
}

.message-content {
  white-space: pre-wrap;
}

.message-content :deep(pre) {
  background: #ffffff;
  padding: 12px;
  border-radius: 4px;
  overflow-x: auto;
  margin: 8px 0;
  font-size: 13px;
  border: 1px solid #e5e5e5;
}

.message-content :deep(code) {
  background: #ffffff;
  padding: 2px 6px;
  border-radius: 4px;
  font-family: 'Courier New', monospace;
  font-size: 13px;
  border: 1px solid #e5e5e5;
}

.user-bubble .message-content :deep(code) {
  background: rgba(255, 255, 255, 0.15);
  border: 1px solid rgba(255, 255, 255, 0.2);
  color: white;
}

.message-content :deep(.chat-img) {
  max-width: 100%;
  height: auto;
  border-radius: 4px;
  margin: 8px 0;
  border: 1px solid #e5e5e5;
}

.cursor {
  display: inline-block;
  animation: blink 1s infinite;
  color: #000000;
  margin-left: 4px;
}

@keyframes blink {
  0%, 50% { opacity: 1; }
  51%, 100% { opacity: 0; }
}

.chat-input {
  flex-shrink: 0;
  display: flex;
  gap: 12px;
  padding: 20px 24px;
  background: #ffffff;
  border-top: 1px solid #e5e5e5;
}

.chat-input input {
  flex: 1;
  padding: 12px 18px;
  border: 1px solid #e5e5e5;
  border-radius: 24px;
  font-size: 15px;
  outline: none;
  transition: all 0.2s ease;
  background: #ffffff;
  min-width: 0;
}

.chat-input input:focus {
  border-color: #000000;
}

.chat-input input:disabled {
  background: #f5f5f5;
  cursor: not-allowed;
}

.chat-input button {
  padding: 12px 28px;
  background: #000000;
  color: #ffffff;
  border: none;
  border-radius: 24px;
  cursor: pointer;
  font-size: 15px;
  font-weight: 500;
  transition: all 0.2s ease;
  white-space: nowrap;
  flex-shrink: 0;
}

.chat-input button:hover:not(:disabled) {
  opacity: 0.85;
}

.chat-input button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.chat-history::-webkit-scrollbar {
  width: 6px;
}

.chat-history::-webkit-scrollbar-track {
  background: transparent;
}

.chat-history::-webkit-scrollbar-thumb {
  background: #e0e0e0;
  border-radius: 3px;
}

.chat-history::-webkit-scrollbar-thumb:hover {
  background: #666666;
}

@media (max-width: 640px) {
  .chat-header {
    padding: 16px 20px;
  }
  
  .chat-header h2 {
    font-size: 17px;
  }
  
  .session-id {
    display: none;
  }
  
  .chat-history {
    padding: 16px;
    gap: 14px;
  }
  
  .message-wrapper {
    max-width: 90%;
  }
  
  .avatar {
    width: 32px;
    height: 32px;
    font-size: 16px;
  }
  
  .message-bubble {
    padding: 12px 16px;
    font-size: 14px;
  }
  
  .chat-input {
    padding: 16px 20px;
    gap: 10px;
  }
  
  .chat-input input {
    padding: 10px 16px;
  }
  
  .chat-input button {
    padding: 10px 20px;
    font-size: 14px;
  }
}
</style>