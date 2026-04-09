<template>
  <div class="app">
    <div class="app-container">
      <nav class="sidebar">
        <div class="sidebar-header">
          <h1 style="text-align: center;">Agent Service</h1>
        </div>
        <ul class="nav-links">
          <li><router-link to="/">Chat</router-link></li>
          <li><router-link to="/tasks">Tasks</router-link></li>
          <li><router-link to="/datasets">Datasets</router-link></li>
        </ul>
      </nav>
      <main class="main-content">
        <router-view />
      </main>
    </div>
  </div>
</template>

<script>
export default {
  name: 'App'
}
</script>

<style>
/* Global variables */
:root {
  /* Black and white minimalist color scheme */
  --bg-white: #ffffff;
  --bg-light: #f5f5f5;
  --bg-dark: #e0e0e0;
  --text-primary: #000000;
  --text-secondary: #666666;
  --text-tertiary: #999999;
  --border-color: #e5e5e5;
  --border-dark: #333333;
  
  /* Shadows */
  --shadow-sm: 0 1px 3px rgba(0, 0, 0, 0.08);
  --shadow-md: 0 4px 12px rgba(0, 0, 0, 0.1);
  --shadow-lg: 0 8px 24px rgba(0, 0, 0, 0.12);
  
  /* Border radius */
  --border-radius-sm: 4px;
  --border-radius-md: 8px;
  --border-radius-lg: 16px;
  
  /* Transitions */
  --transition: all 0.2s ease;
}

/* Global style reset */
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
  line-height: 1.6;
  color: var(--text-primary);
  background: var(--bg-white);
  min-height: 100vh;
}

.container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 20px;
}

/* App container */
.app-container {
  display: flex;
  min-height: 100vh;
}

/* Sidebar styles */
.sidebar {
  width: 220px;
  background: var(--bg-white);
  color: var(--text-primary);
  box-shadow: 1px 0 0 var(--border-color);
  position: sticky;
  top: 0;
  height: 100vh;
  z-index: 1000;
  display: flex;
  flex-direction: column;
}

.sidebar-header {
  padding: 28px 24px;
  border-bottom: 1px solid var(--border-color);
}

.sidebar-header h1 {
  font-size: 18px;
  font-weight: 600;
  letter-spacing: -0.02em;
  margin: 0;
}

.nav-links {
  list-style: none;
  padding: 16px 0;
  flex: 1;
}

.nav-links li {
  margin: 0;
}

.nav-links a {
  color: var(--text-secondary);
  text-decoration: none;
  font-size: 15px;
  font-weight: 500;
  padding: 14px 24px;
  display: block;
  transition: var(--transition);
  border-left: 3px solid transparent;
}

.nav-links a:hover {
  color: var(--text-primary);
  background: var(--bg-light);
}

.nav-links a.active {
  color: var(--text-primary);
  background: var(--bg-light);
  border-left: 3px solid var(--text-primary);
}

/* Main content area */
.main-content {
  flex: 1;
  padding: 32px;
  background: var(--bg-white);
  min-height: 100vh;
}

.container {
  max-width: 100%;
  margin: 0;
  padding: 0;
}

/* Chat page styles */
.chat-container {
  display: flex;
  flex-direction: column;
  height: calc(100vh - 64px);
  border: 1px solid var(--border-color);
  border-radius: var(--border-radius-md);
  overflow: hidden;
  background: var(--bg-white);
}

.chat-header {
  background: var(--bg-white);
  padding: 20px 24px;
  border-bottom: 1px solid var(--border-color);
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.chat-header h2 {
  font-size: 18px;
  font-weight: 600;
  margin: 0;
  letter-spacing: -0.02em;
}

.chat-history {
  flex: 1;
  padding: 24px;
  overflow-y: auto;
  background: var(--bg-white);
}

.message {
  margin-bottom: 24px;
  max-width: 70%;
  animation: fadeIn 0.2s ease;
  display: flex;
  flex-direction: column;
  gap: 8px;
}

@keyframes fadeIn {
  from { opacity: 0; transform: translateY(4px); }
  to { opacity: 1; transform: translateY(0); }
}

.user-message {
  align-self: flex-end;
  background: var(--text-primary);
  color: var(--bg-white);
  border-radius: 18px 18px 4px 18px;
  padding: 14px 18px;
  margin-left: auto;
  max-width: 70%;
  word-wrap: break-word;
}

.ai-message {
  align-self: flex-start;
  background: var(--bg-light);
  color: var(--text-primary);
  border-radius: 18px 18px 18px 4px;
  padding: 14px 18px;
  margin-right: auto;
  max-width: 70%;
  word-wrap: break-word;
}

.chat-input {
  display: flex;
  padding: 20px 24px;
  border-top: 1px solid var(--border-color);
  background: var(--bg-white);
}

.chat-input input {
  flex: 1;
  padding: 12px 18px;
  border: 1px solid var(--border-color);
  border-radius: 24px;
  margin-right: 12px;
  font-size: 15px;
  transition: var(--transition);
  background: var(--bg-white);
}

.chat-input input:focus {
  outline: none;
  border-color: var(--text-primary);
}

.chat-input button {
  padding: 12px 28px;
  background: var(--text-primary);
  color: var(--bg-white);
  border: none;
  border-radius: 24px;
  cursor: pointer;
  font-size: 15px;
  font-weight: 500;
  transition: var(--transition);
}

.chat-input button:hover {
  opacity: 0.85;
}

/* Task and dataset page styles */
.page-header {
  margin-bottom: 28px;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.page-header h2 {
  font-size: 24px;
  font-weight: 600;
  letter-spacing: -0.02em;
  margin: 0;
}

.btn {
  padding: 10px 20px;
  background: var(--text-primary);
  color: var(--bg-white);
  border: none;
  border-radius: var(--border-radius-sm);
  cursor: pointer;
  text-decoration: none;
  display: inline-block;
  text-align: center;
  font-size: 14px;
  font-weight: 500;
  transition: var(--transition);
  margin-right: 8px;
}

.btn:hover {
  opacity: 0.85;
}

.delete-btn {
  background: var(--bg-white);
  color: var(--text-primary);
  border: 1px solid var(--border-dark);
}

.delete-btn:hover {
  background: var(--bg-light);
}

.table-container {
  overflow-x: auto;
  border-radius: var(--border-radius-md);
  border: 1px solid var(--border-color);
  background: var(--bg-white);
}

table {
  width: 100%;
  border-collapse: collapse;
}

table th, table td {
  padding: 16px;
  text-align: left;
  border-bottom: 1px solid var(--border-color);
}

table th {
  background: var(--bg-light);
  font-weight: 600;
  color: var(--text-primary);
  font-size: 13px;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  position: sticky;
  top: 0;
  z-index: 10;
}

table tr:hover {
  background: var(--bg-light);
}

table tr:last-child td {
  border-bottom: none;
}

/* Form styles */
.form-group {
  margin-bottom: 22px;
}

.form-group label {
  display: block;
  margin-bottom: 8px;
  font-weight: 500;
  color: var(--text-primary);
  font-size: 14px;
}

.form-group input, .form-group textarea, .form-group select {
  width: 100%;
  padding: 12px 14px;
  border: 1px solid var(--border-color);
  border-radius: var(--border-radius-sm);
  font-size: 15px;
  transition: var(--transition);
  background: var(--bg-white);
}

.form-group input:focus, .form-group textarea:focus, .form-group select:focus {
  outline: none;
  border-color: var(--text-primary);
}

.form-group textarea {
  resize: vertical;
  min-height: 120px;
}

.form-actions {
  margin-top: 28px;
  display: flex;
  gap: 12px;
  justify-content: flex-end;
}

/* Modal styles */
.modal {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background-color: rgba(0, 0, 0, 0.4);
  display: flex;
  justify-content: center;
  align-items: center;
  z-index: 1000;
  animation: fadeIn 0.2s ease;
}

.modal-content {
  background: var(--bg-white);
  padding: 32px;
  border-radius: var(--border-radius-md);
  width: 90%;
  max-width: 480px;
  box-shadow: var(--shadow-lg);
  animation: slideIn 0.2s ease;
}

@keyframes slideIn {
  from { opacity: 0; transform: translateY(-8px); }
  to { opacity: 1; transform: translateY(0); }
}

.modal-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 24px;
  padding-bottom: 16px;
  border-bottom: 1px solid var(--border-color);
}

.modal-header h3 {
  font-size: 18px;
  font-weight: 600;
  letter-spacing: -0.02em;
  margin: 0;
}

.modal-close {
  background: none;
  border: none;
  font-size: 28px;
  cursor: pointer;
  color: var(--text-tertiary);
  transition: var(--transition);
  padding: 4px;
  display: flex;
  align-items: center;
  justify-content: center;
  width: 28px;
  height: 28px;
  line-height: 1;
}

.modal-close:hover {
  color: var(--text-primary);
}

/* Task result styles */
.result-container {
  margin-top: 12px;
}

.result-image {
  margin: 16px 0;
}

.result-image label {
  display: block;
  margin-bottom: 8px;
  font-weight: 500;
  color: var(--text-primary);
  font-size: 14px;
}

.result-img {
  max-width: 100%;
  height: auto;
  border-radius: var(--border-radius-sm);
  border: 1px solid var(--border-color);
}

.result-metrics {
  margin-top: 20px;
  padding: 18px;
  background: var(--bg-light);
  border-radius: var(--border-radius-sm);
  border: 1px solid var(--border-color);
}

.result-metrics h4 {
  margin-top: 0;
  margin-bottom: 12px;
  font-size: 14px;
  font-weight: 600;
  color: var(--text-primary);
}

.metric-item {
  display: flex;
  justify-content: space-between;
  padding: 8px 0;
  border-bottom: 1px solid var(--border-color);
  font-size: 14px;
}

.metric-item:last-child {
  border-bottom: none;
}

/* Loading state */
.loading {
  text-align: center;
  padding: 48px;
  color: var(--text-secondary);
  font-size: 15px;
  background: var(--bg-white);
  border-radius: var(--border-radius-md);
  border: 1px solid var(--border-color);
  margin: 24px 0;
}

/* Error message */
.error {
  color: var(--text-primary);
  margin: 16px 0;
  padding: 16px;
  background: var(--bg-light);
  border-radius: var(--border-radius-sm);
  border-left: 3px solid var(--text-primary);
  font-size: 14px;
}

/* Task and dataset card styles */
.task-grid,
.dataset-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(360px, 1fr));
  gap: 20px;
  margin-top: 24px;
}

.task-card,
.dataset-card {
  background: var(--bg-white);
  border-radius: var(--border-radius-md);
  overflow: hidden;
  transition: var(--transition);
  border: 1px solid var(--border-color);
  display: flex;
  flex-direction: column;
}

.task-card-body,
.dataset-card-body {
  flex: 1;
  padding: 20px;
}

.task-card:hover,
.dataset-card:hover {
  border-color: var(--text-primary);
}

.task-card-header,
.dataset-card-header {
  background: var(--bg-light);
  color: var(--text-primary);
  padding: 18px 20px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  border-bottom: 1px solid var(--border-color);
}

.task-card-header h3,
.dataset-card-header h3 {
  font-size: 16px;
  font-weight: 600;
  margin: 0;
  letter-spacing: -0.01em;
}

.task-status,
.dataset-status {
  padding: 4px 10px;
  border-radius: 12px;
  font-size: 12px;
  font-weight: 500;
  border: 1px solid var(--border-color);
  background: var(--bg-white);
  transition: var(--transition);
}

.task-status.completed {
  border-color: var(--text-primary);
  background: var(--text-primary);
  color: var(--bg-white);
}

.dataset-status.processed {
  border-color: var(--text-primary);
  background: var(--text-primary);
  color: var(--bg-white);
}

.task-description,
.dataset-description {
  margin-bottom: 16px;
  font-size: 14px;
  line-height: 1.6;
  color: var(--text-secondary);
}

.task-meta,
.dataset-meta {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.meta-item {
  display: flex;
  justify-content: space-between;
  font-size: 13px;
  padding: 4px 0;
}

.meta-item label {
  font-weight: 500;
  color: var(--text-tertiary);
}

.meta-item span {
  color: var(--text-primary);
  text-align: right;
}

.path-text {
  font-size: 12px;
  color: var(--text-tertiary);
  word-break: break-all;
  text-align: right;
}

.task-card-footer,
.dataset-card-footer {
  padding: 16px 20px;
  background: var(--bg-white);
  border-top: 1px solid var(--border-color);
  display: flex;
  gap: 8px;
  justify-content: flex-end;
}

/* Responsive design */
@media (max-width: 768px) {
  .app-container {
    flex-direction: column;
  }
  
  .sidebar {
    width: 100%;
    height: auto;
    position: relative;
  }
  
  .sidebar-header {
    padding: 20px;
  }
  
  .nav-links {
    padding: 8px 0;
    display: flex;
    flex-direction: row;
    overflow-x: auto;
  }
  
  .nav-links li {
    flex: 1;
    min-width: 100px;
    text-align: center;
  }
  
  .nav-links a {
    padding: 12px;
    border-left: none;
    border-bottom: 3px solid transparent;
  }
  
  .nav-links a.active {
    border-left: none;
    border-bottom: 3px solid var(--text-primary);
  }
  
  .main-content {
    padding: 20px;
  }
  
  .chat-container {
    height: 70vh;
  }
  
  .message {
    max-width: 85%;
  }
  
  .user-message,
  .ai-message {
    max-width: 85%;
  }
  
  .page-header {
    flex-direction: column;
    align-items: flex-start;
    gap: 16px;
  }
  
  .task-grid,
  .dataset-grid {
    grid-template-columns: 1fr;
  }
  
  .modal-content {
    width: 95%;
    padding: 24px;
  }
}

/* Scrollbar styles */
::-webkit-scrollbar {
  width: 6px;
  height: 6px;
}

::-webkit-scrollbar-track {
  background: var(--bg-white);
}

::-webkit-scrollbar-thumb {
  background: var(--bg-dark);
  border-radius: 3px;
  transition: var(--transition);
}

::-webkit-scrollbar-thumb:hover {
  background: var(--text-secondary);
}
</style>
