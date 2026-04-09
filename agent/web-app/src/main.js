import { createApp } from 'vue'
import { createRouter, createWebHistory } from 'vue-router'
import App from './App.vue'
import ChatPage from './views/ChatPage.vue'
import TaskPage from './views/TaskPage.vue'
import DatasetPage from './views/DatasetPage.vue'

const routes = [
  {
    path: '/',
    name: 'Chat',
    component: ChatPage
  },
  {
    path: '/tasks',
    name: 'Tasks',
    component: TaskPage
  },
  {
    path: '/datasets',
    name: 'Datasets',
    component: DatasetPage
  }
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

createApp(App).use(router).mount('#app')
