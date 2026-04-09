<template>
  <div>
    <div class="page-header">
      <h2>Task Management</h2>
    </div>
    
    <!-- View Task Status Modal -->
    <div v-if="showStatusModal" class="modal">
      <div class="modal-content">
        <div class="modal-header">
          <h3>Task Status</h3>
          <button class="modal-close" @click="showStatusModal = false">&times;</button>
        </div>
        <div v-if="taskStatus">
          <div class="form-group">
            <label>Task ID</label>
            <div>{{ taskStatus.task_info.id }}</div>
          </div>
          <div class="form-group">
            <label>Task Description</label>
            <div>{{ taskStatus.task_info.description }}</div>
          </div>
          <div class="form-group">
            <label>Task Status</label>
            <div>{{ taskStatus.task_status === 'completed' ? 'Completed' : 'Pending' }}</div>
          </div>
          <div class="form-group" v-if="taskStatus.task_info.startTime">
            <label>Start Time</label>
            <div>{{ taskStatus.task_info.startTime }}</div>
          </div>
          <div class="form-group" v-if="taskStatus.task_info.completeTime">
            <label>Complete Time</label>
            <div>{{ taskStatus.task_info.completeTime }}</div>
          </div>
          <div class="form-group">
            <label>Dataset ID</label>
            <div>{{ taskStatus.task_info.datasetId }}</div>
          </div>
          <div class="form-group" v-if="taskStatus.task_info.modelWeightPath">
            <label>Model Weight Path</label>
            <div>{{ taskStatus.task_info.modelWeightPath }}</div>
          </div>
          <div class="form-group" v-if="taskStatus.task_info.testResultPath">
            <label>Test Result Path</label>
            <div>{{ taskStatus.task_info.testResultPath }}</div>
          </div>
          <div class="form-group" v-if="taskStatus.task_info.result">
            <label>Test Result</label>
            <div v-if="parseResult(taskStatus.task_info.result)" class="result-container">
              <div v-if="parseResult(taskStatus.task_info.result).umap_cell_path" class="result-image">
                <label>UMAP Cell Image</label>
                <img :src="getImageUrl(parseResult(taskStatus.task_info.result).umap_cell_path)" alt="UMAP Cell" class="result-img" />
              </div>
              <div v-if="parseResult(taskStatus.task_info.result).umap_batch_path" class="result-image">
                <label>UMAP Batch Image</label>
                <img :src="getImageUrl(parseResult(taskStatus.task_info.result).umap_batch_path)" alt="UMAP Batch" class="result-img" />
              </div>
              <div class="result-metrics" v-if="parseResult(taskStatus.task_info.result).ARI || parseResult(taskStatus.task_info.result).AMI || parseResult(taskStatus.task_info.result).NMI || parseResult(taskStatus.task_info.result).HOM || parseResult(taskStatus.task_info.result).Cell_ASW || parseResult(taskStatus.task_info.result).Batch_ASW || parseResult(taskStatus.task_info.result).Graph_Connectivity">
                <h4>Metrics</h4>
                <div v-if="parseResult(taskStatus.task_info.result).ARI" class="metric-item">
                  <span>ARI:</span>
                  <span>{{ parseResult(taskStatus.task_info.result).ARI[0] }}</span>
                </div>
                <div v-if="parseResult(taskStatus.task_info.result).AMI" class="metric-item">
                  <span>AMI:</span>
                  <span>{{ parseResult(taskStatus.task_info.result).AMI[0] }}</span>
                </div>
                <div v-if="parseResult(taskStatus.task_info.result).NMI" class="metric-item">
                  <span>NMI:</span>
                  <span>{{ parseResult(taskStatus.task_info.result).NMI[0] }}</span>
                </div>
                <div v-if="parseResult(taskStatus.task_info.result).HOM" class="metric-item">
                  <span>HOM:</span>
                  <span>{{ parseResult(taskStatus.task_info.result).HOM[0] }}</span>
                </div>
                <div v-if="parseResult(taskStatus.task_info.result).Cell_ASW" class="metric-item">
                  <span>Cell_ASW:</span>
                  <span>{{ parseResult(taskStatus.task_info.result).Cell_ASW[0] }}</span>
                </div>
                <div v-if="parseResult(taskStatus.task_info.result).Batch_ASW" class="metric-item">
                  <span>Batch_ASW:</span>
                  <span>{{ parseResult(taskStatus.task_info.result).Batch_ASW[0] }}</span>
                </div>
                <div v-if="parseResult(taskStatus.task_info.result).Graph_Connectivity" class="metric-item">
                  <span>Graph_Connectivity:</span>
                  <span>{{ parseResult(taskStatus.task_info.result).Graph_Connectivity[0] }}</span>
                </div>
              </div>
            </div>
            <div v-else>{{ taskStatus.task_info.result }}</div>
          </div>
        </div>
        <div class="form-actions">
          <button type="button" class="btn" @click="showStatusModal = false">Close</button>
        </div>
      </div>
    </div>
    
    <!-- Edit Task Modal -->
    <div v-if="showEditModal" class="modal">
      <div class="modal-content">
        <div class="modal-header">
          <h3>Edit Task</h3>
          <button class="modal-close" @click="showEditModal = false">&times;</button>
        </div>
        <form @submit.prevent="editTask">
          <div class="form-group">
            <label for="description">Task Description</label>
            <textarea id="description" v-model="currentTask.description" required></textarea>
          </div>
          <div class="form-group">
            <label for="datasetId">Dataset ID</label>
            <input type="number" id="datasetId" v-model="currentTask.datasetId" required />
          </div>
          <div class="form-group">
            <label for="modelWeightPath">Model Weight Path</label>
            <input type="text" id="modelWeightPath" v-model="currentTask.modelWeightPath" />
          </div>
          <div class="form-group">
            <label for="testResultPath">Test Result Path</label>
            <input type="text" id="testResultPath" v-model="currentTask.testResultPath" />
          </div>
          <div class="form-actions">
            <button type="button" class="btn" @click="showEditModal = false">Cancel</button>
            <button type="submit" class="btn">Save</button>
          </div>
        </form>
      </div>
    </div>
    
    <div v-if="loading" class="loading">Loading...</div>
    <div v-else-if="error" class="error">{{ error }}</div>
    <div v-else>
      <div class="task-grid">
        <div v-for="task in tasks" :key="task.id" class="task-card">
          <div class="task-card-header">
            <h3>Task #{{ task.id }}</h3>
            <span class="task-status" :class="{ 'completed': task.completeTime }">
              {{ task.completeTime ? 'Completed' : 'Pending' }}
            </span>
          </div>
          <div class="task-card-body">
            <p class="task-description">{{ task.description }}</p>
            <div class="task-meta">
              <div class="meta-item">
                <label>Start Time:</label>
                <span>{{ task.startTime || '-' }}</span>
              </div>
              <div class="meta-item">
                <label>Complete Time:</label>
                <span>{{ task.completeTime || '-' }}</span>
              </div>
              <div class="meta-item">
                <label>Dataset ID:</label>
                <span>{{ task.datasetId }}</span>
              </div>
            </div>
          </div>
          <div class="task-card-footer">
            <button class="btn" @click="checkTaskStatus(task.id)">View</button>
            <button class="btn" @click="openEditModal(task.id)">Edit</button>
            <button class="btn delete-btn" @click="deleteTask(task.id)">Delete</button>
          </div>
        </div>
      </div>
      <div v-if="tasks.length === 0" class="loading">No tasks available</div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'TaskPage',
  data() {
    return {
      tasks: [],
      loading: false,
      error: '',
      showStatusModal: false,
      showEditModal: false,
      currentTask: null,
      taskStatus: null
    }
  },
  mounted() {
    this.getTasks();
  },
  methods: {
    parseResult(result) {
      try {
        return JSON.parse(result);
      } catch (error) {
        console.error('Failed to parse result field:', error);
        return null;
      }
    },
    getImageUrl(path) {
      return path;
    },
    async getTasks() {
      this.loading = true;
      this.error = '';
      try {
        const response = await fetch('/api/tasks');
        if (response.ok) {
          const tasks = await response.json();
          this.tasks = tasks;
        } else {
          throw new Error('Request failed');
        }
      } catch (error) {
        console.error('Failed to fetch tasks:', error);
        this.error = 'Failed to fetch tasks. Please try again later.';
      } finally {
        this.loading = false;
      }
    },
    async editTask() {
      try {
        const response = await fetch(`/api/tasks/${this.currentTask.id}`, {
          method: 'PUT',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify(this.currentTask)
        });
        if (response.ok) {
          this.showEditModal = false;
          this.currentTask = null;
          this.getTasks();
        } else {
          throw new Error('Request failed');
        }
      } catch (error) {
        console.error('Failed to update task:', error);
        this.error = 'Failed to update task. Please try again later.';
      }
    },
    async deleteTask(taskId) {
      if (!confirm('Are you sure you want to delete this task?')) return;
      
      try {
        const response = await fetch(`/api/tasks/${taskId}`, {
          method: 'DELETE'
        });
        if (response.ok) {
          this.getTasks();
        } else {
          throw new Error('Request failed');
        }
      } catch (error) {
        console.error('Failed to delete task:', error);
        this.error = 'Failed to delete task. Please try again later.';
      }
    },
    async checkTaskStatus(taskId) {
      try {
        const response = await fetch(`/api/tasks/${taskId}`);
        if (response.ok) {
          const task = await response.json();
          this.taskStatus = {
            status: 'success',
            task_status: task.completeTime ? 'completed' : 'pending',
            task_info: task
          };
          this.showStatusModal = true;
        } else {
          throw new Error('Request failed');
        }
      } catch (error) {
        console.error('Failed to fetch task status:', error);
        this.error = 'Failed to fetch task status. Please try again later.';
      }
    },
    async openEditModal(taskId) {
      try {
        const response = await fetch(`/api/tasks/${taskId}`);
        if (response.ok) {
          this.currentTask = await response.json();
          this.showEditModal = true;
        } else {
          throw new Error('Request failed');
        }
      } catch (error) {
        console.error('Failed to fetch task:', error);
        this.error = 'Failed to fetch task. Please try again later.';
      }
    }
  }
}
</script>
