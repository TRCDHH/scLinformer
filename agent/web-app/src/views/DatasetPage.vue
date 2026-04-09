<template>
  <div>
    <div class="page-header">
      <h2>Dataset Management</h2>
      <button class="btn" @click="showCreateModal = true">Create Dataset</button>
    </div>
    
    <!-- Create Dataset Modal -->
    <div v-if="showCreateModal" class="modal">
      <div class="modal-content">
        <div class="modal-header">
          <h3>Create Dataset</h3>
          <button class="modal-close" @click="showCreateModal = false">&times;</button>
        </div>
        <form @submit.prevent="createDataset">
          <div class="form-group">
            <label for="name">Name</label>
            <input type="text" id="name" v-model="newDataset.name" required />
          </div>
          <div class="form-group">
            <label for="description">Description</label>
            <textarea id="description" v-model="newDataset.description" required></textarea>
          </div>
          <div class="form-group">
            <label for="status">Status</label>
            <select id="status" v-model="newDataset.status" required>
              <option value="Processed">Processed</option>
              <option value="Unprocessed">Unprocessed</option>
            </select>
          </div>
          <div class="form-group">
            <label for="datasetPath">Dataset Path</label>
            <input type="text" id="datasetPath" v-model="newDataset.datasetPath" required />
          </div>
          <div class="form-group">
            <label for="processedPath">Processed Path</label>
            <input type="text" id="processedPath" v-model="newDataset.processedPath" />
          </div>
          <div class="form-actions">
            <button type="button" class="btn" @click="showCreateModal = false">Cancel</button>
            <button type="submit" class="btn">Create</button>
          </div>
        </form>
      </div>
    </div>
    
    <!-- Edit Dataset Modal -->
    <div v-if="showEditModal" class="modal">
      <div class="modal-content">
        <div class="modal-header">
          <h3>Edit Dataset</h3>
          <button class="modal-close" @click="showEditModal = false">&times;</button>
        </div>
        <form @submit.prevent="editDataset">
          <div class="form-group">
            <label for="name">Name</label>
            <input type="text" id="name" v-model="currentDataset.name" required />
          </div>
          <div class="form-group">
            <label for="description">Description</label>
            <textarea id="description" v-model="currentDataset.description" required></textarea>
          </div>
          <div class="form-group">
            <label for="status">Status</label>
            <select id="status" v-model="currentDataset.status" required>
              <option value="Processed">Processed</option>
              <option value="Unprocessed">Unprocessed</option>
            </select>
          </div>
          <div class="form-group">
            <label for="datasetPath">Dataset Path</label>
            <input type="text" id="datasetPath" v-model="currentDataset.datasetPath" required />
          </div>
          <div class="form-group">
            <label for="processedPath">Processed Path</label>
            <input type="text" id="processedPath" v-model="currentDataset.processedPath" />
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
      <div class="dataset-grid">
        <div v-for="dataset in datasets" :key="dataset.id" class="dataset-card">
          <div class="dataset-card-header">
            <h3>{{ dataset.name }}</h3>
            <span class="dataset-status" :class="{ 'processed': dataset.status === 'Processed' }">
              {{ dataset.status }}
            </span>
          </div>
          <div class="dataset-card-body">
            <p class="dataset-description">{{ dataset.description }}</p>
            <div class="dataset-meta">
              <div class="meta-item">
                <label>ID:</label>
                <span>{{ dataset.id }}</span>
              </div>
              <div class="meta-item">
                <label>Upload Time:</label>
                <span>{{ dataset.uploadTime }}</span>
              </div>
              <div class="meta-item">
                <label>Dataset Path:</label>
                <span class="path-text">{{ dataset.datasetPath }}</span>
              </div>
              <div class="meta-item" v-if="dataset.processedPath">
                <label>Processed Path:</label>
                <span class="path-text">{{ dataset.processedPath }}</span>
              </div>
            </div>
          </div>
          <div class="dataset-card-footer">
            <button class="btn" @click="openEditModal(dataset.id)">Edit</button>
            <button class="btn delete-btn" @click="deleteDataset(dataset.id)">Delete</button>
          </div>
        </div>
      </div>
      <div v-if="datasets.length === 0" class="loading">No datasets available</div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'DatasetPage',
  data() {
    return {
      datasets: [],
      loading: false,
      error: '',
      showCreateModal: false,
      showEditModal: false,
      newDataset: {
        name: '',
        description: '',
        status: 'Unprocessed',
        datasetPath: '',
        processedPath: ''
      },
      currentDataset: null
    }
  },
  mounted() {
    this.getDatasets();
  },
  methods: {
    async getDatasets() {
      this.loading = true;
      this.error = '';
      try {
        const response = await fetch('/api/datasets');
        if (response.ok) {
          const datasets = await response.json();
          this.datasets = datasets;
        } else {
          throw new Error('Request failed');
        }
      } catch (error) {
        console.error('Failed to fetch datasets:', error);
        this.error = 'Failed to fetch datasets. Please try again later.';
      } finally {
        this.loading = false;
      }
    },
    async createDataset() {
      try {
        const response = await fetch('/api/datasets', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify(this.newDataset)
        });
        if (response.ok) {
          this.showCreateModal = false;
          this.newDataset = {
            name: '',
            description: '',
            status: 'Unprocessed',
            datasetPath: '',
            processedPath: ''
          };
          this.getDatasets();
        } else {
          throw new Error('Request failed');
        }
      } catch (error) {
        console.error('Failed to create dataset:', error);
        this.error = 'Failed to create dataset. Please try again later.';
      }
    },
    async openEditModal(datasetId) {
      try {
        const response = await fetch(`/api/datasets/${datasetId}`);
        if (response.ok) {
          this.currentDataset = await response.json();
          this.showEditModal = true;
        } else {
          throw new Error('Request failed');
        }
      } catch (error) {
        console.error('Failed to fetch dataset:', error);
        this.error = 'Failed to fetch dataset. Please try again later.';
      }
    },
    async editDataset() {
      try {
        const response = await fetch(`/api/datasets/${this.currentDataset.id}`, {
          method: 'PUT',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify(this.currentDataset)
        });
        if (response.ok) {
          this.showEditModal = false;
          this.currentDataset = null;
          this.getDatasets();
        } else {
          throw new Error('Request failed');
        }
      } catch (error) {
        console.error('Failed to update dataset:', error);
        this.error = 'Failed to update dataset. Please try again later.';
      }
    },
    async deleteDataset(datasetId) {
      if (!confirm('Are you sure you want to delete this dataset?')) return;
      
      try {
        const response = await fetch(`/api/datasets/${datasetId}`, {
          method: 'DELETE'
        });
        if (response.ok) {
          this.getDatasets();
        } else {
          throw new Error('Request failed');
        }
      } catch (error) {
        console.error('Failed to delete dataset:', error);
        this.error = 'Failed to delete dataset. Please try again later.';
      }
    }
  }
}
</script>
