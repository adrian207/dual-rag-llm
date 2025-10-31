// Model Fine-tuning Dashboard
// Author: Adrian Johnson <adrian207@gmail.com>

const API_BASE = window.location.origin;

// State
let datasets = [];
let jobs = [];
let models = [];
let activeTab = 'datasets';

// Format examples
const FORMAT_EXAMPLES = {
    chat: `[
  {
    "messages": [
      {"role": "system", "content": "You are a helpful coding assistant."},
      {"role": "user", "content": "How do I sort a list in Python?"},
      {"role": "assistant", "content": "Use the sorted() function: sorted_list = sorted(my_list)"}
    ]
  }
]`,
    instruct: `[
  {
    "instruction": "Write a function to reverse a string",
    "input": "hello",
    "output": "def reverse_string(s):\\n    return s[::-1]\\n\\nreverse_string('hello')  # Returns 'olleh'"
  }
]`,
    qa: `[
  {
    "question": "What is a decorator in Python?",
    "answer": "A decorator is a function that modifies another function's behavior..."
  }
]`,
    completion: `[
  {
    "text": "Function to calculate factorial:\\ndef factorial(n):\\n    if n == 0: return 1\\n    return n * factorial(n-1)"
  }
]`
};

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    setupTabs();
    setupEventListeners();
    loadAllData();
    
    // Auto-refresh every 30 seconds
    setInterval(() => {
        if (activeTab === 'jobs') {
            loadJobs();
        }
    }, 30000);
});

function setupTabs() {
    const tabBtns = document.querySelectorAll('.tab-btn');
    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const tab = btn.dataset.tab;
            switchTab(tab);
        });
    });
}

function switchTab(tab) {
    activeTab = tab;
    
    // Update buttons
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.tab === tab);
    });
    
    // Update content
    document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.toggle('active', content.id === `${tab}-tab`);
    });
    
    // Load data if needed
    if (tab === 'datasets') loadDatasets();
    if (tab === 'jobs') loadJobs();
    if (tab === 'models') loadModels();
}

function setupEventListeners() {
    document.getElementById('refreshAll').addEventListener('click', loadAllData);
    document.getElementById('uploadDatasetForm').addEventListener('submit', handleUploadDataset);
    document.getElementById('createJobForm').addEventListener('submit', handleCreateJob);
    document.getElementById('datasetFormat').addEventListener('change', updateFormatExample);
    
    // Filter buttons
    document.querySelectorAll('.filter-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
            e.target.classList.add('active');
            renderJobs(e.target.dataset.filter);
        });
    });
    
    // Modal close on outside click
    window.addEventListener('click', (e) => {
        if (e.target.classList.contains('modal')) {
            e.target.style.display = 'none';
        }
    });
}

// Load Data
async function loadAllData() {
    await Promise.all([
        loadDatasets(),
        loadJobs(),
        loadModels()
    ]);
}

async function loadDatasets() {
    try {
        const response = await fetch(`${API_BASE}/datasets`);
        const data = await response.json();
        datasets = data.datasets || [];
        renderDatasets();
        updateDatasetSelect();
    } catch (error) {
        console.error('Error loading datasets:', error);
        showNotification('Failed to load datasets', 'error');
    }
}

async function loadJobs() {
    try {
        const response = await fetch(`${API_BASE}/finetuning/jobs`);
        const data = await response.json();
        jobs = data.jobs || [];
        renderJobs('all');
    } catch (error) {
        console.error('Error loading jobs:', error);
        showNotification('Failed to load jobs', 'error');
    }
}

async function loadModels() {
    try {
        const response = await fetch(`${API_BASE}/models/finetuned`);
        const data = await response.json();
        models = data.models || [];
        renderModels();
    } catch (error) {
        console.error('Error loading models:', error);
        showNotification('Failed to load models', 'error');
    }
}

// Render Datasets
function renderDatasets() {
    const container = document.getElementById('datasetsList');
    
    if (datasets.length === 0) {
        container.innerHTML = '<div class="empty-state">No datasets uploaded yet. Upload one in the "Create New" tab!</div>';
        return;
    }
    
    container.innerHTML = datasets.map(dataset => `
        <div class="dataset-card">
            <div class="card-header">
                <div>
                    <h3>${dataset.name}</h3>
                    <p class="description">${dataset.description || 'No description'}</p>
                </div>
                <span class="format-badge">${dataset.format.toUpperCase()}</span>
            </div>
            
            <div class="card-stats">
                <div class="stat">
                    <span class="stat-label">Examples:</span>
                    <span class="stat-value">${dataset.num_examples}</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Created:</span>
                    <span class="stat-value">${formatDate(dataset.created_at)}</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Validated:</span>
                    <span class="stat-value ${dataset.validated ? 'success' : 'warning'}">
                        ${dataset.validated ? '✅ Yes' : '⚠️ Pending'}
                    </span>
                </div>
            </div>
            
            ${dataset.validation_errors.length > 0 ? `
                <div class="validation-errors">
                    <strong>⚠️ Validation Issues:</strong>
                    <ul>
                        ${dataset.validation_errors.map(err => `<li>${err}</li>`).join('')}
                    </ul>
                </div>
            ` : ''}
            
            <div class="card-actions">
                <button onclick="viewDataset('${dataset.dataset_id}')" class="btn-secondary btn-sm">
                    👁️ View
                </button>
                <button onclick="useDataset('${dataset.dataset_id}')" class="btn-primary btn-sm">
                    🚀 Train Model
                </button>
                <button onclick="deleteDataset('${dataset.dataset_id}')" class="btn-danger btn-sm">
                    🗑️ Delete
                </button>
            </div>
        </div>
    `).join('');
}

// Render Jobs
function renderJobs(filter = 'all') {
    const container = document.getElementById('jobsList');
    
    let filteredJobs = jobs;
    if (filter !== 'all') {
        filteredJobs = jobs.filter(job => job.status === filter);
    }
    
    if (filteredJobs.length === 0) {
        container.innerHTML = '<div class="empty-state">No training jobs yet. Create one in the "Create New" tab!</div>';
        return;
    }
    
    container.innerHTML = filteredJobs.map(job => `
        <div class="job-card status-${job.status}">
            <div class="card-header">
                <div>
                    <h3>${job.name}</h3>
                    <p class="job-info">
                        ${job.base_model} → ${job.output_model_name}
                    </p>
                </div>
                <span class="status-badge status-${job.status}">
                    ${getStatusIcon(job.status)} ${job.status.toUpperCase()}
                </span>
            </div>
            
            <div class="job-progress">
                ${renderJobProgress(job)}
            </div>
            
            <div class="card-stats">
                <div class="stat">
                    <span class="stat-label">Created:</span>
                    <span class="stat-value">${formatDate(job.created_at)}</span>
                </div>
                ${job.started_at ? `
                    <div class="stat">
                        <span class="stat-label">Started:</span>
                        <span class="stat-value">${formatDate(job.started_at)}</span>
                    </div>
                ` : ''}
                ${job.completed_at ? `
                    <div class="stat">
                        <span class="stat-label">Completed:</span>
                        <span class="stat-value">${formatDate(job.completed_at)}</span>
                    </div>
                ` : ''}
            </div>
            
            ${job.training_loss.length > 0 ? `
                <div class="metrics-preview">
                    <span>📊 Training Loss: ${job.training_loss[job.training_loss.length - 1].toFixed(4)}</span>
                    ${job.eval_loss.length > 0 ? `
                        <span>📈 Eval Loss: ${job.eval_loss[job.eval_loss.length - 1].toFixed(4)}</span>
                    ` : ''}
                </div>
            ` : ''}
            
            ${job.error_message ? `
                <div class="error-message">
                    ❌ Error: ${job.error_message}
                </div>
            ` : ''}
            
            <div class="card-actions">
                <button onclick="viewJob('${job.job_id}')" class="btn-primary btn-sm">
                    📊 Details
                </button>
                ${job.status === 'training' ? `
                    <button onclick="cancelJob('${job.job_id}')" class="btn-warning btn-sm">
                        ⏸️ Cancel
                    </button>
                ` : ''}
                ${job.status === 'completed' ? `
                    <button onclick="deployModel('${job.job_id}')" class="btn-success btn-sm">
                        🚀 Deploy
                    </button>
                ` : ''}
                <button onclick="deleteJob('${job.job_id}')" class="btn-danger btn-sm">
                    🗑️ Delete
                </button>
            </div>
        </div>
    `).join('');
}

function renderJobProgress(job) {
    const statusSteps = ['pending', 'preparing', 'training', 'evaluating', 'completed'];
    const currentIndex = statusSteps.indexOf(job.status);
    const progress = job.status === 'failed' || job.status === 'cancelled' ? 0 :
                    (currentIndex + 1) / statusSteps.length * 100;
    
    return `
        <div class="progress-bar">
            <div class="progress-fill" style="width: ${progress}%"></div>
        </div>
        <div class="progress-label">${Math.round(progress)}% Complete</div>
    `;
}

// Render Models
function renderModels() {
    const container = document.getElementById('modelsList');
    
    if (models.length === 0) {
        container.innerHTML = '<div class="empty-state">No fine-tuned models yet. Complete a training job first!</div>';
        return;
    }
    
    container.innerHTML = models.map(model => `
        <div class="model-card ${model.deployed ? 'deployed' : ''}">
            <div class="card-header">
                <div>
                    <h3>${model.name}</h3>
                    <p class="model-info">
                        Based on: ${model.base_model} | Version: ${model.version}
                    </p>
                    <p class="description">${model.description || 'No description'}</p>
                </div>
                ${model.deployed ? `
                    <span class="deployed-badge">✅ DEPLOYED</span>
                ` : ''}
            </div>
            
            <div class="card-stats">
                <div class="stat">
                    <span class="stat-label">Created:</span>
                    <span class="stat-value">${formatDate(model.created_at)}</span>
                </div>
                ${model.num_parameters ? `
                    <div class="stat">
                        <span class="stat-label">Parameters:</span>
                        <span class="stat-value">${formatNumber(model.num_parameters)}</span>
                    </div>
                ` : ''}
                ${model.model_size_mb ? `
                    <div class="stat">
                        <span class="stat-label">Size:</span>
                        <span class="stat-value">${model.model_size_mb.toFixed(0)} MB</span>
                    </div>
                ` : ''}
            </div>
            
            ${Object.keys(model.evaluation_metrics).length > 0 ? `
                <div class="metrics-preview">
                    ${Object.entries(model.evaluation_metrics).slice(0, 3).map(([key, value]) => `
                        <span>${key}: ${typeof value === 'number' ? value.toFixed(4) : value}</span>
                    `).join('')}
                </div>
            ` : ''}
            
            <div class="card-actions">
                <button onclick="viewModel('${model.model_id}')" class="btn-primary btn-sm">
                    📊 Details
                </button>
                ${model.deployed ? `
                    <button onclick="testModel('${model.model_id}')" class="btn-success btn-sm">
                        🧪 Test
                    </button>
                    <button onclick="undeployModel('${model.model_id}')" class="btn-warning btn-sm">
                        ⏹️ Undeploy
                    </button>
                ` : `
                    <button onclick="deployModel('${model.job_id}')" class="btn-success btn-sm">
                        🚀 Deploy
                    </button>
                `}
                <button onclick="deleteModel('${model.model_id}')" class="btn-danger btn-sm">
                    🗑️ Delete
                </button>
            </div>
        </div>
    `).join('');
}

// Form Handlers
async function handleUploadDataset(e) {
    e.preventDefault();
    
    const formData = new FormData();
    formData.append('name', document.getElementById('datasetName').value);
    formData.append('description', document.getElementById('datasetDescription').value);
    formData.append('format', document.getElementById('datasetFormat').value);
    formData.append('file', document.getElementById('datasetFile').files[0]);
    
    try {
        const response = await fetch(`${API_BASE}/datasets/upload`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Upload failed');
        }
        
        const dataset = await response.json();
        showNotification(`✅ Dataset uploaded: ${dataset.name}`, 'success');
        
        // Reset form
        e.target.reset();
        document.getElementById('formatExample').innerHTML = '';
        
        // Refresh and switch to datasets tab
        await loadDatasets();
        switchTab('datasets');
        
    } catch (error) {
        console.error('Error uploading dataset:', error);
        showNotification(`❌ Error: ${error.message}`, 'error');
    }
}

async function handleCreateJob(e) {
    e.preventDefault();
    
    const jobData = {
        name: document.getElementById('jobName').value,
        base_model: document.getElementById('baseModel').value,
        dataset_id: document.getElementById('trainingDataset').value,
        output_model_name: document.getElementById('outputModelName').value,
        learning_rate: parseFloat(document.getElementById('learningRate').value),
        num_epochs: parseInt(document.getElementById('numEpochs').value),
        batch_size: parseInt(document.getElementById('batchSize').value),
        lora_r: parseInt(document.getElementById('loraR').value),
        lora_alpha: parseInt(document.getElementById('loraAlpha').value),
        lora_dropout: parseFloat(document.getElementById('loraDropout').value),
        max_seq_length: parseInt(document.getElementById('maxSeqLength').value)
    };
    
    try {
        const response = await fetch(`${API_BASE}/finetuning/jobs`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(jobData)
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Job creation failed');
        }
        
        const job = await response.json();
        showNotification(`✅ Training job created: ${job.name}`, 'success');
        
        // Reset form
        e.target.reset();
        
        // Refresh and switch to jobs tab
        await loadJobs();
        switchTab('jobs');
        
    } catch (error) {
        console.error('Error creating job:', error);
        showNotification(`❌ Error: ${error.message}`, 'error');
    }
}

// Helper Functions
function updateFormatExample() {
    const format = document.getElementById('datasetFormat').value;
    const exampleDiv = document.getElementById('formatExample');
    
    if (format && FORMAT_EXAMPLES[format]) {
        exampleDiv.innerHTML = `
            <div class="format-example-content">
                <strong>Example Format:</strong>
                <pre>${FORMAT_EXAMPLES[format]}</pre>
            </div>
        `;
    } else {
        exampleDiv.innerHTML = '';
    }
}

function updateDatasetSelect() {
    const select = document.getElementById('trainingDataset');
    select.innerHTML = '<option value="">Select dataset...</option>' +
        datasets.filter(d => d.validated).map(d => 
            `<option value="${d.dataset_id}">${d.name} (${d.num_examples} examples)</option>`
        ).join('');
}

function getStatusIcon(status) {
    const icons = {
        pending: '⏳',
        preparing: '📦',
        training: '🔄',
        evaluating: '📊',
        completed: '✅',
        failed: '❌',
        cancelled: '🚫'
    };
    return icons[status] || '❓';
}

function formatDate(dateStr) {
    return new Date(dateStr).toLocaleString();
}

function formatNumber(num) {
    if (num >= 1e9) return (num / 1e9).toFixed(1) + 'B';
    if (num >= 1e6) return (num / 1e6).toFixed(1) + 'M';
    if (num >= 1e3) return (num / 1e3).toFixed(1) + 'K';
    return num.toString();
}

// Actions
function showCreateDataset() {
    switchTab('create');
    document.getElementById('datasetName').focus();
}

function showCreateJob() {
    switchTab('create');
    setTimeout(() => {
        document.getElementById('jobName').scrollIntoView({ behavior: 'smooth' });
        document.getElementById('jobName').focus();
    }, 100);
}

async function viewDataset(id) {
    // Implement dataset viewer
    showNotification('Dataset viewer coming soon!', 'info');
}

function useDataset(id) {
    document.getElementById('trainingDataset').value = id;
    switchTab('create');
    setTimeout(() => {
        document.getElementById('jobName').scrollIntoView({ behavior: 'smooth' });
    }, 100);
}

async function deleteDataset(id) {
    if (!confirm('Delete this dataset? This cannot be undone.')) return;
    
    try {
        await fetch(`${API_BASE}/datasets/${id}`, { method: 'DELETE' });
        showNotification('Dataset deleted', 'success');
        await loadDatasets();
    } catch (error) {
        showNotification('Failed to delete dataset', 'error');
    }
}

async function viewJob(id) {
    const job = jobs.find(j => j.job_id === id);
    if (!job) return;
    
    document.getElementById('jobDetails').innerHTML = `
        <div class="job-details-content">
            <h3>${job.name}</h3>
            <div class="details-grid">
                <div><strong>Status:</strong> ${job.status}</div>
                <div><strong>Base Model:</strong> ${job.base_model}</div>
                <div><strong>Output:</strong> ${job.output_model_name}</div>
                <div><strong>Learning Rate:</strong> ${job.learning_rate}</div>
                <div><strong>Epochs:</strong> ${job.num_epochs}</div>
                <div><strong>Batch Size:</strong> ${job.batch_size}</div>
                <div><strong>LoRA Rank:</strong> ${job.lora_r}</div>
                <div><strong>LoRA Alpha:</strong> ${job.lora_alpha}</div>
            </div>
            
            ${job.training_loss.length > 0 ? `
                <div class="metrics-chart">
                    <h4>Training Metrics</h4>
                    <div class="loss-values">
                        ${job.training_loss.map((loss, i) => 
                            `<span>Epoch ${i + 1}: ${loss.toFixed(4)}</span>`
                        ).join('')}
                    </div>
                </div>
            ` : ''}
        </div>
    `;
    
    document.getElementById('jobModal').style.display = 'block';
}

async function cancelJob(id) {
    if (!confirm('Cancel this training job?')) return;
    
    try {
        await fetch(`${API_BASE}/finetuning/jobs/${id}/cancel`, { method: 'POST' });
        showNotification('Job cancelled', 'success');
        await loadJobs();
    } catch (error) {
        showNotification('Failed to cancel job', 'error');
    }
}

async function deleteJob(id) {
    if (!confirm('Delete this job? This cannot be undone.')) return;
    
    try {
        await fetch(`${API_BASE}/finetuning/jobs/${id}`, { method: 'DELETE' });
        showNotification('Job deleted', 'success');
        await loadJobs();
    } catch (error) {
        showNotification('Failed to delete job', 'error');
    }
}

async function deployModel(jobId) {
    try {
        const response = await fetch(`${API_BASE}/models/deploy/${jobId}`, { method: 'POST' });
        const result = await response.json();
        showNotification(`✅ Model deployed: ${result.ollama_model_name}`, 'success');
        await Promise.all([loadJobs(), loadModels()]);
    } catch (error) {
        showNotification('Failed to deploy model', 'error');
    }
}

async function viewModel(id) {
    const model = models.find(m => m.model_id === id);
    if (!model) return;
    
    document.getElementById('modelDetails').innerHTML = `
        <div class="model-details-content">
            <h3>${model.name}</h3>
            <p>${model.description}</p>
            
            <div class="details-grid">
                <div><strong>Base Model:</strong> ${model.base_model}</div>
                <div><strong>Version:</strong> ${model.version}</div>
                <div><strong>Deployed:</strong> ${model.deployed ? 'Yes ✅' : 'No ❌'}</div>
                ${model.ollama_model_name ? `
                    <div><strong>Ollama Name:</strong> ${model.ollama_model_name}</div>
                ` : ''}
            </div>
            
            ${Object.keys(model.evaluation_metrics).length > 0 ? `
                <div class="metrics-section">
                    <h4>Evaluation Metrics</h4>
                    <div class="metrics-grid">
                        ${Object.entries(model.evaluation_metrics).map(([key, value]) => `
                            <div class="metric-item">
                                <span class="metric-label">${key}:</span>
                                <span class="metric-value">${typeof value === 'number' ? value.toFixed(4) : value}</span>
                            </div>
                        `).join('')}
                    </div>
                </div>
            ` : ''}
        </div>
    `;
    
    document.getElementById('modelModal').style.display = 'block';
}

async function testModel(id) {
    const model = models.find(m => m.model_id === id);
    if (!model || !model.ollama_model_name) return;
    
    // Redirect to main chat with this model selected
    window.location.href = `/ui?model=${model.ollama_model_name}`;
}

async function undeployModel(id) {
    if (!confirm('Undeploy this model? It will no longer be available for queries.')) return;
    
    try {
        await fetch(`${API_BASE}/models/undeploy/${id}`, { method: 'POST' });
        showNotification('Model undeployed', 'success');
        await loadModels();
    } catch (error) {
        showNotification('Failed to undeploy model', 'error');
    }
}

async function deleteModel(id) {
    if (!confirm('Delete this model? This cannot be undone.')) return;
    
    try {
        await fetch(`${API_BASE}/models/${id}`, { method: 'DELETE' });
        showNotification('Model deleted', 'success');
        await loadModels();
    } catch (error) {
        showNotification('Failed to delete model', 'error');
    }
}

function closeJobModal() {
    document.getElementById('jobModal').style.display = 'none';
}

function closeModelModal() {
    document.getElementById('modelModal').style.display = 'none';
}

function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.textContent = message;
    document.body.appendChild(notification);
    
    setTimeout(() => {
        notification.classList.add('show');
    }, 10);
    
    setTimeout(() => {
        notification.classList.remove('show');
        setTimeout(() => notification.remove(), 300);
    }, 3000);
}

