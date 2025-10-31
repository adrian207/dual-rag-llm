// Model Ensemble Management
// Author: Adrian Johnson <adrian207@gmail.com>

const API_BASE = window.location.origin;

// State
let ensembles = [];
let availableModels = [];
let currentTestEnsemble = null;
let activeTab = 'ensembles';

// Strategy descriptions
const STRATEGY_DESCRIPTIONS = {
    voting: {
        title: "🗳️ Voting Strategy",
        description: "Models vote on the answer. Majority (or weighted) vote wins. Best for classification and binary decisions.",
        useCase: "Example: 3 models answer yes/no. 2 say 'yes', 1 says 'no' → Result: 'yes'"
    },
    averaging: {
        title: "📊 Averaging Strategy",
        description: "Combines responses using weighted average based on confidence. Best for numeric values.",
        useCase: "Example: Selects response from most confident model with optional weighting"
    },
    cascade: {
        title: "⚡ Cascade Strategy",
        description: "Try fast models first. If confidence is low, fall back to slower models. Optimizes speed/quality.",
        useCase: "Example: Try Qwen 7B → if confidence < 50%, use DeepSeek 33B"
    },
    best_of_n: {
        title: "🏆 Best-of-N Strategy",
        description: "Run all models and select the response with highest confidence. Maximizes quality.",
        useCase: "Example: 3 models generate code → Select most confident implementation"
    },
    specialist: {
        title: "🎓 Specialist Strategy",
        description: "Route different question types to specialized models based on domain.",
        useCase: "Example: Code → CodeLlama, Explanations → Llama3, Debugging → DeepSeek"
    },
    consensus: {
        title: "🤝 Consensus Strategy",
        description: "Requires threshold% of models to agree. High confidence for critical decisions.",
        useCase: "Example: 4 models, 70% threshold → Need 3 to agree"
    }
};

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    setupTabs();
    setupEventListeners();
    loadAllData();
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
    if (tab === 'ensembles') loadEnsembles();
}

function setupEventListeners() {
    document.getElementById('refreshAll').addEventListener('click', loadAllData);
    document.getElementById('createEnsembleForm').addEventListener('submit', handleCreateEnsemble);
    document.getElementById('strategy').addEventListener('change', updateStrategyUI);
    
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
        loadEnsembles(),
        loadAvailableModels()
    ]);
}

async function loadEnsembles() {
    try {
        const response = await fetch(`${API_BASE}/ensembles`);
        const data = await response.json();
        ensembles = data.ensembles || [];
        renderEnsembles();
    } catch (error) {
        console.error('Error loading ensembles:', error);
        showNotification('Failed to load ensembles', 'error');
    }
}

async function loadAvailableModels() {
    try {
        const response = await fetch(`${API_BASE}/models/available`);
        const data = await response.json();
        availableModels = data.available || [];
        renderModelCheckboxes();
    } catch (error) {
        console.error('Error loading models:', error);
        availableModels = [];
    }
}

// Render Ensembles
function renderEnsembles() {
    const container = document.getElementById('ensemblesList');
    
    if (ensembles.length === 0) {
        container.innerHTML = '<div class="empty-state">No ensembles created yet. Create one in the "Create Ensemble" tab!</div>';
        return;
    }
    
    container.innerHTML = ensembles.map(ensemble => `
        <div class="ensemble-card ${ensemble.enabled ? '' : 'disabled'}">
            <div class="card-header">
                <div>
                    <h3>${ensemble.name}</h3>
                    <p class="description">${ensemble.description || 'No description'}</p>
                </div>
                <div class="ensemble-badges">
                    <span class="strategy-badge strategy-${ensemble.strategy}">
                        ${getStrategyIcon(ensemble.strategy)} ${ensemble.strategy.toUpperCase().replace('_', '-')}
                    </span>
                    ${ensemble.enabled ? '<span class="status-badge enabled">✅ ENABLED</span>' : '<span class="status-badge disabled">⏸️ DISABLED</span>'}
                </div>
            </div>
            
            <div class="ensemble-models">
                <strong>Models (${ensemble.models.length}):</strong>
                <div class="model-chips">
                    ${ensemble.models.map((model, i) => `
                        <span class="model-chip">
                            ${model}
                            ${ensemble.weights ? `<span class="weight">${(ensemble.weights[i] * 100).toFixed(0)}%</span>` : ''}
                        </span>
                    `).join('')}
                </div>
            </div>
            
            <div class="card-stats">
                <div class="stat">
                    <span class="stat-label">Usage:</span>
                    <span class="stat-value">${ensemble.usage_count}</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Parallel:</span>
                    <span class="stat-value">${ensemble.parallel ? '✅ Yes' : '❌ No'}</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Timeout:</span>
                    <span class="stat-value">${ensemble.timeout}s</span>
                </div>
                ${ensemble.avg_response_time ? `
                    <div class="stat">
                        <span class="stat-label">Avg Time:</span>
                        <span class="stat-value">${ensemble.avg_response_time.toFixed(2)}s</span>
                    </div>
                ` : ''}
                ${ensemble.threshold ? `
                    <div class="stat">
                        <span class="stat-label">Threshold:</span>
                        <span class="stat-value">${(ensemble.threshold * 100).toFixed(0)}%</span>
                    </div>
                ` : ''}
            </div>
            
            ${ensemble.routing_rules && Object.keys(ensemble.routing_rules).length > 0 ? `
                <div class="routing-info">
                    <strong>Routing Rules:</strong>
                    ${Object.entries(ensemble.routing_rules).map(([type, model]) => 
                        `<span class="routing-badge">${type} → ${model}</span>`
                    ).join('')}
                </div>
            ` : ''}
            
            <div class="card-actions">
                <button onclick="testEnsemble('${ensemble.ensemble_id}')" class="btn-primary btn-sm">
                    🧪 Test
                </button>
                <button onclick="viewResults('${ensemble.ensemble_id}')" class="btn-secondary btn-sm">
                    📊 Results
                </button>
                <button onclick="toggleEnsemble('${ensemble.ensemble_id}')" class="btn-warning btn-sm">
                    ${ensemble.enabled ? '⏸️ Disable' : '▶️ Enable'}
                </button>
                <button onclick="useInChat('${ensemble.ensemble_id}')" class="btn-success btn-sm">
                    💬 Use in Chat
                </button>
                <button onclick="deleteEnsemble('${ensemble.ensemble_id}')" class="btn-danger btn-sm">
                    🗑️ Delete
                </button>
            </div>
        </div>
    `).join('');
}

function renderModelCheckboxes() {
    const container = document.getElementById('modelCheckboxes');
    
    if (availableModels.length === 0) {
        container.innerHTML = '<p class="help-text">No models available. Make sure Ollama is running.</p>';
        return;
    }
    
    container.innerHTML = availableModels.map(model => `
        <label class="model-checkbox">
            <input type="checkbox" value="${model}" onchange="updateWeightsUI()">
            <span>${model}</span>
        </label>
    `).join('');
}

function updateStrategyUI() {
    const strategy = document.getElementById('strategy').value;
    const descContainer = document.getElementById('strategyDescription');
    
    // Show description
    if (strategy && STRATEGY_DESCRIPTIONS[strategy]) {
        const desc = STRATEGY_DESCRIPTIONS[strategy];
        descContainer.innerHTML = `
            <div class="strategy-description">
                <h4>${desc.title}</h4>
                <p>${desc.description}</p>
                <div class="use-case">${desc.useCase}</div>
            </div>
        `;
    } else {
        descContainer.innerHTML = '';
    }
    
    // Show/hide relevant sections
    document.getElementById('weightsSection').style.display = 
        (strategy === 'voting' || strategy === 'averaging') ? 'block' : 'none';
    
    document.getElementById('thresholdSection').style.display =
        (strategy === 'cascade' || strategy === 'consensus') ? 'block' : 'none';
    
    document.getElementById('routingSection').style.display =
        strategy === 'specialist' ? 'block' : 'none';
    
    // Update weights UI if needed
    if (strategy === 'voting' || strategy === 'averaging') {
        updateWeightsUI();
    }
    
    // Update routing selects
    if (strategy === 'specialist') {
        updateRoutingSelects();
    }
}

function updateWeightsUI() {
    const selectedModels = getSelectedModels();
    const weightsContainer = document.getElementById('modelWeights');
    
    if (selectedModels.length === 0) {
        weightsContainer.innerHTML = '<p class="help-text">Select models first</p>';
        return;
    }
    
    weightsContainer.innerHTML = selectedModels.map((model, i) => `
        <div class="weight-input">
            <label>${model}:</label>
            <input type="number" id="weight_${i}" value="1.0" min="0.1" max="10" step="0.1">
        </div>
    `).join('');
}

function updateRoutingSelects() {
    const selectedModels = getSelectedModels();
    
    ['code', 'explanation', 'debugging'].forEach(type => {
        const select = document.getElementById(`routing_${type}`);
        select.innerHTML = '<option value="">Auto-select</option>' +
            selectedModels.map(model => `<option value="${model}">${model}</option>`).join('');
    });
}

function getSelectedModels() {
    const checkboxes = document.querySelectorAll('#modelCheckboxes input[type="checkbox"]:checked');
    return Array.from(checkboxes).map(cb => cb.value);
}

// Form Handler
async function handleCreateEnsemble(e) {
    e.preventDefault();
    
    const selectedModels = getSelectedModels();
    
    if (selectedModels.length < 2) {
        showNotification('Select at least 2 models', 'error');
        return;
    }
    
    const strategy = document.getElementById('strategy').value;
    
    // Collect weights if applicable
    let weights = null;
    if (strategy === 'voting' || strategy === 'averaging') {
        weights = selectedModels.map((_, i) => {
            const input = document.getElementById(`weight_${i}`);
            return input ? parseFloat(input.value) : 1.0;
        });
    }
    
    // Collect routing rules if applicable
    let routingRules = null;
    if (strategy === 'specialist') {
        routingRules = {};
        ['code', 'explanation', 'debugging'].forEach(type => {
            const value = document.getElementById(`routing_${type}`).value;
            if (value) routingRules[type] = value;
        });
    }
    
    const ensembleData = {
        name: document.getElementById('ensembleName').value,
        description: document.getElementById('ensembleDescription').value,
        strategy: strategy,
        models: selectedModels,
        weights: weights,
        threshold: parseFloat(document.getElementById('threshold').value) || null,
        routing_rules: routingRules,
        parallel: document.getElementById('parallel').checked,
        timeout: parseInt(document.getElementById('timeout').value),
        min_responses: parseInt(document.getElementById('minResponses').value)
    };
    
    try {
        const response = await fetch(`${API_BASE}/ensembles`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(ensembleData)
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to create ensemble');
        }
        
        const ensemble = await response.json();
        showNotification(`✅ Ensemble created: ${ensemble.name}`, 'success');
        
        // Reset form
        e.target.reset();
        document.getElementById('strategyDescription').innerHTML = '';
        document.getElementById('weightsSection').style.display = 'none';
        document.getElementById('thresholdSection').style.display = 'none';
        document.getElementById('routingSection').style.display = 'none';
        
        // Refresh and switch
        await loadEnsembles();
        switchTab('ensembles');
        
    } catch (error) {
        console.error('Error creating ensemble:', error);
        showNotification(`❌ Error: ${error.message}`, 'error');
    }
}

// Actions
function showCreateEnsemble() {
    switchTab('create');
    document.getElementById('ensembleName').focus();
}

async function testEnsemble(id) {
    currentTestEnsemble = id;
    document.getElementById('testResults').style.display = 'none';
    document.getElementById('testResultsContent').innerHTML = '';
    document.getElementById('testModal').style.display = 'block';
}

async function executeTest() {
    if (!currentTestEnsemble) return;
    
    const question = document.getElementById('testQuestion').value;
    if (!question.trim()) {
        showNotification('Enter a test question', 'error');
        return;
    }
    
    try {
        showNotification('Running test...', 'info');
        
        const response = await fetch(`${API_BASE}/ensembles/${currentTestEnsemble}/test?question=${encodeURIComponent(question)}`, {
            method: 'POST'
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Test failed');
        }
        
        const result = await response.json();
        displayTestResults(result);
        showNotification('✅ Test completed', 'success');
        
    } catch (error) {
        console.error('Error testing ensemble:', error);
        showNotification(`❌ Error: ${error.message}`, 'error');
    }
}

function displayTestResults(result) {
    const container = document.getElementById('testResultsContent');
    
    container.innerHTML = `
        <div class="test-results">
            <div class="result-header">
                <h4>Ensemble Result</h4>
                <span class="confidence-score">Confidence: ${(result.confidence_score * 100).toFixed(1)}%</span>
            </div>
            
            <div class="final-answer">
                <strong>Final Answer:</strong>
                <p>${result.final_answer}</p>
            </div>
            
            <div class="result-meta">
                <span>Strategy: ${result.strategy}</span>
                <span>Total Time: ${result.total_time.toFixed(2)}s</span>
                <span>Successful: ${result.successful_models}/${result.models_used.length}</span>
            </div>
            
            ${result.voting_breakdown ? `
                <div class="voting-breakdown">
                    <strong>Voting Breakdown:</strong>
                    ${Object.entries(result.voting_breakdown).map(([answer, votes]) =>
                        `<div>${answer.substring(0, 50)}... : ${votes} votes</div>`
                    ).join('')}
                </div>
            ` : ''}
            
            ${result.agreement_score ? `
                <div class="agreement">
                    <strong>Agreement Score:</strong> ${(result.agreement_score * 100).toFixed(1)}%
                </div>
            ` : ''}
            
            <div class="individual-responses">
                <h4>Individual Model Responses</h4>
                ${result.model_responses.map(resp => `
                    <div class="model-response ${resp.success ? 'success' : 'failed'}">
                        <div class="model-name">${resp.model}</div>
                        <div class="model-stats">
                            <span>Time: ${resp.response_time.toFixed(2)}s</span>
                            <span>Confidence: ${(resp.confidence * 100).toFixed(1)}%</span>
                            ${resp.success ? '✅' : '❌'}
                        </div>
                        ${resp.success ? `
                            <div class="model-answer">${resp.answer.substring(0, 200)}...</div>
                        ` : `
                            <div class="model-error">Error: ${resp.error}</div>
                        `}
                    </div>
                `).join('')}
            </div>
        </div>
    `;
    
    document.getElementById('testResults').style.display = 'block';
}

async function viewResults(id) {
    try {
        const response = await fetch(`${API_BASE}/ensembles/${id}/results`);
        const data = await response.json();
        const results = data.results || [];
        
        const ensemble = ensembles.find(e => e.ensemble_id === id);
        
        const container = document.getElementById('resultsContent');
        container.innerHTML = `
            <div class="results-list">
                <h3>${ensemble?.name || 'Ensemble'} Results (${results.length})</h3>
                
                ${results.length === 0 ? '<p class="empty-state">No results yet</p>' : 
                    results.map(result => `
                        <div class="result-item">
                            <div class="result-question"><strong>Q:</strong> ${result.question}</div>
                            <div class="result-answer"><strong>A:</strong> ${result.final_answer.substring(0, 200)}...</div>
                            <div class="result-meta">
                                <span>Confidence: ${(result.confidence_score * 100).toFixed(1)}%</span>
                                <span>Time: ${result.total_time.toFixed(2)}s</span>
                                <span>Models: ${result.successful_models}/${result.models_used.length}</span>
                                <span>${result.timestamp}</span>
                            </div>
                        </div>
                    `).join('')
                }
            </div>
        `;
        
        document.getElementById('resultsModal').style.display = 'block';
        
    } catch (error) {
        showNotification('Failed to load results', 'error');
    }
}

async function toggleEnsemble(id) {
    try {
        const response = await fetch(`${API_BASE}/ensembles/${id}/toggle`, { method: 'PUT' });
        const result = await response.json();
        showNotification(`Ensemble ${result.enabled ? 'enabled' : 'disabled'}`, 'success');
        await loadEnsembles();
    } catch (error) {
        showNotification('Failed to toggle ensemble', 'error');
    }
}

async function deleteEnsemble(id) {
    if (!confirm('Delete this ensemble? This cannot be undone.')) return;
    
    try {
        await fetch(`${API_BASE}/ensembles/${id}`, { method: 'DELETE' });
        showNotification('Ensemble deleted', 'success');
        await loadEnsembles();
    } catch (error) {
        showNotification('Failed to delete ensemble', 'error');
    }
}

function useInChat(id) {
    // Redirect to main chat with ensemble selected
    window.location.href = `/ui?ensemble_id=${id}`;
}

// Helper Functions
function getStrategyIcon(strategy) {
    const icons = {
        voting: '🗳️',
        averaging: '📊',
        cascade: '⚡',
        best_of_n: '🏆',
        specialist: '🎓',
        consensus: '🤝'
    };
    return icons[strategy] || '🎯';
}

function closeTestModal() {
    document.getElementById('testModal').style.display = 'none';
}

function closeResultsModal() {
    document.getElementById('resultsModal').style.display = 'none';
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

