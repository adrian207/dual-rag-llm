// Automatic Model Selection Dashboard
// Author: Adrian Johnson <adrian207@gmail.com>

const API_BASE = window.location.origin;
let config = null;
let routingMatrix = [];
let performance = [];
let decisions = [];

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    setupTabs();
    setupEventListeners();
    loadAllData();
});

function setupTabs() {
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const tab = btn.dataset.tab;
            switchTab(tab);
        });
    });
}

function switchTab(tab) {
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.tab === tab);
    });
    document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.toggle('active', content.id === `${tab}-tab`);
    });
    
    if (tab === 'routing') loadRouting();
    if (tab === 'performance') loadPerformance();
    if (tab === 'decisions') loadDecisions();
}

function setupEventListeners() {
    document.getElementById('refreshAll').addEventListener('click', loadAllData);
    document.getElementById('saveConfig').addEventListener('click', saveConfiguration);
    document.getElementById('confidenceThreshold').addEventListener('input', (e) => {
        document.getElementById('confidenceValue').textContent = e.target.value;
    });
    document.getElementById('learningRate').addEventListener('input', (e) => {
        document.getElementById('learningValue').textContent = e.target.value;
    });
}

async function loadAllData() {
    await Promise.all([
        loadConfig(),
        loadDecisions()
    ]);
}

async function loadConfig() {
    try {
        const response = await fetch(`${API_BASE}/auto-selection/config`);
        config = await response.json();
        renderConfig();
    } catch (error) {
        console.error('Error loading config:', error);
    }
}

function renderConfig() {
    if (!config) return;
    
    document.getElementById('autoEnabled').checked = config.enabled;
    document.getElementById('usePerformance').checked = config.use_performance_data;
    document.getElementById('confidenceThreshold').value = config.confidence_threshold;
    document.getElementById('confidenceValue').textContent = config.confidence_threshold;
    document.getElementById('learningRate').value = config.learning_rate;
    document.getElementById('learningValue').textContent = config.learning_rate;
    document.getElementById('minQueries').value = config.min_queries_for_routing;
    document.getElementById('fallbackModel').value = config.fallback_model;
}

async function saveConfiguration() {
    const newConfig = {
        enabled: document.getElementById('autoEnabled').checked,
        use_performance_data: document.getElementById('usePerformance').checked,
        confidence_threshold: parseFloat(document.getElementById('confidenceThreshold').value),
        learning_rate: parseFloat(document.getElementById('learningRate').value),
        min_queries_for_routing: parseInt(document.getElementById('minQueries').value),
        fallback_model: document.getElementById('fallbackModel').value
    };
    
    try {
        await fetch(`${API_BASE}/auto-selection/config`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(newConfig)
        });
        showNotification('✅ Configuration saved', 'success');
        await loadConfig();
    } catch (error) {
        showNotification('❌ Failed to save configuration', 'error');
    }
}

async function loadRouting() {
    try {
        const response = await fetch(`${API_BASE}/auto-selection/routing`);
        const data = await response.json();
        routingMatrix = data.routing || [];
        renderRouting();
    } catch (error) {
        console.error('Error loading routing:', error);
    }
}

function renderRouting() {
    const container = document.getElementById('routingMatrix');
    
    if (routingMatrix.length === 0) {
        container.innerHTML = '<p class="empty-state">No routing configured</p>';
        return;
    }
    
    container.innerHTML = routingMatrix.map(route => `
        <div class="routing-card">
            <div class="route-header">
                <h3>${formatQueryType(route.query_type)}</h3>
                <span class="confidence-badge">${(route.min_confidence * 100).toFixed(0)}% min</span>
            </div>
            <div class="route-model">
                <strong>Primary:</strong> <code>${route.primary_model}</code>
            </div>
            <div class="route-fallbacks">
                <strong>Fallbacks:</strong>
                ${route.fallback_models.map(m => `<code>${m}</code>`).join(' → ')}
            </div>
            <div class="route-reasoning">${route.reasoning}</div>
        </div>
    `).join('');
}

async function loadPerformance() {
    try {
        const response = await fetch(`${API_BASE}/auto-selection/performance`);
        const data = await response.json();
        performance = data.performance || [];
        renderPerformance();
    } catch (error) {
        console.error('Error loading performance:', error);
    }
}

function renderPerformance() {
    const container = document.getElementById('performanceList');
    
    if (performance.length === 0) {
        container.innerHTML = '<p class="empty-state">No performance data yet</p>';
        return;
    }
    
    // Group by query type
    const grouped = {};
    performance.forEach(p => {
        if (!grouped[p.query_type]) grouped[p.query_type] = [];
        grouped[p.query_type].push(p);
    });
    
    container.innerHTML = Object.entries(grouped).map(([type, perfs]) => `
        <div class="performance-section">
            <h3>${formatQueryType(type)}</h3>
            <div class="performance-cards">
                ${perfs.map(p => `
                    <div class="performance-card">
                        <div class="perf-model">${p.model}</div>
                        <div class="perf-stats">
                            <div class="perf-stat">
                                <span class="label">Queries:</span>
                                <span class="value">${p.queries_handled}</span>
                            </div>
                            <div class="perf-stat">
                                <span class="label">Avg Time:</span>
                                <span class="value">${p.avg_response_time.toFixed(2)}s</span>
                            </div>
                            <div class="perf-stat">
                                <span class="label">Success:</span>
                                <span class="value">${(p.success_rate * 100).toFixed(0)}%</span>
                            </div>
                            <div class="perf-stat">
                                <span class="label">Rating:</span>
                                <span class="value ${getRatingClass(p.avg_rating)}">${p.avg_rating.toFixed(1)} ⭐</span>
                            </div>
                        </div>
                    </div>
                `).join('')}
            </div>
        </div>
    `).join('');
}

async function loadDecisions() {
    try {
        const response = await fetch(`${API_BASE}/auto-selection/decisions?limit=50`);
        const data = await response.json();
        decisions = data.decisions || [];
        renderDecisions();
        updateStats();
    } catch (error) {
        console.error('Error loading decisions:', error);
    }
}

function renderDecisions() {
    const container = document.getElementById('decisionsList');
    
    if (decisions.length === 0) {
        container.innerHTML = '<p class="empty-state">No decisions yet</p>';
        return;
    }
    
    container.innerHTML = decisions.slice().reverse().map(d => `
        <div class="decision-card">
            <div class="decision-header">
                <span class="query-type-badge">${formatQueryType(d.classification.query_type)}</span>
                <span class="confidence-badge">${(d.classification.confidence * 100).toFixed(0)}%</span>
                <span class="timestamp">${formatDate(d.timestamp)}</span>
            </div>
            <div class="decision-query">${d.query}</div>
            <div class="decision-result">
                <strong>Selected:</strong> <code>${d.selected_model}</code>
                ${d.fallback_used ? '<span class="fallback-badge">⚠️ Fallback</span>' : ''}
                ${d.user_override ? '<span class="override-badge">👤 Override</span>' : ''}
            </div>
            <div class="decision-reasoning">${d.reasoning}</div>
            ${d.classification.language ? `<div class="language-badge">${d.classification.language}</div>` : ''}
        </div>
    `).join('');
}

function updateStats() {
    if (decisions.length === 0) return;
    
    const totalDecisions = decisions.length;
    const avgConfidence = decisions.reduce((sum, d) => sum + d.classification.confidence, 0) / totalDecisions;
    const fallbackCount = decisions.filter(d => d.fallback_used).length;
    const overrideCount = decisions.filter(d => d.user_override).length;
    
    document.getElementById('totalDecisions').textContent = totalDecisions;
    document.getElementById('avgConfidence').textContent = (avgConfidence * 100).toFixed(1) + '%';
    document.getElementById('fallbackRate').textContent = ((fallbackCount / totalDecisions) * 100).toFixed(1) + '%';
    document.getElementById('overrideRate').textContent = ((overrideCount / totalDecisions) * 100).toFixed(1) + '%';
}

async function testClassification() {
    const query = document.getElementById('testQuery').value;
    if (!query.trim()) {
        showNotification('Enter a query', 'error');
        return;
    }
    
    try {
        const response = await fetch(`${API_BASE}/auto-selection/select?query=${encodeURIComponent(query)}`, {
            method: 'POST'
        });
        const result = await response.json();
        
        const container = document.getElementById('testResults');
        container.style.display = 'block';
        container.innerHTML = `
            <h3>Classification Result</h3>
            <div class="test-result-grid">
                <div class="test-item">
                    <strong>Query Type:</strong>
                    <span class="query-type-badge">${formatQueryType(result.classification.query_type)}</span>
                </div>
                <div class="test-item">
                    <strong>Confidence:</strong>
                    <span class="confidence-badge">${(result.classification.confidence * 100).toFixed(1)}%</span>
                </div>
                <div class="test-item">
                    <strong>Selected Model:</strong>
                    <code>${result.selected_model}</code>
                </div>
                <div class="test-item">
                    <strong>Complexity:</strong>
                    <span class="complexity-badge">${result.classification.complexity}</span>
                </div>
            </div>
            
            ${result.classification.language ? `
                <div class="test-language">
                    <strong>Detected Language:</strong> ${result.classification.language}
                </div>
            ` : ''}
            
            <div class="test-keywords">
                <strong>Keywords Matched:</strong>
                ${result.classification.keywords_matched.map(k => `<span class="keyword-badge">${k}</span>`).join('')}
            </div>
            
            <div class="test-reasoning">
                <strong>Reasoning:</strong> ${result.reasoning}
            </div>
            
            ${result.classification.secondary_types.length > 0 ? `
                <div class="test-secondary">
                    <strong>Alternative Types:</strong>
                    ${result.classification.secondary_types.map(t => `<span class="secondary-badge">${formatQueryType(t)}</span>`).join(' ')}
                </div>
            ` : ''}
        `;
        
        showNotification('✅ Classification complete', 'success');
    } catch (error) {
        showNotification('❌ Classification failed', 'error');
    }
}

// Utility Functions
function formatQueryType(type) {
    return type.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
}

function formatDate(dateStr) {
    return new Date(dateStr).toLocaleString();
}

function getRatingClass(rating) {
    if (rating >= 4) return 'rating-good';
    if (rating >= 3) return 'rating-ok';
    return 'rating-bad';
}

function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.textContent = message;
    document.body.appendChild(notification);
    
    setTimeout(() => notification.classList.add('show'), 10);
    setTimeout(() => {
        notification.classList.remove('show');
        setTimeout(() => notification.remove(), 300);
    }, 3000);
}

