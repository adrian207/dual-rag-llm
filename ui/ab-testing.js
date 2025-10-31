// A/B Testing Dashboard
// Author: Adrian Johnson <adrian207@gmail.com>

const API_BASE = window.location.origin;

// DOM Elements
const createTestForm = document.getElementById('createTestForm');
const testsList = document.getElementById('testsList');
const testDetailsPanel = document.getElementById('testDetailsPanel');
const testDetailsContent = document.getElementById('testDetailsContent');
const closeDetailsBtn = document.getElementById('closeDetails');
const refreshTestsBtn = document.getElementById('refreshTests');
const trafficSplitInput = document.getElementById('trafficSplit');
const trafficSplitValue = document.getElementById('trafficSplitValue');

// State
let activeTests = [];
let selectedTestId = null;

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    setupEventListeners();
    loadTests();
    
    // Auto-refresh every 30 seconds
    setInterval(loadTests, 30000);
});

function setupEventListeners() {
    createTestForm.addEventListener('submit', handleCreateTest);
    closeDetailsBtn.addEventListener('click', closeTestDetails);
    refreshTestsBtn.addEventListener('click', loadTests);
    
    // Traffic split slider
    trafficSplitInput.addEventListener('input', (e) => {
        trafficSplitValue.textContent = `${e.target.value}%`;
    });
}

// Create Test
async function handleCreateTest(e) {
    e.preventDefault();
    
    const formData = new FormData(createTestForm);
    const data = {
        name: document.getElementById('testName').value,
        description: document.getElementById('testDescription').value,
        model_a: document.getElementById('modelA').value,
        model_b: document.getElementById('modelB').value,
        traffic_split: parseFloat(trafficSplitInput.value) / 100,
        min_samples: parseInt(document.getElementById('minSamples').value),
        confidence_level: parseFloat(document.getElementById('confidenceLevel').value)
    };
    
    // Validate
    if (data.model_a === data.model_b) {
        alert('Model A and Model B must be different');
        return;
    }
    
    try {
        const response = await fetch(`${API_BASE}/ab-tests?${new URLSearchParams(data)}`, {
            method: 'POST'
        });
        
        if (!response.ok) {
            throw new Error('Failed to create test');
        }
        
        const test = await response.json();
        showNotification(`✅ Test created: ${test.name}`, 'success');
        
        // Reset form
        createTestForm.reset();
        trafficSplitValue.textContent = '50%';
        
        // Reload tests
        await loadTests();
        
        // Show details of new test
        showTestDetails(test.test_id);
        
    } catch (error) {
        console.error('Error creating test:', error);
        showNotification(`❌ Error: ${error.message}`, 'error');
    }
}

// Load Tests
async function loadTests() {
    try {
        const response = await fetch(`${API_BASE}/ab-tests`);
        const data = await response.json();
        
        activeTests = data.tests || [];
        renderTestsList();
        
        // Refresh details if a test is selected
        if (selectedTestId) {
            await loadTestDetails(selectedTestId);
        }
        
    } catch (error) {
        console.error('Error loading tests:', error);
        testsList.innerHTML = '<p class="error">Failed to load tests</p>';
    }
}

// Render Tests List
function renderTestsList() {
    if (activeTests.length === 0) {
        testsList.innerHTML = '<p class="empty-state">No tests yet. Create one above!</p>';
        return;
    }
    
    const groupedTests = {
        running: activeTests.filter(t => t.status === 'running'),
        draft: activeTests.filter(t => t.status === 'draft'),
        paused: activeTests.filter(t => t.status === 'paused'),
        completed: activeTests.filter(t => t.status === 'completed'),
        cancelled: activeTests.filter(t => t.status === 'cancelled')
    };
    
    let html = '';
    
    // Running tests (priority)
    if (groupedTests.running.length > 0) {
        html += '<h3>🟢 Running Tests</h3>';
        groupedTests.running.forEach(test => {
            html += renderTestCard(test);
        });
    }
    
    // Draft tests
    if (groupedTests.draft.length > 0) {
        html += '<h3>📝 Draft Tests</h3>';
        groupedTests.draft.forEach(test => {
            html += renderTestCard(test);
        });
    }
    
    // Paused tests
    if (groupedTests.paused.length > 0) {
        html += '<h3>⏸️ Paused Tests</h3>';
        groupedTests.paused.forEach(test => {
            html += renderTestCard(test);
        });
    }
    
    // Completed tests
    if (groupedTests.completed.length > 0) {
        html += '<h3>✅ Completed Tests</h3>';
        groupedTests.completed.forEach(test => {
            html += renderTestCard(test);
        });
    }
    
    testsList.innerHTML = html;
}

function renderTestCard(test) {
    const statusIcons = {
        draft: '📝',
        running: '🟢',
        paused: '⏸️',
        completed: '✅',
        cancelled: '🚫'
    };
    
    const statusColors = {
        draft: 'var(--text-secondary)',
        running: 'var(--success)',
        paused: 'var(--warning)',
        completed: 'var(--primary)',
        cancelled: 'var(--error)'
    };
    
    return `
        <div class="test-card" data-test-id="${test.test_id}">
            <div class="test-card-header">
                <div>
                    <h4>${test.name}</h4>
                    <p class="test-description">${test.description || 'No description'}</p>
                </div>
                <span class="status-badge" style="color: ${statusColors[test.status]}">
                    ${statusIcons[test.status]} ${test.status.toUpperCase()}
                </span>
            </div>
            
            <div class="test-card-models">
                <div class="model-info">
                    <span class="model-label">Model A</span>
                    <span class="model-name">${formatModelName(test.model_a)}</span>
                </div>
                <span class="vs-separator">VS</span>
                <div class="model-info">
                    <span class="model-label">Model B</span>
                    <span class="model-name">${formatModelName(test.model_b)}</span>
                </div>
            </div>
            
            <div class="test-card-meta">
                <span>Split: ${Math.round(test.traffic_split * 100)}% / ${Math.round((1 - test.traffic_split) * 100)}%</span>
                <span>Min Samples: ${test.min_samples}</span>
                <span>Confidence: ${Math.round(test.confidence_level * 100)}%</span>
            </div>
            
            <div class="test-card-actions">
                <button onclick="showTestDetails('${test.test_id}')" class="btn-primary btn-sm">
                    📊 View Details
                </button>
                ${renderActionButtons(test)}
            </div>
        </div>
    `;
}

function renderActionButtons(test) {
    let buttons = '';
    
    switch (test.status) {
        case 'draft':
            buttons = `
                <button onclick="startTest('${test.test_id}')" class="btn-success btn-sm">
                    ▶️ Start
                </button>
                <button onclick="deleteTest('${test.test_id}')" class="btn-danger btn-sm">
                    🗑️ Delete
                </button>
            `;
            break;
        case 'running':
            buttons = `
                <button onclick="pauseTest('${test.test_id}')" class="btn-warning btn-sm">
                    ⏸️ Pause
                </button>
                <button onclick="completeTest('${test.test_id}')" class="btn-primary btn-sm">
                    ✅ Complete
                </button>
            `;
            break;
        case 'paused':
            buttons = `
                <button onclick="resumeTest('${test.test_id}')" class="btn-success btn-sm">
                    ▶️ Resume
                </button>
                <button onclick="completeTest('${test.test_id}')" class="btn-primary btn-sm">
                    ✅ Complete
                </button>
            `;
            break;
        case 'completed':
            buttons = `
                <button onclick="deleteTest('${test.test_id}')" class="btn-danger btn-sm">
                    🗑️ Delete
                </button>
            `;
            break;
    }
    
    return buttons;
}

// Test Details
async function showTestDetails(testId) {
    selectedTestId = testId;
    await loadTestDetails(testId);
    testDetailsPanel.style.display = 'block';
    testDetailsPanel.scrollIntoView({ behavior: 'smooth' });
}

async function loadTestDetails(testId) {
    try {
        // Load test config
        const testResponse = await fetch(`${API_BASE}/ab-tests/${testId}`);
        const test = await testResponse.json();
        
        // Load statistics if available
        let stats = null;
        try {
            const statsResponse = await fetch(`${API_BASE}/ab-tests/${testId}/statistics`);
            if (statsResponse.ok) {
                stats = await statsResponse.json();
            }
        } catch (e) {
            // No stats yet
        }
        
        // Load results
        const resultsResponse = await fetch(`${API_BASE}/ab-tests/${testId}/results`);
        const resultsData = await resultsResponse.json();
        const results = resultsData.results || [];
        
        renderTestDetails(test, stats, results);
        
    } catch (error) {
        console.error('Error loading test details:', error);
        testDetailsContent.innerHTML = '<p class="error">Failed to load test details</p>';
    }
}

function renderTestDetails(test, stats, results) {
    let html = `
        <div class="test-details">
            <div class="test-info">
                <h3>${test.name}</h3>
                <p>${test.description || 'No description'}</p>
                <div class="test-metadata">
                    <div class="meta-item">
                        <span class="meta-label">Status:</span>
                        <span class="meta-value">${test.status.toUpperCase()}</span>
                    </div>
                    <div class="meta-item">
                        <span class="meta-label">Created:</span>
                        <span class="meta-value">${formatDate(test.created_at)}</span>
                    </div>
                    ${test.started_at ? `
                        <div class="meta-item">
                            <span class="meta-label">Started:</span>
                            <span class="meta-value">${formatDate(test.started_at)}</span>
                        </div>
                    ` : ''}
                    ${test.ended_at ? `
                        <div class="meta-item">
                            <span class="meta-label">Ended:</span>
                            <span class="meta-value">${formatDate(test.ended_at)}</span>
                        </div>
                    ` : ''}
                </div>
            </div>
            
            <div class="models-comparison">
                <div class="model-column">
                    <h4>Model A</h4>
                    <p class="model-name-full">${test.model_a}</p>
                    <div class="traffic-info">
                        <span class="traffic-label">Traffic:</span>
                        <span class="traffic-value">${Math.round(test.traffic_split * 100)}%</span>
                    </div>
                </div>
                
                <div class="vs-divider">VS</div>
                
                <div class="model-column">
                    <h4>Model B</h4>
                    <p class="model-name-full">${test.model_b}</p>
                    <div class="traffic-info">
                        <span class="traffic-label">Traffic:</span>
                        <span class="traffic-value">${Math.round((1 - test.traffic_split) * 100)}%</span>
                    </div>
                </div>
            </div>
    `;
    
    // Statistics
    if (stats) {
        html += renderStatistics(test, stats);
    } else if (results.length > 0) {
        html += '<div class="stats-placeholder">📊 Collecting data... (need more samples for statistics)</div>';
    } else {
        html += '<div class="stats-placeholder">⏳ No data yet. Start the test and run queries!</div>';
    }
    
    // Results Table
    if (results.length > 0) {
        html += renderResultsTable(test, results);
    }
    
    html += '</div>';
    testDetailsContent.innerHTML = html;
}

function renderStatistics(test, stats) {
    const aStats = stats.model_a_stats;
    const bStats = stats.model_b_stats;
    
    const winner = stats.winner;
    const aIsWinner = winner === test.model_a;
    const bIsWinner = winner === test.model_b;
    
    return `
        <div class="statistics">
            <h4>📊 Statistical Analysis</h4>
            
            ${stats.recommendation ? `
                <div class="recommendation ${winner ? 'has-winner' : ''}">
                    ${stats.recommendation}
                </div>
            ` : ''}
            
            <div class="stats-grid">
                <div class="stat-card ${aIsWinner ? 'winner' : ''}">
                    <h5>${formatModelName(test.model_a)} ${aIsWinner ? '🏆' : ''}</h5>
                    <div class="stat-row">
                        <span class="stat-label">Sample Size:</span>
                        <span class="stat-value">${aStats.count}</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">Response Time:</span>
                        <span class="stat-value">${aStats.response_time.mean.toFixed(2)}s ± ${aStats.response_time.std.toFixed(2)}</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">Tokens/Sec:</span>
                        <span class="stat-value">${aStats.tokens_per_sec.mean.toFixed(1)} ± ${aStats.tokens_per_sec.std.toFixed(1)}</span>
                    </div>
                    ${aStats.user_rating.count > 0 ? `
                        <div class="stat-row">
                            <span class="stat-label">User Rating:</span>
                            <span class="stat-value">${renderStars(aStats.user_rating.mean)} (${aStats.user_rating.count} ratings)</span>
                        </div>
                    ` : ''}
                </div>
                
                <div class="stat-card ${bIsWinner ? 'winner' : ''}">
                    <h5>${formatModelName(test.model_b)} ${bIsWinner ? '🏆' : ''}</h5>
                    <div class="stat-row">
                        <span class="stat-label">Sample Size:</span>
                        <span class="stat-value">${bStats.count}</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">Response Time:</span>
                        <span class="stat-value">${bStats.response_time.mean.toFixed(2)}s ± ${bStats.response_time.std.toFixed(2)}</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">Tokens/Sec:</span>
                        <span class="stat-value">${bStats.tokens_per_sec.mean.toFixed(1)} ± ${bStats.tokens_per_sec.std.toFixed(1)}</span>
                    </div>
                    ${bStats.user_rating.count > 0 ? `
                        <div class="stat-row">
                            <span class="stat-label">User Rating:</span>
                            <span class="stat-value">${renderStars(bStats.user_rating.mean)} (${bStats.user_rating.count} ratings)</span>
                        </div>
                    ` : ''}
                </div>
            </div>
            
            <div class="significance-info">
                <h5>Statistical Significance (${Math.round(test.confidence_level * 100)}% confidence)</h5>
                <div class="significance-row">
                    <span>Response Time:</span>
                    <span class="${stats.statistical_significance.response_time ? 'significant' : 'not-significant'}">
                        ${stats.statistical_significance.response_time ? '✅ Significant' : '❌ Not Significant'}
                    </span>
                </div>
                <div class="significance-row">
                    <span>Tokens/Sec:</span>
                    <span class="${stats.statistical_significance.tokens_per_sec ? 'significant' : 'not-significant'}">
                        ${stats.statistical_significance.tokens_per_sec ? '✅ Significant' : '❌ Not Significant'}
                    </span>
                </div>
                ${stats.statistical_significance.user_rating !== undefined ? `
                    <div class="significance-row">
                        <span>User Rating:</span>
                        <span class="${stats.statistical_significance.user_rating ? 'significant' : 'not-significant'}">
                            ${stats.statistical_significance.user_rating ? '✅ Significant' : '❌ Not Significant'}
                        </span>
                    </div>
                ` : ''}
            </div>
        </div>
    `;
}

function renderResultsTable(test, results) {
    return `
        <div class="results-table">
            <h4>📝 Recent Results (${results.length} total)</h4>
            <table>
                <thead>
                    <tr>
                        <th>Time</th>
                        <th>Model</th>
                        <th>Question</th>
                        <th>Response Time</th>
                        <th>Tokens/Sec</th>
                        <th>Rating</th>
                    </tr>
                </thead>
                <tbody>
                    ${results.slice(-20).reverse().map(r => `
                        <tr>
                            <td>${formatTime(r.timestamp)}</td>
                            <td>${formatModelName(r.model)}</td>
                            <td class="question-cell" title="${escapeHtml(r.question)}">
                                ${escapeHtml(r.question.substring(0, 50))}${r.question.length > 50 ? '...' : ''}
                            </td>
                            <td>${r.response_time.toFixed(2)}s</td>
                            <td>${r.tokens_per_sec.toFixed(1)}</td>
                            <td>
                                ${r.user_rating ? renderStars(r.user_rating) : 
                                  `<button onclick="rateResponse('${test.test_id}', '${r.query_id}')" class="btn-rate">Rate</button>`}
                            </td>
                        </tr>
                    `).join('')}
                </tbody>
            </table>
        </div>
    `;
}

// Test Actions
async function startTest(testId) {
    if (!confirm('Start this test? It will begin routing traffic to both models.')) return;
    
    try {
        await fetch(`${API_BASE}/ab-tests/${testId}/start`, { method: 'POST' });
        showNotification('✅ Test started', 'success');
        await loadTests();
    } catch (error) {
        showNotification('❌ Failed to start test', 'error');
    }
}

async function pauseTest(testId) {
    try {
        await fetch(`${API_BASE}/ab-tests/${testId}/pause`, { method: 'POST' });
        showNotification('⏸️ Test paused', 'success');
        await loadTests();
    } catch (error) {
        showNotification('❌ Failed to pause test', 'error');
    }
}

async function resumeTest(testId) {
    try {
        await fetch(`${API_BASE}/ab-tests/${testId}/resume`, { method: 'POST' });
        showNotification('▶️ Test resumed', 'success');
        await loadTests();
    } catch (error) {
        showNotification('❌ Failed to resume test', 'error');
    }
}

async function completeTest(testId) {
    if (!confirm('Complete this test and declare winner?')) return;
    
    try {
        const response = await fetch(`${API_BASE}/ab-tests/${testId}/complete`, { method: 'POST' });
        const data = await response.json();
        
        if (data.statistics && data.statistics.winner) {
            showNotification(`🏆 Winner: ${formatModelName(data.statistics.winner)}`, 'success');
        } else {
            showNotification('✅ Test completed', 'success');
        }
        
        await loadTests();
        if (selectedTestId === testId) {
            await loadTestDetails(testId);
        }
    } catch (error) {
        showNotification('❌ Failed to complete test', 'error');
    }
}

async function deleteTest(testId) {
    if (!confirm('Delete this test? This cannot be undone.')) return;
    
    try {
        await fetch(`${API_BASE}/ab-tests/${testId}`, { method: 'DELETE' });
        showNotification('🗑️ Test deleted', 'success');
        
        if (selectedTestId === testId) {
            closeTestDetails();
        }
        
        await loadTests();
    } catch (error) {
        showNotification('❌ Failed to delete test', 'error');
    }
}

async function rateResponse(testId, queryId) {
    const rating = prompt('Rate this response (1-5 stars):', '5');
    if (!rating) return;
    
    const ratingNum = parseInt(rating);
    if (isNaN(ratingNum) || ratingNum < 1 || ratingNum > 5) {
        alert('Please enter a number between 1 and 5');
        return;
    }
    
    try {
        await fetch(`${API_BASE}/ab-tests/${testId}/rate?query_id=${queryId}&rating=${ratingNum}`, {
            method: 'POST'
        });
        showNotification(`⭐ Rated ${ratingNum} stars`, 'success');
        await loadTestDetails(testId);
    } catch (error) {
        showNotification('❌ Failed to rate response', 'error');
    }
}

function closeTestDetails() {
    testDetailsPanel.style.display = 'none';
    selectedTestId = null;
}

// Utility Functions
function formatModelName(model) {
    return model.split(':')[0].replace(/-/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
}

function formatDate(dateStr) {
    const date = new Date(dateStr);
    return date.toLocaleString();
}

function formatTime(dateStr) {
    const date = new Date(dateStr);
    return date.toLocaleTimeString();
}

function renderStars(rating) {
    const fullStars = Math.floor(rating);
    const halfStar = rating % 1 >= 0.5;
    const emptyStars = 5 - fullStars - (halfStar ? 1 : 0);
    
    return '⭐'.repeat(fullStars) + 
           (halfStar ? '½' : '') + 
           '☆'.repeat(emptyStars) + 
           ` ${rating.toFixed(1)}`;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
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

