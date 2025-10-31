// Dual RAG LLM System - Interactive UI
// Author: Adrian Johnson <adrian207@gmail.com>

const API_BASE = window.location.origin;
let currentEventSource = null;
let currentMessageElement = null;

// DOM Elements
const messagesContainer = document.getElementById('messages');
const questionInput = document.getElementById('questionInput');
const sendBtn = document.getElementById('sendBtn');
const fileExtSelect = document.getElementById('fileExt');
const useWebSearchCheckbox = document.getElementById('useWebSearch');
const useGitHubCheckbox = document.getElementById('useGitHub');
const compareModelsCheckbox = document.getElementById('compareModels');
const modelSelect = document.getElementById('modelSelect');
const githubRepoInput = document.getElementById('githubRepo');
const githubRepoContainer = document.getElementById('githubRepoContainer');
const statusIndicator = document.getElementById('status');
const cacheHitRate = document.getElementById('cacheHitRate');
const clearCacheBtn = document.getElementById('clearCacheBtn');
const refreshStatsBtn = document.getElementById('refreshStatsBtn');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    checkHealth();
    fetchStats();
    setupEventListeners();
    setInterval(fetchStats, 30000); // Update stats every 30s
});

function setupEventListeners() {
    sendBtn.addEventListener('click', () => {
        if (compareModelsCheckbox.checked) {
            compareModels();
        } else {
            sendQuery();
        }
    });
    
    questionInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            if (compareModelsCheckbox.checked) {
                compareModels();
            } else {
                sendQuery();
            }
        }
    });
    
    useGitHubCheckbox.addEventListener('change', (e) => {
        githubRepoContainer.style.display = e.target.checked ? 'block' : 'none';
    });
    
    compareModelsCheckbox.addEventListener('change', (e) => {
        if (e.target.checked) {
            sendBtn.textContent = '🔄 Compare Models';
        } else {
            sendBtn.textContent = '📤 Send';
        }
    });
    
    clearCacheBtn.addEventListener('click', clearCache);
    refreshStatsBtn.addEventListener('click', fetchStats);
    
    // Example buttons
    document.querySelectorAll('.example-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            questionInput.value = btn.dataset.question;
            if (btn.dataset.ext) {
                fileExtSelect.value = btn.dataset.ext;
            }
            if (btn.dataset.web) {
                useWebSearchCheckbox.checked = true;
            }
            if (btn.dataset.github) {
                useGitHubCheckbox.checked = true;
                githubRepoContainer.style.display = 'block';
            }
            questionInput.focus();
        });
    });
}

async function checkHealth() {
    try {
        const response = await fetch(`${API_BASE}/health`);
        const data = await response.json();
        
        if (data.status === 'healthy') {
            statusIndicator.textContent = 'Online';
            statusIndicator.classList.add('online');
            statusIndicator.classList.remove('offline');
        } else {
            statusIndicator.textContent = 'Degraded';
            statusIndicator.classList.remove('online');
            statusIndicator.classList.add('offline');
        }
    } catch (error) {
        statusIndicator.textContent = 'Offline';
        statusIndicator.classList.remove('online');
        statusIndicator.classList.add('offline');
    }
}

async function fetchStats() {
    try {
        const response = await fetch(`${API_BASE}/stats`);
        const data = await response.json();
        
        // Update stats display
        cacheHitRate.textContent = `${data.cache.hit_rate}%`;
        document.getElementById('totalQueries').textContent = data.cache.total_requests;
        document.getElementById('cacheHits').textContent = data.cache.hits;
        document.getElementById('webSearches').textContent = data.tools.web_search;
        document.getElementById('githubQueries').textContent = data.tools.github;
        
        // Display model performance if available
        if (data.model_performance && Object.keys(data.model_performance).length > 0) {
            displayModelPerformance(data.model_performance);
        }
    } catch (error) {
        console.error('Failed to fetch stats:', error);
    }
}

function displayModelPerformance(performance) {
    let perfContainer = document.getElementById('modelPerformance');
    if (!perfContainer) {
        const sidebar = document.querySelector('.sidebar');
        if (sidebar) {
            perfContainer = document.createElement('div');
            perfContainer.id = 'modelPerformance';
            perfContainer.style.cssText = 'margin-top: 1rem; padding: 1rem; background: var(--bg-tertiary); border-radius: 0.5rem;';
            perfContainer.innerHTML = '<h4 style="margin-bottom: 0.5rem;">📊 Model Performance</h4><div id="perfList"></div>';
            sidebar.appendChild(perfContainer);
        }
    }
    
    const perfList = document.getElementById('perfList');
    if (perfList) {
        const sortedModels = Object.entries(performance).sort((a, b) => 
            (b[1].avg_tokens_per_sec || 0) - (a[1].avg_tokens_per_sec || 0)
        );
        
        perfList.innerHTML = sortedModels.map(([model, stats]) => `
            <div style="margin: 0.5rem 0; padding: 0.5rem; background: var(--bg-secondary); border-radius: 0.5rem; font-size: 0.85rem;">
                <div style="font-weight: 600; margin-bottom: 0.25rem;">${model.split(':')[0]}</div>
                <div style="color: var(--text-secondary);">
                    ⚡ ${stats.avg_tokens_per_sec.toFixed(1)} tok/s | 
                    ⏱️ ${stats.avg_response_time.toFixed(1)}s | 
                    📊 ${stats.queries} queries
                </div>
            </div>
        `).join('');
    }
}

async function clearCache() {
    if (!confirm('Are you sure you want to clear the cache?')) {
        return;
    }
    
    try {
        const response = await fetch(`${API_BASE}/cache/clear`, { method: 'POST' });
        const data = await response.json();
        
        if (data.status === 'success') {
            showNotification('Cache cleared successfully', 'success');
            fetchStats();
        }
    } catch (error) {
        showNotification('Failed to clear cache', 'error');
    }
}

function sendQuery() {
    const question = questionInput.value.trim();
    if (!question) return;
    
    // Cancel any existing stream
    if (currentEventSource) {
        currentEventSource.close();
    }
    
    // Disable input
    sendBtn.disabled = true;
    questionInput.disabled = true;
    
    // Add user message
    addUserMessage(question);
    
    // Clear input
    questionInput.value = '';
    
    // Prepare request
    const requestBody = {
        question: question,
        file_ext: fileExtSelect.value,
        use_web_search: useWebSearchCheckbox.checked,
        use_github: useGitHubCheckbox.checked
    };
    
    // Add model override if selected
    if (modelSelect.value) {
        requestBody.model_override = modelSelect.value;
    }
    
    if (useGitHubCheckbox.checked && githubRepoInput.value) {
        requestBody.github_repo = githubRepoInput.value;
    }
    
    // Create assistant message container
    currentMessageElement = createAssistantMessage();
    const contentElement = currentMessageElement.querySelector('.message-text');
    const metaElement = currentMessageElement.querySelector('.message-meta');
    const statusElement = currentMessageElement.querySelector('.message-status');
    
    // Start streaming
    startStreaming(requestBody, contentElement, metaElement, statusElement);
}

function startStreaming(requestBody, contentElement, metaElement, statusElement) {
    // Use EventSource for Server-Sent Events
    const url = `${API_BASE}/query/stream`;
    
    fetch(url, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(requestBody)
    }).then(response => {
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let fullAnswer = '';
        let metadata = {};
        
        function processEvents() {
            reader.read().then(({ done, value }) => {
                if (done) {
                    sendBtn.disabled = false;
                    questionInput.disabled = false;
                    questionInput.focus();
                    fetchStats();
                    return;
                }
                
                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n\n');
                buffer = lines.pop();
                
                lines.forEach(line => {
                    if (line.trim()) {
                        try {
                            const [eventLine, dataLine] = line.split('\n');
                            const event = eventLine.replace('event: ', '');
                            const data = JSON.parse(dataLine.replace('data: ', ''));
                            
                            handleStreamEvent(event, data, contentElement, metaElement, statusElement, (answer) => {
                                fullAnswer = answer;
                                metadata = data;
                            });
                        } catch (error) {
                            console.error('Failed to parse event:', error);
                        }
                    }
                });
                
                processEvents();
            }).catch(error => {
                console.error('Stream error:', error);
                statusElement.textContent = `❌ Error: ${error.message}`;
                sendBtn.disabled = false;
                questionInput.disabled = false;
            });
        }
        
        processEvents();
    }).catch(error => {
        console.error('Request error:', error);
        statusElement.textContent = `❌ Error: ${error.message}`;
        sendBtn.disabled = false;
        questionInput.disabled = false;
    });
}

function handleStreamEvent(event, data, contentElement, metaElement, statusElement, callback) {
    switch (event) {
        case 'status':
            const statusMessages = {
                'starting': '🚀 Starting query...',
                'searching_web': '🌐 Searching the web...',
                'searching_github': '🐙 Searching GitHub...',
                'loading_model': '🧠 Loading AI model...',
                'retrieving_context': '📚 Finding relevant context...',
                'generating': `✨ Generating answer (${data.chunks || 0} docs found)...`
            };
            statusElement.textContent = statusMessages[data.status] || data.status;
            statusElement.style.display = 'block';
            break;
        
        case 'cached':
            statusElement.textContent = '⚡ Retrieved from cache';
            statusElement.style.borderLeftColor = 'var(--success)';
            contentElement.textContent = data.answer;
            formatMessageContent(contentElement);
            updateMetadata(metaElement, data);
            break;
        
        case 'tool':
            addToolBadge(metaElement, data.tool, data.count);
            break;
        
        case 'token':
            statusElement.style.display = 'none';
            contentElement.textContent += data.token;
            // Auto-scroll
            contentElement.scrollTop = contentElement.scrollHeight;
            callback(contentElement.textContent);
            break;
        
        case 'done':
            statusElement.style.display = 'none';
            formatMessageContent(contentElement);
            updateMetadata(metaElement, data);
            
            // Add "try another model" button after response is complete
            if (data.model && !data.cached) {
                const messageElement = contentElement.closest('.message');
                const question = document.querySelector('.message.user:last-of-type .message-text')?.textContent || '';
                addTryAnotherModelButton(messageElement, question, data.model);
            }
            
            callback(contentElement.textContent);
            break;
        
        case 'error':
            statusElement.textContent = `❌ Error: ${data.error}`;
            statusElement.style.borderLeftColor = 'var(--error)';
            
            // Implement automatic fallback if model not available
            if (data.error && data.error.includes('not available') && !data.fallback_attempted) {
                statusElement.textContent += ' - Trying fallback model...';
                const fallbackModel = modelSelect.value;
                if (fallbackModel) {
                    attemptFallback(fallbackModel);
                }
            }
            break;
    }
}

async function attemptFallback(failedModel) {
    // Fallback chain
    const fallbackChain = {
        'qwen2.5-coder:32b-q4_K_M': 'qwen2.5-coder:14b',
        'qwen2.5-coder:14b': 'qwen2.5-coder:7b',
        'deepseek-coder-v2:33b-q4_K_M': 'deepseek-coder-v2:16b',
        'deepseek-coder-v2:16b': 'codellama:13b',
        'codellama:34b': 'codellama:13b',
        'codellama:13b': 'llama3.1:8b',
        'llama3.1:70b': 'llama3.1:8b'
    };
    
    const fallbackModel = fallbackChain[failedModel];
    if (fallbackModel) {
        showNotification(`Model ${failedModel} unavailable. Falling back to ${fallbackModel}...`, 'warning');
        modelSelect.value = fallbackModel;
        
        // Re-trigger the query after a short delay
        setTimeout(() => {
            const lastQuestion = document.querySelector('.message.user:last-of-type .message-text')?.textContent;
            if (lastQuestion) {
                questionInput.value = lastQuestion;
                sendQuery();
            }
        }, 1000);
    } else {
        showNotification(`No fallback available for ${failedModel}. Please select a different model.`, 'error');
    }
}

function addUserMessage(text) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message user';
    messageDiv.innerHTML = `
        <div class="message-avatar">👤</div>
        <div class="message-content">
            <div class="message-text">${escapeHtml(text)}</div>
        </div>
    `;
    messagesContainer.appendChild(messageDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

function createAssistantMessage() {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message assistant';
    messageDiv.innerHTML = `
        <div class="message-avatar">🤖</div>
        <div class="message-content">
            <div class="message-status"></div>
            <div class="message-text"></div>
            <div class="message-meta"></div>
        </div>
    `;
    messagesContainer.appendChild(messageDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
    return messageDiv;
}

function updateMetadata(metaElement, data) {
    const parts = [];
    
    if (data.model) {
        parts.push(`<span>🤖 ${data.model.split(':')[0]}</span>`);
    }
    
    if (data.source) {
        parts.push(`<span>📚 ${data.source}</span>`);
    }
    
    if (data.chunks_retrieved !== undefined) {
        parts.push(`<span>📄 ${data.chunks_retrieved} docs</span>`);
    }
    
    if (data.cached) {
        parts.push(`<span>⚡ Cached</span>`);
    }
    
    if (data.response_time_ms) {
        parts.push(`<span>⏱️ ${data.response_time_ms}ms</span>`);
    }
    
    metaElement.innerHTML = parts.join('');
}

function addToolBadge(metaElement, tool, count) {
    const badge = document.createElement('span');
    badge.className = 'tool-badge';
    
    const icons = {
        'web_search': '🌐',
        'github': '🐙',
        'rag': '📚'
    };
    
    badge.textContent = `${icons[tool] || ''} ${tool} (${count})`;
    
    let toolsContainer = metaElement.querySelector('.message-tools');
    if (!toolsContainer) {
        toolsContainer = document.createElement('div');
        toolsContainer.className = 'message-tools';
        metaElement.appendChild(toolsContainer);
    }
    
    toolsContainer.appendChild(badge);
}

function formatMessageContent(element) {
    let html = element.textContent;
    
    // Format code blocks (```...```)
    html = html.replace(/```(\w+)?\n([\s\S]*?)```/g, (match, lang, code) => {
        return `<pre><code class="language-${lang || 'text'}">${escapeHtml(code.trim())}</code></pre>`;
    });
    
    // Format inline code (`...`)
    html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
    
    // Format bold (**...**)
    html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    
    // Format paragraphs
    html = html.split('\n\n').map(p => `<p>${p}</p>`).join('');
    
    element.innerHTML = html;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function showNotification(message, type = 'info') {
    // Simple notification (can be enhanced with a proper notification library)
    alert(message);
}

// Model Comparison
async function compareModels() {
    const question = questionInput.value.trim();
    if (!question) return;
    
    // Disable input
    sendBtn.disabled = true;
    questionInput.disabled = true;
    
    // Add user message
    addUserMessage(question);
    
    // Clear input
    questionInput.value = '';
    
    // Prepare request
    const requestBody = {
        question: question,
        file_ext: fileExtSelect.value,
        use_web_search: useWebSearchCheckbox.checked,
        use_github: useGitHubCheckbox.checked,
        compare_models: true
    };
    
    if (modelSelect.value) {
        requestBody.model_override = modelSelect.value;
    }
    
    if (useGitHubCheckbox.checked && githubRepoInput.value) {
        requestBody.github_repo = githubRepoInput.value;
    }
    
    // Create comparison container
    const comparisonContainer = document.createElement('div');
    comparisonContainer.className = 'message assistant';
    comparisonContainer.innerHTML = `
        <div class="message-avatar">🤖</div>
        <div class="message-content">
            <div class="message-status">🔄 Comparing models...</div>
            <div class="comparison-container"></div>
        </div>
    `;
    
    messagesContainer.appendChild(comparisonContainer);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
    
    const statusElement = comparisonContainer.querySelector('.message-status');
    const comparisonResults = comparisonContainer.querySelector('.comparison-container');
    
    // Make API call
    try {
        const response = await fetch(`${API_BASE}/query/compare`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestBody)
        });
        
        const data = await response.json();
        statusElement.textContent = `✅ Compared ${data.comparison_count} models`;
        
        // Display results side-by-side
        data.results.forEach((result, index) => {
            const resultDiv = document.createElement('div');
            resultDiv.className = 'comparison-response';
            
            // Highlight fastest response
            if (index === 0 || result.response_time === Math.min(...data.results.map(r => r.response_time || Infinity))) {
                if (!result.error) resultDiv.classList.add('winner');
            }
            
            const perfMetrics = result.performance || {};
            const perfDisplay = perfMetrics.avg_tokens_per_sec ? 
                `<div class="metric">⚡ ${perfMetrics.avg_tokens_per_sec.toFixed(1)} tok/s avg</div>` : '';
            
            resultDiv.innerHTML = `
                <div class="comparison-header">
                    <div class="model-badge">${result.model.split(':')[0]}</div>
                    <div class="performance-metrics">
                        <div class="metric">⏱️ ${result.response_time}s</div>
                        ${perfDisplay}
                        <div class="metric">📄 ${result.chunks_retrieved || 0} docs</div>
                    </div>
                </div>
                <div class="comparison-answer">
                    ${result.error ? `<p class="error">❌ ${result.error}</p>` : `<p>${escapeHtml(result.answer.substring(0, 500))}${result.answer.length > 500 ? '...' : ''}</p>`}
                </div>
                <div class="message-actions">
                    <button class="action-button expand-btn" data-full-answer="${escapeHtml(result.answer || '')}" data-model="${result.model}">
                        📖 View Full Response
                    </button>
                    <button class="action-button use-model-btn" data-model="${result.model}">
                        ✅ Use This Model
                    </button>
                </div>
            `;
            
            comparisonResults.appendChild(resultDiv);
        });
        
        // Add event listeners for expand and use buttons
        comparisonContainer.querySelectorAll('.expand-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const fullAnswer = btn.dataset.fullAnswer;
                const model = btn.dataset.model;
                showFullAnswer(model, fullAnswer);
            });
        });
        
        comparisonContainer.querySelectorAll('.use-model-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                modelSelect.value = btn.dataset.model;
                showNotification(`Model set to: ${btn.dataset.model}`, 'success');
            });
        });
        
    } catch (error) {
        console.error('Comparison error:', error);
        statusElement.textContent = `❌ Error: ${error.message}`;
    } finally {
        sendBtn.disabled = false;
        questionInput.disabled = false;
        questionInput.focus();
        fetchStats();
    }
}

function showFullAnswer(model, answer) {
    const modal = document.createElement('div');
    modal.style.cssText = 'position:fixed;top:0;left:0;right:0;bottom:0;background:rgba(0,0,0,0.8);display:flex;align-items:center;justify-content:center;z-index:1000;padding:2rem;';
    modal.innerHTML = `
        <div style="background:var(--bg-secondary);padding:2rem;border-radius:1rem;max-width:900px;max-height:80vh;overflow-y:auto;">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:1rem;">
                <h3 style="margin:0;">Full Response - ${model}</h3>
                <button style="background:var(--error);color:white;border:none;padding:0.5rem 1rem;border-radius:0.5rem;cursor:pointer;">Close</button>
            </div>
            <pre style="white-space:pre-wrap;word-wrap:break-word;">${escapeHtml(answer)}</pre>
        </div>
    `;
    document.body.appendChild(modal);
    
    modal.querySelector('button').addEventListener('click', () => {
        document.body.removeChild(modal);
    });
    
    modal.addEventListener('click', (e) => {
        if (e.target === modal) {
            document.body.removeChild(modal);
        }
    });
}

// Add "Try Another Model" button to assistant messages
function addTryAnotherModelButton(messageElement, question, currentModel) {
    const actionsDiv = document.createElement('div');
    actionsDiv.className = 'message-actions';
    actionsDiv.innerHTML = `
        <button class="action-button try-another-btn">
            🔄 Try Another Model
        </button>
        <button class="action-button compare-btn">
            📊 Compare Models
        </button>
    `;
    
    messageElement.querySelector('.message-content').appendChild(actionsDiv);
    
    actionsDiv.querySelector('.try-another-btn').addEventListener('click', () => {
        // Set a different model and re-run query
        const models = ['qwen2.5-coder:32b-q4_K_M', 'deepseek-coder-v2:33b-q4_K_M', 'codellama:34b'];
        const otherModels = models.filter(m => m !== currentModel);
        const nextModel = otherModels[0];
        
        modelSelect.value = nextModel;
        questionInput.value = question;
        sendQuery();
    });
    
    actionsDiv.querySelector('.compare-btn').addEventListener('click', () => {
        questionInput.value = question;
        compareModelsCheckbox.checked = true;
        compareModels();
    });
}

