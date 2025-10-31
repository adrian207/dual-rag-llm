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
    sendBtn.addEventListener('click', sendQuery);
    
    questionInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendQuery();
        }
    });
    
    useGitHubCheckbox.addEventListener('change', (e) => {
        githubRepoContainer.style.display = e.target.checked ? 'block' : 'none';
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
    } catch (error) {
        console.error('Failed to fetch stats:', error);
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
            callback(contentElement.textContent);
            break;
        
        case 'error':
            statusElement.textContent = `❌ Error: ${data.error}`;
            statusElement.style.borderLeftColor = 'var(--error)';
            break;
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

