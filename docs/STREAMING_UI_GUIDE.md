# Streaming UI Guide

**Author:** Adrian Johnson <adrian207@gmail.com>

Complete guide to the interactive web interface with real-time streaming.

## Overview

Version 1.2.0 introduces a modern web interface with real-time streaming responses, providing instant feedback as the AI generates answers.

## Features

### 🚀 Real-Time Streaming
- **Token-by-token display** - See responses as they're generated
- **Status updates** - Real-time progress indicators
- **Sub-second latency** - First tokens appear instantly
- **Cancellable** - Stop generation at any time

### 🎨 Modern Interface
- **Dark theme** - Easy on the eyes
- **Responsive design** - Works on desktop, tablet, and mobile
- **Markdown formatting** - Code blocks, bold, inline code
- **Chat-style layout** - Familiar conversation interface

### 🔧 Interactive Tools
- **Web search toggle** - Enable Brave Search with one click
- **GitHub integration** - Search code repositories
- **File type selector** - Route to appropriate model
- **Example queries** - Quick start templates

### 📊 Live Statistics
- **Cache hit rate** - See caching effectiveness
- **Query counts** - Track usage
- **Tool statistics** - Monitor web and GitHub searches
- **System status** - Connection health

## Accessing the Interface

### Local Development

```bash
# Start services
docker compose up -d

# Access UI
http://localhost:8000/ui
```

The UI is served directly from the RAG service at `/ui`.

### Azure Deployment

```bash
# AKS
http://<rag-service-ip>:8000/ui

# VM
http://<vm-ip>:8000/ui
```

## Using the Interface

### 1. Basic Query

1. Type your question in the text area
2. Optionally select a file type (Python, C#, etc.)
3. Click **Send** or press **Enter**
4. Watch the response stream in real-time!

**Example:**
```
Question: How do I create an async function in Python?
File Type: Python (.py)
```

### 2. With Web Search

1. Check the **🌐 Web Search** checkbox
2. Type your question
3. The system will search the web first, then generate an answer

**Use case:** Getting current information
```
Question: Latest Python 3.12 features
☑️ Web Search
```

### 3. With GitHub Integration

1. Check the **🐙 GitHub** checkbox
2. Optionally specify a repository (e.g., `fastapi/fastapi`)
3. Type your question
4. See code examples from GitHub

**Use case:** Finding implementation examples
```
Question: FastAPI middleware example
☑️ GitHub
Repository: fastapi/fastapi
```

### 4. Combined Tools

Use all tools together for comprehensive answers:

```
Question: How to implement Redis caching in FastAPI?
☑️ Web Search
☑️ GitHub  
Repository: fastapi/fastapi
File Type: Python (.py)
```

## UI Components

### Header

**Status Indicator**
- 🟢 **Online**: All systems operational
- 🟡 **Degraded**: Some features unavailable
- 🔴 **Offline**: Cannot connect to backend

**Cache Hit Rate**
- Shows percentage of cached responses
- Target: >75% for good performance

### Main Chat Area

**Message Types:**
- **User messages** (blue) - Your questions
- **Assistant messages** (green) - AI responses
- **Status messages** - Progress updates

**Real-time Updates:**
- "🚀 Starting query..."
- "🌐 Searching the web..."
- "🐙 Searching GitHub..."
- "🧠 Loading AI model..."
- "📚 Finding relevant context..."
- "✨ Generating answer..."

### Tool Controls

**Web Search Checkbox**
- Enables Brave Search API
- Searches for current information
- Adds web results to context

**GitHub Checkbox**
- Enables GitHub code search
- Optional repository targeting
- Finds real code examples

**File Type Selector**
- Routes to appropriate model
- MS technologies → Qwen 2.5 Coder
- OSS technologies → DeepSeek Coder V2

### Sidebar

**Quick Examples**
- Pre-configured example queries
- One-click to populate input
- Learn by example

**System Info**
- Total queries processed
- Cache hits count
- Web searches count
- GitHub queries count

**Actions**
- **Clear Cache**: Remove all cached responses
- **Refresh Stats**: Update statistics

## Streaming Events

The UI uses Server-Sent Events (SSE) for real-time streaming:

### Event Types

| Event | Description | UI Action |
|-------|-------------|-----------|
| `status` | Processing stage | Show progress message |
| `cached` | Cache hit | Display instantly |
| `tool` | Tool result | Show tool badge |
| `token` | LLM output token | Append to message |
| `done` | Complete | Show metadata |
| `error` | Failure | Show error message |

### Example Event Flow

```
1. status: "starting"              → "🚀 Starting query..."
2. status: "searching_web"         → "🌐 Searching the web..."
3. tool: {tool: "web_search", count: 5}  → Badge: "🌐 web_search (5)"
4. status: "loading_model"         → "🧠 Loading AI model..."
5. status: "retrieving_context"    → "📚 Finding relevant context..."
6. status: "generating"            → "✨ Generating answer..."
7. token: {token: "To"}            → "To"
8. token: {token: " create"}       → "To create"
9. token: {token: " an"}           → "To create an"
... (continues token by token)
10. done: {model, source, chunks}  → Show metadata
```

## Performance

### Streaming Benefits

| Metric | Non-Streaming | Streaming | Improvement |
|--------|---------------|-----------|-------------|
| Time to first byte | ~3-5s | <500ms | **6-10x faster** |
| Perceived latency | ~10s | <1s | **10x better** |
| User experience | Wait and see | Interactive | **Significantly better** |

### Response Times

**Cached queries**: <100ms total
**Uncached queries**: First tokens in <500ms, complete in 5-10s
**With tools**: +2-3s for web/GitHub search

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Enter` | Send message |
| `Shift+Enter` | New line in input |
| `Ctrl/Cmd+K` | Focus input (planned) |

## Customization

### Changing Colors

Edit `ui/styles.css`:

```css
:root {
    --primary: #3b82f6;      /* Main brand color */
    --bg-primary: #0f172a;   /* Background */
    --text-primary: #f8fafc; /* Text color */
}
```

### Adding Custom Examples

Edit `ui/index.html`:

```html
<button class="example-btn" 
        data-question="Your question here" 
        data-ext=".py"
        data-web="true">
    Button Text
</button>
```

### Modifying Layout

The UI is responsive and uses CSS Grid:
- Desktop: Chat + Sidebar
- Tablet: Chat only (sidebar hidden)
- Mobile: Single column

## Troubleshooting

### UI Not Loading

**Check RAG service:**
```bash
docker logs rag-service

# Should see:
# "ui_directory_not_found" warning (normal if UI not mounted)
# or successful UI mount
```

**Verify files:**
```bash
ls -la ui/
# Should show: index.html, styles.css, app.js
```

**Check volume mount:**
```bash
docker inspect rag-service | grep ui
# Should show volume mount
```

### Streaming Not Working

**Check browser console:**
```javascript
// Should see EventSource connection
// Look for errors
```

**Verify endpoint:**
```bash
curl -X POST http://localhost:8000/query/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "test", "file_ext": ".py"}'

# Should return event stream
```

### Styling Issues

**Clear browser cache:**
- Hard refresh: `Ctrl+Shift+R` (Windows/Linux) or `Cmd+Shift+R` (Mac)
- Clear cache in browser settings

**Check CSS loading:**
```bash
curl http://localhost:8000/ui/styles.css
# Should return CSS content
```

### CORS Errors

CORS is enabled for all origins in development. For production:

```python
# In rag_dual.py, change:
allow_origins=["*"]
# To:
allow_origins=["https://yourdomain.com"]
```

## Mobile Support

The UI is fully responsive:

### Desktop (>1024px)
- Full sidebar visible
- Wide chat area
- All features accessible

### Tablet (768px-1024px)
- Sidebar hidden
- Full-width chat
- Compact controls

### Mobile (<768px)
- Vertical layout
- Stacked controls
- Touch-optimized

## Security Considerations

### Production Deployment

**1. Enable authentication**
```python
# Add authentication middleware
# Verify user tokens
# Rate limit per user
```

**2. Restrict CORS**
```python
allow_origins=["https://yourdomain.com"]
```

**3. Add HTTPS**
```bash
# Use reverse proxy (nginx, Caddy)
# Enable SSL certificates
# Redirect HTTP to HTTPS
```

**4. Input validation**
- Already implemented in backend
- 2000 character max
- XSS protection via escapeHtml()

## API Reference

### Streaming Endpoint

```http
POST /query/stream
Content-Type: application/json

{
  "question": "string",
  "file_ext": "string",
  "use_web_search": boolean,
  "use_github": boolean,
  "github_repo": "string" (optional)
}
```

**Response:** Server-Sent Events stream

```
event: status
data: {"status": "starting", "model": "..."}

event: token
data: {"token": "Hello"}

event: done
data: {"answer": "...", "model": "...", ...}
```

### Non-Streaming Endpoint

```http
POST /query
Content-Type: application/json

{
  "question": "string",
  "file_ext": "string",
  "use_web_search": boolean,
  "use_github": boolean
}
```

**Response:** Complete JSON response

## Future Enhancements

- [ ] Voice input (speech-to-text)
- [ ] Dark/light theme toggle
- [ ] Export conversations
- [ ] Share conversations via link
- [ ] Collaborative sessions
- [ ] Custom color themes
- [ ] Keyboard shortcut customization
- [ ] Mobile app (PWA)

## Accessibility

### Current Features
- ✅ Semantic HTML
- ✅ Keyboard navigation
- ✅ High contrast text
- ✅ Responsive design

### Planned Improvements
- [ ] Screen reader optimization
- [ ] ARIA labels
- [ ] Focus indicators
- [ ] Reduced motion support

---

For questions: adrian207@gmail.com

