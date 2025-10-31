# Dual RAG LLM System - User Guide

**Version:** 1.11.0  
**Author:** Adrian Johnson <adrian207@gmail.com>  
**Last Updated:** October 31, 2024

Welcome to the Dual RAG LLM System! This guide will help you get started and make the most of all features.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
3. [Core Features](#core-features)
4. [Advanced Features](#advanced-features)
5. [Web Interface Guide](#web-interface-guide)
6. [API Usage](#api-usage)
7. [Configuration](#configuration)
8. [Troubleshooting](#troubleshooting)
9. [Best Practices](#best-practices)
10. [FAQ](#faq)

---

## Introduction

### What is Dual RAG LLM?

The Dual RAG LLM System is an intelligent AI assistant that combines:
- **Dual Knowledge Bases**: Separate stores for Microsoft Technologies and General Software Development
- **Large Language Models (LLMs)**: Multiple AI models through Ollama
- **Advanced Features**: Caching, web search, code highlighting, and more

### Key Benefits

✅ **Accurate Answers**: Retrieves relevant context from curated knowledge bases  
✅ **Fast Responses**: Redis caching for instant results  
✅ **Web-Enhanced**: Access current information through web search  
✅ **Code-Aware**: Syntax highlighting for 22 programming languages  
✅ **Quality Assured**: Built-in validation and fact-checking  
✅ **Flexible**: Choose from multiple AI models  

---

## Getting Started

### Quick Start (5 Minutes)

#### Prerequisites
- Docker and Docker Compose installed
- NVIDIA GPU (optional but recommended)
- 16GB RAM minimum
- 50GB disk space

#### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/adrian207/dual-rag-llm.git
cd dual-rag-llm
```

2. **Create environment file:**
```bash
cp .env.example .env
```

3. **Edit `.env` with your API keys (optional):**
```bash
BRAVE_API_KEY=your_brave_api_key  # For web search
GITHUB_TOKEN=your_github_token    # For GitHub integration
```

4. **Start the system:**
```bash
docker-compose up -d
```

5. **Wait for services to start** (2-3 minutes):
```bash
docker-compose logs -f
```

6. **Open your browser:**
```
http://localhost
```

### First Query

1. Navigate to `http://localhost`
2. Type a question: "How do I create an async function in Python?"
3. Click **Send** or press **Enter**
4. Watch the response stream in real-time!

---

## Core Features

### 1. Dual Knowledge Bases

The system uses two specialized knowledge bases:

#### **MS Store** - Microsoft Technologies
- PowerShell
- C# / .NET
- Azure
- Visual Studio
- Windows Server

#### **General Store** - Software Development
- Python
- JavaScript/TypeScript
- Docker
- Git
- Web frameworks

**How it works**: Your query searches both stores automatically, combining relevant context for comprehensive answers.

### 2. Smart Caching

**Redis-powered caching** makes repeated queries instant:

- **Cache Duration**: 1 hour
- **Cache Key**: Hash of query + file extension + tools used
- **Benefits**: 10-100x faster for common questions

**Example:**
- First query: 2.5 seconds
- Cached query: 0.05 seconds (50ms!)

**Clear cache** if you need fresh data:
```bash
curl -X POST http://localhost:8000/cache/clear
```

### 3. External Tools

#### Web Search (Brave)
- Search the internet for current information
- Useful for: Latest news, recent updates, trending topics
- **Enable**: Check "🌐 Web Search" in UI

**Example queries:**
- "What are the latest Python 3.12 features?"
- "Current best practices for React hooks"

#### GitHub Integration
- Search GitHub repositories
- Find code examples
- Check documentation
- **Enable**: Check "🐙 GitHub" in UI
- **Optional**: Specify repo like `fastapi/fastapi`

**Example queries:**
- "How does FastAPI handle websockets?"
- "Show me authentication examples from fastapi/fastapi"

### 4. Model Selection

Choose from **10+ AI models**:

| Category | Model | Best For | Speed |
|----------|-------|----------|-------|
| **Microsoft** | Qwen 2.5 Coder 32B | Complex code | Medium |
| **Microsoft** | Qwen 2.5 Coder 14B | General coding | Fast |
| **Microsoft** | Qwen 2.5 Coder 7B | Simple tasks | Fastest |
| **Open Source** | DeepSeek Coder V2 33B | Code generation | Medium |
| **Open Source** | CodeLlama 34B | Code understanding | Medium |
| **General** | Llama 3.1 70B | Complex reasoning | Slow |
| **General** | Llama 3.1 8B | General questions | Fastest |

**How to choose:**
1. **Auto** (default): System selects best model for your query
2. **Manual**: Select from dropdown
3. **Compare**: Check "Compare Models" to see multiple responses side-by-side

---

## Advanced Features

### 1. Code Syntax Highlighting

Automatic highlighting for **22 languages**:

Python • JavaScript • TypeScript • Java • C# • C • C++ • Go • Rust • PHP • Ruby • Swift • Kotlin • **PowerShell** • Bash • SQL • HTML • CSS • JSON • YAML • XML • Markdown

**Features:**
- Auto language detection
- 9 color themes (VS Dark default)
- Line numbers
- Copy button on hover

**Change theme**: Use sidebar dropdown

### 2. Model Comparison

Compare responses from multiple models:

1. Check "🔄 Compare Models"
2. Ask your question
3. See responses side-by-side
4. Choose the best answer

**Use cases:**
- Verify answer accuracy
- Find different approaches
- Evaluate model performance

### 3. Answer Validation

Every response is automatically validated:

✅ **Factuality**: Are claims accurate?  
✅ **Sources**: Backed by context?  
✅ **Consistency**: No contradictions?  
✅ **Completeness**: Query fully answered?  
✅ **Relevance**: On-topic?  
✅ **Clarity**: Easy to understand?  
✅ **Code Quality**: Syntax and best practices?  

**Quality Score**: 0-1 (higher is better)
- 0.8+: Excellent
- 0.6-0.8: Good
- 0.4-0.6: Acceptable
- <0.4: Review needed

### 4. Hallucination Detection

The system detects potential AI hallucinations:

🚫 **False Confidence**: "definitely", "always", "never"  
🚫 **Fabricated Sources**: Claims without citations  
🚫 **Inconsistent Details**: Numbers not in context  
🚫 **Unsupported Claims**: Specific dates without source  
🚫 **Contradictions**: Internal conflicts  

**Hallucination Risk**: 0-1 score
- 0-0.3: Low risk ✅
- 0.3-0.7: Medium risk ⚠️
- 0.7-1.0: High risk ❌

### 5. Response Formatting

Responses are automatically formatted:

- **Section Headers**: Auto-added for long responses
- **Lists**: Numbered/bulleted where appropriate
- **Code Blocks**: Properly formatted with language hints
- **Readability**: Line breaks at sentence boundaries
- **Emoji Headers** (optional): Visual enhancement

**Formatting Styles:**
- Plain
- Markdown (default)
- Structured
- Professional
- Concise
- Detailed

### 6. Model Ensembles

Combine multiple models for better answers:

#### **Voting Strategy**
Democratic vote - majority wins

**Use for**: General questions, fact verification

#### **Averaging Strategy**
Weighted average of scores

**Use for**: Numerical answers, rankings

#### **Cascade Strategy**
Sequential refinement - each model improves the last

**Use for**: Complex tasks, multi-step problems

#### **Best-of-N Strategy**
Run N models, pick highest confidence

**Use for**: Critical decisions, high-stakes queries

#### **Specialist Strategy**
Route to domain expert

**Use for**: Technical questions with clear domains

#### **Consensus Strategy**
Require agreement above threshold

**Use for**: Safety-critical answers

### 7. A/B Testing

Compare model performance scientifically:

1. **Create Test**: Define two models to compare
2. **Set Traffic**: Split percentage (e.g., 50/50)
3. **Collect Data**: Response time, user ratings
4. **Analyze**: Statistical significance (t-test)
5. **Choose Winner**: Automatic or manual

**Metrics:**
- Response time
- Token count
- Success rate
- User ratings (1-5 stars)
- Statistical significance (p-value)

---

## Web Interface Guide

### Layout

```
┌─────────────────────────────────────────────────────┐
│ Header: Title + Stats (Cache Hit Rate, Status)     │
├─────────────────────────────────────────┬───────────┤
│                                         │           │
│ Chat Area                               │ Sidebar   │
│ - Messages                              │ - Examples│
│ - Input Box                             │ - Stats   │
│ - Tools                                 │ - Theme   │
│ - Model Selector                        │ - Actions │
│                                         │           │
└─────────────────────────────────────────┴───────────┘
```

### Input Options

**Top Row (Tools):**
- ☐ **Web Search**: Enable Brave web search
- ☐ **GitHub**: Enable GitHub integration
- ☐ **Compare Models**: Side-by-side comparison
- **File Extension**: Auto-detect or specify (Python, JS, etc.)
- **Model Selector**: Choose AI model or Auto

**Input Box:**
- Type your question
- **Enter**: Send query
- **Shift+Enter**: New line
- **Send Button**: Click to submit

**GitHub Repo** (if enabled):
- Optional: Specify `user/repo` format
- Example: `fastapi/fastapi`

### Quick Examples

Click any example to auto-fill:
- **Python async function**
- **C# 12 features** (with web search)
- **FastAPI middleware** (with GitHub)
- **PowerShell errors**

### System Info

Monitor real-time statistics:
- **Total Queries**: Lifetime count
- **Cache Hits**: How many cached
- **Web Searches**: External searches
- **GitHub Queries**: Repository queries

### Actions

**Clear Cache**: Remove all cached responses
**Refresh Stats**: Update statistics display

---

## API Usage

### Base URL
```
http://localhost:8000
```

### Authentication
None required for local deployment. For production, add authentication middleware.

### Core Endpoints

#### 1. Query (Streaming)
```http
POST /query/stream
Content-Type: application/json

{
  "question": "How do I use async/await in Python?",
  "file_ext": ".py",
  "use_web_search": false,
  "use_github": false,
  "model_override": "",
  "compare_models": false
}
```

**Response**: Server-Sent Events (SSE)
```
event: token
data: "To"

event: token
data: " use"

event: done
data: {"metadata": {...}}
```

#### 2. Query (Non-Streaming)
```http
POST /query
Content-Type: application/json

{
  "question": "Explain Python decorators",
  "file_ext": ".py"
}
```

**Response:**
```json
{
  "answer": "Python decorators are...",
  "sources": ["doc1", "doc2"],
  "metadata": {
    "model": "llama3.1:8b",
    "cached": false,
    "ms_docs_used": 2,
    "general_docs_used": 3
  }
}
```

#### 3. Health Check
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "ollama_available": true,
  "redis_available": true
}
```

#### 4. System Stats
```http
GET /stats
```

**Response:**
```json
{
  "cache": {
    "total_queries": 150,
    "cache_hits": 45,
    "hit_rate": 0.3
  },
  "tools": {
    "web_searches": 12,
    "github_queries": 5
  },
  "models_cached": ["llama3.1:8b", "codellama:7b"]
}
```

### Advanced Endpoints

#### Syntax Highlighting
```http
POST /syntax/highlight
Content-Type: application/json

{
  "code": "def hello(): print('world')",
  "language": "python",
  "theme": "vs-dark"
}
```

#### Factuality Check
```http
POST /factuality/check
Content-Type: application/json

{
  "query_id": "123",
  "query": "What is Python?",
  "answer": "Python is a programming language...",
  "model": "llama3.1:8b",
  "context": "Python documentation..."
}
```

#### Model Performance
```http
GET /models/performance
```

#### A/B Test Creation
```http
POST /ab-tests
Content-Type: application/json

{
  "name": "Llama vs CodeLlama",
  "model_a": "llama3.1:8b",
  "model_b": "codellama:7b",
  "traffic_split": 0.5,
  "min_samples": 50
}
```

---

## Configuration

### Environment Variables

Create `.env` file in project root:

```bash
# LLM Configuration
OLLAMA_API=http://ollama:11434
DEFAULT_MODEL=llama3.1:8b

# Redis Cache
REDIS_HOST=redis
REDIS_PORT=6379
CACHE_TTL=3600  # 1 hour in seconds

# External APIs
BRAVE_API_KEY=your_brave_api_key_here
GITHUB_TOKEN=your_github_token_here

# Feature Flags
ENABLE_WEB_SEARCH=true
ENABLE_GITHUB=true
ENABLE_CACHING=true
```

### Docker Compose

Customize `docker-compose.yaml`:

```yaml
services:
  ollama:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1  # Number of GPUs
              capabilities: [gpu]
    volumes:
      - ollama-data:/root/.ollama
    environment:
      - OLLAMA_NUM_PARALLEL=4  # Concurrent requests
```

### Feature Configuration

All features can be configured via API:

```bash
# Update syntax highlighting config
curl -X PUT http://localhost:8000/syntax/config \
  -H "Content-Type: application/json" \
  -d '{"enabled": true, "default_theme": "monokai"}'

# Update validation config
curl -X PUT http://localhost:8000/validation/config \
  -H "Content-Type: application/json" \
  -d '{"min_confidence_threshold": 0.7}'

# Update factuality config
curl -X PUT http://localhost:8000/factuality/config \
  -H "Content-Type: application/json" \
  -d '{"hallucination_threshold": 0.6}'
```

---

## Troubleshooting

### Common Issues

#### 1. Services won't start

**Symptom**: `docker-compose up` fails

**Solutions**:
```bash
# Check Docker is running
docker ps

# Check logs
docker-compose logs

# Restart services
docker-compose down
docker-compose up -d
```

#### 2. Slow responses

**Possible causes**:
- No GPU available
- Model not downloaded
- Large query

**Solutions**:
```bash
# Check GPU
nvidia-smi

# Pre-download models
docker exec -it ollama ollama pull llama3.1:8b

# Use smaller model
# Select "Llama 3.1 8B" in UI
```

#### 3. Cache not working

**Check Redis**:
```bash
# Test Redis connection
docker exec -it redis redis-cli ping
# Should return: PONG

# Check cache stats
curl http://localhost:8000/stats
```

#### 4. Web search not working

**Check API key**:
```bash
# Verify .env file
cat .env | grep BRAVE_API_KEY

# Test endpoint
curl -X POST http://localhost:8000/tools/web_search \
  -H "Content-Type: application/json" \
  -d '{"query": "test"}'
```

#### 5. UI not loading

**Check services**:
```bash
# Verify all services running
docker-compose ps

# Check nginx logs
docker-compose logs nginx

# Try direct API
curl http://localhost:8000/health
```

### Performance Optimization

#### GPU Utilization
```bash
# Monitor GPU
nvidia-smi -l 1

# Increase parallel requests in docker-compose.yaml
OLLAMA_NUM_PARALLEL=8
```

#### Cache Tuning
```bash
# Increase cache TTL (in .env)
CACHE_TTL=7200  # 2 hours

# Check cache memory
docker exec redis redis-cli INFO memory
```

#### Model Selection
- Use smaller models for simple queries (7B-8B)
- Use larger models for complex tasks (30B+)
- Enable model comparison to find best fit

---

## Best Practices

### Writing Effective Queries

✅ **Be Specific**
```
❌ "Tell me about Python"
✅ "How do I handle exceptions in Python async functions?"
```

✅ **Provide Context**
```
❌ "Fix this code"
✅ "Fix this Python code that's raising a KeyError when accessing dict keys"
```

✅ **Use File Extensions**
```
Select ".py" for Python questions
Select ".cs" for C# questions
```

✅ **Enable Relevant Tools**
```
For current info: Enable Web Search
For code examples: Enable GitHub
```

### Choosing Models

| Query Type | Recommended Model |
|------------|-------------------|
| Quick Python question | Llama 3.1 8B |
| Complex algorithm | Llama 3.1 70B |
| Code generation | CodeLlama 34B |
| Microsoft tech | Qwen 2.5 Coder |
| General question | Auto (smart routing) |

### Cache Strategy

**When cache helps**:
- Documentation lookups
- Common "how-to" questions
- Repeated debugging

**When to clear cache**:
- After updating knowledge base
- When answers are stale
- Testing new features

### Quality Assurance

1. **Check validation scores** - Aim for 0.8+
2. **Review hallucination risk** - Should be <0.3
3. **Verify sources** - Check "Sources Used"
4. **Compare models** - For critical decisions
5. **Rate responses** - Help improve A/B tests

---

## FAQ

### General Questions

**Q: Do I need a GPU?**  
A: Not required, but strongly recommended. CPU-only will be 10-50x slower.

**Q: How much does it cost to run?**  
A: Free for self-hosted! Only costs are:
- Optional Brave API (~$5/month for 2000 searches)
- Optional GitHub API (free tier sufficient)
- Cloud hosting if using Azure/AWS

**Q: Can I add my own documents?**  
A: Yes! Place documents in:
- `data/ms/` for Microsoft technologies
- `data/general/` for general dev topics
- Restart services to re-index

**Q: Which model is best?**  
A: Depends on use case:
- General: Llama 3.1 8B (fast, good)
- Coding: CodeLlama 34B or Qwen 2.5 Coder
- Complex: Llama 3.1 70B (slow, excellent)
- Auto-select handles this for you!

### Technical Questions

**Q: How does caching work?**  
A: Redis stores query hash → response. TTL is 1 hour. Cache key includes query, file extension, and enabled tools.

**Q: Can I use other LLMs?**  
A: Yes! Ollama supports 50+ models. Edit `docker-compose.yaml` to add models.

**Q: Is my data private?**  
A: Yes, everything runs locally. No external calls except:
- Brave API (only if enabled)
- GitHub API (only if enabled)

**Q: How do I update?**  
A: ```bash
git pull
docker-compose pull
docker-compose up -d
```

**Q: Can I deploy to production?**  
A: Yes! See:
- [Azure Deployment Guide](../azure/README.md)
- [Kubernetes Guide](../azure/AKS_DEPLOYMENT.md)
- [PowerShell Scripts](../azure/POWERSHELL_DEPLOYMENT.md)

### Feature Questions

**Q: How accurate is hallucination detection?**  
A: ~85-90% accuracy for obvious hallucinations. Not perfect - always verify critical information.

**Q: What's the difference between validation and factuality checking?**  
A: 
- **Validation**: General quality (7 checks)
- **Factuality**: Specifically checks truthfulness and sources

**Q: Can I customize themes?**  
A: Yes, 9 built-in themes. Can add custom CSS in `ui/styles.css`.

**Q: How do ensembles work?**  
A: Multiple models process your query, then results are combined using voting, averaging, or consensus strategies.

**Q: What's the model performance overhead?**  
A: Minimal - statistics collection adds <10ms per query.

---

## Getting Help

### Resources

📖 **Documentation**
- [API Reference](./API_REFERENCE.md)
- [Azure Deployment](../azure/README.md)
- [PowerShell Guide](../azure/POWERSHELL_DEPLOYMENT.md)
- [Model Ensembles](./MODEL_ENSEMBLES.md)

🐛 **Report Issues**
- GitHub: https://github.com/adrian207/dual-rag-llm/issues
- Email: adrian207@gmail.com

💬 **Community**
- Discussions: GitHub Discussions (coming soon)
- Examples: See `examples/` directory

🚀 **Contributing**
- See [CONTRIBUTING.md](../CONTRIBUTING.md)
- Pull requests welcome!

---

## Next Steps

Now that you understand the system, try these:

1. **Explore Features**: Try each tool and setting
2. **Compare Models**: See which works best for your use case
3. **Add Documents**: Build your own knowledge base
4. **Deploy to Production**: Use Azure deployment scripts
5. **Contribute**: Share improvements with the community

**Happy querying!** 🎉

---

*Last updated: October 31, 2024*  
*Version: 1.11.0*  
*Author: Adrian Johnson <adrian207@gmail.com>*

