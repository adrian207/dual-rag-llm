# Web Tools Guide - Brave Search & GitHub Integration

**Author:** Adrian Johnson <adrian207@gmail.com>

Complete guide to using web search and GitHub integration features.

## Overview

The Dual RAG system now includes two powerful external tools:
1. **Brave Search** - Web search for current information
2. **GitHub API** - Code search across repositories

## Brave Search Integration

### Setup

#### 1. Get API Key

1. Visit https://brave.com/search/api/
2. Sign up for Brave Search API
3. Choose plan:
   - **Free**: 2,000 queries/month
   - **Paid**: From $5/month for 20,000 queries

#### 2. Configure

```bash
# Add to .env
BRAVE_API_KEY=your-brave-api-key-here
ENABLE_WEB_SEARCH=true
WEB_SEARCH_MAX_RESULTS=5
WEB_SEARCH_TIMEOUT=10
```

###Usage

#### Basic Web Search

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the latest features in Python 3.12?",
    "file_ext": ".py",
    "use_web_search": true
  }'
```

#### Response with Web Results

```json
{
  "answer": "Python 3.12 introduces...",
  "model": "deepseek-coder-v2:33b-q4_K_M",
  "source": "OpenSource",
  "chunks_retrieved": 3,
  "cached": false,
  "tools_used": ["web_search", "rag"],
  "tool_results": [
    {
      "tool": "web_search",
      "count": 5,
      "results": [
        {
          "title": "What's New In Python 3.12",
          "url": "https://docs.python.org/3.12/whatsnew/3.12.html",
          "description": "Python 3.12 is the latest stable release...",
          "age": "2 days ago"
        }
      ]
    }
  ]
}
```

### Use Cases

**1. Current Events & News**
```bash
{"question": "Latest Azure AI updates", "use_web_search": true}
```

**2. Package/Library Information**
```bash
{"question": "FastAPI 0.109 new features", "use_web_search": true}
```

**3. Error Messages & Troubleshooting**
```bash
{"question": "ModuleNotFoundError: No module named 'redis'", "use_web_search": true}
```

**4. Best Practices & Tutorials**
```bash
{"question": "Python async/await best practices 2024", "use_web_search": true}
```

### Configuration

```python
# In .env
WEB_SEARCH_MAX_RESULTS=5  # Number of results to fetch
WEB_SEARCH_TIMEOUT=10      # Request timeout in seconds
```

### Rate Limits

| Plan | Queries/Month | Cost |
|------|---------------|------|
| Free | 2,000 | $0 |
| Basic | 20,000 | $5/month |
| Pro | 200,000 | $50/month |
| Enterprise | Custom | Contact Brave |

## GitHub Integration

### Setup

#### 1. Create Personal Access Token

1. Go to https://github.com/settings/tokens
2. Click "Generate new token (classic)"
3. Select scopes:
   - ✅ `repo` (for private repositories)
   - ✅ `public_repo` (for public repositories)
4. Generate token
5. Copy token (you won't see it again!)

#### 2. Configure

```bash
# Add to .env
GITHUB_TOKEN=ghp_your_token_here
ENABLE_GITHUB=true
GITHUB_MAX_RESULTS=10
GITHUB_DEFAULT_BRANCH=main
```

### Usage

#### Search All GitHub

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "async redis connection pool python",
    "file_ext": ".py",
    "use_github": true
  }'
```

#### Search Specific Repository

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "caching middleware",
    "file_ext": ".py",
    "use_github": true,
    "github_repo": "fastapi/fastapi"
  }'
```

#### Response with GitHub Results

```json
{
  "answer": "Here's how to implement async Redis connection pool...",
  "tools_used": ["github", "rag"],
  "tool_results": [
    {
      "tool": "github",
      "count": 10,
      "results": [
        {
          "name": "redis_client.py",
          "path": "src/cache/redis_client.py",
          "url": "https://github.com/user/repo/blob/main/src/cache/redis_client.py",
          "repository": "user/repo",
          "sha": "abc123..."
        }
      ]
    }
  ]
}
```

### Use Cases

**1. Code Examples**
```bash
{"question": "FastAPI middleware example", "use_github": true}
```

**2. Implementation Patterns**
```bash
{"question": "Redis caching decorator", "use_github": true, "github_repo": "redis/redis-py"}
```

**3. Bug Fixes & Solutions**
```bash
{"question": "ConnectionError handling", "use_github": true}
```

**4. API Usage Examples**
```bash
{"question": "Ollama Python client", "use_github": true, "github_repo": "ollama/ollama-python"}
```

### Best Practices

**1. Specific Repositories**
- More relevant results
- Faster searches
- Better context

**2. Combine with RAG**
- Use both tools together
- RAG for documentation
- GitHub for real implementations

**3. Rate Limiting**
- GitHub: 5,000 requests/hour (authenticated)
- Cache results when possible
- Be respectful of API limits

## Combined Tool Usage

### Maximum Context

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How to implement Redis caching in FastAPI?",
    "file_ext": ".py",
    "use_web_search": true,
    "use_github": true,
    "github_repo": "fastapi/fastapi"
  }'
```

**Response includes:**
- RAG context from local documentation
- Web search results (tutorials, articles)
- GitHub code examples
- LLM synthesis of all sources

### Recommended Combinations

| Query Type | RAG | Web | GitHub |
|------------|-----|-----|--------|
| API Documentation | ✅ | ❌ | ❌ |
| Current Best Practices | ✅ | ✅ | ❌ |
| Implementation Examples | ✅ | ❌ | ✅ |
| Troubleshooting | ✅ | ✅ | ✅ |
| Latest Updates | ❌ | ✅ | ❌ |
| Code Patterns | ✅ | ❌ | ✅ |

## Monitoring

### Tool Usage Statistics

```bash
curl http://localhost:8000/stats

# Response:
{
  "tools": {
    "web_search": 245,
    "github": 189,
    "rag": 1542,
    "total": 1976
  }
}
```

### Performance Impact

| Tool | Latency | Cost |
|------|---------|------|
| RAG only | ~5s | Free |
| + Web Search | +2s | API quota |
| + GitHub | +1.5s | API quota |
| All tools | ~8.5s | API quotas |

## Troubleshooting

### Brave Search Issues

**"Invalid API key"**
```bash
# Check API key is set
echo $BRAVE_API_KEY

# Verify in .env
grep BRAVE_API_KEY .env

# Test API key
curl -H "X-Subscription-Token: $BRAVE_API_KEY" \
  "https://api.search.brave.com/res/v1/web/search?q=test"
```

**Rate limit exceeded**
```bash
# Check usage at https://brave.com/search/api/
# Upgrade plan or reduce usage
WEB_SEARCH_MAX_RESULTS=3  # Reduce results
```

### GitHub Issues

**"Bad credentials"**
```bash
# Regenerate token
# Go to https://github.com/settings/tokens
# Delete old token
# Create new token with correct scopes

# Update .env
GITHUB_TOKEN=ghp_new_token_here

# Restart service
docker compose restart rag
```

**Rate limit exceeded**
```bash
# Check current limit
curl -H "Authorization: token $GITHUB_TOKEN" \
  https://api.github.com/rate_limit

# Response shows remaining requests
# Wait for reset or use caching
```

**Repository not found**
```bash
# Verify repository exists
# Check spelling: "user/repo"
# Ensure token has access (public/private)
```

## Security

### API Key Management

**DON'T:**
- ❌ Commit API keys to git
- ❌ Share API keys publicly
- ❌ Use same key across environments

**DO:**
- ✅ Use .env files (gitignored)
- ✅ Separate keys for dev/prod
- ✅ Rotate keys regularly
- ✅ Use Azure Key Vault in production

### Kubernetes Secrets

```bash
# Create secret
kubectl create secret generic api-keys \
  --from-literal=brave-api-key=your-key \
  --from-literal=github-token=your-token \
  -n dual-rag

# Reference in deployment
env:
  - name: BRAVE_API_KEY
    valueFrom:
      secretKeyRef:
        name: api-keys
        key: brave-api-key
```

## Cost Optimization

### Brave Search

**Reduce Usage:**
```bash
# Enable only when needed
ENABLE_WEB_SEARCH=false  # Default off

# Require explicit flag
use_web_search=true  # User must request

# Cache results
# Web search results are cached with query
```

**Estimated Costs:**
```
Free tier: 2,000 queries/month
Avg usage: 100 queries/day = 3,000/month
Cost: $5/month (Basic plan)
```

### GitHub

**Free for most use cases:**
- 5,000 requests/hour authenticated
- Typical usage: <100/hour
- Cost: $0

## Future Enhancements

- [ ] Wikipedia integration
- [ ] Stack Overflow search
- [ ] arXiv paper search
- [ ] Package registry search (PyPI, npm)
- [ ] Documentation site scraping
- [ ] Custom search engines

---

For questions: adrian207@gmail.com

