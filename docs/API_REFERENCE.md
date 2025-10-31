# API Reference

**Version:** 1.11.0  
**Author:** Adrian Johnson <adrian207@gmail.com>  
**Base URL:** `http://localhost:8000`

Complete API reference for the Dual RAG LLM System.

---

## Table of Contents

1. [Authentication](#authentication)
2. [Core Endpoints](#core-endpoints)
3. [Tool Endpoints](#tool-endpoints)
4. [Cache Endpoints](#cache-endpoints)
5. [Model Management](#model-management)
6. [Validation & Quality](#validation--quality)
7. [Syntax Highlighting](#syntax-highlighting)
8. [A/B Testing](#ab-testing)
9. [Fine-tuning](#fine-tuning)
10. [Ensembles](#ensembles)
11. [Auto-Selection](#auto-selection)
12. [Error Handling](#error-handling)

---

## Authentication

Currently, no authentication is required for local deployment. For production, implement one of:
- API Keys (add middleware)
- OAuth 2.0
- JWT tokens

---

## Core Endpoints

### POST /query

Standard query endpoint with full response.

**Request:**
```http
POST /query
Content-Type: application/json

{
  "question": "How do I use async/await in Python?",
  "file_ext": ".py",
  "use_web_search": false,
  "use_github": false,
  "github_repo": "",
  "model_override": "",
  "compare_models": false,
  "ab_test_id": "",
  "query_id": "",
  "ensemble_id": ""
}
```

**Parameters:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `question` | string | Yes | - | The query to process |
| `file_ext` | string | No | "" | File extension hint (.py, .js, etc.) |
| `use_web_search` | boolean | No | false | Enable Brave web search |
| `use_github` | boolean | No | false | Enable GitHub integration |
| `github_repo` | string | No | "" | Specific repo (user/repo format) |
| `model_override` | string | No | "" | Specific model to use |
| `compare_models` | boolean | No | false | Compare multiple models |
| `ab_test_id` | string | No | "" | A/B test identifier |
| `query_id` | string | No | auto | Unique query ID |
| `ensemble_id` | string | No | "" | Ensemble configuration ID |

**Response:**
```json
{
  "answer": "To use async/await in Python...",
  "sources": [
    "ms_doc_123.txt",
    "general_doc_456.txt"
  ],
  "metadata": {
    "model": "llama3.1:8b",
    "cached": false,
    "cache_key": "abc123...",
    "ms_docs_used": 2,
    "general_docs_used": 3,
    "web_results": 0,
    "github_results": 0,
    "response_time_ms": 2543,
    "tokens_generated": 456
  }
}
```

**Status Codes:**
- `200`: Success
- `400`: Invalid request
- `500`: Server error
- `503`: Ollama unavailable

---

### POST /query/stream

Streaming query endpoint using Server-Sent Events (SSE).

**Request:** Same as `/query`

**Response:** Server-Sent Events

```
event: token
data: "To"

event: token
data: " use"

event: token
data: " async"

event: done
data: {"metadata": {...}, "sources": [...]}

event: error
data: {"error": "Error message"}
```

**Event Types:**

| Event | Description | Data Format |
|-------|-------------|-------------|
| `token` | Single token | string |
| `done` | Stream complete | JSON object |
| `error` | Error occurred | JSON with error message |

**JavaScript Example:**
```javascript
const eventSource = new EventSource('/query/stream');

eventSource.addEventListener('token', (e) => {
  console.log('Token:', e.data);
});

eventSource.addEventListener('done', (e) => {
  const metadata = JSON.parse(e.data);
  console.log('Complete:', metadata);
  eventSource.close();
});

eventSource.addEventListener('error', (e) => {
  console.error('Error:', e.data);
  eventSource.close();
});
```

---

### GET /health

Health check endpoint.

**Request:**
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "ollama_available": true,
  "redis_available": true,
  "ms_index_loaded": true,
  "general_index_loaded": true,
  "version": "1.11.0"
}
```

**Status Values:**
- `healthy`: All services operational
- `degraded`: Some services unavailable
- `unhealthy`: Critical services down

---

### GET /stats

System statistics.

**Request:**
```http
GET /stats
```

**Response:**
```json
{
  "cache": {
    "total_queries": 1523,
    "cache_hits": 487,
    "cache_misses": 1036,
    "hit_rate": 0.32
  },
  "tools": {
    "web_searches": 234,
    "github_queries": 89
  },
  "models_cached": [
    "llama3.1:8b",
    "codellama:7b",
    "qwen2.5-coder:14b"
  ],
  "ms_index_loaded": true,
  "general_index_loaded": true,
  "model_performance": {
    "llama3.1:8b": {
      "queries": 856,
      "avg_tokens_per_sec": 45.3,
      "avg_response_time_ms": 2341
    }
  }
}
```

---

## Tool Endpoints

### POST /tools/web_search

Execute web search using Brave API.

**Request:**
```json
{
  "query": "latest Python features",
  "count": 5
}
```

**Response:**
```json
{
  "results": [
    {
      "title": "Python 3.12 Release",
      "url": "https://...",
      "description": "...",
      "published_date": "2024-10-15"
    }
  ],
  "total_results": 5
}
```

---

### POST /tools/github_search

Search GitHub repositories.

**Request:**
```json
{
  "query": "fastapi authentication",
  "repo": "fastapi/fastapi"
}
```

**Response:**
```json
{
  "results": [
    {
      "path": "docs/tutorial/security.md",
      "url": "https://github.com/...",
      "content": "...",
      "score": 0.95
    }
  ],
  "total_results": 3,
  "repo_info": {
    "full_name": "fastapi/fastapi",
    "description": "...",
    "stars": 67890
  }
}
```

---

## Cache Endpoints

### POST /cache/clear

Clear all cached responses.

**Request:**
```http
POST /cache/clear
```

**Response:**
```json
{
  "status": "success",
  "message": "Cache cleared",
  "keys_deleted": 487
}
```

---

### GET /cache/stats

Get cache statistics.

**Request:**
```http
GET /cache/stats
```

**Response:**
```json
{
  "total_keys": 487,
  "memory_used_mb": 245.3,
  "hit_rate": 0.32,
  "avg_ttl_seconds": 2145
}
```

---

## Model Management

### GET /models/available

List available models.

**Response:**
```json
{
  "models": [
    {
      "name": "llama3.1:8b",
      "size": "4.7GB",
      "family": "llama",
      "loaded": true
    },
    {
      "name": "codellama:7b",
      "size": "3.8GB",
      "family": "llama",
      "loaded": false
    }
  ]
}
```

---

### GET /models/performance

Get model performance metrics.

**Response:**
```json
{
  "llama3.1:8b": {
    "total_queries": 856,
    "avg_tokens_per_sec": 45.3,
    "avg_response_time_ms": 2341,
    "success_rate": 0.98,
    "last_used": "2024-10-31T10:30:00Z"
  }
}
```

---

## Validation & Quality

### POST /validation/validate

Validate an answer.

**Request:**
```json
{
  "query_id": "abc123",
  "query": "What is Python?",
  "answer": "Python is...",
  "model": "llama3.1:8b",
  "context": "Python documentation..."
}
```

**Response:**
```json
{
  "query_id": "abc123",
  "overall_score": 0.85,
  "approved": true,
  "checks": [
    {
      "check_type": "factuality",
      "passed": true,
      "confidence": 0.92,
      "details": "Claims supported by context"
    },
    {
      "check_type": "completeness",
      "passed": true,
      "confidence": 0.88,
      "details": "Query fully answered"
    }
  ],
  "suggestions": []
}
```

---

### GET /validation/config

Get validation configuration.

**Response:**
```json
{
  "enabled": true,
  "min_confidence_threshold": 0.5,
  "enable_factuality_check": true,
  "enable_source_verification": true,
  "enable_consistency_check": true,
  "enable_completeness_check": true,
  "enable_code_validation": true,
  "auto_reject_threshold": 0.3
}
```

---

### PUT /validation/config

Update validation configuration.

**Request:** Same structure as GET response

---

## Syntax Highlighting

### POST /syntax/highlight

Highlight code with syntax.

**Request:**
```json
{
  "code": "def hello():\n    print('world')",
  "language": "python",
  "theme": "vs-dark"
}
```

**Response:**
```json
{
  "original": "def hello()...",
  "language": "python",
  "theme": "vs-dark",
  "tokens": [
    {"type": "keyword", "value": "def"},
    {"type": "function", "value": "hello"}
  ],
  "line_count": 2,
  "has_syntax_errors": false
}
```

---

### POST /syntax/detect

Detect code language.

**Request:**
```json
{
  "code": "const x = 5;",
  "hint": ""
}
```

**Response:**
```json
{
  "language": "javascript",
  "confidence": 0.95,
  "aliases": ["js", "jsx"],
  "file_extension": ".js"
}
```

---

### GET /syntax/languages

List supported languages.

**Response:**
```json
{
  "languages": [
    {
      "name": "Python",
      "id": "python",
      "extensions": [".py"]
    },
    {
      "name": "PowerShell",
      "id": "powershell",
      "extensions": [".ps1", ".psm1"]
    }
  ]
}
```

---

## A/B Testing

### POST /ab-tests

Create A/B test.

**Request:**
```json
{
  "name": "Llama vs CodeLlama",
  "description": "Compare coding models",
  "model_a": "llama3.1:8b",
  "model_b": "codellama:7b",
  "traffic_split": 0.5,
  "min_samples": 50,
  "max_samples": 200
}
```

**Response:**
```json
{
  "test_id": "test_abc123",
  "name": "Llama vs CodeLlama",
  "status": "draft",
  "created_at": "2024-10-31T10:00:00Z"
}
```

---

### POST /ab-tests/{test_id}/start

Start an A/B test.

**Response:**
```json
{
  "test_id": "test_abc123",
  "status": "running",
  "started_at": "2024-10-31T10:05:00Z"
}
```

---

### GET /ab-tests/{test_id}/results

Get test results.

**Response:**
```json
{
  "test_id": "test_abc123",
  "status": "completed",
  "model_a_stats": {
    "avg_response_time": 2341,
    "avg_rating": 4.2,
    "sample_size": 52
  },
  "model_b_stats": {
    "avg_response_time": 1987,
    "avg_rating": 4.5,
    "sample_size": 48
  },
  "statistical_analysis": {
    "p_value": 0.032,
    "significant": true,
    "confidence_level": 0.95,
    "winner": "model_b"
  }
}
```

---

## Fine-tuning

### POST /finetuning/datasets

Create training dataset.

**Request:**
```json
{
  "name": "My Custom Dataset",
  "description": "Domain-specific training data",
  "format": "alpaca",
  "file_path": "/data/train.json",
  "num_examples": 1000
}
```

---

### POST /finetuning/jobs

Create training job.

**Request:**
```json
{
  "name": "Custom Model Training",
  "base_model": "llama3.1:8b",
  "dataset_id": "dataset_123",
  "config": {
    "learning_rate": 0.0001,
    "epochs": 3,
    "batch_size": 4,
    "use_lora": true
  }
}
```

---

## Ensembles

### POST /ensembles

Create ensemble configuration.

**Request:**
```json
{
  "name": "Code Review Ensemble",
  "description": "Multiple models for code review",
  "strategy": "voting",
  "models": ["llama3.1:8b", "codellama:7b", "qwen2.5-coder:14b"],
  "weights": [1.0, 1.0, 1.0],
  "config": {
    "threshold": 0.7,
    "min_agreement": 2
  }
}
```

---

### POST /ensembles/{ensemble_id}/test

Test ensemble configuration.

**Request:**
```json
{
  "query": "Review this Python code...",
  "context": "def process()..."
}
```

---

## Auto-Selection

### GET /auto-selection/config

Get auto-selection configuration.

**Response:**
```json
{
  "enabled": true,
  "default_model": "llama3.1:8b",
  "fallback_model": "codellama:7b",
  "confidence_threshold": 0.6,
  "learning_enabled": true,
  "learning_rate": 0.1
}
```

---

### GET /auto-selection/routing

Get routing matrix.

**Response:**
```json
{
  "code_generation": {
    "primary_model": "codellama:34b",
    "secondary_model": "qwen2.5-coder:14b",
    "confidence_threshold": 0.7
  },
  "general_question": {
    "primary_model": "llama3.1:8b",
    "secondary_model": "mistral:7b",
    "confidence_threshold": 0.6
  }
}
```

---

## Error Handling

### Error Response Format

All errors return consistent format:

```json
{
  "error": {
    "code": "INVALID_REQUEST",
    "message": "The 'question' field is required",
    "details": {
      "field": "question",
      "constraint": "required"
    },
    "request_id": "req_abc123"
  }
}
```

### Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `INVALID_REQUEST` | 400 | Request validation failed |
| `MODEL_UNAVAILABLE` | 503 | Specified model not available |
| `CACHE_ERROR` | 500 | Redis connection failed |
| `OLLAMA_ERROR` | 503 | Ollama service unavailable |
| `RATE_LIMIT_EXCEEDED` | 429 | Too many requests |
| `INTERNAL_ERROR` | 500 | Unexpected server error |

### Rate Limiting

Default limits (can be configured):
- 100 requests per minute per IP
- 1000 requests per hour per IP

**Headers:**
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 73
X-RateLimit-Reset: 1698765432
```

---

## Pagination

For endpoints returning lists:

**Request:**
```http
GET /api/endpoint?page=2&per_page=50
```

**Response:**
```json
{
  "items": [...],
  "pagination": {
    "page": 2,
    "per_page": 50,
    "total_items": 237,
    "total_pages": 5,
    "has_next": true,
    "has_prev": true
  }
}
```

---

## Webhooks

Register webhooks for events:

**POST /webhooks**
```json
{
  "url": "https://your-server.com/webhook",
  "events": ["query.completed", "test.finished"],
  "secret": "your_webhook_secret"
}
```

**Webhook Payload:**
```json
{
  "event": "query.completed",
  "timestamp": "2024-10-31T10:00:00Z",
  "data": {
    "query_id": "abc123",
    "model": "llama3.1:8b",
    "cached": false
  }
}
```

---

## SDKs & Libraries

### Python SDK

```python
from dual_rag import DualRAGClient

client = DualRAGClient("http://localhost:8000")

# Simple query
response = client.query("How do I use async in Python?")
print(response.answer)

# Streaming query
for token in client.query_stream("Explain decorators"):
    print(token, end='')

# With options
response = client.query(
    "Latest React features",
    use_web_search=True,
    model="llama3.1:70b"
)
```

### JavaScript SDK

```javascript
import { DualRAGClient } from '@dual-rag/client';

const client = new DualRAGClient('http://localhost:8000');

// Simple query
const response = await client.query('How do I use async in JavaScript?');
console.log(response.answer);

// Streaming
const stream = client.queryStream('Explain promises');
stream.on('token', token => console.log(token));
stream.on('done', metadata => console.log('Done:', metadata));
```

---

## API Versioning

Current version: **v1**

Future versions will use URL prefixing:
```
http://localhost:8000/v1/query
http://localhost:8000/v2/query  (when available)
```

Version headers:
```
X-API-Version: 1.11.0
X-Min-Client-Version: 1.0.0
```

---

## Support

- **Documentation**: [docs/](https://github.com/adrian207/dual-rag-llm/tree/main/docs)
- **Issues**: [GitHub Issues](https://github.com/adrian207/dual-rag-llm/issues)
- **Email**: adrian207@gmail.com

---

*Last updated: October 31, 2024*  
*Version: 1.11.0*  
*Author: Adrian Johnson <adrian207@gmail.com>*

