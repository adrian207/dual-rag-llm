# Dynamic Model Switching Guide

**Version:** 1.3.0  
**Author:** Adrian Johnson <adrian207@gmail.com>  
**Date:** 2025-10-31

## Overview

Dynamic Model Switching transforms the Dual RAG LLM System into a flexible, multi-model platform that lets you choose, compare, and optimize between different LLMs in real-time.

### Key Benefits

- **Model Choice**: Select from 10+ available models
- **Performance Comparison**: Run parallel queries to compare models
- **Automatic Fallback**: Seamless fallback when models are unavailable  
- **Performance Tracking**: Real-time metrics for each model
- **One-Click Retry**: Try same query with different models

---

## Features

### 1. Model Selector

Choose your preferred model from an organized dropdown:

**Microsoft Technologies:**
- Qwen 2.5 Coder 32B (Best for C#, PowerShell, YAML)
- Qwen 2.5 Coder 14B (Balanced)
- Qwen 2.5 Coder 7B (Fastest)

**Open Source:**
- DeepSeek Coder V2 33B (Best for Python, JavaScript)
- DeepSeek Coder V2 16B (Balanced)
- CodeLlama 34B (Traditional)
- CodeLlama 13B (Fast)

**General Purpose:**
- Llama 3.1 70B (Best overall)
- Llama 3.1 8B (Fastest)
- Mistral 7B (Efficient)

### 2. Model Comparison

Enable "Compare Models" checkbox to run your query against multiple models simultaneously.

**What Gets Compared:**
- Response quality and accuracy
- Response time
- Tokens per second
- Context retrieval performance

**Comparison Display:**
- Side-by-side responses
- Performance metrics highlighted
- "Winner" badge for fastest model
- Full response expansion
- One-click model selection

### 3. Try Another Model

After receiving a response, click "🔄 Try Another Model" to:
- Re-run same query with different model
- Compare results instantly
- Find the best model for your use case

### 4. Automatic Fallback

If a model is unavailable, the system automatically falls back to a smaller, faster model:

**Fallback Chain:**
```
qwen2.5-coder:32b → qwen2.5-coder:14b → qwen2.5-coder:7b
deepseek-coder-v2:33b → deepseek-coder-v2:16b → codellama:13b
codellama:34b → codellama:13b → llama3.1:8b
llama3.1:70b → llama3.1:8b
```

### 5. Performance Tracking

Real-time performance metrics displayed in sidebar:
- **Tokens per second** - Generation speed
- **Average response time** - Query completion
- **Query count** - Usage statistics

---

## Usage Examples

### Example 1: Selecting a Specific Model

```javascript
// In the UI
1. Open http://localhost:8000/ui
2. Select "Qwen 2.5 Coder 32B" from model dropdown
3. Ask: "How do I implement async/await in C#?"
4. Model override applied automatically
```

### Example 2: Comparing Models

```javascript
// In the UI
1. Check "🔄 Compare Models"
2. Ask: "Explain Python decorators"
3. System runs query on:
   - Qwen 2.5 Coder 32B
   - DeepSeek Coder V2 33B
   - (+ your selected model if different)
4. View side-by-side comparison
5. Click "✅ Use This Model" on best result
```

### Example 3: API Usage with Model Override

```python
import requests

response = requests.post("http://localhost:8000/query/stream", json={
    "question": "How do I use Redis with Python?",
    "file_ext": ".py",
    "model_override": "deepseek-coder-v2:33b-q4_K_M",
    "use_web_search": True
})

# Streams response from DeepSeek model
```

### Example 4: Model Comparison via API

```python
response = requests.post("http://localhost:8000/query/compare", json={
    "question": "Best practices for error handling in Go",
    "file_ext": ".go",
    "use_web_search": False
})

data = response.json()
print(f"Compared {data['comparison_count']} models")

for result in data['results']:
    print(f"{result['model']}: {result['response_time']}s")
    print(f"Performance: {result['performance']['avg_tokens_per_sec']} tok/s")
```

---

## API Reference

### POST /query/stream

**New Parameters:**
```json
{
  "question": "string",
  "file_ext": "string",
  "model_override": "string (optional)",
  "compare_models": false,
  "use_web_search": false,
  "use_github": false,
  "github_repo": "string (optional)"
}
```

**Response Events:**
- `status` - Processing updates (includes model name)
- `token` - Streaming tokens
- `done` - Complete response with performance metrics
- `error` - Errors with fallback suggestions

**Performance Data:**
```json
{
  "event": "done",
  "data": {
    "model": "qwen2.5-coder:32b-q4_K_M",
    "performance": {
      "response_time": 5.23,
      "tokens": 487,
      "tokens_per_sec": 93.1
    }
  }
}
```

### POST /query/compare

Compares responses from multiple models.

**Request:**
```json
{
  "question": "Explain async programming",
  "file_ext": ".py",
  "model_override": "llama3.1:70b" // Optional 3rd model
}
```

**Response:**
```json
{
  "question": "Explain async programming",
  "comparison_count": 2,
  "results": [
    {
      "model": "qwen2.5-coder:32b-q4_K_M",
      "source": "Microsoft",
      "answer": "...",
      "chunks_retrieved": 3,
      "response_time": 4.2,
      "performance": {
        "queries": 15,
        "avg_tokens_per_sec": 95.3,
        "avg_response_time": 4.5
      }
    },
    {
      "model": "deepseek-coder-v2:33b-q4_K_M",
      "source": "Open Source",
      "answer": "...",
      "chunks_retrieved": 3,
      "response_time": 3.8,
      "performance": {...}
    }
  ]
}
```

### GET /models/available

Get list of all available models from Ollama.

**Response:**
```json
{
  "available": [
    "qwen2.5-coder:32b-q4_K_M",
    "deepseek-coder-v2:33b-q4_K_M",
    "llama3.1:70b",
    ...
  ],
  "loaded": ["qwen2.5-coder:32b-q4_K_M"],
  "performance": {
    "qwen2.5-coder:32b-q4_K_M": {
      "queries": 25,
      "total_time": 125.5,
      "total_tokens": 12450,
      "avg_tokens_per_sec": 99.2,
      "avg_response_time": 5.02
    }
  }
}
```

### GET /stats

Enhanced with model performance data.

**Response:**
```json
{
  "cache": {...},
  "tools": {...},
  "models_cached": ["qwen2.5-coder:32b-q4_K_M"],
  "ms_index_loaded": true,
  "oss_index_loaded": true,
  "model_performance": {
    "qwen2.5-coder:32b-q4_K_M": {
      "queries": 25,
      "total_time": 125.5,
      "total_tokens": 12450,
      "avg_tokens_per_sec": 99.2,
      "avg_response_time": 5.02
    }
  }
}
```

---

## Architecture

### Backend Changes

**1. Query Model (`rag_dual.py`):**
```python
class Query(BaseModel):
    question: str
    file_ext: str = ""
    model_override: Optional[str] = None  # NEW
    compare_models: bool = False          # NEW
    use_web_search: bool = False
    use_github: bool = False
```

**2. State Management:**
```python
class AppState:
    model_performance: Dict[str, Dict[str, Any]] = {}
    
    # Tracks per model:
    # - queries: Total queries processed
    # - total_time: Cumulative response time
    # - total_tokens: Total tokens generated
    # - avg_tokens_per_sec: Generation speed
    # - avg_response_time: Average latency
```

**3. Streaming Response:**
- Checks `model_override` first
- Falls back to automatic routing
- Tracks performance metrics
- Updates stats in real-time

**4. Comparison Endpoint:**
- Runs queries in parallel
- Collects all responses
- Aggregates performance data
- Returns structured comparison

### Frontend Changes

**1. Model Selector UI:**
- Organized dropdown by category
- Highlighted with primary color
- Displays speed indicators
- Auto-updates from Ollama

**2. Comparison View:**
- CSS Grid layout (2 columns)
- Winner highlighting
- Performance badges
- Expand/collapse responses

**3. Performance Sidebar:**
- Real-time stats update
- Sorted by tokens/sec
- Color-coded indicators
- Click to select model

**4. Action Buttons:**
- Try Another Model
- Compare Models
- View Full Response
- Use This Model

---

## Performance Metrics

### Typical Response Times (GPU: RTX 4090)

| Model | Tokens/Sec | Avg Response | Quality |
|-------|------------|--------------|---------|
| Qwen 2.5 Coder 7B | 150-180 | 2-3s | Good |
| Qwen 2.5 Coder 14B | 110-130 | 4-5s | Very Good |
| Qwen 2.5 Coder 32B | 80-100 | 6-8s | Excellent |
| DeepSeek V2 16B | 120-140 | 3-4s | Very Good |
| DeepSeek V2 33B | 70-90 | 7-9s | Excellent |
| CodeLlama 13B | 130-150 | 3-4s | Good |
| CodeLlama 34B | 75-95 | 7-9s | Very Good |
| Llama 3.1 8B | 160-200 | 2-3s | Good |
| Llama 3.1 70B | 50-70 | 10-15s | Excellent |

### Cache Impact

- **Cache Hit**: <500ms (99.5% faster)
- **Cache Miss + Model**: 3-15s depending on model
- **Comparison (2 models)**: 6-20s parallel execution

---

## Best Practices

### Choosing the Right Model

**For Microsoft Technologies (.cs, .ps1, .yaml):**
- **Best**: Qwen 2.5 Coder 32B
- **Balanced**: Qwen 2.5 Coder 14B
- **Fast**: Qwen 2.5 Coder 7B

**For Open Source (.py, .js, .go):**
- **Best**: DeepSeek Coder V2 33B
- **Balanced**: DeepSeek Coder V2 16B
- **Fast**: CodeLlama 13B

**For General Queries:**
- **Best**: Llama 3.1 70B
- **Fast**: Llama 3.1 8B

### Optimization Tips

1. **Use Auto-Routing**: Let system choose based on file extension
2. **Enable Caching**: Repeat queries are instant
3. **Compare Strategically**: Use for critical queries only
4. **Monitor Performance**: Check sidebar for model efficiency
5. **Fallback Awareness**: Know your fallback chain

### Production Deployment

**Model Loading Strategy:**
```yaml
# Preload frequently used models
PRELOAD_MODELS:
  - qwen2.5-coder:14b      # Fast, versatile
  - deepseek-coder-v2:16b  # Good balance
  - llama3.1:8b            # General purpose
```

**Resource Allocation:**
- **32B+ models**: 24GB+ VRAM
- **14-16B models**: 16GB VRAM
- **7-13B models**: 8GB VRAM
- **Concurrent models**: Plan for 2-3x peak memory

---

## Troubleshooting

### Issue: Model Not Loading

**Error:** `Model qwen2.5-coder:32b-q4_K_M not available`

**Solution:**
```bash
# Pull model from Ollama
ollama pull qwen2.5-coder:32b-q4_K_M

# Or set fallback
# System will automatically use qwen2.5-coder:14b
```

### Issue: Slow Comparison

**Symptom:** Comparison takes >30s

**Causes:**
- Both models are large (32B+)
- Cold start (models not loaded)
- Insufficient VRAM

**Solutions:**
```python
# Compare smaller models
{
  "model_override": "qwen2.5-coder:7b",
  "compare_models": true
}

# Preload models
curl -X POST http://localhost:11434/api/generate \
  -d '{"model": "qwen2.5-coder:32b-q4_K_M", "prompt": "test"}'
```

### Issue: Performance Stats Not Showing

**Check:**
1. Run at least one query per model
2. Check `/stats` endpoint for data
3. Refresh stats sidebar
4. Clear browser cache

---

## Future Enhancements

### Planned for v1.4.0

1. **Model Voting**: Rate responses, crowd-source best models
2. **A/B Testing**: Automatic comparison for quality assurance
3. **Cost Tracking**: Monitor inference costs per model
4. **Model Scheduling**: Queue multiple model queries
5. **Custom Fallbacks**: User-defined fallback chains

### Ideas for v2.0.0

- Multi-model ensemble (combine outputs)
- Model fine-tuning integration
- Real-time model switching mid-stream
- GPU resource prediction and allocation

---

## Conclusion

Dynamic Model Switching gives you complete control over your RAG system's intelligence layer. Whether you need speed, quality, or cost optimization, you can now choose the perfect model for every query.

**Quick Start:**
```bash
# Start system
docker compose up -d

# Open UI
http://localhost:8000/ui

# Select a model
Choose from dropdown

# Try comparison
Enable "Compare Models" checkbox
```

For questions or feedback: adrian207@gmail.com

---

**Related Documentation:**
- [STREAMING_UI_GUIDE.md](./STREAMING_UI_GUIDE.md) - Real-time streaming
- [WEB_TOOLS_GUIDE.md](./WEB_TOOLS_GUIDE.md) - Brave & GitHub integration
- [REDIS_CACHING_GUIDE.md](./REDIS_CACHING_GUIDE.md) - Caching strategy

**Version:** 1.3.0  
**Last Updated:** 2025-10-31  
**Author:** Adrian Johnson <adrian207@gmail.com>

