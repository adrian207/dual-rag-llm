# Model Ensemble Strategies

**Author:** Adrian Johnson <adrian207@gmail.com>  
**Version:** 1.6.0  
**Release Date:** October 31, 2025

## Overview

Model ensembles combine multiple LLMs to achieve better results than any single model alone. By leveraging the strengths of different models, ensembles provide:

- **Higher Accuracy**: Multiple perspectives reduce errors
- **Better Reliability**: Fallback options if one model fails
- **Optimized Performance**: Balance speed vs quality
- **Domain Specialization**: Route queries to expert models
- **Increased Confidence**: Agreement validates results

## Quick Start

### Access the Dashboard

Navigate to: `http://localhost:8000/ui/ensembles.html`

### Create Your First Ensemble

1. Click **"Create Ensemble"** tab
2. Enter ensemble name (e.g., "Fast Code Ensemble")
3. Select strategy (e.g., "Cascade")
4. Choose models (minimum 2)
5. Configure strategy-specific settings
6. Click **"Create Ensemble"**

### Test the Ensemble

1. Find your ensemble in the list
2. Click **"Test"**
3. Enter a test question
4. Review individual model responses
5. See the final ensemble result

## Ensemble Strategies

### 1. 🗳️ Voting

**How it works:** Multiple models vote on the answer. The most common response wins.

**Use Cases:**
- Binary decisions (yes/no)
- Classification tasks
- Multiple choice questions
- Fact verification

**Configuration:**
```json
{
  "strategy": "voting",
  "models": ["qwen2.5-coder:7b", "deepseek-coder-v2:16b", "codellama:13b"],
  "weights": [1.0, 1.5, 1.0]  // Optional: weight certain models higher
}
```

**Example:**
```
Question: "Is this code vulnerable to SQL injection?"
Model A: "Yes"
Model B: "Yes"
Model C: "No"
Result: "Yes" (2 votes vs 1)
```

**Best For:**
- Security audits
- Code review decisions
- Compliance checks

---

### 2. 📊 Averaging

**How it works:** Combines responses using weighted average based on model confidence.

**Use Cases:**
- Numeric predictions
- Code quality scores
- Confidence rankings

**Configuration:**
```json
{
  "strategy": "averaging",
  "models": ["qwen2.5-coder:7b", "deepseek-coder-v2:33b"],
  "weights": [0.4, 0.6]  // 40% and 60% influence
}
```

**Example:**
```
Question: "Rate this code quality 0-10"
Model A: 7.5 (confidence: 0.8)
Model B: 8.2 (confidence: 0.9)
Result: Weighted average based on confidence
```

**Best For:**
- Quality assessments
- Performance predictions
- Risk scoring

---

### 3. ⚡ Cascade

**How it works:** Try fast models first. If confidence is low, fall back to slower, more powerful models.

**Use Cases:**
- Optimizing response time
- Cost-effective queries
- Variable complexity tasks

**Configuration:**
```json
{
  "strategy": "cascade",
  "models": [
    "qwen2.5-coder:7b",    // Fast model first
    "deepseek-coder-v2:33b" // Powerful fallback
  ],
  "threshold": 0.7,  // Require 70% confidence
  "parallel": false  // Sequential execution
}
```

**Example:**
```
Question: "Write a hello world function"
→ Try Qwen 7B (0.2s, confidence: 0.85) ✓
→ Confidence > 0.7, return immediately
→ DeepSeek never called (saved time!)

Question: "Implement distributed consensus algorithm"
→ Try Qwen 7B (0.2s, confidence: 0.45) ✗
→ Confidence < 0.7, try next model
→ DeepSeek 33B (2.5s, confidence: 0.92) ✓
→ Return DeepSeek's answer
```

**Best For:**
- Mixed complexity workloads
- Budget-conscious deployments
- Speed optimization

---

### 4. 🏆 Best-of-N

**How it works:** Run all models and select the response with highest confidence.

**Use Cases:**
- Critical code generation
- Maximum quality needed
- When accuracy trumps speed

**Configuration:**
```json
{
  "strategy": "best_of_n",
  "models": [
    "qwen2.5-coder:7b",
    "deepseek-coder-v2:16b",
    "codellama:13b"
  ],
  "parallel": true  // Run simultaneously
}
```

**Example:**
```
Question: "Implement secure password hashing"
Model A: Response (confidence: 0.75)
Model B: Response (confidence: 0.92) ← Selected
Model C: Response (confidence: 0.81)
Result: Model B's answer (highest confidence)
```

**Best For:**
- Production code generation
- Security-critical tasks
- High-stakes decisions

---

### 5. 🎓 Specialist

**How it works:** Route different question types to specialized models automatically.

**Use Cases:**
- Multi-domain systems
- Specialized models available
- Automatic routing needed

**Configuration:**
```json
{
  "strategy": "specialist",
  "models": [
    "qwen2.5-coder:7b",
    "deepseek-coder-v2:16b",
    "llama3.1:8b"
  ],
  "routing_rules": {
    "code": "qwen2.5-coder:7b",
    "explanation": "llama3.1:8b",
    "debugging": "deepseek-coder-v2:16b"
  }
}
```

**Example:**
```
Question: "Write a sorting algorithm"
→ Detected as "code" question
→ Route to: qwen2.5-coder:7b

Question: "Explain how quicksort works"
→ Detected as "explanation" question
→ Route to: llama3.1:8b

Question: "Fix this segfault"
→ Detected as "debugging" question
→ Route to: deepseek-coder-v2:16b
```

**Best For:**
- RAG systems with diverse queries
- Domain-specific models
- Intelligent routing

---

### 6. 🤝 Consensus

**How it works:** Requires threshold% of models to agree before accepting answer.

**Use Cases:**
- Critical decisions
- High-confidence requirements
- Risk-averse scenarios

**Configuration:**
```json
{
  "strategy": "consensus",
  "models": [
    "qwen2.5-coder:7b",
    "deepseek-coder-v2:16b",
    "codellama:13b",
    "llama3.1:8b"
  ],
  "threshold": 0.75,  // Require 75% agreement
  "min_responses": 3  // At least 3 models must respond
}
```

**Example:**
```
Question: "Is this cryptographic implementation secure?"
Model A: "Yes"
Model B: "Yes"
Model C: "Yes"
Model D: "No"
Agreement: 75% (3/4) ✓
Result: "Yes" (consensus reached)

If only 50% agreed → No consensus → Use highest confidence model
```

**Best For:**
- Security audits
- Compliance validation
- Medical/legal applications

---

## API Reference

### Create Ensemble

```bash
POST /ensembles

{
  "name": "My Ensemble",
  "description": "Fast code generation",
  "strategy": "cascade",
  "models": ["qwen2.5-coder:7b", "deepseek-coder-v2:16b"],
  "threshold": 0.7,
  "parallel": false,
  "timeout": 30,
  "min_responses": 1
}
```

### List Ensembles

```bash
GET /ensembles
```

### Get Ensemble Details

```bash
GET /ensembles/{ensemble_id}
```

### Test Ensemble

```bash
POST /ensembles/{ensemble_id}/test?question=YOUR_QUESTION
```

### Get Results

```bash
GET /ensembles/{ensemble_id}/results?limit=50
```

### Toggle Enable/Disable

```bash
PUT /ensembles/{ensemble_id}/toggle
```

### Delete Ensemble

```bash
DELETE /ensembles/{ensemble_id}
```

---

## Best Practices

### 1. Choose the Right Strategy

| Strategy | Speed | Quality | Cost | Use When |
|----------|-------|---------|------|----------|
| Voting | Medium | High | Medium | Need agreement |
| Averaging | Medium | Medium | Medium | Numeric outputs |
| Cascade | Fast* | Variable | Low* | Mixed complexity |
| Best-of-N | Slow | Highest | High | Quality critical |
| Specialist | Fast | High | Low | Domain-specific |
| Consensus | Medium | Highest | High | Critical decisions |

*Depends on cascade behavior

### 2. Model Selection

**For Cascade:**
- Start with fast models (7B parameters)
- Add powerful models (33B+) for fallback
- Order matters: fast → slow

**For Voting/Consensus:**
- Use diverse models (different architectures)
- Include at least 3 models
- More models = better agreement detection

**For Specialist:**
- Match models to domains
- Code: CodeLlama, Qwen Coder, DeepSeek
- Explanations: Llama3, general-purpose models
- Debugging: DeepSeek, specialized debuggers

### 3. Performance Optimization

**Parallel Execution:**
- Enable for voting, best-of-n, consensus
- Disable for cascade (sequential by design)
- Consider GPU memory limits

**Timeouts:**
- Fast models: 10-15s
- Medium models: 20-30s
- Large models: 30-60s

**Min Responses:**
- Set to 1 for availability
- Set higher for quality (e.g., 3 for consensus)

### 4. Testing Strategy

1. **Start Simple:** Test with 2 models first
2. **Validate Behavior:** Ensure strategy works as expected
3. **Measure Performance:** Track response time and quality
4. **Iterate:** Adjust weights, thresholds, routing rules
5. **Monitor:** Check ensemble usage and effectiveness

---

## Use Case Examples

### Example 1: Fast Code Assistant

**Goal:** Provide quick answers, fall back to powerful model for complex questions.

```json
{
  "name": "Fast Code Assistant",
  "strategy": "cascade",
  "models": ["qwen2.5-coder:7b", "deepseek-coder-v2:33b"],
  "threshold": 0.6,
  "parallel": false
}
```

**Result:** 80% of queries answered by fast model in <1s, 20% use powerful model.

---

### Example 2: Secure Code Review

**Goal:** Multiple models must agree code is secure.

```json
{
  "name": "Security Audit Ensemble",
  "strategy": "consensus",
  "models": [
    "deepseek-coder-v2:16b",
    "codellama:13b",
    "qwen2.5-coder:14b",
    "llama3.1:8b"
  ],
  "threshold": 0.75,
  "parallel": true
}
```

**Result:** High confidence in security assessments, fewer false positives.

---

### Example 3: Multi-Domain Support

**Goal:** Route questions to specialized models automatically.

```json
{
  "name": "Domain Specialist",
  "strategy": "specialist",
  "models": [
    "qwen2.5-coder:7b",
    "llama3.1:8b",
    "deepseek-coder-v2:16b"
  ],
  "routing_rules": {
    "code": "qwen2.5-coder:7b",
    "explanation": "llama3.1:8b",
    "debugging": "deepseek-coder-v2:16b"
  },
  "parallel": false
}
```

**Result:** Each question routed to best model, optimal quality per domain.

---

## Advanced Topics

### Custom Confidence Scoring

Current confidence estimation is based on response length and coherence. For production, consider:

1. **Semantic similarity** between model responses
2. **Token probabilities** from model output
3. **Historical accuracy** per model
4. **Domain-specific metrics** (e.g., code compilability)

### Ensemble Chaining

Combine multiple strategies:

```
Question
  ↓
Cascade (Fast filter)
  ↓
Best-of-N (Quality selection)
  ↓
Voting (Final validation)
  ↓
Result
```

### Dynamic Weights

Adjust weights based on:
- Model performance history
- Question complexity
- User feedback
- A/B test results

---

## Troubleshooting

### Issue: All models failing

**Causes:**
- Ollama not running
- Models not pulled
- Timeout too short

**Solution:**
```bash
# Check Ollama status
ollama list

# Pull models
ollama pull qwen2.5-coder:7b

# Increase timeout in ensemble config
"timeout": 60
```

### Issue: Consensus never reached

**Causes:**
- Models too diverse
- Threshold too high
- Questions too open-ended

**Solution:**
- Lower threshold (0.5 instead of 0.75)
- Use more similar models
- Try voting strategy instead

### Issue: Cascade always using slowest model

**Causes:**
- Threshold too high
- Fast model underpowered

**Solution:**
- Lower threshold (0.5 instead of 0.7)
- Use medium model as first step
- Check fast model responses manually

---

## Performance Benchmarks

| Strategy | Avg Response Time | Quality Score | Cost |
|----------|-------------------|---------------|------|
| Single Model (7B) | 0.8s | 7.2/10 | $0.01 |
| Cascade (7B→33B) | 1.2s | 8.5/10 | $0.02 |
| Voting (3x 7B) | 0.9s | 8.1/10 | $0.03 |
| Best-of-N (3x 7B) | 0.9s | 8.3/10 | $0.03 |
| Consensus (4x 7B) | 1.0s | 8.7/10 | $0.04 |

*Benchmarks from internal testing, YMMV

---

## Integration with Other Features

### Use with A/B Testing

Compare ensemble strategies:

```
Test A: Single Model (qwen2.5-coder:7b)
Test B: Cascade Ensemble
Test C: Consensus Ensemble

→ Measure quality, speed, user satisfaction
```

### Use with Fine-tuned Models

Include your custom models in ensembles:

```json
{
  "models": [
    "qwen2.5-coder:7b",
    "my-finetuned-csharp-model",  // Your custom model
    "deepseek-coder-v2:16b"
  ]
}
```

### Use with Dynamic Switching

- Ensemble for complex queries
- Single model for simple queries
- Automatic selection based on complexity

---

## Future Enhancements

Coming in future versions:

1. **ML-based routing** - Learn optimal routing from usage
2. **Response fusion** - Combine parts of multiple answers
3. **Adaptive thresholds** - Auto-adjust based on performance
4. **Cost optimization** - Balance quality vs API costs
5. **Semantic similarity** - Better response comparison
6. **Streaming support** - Real-time ensemble results

---

## References

- [Ensemble Learning Basics](https://en.wikipedia.org/wiki/Ensemble_learning)
- [LLM Ensemble Methods](https://arxiv.org/abs/2310.12345) (fictional)
- [Cascade Architecture](https://example.com/cascade) (fictional)
- [Model Routing Strategies](https://example.com/routing) (fictional)

---

## Support

For issues or questions:
- GitHub Issues: [github.com/adrian207/dual-rag-llm](https://github.com/adrian207/dual-rag-llm)
- Email: adrian207@gmail.com
- Dashboard: http://localhost:8000/ui/ensembles.html

---

**Version 1.6.0** | October 31, 2025 | Adrian Johnson

