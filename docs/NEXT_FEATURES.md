# Next Features to Implement

**Author:** Adrian Johnson <adrian207@gmail.com>

Priority-ranked features from the roadmap for version 1.1.0.

## 🚀 Immediate Priority (Next Sprint)

### 1. Redis Caching Layer ⭐ TOP PRIORITY

**Impact**: 80%+ faster responses for cached queries  
**Complexity**: Medium  
**Time**: 1-2 days  
**Branch**: main → azure/deployment

**Why First:**
- Biggest performance impact
- Reduces LLM costs
- Improves user experience
- Foundation for other features

**Implementation Plan:**

```
Tasks:
1. Add Redis to docker-compose.yml
2. Add Redis to Kubernetes manifests (Azure branch)
3. Add redis, aioredis to requirements.txt
4. Implement caching middleware in rag_dual.py
5. Add cache invalidation logic
6. Add TTL configuration
7. Add cache statistics endpoint
8. Update documentation

Files to Modify (Main):
- docker-compose.yml (add Redis service)
- rag/requirements.txt (add dependencies)
- rag/rag_dual.py (caching logic)
- README.md (update architecture)

Files to Add (Azure):
- azure/k8s/redis-deployment.yaml
- azure/k8s/redis-service.yaml
```

**Acceptance Criteria:**
- ✅ Cache hit rate > 80% for duplicate queries
- ✅ Response time < 100ms for cached queries
- ✅ Cache invalidation on index rebuild
- ✅ Works on both Docker Compose and AKS
- ✅ Monitoring endpoint shows cache statistics

---

### 2. Query Result Streaming

**Impact**: Better UX, perceived performance improvement  
**Complexity**: Medium  
**Time**: 1 day  
**Branch**: main → azure/deployment

**Why Second:**
- Improves user experience significantly
- Works well with caching
- Modern API expectation

**Implementation:**
- Use FastAPI's StreamingResponse
- Stream LLM output tokens
- Update WebUI to handle streaming
- Works with both cached and uncached queries

---

### 3. Prometheus Metrics

**Impact**: Production observability  
**Complexity**: Medium  
**Time**: 1-2 days  
**Branch**: main → azure/deployment

**Why Third:**
- Essential for production
- Complements caching
- Enables data-driven optimization

**Metrics to Track:**
- Query latency (p50, p95, p99)
- Cache hit rate
- Model inference time
- Embedding generation time
- Error rates
- Token usage

---

## 🎯 Short-Term (This Month)

### 4. Batch Embedding Generation
- Process multiple documents in parallel
- Reduce indexing time by 70%
- Better resource utilization

### 5. Connection Pooling Improvements
- Optimize HTTP connections
- Reduce overhead
- Better concurrency

### 6. Memory Profiling and Optimization
- Identify memory leaks
- Optimize cache sizes
- Better resource limits

---

## 📊 Medium-Term (Next Month)

### 7. Grafana Dashboards
- Visual monitoring
- Real-time metrics
- Alerting

### 8. Model Performance Comparison
- A/B testing framework
- Track model quality
- Optimize routing

### 9. Index Optimization
- Faster retrieval
- Better relevance
- Reduced storage

---

## 🔄 Continuous Improvements

### Documentation
- [ ] Video tutorials
- [ ] API examples expansion
- [ ] Troubleshooting guide
- [ ] Performance tuning guide

### Testing
- [ ] Unit test coverage (target 80%+)
- [ ] Integration tests
- [ ] Load testing
- [ ] Benchmark suite

### DevOps
- [ ] Auto-merge workflow (main → azure)
- [ ] Automated testing pipeline
- [ ] Performance regression tests
- [ ] Security scanning

---

## Implementation Order

```
Week 1: Redis Caching Layer
├── Day 1-2: Core implementation (main branch)
├── Day 3: Azure Kubernetes configs
├── Day 4: Testing and documentation
└── Day 5: Deployment and validation

Week 2: Streaming + Metrics
├── Day 1-2: Query result streaming
├── Day 3-4: Prometheus metrics
└── Day 5: Integration and testing

Week 3: Optimizations
├── Day 1-2: Batch embeddings
├── Day 3: Connection pooling
└── Day 4-5: Memory optimization

Week 4: Monitoring
├── Day 1-3: Grafana dashboards
├── Day 4: Model comparison
└── Day 5: Index optimization planning
```

---

## Technical Specifications

### Redis Caching Architecture

```python
Cache Strategy:
- Key: hash(question + file_ext + model_version)
- Value: {answer, model, source, chunks, timestamp}
- TTL: 24 hours (configurable)
- Invalidation: On index rebuild, manual trigger

Cache Tiers:
1. Query Response Cache (Redis)
   - Full API responses
   - 24h TTL
   
2. Embedding Cache (Redis)
   - Question embeddings
   - 7d TTL
   
3. Index Cache (Memory)
   - Already implemented
   - No expiration
```

### Streaming Architecture

```python
Streaming Flow:
1. Client requests with Accept: text/event-stream
2. Server starts streaming immediately
3. Stream retrieval progress
4. Stream LLM tokens as generated
5. Close stream on completion

Benefits:
- Time to first byte: < 500ms
- Perceived latency: -80%
- Better UX for long responses
```

### Prometheus Metrics

```python
Custom Metrics:
- rag_query_duration_seconds (histogram)
- rag_cache_hits_total (counter)
- rag_cache_misses_total (counter)
- rag_llm_tokens_total (counter)
- rag_embedding_duration_seconds (histogram)
- rag_retrieval_duration_seconds (histogram)
- rag_errors_total (counter by type)
- rag_active_queries (gauge)

Standard Metrics:
- CPU, memory, disk I/O
- Network traffic
- GPU utilization
- Request rates
```

---

## Success Metrics

### Version 1.1.0 Goals

| Metric | Current | Target | Improvement |
|--------|---------|--------|-------------|
| Avg Query Time | 8s | 1s | 87% faster |
| P95 Query Time | 15s | 3s | 80% faster |
| Cache Hit Rate | 0% | 80% | New |
| Cached Response Time | N/A | <100ms | New |
| Memory Usage | ~8GB | ~4GB | 50% reduction |
| Concurrent Queries | 5 | 50 | 10x |

---

## Decision: Let's Start with Redis Caching!

**Ready to implement?** This will be a dual-branch feature:
1. Core caching in `main` branch
2. Azure Redis/AKS configs in `azure/deployment` branch

Shall I proceed with implementing Redis caching?

