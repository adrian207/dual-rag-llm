# Dual RAG LLM System

**Author:** Adrian Johnson <adrian207@gmail.com>

An intelligent RAG (Retrieval-Augmented Generation) system that automatically routes queries to specialized Large Language Models and knowledge bases based on technology context. The system distinguishes between Microsoft technologies (C#, PowerShell, YAML, XAML) and Open Source technologies, providing optimized responses for each domain.

## Architecture Overview

The system consists of three containerized services orchestrated by Docker Compose:

**1. Ollama Service**
   - GPU-accelerated LLM inference server
   - Hosts Qwen 2.5 Coder (32B) for Microsoft technologies
   - Hosts DeepSeek Coder V2 (33B) for Open Source technologies

**2. RAG Service (FastAPI)**
   - Intelligent query routing based on file extensions
   - Dual ChromaDB vector stores (MS docs + OSS docs)
   - HuggingFace embeddings (all-MiniLM-L6-v2)
   - Async operations with caching for performance
   - Comprehensive error handling and logging

**3. Open-WebUI**
   - User-friendly web interface
   - Direct integration with Ollama
   - Optional connection to RAG service

### Request Flow

```
User Query → File Extension Detection → Route Decision
                                        ↓
    ┌──────────────────────────────────┴──────────────────────────────────┐
    │                                                                      │
MS Path (.cs, .ps1, .yaml, .yml, .xaml)          OSS Path (other)
    │                                                                      │
    ├─→ Qwen 2.5 Coder (32B)                     ├─→ DeepSeek Coder V2 (33B)
    ├─→ ChromaDB: msdocs collection              ├─→ ChromaDB: ossdocs collection
    ├─→ Microsoft documentation                  ├─→ Open source documentation
    │                                                                      │
    └──────────────────────────────────┬──────────────────────────────────┘
                                        ↓
                            Contextual Response
```

## Key Features

### Performance Optimizations
- **Async/await throughout** - Non-blocking I/O operations
- **Index caching** - Avoid repeated disk reads
- **Model verification** - Skip redundant model loads
- **HTTP connection pooling** - Reuse connections to Ollama
- **Startup preloading** - Warm caches during initialization

### Robustness
- **Comprehensive error handling** - Graceful degradation
- **Health checks** - Container orchestration readiness
- **Service dependencies** - Proper startup ordering
- **Structured logging** - JSON output for monitoring
- **Input validation** - Pydantic models with constraints

### Intelligence
- **Context-aware routing** - File extension detection
- **Dual knowledge bases** - Specialized documentation
- **Top-K retrieval** - Most relevant context chunks
- **Model specialization** - Domain-specific LLMs

## Prerequisites

- Docker 20.10+
- Docker Compose 2.0+
- NVIDIA GPU with CUDA support
- NVIDIA Container Toolkit
- 32GB+ RAM recommended
- 100GB+ free disk space (for models)

## Quick Start

### 1. Clone and Setup

```bash
git clone <repository-url>
cd dual-rag-llm
```

### 2. Configure Environment

```bash
cp env.example .env
# Edit .env and change WEBUI_SECRET_KEY
```

### 3. Prepare Documentation

Place your documentation files in the appropriate directories:

```bash
mkdir -p rag/data/ms_docs rag/data/oss_docs

# Microsoft documentation (.pdf, .md, .txt, .html, .docx)
cp /path/to/csharp-docs/* rag/data/ms_docs/
cp /path/to/powershell-docs/* rag/data/ms_docs/

# Open Source documentation
cp /path/to/python-docs/* rag/data/oss_docs/
cp /path/to/javascript-docs/* rag/data/oss_docs/
```

**Note:** If directories are empty, sample documents will be created automatically for testing.

### 4. Pull LLM Models

Before starting the RAG service, pull the required models:

```bash
# Start only Ollama first
docker compose up -d ollama

# Pull models (this takes time - ~20GB each)
docker exec ollama ollama pull qwen2.5-coder:32b-q4_K_M
docker exec ollama ollama pull deepseek-coder-v2:33b-q4_K_M

# Verify models
docker exec ollama ollama list
```

### 5. Start All Services

```bash
docker compose up -d
```

### 6. Verify Services

```bash
# Check service health
docker compose ps

# Check RAG service health
curl http://localhost:8000/health

# View logs
docker compose logs -f rag
```

### 7. Access Interfaces

- **Open-WebUI**: http://localhost:3000
- **RAG API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Ollama API**: http://localhost:11434

## Usage

### Via API

#### Query Endpoint

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How do I create a list in C#?",
    "file_ext": ".cs"
  }'
```

**Response:**
```json
{
  "answer": "To create a list in C#, use the List<T> class...",
  "model": "qwen2.5-coder:32b-q4_K_M",
  "source": "Microsoft",
  "chunks_retrieved": 3
}
```

#### Health Check

```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "ollama_connected": true,
  "ms_index_loaded": true,
  "oss_index_loaded": true,
  "models_available": [
    "qwen2.5-coder:32b-q4_K_M",
    "deepseek-coder-v2:33b-q4_K_M"
  ]
}
```

#### System Statistics

```bash
curl http://localhost:8000/stats
```

### Via Open-WebUI

1. Navigate to http://localhost:3000
2. Create an account (first user becomes admin)
3. Select your model from the dropdown
4. Start chatting

### File Extension Routing

The system routes based on file extensions in the query:

| Extension | Model | Knowledge Base |
|-----------|-------|----------------|
| `.cs` | Qwen 2.5 Coder | Microsoft Docs |
| `.ps1` | Qwen 2.5 Coder | Microsoft Docs |
| `.yaml`, `.yml` | Qwen 2.5 Coder | Microsoft Docs |
| `.xaml` | Qwen 2.5 Coder | Microsoft Docs |
| Others | DeepSeek Coder V2 | OSS Docs |

## Document Management

### Adding New Documents

1. **Add files to appropriate directory:**
   ```bash
   cp new-docs/* rag/data/ms_docs/
   ```

2. **Rebuild indexes:**
   ```bash
   docker compose exec rag python ingest_docs.py
   ```

3. **Restart RAG service to reload:**
   ```bash
   docker compose restart rag
   ```

### Supported Document Formats

- PDF (`.pdf`)
- Markdown (`.md`)
- Text (`.txt`)
- HTML (`.html`)
- reStructuredText (`.rst`)
- Word Documents (`.docx`)

### Manual Index Building

```bash
# Create sample documents
docker compose exec rag python ingest_docs.py --create-samples

# Build all indexes
docker compose exec rag python ingest_docs.py

# Build specific index (modify script as needed)
```

## Monitoring and Maintenance

### View Logs

```bash
# All services
docker compose logs -f

# Specific service
docker compose logs -f rag
docker compose logs -f ollama

# With timestamps
docker compose logs -f --timestamps
```

### Resource Usage

```bash
# Container stats
docker stats

# GPU usage
nvidia-smi
```

### Clear Caches

```bash
# Remove indexes (will rebuild on next start)
rm -rf rag/indexes/*

# Remove models from Ollama
docker compose exec ollama ollama rm qwen2.5-coder:32b-q4_K_M
```

### Backup Data

```bash
# Backup indexes
tar -czf indexes-backup.tar.gz rag/indexes/

# Backup Ollama models
docker run --rm -v dual-rag-llm_ollama_models:/data \
  -v $(pwd):/backup \
  alpine tar -czf /backup/ollama-models-backup.tar.gz /data
```

## Troubleshooting

### Service Won't Start

**Check logs:**
```bash
docker compose logs rag
```

**Common issues:**
- Ollama not ready → Wait for health check
- Models not pulled → Pull models first
- GPU not available → Check NVIDIA Container Toolkit

### Models Not Loading

```bash
# List available models
docker exec ollama ollama list

# Pull missing model
docker exec ollama ollama pull qwen2.5-coder:32b-q4_K_M
```

### Indexes Not Building

```bash
# Check data directories
ls -la rag/data/ms_docs/
ls -la rag/data/oss_docs/

# Manually trigger build
docker compose exec rag python ingest_docs.py --create-samples
docker compose exec rag python ingest_docs.py
```

### Out of Memory

**Reduce model sizes in code:**
- Use smaller quantization (e.g., q4_K_S instead of q4_K_M)
- Reduce context window
- Lower `similarity_top_k`

**Increase Docker resources:**
```bash
# Edit docker-compose.yml, add under each service:
deploy:
  resources:
    limits:
      memory: 16g
```

### Slow Responses

**Check bottlenecks:**
```bash
# CPU usage
docker stats

# GPU usage
nvidia-smi

# Disk I/O
iostat -x 1
```

**Optimizations:**
- Ensure SSD for indexes
- Increase RAM allocation
- Check network latency to Ollama

## Development

### Running Tests

```bash
# Install dev dependencies
pip install pytest pytest-asyncio httpx

# Run tests
pytest tests/
```

### API Documentation

Interactive API docs available at:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Project Structure

```
dual-rag-llm/
├── docker-compose.yml          # Service orchestration
├── env.example                 # Environment template
├── README.md                   # This file
├── rag/
│   ├── Dockerfile              # RAG service container
│   ├── entrypoint.sh           # Startup script
│   ├── requirements.txt        # Python dependencies
│   ├── rag_dual.py             # Main FastAPI application
│   ├── ingest_docs.py          # Document indexing
│   ├── logging_config.py       # Logging setup
│   ├── data/                   # Documentation files
│   │   ├── ms_docs/            # Microsoft docs
│   │   └── oss_docs/           # OSS docs
│   ├── indexes/                # Vector store indexes
│   │   ├── chroma_ms/          # MS index
│   │   └── chroma_oss/         # OSS index
│   └── logs/                   # Application logs
└── workspace/                  # Shared workspace
```

## Configuration Reference

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_API` | `http://ollama:11434` | Ollama API endpoint |
| `LOG_LEVEL` | `INFO` | Logging level |
| `WEBUI_SECRET_KEY` | (required) | Secret key for web UI |
| `WEBUI_AUTH` | `true` | Enable authentication |

### Model Configuration

Edit `rag/rag_dual.py` to customize:

```python
MS_MODEL = "qwen2.5-coder:32b-q4_K_M"
OSS_MODEL = "deepseek-coder-v2:33b-q4_K_M"
MS_EXTENSIONS = {'.cs', '.ps1', '.yaml', '.yml', '.xaml'}
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
```

### RAG Parameters

```python
similarity_top_k = 3          # Number of context chunks
chunk_size = 512              # Token size per chunk
chunk_overlap = 50            # Overlap between chunks
request_timeout = 120.0       # LLM request timeout (seconds)
```

## Performance Benchmarks

[Inference] Based on typical hardware configurations:

| Operation | Latency | Notes |
|-----------|---------|-------|
| Index query | 50-200ms | Depends on index size |
| Embedding generation | 100-300ms | GPU-accelerated |
| LLM inference | 2-10s | Varies by prompt length |
| Total query time | 3-15s | End-to-end |

## Security Considerations

1. **Enable authentication** - Set `WEBUI_AUTH=true`
2. **Change secret keys** - Update `WEBUI_SECRET_KEY`
3. **Network isolation** - Use Docker networks
4. **Input validation** - Pydantic models enforce limits
5. **Rate limiting** - [Unverified] Consider adding nginx reverse proxy
6. **HTTPS** - [Unverified] Use reverse proxy with SSL certificates

## Future Enhancements

- [ ] Response caching with Redis
- [ ] Multi-model ensemble responses
- [ ] Custom model fine-tuning pipeline
- [ ] Metrics dashboard (Prometheus + Grafana)
- [ ] A/B testing framework
- [ ] Streaming responses
- [ ] Multi-language support
- [ ] User feedback collection
- [ ] Query analytics

## License

[Specify your license]

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Submit a pull request

## Support

For issues and questions:
- GitHub Issues: [repository-url]/issues
- Email: adrian207@gmail.com

## Acknowledgments

- Ollama team for the excellent LLM server
- LlamaIndex for the RAG framework
- Open-WebUI for the user interface
- Qwen and DeepSeek teams for the models

