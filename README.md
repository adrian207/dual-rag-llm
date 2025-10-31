# Dual RAG LLM System

[![Version](https://img.shields.io/badge/version-1.20.0-blue.svg)](https://github.com/adrian207/dual-rag-llm/releases)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-20.10+-blue.svg)](https://www.docker.com/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

> **Enterprise-grade RAG system with 12 major features, 10 language support, and production-ready Azure deployment**

**Author:** Adrian Johnson <adrian207@gmail.com>

An intelligent, production-ready RAG (Retrieval-Augmented Generation) system featuring automatic model selection, comprehensive quality assurance, and multi-language support. Built for scalability with Azure deployment guides and enterprise features including A/B testing, ensemble strategies, and factuality checking.

## Table of Contents

- [Key Features](#key-features)
- [Architecture](#architecture-overview)
- [Quick Start](#quick-start)
- [Azure Deployment](#️-azure-deployment)
- [Documentation](#documentation)
- [Usage](#usage)
- [Configuration](#configuration-reference)
- [Contributing](#contributing)
- [License](#license)

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

### 🗄️ Operations & Resilience (v1.20.0)
- **PostgreSQL Vector Database**: Scalable vector storage with pgvector extension
- **Automated Backup System**: Kubernetes CronJob for scheduled backups (every 6 hours)
- **Disaster Recovery Automation**: Automated DR procedures with continuous health monitoring
- **RPO & RTO Targets**: 60-minute RPO, 15-minute RTO with automated recovery
- **Backup Verification**: SHA-256 checksum validation and integrity checks
- **DR Scripts**: Comprehensive bash scripts for backup, restore, and DR management
- **Point-in-Time Recovery**: Transaction log-based recovery capabilities
- **Backup Retention**: 30-day retention with configurable policies (up to 100 backups)
- **Health Monitoring**: Continuous monitoring (60-second intervals) of database, backups, and disk space
- **Automatic Failover**: Intelligent recovery with manual and automatic modes

### ☸️ Production-Ready Kubernetes Deployment (v1.19.0)
- **Complete Helm Chart**: Production-grade Helm chart with 15+ templates
- **Auto-scaling**: HPA with CPU/memory-based scaling (2-10 replicas)
- **High Availability**: Pod anti-affinity, Pod Disruption Budget, multi-replica setup
- **Persistent Storage**: PVC support for data and Ollama models
- **Ingress & TLS**: NGINX ingress with cert-manager integration
- **Monitoring**: Prometheus ServiceMonitor, health probes (liveness/readiness/startup)
- **Security**: Network policies, security contexts, secrets management
- **GPU Support**: NVIDIA GPU integration for Ollama
- **Redis Integration**: Bitnami Redis chart dependency
- **One-Command Deployment**: Automated deployment script

### 💰 Cost Tracking & Budget Management (v1.18.0)
- **Comprehensive Cost Tracking**: Track all query costs with token-level precision
- **Model Pricing Database**: Pre-configured pricing for Ollama (free) and cloud providers (comparison)
- **Cost Analytics**: Daily trends, cost by model, cost by query type, top cost queries
- **Budget Alerts**: Set daily/weekly/monthly budgets with threshold notifications
- **Cost Forecasting**: 30/90-day cost predictions based on historical trends
- **Savings Calculator**: Compare Ollama costs vs cloud provider costs
- **Cost Breakdown**: Input tokens, output tokens, and request-level costs
- **12 API Endpoints**: Complete cost management and analysis

### 🎛️ Enterprise Admin Dashboard (v1.17.0)
- **Comprehensive Management Interface**: Professional React-based admin dashboard with 6 major sections
- **System Overview**: Real-time monitoring of services, cached models, and recent activity
- **Audit Log Viewer**: Advanced filtering, search, export to JSON/CSV, severity badges
- **Encryption Management**: Encrypt/decrypt tools, key rotation, key generation, status monitoring
- **Model Performance**: Usage statistics, response times, success rates, cache hit rates per model
- **Configuration Editor**: Live editing of analytics, audit logging, and language configurations
- **Analytics Dashboard**: Comprehensive reports with insights, query patterns, and performance metrics
- **Responsive Design**: Dark mode support, modern Tailwind CSS styling
- **Real-time Updates**: Auto-refreshing data with React Query

### 📊 Usage Analytics (v1.16.0)
- **Comprehensive Tracking**: Query patterns, model usage, API calls, cache efficiency
- **Time-Series Data**: Historical trends with configurable time periods (hour/day/week/month/year)
- **Performance Metrics**: P50/P95/P99 latency, uptime percentage, error rates
- **Model Analytics**: Usage counts, response times, success rates, token counts, cache hit rates
- **Query Analytics**: Peak hours, query types, queries per hour/day, success rates
- **Automatic Insights**: AI-generated recommendations based on usage patterns
- **8 API Endpoints**: Complete analytics management and reporting
- **Configurable Retention**: 90-day default with automatic cleanup

### 🔐 Data Encryption at Rest & in Transit (v1.15.0)
- **AES-256 Encryption**: Military-grade encryption via Fernet
- **Data-at-Rest**: Encrypt sensitive data in databases and storage
- **Data-in-Transit**: TLS 1.2+ with configurable cipher suites
- **Field-Level Encryption**: Selective encryption of sensitive fields
- **Key Management**: Master key generation, derivation, and rotation
- **Password Hashing**: PBKDF2 with SHA-256 (100K iterations)
- **TLS/HTTPS Support**: Full SSL/TLS configuration for secure communication
- **mTLS Support**: Mutual TLS with client certificate verification
- **Key Rotation**: Periodic key updates without service disruption
- **7 API Endpoints**: Complete encryption management

### 🔒 Enterprise Audit Logging (v1.14.0)
- **24 Event Types**: API requests, queries, system events, security events, data operations
- **5 Severity Levels**: Debug, Info, Warning, Error, Critical
- **Comprehensive Tracking**: Timestamps, IP addresses, endpoints, durations, error messages
- **Advanced Filtering**: By event type, severity, user, date range, endpoint, duration
- **Export Capabilities**: JSON and CSV formats with filters
- **Configurable Retention**: Automatic log rotation with configurable limits (default: 90 days, 100K logs)
- **Statistics Dashboard**: Event counts, error rates, average durations, top endpoints
- **Privacy Controls**: Configurable logging of request bodies and response data
- **Compliance Ready**: Complete audit trail for security and regulatory requirements

### ⚛️ Modern React Frontend (v1.13.0)
- **React 18 + TypeScript**: Modern, type-safe development
- **Vite Build System**: Lightning-fast development and optimized production builds
- **Tailwind CSS**: Beautiful, responsive design with dark mode support
- **Real-time Streaming**: Token-by-token response display with EventSource API
- **State Management**: Zustand for efficient global state
- **React Query**: Smart data fetching with caching
- **Syntax Highlighting**: Code blocks with 22+ language support
- **Markdown Rendering**: Rich text formatting with react-markdown
- **Component Architecture**: Modular, reusable components
- **Production Ready**: Optimized bundle splitting and lazy loading

### 🌍 Multi-language Support (v1.12.0)
- **10 Languages Supported**: English, Spanish, French, German, Japanese, Chinese (Simplified), Portuguese, Russian, Italian, Korean
- **Auto Language Detection**: Recognize input language automatically
- **LLM-Powered Translation**: High-quality translation using Ollama
- **Translation Caching**: Fast repeated translations
- **UI Localization**: 5 UI language packs (expandable)
- **Confidence Scoring**: Translation quality metrics
- **Native Language Names**: Display languages in their native script

### 💻 Code Syntax Highlighting (v1.11.0)
- **22 Languages Supported**: Python, JavaScript, TypeScript, Java, C#, C, C++, Go, Rust, PHP, Ruby, Swift, Kotlin, **PowerShell**, Bash, SQL, HTML, CSS, JSON, YAML, XML, Markdown
- **9 Themes**: VS Dark, VS Light, GitHub Dark, GitHub Light, Monokai, Dracula, Nord, Solarized Dark/Light
- **Advanced Detection**: Pattern-based language detection with 95% confidence
- **Auto-Enhancement**: Automatic language hints for code blocks
- **Metadata Extraction**: Comments, imports, complexity analysis
- **Syntax Error Detection**: Basic validation for common issues
- **Real-time Highlighting**: highlight.js + marked.js integration

### 🎨 Response Formatting (v1.10.0)
- **6 Formatting Styles**: Plain, Markdown, Structured, Professional, Concise, Detailed
- **Auto-Sections**: Intelligent header insertion for long responses
- **Code Enhancement**: Language detection and syntax hints
- **List Creation**: Convert text to structured lists
- **Readability**: Line length optimization and sentence breaking
- **Emoji Headers**: Optional visual enhancement
- **Improvement Scoring**: Track formatting quality (0-1)

### 🔍 Factuality Checking (v1.9.0)
- **Claim Extraction**: Automatically identify factual claims in answers
- **Hallucination Detection**: 5 indicators (false confidence, fabricated sources, inconsistent details, etc.)
- **Source Verification**: Cross-reference claims with RAG context
- **Verdict System**: Supported/Contradicted/Uncertain/Unverifiable
- **Risk Scoring**: 0-1 hallucination risk calculation
- **Reliability Assessment**: Binary reliable/unreliable determination
- **Statistics API**: Track factuality across all responses

### ✅ Answer Validation & Verification (v1.8.0)
- **7 Validation Checks**: Factuality, source verification, consistency, completeness, relevance, clarity, code validity
- **Confidence Scoring**: Overall quality score (0-1) with detailed breakdown
- **Source Verification**: Compare answers against RAG context
- **Auto-Suggestions**: Actionable improvements for failed checks
- **Quality Metrics**: Length, structure, examples, code blocks
- **Approval System**: Auto-approve/reject based on threshold
- **Real-time Stats**: Pass rates, approval rates, avg scores

### 🤖 Automatic Model Selection (v1.7.0)
- **12 Query Types**: Intelligent classification (code gen, debug, explain, review, etc.)
- **Smart Routing**: Automatically select best model for each query type
- **Performance Learning**: Adapts based on success rates and user ratings
- **Confidence Scoring**: Keyword-based classification with confidence levels
- **Fallback Logic**: Graceful degradation if primary model unavailable
- **Language Detection**: Auto-detect programming languages
- **Complexity Estimation**: Route simple/complex queries appropriately
- **Interactive Dashboard**: Configure routing, monitor performance, test classification

### 🎯 Model Ensemble Strategies (v1.6.0)
- **6 Ensemble Strategies**: Voting, Averaging, Cascade, Best-of-N, Specialist, Consensus
- **Parallel Execution**: Run multiple models simultaneously for speed
- **Weighted Combinations**: Assign importance to different models
- **Cascade Fallback**: Fast models first, powerful models if needed
- **Specialist Routing**: Auto-route questions to domain experts
- **Consensus Building**: Require agreement threshold for critical decisions
- **Performance Tracking**: Monitor ensemble effectiveness
- **Interactive Dashboard**: Create, test, and manage ensembles

### 🔧 Custom Model Fine-tuning (v1.5.0)
- **Dataset Management**: Upload and validate training datasets (Chat, Instruct, Q&A, Completion formats)
- **LoRA/QLoRA Training**: Efficient fine-tuning with Low-Rank Adaptation
- **Job Orchestration**: Create, monitor, and manage training jobs
- **Model Registry**: Version control for fine-tuned models
- **Performance Tracking**: Training/eval loss monitoring
- **One-Click Deployment**: Export to Ollama format
- **Interactive Dashboard**: Complete UI for managing pipeline
- **Hyperparameter Configuration**: Learning rate, epochs, batch size, LoRA params

### 🧪 A/B Testing Framework (v1.4.0)
- **Systematic Testing**: Compare models with statistical rigor
- **Traffic Splitting**: Route % of queries to each model (customizable)
- **Statistical Analysis**: T-tests, confidence intervals, significance testing
- **Automatic Winner Detection**: Declare winners based on multiple metrics
- **Performance Tracking**: Response time, tokens/sec, user ratings
- **Test Management**: Create, start, pause, resume, complete tests
- **Results Dashboard**: Visualize statistics and track progress
- **Rating System**: 1-5 star ratings for subjective quality assessment

### 🔄 Dynamic Model Switching (v1.3.0)
- **Model Selector**: Choose from 10+ available LLMs (Qwen, DeepSeek, CodeLlama, Llama 3.1)
- **Model Comparison**: Run parallel queries across multiple models
- **Performance Tracking**: Real-time metrics (tokens/sec, response time, query count)
- **Try Another Model**: One-click retry with different models
- **Automatic Fallback**: Seamless fallback when models unavailable
- **Side-by-Side Comparison**: Visual comparison with winner highlighting

### 🚀 Performance (v1.1.0)
- **Redis caching** - 80-90% faster responses for cached queries (<100ms)
- **Async/await throughout** - Non-blocking I/O operations
- **Index caching** - Avoid repeated disk reads
- **Model verification** - Skip redundant model loads
- **HTTP connection pooling** - Reuse connections to Ollama
- **Startup preloading** - Warm caches during initialization

### 🌐 Web Tools (v1.1.0)
- **Brave Search integration** - Web search for current information
- **GitHub API integration** - Code search across repositories
- **Combined tool usage** - RAG + Web + GitHub in single query
- **Tool statistics** - Track usage and performance

### 🎨 Interactive UI (v1.2.0)
- **Real-time streaming** - Token-by-token response generation
- **Modern web interface** - Dark theme, responsive design
- **Live statistics** - Cache hit rate, query counts
- **Tool toggles** - One-click web search and GitHub integration
- **Chat-style layout** - Familiar conversation interface
- **Markdown formatting** - Code blocks, syntax highlighting

### 🛡️ Robustness
- **Comprehensive error handling** - Graceful degradation
- **Health checks** - Container orchestration readiness
- **Service dependencies** - Proper startup ordering
- **Structured logging** - JSON output for monitoring
- **Input validation** - Pydantic models with constraints

### 🧠 Intelligence
- **Context-aware routing** - File extension detection
- **Dual knowledge bases** - Specialized documentation
- **Top-K retrieval** - Most relevant context chunks
- **Model specialization** - Domain-specific LLMs
- **Multi-source synthesis** - Combines RAG, web, and code sources

## Prerequisites

- Docker 20.10+
- Docker Compose 2.0+
- NVIDIA GPU with CUDA support
- NVIDIA Container Toolkit
- 32GB+ RAM recommended
- 100GB+ free disk space (for models)

## ☁️ Azure Deployment

**Want to deploy to Microsoft Azure?** We have a complete Azure deployment branch with:

- **Azure VM Deployment** - Simplest option (~$900/month)
- **Azure Kubernetes Service (AKS)** - Production-ready (~$960/month)
- **Complete Kubernetes manifests** with GPU support
- **Automated deployment scripts**
- **Model update strategies** (rolling, blue-green, automated)
- **Cost optimization guides** (save 60-70% with scheduling)

### Quick Azure Deploy

```bash
# Clone the Azure deployment branch
git checkout azure/deployment

# Option 1: Deploy to Azure VM (simplest)
cd azure
# Follow VM_DEPLOYMENT.md

# Option 2: Deploy to AKS (production)
./azure/scripts/deploy-to-aks.sh
```

**Documentation:**
- [Azure Deployment Guide](https://github.com/adrian207/dual-rag-llm/tree/azure/deployment/azure) - Complete guide
- [Model Update Strategies](https://github.com/adrian207/dual-rag-llm/tree/azure/deployment/azure/AZURE_DEPLOYMENT.md#model-update-strategy) - Rolling, blue-green, automated

**Key Features:**
- ✅ Native Azure support (VM, AKS, ACI)
- ✅ Full GPU support (NC-series VMs, T4/V100)
- ✅ Zero-downtime model updates
- ✅ Auto-scaling and high availability
- ✅ Cost optimization (scheduled shutdown, spot instances)

See the [`azure/deployment` branch](https://github.com/adrian207/dual-rag-llm/tree/azure/deployment) for full Azure infrastructure.

---

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

- **React Frontend** ⚛️: http://localhost:3001 (v1.13.0 - Modern UI)
- **Interactive UI**: http://localhost:8000/ui (Legacy HTML/JS)
- **RAG API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Open-WebUI**: http://localhost:3000 (optional)
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

## Documentation

### 📚 Complete Guides

- **[CHANGELOG.md](CHANGELOG.md)** - Complete release history for all 20 versions
- **[User Guide](docs/USER_GUIDE.md)** - Comprehensive 870+ line user manual
- **[API Reference](docs/API_REFERENCE.md)** - Complete API documentation (90+ endpoints)
- **[Contributing Guide](CONTRIBUTING.md)** - How to contribute
- **[Roadmap](ROADMAP.md)** - Future features and plans
- **[Disaster Recovery Guide](docs/DISASTER_RECOVERY.md)** - 690+ line comprehensive DR procedures

### 🚀 Deployment Guides

- **[Quick Start](QUICKSTART.md)** - Get running in 5 minutes
- **[Azure Deployment](azure/README.md)** - Complete Azure guide
- **[AKS Guide](azure/AKS_DEPLOYMENT.md)** - Kubernetes deployment
- **[Kubernetes Guide](k8s/README.md)** - Production Kubernetes deployment
- **[PowerShell Scripts](azure/POWERSHELL_DEPLOYMENT.md)** - Windows deployment
- **[Model Ensembles](docs/MODEL_ENSEMBLES.md)** - Ensemble strategies guide

### 🔧 Feature Documentation

- **Dynamic Model Switching** - `docs/DYNAMIC_MODEL_SWITCHING.md`
- **Web Tools Integration** - `docs/WEB_TOOLS_GUIDE.md` 
- **Development Workflow** - `DEVELOPMENT_WORKFLOW.md`

## Version History

**Current Version:** 1.20.0 (October 31, 2024)

See [CHANGELOG.md](CHANGELOG.md) for complete release notes.

**Recent Releases:**
- **v1.20.0** - Operations & Resilience (PostgreSQL, backup, DR automation)
- **v1.19.0** - Kubernetes Deployment (Helm charts, auto-scaling, HA, GPU support)
- **v1.18.0** - Cost Tracking & Budget Management (forecasting, alerts, analytics)
- **v1.17.0** - Enterprise Admin Dashboard (comprehensive management interface)
- **v1.16.0** - Usage Analytics (comprehensive tracking, insights, time-series data)

[View all 20 releases →](CHANGELOG.md)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for:
- Code of Conduct
- Development setup
- Coding standards
- Pull request process
- Release guidelines

**Quick Start:**
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Follow coding standards (PEP 8, type hints, docstrings)
4. Add tests for new features
5. Submit a pull request

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## Support

### Getting Help

- **📖 Documentation**: Start with [User Guide](docs/USER_GUIDE.md) or [API Reference](docs/API_REFERENCE.md)
- **🐛 Bug Reports**: [GitHub Issues](https://github.com/adrian207/dual-rag-llm/issues)
- **💡 Feature Requests**: [GitHub Issues](https://github.com/adrian207/dual-rag-llm/issues)
- **📧 Email**: adrian207@gmail.com
- **💬 Discussions**: GitHub Discussions (coming soon)

### Troubleshooting

See the [User Guide - Troubleshooting](docs/USER_GUIDE.md#troubleshooting) section for common issues and solutions.

## Acknowledgments

This project builds upon excellent open-source work:

- **[Ollama](https://ollama.ai/)** - LLM inference server
- **[ChromaDB](https://www.trychroma.com/)** - Vector database
- **[FastAPI](https://fastapi.tiangolo.com/)** - Modern web framework
- **[HuggingFace](https://huggingface.co/)** - Embedding models
- **[Qwen Team](https://github.com/QwenLM/Qwen)** - Qwen 2.5 Coder models
- **[DeepSeek](https://www.deepseek.com/)** - DeepSeek Coder models
- **[highlight.js](https://highlightjs.org/)** - Syntax highlighting
- **[marked.js](https://marked.js.org/)** - Markdown parsing

## Project Stats

- **20 Major Versions** released
- **PostgreSQL Vector Database** with pgvector
- **Automated Backup & DR** with Kubernetes CronJobs
- **Production-Ready Kubernetes** with Helm charts
- **Cost Tracking & Forecasting** with budget alerts
- **Enterprise Admin Dashboard** with 7 management sections
- **Comprehensive Usage Analytics** with AI insights
- **Modern React Frontend** with TypeScript
- **Enterprise-Grade Encryption** (AES-256 + TLS 1.2+)
- **Enterprise Audit Logging** with 24 event types
- **22 Programming Languages** supported (syntax highlighting)
- **10 Human Languages** supported (UI + translation)
- **120+ API Endpoints** documented
- **8,500+ Lines** of production code
- **3,500+ Lines** of documentation
- **3+ Billion** potential users (language reach)

## Citation

If you use this project in your research or production, please cite:

```bibtex
@software{dual_rag_llm_2024,
  author = {Johnson, Adrian},
  title = {Dual RAG LLM System: Enterprise RAG with Multi-language Support},
  year = {2024},
  version = {1.19.0},
  url = {https://github.com/adrian207/dual-rag-llm}
}
```

---

**Built with ❤️ by Adrian Johnson**  
**Last Updated:** October 31, 2024 | **Version:** 1.19.0

