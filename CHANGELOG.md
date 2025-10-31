# Changelog - Dual RAG LLM System

**Author:** Adrian Johnson <adrian207@gmail.com>

All notable changes to this project will be documented in this file.

## [1.0.0] - 2025-10-31

### Fixed
- **Critical**: Renamed `dockerfile.txt` to `Dockerfile` for proper Docker build
- **Critical**: Added missing `build_indexes()` function via `ingest_docs.py`
- **Critical**: Added missing `llama-index-vector-stores-chroma` dependency
- **Performance**: Removed per-request model pulling (severe latency issue)
- **Architecture**: Fixed entrypoint script to work with containerized Ollama
- **Security**: Enabled authentication in Open-WebUI by default

### Added
- **Async Operations**: Full async/await support throughout FastAPI app
- **Caching**: Index and model caching to avoid repeated disk/network I/O
- **Error Handling**: Comprehensive try-except blocks and HTTP exception handling
- **Health Checks**: Container health endpoints for proper orchestration
- **Logging**: Structured JSON logging with structlog
- **Documentation**: Complete README, QUICKSTART, and inline documentation
- **Validation**: Pydantic models with field constraints
- **Scripts**: Setup, rebuild indexes, and test API helper scripts
- **CPU Support**: Docker Compose override for CPU-only systems

### Enhanced
- **Docker Compose**: Added health checks, proper dependencies, and networks
- **Service Coordination**: Proper startup ordering and readiness checks
- **API Endpoints**: Added `/health`, `/stats`, and improved `/query`
- **Document Support**: Multiple formats (PDF, MD, TXT, HTML, RST, DOCX)
- **Sample Data**: Auto-generation of sample documents for testing

### Performance Improvements
- **HTTP Connection Pooling**: Reuse connections to Ollama API
- **Startup Preloading**: Warm caches during application initialization
- **Lazy Loading**: Load indexes only when needed
- **Model Verification**: Skip redundant model availability checks

### Security Improvements
- **Input Validation**: Request size limits and type checking
- **Authentication**: Enabled by default in Web UI
- **Secret Management**: Environment variable for sensitive data
- **Network Isolation**: Docker network for service communication

### Documentation
- Comprehensive README with architecture diagrams
- Quick start guide for rapid deployment
- API documentation via FastAPI's built-in docs
- Troubleshooting section with common issues
- Code comments following best practices

## [0.1.0] - Initial Version

### Initial Implementation
- Basic FastAPI application
- Dual model routing (Qwen/DeepSeek)
- ChromaDB vector stores
- Docker Compose setup
- Open-WebUI integration

### Known Issues (Resolved in 1.0.0)
- Dockerfile named incorrectly
- Missing index building logic
- No error handling
- Synchronous blocking operations
- Per-request model downloads
- No health checks
- Missing dependencies

