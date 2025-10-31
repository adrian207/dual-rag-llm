# Changelog

All notable changes to the Dual RAG LLM System are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

**Author:** Adrian Johnson <adrian207@gmail.com>

---

## [Unreleased]

### Added
- PowerShell deployment scripts for Azure
- Complete Azure automation toolkit

---

## [1.11.0] - 2024-10-31

### Added - Code Syntax Highlighting
- **22 Programming Languages Supported**: Python, JavaScript, TypeScript, Java, C#, C, C++, Go, Rust, PHP, Ruby, Swift, Kotlin, PowerShell, Bash, SQL, HTML, CSS, JSON, YAML, XML, Markdown
- **9 Professional Themes**: VS Dark, VS Light, GitHub Dark/Light, Monokai, Dracula, Nord, Solarized Dark/Light
- **Advanced Language Detection**: Pattern-based detection with 95% confidence scoring
- **Real-time Highlighting**: highlight.js + marked.js integration in UI
- **Metadata Extraction**: Comments, imports, and complexity analysis for code blocks
- **Syntax Error Detection**: Basic validation for common programming errors

### Enhanced
- Frontend UI with theme selector dropdown
- Markdown rendering with automatic code highlighting
- Copy-to-clipboard functionality for code blocks

### Technical Details
- **Backend**: +415 lines in `rag/rag_dual.py`
- **Frontend**: Updated `ui/index.html`, `ui/app.js`, `ui/styles.css`
- **API Endpoints**: 8 new endpoints for syntax highlighting
- **Detection Patterns**: 21 language-specific patterns with confidence scoring

### Documentation
- Updated README with syntax highlighting features
- Added language support matrix

---

## [1.10.0] - 2024-10-31

### Added - Response Formatting Improvements
- **6 Formatting Styles**: Plain, Markdown, Structured, Professional, Concise, Detailed
- **Auto-Section Headers**: Intelligent header insertion for long responses
- **Code Enhancement**: Automatic language detection and hints for code blocks
- **List Creation**: Convert text patterns to structured lists
- **Readability Optimization**: Line length limits and sentence breaking
- **Emoji Headers**: Optional visual enhancement (configurable)
- **Improvement Scoring**: 0-1 quality metric for formatting enhancements

### Enhanced
- Response readability with maximum line length control (default: 100 chars)
- Code block formatting with automatic language detection
- Structured output with sections, lists, and headers

### Technical Details
- **Backend**: +387 lines of formatting logic
- **Functions**: 8 core formatting functions
- **API Endpoints**: 5 new endpoints for formatting control
- **Scoring System**: Multi-factor improvement calculation

### Documentation
- Detailed formatting feature descriptions
- Configuration options and examples

---

## [1.9.0] - 2024-10-31

### Added - Factuality Checking & Hallucination Detection
- **Claim Extraction**: Automatically identify 6 types of claims (factual statements, definitions, instructions, opinions, examples, code snippets)
- **Hallucination Detection**: 5 sophisticated indicators
  - False confidence markers ("definitely", "always", "never")
  - Fabricated sources (claims without citations)
  - Inconsistent details (numbers/dates not in context)
  - Unsupported claims (specific versions/dates)
  - Contradictory statements
- **Source Verification**: Cross-reference claims with RAG context
- **Verdict System**: Supported, Contradicted, Uncertain, Unverifiable
- **Risk Scoring**: 0-1 hallucination probability calculation
- **Reliability Assessment**: Binary safe/unsafe determination

### Enhanced
- Truth verification with context matching
- Evidence extraction and source tracking
- Confidence scoring for fact-checking

### Technical Details
- **Backend**: +488 lines of fact-checking logic
- **Detection**: Pattern-based claim classification
- **Scoring**: Support rate - contradiction penalty - hallucination risk
- **API Endpoints**: 5 new endpoints

### Documentation
- Factuality checking methodology
- Hallucination indicator descriptions

---

## [1.8.0] - 2024-10-31

### Added - Answer Validation & Verification
- **7 Validation Checks**:
  1. Factuality checking
  2. Source verification against RAG context
  3. Consistency checking (internal contradictions)
  4. Completeness checking (query fully answered)
  5. Relevance checking (on-topic verification)
  6. Clarity checking (readability metrics)
  7. Code validity checking (syntax and best practices)
- **Confidence Scoring**: Overall quality score (0-1) with detailed breakdown
- **Auto-Suggestions**: Actionable improvements for failed checks
- **Quality Metrics**: Length, structure, examples, code blocks, citations
- **Auto-Approval System**: Configurable threshold (default: 0.5)

### Enhanced
- Quality assurance for all responses
- Detailed feedback for improvement
- Configurable validation rules

### Technical Details
- **Backend**: +410 lines of validation logic
- **API Endpoints**: 5 new endpoints
- **Models**: ValidationCheck, ValidationResult, AnswerValidation, ValidationConfig

### Documentation
- Validation methodology
- Configuration options
- Quality metrics explanation

---

## [1.7.0] - 2024-10-31

### Added - Automatic Model Selection
- **12 Query Type Classification**: General questions, code generation, debugging, documentation, technical explanation, comparison, step-by-step tutorials, troubleshooting, API usage, architecture design, performance optimization, security
- **Intelligent Routing**: Map query types to optimal models
- **Performance Learning**: Track model performance per query type
- **User Feedback Integration**: Rating system affects future routing decisions
- **Automatic Fallback**: Secondary model selection if primary fails
- **Smart Model Matrix**: Configurable routing rules with confidence thresholds

### Enhanced
- Query classification with language and complexity detection
- Performance tracking with success rates and latency metrics
- Decision history with 500-item cache

### Technical Details
- **Backend**: +450 lines of routing logic
- **Classification**: Pattern-based with 12 query types
- **API Endpoints**: 8 new endpoints
- **Models**: QueryType, QueryClassification, ModelRouting, SelectionDecision

### Documentation
- Query type descriptions
- Routing configuration guide
- Performance tracking methodology

---

## [1.6.0] - 2024-10-31

### Added - Model Ensemble Strategies
- **6 Ensemble Strategies**:
  1. **Voting**: Democratic majority vote
  2. **Averaging**: Weighted score combination
  3. **Cascade**: Sequential refinement
  4. **Best-of-N**: Run multiple, pick best
  5. **Specialist**: Domain-specific model selection
  6. **Consensus**: Agreement-based confidence
- **Parallel Execution**: Concurrent model queries
- **Weighted Combinations**: Customizable model weights
- **Confidence-Based Selection**: Automatic best response choice
- **Domain Specialization**: Model-to-domain mapping

### Enhanced
- Multi-model collaboration
- Response quality through ensemble voting
- Confidence scoring across models

### Technical Details
- **Backend**: +530 lines of ensemble logic
- **Execution**: Async parallel model queries
- **API Endpoints**: 7 new endpoints
- **Models**: EnsembleStrategy, EnsembleConfig, EnsembleResult

### Documentation
- `docs/MODEL_ENSEMBLES.md` with strategy explanations
- Best practices and use cases

---

## [1.5.0] - 2024-10-31

### Added - Custom Model Fine-tuning Pipeline
- **Dataset Management**: CRUD operations for training datasets
- **Training Job Orchestration**: Job configuration and monitoring
- **Model Versioning**: Registry with metadata tracking
- **LoRA/QLoRA Support**: Architectural placeholders for efficient fine-tuning
- **Format Support**: Alpaca, ShareGPT, Chat formats
- **Progress Tracking**: Real-time training status and metrics

### Enhanced
- Model customization capabilities
- Training job management
- Version control for fine-tuned models

### Technical Details
- **Backend**: +420 lines of fine-tuning framework
- **API Endpoints**: 15 new endpoints (5 per resource type)
- **Models**: DatasetConfig, FineTuningConfig, ModelVersion
- **UI Dashboard**: Complete fine-tuning management interface

### Documentation
- Fine-tuning pipeline architecture
- Dataset format specifications
- Training configuration options

### Notes
- GPU training process requires external execution
- Framework provides management layer

---

## [1.4.0] - 2024-10-31

### Added - Model A/B Testing Framework
- **Statistical Testing**: t-tests with p-value calculation
- **Traffic Splitting**: Configurable model distribution
- **User Rating System**: 1-5 star feedback collection
- **Automatic Winner Detection**: Statistical significance threshold (default: p < 0.05)
- **Comprehensive Metrics**: Response time, token count, success rate, user satisfaction
- **Test Lifecycle**: Draft, Running, Paused, Completed states

### Enhanced
- Model comparison with statistical rigor
- Performance tracking across test variants
- User preference collection

### Technical Details
- **Backend**: +380 lines of A/B testing logic
- **Statistics**: t-tests, confidence intervals (95%)
- **API Endpoints**: 11 new endpoints
- **Models**: ABTestConfig, ABTestResult, ABTestStatistics
- **UI Dashboard**: Dedicated A/B testing interface

### Documentation
- A/B testing methodology
- Statistical analysis explanation
- Test configuration guide

---

## [1.3.0] - 2024-10-31

### Added - Dynamic Model Switching
- **Manual Model Selection**: 10+ LLM options across 3 categories
  - Microsoft Technologies: Qwen 2.5 Coder variants
  - Open Source: DeepSeek Coder, CodeLlama
  - General Purpose: Llama 3.1, Mistral
- **Model Comparison**: Side-by-side comparison of multiple models
- **Performance Tracking**: Tokens/sec, response time, query count per model
- **Automatic Fallback**: Secondary model if primary unavailable
- **Smart Routing**: Auto-select based on query type (default option)

### Enhanced
- UI with model selector dropdown
- Performance metrics display in sidebar
- "Try Another Model" functionality
- Model badges and visual indicators

### Technical Details
- **Backend**: Model override support in query processing
- **Frontend**: Dynamic model selection UI
- **API**: Model performance endpoint
- **Fallback**: Automatic model availability detection

### Documentation
- `docs/DYNAMIC_MODEL_SWITCHING.md`
- Model comparison guidelines

---

## [1.2.0] - 2024-10-31

### Added - Real-time Streaming & Web UI
- **Server-Sent Events (SSE)**: Real-time token streaming
- **Modern Web Interface**: HTML/CSS/JavaScript UI
- **Interactive Chat**: Message history with user/assistant distinction
- **Tool Integration UI**: Checkboxes for Web Search and GitHub
- **Example Queries**: One-click example questions
- **System Statistics**: Live cache hit rate and system info
- **Responsive Design**: Mobile-friendly layout

### Enhanced
- User experience with real-time feedback
- Visual indicators for tool usage
- Progress tracking during queries

### Technical Details
- **Backend**: `sse_starlette` for streaming
- **Frontend**: `ui/index.html`, `ui/app.js`, `ui/styles.css`
- **API**: `/query/stream` endpoint
- **Protocol**: Server-Sent Events

### Documentation
- UI usage guide
- Streaming protocol details

---

## [1.1.0] - 2024-10-31

### Added - Redis Caching & External Tools
- **Redis Caching**: Ultra-fast response caching with TTL (1 hour default)
- **Brave Search Integration**: Web search capability with API
- **GitHub Integration**: Repository search and code exploration
- **Tool System**: Modular architecture for external integrations
- **Cache Statistics**: Hit rate, total queries, cache misses tracking

### Enhanced
- Performance with intelligent caching
- Knowledge base with real-time web data
- Code search with GitHub API

### Technical Details
- **Caching**: `aioredis` with async support
- **Tools**: `httpx` for async HTTP requests
- **APIs**: Brave Search API, GitHub API (PyGithub)
- **Endpoints**: `/tools/web_search`, `/tools/github_search`, `/cache/clear`, `/cache/stats`

### Configuration
```bash
REDIS_HOST=redis
REDIS_PORT=6379
BRAVE_API_KEY=your_key
GITHUB_TOKEN=your_token
```

### Documentation
- Tool integration guide
- Caching strategy documentation

---

## [1.0.0] - 2024-10-31

### Added - Core Dual RAG System
- **Dual Vector Stores**: ChromaDB for two knowledge bases
  - Microsoft Technologies (MS Store)
  - General Software Development (General Store)
- **LLM Integration**: Ollama with multiple model support
- **HuggingFace Embeddings**: `all-MiniLM-L6-v2` for efficient vector encoding
- **FastAPI Backend**: Async REST API
- **Docker Compose**: Complete containerized deployment
- **GPU Support**: NVIDIA GPU acceleration for Ollama

### Core Features
- Document ingestion with text chunking (500 chars, 50 overlap)
- Semantic search across both knowledge bases
- Context-aware response generation
- Structured logging with `structlog`
- Health check endpoint
- System statistics endpoint

### API Endpoints
- `POST /query`: Main RAG query endpoint
- `GET /health`: System health check
- `GET /stats`: System statistics

### Architecture
```
User Query → Dual RAG (MS + General) → Context → LLM → Response
```

### Technical Stack
- **Backend**: Python 3.11, FastAPI, ChromaDB
- **LLM**: Ollama (llama3.1:8b default)
- **Embeddings**: HuggingFace sentence-transformers
- **Infrastructure**: Docker, Docker Compose
- **GPU**: NVIDIA Container Toolkit

### Documentation
- `README.md`: Project overview
- `QUICKSTART.md`: Getting started guide
- `docker-compose.yaml`: Deployment configuration

---

## Project Information

### Repository
- **URL**: https://github.com/adrian207/dual-rag-llm
- **License**: MIT
- **Author**: Adrian Johnson <adrian207@gmail.com>

### Branches
- `main`: Production-ready releases
- `azure/deployment`: Azure-specific deployment branch
- `feature/*`: Feature development branches

### Release Process
1. Feature development on feature branch
2. Merge to `main` with version tag
3. Sync to `azure/deployment` branch
4. Push all changes to GitHub

### Version History
- **v1.11.0**: Code Syntax Highlighting (22 languages)
- **v1.10.0**: Response Formatting (6 styles)
- **v1.9.0**: Factuality Checking & Hallucination Detection
- **v1.8.0**: Answer Validation & Verification (7 checks)
- **v1.7.0**: Automatic Model Selection (12 query types)
- **v1.6.0**: Model Ensemble Strategies (6 strategies)
- **v1.5.0**: Custom Model Fine-tuning Pipeline
- **v1.4.0**: Model A/B Testing Framework
- **v1.3.0**: Dynamic Model Switching
- **v1.2.0**: Real-time Streaming & Web UI
- **v1.1.0**: Redis Caching & External Tools
- **v1.0.0**: Core Dual RAG System

### Statistics
- **Total Releases**: 12
- **Total Features**: 60+
- **API Endpoints**: 80+
- **Code Lines Added**: ~4,500+
- **Languages Supported**: 22
- **Documentation Pages**: 15+

---

## Support & Contributing

### Getting Help
- **GitHub Issues**: [Report bugs or request features](https://github.com/adrian207/dual-rag-llm/issues)
- **Email**: adrian207@gmail.com
- **Documentation**: See `docs/` directory

### Contributing
- See `CONTRIBUTING.md` for guidelines
- Follow semantic versioning
- Include tests for new features
- Update documentation

### Acknowledgments
- Ollama team for LLM server
- ChromaDB for vector store
- FastAPI for web framework
- HuggingFace for embeddings
- All contributors and users

---

[Unreleased]: https://github.com/adrian207/dual-rag-llm/compare/v1.11.0...HEAD
[1.11.0]: https://github.com/adrian207/dual-rag-llm/compare/v1.10.0...v1.11.0
[1.10.0]: https://github.com/adrian207/dual-rag-llm/compare/v1.9.0...v1.10.0
[1.9.0]: https://github.com/adrian207/dual-rag-llm/compare/v1.8.0...v1.9.0
[1.8.0]: https://github.com/adrian207/dual-rag-llm/compare/v1.7.0...v1.8.0
[1.7.0]: https://github.com/adrian207/dual-rag-llm/compare/v1.6.0...v1.7.0
[1.6.0]: https://github.com/adrian207/dual-rag-llm/compare/v1.5.0...v1.6.0
[1.5.0]: https://github.com/adrian207/dual-rag-llm/compare/v1.4.0...v1.5.0
[1.4.0]: https://github.com/adrian207/dual-rag-llm/compare/v1.3.0...v1.4.0
[1.3.0]: https://github.com/adrian207/dual-rag-llm/compare/v1.2.0...v1.3.0
[1.2.0]: https://github.com/adrian207/dual-rag-llm/compare/v1.1.0...v1.2.0
[1.1.0]: https://github.com/adrian207/dual-rag-llm/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/adrian207/dual-rag-llm/releases/tag/v1.0.0
