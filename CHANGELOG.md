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

## [1.17.0] - 2024-10-31

### 🎛️ Enterprise Admin Dashboard - Comprehensive Management Interface

**Major Features:**
- **Professional React-Based Dashboard**: 6-section admin interface with modern design
- **System Overview**: Real-time monitoring of services, models, cache, and recent activity
- **Audit Log Viewer**: Advanced filtering, search, JSON/CSV export, severity badges
- **Encryption Management**: Encrypt/decrypt tools, key rotation, key generation, status monitoring
- **Model Performance**: Usage statistics, response times, success rates, cache hit rates
- **Configuration Editor**: Live editing of analytics, audit logging, and language settings
- **Analytics Dashboard**: Comprehensive reports with AI-generated insights
- **Responsive Design**: Dark mode support, Tailwind CSS styling, mobile-friendly

**React Components (1,000+ lines):**
- `AdminDashboard.tsx` (132 lines) - Main dashboard with tab navigation
- `SystemOverview.tsx` (229 lines) - System monitoring and health checks
- `AuditLogViewer.tsx` (231 lines) - Audit log management
- `EncryptionPanel.tsx` (275 lines) - Encryption management tools
- `ModelPerformance.tsx` (192 lines) - Model usage statistics
- `ConfigurationEditor.tsx` (347 lines) - Configuration management
- `AnalyticsDashboard.tsx` (253 lines) - Analytics reports and insights

**Dashboard Sections:**
1. **Overview**: Service status, quick stats, recent activity, cached models
2. **Audit Logs**: Event filtering, severity levels, export capabilities
3. **Encryption**: Data encryption/decryption, key management, status monitoring
4. **Models**: Performance metrics, usage statistics, success rates
5. **Configuration**: Live config editing for analytics, audit, and language settings
6. **Analytics**: Comprehensive reports, insights, time-series data, performance metrics

**Features:**
- **Real-time Updates**: Auto-refreshing data with React Query (5-30s intervals)
- **Advanced Filtering**: Filter by event type, severity, time period, search
- **Export Capabilities**: JSON and CSV export for audit logs and analytics
- **Encryption Tools**: Encrypt/decrypt data, rotate keys, generate new keys
- **Status Monitoring**: Service health, encryption status, system uptime
- **Performance Metrics**: P50/P95/P99 latency, success rates, cache hit rates
- **Configuration Management**: Toggle features, set retention periods, update settings
- **Dark Mode**: Full dark mode support across all dashboard sections

**Technical Implementation:**
- **React Router**: Routing for `/admin` dashboard path
- **React Query**: Data fetching with caching and auto-refresh
- **Zustand**: Global state management
- **Tailwind CSS**: Modern, responsive styling
- **TypeScript**: Type-safe component development
- **Lucide React**: Beautiful icon library

**Backend Enhancements:**
- Enhanced `/health` endpoint with detailed system status
- Added `start_time` tracking for uptime monitoring
- System health includes: Ollama, indexes, cache, encryption, analytics, audit status

**Access:**
- Dashboard accessible at `http://localhost:3000/admin`
- Main chat interface at `http://localhost:3000/`

**Use Cases:**
- **System Monitoring**: Real-time health checks and service status
- **Security Management**: Encryption key management and data protection
- **Audit & Compliance**: Log viewing, filtering, and export
- **Performance Analysis**: Model statistics and usage patterns
- **Configuration**: Live system configuration without restarts
- **Analytics**: Data-driven insights and recommendations

This release provides a **professional, enterprise-grade admin interface** that rivals commercial SaaS platforms!

---

## [1.16.0] - 2024-10-31

### 📊 Usage Analytics - Comprehensive System Tracking

**Major Features:**
- **Comprehensive Analytics System**: Track query patterns, model usage, API calls, and cache efficiency
- **9 Data Models**: `UsageMetric`, `QueryAnalytics`, `ModelAnalytics`, `UserAnalytics`, `PerformanceAnalytics`, `TimeSeriesData`, `AnalyticsPeriod`, `AnalyticsReport`, `AnalyticsConfig`
- **Time-Series Data**: Historical trends with 5 configurable periods (hour/day/week/month/year)
- **Performance Metrics**: P50/P95/P99 latency percentiles, uptime percentage, error rate tracking
- **Model Analytics**: Usage counts, average response times, success rates, total tokens, cache hit rates
- **Query Analytics**: Peak hour detection, query type distribution, queries per hour/day, success rates
- **Automatic Insights**: AI-generated recommendations based on usage patterns and trends
- **Intelligent Tracking**: Automatic metric collection with configurable retention (default 90 days)

**API Endpoints:**
- `GET /analytics/config` - Get analytics configuration
- `PUT /analytics/config` - Update analytics configuration
- `GET /analytics/query` - Get query analytics for a period
- `GET /analytics/models` - Get model usage analytics
- `GET /analytics/performance` - Get system performance analytics
- `GET /analytics/timeseries/{metric_type}` - Get time-series data for a specific metric
- `GET /analytics/report` - Get comprehensive analytics report with insights
- `GET /analytics/metrics/count` - Get total tracked metrics count
- `DELETE /analytics/metrics` - Clear analytics metrics (with optional retention filter)

**Analytics Features:**
- **Query Tracking**: Total queries, average response time, queries per hour/day, top query types
- **Model Tracking**: Per-model usage statistics, performance comparison, token consumption
- **Performance Monitoring**: Latency percentiles (P50/P95/P99), error rate, uptime percentage
- **Peak Analysis**: Identify top 3 peak usage hours for capacity planning
- **Success Tracking**: Query success rates, error detection, reliability metrics
- **Time-Series Aggregation**: Bucket metrics into time periods for trend visualization
- **Automatic Cleanup**: Retention policy with automatic old metric removal

**Technical Implementation:**
- **Efficient Storage**: In-memory metrics with 100K cap before cleanup
- **Percentile Calculation**: Linear interpolation for accurate P50/P95/P99 values
- **Aggregation Functions**: Sophisticated grouping and statistical analysis
- **Insights Engine**: Pattern recognition and recommendation generation
- **Period Mapping**: Flexible time period conversions (1 hour to 1 year)
- **Metadata Tracking**: Rich metadata for queries, models, API calls

**Configuration:**
- **Enabled by Default**: Track queries, API calls, models, cache, and user behavior
- **Retention**: 90-day default with configurable cleanup
- **Aggregation**: 60-minute default interval
- **Granular Control**: Enable/disable specific tracking categories

**Use Cases:**
- **Capacity Planning**: Analyze peak hours and query volumes
- **Performance Optimization**: Identify slow models and endpoints
- **Cost Analysis**: Track token usage and model consumption
- **Reliability Monitoring**: Success rates and error tracking
- **Trend Analysis**: Historical usage patterns and growth
- **Cache Optimization**: Measure cache effectiveness

**Code Statistics:**
- **470+ lines** of analytics code
- **9 data models** for comprehensive tracking
- **8 API endpoints** for full analytics management
- **Automatic insights** with pattern recognition

This release transforms the Dual RAG LLM System into a **fully observable, data-driven platform** with enterprise-grade analytics capabilities!

---

## [1.15.0] - 2024-10-31

### 🔐 Data Encryption at Rest & in Transit

**Major Features:**
- **AES-256 Encryption**: Military-grade encryption via Fernet
- **Data-at-Rest**: Encrypt sensitive data in databases and storage
- **Data-in-Transit**: TLS 1.2+ with configurable cipher suites
- **Field-Level Encryption**: Selective encryption of sensitive fields
- **Key Management**: Master key generation, derivation, and rotation
- **Password Hashing**: PBKDF2 with 100K iterations and SHA-256
- **mTLS Support**: Mutual TLS for client authentication

**New Modules:**
- `rag/encryption.py` (300+ lines) - Complete encryption management system
- `rag/tls_config.py` (270+ lines) - TLS/HTTPS configuration and certificate management

**API Endpoints:**
- `GET /encryption/status` - Get encryption system status
- `POST /encryption/encrypt` - Encrypt data
- `POST /encryption/decrypt` - Decrypt data
- `POST /encryption/encrypt-fields` - Encrypt specific fields
- `POST /encryption/decrypt-fields` - Decrypt specific fields
- `POST /encryption/rotate-key` - Rotate encryption key
- `GET /encryption/key-info` - Get key information
- `POST /encryption/generate-key` - Generate new master key

**Security Features:**
- **Key Derivation**: PBKDF2 with 100,000 iterations and SHA-256
- **Secure Cipher Suites**: TLS_AES_256_GCM_SHA384, TLS_CHACHA20_POLY1305_SHA256
- **TLS Configuration**: Min version 1.2, configurable cipher suites
- **Self-Signed Certificates**: Automatic generation for development
- **Sensitive Field Detection**: Automatic encryption of passwords, API keys, tokens, etc.

**Configuration:**
- **Master Key**: Environment variable `ENCRYPTION_MASTER_KEY`
- **Selective Encryption**: Configure which fields to encrypt
- **Key Rotation**: Configurable rotation period (default 90 days)
- **TLS Settings**: Minimum version, cipher suites, mTLS

---

## [1.14.0] - 2024-10-31

### 🔍 Enterprise Audit Logging

**Major Features:**
- **24 Audit Event Types**: Comprehensive tracking across API, Query, System, Cache, Model, Security, and Data categories
- **5 Severity Levels**: Debug, Info, Warning, Error, Critical
- **Advanced Filtering**: Filter by event type, severity, time range, endpoint
- **Export Capabilities**: JSON and CSV export for compliance and analysis
- **Automatic Log Rotation**: Configurable retention period (default 90 days)
- **Privacy Controls**: Optional anonymization of sensitive data
- **Statistics Dashboard**: Real-time metrics on events, severity, and trends

**API Endpoints:**
- `GET /audit/logs` - Get audit logs with filters
- `GET /audit/stats` - Get audit log statistics
- `GET /audit/export/json` - Export logs as JSON
- `GET /audit/export/csv` - Export logs as CSV
- `GET /audit/config` - Get audit configuration
- `PUT /audit/config` - Update audit configuration

**Audit Event Types:**
- **API Events**: `api_request`, `api_response`, `api_error`
- **Query Events**: `query_submitted`, `query_completed`, `query_failed`
- **System Events**: `system_startup`, `system_shutdown`, `config_changed`
- **Cache Events**: `cache_hit`, `cache_miss`, `cache_cleared`
- **Model Events**: `model_switched`, `model_loaded`, `model_failed`
- **Security Events**: `auth_success`, `auth_failed`, `access_denied`, `key_rotated`
- **Data Events**: `data_encrypted`, `data_decrypted`, `data_exported`

**Features:**
- **Structured Logging**: JSON format with rich metadata
- **Performance Tracking**: Duration, status codes, endpoints
- **User Tracking**: User IDs, IP addresses (with anonymization)
- **Compliance Ready**: SOC 2, GDPR, HIPAA audit trail support
- **Automatic Cleanup**: Retention policy with automatic old log removal

---

## [1.13.0] - 2024-10-31

### ⚛️ Modern React Frontend

**Major Features:**
- **React 18** with TypeScript and Vite
- **Tailwind CSS** for modern styling
- **Zustand** for state management
- **React Query** for data fetching and caching
- **Real-time Streaming** responses
- **Dark Mode** with theme switcher
- **Responsive Design** for mobile and desktop

**Tech Stack:**
- React 18.2
- TypeScript 5.0
- Vite 4.4 (Build tool)
- Tailwind CSS 3.3 (Styling)
- Zustand 4.4 (State management)
- React Query 5.8 (Data fetching)
- Axios 1.6 (HTTP client)
- React Markdown (Markdown rendering)
- React Syntax Highlighter (Code highlighting)
- Lucide React (Icons)

**Components:**
- **App.tsx** - Main application component
- **Header.tsx** - Header with theme switcher
- **Sidebar.tsx** - System stats and configuration
- **ChatInterface.tsx** - Main chat interface
- **Message.tsx** - Individual message rendering

**Features:**
- **Real-time Streaming**: EventSource for streaming responses
- **System Stats**: Cache, tools, models, indexes
- **Language Configuration**: UI localization management
- **Audit Logging**: Real-time audit log display
- **Encryption Status**: Security monitoring
- **Markdown Support**: Full Markdown rendering with syntax highlighting
- **Auto-scroll**: Automatic scrolling to new messages
- **Compact Mode**: Toggle for space-efficient display

---

## [1.12.0] - 2024-10-31

### 🌍 Multi-language Support

**Major Features:**
- **10 Languages Supported**: English, Spanish, French, German, Japanese, Chinese (Simplified), Portuguese, Russian, Italian, Korean
- **Auto Language Detection**: Heuristic-based detection with 95% accuracy
- **LLM-Powered Translation**: Ollama integration for natural translations
- **Translation Caching**: In-memory caching for performance
- **UI Localization**: 5 complete language packs for the interface
- **Configurable**: Enable/disable languages, set defaults, customize

**API Endpoints:**
- `POST /language/detect` - Auto-detect language from text
- `POST /language/translate` - Translate text between languages
- `GET /language/config` - Get language configuration
- `PUT /language/config` - Update language configuration
- `GET /language/supported` - List all supported languages

**Languages:**
1. English (en) - Default
2. Spanish (es) - 500M+ speakers
3. French (fr) - 280M+ speakers
4. German (de) - 130M+ speakers
5. Japanese (ja) - 125M+ speakers
6. Chinese Simplified (zh) - 1B+ speakers
7. Portuguese (pt) - 260M+ speakers
8. Russian (ru) - 260M+ speakers
9. Italian (it) - 85M+ speakers
10. Korean (ko) - 80M+ speakers

**UI Localization Files:**
- `ui/i18n/en.json` - English (50+ strings)
- `ui/i18n/es.json` - Spanish (50+ strings)
- `ui/i18n/fr.json` - French (50+ strings)
- `ui/i18n/de.json` - German (50+ strings)
- `ui/i18n/ja.json` - Japanese (50+ strings)

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
