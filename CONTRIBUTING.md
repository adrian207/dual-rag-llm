# Contributing to Dual RAG LLM System

**Author:** Adrian Johnson <adrian207@gmail.com>

This guide outlines the development workflow, branching strategy, and contribution guidelines.

## Development Setup

### Prerequisites

- Docker with GPU support
- Python 3.11+
- Git
- NVIDIA Container Toolkit (for GPU)

### Local Development Setup

```bash
# Clone the repository
git clone https://github.com/adrian207/dual-rag-llm.git
cd dual-rag-llm

# Create development branch
git checkout -b feature/your-feature-name

# Set up development environment
make setup

# Start services in development mode
docker compose up -d
```

## Branching Strategy

### Main Branches

- **`main`** - Production-ready code, protected branch
- **`develop`** - Integration branch for features (create if needed)

### Feature Branches

Create from `main` or `develop`:

```bash
git checkout -b feature/feature-name
git checkout -b bugfix/bug-description
git checkout -b hotfix/critical-fix
git checkout -b docs/documentation-update
```

### Branch Naming Convention

| Type | Pattern | Example |
|------|---------|---------|
| Feature | `feature/description` | `feature/add-redis-cache` |
| Bug Fix | `bugfix/description` | `bugfix/fix-index-loading` |
| Hot Fix | `hotfix/description` | `hotfix/memory-leak` |
| Documentation | `docs/description` | `docs/api-examples` |
| Refactor | `refactor/description` | `refactor/async-handlers` |
| Performance | `perf/description` | `perf/optimize-embeddings` |

## Versioning Strategy

We follow [Semantic Versioning](https://semver.org/) (SemVer):

**Format:** `MAJOR.MINOR.PATCH`

- **MAJOR** - Breaking changes
- **MINOR** - New features (backward compatible)
- **PATCH** - Bug fixes (backward compatible)

### Version Bumping

Update these files when changing version:

1. `VERSION`
2. `rag/__init__.py` (`__version__`)
3. `CHANGELOG.md`

```bash
# Example: Bumping to 1.1.0
echo "1.1.0" > VERSION
sed -i 's/__version__ = ".*"/__version__ = "1.1.0"/' rag/__init__.py
# Update CHANGELOG.md manually
```

## Development Workflow

### 1. Create Feature Branch

```bash
git checkout main
git pull origin main
git checkout -b feature/your-feature
```

### 2. Make Changes

```bash
# Edit files
# Test locally
make test

# Check logs
make logs-rag
```

### 3. Test Thoroughly

```bash
# Run linting (if you add Python linting)
# pylint rag/*.py

# Run tests
make test

# Check health
make health

# Manual testing
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "test", "file_ext": ".py"}'
```

### 4. Commit Changes

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```bash
git add -A
git commit -m "feat: add Redis caching layer

- Implement Redis connection pooling
- Add cache middleware for query responses
- Update docker-compose with Redis service
- Add cache invalidation on index rebuild

Closes #123"
```

**Commit Types:**
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation
- `style:` - Formatting
- `refactor:` - Code restructuring
- `perf:` - Performance improvement
- `test:` - Tests
- `chore:` - Maintenance

### 5. Push and Create PR

```bash
git push origin feature/your-feature
```

Then create a Pull Request on GitHub.

## Release Process

### For Maintainers

#### Minor/Major Release

```bash
# 1. Update version
echo "1.1.0" > VERSION
sed -i 's/__version__ = ".*"/__version__ = "1.1.0"/' rag/__init__.py

# 2. Update CHANGELOG.md
# Add new section with changes

# 3. Commit
git add VERSION rag/__init__.py CHANGELOG.md
git commit -m "chore: bump version to 1.1.0"

# 4. Tag
git tag -a v1.1.0 -m "Release v1.1.0: Description

- Feature 1
- Feature 2
- Bug fix 1

Author: Adrian Johnson <adrian207@gmail.com>
Date: $(date +%Y-%m-%d)"

# 5. Push
git push origin main
git push origin v1.1.0
```

#### Patch Release

```bash
# Quick fix process
git checkout -b hotfix/critical-bug
# Fix the bug
git commit -m "fix: resolve critical memory leak"

# Bump patch version
echo "1.0.1" > VERSION
sed -i 's/__version__ = ".*"/__version__ = "1.0.1"/' rag/__init__.py
git commit -m "chore: bump version to 1.0.1"

# Merge and tag
git checkout main
git merge hotfix/critical-bug
git tag -a v1.0.1 -m "Hotfix v1.0.1: Memory leak fix"
git push origin main v1.0.1
```

## Code Quality Standards

### Python Code

- Follow PEP 8 style guide
- Use type hints where possible
- Add docstrings to all functions
- Maximum line length: 100 characters
- Use meaningful variable names

```python
def process_query(
    question: str,
    file_ext: str,
    top_k: int = 3
) -> QueryResponse:
    """
    Process user query with context retrieval.
    
    Args:
        question: User's question text
        file_ext: File extension for routing
        top_k: Number of context chunks to retrieve
        
    Returns:
        QueryResponse with answer and metadata
        
    Raises:
        HTTPException: If processing fails
    """
    # Implementation
```

### Shell Scripts

- Use `#!/bin/bash` shebang
- Add `set -e` for error handling
- Comment complex sections
- Test on both Linux and macOS

### Docker

- Multi-stage builds when possible
- Minimize layer count
- Use specific version tags, not `latest`
- Add health checks
- Optimize for caching

## Testing Guidelines

### Manual Testing Checklist

- [ ] Service starts without errors
- [ ] Health endpoint returns 200
- [ ] Query endpoint works for both MS and OSS paths
- [ ] Indexes build successfully
- [ ] Logs are structured and readable
- [ ] Error cases handled gracefully
- [ ] Performance is acceptable (< 10s per query)

### Adding Automated Tests

[Unverified] When adding tests:

```python
# tests/test_rag_dual.py
import pytest
from rag_dual import get_model_for_extension

def test_model_routing_ms_extensions():
    """Test that MS extensions route to Qwen model"""
    model, source = get_model_for_extension(".cs")
    assert model == "qwen2.5-coder:32b-q4_K_M"
    assert source == "Microsoft"

def test_model_routing_oss_extensions():
    """Test that OSS extensions route to DeepSeek model"""
    model, source = get_model_for_extension(".py")
    assert model == "deepseek-coder-v2:33b-q4_K_M"
    assert source == "OpenSource"
```

## Documentation Standards

- Update README.md for user-facing changes
- Update CHANGELOG.md for every release
- Add inline comments for complex logic
- Update API docs (docstrings) when changing endpoints
- Include examples for new features

## Development Tips

### Quick Iteration

```bash
# Rebuild only RAG service
docker compose build rag
docker compose up -d rag

# View live logs
make dev-logs

# Shell into container
make dev-shell
```

### Debugging

```bash
# Check service logs
docker compose logs -f rag

# Check Ollama logs
docker compose logs -f ollama

# Inspect container
docker compose exec rag /bin/bash
python -c "from rag_dual import *; print(app_state)"

# Check GPU usage
nvidia-smi

# Check disk usage
du -sh rag/indexes/*
```

### Performance Profiling

```python
# Add to rag_dual.py for profiling
import time

def profile_query(q: Query):
    start = time.time()
    
    # ... existing code ...
    
    logger.info(
        "query_performance",
        total_time=time.time() - start,
        retrieval_time=retrieval_time,
        llm_time=llm_time
    )
```

## Common Development Tasks

### Adding a New Model

1. Pull the model: `docker exec ollama ollama pull model-name`
2. Update constants in `rag/rag_dual.py`
3. Update routing logic in `get_model_for_extension()`
4. Update documentation
5. Test thoroughly

### Adding a New Document Format

1. Check if LlamaIndex supports it (likely does)
2. Update `required_exts` in `ingest_docs.py`
3. Test with sample documents
4. Update documentation

### Adding a New API Endpoint

1. Add endpoint to `rag/rag_dual.py`
2. Add Pydantic models for request/response
3. Add error handling
4. Add logging
5. Test endpoint
6. Update README.md

### Optimizing Performance

1. Profile with `time.perf_counter()`
2. Check for blocking operations
3. Add caching where appropriate
4. Consider async alternatives
5. Monitor memory usage
6. Test with realistic data volumes

## Getting Help

- **Issues**: Create GitHub issue with detailed description
- **Questions**: Discussions tab on GitHub
- **Email**: adrian207@gmail.com

## License

[Specify your license here]

---

Thank you for contributing to the Dual RAG LLM System! 🚀

