# Development Guide - Quick Reference

**Author:** Adrian Johnson <adrian207@gmail.com>

Quick reference for developing the Dual RAG LLM System.

## 🚀 Quick Start Development

```bash
# Clone and setup
git clone https://github.com/adrian207/dual-rag-llm.git
cd dual-rag-llm
make setup

# Create feature branch
git checkout -b feature/my-feature

# Make changes, test locally
make start
make test
make logs-rag

# Commit and push
git add -A
git commit -m "feat: description of feature"
git push origin feature/my-feature
```

## 📁 Project Structure

```
dual-rag-llm/
├── .github/               # CI/CD workflows and templates
│   ├── workflows/         # GitHub Actions
│   └── ISSUE_TEMPLATE/    # Issue templates
├── docs/                  # Additional documentation
├── rag/                   # Main application
│   ├── __init__.py        # Version metadata
│   ├── Dockerfile         # Container build
│   ├── rag_dual.py        # FastAPI application ⭐
│   ├── ingest_docs.py     # Document indexing ⭐
│   ├── logging_config.py  # Logging setup
│   └── requirements.txt   # Python dependencies
├── scripts/               # Helper scripts
│   ├── setup.sh          # Initial setup
│   ├── rebuild_indexes.sh # Index rebuilding
│   └── test_api.sh       # API testing
├── CHANGELOG.md          # Version history
├── CONTRIBUTING.md       # Contribution guide
├── docker-compose.yml    # Service orchestration
├── Makefile              # Convenient commands
├── README.md             # Main documentation
├── ROADMAP.md            # Future plans
└── VERSION               # Current version
```

## 🎯 Key Files to Know

### Core Application
- **`rag/rag_dual.py`** - Main FastAPI app, routing logic, API endpoints
- **`rag/ingest_docs.py`** - Document loading and index building
- **`docker-compose.yml`** - Service configuration

### Configuration
- **`VERSION`** - Current version (update on release)
- **`env.example`** - Environment variables template
- **`rag/requirements.txt`** - Python dependencies

### Documentation
- **`README.md`** - User-facing documentation
- **`CONTRIBUTING.md`** - Developer guide
- **`ROADMAP.md`** - Future plans
- **`CHANGELOG.md`** - Release notes

## 🔧 Common Tasks

### Adding a New Feature

```bash
# 1. Create branch
git checkout -b feature/redis-cache

# 2. Make changes
# Edit files...

# 3. Test locally
make start
curl http://localhost:8000/health

# 4. Commit
git add -A
git commit -m "feat: add Redis caching layer

- Add Redis service to docker-compose
- Implement cache middleware
- Add cache invalidation logic

Improves query performance by 80% for cached queries"

# 5. Push and create PR
git push origin feature/redis-cache
# Go to GitHub and create Pull Request
```

### Fixing a Bug

```bash
# 1. Create branch
git checkout -b bugfix/memory-leak

# 2. Fix the bug
# Edit files...

# 3. Test
make test
make logs-rag

# 4. Commit with clear description
git commit -m "fix: resolve memory leak in index caching

- Clear index cache on rebuild
- Add garbage collection trigger
- Limit cache size to 1GB

Fixes #42"

# 5. Push
git push origin bugfix/memory-leak
```

### Updating Documentation

```bash
git checkout -b docs/api-examples
# Update README.md or other docs
git commit -m "docs: add API usage examples"
git push origin docs/api-examples
```

## 🏷️ Version Management

### When to Bump Version

- **MAJOR** (1.0.0 → 2.0.0): Breaking changes
- **MINOR** (1.0.0 → 1.1.0): New features, backward compatible
- **PATCH** (1.0.0 → 1.0.1): Bug fixes, backward compatible

### How to Release

```bash
# 1. Update version files
echo "1.1.0" > VERSION
sed -i 's/__version__ = ".*"/__version__ = "1.1.0"/' rag/__init__.py

# 2. Update CHANGELOG.md
# Add new section at top:
## [1.1.0] - 2026-01-15
### Added
- Feature 1
- Feature 2
### Fixed
- Bug 1

# 3. Commit version bump
git add VERSION rag/__init__.py CHANGELOG.md
git commit -m "chore: bump version to 1.1.0"

# 4. Create and push tag
git tag -a v1.1.0 -m "Release v1.1.0: Description

- Key feature 1
- Key feature 2

Author: Adrian Johnson <adrian207@gmail.com>
Date: $(date +%Y-%m-%d)"

git push origin main
git push origin v1.1.0
```

GitHub Actions will automatically create a release!

## 🧪 Testing

### Local Testing

```bash
# Start services
make start

# Health check
make health

# Run API tests
make test

# Manual query test
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How do I create a list?",
    "file_ext": ".py"
  }' | jq
```

### Testing Changes

```bash
# Rebuild after code changes
docker compose build rag
docker compose up -d rag

# Watch logs
make logs-rag

# Test specific endpoint
curl http://localhost:8000/stats | jq
```

## 📊 Monitoring

### View Logs

```bash
make logs              # All services
make logs-rag          # RAG service only
make logs-ollama       # Ollama only
```

### Check Status

```bash
docker compose ps
make health
docker stats          # Resource usage
nvidia-smi            # GPU usage
```

### Debugging

```bash
# Shell into container
make dev-shell
# or
docker compose exec rag /bin/bash

# Python REPL in container
docker compose exec rag python
>>> from rag_dual import app_state
>>> print(app_state.ms_index)

# Check for errors
docker compose logs rag | grep ERROR
```

## 🔥 Hot Tips

### Fast Iteration

```bash
# Only rebuild RAG service
docker compose build rag && docker compose up -d rag

# Tail logs for immediate feedback
make dev-logs
```

### Performance Profiling

Add timing to endpoints:

```python
import time

@app.post("/query")
async def query_endpoint(q: Query):
    start = time.time()
    
    # ... existing code ...
    
    logger.info("timing", total=time.time() - start)
```

### Debugging Docker Issues

```bash
# Check Docker resources
docker system df

# Clean up
docker system prune -a

# Rebuild from scratch
docker compose down -v
docker compose build --no-cache
docker compose up -d
```

### Git Workflow Tips

```bash
# Stash changes temporarily
git stash
git checkout main
git pull origin main
git checkout -
git stash pop

# Interactive rebase (clean up commits)
git rebase -i main

# Amend last commit
git commit --amend

# View commit history nicely
git log --oneline --graph --all
```

## 📚 Resources

### Internal Documentation
- [README.md](../README.md) - Main documentation
- [CONTRIBUTING.md](../CONTRIBUTING.md) - Full contribution guide
- [ROADMAP.md](../ROADMAP.md) - Future plans
- [CHANGELOG.md](../CHANGELOG.md) - Version history

### External Resources
- [LlamaIndex Docs](https://docs.llamaindex.ai/)
- [FastAPI Docs](https://fastapi.tiangolo.com/)
- [Ollama Docs](https://ollama.ai/docs)
- [ChromaDB Docs](https://docs.trychroma.com/)

### Community
- GitHub Issues: Report bugs
- GitHub Discussions: Ask questions
- Email: adrian207@gmail.com

## ✅ Pre-Push Checklist

Before pushing code:

- [ ] Code follows style guidelines
- [ ] Tests pass locally (`make test`)
- [ ] Services start without errors (`make start`)
- [ ] Logs are clean (no errors)
- [ ] Documentation updated if needed
- [ ] Commit message follows conventions
- [ ] Branch name is descriptive

## 🚨 Common Issues

**Issue:** Docker build fails
```bash
# Solution: Clean build
docker compose build --no-cache
```

**Issue:** Port already in use
```bash
# Solution: Find and kill process
# Windows:
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

**Issue:** Out of memory
```bash
# Solution: Increase Docker memory
# Docker Desktop → Settings → Resources
# Or use CPU-only mode:
docker compose -f docker-compose.yml -f docker-compose.cpu.yml up -d
```

**Issue:** Git conflicts
```bash
# Solution: Rebase on main
git fetch origin
git rebase origin/main
# Resolve conflicts, then:
git rebase --continue
```

---

Happy coding! 🚀

For questions: adrian207@gmail.com

