# Quick Start Guide - Dual RAG LLM System

**Author:** Adrian Johnson <adrian207@gmail.com>

Get up and running in 5 steps.

## Prerequisites

- Docker with GPU support
- NVIDIA Container Toolkit
- 32GB+ RAM
- 100GB+ free disk space

## Setup Steps

### 1. Automated Setup (Recommended)

```bash
./scripts/setup.sh
```

This script will:
- Verify prerequisites
- Create required directories
- Set up environment variables
- Pull Docker images
- Start Ollama
- Download LLM models (~40GB)
- Start all services

**Time required:** 30-60 minutes (mostly downloading models)

### 2. Manual Setup (Alternative)

```bash
# Create directories
mkdir -p rag/data/{ms_docs,oss_docs} rag/indexes rag/logs

# Setup environment
cp env.example .env
# Edit .env and change WEBUI_SECRET_KEY

# Start Ollama
docker compose up -d ollama

# Pull models (large downloads!)
docker exec ollama ollama pull qwen2.5-coder:32b-q4_K_M
docker exec ollama ollama pull deepseek-coder-v2:33b-q4_K_M

# Start all services
docker compose up -d
```

## Verify Installation

```bash
# Check all services are running
docker compose ps

# Test the API
./scripts/test_api.sh

# View logs
docker compose logs -f rag
```

## Access Points

- **Web UI**: http://localhost:3000 (create account on first visit)
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

## First Query

### Via API

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How do I iterate over a list?",
    "file_ext": ".py"
  }'
```

### Via Web UI

1. Go to http://localhost:3000
2. Create an account
3. Select a model (qwen2.5-coder or deepseek-coder-v2)
4. Ask your question

## Add Your Documentation

```bash
# Copy your docs
cp /path/to/docs/* rag/data/ms_docs/    # For C#, PowerShell, YAML
cp /path/to/docs/* rag/data/oss_docs/   # For Python, JS, etc.

# Rebuild indexes
./scripts/rebuild_indexes.sh

# Or manually:
docker compose exec rag python ingest_docs.py
docker compose restart rag
```

## Supported Document Formats

- PDF, Markdown, Text, HTML, reStructuredText, DOCX

## Common Issues

### "Ollama not available"
- Wait 30 seconds and retry
- Check: `docker compose logs ollama`

### "Models not found"
- Pull models: `docker exec ollama ollama pull <model-name>`
- Check: `docker exec ollama ollama list`

### "Index not found"
- Run: `./scripts/rebuild_indexes.sh`
- Or: `docker compose exec rag python ingest_docs.py --create-samples`

### Out of memory
- Close other applications
- Check Docker memory limits
- Consider smaller model quantizations

## Next Steps

1. Read the full [README.md](README.md)
2. Add your documentation files
3. Customize routing rules in `rag/rag_dual.py`
4. Explore the API at http://localhost:8000/docs

## Get Help

- View logs: `docker compose logs -f`
- Check health: `curl http://localhost:8000/health`
- Email: adrian207@gmail.com

