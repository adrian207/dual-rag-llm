#!/bin/bash
# Setup script for Dual RAG LLM System
# Author: Adrian Johnson <adrian207@gmail.com>

set -e

echo "=== Dual RAG LLM System Setup ==="
echo ""

# Check prerequisites
echo "Checking prerequisites..."

if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker is not installed"
    exit 1
fi

if ! command -v docker compose &> /dev/null; then
    echo "ERROR: Docker Compose is not installed"
    exit 1
fi

if ! command -v nvidia-smi &> /dev/null; then
    echo "WARNING: nvidia-smi not found. GPU support may not be available."
fi

echo "✓ Prerequisites OK"
echo ""

# Create directories
echo "Creating required directories..."
mkdir -p rag/data/ms_docs
mkdir -p rag/data/oss_docs
mkdir -p rag/indexes
mkdir -p rag/logs
mkdir -p workspace
echo "✓ Directories created"
echo ""

# Setup environment file
if [ ! -f .env ]; then
    echo "Creating .env file from template..."
    cp env.example .env
    
    # Generate random secret key
    SECRET=$(openssl rand -hex 32 2>/dev/null || echo "change-this-secret-key-$(date +%s)")
    sed -i "s/your-secret-key-here-change-this/$SECRET/g" .env
    
    echo "✓ .env file created with random secret key"
else
    echo "✓ .env file already exists"
fi
echo ""

# Pull base images
echo "Pulling Docker images (this may take a while)..."
docker compose pull
echo "✓ Images pulled"
echo ""

# Start Ollama
echo "Starting Ollama service..."
docker compose up -d ollama
echo "Waiting for Ollama to be ready..."
sleep 10

MAX_RETRIES=30
RETRY_COUNT=0
until curl -s http://localhost:11434/api/tags > /dev/null 2>&1; do
    RETRY_COUNT=$((RETRY_COUNT + 1))
    if [ $RETRY_COUNT -ge $MAX_RETRIES ]; then
        echo "ERROR: Ollama did not start in time"
        exit 1
    fi
    echo "Waiting... (${RETRY_COUNT}/${MAX_RETRIES})"
    sleep 2
done
echo "✓ Ollama is ready"
echo ""

# Pull models
echo "Pulling LLM models (this will take 20-40 minutes)..."
echo "Pulling Qwen 2.5 Coder (32B)..."
docker exec ollama ollama pull qwen2.5-coder:32b-q4_K_M

echo "Pulling DeepSeek Coder V2 (33B)..."
docker exec ollama ollama pull deepseek-coder-v2:33b-q4_K_M

echo "✓ Models pulled"
echo ""

# Verify models
echo "Available models:"
docker exec ollama ollama list
echo ""

# Start remaining services
echo "Starting all services..."
docker compose up -d
echo "✓ All services started"
echo ""

# Wait for services
echo "Waiting for services to be ready..."
sleep 15

# Check health
echo "Checking service health..."
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "✓ RAG service is healthy"
else
    echo "⚠ RAG service may not be ready yet (check logs)"
fi
echo ""

echo "=== Setup Complete ==="
echo ""
echo "Services:"
echo "  - Open-WebUI: http://localhost:3000"
echo "  - RAG API: http://localhost:8000"
echo "  - API Docs: http://localhost:8000/docs"
echo ""
echo "Next steps:"
echo "  1. Add documentation to rag/data/ms_docs/ and rag/data/oss_docs/"
echo "  2. Rebuild indexes: docker compose exec rag python ingest_docs.py"
echo "  3. Access Open-WebUI at http://localhost:3000"
echo ""
echo "View logs: docker compose logs -f"
echo ""

