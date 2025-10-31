#!/bin/bash
# Rebuild indexes for Dual RAG LLM System
# Author: Adrian Johnson <adrian207@gmail.com>

set -e

echo "=== Rebuilding Vector Indexes ==="
echo ""

# Check if RAG service is running
if ! docker compose ps rag | grep -q "running"; then
    echo "ERROR: RAG service is not running"
    echo "Start it with: docker compose up -d rag"
    exit 1
fi

# Check data directories
echo "Checking data directories..."

MS_DOCS_COUNT=$(find rag/data/ms_docs -type f 2>/dev/null | wc -l)
OSS_DOCS_COUNT=$(find rag/data/oss_docs -type f 2>/dev/null | wc -l)

echo "  MS docs: $MS_DOCS_COUNT files"
echo "  OSS docs: $OSS_DOCS_COUNT files"
echo ""

if [ "$MS_DOCS_COUNT" -eq 0 ] && [ "$OSS_DOCS_COUNT" -eq 0 ]; then
    echo "No documents found. Creating sample documents..."
    docker compose exec rag python ingest_docs.py --create-samples
    echo ""
fi

# Backup existing indexes
if [ -d "rag/indexes/chroma_ms" ] || [ -d "rag/indexes/chroma_oss" ]; then
    BACKUP_NAME="indexes_backup_$(date +%Y%m%d_%H%M%S).tar.gz"
    echo "Backing up existing indexes to $BACKUP_NAME..."
    tar -czf "$BACKUP_NAME" rag/indexes/ 2>/dev/null || true
    echo "✓ Backup created"
    echo ""
fi

# Rebuild indexes
echo "Building indexes (this may take several minutes)..."
docker compose exec rag python ingest_docs.py

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Indexes rebuilt successfully"
    echo ""
    echo "Restarting RAG service to reload indexes..."
    docker compose restart rag
    echo "✓ Service restarted"
else
    echo ""
    echo "ERROR: Index building failed"
    exit 1
fi

echo ""
echo "=== Index Rebuild Complete ==="
echo ""

