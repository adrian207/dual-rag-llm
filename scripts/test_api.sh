#!/bin/bash
# Test API endpoints for Dual RAG LLM System
# Author: Adrian Johnson <adrian207@gmail.com>

set -e

API_URL="${API_URL:-http://localhost:8000}"

echo "=== Testing Dual RAG API ==="
echo "API URL: $API_URL"
echo ""

# Test health endpoint
echo "1. Testing health endpoint..."
HEALTH=$(curl -s "$API_URL/health")
echo "$HEALTH" | python -m json.tool
echo ""

# Test root endpoint
echo "2. Testing root endpoint..."
ROOT=$(curl -s "$API_URL/")
echo "$ROOT" | python -m json.tool
echo ""

# Test stats endpoint
echo "3. Testing stats endpoint..."
STATS=$(curl -s "$API_URL/stats")
echo "$STATS" | python -m json.tool
echo ""

# Test query with MS extension
echo "4. Testing query (C# example)..."
curl -s -X POST "$API_URL/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How do I create a list in C#?",
    "file_ext": ".cs"
  }' | python -m json.tool
echo ""

# Test query with OSS extension
echo "5. Testing query (Python example)..."
curl -s -X POST "$API_URL/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How do I create a list in Python?",
    "file_ext": ".py"
  }' | python -m json.tool
echo ""

echo "=== Tests Complete ==="
echo ""

