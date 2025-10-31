#!/bin/bash
set -e

echo "=== RAG Service Starting ==="

# Wait for Ollama to be ready
echo "Waiting for Ollama service..."
MAX_RETRIES=30
RETRY_COUNT=0
until curl -s http://ollama:11434/api/tags > /dev/null 2>&1; do
  RETRY_COUNT=$((RETRY_COUNT + 1))
  if [ $RETRY_COUNT -ge $MAX_RETRIES ]; then
    echo "ERROR: Ollama service not available after ${MAX_RETRIES} attempts"
    exit 1
  fi
  echo "Attempt ${RETRY_COUNT}/${MAX_RETRIES}: Ollama not ready yet, waiting..."
  sleep 2
done
echo "✓ Ollama service is ready"

# Check if indexes exist, build if needed
if [ ! -d "/app/indexes/chroma_ms" ] || [ ! -d "/app/indexes/chroma_oss" ]; then
  echo "Building vector indexes..."
  
  # Create sample documents if data directories are empty
  if [ ! -d "/app/data/ms_docs" ] || [ -z "$(ls -A /app/data/ms_docs 2>/dev/null)" ]; then
    echo "Creating sample documents for testing..."
    python ingest_docs.py --create-samples
  fi
  
  # Build the indexes
  python ingest_docs.py
  
  if [ $? -eq 0 ]; then
    echo "✓ Indexes built successfully"
  else
    echo "⚠ Warning: Index build had errors, but continuing..."
  fi
else
  echo "✓ Indexes already exist"
fi

# Verify models are available in Ollama
echo "Checking available models in Ollama..."
curl -s http://ollama:11434/api/tags | python -c "
import sys, json
try:
    data = json.load(sys.stdin)
    models = [m['name'] for m in data.get('models', [])]
    print(f'Available models: {models}')
    
    required = ['qwen2.5-coder:32b-q4_K_M', 'deepseek-coder-v2:33b-q4_K_M']
    missing = [m for m in required if m not in models]
    
    if missing:
        print(f'⚠ Warning: Missing models: {missing}')
        print('Models need to be pulled in Ollama container first.')
        print('Run: docker exec ollama ollama pull <model_name>')
    else:
        print('✓ All required models available')
except Exception as e:
    print(f'Could not parse models: {e}')
" || echo "Could not check models"

echo "=== Starting FastAPI Server ==="
exec uvicorn rag_dual:app --host 0.0.0.0 --port 8000 --log-level info