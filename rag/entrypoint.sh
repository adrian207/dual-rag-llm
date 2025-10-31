#!/bin/bash
set -e

# Download models
if [ ! -f "/app/data/qwen.gguf" ]; then
  curl -L -o /app/data/qwen.gguf \
    "https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct-GGUF/resolve/main/Qwen2.5-Coder-32B-Instruct-Q4_K_M.gguf"
fi
if [ ! -f "/app/data/deepseek.gguf" ]; then
  curl -L -o /app/data/deepseek.gguf \
    "https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Instruct-GGUF/resolve/main/DeepSeek-Coder-V2-033B-Instruct-Q4_K_M.gguf"
fi

# Create in Ollama
ollama create qwen2.5-coder:32b-q4_K_M -f /app/data/qwen.gguf || true
ollama create deepseek-coder-v2:33b-q4_K_M -f /app/data/deepseek.gguf || true

# Build indexes (if not exist)
python -c "
from rag_dual import build_indexes
if not os.path.exists('/app/indexes/msdocs_index'):
    build_indexes()
" || echo "Indexes already built"

exec uvicorn rag_dual:app --host 0.0.0.0 --port 8000