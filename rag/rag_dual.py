"""
Dual-Strategy RAG System for Microsoft and Open Source Technologies

Author: Adrian Johnson <adrian207@gmail.com>

Routes queries to specialized LLMs and knowledge bases based on file extensions.
Microsoft technologies use Qwen 2.5 Coder, OSS uses DeepSeek Coder V2.
"""

import os
import asyncio
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager

import structlog
import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from llama_index.core import VectorStoreIndex, Settings, StorageContext
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from chromadb import PersistentClient

# Configure structured logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.JSONRenderer()
    ]
)
logger = structlog.get_logger()

# Global state for caching
class AppState:
    """Application state container for cached resources"""
    ms_index: Optional[VectorStoreIndex] = None
    oss_index: Optional[VectorStoreIndex] = None
    http_client: Optional[httpx.AsyncClient] = None
    models_loaded: set = set()

app_state = AppState()

# Configuration
OLLAMA_API = os.getenv("OLLAMA_API", "http://ollama:11434")
MS_MODEL = "qwen2.5-coder:32b-q4_K_M"
OSS_MODEL = "deepseek-coder-v2:33b-q4_K_M"
MS_EXTENSIONS = {'.cs', '.ps1', '.yaml', '.yml', '.xaml'}
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Request/Response Models
class Query(BaseModel):
    """Query request model"""
    question: str = Field(..., min_length=1, max_length=2000, description="User question")
    file_ext: str = Field(default="", description="File extension to route query")

class QueryResponse(BaseModel):
    """Query response model"""
    answer: str
    model: str
    source: str
    chunks_retrieved: int
    tokens_used: Optional[int] = None

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    ollama_connected: bool
    ms_index_loaded: bool
    oss_index_loaded: bool
    models_available: list

async def check_ollama_health() -> bool:
    """Check if Ollama service is reachable"""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{OLLAMA_API}/api/tags")
            return response.status_code == 200
    except Exception as e:
        logger.error("ollama_health_check_failed", error=str(e))
        return False

async def ensure_model_loaded(model_name: str) -> bool:
    """
    Ensure model is loaded in Ollama. Returns True if successful.
    Uses caching to avoid repeated checks.
    """
    if model_name in app_state.models_loaded:
        return True
    
    try:
        if not app_state.http_client:
            app_state.http_client = httpx.AsyncClient(timeout=300.0)
        
        # Check if model exists
        response = await app_state.http_client.get(f"{OLLAMA_API}/api/tags")
        response.raise_for_status()
        
        models = response.json().get("models", [])
        model_names = [m.get("name") for m in models]
        
        if model_name not in model_names:
            logger.warning("model_not_found", model=model_name, available=model_names)
            return False
        
        app_state.models_loaded.add(model_name)
        logger.info("model_verified", model=model_name)
        return True
        
    except Exception as e:
        logger.error("model_check_failed", model=model_name, error=str(e))
        return False

def get_model_for_extension(file_ext: str) -> tuple[str, str]:
    """
    Determine which model and source to use based on file extension.
    Returns (model_name, source_type)
    """
    if file_ext.lower() in MS_EXTENSIONS:
        return MS_MODEL, "Microsoft"
    return OSS_MODEL, "OpenSource"

def load_index(is_ms: bool) -> Optional[VectorStoreIndex]:
    """
    Load vector index from ChromaDB.
    Uses caching to avoid repeated disk reads.
    """
    try:
        # Check cache first
        if is_ms and app_state.ms_index:
            return app_state.ms_index
        if not is_ms and app_state.oss_index:
            return app_state.oss_index
        
        # Load from disk
        path = "/app/indexes/chroma_ms" if is_ms else "/app/indexes/chroma_oss"
        collection_name = "msdocs" if is_ms else "ossdocs"
        
        if not os.path.exists(path):
            logger.error("index_not_found", path=path)
            return None
        
        client = PersistentClient(path=path)
        collection = client.get_collection(collection_name)
        vector_store = ChromaVectorStore(chroma_collection=collection)
        index = VectorStoreIndex.from_vector_store(vector_store)
        
        # Cache the index
        if is_ms:
            app_state.ms_index = index
        else:
            app_state.oss_index = index
        
        logger.info("index_loaded", source="MS" if is_ms else "OSS", path=path)
        return index
        
    except Exception as e:
        logger.error("index_load_failed", is_ms=is_ms, error=str(e))
        return None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown"""
    logger.info("application_starting")
    
    # Initialize global settings
    Settings.embed_model = HuggingFaceEmbedding(EMBEDDING_MODEL)
    logger.info("embedding_model_initialized", model=EMBEDDING_MODEL)
    
    # Initialize HTTP client
    app_state.http_client = httpx.AsyncClient(timeout=300.0)
    
    # Pre-warm indexes (don't fail if missing)
    try:
        app_state.ms_index = load_index(is_ms=True)
        app_state.oss_index = load_index(is_ms=False)
    except Exception as e:
        logger.warning("index_preload_failed", error=str(e))
    
    # Verify Ollama connection
    if await check_ollama_health():
        logger.info("ollama_connected")
    else:
        logger.warning("ollama_not_available")
    
    yield
    
    # Shutdown
    logger.info("application_shutting_down")
    if app_state.http_client:
        await app_state.http_client.aclose()

app = FastAPI(
    title="Dual RAG LLM System",
    description="Intelligent routing between MS and OSS LLM models",
    version="1.0.0",
    lifespan=lifespan
)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled errors"""
    logger.error("unhandled_exception", path=request.url.path, error=str(exc))
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "type": type(exc).__name__}
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for container orchestration"""
    ollama_ok = await check_ollama_health()
    
    models_available = []
    if ollama_ok and app_state.http_client:
        try:
            response = await app_state.http_client.get(f"{OLLAMA_API}/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                models_available = [m.get("name") for m in models]
        except:
            pass
    
    return HealthResponse(
        status="healthy" if ollama_ok else "degraded",
        ollama_connected=ollama_ok,
        ms_index_loaded=app_state.ms_index is not None,
        oss_index_loaded=app_state.oss_index is not None,
        models_available=models_available
    )

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "service": "Dual RAG LLM System",
        "version": "1.0.0",
        "endpoints": {
            "POST /query": "Submit a question with optional file_ext",
            "GET /health": "Health check",
            "GET /stats": "System statistics"
        }
    }

@app.get("/stats")
async def get_stats():
    """Get system statistics"""
    return {
        "models_cached": list(app_state.models_loaded),
        "ms_index_loaded": app_state.ms_index is not None,
        "oss_index_loaded": app_state.oss_index is not None,
        "ollama_api": OLLAMA_API
    }

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(q: Query):
    """
    Main query endpoint. Routes to appropriate model and knowledge base.
    
    Args:
        q: Query object with question and optional file_ext
        
    Returns:
        QueryResponse with answer, model used, and metadata
        
    Raises:
        HTTPException: If query processing fails
    """
    logger.info("query_received", question_length=len(q.question), file_ext=q.file_ext)
    
    try:
        # Determine routing
        model_name, source_type = get_model_for_extension(q.file_ext)
        is_ms = source_type == "Microsoft"
        
        # Ensure model is available
        model_available = await ensure_model_loaded(model_name)
        if not model_available:
            raise HTTPException(
                status_code=503,
                detail=f"Model {model_name} not available in Ollama"
            )
        
        # Load appropriate index
        index = load_index(is_ms)
        if index is None:
            raise HTTPException(
                status_code=503,
                detail=f"Index for {source_type} not available. Run index builder first."
            )
        
        # Retrieve relevant context
        retriever = index.as_retriever(similarity_top_k=3)
        nodes = retriever.retrieve(q.question)
        
        if not nodes:
            logger.warning("no_nodes_retrieved", question=q.question[:50])
            context = "No relevant context found."
        else:
            context = "\n\n".join([f"[Document {i+1}]\n{n.text}" for i, n in enumerate(nodes)])
        
        # Generate response using LLM
        llm = Ollama(model=model_name, request_timeout=120.0, base_url=OLLAMA_API)
        
        prompt = f"""Context from documentation:
{context}

Question: {q.question}

Please provide a detailed answer based on the context above. Include code examples if relevant."""
        
        response = llm.complete(prompt)
        
        logger.info(
            "query_completed",
            model=model_name,
            source=source_type,
            chunks=len(nodes),
            answer_length=len(response.text)
        )
        
        return QueryResponse(
            answer=response.text,
            model=model_name,
            source=source_type,
            chunks_retrieved=len(nodes)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("query_failed", error=str(e), error_type=type(e).__name__)
        raise HTTPException(
            status_code=500,
            detail=f"Query processing failed: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
