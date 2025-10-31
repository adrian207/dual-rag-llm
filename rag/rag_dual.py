"""
Dual-Strategy RAG System with Caching and Web Tools

Author: Adrian Johnson <adrian207@gmail.com>

Features:
- Routes queries to specialized LLMs based on file extensions
- Redis caching for fast responses
- Brave Search for web queries
- GitHub integration for code search
"""

import os
import asyncio
import hashlib
import json
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager
from datetime import datetime, timedelta

import structlog
import httpx
import redis.asyncio as aioredis
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse
from github import Github, GithubException

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
    redis_client: Optional[aioredis.Redis] = None
    github_client: Optional[Github] = None
    models_loaded: set = set()
    cache_stats: Dict[str, int] = {"hits": 0, "misses": 0, "errors": 0}
    tool_usage: Dict[str, int] = {"web_search": 0, "github": 0, "rag": 0}
    model_performance: Dict[str, Dict[str, Any]] = {}  # Track performance per model

app_state = AppState()

# Configuration
OLLAMA_API = os.getenv("OLLAMA_API", "http://ollama:11434")
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379")
REDIS_CACHE_TTL = int(os.getenv("REDIS_CACHE_TTL", "86400"))  # 24 hours
BRAVE_API_KEY = os.getenv("BRAVE_API_KEY", "")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
ENABLE_WEB_SEARCH = os.getenv("ENABLE_WEB_SEARCH", "true").lower() == "true"
ENABLE_GITHUB = os.getenv("ENABLE_GITHUB", "true").lower() == "true"
ENABLE_CACHING = os.getenv("ENABLE_CACHING", "true").lower() == "true"

MS_MODEL = os.getenv("MS_MODEL", "qwen2.5-coder:32b-q4_K_M")
OSS_MODEL = os.getenv("OSS_MODEL", "deepseek-coder-v2:33b-q4_K_M")
MS_EXTENSIONS = {'.cs', '.ps1', '.yaml', '.yml', '.xaml'}
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# Request/Response Models
class Query(BaseModel):
    """Query request model"""
    question: str = Field(..., min_length=1, max_length=2000)
    file_ext: str = Field(default="")
    use_web_search: bool = Field(default=False)
    use_github: bool = Field(default=False)
    github_repo: Optional[str] = Field(default=None)
    model_override: Optional[str] = Field(default=None, description="Override automatic model selection")
    compare_models: bool = Field(default=False, description="Compare responses from multiple models")

class ToolResult(BaseModel):
    """Result from a tool (web search or GitHub)"""
    tool: str
    results: List[Dict[str, Any]]
    count: int

class QueryResponse(BaseModel):
    """Query response model"""
    answer: str
    model: str
    source: str
    chunks_retrieved: int
    cached: bool = False
    cache_age_seconds: Optional[int] = None
    tools_used: List[str] = []
    tool_results: List[ToolResult] = []
    response_time_ms: Optional[int] = None

class CacheStats(BaseModel):
    """Cache statistics"""
    hits: int
    misses: int
    errors: int
    hit_rate: float
    total_requests: int

class ToolStats(BaseModel):
    """Tool usage statistics"""
    web_search: int
    github: int
    rag: int
    total: int

class ModelPerformance(BaseModel):
    """Model performance metrics"""
    queries: int
    total_time: float
    total_tokens: int
    avg_tokens_per_sec: float
    avg_response_time: float

class SystemStats(BaseModel):
    """System statistics"""
    cache: CacheStats
    tools: ToolStats
    models_cached: List[str]
    ms_index_loaded: bool
    oss_index_loaded: bool
    model_performance: Dict[str, ModelPerformance] = {}

# Helper functions
def generate_cache_key(question: str, file_ext: str, model: str) -> str:
    """Generate cache key for a query"""
    content = f"{question}:{file_ext}:{model}"
    return f"query:{hashlib.sha256(content.encode()).hexdigest()}"

async def get_cached_response(cache_key: str) -> Optional[Dict]:
    """Get cached response from Redis"""
    if not ENABLE_CACHING or not app_state.redis_client:
        return None
    
    try:
        cached = await app_state.redis_client.get(cache_key)
        if cached:
            app_state.cache_stats["hits"] += 1
            data = json.loads(cached)
            # Calculate cache age
            cached_time = datetime.fromisoformat(data.get("cached_at", datetime.now().isoformat()))
            age = (datetime.now() - cached_time).total_seconds()
            data["cache_age_seconds"] = int(age)
            logger.info("cache_hit", key=cache_key, age_seconds=age)
            return data
        app_state.cache_stats["misses"] += 1
        return None
    except Exception as e:
        app_state.cache_stats["errors"] += 1
        logger.error("cache_get_error", error=str(e))
        return None

async def set_cached_response(cache_key: str, data: Dict) -> None:
    """Set cached response in Redis"""
    if not ENABLE_CACHING or not app_state.redis_client:
        return
    
    try:
        data["cached_at"] = datetime.now().isoformat()
        await app_state.redis_client.setex(
            cache_key,
            REDIS_CACHE_TTL,
            json.dumps(data)
        )
        logger.info("cache_set", key=cache_key, ttl=REDIS_CACHE_TTL)
    except Exception as e:
        logger.error("cache_set_error", error=str(e))

async def search_web(query: str, max_results: int = 5) -> ToolResult:
    """Search the web using Brave Search API"""
    if not ENABLE_WEB_SEARCH or not BRAVE_API_KEY:
        return ToolResult(tool="web_search", results=[], count=0)
    
    app_state.tool_usage["web_search"] += 1
    
    try:
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": BRAVE_API_KEY
        }
        
        params = {
            "q": query,
            "count": max_results
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://api.search.brave.com/res/v1/web/search",
                headers=headers,
                params=params,
                timeout=10.0
            )
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            for item in data.get("web", {}).get("results", [])[:max_results]:
                results.append({
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "description": item.get("description", ""),
                    "age": item.get("age", "")
                })
            
            logger.info("web_search_completed", query=query, results=len(results))
            return ToolResult(tool="web_search", results=results, count=len(results))
            
    except Exception as e:
        logger.error("web_search_error", error=str(e))
        return ToolResult(tool="web_search", results=[], count=0)

async def search_github(query: str, repo: Optional[str] = None, max_results: int = 10) -> ToolResult:
    """Search GitHub for code or repositories"""
    if not ENABLE_GITHUB or not GITHUB_TOKEN or not app_state.github_client:
        return ToolResult(tool="github", results=[], count=0)
    
    app_state.tool_usage["github"] += 1
    
    try:
        results = []
        
        if repo:
            # Search within a specific repository
            repo_obj = app_state.github_client.get_repo(repo)
            contents = repo_obj.search_code(query)
            
            for item in contents[:max_results]:
                results.append({
                    "name": item.name,
                    "path": item.path,
                    "url": item.html_url,
                    "repository": repo,
                    "sha": item.sha
                })
        else:
            # Search across GitHub
            code_results = app_state.github_client.search_code(query)
            
            for item in code_results[:max_results]:
                results.append({
                    "name": item.name,
                    "path": item.path,
                    "url": item.html_url,
                    "repository": item.repository.full_name,
                    "sha": item.sha
                })
        
        logger.info("github_search_completed", query=query, repo=repo, results=len(results))
        return ToolResult(tool="github", results=results, count=len(results))
        
    except GithubException as e:
        logger.error("github_search_error", error=str(e))
        return ToolResult(tool="github", results=[], count=0)
    except Exception as e:
        logger.error("github_search_error", error=str(e))
        return ToolResult(tool="github", results=[], count=0)

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
    """Ensure model is loaded in Ollama"""
    if model_name in app_state.models_loaded:
        return True
    
    try:
        if not app_state.http_client:
            app_state.http_client = httpx.AsyncClient(timeout=300.0)
        
        response = await app_state.http_client.get(f"{OLLAMA_API}/api/tags")
        response.raise_for_status()
        
        models = response.json().get("models", [])
        model_names = [m.get("name") for m in models]
        
        if model_name not in model_names:
            logger.warning("model_not_found", model=model_name)
            return False
        
        app_state.models_loaded.add(model_name)
        logger.info("model_verified", model=model_name)
        return True
        
    except Exception as e:
        logger.error("model_check_failed", model=model_name, error=str(e))
        return False

def get_model_for_extension(file_ext: str) -> tuple[str, str]:
    """Determine which model and source to use"""
    if file_ext.lower() in MS_EXTENSIONS:
        return MS_MODEL, "Microsoft"
    return OSS_MODEL, "OpenSource"

def load_index(is_ms: bool) -> Optional[VectorStoreIndex]:
    """Load vector index from ChromaDB"""
    try:
        if is_ms and app_state.ms_index:
            return app_state.ms_index
        if not is_ms and app_state.oss_index:
            return app_state.oss_index
        
        path = "/app/indexes/chroma_ms" if is_ms else "/app/indexes/chroma_oss"
        collection_name = "msdocs" if is_ms else "ossdocs"
        
        if not os.path.exists(path):
            logger.error("index_not_found", path=path)
            return None
        
        client = PersistentClient(path=path)
        collection = client.get_collection(collection_name)
        vector_store = ChromaVectorStore(chroma_collection=collection)
        index = VectorStoreIndex.from_vector_store(vector_store)
        
        if is_ms:
            app_state.ms_index = index
        else:
            app_state.oss_index = index
        
        logger.info("index_loaded", source="MS" if is_ms else "OSS")
        return index
        
    except Exception as e:
        logger.error("index_load_failed", is_ms=is_ms, error=str(e))
        return None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    logger.info("application_starting")
    
    # Initialize settings
    Settings.embed_model = HuggingFaceEmbedding(EMBEDDING_MODEL)
    logger.info("embedding_model_initialized")
    
    # Initialize HTTP client
    app_state.http_client = httpx.AsyncClient(timeout=300.0)
    
    # Initialize Redis
    if ENABLE_CACHING:
        try:
            app_state.redis_client = aioredis.from_url(
                REDIS_URL,
                encoding="utf-8",
                decode_responses=False
            )
            await app_state.redis_client.ping()
            logger.info("redis_connected")
        except Exception as e:
            logger.error("redis_connection_failed", error=str(e))
    
    # Initialize GitHub client
    if ENABLE_GITHUB and GITHUB_TOKEN:
        try:
            app_state.github_client = Github(GITHUB_TOKEN)
            app_state.github_client.get_user().login  # Test connection
            logger.info("github_connected")
        except Exception as e:
            logger.error("github_connection_failed", error=str(e))
    
    # Pre-warm indexes
    try:
        app_state.ms_index = load_index(is_ms=True)
        app_state.oss_index = load_index(is_ms=False)
    except Exception as e:
        logger.warning("index_preload_failed", error=str(e))
    
    # Verify Ollama
    if await check_ollama_health():
        logger.info("ollama_connected")
    else:
        logger.warning("ollama_not_available")
    
    yield
    
    # Shutdown
    logger.info("application_shutting_down")
    if app_state.http_client:
        await app_state.http_client.aclose()
    if app_state.redis_client:
        await app_state.redis_client.close()

app = FastAPI(
    title="Dual RAG LLM System with Web Tools",
    description="Intelligent routing with caching, web search, and GitHub integration",
    version="1.2.0",
    lifespan=lifespan
)

# Enable CORS for web UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for web UI
try:
    app.mount("/ui", StaticFiles(directory="/app/ui", html=True), name="ui")
except RuntimeError:
    logger.warning("ui_directory_not_found", path="/app/ui")

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error("unhandled_exception", path=request.url.path, error=str(exc))
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "type": type(exc).__name__}
    )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    ollama_ok = await check_ollama_health()
    redis_ok = False
    
    if app_state.redis_client:
        try:
            await app_state.redis_client.ping()
            redis_ok = True
        except:
            pass
    
    return {
        "status": "healthy" if ollama_ok else "degraded",
        "ollama_connected": ollama_ok,
        "redis_connected": redis_ok,
        "ms_index_loaded": app_state.ms_index is not None,
        "oss_index_loaded": app_state.oss_index is not None,
        "features": {
            "caching": ENABLE_CACHING and redis_ok,
            "web_search": ENABLE_WEB_SEARCH and bool(BRAVE_API_KEY),
            "github": ENABLE_GITHUB and app_state.github_client is not None
        }
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Dual RAG LLM System with Web Tools",
        "version": "1.1.0",
        "features": ["RAG", "Caching", "Web Search", "GitHub Integration"],
        "endpoints": {
            "POST /query": "Submit a question with optional tools",
            "GET /health": "Health check",
            "GET /stats": "System statistics",
            "POST /cache/clear": "Clear cache",
            "GET /cache/stats": "Cache statistics"
        }
    }

@app.get("/stats", response_model=SystemStats)
async def get_stats():
    """Get system statistics"""
    total_cache = sum(app_state.cache_stats.values())
    hits = app_state.cache_stats["hits"]
    hit_rate = (hits / total_cache * 100) if total_cache > 0 else 0.0
    
    total_tools = sum(app_state.tool_usage.values())
    
    # Convert model_performance to ModelPerformance objects
    model_perf_objects = {}
    for model, stats in app_state.model_performance.items():
        model_perf_objects[model] = ModelPerformance(**stats)
    
    return SystemStats(
        cache=CacheStats(
            hits=app_state.cache_stats["hits"],
            misses=app_state.cache_stats["misses"],
            errors=app_state.cache_stats["errors"],
            hit_rate=round(hit_rate, 2),
            total_requests=total_cache
        ),
        tools=ToolStats(
            web_search=app_state.tool_usage["web_search"],
            github=app_state.tool_usage["github"],
            rag=app_state.tool_usage["rag"],
            total=total_tools
        ),
        models_cached=list(app_state.models_loaded),
        ms_index_loaded=app_state.ms_index is not None,
        oss_index_loaded=app_state.oss_index is not None,
        model_performance=model_perf_objects
    )

@app.post("/cache/clear")
async def clear_cache():
    """Clear Redis cache"""
    if not app_state.redis_client:
        raise HTTPException(status_code=503, detail="Redis not available")
    
    try:
        await app_state.redis_client.flushdb()
        app_state.cache_stats = {"hits": 0, "misses": 0, "errors": 0}
        logger.info("cache_cleared")
        return {"status": "success", "message": "Cache cleared"}
    except Exception as e:
        logger.error("cache_clear_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models/available")
async def get_available_models():
    """Get list of available models from Ollama"""
    try:
        async with app_state.http_client.stream("GET", f"{OLLAMA_API}/api/tags") as response:
            if response.status_code == 200:
                data = await response.aread()
                models_data = json.loads(data)
                models = [m["name"] for m in models_data.get("models", [])]
                return {
                    "available": models,
                    "loaded": list(app_state.models_loaded),
                    "performance": app_state.model_performance
                }
            else:
                return {"available": [], "loaded": list(app_state.models_loaded)}
    except Exception as e:
        logger.error("failed_to_fetch_models", error=str(e))
        return {"available": [], "loaded": list(app_state.models_loaded)}

@app.post("/query/compare")
async def compare_models(q: Query):
    """Compare responses from multiple models"""
    logger.info("comparison_request", question_length=len(q.question))
    
    # Define models to compare
    comparison_models = [
        ("qwen2.5-coder:32b-q4_K_M", "Microsoft"),
        ("deepseek-coder-v2:33b-q4_K_M", "Open Source")
    ]
    
    # If user specified a model, use it for one of the comparisons
    if q.model_override and q.model_override not in [m[0] for m in comparison_models]:
        comparison_models.append((q.model_override, "User Selected"))
    
    results = []
    
    for model_name, source_type in comparison_models:
        try:
            # Track time
            start_time = time.time()
            
            # Ensure model is available
            model_available = await ensure_model_loaded(model_name)
            if not model_available:
                results.append({
                    "model": model_name,
                    "source": source_type,
                    "error": "Model not available",
                    "response_time": 0
                })
                continue
            
            # Use the same logic as regular query
            is_ms = "qwen" in model_name.lower()
            index = load_index(is_ms)
            
            if index is None:
                results.append({
                    "model": model_name,
                    "source": source_type,
                    "error": "Index not available",
                    "response_time": 0
                })
                continue
            
            # Retrieve context
            retriever = index.as_retriever(similarity_top_k=3)
            nodes = retriever.retrieve(q.question)
            context = "\n\n".join([f"[Doc {i+1}]\n{n.text}" for i, n in enumerate(nodes)]) if nodes else "No context available."
            
            # Generate response
            llm = Ollama(model=model_name, request_timeout=120.0, base_url=OLLAMA_API)
            prompt = f"""Context from documentation:
{context}

Question: {q.question}

Please provide a detailed answer based on the context above. Include code examples if relevant."""
            
            response = llm.complete(prompt)
            response_time = time.time() - start_time
            
            results.append({
                "model": model_name,
                "source": source_type,
                "answer": response.text,
                "chunks_retrieved": len(nodes),
                "response_time": round(response_time, 2),
                "performance": app_state.model_performance.get(model_name, {})
            })
            
        except Exception as e:
            logger.error("comparison_model_failed", model=model_name, error=str(e))
            results.append({
                "model": model_name,
                "source": source_type,
                "error": str(e),
                "response_time": 0
            })
    
    return {
        "question": q.question,
        "results": results,
        "comparison_count": len(results)
    }

async def generate_streaming_response(q: Query):
    """Generate streaming response for query"""
    try:
        # Determine routing (with override support)
        if q.model_override:
            model_name = q.model_override
            # Determine source type based on model name
            if "qwen" in model_name.lower():
                source_type = "Microsoft"
                is_ms = True
            elif "deepseek" in model_name.lower() or "codellama" in model_name.lower():
                source_type = "Open Source"
                is_ms = False
            else:
                source_type = "General"
                is_ms = False
        else:
            model_name, source_type = get_model_for_extension(q.file_ext)
            is_ms = source_type == "Microsoft"
        
        # Send initial status
        yield {
            "event": "status",
            "data": json.dumps({"status": "starting", "model": model_name, "source": source_type})
        }
        
        # Track start time for performance metrics
        start_time = time.time()
        
        # Check cache
        cache_key = generate_cache_key(q.question, q.file_ext, model_name)
        cached_response = await get_cached_response(cache_key)
        
        if cached_response:
            yield {
                "event": "cached",
                "data": json.dumps(cached_response)
            }
            yield {
                "event": "done",
                "data": json.dumps({"cached": True})
            }
            return
        
        # Use tools if requested
        tools_used = []
        tool_results = []
        
        if q.use_web_search:
            yield {
                "event": "status",
                "data": json.dumps({"status": "searching_web"})
            }
            web_result = await search_web(q.question)
            if web_result.count > 0:
                tools_used.append("web_search")
                tool_results.append(web_result)
                yield {
                    "event": "tool",
                    "data": json.dumps({"tool": "web_search", "count": web_result.count})
                }
        
        if q.use_github:
            yield {
                "event": "status",
                "data": json.dumps({"status": "searching_github"})
            }
            github_result = await search_github(q.question, q.github_repo)
            if github_result.count > 0:
                tools_used.append("github")
                tool_results.append(github_result)
                yield {
                    "event": "tool",
                    "data": json.dumps({"tool": "github", "count": github_result.count})
                }
        
        # Ensure model available
        yield {
            "event": "status",
            "data": json.dumps({"status": "loading_model"})
        }
        model_available = await ensure_model_loaded(model_name)
        if not model_available:
            yield {
                "event": "error",
                "data": json.dumps({"error": f"Model {model_name} not available"})
            }
            return
        
        # Load RAG index
        yield {
            "event": "status",
            "data": json.dumps({"status": "retrieving_context"})
        }
        index = load_index(is_ms)
        if index is None:
            yield {
                "event": "error",
                "data": json.dumps({"error": f"Index for {source_type} not available"})
            }
            return
        
        app_state.tool_usage["rag"] += 1
        
        # Retrieve context
        retriever = index.as_retriever(similarity_top_k=3)
        nodes = retriever.retrieve(q.question)
        
        context_parts = []
        if nodes:
            context_parts = [f"[Doc {i+1}]\n{n.text}" for i, n in enumerate(nodes)]
        
        # Add tool context
        for result in tool_results:
            if result.tool == "web_search":
                for r in result.results[:3]:
                    context_parts.append(f"[Web] {r['title']}: {r['description']}")
            elif result.tool == "github":
                for r in result.results[:3]:
                    context_parts.append(f"[GitHub] {r['repository']}/{r['path']}")
        
        context = "\n\n".join(context_parts) if context_parts else "No context available."
        
        # Generate streaming response from LLM
        yield {
            "event": "status",
            "data": json.dumps({"status": "generating", "chunks": len(nodes)})
        }
        
        llm = Ollama(model=model_name, request_timeout=120.0, base_url=OLLAMA_API)
        
        prompt = f"""Context from documentation and tools:
{context}

Question: {q.question}

Please provide a detailed answer based on the context above. Include code examples if relevant."""
        
        # Stream the LLM response
        full_answer = ""
        token_count = 0
        async for token in llm.astream_complete(prompt):
            full_answer += token.delta
            token_count += 1
            yield {
                "event": "token",
                "data": json.dumps({"token": token.delta})
            }
        
        # Calculate performance metrics
        end_time = time.time()
        total_time = end_time - start_time
        tokens_per_second = token_count / total_time if total_time > 0 else 0
        
        # Update model performance stats
        if model_name not in app_state.model_performance:
            app_state.model_performance[model_name] = {
                "queries": 0,
                "total_time": 0,
                "total_tokens": 0,
                "avg_tokens_per_sec": 0,
                "avg_response_time": 0
            }
        
        stats = app_state.model_performance[model_name]
        stats["queries"] += 1
        stats["total_time"] += total_time
        stats["total_tokens"] += token_count
        stats["avg_tokens_per_sec"] = stats["total_tokens"] / stats["total_time"]
        stats["avg_response_time"] = stats["total_time"] / stats["queries"]
        
        # Cache the complete response
        result = {
            "answer": full_answer,
            "model": model_name,
            "source": source_type,
            "chunks_retrieved": len(nodes),
            "tools_used": tools_used,
            "tool_results": [tr.dict() for tr in tool_results],
            "performance": {
                "response_time": round(total_time, 2),
                "tokens": token_count,
                "tokens_per_sec": round(tokens_per_second, 2)
            }
        }
        
        await set_cached_response(cache_key, result)
        
        # Send completion
        yield {
            "event": "done",
            "data": json.dumps(result)
        }
        
    except Exception as e:
        logger.error("streaming_error", error=str(e))
        yield {
            "event": "error",
            "data": json.dumps({"error": str(e)})
        }

@app.post("/query/stream")
async def query_stream_endpoint(q: Query):
    """Streaming query endpoint"""
    logger.info("stream_query_received", question_length=len(q.question))
    
    async def event_generator():
        async for event in generate_streaming_response(q):
            yield f"event: {event['event']}\ndata: {event['data']}\n\n"
    
    return EventSourceResponse(event_generator())

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(q: Query):
    """Main query endpoint with tools (non-streaming)"""
    start_time = asyncio.get_event_loop().time()
    logger.info("query_received", question_length=len(q.question), file_ext=q.file_ext)
    
    try:
        # Determine routing
        model_name, source_type = get_model_for_extension(q.file_ext)
        is_ms = source_type == "Microsoft"
        
        # Check cache
        cache_key = generate_cache_key(q.question, q.file_ext, model_name)
        cached_response = await get_cached_response(cache_key)
        
        if cached_response:
            response_time = int((asyncio.get_event_loop().time() - start_time) * 1000)
            return QueryResponse(
                **cached_response,
                cached=True,
                response_time_ms=response_time
            )
        
        # Use tools if requested
        tools_used = []
        tool_results = []
        additional_context = []
        
        if q.use_web_search:
            web_result = await search_web(q.question)
            if web_result.count > 0:
                tools_used.append("web_search")
                tool_results.append(web_result)
                for r in web_result.results[:3]:
                    additional_context.append(f"[Web] {r['title']}: {r['description']}")
        
        if q.use_github:
            github_result = await search_github(q.question, q.github_repo)
            if github_result.count > 0:
                tools_used.append("github")
                tool_results.append(github_result)
                for r in github_result.results[:3]:
                    additional_context.append(f"[GitHub] {r['repository']}/{r['path']}")
        
        # Ensure model available
        model_available = await ensure_model_loaded(model_name)
        if not model_available:
            raise HTTPException(
                status_code=503,
                detail=f"Model {model_name} not available"
            )
        
        # Load RAG index
        index = load_index(is_ms)
        if index is None:
            raise HTTPException(
                status_code=503,
                detail=f"Index for {source_type} not available"
            )
        
        app_state.tool_usage["rag"] += 1
        
        # Retrieve context
        retriever = index.as_retriever(similarity_top_k=3)
        nodes = retriever.retrieve(q.question)
        
        if not nodes:
            context = "No relevant context found in documentation."
        else:
            context = "\n\n".join([f"[Doc {i+1}]\n{n.text}" for i, n in enumerate(nodes)])
        
        # Add tool context
        if additional_context:
            context += "\n\n" + "\n".join(additional_context)
        
        # Generate response
        llm = Ollama(model=model_name, request_timeout=120.0, base_url=OLLAMA_API)
        
        prompt = f"""Context from documentation and tools:
{context}

Question: {q.question}

Please provide a detailed answer based on the context above. Include code examples if relevant."""
        
        response = llm.complete(prompt)
        
        # Prepare response
        result = {
            "answer": response.text,
            "model": model_name,
            "source": source_type,
            "chunks_retrieved": len(nodes),
            "tools_used": tools_used,
            "tool_results": [tr.dict() for tr in tool_results]
        }
        
        # Cache response
        await set_cached_response(cache_key, result)
        
        response_time = int((asyncio.get_event_loop().time() - start_time) * 1000)
        
        logger.info(
            "query_completed",
            model=model_name,
            source=source_type,
            chunks=len(nodes),
            tools=tools_used,
            response_time_ms=response_time
        )
        
        return QueryResponse(**result, response_time_ms=response_time)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("query_failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

