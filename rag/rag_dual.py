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
import random
import math
import time
from typing import Optional, Dict, Any, List, Literal
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from enum import Enum

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
    ab_tests: Dict[str, ABTestConfig] = {}  # Active A/B tests
    ab_results: Dict[str, List[ABTestResult]] = {}  # Test results
    datasets: Dict[str, DatasetConfig] = {}  # Fine-tuning datasets
    finetuning_jobs: Dict[str, FineTuningConfig] = {}  # Fine-tuning jobs
    model_versions: Dict[str, ModelVersion] = {}  # Fine-tuned model registry
    ensembles: Dict[str, EnsembleConfig] = {}  # Model ensembles
    ensemble_results: Dict[str, List[EnsembleResult]] = {}  # Ensemble results
    routing_matrix: Dict[QueryType, ModelRouting] = {}  # Query type → model routing
    query_type_performance: Dict[str, QueryTypePerformance] = {}  # key: f"{query_type}_{model}"
    selection_decisions: List[SelectionDecision] = []  # History of selections
    auto_selection_config: AutoSelectionConfig = AutoSelectionConfig()
    validations: List[AnswerValidation] = []  # Validation history
    validation_config: ValidationConfig = ValidationConfig()
    factuality_checks: List[FactualityCheck] = []  # Factuality check history
    factuality_config: FactualityConfig = FactualityConfig()
    formatted_responses: List[FormattedResponse] = []  # Formatting history
    formatting_config: FormattingConfig = FormattingConfig()

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
    ab_test_id: Optional[str] = Field(default=None, description="Participate in A/B test")
    query_id: Optional[str] = Field(default=None, description="Unique query ID for tracking")
    ensemble_id: Optional[str] = Field(default=None, description="Use model ensemble")

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

class TestStatus(str, Enum):
    """A/B test status"""
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"

class ABTestConfig(BaseModel):
    """A/B test configuration"""
    test_id: str
    name: str
    description: str
    model_a: str
    model_b: str
    traffic_split: float = Field(default=0.5, ge=0.0, le=1.0)  # % to model_a
    status: TestStatus = TestStatus.DRAFT
    created_at: str
    started_at: Optional[str] = None
    ended_at: Optional[str] = None
    min_samples: int = Field(default=30, ge=10)
    confidence_level: float = Field(default=0.95, ge=0.8, le=0.99)
    metrics: List[str] = ["response_time", "tokens_per_sec", "user_rating"]

class ABTestResult(BaseModel):
    """A/B test result for a single query"""
    test_id: str
    query_id: str
    model: str
    question: str
    response_time: float
    tokens: int
    tokens_per_sec: float
    user_rating: Optional[int] = None  # 1-5 stars
    timestamp: str

class ABTestStatistics(BaseModel):
    """Statistical analysis of A/B test"""
    test_id: str
    model_a_stats: Dict[str, Any]
    model_b_stats: Dict[str, Any]
    statistical_significance: Dict[str, bool]
    confidence_intervals: Dict[str, Dict[str, float]]
    winner: Optional[str] = None
    winner_metric: Optional[str] = None
    sample_sizes: Dict[str, int]
    recommendation: str

# Fine-tuning Models
class DatasetFormat(str, Enum):
    """Dataset formats"""
    CHAT = "chat"  # ChatML format
    INSTRUCT = "instruct"  # Instruction-response pairs
    COMPLETION = "completion"  # Raw text completion
    QA = "qa"  # Question-answer pairs

class TrainingStatus(str, Enum):
    """Training job status"""
    PENDING = "pending"
    PREPARING = "preparing"
    TRAINING = "training"
    EVALUATING = "evaluating"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class DatasetConfig(BaseModel):
    """Fine-tuning dataset configuration"""
    dataset_id: str
    name: str
    description: str
    format: DatasetFormat
    file_path: str
    num_examples: int = 0
    created_at: str
    validated: bool = False
    validation_errors: List[str] = []

class FineTuningConfig(BaseModel):
    """Fine-tuning job configuration"""
    job_id: str
    name: str
    base_model: str  # e.g., "qwen2.5-coder:7b"
    dataset_id: str
    status: TrainingStatus = TrainingStatus.PENDING
    
    # Training parameters
    learning_rate: float = Field(default=2e-4, gt=0)
    num_epochs: int = Field(default=3, ge=1, le=10)
    batch_size: int = Field(default=4, ge=1, le=32)
    max_seq_length: int = Field(default=512, ge=128, le=4096)
    
    # LoRA parameters
    lora_r: int = Field(default=16, ge=8, le=128)
    lora_alpha: int = Field(default=32, ge=8, le=128)
    lora_dropout: float = Field(default=0.05, ge=0, le=0.5)
    
    # Output
    output_model_name: str
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    
    # Metrics
    training_loss: List[float] = []
    eval_loss: List[float] = []
    best_eval_loss: Optional[float] = None
    
    # Error handling
    error_message: Optional[str] = None

class ModelVersion(BaseModel):
    """Fine-tuned model version"""
    model_id: str
    name: str
    base_model: str
    version: str
    description: str
    created_at: str
    job_id: str
    
    # Performance metrics
    training_metrics: Dict[str, Any] = {}
    evaluation_metrics: Dict[str, Any] = {}
    
    # Deployment
    deployed: bool = False
    ollama_model_name: Optional[str] = None
    
    # Metadata
    dataset_id: str
    num_parameters: Optional[int] = None
    model_size_mb: Optional[float] = None

# Model Ensemble Models
class EnsembleStrategy(str, Enum):
    """Ensemble combination strategies"""
    VOTING = "voting"  # Majority vote or weighted voting
    AVERAGING = "averaging"  # Average or weighted average
    CASCADE = "cascade"  # Try models in sequence
    BEST_OF_N = "best_of_n"  # Pick best response
    SPECIALIST = "specialist"  # Route by domain/type
    CONSENSUS = "consensus"  # Require agreement threshold

class EnsembleConfig(BaseModel):
    """Model ensemble configuration"""
    ensemble_id: str
    name: str
    description: str
    strategy: EnsembleStrategy
    models: List[str]  # List of model names
    
    # Strategy-specific parameters
    weights: Optional[List[float]] = None  # For weighted strategies
    threshold: Optional[float] = None  # For consensus
    routing_rules: Optional[Dict[str, str]] = None  # For specialist
    
    # Configuration
    parallel: bool = True  # Run models in parallel
    timeout: int = 30  # Timeout per model in seconds
    min_responses: int = 1  # Minimum successful responses
    
    # Metadata
    created_at: str
    enabled: bool = True
    usage_count: int = 0
    
    # Performance
    avg_response_time: Optional[float] = None
    avg_quality_score: Optional[float] = None

class EnsembleResult(BaseModel):
    """Result from ensemble query"""
    ensemble_id: str
    query_id: str
    question: str
    strategy: EnsembleStrategy
    
    # Individual model responses
    model_responses: List[Dict[str, Any]]  # [{model, answer, time, confidence}]
    
    # Ensemble result
    final_answer: str
    confidence_score: float
    
    # Metadata
    models_used: List[str]
    total_time: float
    successful_models: int
    timestamp: str
    
    # Strategy-specific
    voting_breakdown: Optional[Dict[str, int]] = None
    agreement_score: Optional[float] = None

# Automatic Model Selection Models
class QueryType(str, Enum):
    """Types of queries for automatic routing"""
    CODE_GENERATION = "code_generation"
    CODE_EXPLANATION = "code_explanation"
    CODE_DEBUGGING = "code_debugging"
    CODE_REVIEW = "code_review"
    CODE_REFACTORING = "code_refactoring"
    CODE_TESTING = "code_testing"
    CODE_DOCUMENTATION = "code_documentation"
    ARCHITECTURE_DESIGN = "architecture_design"
    GENERAL_QUESTION = "general_question"
    TROUBLESHOOTING = "troubleshooting"
    BEST_PRACTICES = "best_practices"
    COMPARISON = "comparison"

class QueryClassification(BaseModel):
    """Classification result for a query"""
    query: str
    query_type: QueryType
    confidence: float  # 0.0 to 1.0
    keywords_matched: List[str]
    secondary_types: List[QueryType] = []  # Alternative classifications
    language: Optional[str] = None  # Programming language if detected
    complexity: str = "medium"  # low, medium, high

class ModelRouting(BaseModel):
    """Model routing configuration per query type"""
    query_type: QueryType
    primary_model: str
    fallback_models: List[str]
    min_confidence: float = 0.5
    reasoning: str = ""

class QueryTypePerformance(BaseModel):
    """Track model performance per query type"""
    query_type: QueryType
    model: str
    queries_handled: int = 0
    avg_response_time: float = 0.0
    avg_tokens_per_sec: float = 0.0
    success_rate: float = 1.0
    user_ratings: List[int] = []
    avg_rating: float = 0.0
    last_updated: str

class AutoSelectionConfig(BaseModel):
    """Configuration for automatic model selection"""
    enabled: bool = True
    use_performance_data: bool = True
    learning_rate: float = 0.1  # How quickly to adapt to feedback
    min_queries_for_routing: int = 10  # Min queries before using performance data
    confidence_threshold: float = 0.6  # Min confidence for auto-selection
    fallback_model: str = "qwen2.5-coder:7b"  # Default if no match

class SelectionDecision(BaseModel):
    """Record of an automatic selection decision"""
    query_id: str
    query: str
    classification: QueryClassification
    selected_model: str
    reasoning: str
    fallback_used: bool = False
    user_override: bool = False
    timestamp: str
    response_time: Optional[float] = None
    user_rating: Optional[int] = None

# Answer Validation Models
class ValidationCheck(str, Enum):
    """Types of validation checks"""
    FACTUALITY = "factuality"
    SOURCE_VERIFICATION = "source_verification"
    CONSISTENCY = "consistency"
    COMPLETENESS = "completeness"
    RELEVANCE = "relevance"
    CLARITY = "clarity"
    CODE_VALIDITY = "code_validity"

class ValidationResult(BaseModel):
    """Result of a validation check"""
    check_type: ValidationCheck
    passed: bool
    score: float  # 0.0 to 1.0
    issues: List[str] = []
    suggestions: List[str] = []
    details: Dict[str, Any] = {}

class AnswerValidation(BaseModel):
    """Complete validation of an answer"""
    query_id: str
    query: str
    answer: str
    model: str
    timestamp: str
    
    # Validation results
    checks: List[ValidationResult]
    overall_score: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    
    # Metadata
    source_verified: bool = False
    sources_used: List[str] = []
    context_matches: int = 0
    
    # Quality metrics
    answer_length: int
    code_blocks: int = 0
    has_examples: bool = False
    
    # Recommendations
    approved: bool = True
    warnings: List[str] = []
    corrections: List[Dict[str, str]] = []

class ValidationConfig(BaseModel):
    """Configuration for validation system"""
    enabled: bool = True
    min_confidence_threshold: float = 0.5
    enable_factuality_check: bool = True
    enable_source_verification: bool = True
    enable_consistency_check: bool = True
    enable_completeness_check: bool = True
    enable_code_validation: bool = True
    auto_reject_threshold: float = 0.3

# Factuality Checking Models
class ClaimType(str, Enum):
    """Types of claims in answers"""
    FACTUAL_STATEMENT = "factual_statement"
    DEFINITION = "definition"
    INSTRUCTION = "instruction"
    OPINION = "opinion"
    EXAMPLE = "example"
    CODE_SNIPPET = "code_snippet"

class Claim(BaseModel):
    """An extracted claim from an answer"""
    text: str
    claim_type: ClaimType
    confidence: float  # How confident extraction was
    verifiable: bool  # Can this be fact-checked
    context_required: bool  # Needs domain context to verify

class FactCheckResult(BaseModel):
    """Result of fact-checking a single claim"""
    claim: Claim
    verdict: str  # "supported", "contradicted", "unverifiable", "uncertain"
    confidence: float  # 0.0 to 1.0
    evidence: List[str] = []  # Supporting or contradicting evidence
    sources: List[str] = []  # Where evidence came from
    explanation: str = ""

class HallucinationIndicator(BaseModel):
    """Indicators of potential hallucination"""
    indicator_type: str  # "unsupported_claim", "false_confidence", "inconsistent_detail", "fabricated_source"
    severity: float  # 0.0 to 1.0
    description: str
    location: str  # Where in answer

class FactualityCheck(BaseModel):
    """Complete factuality assessment"""
    query_id: str
    query: str
    answer: str
    model: str
    timestamp: str
    
    # Claims analysis
    claims_extracted: List[Claim]
    claims_verified: int
    claims_supported: int
    claims_contradicted: int
    claims_uncertain: int
    
    # Fact check results
    fact_checks: List[FactCheckResult]
    
    # Hallucination detection
    hallucination_indicators: List[HallucinationIndicator]
    hallucination_risk: float  # 0.0 to 1.0
    
    # Overall assessment
    factuality_score: float  # 0.0 to 1.0
    confidence: float
    reliable: bool
    
    # Metadata
    context_available: bool
    sources_cited: int
    unsupported_claims: List[str] = []
    corrections: List[Dict[str, str]] = []
    
class FactualityConfig(BaseModel):
    """Configuration for factuality checking"""
    enabled: bool = True
    min_confidence_threshold: float = 0.6
    enable_claim_extraction: bool = True
    enable_hallucination_detection: bool = True
    enable_source_verification: bool = True
    strict_mode: bool = False  # Require evidence for all claims
    hallucination_threshold: float = 0.7

# Response Formatting Models
class FormattingStyle(str, Enum):
    """Response formatting styles"""
    PLAIN = "plain"
    MARKDOWN = "markdown"
    STRUCTURED = "structured"
    PROFESSIONAL = "professional"
    CONCISE = "concise"
    DETAILED = "detailed"

class FormattedResponse(BaseModel):
    """A formatted response"""
    original: str
    formatted: str
    style: FormattingStyle
    improvements: List[str] = []
    
    # Structure enhancements
    has_sections: bool = False
    has_code_blocks: int = 0
    has_lists: bool = False
    has_tables: bool = False
    
    # Readability metrics
    original_length: int
    formatted_length: int
    improvement_score: float  # 0.0 to 1.0

class FormattingConfig(BaseModel):
    """Configuration for response formatting"""
    enabled: bool = True
    default_style: FormattingStyle = FormattingStyle.MARKDOWN
    auto_add_sections: bool = True
    auto_format_code: bool = True
    auto_create_lists: bool = True
    improve_readability: bool = True
    add_emoji_headers: bool = False
    max_line_length: int = 100
    
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

# A/B Testing Functions
def select_model_for_test(test: ABTestConfig) -> str:
    """Select model based on traffic split"""
    return test.model_a if random.random() < test.traffic_split else test.model_b

async def record_ab_test_result(result: ABTestResult):
    """Record A/B test result"""
    test_id = result.test_id
    
    if test_id not in app_state.ab_results:
        app_state.ab_results[test_id] = []
    
    app_state.ab_results[test_id].append(result)
    
    # Store in Redis for persistence
    if app_state.redis_client:
        try:
            key = f"ab_test_result:{test_id}:{result.query_id}"
            await app_state.redis_client.setex(
                key,
                86400 * 30,  # 30 days
                json.dumps(result.dict())
            )
        except Exception as e:
            logger.error("failed_to_store_ab_result", error=str(e))

def calculate_statistics(results: List[ABTestResult], model: str) -> Dict[str, Any]:
    """Calculate statistics for a model's results"""
    model_results = [r for r in results if r.model == model]
    
    if not model_results:
        return {
            "count": 0,
            "response_time": {"mean": 0, "std": 0},
            "tokens_per_sec": {"mean": 0, "std": 0},
            "user_rating": {"mean": 0, "std": 0, "count": 0}
        }
    
    response_times = [r.response_time for r in model_results]
    tokens_per_sec = [r.tokens_per_sec for r in model_results]
    user_ratings = [r.user_rating for r in model_results if r.user_rating is not None]
    
    return {
        "count": len(model_results),
        "response_time": {
            "mean": sum(response_times) / len(response_times),
            "std": math.sqrt(sum((x - sum(response_times) / len(response_times)) ** 2 for x in response_times) / len(response_times)) if len(response_times) > 1 else 0
        },
        "tokens_per_sec": {
            "mean": sum(tokens_per_sec) / len(tokens_per_sec),
            "std": math.sqrt(sum((x - sum(tokens_per_sec) / len(tokens_per_sec)) ** 2 for x in tokens_per_sec) / len(tokens_per_sec)) if len(tokens_per_sec) > 1 else 0
        },
        "user_rating": {
            "mean": sum(user_ratings) / len(user_ratings) if user_ratings else 0,
            "std": math.sqrt(sum((x - sum(user_ratings) / len(user_ratings)) ** 2 for x in user_ratings) / len(user_ratings)) if len(user_ratings) > 1 else 0,
            "count": len(user_ratings)
        }
    }

def calculate_confidence_interval(data: List[float], confidence_level: float = 0.95) -> Dict[str, float]:
    """Calculate confidence interval using t-distribution"""
    if len(data) < 2:
        return {"lower": 0, "upper": 0, "margin": 0}
    
    n = len(data)
    mean = sum(data) / n
    std = math.sqrt(sum((x - mean) ** 2 for x in data) / (n - 1))
    
    # t-value approximation for 95% confidence
    t_value = 1.96 if n > 30 else 2.262  # Simplified
    margin = t_value * (std / math.sqrt(n))
    
    return {
        "lower": mean - margin,
        "upper": mean + margin,
        "margin": margin
    }

def test_statistical_significance(data_a: List[float], data_b: List[float], confidence_level: float = 0.95) -> bool:
    """Two-sample t-test for statistical significance"""
    if len(data_a) < 2 or len(data_b) < 2:
        return False
    
    n_a, n_b = len(data_a), len(data_b)
    mean_a = sum(data_a) / n_a
    mean_b = sum(data_b) / n_b
    
    var_a = sum((x - mean_a) ** 2 for x in data_a) / (n_a - 1)
    var_b = sum((x - mean_b) ** 2 for x in data_b) / (n_b - 1)
    
    # Welch's t-test
    t_stat = abs(mean_a - mean_b) / math.sqrt(var_a / n_a + var_b / n_b)
    
    # Critical value for 95% confidence (simplified)
    t_critical = 1.96
    
    return t_stat > t_critical

async def analyze_ab_test(test_id: str) -> ABTestStatistics:
    """Perform statistical analysis of A/B test"""
    if test_id not in app_state.ab_tests:
        raise HTTPException(status_code=404, detail="Test not found")
    
    test = app_state.ab_tests[test_id]
    results = app_state.ab_results.get(test_id, [])
    
    if not results:
        raise HTTPException(status_code=400, detail="No results yet")
    
    # Calculate statistics for each model
    stats_a = calculate_statistics(results, test.model_a)
    stats_b = calculate_statistics(results, test.model_b)
    
    # Extract data for significance testing
    results_a = [r for r in results if r.model == test.model_a]
    results_b = [r for r in results if r.model == test.model_b]
    
    response_times_a = [r.response_time for r in results_a]
    response_times_b = [r.response_time for r in results_b]
    
    tokens_per_sec_a = [r.tokens_per_sec for r in results_a]
    tokens_per_sec_b = [r.tokens_per_sec for r in results_b]
    
    ratings_a = [r.user_rating for r in results_a if r.user_rating is not None]
    ratings_b = [r.user_rating for r in results_b if r.user_rating is not None]
    
    # Test statistical significance
    significance = {
        "response_time": test_statistical_significance(response_times_a, response_times_b, test.confidence_level),
        "tokens_per_sec": test_statistical_significance(tokens_per_sec_a, tokens_per_sec_b, test.confidence_level),
        "user_rating": test_statistical_significance(ratings_a, ratings_b, test.confidence_level) if ratings_a and ratings_b else False
    }
    
    # Calculate confidence intervals
    confidence_intervals = {
        "response_time_a": calculate_confidence_interval(response_times_a, test.confidence_level),
        "response_time_b": calculate_confidence_interval(response_times_b, test.confidence_level),
        "tokens_per_sec_a": calculate_confidence_interval(tokens_per_sec_a, test.confidence_level),
        "tokens_per_sec_b": calculate_confidence_interval(tokens_per_sec_b, test.confidence_level)
    }
    
    # Determine winner
    winner = None
    winner_metric = None
    recommendation = "Continue test - insufficient data or no clear winner"
    
    min_samples = test.min_samples
    if stats_a["count"] >= min_samples and stats_b["count"] >= min_samples:
        # Check for winner based on multiple metrics
        a_wins = 0
        b_wins = 0
        
        # Response time (lower is better)
        if significance["response_time"]:
            if stats_a["response_time"]["mean"] < stats_b["response_time"]["mean"]:
                a_wins += 1
            else:
                b_wins += 1
        
        # Tokens per second (higher is better)
        if significance["tokens_per_sec"]:
            if stats_a["tokens_per_sec"]["mean"] > stats_b["tokens_per_sec"]["mean"]:
                a_wins += 1
            else:
                b_wins += 1
        
        # User rating (higher is better)
        if significance["user_rating"]:
            if stats_a["user_rating"]["mean"] > stats_b["user_rating"]["mean"]:
                a_wins += 1
                winner_metric = "user_rating"
            else:
                b_wins += 1
                winner_metric = "user_rating"
        
        if a_wins > b_wins:
            winner = test.model_a
            recommendation = f"✅ Clear winner: {test.model_a} performs better on {a_wins} out of {a_wins + b_wins} significant metrics"
        elif b_wins > a_wins:
            winner = test.model_b
            recommendation = f"✅ Clear winner: {test.model_b} performs better on {b_wins} out of {a_wins + b_wins} significant metrics"
        elif a_wins == b_wins and a_wins > 0:
            recommendation = f"⚖️ Tie: Both models perform similarly. Consider other factors (cost, latency)"
    else:
        needed_a = max(0, min_samples - stats_a["count"])
        needed_b = max(0, min_samples - stats_b["count"])
        recommendation = f"⏳ Need {needed_a + needed_b} more samples ({needed_a} for {test.model_a}, {needed_b} for {test.model_b})"
    
    return ABTestStatistics(
        test_id=test_id,
        model_a_stats=stats_a,
        model_b_stats=stats_b,
        statistical_significance=significance,
        confidence_intervals=confidence_intervals,
        winner=winner,
        winner_metric=winner_metric,
        sample_sizes={test.model_a: stats_a["count"], test.model_b: stats_b["count"]},
        recommendation=recommendation
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

# Model Ensemble Functions
async def execute_model_query(model: str, question: str, context: str, timeout: int = 30) -> Dict[str, Any]:
    """Execute query on a single model"""
    start_time = time.time()
    
    try:
        prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        
        async with asyncio.timeout(timeout):
            async with app_state.http_client.stream(
                "POST",
                f"{OLLAMA_API}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=timeout
            ) as response:
                if response.status_code != 200:
                    raise Exception(f"Model returned status {response.status_code}")
                
                data = await response.aread()
                result = json.loads(data)
                answer = result.get("response", "")
                
                response_time = time.time() - start_time
                
                # Simple confidence estimation based on response length and coherence
                confidence = min(1.0, len(answer.split()) / 100)  # Simple heuristic
                
                return {
                    "model": model,
                    "answer": answer,
                    "response_time": response_time,
                    "confidence": confidence,
                    "success": True,
                    "error": None
                }
    
    except Exception as e:
        return {
            "model": model,
            "answer": "",
            "response_time": time.time() - start_time,
            "confidence": 0.0,
            "success": False,
            "error": str(e)
        }

async def execute_ensemble(
    ensemble: EnsembleConfig,
    question: str,
    context: str
) -> EnsembleResult:
    """Execute ensemble query with specified strategy"""
    start_time = time.time()
    query_id = str(uuid.uuid4())
    
    model_responses = []
    
    if ensemble.strategy == EnsembleStrategy.CASCADE:
        # Try models in sequence until success
        for model in ensemble.models:
            response = await execute_model_query(model, question, context, ensemble.timeout)
            model_responses.append(response)
            
            if response["success"] and response["confidence"] >= (ensemble.threshold or 0.5):
                # Sufficient response, stop cascade
                break
    
    elif ensemble.parallel:
        # Execute all models in parallel
        tasks = [
            execute_model_query(model, question, context, ensemble.timeout)
            for model in ensemble.models
        ]
        model_responses = await asyncio.gather(*tasks, return_exceptions=False)
    
    else:
        # Execute models sequentially
        for model in ensemble.models:
            response = await execute_model_query(model, question, context, ensemble.timeout)
            model_responses.append(response)
    
    # Filter successful responses
    successful_responses = [r for r in model_responses if r["success"]]
    
    if len(successful_responses) < ensemble.min_responses:
        raise HTTPException(
            status_code=500,
            detail=f"Only {len(successful_responses)} successful responses, need {ensemble.min_responses}"
        )
    
    # Apply ensemble strategy to combine responses
    if ensemble.strategy == EnsembleStrategy.VOTING:
        final_answer, confidence, metadata = apply_voting(successful_responses, ensemble.weights)
    
    elif ensemble.strategy == EnsembleStrategy.AVERAGING:
        final_answer, confidence, metadata = apply_averaging(successful_responses, ensemble.weights)
    
    elif ensemble.strategy == EnsembleStrategy.CASCADE:
        # Use the first successful response
        final_answer = successful_responses[0]["answer"]
        confidence = successful_responses[0]["confidence"]
        metadata = {"cascade_stopped_at": successful_responses[0]["model"]}
    
    elif ensemble.strategy == EnsembleStrategy.BEST_OF_N:
        final_answer, confidence, metadata = apply_best_of_n(successful_responses)
    
    elif ensemble.strategy == EnsembleStrategy.SPECIALIST:
        final_answer, confidence, metadata = apply_specialist(successful_responses, ensemble.routing_rules, question)
    
    elif ensemble.strategy == EnsembleStrategy.CONSENSUS:
        final_answer, confidence, metadata = apply_consensus(successful_responses, ensemble.threshold or 0.7)
    
    else:
        # Default: use first successful response
        final_answer = successful_responses[0]["answer"]
        confidence = successful_responses[0]["confidence"]
        metadata = {}
    
    total_time = time.time() - start_time
    
    result = EnsembleResult(
        ensemble_id=ensemble.ensemble_id,
        query_id=query_id,
        question=question,
        strategy=ensemble.strategy,
        model_responses=model_responses,
        final_answer=final_answer,
        confidence_score=confidence,
        models_used=ensemble.models,
        total_time=total_time,
        successful_models=len(successful_responses),
        timestamp=datetime.now().isoformat(),
        voting_breakdown=metadata.get("voting_breakdown"),
        agreement_score=metadata.get("agreement_score")
    )
    
    # Store result
    if ensemble.ensemble_id not in app_state.ensemble_results:
        app_state.ensemble_results[ensemble.ensemble_id] = []
    app_state.ensemble_results[ensemble.ensemble_id].append(result)
    
    # Update ensemble stats
    ensemble.usage_count += 1
    if ensemble.avg_response_time is None:
        ensemble.avg_response_time = total_time
    else:
        ensemble.avg_response_time = (ensemble.avg_response_time * 0.9) + (total_time * 0.1)
    
    return result

def apply_voting(responses: List[Dict[str, Any]], weights: Optional[List[float]] = None) -> tuple:
    """Voting strategy: select most common answer or weighted vote"""
    if not responses:
        raise ValueError("No responses to vote on")
    
    # Simple voting: count similar answers
    answer_votes = {}
    for i, resp in enumerate(responses):
        answer = resp["answer"]
        weight = weights[i] if weights and i < len(weights) else 1.0
        
        # Group similar answers (simplified)
        answer_key = answer[:100]  # Use first 100 chars as key
        if answer_key not in answer_votes:
            answer_votes[answer_key] = {"full_answer": answer, "votes": 0, "confidence": []}
        
        answer_votes[answer_key]["votes"] += weight
        answer_votes[answer_key]["confidence"].append(resp["confidence"])
    
    # Select winner
    winner = max(answer_votes.items(), key=lambda x: x[1]["votes"])
    final_answer = winner[1]["full_answer"]
    avg_confidence = sum(winner[1]["confidence"]) / len(winner[1]["confidence"])
    
    voting_breakdown = {k[:50]: v["votes"] for k, v in answer_votes.items()}
    
    return final_answer, avg_confidence, {"voting_breakdown": voting_breakdown}

def apply_averaging(responses: List[Dict[str, Any]], weights: Optional[List[float]] = None) -> tuple:
    """Averaging strategy: combine answers (for compatible responses)"""
    if not responses:
        raise ValueError("No responses to average")
    
    # For text, we use weighted selection based on confidence
    if weights:
        total_weight = sum(weights[:len(responses)])
        weighted_confidences = [
            resp["confidence"] * (weights[i] / total_weight)
            for i, resp in enumerate(responses)
        ]
    else:
        weighted_confidences = [resp["confidence"] for resp in responses]
    
    # Select response with highest weighted confidence
    best_idx = weighted_confidences.index(max(weighted_confidences))
    final_answer = responses[best_idx]["answer"]
    avg_confidence = sum(r["confidence"] for r in responses) / len(responses)
    
    return final_answer, avg_confidence, {"selected_model": responses[best_idx]["model"]}

def apply_best_of_n(responses: List[Dict[str, Any]]) -> tuple:
    """Best-of-N: select response with highest confidence"""
    if not responses:
        raise ValueError("No responses to select from")
    
    best = max(responses, key=lambda x: x["confidence"])
    
    return best["answer"], best["confidence"], {
        "selected_model": best["model"],
        "confidence_scores": {r["model"]: r["confidence"] for r in responses}
    }

def apply_specialist(responses: List[Dict[str, Any]], routing_rules: Optional[Dict[str, str]], question: str) -> tuple:
    """Specialist: route based on question type"""
    if not responses:
        raise ValueError("No responses available")
    
    # Detect question type
    question_lower = question.lower()
    question_type = "general"
    
    if any(word in question_lower for word in ["code", "function", "debug", "implement"]):
        question_type = "code"
    elif any(word in question_lower for word in ["explain", "what is", "how does"]):
        question_type = "explanation"
    elif any(word in question_lower for word in ["debug", "error", "fix", "problem"]):
        question_type = "debugging"
    
    # Use routing rules if provided
    if routing_rules and question_type in routing_rules:
        preferred_model = routing_rules[question_type]
        for resp in responses:
            if resp["model"] == preferred_model:
                return resp["answer"], resp["confidence"], {
                    "question_type": question_type,
                    "selected_model": preferred_model
                }
    
    # Fall back to best confidence
    best = max(responses, key=lambda x: x["confidence"])
    return best["answer"], best["confidence"], {
        "question_type": question_type,
        "selected_model": best["model"]
    }

def apply_consensus(responses: List[Dict[str, Any]], threshold: float = 0.7) -> tuple:
    """Consensus: require agreement between models"""
    if not responses:
        raise ValueError("No responses available")
    
    # Calculate similarity between responses (simplified)
    answer_groups = {}
    for resp in responses:
        answer_key = resp["answer"][:100]
        if answer_key not in answer_groups:
            answer_groups[answer_key] = []
        answer_groups[answer_key].append(resp)
    
    # Check if any group meets threshold
    for answer_key, group in answer_groups.items():
        agreement = len(group) / len(responses)
        if agreement >= threshold:
            # Use answer from highest confidence in group
            best = max(group, key=lambda x: x["confidence"])
            return best["answer"], best["confidence"], {
                "agreement_score": agreement,
                "models_agreed": [r["model"] for r in group]
            }
    
    # No consensus, use highest confidence
    best = max(responses, key=lambda x: x["confidence"])
    return best["answer"], best["confidence"], {
        "agreement_score": 1/len(responses),
        "models_agreed": [best["model"]],
        "consensus_failed": True
    }

# Automatic Model Selection Functions
def classify_query(query: str) -> QueryClassification:
    """Classify query into type with confidence score"""
    query_lower = query.lower()
    keywords_matched = []
    scores = {query_type: 0.0 for query_type in QueryType}
    
    # Define keywords for each query type
    type_keywords = {
        QueryType.CODE_GENERATION: {
            "write": 2.0, "create": 2.0, "implement": 2.0, "generate": 2.0,
            "build": 1.5, "make": 1.5, "function": 1.5, "class": 1.5,
            "method": 1.5, "algorithm": 2.0, "code": 1.0
        },
        QueryType.CODE_EXPLANATION: {
            "explain": 3.0, "what is": 2.5, "what does": 2.5, "how does": 2.5,
            "describe": 2.0, "understand": 2.0, "meaning": 2.0, "purpose": 2.0,
            "works": 1.5, "means": 1.5
        },
        QueryType.CODE_DEBUGGING: {
            "debug": 3.0, "fix": 3.0, "error": 2.5, "bug": 2.5, "issue": 2.0,
            "problem": 2.0, "not working": 2.5, "fails": 2.0, "crash": 2.5,
            "exception": 2.0, "traceback": 2.0, "stack trace": 2.5
        },
        QueryType.CODE_REVIEW: {
            "review": 3.0, "check": 2.0, "analyze": 2.0, "assess": 2.0,
            "evaluate": 2.0, "quality": 2.0, "improve": 1.5, "better": 1.5,
            "secure": 2.0, "vulnerable": 2.5
        },
        QueryType.CODE_REFACTORING: {
            "refactor": 3.0, "optimize": 2.5, "improve": 2.0, "clean": 2.0,
            "rewrite": 2.5, "restructure": 2.5, "simplify": 2.0,
            "performance": 2.0, "efficient": 2.0
        },
        QueryType.CODE_TESTING: {
            "test": 3.0, "unit test": 3.0, "integration test": 3.0,
            "testing": 2.5, "mock": 2.0, "assert": 2.0, "coverage": 2.0,
            "pytest": 2.0, "jest": 2.0, "junit": 2.0
        },
        QueryType.CODE_DOCUMENTATION: {
            "document": 3.0, "docstring": 3.0, "comment": 2.5,
            "documentation": 3.0, "readme": 2.5, "api doc": 2.5,
            "javadoc": 2.5, "jsdoc": 2.5
        },
        QueryType.ARCHITECTURE_DESIGN: {
            "architecture": 3.0, "design": 2.5, "pattern": 2.5,
            "structure": 2.0, "system": 1.5, "microservice": 2.5,
            "api": 1.5, "database": 1.5, "scalable": 2.0
        },
        QueryType.TROUBLESHOOTING: {
            "troubleshoot": 3.0, "diagnose": 2.5, "investigate": 2.0,
            "why": 2.0, "not working": 2.5, "doesn't work": 2.5,
            "won't": 2.0, "can't": 2.0
        },
        QueryType.BEST_PRACTICES: {
            "best practice": 3.0, "recommended": 2.5, "should": 2.0,
            "convention": 2.5, "standard": 2.0, "guideline": 2.5,
            "proper way": 2.5, "right way": 2.5
        },
        QueryType.COMPARISON: {
            "compare": 3.0, "difference": 3.0, "vs": 2.5, "versus": 2.5,
            "better": 2.0, "which": 2.0, "between": 1.5, "or": 1.0
        }
    }
    
    # Score each query type based on keyword matches
    for query_type, keywords in type_keywords.items():
        for keyword, weight in keywords.items():
            if keyword in query_lower:
                scores[query_type] += weight
                if keyword not in keywords_matched:
                    keywords_matched.append(keyword)
    
    # Default to general question if no strong matches
    if max(scores.values()) == 0:
        return QueryClassification(
            query=query,
            query_type=QueryType.GENERAL_QUESTION,
            confidence=0.5,
            keywords_matched=[],
            complexity="medium"
        )
    
    # Get top scoring type
    sorted_types = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    primary_type = sorted_types[0][0]
    primary_score = sorted_types[0][1]
    
    # Calculate confidence (normalize to 0-1)
    max_possible_score = 10.0  # Rough estimate
    confidence = min(1.0, primary_score / max_possible_score)
    
    # Get secondary types (score > 0)
    secondary_types = [t for t, s in sorted_types[1:3] if s > 0]
    
    # Detect programming language
    language = detect_language(query)
    
    # Estimate complexity
    complexity = estimate_complexity(query)
    
    return QueryClassification(
        query=query,
        query_type=primary_type,
        confidence=confidence,
        keywords_matched=keywords_matched,
        secondary_types=secondary_types,
        language=language,
        complexity=complexity
    )

def detect_language(query: str) -> Optional[str]:
    """Detect programming language from query"""
    query_lower = query.lower()
    
    languages = {
        "python": ["python", "py", "django", "flask", "pandas", "numpy"],
        "javascript": ["javascript", "js", "node", "react", "vue", "angular", "typescript", "ts"],
        "java": ["java", "spring", "maven", "gradle"],
        "csharp": ["c#", "csharp", ".net", "dotnet", "xaml", "blazor"],
        "go": ["golang", "go "],
        "rust": ["rust", "cargo"],
        "cpp": ["c++", "cpp"],
        "c": [" c ", "c programming"],
        "ruby": ["ruby", "rails"],
        "php": ["php", "laravel"],
        "sql": ["sql", "mysql", "postgres", "oracle"],
        "powershell": ["powershell", "ps1"],
        "bash": ["bash", "shell", "sh "]
    }
    
    for lang, keywords in languages.items():
        if any(kw in query_lower for kw in keywords):
            return lang
    
    return None

def estimate_complexity(query: str) -> str:
    """Estimate query complexity"""
    word_count = len(query.split())
    
    # Simple heuristics
    if word_count < 10:
        return "low"
    elif word_count > 30:
        return "high"
    
    # Check for complexity indicators
    complex_keywords = [
        "distributed", "concurrent", "parallel", "scalable", "enterprise",
        "microservice", "architecture", "design pattern", "algorithm",
        "optimization", "performance"
    ]
    
    if any(kw in query.lower() for kw in complex_keywords):
        return "high"
    
    return "medium"

def initialize_default_routing():
    """Initialize default routing matrix"""
    if app_state.routing_matrix:
        return  # Already initialized
    
    default_routes = {
        QueryType.CODE_GENERATION: ModelRouting(
            query_type=QueryType.CODE_GENERATION,
            primary_model="qwen2.5-coder:7b",
            fallback_models=["deepseek-coder-v2:16b", "codellama:13b"],
            min_confidence=0.6,
            reasoning="Fast code generation model"
        ),
        QueryType.CODE_EXPLANATION: ModelRouting(
            query_type=QueryType.CODE_EXPLANATION,
            primary_model="llama3.1:8b",
            fallback_models=["qwen2.5-coder:7b", "deepseek-coder-v2:16b"],
            min_confidence=0.5,
            reasoning="General purpose explanation model"
        ),
        QueryType.CODE_DEBUGGING: ModelRouting(
            query_type=QueryType.CODE_DEBUGGING,
            primary_model="deepseek-coder-v2:16b",
            fallback_models=["qwen2.5-coder:14b", "codellama:13b"],
            min_confidence=0.6,
            reasoning="Powerful debugging model"
        ),
        QueryType.CODE_REVIEW: ModelRouting(
            query_type=QueryType.CODE_REVIEW,
            primary_model="deepseek-coder-v2:16b",
            fallback_models=["qwen2.5-coder:14b", "llama3.1:8b"],
            min_confidence=0.6,
            reasoning="Thorough code analysis"
        ),
        QueryType.CODE_REFACTORING: ModelRouting(
            query_type=QueryType.CODE_REFACTORING,
            primary_model="qwen2.5-coder:14b",
            fallback_models=["deepseek-coder-v2:16b", "qwen2.5-coder:7b"],
            min_confidence=0.6,
            reasoning="Advanced refactoring capabilities"
        ),
        QueryType.CODE_TESTING: ModelRouting(
            query_type=QueryType.CODE_TESTING,
            primary_model="qwen2.5-coder:7b",
            fallback_models=["codellama:13b", "deepseek-coder-v2:16b"],
            min_confidence=0.5,
            reasoning="Test generation specialist"
        ),
        QueryType.CODE_DOCUMENTATION: ModelRouting(
            query_type=QueryType.CODE_DOCUMENTATION,
            primary_model="llama3.1:8b",
            fallback_models=["qwen2.5-coder:7b", "codellama:13b"],
            min_confidence=0.5,
            reasoning="Clear documentation writing"
        ),
        QueryType.ARCHITECTURE_DESIGN: ModelRouting(
            query_type=QueryType.ARCHITECTURE_DESIGN,
            primary_model="deepseek-coder-v2:16b",
            fallback_models=["qwen2.5-coder:14b", "llama3.1:8b"],
            min_confidence=0.6,
            reasoning="High-level design thinking"
        ),
        QueryType.GENERAL_QUESTION: ModelRouting(
            query_type=QueryType.GENERAL_QUESTION,
            primary_model="llama3.1:8b",
            fallback_models=["qwen2.5-coder:7b", "deepseek-coder-v2:16b"],
            min_confidence=0.4,
            reasoning="General purpose model"
        ),
        QueryType.TROUBLESHOOTING: ModelRouting(
            query_type=QueryType.TROUBLESHOOTING,
            primary_model="deepseek-coder-v2:16b",
            fallback_models=["qwen2.5-coder:14b", "llama3.1:8b"],
            min_confidence=0.5,
            reasoning="Problem diagnosis expert"
        ),
        QueryType.BEST_PRACTICES: ModelRouting(
            query_type=QueryType.BEST_PRACTICES,
            primary_model="llama3.1:8b",
            fallback_models=["deepseek-coder-v2:16b", "qwen2.5-coder:14b"],
            min_confidence=0.5,
            reasoning="Knowledgeable about conventions"
        ),
        QueryType.COMPARISON: ModelRouting(
            query_type=QueryType.COMPARISON,
            primary_model="llama3.1:8b",
            fallback_models=["deepseek-coder-v2:16b", "qwen2.5-coder:14b"],
            min_confidence=0.5,
            reasoning="Balanced comparison analysis"
        )
    }
    
    app_state.routing_matrix = default_routes
    logger.info("initialized_default_routing", routes=len(default_routes))

async def select_model_automatically(query: str, available_models: Optional[List[str]] = None) -> tuple[str, QueryClassification, str]:
    """Automatically select best model for query"""
    # Initialize routing if needed
    initialize_default_routing()
    
    # Classify query
    classification = classify_query(query)
    
    # Check if auto-selection is enabled
    if not app_state.auto_selection_config.enabled:
        fallback = app_state.auto_selection_config.fallback_model
        return fallback, classification, "Auto-selection disabled, using fallback"
    
    # Check confidence threshold
    if classification.confidence < app_state.auto_selection_config.confidence_threshold:
        fallback = app_state.auto_selection_config.fallback_model
        return fallback, classification, f"Low confidence ({classification.confidence:.2f}), using fallback"
    
    # Get routing for query type
    routing = app_state.routing_matrix.get(classification.query_type)
    if not routing:
        fallback = app_state.auto_selection_config.fallback_model
        return fallback, classification, "No routing found for query type"
    
    # Use performance data if enabled and available
    if app_state.auto_selection_config.use_performance_data:
        perf_key = f"{classification.query_type}_{routing.primary_model}"
        perf_data = app_state.query_type_performance.get(perf_key)
        
        if perf_data and perf_data.queries_handled >= app_state.auto_selection_config.min_queries_for_routing:
            # Check if performance is good
            if perf_data.success_rate < 0.7 or perf_data.avg_rating < 3.0:
                # Try fallback models
                for fallback_model in routing.fallback_models:
                    fallback_key = f"{classification.query_type}_{fallback_model}"
                    fallback_perf = app_state.query_type_performance.get(fallback_key)
                    
                    if fallback_perf and fallback_perf.avg_rating > perf_data.avg_rating:
                        return fallback_model, classification, f"Performance-based: {fallback_model} rated higher"
    
    # Check model availability
    if available_models and routing.primary_model not in available_models:
        # Try fallbacks
        for fallback_model in routing.fallback_models:
            if fallback_model in available_models:
                return fallback_model, classification, f"Primary unavailable, using fallback: {fallback_model}"
        
        # Use any available model
        if available_models:
            return available_models[0], classification, "Using first available model"
    
    # Return primary model
    return routing.primary_model, classification, f"Matched {classification.query_type.value}: {routing.reasoning}"

async def record_selection_decision(
    query_id: str,
    query: str,
    classification: QueryClassification,
    selected_model: str,
    reasoning: str,
    fallback_used: bool = False,
    user_override: bool = False
) -> SelectionDecision:
    """Record an automatic selection decision"""
    decision = SelectionDecision(
        query_id=query_id,
        query=query,
        classification=classification,
        selected_model=selected_model,
        reasoning=reasoning,
        fallback_used=fallback_used,
        user_override=user_override,
        timestamp=datetime.now().isoformat()
    )
    
    app_state.selection_decisions.append(decision)
    
    # Keep only last 1000 decisions
    if len(app_state.selection_decisions) > 1000:
        app_state.selection_decisions = app_state.selection_decisions[-1000:]
    
    return decision

async def update_query_type_performance(
    query_type: QueryType,
    model: str,
    response_time: float,
    tokens_per_sec: float,
    success: bool,
    user_rating: Optional[int] = None
):
    """Update performance metrics for query type + model combination"""
    key = f"{query_type}_{model}"
    
    if key not in app_state.query_type_performance:
        app_state.query_type_performance[key] = QueryTypePerformance(
            query_type=query_type,
            model=model,
            last_updated=datetime.now().isoformat()
        )
    
    perf = app_state.query_type_performance[key]
    
    # Update metrics with exponential moving average
    learning_rate = app_state.auto_selection_config.learning_rate
    
    if perf.queries_handled == 0:
        perf.avg_response_time = response_time
        perf.avg_tokens_per_sec = tokens_per_sec
    else:
        perf.avg_response_time = (1 - learning_rate) * perf.avg_response_time + learning_rate * response_time
        perf.avg_tokens_per_sec = (1 - learning_rate) * perf.avg_tokens_per_sec + learning_rate * tokens_per_sec
    
    perf.queries_handled += 1
    
    # Update success rate
    if perf.queries_handled == 1:
        perf.success_rate = 1.0 if success else 0.0
    else:
        perf.success_rate = (1 - learning_rate) * perf.success_rate + learning_rate * (1.0 if success else 0.0)
    
    # Update ratings
    if user_rating:
        perf.user_ratings.append(user_rating)
        # Keep only last 100 ratings
        if len(perf.user_ratings) > 100:
            perf.user_ratings = perf.user_ratings[-100:]
        perf.avg_rating = sum(perf.user_ratings) / len(perf.user_ratings)
    
    perf.last_updated = datetime.now().isoformat()
    
    logger.info("updated_query_type_performance", key=key, queries=perf.queries_handled, rating=perf.avg_rating)

# Automatic Model Selection API Endpoints
@app.post("/auto-selection/classify")
async def classify_query_endpoint(query: str):
    """Classify a query into type"""
    classification = classify_query(query)
    return classification

@app.post("/auto-selection/select")
async def select_model_endpoint(query: str):
    """Automatically select best model for query"""
    model, classification, reasoning = await select_model_automatically(query)
    return {
        "selected_model": model,
        "classification": classification,
        "reasoning": reasoning
    }

@app.get("/auto-selection/config")
async def get_auto_selection_config():
    """Get automatic selection configuration"""
    return app_state.auto_selection_config

@app.put("/auto-selection/config")
async def update_auto_selection_config(config: AutoSelectionConfig):
    """Update automatic selection configuration"""
    app_state.auto_selection_config = config
    logger.info("updated_auto_selection_config", enabled=config.enabled)
    return config

@app.get("/auto-selection/routing")
async def get_routing_matrix():
    """Get model routing matrix"""
    initialize_default_routing()
    return {"routing": list(app_state.routing_matrix.values())}

@app.put("/auto-selection/routing/{query_type}")
async def update_routing(query_type: QueryType, routing: ModelRouting):
    """Update routing for a query type"""
    app_state.routing_matrix[query_type] = routing
    logger.info("updated_routing", query_type=query_type, model=routing.primary_model)
    return routing

@app.get("/auto-selection/performance")
async def get_performance_stats(query_type: Optional[QueryType] = None):
    """Get performance statistics"""
    if query_type:
        # Filter by query type
        stats = {k: v for k, v in app_state.query_type_performance.items() if k.startswith(query_type.value)}
    else:
        stats = app_state.query_type_performance
    
    return {"performance": list(stats.values())}

@app.get("/auto-selection/decisions")
async def get_selection_decisions(limit: int = 50):
    """Get recent selection decisions"""
    return {"decisions": app_state.selection_decisions[-limit:]}

@app.post("/auto-selection/feedback")
async def submit_selection_feedback(query_id: str, rating: int, correct_model: Optional[str] = None):
    """Submit feedback on automatic selection"""
    # Find the decision
    decision = next((d for d in app_state.selection_decisions if d.query_id == query_id), None)
    
    if not decision:
        raise HTTPException(status_code=404, detail="Decision not found")
    
    decision.user_rating = rating
    
    # If user suggests different model, learn from it
    if correct_model and correct_model != decision.selected_model:
        # Decrease confidence in selected model
        await update_query_type_performance(
            decision.classification.query_type,
            decision.selected_model,
            decision.response_time or 0.0,
            0.0,
            False,
            rating
        )
        
        # Increase confidence in correct model
        await update_query_type_performance(
            decision.classification.query_type,
            correct_model,
            0.0,
            0.0,
            True,
            5  # Boost rating
        )
        
        logger.info("learned_from_feedback", wrong=decision.selected_model, correct=correct_model)
    
    return {"status": "success", "message": "Feedback recorded"}

# Answer Validation Functions
async def validate_answer(
    query_id: str,
    query: str,
    answer: str,
    model: str,
    context: Optional[str] = None
) -> AnswerValidation:
    """Comprehensive answer validation"""
    
    if not app_state.validation_config.enabled:
        # Return minimal validation if disabled
        return AnswerValidation(
            query_id=query_id,
            query=query,
            answer=answer,
            model=model,
            timestamp=datetime.now().isoformat(),
            checks=[],
            overall_score=1.0,
            confidence=1.0,
            answer_length=len(answer)
        )
    
    checks = []
    
    # 1. Source Verification (if context available)
    if app_state.validation_config.enable_source_verification and context:
        source_result = check_source_verification(answer, context)
        checks.append(source_result)
    
    # 2. Consistency Check
    if app_state.validation_config.enable_consistency_check:
        consistency_result = check_consistency(answer)
        checks.append(consistency_result)
    
    # 3. Completeness Check
    if app_state.validation_config.enable_completeness_check:
        completeness_result = check_completeness(query, answer)
        checks.append(completeness_result)
    
    # 4. Relevance Check
    relevance_result = check_relevance(query, answer)
    checks.append(relevance_result)
    
    # 5. Clarity Check
    clarity_result = check_clarity(answer)
    checks.append(clarity_result)
    
    # 6. Code Validity (if code present)
    if app_state.validation_config.enable_code_validation:
        code_result = check_code_validity(answer)
        if code_result:
            checks.append(code_result)
    
    # Calculate overall score
    if checks:
        overall_score = sum(check.score for check in checks) / len(checks)
    else:
        overall_score = 0.8  # Default if no checks
    
    # Calculate confidence
    passed_checks = sum(1 for check in checks if check.passed)
    confidence = passed_checks / len(checks) if checks else 0.8
    
    # Extract metadata
    code_blocks = answer.count("```")
    has_examples = any(word in answer.lower() for word in ["example", "for instance", "such as"])
    
    # Collect warnings and determine approval
    warnings = []
    approved = True
    
    for check in checks:
        if not check.passed:
            warnings.extend(check.issues)
    
    if overall_score < app_state.validation_config.auto_reject_threshold:
        approved = False
        warnings.append(f"Overall score ({overall_score:.2f}) below threshold")
    
    validation = AnswerValidation(
        query_id=query_id,
        query=query,
        answer=answer,
        model=model,
        timestamp=datetime.now().isoformat(),
        checks=checks,
        overall_score=overall_score,
        confidence=confidence,
        source_verified=context is not None,
        sources_used=[],
        answer_length=len(answer),
        code_blocks=code_blocks,
        has_examples=has_examples,
        approved=approved,
        warnings=warnings
    )
    
    # Store validation
    app_state.validations.append(validation)
    
    # Keep only last 500 validations
    if len(app_state.validations) > 500:
        app_state.validations = app_state.validations[-500:]
    
    logger.info("validated_answer", query_id=query_id, score=overall_score, approved=approved)
    
    return validation

def check_source_verification(answer: str, context: str) -> ValidationResult:
    """Verify answer against provided context"""
    answer_lower = answer.lower()
    context_lower = context.lower()
    
    # Simple overlap check - count matching phrases
    answer_words = set(answer_lower.split())
    context_words = set(context_lower.split())
    
    overlap = len(answer_words & context_words)
    total = len(answer_words)
    
    overlap_ratio = overlap / total if total > 0 else 0
    
    passed = overlap_ratio > 0.2  # At least 20% overlap
    score = min(1.0, overlap_ratio * 2)  # Scale to 0-1
    
    issues = []
    suggestions = []
    
    if not passed:
        issues.append("Answer may not be grounded in provided context")
        suggestions.append("Ensure answer references the context sources")
    
    return ValidationResult(
        check_type=ValidationCheck.SOURCE_VERIFICATION,
        passed=passed,
        score=score,
        issues=issues,
        suggestions=suggestions,
        details={"overlap_ratio": overlap_ratio, "overlapping_words": overlap}
    )

def check_consistency(answer: str) -> ValidationResult:
    """Check for internal contradictions"""
    issues = []
    
    # Look for contradiction indicators
    contradictions = [
        ("yes", "no"), ("true", "false"), ("always", "never"),
        ("correct", "incorrect"), ("should", "shouldn't"),
        ("can", "cannot"), ("will", "won't")
    ]
    
    answer_lower = answer.lower()
    found_contradictions = []
    
    for word1, word2 in contradictions:
        if word1 in answer_lower and word2 in answer_lower:
            # Check if they're in the same sentence (simplified)
            found_contradictions.append((word1, word2))
    
    passed = len(found_contradictions) == 0
    score = 1.0 if passed else max(0.3, 1.0 - (len(found_contradictions) * 0.2))
    
    if found_contradictions:
        for w1, w2 in found_contradictions:
            issues.append(f"Potential contradiction: '{w1}' and '{w2}' both present")
    
    return ValidationResult(
        check_type=ValidationCheck.CONSISTENCY,
        passed=passed,
        score=score,
        issues=issues,
        suggestions=["Review answer for contradictory statements"] if issues else [],
        details={"contradictions_found": len(found_contradictions)}
    )

def check_completeness(query: str, answer: str) -> ValidationResult:
    """Check if answer fully addresses the question"""
    query_lower = query.lower()
    answer_lower = answer.lower()
    
    # Extract key question words
    question_words = ["what", "how", "why", "when", "where", "who", "which"]
    has_question_word = any(word in query_lower for word in question_words)
    
    # Check answer length relative to query complexity
    query_words = len(query.split())
    answer_words = len(answer.split())
    
    # Simple heuristic: answer should be at least 2x query length for complex questions
    min_expected = query_words * 2 if query_words > 10 else 20
    
    length_adequate = answer_words >= min_expected
    
    # Check if answer addresses the question type
    addresses_question = True
    issues = []
    suggestions = []
    
    if has_question_word:
        # For "how" questions, expect steps or explanations
        if "how" in query_lower:
            has_steps = any(word in answer_lower for word in ["first", "then", "next", "finally", "step"])
            if not has_steps and answer_words < 50:
                addresses_question = False
                issues.append("'How' question may need more detailed explanation")
                suggestions.append("Add step-by-step instructions or examples")
        
        # For "why" questions, expect reasoning
        if "why" in query_lower:
            has_reasoning = any(word in answer_lower for word in ["because", "since", "due to", "reason"])
            if not has_reasoning:
                issues.append("'Why' question should include reasoning")
                suggestions.append("Explain the reasoning behind the answer")
    
    if not length_adequate:
        issues.append(f"Answer may be too brief ({answer_words} words, expected ~{min_expected})")
        suggestions.append("Provide more detailed explanation or examples")
    
    passed = length_adequate and addresses_question
    score = 0.5 if not length_adequate else (1.0 if addresses_question else 0.7)
    
    return ValidationResult(
        check_type=ValidationCheck.COMPLETENESS,
        passed=passed,
        score=score,
        issues=issues,
        suggestions=suggestions,
        details={"query_words": query_words, "answer_words": answer_words}
    )

def check_relevance(query: str, answer: str) -> ValidationResult:
    """Check if answer is relevant to the question"""
    query_words = set(query.lower().split())
    answer_words = set(answer.lower().split())
    
    # Remove common stop words
    stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for"}
    query_words -= stop_words
    answer_words -= stop_words
    
    # Calculate relevance
    common_words = query_words & answer_words
    relevance_ratio = len(common_words) / len(query_words) if query_words else 0
    
    passed = relevance_ratio > 0.2  # At least 20% of query words in answer
    score = min(1.0, relevance_ratio * 2)
    
    issues = []
    suggestions = []
    
    if not passed:
        issues.append("Answer may not directly address the question")
        suggestions.append("Ensure answer stays focused on the specific question asked")
    
    return ValidationResult(
        check_type=ValidationCheck.RELEVANCE,
        passed=passed,
        score=score,
        issues=issues,
        suggestions=suggestions,
        details={"relevance_ratio": relevance_ratio, "common_words": len(common_words)}
    )

def check_clarity(answer: str) -> ValidationResult:
    """Check answer clarity and readability"""
    words = answer.split()
    sentences = answer.count('.') + answer.count('!') + answer.count('?')
    
    if sentences == 0:
        sentences = 1  # Avoid division by zero
    
    avg_sentence_length = len(words) / sentences
    
    # Check for clarity indicators
    has_structure = any(word in answer.lower() for word in ["first", "second", "finally", "however", "therefore"])
    has_formatting = "```" in answer or "\n\n" in answer
    
    issues = []
    suggestions = []
    
    # Too long sentences reduce clarity
    if avg_sentence_length > 30:
        issues.append(f"Average sentence length is high ({avg_sentence_length:.1f} words)")
        suggestions.append("Break long sentences into shorter ones for clarity")
    
    score = 1.0
    
    # Adjust score based on factors
    if avg_sentence_length > 30:
        score -= 0.2
    if not has_structure and len(words) > 100:
        score -= 0.1
        issues.append("Long answer could benefit from structure (e.g., numbered steps)")
    
    score = max(0.5, score)  # Minimum 0.5
    passed = score >= 0.7
    
    return ValidationResult(
        check_type=ValidationCheck.CLARITY,
        passed=passed,
        score=score,
        issues=issues,
        suggestions=suggestions,
        details={"avg_sentence_length": avg_sentence_length, "has_structure": has_structure}
    )

def check_code_validity(answer: str) -> Optional[ValidationResult]:
    """Check if code blocks are properly formatted"""
    if "```" not in answer:
        return None  # No code blocks
    
    code_blocks = answer.count("```") // 2  # Pairs of ```
    
    issues = []
    suggestions = []
    
    # Check for unclosed code blocks
    if answer.count("```") % 2 != 0:
        issues.append("Unclosed code block detected")
        suggestions.append("Ensure all code blocks are properly closed with ```")
    
    # Check for syntax in code blocks
    # Extract code between ```
    code_parts = answer.split("```")
    for i in range(1, len(code_parts), 2):
        code = code_parts[i].strip()
        if not code:
            issues.append("Empty code block found")
    
    passed = len(issues) == 0
    score = 1.0 if passed else 0.6
    
    return ValidationResult(
        check_type=ValidationCheck.CODE_VALIDITY,
        passed=passed,
        score=score,
        issues=issues,
        suggestions=suggestions,
        details={"code_blocks": code_blocks}
    )

# Answer Validation API Endpoints
@app.post("/validation/validate")
async def validate_answer_endpoint(
    query_id: str,
    query: str,
    answer: str,
    model: str,
    context: Optional[str] = None
):
    """Validate an answer"""
    validation = await validate_answer(query_id, query, answer, model, context)
    return validation

@app.get("/validation/config")
async def get_validation_config():
    """Get validation configuration"""
    return app_state.validation_config

@app.put("/validation/config")
async def update_validation_config(config: ValidationConfig):
    """Update validation configuration"""
    app_state.validation_config = config
    logger.info("updated_validation_config", enabled=config.enabled)
    return config

@app.get("/validation/history")
async def get_validation_history(limit: int = 50):
    """Get recent validations"""
    return {"validations": app_state.validations[-limit:]}

@app.get("/validation/stats")
async def get_validation_stats():
    """Get validation statistics"""
    if not app_state.validations:
        return {
            "total_validations": 0,
            "avg_score": 0,
            "approval_rate": 0,
            "check_pass_rates": {}
        }
    
    total = len(app_state.validations)
    avg_score = sum(v.overall_score for v in app_state.validations) / total
    approved = sum(1 for v in app_state.validations if v.approved)
    approval_rate = approved / total
    
    # Calculate pass rates per check type
    check_counts = {}
    check_passes = {}
    
    for validation in app_state.validations:
        for check in validation.checks:
            check_type = check.check_type.value
            check_counts[check_type] = check_counts.get(check_type, 0) + 1
            if check.passed:
                check_passes[check_type] = check_passes.get(check_type, 0) + 1
    
    check_pass_rates = {
        check_type: (check_passes.get(check_type, 0) / count)
        for check_type, count in check_counts.items()
    }
    
    return {
        "total_validations": total,
        "avg_score": round(avg_score, 3),
        "approval_rate": round(approval_rate, 3),
        "check_pass_rates": {k: round(v, 3) for k, v in check_pass_rates.items()}
    }

# Factuality Checking Functions
def extract_claims(answer: str) -> List[Claim]:
    """Extract verifiable claims from answer"""
    claims = []
    
    # Split into sentences
    sentences = []
    for delimiter in ['. ', '! ', '? ', '.\n', '\n\n']:
        if delimiter in answer:
            parts = answer.split(delimiter)
            sentences.extend([s.strip() for s in parts if s.strip()])
    
    if not sentences:
        sentences = [answer]
    
    for sentence in sentences:
        if len(sentence) < 10:
            continue  # Skip very short sentences
        
        sentence_lower = sentence.lower()
        
        # Classify claim type
        claim_type = ClaimType.FACTUAL_STATEMENT
        verifiable = True
        context_required = True
        
        # Check for definitions
        if any(word in sentence_lower for word in ["is a", "is an", "refers to", "means", "defined as"]):
            claim_type = ClaimType.DEFINITION
            verifiable = True
            context_required = False
        
        # Check for instructions
        elif any(word in sentence_lower[:20] for word in ["use", "run", "execute", "call", "create", "write"]):
            claim_type = ClaimType.INSTRUCTION
            verifiable = False
        
        # Check for examples
        elif any(word in sentence_lower for word in ["for example", "such as", "like", "e.g."]):
            claim_type = ClaimType.EXAMPLE
            verifiable = True
        
        # Check for code
        elif "```" in sentence or any(word in sentence_lower for word in ["function", "class", "method", "variable"]):
            claim_type = ClaimType.CODE_SNIPPET
            verifiable = True
        
        # Check for opinions
        elif any(word in sentence_lower for word in ["should", "better", "prefer", "recommend", "suggest", "think", "believe"]):
            claim_type = ClaimType.OPINION
            verifiable = False
        
        # Estimate extraction confidence
        confidence = 0.8
        if len(sentence.split()) < 5:
            confidence = 0.5
        elif len(sentence.split()) > 30:
            confidence = 0.6
        
        claims.append(Claim(
            text=sentence,
            claim_type=claim_type,
            confidence=confidence,
            verifiable=verifiable,
            context_required=context_required
        ))
    
    return claims

def detect_hallucinations(answer: str, context: Optional[str] = None) -> tuple[List[HallucinationIndicator], float]:
    """Detect potential hallucinations in answer"""
    indicators = []
    
    answer_lower = answer.lower()
    
    # 1. Check for false confidence markers
    false_confidence_phrases = [
        "definitely", "certainly", "always", "never", "impossible",
        "guaranteed", "absolutely", "100%", "without a doubt"
    ]
    
    for phrase in false_confidence_phrases:
        if phrase in answer_lower:
            indicators.append(HallucinationIndicator(
                indicator_type="false_confidence",
                severity=0.5,
                description=f"Uses absolute language: '{phrase}'",
                location=f"Contains '{phrase}'"
            ))
    
    # 2. Check for fabricated sources
    source_indicators = ["according to", "research shows", "studies indicate", "experts say"]
    has_source_claim = any(ind in answer_lower for ind in source_indicators)
    
    if has_source_claim:
        # Check if actual sources are cited
        has_citation = any(word in answer for word in ["[", "(", "http", "www", "source:", "ref:"]))
        if not has_citation:
            indicators.append(HallucinationIndicator(
                indicator_type="fabricated_source",
                severity=0.8,
                description="References sources without citations",
                location="Claims research/studies without references"
            ))
    
    # 3. Check for inconsistent details
    if context:
        # Look for specific numbers/dates that don't appear in context
        import re
        answer_numbers = set(re.findall(r'\b\d+\.?\d*\b', answer))
        context_numbers = set(re.findall(r'\b\d+\.?\d*\b', context))
        
        unsupported_numbers = answer_numbers - context_numbers
        if len(unsupported_numbers) > 3:
            indicators.append(HallucinationIndicator(
                indicator_type="inconsistent_detail",
                severity=0.6,
                description=f"Contains {len(unsupported_numbers)} numbers not in context",
                location="Specific numerical claims"
            ))
    
    # 4. Check for overly specific claims without context
    specific_indicators = [
        "in 2020", "in 2021", "in 2022", "in 2023", "in 2024",
        "version 1.", "version 2.", "version 3.",
        "released in", "published in"
    ]
    
    specific_claims = sum(1 for ind in specific_indicators if ind in answer_lower)
    if specific_claims > 2 and not context:
        indicators.append(HallucinationIndicator(
            indicator_type="unsupported_claim",
            severity=0.7,
            description="Makes specific version/date claims without source",
            location="Temporal or version-specific claims"
        ))
    
    # 5. Check for contradictory statements
    contradictions = [
        ("can", "cannot"), ("will", "will not"), ("is", "is not"),
        ("does", "does not"), ("has", "has not")
    ]
    
    for pos, neg in contradictions:
        if f" {pos} " in answer_lower and f" {neg} " in answer_lower:
            indicators.append(HallucinationIndicator(
                indicator_type="inconsistent_detail",
                severity=0.7,
                description=f"Contains both '{pos}' and '{neg}'",
                location="Contradictory statements"
            ))
    
    # Calculate overall hallucination risk
    if indicators:
        avg_severity = sum(ind.severity for ind in indicators) / len(indicators)
        hallucination_risk = min(1.0, avg_severity * (len(indicators) / 3))
    else:
        hallucination_risk = 0.0
    
    return indicators, hallucination_risk

def verify_claim(claim: Claim, context: Optional[str] = None) -> FactCheckResult:
    """Verify a single claim"""
    
    if not claim.verifiable:
        return FactCheckResult(
            claim=claim,
            verdict="unverifiable",
            confidence=0.5,
            explanation="Claim type is not factually verifiable (opinion/instruction)"
        )
    
    if not context:
        return FactCheckResult(
            claim=claim,
            verdict="uncertain",
            confidence=0.3,
            explanation="No context available for verification"
        )
    
    # Check if claim appears in context
    claim_lower = claim.text.lower()
    context_lower = context.lower()
    
    # Extract key terms from claim
    claim_words = set(claim_lower.split())
    context_words = set(context_lower.split())
    
    # Remove stop words
    stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "is", "are", "was", "were"}
    claim_words -= stop_words
    context_words -= stop_words
    
    # Calculate overlap
    overlap = claim_words & context_words
    support_ratio = len(overlap) / len(claim_words) if claim_words else 0
    
    # Determine verdict
    if support_ratio > 0.6:
        verdict = "supported"
        confidence = min(0.95, support_ratio * 1.2)
        explanation = f"{len(overlap)} key terms found in context"
        evidence = [f"Context contains: {', '.join(list(overlap)[:5])}"]
    elif support_ratio > 0.3:
        verdict = "uncertain"
        confidence = 0.5
        explanation = f"Partial support ({len(overlap)} terms match)"
        evidence = [f"Some overlap: {', '.join(list(overlap)[:3])}"]
    else:
        verdict = "contradicted"
        confidence = 0.7
        explanation = "Minimal overlap with context"
        evidence = ["Claim not found in provided context"]
    
    return FactCheckResult(
        claim=claim,
        verdict=verdict,
        confidence=confidence,
        evidence=evidence,
        sources=["RAG Context"],
        explanation=explanation
    )

async def check_factuality(
    query_id: str,
    query: str,
    answer: str,
    model: str,
    context: Optional[str] = None
) -> FactualityCheck:
    """Comprehensive factuality check"""
    
    if not app_state.factuality_config.enabled:
        # Return minimal check if disabled
        return FactualityCheck(
            query_id=query_id,
            query=query,
            answer=answer,
            model=model,
            timestamp=datetime.now().isoformat(),
            claims_extracted=[],
            claims_verified=0,
            claims_supported=0,
            claims_contradicted=0,
            claims_uncertain=0,
            fact_checks=[],
            hallucination_indicators=[],
            hallucination_risk=0.0,
            factuality_score=0.8,
            confidence=0.5,
            reliable=True,
            context_available=False,
            sources_cited=0
        )
    
    # 1. Extract claims
    claims = []
    if app_state.factuality_config.enable_claim_extraction:
        claims = extract_claims(answer)
    
    # 2. Verify claims against context
    fact_checks = []
    if app_state.factuality_config.enable_source_verification and context:
        for claim in claims:
            if claim.verifiable:
                fact_check = verify_claim(claim, context)
                fact_checks.append(fact_check)
    
    # 3. Detect hallucinations
    hallucination_indicators = []
    hallucination_risk = 0.0
    if app_state.factuality_config.enable_hallucination_detection:
        hallucination_indicators, hallucination_risk = detect_hallucinations(answer, context)
    
    # Calculate metrics
    claims_verified = len([fc for fc in fact_checks if fc.verdict != "unverifiable"])
    claims_supported = len([fc for fc in fact_checks if fc.verdict == "supported"])
    claims_contradicted = len([fc for fc in fact_checks if fc.verdict == "contradicted"])
    claims_uncertain = len([fc for fc in fact_checks if fc.verdict == "uncertain"])
    
    # Calculate factuality score
    if claims_verified > 0:
        support_rate = claims_supported / claims_verified
        contradiction_penalty = claims_contradicted / claims_verified
        factuality_score = max(0.0, support_rate - contradiction_penalty - (hallucination_risk * 0.3))
    else:
        # No verifiable claims - base on hallucination risk
        factuality_score = max(0.3, 0.7 - hallucination_risk)
    
    # Calculate confidence
    if fact_checks:
        avg_confidence = sum(fc.confidence for fc in fact_checks) / len(fact_checks)
        confidence = avg_confidence
    else:
        confidence = 0.5
    
    # Determine reliability
    reliable = True
    if factuality_score < 0.5:
        reliable = False
    if hallucination_risk > app_state.factuality_config.hallucination_threshold:
        reliable = False
    if claims_contradicted > claims_supported:
        reliable = False
    
    # Collect unsupported claims
    unsupported_claims = [
        fc.claim.text for fc in fact_checks
        if fc.verdict in ["contradicted", "uncertain"]
    ]
    
    # Count sources
    sources_cited = answer.count("[") + answer.count("(http")
    
    factuality_check = FactualityCheck(
        query_id=query_id,
        query=query,
        answer=answer,
        model=model,
        timestamp=datetime.now().isoformat(),
        claims_extracted=claims,
        claims_verified=claims_verified,
        claims_supported=claims_supported,
        claims_contradicted=claims_contradicted,
        claims_uncertain=claims_uncertain,
        fact_checks=fact_checks,
        hallucination_indicators=hallucination_indicators,
        hallucination_risk=hallucination_risk,
        factuality_score=factuality_score,
        confidence=confidence,
        reliable=reliable,
        context_available=context is not None,
        sources_cited=sources_cited,
        unsupported_claims=unsupported_claims
    )
    
    # Store check
    app_state.factuality_checks.append(factuality_check)
    
    # Keep only last 500
    if len(app_state.factuality_checks) > 500:
        app_state.factuality_checks = app_state.factuality_checks[-500:]
    
    logger.info("factuality_check", query_id=query_id, score=factuality_score, reliable=reliable)
    
    return factuality_check

# Factuality Checking API Endpoints
@app.post("/factuality/check")
async def check_factuality_endpoint(
    query_id: str,
    query: str,
    answer: str,
    model: str,
    context: Optional[str] = None
):
    """Check factuality of an answer"""
    check = await check_factuality(query_id, query, answer, model, context)
    return check

@app.get("/factuality/config")
async def get_factuality_config():
    """Get factuality checking configuration"""
    return app_state.factuality_config

@app.put("/factuality/config")
async def update_factuality_config(config: FactualityConfig):
    """Update factuality checking configuration"""
    app_state.factuality_config = config
    logger.info("updated_factuality_config", enabled=config.enabled)
    return config

@app.get("/factuality/history")
async def get_factuality_history(limit: int = 50):
    """Get recent factuality checks"""
    return {"checks": app_state.factuality_checks[-limit:]}

@app.get("/factuality/stats")
async def get_factuality_stats():
    """Get factuality checking statistics"""
    if not app_state.factuality_checks:
        return {
            "total_checks": 0,
            "avg_factuality_score": 0,
            "reliability_rate": 0,
            "avg_hallucination_risk": 0,
            "claims_verified": 0,
            "support_rate": 0
        }
    
    total = len(app_state.factuality_checks)
    avg_score = sum(fc.factuality_score for fc in app_state.factuality_checks) / total
    reliable_count = sum(1 for fc in app_state.factuality_checks if fc.reliable)
    reliability_rate = reliable_count / total
    avg_hallucination = sum(fc.hallucination_risk for fc in app_state.factuality_checks) / total
    
    total_verified = sum(fc.claims_verified for fc in app_state.factuality_checks)
    total_supported = sum(fc.claims_supported for fc in app_state.factuality_checks)
    support_rate = total_supported / total_verified if total_verified > 0 else 0
    
    return {
        "total_checks": total,
        "avg_factuality_score": round(avg_score, 3),
        "reliability_rate": round(reliability_rate, 3),
        "avg_hallucination_risk": round(avg_hallucination, 3),
        "claims_verified": total_verified,
        "support_rate": round(support_rate, 3)
    }

# Response Formatting Functions
def format_response(response: str, style: FormattingStyle = FormattingStyle.MARKDOWN) -> FormattedResponse:
    """Format and enhance response"""
    
    if not app_state.formatting_config.enabled:
        return FormattedResponse(
            original=response,
            formatted=response,
            style=style,
            original_length=len(response),
            formatted_length=len(response),
            improvement_score=0.0
        )
    
    formatted = response
    improvements = []
    
    # 1. Add sections for long responses
    if app_state.formatting_config.auto_add_sections and len(response) > 300:
        formatted = add_sections(formatted)
        if formatted != response:
            improvements.append("Added section headers")
    
    # 2. Format code blocks
    if app_state.formatting_config.auto_format_code:
        formatted = enhance_code_blocks(formatted)
        if "```" in formatted and formatted != response:
            improvements.append("Enhanced code blocks")
    
    # 3. Create lists
    if app_state.formatting_config.auto_create_lists:
        formatted = create_lists(formatted)
        if ("-" in formatted or "•" in formatted) and formatted != response:
            improvements.append("Created structured lists")
    
    # 4. Improve readability
    if app_state.formatting_config.improve_readability:
        formatted = improve_readability(formatted, app_state.formatting_config.max_line_length)
        if formatted != response:
            improvements.append("Improved readability")
    
    # 5. Add emoji headers (if enabled)
    if app_state.formatting_config.add_emoji_headers:
        formatted = add_emoji_headers(formatted)
        if formatted != response:
            improvements.append("Added emoji headers")
    
    # Calculate metrics
    has_sections = any(marker in formatted for marker in ["##", "###", "**"])
    has_code_blocks = formatted.count("```") // 2
    has_lists = "-" in formatted or "•" in formatted or any(f"{i}." in formatted for i in range(1, 10))
    has_tables = "|" in formatted and "---" in formatted
    
    # Calculate improvement score
    improvement_score = calculate_improvement_score(response, formatted, improvements)
    
    formatted_response = FormattedResponse(
        original=response,
        formatted=formatted,
        style=style,
        improvements=improvements,
        has_sections=has_sections,
        has_code_blocks=has_code_blocks,
        has_lists=has_lists,
        has_tables=has_tables,
        original_length=len(response),
        formatted_length=len(formatted),
        improvement_score=improvement_score
    )
    
    # Store formatting
    app_state.formatted_responses.append(formatted_response)
    
    # Keep only last 200
    if len(app_state.formatted_responses) > 200:
        app_state.formatted_responses = app_state.formatted_responses[-200:]
    
    return formatted_response

def add_sections(text: str) -> str:
    """Add section headers to long text"""
    lines = text.split('\n')
    if len(lines) < 5:
        return text
    
    # Look for natural section breaks
    formatted_lines = []
    in_code_block = False
    
    for i, line in enumerate(lines):
        # Track code blocks
        if "```" in line:
            in_code_block = not in_code_block
        
        # Don't modify code blocks
        if in_code_block:
            formatted_lines.append(line)
            continue
        
        # Add section headers before key phrases
        line_lower = line.lower().strip()
        
        if i > 0 and not formatted_lines[-1].startswith("#"):
            if line_lower.startswith(("step ", "1.", "2.", "3.", "first,", "second,", "finally,")):
                if not formatted_lines[-1].strip().startswith("##"):
                    formatted_lines.append("\n## Steps")
            elif line_lower.startswith(("example:", "for example", "e.g.")):
                if not formatted_lines[-1].strip().startswith("##"):
                    formatted_lines.append("\n## Example")
            elif line_lower.startswith(("note:", "important:", "warning:")):
                formatted_lines.append("\n## ⚠️ Important")
        
        formatted_lines.append(line)
    
    return '\n'.join(formatted_lines)

def enhance_code_blocks(text: str) -> str:
    """Enhance code block formatting"""
    if "```" not in text:
        return text
    
    parts = text.split("```")
    enhanced = []
    
    for i, part in enumerate(parts):
        if i % 2 == 1:  # Inside code block
            # Add language hint if missing
            lines = part.split('\n', 1)
            if lines[0].strip() and not any(lang in lines[0].lower() for lang in ["python", "javascript", "java", "c#", "go", "rust"]):
                # Detect language
                code = lines[1] if len(lines) > 1 else lines[0]
                lang = detect_code_language(code)
                if lang:
                    enhanced.append(f"{lang}\n{part}")
                else:
                    enhanced.append(part)
            else:
                enhanced.append(part)
        else:
            enhanced.append(part)
    
    return "```".join(enhanced)

def detect_code_language(code: str) -> str:
    """Simple language detection"""
    code_lower = code.lower()
    
    if "def " in code_lower or "import " in code_lower or "print(" in code_lower:
        return "python"
    elif "function " in code_lower or "const " in code_lower or "=>" in code:
        return "javascript"
    elif "class " in code and ("public " in code_lower or "private " in code_lower):
        return "java"
    elif "namespace" in code_lower or "using System" in code:
        return "csharp"
    elif "fn " in code_lower or "let mut" in code_lower:
        return "rust"
    elif "func " in code_lower and "package " in code_lower:
        return "go"
    
    return ""

def create_lists(text: str) -> str:
    """Convert text to lists where appropriate"""
    lines = text.split('\n')
    formatted = []
    in_code_block = False
    
    for line in lines:
        if "```" in line:
            in_code_block = not in_code_block
            formatted.append(line)
            continue
        
        if in_code_block:
            formatted.append(line)
            continue
        
        # Convert numbered patterns to lists
        line_stripped = line.strip()
        
        # Check for implicit list items
        if line_stripped.startswith(("- ", "* ", "• ")):
            formatted.append(line)
        elif any(line_stripped.startswith(f"{i}. ") or line_stripped.startswith(f"{i}) ") for i in range(1, 20)):
            # Already a list
            formatted.append(line)
        elif line_stripped and len(line_stripped) < 100:
            # Check if it's a sentence that should be a list item
            if line_stripped.endswith((',', ';', ':')) and len(formatted) > 0:
                # Might be start of a list
                formatted.append(line)
            else:
                formatted.append(line)
        else:
            formatted.append(line)
    
    return '\n'.join(formatted)

def improve_readability(text: str, max_line_length: int = 100) -> str:
    """Improve text readability"""
    lines = text.split('\n')
    formatted = []
    in_code_block = False
    
    for line in lines:
        if "```" in line:
            in_code_block = not in_code_block
            formatted.append(line)
            continue
        
        if in_code_block or len(line) <= max_line_length:
            formatted.append(line)
            continue
        
        # Break long lines at sentence boundaries
        if '. ' in line and len(line) > max_line_length:
            sentences = line.split('. ')
            current_line = ""
            
            for sentence in sentences:
                if len(current_line) + len(sentence) > max_line_length and current_line:
                    formatted.append(current_line.strip())
                    current_line = sentence + '. '
                else:
                    current_line += sentence + '. '
            
            if current_line:
                formatted.append(current_line.strip())
        else:
            formatted.append(line)
    
    return '\n'.join(formatted)

def add_emoji_headers(text: str) -> str:
    """Add emojis to headers"""
    lines = text.split('\n')
    formatted = []
    
    emoji_map = {
        "step": "📝", "example": "💡", "note": "📌", "important": "⚠️",
        "warning": "⚠️", "tip": "💡", "conclusion": "✅", "summary": "📊",
        "code": "💻", "result": "✨", "error": "❌", "success": "✅"
    }
    
    for line in lines:
        if line.startswith("##"):
            # Check if header already has emoji
            if not any(char in line[:5] for char in "🎯📝💡⚠️✅❌💻✨"):
                # Find matching emoji
                line_lower = line.lower()
                for keyword, emoji in emoji_map.items():
                    if keyword in line_lower:
                        # Add emoji after ##
                        parts = line.split(" ", 1)
                        if len(parts) > 1:
                            line = f"{parts[0]} {emoji} {parts[1]}"
                        break
        
        formatted.append(line)
    
    return '\n'.join(formatted)

def calculate_improvement_score(original: str, formatted: str, improvements: List[str]) -> float:
    """Calculate formatting improvement score"""
    score = 0.0
    
    # Base score on number of improvements
    score += min(0.4, len(improvements) * 0.1)
    
    # Check for structure
    if "##" in formatted and "##" not in original:
        score += 0.2
    
    # Check for code blocks
    if formatted.count("```") > original.count("```"):
        score += 0.1
    
    # Check for lists
    original_lists = original.count("-") + original.count("•")
    formatted_lists = formatted.count("-") + formatted.count("•")
    if formatted_lists > original_lists:
        score += 0.15
    
    # Check readability (fewer very long lines)
    original_long_lines = len([l for l in original.split('\n') if len(l) > 150])
    formatted_long_lines = len([l for l in formatted.split('\n') if len(l) > 150])
    if formatted_long_lines < original_long_lines:
        score += 0.15
    
    return min(1.0, score)

# Response Formatting API Endpoints
@app.post("/formatting/format")
async def format_response_endpoint(
    response: str,
    style: FormattingStyle = FormattingStyle.MARKDOWN
):
    """Format a response"""
    formatted = format_response(response, style)
    return formatted

@app.get("/formatting/config")
async def get_formatting_config():
    """Get formatting configuration"""
    return app_state.formatting_config

@app.put("/formatting/config")
async def update_formatting_config(config: FormattingConfig):
    """Update formatting configuration"""
    app_state.formatting_config = config
    logger.info("updated_formatting_config", enabled=config.enabled, style=config.default_style)
    return config

@app.get("/formatting/history")
async def get_formatting_history(limit: int = 50):
    """Get recent formatted responses"""
    return {"formatted_responses": app_state.formatted_responses[-limit:]}

@app.get("/formatting/stats")
async def get_formatting_stats():
    """Get formatting statistics"""
    if not app_state.formatted_responses:
        return {
            "total_formatted": 0,
            "avg_improvement_score": 0,
            "most_common_improvements": []
        }
    
    total = len(app_state.formatted_responses)
    avg_score = sum(fr.improvement_score for fr in app_state.formatted_responses) / total
    
    # Count improvement types
    improvement_counts = {}
    for fr in app_state.formatted_responses:
        for imp in fr.improvements:
            improvement_counts[imp] = improvement_counts.get(imp, 0) + 1
    
    most_common = sorted(improvement_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    
    return {
        "total_formatted": total,
        "avg_improvement_score": round(avg_score, 3),
        "most_common_improvements": [{"type": k, "count": v} for k, v in most_common]
    }

# Model Ensemble Management Endpoints
@app.post("/ensembles", response_model=EnsembleConfig)
async def create_ensemble(
    name: str,
    description: str,
    strategy: EnsembleStrategy,
    models: List[str],
    weights: Optional[List[float]] = None,
    threshold: Optional[float] = None,
    routing_rules: Optional[Dict[str, str]] = None,
    parallel: bool = True,
    timeout: int = 30,
    min_responses: int = 1
):
    """Create a new model ensemble"""
    ensemble_id = str(uuid.uuid4())
    
    # Validate weights if provided
    if weights and len(weights) != len(models):
        raise HTTPException(status_code=400, detail="Weights must match number of models")
    
    # Normalize weights
    if weights:
        total = sum(weights)
        weights = [w / total for w in weights]
    
    ensemble = EnsembleConfig(
        ensemble_id=ensemble_id,
        name=name,
        description=description,
        strategy=strategy,
        models=models,
        weights=weights,
        threshold=threshold,
        routing_rules=routing_rules,
        parallel=parallel,
        timeout=timeout,
        min_responses=min_responses,
        created_at=datetime.now().isoformat()
    )
    
    app_state.ensembles[ensemble_id] = ensemble
    logger.info("ensemble_created", ensemble_id=ensemble_id, name=name, strategy=strategy)
    
    return ensemble

@app.get("/ensembles")
async def list_ensembles():
    """List all model ensembles"""
    return {"ensembles": list(app_state.ensembles.values())}

@app.get("/ensembles/{ensemble_id}", response_model=EnsembleConfig)
async def get_ensemble(ensemble_id: str):
    """Get ensemble details"""
    if ensemble_id not in app_state.ensembles:
        raise HTTPException(status_code=404, detail="Ensemble not found")
    return app_state.ensembles[ensemble_id]

@app.put("/ensembles/{ensemble_id}/toggle")
async def toggle_ensemble(ensemble_id: str):
    """Enable/disable ensemble"""
    if ensemble_id not in app_state.ensembles:
        raise HTTPException(status_code=404, detail="Ensemble not found")
    
    ensemble = app_state.ensembles[ensemble_id]
    ensemble.enabled = not ensemble.enabled
    
    return {"ensemble_id": ensemble_id, "enabled": ensemble.enabled}

@app.delete("/ensembles/{ensemble_id}")
async def delete_ensemble(ensemble_id: str):
    """Delete an ensemble"""
    if ensemble_id not in app_state.ensembles:
        raise HTTPException(status_code=404, detail="Ensemble not found")
    
    del app_state.ensembles[ensemble_id]
    if ensemble_id in app_state.ensemble_results:
        del app_state.ensemble_results[ensemble_id]
    
    logger.info("ensemble_deleted", ensemble_id=ensemble_id)
    return {"status": "success", "message": "Ensemble deleted"}

@app.get("/ensembles/{ensemble_id}/results")
async def get_ensemble_results(ensemble_id: str, limit: int = 50):
    """Get ensemble query results"""
    if ensemble_id not in app_state.ensembles:
        raise HTTPException(status_code=404, detail="Ensemble not found")
    
    results = app_state.ensemble_results.get(ensemble_id, [])
    return {"results": results[-limit:]}

@app.post("/ensembles/{ensemble_id}/test")
async def test_ensemble(ensemble_id: str, question: str):
    """Test an ensemble with a sample question"""
    if ensemble_id not in app_state.ensembles:
        raise HTTPException(status_code=404, detail="Ensemble not found")
    
    ensemble = app_state.ensembles[ensemble_id]
    if not ensemble.enabled:
        raise HTTPException(status_code=400, detail="Ensemble is disabled")
    
    # Simple context for testing
    context = "This is a test query to evaluate the ensemble."
    
    result = await execute_ensemble(ensemble, question, context)
    
    return result

# A/B Test Management Endpoints
@app.post("/ab-tests", response_model=ABTestConfig)
async def create_ab_test(
    name: str,
    description: str,
    model_a: str,
    model_b: str,
    traffic_split: float = 0.5,
    min_samples: int = 30,
    confidence_level: float = 0.95
):
    """Create a new A/B test"""
    test_id = f"test_{int(time.time())}_{random.randint(1000, 9999)}"
    
    test = ABTestConfig(
        test_id=test_id,
        name=name,
        description=description,
        model_a=model_a,
        model_b=model_b,
        traffic_split=traffic_split,
        status=TestStatus.DRAFT,
        created_at=datetime.now().isoformat(),
        min_samples=min_samples,
        confidence_level=confidence_level
    )
    
    app_state.ab_tests[test_id] = test
    app_state.ab_results[test_id] = []
    
    logger.info("ab_test_created", test_id=test_id, name=name)
    return test

@app.get("/ab-tests")
async def list_ab_tests():
    """List all A/B tests"""
    return {
        "tests": list(app_state.ab_tests.values()),
        "count": len(app_state.ab_tests)
    }

@app.get("/ab-tests/{test_id}", response_model=ABTestConfig)
async def get_ab_test(test_id: str):
    """Get A/B test configuration"""
    if test_id not in app_state.ab_tests:
        raise HTTPException(status_code=404, detail="Test not found")
    return app_state.ab_tests[test_id]

@app.post("/ab-tests/{test_id}/start")
async def start_ab_test(test_id: str):
    """Start an A/B test"""
    if test_id not in app_state.ab_tests:
        raise HTTPException(status_code=404, detail="Test not found")
    
    test = app_state.ab_tests[test_id]
    if test.status != TestStatus.DRAFT:
        raise HTTPException(status_code=400, detail="Test must be in DRAFT status")
    
    test.status = TestStatus.RUNNING
    test.started_at = datetime.now().isoformat()
    
    logger.info("ab_test_started", test_id=test_id)
    return {"status": "started", "test": test}

@app.post("/ab-tests/{test_id}/pause")
async def pause_ab_test(test_id: str):
    """Pause an A/B test"""
    if test_id not in app_state.ab_tests:
        raise HTTPException(status_code=404, detail="Test not found")
    
    test = app_state.ab_tests[test_id]
    if test.status != TestStatus.RUNNING:
        raise HTTPException(status_code=400, detail="Test must be RUNNING")
    
    test.status = TestStatus.PAUSED
    logger.info("ab_test_paused", test_id=test_id)
    return {"status": "paused", "test": test}

@app.post("/ab-tests/{test_id}/resume")
async def resume_ab_test(test_id: str):
    """Resume a paused A/B test"""
    if test_id not in app_state.ab_tests:
        raise HTTPException(status_code=404, detail="Test not found")
    
    test = app_state.ab_tests[test_id]
    if test.status != TestStatus.PAUSED:
        raise HTTPException(status_code=400, detail="Test must be PAUSED")
    
    test.status = TestStatus.RUNNING
    logger.info("ab_test_resumed", test_id=test_id)
    return {"status": "resumed", "test": test}

@app.post("/ab-tests/{test_id}/complete")
async def complete_ab_test(test_id: str):
    """Complete an A/B test and declare winner"""
    if test_id not in app_state.ab_tests:
        raise HTTPException(status_code=404, detail="Test not found")
    
    test = app_state.ab_tests[test_id]
    test.status = TestStatus.COMPLETED
    test.ended_at = datetime.now().isoformat()
    
    # Analyze results
    try:
        stats = await analyze_ab_test(test_id)
    except HTTPException:
        stats = None
    
    logger.info("ab_test_completed", test_id=test_id, winner=stats.winner if stats else None)
    return {"status": "completed", "test": test, "statistics": stats}

@app.get("/ab-tests/{test_id}/statistics", response_model=ABTestStatistics)
async def get_ab_test_statistics(test_id: str):
    """Get statistical analysis of A/B test"""
    return await analyze_ab_test(test_id)

@app.get("/ab-tests/{test_id}/results")
async def get_ab_test_results(test_id: str):
    """Get all results for an A/B test"""
    if test_id not in app_state.ab_tests:
        raise HTTPException(status_code=404, detail="Test not found")
    
    results = app_state.ab_results.get(test_id, [])
    return {
        "test_id": test_id,
        "results": [r.dict() for r in results],
        "count": len(results)
    }

@app.post("/ab-tests/{test_id}/rate")
async def rate_ab_test_response(test_id: str, query_id: str, rating: int):
    """Rate a response from A/B test (1-5 stars)"""
    if test_id not in app_state.ab_tests:
        raise HTTPException(status_code=404, detail="Test not found")
    
    if rating < 1 or rating > 5:
        raise HTTPException(status_code=400, detail="Rating must be between 1 and 5")
    
    results = app_state.ab_results.get(test_id, [])
    for result in results:
        if result.query_id == query_id:
            result.user_rating = rating
            logger.info("ab_test_rated", test_id=test_id, query_id=query_id, rating=rating)
            return {"status": "rated", "query_id": query_id, "rating": rating}
    
    raise HTTPException(status_code=404, detail="Query not found")

@app.delete("/ab-tests/{test_id}")
async def delete_ab_test(test_id: str):
    """Delete an A/B test"""
    if test_id not in app_state.ab_tests:
        raise HTTPException(status_code=404, detail="Test not found")
    
    del app_state.ab_tests[test_id]
    if test_id in app_state.ab_results:
        del app_state.ab_results[test_id]
    
    logger.info("ab_test_deleted", test_id=test_id)
    return {"status": "deleted", "test_id": test_id}

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
        # Check for A/B test participation
        ab_test_model = None
        participating_in_test = False
        if q.ab_test_id and q.ab_test_id in app_state.ab_tests:
            test = app_state.ab_tests[q.ab_test_id]
            if test.status == TestStatus.RUNNING:
                ab_test_model = select_model_for_test(test)
                participating_in_test = True
                logger.info("ab_test_query", test_id=q.ab_test_id, model=ab_test_model)
        
        # Determine routing (with A/B test or override support)
        if participating_in_test:
            model_name = ab_test_model
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
        elif q.model_override:
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
        
        # Record A/B test result if participating
        if participating_in_test and q.ab_test_id:
            query_id = q.query_id or f"query_{int(time.time())}_{random.randint(1000, 9999)}"
            ab_result = ABTestResult(
                test_id=q.ab_test_id,
                query_id=query_id,
                model=model_name,
                question=q.question,
                response_time=total_time,
                tokens=token_count,
                tokens_per_sec=tokens_per_second,
                timestamp=datetime.now().isoformat()
            )
            await record_ab_test_result(ab_result)
            result["ab_test"] = {
                "test_id": q.ab_test_id,
                "query_id": query_id,
                "model_selected": model_name
            }
        
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

