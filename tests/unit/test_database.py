"""
Unit Tests for Database Module
Tests for PostgreSQL vector database operations

Author: Adrian Johnson <adrian207@gmail.com>
"""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from rag.database import (
    VectorDatabase,
    VectorDocument,
    DatabaseConfig,
    get_database,
    close_database
)


@pytest.mark.unit
@pytest.mark.database
class TestDatabaseConfig:
    """Test database configuration"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = DatabaseConfig()
        assert config.host == "localhost"
        assert config.port == 5432
        assert config.database == "dual_rag"
        assert config.pool_min_size == 5
        assert config.pool_max_size == 20
    
    def test_custom_config(self):
        """Test custom configuration"""
        config = DatabaseConfig(
            host="db.example.com",
            port=5433,
            database="custom_db",
            pool_min_size=10
        )
        assert config.host == "db.example.com"
        assert config.port == 5433
        assert config.database == "custom_db"
        assert config.pool_min_size == 10


@pytest.mark.unit
@pytest.mark.database
class TestVectorDocument:
    """Test vector document model"""
    
    def test_create_document(self):
        """Test document creation"""
        doc = VectorDocument(
            id="test_doc",
            content="Test content",
            embedding=[0.1, 0.2, 0.3],
            metadata={"source": "test.py"},
            collection="test_collection",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        assert doc.id == "test_doc"
        assert doc.content == "Test content"
        assert len(doc.embedding) == 3
        assert doc.metadata["source"] == "test.py"
    
    def test_document_validation(self):
        """Test document validation"""
        with pytest.raises(Exception):
            VectorDocument(
                id="test",
                content="",  # Empty content should fail
                embedding=[],
                collection="test",
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )


@pytest.mark.unit
@pytest.mark.database
@pytest.mark.asyncio
class TestVectorDatabase:
    """Test vector database operations"""
    
    @patch('rag.database.asyncpg.create_pool')
    async def test_initialize(self, mock_pool):
        """Test database initialization"""
        mock_pool_instance = AsyncMock()
        mock_pool.return_value = mock_pool_instance
        
        mock_conn = AsyncMock()
        mock_pool_instance.acquire.return_value.__aenter__.return_value = mock_conn
        
        db = VectorDatabase()
        await db.initialize()
        
        assert db._initialized
        assert db.pool is not None
        mock_pool.assert_called_once()
    
    @patch('rag.database.asyncpg.create_pool')
    async def test_health_check(self, mock_pool):
        """Test database health check"""
        mock_pool_instance = AsyncMock()
        mock_pool.return_value = mock_pool_instance
        
        mock_conn = AsyncMock()
        mock_conn.fetchval.return_value = 1
        mock_pool_instance.acquire.return_value.__aenter__.return_value = mock_conn
        mock_pool_instance.get_size.return_value = 10
        mock_pool_instance.get_idle_size.return_value = 5
        
        db = VectorDatabase()
        db.pool = mock_pool_instance
        db._initialized = True
        
        health = await db.health_check()
        
        assert health["healthy"] is True
        assert health["pool_size"] == 10
        assert health["pool_free"] == 5
        assert health["pool_used"] == 5
    
    @patch('rag.database.asyncpg.create_pool')
    async def test_insert_document(self, mock_pool):
        """Test document insertion"""
        mock_pool_instance = AsyncMock()
        mock_pool.return_value = mock_pool_instance
        
        mock_conn = AsyncMock()
        mock_pool_instance.acquire.return_value.__aenter__.return_value = mock_conn
        
        db = VectorDatabase()
        db.pool = mock_pool_instance
        db._initialized = True
        
        doc = VectorDocument(
            id="test_doc",
            content="Test content",
            embedding=[0.1, 0.2, 0.3],
            metadata={"source": "test.py"},
            collection="test_collection",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        result = await db.insert_document(doc)
        
        assert result is True
        assert mock_conn.execute.call_count == 2  # Insert + update collection
    
    @patch('rag.database.asyncpg.create_pool')
    async def test_search_similar(self, mock_pool):
        """Test similarity search"""
        mock_pool_instance = AsyncMock()
        mock_pool.return_value = mock_pool_instance
        
        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = [
            {
                'id': 'doc1',
                'content': 'Test content',
                'embedding': '[0.1,0.2,0.3]',
                'metadata': {},
                'collection': 'test',
                'created_at': datetime.utcnow(),
                'updated_at': datetime.utcnow(),
                'similarity': 0.95
            }
        ]
        mock_pool_instance.acquire.return_value.__aenter__.return_value = mock_conn
        
        db = VectorDatabase()
        db.pool = mock_pool_instance
        db._initialized = True
        
        results = await db.search_similar(
            embedding=[0.1, 0.2, 0.3],
            collection="test",
            limit=10
        )
        
        assert len(results) == 1
        doc, similarity = results[0]
        assert doc.id == 'doc1'
        assert similarity == 0.95
    
    async def test_close(self):
        """Test database connection closing"""
        mock_pool = AsyncMock()
        
        db = VectorDatabase()
        db.pool = mock_pool
        db._initialized = True
        
        await db.close()
        
        mock_pool.close.assert_called_once()
        assert db._initialized is False


@pytest.mark.unit
@pytest.mark.database
@pytest.mark.asyncio
async def test_get_database_singleton():
    """Test database singleton pattern"""
    with patch('rag.database._db_instance', None):
        with patch.object(VectorDatabase, 'initialize', new=AsyncMock()):
            db1 = await get_database()
            db2 = await get_database()
            assert db1 is db2  # Should be same instance


@pytest.mark.unit
@pytest.mark.database
@pytest.mark.asyncio
async def test_close_database():
    """Test closing database singleton"""
    with patch('rag.database._db_instance') as mock_instance:
        mock_instance.close = AsyncMock()
        await close_database()
        mock_instance.close.assert_called_once()

