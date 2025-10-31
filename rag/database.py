"""
Database Backend for Vector Indexes
PostgreSQL with pgvector for scalable vector storage

Author: Adrian Johnson <adrian207@gmail.com>
"""

import os
import asyncio
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
import json

import asyncpg
import numpy as np
from pydantic import BaseModel
import structlog

logger = structlog.get_logger()


class VectorDocument(BaseModel):
    """Vector document model"""
    id: str
    content: str
    embedding: List[float]
    metadata: Dict[str, Any] = {}
    collection: str
    created_at: datetime
    updated_at: datetime


class DatabaseConfig(BaseModel):
    """Database configuration"""
    host: str = os.getenv("POSTGRES_HOST", "localhost")
    port: int = int(os.getenv("POSTGRES_PORT", "5432"))
    database: str = os.getenv("POSTGRES_DB", "dual_rag")
    user: str = os.getenv("POSTGRES_USER", "postgres")
    password: str = os.getenv("POSTGRES_PASSWORD", "postgres")
    pool_min_size: int = 5
    pool_max_size: int = 20
    ssl_mode: str = os.getenv("POSTGRES_SSL_MODE", "prefer")


class VectorDatabase:
    """PostgreSQL vector database with pgvector"""
    
    def __init__(self, config: Optional[DatabaseConfig] = None):
        self.config = config or DatabaseConfig()
        self.pool: Optional[asyncpg.Pool] = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize database connection and schema"""
        if self._initialized:
            return
        
        try:
            # Create connection pool
            self.pool = await asyncpg.create_pool(
                host=self.config.host,
                port=self.config.port,
                database=self.config.database,
                user=self.config.user,
                password=self.config.password,
                min_size=self.config.pool_min_size,
                max_size=self.config.pool_max_size,
                command_timeout=60,
            )
            
            # Initialize schema
            await self._init_schema()
            
            self._initialized = True
            logger.info("database_initialized", 
                       host=self.config.host, 
                       database=self.config.database)
        
        except Exception as e:
            logger.error("database_init_failed", error=str(e))
            raise
    
    async def _init_schema(self):
        """Initialize database schema"""
        async with self.pool.acquire() as conn:
            # Enable pgvector extension
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            
            # Create vector documents table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS vector_documents (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    embedding vector(384),  -- MiniLM embedding size
                    metadata JSONB DEFAULT '{}',
                    collection TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW()
                );
            """)
            
            # Create indexes
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_vector_collection 
                ON vector_documents(collection);
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_vector_created 
                ON vector_documents(created_at DESC);
            """)
            
            # Create IVFFLAT index for fast similarity search
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_vector_embedding 
                ON vector_documents 
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100);
            """)
            
            # Create collections metadata table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS collections (
                    name TEXT PRIMARY KEY,
                    description TEXT,
                    document_count INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW()
                );
            """)
            
            logger.info("database_schema_initialized")
    
    async def close(self):
        """Close database connections"""
        if self.pool:
            await self.pool.close()
            self._initialized = False
            logger.info("database_closed")
    
    async def insert_document(self, document: VectorDocument) -> bool:
        """Insert a vector document"""
        try:
            async with self.pool.acquire() as conn:
                # Convert embedding to string format for pgvector
                embedding_str = "[" + ",".join(map(str, document.embedding)) + "]"
                
                await conn.execute("""
                    INSERT INTO vector_documents 
                    (id, content, embedding, metadata, collection, created_at, updated_at)
                    VALUES ($1, $2, $3::vector, $4, $5, $6, $7)
                    ON CONFLICT (id) DO UPDATE SET
                        content = EXCLUDED.content,
                        embedding = EXCLUDED.embedding,
                        metadata = EXCLUDED.metadata,
                        updated_at = EXCLUDED.updated_at
                """, document.id, document.content, embedding_str, 
                json.dumps(document.metadata), document.collection,
                document.created_at, document.updated_at)
                
                # Update collection count
                await conn.execute("""
                    INSERT INTO collections (name, document_count, updated_at)
                    VALUES ($1, 1, NOW())
                    ON CONFLICT (name) DO UPDATE SET
                        document_count = collections.document_count + 1,
                        updated_at = NOW()
                """, document.collection)
                
                logger.info("document_inserted", 
                           doc_id=document.id, 
                           collection=document.collection)
                return True
        
        except Exception as e:
            logger.error("document_insert_failed", 
                        doc_id=document.id, 
                        error=str(e))
            return False
    
    async def search_similar(
        self, 
        embedding: List[float], 
        collection: str,
        limit: int = 10,
        threshold: float = 0.7
    ) -> List[Tuple[VectorDocument, float]]:
        """Search for similar documents using cosine similarity"""
        try:
            async with self.pool.acquire() as conn:
                # Convert embedding to string format
                embedding_str = "[" + ",".join(map(str, embedding)) + "]"
                
                # Search using cosine similarity
                rows = await conn.fetch("""
                    SELECT 
                        id, content, embedding::text, metadata, collection,
                        created_at, updated_at,
                        1 - (embedding <=> $1::vector) as similarity
                    FROM vector_documents
                    WHERE collection = $2
                    AND 1 - (embedding <=> $1::vector) >= $3
                    ORDER BY embedding <=> $1::vector
                    LIMIT $4
                """, embedding_str, collection, threshold, limit)
                
                results = []
                for row in rows:
                    # Parse embedding back to list
                    embedding_list = json.loads(row['embedding'].replace('vector', ''))
                    
                    doc = VectorDocument(
                        id=row['id'],
                        content=row['content'],
                        embedding=embedding_list,
                        metadata=row['metadata'],
                        collection=row['collection'],
                        created_at=row['created_at'],
                        updated_at=row['updated_at']
                    )
                    results.append((doc, row['similarity']))
                
                logger.info("similarity_search_completed",
                           collection=collection,
                           results_count=len(results))
                
                return results
        
        except Exception as e:
            logger.error("similarity_search_failed", error=str(e))
            return []
    
    async def get_document(self, doc_id: str) -> Optional[VectorDocument]:
        """Get a document by ID"""
        try:
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow("""
                    SELECT id, content, embedding::text, metadata, collection,
                           created_at, updated_at
                    FROM vector_documents
                    WHERE id = $1
                """, doc_id)
                
                if not row:
                    return None
                
                embedding_list = json.loads(row['embedding'].replace('vector', ''))
                
                return VectorDocument(
                    id=row['id'],
                    content=row['content'],
                    embedding=embedding_list,
                    metadata=row['metadata'],
                    collection=row['collection'],
                    created_at=row['created_at'],
                    updated_at=row['updated_at']
                )
        
        except Exception as e:
            logger.error("get_document_failed", doc_id=doc_id, error=str(e))
            return None
    
    async def delete_document(self, doc_id: str) -> bool:
        """Delete a document"""
        try:
            async with self.pool.acquire() as conn:
                # Get collection before deleting
                row = await conn.fetchrow("""
                    SELECT collection FROM vector_documents WHERE id = $1
                """, doc_id)
                
                if not row:
                    return False
                
                collection = row['collection']
                
                # Delete document
                await conn.execute("""
                    DELETE FROM vector_documents WHERE id = $1
                """, doc_id)
                
                # Update collection count
                await conn.execute("""
                    UPDATE collections 
                    SET document_count = document_count - 1,
                        updated_at = NOW()
                    WHERE name = $1
                """, collection)
                
                logger.info("document_deleted", doc_id=doc_id)
                return True
        
        except Exception as e:
            logger.error("document_delete_failed", doc_id=doc_id, error=str(e))
            return False
    
    async def get_collection_stats(self, collection: str) -> Dict[str, Any]:
        """Get statistics for a collection"""
        try:
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow("""
                    SELECT name, description, document_count, created_at, updated_at
                    FROM collections
                    WHERE name = $1
                """, collection)
                
                if not row:
                    return {"name": collection, "document_count": 0}
                
                return {
                    "name": row['name'],
                    "description": row['description'],
                    "document_count": row['document_count'],
                    "created_at": row['created_at'].isoformat(),
                    "updated_at": row['updated_at'].isoformat()
                }
        
        except Exception as e:
            logger.error("get_collection_stats_failed", error=str(e))
            return {"name": collection, "document_count": 0}
    
    async def list_collections(self) -> List[Dict[str, Any]]:
        """List all collections"""
        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT name, description, document_count, created_at, updated_at
                    FROM collections
                    ORDER BY updated_at DESC
                """)
                
                return [
                    {
                        "name": row['name'],
                        "description": row['description'],
                        "document_count": row['document_count'],
                        "created_at": row['created_at'].isoformat(),
                        "updated_at": row['updated_at'].isoformat()
                    }
                    for row in rows
                ]
        
        except Exception as e:
            logger.error("list_collections_failed", error=str(e))
            return []
    
    async def health_check(self) -> Dict[str, Any]:
        """Check database health"""
        try:
            async with self.pool.acquire() as conn:
                # Simple query to check connectivity
                result = await conn.fetchval("SELECT 1")
                
                # Get pool stats
                pool_size = self.pool.get_size()
                pool_free = self.pool.get_idle_size()
                
                return {
                    "healthy": result == 1,
                    "pool_size": pool_size,
                    "pool_free": pool_free,
                    "pool_used": pool_size - pool_free,
                    "timestamp": datetime.utcnow().isoformat()
                }
        
        except Exception as e:
            logger.error("health_check_failed", error=str(e))
            return {
                "healthy": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }


# Global database instance
_db_instance: Optional[VectorDatabase] = None


async def get_database() -> VectorDatabase:
    """Get or create database instance"""
    global _db_instance
    
    if _db_instance is None:
        _db_instance = VectorDatabase()
        await _db_instance.initialize()
    
    return _db_instance


async def close_database():
    """Close database instance"""
    global _db_instance
    
    if _db_instance:
        await _db_instance.close()
        _db_instance = None

