"""
Document Ingestion and Index Building for Dual RAG System

Author: Adrian Johnson <adrian207@gmail.com>

Builds separate ChromaDB indexes for Microsoft and Open Source documentation.
Supports multiple document formats: PDF, Markdown, HTML, TXT.
"""

import os
import sys
from pathlib import Path
from typing import List, Optional
import structlog

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
    StorageContext,
    Document
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from chromadb import PersistentClient

# Configure logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.JSONRenderer()
    ]
)
logger = structlog.get_logger()

# Configuration
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
MS_DATA_DIR = "/app/data/ms_docs"
OSS_DATA_DIR = "/app/data/oss_docs"
MS_INDEX_DIR = "/app/indexes/chroma_ms"
OSS_INDEX_DIR = "/app/indexes/chroma_oss"

def initialize_settings():
    """Initialize global LlamaIndex settings"""
    Settings.embed_model = HuggingFaceEmbedding(EMBEDDING_MODEL)
    Settings.chunk_size = 512
    Settings.chunk_overlap = 50
    logger.info("settings_initialized", embedding_model=EMBEDDING_MODEL)

def load_documents(data_dir: str) -> Optional[List[Document]]:
    """
    Load documents from directory with support for multiple formats.
    
    Args:
        data_dir: Path to directory containing documents
        
    Returns:
        List of Document objects or None if directory doesn't exist
    """
    if not os.path.exists(data_dir):
        logger.warning("data_directory_not_found", path=data_dir)
        return None
    
    try:
        # Check if directory is empty
        files = list(Path(data_dir).rglob("*"))
        files = [f for f in files if f.is_file()]
        
        if not files:
            logger.warning("data_directory_empty", path=data_dir)
            return None
        
        logger.info("loading_documents", path=data_dir, file_count=len(files))
        
        # Load documents with support for multiple formats
        reader = SimpleDirectoryReader(
            input_dir=data_dir,
            recursive=True,
            required_exts=[".pdf", ".md", ".txt", ".html", ".rst", ".docx"]
        )
        documents = reader.load_data()
        
        logger.info(
            "documents_loaded",
            path=data_dir,
            document_count=len(documents),
            total_chars=sum(len(doc.text) for doc in documents)
        )
        
        return documents
        
    except Exception as e:
        logger.error("document_loading_failed", path=data_dir, error=str(e))
        return None

def build_index(
    documents: List[Document],
    index_dir: str,
    collection_name: str
) -> bool:
    """
    Build and persist ChromaDB vector index.
    
    Args:
        documents: List of documents to index
        index_dir: Directory to store the index
        collection_name: Name for the ChromaDB collection
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create directory if needed
        os.makedirs(index_dir, exist_ok=True)
        
        logger.info(
            "building_index",
            collection=collection_name,
            document_count=len(documents)
        )
        
        # Initialize ChromaDB client
        client = PersistentClient(path=index_dir)
        
        # Delete existing collection if it exists
        try:
            client.delete_collection(collection_name)
            logger.info("deleted_existing_collection", collection=collection_name)
        except:
            pass
        
        # Create new collection
        collection = client.create_collection(collection_name)
        vector_store = ChromaVectorStore(chroma_collection=collection)
        
        # Build index with storage context
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            show_progress=True
        )
        
        logger.info(
            "index_built_successfully",
            collection=collection_name,
            index_dir=index_dir
        )
        
        return True
        
    except Exception as e:
        logger.error(
            "index_build_failed",
            collection=collection_name,
            error=str(e)
        )
        return False

def build_ms_index() -> bool:
    """Build Microsoft documentation index"""
    logger.info("starting_ms_index_build")
    
    documents = load_documents(MS_DATA_DIR)
    if not documents:
        logger.warning("no_ms_documents_found")
        return False
    
    success = build_index(documents, MS_INDEX_DIR, "msdocs")
    
    if success:
        logger.info("ms_index_completed")
    else:
        logger.error("ms_index_failed")
    
    return success

def build_oss_index() -> bool:
    """Build Open Source documentation index"""
    logger.info("starting_oss_index_build")
    
    documents = load_documents(OSS_DATA_DIR)
    if not documents:
        logger.warning("no_oss_documents_found")
        return False
    
    success = build_index(documents, OSS_INDEX_DIR, "ossdocs")
    
    if success:
        logger.info("oss_index_completed")
    else:
        logger.error("oss_index_failed")
    
    return success

def build_all_indexes() -> int:
    """
    Build all indexes.
    
    Returns:
        Exit code: 0 if all successful, 1 if any failed, 2 if all failed
    """
    logger.info("starting_index_build_process")
    initialize_settings()
    
    ms_success = build_ms_index()
    oss_success = build_oss_index()
    
    if ms_success and oss_success:
        logger.info("all_indexes_built_successfully")
        return 0
    elif ms_success or oss_success:
        logger.warning("partial_index_build_success")
        return 1
    else:
        logger.error("all_indexes_failed")
        return 2

def create_sample_documents():
    """Create sample documents for testing if data directories are empty"""
    logger.info("creating_sample_documents")
    
    # Create MS sample
    os.makedirs(MS_DATA_DIR, exist_ok=True)
    ms_sample = """
# C# Programming Guide

## Introduction to C#
C# is a modern, object-oriented programming language developed by Microsoft.

## Basic Syntax
```csharp
using System;

class Program
{
    static void Main()
    {
        Console.WriteLine("Hello, World!");
    }
}
```

## Key Features
- Type safety
- Garbage collection
- LINQ (Language Integrated Query)
- Async/await pattern
- Properties and indexers

## PowerShell Integration
C# and PowerShell work together seamlessly. You can call .NET methods from PowerShell:
```powershell
[System.DateTime]::Now
```
"""
    
    with open(f"{MS_DATA_DIR}/csharp_guide.md", "w") as f:
        f.write(ms_sample)
    
    # Create OSS sample
    os.makedirs(OSS_DATA_DIR, exist_ok=True)
    oss_sample = """
# Python Programming Guide

## Introduction to Python
Python is a high-level, interpreted programming language known for its simplicity.

## Basic Syntax
```python
def hello_world():
    print("Hello, World!")

if __name__ == "__main__":
    hello_world()
```

## Key Features
- Dynamic typing
- Extensive standard library
- List comprehensions
- Decorators
- Context managers

## Popular Frameworks
- Django and Flask for web development
- NumPy and Pandas for data science
- TensorFlow and PyTorch for machine learning
"""
    
    with open(f"{OSS_DATA_DIR}/python_guide.md", "w") as f:
        f.write(oss_sample)
    
    logger.info("sample_documents_created")

if __name__ == "__main__":
    # Check if we should create sample documents
    if "--create-samples" in sys.argv:
        create_sample_documents()
    
    # Build indexes
    exit_code = build_all_indexes()
    sys.exit(exit_code)

