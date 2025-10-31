"""
Simple Unit Tests
Basic tests to verify test framework is working

Author: Adrian Johnson <adrian207@gmail.com>
"""

import pytest


@pytest.mark.unit
class TestBasicPython:
    """Test basic Python functionality"""
    
    def test_addition(self):
        """Test basic addition"""
        assert 1 + 1 == 2
    
    def test_string_operations(self):
        """Test string operations"""
        text = "Dual RAG LLM"
        assert "RAG" in text
        assert text.lower() == "dual rag llm"
    
    def test_list_operations(self):
        """Test list operations"""
        items = [1, 2, 3, 4, 5]
        assert len(items) == 5
        assert sum(items) == 15
        assert max(items) == 5


@pytest.mark.unit
class TestVersionInfo:
    """Test version information"""
    
    def test_version_import(self):
        """Test importing version from rag module"""
        from rag import __version__
        assert __version__ == "1.21.0"
    
    def test_author_info(self):
        """Test author information"""
        from rag import __author__
        assert "Adrian Johnson" in __author__


@pytest.mark.unit
@pytest.mark.asyncio
async def test_async_functionality():
    """Test async support in tests"""
    import asyncio
    
    async def async_add(a, b):
        await asyncio.sleep(0.001)  # Simulate async operation
        return a + b
    
    result = await async_add(5, 3)
    assert result == 8

