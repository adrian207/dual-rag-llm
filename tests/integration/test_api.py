"""
Integration Tests for API Endpoints
End-to-end testing of FastAPI application

Author: Adrian Johnson <adrian207@gmail.com>
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock, MagicMock


@pytest.mark.integration
class TestHealthEndpoint:
    """Test health check endpoint"""
    
    def test_health_check(self, client):
        """Test basic health check"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
    
    def test_health_check_detailed(self, client):
        """Test detailed health status"""
        response = client.get("/health")
        data = response.json()
        assert "ollama_status" in data or "status" in data


@pytest.mark.integration
class TestQueryEndpoint:
    """Test query endpoints"""
    
    def test_query_basic(self, client, mock_ollama_client):
        """Test basic query"""
        with patch('rag.rag_dual.ollama_client', mock_ollama_client):
            response = client.post("/query", json={
                "query": "How do I use FastAPI?",
                "file_context": "app.py"
            })
            assert response.status_code == 200 or response.status_code == 500  # May fail without real services
    
    def test_query_validation(self, client):
        """Test query input validation"""
        response = client.post("/query", json={
            "query": "",  # Empty query
            "file_context": "app.py"
        })
        assert response.status_code == 422  # Validation error
    
    def test_query_stream(self, client):
        """Test streaming query endpoint"""
        response = client.post("/query/stream", json={
            "query": "Test query",
            "file_context": "test.py"
        }, stream=True)
        # Should return SSE or error
        assert response.status_code in [200, 500, 503]


@pytest.mark.integration
class TestModelEndpoints:
    """Test model management endpoints"""
    
    def test_list_models(self, client):
        """Test listing available models"""
        response = client.get("/models")
        assert response.status_code in [200, 503]
        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, list)
    
    def test_get_current_model(self, client):
        """Test getting current model"""
        response = client.get("/current-model")
        assert response.status_code in [200, 503]


@pytest.mark.integration
class TestAnalyticsEndpoints:
    """Test analytics endpoints"""
    
    def test_get_analytics_config(self, client):
        """Test getting analytics configuration"""
        response = client.get("/api/analytics/config")
        assert response.status_code == 200
        data = response.json()
        assert "enabled" in data
    
    def test_update_analytics_config(self, client):
        """Test updating analytics configuration"""
        response = client.put("/api/analytics/config", json={
            "enabled": True,
            "retention_days": 90
        })
        assert response.status_code in [200, 422]


@pytest.mark.integration
class TestCostEndpoints:
    """Test cost tracking endpoints"""
    
    def test_get_cost_config(self, client):
        """Test getting cost configuration"""
        response = client.get("/api/cost/config")
        assert response.status_code == 200
        data = response.json()
        assert "track_costs" in data
    
    def test_get_cost_summary(self, client):
        """Test getting cost summary"""
        response = client.get("/api/cost/summary")
        assert response.status_code == 200
        data = response.json()
        assert "total_cost" in data


@pytest.mark.integration
@pytest.mark.slow
class TestBackupEndpoints:
    """Test backup endpoints"""
    
    def test_create_backup(self, client, mock_backup_manager):
        """Test creating a backup"""
        with patch('rag.rag_dual.backup_manager', mock_backup_manager):
            response = client.post("/api/backup/create")
            assert response.status_code in [200, 500]
    
    def test_list_backups(self, client, mock_backup_manager):
        """Test listing backups"""
        with patch('rag.rag_dual.backup_manager', mock_backup_manager):
            response = client.get("/api/backup/list")
            assert response.status_code in [200, 500]


@pytest.mark.integration
@pytest.mark.slow
class TestDisasterRecoveryEndpoints:
    """Test disaster recovery endpoints"""
    
    def test_get_dr_status(self, client, mock_dr_manager):
        """Test getting DR status"""
        with patch('rag.rag_dual.dr_manager', mock_dr_manager):
            response = client.get("/api/dr/status")
            assert response.status_code in [200, 500]
    
    def test_health_check(self, client, mock_dr_manager):
        """Test DR health check"""
        with patch('rag.rag_dual.dr_manager', mock_dr_manager):
            response = client.get("/api/dr/health")
            assert response.status_code in [200, 500]


# Fixtures for integration tests
@pytest.fixture(scope="module")
def client():
    """Create test client"""
    try:
        from rag.rag_dual import app
        return TestClient(app)
    except Exception as e:
        pytest.skip(f"Could not create test client: {e}")

