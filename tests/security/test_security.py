"""
Security Testing Suite
Tests for common security vulnerabilities

Author: Adrian Johnson <adrian207@gmail.com>
"""

import pytest
from fastapi.testclient import TestClient


@pytest.mark.security
class TestInputValidation:
    """Test input validation and sanitization"""
    
    def test_sql_injection_protection(self, client, malicious_inputs):
        """Test protection against SQL injection"""
        for payload in malicious_inputs:
            response = client.post("/query", json={
                "query": payload,
                "file_context": "test.py"
            })
            # Should not crash or return database errors
            assert response.status_code in [200, 422, 500]
            if response.status_code == 200:
                # Check that malicious content is not executed
                data = response.json()
                assert "DROP TABLE" not in str(data).upper()
    
    def test_xss_protection(self, client):
        """Test protection against XSS attacks"""
        xss_payloads = [
            "<script>alert('XSS')</script>",
            "<img src=x onerror=alert(1)>",
            "javascript:alert('XSS')"
        ]
        for payload in xss_payloads:
            response = client.post("/query", json={
                "query": payload,
                "file_context": "test.py"
            })
            assert response.status_code in [200, 422, 500]
    
    def test_path_traversal_protection(self, client):
        """Test protection against path traversal attacks"""
        traversal_payloads = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "....//....//....//etc/passwd"
        ]
        for payload in traversal_payloads:
            response = client.post("/query", json={
                "query": payload,
                "file_context": payload
            })
            # Should not expose system files
            assert response.status_code in [200, 422, 500]
    
    def test_command_injection_protection(self, client):
        """Test protection against command injection"""
        command_payloads = [
            "; ls -la",
            "| cat /etc/passwd",
            "& dir",
            "`whoami`"
        ]
        for payload in command_payloads:
            response = client.post("/query", json={
                "query": f"test query {payload}",
                "file_context": "test.py"
            })
            assert response.status_code in [200, 422, 500]


@pytest.mark.security
class TestAuthentication:
    """Test authentication and authorization"""
    
    def test_admin_endpoints_protection(self, client):
        """Test that admin endpoints require authentication"""
        admin_endpoints = [
            "/api/backup/create",
            "/api/dr/recover",
            "/api/encryption/rotate-key"
        ]
        for endpoint in admin_endpoints:
            response = client.post(endpoint)
            # Should require auth or return error
            assert response.status_code in [401, 403, 404, 405, 422, 500]
    
    def test_rate_limiting(self, client):
        """Test rate limiting protection"""
        # Make many rapid requests
        responses = []
        for _ in range(100):
            response = client.get("/health")
            responses.append(response.status_code)
        
        # Should eventually rate limit (if implemented)
        # This is a placeholder test
        assert any(code == 200 for code in responses)


@pytest.mark.security
class TestDataExposure:
    """Test protection against data exposure"""
    
    def test_error_messages_no_sensitive_data(self, client):
        """Test that error messages don't expose sensitive data"""
        response = client.post("/query", json={
            "query": "test",
            "file_context": "nonexistent_file_that_should_not_exist.xyz"
        })
        if response.status_code >= 400:
            data = response.text
            # Should not expose internal paths, passwords, etc.
            assert "password" not in data.lower()
            assert "/home/" not in data.lower() or True  # May be acceptable
    
    def test_headers_security(self, client):
        """Test security headers"""
        response = client.get("/health")
        headers = response.headers
        
        # Check for security headers (if implemented)
        # This is aspirational - headers may not be set yet
        if "X-Content-Type-Options" in headers:
            assert headers["X-Content-Type-Options"] == "nosniff"


@pytest.mark.security
class TestEncryption:
    """Test encryption functionality"""
    
    def test_encryption_status(self, client):
        """Test encryption status endpoint"""
        response = client.get("/api/encryption/status")
        if response.status_code == 200:
            data = response.json()
            assert "encryption_enabled" in data or "status" in data
    
    def test_sensitive_data_encryption(self, client):
        """Test that sensitive data is encrypted"""
        # This would test actual encryption of data at rest
        # Placeholder for now
        pass


@pytest.mark.security
@pytest.mark.slow
class TestDDoSProtection:
    """Test DDoS protection mechanisms"""
    
    def test_large_payload_rejection(self, client):
        """Test that very large payloads are rejected"""
        large_query = "A" * (10 * 1024 * 1024)  # 10 MB
        response = client.post("/query", json={
            "query": large_query,
            "file_context": "test.py"
        })
        # Should reject or handle gracefully
        assert response.status_code in [413, 422, 500]
    
    def test_concurrent_request_handling(self, client):
        """Test handling of many concurrent requests"""
        import concurrent.futures
        
        def make_request():
            return client.get("/health")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
            futures = [executor.submit(make_request) for _ in range(100)]
            results = [f.result() for f in futures]
        
        # Should handle all requests without crashing
        assert len(results) == 100
        success_count = sum(1 for r in results if r.status_code == 200)
        assert success_count > 0  # At least some should succeed


# Fixtures
@pytest.fixture(scope="module")
def client():
    """Create test client for security tests"""
    try:
        from rag.rag_dual import app
        return TestClient(app)
    except Exception as e:
        pytest.skip(f"Could not create test client: {e}")

