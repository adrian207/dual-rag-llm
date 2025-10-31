# Testing Guide

**Comprehensive testing framework for Dual RAG LLM System**

Author: Adrian Johnson <adrian207@gmail.com>

---

## Table of Contents

1. [Overview](#overview)
2. [Test Types](#test-types)
3. [Running Tests](#running-tests)
4. [Writing Tests](#writing-tests)
5. [Coverage Reports](#coverage-reports)
6. [CI/CD Integration](#cicd-integration)
7. [Best Practices](#best-practices)

---

## Overview

The Dual RAG LLM testing framework provides comprehensive quality assurance with **6 test types**:

1. **Unit Tests** - Individual component testing (80%+ coverage target)
2. **Integration Tests** - End-to-end API testing
3. **Load Tests** - Performance and stress testing with Locust
4. **Security Tests** - Vulnerability and penetration testing
5. **Performance Tests** - Regression testing and benchmarking
6. **Chaos Tests** - Resilience testing with Chaos Mesh

### Test Coverage Goals

| Component | Target Coverage | Current Status |
|-----------|----------------|----------------|
| **Database Module** | 85% | ✅ Implemented |
| **Backup Module** | 85% | ✅ Implemented |
| **API Endpoints** | 80% | ✅ Implemented |
| **Disaster Recovery** | 80% | ✅ Implemented |
| **Overall** | **80%+** | **Target** |

---

## Test Types

### 1. Unit Tests

**Purpose:** Test individual functions and classes in isolation.

**Location:** `tests/unit/`

**Technologies:** pytest, pytest-asyncio, unittest.mock

**Run Command:**
```bash
pytest tests/unit/ -v --cov=rag --cov-report=html
```

**What's Tested:**
- Database operations (insert, search, delete)
- Backup creation and restoration
- Configuration validation
- Data models
- Helper functions

**Example Test:**
```python
@pytest.mark.unit
@pytest.mark.asyncio
async def test_database_health_check(mock_database):
    health = await mock_database.health_check()
    assert health["healthy"] is True
```

---

### 2. Integration Tests

**Purpose:** Test API endpoints and service interactions.

**Location:** `tests/integration/`

**Technologies:** FastAPI TestClient, httpx

**Run Command:**
```bash
pytest tests/integration/ -v
```

**What's Tested:**
- `/query` endpoint
- `/health` endpoint
- `/api/analytics/*` endpoints
- `/api/cost/*` endpoints
- `/api/backup/*` endpoints
- `/api/dr/*` endpoints

**Services Required:**
- PostgreSQL (via Docker)
- Redis (via Docker)

**Example Test:**
```python
@pytest.mark.integration
def test_query_endpoint(client):
    response = client.post("/query", json={
        "query": "How do I use FastAPI?",
        "file_context": "app.py"
    })
    assert response.status_code == 200
```

---

### 3. Load Tests

**Purpose:** Test system performance under load.

**Location:** `tests/load/locustfile.py`

**Technologies:** Locust

**Run Command:**
```bash
# Web UI mode
locust -f tests/load/locustfile.py --host=http://localhost:8000

# Headless mode
locust -f tests/load/locustfile.py \
  --host=http://localhost:8000 \
  --headless \
  --users 100 \
  --spawn-rate 10 \
  --run-time 5m
```

**Test Scenarios:**
- **DualRAGUser** - Normal user behavior (10 tasks)
- **AdminUser** - Admin operations (5 tasks)
- **StressTestUser** - Heavy load testing

**Metrics Tracked:**
- Requests per second (RPS)
- Response time percentiles (P50, P95, P99)
- Error rates
- Concurrent users

**Example Results:**
```
Type     Name                                  # reqs    # fails  Avg   Min   Max  Median  req/s
--------|-------------------------------------|--------|---------|-----|-----|-----|-------|------
POST     /query                                 1000      0     245   120   890    210    20.0
GET      /health                                5000      0      12     5    45     10   100.0
```

---

### 4. Security Tests

**Purpose:** Test for common vulnerabilities.

**Location:** `tests/security/`

**Technologies:** pytest, OWASP testing patterns

**Run Command:**
```bash
pytest tests/security/ -v -m security
```

**What's Tested:**
- ✅ SQL Injection protection
- ✅ XSS (Cross-Site Scripting) protection
- ✅ Path Traversal protection
- ✅ Command Injection protection
- ✅ Rate Limiting
- ✅ Authentication/Authorization
- ✅ Data Exposure prevention
- ✅ DDoS protection

**Example Test:**
```python
@pytest.mark.security
def test_sql_injection_protection(client, malicious_inputs):
    for payload in malicious_inputs:
        response = client.post("/query", json={
            "query": payload,
            "file_context": "test.py"
        })
        assert "DROP TABLE" not in str(response.json()).upper()
```

**Additional Security Tools:**
```bash
# Check dependencies for vulnerabilities
safety check

# Run Bandit security linter
bandit -r rag/ -f json

# OWASP ZAP scan (requires ZAP installation)
zap-cli quick-scan http://localhost:8000
```

---

### 5. Performance Tests

**Purpose:** Benchmark performance and detect regressions.

**Location:** `tests/performance/`

**Technologies:** pytest-benchmark, tracemalloc

**Run Command:**
```bash
pytest tests/performance/ -v --benchmark-only
```

**What's Tested:**
- Query response times
- Database operation speed
- Cache hit/miss performance
- Concurrent request handling
- Memory usage patterns

**Baseline Metrics:**
| Operation | Target | P50 | P95 | P99 |
|-----------|--------|-----|-----|-----|
| Query | < 500ms | 250ms | 800ms | 2s |
| Health Check | < 50ms | 12ms | 25ms | 45ms |
| Cache Hit | < 10ms | 2ms | 5ms | 10ms |
| DB Search | < 1s | 450ms | 900ms | 1.5s |

**Example Test:**
```python
@pytest.mark.performance
def test_query_performance(client, benchmark):
    result = benchmark(lambda: client.post("/query", json={
        "query": "Test",
        "file_context": "test.py"
    }))
    assert benchmark.stats['mean'] < 0.5  # < 500ms
```

---

### 6. Chaos Engineering Tests

**Purpose:** Test system resilience under failure conditions.

**Location:** `tests/chaos/`

**Technologies:** Chaos Mesh (Kubernetes)

**Prerequisites:**
```bash
# Install Chaos Mesh
kubectl create ns chaos-mesh
helm repo add chaos-mesh https://charts.chaos-mesh.org
helm install chaos-mesh chaos-mesh/chaos-mesh -n chaos-mesh
```

**Run Command:**
```bash
# Apply pod failure chaos
kubectl apply -f tests/chaos/pod-failure.yaml

# Apply network latency
kubectl apply -f tests/chaos/network-latency.yaml

# Apply stress chaos
kubectl apply -f tests/chaos/stress-chaos.yaml

# Monitor experiments
kubectl get podchaos -n default
```

**Test Scenarios:**
1. **Pod Failure** - Random pod termination every 5 minutes
2. **Network Latency** - 100ms latency every 10 minutes
3. **Stress CPU/Memory** - Resource exhaustion every 15 minutes

**Expected Outcomes:**
- ✅ System recovers automatically within RTO (15 minutes)
- ✅ No data loss
- ✅ Graceful degradation
- ✅ Monitoring alerts triggered

**Cleanup:**
```bash
kubectl delete -f tests/chaos/
```

---

## Running Tests

### Quick Start

```bash
# Install test dependencies
pip install -r rag/requirements.txt
pip install -r requirements-test.txt

# Run all tests
pytest

# Run with coverage
pytest --cov=rag --cov-report=html

# Run specific test types
pytest tests/unit/         # Unit tests only
pytest tests/integration/  # Integration tests only
pytest -m security         # Security tests only
pytest -m performance      # Performance tests only
```

### Test Markers

```bash
# Run by marker
pytest -m unit             # Unit tests
pytest -m integration      # Integration tests
pytest -m slow             # Slow tests (> 1s)
pytest -m database         # Database tests
pytest -m cache            # Cache tests
pytest -m security         # Security tests
pytest -m performance      # Performance tests
pytest -m chaos            # Chaos tests
pytest -m smoke            # Smoke tests for CI/CD

# Exclude markers
pytest -m "not slow"       # Skip slow tests
pytest -m "not chaos"      # Skip chaos tests
```

### Docker Testing Environment

```bash
# Start test services
docker-compose -f docker-compose.test.yml up -d

# Run tests
pytest

# Stop services
docker-compose -f docker-compose.test.yml down
```

---

## Writing Tests

### Test Structure

```python
"""
Test Module Name
Brief description

Author: Adrian Johnson <adrian207@gmail.com>
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch

@pytest.mark.unit  # or integration, security, performance
class TestFeatureName:
    """Test suite for specific feature"""
    
    def test_basic_functionality(self):
        """Test basic behavior"""
        # Arrange
        input_data = "test"
        
        # Act
        result = function_under_test(input_data)
        
        # Assert
        assert result == expected_output
    
    @pytest.mark.asyncio
    async def test_async_functionality(self, mock_dependency):
        """Test async behavior"""
        result = await async_function(mock_dependency)
        assert result is not None
    
    @pytest.mark.slow
    def test_expensive_operation(self):
        """Test that takes > 1 second"""
        pass
```

### Using Fixtures

```python
@pytest.fixture
def sample_data():
    """Provide sample data for tests"""
    return {"id": 1, "name": "test"}

def test_with_fixture(sample_data):
    """Use fixture in test"""
    assert sample_data["id"] == 1
```

### Mocking

```python
from unittest.mock import AsyncMock, patch

@pytest.mark.asyncio
async def test_with_mock():
    """Test with mocked dependency"""
    mock_db = AsyncMock()
    mock_db.query.return_value = [{"result": "test"}]
    
    result = await function_that_uses_db(mock_db)
    
    mock_db.query.assert_called_once()
    assert result is not None
```

---

## Coverage Reports

### Generating Coverage

```bash
# Generate HTML coverage report
pytest --cov=rag --cov-report=html

# Open report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
start htmlcov/index.html  # Windows

# Generate XML for CI/CD
pytest --cov=rag --cov-report=xml

# Terminal report
pytest --cov=rag --cov-report=term-missing
```

### Coverage Requirements

```ini
# pytest.ini
[pytest]
addopts = --cov-fail-under=80
```

If coverage falls below 80%, tests will fail.

### Viewing Coverage

```bash
Name                    Stmts   Miss  Cover   Missing
-----------------------------------------------------
rag/__init__.py            2      0   100%
rag/database.py          397     40    90%   145-152, 200-205
rag/backup.py            520     52    90%   300-310, 450-460
rag/rag_dual.py          800    100    87%   500-550, 700-720
-----------------------------------------------------
TOTAL                   1719    192    89%
```

---

## CI/CD Integration

### GitHub Actions

Tests run automatically on:
- ✅ Push to `main` or `develop` branches
- ✅ Pull requests
- ✅ Daily schedule (cron: '0 0 * * *')

**Workflow:** `.github/workflows/tests.yml`

**Jobs:**
1. **unit-tests** - Python 3.11 and 3.12
2. **integration-tests** - With PostgreSQL and Redis services
3. **security-tests** - Safety, Bandit, security test suite
4. **performance-tests** - Benchmarking with result tracking
5. **docker-build** - Docker image build verification
6. **lint** - flake8, black, isort, mypy

**View Results:**
```
GitHub → Actions → Test Suite → Latest run
```

### Running Tests Locally (CI Simulation)

```bash
# Run all CI checks locally
./scripts/run-ci-checks.sh

# Or manually:
pytest tests/unit/ --cov=rag
pytest tests/integration/
pytest tests/security/ -m security
safety check
bandit -r rag/
flake8 rag/
black --check rag/
```

---

## Best Practices

### ✅ DO

1. **Write tests first** (TDD approach)
2. **Use descriptive test names** - `test_user_registration_with_invalid_email`
3. **Follow AAA pattern** - Arrange, Act, Assert
4. **Mock external dependencies** - Don't call real APIs in unit tests
5. **Use fixtures** for common setup
6. **Tag tests appropriately** - `@pytest.mark.unit`, `@pytest.mark.slow`
7. **Test edge cases** - Empty strings, None values, large inputs
8. **Keep tests independent** - No shared state between tests
9. **Run tests frequently** - Before every commit
10. **Maintain 80%+ coverage** - Coverage != quality, but it helps

### ❌ DON'T

1. **Don't skip tests** - Fix failing tests, don't skip them
2. **Don't test implementation details** - Test behavior, not internals
3. **Don't have flaky tests** - Tests should be deterministic
4. **Don't have slow unit tests** - Unit tests should be fast (< 1s)
5. **Don't ignore security tests** - Security is not optional
6. **Don't commit without running tests** - CI is not a substitute
7. **Don't write tests without assertions** - Tests must verify something
8. **Don't mix test types** - Keep unit/integration/performance separate

### Test Naming Convention

```python
# Good
def test_user_login_with_valid_credentials():
    pass

def test_user_login_with_invalid_password_returns_401():
    pass

def test_backup_creation_generates_checksum():
    pass

# Bad
def test_1():
    pass

def test_user():
    pass

def test():
    pass
```

---

## Troubleshooting

### Common Issues

**1. Tests fail with "ModuleNotFoundError"**
```bash
# Install test dependencies
pip install -r requirements-test.txt
```

**2. Integration tests fail with "Connection refused"**
```bash
# Start required services
docker-compose -f docker-compose.test.yml up -d
```

**3. Async tests don't run**
```bash
# Install pytest-asyncio
pip install pytest-asyncio
```

**4. Coverage below 80%**
```bash
# Generate detailed report
pytest --cov=rag --cov-report=term-missing
# Add tests for missing lines
```

**5. Chaos tests don't work**
```bash
# Install Chaos Mesh first
kubectl apply -f https://mirrors.chaos-mesh.org/latest/crd.yaml
helm install chaos-mesh chaos-mesh/chaos-mesh -n chaos-mesh
```

---

## Test Metrics

### Current Status (v1.21.0)

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| **Total Tests** | 150+ | 100+ | ✅ |
| **Unit Test Coverage** | 85% | 80%+ | ✅ |
| **Integration Tests** | 40+ | 30+ | ✅ |
| **Security Tests** | 20+ | 15+ | ✅ |
| **Performance Tests** | 15+ | 10+ | ✅ |
| **Chaos Tests** | 3 scenarios | 3+ | ✅ |
| **CI/CD Jobs** | 6 jobs | 5+ | ✅ |
| **Test Execution Time** | < 5 min | < 10 min | ✅ |

---

## Resources

### Documentation
- [pytest Documentation](https://docs.pytest.org/)
- [FastAPI Testing](https://fastapi.tiangolo.com/tutorial/testing/)
- [Locust Documentation](https://docs.locust.io/)
- [Chaos Mesh Documentation](https://chaos-mesh.org/docs/)

### Tools
- **pytest** - Test framework
- **pytest-cov** - Coverage plugin
- **pytest-asyncio** - Async test support
- **pytest-benchmark** - Benchmarking plugin
- **Locust** - Load testing
- **Chaos Mesh** - Chaos engineering
- **Safety** - Dependency vulnerability scanner
- **Bandit** - Security linter

---

## Support

For testing questions or issues:
- **📧 Email:** adrian207@gmail.com
- **🐛 Issues:** [GitHub Issues](https://github.com/adrian207/dual-rag-llm/issues)
- **📖 Docs:** [Testing Guide](docs/TESTING.md)

---

**Version:** 1.21.0  
**Last Updated:** October 31, 2024  
**Author:** Adrian Johnson <adrian207@gmail.com>

