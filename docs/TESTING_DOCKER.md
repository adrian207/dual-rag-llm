# Docker Testing Environment

**Run comprehensive tests locally using Docker**

Author: Adrian Johnson <adrian207@gmail.com>

---

## Quick Start

### Prerequisites

- Docker Desktop installed and running
- Docker Compose v1.27+

### Run All Tests

**Linux/Mac:**
```bash
chmod +x scripts/run-tests.sh
./scripts/run-tests.sh all
```

**Windows (PowerShell):**
```powershell
.\scripts\run-tests.ps1 all
```

---

## Test Environment Architecture

```
┌─────────────────────────────────────────────┐
│         Docker Test Environment             │
├─────────────────────────────────────────────┤
│                                             │
│  ┌──────────────┐  ┌──────────────┐       │
│  │  PostgreSQL  │  │    Redis     │       │
│  │  (pgvector)  │  │   (cache)    │       │
│  │  Port: 5433  │  │  Port: 6380  │       │
│  └──────────────┘  └──────────────┘       │
│         ↓                  ↓                │
│  ┌─────────────────────────────────────┐  │
│  │       Test Runner Container         │  │
│  │  - Python 3.11                      │  │
│  │  - pytest + all dependencies        │  │
│  │  - Your code mounted                │  │
│  │  - Runs test suite                  │  │
│  └─────────────────────────────────────┘  │
│                                             │
└─────────────────────────────────────────────┘
```

---

## Available Commands

### Run Specific Test Types

```bash
# Unit tests only
./scripts/run-tests.sh unit

# Integration tests
./scripts/run-tests.sh integration

# Security tests
./scripts/run-tests.sh security

# Performance benchmarks
./scripts/run-tests.sh performance

# Quick smoke tests
./scripts/run-tests.sh smoke

# Coverage report
./scripts/run-tests.sh coverage
```

### Manual Docker Commands

**Start services:**
```bash
docker-compose -f docker-compose.test.yml up -d postgres redis
```

**Run tests:**
```bash
docker-compose -f docker-compose.test.yml run --rm test-runner pytest tests/ -v
```

**Interactive shell:**
```bash
./scripts/run-tests.sh shell
# Or:
docker-compose -f docker-compose.test.yml run --rm test-runner bash
```

**View coverage report:**
```bash
docker-compose -f docker-compose.test.yml run --rm test-runner \
  python -m http.server --directory htmlcov 8080

# Then open: http://localhost:8080
```

**Cleanup:**
```bash
./scripts/run-tests.sh clean
# Or:
docker-compose -f docker-compose.test.yml down -v
```

---

## Test Services

### PostgreSQL (pgvector)

- **Image:** `pgvector/pgvector:pg15`
- **Port:** 5433 (host) → 5432 (container)
- **Database:** `test_dual_rag`
- **User:** `testuser`
- **Password:** `testpass`

**Connect directly:**
```bash
docker-compose -f docker-compose.test.yml exec postgres \
  psql -U testuser -d test_dual_rag
```

### Redis

- **Image:** `redis:7-alpine`
- **Port:** 6380 (host) → 6379 (container)

**Connect directly:**
```bash
docker-compose -f docker-compose.test.yml exec redis redis-cli
```

---

## Environment Variables

The test runner sets these automatically:

```bash
TESTING=true
LOG_LEVEL=DEBUG

# PostgreSQL
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_DB=test_dual_rag
POSTGRES_USER=testuser
POSTGRES_PASSWORD=testpass

# Redis
REDIS_HOST=redis
REDIS_PORT=6379
```

---

## Coverage Reports

### Generate HTML Coverage

```bash
./scripts/run-tests.sh coverage
```

The report will be available in `htmlcov/index.html`.

### View Coverage in Container

```bash
docker-compose -f docker-compose.test.yml run --rm test-runner \
  pytest tests/ --cov=rag --cov-report=html

# Serve the report
docker-compose -f docker-compose.test.yml run --rm -p 8080:8080 test-runner \
  python -m http.server --directory htmlcov 8080
```

Open http://localhost:8080 in your browser.

---

## Troubleshooting

### Services Won't Start

**Check Docker is running:**
```bash
docker ps
```

**Check logs:**
```bash
docker-compose -f docker-compose.test.yml logs postgres
docker-compose -f docker-compose.test.yml logs redis
```

**Reset environment:**
```bash
docker-compose -f docker-compose.test.yml down -v
docker-compose -f docker-compose.test.yml up -d
```

### Tests Fail with Connection Errors

**Wait for services to be healthy:**
```bash
docker-compose -f docker-compose.test.yml up -d
sleep 10  # Wait for health checks
./scripts/run-tests.sh unit
```

**Check service health:**
```bash
docker-compose -f docker-compose.test.yml ps
```

All services should show `healthy` status.

### Permission Errors

**Linux/Mac - Make scripts executable:**
```bash
chmod +x scripts/run-tests.sh
```

**Windows - Run PowerShell as Administrator:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Port Conflicts

If ports 5433 or 6380 are in use, edit `docker-compose.test.yml`:

```yaml
postgres:
  ports:
    - "5434:5432"  # Change 5433 to 5434

redis:
  ports:
    - "6381:6379"  # Change 6380 to 6381
```

---

## CI/CD Integration

The same Docker environment is used in GitHub Actions:

```yaml
# .github/workflows/tests.yml
services:
  postgres:
    image: pgvector/pgvector:pg15
    env:
      POSTGRES_PASSWORD: postgres
    options: >-
      --health-cmd pg_isready
      --health-interval 10s
```

This ensures **test parity** between local and CI environments.

---

## Performance Tips

### Speed Up Rebuilds

**Use BuildKit:**
```bash
DOCKER_BUILDKIT=1 docker-compose -f docker-compose.test.yml build
```

**Cache Python packages:**
```bash
# Packages are cached in the Docker image
# Rebuilds only when requirements change
```

### Parallel Testing

```bash
# Run tests in parallel (requires pytest-xdist)
docker-compose -f docker-compose.test.yml run --rm test-runner \
  pytest tests/ -n auto
```

### Faster Test Iterations

**Keep services running:**
```bash
# Start once
docker-compose -f docker-compose.test.yml up -d postgres redis

# Run tests multiple times (faster)
docker-compose -f docker-compose.test.yml run --rm test-runner pytest tests/unit/
docker-compose -f docker-compose.test.yml run --rm test-runner pytest tests/integration/

# Cleanup when done
docker-compose -f docker-compose.test.yml down
```

---

## Advanced Usage

### Custom Test Commands

```bash
# Run specific test file
docker-compose -f docker-compose.test.yml run --rm test-runner \
  pytest tests/unit/test_database.py -v

# Run with debugging
docker-compose -f docker-compose.test.yml run --rm test-runner \
  pytest tests/ -v --pdb

# Run with markers
docker-compose -f docker-compose.test.yml run --rm test-runner \
  pytest tests/ -m "unit and not slow"

# Generate JUnit XML
docker-compose -f docker-compose.test.yml run --rm test-runner \
  pytest tests/ --junitxml=test-results.xml
```

### Debugging Tests

**Interactive Python shell:**
```bash
docker-compose -f docker-compose.test.yml run --rm test-runner python

>>> from rag.database import VectorDatabase
>>> # Test your code interactively
```

**Test with breakpoints:**
```python
# Add to your test
import pdb; pdb.set_trace()
```

```bash
docker-compose -f docker-compose.test.yml run --rm test-runner \
  pytest tests/unit/test_database.py -s
```

### Load Testing with Locust

```bash
# Start Locust web UI
docker-compose -f docker-compose.test.yml run --rm -p 8089:8089 test-runner \
  locust -f tests/load/locustfile.py --host=http://localhost:8000

# Open http://localhost:8089
```

---

## Best Practices

1. **Always use the Docker environment for pre-commit testing**
   ```bash
   ./scripts/run-tests.sh unit
   git commit -m "feat: add new feature"
   ```

2. **Run full test suite before merging PRs**
   ```bash
   ./scripts/run-tests.sh all
   ```

3. **Generate coverage reports regularly**
   ```bash
   ./scripts/run-tests.sh coverage
   ```

4. **Clean up after testing**
   ```bash
   ./scripts/run-tests.sh clean
   ```

5. **Use markers to run relevant tests**
   ```bash
   # When working on database code
   docker-compose -f docker-compose.test.yml run --rm test-runner \
     pytest -m database -v
   ```

---

## Comparison: Local vs Docker vs CI

| Feature | Local | Docker | CI (GitHub Actions) |
|---------|-------|--------|---------------------|
| **Setup Time** | Hours | Minutes | Automatic |
| **Consistency** | ❌ Varies | ✅ Same | ✅ Same |
| **Dependencies** | Manual | Automatic | Automatic |
| **PostgreSQL** | Complex | ✅ Easy | ✅ Easy |
| **Redis** | Complex | ✅ Easy | ✅ Easy |
| **asyncpg** | Windows issues | ✅ Works | ✅ Works |
| **Isolation** | ❌ Shared | ✅ Isolated | ✅ Isolated |
| **Cleanup** | Manual | ✅ Auto | ✅ Auto |

**Recommendation:** Use Docker for local testing! 🐳

---

## Support

For issues with the Docker test environment:

- **📖 Main Testing Guide:** [docs/TESTING.md](TESTING.md)
- **🐛 Report Issues:** [GitHub Issues](https://github.com/adrian207/dual-rag-llm/issues)
- **📧 Email:** adrian207@gmail.com

---

**Version:** 1.21.0  
**Last Updated:** October 31, 2024

