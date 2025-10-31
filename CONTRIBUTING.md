# Contributing to Dual RAG LLM System

**Author:** Adrian Johnson <adrian207@gmail.com>

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

---

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [How to Contribute](#how-to-contribute)
3. [Development Setup](#development-setup)
4. [Coding Standards](#coding-standards)
5. [Testing Guidelines](#testing-guidelines)
6. [Documentation](#documentation)
7. [Pull Request Process](#pull-request-process)
8. [Release Process](#release-process)

---

## Code of Conduct

###  Our Pledge

We are committed to providing a welcoming and inclusive environment for all contributors, regardless of:
- Experience level
- Gender identity
- Sexual orientation
- Disability
- Personal appearance
- Race or ethnicity
- Age
- Religion

### Our Standards

**Examples of positive behavior:**
✅ Using welcoming and inclusive language  
✅ Being respectful of differing viewpoints  
✅ Gracefully accepting constructive criticism  
✅ Focusing on what is best for the community  
✅ Showing empathy towards others  

**Examples of unacceptable behavior:**
❌ Trolling, insulting, or derogatory comments  
❌ Public or private harassment  
❌ Publishing others' private information  
❌ Other conduct which could be considered inappropriate  

### Enforcement

Instances of abusive behavior may be reported to adrian207@gmail.com. All complaints will be reviewed and investigated promptly.

---

## How to Contribute

### Reporting Bugs

Before creating a bug report:
1. **Search existing issues** to avoid duplicates
2. **Verify** the bug exists in the latest version
3. **Collect** relevant information (logs, screenshots, environment)

**Bug Report Template:**
```markdown
## Bug Description
Clear description of the bug

## Steps to Reproduce
1. Step 1
2. Step 2
3. ...

## Expected Behavior
What you expected to happen

## Actual Behavior
What actually happened

## Environment
- OS: [e.g., Ubuntu 22.04]
- Docker version: [e.g., 24.0.5]
- GPU: [e.g., NVIDIA RTX 3090]
- Version: [e.g., 1.11.0]

## Logs
```
Relevant log output
```

## Additional Context
Screenshots, error messages, etc.
```

### Suggesting Features

Feature requests are welcome! Please provide:
1. **Clear use case**: Why is this feature needed?
2. **Proposed solution**: How should it work?
3. **Alternatives considered**: Other approaches you thought about
4. **Additional context**: Examples, mockups, references

**Feature Request Template:**
```markdown
## Feature Description
Clear description of the feature

## Problem It Solves
What problem does this address?

## Proposed Solution
How should this work?

## Alternatives Considered
Other approaches you considered

## Additional Context
Examples, mockups, references
```

### Improving Documentation

Documentation improvements are always appreciated:
- Fix typos or unclear wording
- Add examples
- Improve formatting
- Translate to other languages
- Add diagrams or screenshots

---

## Development Setup

### Prerequisites

- **Python 3.11+**
- **Docker & Docker Compose**
- **Git**
- **NVIDIA GPU** (optional but recommended)

### Fork & Clone

```bash
# Fork the repository on GitHub first, then:

git clone https://github.com/YOUR_USERNAME/dual-rag-llm.git
cd dual-rag-llm

# Add upstream remote
git remote add upstream https://github.com/adrian207/dual-rag-llm.git
```

### Local Development

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r rag/requirements.txt
pip install -r rag/requirements-dev.txt  # If available

# Start services
docker-compose up -d

# Run in development mode
cd rag
uvicorn rag_dual:app --reload --host 0.0.0.0 --port 8000
```

### Branch Strategy

- `main`: Production-ready code
- `azure/deployment`: Azure-specific branch
- `feature/*`: New features
- `bugfix/*`: Bug fixes
- `docs/*`: Documentation updates

**Create a feature branch:**
```bash
git checkout main
git pull upstream main
git checkout -b feature/your-feature-name
```

---

## Coding Standards

### Python Style Guide

Follow **PEP 8** with these specifics:

**Formatting:**
- **Line length**: 100 characters (not 79)
- **Indentation**: 4 spaces
- **Quotes**: Double quotes for strings
- **Imports**: Organized (stdlib, third-party, local)

**Example:**
```python
"""
Module docstring.

Author: Your Name <your.email@example.com>
"""

import os
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .utils import helper_function


class MyModel(BaseModel):
    """Model docstring."""
    
    name: str
    value: int
    optional_field: Optional[str] = None


async def my_function(param: str) -> dict:
    """
    Function docstring.
    
    Args:
        param: Description of param
        
    Returns:
        Dictionary with results
        
    Raises:
        HTTPException: If something goes wrong
    """
    if not param:
        raise HTTPException(status_code=400, detail="param required")
    
    result = await helper_function(param)
    return {"status": "success", "data": result}
```

### Type Hints

Use type hints for all function signatures:

```python
# Good
def process_query(query: str, max_results: int = 10) -> List[dict]:
    ...

# Bad
def process_query(query, max_results=10):
    ...
```

### Docstrings

Use **Google-style docstrings**:

```python
def complex_function(arg1: str, arg2: int, arg3: Optional[bool] = None) -> dict:
    """
    One-line summary.
    
    Longer description if needed. Can span multiple lines
    and include code examples.
    
    Args:
        arg1: Description of arg1
        arg2: Description of arg2
        arg3: Description of arg3. Defaults to None.
    
    Returns:
        Dictionary containing:
        - key1: Description
        - key2: Description
    
    Raises:
        ValueError: If arg2 is negative
        KeyError: If required key missing
        
    Example:
        >>> result = complex_function("test", 5)
        >>> print(result)
        {'key1': 'value1', 'key2': 'value2'}
    """
    pass
```

### Logging

Use **structlog** for structured logging:

```python
import structlog

logger = structlog.get_logger()

# Good
logger.info("query_processed", query_id=query_id, model=model, cached=cached)

# Bad
logger.info(f"Processed query {query_id} with model {model}")
```

### Error Handling

```python
# Good - Specific exceptions
try:
    result = await process_query(query)
except ValueError as e:
    logger.error("invalid_query", error=str(e))
    raise HTTPException(status_code=400, detail=str(e))
except Exception as e:
    logger.exception("unexpected_error")
    raise HTTPException(status_code=500, detail="Internal error")

# Bad - Bare except
try:
    result = await process_query(query)
except:
    pass
```

### Async Best Practices

```python
# Good - Proper async
async def fetch_data(url: str) -> dict:
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        return response.json()

# Bad - Blocking in async
async def fetch_data(url: str) -> dict:
    response = requests.get(url)  # Blocks event loop!
    return response.json()
```

---

## Testing Guidelines

### Test Structure

```
tests/
├── __init__.py
├── conftest.py           # Fixtures
├── test_api.py          # API tests
├── test_rag.py          # RAG functionality
├── test_tools.py        # Tool integrations
└── test_validation.py   # Validation/quality
```

### Writing Tests

```python
import pytest
from fastapi.testclient import TestClient

from rag.rag_dual import app


client = TestClient(app)


def test_health_check():
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] in ["healthy", "degraded"]


@pytest.mark.asyncio
async def test_query_endpoint():
    """Test query endpoint."""
    response = client.post("/query", json={
        "question": "What is Python?",
        "file_ext": ".py"
    })
    
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "metadata" in data
    assert len(data["answer"]) > 0


@pytest.fixture
def mock_ollama(monkeypatch):
    """Mock Ollama API calls."""
    async def mock_generate(*args, **kwargs):
        return {"response": "Mocked response"}
    
    monkeypatch.setattr("ollama.AsyncClient.generate", mock_generate)
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific file
pytest tests/test_api.py

# Run with coverage
pytest --cov=rag --cov-report=html

# Run only fast tests
pytest -m "not slow"
```

### Test Coverage

Aim for **80%+ coverage** for new code:

```bash
pytest --cov=rag --cov-report=term-missing
```

---

## Documentation

### Code Documentation

- **All public functions** must have docstrings
- **Complex logic** should have inline comments
- **Type hints** for all parameters and returns

### User Documentation

When adding features, update:
- `README.md` - Overview and quick start
- `docs/USER_GUIDE.md` - Detailed usage
- `docs/API_REFERENCE.md` - API documentation
- `CHANGELOG.md` - Release notes

### Documentation Format

Use **Markdown** with clear structure:

```markdown
# Title (H1 - once per document)

## Section (H2 - main sections)

### Subsection (H3 - details)

**Bold** for emphasis
*Italic* for terms
`code` for inline code

```python
# Code blocks with language
def example():
    pass
```

[Links](https://example.com)

- Bullet points
- For lists

1. Numbered lists
2. For steps

> Blockquotes for important notes

| Tables | For |
|--------|-----|
| Structured | Data |
```

---

## Pull Request Process

### Before Submitting

1. ✅ **Code compiles** without errors
2. ✅ **Tests pass** locally
3. ✅ **Documentation updated**
4. ✅ **CHANGELOG.md** updated (if user-facing change)
5. ✅ **Code follows** style guidelines
6. ✅ **Commits are** descriptive

### PR Title Format

```
<type>(<scope>): <description>

Examples:
feat(api): add new validation endpoint
fix(cache): resolve redis connection timeout
docs(readme): update installation instructions
refactor(rag): improve query processing performance
test(api): add integration tests for tools
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Formatting changes
- `refactor`: Code refactoring
- `test`: Adding tests
- `chore`: Maintenance tasks

### PR Description Template

```markdown
## Description
Clear description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
Describe testing done

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex code
- [ ] Documentation updated
- [ ] Tests added/updated
- [ ] All tests pass
- [ ] No new warnings
```

### Review Process

1. **Automated checks** must pass (if CI/CD configured)
2. **At least one review** required
3. **Address feedback** promptly
4. **Squash commits** if requested
5. **Maintainer merges** after approval

---

## Release Process

### Version Numbering

Follow **Semantic Versioning** (SemVer):

```
MAJOR.MINOR.PATCH

1.11.0
│  │  └─ Patch: Bug fixes
│  └──── Minor: New features (backward compatible)
└─────── Major: Breaking changes
```

### Release Steps

1. **Update version** in:
   - `rag/__init__.py`
   - `VERSION` file
   - `CHANGELOG.md`

2. **Create release branch:**
```bash
git checkout -b release/v1.12.0
```

3. **Update CHANGELOG.md:**
```markdown
## [1.12.0] - 2024-11-01

### Added
- New feature X
- New feature Y

### Changed
- Improved feature Z

### Fixed
- Bug fix A
```

4. **Commit and tag:**
```bash
git add -A
git commit -m "chore: release v1.12.0"
git tag -a v1.12.0 -m "Release v1.12.0: Description"
```

5. **Merge to main:**
```bash
git checkout main
git merge release/v1.12.0 --no-ff
git push origin main
git push origin v1.12.0
```

6. **Sync to azure/deployment:**
```bash
git checkout azure/deployment
git merge main -m "merge: sync v1.12.0 from main"
git push origin azure/deployment
```

7. **Create GitHub Release:**
   - Go to repository → Releases
   - Click "Draft a new release"
   - Select tag v1.12.0
   - Copy from CHANGELOG.md
   - Publish release

---

## Recognition

Contributors will be:
- Listed in `CONTRIBUTORS.md`
- Mentioned in release notes
- Credited in relevant documentation

---

## Questions?

- **GitHub Discussions**: (Coming soon)
- **GitHub Issues**: For bug reports/feature requests
- **Email**: adrian207@gmail.com

---

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for contributing!** 🎉

Your contributions help make this project better for everyone.

---

*Last updated: October 31, 2024*  
*Author: Adrian Johnson <adrian207@gmail.com>*
