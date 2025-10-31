# Dual-Branch Development Workflow

**Author:** Adrian Johnson <adrian207@gmail.com>

This document outlines the workflow for maintaining both `main` and `azure/deployment` branches.

## Branch Strategy

```
main (v1.x.x)
├── Core application code
├── Docker Compose for local/VM deployment
├── Documentation
└── Universal features
    │
    ├─→ Merge to azure/deployment
    │
azure/deployment (v1.x.x-azure)
├── All code from main
├── Azure-specific configurations
├── Kubernetes manifests
├── Azure deployment scripts
└── Azure documentation
```

## Development Workflow

### Feature Development Process

#### Step 1: Develop on Main Branch

```bash
# Always start on main for core features
git checkout main
git pull origin main

# Create feature branch
git checkout -b feature/redis-cache

# Develop and test
# ... make changes ...

# Commit
git add -A
git commit -m "feat: add Redis caching layer"

# Push and create PR
git push origin feature/redis-cache
```

#### Step 2: Merge to Main

```bash
# After PR approval
git checkout main
git merge feature/redis-cache
git push origin main
```

#### Step 3: Merge to Azure Branch

```bash
# Switch to Azure branch
git checkout azure/deployment
git pull origin azure/deployment

# Merge main into Azure branch
git merge main

# Resolve conflicts if any (usually none for core features)
git push origin azure/deployment
```

#### Step 4: Add Azure-Specific Changes (if needed)

```bash
# Still on azure/deployment branch
# Add Kubernetes configs, Azure scripts, etc.

# For example, update Kubernetes manifests for new Redis service
vim azure/k8s/redis-deployment.yaml

git add azure/
git commit -m "feat(azure): add Redis deployment for AKS"
git push origin azure/deployment
```

## What Goes Where?

### Always on MAIN Branch
- ✅ Core application code (`rag/rag_dual.py`, `rag/ingest_docs.py`, etc.)
- ✅ Docker Compose files
- ✅ Python dependencies (`requirements.txt`)
- ✅ FastAPI endpoints and logic
- ✅ Documentation (README, CONTRIBUTING, etc.)
- ✅ Base Dockerfile
- ✅ Helper scripts that work everywhere

### Only on AZURE Branch
- ✅ Kubernetes manifests (`azure/k8s/*.yaml`)
- ✅ Azure deployment scripts (`azure/scripts/*.sh`)
- ✅ Azure-specific documentation (`azure/*.md`)
- ✅ AKS configurations
- ✅ Azure Resource Manager templates
- ✅ Azure DevOps pipelines (if added)

### On BOTH Branches (via merge)
- ✅ Core features (Redis, monitoring, etc.)
- ✅ Bug fixes
- ✅ Performance improvements
- ✅ Security updates
- ✅ API changes
- ✅ Documentation updates (general)

## Automatic Merge Strategy

### Option 1: Manual Merge (Current)

After each main update:
```bash
git checkout azure/deployment
git merge main
git push origin azure/deployment
```

### Option 2: GitHub Actions Auto-Merge (Future)

Create `.github/workflows/sync-azure-branch.yml`:
```yaml
name: Sync Azure Branch
on:
  push:
    branches: [main]
jobs:
  sync:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
      - name: Merge main to azure/deployment
        run: |
          git checkout azure/deployment
          git merge main
          git push origin azure/deployment
```

## Version Numbering

### Main Branch
```
v1.0.0 - Initial release
v1.1.0 - Redis caching
v1.2.0 - Advanced RAG
v2.0.0 - Breaking changes
```

### Azure Branch
```
v1.0.0-azure - Initial Azure support
v1.1.0-azure - Redis + Azure updates
```

**Both share the same major.minor version, Azure adds suffix**

## Example: Adding Redis Caching

### On Main Branch

```bash
git checkout main
git checkout -b feature/redis-cache

# 1. Update docker-compose.yml
# Add Redis service

# 2. Update rag/requirements.txt
# Add redis, aioredis

# 3. Update rag/rag_dual.py
# Add caching logic

# 4. Test locally
docker compose up -d
make test

# 5. Commit and merge
git commit -m "feat: add Redis caching layer"
git checkout main
git merge feature/redis-cache
git push origin main
```

### On Azure Branch

```bash
git checkout azure/deployment
git merge main  # Get all main changes

# Add Azure-specific configs
cat > azure/k8s/redis-deployment.yaml << EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: redis
  namespace: dual-rag
spec:
  # ... Redis config for AKS
EOF

git add azure/k8s/redis-deployment.yaml
git commit -m "feat(azure): add Redis deployment for AKS"
git push origin azure/deployment
```

## Testing Strategy

### Test on Main
- Local Docker Compose
- VM deployment
- Unit tests
- Integration tests

### Test on Azure Branch
- AKS deployment
- Azure VM deployment
- Load testing
- Cost validation

## Release Process

### For Both Branches

```bash
# 1. Update version on main
echo "1.1.0" > VERSION
git commit -m "chore: bump version to 1.1.0"
git tag -a v1.1.0 -m "Release v1.1.0"
git push origin main v1.1.0

# 2. Merge to Azure and tag
git checkout azure/deployment
git merge main
git tag -a v1.1.0-azure -m "Release v1.1.0 for Azure"
git push origin azure/deployment v1.1.0-azure
```

## Conflict Resolution

### Common Conflicts

1. **docker-compose.yml** - Azure may have additional services
   - Keep both changes
   - Azure version can be more comprehensive

2. **README.md** - Azure section in main
   - Keep both
   - Azure branch has more detailed Azure docs

3. **File locations** - Azure has `azure/` directory
   - No conflict, Azure files are separate

### Resolution Strategy

```bash
# When conflicts occur during merge
git checkout azure/deployment
git merge main

# If conflicts:
git status
# Edit conflicting files
git add <resolved-files>
git commit -m "merge: resolve conflicts from main"
git push origin azure/deployment
```

## Communication

### Commit Message Prefixes

- `feat:` - Feature for both branches
- `feat(azure):` - Azure-specific feature
- `fix:` - Bug fix for both
- `fix(azure):` - Azure-specific fix
- `docs:` - Documentation (both)
- `docs(azure):` - Azure documentation

### PR Labels

- `enhancement` - New feature
- `bug` - Bug fix
- `azure` - Azure-specific
- `core` - Core application
- `documentation` - Docs only

## Quick Reference

```bash
# Start new feature
git checkout main
git checkout -b feature/name

# Develop and test
# ...

# Merge to main
git checkout main
git merge feature/name
git push origin main

# Sync to Azure
git checkout azure/deployment
git merge main
git push origin azure/deployment

# Add Azure-specific changes if needed
# Edit azure/ files
git commit -m "feat(azure): description"
git push origin azure/deployment

# Release
git checkout main
git tag -a vX.Y.Z -m "Release X.Y.Z"
git push origin main vX.Y.Z

git checkout azure/deployment
git tag -a vX.Y.Z-azure -m "Release X.Y.Z for Azure"
git push origin azure/deployment vX.Y.Z-azure
```

---

**Remember**: Develop features on `main`, merge to `azure/deployment`, add Azure-specific configs on `azure/deployment`.

