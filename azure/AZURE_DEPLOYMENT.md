# Azure Deployment Guide - Dual RAG LLM System

**Author:** Adrian Johnson <adrian207@gmail.com>

This guide covers deploying the Dual RAG LLM system to Microsoft Azure with GPU support.

## Can This Run Natively in Azure?

**Yes, with caveats.** The system requires GPU compute for LLM inference, which limits deployment options:

### ✅ Recommended Azure Services

#### 1. **Azure Kubernetes Service (AKS) with GPU Nodes** ⭐ RECOMMENDED
- **Best for**: Production deployments, scalability, high availability
- **GPU Support**: Full NVIDIA GPU support (NC-series VMs)
- **Cost**: $$$$ (~$500-2000/month depending on GPU tier)
- **Complexity**: Medium-High
- **Advantages**:
  - Native Kubernetes orchestration
  - Auto-scaling capabilities
  - Load balancing built-in
  - High availability
  - Easy updates and rollbacks

#### 2. **Azure Container Instances (ACI) with GPU**
- **Best for**: Development, testing, simple deployments
- **GPU Support**: Limited (V100 GPUs available)
- **Cost**: $$$ (~$300-1000/month)
- **Complexity**: Low
- **Advantages**:
  - Simplest deployment
  - Pay-per-second billing
  - No cluster management
  - Fast startup

#### 3. **Azure Container Apps with GPU** (Preview)
- **Best for**: Serverless workloads
- **GPU Support**: Limited availability
- **Cost**: $$ (Pay for use)
- **Complexity**: Low-Medium
- **Status**: Check current GPU support status

#### 4. **Azure VMs with GPU + Docker**
- **Best for**: Full control, testing, development
- **GPU Support**: Full (NC, ND, NV series)
- **Cost**: $$$ (~$200-1500/month)
- **Complexity**: Low
- **Advantages**:
  - Maximum control
  - Direct Docker Compose usage
  - Familiar environment

### ❌ Not Suitable

- **Azure App Service**: No GPU support
- **Azure Functions**: No GPU support, size limits
- **Azure Batch**: Possible but overcomplicated for this use case

## Recommended Approach: AKS with GPU Nodes

Azure Kubernetes Service provides the best balance of:
- Production readiness
- Scalability
- GPU support
- Azure integration
- Cost management

## Architecture Options

### Option 1: Single GPU VM (Development/Small Scale)

```
Internet → Azure Load Balancer → Single NC6s_v3 VM
                                    ├── Ollama (GPU)
                                    ├── RAG Service (GPU)
                                    └── Open-WebUI
```

**Cost**: ~$500/month
**Use Case**: Testing, development, small teams

### Option 2: AKS Cluster (Production)

```
Internet → Application Gateway → AKS Cluster
                                   ├── GPU Node Pool (NC-series)
                                   │   ├── Ollama Pod
                                   │   └── RAG Pod
                                   └── CPU Node Pool
                                       └── WebUI Pod
```

**Cost**: ~$1000-2000/month
**Use Case**: Production, high availability, scalability

### Option 3: Hybrid (Cost Optimized)

```
Internet → Traffic Manager
           ├── Azure VM (GPU) → Ollama + RAG
           └── Container Instances → WebUI
```

**Cost**: ~$300-700/month
**Use Case**: Cost-sensitive production

## GPU Requirements

### Azure GPU VM Series

| VM Series | GPU | vCPUs | RAM | Best For | Cost/Month* |
|-----------|-----|-------|-----|----------|-------------|
| NC6s_v3 | V100 16GB | 6 | 112GB | Development | ~$1,300 |
| NC12s_v3 | 2x V100 | 12 | 224GB | Production | ~$2,600 |
| NC24s_v3 | 4x V100 | 24 | 448GB | High Scale | ~$5,200 |
| NC4as_T4_v3 | T4 16GB | 4 | 28GB | Cost-Optimized | ~$450 |
| NC8as_T4_v3 | T4 16GB | 8 | 56GB | Balanced | ~$900 |

*Approximate costs for US East, subject to change

### Recommended Starting Point

**NC8as_T4_v3** - Best balance of cost and performance
- T4 GPU (sufficient for 32B quantized models)
- 8 vCPUs
- 56GB RAM
- ~$900/month (~$0.45/hour)

## Model Update Strategy

### 1. Zero-Downtime Updates (Blue-Green Deployment)

```bash
# Current production (Green)
Pod A: Ollama with v1 models → Live Traffic

# Deploy new version (Blue)
Pod B: Ollama with v2 models → Testing

# Switch traffic
Pod B → Live Traffic
Pod A → Terminate
```

**Implementation:**
- Use Kubernetes deployment strategies
- Health checks ensure new models are ready
- Instant rollback if issues detected

### 2. Rolling Updates (Gradual)

```bash
# Update one replica at a time
Replica 1 → Pull new model → Test → Live
Replica 2 → Pull new model → Test → Live
Replica 3 → Pull new model → Test → Live
```

**Advantages:**
- No downtime
- Gradual rollout
- Easy rollback

### 3. Scheduled Maintenance Window

```bash
# During low-usage period
1. Announce maintenance window
2. Stop services
3. Pull new models
4. Update configurations
5. Restart services
6. Verify functionality
```

**Best for:**
- Internal tools
- Known low-usage periods
- Major updates

### 4. Model Version Management

```yaml
# Store in Azure Blob Storage
models/
  ├── qwen2.5-coder/
  │   ├── v32b-q4-2024-10/
  │   ├── v32b-q4-2024-11/
  │   └── v32b-q4-2024-12/
  └── deepseek-coder-v2/
      ├── v33b-q4-2024-10/
      └── v33b-q4-2024-11/
```

**Benefits:**
- Version control for models
- Fast rollback capability
- Shared storage across pods
- Reduced pull times

### 5. Automated Update Pipeline

```yaml
# GitHub Actions or Azure DevOps
Schedule: Weekly on Sunday 2 AM

1. Check for new model versions
2. Pull models to staging environment
3. Run automated tests
4. Deploy to production with blue-green
5. Monitor metrics for 1 hour
6. Rollback if metrics degrade
7. Notify team of results
```

## Model Update Methods

### Method A: Ollama API (Dynamic)

```bash
# Update model on running instance
curl -X POST http://ollama:11434/api/pull \
  -d '{"name": "qwen2.5-coder:32b-q4_K_M"}'

# No restart required
# Updates in-place
```

**Pros**: No downtime, simple
**Cons**: Uses instance resources during update

### Method B: Azure Blob Storage (Persistent)

```bash
# Store models in Azure Blob
az storage blob upload \
  --account-name ragmodels \
  --container-name ollama-models \
  --name qwen-32b.gguf \
  --file qwen-32b.gguf

# Mount in Kubernetes
volumes:
  - name: models
    azureFile:
      shareName: ollama-models
      readOnly: true
```

**Pros**: Persistent, shared, fast
**Cons**: Storage costs, complexity

### Method C: Container Image with Models (Not Recommended)

```dockerfile
FROM ollama/ollama:latest
COPY models/ /root/.ollama/models/
```

**Pros**: Self-contained
**Cons**: Huge images (~40GB+), slow updates

### Method D: Init Container Pattern (RECOMMENDED)

```yaml
# Kubernetes init container
initContainers:
  - name: model-loader
    image: azure-cli
    command:
      - /bin/sh
      - -c
      - |
        az storage blob download-batch \
          --source ollama-models \
          --destination /models \
          --account-name $STORAGE_ACCOUNT
    volumeMounts:
      - name: models
        mountPath: /models
```

**Pros**: Clean separation, versioned, fast startup
**Cons**: Requires Azure Storage setup

## Recommended Update Strategy

### For Production (Best Practice)

```yaml
Update Frequency: Monthly (scheduled)
Method: Blue-Green with Init Containers
Storage: Azure Blob Storage
Rollback: Automatic on health check failure
Testing: 1 hour monitoring period
Notifications: Teams/Slack alerts

Steps:
1. Upload new model versions to Azure Blob
2. Update deployment YAML with new model path
3. Deploy blue environment with new models
4. Health checks validate model availability
5. Gradual traffic shift (10% → 50% → 100%)
6. Monitor metrics (latency, error rate)
7. Complete switch or rollback
8. Cleanup old environment after 24h
```

## Cost Optimization for Model Updates

### 1. Use Azure Spot Instances for Testing

```bash
# Test new models on spot VMs (70% discount)
az vm create \
  --name rag-test \
  --priority Spot \
  --eviction-policy Deallocate \
  --max-price -1
```

### 2. Schedule GPU Resources

```bash
# Scale down during low-usage hours
# Scale up during business hours
# Use Azure Automation or Kubernetes HPA
```

### 3. Model Caching

```bash
# Cache models in Azure CDN
# Share across multiple deployments
# Reduce pull times from hours to minutes
```

### 4. Differential Updates

```bash
# Only update changed model files
# Use rsync or Azure Blob incremental copy
# Reduces update time and bandwidth
```

## Quick Start: Deploy to Azure VM (Simplest)

```bash
# 1. Create GPU VM
az vm create \
  --resource-group rag-llm-rg \
  --name rag-llm-vm \
  --size Standard_NC8as_T4_v3 \
  --image Ubuntu2204 \
  --admin-username azureuser \
  --generate-ssh-keys

# 2. Install NVIDIA drivers and Docker
az vm extension set \
  --resource-group rag-llm-rg \
  --vm-name rag-llm-vm \
  --name NvidiaGpuDriverLinux \
  --publisher Microsoft.HpcCompute

# 3. SSH and deploy
ssh azureuser@<vm-ip>
git clone https://github.com/adrian207/dual-rag-llm.git
cd dual-rag-llm
./scripts/setup.sh

# 4. Configure firewall
az vm open-port --port 3000 --resource-group rag-llm-rg --name rag-llm-vm
az vm open-port --port 8000 --resource-group rag-llm-rg --name rag-llm-vm
```

**Time to Deploy**: ~30 minutes
**Monthly Cost**: ~$900 (NC8as_T4_v3)

## Next Steps

See detailed guides:
- [AKS Deployment](./AKS_DEPLOYMENT.md) - Full Kubernetes setup
- [Azure VM Setup](./VM_DEPLOYMENT.md) - Simple VM deployment
- [CI/CD Pipeline](./AZURE_DEVOPS_PIPELINE.md) - Automated deployments
- [Cost Optimization](./COST_OPTIMIZATION.md) - Reduce Azure spending

## Summary: Your Questions Answered

### Can this run natively in Azure?
**Yes!** Multiple options:
- ✅ AKS with GPU nodes (best for production)
- ✅ Azure VMs with GPU + Docker Compose (simplest)
- ✅ Container Instances with GPU (serverless)

### Azure Container Service?
**Yes, recommended!** Use AKS with GPU node pools for:
- Production-grade orchestration
- Auto-scaling
- High availability
- Easy updates

### How to update models?
**Multiple strategies:**
1. **Blue-Green** - Zero downtime, instant rollback
2. **Rolling** - Gradual, safe
3. **Storage-based** - Fast, efficient (Azure Blob)
4. **Automated** - Scheduled, tested, monitored

**Recommended**: Blue-Green deployment with Azure Blob Storage and automated testing pipeline.

---

**Estimated Total Cost for Production:**
- AKS cluster: $200-400/month
- GPU node (NC8as_T4_v3): $900/month
- Storage & networking: $50-100/month
- **Total: ~$1,200-1,500/month**

For development/testing: ~$500-700/month using spot instances and scheduled scaling.

---

Need help with deployment? adrian207@gmail.com

