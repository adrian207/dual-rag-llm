# Azure Deployment - Dual RAG LLM System

**Author:** Adrian Johnson <adrian207@gmail.com>

Complete Azure deployment guides and automation scripts for the Dual RAG LLM system.

## Quick Answers to Your Questions

### ✅ Can this run natively in Azure?

**Yes!** Multiple options available:
- Azure Kubernetes Service (AKS) with GPU - **RECOMMENDED for production**
- Azure VMs with GPU + Docker - **Simplest option**
- Azure Container Instances with GPU - For serverless
- Azure Container Apps - Emerging option

### ✅ Azure Container Service Deployment?

**Absolutely!** We provide full AKS (Azure Kubernetes Service) support:
- Complete Kubernetes manifests in `k8s/` directory
- Automated deployment script: `scripts/deploy-to-aks.sh`
- GPU node pools with NVIDIA support
- Auto-scaling and high availability
- Production-grade orchestration

### ✅ How to Update Models?

**Multiple strategies available:**

1. **Rolling Updates** - Zero downtime, gradual
2. **Blue-Green Deployment** - Instant switch, easy rollback
3. **Scheduled Maintenance** - Planned updates
4. **Automated Pipeline** - CI/CD integration

See [Model Update Guide](#model-update-strategies) below.

---

## Deployment Options Comparison

| Option | Complexity | Cost/Month | Best For | Deployment Time |
|--------|-----------|------------|----------|-----------------|
| **Azure VM** | Low | ~$900 | Dev/Test, Small teams | 1 hour |
| **AKS** | Medium | ~$960 | Production, Scale | 2-3 hours |
| **ACI** | Low | ~$700 | Simple workloads | 30 minutes |
| **Azure Container Apps** | Medium | ~$500 | Serverless | 1 hour |

## Quick Start Options

### Option 1: Azure VM (Simplest)

```bash
# One-command deployment
curl -sL https://raw.githubusercontent.com/adrian207/dual-rag-llm/azure/deployment/azure/scripts/quick-vm-deploy.sh | bash
```

**Or follow manual steps:**
- [Complete VM Deployment Guide](./VM_DEPLOYMENT.md)

**Time**: ~1 hour  
**Cost**: ~$900/month  
**Best for**: Development, testing, small teams

### Option 2: AKS (Production)

```bash
# Automated AKS deployment
./azure/scripts/deploy-to-aks.sh
```

**Or follow manual steps:**
- [Complete AKS Deployment Guide](./AKS_DEPLOYMENT.md)

**Time**: ~2-3 hours  
**Cost**: ~$960/month  
**Best for**: Production, high availability, scalability

## 💻 PowerShell Deployment Support

**Windows users rejoice!** Complete PowerShell deployment scripts available:

- **Deploy-DualRAG-VM.ps1** - Azure VM deployment
- **Deploy-DualRAG-AKS.ps1** - AKS cluster deployment
- **Monitor-DualRAG.ps1** - System monitoring & management

📘 **Complete PowerShell Guide**: [POWERSHELL_DEPLOYMENT.md](./POWERSHELL_DEPLOYMENT.md)

### Quick PowerShell Examples

**VM Deployment:**
```powershell
.\azure\scripts\Deploy-DualRAG-VM.ps1 `
    -ResourceGroupName "rg-dualrag-prod" `
    -Location "eastus" `
    -VMName "vm-dualrag-01" `
    -AdminUsername "azureuser"
```

**AKS Deployment:**
```powershell
.\azure\scripts\Deploy-DualRAG-AKS.ps1 `
    -ResourceGroupName "rg-dualrag-prod" `
    -ClusterName "aks-dualrag" `
    -EnableAutoScaling
```

**Monitoring:**
```powershell
.\azure\scripts\Monitor-DualRAG.ps1 `
    -DeploymentType "AKS" `
    -ResourceGroupName "rg-dualrag-prod" `
    -ClusterName "aks-dualrag" `
    -Action "Status"
```

---

## Detailed Documentation

### Deployment Guides
1. **[AZURE_DEPLOYMENT.md](./AZURE_DEPLOYMENT.md)** - Architecture overview and all options
2. **[VM_DEPLOYMENT.md](./VM_DEPLOYMENT.md)** - Step-by-step VM deployment
3. **[AKS_DEPLOYMENT.md](./AKS_DEPLOYMENT.md)** - Kubernetes deployment guide

### Kubernetes Resources
- `k8s/namespace.yaml` - Namespace configuration
- `k8s/storage.yaml` - Persistent volumes and storage
- `k8s/ollama-statefulset.yaml` - Ollama with GPU
- `k8s/rag-deployment.yaml` - RAG service deployment
- `k8s/webui-deployment.yaml` - Web UI deployment

### Automation Scripts
- `scripts/deploy-to-aks.sh` - Full AKS deployment automation
- `scripts/update-models.sh` - Model update automation
- `scripts/quick-vm-deploy.sh` - Quick VM setup

## Model Update Strategies

### Strategy 1: Rolling Updates (Recommended)

Zero-downtime updates with gradual rollout:

```bash
./azure/scripts/update-models.sh rolling
```

**Process:**
1. Update one pod at a time
2. Health checks ensure model is ready
3. Move to next pod
4. No service interruption

**Use when:** Regular monthly updates

### Strategy 2: Blue-Green Deployment

Instant switch with easy rollback:

```bash
./azure/scripts/update-models.sh blue-green
```

**Process:**
1. Deploy new version alongside current
2. Pull and test new models
3. Switch traffic instantly
4. Rollback in seconds if needed

**Use when:** Major model updates, critical changes

### Strategy 3: Automated Schedule

Set up automated monthly updates:

```bash
# Create Azure DevOps or GitHub Actions pipeline
# Schedule: First Sunday of month at 2 AM

1. Check for new models
2. Deploy to staging
3. Run tests
4. Deploy to production (blue-green)
5. Monitor for 1 hour
6. Rollback if issues
```

### Strategy 4: Manual Ollama API

Update on-demand without restart:

```bash
# For Azure VM
docker exec ollama ollama pull qwen2.5-coder:32b-q4_K_M

# For AKS
kubectl exec -n dual-rag ollama-0 -- \
  ollama pull qwen2.5-coder:32b-q4_K_M
```

## Cost Breakdown

### Azure VM Option
- NC8as_T4_v3 VM: $650/month
- Managed Disks (256GB): $40/month
- Network: $10/month
- **Total**: ~$700/month

**Optimization**: Use scheduled shutdown (business hours only) → ~$200/month

### AKS Option
- System nodes (2x D4s_v3): $280/month
- GPU node (NC8as_T4_v3): $650/month
- Load balancers: $40/month
- Storage: $20/month
- **Total**: ~$990/month

**Optimization**: Use autoscaling and spot instances → ~$400/month

## Cost Optimization Tips

1. **Schedule GPU Resources**
   - Scale down nights/weekends
   - Save 60-70%

2. **Use Spot Instances**
   - 70% discount on compute
   - Good for dev/test

3. **Right-Size VMs**
   - Start with T4 GPUs
   - Upgrade only if needed

4. **Reserved Instances**
   - 1-year commitment: 40% off
   - 3-year commitment: 60% off

5. **Azure Hybrid Benefit**
   - Use existing Windows licenses
   - Additional 30-40% savings

## Architecture Diagrams

### Azure VM Architecture
```
Internet
   │
   ↓
Azure Load Balancer
   │
   ↓
NC8as_T4_v3 VM
   ├── Ollama (GPU)
   ├── RAG Service (GPU)
   └── Open-WebUI
```

### AKS Architecture
```
Internet
   │
   ↓
Application Gateway + WAF
   │
   ↓
AKS Cluster
   ├── GPU Node Pool
   │   ├── Ollama StatefulSet
   │   └── RAG Deployment
   │
   └── CPU Node Pool
       └── WebUI Deployment
```

## Monitoring

### Azure Monitor
```bash
# Enable for VM
az vm enable-insights \
  --resource-group rag-llm-rg \
  --vm-name rag-llm-vm

# Enable for AKS
az aks enable-addons \
  --resource-group rag-llm-aks-rg \
  --name rag-llm-cluster \
  --addons monitoring
```

### Application Insights
```bash
# Get instrumentation key
INSTRUMENTATION_KEY=$(az monitor app-insights component show \
  --app rag-llm-insights \
  --resource-group rag-llm-rg \
  --query instrumentationKey \
  --output tsv)

# Add to environment
export APPINSIGHTS_INSTRUMENTATIONKEY=$INSTRUMENTATION_KEY
```

## Security Best Practices

1. **Network Security**
   - Use Azure Firewall
   - Restrict NSG rules to known IPs
   - Enable DDoS protection

2. **Identity Management**
   - Use Azure AD for authentication
   - Enable managed identities
   - Implement RBAC

3. **Data Protection**
   - Enable disk encryption
   - Use Azure Key Vault for secrets
   - Configure private endpoints

4. **Monitoring & Compliance**
   - Enable Azure Security Center
   - Configure audit logs
   - Set up alerts

## Backup and Disaster Recovery

### VM Backup
```bash
# Enable Azure Backup
az backup protection enable-for-vm \
  --resource-group rag-llm-rg \
  --vault-name rag-backup-vault \
  --vm rag-llm-vm \
  --policy-name DefaultPolicy
```

### AKS Backup
```bash
# Install Velero
velero install --provider azure \
  --bucket rag-backups \
  --secret-file ./credentials

# Schedule daily backups
velero schedule create daily \
  --schedule="0 2 * * *" \
  --include-namespaces dual-rag
```

## Support & Troubleshooting

### Common Issues

**Issue: GPU not detected**
```bash
# VM: Check NVIDIA driver
nvidia-smi

# AKS: Check device plugin
kubectl get pods -n kube-system | grep nvidia
```

**Issue: Out of memory**
```bash
# Increase VM size or add swap
# Or use smaller model quantizations
```

**Issue: Slow model downloads**
```bash
# Use Azure Blob Storage for model caching
# Pre-download models to persistent volumes
```

### Get Help
- Email: adrian207@gmail.com
- GitHub Issues: [repository]/issues
- Azure Support: Open ticket in portal

## Next Steps

1. **Choose Your Deployment Option**
   - Quick test? → Azure VM
   - Production? → AKS

2. **Review Cost Estimates**
   - Use Azure Pricing Calculator
   - Plan for optimization

3. **Deploy**
   - Follow guide for your chosen option
   - Run automated script or manual steps

4. **Configure**
   - Add your documentation
   - Customize model routing
   - Set up monitoring

5. **Optimize**
   - Implement cost-saving measures
   - Configure auto-scaling
   - Set up backups

## Resources

- [Azure GPU VMs](https://docs.microsoft.com/azure/virtual-machines/sizes-gpu)
- [AKS GPU Support](https://docs.microsoft.com/azure/aks/gpu-cluster)
- [Azure Pricing Calculator](https://azure.microsoft.com/pricing/calculator/)
- [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker)

---

**Ready to deploy?** Choose your option and follow the guide!

For questions or support: adrian207@gmail.com

