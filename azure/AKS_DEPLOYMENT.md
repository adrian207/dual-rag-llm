# Azure Kubernetes Service (AKS) Deployment

**Author:** Adrian Johnson <adrian207@gmail.com>

Deploy the Dual RAG system to AKS for production-grade scalability and reliability.

## Architecture

```
Internet
   │
   ↓
Azure Application Gateway (WAF)
   │
   ↓
AKS Cluster
   ├── GPU Node Pool (NC-series)
   │   ├── Ollama Deployment (StatefulSet)
   │   └── RAG Service Deployment
   │
   └── CPU Node Pool (Standard D-series)
       └── WebUI Deployment
```

## Prerequisites

- Azure subscription
- Azure CLI with AKS preview features
- kubectl installed
- Helm 3 installed

## Quick Start Deployment

### 1. Set Variables

```bash
RESOURCE_GROUP="rag-llm-aks-rg"
LOCATION="eastus"
CLUSTER_NAME="rag-llm-cluster"
ACR_NAME="ragllmacr$(date +%s)"  # Unique name
```

### 2. Create Resource Group

```bash
az group create \
  --name $RESOURCE_GROUP \
  --location $LOCATION
```

### 3. Create Azure Container Registry

```bash
az acr create \
  --resource-group $RESOURCE_GROUP \
  --name $ACR_NAME \
  --sku Standard

# Enable admin access
az acr update \
  --name $ACR_NAME \
  --admin-enabled true

# Get credentials
ACR_LOGIN_SERVER=$(az acr show --name $ACR_NAME --query loginServer --output tsv)
```

### 4. Create AKS Cluster

```bash
# Create cluster with system node pool
az aks create \
  --resource-group $RESOURCE_GROUP \
  --name $CLUSTER_NAME \
  --node-count 2 \
  --node-vm-size Standard_D4s_v3 \
  --enable-managed-identity \
  --attach-acr $ACR_NAME \
  --enable-addons monitoring \
  --generate-ssh-keys

# Add GPU node pool
az aks nodepool add \
  --resource-group $RESOURCE_GROUP \
  --cluster-name $CLUSTER_NAME \
  --name gpunodepool \
  --node-count 1 \
  --node-vm-size Standard_NC8as_T4_v3 \
  --node-taints sku=gpu:NoSchedule \
  --labels sku=gpu

# Get credentials
az aks get-credentials \
  --resource-group $RESOURCE_GROUP \
  --name $CLUSTER_NAME
```

### 5. Install NVIDIA Device Plugin

```bash
kubectl create -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.0/nvidia-device-plugin.yml

# Verify GPU nodes
kubectl get nodes -o wide
kubectl describe node -l sku=gpu
```

### 6. Build and Push Container Images

```bash
cd ~/dual-rag-llm

# Login to ACR
az acr login --name $ACR_NAME

# Build and push RAG service
docker build -t $ACR_LOGIN_SERVER/rag-service:v1.0.0 ./rag
docker push $ACR_LOGIN_SERVER/rag-service:v1.0.0

# Ollama (use official image, or customize)
docker pull ollama/ollama:latest
docker tag ollama/ollama:latest $ACR_LOGIN_SERVER/ollama:latest
docker push $ACR_LOGIN_SERVER/ollama:latest

# WebUI
docker pull ghcr.io/open-webui/open-webui:main
docker tag ghcr.io/open-webui/open-webui:main $ACR_LOGIN_SERVER/open-webui:main
docker push $ACR_LOGIN_SERVER/open-webui:main
```

### 7. Create Kubernetes Manifests

See `./k8s/` directory for all manifests. Key files:

- `namespace.yaml` - Namespace
- `storage.yaml` - Persistent volumes
- `ollama-statefulset.yaml` - Ollama with GPU
- `rag-deployment.yaml` - RAG service
- `webui-deployment.yaml` - Web interface
- `ingress.yaml` - External access

### 8. Deploy to Kubernetes

```bash
# Create namespace
kubectl create namespace dual-rag

# Deploy storage
kubectl apply -f azure/k8s/storage.yaml -n dual-rag

# Deploy Ollama
kubectl apply -f azure/k8s/ollama-statefulset.yaml -n dual-rag

# Wait for Ollama to be ready
kubectl wait --for=condition=ready pod -l app=ollama -n dual-rag --timeout=300s

# Pull models into Ollama
kubectl exec -n dual-rag ollama-0 -- ollama pull qwen2.5-coder:32b-q4_K_M
kubectl exec -n dual-rag ollama-0 -- ollama pull deepseek-coder-v2:33b-q4_K_M

# Deploy RAG service
kubectl apply -f azure/k8s/rag-deployment.yaml -n dual-rag

# Deploy WebUI
kubectl apply -f azure/k8s/webui-deployment.yaml -n dual-rag

# Deploy ingress
kubectl apply -f azure/k8s/ingress.yaml -n dual-rag
```

### 9. Verify Deployment

```bash
# Check all pods
kubectl get pods -n dual-rag

# Check services
kubectl get services -n dual-rag

# Check ingress
kubectl get ingress -n dual-rag

# View logs
kubectl logs -n dual-rag -l app=rag -f
```

### 10. Access Application

```bash
# Get external IP
EXTERNAL_IP=$(kubectl get service -n dual-rag webui-service -o jsonpath='{.status.loadBalancer.ingress[0].ip}')

echo "Web UI: http://$EXTERNAL_IP:3000"
echo "API: http://$EXTERNAL_IP:8000"
```

## Model Update Strategy

### Method 1: Rolling Update

```bash
# Update image with new model version
kubectl set image statefulset/ollama \
  ollama=$ACR_LOGIN_SERVER/ollama:v2 \
  -n dual-rag

# Or update deployment
kubectl rollout restart statefulset/ollama -n dual-rag

# Check status
kubectl rollout status statefulset/ollama -n dual-rag
```

### Method 2: Blue-Green Deployment

```yaml
# Deploy new version alongside current
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: ollama-v2
  namespace: dual-rag
spec:
  # ... same as ollama but with new models
  
# Switch service selector
kubectl patch service ollama-service -n dual-rag \
  -p '{"spec":{"selector":{"version":"v2"}}}'

# Cleanup old version
kubectl delete statefulset ollama -n dual-rag
```

### Method 3: Persistent Volume with Model Manager

```yaml
# Create init container to sync models
initContainers:
  - name: model-sync
    image: azure/cli
    command:
      - sh
      - -c
      - |
        az storage blob download-batch \
          --source models \
          --destination /models \
          --pattern "*.gguf"
    volumeMounts:
      - name: models
        mountPath: /models
    env:
      - name: AZURE_STORAGE_ACCOUNT
        valueFrom:
          secretKeyRef:
            name: azure-storage
            key: account
```

## Auto-Scaling Configuration

### Horizontal Pod Autoscaler

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: rag-hpa
  namespace: dual-rag
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: rag-deployment
  minReplicas: 2
  maxReplicas: 10
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
    - type: Resource
      resource:
        name: memory
        target:
          type: Utilization
          averageUtilization: 80
```

### Cluster Autoscaler

```bash
# Enable cluster autoscaler
az aks update \
  --resource-group $RESOURCE_GROUP \
  --name $CLUSTER_NAME \
  --enable-cluster-autoscaler \
  --min-count 1 \
  --max-count 5 \
  --nodepool-name gpunodepool
```

## Monitoring and Logging

### Azure Monitor

```bash
# Enable Container Insights
az aks enable-addons \
  --resource-group $RESOURCE_GROUP \
  --name $CLUSTER_NAME \
  --addons monitoring

# View in Azure Portal:
# AKS → Monitoring → Insights
```

### Prometheus + Grafana

```bash
# Install Prometheus
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm install prometheus prometheus-community/kube-prometheus-stack -n monitoring

# Port forward Grafana
kubectl port-forward -n monitoring svc/prometheus-grafana 3001:80

# Access: http://localhost:3001
# Default: admin/prom-operator
```

## Cost Optimization

### Current Costs (Estimated)

- System node pool (2x D4s_v3): ~$280/month
- GPU node pool (1x NC8as_T4_v3): ~$650/month
- Load balancer: ~$20/month
- Managed disk: ~$10/month
- **Total**: ~$960/month

### Optimization Strategies

1. **Schedule GPU nodes**
   ```bash
   # Scale down during off-hours
   kubectl scale statefulset ollama --replicas=0 -n dual-rag
   az aks nodepool scale \
     --resource-group $RESOURCE_GROUP \
     --cluster-name $CLUSTER_NAME \
     --name gpunodepool \
     --node-count 0
   ```

2. **Use spot instances**
   ```bash
   az aks nodepool add \
     --resource-group $RESOURCE_GROUP \
     --cluster-name $CLUSTER_NAME \
     --name spotnodepool \
     --priority Spot \
     --eviction-policy Delete \
     --spot-max-price -1 \
     --node-vm-size Standard_NC8as_T4_v3 \
     --node-count 1
   ```

3. **Right-size nodes** - Start with T4 GPUs, upgrade only if needed

## Backup and Disaster Recovery

### Backup Strategy

```bash
# Install Velero for backups
velero install \
  --provider azure \
  --plugins velero/velero-plugin-for-microsoft-azure:v1.5.0 \
  --bucket velero \
  --secret-file ./credentials-velero

# Create backup
velero backup create rag-backup --include-namespaces dual-rag

# Schedule daily backups
velero schedule create daily-backup \
  --schedule="0 2 * * *" \
  --include-namespaces dual-rag
```

## Security

### Network Policies

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: rag-network-policy
  namespace: dual-rag
spec:
  podSelector:
    matchLabels:
      app: rag
  policyTypes:
    - Ingress
    - Egress
  ingress:
    - from:
        - podSelector:
            matchLabels:
              app: webui
      ports:
        - protocol: TCP
          port: 8000
  egress:
    - to:
        - podSelector:
            matchLabels:
              app: ollama
      ports:
        - protocol: TCP
          port: 11434
```

### Azure Key Vault Integration

```bash
# Enable Key Vault secrets
az aks enable-addons \
  --addons azure-keyvault-secrets-provider \
  --name $CLUSTER_NAME \
  --resource-group $RESOURCE_GROUP
```

## Clean Up

```bash
# Delete entire resource group
az group delete \
  --name $RESOURCE_GROUP \
  --yes \
  --no-wait
```

## Summary

**Complexity**: Medium-High
**Monthly Cost**: ~$960 (can optimize to ~$400)
**Best For**: Production, high availability, scalability
**Deployment Time**: 2-3 hours

---

See also:
- [Model Update Automation](./MODEL_UPDATE_PIPELINE.md)
- [Cost Optimization Guide](./COST_OPTIMIZATION.md)
- [Monitoring Setup](./MONITORING_SETUP.md)

