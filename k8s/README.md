# Kubernetes Deployment Guide

Complete guide for deploying Dual RAG LLM System to Kubernetes.

**Author:** Adrian Johnson <adrian207@gmail.com>

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Helm Installation](#helm-installation)
- [Configuration](#configuration)
- [Advanced Deployment](#advanced-deployment)
- [Monitoring](#monitoring)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required Tools

- **Kubernetes 1.24+** - Production-ready cluster
- **Helm 3.12+** - Package manager for Kubernetes
- **kubectl 1.24+** - Kubernetes CLI
- **Container Registry** - For custom images (optional)

### Cluster Requirements

**Minimum Resources:**
- 3 nodes (for high availability)
- 8 CPU cores per node
- 16 GB RAM per node
- 100 GB disk space per node
- NVIDIA GPU support (optional, for Ollama)

**Recommended Resources:**
- 5+ nodes
- 16+ CPU cores per node
- 32+ GB RAM per node
- 500+ GB SSD per node
- NVIDIA A100/V100 GPU (for production)

---

## Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/adrian207/dual-rag-llm.git
cd dual-rag-llm/k8s
```

### 2. Configure Values

```bash
cp helm/dual-rag-llm/values.yaml my-values.yaml
# Edit my-values.yaml with your configuration
```

### 3. Deploy

```bash
# Automated deployment
./deploy.sh

# Or manual Helm installation
helm install dual-rag-llm ./helm/dual-rag-llm \
  --namespace dual-rag-llm \
  --create-namespace \
  --values my-values.yaml
```

### 4. Verify Deployment

```bash
kubectl get pods -n dual-rag-llm
kubectl get svc -n dual-rag-llm
kubectl get ingress -n dual-rag-llm
```

---

## Helm Installation

### Install from Chart

```bash
# Add Helm repository (if published)
helm repo add dual-rag-llm https://charts.dual-rag-llm.io
helm repo update

# Install
helm install my-release dual-rag-llm/dual-rag-llm \
  --namespace dual-rag-llm \
  --create-namespace
```

### Install from Source

```bash
# From local chart
helm install my-release ./helm/dual-rag-llm \
  --namespace dual-rag-llm \
  --create-namespace \
  --values my-values.yaml
```

### Upgrade Release

```bash
helm upgrade my-release ./helm/dual-rag-llm \
  --namespace dual-rag-llm \
  --values my-values.yaml
```

### Uninstall

```bash
helm uninstall my-release --namespace dual-rag-llm
kubectl delete namespace dual-rag-llm
```

---

## Configuration

### Essential Values

**Image Configuration:**
```yaml
image:
  repository: your-registry/dual-rag-llm
  tag: "1.19.0"
  pullPolicy: IfNotPresent
```

**Resource Limits:**
```yaml
resources:
  limits:
    cpu: 2000m
    memory: 4Gi
  requests:
    cpu: 500m
    memory: 2Gi
```

**Autoscaling:**
```yaml
autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70
```

**Ingress:**
```yaml
ingress:
  enabled: true
  className: "nginx"
  hosts:
    - host: dual-rag.example.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: dual-rag-tls
      hosts:
        - dual-rag.example.com
```

**Redis:**
```yaml
redis:
  enabled: true
  auth:
    password: "your-secure-password"
  master:
    persistence:
      size: 8Gi
```

**Ollama:**
```yaml
ollama:
  enabled: true
  resources:
    limits:
      nvidia.com/gpu: "1"
  persistence:
    size: 50Gi
  models:
    - llama3.1
    - deepseek-coder
```

### Secrets Configuration

Create secrets before deployment:

```bash
# Generate encryption key
ENCRYPTION_KEY=$(openssl rand -base64 32)

# Create secrets
kubectl create secret generic dual-rag-llm-secrets \
  --namespace=dual-rag-llm \
  --from-literal=encryption-master-key=$ENCRYPTION_KEY \
  --from-literal=brave-search-api-key=YOUR_BRAVE_KEY \
  --from-literal=github-token=YOUR_GITHUB_TOKEN
```

Or use values.yaml:

```yaml
secrets:
  ENCRYPTION_MASTER_KEY: "base64-encoded-key"
  BRAVE_SEARCH_API_KEY: "your-api-key"
  GITHUB_TOKEN: "your-github-token"
```

---

## Advanced Deployment

### Multi-Environment Setup

**Development:**
```bash
helm install dual-rag-dev ./helm/dual-rag-llm \
  --namespace dual-rag-dev \
  --values values-dev.yaml
```

**Staging:**
```bash
helm install dual-rag-staging ./helm/dual-rag-llm \
  --namespace dual-rag-staging \
  --values values-staging.yaml
```

**Production:**
```bash
helm install dual-rag-prod ./helm/dual-rag-llm \
  --namespace dual-rag-prod \
  --values values-prod.yaml
```

### GPU Node Configuration

Label GPU nodes:
```bash
kubectl label nodes gpu-node-1 nvidia.com/gpu=true
kubectl label nodes gpu-node-1 workload=gpu-intensive
```

Configure node selector in values.yaml:
```yaml
ollama:
  nodeSelector:
    nvidia.com/gpu: "true"
    workload: gpu-intensive
```

### High Availability Setup

```yaml
replicaCount: 3

affinity:
  podAntiAffinity:
    requiredDuringSchedulingIgnoredDuringExecution:
      - labelSelector:
          matchExpressions:
            - key: app.kubernetes.io/name
              operator: In
              values:
                - dual-rag-llm
        topologyKey: kubernetes.io/hostname

podDisruptionBudget:
  enabled: true
  minAvailable: 2
```

### Persistent Storage

**Using AWS EBS:**
```yaml
persistence:
  enabled: true
  storageClass: "gp3"
  size: 100Gi
```

**Using Azure Disk:**
```yaml
persistence:
  enabled: true
  storageClass: "managed-premium"
  size: 100Gi
```

**Using GCP Persistent Disk:**
```yaml
persistence:
  enabled: true
  storageClass: "standard-rwo"
  size: 100Gi
```

---

## Monitoring

### Prometheus Integration

```yaml
monitoring:
  enabled: true
  serviceMonitor:
    enabled: true
    interval: 30s
```

### Grafana Dashboards

Install Grafana:
```bash
helm install grafana grafana/grafana \
  --namespace monitoring \
  --set persistence.enabled=true
```

Import dashboard: [Dashboard JSON available in `/monitoring` directory]

### Logging with ELK

```bash
# Install Elasticsearch
helm install elasticsearch elastic/elasticsearch \
  --namespace logging

# Install Kibana
helm install kibana elastic/kibana \
  --namespace logging

# Configure Filebeat
kubectl apply -f logging/filebeat-config.yaml
```

---

## Troubleshooting

### Common Issues

**Pods not starting:**
```bash
kubectl describe pod <pod-name> -n dual-rag-llm
kubectl logs <pod-name> -n dual-rag-llm
```

**Ingress not working:**
```bash
kubectl describe ingress -n dual-rag-llm
kubectl logs -n ingress-nginx deployment/ingress-nginx-controller
```

**Redis connection issues:**
```bash
kubectl logs -n dual-rag-llm deployment/dual-rag-llm | grep redis
kubectl exec -it -n dual-rag-llm <pod-name> -- redis-cli -h dual-rag-llm-redis-master ping
```

**GPU not available:**
```bash
kubectl describe node <gpu-node>
kubectl get pods -n gpu-operator
nvidia-smi
```

### Debug Commands

```bash
# Check all resources
kubectl get all -n dual-rag-llm

# Check events
kubectl get events -n dual-rag-llm --sort-by='.lastTimestamp'

# Check logs
kubectl logs -f deployment/dual-rag-llm -n dual-rag-llm

# Port forward for debugging
kubectl port-forward svc/dual-rag-llm 8000:8000 -n dual-rag-llm

# Shell into pod
kubectl exec -it deployment/dual-rag-llm -n dual-rag-llm -- /bin/bash
```

---

## Production Checklist

- [ ] Configure proper resource limits
- [ ] Enable autoscaling
- [ ] Set up persistent storage
- [ ] Configure ingress with TLS
- [ ] Create secure secrets
- [ ] Enable monitoring
- [ ] Configure logging
- [ ] Set up backups
- [ ] Configure network policies
- [ ] Test disaster recovery
- [ ] Document runbooks
- [ ] Set up alerts

---

## Support

For issues and questions:
- GitHub Issues: https://github.com/adrian207/dual-rag-llm/issues
- Email: adrian207@gmail.com
- Documentation: https://github.com/adrian207/dual-rag-llm/tree/main/docs

---

**Version:** 1.19.0  
**Last Updated:** October 31, 2024  
**Author:** Adrian Johnson <adrian207@gmail.com>

