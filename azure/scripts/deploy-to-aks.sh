#!/bin/bash
# Deploy Dual RAG LLM to Azure Kubernetes Service
# Author: Adrian Johnson <adrian207@gmail.com>

set -e

echo "=== Dual RAG LLM - AKS Deployment ==="
echo ""

# Configuration
RESOURCE_GROUP="${RESOURCE_GROUP:-rag-llm-aks-rg}"
LOCATION="${LOCATION:-eastus}"
CLUSTER_NAME="${CLUSTER_NAME:-rag-llm-cluster}"
ACR_NAME="${ACR_NAME:-ragllmacr$(date +%s)}"

echo "Configuration:"
echo "  Resource Group: $RESOURCE_GROUP"
echo "  Location: $LOCATION"
echo "  Cluster Name: $CLUSTER_NAME"
echo "  ACR Name: $ACR_NAME"
echo ""

# Check prerequisites
echo "Checking prerequisites..."
command -v az >/dev/null 2>&1 || { echo "Azure CLI is required but not installed."; exit 1; }
command -v kubectl >/dev/null 2>&1 || { echo "kubectl is required but not installed."; exit 1; }
echo "✓ Prerequisites OK"
echo ""

# Create resource group
echo "Creating resource group..."
az group create --name $RESOURCE_GROUP --location $LOCATION
echo "✓ Resource group created"
echo ""

# Create ACR
echo "Creating Azure Container Registry..."
az acr create \
  --resource-group $RESOURCE_GROUP \
  --name $ACR_NAME \
  --sku Standard \
  --admin-enabled true
ACR_LOGIN_SERVER=$(az acr show --name $ACR_NAME --query loginServer --output tsv)
echo "✓ ACR created: $ACR_LOGIN_SERVER"
echo ""

# Create AKS cluster
echo "Creating AKS cluster (this takes 10-15 minutes)..."
az aks create \
  --resource-group $RESOURCE_GROUP \
  --name $CLUSTER_NAME \
  --node-count 2 \
  --node-vm-size Standard_D4s_v3 \
  --enable-managed-identity \
  --attach-acr $ACR_NAME \
  --enable-addons monitoring \
  --generate-ssh-keys
echo "✓ AKS cluster created"
echo ""

# Add GPU node pool
echo "Adding GPU node pool..."
az aks nodepool add \
  --resource-group $RESOURCE_GROUP \
  --cluster-name $CLUSTER_NAME \
  --name gpunodepool \
  --node-count 1 \
  --node-vm-size Standard_NC8as_T4_v3 \
  --node-taints sku=gpu:NoSchedule \
  --labels sku=gpu
echo "✓ GPU node pool added"
echo ""

# Get AKS credentials
echo "Getting AKS credentials..."
az aks get-credentials \
  --resource-group $RESOURCE_GROUP \
  --name $CLUSTER_NAME \
  --overwrite-existing
echo "✓ Credentials configured"
echo ""

# Install NVIDIA device plugin
echo "Installing NVIDIA device plugin..."
kubectl create -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.0/nvidia-device-plugin.yml
echo "✓ NVIDIA plugin installed"
echo ""

# Build and push images
echo "Building and pushing container images..."
cd "$(dirname "$0")/../.."

az acr login --name $ACR_NAME

docker build -t $ACR_LOGIN_SERVER/rag-service:v1.0.0 ./rag
docker push $ACR_LOGIN_SERVER/rag-service:v1.0.0

echo "✓ Images pushed"
echo ""

# Deploy to Kubernetes
echo "Deploying to Kubernetes..."

kubectl create namespace dual-rag || true

# Update image in deployment
sed "s|ragllmacr.azurecr.io|$ACR_LOGIN_SERVER|g" \
  azure/k8s/rag-deployment.yaml > /tmp/rag-deployment.yaml

kubectl apply -f azure/k8s/storage.yaml
kubectl apply -f azure/k8s/ollama-statefulset.yaml
sleep 30  # Wait for Ollama to start

kubectl apply -f /tmp/rag-deployment.yaml
kubectl apply -f azure/k8s/webui-deployment.yaml

echo "✓ Deployed to Kubernetes"
echo ""

# Wait for pods
echo "Waiting for pods to be ready..."
kubectl wait --for=condition=ready pod -l app=ollama -n dual-rag --timeout=300s
kubectl wait --for=condition=ready pod -l app=rag -n dual-rag --timeout=300s
kubectl wait --for=condition=ready pod -l app=webui -n dual-rag --timeout=300s
echo "✓ All pods ready"
echo ""

# Pull models
echo "Pulling LLM models (this takes 30-60 minutes)..."
kubectl exec -n dual-rag ollama-0 -- ollama pull qwen2.5-coder:32b-q4_K_M &
kubectl exec -n dual-rag ollama-0 -- ollama pull deepseek-coder-v2:33b-q4_K_M &
wait
echo "✓ Models pulled"
echo ""

# Get external IPs
echo "Getting service endpoints..."
sleep 30  # Wait for load balancers

WEBUI_IP=$(kubectl get service -n dual-rag webui-service -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
RAG_IP=$(kubectl get service -n dual-rag rag-service -o jsonpath='{.status.loadBalancer.ingress[0].ip}')

echo ""
echo "=== Deployment Complete ==="
echo ""
echo "Access your services:"
echo "  Web UI: http://$WEBUI_IP:3000"
echo "  RAG API: http://$RAG_IP:8000"
echo "  API Docs: http://$RAG_IP:8000/docs"
echo ""
echo "Verify deployment:"
echo "  kubectl get pods -n dual-rag"
echo "  kubectl logs -n dual-rag -l app=rag -f"
echo ""
echo "Estimated monthly cost: ~$960 USD"
echo ""

