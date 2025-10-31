#!/bin/bash

# Dual RAG LLM Kubernetes Deployment Script
# Author: Adrian Johnson <adrian207@gmail.com>

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
NAMESPACE="${NAMESPACE:-dual-rag-llm}"
RELEASE_NAME="${RELEASE_NAME:-dual-rag-llm}"
HELM_CHART="./helm/dual-rag-llm"
VALUES_FILE="${VALUES_FILE:-./helm/dual-rag-llm/values.yaml}"

# Functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed"
        exit 1
    fi
    
    # Check helm
    if ! command -v helm &> /dev/null; then
        log_error "helm is not installed"
        exit 1
    fi
    
    # Check cluster connection
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    log_info "Prerequisites check passed"
}

create_namespace() {
    log_info "Creating namespace: $NAMESPACE"
    kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -
}

install_cert_manager() {
    log_info "Installing cert-manager (if not already installed)..."
    
    if ! kubectl get namespace cert-manager &> /dev/null; then
        kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml
        log_info "Waiting for cert-manager to be ready..."
        kubectl wait --for=condition=Available --timeout=300s deployment/cert-manager -n cert-manager
        kubectl wait --for=condition=Available --timeout=300s deployment/cert-manager-webhook -n cert-manager
    else
        log_info "cert-manager is already installed"
    fi
}

install_ingress_nginx() {
    log_info "Installing ingress-nginx (if not already installed)..."
    
    if ! kubectl get namespace ingress-nginx &> /dev/null; then
        kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.8.1/deploy/static/provider/cloud/deploy.yaml
        log_info "Waiting for ingress-nginx to be ready..."
        kubectl wait --for=condition=Available --timeout=300s deployment/ingress-nginx-controller -n ingress-nginx
    else
        log_info "ingress-nginx is already installed"
    fi
}

setup_secrets() {
    log_info "Setting up secrets..."
    
    # Prompt for secrets if not set
    if [ -z "$ENCRYPTION_MASTER_KEY" ]; then
        log_warn "ENCRYPTION_MASTER_KEY not set. Generating random key..."
        ENCRYPTION_MASTER_KEY=$(openssl rand -base64 32)
    fi
    
    # Create secrets
    kubectl create secret generic ${RELEASE_NAME}-secrets \
        --namespace=$NAMESPACE \
        --from-literal=encryption-master-key=$ENCRYPTION_MASTER_KEY \
        --from-literal=brave-search-api-key=${BRAVE_SEARCH_API_KEY:-""} \
        --from-literal=github-token=${GITHUB_TOKEN:-""} \
        --dry-run=client -o yaml | kubectl apply -f -
    
    log_info "Secrets created successfully"
}

install_redis() {
    log_info "Adding Bitnami Helm repository..."
    helm repo add bitnami https://charts.bitnami.com/bitnami
    helm repo update
}

deploy_application() {
    log_info "Deploying Dual RAG LLM..."
    
    helm upgrade --install $RELEASE_NAME $HELM_CHART \
        --namespace=$NAMESPACE \
        --values=$VALUES_FILE \
        --set image.tag=${IMAGE_TAG:-latest} \
        --wait \
        --timeout=10m
    
    log_info "Application deployed successfully"
}

wait_for_pods() {
    log_info "Waiting for pods to be ready..."
    kubectl wait --for=condition=Ready pods \
        --selector=app.kubernetes.io/name=dual-rag-llm \
        --namespace=$NAMESPACE \
        --timeout=5m
}

show_status() {
    log_info "Deployment Status:"
    echo ""
    kubectl get all -n $NAMESPACE
    echo ""
    
    # Get ingress URL
    INGRESS_HOST=$(kubectl get ingress -n $NAMESPACE -o jsonpath='{.items[0].spec.rules[0].host}' 2>/dev/null || echo "Not configured")
    log_info "Access URL: https://$INGRESS_HOST"
    
    # Get load balancer IP (if available)
    LB_IP=$(kubectl get svc -n ingress-nginx ingress-nginx-controller -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "Pending...")
    log_info "Load Balancer IP: $LB_IP"
}

# Main deployment
main() {
    log_info "Starting Dual RAG LLM Kubernetes deployment..."
    
    check_prerequisites
    create_namespace
    install_cert_manager
    install_ingress_nginx
    install_redis
    setup_secrets
    deploy_application
    wait_for_pods
    show_status
    
    log_info "Deployment completed successfully!"
}

# Run main function
main

