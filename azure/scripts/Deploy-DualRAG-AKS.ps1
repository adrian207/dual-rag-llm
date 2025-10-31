<#
.SYNOPSIS
    Deploy Dual RAG LLM System to Azure Kubernetes Service (AKS)

.DESCRIPTION
    Complete PowerShell script to deploy the Dual RAG LLM system to Azure Kubernetes Service
    with GPU node pools, monitoring, and auto-scaling.

.PARAMETER ResourceGroupName
    Name of the Azure Resource Group

.PARAMETER Location
    Azure region for deployment

.PARAMETER ClusterName
    Name of the AKS cluster

.PARAMETER NodeCount
    Initial node count (default: 2)

.PARAMETER EnableAutoScaling
    Enable cluster autoscaling

.PARAMETER MinNodeCount
    Minimum nodes for autoscaling

.PARAMETER MaxNodeCount
    Maximum nodes for autoscaling

.EXAMPLE
    .\Deploy-DualRAG-AKS.ps1 -ResourceGroupName "rg-dualrag-prod" -Location "eastus" -ClusterName "aks-dualrag" -EnableAutoScaling

.NOTES
    Author: Adrian Johnson <adrian207@gmail.com>
    Requires: Azure PowerShell Module (Az), kubectl, helm
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$ResourceGroupName,

    [Parameter(Mandatory = $false)]
    [string]$Location = "eastus",

    [Parameter(Mandatory = $false)]
    [string]$ClusterName = "aks-dualrag",

    [Parameter(Mandatory = $false)]
    [int]$NodeCount = 2,

    [Parameter(Mandatory = $false)]
    [switch]$EnableAutoScaling,

    [Parameter(Mandatory = $false)]
    [int]$MinNodeCount = 1,

    [Parameter(Mandatory = $false)]
    [int]$MaxNodeCount = 5,

    [Parameter(Mandatory = $false)]
    [string]$NodeVMSize = "Standard_NC6s_v3",

    [Parameter(Mandatory = $false)]
    [string]$KubernetesVersion = "1.28"
)

#Requires -Version 5.1
#Requires -Modules Az.Aks, Az.Resources

$ErrorActionPreference = "Stop"

# Colors for output
function Write-ColorOutput {
    param(
        [string]$Message,
        [string]$Color = "White"
    )
    $currentColor = $host.UI.RawUI.ForegroundColor
    $host.UI.RawUI.ForegroundColor = $Color
    Write-Output $Message
    $host.UI.RawUI.ForegroundColor = $currentColor
}

Write-ColorOutput "`n🚀 Dual RAG LLM - Azure AKS Deployment Script" -Color "Cyan"
Write-ColorOutput "==========================================`n" -Color "Cyan"

# Check prerequisites
Write-ColorOutput "🔍 Checking prerequisites..." -Color "Cyan"

# Check kubectl
try {
    $kubectlVersion = kubectl version --client --short 2>$null
    Write-ColorOutput "✓ kubectl installed: $kubectlVersion" -Color "Green"
} catch {
    Write-ColorOutput "❌ kubectl not found. Please install: https://kubernetes.io/docs/tasks/tools/" -Color "Red"
    exit 1
}

# Check helm
try {
    $helmVersion = helm version --short 2>$null
    Write-ColorOutput "✓ helm installed: $helmVersion" -Color "Green"
} catch {
    Write-ColorOutput "❌ helm not found. Please install: https://helm.sh/docs/intro/install/" -Color "Red"
    exit 1
}

# Check Azure login
try {
    $context = Get-AzContext
    if (-not $context) {
        throw "Not logged in"
    }
    Write-ColorOutput "✓ Logged in as: $($context.Account.Id)" -Color "Green"
} catch {
    Write-ColorOutput "⚠ Not logged in to Azure. Please run Connect-AzAccount" -Color "Yellow"
    Connect-AzAccount
}

# Create Resource Group
Write-ColorOutput "`n📦 Creating Resource Group: $ResourceGroupName" -Color "Cyan"
$rg = Get-AzResourceGroup -Name $ResourceGroupName -ErrorAction SilentlyContinue
if (-not $rg) {
    $rg = New-AzResourceGroup -Name $ResourceGroupName -Location $Location
    Write-ColorOutput "✓ Resource Group created" -Color "Green"
} else {
    Write-ColorOutput "✓ Resource Group already exists" -Color "Yellow"
}

# Create AKS Cluster
Write-ColorOutput "`n☸️  Creating AKS Cluster: $ClusterName" -Color "Cyan"
$aks = Get-AzAksCluster -ResourceGroupName $ResourceGroupName -Name $ClusterName -ErrorAction SilentlyContinue

if (-not $aks) {
    Write-ColorOutput "⏳ Creating cluster (this may take 10-15 minutes)..." -Color "Yellow"
    
    $aksParams = @{
        ResourceGroupName = $ResourceGroupName
        Name = $ClusterName
        Location = $Location
        NodeCount = $NodeCount
        NodeVmSize = $NodeVMSize
        KubernetesVersion = $KubernetesVersion
        EnableManagedIdentity = $true
        GenerateSshKey = $true
    }

    if ($EnableAutoScaling) {
        $aksParams['EnableClusterAutoscaler'] = $true
        $aksParams['MinNodeCount'] = $MinNodeCount
        $aksParams['MaxNodeCount'] = $MaxNodeCount
    }

    $aks = New-AzAksCluster @aksParams
    Write-ColorOutput "✓ AKS Cluster created" -Color "Green"
} else {
    Write-ColorOutput "✓ AKS Cluster already exists" -Color "Yellow"
}

# Get credentials
Write-ColorOutput "`n🔑 Getting cluster credentials..." -Color "Cyan"
Import-AzAksCredential -ResourceGroupName $ResourceGroupName -Name $ClusterName -Force
Write-ColorOutput "✓ Credentials configured" -Color "Green"

# Verify connection
Write-ColorOutput "`n✅ Verifying cluster connection..." -Color "Cyan"
$nodes = kubectl get nodes -o json | ConvertFrom-Json
Write-ColorOutput "✓ Connected to cluster with $($nodes.items.Count) nodes" -Color "Green"

# Install NVIDIA Device Plugin
Write-ColorOutput "`n🎮 Installing NVIDIA GPU Device Plugin..." -Color "Cyan"
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.0/nvidia-device-plugin.yml
Start-Sleep -Seconds 10
Write-ColorOutput "✓ NVIDIA plugin installed" -Color "Green"

# Create namespace
Write-ColorOutput "`n📦 Creating namespace: dualrag" -Color "Cyan"
kubectl create namespace dualrag --dry-run=client -o yaml | kubectl apply -f -
Write-ColorOutput "✓ Namespace created" -Color "Green"

# Create secrets (placeholders)
Write-ColorOutput "`n🔐 Creating secrets..." -Color "Cyan"
$secretYaml = @"
apiVersion: v1
kind: Secret
metadata:
  name: dualrag-secrets
  namespace: dualrag
type: Opaque
stringData:
  brave-api-key: "REPLACE_WITH_YOUR_BRAVE_API_KEY"
  github-token: "REPLACE_WITH_YOUR_GITHUB_TOKEN"
"@

$secretYaml | kubectl apply -f -
Write-ColorOutput "✓ Secrets created (please update with actual values)" -Color "Yellow"

# Apply Kubernetes manifests
Write-ColorOutput "`n📝 Applying Kubernetes manifests..." -Color "Cyan"

$manifestPath = Join-Path (Get-Location) "azure\k8s"
if (Test-Path $manifestPath) {
    # Apply in order
    kubectl apply -f "$manifestPath\redis-statefulset.yaml"
    kubectl apply -f "$manifestPath\ollama-statefulset.yaml"
    kubectl apply -f "$manifestPath\chroma-statefulset.yaml"
    kubectl apply -f "$manifestPath\api-deployment.yaml"
    kubectl apply -f "$manifestPath\nginx-deployment.yaml"
    kubectl apply -f "$manifestPath\ingress.yaml"
    
    Write-ColorOutput "✓ Manifests applied" -Color "Green"
} else {
    Write-ColorOutput "⚠ Manifest directory not found at $manifestPath" -Color "Yellow"
    Write-ColorOutput "  Please ensure you're running from the project root" -Color "Yellow"
}

# Wait for deployments
Write-ColorOutput "`n⏳ Waiting for deployments to be ready..." -Color "Cyan"
kubectl wait --for=condition=ready pod -l app=redis -n dualrag --timeout=300s
kubectl wait --for=condition=ready pod -l app=ollama -n dualrag --timeout=300s
kubectl wait --for=condition=ready pod -l app=dualrag-api -n dualrag --timeout=300s
Write-ColorOutput "✓ Deployments ready" -Color "Green"

# Get service information
Write-ColorOutput "`n📋 Getting service information..." -Color "Cyan"
$services = kubectl get services -n dualrag -o json | ConvertFrom-Json

foreach ($svc in $services.items) {
    if ($svc.spec.type -eq "LoadBalancer") {
        $lb = $svc.status.loadBalancer.ingress[0]
        if ($lb.ip) {
            Write-ColorOutput "  $($svc.metadata.name): http://$($lb.ip)" -Color "White"
        }
    }
}

# Install monitoring (optional)
Write-ColorOutput "`n📊 Installing monitoring stack..." -Color "Cyan"
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

helm upgrade --install prometheus prometheus-community/kube-prometheus-stack `
    --namespace monitoring --create-namespace `
    --set prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues=false

Write-ColorOutput "✓ Monitoring installed" -Color "Green"

# Output summary
Write-ColorOutput "`n✅ Deployment Complete!" -Color "Green"
Write-ColorOutput "==========================================`n" -Color "Green"

Write-ColorOutput "📋 Cluster Information:" -Color "Cyan"
Write-ColorOutput "  Resource Group: $ResourceGroupName" -Color "White"
Write-ColorOutput "  Cluster Name: $ClusterName" -Color "White"
Write-ColorOutput "  Location: $Location" -Color "White"
Write-ColorOutput "  Nodes: $NodeCount" -Color "White"

Write-ColorOutput "`n📝 Useful Commands:" -Color "Cyan"
Write-ColorOutput "  View pods: kubectl get pods -n dualrag" -Color "White"
Write-ColorOutput "  View logs: kubectl logs -f <pod-name> -n dualrag" -Color "White"
Write-ColorOutput "  Scale: kubectl scale deployment dualrag-api --replicas=3 -n dualrag" -Color "White"
Write-ColorOutput "  Port-forward: kubectl port-forward svc/dualrag-api 8000:8000 -n dualrag" -Color "White"

Write-ColorOutput "`n⚠️  Important:" -Color "Yellow"
Write-ColorOutput "  1. Update secrets with actual API keys:" -Color "White"
Write-ColorOutput "     kubectl edit secret dualrag-secrets -n dualrag" -Color "White"
Write-ColorOutput "  2. Configure ingress domain in azure\k8s\ingress.yaml" -Color "White"
Write-ColorOutput "  3. Monitor GPU usage: kubectl top nodes" -Color "White"

Write-ColorOutput "`n🎉 AKS deployment completed successfully!" -Color "Green"

