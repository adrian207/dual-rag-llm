# PowerShell Deployment Scripts

**Author:** Adrian Johnson <adrian207@gmail.com>

Complete PowerShell scripts for deploying and managing the Dual RAG LLM System on Azure.

## 📋 Prerequisites

### Software Requirements
- **PowerShell 5.1+** or **PowerShell Core 7+**
- **Azure PowerShell Module** (`Az`)
- **kubectl** (for AKS deployments)
- **helm** (for AKS deployments)

### Install Prerequisites

```powershell
# Install Azure PowerShell Module
Install-Module -Name Az -Repository PSGallery -Force

# Install kubectl (using Chocolatey)
choco install kubernetes-cli

# Or download from: https://kubernetes.io/docs/tasks/tools/

# Install helm
choco install kubernetes-helm

# Or download from: https://helm.sh/docs/intro/install/
```

### Azure Login

```powershell
Connect-AzAccount

# Select subscription
Set-AzContext -Subscription "Your-Subscription-Name"
```

---

## 🖥️ VM Deployment

Deploy to a single Azure Virtual Machine with GPU support.

### Basic Deployment

```powershell
.\azure\scripts\Deploy-DualRAG-VM.ps1 `
    -ResourceGroupName "rg-dualrag-prod" `
    -Location "eastus" `
    -VMName "vm-dualrag-01" `
    -AdminUsername "azureuser"
```

### Advanced Deployment

```powershell
$password = ConvertTo-SecureString "YourStrongPassword123!" -AsPlainText -Force

.\azure\scripts\Deploy-DualRAG-VM.ps1 `
    -ResourceGroupName "rg-dualrag-prod" `
    -Location "eastus2" `
    -VMName "vm-dualrag-gpu" `
    -VMSize "Standard_NC6s_v3" `
    -AdminUsername "azureuser" `
    -AdminPassword $password `
    -DataDiskSizeGB 1024 `
    -VNetName "vnet-dualrag-custom" `
    -SubnetName "subnet-ai" `
    -PublicIPName "pip-dualrag-external"
```

### Parameters

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `ResourceGroupName` | Yes | - | Azure Resource Group name |
| `Location` | No | eastus | Azure region |
| `VMName` | No | vm-dualrag-01 | Virtual Machine name |
| `VMSize` | No | Standard_NC6s_v3 | VM size (GPU recommended) |
| `AdminUsername` | Yes | - | VM administrator username |
| `AdminPassword` | No | Prompted | VM administrator password |
| `DataDiskSizeGB` | No | 512 | Data disk size for Docker |
| `VNetName` | No | vnet-dualrag | Virtual Network name |
| `SubnetName` | No | subnet-compute | Subnet name |
| `NSGName` | No | nsg-dualrag | Network Security Group |
| `PublicIPName` | No | pip-dualrag | Public IP name |

### GPU-Optimized VM Sizes

| Size | vCPUs | RAM | GPU | Best For |
|------|-------|-----|-----|----------|
| Standard_NC6s_v3 | 6 | 112 GB | 1x V100 | Development/Testing |
| Standard_NC12s_v3 | 12 | 224 GB | 2x V100 | Production |
| Standard_NC24s_v3 | 24 | 448 GB | 4x V100 | High Performance |

### Post-Deployment

After deployment completes:

```powershell
# SSH into the VM
ssh azureuser@<public-ip>

# Navigate to project directory
cd /opt/dual-rag-llm

# Set environment variables
nano .env
# Add:
# BRAVE_API_KEY=your_key
# GITHUB_TOKEN=your_token

# Start services
docker-compose up -d

# Verify
docker-compose ps
curl http://localhost:8000/health
```

---

## ☸️ AKS Deployment

Deploy to Azure Kubernetes Service for production scalability.

### Basic Deployment

```powershell
.\azure\scripts\Deploy-DualRAG-AKS.ps1 `
    -ResourceGroupName "rg-dualrag-prod" `
    -Location "eastus" `
    -ClusterName "aks-dualrag"
```

### With Auto-Scaling

```powershell
.\azure\scripts\Deploy-DualRAG-AKS.ps1 `
    -ResourceGroupName "rg-dualrag-prod" `
    -Location "westus2" `
    -ClusterName "aks-dualrag-prod" `
    -NodeCount 3 `
    -EnableAutoScaling `
    -MinNodeCount 2 `
    -MaxNodeCount 10 `
    -NodeVMSize "Standard_NC6s_v3"
```

### Parameters

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `ResourceGroupName` | Yes | - | Azure Resource Group name |
| `Location` | No | eastus | Azure region |
| `ClusterName` | No | aks-dualrag | AKS cluster name |
| `NodeCount` | No | 2 | Initial node count |
| `EnableAutoScaling` | No | false | Enable cluster autoscaling |
| `MinNodeCount` | No | 1 | Minimum nodes (if autoscaling) |
| `MaxNodeCount` | No | 5 | Maximum nodes (if autoscaling) |
| `NodeVMSize` | No | Standard_NC6s_v3 | Node VM size |
| `KubernetesVersion` | No | 1.28 | Kubernetes version |

### Post-Deployment

```powershell
# Get cluster credentials
Import-AzAksCredential -ResourceGroupName "rg-dualrag-prod" -Name "aks-dualrag"

# Update secrets with actual values
kubectl edit secret dualrag-secrets -n dualrag

# Verify deployment
kubectl get pods -n dualrag
kubectl get services -n dualrag

# Port-forward for testing
kubectl port-forward svc/dualrag-api 8000:8000 -n dualrag

# Test
curl http://localhost:8000/health
```

---

## 📊 Monitoring & Management

Monitor and manage deployed systems.

### VM Monitoring

```powershell
# Check VM status
.\azure\scripts\Monitor-DualRAG.ps1 `
    -DeploymentType "VM" `
    -ResourceGroupName "rg-dualrag-prod" `
    -VMName "vm-dualrag-01" `
    -Action "Status"

# View metrics
.\azure\scripts\Monitor-DualRAG.ps1 `
    -DeploymentType "VM" `
    -ResourceGroupName "rg-dualrag-prod" `
    -VMName "vm-dualrag-01" `
    -Action "Metrics"

# Health check
.\azure\scripts\Monitor-DualRAG.ps1 `
    -DeploymentType "VM" `
    -ResourceGroupName "rg-dualrag-prod" `
    -VMName "vm-dualrag-01" `
    -Action "Health"

# Restart VM
.\azure\scripts\Monitor-DualRAG.ps1 `
    -DeploymentType "VM" `
    -ResourceGroupName "rg-dualrag-prod" `
    -VMName "vm-dualrag-01" `
    -Action "Restart"
```

### AKS Monitoring

```powershell
# Check cluster status
.\azure\scripts\Monitor-DualRAG.ps1 `
    -DeploymentType "AKS" `
    -ResourceGroupName "rg-dualrag-prod" `
    -ClusterName "aks-dualrag" `
    -Action "Status"

# View pod logs
.\azure\scripts\Monitor-DualRAG.ps1 `
    -DeploymentType "AKS" `
    -ResourceGroupName "rg-dualrag-prod" `
    -ClusterName "aks-dualrag" `
    -Action "Logs"

# Scale deployment
.\azure\scripts\Monitor-DualRAG.ps1 `
    -DeploymentType "AKS" `
    -ResourceGroupName "rg-dualrag-prod" `
    -ClusterName "aks-dualrag" `
    -Action "Scale"

# View metrics
.\azure\scripts\Monitor-DualRAG.ps1 `
    -DeploymentType "AKS" `
    -ResourceGroupName "rg-dualrag-prod" `
    -ClusterName "aks-dualrag" `
    -Action "Metrics"

# Health check
.\azure\scripts\Monitor-DualRAG.ps1 `
    -DeploymentType "AKS" `
    -ResourceGroupName "rg-dualrag-prod" `
    -ClusterName "aks-dualrag" `
    -Action "Health"

# Restart deployments
.\azure\scripts\Monitor-DualRAG.ps1 `
    -DeploymentType "AKS" `
    -ResourceGroupName "rg-dualrag-prod" `
    -ClusterName "aks-dualrag" `
    -Action "Restart"
```

---

## 🔧 Troubleshooting

### Common Issues

#### 1. Azure Login Failed

```powershell
# Clear cached credentials
Disconnect-AzAccount
Clear-AzContext -Force

# Login again
Connect-AzAccount
```

#### 2. VM Deployment Timeout

```powershell
# Check deployment status
Get-AzResourceGroupDeployment -ResourceGroupName "rg-dualrag-prod"

# View detailed error
(Get-AzResourceGroupDeployment -ResourceGroupName "rg-dualrag-prod").Properties.Error
```

#### 3. AKS Connection Issues

```powershell
# Re-import credentials
Import-AzAksCredential -ResourceGroupName "rg-dualrag-prod" -Name "aks-dualrag" -Force

# Test connection
kubectl cluster-info

# Check authentication
kubectl auth can-i get pods --all-namespaces
```

#### 4. GPU Not Detected

```powershell
# For VM:
# SSH and check
nvidia-smi

# For AKS:
kubectl describe nodes | grep nvidia

# Reinstall device plugin
kubectl delete -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.0/nvidia-device-plugin.yml
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.0/nvidia-device-plugin.yml
```

---

## 💰 Cost Optimization

### VM Cost Savings

```powershell
# Stop VM when not in use
Stop-AzVM -ResourceGroupName "rg-dualrag-prod" -Name "vm-dualrag-01"

# Start VM
Start-AzVM -ResourceGroupName "rg-dualrag-prod" -Name "vm-dualrag-01"

# Deallocate to stop billing
Stop-AzVM -ResourceGroupName "rg-dualrag-prod" -Name "vm-dualrag-01" -Force
```

### AKS Cost Savings

```powershell
# Scale down during off-hours
kubectl scale deployment/dualrag-api --replicas=1 -n dualrag
kubectl scale deployment/nginx --replicas=1 -n dualrag

# Scale up for production
kubectl scale deployment/dualrag-api --replicas=3 -n dualrag
kubectl scale deployment/nginx --replicas=2 -n dualrag
```

---

## 🔐 Security Best Practices

1. **Use Managed Identities**
   ```powershell
   # Enable managed identity on VM
   $vm = Get-AzVM -ResourceGroupName "rg-dualrag-prod" -Name "vm-dualrag-01"
   Update-AzVM -ResourceGroupName "rg-dualrag-prod" -VM $vm -IdentityType SystemAssigned
   ```

2. **Restrict Network Access**
   ```powershell
   # Update NSG to allow only specific IPs
   Get-AzNetworkSecurityGroup -ResourceGroupName "rg-dualrag-prod" -Name "nsg-dualrag" | `
       Add-AzNetworkSecurityRuleConfig -Name "Allow-MyIP-SSH" `
       -Access Allow -Protocol Tcp -Direction Inbound -Priority 90 `
       -SourceAddressPrefix "YOUR_IP/32" -SourcePortRange * `
       -DestinationAddressPrefix * -DestinationPortRange 22 | `
       Set-AzNetworkSecurityGroup
   ```

3. **Enable Encryption**
   ```powershell
   # Enable disk encryption
   Set-AzVMDiskEncryptionExtension -ResourceGroupName "rg-dualrag-prod" `
       -VMName "vm-dualrag-01" -DiskEncryptionKeyVaultUrl $KeyVault.VaultUri `
       -DiskEncryptionKeyVaultId $KeyVault.ResourceId
   ```

---

## 📚 Additional Resources

- [Azure PowerShell Documentation](https://docs.microsoft.com/en-us/powershell/azure/)
- [AKS Documentation](https://docs.microsoft.com/en-us/azure/aks/)
- [Azure VM Sizes](https://docs.microsoft.com/en-us/azure/virtual-machines/sizes)
- [NVIDIA GPU on Azure](https://docs.microsoft.com/en-us/azure/virtual-machines/sizes-gpu)

---

## 🆘 Support

For issues or questions:
- **GitHub Issues**: [dual-rag-llm/issues](https://github.com/adrian207/dual-rag-llm/issues)
- **Email**: adrian207@gmail.com
- **Azure Support**: [https://azure.microsoft.com/support](https://azure.microsoft.com/support)

