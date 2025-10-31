# Azure VM Deployment - Quick Start

**Author:** Adrian Johnson <adrian207@gmail.com>

Deploy the Dual RAG system to a single Azure VM with GPU. This is the **simplest** Azure deployment option.

## Prerequisites

- Azure subscription
- Azure CLI installed locally
- SSH key pair (or will be generated)

## Step-by-Step Deployment

### 1. Set Variables

```bash
# Configure your deployment
RESOURCE_GROUP="rag-llm-rg"
LOCATION="eastus"
VM_NAME="rag-llm-vm"
VM_SIZE="Standard_NC8as_T4_v3"  # T4 GPU, 8 vCPU, 56GB RAM
ADMIN_USER="azureuser"
```

### 2. Create Resource Group

```bash
az group create \
  --name $RESOURCE_GROUP \
  --location $LOCATION
```

### 3. Create Virtual Network

```bash
az network vnet create \
  --resource-group $RESOURCE_GROUP \
  --name rag-vnet \
  --address-prefix 10.0.0.0/16 \
  --subnet-name rag-subnet \
  --subnet-prefix 10.0.1.0/24
```

### 4. Create Network Security Group

```bash
# Create NSG
az network nsg create \
  --resource-group $RESOURCE_GROUP \
  --name rag-nsg

# Allow SSH
az network nsg rule create \
  --resource-group $RESOURCE_GROUP \
  --nsg-name rag-nsg \
  --name AllowSSH \
  --priority 1000 \
  --source-address-prefixes '*' \
  --destination-port-ranges 22 \
  --protocol Tcp \
  --access Allow

# Allow Open-WebUI
az network nsg rule create \
  --resource-group $RESOURCE_GROUP \
  --nsg-name rag-nsg \
  --name AllowWebUI \
  --priority 1001 \
  --source-address-prefixes '*' \
  --destination-port-ranges 3000 \
  --protocol Tcp \
  --access Allow

# Allow RAG API
az network nsg rule create \
  --resource-group $RESOURCE_GROUP \
  --nsg-name rag-nsg \
  --name AllowRAGAPI \
  --priority 1002 \
  --source-address-prefixes '*' \
  --destination-port-ranges 8000 \
  --protocol Tcp \
  --access Allow
```

### 5. Create GPU-Enabled VM

```bash
az vm create \
  --resource-group $RESOURCE_GROUP \
  --name $VM_NAME \
  --size $VM_SIZE \
  --image Ubuntu2204 \
  --admin-username $ADMIN_USER \
  --generate-ssh-keys \
  --vnet-name rag-vnet \
  --subnet rag-subnet \
  --nsg rag-nsg \
  --public-ip-sku Standard \
  --os-disk-size-gb 256
```

**Note**: VM creation takes 5-10 minutes

### 6. Install NVIDIA GPU Drivers

```bash
# Install NVIDIA GPU extension
az vm extension set \
  --resource-group $RESOURCE_GROUP \
  --vm-name $VM_NAME \
  --name NvidiaGpuDriverLinux \
  --publisher Microsoft.HpcCompute \
  --version 1.9
```

**Note**: Driver installation takes 10-15 minutes

### 7. Get VM Public IP

```bash
VM_IP=$(az vm show \
  --resource-group $RESOURCE_GROUP \
  --name $VM_NAME \
  --show-details \
  --query publicIps \
  --output tsv)

echo "VM Public IP: $VM_IP"
```

### 8. SSH into VM

```bash
ssh $ADMIN_USER@$VM_IP
```

### 9. Install Docker and NVIDIA Container Toolkit

```bash
# Update system
sudo apt-get update
sudo apt-get upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Verify GPU
nvidia-smi
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

### 10. Clone and Deploy Application

```bash
# Clone repository
git clone https://github.com/adrian207/dual-rag-llm.git
cd dual-rag-llm

# Run automated setup
./scripts/setup.sh
```

**Note**: Setup takes 30-60 minutes (mostly downloading models)

### 11. Verify Deployment

```bash
# Check services
docker compose ps

# Test health
curl http://localhost:8000/health

# View logs
docker compose logs -f rag
```

### 12. Access from Browser

Open in your browser:
- **Web UI**: http://$VM_IP:3000
- **API**: http://$VM_IP:8000
- **API Docs**: http://$VM_IP:8000/docs

## Post-Deployment Configuration

### Set Up HTTPS with Let's Encrypt

```bash
# Install Certbot
sudo apt-get install -y certbot

# Get certificate (requires domain name)
sudo certbot certonly --standalone -d your-domain.com

# Update docker-compose.yml to use certificates
# Add reverse proxy (nginx) for HTTPS
```

### Enable Automatic Updates

```bash
# Create update script
cat > ~/update-models.sh << 'EOF'
#!/bin/bash
cd ~/dual-rag-llm
docker exec ollama ollama pull qwen2.5-coder:32b-q4_K_M
docker exec ollama ollama pull deepseek-coder-v2:33b-q4_K_M
docker compose restart rag
EOF

chmod +x ~/update-models.sh

# Schedule monthly updates
crontab -e
# Add: 0 2 1 * * /home/azureuser/update-models.sh
```

### Set Up Monitoring

```bash
# Install Azure Monitor agent
wget https://aka.ms/dependencyagentlinux -O InstallDependencyAgent-Linux64.bin
sudo sh InstallDependencyAgent-Linux64.bin -s

# Enable VM insights in Azure Portal
# Portal → VM → Monitoring → Insights → Enable
```

### Configure Backups

```bash
# Backup indexes
cat > ~/backup-indexes.sh << 'EOF'
#!/bin/bash
DATE=$(date +%Y%m%d)
tar -czf /backup/indexes-$DATE.tar.gz ~/dual-rag-llm/rag/indexes/
# Upload to Azure Blob (configure az CLI first)
az storage blob upload \
  --account-name yourstorageaccount \
  --container-name backups \
  --name indexes-$DATE.tar.gz \
  --file /backup/indexes-$DATE.tar.gz
EOF

chmod +x ~/backup-indexes.sh

# Schedule daily backups
crontab -e
# Add: 0 3 * * * /home/azureuser/backup-indexes.sh
```

## Cost Management

### Current VM Cost
- **NC8as_T4_v3**: ~$0.45/hour = ~$324/month (24/7)

### Optimization Strategies

#### 1. Scheduled Shutdown
```bash
# Stop VM during non-business hours
az vm deallocate \
  --resource-group $RESOURCE_GROUP \
  --name $VM_NAME

# Start VM
az vm start \
  --resource-group $RESOURCE_GROUP \
  --name $VM_NAME
```

**Savings**: ~60% if running only 8 hours/day

#### 2. Use Azure Automation

Create runbooks to:
- Start VM at 8 AM
- Stop VM at 6 PM
- Weekend shutdown

**Estimated cost**: ~$130/month (business hours only)

#### 3. Use Spot Instances (Dev/Test)

```bash
# Create spot instance VM (70% discount)
az vm create \
  --resource-group $RESOURCE_GROUP \
  --name $VM_NAME-spot \
  --size $VM_SIZE \
  --priority Spot \
  --max-price -1 \
  --eviction-policy Deallocate \
  --image Ubuntu2204 \
  --admin-username $ADMIN_USER \
  --generate-ssh-keys
```

**Cost**: ~$0.14/hour = ~$100/month
**Caveat**: Can be evicted when Azure needs capacity

## Maintenance

### Update Application

```bash
cd ~/dual-rag-llm
git pull origin main
docker compose build
docker compose up -d
```

### Update Models

```bash
docker exec ollama ollama pull qwen2.5-coder:32b-q4_K_M
docker exec ollama ollama pull deepseek-coder-v2:33b-q4_K_M
docker compose restart rag
```

### View Logs

```bash
docker compose logs -f rag
docker compose logs -f ollama
```

### Monitor Resources

```bash
# CPU and memory
htop

# GPU usage
watch -n 1 nvidia-smi

# Disk usage
df -h

# Docker stats
docker stats
```

## Troubleshooting

### GPU Not Detected

```bash
# Check NVIDIA driver
nvidia-smi

# Reinstall if needed
sudo apt-get purge -y nvidia-*
az vm extension set \
  --resource-group $RESOURCE_GROUP \
  --vm-name $VM_NAME \
  --name NvidiaGpuDriverLinux \
  --publisher Microsoft.HpcCompute
```

### Services Not Starting

```bash
# Check Docker
sudo systemctl status docker

# Check logs
docker compose logs

# Restart services
docker compose down
docker compose up -d
```

### Out of Disk Space

```bash
# Clean Docker
docker system prune -a

# Clean logs
docker compose down
rm -rf rag/logs/*
docker compose up -d

# Expand disk (in Azure Portal)
```

## Security Best Practices

1. **Restrict SSH access**
   ```bash
   # Update NSG to allow only your IP
   az network nsg rule update \
     --resource-group $RESOURCE_GROUP \
     --nsg-name rag-nsg \
     --name AllowSSH \
     --source-address-prefixes YOUR_IP
   ```

2. **Enable Azure Firewall**
3. **Use Azure Key Vault for secrets**
4. **Enable disk encryption**
5. **Set up Azure Backup**

## Clean Up (When Done Testing)

```bash
# Delete entire resource group
az group delete \
  --name $RESOURCE_GROUP \
  --yes \
  --no-wait
```

## Summary

**Total Time**: ~1 hour
**Monthly Cost**: ~$324 (24/7) or ~$130 (business hours)
**Complexity**: Low
**Best For**: Development, testing, small teams

---

Next: Consider [AKS deployment](./AKS_DEPLOYMENT.md) for production scale.

