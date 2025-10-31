<#
.SYNOPSIS
    Deploy Dual RAG LLM System to Azure VM

.DESCRIPTION
    Complete PowerShell script to deploy the Dual RAG LLM system to an Azure Virtual Machine
    with all required dependencies (Docker, NVIDIA drivers, Ollama).

.PARAMETER ResourceGroupName
    Name of the Azure Resource Group

.PARAMETER Location
    Azure region for deployment

.PARAMETER VMName
    Name of the Virtual Machine

.PARAMETER VMSize
    VM size (default: Standard_NC6s_v3 with GPU)

.PARAMETER AdminUsername
    VM administrator username

.PARAMETER AdminPassword
    VM administrator password (secure string)

.EXAMPLE
    .\Deploy-DualRAG-VM.ps1 -ResourceGroupName "rg-dualrag-prod" -Location "eastus" -VMName "vm-dualrag-01" -AdminUsername "azureuser"

.NOTES
    Author: Adrian Johnson <adrian207@gmail.com>
    Requires: Azure PowerShell Module (Az)
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$ResourceGroupName,

    [Parameter(Mandatory = $false)]
    [string]$Location = "eastus",

    [Parameter(Mandatory = $false)]
    [string]$VMName = "vm-dualrag-01",

    [Parameter(Mandatory = $false)]
    [string]$VMSize = "Standard_NC6s_v3",

    [Parameter(Mandatory = $true)]
    [string]$AdminUsername,

    [Parameter(Mandatory = $false)]
    [securestring]$AdminPassword,

    [Parameter(Mandatory = $false)]
    [string]$VNetName = "vnet-dualrag",

    [Parameter(Mandatory = $false)]
    [string]$SubnetName = "subnet-compute",

    [Parameter(Mandatory = $false)]
    [string]$NSGName = "nsg-dualrag",

    [Parameter(Mandatory = $false)]
    [string]$PublicIPName = "pip-dualrag",

    [Parameter(Mandatory = $false)]
    [int]$DataDiskSizeGB = 512
)

#Requires -Version 5.1
#Requires -Modules Az.Compute, Az.Network, Az.Resources

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

Write-ColorOutput "`n🚀 Dual RAG LLM - Azure VM Deployment Script" -Color "Cyan"
Write-ColorOutput "============================================`n" -Color "Cyan"

# Check if logged in to Azure
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

# Prompt for password if not provided
if (-not $AdminPassword) {
    $AdminPassword = Read-Host -AsSecureString -Prompt "Enter VM Administrator Password"
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

# Create Virtual Network
Write-ColorOutput "`n🌐 Creating Virtual Network: $VNetName" -Color "Cyan"
$vnet = Get-AzVirtualNetwork -ResourceGroupName $ResourceGroupName -Name $VNetName -ErrorAction SilentlyContinue
if (-not $vnet) {
    $subnetConfig = New-AzVirtualNetworkSubnetConfig -Name $SubnetName -AddressPrefix "10.0.1.0/24"
    $vnet = New-AzVirtualNetwork -Name $VNetName -ResourceGroupName $ResourceGroupName `
        -Location $Location -AddressPrefix "10.0.0.0/16" -Subnet $subnetConfig
    Write-ColorOutput "✓ Virtual Network created" -Color "Green"
} else {
    Write-ColorOutput "✓ Virtual Network already exists" -Color "Yellow"
}

# Create Network Security Group
Write-ColorOutput "`n🔒 Creating Network Security Group: $NSGName" -Color "Cyan"
$nsg = Get-AzNetworkSecurityGroup -ResourceGroupName $ResourceGroupName -Name $NSGName -ErrorAction SilentlyContinue
if (-not $nsg) {
    # SSH Rule
    $sshRule = New-AzNetworkSecurityRuleConfig -Name "Allow-SSH" -Description "Allow SSH" `
        -Access Allow -Protocol Tcp -Direction Inbound -Priority 100 `
        -SourceAddressPrefix Internet -SourcePortRange * `
        -DestinationAddressPrefix * -DestinationPortRange 22

    # HTTP Rule
    $httpRule = New-AzNetworkSecurityRuleConfig -Name "Allow-HTTP" -Description "Allow HTTP" `
        -Access Allow -Protocol Tcp -Direction Inbound -Priority 110 `
        -SourceAddressPrefix Internet -SourcePortRange * `
        -DestinationAddressPrefix * -DestinationPortRange 80

    # HTTPS Rule
    $httpsRule = New-AzNetworkSecurityRuleConfig -Name "Allow-HTTPS" -Description "Allow HTTPS" `
        -Access Allow -Protocol Tcp -Direction Inbound -Priority 120 `
        -SourceAddressPrefix Internet -SourcePortRange * `
        -DestinationAddressPrefix * -DestinationPortRange 443

    # API Rule (8000)
    $apiRule = New-AzNetworkSecurityRuleConfig -Name "Allow-API" -Description "Allow API" `
        -Access Allow -Protocol Tcp -Direction Inbound -Priority 130 `
        -SourceAddressPrefix Internet -SourcePortRange * `
        -DestinationAddressPrefix * -DestinationPortRange 8000

    $nsg = New-AzNetworkSecurityGroup -ResourceGroupName $ResourceGroupName -Location $Location `
        -Name $NSGName -SecurityRules $sshRule, $httpRule, $httpsRule, $apiRule
    Write-ColorOutput "✓ Network Security Group created" -Color "Green"
} else {
    Write-ColorOutput "✓ Network Security Group already exists" -Color "Yellow"
}

# Create Public IP
Write-ColorOutput "`n🌍 Creating Public IP: $PublicIPName" -Color "Cyan"
$pip = Get-AzPublicIpAddress -ResourceGroupName $ResourceGroupName -Name $PublicIPName -ErrorAction SilentlyContinue
if (-not $pip) {
    $pip = New-AzPublicIpAddress -Name $PublicIPName -ResourceGroupName $ResourceGroupName `
        -Location $Location -AllocationMethod Static -Sku Standard
    Write-ColorOutput "✓ Public IP created: $($pip.IpAddress)" -Color "Green"
} else {
    Write-ColorOutput "✓ Public IP already exists: $($pip.IpAddress)" -Color "Yellow"
}

# Create Network Interface
Write-ColorOutput "`n🔌 Creating Network Interface" -Color "Cyan"
$nicName = "$VMName-nic"
$nic = Get-AzNetworkInterface -ResourceGroupName $ResourceGroupName -Name $nicName -ErrorAction SilentlyContinue
if (-not $nic) {
    $subnet = Get-AzVirtualNetworkSubnetConfig -Name $SubnetName -VirtualNetwork $vnet
    $nic = New-AzNetworkInterface -Name $nicName -ResourceGroupName $ResourceGroupName `
        -Location $Location -SubnetId $subnet.Id -PublicIpAddressId $pip.Id `
        -NetworkSecurityGroupId $nsg.Id
    Write-ColorOutput "✓ Network Interface created" -Color "Green"
} else {
    Write-ColorOutput "✓ Network Interface already exists" -Color "Yellow"
}

# Create VM Configuration
Write-ColorOutput "`n💻 Creating Virtual Machine: $VMName" -Color "Cyan"
$vm = Get-AzVM -ResourceGroupName $ResourceGroupName -Name $VMName -ErrorAction SilentlyContinue
if (-not $vm) {
    $vmConfig = New-AzVMConfig -VMName $VMName -VMSize $VMSize
    
    # Set Ubuntu 22.04 LTS image
    $vmConfig = Set-AzVMOperatingSystem -VM $vmConfig -Linux -ComputerName $VMName `
        -Credential (New-Object PSCredential($AdminUsername, $AdminPassword)) `
        -DisablePasswordAuthentication:$false

    $vmConfig = Set-AzVMSourceImage -VM $vmConfig -PublisherName "Canonical" `
        -Offer "0001-com-ubuntu-server-jammy" -Skus "22_04-lts-gen2" -Version "latest"

    $vmConfig = Add-AzVMNetworkInterface -VM $vmConfig -Id $nic.Id

    # Add data disk for Docker volumes
    $diskConfig = New-AzDiskConfig -SkuName Premium_LRS -Location $Location `
        -CreateOption Empty -DiskSizeGB $DataDiskSizeGB
    $dataDisk = New-AzDisk -DiskName "$VMName-data-disk" -Disk $diskConfig `
        -ResourceGroupName $ResourceGroupName
    $vmConfig = Add-AzVMDataDisk -VM $vmConfig -Name "$VMName-data-disk" `
        -CreateOption Attach -ManagedDiskId $dataDisk.Id -Lun 0

    # Create the VM
    Write-ColorOutput "⏳ Creating VM (this may take several minutes)..." -Color "Yellow"
    New-AzVM -ResourceGroupName $ResourceGroupName -Location $Location -VM $vmConfig | Out-Null
    Write-ColorOutput "✓ Virtual Machine created successfully" -Color "Green"
} else {
    Write-ColorOutput "✓ Virtual Machine already exists" -Color "Yellow"
}

# Custom Script Extension for setup
Write-ColorOutput "`n⚙️ Installing Docker and dependencies..." -Color "Cyan"

$setupScript = @'
#!/bin/bash
set -e

# Update system
apt-get update && apt-get upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh
usermod -aG docker $USER

# Install Docker Compose
curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose

# Format and mount data disk
if [ -b /dev/sdc ]; then
    mkfs.ext4 /dev/sdc
    mkdir -p /mnt/data
    echo "/dev/sdc /mnt/data ext4 defaults 0 0" >> /etc/fstab
    mount -a
    mkdir -p /mnt/data/docker /mnt/data/ollama
    ln -s /mnt/data/docker /var/lib/docker
fi

# Install NVIDIA drivers (if GPU present)
if lspci | grep -i nvidia > /dev/null; then
    apt-get install -y ubuntu-drivers-common
    ubuntu-drivers autoinstall
fi

# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | tee /etc/apt/sources.list.d/nvidia-docker.list
apt-get update && apt-get install -y nvidia-container-toolkit
systemctl restart docker

# Clone repository
cd /opt
git clone https://github.com/adrian207/dual-rag-llm.git
cd dual-rag-llm

# Create .env file
cat > .env << EOF
OLLAMA_API=http://ollama:11434
REDIS_HOST=redis
REDIS_PORT=6379
BRAVE_API_KEY=\${BRAVE_API_KEY}
GITHUB_TOKEN=\${GITHUB_TOKEN}
EOF

echo "✓ Setup complete!"
'@

$scriptPath = [System.IO.Path]::GetTempFileName() + ".sh"
Set-Content -Path $scriptPath -Value $setupScript

# Upload and run the script
$null = Set-AzVMCustomScriptExtension -ResourceGroupName $ResourceGroupName `
    -VMName $VMName -Location $Location -FileUri $scriptPath `
    -Run "bash setup.sh" -Name "DualRAGSetup"

Remove-Item $scriptPath

Write-ColorOutput "✓ Dependencies installed" -Color "Green"

# Output connection information
Write-ColorOutput "`n✅ Deployment Complete!" -Color "Green"
Write-ColorOutput "==========================================`n" -Color "Green"

$publicIP = (Get-AzPublicIpAddress -ResourceGroupName $ResourceGroupName -Name $PublicIPName).IpAddress

Write-ColorOutput "📋 Connection Information:" -Color "Cyan"
Write-ColorOutput "  SSH: ssh $AdminUsername@$publicIP" -Color "White"
Write-ColorOutput "  API: http://$publicIP:8000" -Color "White"
Write-ColorOutput "  UI:  http://$publicIP" -Color "White"
Write-ColorOutput "`n📝 Next Steps:" -Color "Cyan"
Write-ColorOutput "  1. SSH into the VM" -Color "White"
Write-ColorOutput "  2. cd /opt/dual-rag-llm" -Color "White"
Write-ColorOutput "  3. Set environment variables in .env" -Color "White"
Write-ColorOutput "  4. docker-compose up -d" -Color "White"

Write-ColorOutput "`n🎉 Deployment script completed successfully!" -Color "Green"

