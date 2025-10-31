<#
.SYNOPSIS
    Monitor and manage Dual RAG LLM System on Azure

.DESCRIPTION
    PowerShell script for monitoring, managing, and troubleshooting the Dual RAG system
    deployed on Azure (VM or AKS).

.PARAMETER DeploymentType
    Type of deployment: VM or AKS

.PARAMETER ResourceGroupName
    Name of the Azure Resource Group

.PARAMETER VMName
    Name of the VM (if DeploymentType is VM)

.PARAMETER ClusterName
    Name of the AKS cluster (if DeploymentType is AKS)

.PARAMETER Action
    Action to perform: Status, Logs, Restart, Scale, Metrics

.EXAMPLE
    .\Monitor-DualRAG.ps1 -DeploymentType "AKS" -ResourceGroupName "rg-dualrag" -ClusterName "aks-dualrag" -Action "Status"

.NOTES
    Author: Adrian Johnson <adrian207@gmail.com>
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("VM", "AKS")]
    [string]$DeploymentType,

    [Parameter(Mandatory = $true)]
    [string]$ResourceGroupName,

    [Parameter(Mandatory = $false)]
    [string]$VMName,

    [Parameter(Mandatory = $false)]
    [string]$ClusterName,

    [Parameter(Mandatory = $true)]
    [ValidateSet("Status", "Logs", "Restart", "Scale", "Metrics", "Health")]
    [string]$Action
)

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

Write-ColorOutput "`n📊 Dual RAG LLM - Monitoring & Management" -Color "Cyan"
Write-ColorOutput "========================================`n" -Color "Cyan"

# Check Azure login
try {
    $context = Get-AzContext
    if (-not $context) {
        throw "Not logged in"
    }
} catch {
    Connect-AzAccount
}

if ($DeploymentType -eq "VM") {
    if (-not $VMName) {
        Write-ColorOutput "❌ VMName is required for VM deployment type" -Color "Red"
        exit 1
    }

    switch ($Action) {
        "Status" {
            Write-ColorOutput "📋 VM Status" -Color "Cyan"
            $vm = Get-AzVM -ResourceGroupName $ResourceGroupName -Name $VMName -Status
            
            Write-ColorOutput "`nVM Information:" -Color "White"
            Write-ColorOutput "  Name: $($vm.Name)" -Color "White"
            Write-ColorOutput "  Size: $($vm.HardwareProfile.VmSize)" -Color "White"
            Write-ColorOutput "  Location: $($vm.Location)" -Color "White"
            
            $powerState = ($vm.Statuses | Where-Object { $_.Code -like "PowerState/*" }).DisplayStatus
            $provisioningState = ($vm.Statuses | Where-Object { $_.Code -like "ProvisioningState/*" }).DisplayStatus
            
            Write-ColorOutput "  Power State: $powerState" -Color $(if ($powerState -eq "VM running") { "Green" } else { "Yellow" })
            Write-ColorOutput "  Provisioning State: $provisioningState" -Color "White"

            # Get Public IP
            $nic = Get-AzNetworkInterface -ResourceId $vm.NetworkProfile.NetworkInterfaces[0].Id
            $pipId = $nic.IpConfigurations[0].PublicIpAddress.Id
            if ($pipId) {
                $pip = Get-AzPublicIpAddress -ResourceId $pipId
                Write-ColorOutput "  Public IP: $($pip.IpAddress)" -Color "White"
            }
        }

        "Logs" {
            Write-ColorOutput "📜 Fetching VM logs..." -Color "Cyan"
            $pip = Get-AzPublicIpAddress -ResourceGroupName $ResourceGroupName | Where-Object { $_.Name -like "*$VMName*" } | Select-Object -First 1
            
            if ($pip) {
                Write-ColorOutput "`nTo view logs, SSH into the VM:" -Color "Yellow"
                Write-ColorOutput "  ssh <username>@$($pip.IpAddress)" -Color "White"
                Write-ColorOutput "  cd /opt/dual-rag-llm" -Color "White"
                Write-ColorOutput "  docker-compose logs -f" -Color "White"
            }
        }

        "Restart" {
            Write-ColorOutput "🔄 Restarting VM..." -Color "Cyan"
            Restart-AzVM -ResourceGroupName $ResourceGroupName -Name $VMName
            Write-ColorOutput "✓ VM restarted" -Color "Green"
        }

        "Metrics" {
            Write-ColorOutput "📈 VM Metrics" -Color "Cyan"
            $endTime = Get-Date
            $startTime = $endTime.AddHours(-1)
            
            # CPU
            $cpuMetrics = Get-AzMetric -ResourceId (Get-AzVM -ResourceGroupName $ResourceGroupName -Name $VMName).Id `
                -MetricName "Percentage CPU" -StartTime $startTime -EndTime $endTime -TimeGrain 00:05:00
            
            Write-ColorOutput "`nCPU Usage (last hour):" -Color "White"
            $avgCpu = ($cpuMetrics.Data.Average | Measure-Object -Average).Average
            Write-ColorOutput "  Average: $([math]::Round($avgCpu, 2))%" -Color "White"
            
            # Memory
            $memMetrics = Get-AzMetric -ResourceId (Get-AzVM -ResourceGroupName $ResourceGroupName -Name $VMName).Id `
                -MetricName "Available Memory Bytes" -StartTime $startTime -EndTime $endTime -TimeGrain 00:05:00
            
            if ($memMetrics.Data) {
                $avgMem = ($memMetrics.Data.Average | Measure-Object -Average).Average
                Write-ColorOutput "  Available Memory: $([math]::Round($avgMem / 1GB, 2)) GB" -Color "White"
            }
        }

        "Health" {
            Write-ColorOutput "🏥 Health Check" -Color "Cyan"
            $pip = Get-AzPublicIpAddress -ResourceGroupName $ResourceGroupName | Where-Object { $_.Name -like "*$VMName*" } | Select-Object -First 1
            
            if ($pip) {
                try {
                    $health = Invoke-RestMethod -Uri "http://$($pip.IpAddress):8000/health" -TimeoutSec 10
                    Write-ColorOutput "✓ API is healthy" -Color "Green"
                    Write-ColorOutput "  Status: $($health.status)" -Color "White"
                    Write-ColorOutput "  Ollama: $(if ($health.ollama_available) { '✓' } else { '✗' })" -Color $(if ($health.ollama_available) { "Green" } else { "Red" })
                } catch {
                    Write-ColorOutput "❌ API is not responding" -Color "Red"
                }
            }
        }
    }

} elseif ($DeploymentType -eq "AKS") {
    if (-not $ClusterName) {
        Write-ColorOutput "❌ ClusterName is required for AKS deployment type" -Color "Red"
        exit 1
    }

    # Get AKS credentials
    Import-AzAksCredential -ResourceGroupName $ResourceGroupName -Name $ClusterName -Force | Out-Null

    switch ($Action) {
        "Status" {
            Write-ColorOutput "📋 AKS Cluster Status" -Color "Cyan"
            
            Write-ColorOutput "`nNodes:" -Color "White"
            kubectl get nodes
            
            Write-ColorOutput "`nPods in dualrag namespace:" -Color "White"
            kubectl get pods -n dualrag
            
            Write-ColorOutput "`nServices:" -Color "White"
            kubectl get services -n dualrag
        }

        "Logs" {
            Write-ColorOutput "📜 Pod Logs" -Color "Cyan"
            
            $pods = kubectl get pods -n dualrag -o json | ConvertFrom-Json
            
            Write-ColorOutput "`nAvailable pods:" -Color "White"
            foreach ($pod in $pods.items) {
                Write-ColorOutput "  - $($pod.metadata.name) ($($pod.status.phase))" -Color "White"
            }
            
            Write-ColorOutput "`nTo view logs for a specific pod:" -Color "Yellow"
            Write-ColorOutput "  kubectl logs -f <pod-name> -n dualrag" -Color "White"
        }

        "Restart" {
            Write-ColorOutput "🔄 Restarting deployments..." -Color "Cyan"
            kubectl rollout restart deployment/dualrag-api -n dualrag
            kubectl rollout restart deployment/nginx -n dualrag
            Write-ColorOutput "✓ Deployments restarted" -Color "Green"
        }

        "Scale" {
            $replicas = Read-Host "Enter number of API replicas"
            Write-ColorOutput "📈 Scaling to $replicas replicas..." -Color "Cyan"
            kubectl scale deployment/dualrag-api --replicas=$replicas -n dualrag
            Write-ColorOutput "✓ Scaled successfully" -Color "Green"
        }

        "Metrics" {
            Write-ColorOutput "📊 Cluster Metrics" -Color "Cyan"
            
            Write-ColorOutput "`nNode Resource Usage:" -Color "White"
            kubectl top nodes
            
            Write-ColorOutput "`nPod Resource Usage:" -Color "White"
            kubectl top pods -n dualrag
        }

        "Health" {
            Write-ColorOutput "🏥 Health Check" -Color "Cyan"
            
            # Port forward temporarily
            $job = Start-Job -ScriptBlock {
                kubectl port-forward svc/dualrag-api 18000:8000 -n dualrag
            }
            
            Start-Sleep -Seconds 5
            
            try {
                $health = Invoke-RestMethod -Uri "http://localhost:18000/health" -TimeoutSec 10
                Write-ColorOutput "✓ API is healthy" -Color "Green"
                Write-ColorOutput "  Status: $($health.status)" -Color "White"
            } catch {
                Write-ColorOutput "❌ API is not responding" -Color "Red"
            } finally {
                Stop-Job $job
                Remove-Job $job
            }
        }
    }
}

Write-ColorOutput "`n✅ Operation completed!" -Color "Green"

