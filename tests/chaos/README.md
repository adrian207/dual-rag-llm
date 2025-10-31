# Chaos Engineering Tests

Chaos engineering tests for Kubernetes deployments using Chaos Mesh.

**Author:** Adrian Johnson <adrian207@gmail.com>

## Prerequisites

```bash
# Install Chaos Mesh
kubectl create ns chaos-mesh
helm repo add chaos-mesh https://charts.chaos-mesh.org
helm install chaos-mesh chaos-mesh/chaos-mesh -n chaos-mesh --version 2.6.0
```

## Running Tests

```bash
# Apply pod failure chaos
kubectl apply -f tests/chaos/pod-failure.yaml

# Apply network latency chaos
kubectl apply -f tests/chaos/network-latency.yaml

# Apply stress chaos
kubectl apply -f tests/chaos/stress-chaos.yaml

# Monitor chaos experiments
kubectl get podchaos -n default
kubectl get networkchaos -n default
kubectl get stresschaos -n default

# Cleanup
kubectl delete -f tests/chaos/
```

## Test Scenarios

1. **Pod Failure** - Random pod termination
2. **Network Latency** - Introduce network delays
3. **Network Partition** - Network isolation
4. **Stress CPU** - CPU resource exhaustion
5. **Stress Memory** - Memory pressure
6. **I/O Stress** - Disk I/O pressure

## Expected Outcomes

- System should recover automatically
- No data loss
- Graceful degradation
- Monitoring alerts triggered

