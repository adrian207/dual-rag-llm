#!/bin/bash
# Update LLM models in Kubernetes deployment
# Author: Adrian Johnson <adrian207@gmail.com>

set -e

echo "=== Model Update Script ==="
echo ""

NAMESPACE="dual-rag"
STRATEGY="${1:-rolling}"  # rolling or blue-green

case $STRATEGY in
  rolling)
    echo "Using rolling update strategy..."
    echo ""
    
    # Get current Ollama pods
    PODS=$(kubectl get pods -n $NAMESPACE -l app=ollama -o name)
    
    for POD in $PODS; do
      echo "Updating models in $POD..."
      
      # Pull new model versions
      kubectl exec -n $NAMESPACE $POD -- ollama pull qwen2.5-coder:32b-q4_K_M
      kubectl exec -n $NAMESPACE $POD -- ollama pull deepseek-coder-v2:33b-q4_K_M
      
      echo "✓ Models updated in $POD"
    done
    
    # Restart RAG service to reload model cache
    echo ""
    echo "Restarting RAG service..."
    kubectl rollout restart deployment/rag-deployment -n $NAMESPACE
    kubectl rollout status deployment/rag-deployment -n $NAMESPACE
    
    echo "✓ Rolling update complete"
    ;;
    
  blue-green)
    echo "Using blue-green deployment strategy..."
    echo ""
    
    # Check if green (current) deployment exists
    if kubectl get statefulset ollama-green -n $NAMESPACE >/dev/null 2>&1; then
      CURRENT="green"
      NEW="blue"
    else
      CURRENT="blue"
      NEW="green"
    fi
    
    echo "Current deployment: $CURRENT"
    echo "New deployment: $NEW"
    echo ""
    
    # Copy current StatefulSet and modify
    kubectl get statefulset ollama-$CURRENT -n $NAMESPACE -o yaml | \
      sed "s/ollama-$CURRENT/ollama-$NEW/g" | \
      kubectl apply -f -
    
    # Wait for new deployment
    kubectl wait --for=condition=ready pod -l app=ollama-$NEW -n $NAMESPACE --timeout=300s
    
    # Pull models in new deployment
    echo "Pulling models in new deployment..."
    NEW_POD=$(kubectl get pods -n $NAMESPACE -l app=ollama-$NEW -o name | head -1)
    kubectl exec -n $NAMESPACE $NEW_POD -- ollama pull qwen2.5-coder:32b-q4_K_M
    kubectl exec -n $NAMESPACE $NEW_POD -- ollama pull deepseek-coder-v2:33b-q4_K_M
    
    # Test new deployment
    echo "Testing new deployment..."
    kubectl run test-pod --rm -i --restart=Never \
      --image=curlimages/curl \
      -- curl -f http://ollama-$NEW-service:11434/api/tags
    
    # Switch service
    echo "Switching traffic to new deployment..."
    kubectl patch service ollama-service -n $NAMESPACE \
      -p "{\"spec\":{\"selector\":{\"app\":\"ollama-$NEW\"}}}"
    
    # Wait and verify
    sleep 30
    
    # Cleanup old deployment
    echo "Cleaning up old deployment..."
    kubectl delete statefulset ollama-$CURRENT -n $NAMESPACE
    
    echo "✓ Blue-green deployment complete"
    ;;
    
  *)
    echo "Unknown strategy: $STRATEGY"
    echo "Usage: $0 [rolling|blue-green]"
    exit 1
    ;;
esac

echo ""
echo "Model update complete!"
echo ""
echo "Verify:"
echo "  kubectl exec -n $NAMESPACE ollama-0 -- ollama list"
echo "  curl http://\$(kubectl get svc -n $NAMESPACE rag-service -o jsonpath='{.status.loadBalancer.ingress[0].ip}'):8000/health"
echo ""

