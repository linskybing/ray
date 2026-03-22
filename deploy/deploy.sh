#!/usr/bin/env bash
# Master deployment script — orchestrates the full KubeRay pipeline deployment.
#
# Usage:
#   bash deploy/deploy.sh
#
# Steps:
#   1. Create namespace
#   2. Install KubeRay operator (idempotent)
#   3. Build and push Docker image to Harbor
#   4. Deploy RayCluster
#   5. Submit RayJob
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
IMAGE="gpu1:30003/library/ray-pipeline:latest"

echo "=== Step 1: Create namespace ==="
kubectl apply -f "$SCRIPT_DIR/00-namespace.yaml"

echo ""
echo "=== Step 2: Install KubeRay operator ==="
bash "$SCRIPT_DIR/01-install-kuberay-operator.sh"

echo ""
echo "=== Step 3: Build and push Docker image ==="
docker build -t "$IMAGE" "$PROJECT_DIR"
docker push "$IMAGE"

echo ""
echo "=== Step 4: Deploy RayCluster ==="
kubectl apply -f "$SCRIPT_DIR/02-raycluster.yaml"
echo "Waiting for head pod to be ready..."
kubectl -n ray-pipeline wait --for=condition=Ready pod \
    -l ray.io/node-type=head --timeout=300s

echo ""
echo "=== Step 5: Submit RayJob ==="
kubectl apply -f "$SCRIPT_DIR/03-rayjob.yaml"

echo ""
echo "============================================"
echo "  Deployment complete!"
echo ""
echo "  Monitor pods:    kubectl -n ray-pipeline get pods -w"
echo "  Job logs:        kubectl -n ray-pipeline logs -f -l ray.io/job-type=submitter"
echo "  Dashboard:       kubectl -n ray-pipeline port-forward svc/inference-pipeline-head-svc 8265:8265"
echo "  Copy trace:      kubectl cp ray-pipeline/<head-pod>:/tmp/ray/pipeline_trace.json ./pipeline_trace.json"
echo "  View trace:      https://ui.perfetto.dev"
echo "============================================"
