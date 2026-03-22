#!/usr/bin/env bash
# Full teardown — removes RayJob, RayCluster, namespace, and KubeRay operator.
#
# Usage:
#   bash deploy/teardown.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== Deleting RayJob ==="
kubectl delete -f "$SCRIPT_DIR/03-rayjob.yaml" --ignore-not-found

echo "=== Deleting RayCluster ==="
kubectl delete -f "$SCRIPT_DIR/02-raycluster.yaml" --ignore-not-found

echo "=== Deleting namespace ==="
kubectl delete -f "$SCRIPT_DIR/00-namespace.yaml" --ignore-not-found

echo "=== Uninstalling KubeRay operator ==="
helm uninstall kuberay-operator -n kuberay-system 2>/dev/null || true
kubectl delete ns kuberay-system --ignore-not-found

echo ""
echo "Teardown complete."
