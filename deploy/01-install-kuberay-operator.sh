#!/usr/bin/env bash
set -euo pipefail

KUBERAY_VERSION="1.5.1"

# Idempotent — safe to run multiple times
if helm repo list 2>/dev/null | grep -q kuberay; then
    echo "kuberay repo already added, updating..."
    helm repo update kuberay
else
    echo "Adding kuberay helm repo..."
    helm repo add kuberay https://ray-project.github.io/kuberay-helm/
    helm repo update
fi

if helm status kuberay-operator -n kuberay-system &>/dev/null; then
    echo "kuberay-operator already installed, upgrading..."
    helm upgrade kuberay-operator kuberay/kuberay-operator \
        --version "${KUBERAY_VERSION}" \
        --namespace kuberay-system \
        --wait
else
    echo "Installing kuberay-operator v${KUBERAY_VERSION}..."
    helm install kuberay-operator kuberay/kuberay-operator \
        --version "${KUBERAY_VERSION}" \
        --namespace kuberay-system \
        --create-namespace \
        --wait
fi

echo "Verifying operator pod..."
kubectl get pods -n kuberay-system
echo ""
echo "Verifying CRDs..."
kubectl get crd | grep ray || true
echo ""
echo "KubeRay operator v${KUBERAY_VERSION} ready."
