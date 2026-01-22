#!/bin/bash
# Bicleaner Service Launcher
# Auto-detects GPU and starts appropriate mode

set -e

cd "$(dirname "$0")/.."

# Check if nvidia-docker/GPU is available
check_gpu() {
    docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi > /dev/null 2>&1
}

echo "Bicleaner Service Launcher"
echo "=========================="

if check_gpu; then
    echo "[OK] GPU detected, starting with GPU support..."
    docker compose up -d
else
    echo "[!] GPU not available, starting in CPU mode..."
    docker compose --profile cpu up -d bicleaner-service-cpu
fi

echo ""
echo "Waiting for service to start..."
sleep 10

# Check health
if curl -sf http://localhost:8057/health > /dev/null; then
    echo "[OK] Service is healthy!"
    curl -s http://localhost:8057/health | python -m json.tool
else
    echo "[!] Service not ready yet. Check logs with: docker compose logs -f"
fi
