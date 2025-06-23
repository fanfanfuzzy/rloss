#!/bin/bash

set -e

echo "🔧 Fixing Docker GPU access for A100 GPUs..."

if [[ $EUID -eq 0 ]]; then
    SUDO=""
else
    SUDO="sudo"
fi

echo "📋 Current Docker configuration:"
if [ -f /etc/docker/daemon.json ]; then
    echo "Current daemon.json:"
    cat /etc/docker/daemon.json
else
    echo "No daemon.json found, creating new one..."
fi

if [ -f /etc/docker/daemon.json ]; then
    echo "📁 Backing up existing daemon.json..."
    $SUDO cp /etc/docker/daemon.json /etc/docker/daemon.json.backup.$(date +%Y%m%d_%H%M%S)
fi

if ! command -v nvidia-ctk >/dev/null 2>&1; then
    echo "❌ nvidia-container-toolkit not found. Installing..."
    ./install_nvidia_container_toolkit.sh
    exit 0
fi

echo "🐳 Reconfiguring Docker runtime..."
$SUDO nvidia-ctk runtime configure --runtime=docker

echo "📋 Updated Docker configuration:"
cat /etc/docker/daemon.json

echo "🔄 Restarting Docker daemon..."
$SUDO systemctl restart docker

sleep 3

echo "📋 Docker runtime info:"
docker info | grep -i runtime || echo "No runtime info found"

echo "🧪 Testing GPU access..."

echo "Test 1: CUDA + cuDNN development container"
if docker run --rm --gpus all nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 nvidia-smi; then
    echo "✅ CUDA + cuDNN development GPU access working"
elif docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi; then
    echo "✅ CUDA 11.8.0 base access working"
else
    echo "❌ GPU access failed"
fi

echo "Test 2: PyTorch GPU detection"
if docker run --rm --gpus all pytorch/pytorch:2.5.1-cuda11.8-cudnn9-devel python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"; then
    echo "✅ PyTorch GPU detection working"
else
    echo "❌ PyTorch GPU detection failed"
fi

echo "Test 3: rloss container GPU access"
if docker images | grep -q "rloss:a100-ubuntu22.04"; then
    if docker run --rm --gpus all rloss:a100-ubuntu22.04 python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"; then
        echo "✅ rloss container GPU access working"
    else
        echo "❌ rloss container GPU access failed"
    fi
else
    echo "⚠️  rloss container not built yet. Run 'make build' first."
fi

echo "🎉 Docker GPU access fix completed!"
echo ""
echo "If tests still fail, try:"
echo "1. Reboot the system to ensure all drivers are loaded properly"
echo "2. Check NVIDIA driver version: nvidia-smi"
echo "3. Check Docker version: docker --version"
echo "4. Run: make test-env"
