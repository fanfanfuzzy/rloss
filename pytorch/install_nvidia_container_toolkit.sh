#!/bin/bash

set -e

echo "🔧 Installing NVIDIA Container Toolkit for A100 + Ubuntu 22.04..."

if [[ $EUID -eq 0 ]]; then
    SUDO=""
else
    SUDO="sudo"
fi

echo "📦 Configuring NVIDIA Container Toolkit repository..."
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | $SUDO gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
$SUDO tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

echo "🔄 Updating package list..."
$SUDO apt-get update

echo "📥 Installing NVIDIA Container Toolkit packages..."
export NVIDIA_CONTAINER_TOOLKIT_VERSION=1.17.8-1
$SUDO apt-get install -y \
    nvidia-container-toolkit=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
    nvidia-container-toolkit-base=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
    libnvidia-container-tools=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
    libnvidia-container1=${NVIDIA_CONTAINER_TOOLKIT_VERSION}

echo "🐳 Configuring Docker runtime..."
$SUDO nvidia-ctk runtime configure --runtime=docker

echo "🔄 Restarting Docker daemon..."
$SUDO systemctl restart docker

echo "✅ Verifying NVIDIA Container Toolkit installation..."

if command -v nvidia-ctk >/dev/null 2>&1; then
    echo "✓ nvidia-ctk command available"
    nvidia-ctk --version
else
    echo "❌ nvidia-ctk command not found"
    exit 1
fi

if [ -f /etc/docker/daemon.json ]; then
    echo "✓ Docker daemon.json exists:"
    cat /etc/docker/daemon.json
else
    echo "❌ Docker daemon.json not found"
    exit 1
fi

echo "🧪 Testing GPU access in container..."
if docker run --rm --gpus all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility nvidia/cuda:12.1.0-cudnn8-devel-ubuntu20.04 nvidia-smi >/dev/null 2>&1; then
    echo "✅ GPU access in container working!"
else
    echo "❌ GPU access in container failed"
    echo "Troubleshooting steps:"
    echo "1. Check if NVIDIA drivers are installed: nvidia-smi"
    echo "2. Check Docker info: docker info | grep -i runtime"
    echo "3. Check container runtime: docker run --rm --gpus all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility nvidia/cuda:12.1.0-cudnn8-devel-ubuntu20.04 nvidia-smi"
    exit 1
fi

echo "🎉 NVIDIA Container Toolkit installation completed successfully!"
echo ""
echo "Next steps:"
echo "1. Test with your rloss container: make test-env"
echo "2. Run training: make train-small"
echo "3. Monitor GPU usage: make monitor-gpu"
