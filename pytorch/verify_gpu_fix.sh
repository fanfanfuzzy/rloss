#!/bin/bash

set -e

echo "🔍 Comprehensive GPU Access Verification for A100 + Ubuntu 22.04"
echo "=================================================================="

echo ""
echo "Step 1: Checking host GPU status..."
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "✅ nvidia-smi available"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits
else
    echo "❌ nvidia-smi not available - NVIDIA drivers not installed"
    exit 1
fi

echo ""
echo "Step 2: Checking nvidia-container-toolkit installation..."
if command -v nvidia-ctk >/dev/null 2>&1; then
    echo "✅ nvidia-container-toolkit installed"
    nvidia-ctk --version
else
    echo "❌ nvidia-container-toolkit not installed"
    echo "🔧 Running installation script..."
    ./install_nvidia_container_toolkit.sh
fi

echo ""
echo "Step 3: Checking Docker daemon configuration..."
if [ -f /etc/docker/daemon.json ]; then
    echo "✅ Docker daemon.json exists:"
    cat /etc/docker/daemon.json
    if grep -q "nvidia" /etc/docker/daemon.json; then
        echo "✅ NVIDIA runtime configured in daemon.json"
    else
        echo "❌ NVIDIA runtime not configured"
        echo "🔧 Running GPU access fix..."
        ./fix_docker_gpu_access.sh
    fi
else
    echo "❌ Docker daemon.json not found"
    echo "🔧 Running GPU access fix..."
    ./fix_docker_gpu_access.sh
fi

echo ""
echo "Step 4: Testing basic CUDA container access..."
if docker run --rm --gpus all nvidia/cuda:12.1.0-cudnn8-devel-ubuntu20.04 nvidia-smi >/dev/null 2>&1; then
    echo "✅ CUDA 12.1.0 + cuDNN 8 development container GPU access working"
else
    echo "❌ CUDA 12.1.0 + cuDNN 8 development container GPU access failed"
    echo "Trying alternative CUDA images..."
    if docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu20.04 nvidia-smi >/dev/null 2>&1; then
        echo "✅ CUDA 12.1.0 base container working"
    elif docker run --rm --gpus all nvidia/cuda:12.1.0-runtime-ubuntu20.04 nvidia-smi >/dev/null 2>&1; then
        echo "✅ CUDA 12.1.0 runtime container working"
    else
        echo "❌ All CUDA container tests failed"
        exit 1
    fi
fi

echo ""
echo "Step 5: Testing PyTorch GPU detection..."
if docker run --rm --gpus all pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"; then
    echo "✅ PyTorch GPU detection working"
else
    echo "❌ PyTorch GPU detection failed"
    exit 1
fi

echo ""
echo "Step 6: Testing rloss container GPU access..."
if docker images | grep -q "rloss:a100-ubuntu22.04"; then
    echo "✅ rloss image found"
    if docker run --rm --gpus all rloss:a100-ubuntu22.04 python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"; then
        echo "✅ rloss container GPU access working!"
    else
        echo "❌ rloss container GPU access failed"
        exit 1
    fi
else
    echo "⚠️  rloss image not found. Building..."
    make build
    echo "Testing rloss container after build..."
    if docker run --rm --gpus all rloss:a100-ubuntu22.04 python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"; then
        echo "✅ rloss container GPU access working!"
    else
        echo "❌ rloss container GPU access still failed"
        exit 1
    fi
fi

echo ""
echo "Step 7: Running comprehensive environment test..."
if make test-env; then
    echo "✅ Environment test passed"
else
    echo "❌ Environment test failed"
    exit 1
fi

echo ""
echo "🎉 All GPU access tests passed successfully!"
echo "✅ Your A100 + Docker + PyTorch environment is ready for training!"
echo ""
echo "Next steps:"
echo "1. Run training: make train-small"
echo "2. Monitor GPU usage: make monitor-gpu"
echo "3. Start Jupyter: make jupyter"
