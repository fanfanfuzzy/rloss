# Makefile for rloss Docker environment setup
# Optimized for A100 GPU on Ubuntu 22.04

# Variables
DOCKER_IMAGE_NAME := rloss
DOCKER_TAG        := a100-ubuntu22.04
IMAGE             := $(DOCKER_IMAGE_NAME):$(DOCKER_TAG)
CONTAINER_NAME    := rloss-container
WORKSPACE_DIR     := $(shell pwd)
DATA_DIR          := /data/datasets
RESULTS_DIR       := $(WORKSPACE_DIR)/results

# Build arguments
CUDA_VERSION      := 11.8
PYTORCH_VERSION   := 2.5.1
UBUNTU_VERSION    := 22.04

# Runtime options
GPU_OPTS          := --gpus all
# 古い nvidia-docker2 レガシーランタイムを使いたい場合
ifeq ($(USE_LEGACY_NVIDIA),true)
GPU_OPTS         += --runtime=nvidia
endif
# CPU モード: NO_GPU=true を渡すと GPU_OPTS を空に
ifeq ($(NO_GPU),true)
GPU_OPTS         :=
endif

.PHONY: all help build run run-detached exec clean setup-data test-env train-small train-standard inference jupyter monitor-gpu debug logs stop info status install-nvidia-toolkit fix-gpu-access test-gpu

all: build run

help:
	@echo "rloss Docker Environment - Available Commands:"
	@echo ""
	@echo "🏗️  Build & Setup:"
	@echo "  make build          - Build Docker image for A100 + Ubuntu 22.04"
	@echo "  make setup-data     - Create necessary data directories"
	@echo ""
	@echo "🚀 Container Management:"
	@echo "  make run            - Start interactive container"
	@echo "  make run-detached   - Start container in background"
	@echo "  make exec           - Enter running container"
	@echo "  make stop           - Stop running container"
	@echo "  make clean          - Remove Docker image"
	@echo ""
	@echo "🧪 Testing & Development:"
	@echo "  make test-env       - Test environment setup"
	@echo "  make debug          - Start debug container"
	@echo "  make logs           - Show container logs"
	@echo ""
	@echo "🎯 Training & Inference:"
	@echo "  make train-small    - Train with small images (40×40)"
	@echo "  make train-standard - Train with standard images (513×513)"
	@echo "  make inference IMAGE_PATH=/path/to/image.jpg - Run inference"
	@echo ""
	@echo "📊 Tools & Monitoring:"
	@echo "  make jupyter        - Start Jupyter Lab (port 8888)"
	@echo "  make monitor-gpu    - Monitor GPU usage"
	@echo "  make status         - Show container and GPU status"
	@echo "  make info           - Show system information"
	@echo ""
	@echo "🔧 GPU Troubleshooting:"
	@echo "  make install-nvidia-toolkit - Install NVIDIA Container Toolkit"
	@echo "  make fix-gpu-access - Fix Docker GPU access issues"
	@echo "  make test-gpu       - Test GPU access in container"
	@echo ""
	@echo "💡 Environment Variables:"
	@echo "  NO_GPU=true         - Run without GPU support"
	@echo "  USE_LEGACY_NVIDIA=true - Use legacy nvidia-docker2 runtime"
	@echo ""

build:
	@echo "Building Docker image for A100 + Ubuntu $(UBUNTU_VERSION)..."
	docker build \
		--build-arg CUDA_VERSION=$(CUDA_VERSION) \
		--build-arg PYTORCH_VERSION=$(PYTORCH_VERSION) \
		--build-arg UBUNTU_VERSION=$(UBUNTU_VERSION) \
		-t $(IMAGE) \
		-f Dockerfile.a100 .

# run ターゲットは build の後に実行
run: build setup-data
	@echo "Starting rloss container interactively..."
	docker run -it --rm \
		$(GPU_OPTS) \
		--ipc=host \
		--ulimit memlock=-1 \
		--ulimit stack=67108864 \
		--name $(CONTAINER_NAME) \
		-v $(WORKSPACE_DIR):/workspace \
		-v $(DATA_DIR):/data/datasets \
		-v $(RESULTS_DIR):/workspace/results \
		-w /workspace \
		$(IMAGE) bash

# run-detached も build -> setup-data の順に実行
run-detached: build setup-data
	@echo "Starting rloss container in background..."
	docker run -d \
		$(GPU_OPTS) \
		--ipc=host \
		--ulimit memlock=-1 \
		--ulimit stack=67108864 \
		--name $(CONTAINER_NAME) \
		-v $(WORKSPACE_DIR):/workspace \
		-v $(DATA_DIR):/data/datasets \
		-v $(RESULTS_DIR):/workspace/results \
		-w /workspace \
		$(IMAGE) tail -f /dev/null
	@echo "✅ Container started (background)"

exec:
	@echo "Entering running container..."
	docker exec -it $(CONTAINER_NAME) /bin/bash

clean:
	-@echo "Removing Docker image..."
	-@docker rmi $(IMAGE) || true
	@echo "✅ Cleanup completed."

setup-data:
	@echo "Setting up data directories..."
	mkdir -p $(DATA_DIR)/VOCdevkit/VOC2012 \
	         $(DATA_DIR)/benchmark_RELEASE \
	         $(DATA_DIR)/cityscapes \
	         $(DATA_DIR)/coco \
	         $(RESULTS_DIR)
	@echo "✅ Data directories created."

test-env:
	@echo "Testing environment setup..."
	docker run --rm \
		$(GPU_OPTS) \
		-v $(WORKSPACE_DIR):/workspace \
		-w /workspace \
		$(IMAGE) \
		python test_environment.py

train-small:
	@echo "Starting training for small images (40x40)..."
	docker run --rm \
		$(GPU_OPTS) \
		--ipc=host \
		-v $(WORKSPACE_DIR):/workspace \
		-v $(DATA_DIR):/data/datasets \
		-v $(RESULTS_DIR):/workspace/results \
		-w /workspace/pytorch-deeplab_v3_plus \
		$(IMAGE) \
		python train_withdensecrfloss.py \
		--backbone mobilenet \
		--crop-size 64 \
		--batch-size 32 \
		--lr 0.02 \
		--epochs 100 \
		--densecrfloss 2e-9 \
		--rloss-scale 0.25

train-standard:
	@echo "Starting training for standard images (513x513)..."
	docker run --rm \
		$(GPU_OPTS) \
		--ipc=host \
		-v $(WORKSPACE_DIR):/workspace \
		-v $(DATA_DIR):/data/datasets \
		-v $(RESULTS_DIR):/workspace/results \
		-w /workspace/pytorch-deeplab_v3_plus \
		$(IMAGE) \
		python train_withdensecrfloss.py \
		--backbone resnet \
		--crop-size 513 \
		--batch-size 16 \
		--lr 0.01 \
		--epochs 50 \
		--densecrfloss 2e-9 \
		--rloss-scale 1.0

inference:
	@if [ -z "$(IMAGE_PATH)" ]; then \ 
		echo "Error: Please specify IMAGE_PATH"; \ 
		echo "Usage: make inference IMAGE_PATH=/path/to/image.jpg [CHECKPOINT_PATH=/path/to/model.pth.tar]"; \ 
		exit 1; \ 
	fi
	docker run --rm \
		$(GPU_OPTS) \
		-v $(WORKSPACE_DIR):/workspace \
		-v $(DATA_DIR):/data/datasets \
		-v $(RESULTS_DIR):/workspace/results \
		-w /workspace/pytorch-deeplab_v3_plus \
		$(IMAGE) \
		python inference.py \
		--backbone mobilenet \
		--checkpoint $(CHECKPOINT_PATH) \
		--image_path $(IMAGE_PATH) \
		--output_directory ./results

jupyter:
	@echo "Starting Jupyter Lab..."
	docker run -d \
		$(GPU_OPTS) \
		--ipc=host \
		-p 8888:8888 \
		--name rloss-jupyter \
		-v $(WORKSPACE_DIR):/workspace \
		-v $(DATA_DIR):/data/datasets \
		-v $(RESULTS_DIR):/workspace/results \
		-w /workspace \
		$(IMAGE) \
		jupyter lab --ip=0.0.0.0 --allow-root --no-browser --NotebookApp.token='' --NotebookApp.password=''
	@echo "✅ Jupyter Lab is running on port 8888"

monitor-gpu:
	@echo "Monitoring GPU usage..."
	watch -n 1 nvidia-smi

debug:
	@echo "Starting debug container..."
	docker run -it --rm \
		$(GPU_OPTS) \
		--ipc=host \
		-v $(WORKSPACE_DIR):/workspace \
		-w /workspace \
		$(IMAGE) bash

logs:
	@echo "Showing container logs..."
	docker logs -f $(CONTAINER_NAME)

stop:
	@echo "Stopping container..."
	-@docker stop $(CONTAINER_NAME) || true
	-@docker rm $(CONTAINER_NAME) || true
	@echo "✅ Container stopped and removed."

status:
	@echo "=== Container Status ==="
	@docker ps -a --filter name=$(CONTAINER_NAME) --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
	@echo ""
	@echo "=== GPU Status ==="
	@if command -v nvidia-smi >/dev/null 2>&1; then \
		nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits; \
	else \
		echo "nvidia-smi not available"; \
	fi

info:
	@echo "=== rloss Environment Information ==="
	@echo "Docker Image: $(IMAGE)"
	@echo "Container Name: $(CONTAINER_NAME)"
	@echo "CUDA Version: $(CUDA_VERSION)"
	@echo "PyTorch Version: $(PYTORCH_VERSION)"
	@echo "Ubuntu Version: $(UBUNTU_VERSION)"
	@echo "Workspace: $(WORKSPACE_DIR)"
	@echo "Data Directory: $(DATA_DIR)"
	@echo "Results Directory: $(RESULTS_DIR)"
	@echo ""
	@echo "=== Docker Images ==="
	@docker images | grep -E "(REPOSITORY|$(DOCKER_IMAGE_NAME))" || echo "No rloss images found"

install-nvidia-toolkit:
	@echo "Installing NVIDIA Container Toolkit..."
	@chmod +x install_nvidia_container_toolkit.sh
	@./install_nvidia_container_toolkit.sh

fix-gpu-access:
	@echo "Fixing Docker GPU access..."
	@chmod +x fix_docker_gpu_access.sh
	@./fix_docker_gpu_access.sh

test-gpu:
	@echo "Testing GPU access in containers..."
	@echo "=== Host GPU Status ==="
	@if command -v nvidia-smi >/dev/null 2>&1; then \
		nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits; \
	else \
		echo "nvidia-smi not available on host"; \
	fi
	@echo ""
	@echo "=== Docker Runtime Info ==="
	@docker info | grep -i runtime || echo "No runtime info found"
	@echo ""
	@echo "=== Basic CUDA Container Test ==="
	@docker run --rm $(GPU_OPTS) nvidia/cuda:11.8-base-ubuntu22.04 nvidia-smi || echo "Basic CUDA test failed"
	@echo ""
	@echo "=== PyTorch GPU Test ==="
	@docker run --rm $(GPU_OPTS) pytorch/pytorch:2.5.1-cuda11.8-cudnn9-devel python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')" || echo "PyTorch GPU test failed"
	@echo ""
	@echo "=== rloss Container Test ==="
	@if docker images | grep -q "$(IMAGE)"; then \
		docker run --rm $(GPU_OPTS) -v $(WORKSPACE_DIR):/workspace -w /workspace $(IMAGE) python test_environment.py || echo "rloss container test failed"; \
	else \
		echo "rloss image not found. Run 'make build' first."; \
	fi
