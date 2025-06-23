# Makefile for rloss Docker environment setup
# Optimized for A100 GPU on Ubuntu 22.04

# Configuration
DOCKER_IMAGE_NAME = rloss
DOCKER_TAG = a100-ubuntu22.04
CONTAINER_NAME = rloss-container
WORKSPACE_DIR = $(shell pwd)
DATA_DIR = /data/datasets
RESULTS_DIR = $(WORKSPACE_DIR)/results

# Docker build arguments
CUDA_VERSION = 11.8
PYTORCH_VERSION = 2.5.1
UBUNTU_VERSION = 22.04

# Default target
.PHONY: help
help:
	@echo "=== rloss Docker Environment for A100 + Ubuntu 22.04 ==="
	@echo ""
	@echo "Available targets:"
	@echo "  build          - Build Docker image"
	@echo "  run            - Run container interactively"
	@echo "  run-detached   - Run container in background"
	@echo "  exec           - Execute bash in running container"
	@echo "  stop           - Stop running container"
	@echo "  clean          - Remove container and image"
	@echo "  setup-data     - Create data directories"
	@echo "  test-env       - Test environment setup"
	@echo "  train-small    - Train with small images (40x40)"
	@echo "  train-standard - Train with standard images (513x513)"
	@echo "  inference      - Run inference on test images"
	@echo "  logs           - Show container logs"
	@echo "  status         - Show container status"
	@echo ""
	@echo "Environment:"
	@echo "  CUDA Version:    $(CUDA_VERSION)"
	@echo "  PyTorch Version: $(PYTORCH_VERSION)"
	@echo "  Ubuntu Version:  $(UBUNTU_VERSION)"
	@echo "  Workspace:       $(WORKSPACE_DIR)"
	@echo "  Data Directory:  $(DATA_DIR)"

# Build Docker image
.PHONY: build
build:
	@echo "Building Docker image for A100 + Ubuntu 22.04..."
	docker build \
		--build-arg CUDA_VERSION=$(CUDA_VERSION) \
		--build-arg PYTORCH_VERSION=$(PYTORCH_VERSION) \
		--build-arg UBUNTU_VERSION=$(UBUNTU_VERSION) \
		-t $(DOCKER_IMAGE_NAME):$(DOCKER_TAG) \
		-f Dockerfile.a100 .
	@echo "✅ Docker image built successfully!"

# Run container interactively
.PHONY: run
run: setup-data
	@echo "Starting rloss container interactively..."
	docker run -it --rm \
		--gpus all \
		--ipc=host \
		--ulimit memlock=-1 \
		--ulimit stack=67108864 \
		--name $(CONTAINER_NAME) \
		-v $(WORKSPACE_DIR):/workspace \
		-v $(DATA_DIR):/data/datasets \
		-v $(RESULTS_DIR):/workspace/results \
		-w /workspace \
		$(DOCKER_IMAGE_NAME):$(DOCKER_TAG) \
		/bin/bash

# Run container in background
.PHONY: run-detached
run-detached: setup-data
	@echo "Starting rloss container in background..."
	docker run -d \
		--gpus all \
		--ipc=host \
		--ulimit memlock=-1 \
		--ulimit stack=67108864 \
		--name $(CONTAINER_NAME) \
		-v $(WORKSPACE_DIR):/workspace \
		-v $(DATA_DIR):/data/datasets \
		-v $(RESULTS_DIR):/workspace/results \
		-w /workspace \
		$(DOCKER_IMAGE_NAME):$(DOCKER_TAG) \
		tail -f /dev/null
	@echo "✅ Container started in background"
	@echo "Use 'make exec' to enter the container"

# Execute bash in running container
.PHONY: exec
exec:
	@echo "Entering running container..."
	docker exec -it $(CONTAINER_NAME) /bin/bash

# Stop container
.PHONY: stop
stop:
	@echo "Stopping container..."
	-docker stop $(CONTAINER_NAME)
	-docker rm $(CONTAINER_NAME)
	@echo "✅ Container stopped"

# Clean up containers and images
.PHONY: clean
clean: stop
	@echo "Cleaning up Docker resources..."
	-docker rmi $(DOCKER_IMAGE_NAME):$(DOCKER_TAG)
	-docker system prune -f
	@echo "✅ Cleanup completed"

# Setup data directories
.PHONY: setup-data
setup-data:
	@echo "Setting up data directories..."
	mkdir -p $(DATA_DIR)/VOCdevkit/VOC2012
	mkdir -p $(DATA_DIR)/benchmark_RELEASE
	mkdir -p $(DATA_DIR)/cityscapes
	mkdir -p $(DATA_DIR)/coco
	mkdir -p $(RESULTS_DIR)
	@echo "✅ Data directories created"

# Test environment setup
.PHONY: test-env
test-env:
	@echo "Testing environment setup..."
	docker run --rm \
		--gpus all \
		-v $(WORKSPACE_DIR):/workspace \
		-w /workspace \
		$(DOCKER_IMAGE_NAME):$(DOCKER_TAG) \
		python test_environment.py

# Training targets
.PHONY: train-small
train-small:
	@echo "Starting training for small images (40x40)..."
	docker run --rm \
		--gpus all \
		--ipc=host \
		-v $(WORKSPACE_DIR):/workspace \
		-v $(DATA_DIR):/data/datasets \
		-v $(RESULTS_DIR):/workspace/results \
		-w /workspace/pytorch-deeplab_v3_plus \
		$(DOCKER_IMAGE_NAME):$(DOCKER_TAG) \
		python train_withdensecrfloss.py \
		--backbone mobilenet \
		--crop-size 64 \
		--base-size 64 \
		--batch-size 32 \
		--lr 0.01 \
		--epochs 80 \
		--densecrfloss 2e-9 \
		--rloss-scale 0.25 \
		--sigma-rgb 10 \
		--sigma-xy 20 \
		--dataset pascal \
		--checkname deeplab-small-images

.PHONY: train-standard
train-standard:
	@echo "Starting training for standard images (513x513)..."
	docker run --rm \
		--gpus all \
		--ipc=host \
		-v $(WORKSPACE_DIR):/workspace \
		-v $(DATA_DIR):/data/datasets \
		-v $(RESULTS_DIR):/workspace/results \
		-w /workspace/pytorch-deeplab_v3_plus \
		$(DOCKER_IMAGE_NAME):$(DOCKER_TAG) \
		python train_withdensecrfloss.py \
		--backbone mobilenet \
		--crop-size 513 \
		--base-size 513 \
		--batch-size 8 \
		--lr 0.007 \
		--epochs 60 \
		--densecrfloss 2e-9 \
		--rloss-scale 0.5 \
		--sigma-rgb 15 \
		--sigma-xy 80 \
		--dataset pascal \
		--checkname deeplab-standard

# Inference target
.PHONY: inference
inference:
	@echo "Running inference..."
	@if [ -z "$(IMAGE_PATH)" ]; then \
		echo "Error: Please specify IMAGE_PATH"; \
		echo "Usage: make inference IMAGE_PATH=/path/to/image.jpg [CHECKPOINT_PATH=/path/to/model.pth.tar]"; \
		exit 1; \
	fi
	docker run --rm \
		--gpus all \
		-v $(WORKSPACE_DIR):/workspace \
		-v $(DATA_DIR):/data/datasets \
		-v $(RESULTS_DIR):/workspace/results \
		-w /workspace/pytorch-deeplab_v3_plus \
		$(DOCKER_IMAGE_NAME):$(DOCKER_TAG) \
		python inference.py \
		--backbone mobilenet \
		--checkpoint $(or $(CHECKPOINT_PATH),/workspace/results/model_best.pth.tar) \
		--image_path $(IMAGE_PATH) \
		--output_directory /workspace/results/inference \
		--crop_size 513

# Show container logs
.PHONY: logs
logs:
	docker logs $(CONTAINER_NAME)

# Show container status
.PHONY: status
status:
	@echo "=== Container Status ==="
	@docker ps -a --filter name=$(CONTAINER_NAME) --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
	@echo ""
	@echo "=== GPU Status ==="
	@docker run --rm --gpus all $(DOCKER_IMAGE_NAME):$(DOCKER_TAG) nvidia-smi || echo "GPU not available"

# Development targets
.PHONY: dev-setup
dev-setup: build setup-data
	@echo "Development environment setup completed!"
	@echo "Run 'make run' to start interactive container"

.PHONY: download-data
download-data:
	@echo "Downloading PASCAL VOC2012 dataset..."
	docker run --rm \
		-v $(DATA_DIR):/data/datasets \
		-w /workspace \
		$(DOCKER_IMAGE_NAME):$(DOCKER_TAG) \
		bash -c "cd /workspace/data/VOC2012 && ./fetchVOC2012.sh"
	@echo "Downloading Pascal scribble annotations..."
	docker run --rm \
		-v $(DATA_DIR):/data/datasets \
		-w /workspace \
		$(DOCKER_IMAGE_NAME):$(DOCKER_TAG) \
		bash -c "cd /workspace/data/pascal_scribble && ./fetchPascalScribble.sh"

# Jupyter notebook support
.PHONY: jupyter
jupyter: setup-data
	@echo "Starting Jupyter notebook server..."
	docker run --rm \
		--gpus all \
		--ipc=host \
		-p 8888:8888 \
		-v $(WORKSPACE_DIR):/workspace \
		-v $(DATA_DIR):/data/datasets \
		-v $(RESULTS_DIR):/workspace/results \
		-w /workspace \
		$(DOCKER_IMAGE_NAME):$(DOCKER_TAG) \
		jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root

# Monitoring and debugging
.PHONY: monitor-gpu
monitor-gpu:
	@echo "Monitoring GPU usage..."
	docker run --rm \
		--gpus all \
		$(DOCKER_IMAGE_NAME):$(DOCKER_TAG) \
		watch -n 1 nvidia-smi

.PHONY: debug
debug: setup-data
	@echo "Starting debug container with additional tools..."
	docker run -it --rm \
		--gpus all \
		--ipc=host \
		--cap-add=SYS_PTRACE \
		--security-opt seccomp=unconfined \
		-v $(WORKSPACE_DIR):/workspace \
		-v $(DATA_DIR):/data/datasets \
		-v $(RESULTS_DIR):/workspace/results \
		-w /workspace \
		$(DOCKER_IMAGE_NAME):$(DOCKER_TAG) \
		/bin/bash

# Quick commands
.PHONY: quick-test
quick-test: build test-env

.PHONY: quick-train
quick-train: build train-small

# Show system information
.PHONY: info
info:
	@echo "=== System Information ==="
	@echo "Docker version:"
	@docker --version
	@echo ""
	@echo "NVIDIA Docker runtime:"
	@docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu22.04 nvidia-smi || echo "NVIDIA Docker not available"
	@echo ""
	@echo "Available images:"
	@docker images | grep $(DOCKER_IMAGE_NAME) || echo "No rloss images found"
