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

.PHONY: all build run run-detached exec clean setup-data test-env train-small train-standard inference jupyter monitor-gpu debug logs

all: build run

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
