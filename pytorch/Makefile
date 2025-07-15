# Makefile for rloss Docker environment setup
# Optimized for A100 GPU on Ubuntu 22.04

# Variables
DOCKER_IMAGE_NAME := rloss
DOCKER_TAG        := a100-ubuntu22.04
IMAGE             := $(DOCKER_IMAGE_NAME):$(DOCKER_TAG)
CONTAINER_NAME    := rloss-container
WORKSPACE_DIR     := $(shell pwd)
DATA_DIR          := $(HOME)/datasets
RESULTS_DIR       := $(WORKSPACE_DIR)/results

# Build arguments
CUDA_VERSION      := 12.1
PYTORCH_VERSION   := 2.3.0
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

.PHONY: all help build run run-detached exec clean setup-data test-env train-small train-standard inference jupyter monitor-gpu debug logs stop info status install-nvidia-toolkit fix-gpu-access test-gpu train-small-direct train-standard-direct inference-direct test-env-direct status-direct download-data download-models train-small-wandb train-standard-wandb setup-wandb check-wandb

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
	@echo "  make inference IMAGE_PATH=/path/to/image.jpg - Run inference with bicubic interpolation"
	@echo ""
	@echo "🎯 Container-friendly Training & Inference (use inside container):"
	@echo "  make train-small-direct    - Train with small images (direct execution)"
	@echo "  make train-standard-direct - Train with standard images (direct execution)"
	@echo "  make inference-direct [IMAGE_PATH=/path/to/image.jpg] - Run inference with bicubic interpolation (direct execution)"
	@echo "  make test-env-direct       - Test environment (direct execution)"
	@echo "  make status-direct         - Check container environment status"
	@echo ""
	@echo "📊 Weights & Biases Integration:"
	@echo "  make train-small-wandb     - Train with small images + W&B logging"
	@echo "  make train-standard-wandb  - Train with standard images + W&B logging"
	@echo "  make setup-wandb           - Setup W&B credentials securely"
	@echo "  make check-wandb           - Check W&B setup status"
	@echo ""
	@echo "⚠️  Note: Use '-direct' targets when working inside Docker containers"
	@echo "    Regular targets use nested Docker and will fail inside containers"
	@echo ""
	@echo "📦 Data Management:"
	@echo "  make download-data  - Download and setup PASCAL VOC2012 dataset"
	@echo "  make clean-data     - Clean dataset directories for fresh download"
	@echo "  make download-models - Show instructions for downloading pre-trained models"
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
		-e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
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
		-e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
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
	@echo "⚠️  Note: This target uses nested Docker and may fail inside containers."
	@echo "    Use 'make test-env-direct' when running inside a container."
	@echo "Testing environment setup..."
	docker run --rm \
		$(GPU_OPTS) \
		-e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
		-v $(WORKSPACE_DIR):/workspace \
		-w /workspace \
		$(IMAGE) \
		python test_environment.py

train-small:
	@echo "⚠️  Note: This target uses nested Docker and may fail inside containers."
	@echo "    Use 'make train-small-direct' when running inside a container."
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
	@echo "⚠️  Note: This target uses nested Docker and may fail inside containers."
	@echo "    Use 'make train-standard-direct' when running inside a container."
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
	@echo "⚠️  Note: This target uses nested Docker and may fail inside containers."
	@echo "    Use 'make inference-direct' when running inside a container."
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
		--output_directory ./results \
		--use_bicubic

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
	@echo "=== CUDA 12.1 Base Container Test ==="
	@docker run --rm $(GPU_OPTS) -e NVIDIA_DRIVER_CAPABILITIES=compute,utility nvidia/cuda:12.1.0-base-ubuntu20.04 nvidia-smi || echo "CUDA 12.1 base test failed"
	@echo ""
	@echo "=== PyTorch GPU Test ==="
	@docker run --rm $(GPU_OPTS) -e NVIDIA_DRIVER_CAPABILITIES=compute,utility pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')" || echo "PyTorch GPU test failed"
	@echo ""
	@echo "=== rloss Container Test ==="
	@if docker images | grep -q "$(IMAGE)"; then \
		docker run --rm $(GPU_OPTS) -e NVIDIA_DRIVER_CAPABILITIES=compute,utility -v $(WORKSPACE_DIR):/workspace -w /workspace $(IMAGE) python test_environment.py || echo "rloss container test failed"; \
	else \
		echo "rloss image not found. Run 'make build' first."; \
	fi

# Container-friendly targets (execute inside running container)
train-small-direct:
	@echo "Training with small images (40x40) - direct execution..."
	@echo "=== Folder Structure Debug - Training Setup ==="
	@echo "Current working directory:"
	pwd
	@echo "Dataset path expected by training script:"
	@echo "  /data/datasets/VOCdevkit/VOC2012/"
	@echo "Dataset directory structure:"
	ls -la /data/datasets/VOCdevkit/VOC2012/ 2>/dev/null || echo "  ❌ /data/datasets/VOCdevkit/VOC2012/ does not exist"
	@echo "ImageSets directory:"
	ls -la /data/datasets/VOCdevkit/VOC2012/ImageSets/ 2>/dev/null || echo "  ❌ ImageSets directory does not exist"
	@echo "SegmentationAug directory:"
	ls -la /data/datasets/VOCdevkit/VOC2012/ImageSets/SegmentationAug/ 2>/dev/null || echo "  ❌ SegmentationAug directory does not exist"
	@echo "Required train.txt file:"
	ls -la /data/datasets/VOCdevkit/VOC2012/ImageSets/SegmentationAug/train.txt 2>/dev/null || echo "  ❌ train.txt NOT FOUND"
	@echo "Scribble annotations directory (required for training):"
	ls -la /data/datasets/VOCdevkit/VOC2012/pascal_2012_scribble/ | head -3 2>/dev/null || echo "  ❌ pascal_2012_scribble directory NOT FOUND"
	@echo "Available .txt files in ImageSets:"
	find /data/datasets/VOCdevkit/VOC2012/ImageSets/ -name "*.txt" 2>/dev/null || echo "  No .txt files found"
	@echo ""
	cd pytorch-deeplab_v3_plus && python train_withdensecrfloss.py \
		--backbone mobilenet \
		--crop-size 64 \
		--batch-size 32 \
		--lr 0.02 \
		--epochs 100 \
		--densecrfloss 2e-9 \
		--rloss-scale 0.25

train-standard-direct:
	@echo "Training with standard images (513x513) - direct execution..."
	@echo "=== Folder Structure Debug - Training Setup ==="
	@echo "Current working directory:"
	pwd
	@echo "Dataset path expected by training script:"
	@echo "  /data/datasets/VOCdevkit/VOC2012/"
	@echo "Dataset directory structure:"
	ls -la /data/datasets/VOCdevkit/VOC2012/ 2>/dev/null || echo "  ❌ /data/datasets/VOCdevkit/VOC2012/ does not exist"
	@echo "ImageSets directory:"
	ls -la /data/datasets/VOCdevkit/VOC2012/ImageSets/ 2>/dev/null || echo "  ❌ ImageSets directory does not exist"
	@echo "SegmentationAug directory:"
	ls -la /data/datasets/VOCdevkit/VOC2012/ImageSets/SegmentationAug/ 2>/dev/null || echo "  ❌ SegmentationAug directory does not exist"
	@echo "Required train.txt file:"
	ls -la /data/datasets/VOCdevkit/VOC2012/ImageSets/SegmentationAug/train.txt 2>/dev/null || echo "  ❌ train.txt NOT FOUND"
	@echo "Scribble annotations directory (required for training):"
	ls -la /data/datasets/VOCdevkit/VOC2012/pascal_2012_scribble/ | head -3 2>/dev/null || echo "  ❌ pascal_2012_scribble directory NOT FOUND"
	@echo ""
	cd pytorch-deeplab_v3_plus && python train_withdensecrfloss.py \
		--backbone resnet \
		--crop-size 513 \
		--batch-size 16 \
		--lr 0.01 \
		--epochs 50 \
		--densecrfloss 2e-9 \
		--rloss-scale 1.0

inference-direct:
	@echo "Ensuring test image exists..."
	python create_test_image_if_missing.py
	@echo "=== Folder Structure Debug - Inference Setup ==="
	@echo "Current working directory:"
	pwd
	@echo "Test image location:"
	ls -la pytorch-deeplab_v3_plus/misc/test.png 2>/dev/null || echo "  ❌ Test image does not exist"
	@echo "Results directory:"
	ls -la ./results/ 2>/dev/null || echo "  Results directory does not exist (will be created)"
	@echo "Dataset directory (for model loading):"
	ls -la /data/datasets/VOCdevkit/VOC2012/ 2>/dev/null || echo "  ❌ Dataset directory does not exist"
	@echo ""
	if [ -z "$(IMAGE_PATH)" ]; then \
		echo "Using default test image..."; \
		cd pytorch-deeplab_v3_plus && python inference.py \
			--backbone mobilenet \
			--image_path ./misc/test.png \
			--output_directory ./results; \
	else \
		echo "Running inference on $(IMAGE_PATH)..."; \
		cd pytorch-deeplab_v3_plus && python inference.py \
			--backbone mobilenet \
			--image_path $(IMAGE_PATH) \
			--output_directory ./results; \
	fi

test-env-direct:
	@echo "Testing environment - direct execution..."
	@echo "=== Folder Structure Debug - Environment Test ==="
	@echo "Current working directory:"
	pwd
	@echo "Workspace structure:"
	ls -la /workspace/ 2>/dev/null || echo "  /workspace/ does not exist"
	@echo "Dataset directory:"
	ls -la /data/datasets/ 2>/dev/null || echo "  /data/datasets/ does not exist"
	@echo "Results directory:"
	ls -la ./results/ 2>/dev/null || echo "  ./results/ does not exist"
	@echo ""
	python test_environment.py

status-direct:
	@echo "=== Container Environment Status ==="
	@echo "Current directory: $(shell pwd)"
	@echo "User: $(shell whoami)"
	@echo "Python version: $(shell python --version 2>&1)"
	@echo "PyTorch version: $(shell python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'Not available')"
	@echo "CUDA available: $(shell python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'Unknown')"
	@echo "GPU count: $(shell python -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null || echo 'Unknown')"
	@echo ""
	@echo "=== Folder Structure Debug - Complete Status ==="
	@echo "Workspace directory:"
	ls -la /workspace/ 2>/dev/null || echo "  /workspace/ does not exist"
	@echo "Data directory:"
	ls -la /data/datasets/ 2>/dev/null || echo "  /data/datasets/ does not exist"
	@echo "VOCdevkit structure:"
	ls -la /data/datasets/VOCdevkit/ 2>/dev/null || echo "  /data/datasets/VOCdevkit/ does not exist"
	@echo "VOC2012 structure:"
	ls -la /data/datasets/VOCdevkit/VOC2012/ 2>/dev/null || echo "  /data/datasets/VOCdevkit/VOC2012/ does not exist"
	@echo "ImageSets structure:"
	ls -la /data/datasets/VOCdevkit/VOC2012/ImageSets/ 2>/dev/null || echo "  /data/datasets/VOCdevkit/VOC2012/ImageSets/ does not exist"
	@echo "SegmentationAug structure:"
	ls -la /data/datasets/VOCdevkit/VOC2012/ImageSets/SegmentationAug/ 2>/dev/null || echo "  /data/datasets/VOCdevkit/VOC2012/ImageSets/SegmentationAug/ does not exist"
	@echo "Required files check:"
	ls -la /data/datasets/VOCdevkit/VOC2012/ImageSets/SegmentationAug/train.txt 2>/dev/null || echo "  ❌ train.txt NOT FOUND"
	ls -la /data/datasets/VOCdevkit/VOC2012/ImageSets/SegmentationAug/val.txt 2>/dev/null || echo "  ❌ val.txt NOT FOUND"
	@echo "Results directory:"
	ls -la ./results/ 2>/dev/null || echo "  ./results/ does not exist"
	@echo "Dataset status: $(shell ls -la /data/datasets 2>/dev/null | wc -l) items in data directory"

download-data:
	@echo "Setting up PASCAL VOC2012 dataset..."
	@echo "=== Folder Structure Debug - Before Setup ==="
	@echo "Host datasets directory:"
	ls -la /data/datasets/ 2>/dev/null || echo "  /data/datasets/ does not exist"
	@echo "Container datasets directory:"
	ls -la /root/datasets/ 2>/dev/null || echo "  /root/datasets/ does not exist"
	@echo ""
	mkdir -p /data/datasets
	mkdir -p /root/datasets
	@echo "=== Step 1: Download PASCAL VOC2012 Dataset ==="
	if [ -f /home/ubuntu/browser_downloads/VOCtrainval_11-May-2012.tar ]; then \
		echo "Using existing PASCAL VOC2012 dataset from browser downloads..."; \
		cp /home/ubuntu/browser_downloads/VOCtrainval_11-May-2012.tar /data/datasets/; \
	elif [ ! -f /data/datasets/VOCtrainval_11-May-2012.tar ]; then \
		echo "Downloading PASCAL VOC2012 dataset..."; \
		wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar -O /data/datasets/VOCtrainval_11-May-2012.tar; \
	else \
		echo "PASCAL VOC2012 dataset tar already exists"; \
	fi
	@echo "Verifying tar file integrity..."
	if [ -f /data/datasets/VOCtrainval_11-May-2012.tar ]; then \
		echo "✅ Tar file exists (size: $$(ls -lh /data/datasets/VOCtrainval_11-May-2012.tar | awk '{print $$5}'))"; \
		echo "Testing tar file integrity..."; \
		tar -tf /data/datasets/VOCtrainval_11-May-2012.tar | head -5; \
	else \
		echo "❌ Tar file not found - download failed"; \
		exit 1; \
	fi
	@echo "=== Step 2: Extract PASCAL VOC2012 Dataset ==="
	if [ -f /data/datasets/VOCtrainval_11-May-2012.tar ]; then \
		if [ ! -d /data/datasets/VOCdevkit/VOC2012/ImageSets ]; then \
			echo "Extracting PASCAL VOC2012 dataset..."; \
			rm -rf /data/datasets/VOCdevkit 2>/dev/null || true; \
			tar -xf /data/datasets/VOCtrainval_11-May-2012.tar -C /data/datasets/; \
			echo "✅ Dataset extraction completed"; \
		else \
			echo "Dataset already extracted (ImageSets directory exists)"; \
		fi; \
	else \
		echo "❌ Cannot extract - tar file not found"; \
		exit 1; \
	fi
	@echo "=== Step 3: Verify Core Dataset Structure ==="
	@echo "Checking essential directories..."
	if [ ! -d /data/datasets/VOCdevkit/VOC2012 ]; then \
		echo "❌ VOC2012 directory not found after extraction"; \
		exit 1; \
	fi
	if [ ! -d /data/datasets/VOCdevkit/VOC2012/ImageSets ]; then \
		echo "❌ ImageSets directory not found after extraction"; \
		exit 1; \
	fi
	if [ ! -d /data/datasets/VOCdevkit/VOC2012/JPEGImages ]; then \
		echo "❌ JPEGImages directory not found after extraction"; \
		exit 1; \
	fi
	if [ ! -d /data/datasets/VOCdevkit/VOC2012/SegmentationClass ]; then \
		echo "❌ SegmentationClass directory not found after extraction"; \
		exit 1; \
	fi
	@echo "✅ All essential directories verified"
	@echo "=== Step 4: Create Symbolic Links ==="
	if [ ! -L /root/datasets/VOCdevkit ] && [ -d /data/datasets/VOCdevkit ]; then \
		ln -sf /data/datasets/VOCdevkit /root/datasets/VOCdevkit; \
		echo "✅ Created /root/datasets/VOCdevkit -> /data/datasets/VOCdevkit"; \
	fi
	if [ ! -L /data/datasets/pascal ]; then \
		ln -sf /data/datasets/VOCdevkit/VOC2012 /data/datasets/pascal; \
		echo "✅ Created /data/datasets/pascal -> /data/datasets/VOCdevkit/VOC2012"; \
	fi
	@echo "=== Step 5: Create SegmentationClassAug Link ==="
	if [ ! -L /data/datasets/VOCdevkit/VOC2012/SegmentationClassAug ] && [ -d /data/datasets/VOCdevkit/VOC2012/SegmentationClass ]; then \
		ln -sf /data/datasets/VOCdevkit/VOC2012/SegmentationClass /data/datasets/VOCdevkit/VOC2012/SegmentationClassAug; \
		echo "✅ Created SegmentationClassAug -> SegmentationClass symbolic link"; \
	elif [ -L /data/datasets/VOCdevkit/VOC2012/SegmentationClassAug ]; then \
		echo "✅ SegmentationClassAug symbolic link already exists"; \
	else \
		echo "❌ Cannot create SegmentationClassAug link - SegmentationClass not found"; \
		exit 1; \
	fi
	@echo "=== Step 6: Create SegmentationAug Files ==="
	python create_segmentation_aug_files.py
	@echo "=== Step 7: Download Scribble Annotations ==="
	cd /data/datasets/VOCdevkit/VOC2012 && \
	if [ ! -f pascal_2012_scribble.zip ]; then \
		echo "Downloading scribble annotations from rloss repository..."; \
		wget http://cs.uwaterloo.ca/~m62tang/rloss/pascal_2012_scribble.zip; \
	else \
		echo "Scribble annotations zip already exists"; \
	fi && \
	if [ ! -d pascal_2012_scribble ]; then \
		echo "Extracting scribble annotations..."; \
		python -c "import zipfile; zipfile.ZipFile('pascal_2012_scribble.zip').extractall('.')"; \
	else \
		echo "Scribble annotations directory already exists"; \
	fi
	@echo "=== Final Verification ==="
	@echo "Dataset directory structure:"
	ls -la /data/datasets/ 2>/dev/null || echo "  /data/datasets/ does not exist"
	@echo "VOCdevkit structure:"
	ls -la /data/datasets/VOCdevkit/ 2>/dev/null || echo "  /data/datasets/VOCdevkit/ does not exist"
	@echo "VOC2012 structure:"
	ls -la /data/datasets/VOCdevkit/VOC2012/ 2>/dev/null || echo "  /data/datasets/VOCdevkit/VOC2012/ does not exist"
	@echo "ImageSets structure:"
	ls -la /data/datasets/VOCdevkit/VOC2012/ImageSets/ 2>/dev/null || echo "  /data/datasets/VOCdevkit/VOC2012/ImageSets/ does not exist"
	@echo "SegmentationAug structure:"
	ls -la /data/datasets/VOCdevkit/VOC2012/ImageSets/SegmentationAug/ 2>/dev/null || echo "  /data/datasets/VOCdevkit/VOC2012/ImageSets/SegmentationAug/ does not exist"
	@echo "Required train.txt file:"
	ls -la /data/datasets/VOCdevkit/VOC2012/ImageSets/SegmentationAug/train.txt 2>/dev/null || echo "  ❌ train.txt NOT FOUND"
	@echo "Available ImageSets files:"
	find /data/datasets/VOCdevkit/VOC2012/ImageSets/ -name "*.txt" 2>/dev/null | head -10 || echo "  No .txt files found"
	@echo "Scribble annotations directory:"
	ls -la /data/datasets/VOCdevkit/VOC2012/pascal_2012_scribble/ | head -5 2>/dev/null || echo "  ❌ pascal_2012_scribble directory does not exist"
	@echo "Sample scribble files count:"
	ls /data/datasets/VOCdevkit/VOC2012/pascal_2012_scribble/*.png 2>/dev/null | wc -l || echo "  0 scribble files found"
	@echo ""
	@echo "=== Dataset Setup Summary ==="
	@if [ -f /data/datasets/VOCdevkit/VOC2012/ImageSets/SegmentationAug/train.txt ]; then \
		echo "✅ PASCAL VOC2012 dataset setup SUCCESSFUL"; \
		echo "✅ Dataset ready at /data/datasets/VOCdevkit/VOC2012"; \
		echo "✅ Training can proceed with make train-small-direct"; \
	else \
		echo "❌ PASCAL VOC2012 dataset setup FAILED"; \
		echo "❌ Missing required train.txt file"; \
		exit 1; \
	fi

clean-data:
	@echo "Cleaning dataset directories..."
	@echo "Removing /data/datasets/VOCdevkit..."
	rm -rf /data/datasets/VOCdevkit
	@echo "Removing /data/datasets/VOCtrainval_11-May-2012.tar..."
	rm -f /data/datasets/VOCtrainval_11-May-2012.tar
	@echo "Removing /root/datasets/VOCdevkit symlink..."
	rm -f /root/datasets/VOCdevkit
	@echo "Removing /data/datasets/pascal symlink..."
	rm -f /data/datasets/pascal
	@echo "✅ Dataset directories cleaned - ready for fresh download"

download-models:
	@echo "Setting up pre-trained models..."
	mkdir -p /workspace/models
	@echo "Pre-trained models available at: https://cs.uwaterloo.ca/~m62tang/rloss/pytorch"
	@echo "Please download manually and place in /workspace/models/"
	@echo "Available models:"
	@echo "  - deeplab-resnet.pth.tar (ResNet backbone)"
	@echo "  - deeplab-mobilenet.pth.tar (MobileNet backbone)"
	@echo ""
	@echo "Note: MobileNet backbone uses torchvision pretrained weights automatically"
	@echo "ResNet backbone requires manual download from the URL above"


# W&B-enabled training targets (with Weights & Biases logging)
train-small-wandb:
	@echo "Training with small images (40x40) + W&B logging..."
	@if [ -z "$$WANDB_API_KEY" ]; then \
		echo "⚠️  WANDB_API_KEY not set. Run: python setup_wandb.py"; \
		echo "   Or set manually: export WANDB_API_KEY=your_api_key"; \
		exit 1; \
	fi
	cd pytorch-deeplab_v3_plus && python train_withdensecrfloss.py \
		--backbone mobilenet \
		--crop-size 64 \
		--batch-size 32 \
		--lr 0.02 \
		--epochs 100 \
		--densecrfloss 2e-9 \
		--rloss-scale 0.25 \
		--use-wandb

train-standard-wandb:
	@echo "Training with standard images (513x513) + W&B logging..."
	@if [ -z "$$WANDB_API_KEY" ]; then \
		echo "⚠️  WANDB_API_KEY not set. Run: python setup_wandb.py"; \
		echo "   Or set manually: export WANDB_API_KEY=your_api_key"; \
		exit 1; \
	fi
	cd pytorch-deeplab_v3_plus && python train_withdensecrfloss.py \
		--backbone resnet \
		--crop-size 513 \
		--batch-size 16 \
		--lr 0.01 \
		--epochs 50 \
		--densecrfloss 2e-9 \
		--rloss-scale 1.0 \
		--use-wandb

setup-wandb:
	@echo "Setting up Weights & Biases credentials..."
	python setup_wandb.py

check-wandb:
	@echo "Checking W&B setup status..."
	python setup_wandb.py --check
