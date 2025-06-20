# Updated base image for modern PyTorch with CUDA support
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

# Install system dependencies
RUN apt-get update && apt-get install -y \
    swig \
    build-essential \
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Set working directory
WORKDIR /workspace

# Copy source code
COPY . /workspace/

# Build bilateral filtering extension
WORKDIR /workspace/wrapper/bilateralfilter
RUN swig -python -c++ bilateralfilter.i && \
    python setup.py build_ext --inplace && \
    python setup.py install

# Return to main directory
WORKDIR /workspace

