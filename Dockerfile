# Official PyTorch base image with CUDA 12.1 and cuDNN 8 development headers
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel

# Environment variables for FlashAttention compilation and vLLM multiprocessing
ENV DEBIAN_FRONTEND=noninteractive
ENV MAX_JOBS=4
ENV FLASH_ATTENTION_FORCE_BUILD=TRUE
ENV VLLM_WORKER_MULTIPROCESS_METHOD=spawn

# Install system build dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    ninja-build \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Copy dependency manifests first to leverage Docker layer caching
COPY pyproject.toml setup.py requirements.txt README.md ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy full repository source and install package in editable mode
COPY . .
RUN pip install --no-cache-dir --no-deps -e .

# Default entrypoint
CMD ["/bin/bash"]
