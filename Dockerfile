# Base image: Ubuntu with CUDA and cuDNN support
FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04

# Set working directory
WORKDIR /workspace

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip3 install --upgrade pip

# Install PyTorch and other dependencies
RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

# Clone the DeepLabV3Plus repository
RUN git clone https://github.com/VainF/DeepLabV3Plus-Pytorch.git /workspace/DeepLabV3Plus-Pytorch

# Install required Python packages from the repository
WORKDIR /workspace/DeepLabV3Plus-Pytorch
RUN pip3 install -r requirements.txt

# Set environment variables
ENV PYTHONUNBUFFERED=1


