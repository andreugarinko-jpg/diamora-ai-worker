# Use NVIDIA CUDA base image compatible with PyTorch
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 python3.10-venv python3-pip \
    git wget libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set python3 as default
RUN ln -sf /usr/bin/python3.10 /usr/bin/python
RUN ln -sf /usr/bin/pip3 /usr/bin/pip

WORKDIR /workspace

# Install PyTorch and Xformers
RUN pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
RUN pip install xformers==0.0.23.post1 --index-url https://download.pytorch.org/whl/cu118

# Clone TRELLIS
RUN git clone --recurse-submodules https://github.com/microsoft/TRELLIS.git /workspace/TRELLIS
WORKDIR /workspace/TRELLIS

# Install TRELLIS dependencies
RUN pip install ninja spconv-cu118
RUN pip install imageio imageio-ffmpeg trimesh
RUN pip install runpod boto3 requests

# Install standard python dependencies
RUN pip install numpy scipy Pillow transformers huggingface_hub safetensors accelerate tqdm

# Set PYTHONPATH so python can find the trellis module
ENV PYTHONPATH="/workspace/TRELLIS"
# Copy our custom handler
COPY rp_handler.py /workspace/rp_handler.py
WORKDIR /workspace

# Pre-download the model weights during the Docker build so it starts instantly on RunPod
RUN python -c "from trellis.pipelines import TrellisImageTo3DPipeline; TrellisImageTo3DPipeline.from_pretrained('JeffreyXiang/TRELLIS-image-large')"

# Run the handler
CMD ["python", "-u", "/workspace/rp_handler.py"]
