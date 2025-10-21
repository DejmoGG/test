# CUDA 12.1 + PyTorch 2.4.1 base with ffmpeg present
FROM runpod/pytorch:2.4.1-py3.10-cuda12.1.0-devel-ubuntu22.04

# System deps
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    git git-lfs ffmpeg libgl1 libglib2.0-0 wget && \
    rm -rf /var/lib/apt/lists/* && git lfs install

# Workdir
WORKDIR /workspace

# Copy repo (assuming Dockerfile is at repo root)
COPY . /workspace


# Python deps EXACTLY as upstream
RUN pip install --upgrade pip wheel packaging ninja psutil \
 && pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121 \
 && pip install -U xformers==0.0.28 --index-url https://download.pytorch.org/whl/cu121 \
 && pip install flash_attn==2.7.4.post1 \
 && pip install misaki[en] \
 && pip install -r requirements.txt \
 && pip install runpod requests tqdm huggingface_hub==0.23.5

# Cache dir for models (persist within container disk while warm)
ENV HF_HOME=/workspace/.cache/huggingface
ENV TORCH_HOME=/workspace/.cache/torch
ENV TRANSFORMERS_CACHE=/workspace/.cache/huggingface
ENV WANDB_DISABLED=true

# Where we’ll keep weights (download-once strategy)
ENV WEIGHTS_DIR=/workspace/weights
RUN mkdir -p $WEIGHTS_DIR /out

# Optional: put your HF token at deploy time, not here
# ENV HF_TOKEN=...

# Entrypoint
COPY start.sh /workspace/start.sh
COPY handler.py /workspace/handler.py
RUN chmod +x /workspace/start.sh

CMD ["/workspace/start.sh"]
