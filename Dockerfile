FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DEFAULT_TIMEOUT=120

ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ENV PIP_INDEX_URL=${PIP_INDEX_URL}
ENV PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn

ENV HF_HOME=/root/.cache/huggingface
ENV TRANSFORMERS_CACHE=/root/.cache/huggingface

# Ubuntu apt 源换清华源
RUN sed -i 's#http://archive.ubuntu.com/ubuntu/#https://mirrors.tuna.tsinghua.edu.cn/ubuntu/#g' /etc/apt/sources.list && \
    sed -i 's#http://security.ubuntu.com/ubuntu/#https://mirrors.tuna.tsinghua.edu.cn/ubuntu/#g' /etc/apt/sources.list

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3-pip python3.10-dev \
    git curl wget ca-certificates build-essential ninja-build \
    poppler-utils tesseract-ocr tesseract-ocr-chi-sim \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    python -m pip install --no-cache-dir --upgrade pip setuptools wheel

WORKDIR /workspace/DocVisRAG

COPY requirements-base.txt /tmp/requirements-base.txt

# CUDA 版 PyTorch 仍然用官方 PyTorch 源，不走清华 PyPI
RUN python -m pip install --no-cache-dir \
    torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124

RUN python -m pip install --no-cache-dir -r /tmp/requirements-base.txt

CMD ["/bin/bash"]