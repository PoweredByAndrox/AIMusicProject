# ML Worker Dockerfile
# Python 3.11+ with PyTorch (CUDA), TensorFlow (CUDA), Essentia, FAISS, yt-dlp
# Requirements: 3.1, 13.1
#
# This container handles:
# - Audio analysis (BPM/key/structure via allin1 + Essentia)
# - Audio embeddings for recommendations (Discogs-EffNet frozen .pb via TensorFlow)
# - Vector similarity search (FAISS)
# - Queue task processing from Redis
#
# IMPORTANT:
# TensorFlow (2.21.0) is built against CUDA 12.5.1 + cuDNN 9 (>= 9.3.0 at runtime).
# We use a CUDA base image that ships cuDNN >= 9.3.0 so embedding inference runs on GPU.

FROM nvidia/cuda:12.6.0-cudnn-devel-ubuntu22.04

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive \
    TF_CPP_MIN_LOG_LEVEL=2 \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:$PATH

# Install Python 3.11 and system dependencies (cmake is needed by some builds, e.g. natten)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    ffmpeg \
    libsndfile1 \
    libpq5 \
    libpq-dev \
    build-essential \
    git \
    curl \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Upgrade pip
RUN python -m pip install --upgrade pip

WORKDIR /app

COPY docker/ml-worker/requirements.txt /app/requirements.txt
COPY docker/ml-worker/patch_natten_compat.py /app/patch_natten_compat.py
COPY docker/ml-worker/patch_mamba_ssm_compat.py /app/patch_mamba_ssm_compat.py

# Install Python dependencies (PyTorch + TensorFlow GPU wheels)
RUN pip install --no-cache-dir -r /app/requirements.txt

# BS-Mamba2 vendor checkpoints require mamba_ssm. Build from PyPI source with the
# already-installed PyTorch/CUDA stack, without allowing pip to pull a different torch/CUDA runtime.
# mamba-ssm 2.3.2.post1 imports Mamba3 at package top level, which requires a newer Triton API
# than torch==2.5.1 provides. BS-Mamba2 only imports Mamba/Mamba2/Block/MLP, so patch the top-level
# __init__.py to avoid the unused Mamba3 import while preserving the modules required by the vendor model.
ARG MAMBA_SSM_VERSION=2.3.2.post1
ARG CAUSAL_CONV1D_VERSION=1.5.3.post1
ARG MAMBA_TORCH_CUDA_ARCH_LIST=8.9
ARG MAMBA_BUILD_MAX_JOBS=4
RUN pip install --no-cache-dir ninja \
    && TORCH_CUDA_ARCH_LIST="${MAMBA_TORCH_CUDA_ARCH_LIST}" \
       MAX_JOBS="${MAMBA_BUILD_MAX_JOBS}" \
       pip install --no-cache-dir --no-build-isolation "causal-conv1d==${CAUSAL_CONV1D_VERSION}" \
    && TORCH_CUDA_ARCH_LIST="${MAMBA_TORCH_CUDA_ARCH_LIST}" \
       MAX_JOBS="${MAMBA_BUILD_MAX_JOBS}" \
       MAMBA_FORCE_BUILD=TRUE \
       pip install --no-cache-dir --no-deps --no-build-isolation "mamba-ssm==${MAMBA_SSM_VERSION}" \
    && python /app/patch_mamba_ssm_compat.py \
    && python -c "import importlib.util; assert importlib.util.find_spec('mamba_ssm') is not None; assert importlib.util.find_spec('causal_conv1d') is not None; print('mamba_ssm and causal_conv1d packages available')"

# TEMPORARY third-party workaround:
# basic-pitch metadata hard-pins tensorflow<2.15.1, but the worker image requires
# tensorflow==2.21.0 for Essentia/Discogs-EffNet. Install it without re-resolving
# TensorFlow, then add only the runtime dependencies its note-extraction path needs.
# Real fix path: replace basic-pitch note extraction or move it to an isolated
# sidecar runtime with a compatible TensorFlow stack.
RUN pip install --no-cache-dir --no-deps basic-pitch==0.4.0 \
    && pip install --no-cache-dir "pretty_midi>=0.2.9" "mir_eval>=0.6" "resampy>=0.2.2,<0.4.3" onnxruntime-gpu

# Audio Separator requires numpy>=2 and newer beartype, while the main worker
# keeps numpy<2 for TensorFlow/Essentia and legacy MSST routes. Keep it isolated.
RUN python -m venv /opt/audio-separator-venv \
    && /opt/audio-separator-venv/bin/python -m pip install --upgrade pip \
    && /opt/audio-separator-venv/bin/pip install --no-cache-dir audio-separator==0.44.2 onnxruntime-gpu \
    && /opt/audio-separator-venv/bin/python -c "from audio_separator.separator import Separator; print(Separator.__name__)" \
    && /opt/audio-separator-venv/bin/python -c "import onnxruntime as ort; providers = ort.get_available_providers(); assert 'CUDAExecutionProvider' in providers, f'Audio Separator GPU runtime must expose CUDAExecutionProvider, got {providers}'"
ENV AUDIO_SEPARATOR_PYTHON=/opt/audio-separator-venv/bin/python

# Patch allin1/models/dinat.py to use natten>=0.15 compat shims
# (allin1==1.1.0 uses the old natten<=0.14 split API which was removed in natten>=0.15)
RUN python /app/patch_natten_compat.py

# Create necessary directories
RUN mkdir -p /app/storage \
    /app/faiss_index \
    /app/models \
    /app/logs \
    /app/demix

RUN chmod -R 755 /app

# Copy worker code for model download (runtime code is mounted as a volume)
COPY ml_worker_improved/worker /app/worker

# Download Essentia models during build to speed up first run.
ENV MODELS_PATH=/app/models
RUN python -m worker.download_models || echo "Model download failed, will retry at runtime"

# Health check - verify the worker runtime can still reach its queue and database
HEALTHCHECK --interval=30s --timeout=15s --start-period=20s --retries=3 \
    CMD python -c "import os, redis, psycopg2; redis.Redis(host=os.getenv('REDIS_HOST','redis'), port=int(os.getenv('REDIS_PORT','6379')), password=os.getenv('REDIS_PASSWORD') or None, socket_connect_timeout=1, socket_timeout=1, decode_responses=True).ping(); conn=psycopg2.connect(host=os.getenv('DB_HOST','postgres'), port=int(os.getenv('DB_PORT','5432')), dbname=os.getenv('DB_NAME','music_recommendations'), user=os.getenv('DB_USER','postgres'), password=os.getenv('DB_PASSWORD','secret'), connect_timeout=1); conn.close(); print('OK')" || exit 1

CMD ["python", "-m", "worker.main"]
