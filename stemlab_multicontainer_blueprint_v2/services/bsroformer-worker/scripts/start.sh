#!/bin/sh
set -eu

mkdir -p "${BSROFORMER_CACHE_ROOT:-/app/models/bsroformer-worker-cache}"
mkdir -p "${TORCH_HOME:-/app/models/bsroformer-worker-cache/torch}"
mkdir -p "${HF_HOME:-/app/models/bsroformer-worker-cache/hf}"
mkdir -p "${HUGGINGFACE_HUB_CACHE:-/app/models/bsroformer-worker-cache/hf/hub}"
mkdir -p "${XDG_CACHE_HOME:-/app/models/bsroformer-worker-cache/xdg}"
mkdir -p "${MPLCONFIGDIR:-/app/models/bsroformer-worker-cache/matplotlib}"

if [ "${BSROFORMER_PREFETCH_ON_START:-true}" = "true" ]; then
  python /app/scripts/prefetch_bs_models.py || echo "bsroformer prefetch failed; continuing with cold runtime"
fi

exec uvicorn app.main:app --host 0.0.0.0 --port 8000
