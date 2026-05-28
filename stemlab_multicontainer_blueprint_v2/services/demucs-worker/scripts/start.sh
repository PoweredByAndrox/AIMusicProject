#!/bin/sh
set -eu

mkdir -p "${DEMUCS_CACHE_ROOT:-/app/models/demucs-worker-cache}"
mkdir -p "${TORCH_HOME:-/app/models/demucs-worker-cache/torch}"
mkdir -p "${XDG_CACHE_HOME:-/app/models/demucs-worker-cache/xdg}"
mkdir -p "${MPLCONFIGDIR:-/app/models/demucs-worker-cache/matplotlib}"

if [ "${DEMUCS_PREFETCH_ON_START:-true}" = "true" ]; then
  python /app/scripts/prefetch_demucs_models.py || echo "demucs prefetch failed; continuing with cold runtime"
fi

exec uvicorn app.main:app --host 0.0.0.0 --port 8000
