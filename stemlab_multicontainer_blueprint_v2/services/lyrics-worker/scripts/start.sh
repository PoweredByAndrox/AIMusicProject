#!/bin/sh
set -eu

mkdir -p "${LYRICS_CACHE_ROOT:-/app/models/lyrics-worker-cache}"
mkdir -p "${HF_HOME:-/app/models/lyrics-worker-cache/hf}"
mkdir -p "${HUGGINGFACE_HUB_CACHE:-/app/models/lyrics-worker-cache/hf/hub}"
mkdir -p "${TORCH_HOME:-/app/models/lyrics-worker-cache/torch}"
mkdir -p "${XDG_CACHE_HOME:-/app/models/lyrics-worker-cache/xdg}"
mkdir -p "${MPLCONFIGDIR:-/app/models/lyrics-worker-cache/matplotlib}"

if [ "${LYRICS_PREFETCH_ON_START:-true}" = "true" ]; then
  python /app/scripts/prefetch_lyrics_models.py || echo "lyrics prefetch failed; continuing with cold runtime"
fi

exec uvicorn app.main:app --host 0.0.0.0 --port 8000
