#!/usr/bin/env bash
set -euo pipefail

echo "=== Bootstrap models ==="
echo "This script does not download everything automatically on purpose."
echo "Edit the target directories and uncomment what you actually want."

mkdir -p ./models

cat <<'TXT'
Recommended starting point:
1. BS-RoFormer-Infer:
   pip install bs-roformer-infer
   bs-roformer-download --model roformer-model-bs-roformer-sw-by-jarredou --output-dir ./models
   bs-roformer-download --model roformer-model-bs-roformer-vocals-resurrection-by-unwa --output-dir ./models
   bs-roformer-download --model roformer-model-bs-roformer-instrumental-resurrection-by-unwa --output-dir ./models
   bs-roformer-download --model roformer-model-bs-roformer-de-reverb --output-dir ./models

2. Demucs:
   pip install -U demucs

3. Audio Separator:
   pip install "audio-separator[gpu]"

4. Query-based:
   clone and integrate your chosen research repo manually
TXT
