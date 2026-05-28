Write-Host "=== Bootstrap models ==="
Write-Host "Edit this script before using it in production."

New-Item -ItemType Directory -Force -Path ".\models" | Out-Null

Write-Host @"
Recommended starting point:
1. BS-RoFormer-Infer:
   pip install bs-roformer-infer
   bs-roformer-download --model roformer-model-bs-roformer-sw-by-jarredou --output-dir .\models
   bs-roformer-download --model roformer-model-bs-roformer-vocals-resurrection-by-unwa --output-dir .\models
   bs-roformer-download --model roformer-model-bs-roformer-instrumental-resurrection-by-unwa --output-dir .\models
   bs-roformer-download --model roformer-model-bs-roformer-de-reverb --output-dir .\models

2. Demucs:
   pip install -U demucs

3. Audio Separator:
   pip install "audio-separator[gpu]"

4. Query-based:
   integrate manually
"@
