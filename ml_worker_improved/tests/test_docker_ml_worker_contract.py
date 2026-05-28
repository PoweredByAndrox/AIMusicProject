from __future__ import annotations

from pathlib import Path

import pytest

PROJECT_ROOT = Path("/workspace") if Path("/workspace/docker").exists() else Path(__file__).resolve().parents[2]


def test_gpu_dockerfile_audio_separator_venv_requires_cuda_execution_provider() -> None:
    dockerfile = PROJECT_ROOT / "docker" / "ml-worker" / "Dockerfile.gpu"
    if not dockerfile.exists():
        pytest.skip("Dockerfile.gpu is outside the ml-worker /app bind; mount the project root at /workspace.")
    content = dockerfile.read_text(encoding="utf-8")

    assert "onnxruntime-gpu" in content
    assert "CUDAExecutionProvider" in content
    assert "Audio Separator GPU runtime must expose CUDAExecutionProvider" in content
