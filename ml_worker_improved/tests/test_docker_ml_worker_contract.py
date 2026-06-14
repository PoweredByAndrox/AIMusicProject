from __future__ import annotations

from pathlib import Path

import pytest

PROJECT_ROOT = Path("/workspace") if Path("/workspace/docker").exists() else Path(__file__).resolve().parents[2]


def test_ml_worker_dockerfiles_audio_separator_venv_requires_cuda_execution_provider() -> None:
    dockerfiles = [
        PROJECT_ROOT / "docker" / "ml-worker" / "Dockerfile",
        PROJECT_ROOT / "docker" / "ml-worker" / "Dockerfile.gpu",
    ]
    missing = [str(dockerfile) for dockerfile in dockerfiles if not dockerfile.exists()]
    if missing:
        pytest.skip(f"Dockerfiles are outside the ml-worker /app bind: {missing}")

    for dockerfile in dockerfiles:
        content = dockerfile.read_text(encoding="utf-8")
        audio_separator_block = content.split("RUN python -m venv /opt/audio-separator-venv", 1)[1]
        audio_separator_block = audio_separator_block.split("ENV AUDIO_SEPARATOR_PYTHON", 1)[0]

        assert "audio-separator==0.44.2 onnxruntime-gpu" in audio_separator_block, str(dockerfile)
        assert "CUDAExecutionProvider" in audio_separator_block, str(dockerfile)
        assert "Audio Separator GPU runtime must expose CUDAExecutionProvider" in audio_separator_block, str(dockerfile)


def test_ml_worker_dockerfiles_install_bs_mamba2_required_causal_conv1d() -> None:
    dockerfiles = [
        PROJECT_ROOT / "docker" / "ml-worker" / "Dockerfile",
        PROJECT_ROOT / "docker" / "ml-worker" / "Dockerfile.gpu",
    ]
    missing = [str(dockerfile) for dockerfile in dockerfiles if not dockerfile.exists()]
    if missing:
        pytest.skip(f"Dockerfiles are outside the ml-worker /app bind: {missing}")

    for dockerfile in dockerfiles:
        content = dockerfile.read_text(encoding="utf-8")
        assert "causal-conv1d" in content, str(dockerfile)
        assert "causal_conv1d" in content, str(dockerfile)
        assert "find_spec('mamba_ssm')" in content, str(dockerfile)
        assert "find_spec('causal_conv1d')" in content, str(dockerfile)
