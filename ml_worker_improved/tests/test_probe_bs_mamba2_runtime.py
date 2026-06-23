from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf


TOOLS_ROOT = Path(__file__).resolve().parents[1] / "tools"
ROOT_DIR = Path(__file__).resolve().parents[1]
MODULE_PATH = TOOLS_ROOT / "probe_bs_mamba2_runtime.py"


def _load_probe_module():
    if str(ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(ROOT_DIR))
    spec = importlib.util.spec_from_file_location("probe_bs_mamba2_runtime", MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_tiny_flac(path: Path, *, frames: int = 4800, sample_rate: int = 48000) -> None:
    audio = np.zeros((frames, 2), dtype=np.float32)
    sf.write(str(path), audio, sample_rate, subtype="PCM_24")


def test_probe_writes_runtime_marker_after_valid_bs_mamba2_manifest(tmp_path: Path) -> None:
    module = _load_probe_module()
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    vocals = output_dir / "vocals.flac"
    instrumental = output_dir / "instrumental.flac"
    _write_tiny_flac(vocals)
    _write_tiny_flac(instrumental)
    source_audio_path = tmp_path / "probe.flac"
    _write_tiny_flac(source_audio_path)
    (output_dir / "stem_manifest.json").write_text(
        json.dumps(
            {
                "model_id": "bs_mamba2_vocals",
                "stems": {
                    "vocals": {"path": str(vocals)},
                    "instrumental": {"path": str(instrumental)},
                },
            }
        ),
        encoding="utf-8",
    )

    marker_path = tmp_path / "runtime_probe_passed.json"
    result = module.write_success_marker_from_probe_outputs(
        marker_path=marker_path,
        output_dir=output_dir,
        source_audio_path=source_audio_path,
        repo_path=Path("/app/vendor/music_source_separation_training"),
        model_path=Path("/app/vendor/bs_mamba2_vocals/bs_mamba2_vocals.ckpt"),
        config_path=Path("/app/vendor/bs_mamba2_vocals/config_bs_mamba2_vocals.yaml"),
        elapsed_sec=1.25,
    )

    assert result["status"] == "passed"
    assert marker_path.exists()
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    assert marker["schema"] == "bs_mamba2_runtime_probe.v1"
    assert marker["model_id"] == "bs_mamba2_vocals"
    assert marker["runtime_probe_status"] == "passed"
    assert marker["contract"]["docker_only"] is True
    assert marker["contract"]["synthetic_probe_audio"] is True
    assert marker["contract"]["fallback_used"] is False
    assert marker["validated_outputs"] == ["vocals", "instrumental"]


def test_probe_refuses_marker_when_instrumental_output_is_missing(tmp_path: Path) -> None:
    module = _load_probe_module()
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    vocals = output_dir / "vocals.flac"
    _write_tiny_flac(vocals)
    source_audio_path = tmp_path / "probe.flac"
    _write_tiny_flac(source_audio_path)
    (output_dir / "stem_manifest.json").write_text(
        json.dumps({"model_id": "bs_mamba2_vocals", "stems": {"vocals": {"path": str(vocals)}}}),
        encoding="utf-8",
    )
    marker_path = tmp_path / "runtime_probe_passed.json"

    with pytest.raises(RuntimeError, match="missing_probe_output:instrumental"):
        module.write_success_marker_from_probe_outputs(
            marker_path=marker_path,
            output_dir=output_dir,
            source_audio_path=source_audio_path,
            repo_path=Path("/repo"),
            model_path=Path("/model.ckpt"),
            config_path=Path("/config.yaml"),
            elapsed_sec=1.0,
        )

    assert not marker_path.exists()


def test_probe_refuses_marker_when_source_audio_is_missing(tmp_path: Path) -> None:
    module = _load_probe_module()
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    marker_path = tmp_path / "runtime_probe_passed.json"

    with pytest.raises(RuntimeError, match="missing_probe_source_audio"):
        module.write_success_marker_from_probe_outputs(
            marker_path=marker_path,
            output_dir=output_dir,
            source_audio_path=tmp_path / "missing.flac",
            repo_path=Path("/repo"),
            model_path=Path("/model.ckpt"),
            config_path=Path("/config.yaml"),
            elapsed_sec=1.0,
        )

    assert not marker_path.exists()


def test_probe_refuses_nonempty_keep_output_dir(tmp_path: Path) -> None:
    module = _load_probe_module()
    keep_output_dir = tmp_path / "kept"
    keep_output_dir.mkdir()
    (keep_output_dir / "stem_manifest.json").write_text("{}", encoding="utf-8")

    with pytest.raises(RuntimeError, match="probe_output_dir_must_be_empty"):
        module.prepare_probe_output_dir(keep_output_dir)
