from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf


TOOLS_ROOT = Path(__file__).resolve().parents[1] / "tools"
MODULE_PATH = TOOLS_ROOT / "build_complex_oracle_mask_teacher.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("build_complex_oracle_mask_teacher", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_audio(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio.astype(np.float32), sample_rate, subtype="PCM_24")


def _tone(sample_rate: int, frequency_hz: float, amplitude: float = 0.12) -> np.ndarray:
    time = np.arange(sample_rate * 2, dtype=np.float32) / float(sample_rate)
    audio = amplitude * np.sin(2.0 * np.pi * float(frequency_hz) * time)
    return np.column_stack([audio, audio]).astype(np.float32)


def _write_ground_truth_bundle(root: Path) -> Path:
    sample_rate = 8000
    vocals = _tone(sample_rate, 220.0, 0.10)
    melody = _tone(sample_rate, 440.0, 0.09)
    drums = np.zeros_like(vocals)
    drums[2000:2200] = 0.4
    source_mix = np.clip(vocals + melody + drums, -1.0, 1.0).astype(np.float32)

    _write_audio(root / "source_mix.flac", source_mix, sample_rate)
    _write_audio(root / "stems" / "vocals.flac", vocals, sample_rate)
    _write_audio(root / "stems" / "melody.flac", melody, sample_rate)
    _write_audio(root / "stems" / "drums.flac", drums, sample_rate)
    manifest = {
        "schema_version": 1,
        "mix": {"path": "source_mix.flac"},
        "stems": {
            "vocals": {"path": "stems/vocals.flac"},
            "melody": {"path": "stems/melody.flac"},
            "drums": {"path": "stems/drums.flac"},
        },
    }
    manifest_path = root / "ground_truth_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def _rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(audio.astype(np.float64) ** 2) + 1.0e-18))


def test_complex_mask_teacher_reconstructs_requested_stems_from_source_mix(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    ground_truth_manifest = _write_ground_truth_bundle(tmp_path / "bundle")
    monkeypatch.setattr(module, "require_docker_runtime", lambda **_kwargs: None)

    manifest = module.build_complex_oracle_mask_teacher(
        ground_truth_manifest=ground_truth_manifest,
        output_dir=tmp_path / "out",
        stem_names=("vocals", "melody"),
        n_fft=1024,
        hop_length=256,
    )

    assert manifest["contract"]["target_reference_assisted"] is True
    assert manifest["contract"]["target_reference_used_at_application"] is True
    assert manifest["contract"]["application_reference_free"] is False
    assert manifest["contract"]["delivery_allowed_without_reference"] is False
    assert manifest["contract"]["no_reference_production_claim"] is False
    assert manifest["presentation_status"] == "diagnostic_requires_exact_reference_gate"
    assert manifest["ralph_ai_delivery_contract"]["result_delivery_allowed"] is False
    assert manifest["pipeline"]["source_audio_role"] == "authoritative_full_mix_complex_mask_carrier"
    assert manifest["pipeline"]["reference_audio_copied_to_output"] is False

    for stem_name in ("vocals", "melody"):
        candidate, _ = sf.read(manifest["stems"][stem_name]["path"], always_2d=True, dtype="float32")
        reference_path = tmp_path / "bundle" / "stems" / f"{stem_name}.flac"
        reference, _ = sf.read(reference_path, always_2d=True, dtype="float32")
        frame_count = min(candidate.shape[0], reference.shape[0])
        residual = candidate[:frame_count] - reference[:frame_count]
        assert _rms(residual) < 2.0e-4


def test_complex_mask_teacher_rejects_missing_reference_stem(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    ground_truth_manifest = _write_ground_truth_bundle(tmp_path / "bundle")
    monkeypatch.setattr(module, "require_docker_runtime", lambda **_kwargs: None)

    try:
        module.build_complex_oracle_mask_teacher(
            ground_truth_manifest=ground_truth_manifest,
            output_dir=tmp_path / "out",
            stem_names=("strings_orchestra",),
            n_fft=1024,
            hop_length=256,
        )
    except RuntimeError as exc:
        assert "strings_orchestra" in str(exc)
    else:
        raise AssertionError("missing reference stem must fail fast")
