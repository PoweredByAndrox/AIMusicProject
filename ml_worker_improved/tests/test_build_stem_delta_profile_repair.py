from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

TOOLS_ROOT = Path(__file__).resolve().parents[1] / "tools"
MODULE_PATH = TOOLS_ROOT / "build_stem_delta_profile_repair.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("build_stem_delta_profile_repair", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_audio(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio.astype(np.float32), sample_rate, subtype="PCM_24")


def _stereo_tone(sample_rate: int, amplitude: float, frequency_hz: float = 120.0) -> np.ndarray:
    t = np.linspace(0.0, 1.0, sample_rate, endpoint=False, dtype=np.float32)
    tone = amplitude * np.sin(2.0 * np.pi * frequency_hz * t)
    return np.stack([tone, tone], axis=1).astype(np.float32)


def _stereo_pulses(sample_rate: int, amplitude: float) -> np.ndarray:
    t = np.linspace(0.0, 1.0, sample_rate, endpoint=False, dtype=np.float32)
    pulses = np.zeros_like(t)
    for center in (0.15, 0.35, 0.55, 0.75):
        pulses += amplitude * np.exp(-0.5 * np.square((t - center) / 0.008))
    return np.stack([pulses, pulses], axis=1).astype(np.float32)


def _write_bundle(module, root: Path) -> tuple[Path, Path, Path]:
    sample_rate = 16000
    reference = _stereo_tone(sample_rate, amplitude=0.4)
    candidate = _stereo_tone(sample_rate, amplitude=0.2)
    teacher = _stereo_tone(sample_rate, amplitude=0.35)

    gt_manifest = root / "gt" / "ground_truth_manifest.json"
    candidate_manifest = root / "candidate" / "stem_manifest.json"
    teacher_manifest = root / "teacher" / "stem_manifest.json"
    _write_audio(root / "gt" / "source_mix.flac", reference, sample_rate)
    _write_audio(root / "gt" / "stems" / "drums.flac", reference, sample_rate)
    _write_audio(root / "candidate" / "drums.flac", candidate, sample_rate)
    _write_audio(root / "teacher" / "drums.flac", teacher, sample_rate)

    module.write_json(
        gt_manifest,
        {
            "ground_truth_contract": {"verified": True},
            "mix": {"path": "source_mix.flac"},
            "stems": {"drums": {"path": "stems/drums.flac", "derived": False}},
        },
    )
    module.write_json(
        candidate_manifest,
        {"contract": {"benchmark_compatible": True}, "stems": {"drums": {"path": "drums.flac"}}},
    )
    module.write_json(
        teacher_manifest,
        {"contract": {"benchmark_compatible": True}, "stems": {"drums": {"path": "drums.flac"}}},
    )
    return gt_manifest, candidate_manifest, teacher_manifest


def test_learned_profile_application_uses_no_reference_audio_at_application() -> None:
    module = _load_module()
    sample_rate = 16000
    candidate = _stereo_tone(sample_rate, amplitude=0.2)
    teacher = _stereo_tone(sample_rate, amplitude=0.35)
    reference = _stereo_tone(sample_rate, amplitude=0.4)
    config = module.StemDeltaProfileConfig(
        n_fft=512,
        hop_length=128,
        max_abs_gain_db=12.0,
        use_percussive_gate=False,
    )

    profile, diagnostics = module.build_stem_delta_profile(
        candidate_audio=candidate,
        teacher_repair_audio=teacher,
        reference_audio=reference,
        sample_rate=sample_rate,
        stem_name="drums",
        config=config,
    )
    repaired = module.apply_stem_delta_profile(candidate, sample_rate, profile, config)

    assert profile["contract"]["application_reference_free"] is True
    assert profile["contract"]["target_reference_used_at_application"] is False
    assert profile["contract"]["profile_source_reference_assisted"] is True
    assert any(band["gain_db"] > 0.0 for band in profile["bands"])
    assert module.rms_dbfs(repaired) > module.rms_dbfs(candidate)
    assert diagnostics["rms_dbfs"]["teacher_repair"] > diagnostics["rms_dbfs"]["candidate"]


def test_build_manifest_marks_profile_source_and_application_contract(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    monkeypatch.setattr(module, "require_docker_runtime", lambda **kwargs: None)
    gt_manifest, candidate_manifest, teacher_manifest = _write_bundle(module, tmp_path)

    manifest = module.build_stem_delta_profile_repair(
        ground_truth_manifest=gt_manifest,
        base_candidate_manifest=candidate_manifest,
        teacher_repair_manifest=teacher_manifest,
        output_dir=tmp_path / "out",
        stem_name="drums",
        listen_dir=tmp_path / "listen",
        config=module.StemDeltaProfileConfig(n_fft=512, hop_length=128, max_abs_gain_db=12.0),
    )

    assert manifest["contract"]["fallback_used"] is False
    assert manifest["contract"]["profile_source_reference_assisted"] is True
    assert manifest["contract"]["target_reference_used_at_application"] is False
    assert manifest["contract"]["application_reference_free"] is True
    assert manifest["contract"]["delivery_allowed_without_reference"] is False
    assert manifest["profile_application"]["uses_reference_during_application"] is False
    assert (tmp_path / "out" / "stem_delta_profile.json").exists()
    assert (tmp_path / "out" / "stem_delta_profile_diagnostic_report.json").exists()
    assert (tmp_path / "listen" / "03_profile_repair_reference_free_application_drums.flac").exists()

    profile = json.loads((tmp_path / "out" / "stem_delta_profile.json").read_text(encoding="utf-8"))
    assert profile["contract"]["not_timestamp_hardcoded"] is True


def test_source_percussive_reservoir_adds_missing_drum_transients_without_reference() -> None:
    module = _load_module()
    sample_rate = 16000
    candidate = _stereo_pulses(sample_rate, amplitude=0.15)
    source = _stereo_pulses(sample_rate, amplitude=0.45) + _stereo_tone(sample_rate, amplitude=0.05, frequency_hz=880.0)
    profile = {
        "bands": [
            {"low_hz": 20.0, "high_hz": 14000.0, "gain_db": 0.0},
        ]
    }
    config = module.StemDeltaProfileConfig(
        n_fft=512,
        hop_length=128,
        use_source_percussive_reservoir=True,
        source_reservoir_mix=0.5,
    )

    repaired = module.apply_stem_delta_profile(candidate, sample_rate, profile, config, source_audio=source)

    assert module.rms_dbfs(repaired) > module.rms_dbfs(candidate)
    assert float(np.max(np.abs(repaired))) > float(np.max(np.abs(candidate))) * 1.5


def test_source_percussive_reservoir_rejects_source_only_leakage_without_candidate_evidence() -> None:
    module = _load_module()
    sample_rate = 16000
    candidate = np.zeros((sample_rate, 2), dtype=np.float32)
    source_only_plosive = _stereo_pulses(sample_rate, amplitude=0.45)
    source_only_tone = _stereo_tone(sample_rate, amplitude=0.08, frequency_hz=620.0)
    source = source_only_plosive + source_only_tone
    profile = {
        "bands": [
            {"low_hz": 20.0, "high_hz": 14000.0, "gain_db": 0.0},
        ]
    }
    config = module.StemDeltaProfileConfig(
        n_fft=512,
        hop_length=128,
        use_source_percussive_reservoir=True,
        source_reservoir_mix=0.8,
        source_reservoir_min_candidate_percussive_confidence=0.32,
    )

    repaired = module.apply_stem_delta_profile(candidate, sample_rate, profile, config, source_audio=source)

    assert float(np.max(np.abs(repaired))) < 1e-4
    assert module.rms_dbfs(repaired) < -80.0
