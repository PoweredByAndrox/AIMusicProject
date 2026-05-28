from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import soundfile as sf


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "build_model_stack_melody_candidate.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("build_model_stack_melody_candidate", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _sine(sample_rate: int, duration_sec: float, frequency_hz: float, gain: float) -> np.ndarray:
    samples = np.arange(int(sample_rate * duration_sec), dtype=np.float32)
    tone = np.sin(2.0 * np.pi * frequency_hz * samples / sample_rate).astype(np.float32) * gain
    return np.repeat(tone[:, np.newaxis], 2, axis=1)


def _noise_burst(sample_rate: int, duration_sec: float, gain: float) -> np.ndarray:
    audio = np.zeros((int(sample_rate * duration_sec), 2), dtype=np.float32)
    start = int(0.70 * sample_rate)
    end = int(0.82 * sample_rate)
    rng = np.random.default_rng(321)
    audio[start:end] = rng.normal(0.0, gain, size=audio[start:end].shape).astype(np.float32)
    return audio


def _projection_rms(audio: np.ndarray, reference: np.ndarray) -> float:
    candidate = np.asarray(audio, dtype=np.float32)
    ref = np.asarray(reference, dtype=np.float32)
    gain = float(np.sum(candidate * ref) / (np.sum(ref * ref) + 1e-12))
    projected = ref * gain
    return float(np.sqrt(np.mean(np.square(projected)) + 1e-12))


def _write_stem(path: Path, audio: np.ndarray, sample_rate: int) -> dict[str, object]:
    sf.write(path, audio, sample_rate, subtype="PCM_24")
    return {"path": str(path)}


def test_base_manifest_builder_prepares_hybrid_bs_runtime_before_direct_export(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_module()
    source_path = tmp_path / "source_mix.flac"
    output_dir = tmp_path / "candidate"
    sf.write(source_path, _sine(8000, 1.0, 330.0, 0.2), 8000, subtype="PCM_24")
    runtime_calls: list[str] = []
    export_calls: list[str] = []

    def fake_prepare_runtime(model_name: str) -> None:
        runtime_calls.append(model_name)

    def fake_export_base(source_audio_path, base_output_dir, *, model_name, device, sharpen_bleed) -> dict:
        export_calls.append(str(model_name))
        base_output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = base_output_dir / "stem_manifest.json"
        manifest_path.write_text("{}", encoding="utf-8")
        return {"manifest_path": str(manifest_path)}

    monkeypatch.setattr(module, "_ensure_base_model_runtime", fake_prepare_runtime)
    monkeypatch.setattr(module, "_export_base_stems_aggressive_melody_first", fake_export_base)

    manifest_path = module._build_base_manifest_from_source(
        source_audio_path=source_path,
        output_dir=output_dir,
        model_name="hybrid_htdemucs_ft_bs_roformer_6s",
        device="cuda",
    )

    assert runtime_calls == ["hybrid_htdemucs_ft_bs_roformer_6s"]
    assert export_calls == ["hybrid_htdemucs_ft_bs_roformer_6s"]
    assert manifest_path == output_dir / "_base_model_stack" / "stem_manifest.json"


def test_model_stack_candidate_removes_vocal_drum_and_bass_projection_without_killing_melody() -> None:
    module = _load_module()
    sample_rate = 8000
    duration_sec = 2.0
    melody = _sine(sample_rate, duration_sec, 440.0, 0.28)
    vocal = _sine(sample_rate, duration_sec, 660.0, 0.22)
    bass = _sine(sample_rate, duration_sec, 80.0, 0.28)
    drums = _noise_burst(sample_rate, duration_sec, 0.55)
    raw_model_melody = melody + vocal * 0.75 + drums * 0.50 + bass * 0.35

    candidate, diagnostics = module.build_model_stack_melody_candidate_audio(
        {
            "melody": raw_model_melody,
            "other": raw_model_melody * 0.25,
            "vocals": vocal,
            "drums": drums,
            "bass": bass,
        },
        sample_rate=sample_rate,
        harmonic_n_fft=512,
        harmonic_hop_length=128,
        vocal_suppression=0.92,
        drum_suppression=0.80,
        bass_suppression=0.90,
    )

    assert _projection_rms(candidate, vocal) < _projection_rms(raw_model_melody, vocal) * 0.35
    assert _projection_rms(candidate, drums) < _projection_rms(raw_model_melody, drums) * 0.45
    assert _projection_rms(candidate, bass) < _projection_rms(raw_model_melody, bass) * 0.35
    assert _projection_rms(candidate, melody) > _projection_rms(raw_model_melody, melody) * 0.55
    assert diagnostics["route_name"] == "model_stack_melody_v1"
    assert diagnostics["projection_removal"]["vocals"]["gain"] > 0.0


def test_model_stack_candidate_uses_source_tonal_prior_when_model_melody_is_weak() -> None:
    module = _load_module()
    sample_rate = 8000
    duration_sec = 2.0
    melody = _sine(sample_rate, duration_sec, 440.0, 0.28)
    vocal = _sine(sample_rate, duration_sec, 660.0, 0.22)
    bass = _sine(sample_rate, duration_sec, 80.0, 0.20)
    drums = _noise_burst(sample_rate, duration_sec, 0.45)
    weak_model_melody = melody * 0.20 + drums * 0.60
    source_audio = melody + vocal * 0.70 + drums * 0.30 + bass * 0.20

    candidate, diagnostics = module.build_model_stack_melody_candidate_audio(
        {
            "melody": weak_model_melody,
            "vocals": vocal,
            "drums": drums,
            "bass": bass,
        },
        source_audio=source_audio,
        sample_rate=sample_rate,
        harmonic_n_fft=512,
        harmonic_hop_length=128,
        vocal_suppression=0.95,
        drum_suppression=0.85,
        bass_suppression=0.90,
        source_tonal_weight=0.90,
    )

    assert _projection_rms(candidate, melody) > _projection_rms(weak_model_melody, melody) * 2.0
    assert _projection_rms(candidate, vocal) < _projection_rms(source_audio, vocal) * 0.35
    assert diagnostics["source_contributions"]["source_tonal_prior"]["weight"] == 0.90


def test_model_stack_manifest_requires_no_reference_base_and_writes_melody(tmp_path: Path) -> None:
    module = _load_module()
    sample_rate = 8000
    source_path = tmp_path / "source_mix.flac"
    source_audio = _sine(sample_rate, 1.0, 330.0, 0.25)
    sf.write(source_path, source_audio, sample_rate, subtype="PCM_24")

    stem_dir = tmp_path / "base"
    stem_dir.mkdir()
    melody = _sine(sample_rate, 1.0, 330.0, 0.25)
    vocals = _sine(sample_rate, 1.0, 660.0, 0.18)
    drums = _noise_burst(sample_rate, 1.0, 0.35)
    bass = _sine(sample_rate, 1.0, 80.0, 0.20)
    base_manifest = {
        "audio_path": str(source_path),
        "audio_sha256": module.sha256_file(source_path),
        "model_name": "synthetic_no_reference_base",
        "sample_rate": sample_rate,
        "contract": {
            "target_reference_assisted": False,
            "target_reference_used": False,
            "target_reference_used_at_application": False,
            "application_reference_free": True,
            "delivery_allowed_without_reference": True,
            "registered_model_candidate": True,
            "fallback_used": False,
            "hidden_inference": False,
            "per_track_hardcode_allowed": False,
            "not_timestamp_hardcoded": True,
        },
        "pipeline": {
            "mode": "synthetic_base",
            "target_reference_read": False,
            "reference_sum_mix_read": False,
        },
        "stems": {
            "melody": _write_stem(stem_dir / "melody.flac", melody + vocals * 0.6, sample_rate),
            "vocals": _write_stem(stem_dir / "vocals.flac", vocals, sample_rate),
            "drums": _write_stem(stem_dir / "drums.flac", drums, sample_rate),
            "bass": _write_stem(stem_dir / "bass.flac", bass, sample_rate),
        },
    }
    base_manifest_path = stem_dir / "stem_manifest.json"
    base_manifest_path.write_text(json.dumps(base_manifest, indent=2), encoding="utf-8")

    output_dir = tmp_path / "candidate"
    manifest = module.build_model_stack_melody_candidate(
        source_audio_path=source_path,
        output_dir=output_dir,
        base_manifest_path=base_manifest_path,
        harmonic_n_fft=512,
        harmonic_hop_length=128,
    )

    saved_manifest = json.loads((output_dir / "stem_manifest.json").read_text(encoding="utf-8"))
    assert (output_dir / "melody.flac").exists()
    assert saved_manifest["contract"]["target_reference_assisted"] is False
    assert saved_manifest["contract"]["target_reference_used"] is False
    assert saved_manifest["contract"]["target_reference_used_at_application"] is False
    assert saved_manifest["contract"]["application_reference_free"] is True
    assert saved_manifest["contract"]["fallback_used"] is False
    assert saved_manifest["contract"]["per_track_hardcode_allowed"] is False
    assert saved_manifest["pipeline"]["mode"] == "model_stack_melody_v1"
    assert saved_manifest["pipeline"]["base_manifest_contract_checked"] is True
    assert saved_manifest["pipeline"]["target_reference_read"] is False
    assert saved_manifest["pipeline"]["reference_sum_mix_read"] is False
    assert saved_manifest["pipeline"]["source"] == "authoritative_full_mix_plus_no_reference_model_stack"
    assert saved_manifest["stems"]["melody"]["path"] == str(output_dir / "melody.flac")
    assert manifest["manifest_path"] == str(output_dir / "stem_manifest.json")


def test_model_stack_rejects_base_manifest_without_source_hash_or_hashable_audio_path(tmp_path: Path) -> None:
    module = _load_module()
    sample_rate = 8000
    source_path = tmp_path / "source_mix.flac"
    sf.write(source_path, _sine(sample_rate, 1.0, 330.0, 0.25), sample_rate, subtype="PCM_24")

    stem_dir = tmp_path / "base"
    stem_dir.mkdir()
    melody_path = stem_dir / "melody.flac"
    sf.write(melody_path, _sine(sample_rate, 1.0, 330.0, 0.25), sample_rate, subtype="PCM_24")
    base_manifest = {
        "audio_path": str(tmp_path / "missing_source.flac"),
        "model_name": "synthetic_no_reference_base",
        "sample_rate": sample_rate,
        "contract": {
            "target_reference_assisted": False,
            "target_reference_used": False,
            "target_reference_used_at_application": False,
            "application_reference_free": True,
            "delivery_allowed_without_reference": True,
            "registered_model_candidate": True,
            "fallback_used": False,
            "hidden_inference": False,
            "per_track_hardcode_allowed": False,
            "not_timestamp_hardcoded": True,
        },
        "pipeline": {
            "target_reference_read": False,
            "reference_sum_mix_read": False,
        },
        "stems": {"melody": {"path": str(melody_path)}},
    }
    base_manifest_path = stem_dir / "stem_manifest.json"
    base_manifest_path.write_text(json.dumps(base_manifest, indent=2), encoding="utf-8")

    try:
        module.build_model_stack_melody_candidate(
            source_audio_path=source_path,
            output_dir=tmp_path / "candidate",
            base_manifest_path=base_manifest_path,
            harmonic_n_fft=512,
            harmonic_hop_length=128,
            source_tonal_n_fft=512,
            source_tonal_hop_length=128,
        )
    except ValueError as error:
        assert "base manifest source audio is not hashable" in str(error)
    else:
        raise AssertionError("expected missing base manifest source audio to be rejected")


def test_base_manifest_source_hash_accepts_scnet_source_audio_path(tmp_path: Path) -> None:
    module = _load_module()
    source_path = tmp_path / "source_mix.flac"
    sf.write(source_path, _sine(8000, 1.0, 330.0, 0.25), 8000, subtype="PCM_24")
    manifest_path = tmp_path / "scnet_base" / "stem_manifest.json"
    manifest_path.parent.mkdir()
    manifest = {
        "source_audio_path": str(source_path),
        "model_id": "scnet_xl_ihf_musdb18",
        "contract": {
            "target_reference_assisted": False,
            "target_reference_used": False,
            "target_reference_used_at_application": False,
            "application_reference_free": True,
            "delivery_allowed_without_reference": True,
            "registered_model_candidate": True,
            "fallback_used": False,
            "hidden_inference": False,
            "per_track_hardcode_allowed": False,
            "not_timestamp_hardcoded": True,
        },
        "stems": {},
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    assert module._base_manifest_source_sha256(manifest, manifest_path) == module.sha256_file(source_path)
