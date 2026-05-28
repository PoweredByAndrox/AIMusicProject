from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import soundfile as sf


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "build_joint_vocal_melody_candidate_v3.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("build_joint_vocal_melody_candidate_v3", MODULE_PATH)
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
    start = int(0.52 * sample_rate)
    end = int(0.70 * sample_rate)
    rng = np.random.default_rng(931)
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


def _base_contract() -> dict[str, object]:
    return {
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
    }


def test_joint_v3_selects_stem_specific_candidates_from_multiple_no_reference_bases() -> None:
    module = _load_module()
    sample_rate = 8000
    duration_sec = 2.0
    vocal = _sine(sample_rate, duration_sec, 660.0, 0.22)
    melody = _sine(sample_rate, duration_sec, 440.0, 0.28)
    bass = _sine(sample_rate, duration_sec, 90.0, 0.18)
    drums = _noise_burst(sample_rate, duration_sec, 0.42)
    source_audio = vocal + melody + bass + drums

    candidate_sets = {
        "clean_vocal_dirty_melody": {
            "vocals": vocal + melody * 0.10,
            "melody": melody + vocal * 0.70 + drums * 0.45 + bass * 0.30,
            "drums": drums,
            "bass": bass,
        },
        "dirty_vocal_clean_melody": {
            "vocals": vocal + melody * 0.75 + drums * 0.20,
            "melody": melody + vocal * 0.08,
            "drums": drums,
            "bass": bass,
        },
    }

    outputs, diagnostics = module.build_joint_vocal_melody_v3_candidate_audio(
        candidate_sets,
        source_audio=source_audio,
        sample_rate=sample_rate,
        cross_stem_suppression=0.92,
        non_target_suppression=0.85,
    )

    assert diagnostics["route_name"] == "joint_vocal_melody_v3"
    assert diagnostics["selected_candidates"]["vocals"] == "clean_vocal_dirty_melody"
    assert diagnostics["selected_candidates"]["melody"] == "dirty_vocal_clean_melody"
    assert _projection_rms(outputs["vocals"], melody) < _projection_rms(
        candidate_sets["dirty_vocal_clean_melody"]["vocals"],
        melody,
    ) * 0.30
    assert _projection_rms(outputs["melody"], vocal) < _projection_rms(
        candidate_sets["clean_vocal_dirty_melody"]["melody"],
        vocal,
    ) * 0.25
    assert _projection_rms(outputs["melody"], drums) < _projection_rms(
        candidate_sets["clean_vocal_dirty_melody"]["melody"],
        drums,
    ) * 0.35
    assert _projection_rms(outputs["vocals"], vocal) > _projection_rms(vocal, vocal) * 0.75
    assert _projection_rms(outputs["melody"], melody) > _projection_rms(melody, melody) * 0.75


def test_joint_v3_manifest_writes_no_reference_multi_candidate_provenance(tmp_path: Path) -> None:
    module = _load_module()
    sample_rate = 8000
    source_path = tmp_path / "source_mix.flac"
    vocal = _sine(sample_rate, 1.0, 660.0, 0.22)
    melody = _sine(sample_rate, 1.0, 440.0, 0.25)
    drums = _noise_burst(sample_rate, 1.0, 0.35)
    bass = _sine(sample_rate, 1.0, 90.0, 0.18)
    sf.write(source_path, vocal + melody + drums + bass, sample_rate, subtype="PCM_24")

    manifest_paths: list[Path] = []
    for label, vocals_audio, melody_audio in (
        ("vocal_base", vocal + melody * 0.10, melody + vocal * 0.65 + drums * 0.30),
        ("melody_base", vocal + melody * 0.70, melody + vocal * 0.08),
    ):
        stem_dir = tmp_path / label
        stem_dir.mkdir()
        base_manifest = {
            "audio_path": str(source_path),
            "audio_sha256": module.sha256_file(source_path),
            "model_name": label,
            "sample_rate": sample_rate,
            "contract": _base_contract(),
            "pipeline": {
                "target_reference_read": False,
                "reference_sum_mix_read": False,
            },
            "stems": {
                "vocals": _write_stem(stem_dir / "vocals.flac", vocals_audio, sample_rate),
                "melody": _write_stem(stem_dir / "melody.flac", melody_audio, sample_rate),
                "drums": _write_stem(stem_dir / "drums.flac", drums, sample_rate),
                "bass": _write_stem(stem_dir / "bass.flac", bass, sample_rate),
            },
        }
        manifest_path = stem_dir / "stem_manifest.json"
        manifest_path.write_text(json.dumps(base_manifest, indent=2), encoding="utf-8")
        manifest_paths.append(manifest_path)

    output_dir = tmp_path / "candidate"
    listen_dir = tmp_path / "listen"
    manifest = module.build_joint_vocal_melody_v3_candidate(
        source_audio_path=source_path,
        output_dir=output_dir,
        base_manifest_paths=manifest_paths,
        listen_dir=listen_dir,
    )

    saved_manifest = json.loads((output_dir / "stem_manifest.json").read_text(encoding="utf-8"))
    assert (output_dir / "vocals.flac").exists()
    assert (output_dir / "melody.flac").exists()
    assert saved_manifest["contract"]["target_reference_assisted"] is False
    assert saved_manifest["contract"]["target_reference_used_at_application"] is False
    assert saved_manifest["contract"]["application_reference_free"] is True
    assert saved_manifest["contract"]["fallback_used"] is False
    assert saved_manifest["contract"]["per_track_hardcode_allowed"] is False
    assert saved_manifest["contract"]["route_name"] == "joint_vocal_melody_v3"
    assert saved_manifest["pipeline"]["mode"] == "joint_vocal_melody_v3"
    assert saved_manifest["pipeline"]["target_reference_read"] is False
    assert saved_manifest["pipeline"]["reference_sum_mix_read"] is False
    assert saved_manifest["pipeline"]["postprocess"] == "multi_model_stem_specific_proxy_router_v3"
    assert len(saved_manifest["pipeline"]["base_manifests"]) == 2
    assert saved_manifest["listen"]["status"] == "diagnostic_requires_objective_gate"
    assert (listen_dir / "01_joint_v3_vocals.flac").exists()
    assert (listen_dir / "02_joint_v3_melody.flac").exists()
    assert manifest["manifest_path"] == str(output_dir / "stem_manifest.json")
