from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf


TOOLS_ROOT = Path(__file__).resolve().parents[1] / "tools"
MODULE_PATH = TOOLS_ROOT / "build_percussive_fx_aux_profile_candidate.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("build_percussive_fx_aux_profile_candidate", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_audio(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio.astype(np.float32), sample_rate, format="FLAC", subtype="PCM_24")


def _write_composite_manifest(path: Path, stem_path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "source_audio_path": None,
                "model_id": "teacher_distilled_good_drums_base",
                "contract": {
                    "target_reference_assisted": True,
                    "target_reference_used_at_application": True,
                    "application_reference_free": False,
                    "delivery_allowed_without_reference": False,
                    "fallback_used": False,
                    "per_track_hardcode_allowed": False,
                    "not_timestamp_hardcoded": True,
                },
                "stems": {
                    "drums_plus_percussive_fx_aux": {
                        "path": stem_path.name,
                    }
                },
            }
        ),
        encoding="utf-8",
    )


def _noise_riser(frames: int, start: int, length: int, seed: int, gain: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    event = np.zeros(frames, dtype=np.float32)
    tail = np.linspace(1.0, 0.15, length, dtype=np.float32)
    noise = rng.normal(0.0, 1.0, length).astype(np.float32)
    event[start : start + length] = gain * noise * tail
    return event


def test_profile_route_detects_fullmix_clap_riser_without_copying_teacher_timing(tmp_path: Path) -> None:
    module = _load_module()
    sample_rate = 8000
    frames = sample_rate * 3
    time = np.arange(frames, dtype=np.float32) / float(sample_rate)

    base = np.zeros((frames, 2), dtype=np.float32)
    base[2000:2024] = 0.6
    melody = 0.045 * np.sin(2.0 * np.pi * 440.0 * time).astype(np.float32)
    teacher_event = _noise_riser(frames, start=1200, length=720, seed=3, gain=0.36)
    mix_event = _noise_riser(frames, start=15000, length=760, seed=13, gain=0.32)

    source_mix = base + np.column_stack([melody + mix_event, melody + mix_event]).astype(np.float32)
    teacher_aux = np.column_stack([teacher_event, teacher_event]).astype(np.float32)
    base_dir = tmp_path / "base"
    base_path = base_dir / "drums_plus_percussive_fx_aux.flac"
    source_path = tmp_path / "source_mix.flac"
    teacher_path = tmp_path / "teacher_clap_riser.flac"
    _write_audio(base_path, base, sample_rate)
    _write_audio(source_path, source_mix, sample_rate)
    _write_audio(teacher_path, teacher_aux, sample_rate)
    _write_composite_manifest(base_dir / "stem_manifest.json", base_path)

    manifest = module.build_percussive_fx_aux_profile_candidate(
        base_manifest_path=base_dir / "stem_manifest.json",
        source_audio_path=source_path,
        output_dir=tmp_path / "out",
        teacher_aux_path=teacher_path,
        config=module.ProfilePercussiveAuxConfig(
            frame_size=512,
            hop_size=128,
            min_profile_similarity=0.18,
            min_flatness=0.12,
            flux_percentile=62.0,
            activity_floor_ratio=0.02,
            pre_roll_frames=1,
            tail_frames=3,
            aux_gain=1.0,
            max_aux_rms_ratio=2.0,
            spectral_mask_floor=0.4,
        ),
    )

    aux_audio, _ = sf.read(manifest["stems"]["percussive_fx_aux"]["path"], always_2d=True, dtype="float32")
    aux_mono = np.mean(aux_audio, axis=1)
    teacher_timing_rms = float(np.sqrt(np.mean(aux_audio[1200:1920] * aux_audio[1200:1920]) + 1.0e-12))
    detected_event_rms = float(np.sqrt(np.mean(aux_audio[15000:15760] * aux_audio[15000:15760]) + 1.0e-12))
    melody_only_rms = float(np.sqrt(np.mean(aux_audio[6000:10000] * aux_audio[6000:10000]) + 1.0e-12))
    overlapped_melody = melody[15000:15760]
    overlapped_aux = aux_mono[15000:15760]
    melody_projection_gain = float(
        np.dot(overlapped_aux, overlapped_melody) / (np.dot(overlapped_melody, overlapped_melody) + 1.0e-12)
    )

    assert detected_event_rms > 0.012
    assert melody_only_rms < detected_event_rms * 0.25
    assert abs(melody_projection_gain) < 0.12
    assert teacher_timing_rms < detected_event_rms * 0.1
    assert manifest["contract"]["target_reference_assisted"] is False
    assert manifest["contract"]["profile_source_reference_assisted"] is True
    assert manifest["contract"]["target_reference_used_at_application"] is False
    assert manifest["contract"]["application_reference_free"] is True
    assert manifest["contract"]["delivery_allowed_without_reference"] is False
    assert manifest["pipeline"]["teacher_profile"]["audio_copied_to_output"] is False
    assert manifest["pipeline"]["teacher_profile"]["timing_used_for_detection"] is False


def test_profile_route_rejects_missing_teacher_when_profile_mode_requires_it(tmp_path: Path) -> None:
    module = _load_module()
    sample_rate = 8000
    audio = np.zeros((sample_rate, 2), dtype=np.float32)
    base_dir = tmp_path / "base"
    base_path = base_dir / "drums_plus_percussive_fx_aux.flac"
    source_path = tmp_path / "source_mix.flac"
    _write_audio(base_path, audio, sample_rate)
    _write_audio(source_path, audio, sample_rate)
    _write_composite_manifest(base_dir / "stem_manifest.json", base_path)

    try:
        module.build_percussive_fx_aux_profile_candidate(
            base_manifest_path=base_dir / "stem_manifest.json",
            source_audio_path=source_path,
            output_dir=tmp_path / "out",
            teacher_aux_path=tmp_path / "missing_teacher.flac",
        )
    except FileNotFoundError as exc:
        assert "teacher aux file is missing" in str(exc)
    else:
        raise AssertionError("missing teacher profile source must fail fast")
