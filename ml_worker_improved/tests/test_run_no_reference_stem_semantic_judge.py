from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from tools.run_no_reference_stem_semantic_judge import judge_listen_manifest, run_stem_semantic_judge, main, stem_semantic_result


def write_audio(path: Path, data: np.ndarray, sample_rate: int = 16_000) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, data.astype(np.float32), sample_rate)
    return path


def sine_audio(freq_hz: float, *, duration_sec: float = 2.0, amplitude: float = 0.08) -> np.ndarray:
    sample_rate = 16_000
    t = np.arange(int(sample_rate * duration_sec), dtype=np.float32) / float(sample_rate)
    mono = np.sin(2.0 * np.pi * freq_hz * t).astype(np.float32) * amplitude
    return np.column_stack([mono, mono])


def noise_audio(*, duration_sec: float = 2.0, amplitude: float = 0.08) -> np.ndarray:
    rng = np.random.default_rng(1234)
    mono = rng.normal(0.0, amplitude, int(16_000 * duration_sec)).astype(np.float32)
    return np.column_stack([mono, mono])


def mixed_pair_audio(*, duration_sec: float = 2.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vocal = sine_audio(440.0, duration_sec=duration_sec, amplitude=0.05)
    minus = sine_audio(120.0, duration_sec=duration_sec, amplitude=0.10)
    return vocal + minus, vocal, minus


def impulse_audio(*, duration_sec: float = 2.0, amplitude: float = 0.8, every: int = 2_000) -> np.ndarray:
    sample_rate = 16_000
    mono = np.zeros(int(sample_rate * duration_sec), dtype=np.float32)
    mono[::every] = amplitude
    return np.column_stack([mono, mono])


def write_listen_manifest(path: Path, *, stem_type: str, audio_path: Path, delivery_allowed: bool = True) -> Path:
    payload = {
        "manifest_type": "no_reference_stem_listen_preview",
        "delivery_status": "accepted" if delivery_allowed else "blocked",
        "delivery_allowed": delivery_allowed,
        "presentation_status": "accepted_for_listen_pack" if delivery_allowed else "blocked_from_listen_pack",
        "release_status": "listen_preview_only",
        "stem_type": stem_type,
        "track_id": "track-a",
        "model_name": "unit_model",
        "selected_audio_path": str(audio_path),
        "audio_integrity_gate": {"status": "passed", "duration_sec": 2.0, "rms": 0.05, "peak": 0.5},
        "gpu_provenance_gate": {
            "status": "passed",
            "gpu_inference_required": True,
            "gpu_inference_skipped": False,
            "cuda_available": True,
            "resolved_device": "cuda",
        },
        "source_contract_gate": {"status": "passed"},
        "semantic_quality_verified": False,
        "release_ready": False,
        "can_mark_release_ready": False,
        "can_publish_as_release": False,
        "audio_copied": False,
        "audio_exported": False,
        "safety": {
            "training_started": False,
            "fine_tune_started": False,
            "models_changed": False,
            "model_registry_edited": False,
            "cleanup_routes_added": False,
            "thresholds_relaxed": False,
            "release_ready": False,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_accepts_pitched_melody_preview_without_release_claim(tmp_path: Path) -> None:
    audio = write_audio(tmp_path / "melody.flac", sine_audio(440.0))
    manifest = write_listen_manifest(tmp_path / "melody.json", stem_type="melody", audio_path=audio)

    row = judge_listen_manifest(manifest)

    assert row["decision"] == "accept_candidate"
    assert row["stem_semantic_decision"] == "accept_candidate"
    assert row["stem_semantic_score"] >= 0.70
    assert row["stem_semantic_blockers"] == []
    assert row["semantic_quality_verified"] is True
    assert row["delivery_allowed"] is False
    assert row["release_ready"] is False
    assert row["can_publish_as_release"] is False


def test_blocks_noisy_melody_without_pitch_confidence(tmp_path: Path) -> None:
    audio = write_audio(tmp_path / "melody_noise.flac", noise_audio())
    manifest = write_listen_manifest(tmp_path / "melody_noise.json", stem_type="melody", audio_path=audio)

    row = judge_listen_manifest(manifest)

    assert row["decision"] == "blocked_hard_case"
    assert "melody_low_pitch_confidence" in row["stem_semantic_blockers"]


def test_blocks_bass_dominant_melody_even_with_high_pitch_confidence(tmp_path: Path) -> None:
    audio = write_audio(tmp_path / "melody_bass_dominant.flac", sine_audio(57.0, amplitude=0.12))
    manifest = write_listen_manifest(tmp_path / "melody_bass_dominant.json", stem_type="melody", audio_path=audio)

    row = judge_listen_manifest(manifest)

    assert row["decision"] == "blocked_hard_case"
    assert row["stem_semantic_metrics"]["dominant_frequency_hz"] < 80.0
    assert row["stem_semantic_metrics"]["low_band_ratio"] > 0.70
    assert "melody_bass_dominant" in row["stem_semantic_blockers"]


def test_blocks_drums_without_transient_density(tmp_path: Path) -> None:
    audio = write_audio(tmp_path / "drums_sine.flac", sine_audio(220.0))
    manifest = write_listen_manifest(tmp_path / "drums.json", stem_type="drums", audio_path=audio)

    row = judge_listen_manifest(manifest)

    assert row["decision"] == "blocked_hard_case"
    assert "drums_low_transient_density" in row["stem_semantic_blockers"]


def test_accepts_impulsive_drums(tmp_path: Path) -> None:
    audio = write_audio(tmp_path / "drums.flac", impulse_audio())
    manifest = write_listen_manifest(tmp_path / "drums_manifest.json", stem_type="drums", audio_path=audio)

    row = judge_listen_manifest(manifest)

    assert row["decision"] == "accept_candidate"
    assert row["stem_semantic_blockers"] == []


def test_blocks_bass_without_low_band_energy(tmp_path: Path) -> None:
    audio = write_audio(tmp_path / "bass_high.flac", sine_audio(1200.0))
    manifest = write_listen_manifest(tmp_path / "bass.json", stem_type="bass", audio_path=audio)

    row = judge_listen_manifest(manifest)

    assert row["decision"] == "blocked_hard_case"
    assert "bass_low_band_energy_ratio" in row["stem_semantic_blockers"]


def test_accepts_piano_tonal_preview_without_release_claim(tmp_path: Path) -> None:
    audio = write_audio(tmp_path / "piano.flac", sine_audio(440.0))
    manifest = write_listen_manifest(tmp_path / "piano.json", stem_type="piano", audio_path=audio)

    row = judge_listen_manifest(manifest)

    assert row["decision"] == "accept_candidate"
    assert row["stem_semantic_blockers"] == []
    assert row["semantic_quality_verified"] is True
    assert row["release_ready"] is False


def test_accepts_guitar_tonal_preview_without_release_claim(tmp_path: Path) -> None:
    audio = write_audio(tmp_path / "guitar.flac", sine_audio(330.0))
    manifest = write_listen_manifest(tmp_path / "guitar.json", stem_type="guitar", audio_path=audio)

    row = judge_listen_manifest(manifest)

    assert row["decision"] == "accept_candidate"
    assert row["stem_semantic_blockers"] == []
    assert row["semantic_quality_verified"] is True
    assert row["release_ready"] is False


def test_blocks_piano_noise_without_tonal_confidence(tmp_path: Path) -> None:
    audio = write_audio(tmp_path / "piano_noise.flac", noise_audio())
    manifest = write_listen_manifest(tmp_path / "piano_noise.json", stem_type="piano", audio_path=audio)

    row = judge_listen_manifest(manifest)

    assert row["decision"] == "blocked_hard_case"
    assert "piano_low_tonal_confidence" in row["stem_semantic_blockers"]


def test_accepts_polyphonic_guitar_proxy_metrics_without_single_peak_pitch() -> None:
    score, blockers, reasons = stem_semantic_result(
        "guitar",
        {
            "duration_sec": 150.0,
            "rms": 0.074,
            "peak": 0.64,
            "pitch_confidence": 0.026,
            "low_band_ratio": 0.216,
            "mid_band_ratio": 0.770,
            "spectral_flatness": 0.417,
            "transient_density": 0.701,
            "crest_factor": 8.58,
            "dominant_frequency_hz": 294.0,
        },
    )

    assert score >= 0.70
    assert blockers == []
    assert reasons == ["guitar_semantic_proxy_pass"]


def test_blocks_unsupported_stem_without_erasing_stem_type(tmp_path: Path) -> None:
    audio = write_audio(tmp_path / "symphonic.flac", sine_audio(440.0))
    manifest = write_listen_manifest(tmp_path / "symphonic.json", stem_type="symphonic", audio_path=audio)

    row = judge_listen_manifest(manifest)

    assert row["decision"] == "blocked_hard_case"
    assert row["stem_type"] == "symphonic"
    assert row["stem_semantic_blockers"] == ["unsupported_stem_type"]


def test_vocal_uses_mode_specific_pair_evidence_when_available(tmp_path: Path) -> None:
    mix_audio, vocal_audio, minus_audio = mixed_pair_audio()
    mix = write_audio(tmp_path / "mix.flac", mix_audio)
    vocal = write_audio(tmp_path / "vocal.flac", vocal_audio)
    minus = write_audio(tmp_path / "minus.flac", minus_audio)
    manifest = write_listen_manifest(tmp_path / "vocal.json", stem_type="vocal", audio_path=vocal)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["mode_specific_candidate_row"] = {
        "track_id": "track-a",
        "candidate_id": "pair-a",
        "mix_path": str(mix),
        "vocal_path": str(vocal),
        "instrumental_path": str(minus),
    }
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    row = judge_listen_manifest(manifest)

    assert row["stem_type"] == "vocal"
    assert row["score_mode"] == "vocal"
    assert row["decision"] == "accept_candidate"
    assert row["semantic_quality_verified"] is True
    assert row["mode_specific_quality_verified"] is True
    assert row["stem_semantic_blockers"] == []
    assert row["mode_specific_judge"]["candidate_id"] == "pair-a"
    assert row["release_ready"] is False


def test_minus_uses_mode_specific_pair_evidence_when_available(tmp_path: Path) -> None:
    mix_audio, vocal_audio, minus_audio = mixed_pair_audio()
    mix = write_audio(tmp_path / "mix.flac", mix_audio)
    vocal = write_audio(tmp_path / "vocal.flac", vocal_audio)
    minus = write_audio(tmp_path / "minus.flac", minus_audio)
    manifest = write_listen_manifest(tmp_path / "minus.json", stem_type="minus", audio_path=minus)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["mode_specific_candidate_row"] = {
        "track_id": "track-a",
        "candidate_id": "pair-a",
        "mix_path": str(mix),
        "vocal_path": str(vocal),
        "instrumental_path": str(minus),
    }
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    row = judge_listen_manifest(manifest)

    assert row["stem_type"] == "minus"
    assert row["score_mode"] == "minus"
    assert row["decision"] == "accept_candidate"
    assert row["semantic_quality_verified"] is True
    assert row["mode_specific_quality_verified"] is True
    assert row["stem_semantic_blockers"] == []
    assert row["mode_specific_judge"]["candidate_id"] == "pair-a"
    assert row["release_ready"] is False


def test_vocal_without_pair_evidence_blocks_as_missing_mode_specific_evidence(tmp_path: Path) -> None:
    audio = write_audio(tmp_path / "vocal.flac", sine_audio(440.0))
    manifest = write_listen_manifest(tmp_path / "vocal.json", stem_type="vocal", audio_path=audio)

    row = judge_listen_manifest(manifest)

    assert row["decision"] == "blocked_hard_case"
    assert row["stem_type"] == "vocal"
    assert row["stem_semantic_blockers"] == ["mode_specific_judge_evidence_missing"]


def test_cli_writes_jsonl_summary_and_does_not_export_audio(tmp_path: Path) -> None:
    melody = write_audio(tmp_path / "melody.flac", sine_audio(440.0))
    drums = write_audio(tmp_path / "drums.flac", impulse_audio())
    melody_manifest = write_listen_manifest(tmp_path / "melody.json", stem_type="melody", audio_path=melody)
    drums_manifest = write_listen_manifest(tmp_path / "drums.json", stem_type="drums", audio_path=drums)
    output_root = tmp_path / "judge"

    exit_code = main(
        [
            "--listen-manifest-json",
            str(melody_manifest),
            "--listen-manifest-json",
            str(drums_manifest),
            "--output-root",
            str(output_root),
        ]
    )

    assert exit_code == 0
    jsonl_path = output_root / "no_reference_stem_semantic_judge.jsonl"
    summary_json = output_root / "no_reference_stem_semantic_judge_summary.json"
    summary_md = output_root / "no_reference_stem_semantic_judge_summary.md"
    assert jsonl_path.exists()
    assert summary_json.exists()
    assert summary_md.exists()
    rows = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert [row["decision"] for row in rows] == ["accept_candidate", "accept_candidate"]
    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    assert summary["release_ready"] is False
    assert summary["semantic_quality_verified_count"] == 2
    assert summary["safety"]["audio_exported"] is False
    assert not list(output_root.rglob("*.wav"))
    assert not list(output_root.rglob("*.flac"))


def test_file_script_entrypoint_runs_with_tools_imports(tmp_path: Path) -> None:
    audio = write_audio(tmp_path / "other.flac", sine_audio(330.0))
    manifest = write_listen_manifest(tmp_path / "other.json", stem_type="other", audio_path=audio)
    output_root = tmp_path / "judge"
    repo_root = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        [
            sys.executable,
            "tools/run_no_reference_stem_semantic_judge.py",
            "--listen-manifest-json",
            str(manifest),
            "--output-root",
            str(output_root),
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert (output_root / "no_reference_stem_semantic_judge.jsonl").exists()
